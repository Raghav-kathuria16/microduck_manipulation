"""WebSocket-based Web UI router with click-to-navigate waypoint control."""

from __future__ import annotations
import asyncio
import json
import math
import threading
from typing import Callable, Optional

import numpy as np
import websockets

from teleop.controllers.waypoint import WaypointController


class WebJoystickState:
    """Thread-safe snapshot of the latest state received from the Web UI."""
    def __init__(self):
        self.lock = threading.Lock()
        
        # Navigation target (click-to-navigate)
        self.nav_target_x: float | None = None
        self.nav_target_y: float | None = None
        self.nav_stop: bool = False
        
        # Head aim (right joystick, continuous)
        self.head_pitch: float = 0.0
        self.head_yaw: float = 0.0
        
        # Action toggles/triggers
        self.grasp: bool = False
        self.pick: bool = False
        self.sit: bool = False
        self.reset_head: bool = False
        self.randomize: bool = False
        
    def update(self, data: dict):
        with self.lock:
            # Navigation
            nav = data.get('nav_target')
            if nav is not None:
                self.nav_target_x = float(nav.get('x', 0))
                self.nav_target_y = float(nav.get('y', 0))
            else:
                self.nav_target_x = None
                self.nav_target_y = None
            
            self.nav_stop = bool(data.get('nav_stop', False))
            
            # Head aim
            self.head_pitch = float(data.get('head_pitch', 0.0))
            self.head_yaw = float(data.get('head_yaw', 0.0))
            
            actions = data.get('actions', {})
            self.grasp = bool(actions.get('grasp', False))
            self.pick = bool(actions.get('pick', False))
            self.sit = bool(actions.get('sit', False))
            self.reset_head = bool(actions.get('reset_head', False))
            self.randomize = bool(actions.get('randomize', False))


class WebJoystickRouter:
    """
    Hosts an asyncio WebSocket server and routes navigation/head/action
    inputs to the controllers. Sends robot pose back to the browser.
    """
    def __init__(
        self,
        locomotion_ctrl,
        head_ctrl,
        grasp_ctrl,
        policy_runner,
        host: str = "0.0.0.0",
        port: int = 8765
    ):
        self._loco = locomotion_ctrl
        self._head = head_ctrl
        self._grasp = grasp_ctrl
        self._runner = policy_runner
        
        self.host = host
        self.port = port
        self.state = WebJoystickState()
        self.waypoint = WaypointController()
        
        # Callbacks
        self._cb_randomize: Optional[Callable] = None
        self._cb_record_toggle: Optional[Callable] = None
        self._cb_quit: Optional[Callable] = None

        # Track previous toggle states to detect rising edges
        self._prev_grasp = False
        self._prev_sit = False
        self._prev_nav_target = None
        
        # Latest pose for sending to browser (thread-safe dict)
        self._latest_pose: dict = {'type': 'pose', 'x': 0, 'y': 0, 'yaw': 0, 'vx': 0, 'wz': 0}
        self._pose_lock = threading.Lock()
        
        self._thread = threading.Thread(target=self._start_server, daemon=True)
        self._thread.start()

    def set_app_callbacks(
        self,
        randomize: Optional[Callable] = None,
        reset: Optional[Callable] = None,
        record_toggle: Optional[Callable] = None,
        quit: Optional[Callable] = None,
    ):
        self._cb_randomize = randomize
        self._cb_record_toggle = record_toggle
        self._cb_quit = quit

    def on_key(self, keycode: int):
        if keycode == 32 and self._cb_record_toggle:
            self._cb_record_toggle()
        elif keycode == 256 and self._cb_quit:
            self._cb_quit()
        elif keycode in (82, 114) and self._cb_randomize:
            self._cb_randomize()

    def on_key_release(self, keycode: int):
        pass

    # ── WebSocket server ───────────────────────────────────────────

    async def _run_server(self):
        print(f"[WebUI] WebSocket server listening on ws://{self.host}:{self.port}")
        async with websockets.serve(self._ws_handler, self.host, self.port):
            await asyncio.Future()

    def _start_server(self):
        try:
            asyncio.run(self._run_server())
        except Exception as e:
            print(f"[WebUI] Server failed: {e}")

    async def _ws_handler(self, websocket, path=None):
        """Handle a single WebSocket client. For each command received,
        process it and immediately respond with the latest robot pose."""
        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get('type') == 'command':
                    self.state.update(data)
                    # Respond with current robot pose
                    with self._pose_lock:
                        await websocket.send(json.dumps(self._latest_pose))
        except websockets.exceptions.ConnectionClosed:
            pass

    # ── Main tick (called at 50Hz) ─────────────────────────────────

    def tick(self, dt: float):
        """Per-frame update: process navigation + head + actions."""
        
        # ── Get robot pose from simulation ─────────────────────────
        robot_x, robot_y, robot_yaw = self._get_robot_pose()
        
        with self.state.lock:
            # ── Navigation (waypoint) ──────────────────────────────
            
            # Check for new target
            if self.state.nav_target_x is not None:
                new_target = (self.state.nav_target_x, self.state.nav_target_y)
                if new_target != self._prev_nav_target:
                    self.waypoint.set_target(self.state.nav_target_x, self.state.nav_target_y)
                    self._prev_nav_target = new_target
            
            # Check for stop
            if self.state.nav_stop:
                self.waypoint.stop()
                self._prev_nav_target = None
            
            # Compute velocity from waypoint controller
            vx, wz = self.waypoint.compute(robot_x, robot_y, robot_yaw)
            
            # Set locomotion targets
            from teleop.utils.scene_loader import VEL_FWD, VEL_BACK, VEL_ANG
            self._loco._target_vx = float(np.clip(vx, VEL_BACK, VEL_FWD))
            self._loco._target_wz = float(np.clip(wz, -VEL_ANG, VEL_ANG))

            # ── Actions ────────────────────────────────────────────
            if self.state.grasp and not self._prev_grasp:
                self._grasp.toggle()
                self._runner.toggle_beak()
            elif not self.state.grasp and self._prev_grasp:
                self._grasp.toggle()
                self._runner.toggle_beak()
            self._prev_grasp = self.state.grasp

            if self.state.pick:
                self._runner.trigger_ground_pick()

            if self.state.sit and not self._prev_sit:
                self._runner.toggle_sit()
            elif not self.state.sit and self._prev_sit:
                self._runner.toggle_sit()
            self._prev_sit = self.state.sit

            if self.state.reset_head:
                self._head.reset()
                self._runner.set_head_offset(0, 0, 0, 0)

            if self.state.randomize and self._cb_randomize:
                self._cb_randomize()
                self.waypoint.stop()
                self._prev_nav_target = None
                
            # ── Head aim ───────────────────────────────────────────
            target_pitch = self.state.head_pitch * 1.4
            target_yaw = self.state.head_yaw * 1.4
            self._head.offset[1] = target_pitch
            self._head.offset[2] = target_yaw

        # ── Tick locomotion (ramp smoothing) ───────────────────────
        self._loco.tick(dt)
        cmd_vx, cmd_vy, cmd_wz = self._loco.command
        
        self._runner.set_velocity_command(cmd_vx, cmd_vy, cmd_wz)
        self._runner.set_head_offset(
            self._head.neck_pitch,
            self._head.head_pitch,
            self._head.head_yaw,
            self._head.head_roll,
        )
        
        # ── Update latest pose for browser ─────────────────────────
        with self._pose_lock:
            self._latest_pose = {
                'type': 'pose',
                'x': round(robot_x, 3),
                'y': round(robot_y, 3),
                'yaw': round(robot_yaw, 3),
                'vx': round(cmd_vx, 3),
                'wz': round(cmd_wz, 3),
                'nav_status': self.waypoint.status,
            }

    def _get_robot_pose(self) -> tuple[float, float, float]:
        """Extract robot (x, y, yaw) from the freejoint quaternion."""
        data = self._runner.data
        h = self._runner.handles
        adr = h.trunk_qpos_adr
        
        # Freejoint: qpos[adr:adr+7] = [x, y, z, qw, qx, qy, qz]
        x = float(data.qpos[adr])
        y = float(data.qpos[adr + 1])
        
        # Extract yaw from quaternion (wxyz format)
        qw = float(data.qpos[adr + 3])
        qx = float(data.qpos[adr + 4])
        qy = float(data.qpos[adr + 5])
        qz = float(data.qpos[adr + 6])
        
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return (x, y, yaw)
