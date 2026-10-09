"""🦆 Microduck Hierarchical Teleop App — Phase 3.

Runs the full hierarchical control stack:
  - Layer 1: Walk policy (ONNX) driven by WASD at 50 Hz
  - Layer 2: Head aim via arrow keys (command slots, new-cmd-obs)
  - Layer 3: Grasp toggle (G key → beak actuator)
  - Layer 4: Ground-pick one-shot (P key)

With optional LeRobot-compatible episode recording (--record flag or Space key).

Usage:
    .venv/bin/mjpython -m teleop.app
    .venv/bin/mjpython -m teleop.app --randomize
    .venv/bin/mjpython -m teleop.app --randomize --record --seed 42
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

import mujoco
import mujoco.viewer
import numpy as np

# ── workspace path ──────────────────────────────────────────────────────
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _WORKSPACE)
sys.path.insert(0, os.path.join(_WORKSPACE, "scripts"))

from teleop.utils.scene_loader import load_scene, CTRL_DT, TIMESTEP, SCENE_XML
from teleop.policy_runner import PolicyRunner
from teleop.controllers.locomotion import LocomotionController
from teleop.controllers.head_aim import HeadAimController
from teleop.controllers.grasp import GraspController
from teleop.input.web_joystick import WebJoystickRouter
from teleop.gui.overlay import HUDOverlay
from teleop.recorder.episode_recorder import EpisodeRecorder
from teleop.utils.reset import reset_episode

# Optional: scene randomizer config
try:
    from randomize_scene import load_scene_config
    _RANDOMIZER_AVAILABLE = True
except ImportError:
    _RANDOMIZER_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Microduck Hierarchical Teleop")
    parser.add_argument("--randomize", action="store_true",
                        help="Randomize robot pose, sock, and lighting on start")
    parser.add_argument("--record",  action="store_true",
                        help="Start recording immediately (also toggleable via Space)")
    parser.add_argument("--seed",    type=int, default=None, help="RNG seed")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable beak-cam video recording (faster)")
    parser.add_argument("--lighting", type=str, default=None,
                        help="Force a specific lighting preset name")
    args = parser.parse_args()

    # ── banner ──────────────────────────────────────────────────────────
    print("=" * 62)
    print("🦆  MICRODUCK TELEOP  —  Phase 3 Hierarchical Control")
    print("=" * 62)
    print(f"Scene:   {SCENE_XML}")

    # ── Load scene ──────────────────────────────────────────────────────
    model, data, handles = load_scene()
    print(f"Model:   {model.nbody} bodies, {model.nu} actuators, {model.nq} DOF")

    # ── Load scene config for randomization ─────────────────────────────
    config_path = os.path.join(_WORKSPACE, "configs", "scene_apartment_v1.yaml")
    scene_config = None
    if _RANDOMIZER_AVAILABLE and os.path.exists(config_path):
        scene_config = load_scene_config(config_path)
        print(f"Config:  {len(scene_config.sock_spawn_zones)} spawn zones loaded")
    else:
        print("Config:  not found — randomization disabled")

    rng = np.random.default_rng(args.seed)

    # ── Build controller stack ───────────────────────────────────────────
    runner   = PolicyRunner(model, data, handles)
    loco     = LocomotionController()
    head     = HeadAimController()
    grasp    = GraspController()
    recorder = EpisodeRecorder(
        model, data, handles,
        record_video=not args.no_video,
    )
    hud      = HUDOverlay(fps_target=2.0)

    # ── Serve Web UI ─────────────────────────────────────────────────────
    def serve_web_ui():
        web_dir = os.path.join(_WORKSPACE, "teleop", "gui", "web")
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=web_dir, **kwargs)
            def end_headers(self):
                # Prevent browser caching so code changes take effect immediately
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                super().end_headers()
            def log_message(self, format, *args):
                pass  # suppress HTTP logs
        
        httpd = HTTPServer(("0.0.0.0", 8000), Handler)
        httpd.serve_forever()

    threading.Thread(target=serve_web_ui, daemon=True).start()
    print("\n[app] 🌐 Web UI available at: http://localhost:8000")

    router = WebJoystickRouter(loco, head, grasp, runner)

    # ── App state ────────────────────────────────────────────────────────
    _state = {
        "recording":        args.record,
        "quit":             False,
        "current_ep_config": None,
        "episode_started":  False,
    }

    # ── Callbacks wired into the keyboard router ─────────────────────────

    def do_randomize():
        if scene_config is None:
            print("[app] No scene config — cannot randomize")
            return
        ep = reset_episode(model, data, scene_config, rng, runner, args.lighting)
        _state["current_ep_config"] = ep
        print(f"[app] Randomized: pose={ep.robot_pose.name}, zone={ep.sock_zone_name}")

    def do_record_toggle():
        if _state["recording"]:
            # Stop recording
            _state["recording"] = False
            recorder.finish_episode(outcome="aborted")   # user can mark success via Esc
            hud.mark_episode_end()
            print("[app] ○ Recording stopped")
        else:
            # Start recording
            ep_id = recorder.start_episode(_state["current_ep_config"])
            _state["recording"] = True
            hud.mark_episode_start(ep_id, recording=True)
            print(f"[app] ● Recording episode {ep_id}")

    def do_quit():
        _state["quit"] = True
        if _state["recording"]:
            recorder.finish_episode(outcome="aborted")

    router.set_app_callbacks(
        randomize     = do_randomize,
        record_toggle = do_record_toggle,
        quit          = do_quit,
    )

    # ── Initial randomize (if requested) ────────────────────────────────
    if args.randomize and scene_config is not None:
        ep = reset_episode(model, data, scene_config, rng, runner, args.lighting)
        _state["current_ep_config"] = ep

    # ── Auto-start recording ─────────────────────────────────────────────
    if args.record:
        ep_id = recorder.start_episode(_state["current_ep_config"])
        _state["recording"] = True
        hud.mark_episode_start(ep_id, recording=True)

    # ── Control loop variables ───────────────────────────────────────────
    decimation_counter = 0           # count physics steps; run policy every 4
    render_fps_divider = 0           # capture video at ~30 Hz (every 34 ms)
    RENDER_EVERY_N = max(1, int(round(1.0 / (30 * TIMESTEP))))  # ~6 steps @ 0.005 s

    # ── Print controls ───────────────────────────────────────────────────
    print("\nControls:")
    print("  Use the Web UI at http://localhost:8000 for continuous joystick control.")
    print("  (Space = record, Esc = quit, R = randomize from viewer window)")
    print("=" * 62)
    print()  # blank line for HUD

    # ── Launch viewer ─────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        model, data,
        key_callback=router.on_key,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:

        prev_time = time.perf_counter()

        while viewer.is_running() and not _state["quit"]:
            now = time.perf_counter()
            dt  = min(now - prev_time, 0.1)   # clamp in case of lag
            prev_time = now

            # ── Physics step ───────────────────────────────────────────
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            decimation_counter += 1
            render_fps_divider  += 1

            # ── 50 Hz policy step (every DECIMATION physics steps) ─────
            if decimation_counter >= 4:
                decimation_counter = 0
                router.tick(CTRL_DT)     # update command from held keys
                runner.step(CTRL_DT)     # obs → infer → ctrl

                # ── Record step ────────────────────────────────────────
                if _state["recording"]:
                    do_render = (render_fps_divider >= RENDER_EVERY_N)
                    if do_render:
                        render_fps_divider = 0
                    recorder.record_step(
                        vx          = float(loco.vx),
                        wz          = float(loco.wz),
                        head_offset = head.offset,
                        grasp_state = grasp.state_float,
                        render_frame= do_render,
                    )

            # ── HUD update ─────────────────────────────────────────────
            hud.render(
                mode        = runner.current_mode,
                vx          = float(loco.vx),
                wz          = float(loco.wz),
                head_pitch  = float(head.head_pitch),
                head_yaw    = float(head.head_yaw),
                neck_pitch  = float(head.neck_pitch),
                grasp_open  = grasp.open,
                viewer      = viewer,
            )

            # ── Sync viewer ────────────────────────────────────────────
            viewer.sync()

            # ── Rate-limit to sim timestep ─────────────────────────────
            elapsed = time.perf_counter() - step_start
            sleep   = TIMESTEP - elapsed
            if sleep > 0:
                time.sleep(sleep)

    print("\n[app] Viewer closed. Goodbye! 🦆")
    if _state["recording"]:
        recorder.finish_episode(outcome="aborted")


if __name__ == "__main__":
    main()
