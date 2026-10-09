"""PolicyRunner: ONNX policy inference at 50 Hz.

Exact port of the infer_policy.py PolicyInference class, adapted for the
apartment scene and the hierarchical teleop command interface.

Obs layout (61D, new-cmd-obs flavor — identical to the HF Simulator):
  [base_ang_vel(3), projected_gravity(3), joint_pos_rel(14),
   joint_vel(14), last_action(14), command(13)]

Command layout (13D):
  [vx, vy, wz,                               ← locomotion twist
   neck_pitch, head_pitch, head_yaw, head_roll, ← head aim offsets
   body_x, body_y, body_z, body_roll, body_pitch, body_yaw]  ← body pose

Action → control:
  ctrl = DEFAULT_POSE + action * ACTION_SCALE
"""

from __future__ import annotations
import os
from typing import Optional

import mujoco
import numpy as np
import onnxruntime as ort

from teleop.utils.scene_loader import (
    SceneHandles, DEFAULT_POSE, NUM_JOINTS, OBS_SIZE, CMD_SIZE,
    TIMESTEP, DECIMATION, CTRL_DT, VEL_FWD, VEL_BACK, VEL_ANG,
)

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIR = os.path.join(WORKSPACE_ROOT, "policies", "onnx")

ACTION_SCALE = 1.0
SWITCH_THRESHOLD = 0.05  # vel magnitude below which we use the stand policy

# Ground-pick phase encoding (matches infer_policy.py defaults)
GROUND_PICK_PERIOD_S  = 4.0
GROUND_PICK_END_PHASE = 0.7


class PolicyRunner:
    """Manages ONNX session(s) and produces joint targets at 50 Hz.

    Usage:
        runner = PolicyRunner(model, data, handles)
        runner.set_velocity_command(vx=0.2, wz=0.0)
        runner.set_head_offset(neck=0.3, pitch=0.0, yaw=0.2, roll=0.0)
        # in the control loop (every CTRL_DT seconds):
        runner.step()   # runs obs → policy → apply ctrl
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        handles: SceneHandles,
    ):
        self.model   = model
        self.data    = data
        self.handles = handles

        # Load sessions
        self._walk_sess  = self._load(os.path.join(POLICY_DIR, "BEST_alpha_walking.onnx"), required=True)
        self._stand_sess = self._load(os.path.join(POLICY_DIR, "BEST_alpha_stand.onnx"),   required=False)
        self._ss_sess    = self._load(os.path.join(POLICY_DIR, "BEST_alpha_sitstand.onnx"),required=False)
        self._gp_sess    = self._load(os.path.join(POLICY_DIR, "alpha_ground_pick.onnx"),  required=False)

        # Active session — the walk policy handles balancing and locomotion.
        # At zero velocity it produces a stable stance (minimal stepping).
        self._session     = self._walk_sess
        self.current_mode = "walking"

        # Policy state
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.command     = np.zeros(CMD_SIZE,   dtype=np.float32)

        # Command sub-fields (updated by controllers)
        self._vel_cmd    = np.zeros(3, dtype=np.float32)   # [vx, vy, wz]
        self._head_off   = np.zeros(4, dtype=np.float32)   # [neck, head_pitch, yaw, roll]
        self._body_cmd   = np.zeros(6, dtype=np.float32)   # [x,y,z,roll,pitch,yaw] ← not exposed in Phase 3

        # Ground-pick one-shot
        self._gp_active  = False
        self._gp_phase   = 0.0

        # Sit state
        self._sit_flag   = False

        # Beak state (grasp)
        self.beak_open   = False

        # Internal step counter for 50 Hz decimation (we run mj_step externally)
        self._step_count = 0

        # Cache input/output names
        self._iname = self._walk_sess.get_inputs()[0].name
        self._oname = self._walk_sess.get_outputs()[0].name

        print(f"[PolicyRunner] Loaded — mode={self.current_mode}")
        print(f"  Walk policy: {self._walk_sess.get_inputs()[0].shape} → {self._walk_sess.get_outputs()[0].shape}")

    # ─── public command interface ───────────────────────────────────────

    def set_velocity_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
        """Set locomotion twist."""
        self._vel_cmd[:] = [
            float(np.clip(vx, VEL_BACK, VEL_FWD)),
            float(np.clip(vy, -0.2, 0.2)),
            float(np.clip(wz, -VEL_ANG, VEL_ANG)),
        ]
        self._maybe_switch_mode()
        self._rebuild_command()

    def set_head_offset(
        self,
        neck: float  = 0.0,
        pitch: float = 0.0,
        yaw: float   = 0.0,
        roll: float  = 0.0,
    ):
        """Set head aim offsets (fed into command slots [3:7])."""
        cap = 1.4
        self._head_off[:] = [
            float(np.clip(neck,  -cap, cap)),
            float(np.clip(pitch, -cap, cap)),
            float(np.clip(yaw,   -cap, cap)),
            float(np.clip(roll,  -0.44, 0.44)),
        ]
        self._rebuild_command()

    def trigger_ground_pick(self):
        """Start the blind beak-dip one-shot (if policy loaded)."""
        if self._gp_sess is None:
            print("[PolicyRunner] ground_pick policy not loaded")
            return
        if self._gp_active:
            return
        self._gp_active = True
        self._gp_phase  = 0.0
        self._session   = self._gp_sess
        self.current_mode = "ground_pick"
        print("[PolicyRunner] Ground pick started")

    def toggle_sit(self):
        """Toggle sit/stand (requires sitstand policy)."""
        if self._ss_sess is None:
            print("[PolicyRunner] sitstand policy not loaded")
            return
        self._sit_flag = not self._sit_flag
        if self._sit_flag:
            self._vel_cmd[:] = 0.0
            self._session = self._ss_sess
            self.current_mode = "sit"
        else:
            self.current_mode = "walking"
            self._session = self._walk_sess
        self._rebuild_command()
        print(f"[PolicyRunner] Sit: {'ON' if self._sit_flag else 'OFF'}")

    def toggle_beak(self):
        """Toggle beak (grasp) open/closed."""
        self.beak_open = not self.beak_open
        act_id = self.handles.act_beak_id
        if act_id >= 0:
            self.data.ctrl[act_id] = 0.5 if self.beak_open else 0.0
        print(f"[PolicyRunner] Beak: {'OPEN' if self.beak_open else 'CLOSED'}")

    def reset(self):
        """Zero all policy state (call at episode start)."""
        self.last_action[:] = 0.0
        self.command[:]     = 0.0
        self._vel_cmd[:]    = 0.0
        self._head_off[:]   = 0.0
        self._body_cmd[:]   = 0.0
        self._gp_active     = False
        self._gp_phase      = 0.0
        self._sit_flag      = False
        self.beak_open      = False
        self._session       = self._walk_sess
        self.current_mode   = "walking"
        self._step_count    = 0
        if self.handles.act_beak_id >= 0:
            self.data.ctrl[self.handles.act_beak_id] = 0.0
        self._rebuild_command()

        # Set actuated joints to DEFAULT_POSE so the walk policy starts
        # from exactly the state it was trained on (no correction transient).
        for i, idx in enumerate(self.handles.joint_qpos_indices):
            if i < len(DEFAULT_POSE):
                self.data.qpos[idx] = DEFAULT_POSE[i]
        self.data.qvel[:] = 0.0
        import mujoco
        mujoco.mj_forward(self.model, self.data)

    # ─── main step (call every CTRL_DT = 0.02 s) ───────────────────────

    def step(self, dt: float = CTRL_DT):
        """Run one 50 Hz policy step: build obs → infer → apply ctrl."""
        # Advance ground-pick phase
        if self._gp_active:
            self._advance_ground_pick(dt)

        obs    = self._get_obs()
        action = self._infer(obs)
        self._apply_action(action)
        self._step_count += 1

    # ─── observation ────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build the 61D observation vector."""
        return np.concatenate([
            self._get_base_ang_vel(),    # 3
            self._get_projected_gravity(), # 3
            self._get_joint_pos_rel(),   # 14
            self._get_joint_vel(),       # 14
            self.last_action,            # 14
            self.command,                # 13
        ]).astype(np.float32)

    def _get_base_ang_vel(self) -> np.ndarray:
        sid = self.handles.imu_ang_vel_id
        if sid < 0:
            return np.zeros(3, dtype=np.float32)
        adr = self.model.sensor_adr[sid]
        return self.data.sensordata[adr:adr+3].copy().astype(np.float32)

    def _get_projected_gravity(self) -> np.ndarray:
        """Project world gravity (-Z) into the trunk body frame."""
        bid  = self.handles.trunk_body_id
        quat = self.data.xquat[bid].copy().astype(np.float32)  # wxyz
        grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return _quat_rotate_inverse(quat, grav)

    def _get_joint_pos_rel(self) -> np.ndarray:
        idx = self.handles.joint_qpos_indices
        return (self.data.qpos[idx].copy().astype(np.float32) - DEFAULT_POSE)

    def _get_joint_vel(self) -> np.ndarray:
        idx = self.handles.joint_qvel_indices
        return self.data.qvel[idx].copy().astype(np.float32)

    # ─── inference ──────────────────────────────────────────────────────

    def _infer(self, obs: np.ndarray) -> np.ndarray:
        batch  = obs.reshape(1, -1)
        out    = self._session.run([self._oname], {self._iname: batch})[0]
        action = out.squeeze(0).astype(np.float32)
        self.last_action = action.copy()
        return action

    def _apply_action(self, action: np.ndarray):
        target = DEFAULT_POSE + action * ACTION_SCALE
        # Apply to all actuated joints except beak (beak managed separately)
        beak_id = self.handles.act_beak_id
        for i in range(self.model.nu):
            if i == beak_id:
                continue
            self.data.ctrl[i] = target[i] if i < len(target) else 0.0

    # ─── internal helpers ────────────────────────────────────────────────

    def _rebuild_command(self):
        """Rebuild the 13D command from sub-fields."""
        c = self.command
        c[:] = 0.0
        if self.current_mode == "walking":
            c[0:3] = self._vel_cmd
        elif self.current_mode == "sit" and self._sit_flag:
            c[0] = 1.0  # posture flag = sit
        elif self.current_mode == "ground_pick":
            # Phase encoding written by _advance_ground_pick
            pass
        c[3:7] = self._head_off

    def _maybe_switch_mode(self):
        """Auto-switching disabled — BEST_alpha_stand is a get-up policy,
        not a stand-still policy. The walk policy at zero velocity is stable."""
        if self._gp_active or self._sit_flag:
            return

    def _advance_ground_pick(self, dt: float):
        new_phase = self._gp_phase + dt / GROUND_PICK_PERIOD_S
        if new_phase >= GROUND_PICK_END_PHASE:
            self._gp_active   = False
            self._gp_phase    = 0.0
            self._session     = self._walk_sess
            self.current_mode = "walking"
            self._vel_cmd[:]  = 0.0
            self._rebuild_command()
            print("[PolicyRunner] Ground pick done → walking")
            return
        self._gp_phase = new_phase
        self.command[0] = float(np.cos(2 * np.pi * self._gp_phase))
        self.command[1] = float(np.sin(2 * np.pi * self._gp_phase))
        self.command[2] = 0.0

    @staticmethod
    def _load(path: str, required: bool) -> Optional[ort.InferenceSession]:
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(
                    f"Required policy not found: {path}\n"
                    "Run: python scripts/download_policies.py"
                )
            return None
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


# ─── math helper ────────────────────────────────────────────────────────────

def _quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec by the INVERSE of quaternion [w, x, y, z]."""
    w   = quat_wxyz[0]
    xyz = quat_wxyz[1:4]
    t   = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)
