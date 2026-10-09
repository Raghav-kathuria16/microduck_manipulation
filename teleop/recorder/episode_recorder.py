"""Episode recorder: logs per-timestep data to data/raw/episode_NNNN/.

Each episode produces:
  episode_NNNN/
    metadata.json          - episode config and outcome
    states.npy             - (T, 51) float32: joint_pos(14) + joint_vel(14) +
                                               imu_quat(4) + imu_ang_vel(3) +
                                               imu_accel(3) + timestamp(1) +
                                               trunk_pos(7)
    actions.npy            - (T, 7)  float32: vx, wz, neck, pitch, yaw, roll, grasp
    privileged.npy         - (T, 8)  float32: sock_x, sock_y, sock_z, sock_qw,
                                               sock_qx, sock_qy, sock_qz, in_target
    video_beak.mp4         - beak-cam frames at ~30 fps

The manifest at data/raw/manifest.csv tracks all episodes.
"""

from __future__ import annotations
import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import mujoco
import numpy as np

# Try OpenCV for video writing; fall back gracefully
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print("[EpisodeRecorder] opencv not available — video logging disabled")

WORKSPACE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
RAW_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "raw")
MANIFEST_CSV = os.path.join(RAW_DATA_DIR, "manifest.csv")

# Camera rendering resolution
CAM_W, CAM_H = 640, 480
CAM_FPS = 30

# Target zone detection radius (metres from dock_site)
TARGET_RADIUS = 0.25


@dataclass
class EpisodeMeta:
    """Episode-level metadata written to metadata.json."""
    episode_id:       int    = 0
    task_instruction: str    = "pick up the sock and carry it to the basket"
    scene_variant:    str    = "apartment_v1"
    sock_spawn_zone:  str    = "unknown"
    robot_spawn_pose: str    = "unknown"
    lighting_preset:  str    = "unknown"
    target_position:  list   = field(default_factory=lambda: [4.2, 2.0])
    outcome:          str    = "aborted"   # success | failure_fall | failure_timeout | aborted
    duration_seconds: float  = 0.0
    num_timesteps:    int    = 0
    source:           str    = "teleop"
    timestamp:        str    = ""
    recording_fps:    int    = CAM_FPS


class EpisodeRecorder:
    """Records a single teleop episode to disk.

    Usage:
        rec = EpisodeRecorder(model, data, handles)
        rec.start_episode(episode_config)
        # each control tick:
        rec.record_step(action_vx, action_wz, head_offset, grasp_state)
        # on finish:
        rec.finish_episode(outcome="success")
    """

    def __init__(
        self,
        model:   mujoco.MjModel,
        data:    mujoco.MjData,
        handles,
        record_video: bool = True,
        cam_name: str = "beak_cam",
    ):
        self.model        = model
        self.data         = data
        self.handles      = handles
        self.record_video = record_video and _HAS_CV2
        self.cam_name     = cam_name

        # Buffers (lists → converted to arrays on finish)
        self._states:     list[np.ndarray] = []
        self._actions:    list[np.ndarray] = []
        self._privileged: list[np.ndarray] = []
        self._timestamps: list[float]      = []

        self._meta:        Optional[EpisodeMeta] = None
        self._ep_dir:      Optional[str]          = None
        self._start_time:  Optional[float]        = None
        self._video:       Optional[object]       = None  # cv2.VideoWriter
        self._renderer:    Optional[mujoco.Renderer] = None
        self._cam_id:      int = -1
        self.active:       bool = False
        self.episode_count: int = self._count_existing_episodes()

        # Resolve beak_cam id
        self._cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

    # ── lifecycle ───────────────────────────────────────────────────────

    def start_episode(self, episode_config=None) -> int:
        """Begin recording. Returns episode_id."""
        if self.active:
            self.finish_episode(outcome="aborted")

        self.episode_count += 1
        ep_id = self.episode_count

        self._ep_dir = os.path.join(RAW_DATA_DIR, f"episode_{ep_id:04d}")
        os.makedirs(self._ep_dir, exist_ok=True)

        self._meta = EpisodeMeta(
            episode_id  = ep_id,
            timestamp   = time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        if episode_config is not None:
            self._meta.sock_spawn_zone  = episode_config.sock_zone_name
            self._meta.robot_spawn_pose = episode_config.robot_pose.name
            self._meta.lighting_preset  = episode_config.lighting.name
            self._meta.target_position  = list(episode_config.target.position[:2])

        self._states.clear()
        self._actions.clear()
        self._privileged.clear()
        self._timestamps.clear()

        self._start_time = time.monotonic()
        self.active      = True

        # Video writer
        if self.record_video and self._cam_id >= 0:
            try:
                self._renderer = mujoco.Renderer(self.model, height=CAM_H, width=CAM_W)
                vpath = os.path.join(self._ep_dir, "video_beak.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self._video = cv2.VideoWriter(vpath, fourcc, CAM_FPS, (CAM_W, CAM_H))
            except Exception as e:
                print(f"[EpisodeRecorder] Video init failed: {e}")
                self._video = None

        print(f"[EpisodeRecorder] Episode {ep_id} started → {self._ep_dir}")
        return ep_id

    def record_step(
        self,
        vx:         float,
        wz:         float,
        head_offset: np.ndarray,   # shape (4,): [neck, pitch, yaw, roll]
        grasp_state: float,        # 0.0 = closed, 1.0 = open
        render_frame: bool = False,
    ):
        """Record one control timestep (call at 50 Hz)."""
        if not self.active:
            return

        now = time.monotonic()
        self._timestamps.append(now - self._start_time)  # type: ignore

        # — State —
        state = self._collect_state()
        self._states.append(state)

        # — Action (BC label) —
        action = np.array(
            [vx, wz, *head_offset[:4], grasp_state],
            dtype=np.float32
        )
        self._actions.append(action)

        # — Privileged —
        priv = self._collect_privileged()
        self._privileged.append(priv)

        # — Video (decimated to ~30 fps) —
        if self._video is not None and render_frame:
            self._capture_frame()

    def finish_episode(self, outcome: str = "success"):
        """Save all buffers to disk and update manifest."""
        if not self.active:
            return
        self.active = False

        duration = time.monotonic() - self._start_time  # type: ignore

        # Close video
        if self._video is not None:
            self._video.release()
            self._video = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        if not self._states:
            print(f"[EpisodeRecorder] Episode {self._meta.episode_id} aborted (no steps recorded)")  # type: ignore
            return

        # Save arrays
        np.save(os.path.join(self._ep_dir, "states.npy"),     np.stack(self._states))     # type: ignore
        np.save(os.path.join(self._ep_dir, "actions.npy"),    np.stack(self._actions))    # type: ignore
        np.save(os.path.join(self._ep_dir, "privileged.npy"), np.stack(self._privileged)) # type: ignore
        np.save(os.path.join(self._ep_dir, "timestamps.npy"), np.array(self._timestamps, dtype=np.float32)) # type: ignore

        # Save metadata
        self._meta.outcome          = outcome                   # type: ignore
        self._meta.duration_seconds = duration                  # type: ignore
        self._meta.num_timesteps    = len(self._states)         # type: ignore
        with open(os.path.join(self._ep_dir, "metadata.json"), "w") as f:  # type: ignore
            json.dump(asdict(self._meta), f, indent=2)          # type: ignore

        self._append_manifest()

        print(
            f"[EpisodeRecorder] Episode {self._meta.episode_id} saved "  # type: ignore
            f"({len(self._states)} steps, {duration:.1f}s, outcome={outcome})"
        )

    # ── internal ────────────────────────────────────────────────────────

    def _collect_state(self) -> np.ndarray:
        h = self.handles
        # joint pos + vel (14+14)
        jpos = self.data.qpos[h.joint_qpos_indices].copy().astype(np.float32)
        jvel = self.data.qvel[h.joint_qvel_indices].copy().astype(np.float32)
        # trunk freejoint pose (7: xyz + quat wxyz)
        adr  = h.trunk_qpos_adr
        trunk_pose = self.data.qpos[adr:adr+7].copy().astype(np.float32)
        # IMU
        imu_quat = np.zeros(4, dtype=np.float32)
        try:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")
            if sid >= 0:
                imu_quat[:] = self.data.sensordata[self.model.sensor_adr[sid]:self.model.sensor_adr[sid]+4]
        except Exception:
            pass
        imu_ang_vel = np.zeros(3, dtype=np.float32)
        if h.imu_ang_vel_id >= 0:
            adr2 = self.model.sensor_adr[h.imu_ang_vel_id]
            imu_ang_vel[:] = self.data.sensordata[adr2:adr2+3]
        imu_accel = np.zeros(3, dtype=np.float32)
        if h.imu_accel_id >= 0:
            adr3 = self.model.sensor_adr[h.imu_accel_id]
            imu_accel[:] = self.data.sensordata[adr3:adr3+3]

        return np.concatenate([jpos, jvel, imu_quat, imu_ang_vel, imu_accel, trunk_pose])

    def _collect_privileged(self) -> np.ndarray:
        h = self.handles
        sock_pose = np.zeros(7, dtype=np.float32)
        if h.sock_qpos_adr >= 0:
            sock_pose[:] = self.data.qpos[h.sock_qpos_adr:h.sock_qpos_adr+7]

        # Is sock in target zone?
        in_target = 0.0
        if h.target_site_id >= 0 and h.sock_qpos_adr >= 0:
            site_pos = self.data.site_xpos[h.target_site_id]
            sock_pos = sock_pose[:3]
            dist = float(np.linalg.norm(sock_pos - site_pos))
            in_target = 1.0 if dist < TARGET_RADIUS else 0.0

        return np.array([*sock_pose, in_target], dtype=np.float32)

    def _capture_frame(self):
        """Render the beak camera and write to video."""
        try:
            self._renderer.update_scene(self.data, camera=self.cam_name)
            rgb = self._renderer.render()
            # OpenCV expects BGR
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self._video.write(bgr)
        except Exception as e:
            pass  # Don't crash the sim on render errors

    def _append_manifest(self):
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        write_header = not os.path.exists(MANIFEST_CSV)
        with open(MANIFEST_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["episode_id", "outcome", "duration_s", "num_steps",
                             "sock_zone", "pose", "lighting", "timestamp"])
            m = self._meta
            w.writerow([m.episode_id, m.outcome, f"{m.duration_seconds:.1f}",
                        m.num_timesteps, m.sock_spawn_zone, m.robot_spawn_pose,
                        m.lighting_preset, m.timestamp])

    def _count_existing_episodes(self) -> int:
        if not os.path.exists(RAW_DATA_DIR):
            return 0
        return sum(
            1 for d in os.listdir(RAW_DATA_DIR)
            if d.startswith("episode_") and os.path.isdir(os.path.join(RAW_DATA_DIR, d))
        )
