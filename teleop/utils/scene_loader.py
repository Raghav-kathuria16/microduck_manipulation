"""Scene loader: discovers joints, sensors, and actuators in apartment_v1.xml."""

from __future__ import annotations
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_XML = os.path.join(WORKSPACE_ROOT, "scenes", "apartment_v1", "apartment_v1.xml")

# Matches actuator order in robot_allcollisions.xml (= policy joint order)
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]

# Default (standing) pose that policy actions are offsets from
DEFAULT_POSE = np.array([
    0.0,      # left_hip_yaw
    -0.0873,  # left_hip_roll
    -0.4579,  # left_hip_pitch
    -0.0049,  # left_knee
     0.4530,  # left_ankle
     0.3491,  # neck_pitch
     0.3491,  # head_pitch
     0.0,     # head_yaw
     0.0,     # head_roll
     0.0,     # right_hip_yaw
     0.0873,  # right_hip_roll
     0.4579,  # right_hip_pitch
     0.0049,  # right_knee
    -0.4530,  # right_ankle
], dtype=np.float32)

NUM_JOINTS = 14
OBS_SIZE   = 61
CMD_SIZE   = 13

# 50 Hz control loop
TIMESTEP   = 0.005
DECIMATION = 4
CTRL_DT    = TIMESTEP * DECIMATION  # 0.02 s

# Velocity command limits (from the HF Simulator constants.js)
VEL_FWD  =  1.0    # m/s forward (running)
VEL_BACK = -0.20   # m/s backward
VEL_ANG  =  1.0    # rad/s angular


@dataclass
class SceneHandles:
    """All the MuJoCo IDs/indices we need at runtime, resolved once at load."""
    # Freejoint: trunk_base (root of the robot)
    trunk_body_id: int
    trunk_jnt_id:  int
    trunk_qpos_adr: int

    # Actuated joint qpos / qvel indices (in actuator/policy order)
    joint_qpos_indices: list[int] = field(default_factory=list)
    joint_qvel_indices: list[int] = field(default_factory=list)

    # Beak actuator (open/close grasp)
    act_beak_id: int = -1

    # IMU sensors
    imu_ang_vel_id: int = -1
    imu_accel_id:   int = -1

    # Sock freejoint (for privileged logging)
    sock_jnt_id:    int = -1
    sock_qpos_adr:  int = -1

    # Target zone site (for success detection)
    target_site_id: int = -1


def load_scene(xml_path: str = SCENE_XML) -> tuple[mujoco.MjModel, mujoco.MjData, SceneHandles]:
    """Load the apartment scene and resolve all handles.

    Returns:
        (model, data, handles)
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Scene XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    handles = _resolve_handles(model)
    return model, data, handles


def _resolve_handles(model: mujoco.MjModel) -> SceneHandles:
    """Walk the model and fill a SceneHandles."""

    def jnt_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def sensor_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)

    def act_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def body_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def site_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

    # Robot root
    trunk_bid = body_id("trunk_base")
    trunk_jid = jnt_id("trunk_base_freejoint")
    trunk_qpos_adr = int(model.jnt_qposadr[trunk_jid]) if trunk_jid >= 0 else 0

    # Actuated joints in policy order
    # Use actuator transmission indices so this works even if extra passive
    # joints exist (e.g., future roller wheels)
    joint_qpos_indices = [
        int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)
    ]
    joint_qvel_indices = [
        int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)
    ]

    # Beak actuator (may not exist yet; graceful)
    act_beak = act_id("act_beak")

    # Sensors
    imu_ang_vel = sensor_id("imu_ang_vel")
    imu_accel   = sensor_id("imu_accel")

    # Sock freejoint (set up in scene XML later; optional here)
    sock_jid     = jnt_id("sock_joint")
    sock_qpos    = int(model.jnt_qposadr[sock_jid]) if sock_jid >= 0 else -1

    # Target zone (dock_site in the apartment scene)
    target_site = site_id("dock_site")

    return SceneHandles(
        trunk_body_id       = trunk_bid,
        trunk_jnt_id        = trunk_jid,
        trunk_qpos_adr      = trunk_qpos_adr,
        joint_qpos_indices  = joint_qpos_indices,
        joint_qvel_indices  = joint_qvel_indices,
        act_beak_id         = act_beak,
        imu_ang_vel_id      = imu_ang_vel,
        imu_accel_id        = imu_accel,
        sock_jnt_id         = sock_jid,
        sock_qpos_adr       = sock_qpos,
        target_site_id      = target_site,
    )
