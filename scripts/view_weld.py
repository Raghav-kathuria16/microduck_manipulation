#!/usr/bin/env python3
"""Interactive MuJoCo viewer for the load-carry welded cylinder.

Run from the repository root on macOS with MuJoCo's GUI Python:

    .external_repo/.venv/bin/mjpython scripts/view_weld.py

The default view is static at the official HOME pose, so the horizontal rod
can be inspected without the unactuated raw robot falling. Pass ``--run`` to
step the dynamics after the window opens.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, ".external_repo", "src"))


HOME_JOINTS = {
    "left_hip_roll": -0.0873,
    "right_hip_roll": 0.0873,
    "left_hip_pitch": -0.4579,
    "right_hip_pitch": 0.4579,
    "left_knee": -0.0049,
    "right_knee": 0.0049,
    "left_ankle": 0.4530,
    "right_ankle": -0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
}


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Return q1 * q2 for MuJoCo (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def build_welded_scene() -> tuple[mujoco.MjModel, mujoco.MjData, int, int]:
    """Compile the same robot+cylinder+weld scene used by test_weld.py."""
    from mjlab_microduck.robot.microduck_constants import (
        get_cylinder_spec,
        get_standup_spec,
    )
    from mjlab_microduck.tasks.microduck_loadcarry_env_cfg import (
        _weld_cylinder_to_jaw,
    )

    # Mirror mjlab Scene._add_entities: attach each entity under a prefix.
    spec = mujoco.MjSpec()
    robot_frame = spec.worldbody.add_frame()
    spec.attach(get_standup_spec(), prefix="robot/", frame=robot_frame)
    cylinder_frame = spec.worldbody.add_frame()
    spec.attach(get_cylinder_spec(), prefix="cylinder/", frame=cylinder_frame)
    _weld_cylinder_to_jaw(spec)
    model = spec.compile()
    data = mujoco.MjData(model)

    # Set the official HOME joint pose. Attached entity keyframes are not
    # retained by MjSpec.attach, so this mirrors test_weld.py explicitly.
    for name, value in HOME_JOINTS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")
        if joint_id < 0:
            raise RuntimeError(f"HOME joint missing from compiled model: robot/{name}")
        data.qpos[model.jnt_qposadr[joint_id]] = value

    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "robot/trunk_base_freejoint"
    )
    root_adr = model.jnt_qposadr[root_id]
    data.qpos[root_adr + 2] = 0.1880
    data.qpos[root_adr + 3 : root_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/jaw_soft")
    cylinder_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder/cylinder_joint"
    )
    cylinder_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "cylinder/cylinder"
    )
    if jaw_id < 0 or cylinder_joint_id < 0 or cylinder_body_id < 0:
        raise RuntimeError("Compiled weld scene is missing jaw or cylinder handles")

    # Initialize the free cylinder exactly at the weld's HOME relpose before
    # rendering; mj_forward then shows the rod clamped at the mouth immediately.
    from mjlab_microduck.robot.microduck_constants import (
        CYLINDER_WELD_RELPOS,
        CYLINDER_WELD_RELQUAT,
    )

    jaw_rotation = data.xmat[jaw_id].reshape(3, 3)
    cylinder_position = data.xpos[jaw_id] + jaw_rotation @ np.asarray(CYLINDER_WELD_RELPOS)
    cylinder_quaternion = _quat_mul(data.xquat[jaw_id], np.asarray(CYLINDER_WELD_RELQUAT))
    cylinder_adr = model.jnt_qposadr[cylinder_joint_id]
    data.qpos[cylinder_adr : cylinder_adr + 3] = cylinder_position
    data.qpos[cylinder_adr + 3 : cylinder_adr + 7] = cylinder_quaternion
    mujoco.mj_forward(model, data)

    return model, data, jaw_id, cylinder_body_id


def check_scene() -> int:
    """Headless check used to validate the GUI scene construction."""
    model, data, jaw_id, cylinder_id = build_welded_scene()
    equality_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, "cylinder_jaw_weld"
    )
    if equality_id < 0 or model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_WELD:
        raise RuntimeError("cylinder_jaw_weld equality is missing")

    from mjlab_microduck.robot.microduck_constants import CYLINDER_WELD_RELPOS

    jaw_rotation = data.xmat[jaw_id].reshape(3, 3)
    relative_position = jaw_rotation.T @ (data.xpos[cylinder_id] - data.xpos[jaw_id])
    error = float(np.linalg.norm(relative_position - np.asarray(CYLINDER_WELD_RELPOS)))
    print(f"GUI scene compiled; weld relative-position error: {error * 1000:.3f} mm")
    if error > 5e-3:
        raise RuntimeError("cylinder is not initialized at the welded mouth pose")
    return 0


def run_viewer(run_dynamics: bool) -> int:
    """Open the native MuJoCo viewer and focus it on the mouth/rod."""
    import mujoco.viewer

    model, data, jaw_id, _ = build_welded_scene()
    print("Weld viewer ready: static HOME pose; close the window to exit.", flush=True)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set a close-up free camera on the jaw and cylinder.
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = data.xpos[jaw_id]
            viewer.cam.distance = 0.42
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -12.0
        viewer.sync()

        while viewer.is_running():
            if run_dynamics:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Step the unactuated raw dynamics after opening (default: static pose)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compile and validate the welded pose without opening a GUI",
    )
    args = parser.parse_args()
    if args.check:
        return check_scene()
    return run_viewer(args.run)


if __name__ == "__main__":
    raise SystemExit(main())
