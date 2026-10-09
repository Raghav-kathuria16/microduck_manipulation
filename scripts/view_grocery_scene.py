#!/usr/bin/env python3
"""Interactive 3D Viewer for the Realistic Kraft Grocery Bag and Grounded MicroDuck.

Launch with:
    microduck_rl/.venv/bin/mjpython microduck_manipulation/scripts/view_grocery_scene.py
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")
_ROOT = os.path.dirname(_MANIP_DIR)
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))
sys.path.insert(0, os.path.join(_ROOT, "microduck_rl/scripts"))
sys.path.insert(0, _MANIP_DIR)

from infer_policy import (
    load_bam_model,
    load_mujoco_with_bam,
    PolicyInference,
    BAM_KP_FW,
    BAM_VIN_MIN,
)

XML_PATH = os.path.join(
    _EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_grocery_bag.xml"
)
DEFAULT_STANDING_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/BEST_alpha_stand.onnx")
DEFAULT_GROUND_PICK_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_ground_pick_headup.onnx")

def main():
    bam_model = load_bam_model(kp_fw=BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, names = load_mujoco_with_bam(
        XML_PATH, bam_model, 0.005, 0.0, BAM_VIN_MIN
    )

    policy = PolicyInference(
        model, data,
        standing_onnx_path=DEFAULT_STANDING_POLICY,
        ground_pick_onnx_path=DEFAULT_GROUND_PICK_POLICY,
        ground_pick_period=4.0,
        use_projected_gravity=True,
        new_cmd_obs=True,
        bam_ctrl=bam_ctrl,
    )

    beak_jid = model.joint("beak").id
    beak_qadr = model.jnt_qposadr[beak_jid]
    beak_act_id = policy.beak_act_id

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]
    data.qpos[beak_qadr] = 1.500

    bag_jid = model.joint("bag_free").id
    bag_qadr = model.jnt_qposadr[bag_jid]
    data.qpos[bag_qadr:bag_qadr+3] = [0.200, 0.0, 0.000]
    data.qpos[bag_qadr+3:bag_qadr+7] = [1.0, 0.0, 0.0, 0.0]

    bam_ctrl.reset(data.qpos)
    mujoco.mj_forward(model, data)

    # 80-step standing settle
    for step in range(80):
        if step % 4 == 0:
            act = policy.infer()
            policy.apply_action(act)
        bam_ctrl.q_target[beak_act_id] = 1.500
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    print("=" * 60)
    print("Launching Interactive MuJoCo Viewer for Realistic Grocery Bag")
    print("  - Robot: Grounded with active RL standing balance")
    print("  - Bag:   Realistic folded kraft paper bag with inward gussets")
    print("  - Strap: Slender 1.6mm twine loop")
    print("=" * 60)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat = [0.12, 0.0, 0.08]
        viewer.cam.distance = 0.48
        viewer.cam.elevation = -12.0
        viewer.cam.azimuth = 135.0

        step_count = 0
        while viewer.is_running():
            step_start = time.time()
            if step_count % 4 == 0:
                act = policy.infer()
                policy.apply_action(act)
            bam_ctrl.q_target[beak_act_id] = 1.500
            bam_ctrl.update()
            mujoco.mj_step(model, data)
            viewer.sync()
            step_count += 1

            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    main()
