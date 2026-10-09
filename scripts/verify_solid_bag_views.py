#!/usr/bin/env python3
"""Render all 4 sides of the solid bag + duck alignment to prove the sides are 100% sealed."""

import os
import sys
import mujoco
import mediapy as media
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))

from infer_policy import (
    load_bam_model,
    load_mujoco_with_bam,
    PolicyInference,
    BAM_KP_FW,
    BAM_VIN_MIN,
)

ARTIFACT_DIR = "/Users/raghav/.gemini/antigravity-ide/brain/1dcd9661-c7ab-457a-8bec-a3fc2df3ade3"
IMG_DIR = os.path.join(ARTIFACT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

XML_PATH = os.path.join(
    _EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_grocery_bag.xml"
)
STANDING_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/BEST_alpha_stand.onnx")
GROUND_PICK_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_ground_pick_headup.onnx")

def main():
    bam_model = load_bam_model(kp_fw=BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, names = load_mujoco_with_bam(
        XML_PATH, bam_model, 0.005, 0.0, BAM_VIN_MIN
    )
    policy = PolicyInference(
        model, data,
        standing_onnx_path=STANDING_POLICY,
        ground_pick_onnx_path=GROUND_PICK_POLICY,
        ground_pick_period=4.0,
        use_projected_gravity=True,
        new_cmd_obs=True,
        bam_ctrl=bam_ctrl,
    )

    renderer = mujoco.Renderer(model, height=480, width=640)
    bag_jid = model.joint("bag_free").id
    beak_qadr = model.jnt_qposadr[model.joint("beak").id]

    x_bag = 0.105 # Aligned: string just above top jaw outside it

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]
    data.qpos[beak_qadr] = 1.95 # open

    b_adr = model.jnt_qposadr[bag_jid]
    data.qpos[b_adr:b_adr + 3] = [x_bag, 0.0, 0.001]
    data.qpos[b_adr + 3:b_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    bam_ctrl.reset(data.qpos)
    mujoco.mj_forward(model, data)

    for _ in range(60):
        if _ % 4 == 0:
            act = policy.infer()
            policy.apply_action(act)
        bam_ctrl.q_target[policy.beak_act_id] = 1.95
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    # Crouch to step 160
    policy.trigger_ground_pick()
    for s in range(160):
        policy.update_ground_pick_phase(0.005)
        if s % 4 == 0:
            act = policy.infer()
            policy.apply_action(act)
        bam_ctrl.q_target[policy.beak_act_id] = 1.95
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    # Cameras
    cameras = {
        "solid_proof_1_front_wall": (0.0, [0.105, 0.0, 0.04], 0.25, 0.0),
        "solid_proof_2_back_wall": (180.0, [0.105, 0.0, 0.04], 0.25, 0.0),
        "solid_proof_3_right_side_gusset": (90.0, [0.105, 0.0, 0.06], 0.28, -5.0),
        "solid_proof_4_left_side_gusset": (-90.0, [0.105, 0.0, 0.06], 0.28, -5.0),
        "solid_proof_5_perspective_duck_string": (125.0, [0.105, 0.0, 0.07], 0.35, -12.0),
        "solid_proof_6_top_jaw_string_closeup": (90.0, [0.108, 0.0, 0.10], 0.18, -4.0),
    }

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    for name, (az, lookat, dist, el) in cameras.items():
        cam.azimuth = az
        cam.lookat = lookat
        cam.distance = dist
        cam.elevation = el
        renderer.update_scene(data, cam)
        media.write_image(os.path.join(IMG_DIR, f"{name}.png"), renderer.render())
        print(f"Rendered {name}.png")

    print("\nAll solid proof images successfully written!")

if __name__ == "__main__":
    main()
