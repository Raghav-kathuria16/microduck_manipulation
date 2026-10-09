#!/usr/bin/env python3
"""Render and verify string alignment:
'jus go a lil bit forward like the string is just above the top jaw outside it.'

Tests bag X positions:
- 0.125m (previous default)
- 0.115m
- 0.105m
- 0.095m

Renders high-res side profile and perspective views for visual proof.
"""

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
    
    # Side camera
    cam_side = mujoco.MjvCamera()
    cam_side.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_side.lookat = [0.10, 0.0, 0.08]
    cam_side.distance = 0.30
    cam_side.elevation = -2.0
    cam_side.azimuth = 90.0

    # Perspective 3/4 camera
    cam_persp = mujoco.MjvCamera()
    cam_persp.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_persp.lookat = [0.10, 0.0, 0.08]
    cam_persp.distance = 0.35
    cam_persp.elevation = -12.0
    cam_persp.azimuth = 135.0

    # Bottom camera to prove bag is sealed
    cam_bottom = mujoco.MjvCamera()
    cam_bottom.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_bottom.lookat = [0.10, 0.0, 0.03]
    cam_bottom.distance = 0.28
    cam_bottom.elevation = 20.0
    cam_bottom.azimuth = 90.0

    bag_bid = model.body("grocery_bag").id
    bag_jid = model.joint("bag_free").id
    apex_bid = model.body("strap_apex").id
    beak_qadr = model.jnt_qposadr[model.joint("beak").id]

    positions_to_test = [
        (0.125, "x125_standard"),
        (0.115, "x115_closer"),
        (0.105, "x105_topjaw_under"),
        (0.095, "x095_forward"),
    ]

    for x_bag, tag in positions_to_test:
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

        policy.ground_pick_mode = False
        policy.current_policy = "standing"
        policy.ort_session = policy.standing_session

        # Settle 60 steps
        for _ in range(60):
            if _ % 4 == 0:
                act = policy.infer()
                policy.apply_action(act)
            bam_ctrl.q_target[policy.beak_act_id] = 1.95
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        # Trigger crouch
        policy.trigger_ground_pick()
        # Step to crouch at step 160 (phi = 0.20)
        for s in range(165):
            policy.update_ground_pick_phase(0.005)
            if s % 4 == 0:
                act = policy.infer()
                policy.apply_action(act)
            bam_ctrl.q_target[policy.beak_act_id] = 1.95
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        m_pos = data.site_xpos[model.site("mouth_tip").id]
        l_pos = data.site_xpos[model.site("lower_beak_tip").id]
        a_pos = data.xpos[apex_bid]

        print(f"\n[{tag}] x_bag={x_bag*1000:.1f}mm:")
        print(f"  Mouth tip: X={m_pos[0]*1000:.1f}mm, Z={m_pos[2]*1000:.1f}mm")
        print(f"  Strap apex: X={a_pos[0]*1000:.1f}mm, Z={a_pos[2]*1000:.1f}mm")
        print(f"  Relative X (Mouth - Apex): {(m_pos[0] - a_pos[0])*1000:+.1f}mm")
        print(f"  Relative Z (Apex - Mouth): {(a_pos[2] - m_pos[2])*1000:+.1f}mm")

        # Render Side
        renderer.update_scene(data, cam_side)
        media.write_image(os.path.join(IMG_DIR, f"string_align_side_{tag}.png"), renderer.render())

        # Render Persp
        renderer.update_scene(data, cam_persp)
        media.write_image(os.path.join(IMG_DIR, f"string_align_persp_{tag}.png"), renderer.render())

    # Also render bottom view to PROVE sealed base
    renderer.update_scene(data, cam_bottom)
    media.write_image(os.path.join(IMG_DIR, "sealed_bag_bottom_proof.png"), renderer.render())
    print("\nSaved all renders and bottom seal proof!")

if __name__ == "__main__":
    main()
