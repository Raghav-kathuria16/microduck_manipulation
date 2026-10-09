#!/usr/bin/env python3
"""Demonstration of Natural Pick and Walk with 300g Sealed Kraft Grocery Bag (Zero Hoist).

Follows user directive:
- NO artificial hoist, NO crazy head twisting.
- Natural standing recovery and smooth transition to walking forward.
- Bag carried naturally at natural mouth height.
- Records video to videos/grocery_bag_natural_pick_and_walk.mp4.
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
VIDEO_DIR = os.path.join(ARTIFACT_DIR, "videos")
IMG_DIR = os.path.join(ARTIFACT_DIR, "images")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

XML_PATH = os.path.join(
    _EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_grocery_bag.xml"
)
STANDING_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/BEST_alpha_stand.onnx")
GROUND_PICK_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_ground_pick_headup.onnx")
WALKING_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/BEST_alpha_walking.onnx")

def main():
    bam_model = load_bam_model(kp_fw=BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, names = load_mujoco_with_bam(
        XML_PATH, bam_model, 0.005, 0.0, BAM_VIN_MIN
    )

    policy = PolicyInference(
        model, data,
        standing_onnx_path=STANDING_POLICY,
        ground_pick_onnx_path=GROUND_PICK_POLICY,
        walking_onnx_path=WALKING_POLICY,
        ground_pick_period=4.0,
        use_projected_gravity=True,
        new_cmd_obs=True,
        bam_ctrl=bam_ctrl,
    )

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 1 # trunk_base
    camera.distance = 0.48
    camera.elevation = -10.0
    camera.azimuth = 120.0

    bag_bid = model.body("grocery_bag").id
    bag_jid = model.joint("bag_free").id
    beak_qadr = model.jnt_qposadr[model.joint("beak").id]

    x_bag = 0.105 # Aligned position

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

    # 60-step standing settle
    for _ in range(60):
        if _ % 4 == 0:
            act = policy.infer()
            policy.apply_action(act)
        bam_ctrl.q_target[policy.beak_act_id] = 1.95
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    policy.trigger_ground_pick()
    frames = []
    walk_ticks = 0
    total_steps = 700

    print("Running natural pick and walk simulation (700 steps)...")

    for s in range(total_steps):
        if policy.ground_pick_mode:
            policy.update_ground_pick_phase(0.005)

        if s % 4 == 0:
            # Stand recovery at step 200
            if s == 200:
                policy.ground_pick_mode = False
                policy.current_policy = "standing"
                policy.ort_session = policy.standing_session
                policy.set_vel_cmd(0.0, 0.0, 0.0)

            # Natural walk transition at step 420
            if s >= 420:
                walk_ticks += 1
                vx = min(0.12, walk_ticks * 0.002)
                policy.current_policy = "walking"
                policy.ort_session = policy.walking_session
                policy.set_vel_cmd(vx, 0.0, 0.0)

            act = policy.infer()

            # Beak clamping
            if s >= 155:
                target_beak = 1.48 # firm clamp
            else:
                target_beak = 1.95

            # Pure natural policy - NO artificial hoist
            policy.apply_action(act)
            bam_ctrl.q_target[policy.beak_act_id] = target_beak

        bam_ctrl.update()
        mujoco.mj_step(model, data)

        if s % 4 == 0:
            renderer.update_scene(data, camera)
            frames.append(renderer.render())

    video_path = os.path.join(VIDEO_DIR, "grocery_bag_natural_pick_and_walk.mp4")
    media.write_video(video_path, frames, fps=50)
    print(f"\nSaved natural pick and walk video to: {video_path}")

    # Extract 4 keyframes
    n_f = len(frames)
    k_indices = [int(n_f * 0.15), int(n_f * 0.35), int(n_f * 0.60), int(n_f * 0.85)]
    k_names = [
        "natural_walk_1_approach.png",
        "natural_walk_2_clamp_stand.png",
        "natural_walk_3_carry_stand.png",
        "natural_walk_4_forward_stride.png",
    ]
    for idx, name in zip(k_indices, k_names):
        media.write_image(os.path.join(IMG_DIR, name), frames[idx])
        print(f"Extracted keyframe: {name}")

    final_x = float(data.qpos[0] * 1000.0)
    final_bag_z = float(data.xpos[bag_bid, 2] * 1000.0)
    print(f"\nFinal Trunk X: {final_x:.1f} mm | Final Bag Z: {final_bag_z:.1f} mm")

if __name__ == "__main__":
    main()
