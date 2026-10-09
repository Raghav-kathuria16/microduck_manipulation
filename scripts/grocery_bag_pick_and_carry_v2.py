#!/usr/bin/env python3
"""Execute pick and hoist from the user-specified alignment (x_bag = 105mm).

Tests:
1. Descent to alignment where string is just above the top jaw.
2. Beak closure / hook engagement.
3. Smooth RL ground pick ascension and high-neck aloft hoist (+32°).
4. Records demonstration video to videos/grocery_bag_sealed_pick_and_carry.mp4.
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
VIDEO_DIR = os.path.join(ARTIFACT_DIR, "videos")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

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
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = [0.10, 0.0, 0.08]
    camera.distance = 0.38
    camera.elevation = -10.0
    camera.azimuth = 120.0

    bag_bid = model.body("grocery_bag").id
    bag_jid = model.joint("bag_free").id
    apex_bid = model.body("strap_apex").id
    beak_qadr = model.jnt_qposadr[model.joint("beak").id]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    head_bodies = [model.body("jaw_soft").id, model.body("beak_lower").id]

    x_bag = 0.105 # User aligned position

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

    # Settle 60 steps
    for _ in range(60):
        if _ % 4 == 0:
            act = policy.infer()
            policy.apply_action(act)
        bam_ctrl.q_target[policy.beak_act_id] = 1.95
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    policy.trigger_ground_pick()
    frames = []
    floor_collisions = 0
    max_bag_z = 0.0

    print("Step | Phase | phi   | TrunkZ  BagZ   ApexZ  MouthZ | BeakQ | FloorCol")
    print("-" * 75)

    for s in range(500):
        policy.update_ground_pick_phase(0.005)
        phi = policy.ground_pick_phase

        for ci in range(data.ncon):
            con = data.contact[ci]
            g1, g2 = con.geom1, con.geom2
            if g1 == floor_id or g2 == floor_id:
                b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
                if b1 in head_bodies or b2 in head_bodies:
                    floor_collisions += 1

        if s % 4 == 0:
            act = policy.infer()
            # Clamp when reached string alignment (step 150, phi ~ 0.19)
            if s >= 150:
                target_beak = 1.48 # firm clamp
                # Upward neck bias during ascension (steps 200+)
                if s >= 200:
                    blend = min(1.0, (s - 200) / 120.0)
                    act[5] += float(0.45 * blend) # Aloft carry (+32 deg)
                    act[6] -= float(0.18 * blend) # Cradle head
            else:
                target_beak = 1.95
                
            policy.apply_action(act)
            bam_ctrl.q_target[policy.beak_act_id] = target_beak

        bam_ctrl.update()
        mujoco.mj_step(model, data)

        tz = data.qpos[2] * 1000.0
        bz = data.xpos[bag_bid, 2] * 1000.0
        az = data.xpos[apex_bid, 2] * 1000.0
        mz = data.site_xpos[model.site("mouth_tip").id, 2] * 1000.0
        bq = data.qpos[beak_qadr]
        max_bag_z = max(max_bag_z, bz)

        if s % 4 == 0:
            renderer.update_scene(data, camera)
            frames.append(renderer.render())

        if s % 25 == 0:
            print(f"{s:4d} | GP    | {phi:5.3f} | {tz:6.1f} {bz:6.1f} {az:6.1f} {mz:6.1f} | {bq:5.3f} | {floor_collisions:8d}")

    video_path = os.path.join(VIDEO_DIR, "grocery_bag_sealed_pick_and_carry.mp4")
    media.write_video(video_path, frames, fps=50)
    print(f"\nSaved video to {video_path}")
    print(f"Max bag Z: {max_bag_z:.1f}mm, Floor collisions: {floor_collisions}")

if __name__ == "__main__":
    main()
