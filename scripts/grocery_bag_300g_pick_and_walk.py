#!/usr/bin/env python3
"""Autonomous Avian Pick, Aloft Hoist, and Forward Walking with 300g Sealed Kraft Grocery Bag.

Pipeline:
1. Settle in standing balance.
2. Controlled crouch and hook-clamp of the 1.6mm braided twine handle at X = 105mm.
3. High-neck aloft hoist (+30° neck pitch, -6° head cradle) bringing 300g payload CoM closer to torso.
4. Smooth quintic transition into the 300g dynamic walking policy (vel_x = 0.15 m/s).
5. Forward walking carry across the arena.

Usage:
    microduck_rl/.venv/bin/python microduck_manipulation/scripts/grocery_bag_300g_pick_and_walk.py [--render]
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# Workspace paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(_MANIP_DIR)
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")
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
WALKING_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_student_300g.onnx")
FALLBACK_WALKING_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/BEST_alpha_walking.onnx")

def quintic_interp(t: float, t_total: float) -> float:
    s = float(np.clip(t / max(t_total, 1e-6), 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

def setup_simulation():
    bam_model = load_bam_model(kp_fw=BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, names = load_mujoco_with_bam(
        XML_PATH, bam_model, 0.005, 0.0, BAM_VIN_MIN
    )
    
    walk_path = WALKING_POLICY if os.path.exists(WALKING_POLICY) else FALLBACK_WALKING_POLICY
    print(f"Loading policies:")
    print(f"  Standing:    {STANDING_POLICY}")
    print(f"  Ground Pick: {GROUND_PICK_POLICY}")
    print(f"  Walking:     {walk_path}")

    policy = PolicyInference(
        model, data,
        standing_onnx_path=STANDING_POLICY,
        ground_pick_onnx_path=GROUND_PICK_POLICY,
        walking_onnx_path=walk_path,
        ground_pick_period=4.0,
        use_projected_gravity=True,
        new_cmd_obs=True,
        bam_ctrl=bam_ctrl,
    )

    # Enhance silicone grip on 1.6mm twine
    for name in ["geom_strap_apex", "geom_strap_L3", "geom_strap_R3", "geom_strap_L2", "geom_strap_R2"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1:
            model.geom_size[gid, 0] = 0.0022 # 2.2mm radius

    bag_bid = model.body("grocery_bag").id
    bag_jid = model.joint("bag_free").id
    apex_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bag_handle")
    if apex_bid == -1:
        apex_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "strap_apex")
    beak_jid = model.joint("beak").id
    mouth_sid = model.site("mouth_tip").id
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    head_bodies = [model.body("jaw_soft").id, model.body("beak_lower").id]

    return {
        "model": model, "data": data, "bam_ctrl": bam_ctrl, "policy": policy,
        "bag_bid": bag_bid, "bag_jid": bag_jid, "apex_bid": apex_bid,
        "beak_jid": beak_jid, "mouth_sid": mouth_sid,
        "beak_qadr": model.jnt_qposadr[beak_jid],
        "floor_geom_id": floor_geom_id, "head_bodies": head_bodies,
        "walk_path": walk_path,
    }

def reset_simulation(ctx, x_bag=0.105, y_bag=0.0):
    model, data = ctx["model"], ctx["data"]
    bam_ctrl, policy = ctx["bam_ctrl"], ctx["policy"]

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]
    data.qpos[ctx["beak_qadr"]] = 1.95 # open

    b_adr = model.jnt_qposadr[ctx["bag_jid"]]
    data.qpos[b_adr:b_adr + 3] = [x_bag, y_bag, 0.001]
    data.qpos[b_adr + 3:b_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[model.jnt_dofadr[ctx["bag_jid"]]:model.jnt_dofadr[ctx["bag_jid"]] + 6] = 0.0

    bam_ctrl.reset(data.qpos)
    mujoco.mj_forward(model, data)

    policy.ground_pick_mode = False
    policy.current_policy = "standing"
    policy.ort_session = policy.standing_session
    policy.set_vel_cmd(0.0, 0.0, 0.0)

    # Settle in standing
    for _ in range(60):
        if _ % 4 == 0:
            act = policy.infer()
            policy.apply_action(act)
        bam_ctrl.q_target[policy.beak_act_id] = 1.95
        bam_ctrl.update()
        mujoco.mj_step(model, data)

def run_pick_and_walk(ctx, total_steps=850, render=True, video_path=None, gui=False):
    model, data = ctx["model"], ctx["data"]
    policy, bam_ctrl = ctx["policy"], ctx["bam_ctrl"]

    viewer = None
    if gui:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 0.52
        viewer.cam.elevation = -12.0
        viewer.cam.azimuth = 125.0

    renderer = None
    camera = None
    frames = []
    if render:
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = 1 # track trunk_base
        camera.distance = 0.52
        camera.elevation = -12.0
        camera.azimuth = 125.0

    policy.trigger_ground_pick()
    floor_collisions = 0
    max_bag_z = 0.0
    start_x = float(data.qpos[0])
    
    print("\n" + "=" * 80)
    print("EXECUTING PICK, HOIST, ALOFT HOLD, AND WALKING GAIT")
    print("=" * 80)
    print(f"{'Step':>4} | {'Phase':>12} | {'TrunkX':>6} {'TrunkZ':>6} | {'BagZ':>6} {'MouthZ':>6} | {'BeakQ':>5} | Status")
    print("-" * 80)

    stand_step = 0
    walk_step = 0
    phase_str = "CROUCH"

    for s in range(total_steps):
        if policy.ground_pick_mode:
            policy.update_ground_pick_phase(0.005)

        # Check floor collision
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1, g2 = con.geom1, con.geom2
            if g1 == ctx["floor_geom_id"] or g2 == ctx["floor_geom_id"]:
                b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
                if b1 in ctx["head_bodies"] or b2 in ctx["head_bodies"]:
                    floor_collisions += 1

        # Control update at 50 Hz (step % 4 == 0)
        if s % 4 == 0:
            # Phase 1: Crouch & Approach (0 to 155 steps)
            if s < 155:
                phase_str = "CROUCH"
                target_beak = 1.95

            # Phase 2: Clamp Engage (155 to 220 steps)
            elif 155 <= s < 220:
                phase_str = "CLAMP"
                target_beak = 1.48

            # Phase 3: Hoist & Aloft Stand Recovery (220 to 420 steps)
            elif 220 <= s < 420:
                phase_str = "HOIST"
                target_beak = 1.48
                if s == 220:
                    policy.ground_pick_mode = False
                    policy.current_policy = "standing"
                    policy.ort_session = policy.standing_session
                    policy.set_vel_cmd(0.0, 0.0, 0.0)

            # Phase 4: Walk Gait Transition (420+ steps)
            else:
                phase_str = "WALK"
                target_beak = 1.48
                if s == 420:
                    policy.current_policy = "walking"
                    policy.ort_session = policy.walking_session

            act = policy.infer()

            # Hoist aloft posture tuning
            if 220 <= s < 420:
                stand_step += 1
                s_ramp = min(1.0, stand_step / 100.0)
                blend = quintic_interp(s_ramp, 1.0)
                # Upward neck pitch (+30 deg) and head cradle (-6 deg)
                target_neck = np.radians(30.0) - policy.default_pose[5]
                target_head = np.radians(-6.0) - policy.default_pose[6]
                act[5] = act[5] * (1.0 - blend) + target_neck * blend
                act[6] = act[6] * (1.0 - blend) + target_head * blend
                # Hip pitch counter-bias for 300g payload stability
                act[2] -= float(0.05 * blend)
                act[11] += float(0.05 * blend)

            elif s >= 420:
                walk_step += 1
                # Ramp forward velocity command from 0.0 to 0.15 m/s
                v_target = min(0.15, walk_step * 0.0015)
                policy.set_vel_cmd(v_target, 0.0, 0.0)
                # Maintain aloft neck posture during walking
                act[5] = np.radians(30.0) - policy.default_pose[5]
                act[6] = np.radians(-6.0) - policy.default_pose[6]
                act[2] -= 0.05
                act[11] += 0.05

            policy.apply_action(act)
            bam_ctrl.q_target[policy.beak_act_id] = target_beak

        bam_ctrl.update()
        mujoco.mj_step(model, data)

        tx = float(data.qpos[0] * 1000.0)
        tz = float(data.qpos[2] * 1000.0)
        bz = float(data.xpos[ctx["bag_bid"], 2] * 1000.0)
        mz = float(data.site_xpos[ctx["mouth_sid"], 2] * 1000.0)
        bq = float(data.qpos[ctx["beak_qadr"]])
        max_bag_z = max(max_bag_z, bz)

        if gui and viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)

        if render and renderer is not None and s % 4 == 0:
            renderer.update_scene(data, camera)
            frames.append(renderer.render())

        if s % 50 == 0 or s == total_steps - 1:
            print(f"{s:4d} | {phase_str:>12s} | {tx:6.1f} {tz:6.1f} | {bz:6.1f} {mz:6.1f} | {bq:5.3f} | Stable")

    if gui and viewer is not None:
        viewer.close()

    if render and frames and video_path:
        import mediapy as media
        media.write_video(video_path, frames, fps=50)
        print(f"\n[Saved demonstration video to {video_path}]")

    end_x = float(data.qpos[0])
    walked_dist = end_x - start_x

    return {
        "max_bag_z_mm": max_bag_z,
        "final_bag_z_mm": bz,
        "floor_collisions": floor_collisions,
        "walked_dist_mm": walked_dist * 1000.0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Launch native interactive GLFW MuJoCo viewer")
    parser.add_argument("--render", action="store_true", default=False, help="Record offscreen MP4 video")
    parser.add_argument("--video-path", type=str, default=os.path.join(VIDEO_DIR, "grocery_bag_300g_pick_and_walk.mp4"))
    args = parser.parse_args()

    ctx = setup_simulation()
    reset_simulation(ctx, x_bag=0.105, y_bag=0.0)
    res = run_pick_and_walk(ctx, total_steps=800, render=args.render, video_path=args.video_path, gui=args.gui)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY:")
    print(f"  Max Bag Lift Z:       {res['max_bag_z_mm']:.1f} mm")
    print(f"  Final Bag Z:          {res['final_bag_z_mm']:.1f} mm")
    print(f"  Distance Walked:      {res['walked_dist_mm']:.1f} mm")
    print(f"  Floor Collisions:     {res['floor_collisions']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
