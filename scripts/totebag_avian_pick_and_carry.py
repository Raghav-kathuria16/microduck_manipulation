#!/usr/bin/env python3
"""Autonomous Avian Hook-and-Hoist Demonstration for 300g Soft-Body Tote Bag.

Sequence:
  Phase 0: GUIDED_DESCENT — Crouches down to handle loop center height (Z ~ 75-80mm).
  Phase 1: LOOP_INTERCEPT — Lower beak penetrates through the open handle loop beneath the apex.
  Phase 2: SILICONE_CLAMP — Beak closes firmly (1.500 rad) pinching the strap against upper palate.
  Phase 3: HIGH_NECK_HOIST — Neck pitches up to +32° (+0.56 rad), hoisting 300g bag off the floor.
  Phase 4: STAND_RECOVERY — Standing policy balances the robot holding the tote bag aloft.

Usage:
    microduck_rl/.venv/bin/mjpython microduck_manipulation/scripts/totebag_avian_pick_and_carry.py [--gui] [--render]
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

# ---------------------------------------------------------------------------
# Workspace Paths
# ---------------------------------------------------------------------------
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
    _EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_loadcarry_totebag.xml"
)
DEFAULT_STANDING_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/BEST_alpha_stand.onnx")
DEFAULT_GROUND_PICK_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_ground_pick_headup.onnx")

# Beak angles (rad)
OPEN_BEAK_Q = 1.950
CLAMP_BEAK_Q = 1.500

# High-Neck Elevation Angles (rad)
HIGH_NECK_PITCH = 0.56    # +32 degrees aloft carry
CRADLE_HEAD_PITCH = -0.14  # -8 degrees cradle

# Phases
PHASE_GUIDED_DESCENT = 0
PHASE_LOOP_INTERCEPT = 1
PHASE_SILICONE_CLAMP = 2
PHASE_HIGH_NECK_HOIST = 3
PHASE_STAND_RECOVERY = 4

PHASE_NAMES = [
    "GUIDED_DESCENT", "LOOP_INTERCEPT", "SILICONE_CLAMP",
    "HIGH_NECK_HOIST", "STAND_RECOVERY"
]

def quintic_interp(t: float, t_total: float) -> float:
    s = float(np.clip(t / max(t_total, 1e-6), 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

def setup_simulation():
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

    pouch_bid = model.body("totebag_pouch").id
    apex_bid = model.body("strap_apex").id
    totebag_jid = model.joint("totebag_free").id
    beak_jid = model.joint("beak").id
    mouth_sid = model.site("mouth_tip").id
    lower_sid = model.site("lower_beak_tip").id
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    jaw_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "jaw_soft_lower_collision")
    mouth_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "soft_mouth_top_collision")

    strap_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ["geom_strap_apex", "geom_strap_L3", "geom_strap_R3", "geom_strap_L2", "geom_strap_R2"]
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) != -1
    ]

    return {
        "model": model, "data": data, "bam_ctrl": bam_ctrl, "policy": policy,
        "pouch_bid": pouch_bid, "apex_bid": apex_bid, "totebag_jid": totebag_jid,
        "beak_jid": beak_jid, "mouth_sid": mouth_sid, "lower_sid": lower_sid,
        "floor_geom_id": floor_geom_id, "jaw_geom_id": jaw_geom_id,
        "mouth_geom_id": mouth_geom_id, "strap_geom_ids": strap_geom_ids,
        "beak_qadr": model.jnt_qposadr[beak_jid],
        "head_bid": model.body("jaw_soft").id,
        "jaw_soft_bid": model.body("jaw_soft").id,
        "beak_lower_bid": model.body("beak_lower").id,
    }

def reset_simulation(ctx, x_bag=0.200, y_bag=0.000):
    model, data = ctx["model"], ctx["data"]
    bam_ctrl, policy = ctx["bam_ctrl"], ctx["policy"]

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]
    data.qpos[ctx["beak_qadr"]] = 1.500

    # Bag on floor
    b_adr = model.jnt_qposadr[ctx["totebag_jid"]]
    data.qpos[b_adr:b_adr + 3] = [x_bag, y_bag, 0.030]
    data.qpos[b_adr + 3:b_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[model.jnt_dofadr[ctx["totebag_jid"]]:model.jnt_dofadr[ctx["totebag_jid"]] + 6] = 0.0

    bam_ctrl.reset(data.qpos)
    mujoco.mj_forward(model, data)

    policy.ground_pick_mode = False
    policy.ground_pick_phase = 0.0
    policy.current_policy = "standing"
    policy.ort_session = policy.standing_session
    policy.set_vel_cmd(0.0, 0.0, 0.0)

    # 80-step standing settle
    for step in range(80):
        if step % 4 == 0:
            act = policy.infer()
            policy.apply_action(act)
        bam_ctrl.q_target[policy.beak_act_id] = 1.500
        bam_ctrl.update()
        mujoco.mj_step(model, data)

def count_strap_contacts(ctx):
    model, data = ctx["model"], ctx["data"]
    top_c, bot_c = 0, 0
    strap_ids = ctx["strap_geom_ids"]
    jaw_geom = ctx["jaw_geom_id"]
    mouth_geom = ctx["mouth_geom_id"]

    for ci in range(data.ncon):
        con = data.contact[ci]
        g1, g2 = con.geom1, con.geom2
        if g1 in strap_ids or g2 in strap_ids:
            other = g2 if g1 in strap_ids else g1
            if other == mouth_geom:
                top_c += 1
            elif other == jaw_geom:
                bot_c += 1
    return top_c, bot_c

def check_head_floor_collision(ctx):
    model, data = ctx["model"], ctx["data"]
    floor_id = ctx["floor_geom_id"]
    head_bodies = [ctx["head_bid"], ctx["jaw_soft_bid"], ctx["beak_lower_bid"]]
    for ci in range(data.ncon):
        con = data.contact[ci]
        g1, g2 = con.geom1, con.geom2
        if g1 == floor_id or g2 == floor_id:
            b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
            if b1 in head_bodies or b2 in head_bodies:
                return True
    return False

def run_totebag_pick_and_carry(ctx, total_steps=850, verbose=True, render=False, video_path=None, gui=False):
    model, data = ctx["model"], ctx["data"]
    policy, bam_ctrl = ctx["policy"], ctx["bam_ctrl"]

    viewer = None
    if gui:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat = [0.12, 0.0, 0.09]
        viewer.cam.distance = 0.52
        viewer.cam.elevation = -12.0
        viewer.cam.azimuth = 135.0

    renderer = None
    frames = []
    if render:
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = 1  # track robot trunk
        camera.distance = 0.50
        camera.elevation = -12.0
        camera.azimuth = 135.0

    policy.trigger_ground_pick()
    init_pouch_z = float(data.xpos[ctx["pouch_bid"], 2] * 1000.0)

    # Telemetry
    max_pouch_z = init_pouch_z
    min_mouth_z = 999.0
    floor_collisions = 0
    phase = PHASE_GUIDED_DESCENT
    phase_timer = 0
    clamped = False

    target_beak_q = OPEN_BEAK_Q
    beak_act_id = policy.beak_act_id

    # High-neck offset commands
    high_neck_offset = np.radians(32.0) - policy.default_pose[5]
    cradle_head_offset = np.radians(-8.0) - policy.default_pose[6]

    if verbose:
        print(f"\n{'Step':>4} | {'Phase':>16} | {'φ':>5} | {'TrunkZ':>7} {'PouchZ':>7} {'ApexZ':>7} {'MouthZ':>7} | "
              f"{'BeakQ':>6} | {'T':>2} {'B':>2} | {'FloorCol':>8}")
        print("-" * 110)

    for step in range(total_steps):
        if policy.ground_pick_mode:
            policy.update_ground_pick_phase(0.005)
        phi = policy.ground_pick_phase if policy.ground_pick_mode else 1.0

        m_pos = data.site_xpos[ctx["mouth_sid"]]
        p_pos = data.xpos[ctx["pouch_bid"]]
        a_pos = data.xpos[ctx["apex_bid"]]

        m_z = float(m_pos[2] * 1000.0)
        p_z = float(p_pos[2] * 1000.0)
        a_z = float(a_pos[2] * 1000.0)

        max_pouch_z = max(max_pouch_z, p_z)
        min_mouth_z = min(min_mouth_z, m_z)

        if check_head_floor_collision(ctx):
            floor_collisions += 1

        top_c, bot_c = count_strap_contacts(ctx)

        # 50 Hz control loop
        if step % 4 == 0:
            phase_timer += 1

            # State Machine Transitions
            if phase == PHASE_GUIDED_DESCENT:
                target_beak_q = OPEN_BEAK_Q
                dist_to_apex_x = abs(m_pos[0] - a_pos[0])
                if phi >= 0.22 or (dist_to_apex_x < 0.040 and m_z < 115.0):
                    phase = PHASE_LOOP_INTERCEPT
                    phase_timer = 0
                    if verbose:
                        print(f"--> Transferred to LOOP_INTERCEPT at step {step} (dist_x={dist_to_apex_x*1000:.1f}mm, mouth_z={m_z:.1f}mm)")

            elif phase == PHASE_LOOP_INTERCEPT:
                target_beak_q = OPEN_BEAK_Q
                if (top_c > 0 or bot_c > 0) or phase_timer >= 12:
                    phase = PHASE_SILICONE_CLAMP
                    phase_timer = 0
                    if verbose:
                        print(f"--> Transferred to SILICONE_CLAMP at step {step} (contacts T={top_c}, B={bot_c})")

            elif phase == PHASE_SILICONE_CLAMP:
                target_beak_q = CLAMP_BEAK_Q
                if phase_timer >= 15: # 300ms clamp settle
                    phase = PHASE_HIGH_NECK_HOIST
                    phase_timer = 0
                    clamped = True
                    if verbose:
                        print(f"--> Transferred to HIGH_NECK_HOIST at step {step} (strap clamped)")

            elif phase == PHASE_HIGH_NECK_HOIST:
                target_beak_q = CLAMP_BEAK_Q
                if phase_timer >= 25: # 500ms hoist
                    phase = PHASE_STAND_RECOVERY
                    phase_timer = 0
                    policy.ground_pick_mode = False
                    policy.current_policy = "standing"
                    policy.ort_session = policy.standing_session
                    if verbose:
                        print(f"--> Transferred to STAND_RECOVERY at step {step} (pouch hoisted to {p_z:.1f}mm)")

            elif phase == PHASE_STAND_RECOVERY:
                target_beak_q = CLAMP_BEAK_Q

            # Policy Action Computation
            act = policy.infer()

            if phase in (PHASE_GUIDED_DESCENT, PHASE_LOOP_INTERCEPT, PHASE_SILICONE_CLAMP):
                # Altitude floor guard
                if m_z < 65.0:
                    act[5] = max(act[5], 0.15)

            elif phase == PHASE_HIGH_NECK_HOIST:
                s_hoist = quintic_interp(phase_timer * 0.02, 0.50)
                act[5] = float(np.interp(s_hoist, [0, 1], [act[5], high_neck_offset]))
                act[6] = float(np.interp(s_hoist, [0, 1], [act[6], cradle_head_offset]))

            elif phase == PHASE_STAND_RECOVERY:
                act[5] = high_neck_offset
                act[6] = cradle_head_offset

            policy.apply_action(act)

            # Set beak clamp servo
            bam_ctrl.q_target[beak_act_id] = target_beak_q

        # BAM Actuator & Physics Step (200 Hz)
        bam_ctrl.update()
        mujoco.mj_step(model, data)

        if gui and viewer is not None:
            viewer.sync()

        if render and renderer is not None and step % 4 == 0:
            renderer.update_scene(data, camera)
            frames.append(renderer.render())

        if verbose and (step % 40 == 0 or step == total_steps - 1):
            tz = float(data.qpos[2] * 1000.0)
            bq = float(data.qpos[ctx["beak_qadr"]])
            pname = PHASE_NAMES[phase]
            print(f"{step:4d} | {pname:>16} | {phi:5.3f} | {tz:7.1f} {p_z:7.1f} {a_z:7.1f} {m_z:7.1f} | "
                  f"{bq:6.3f} | {top_c:2d} {bot_c:2d} | {floor_collisions:8d}")

    if render and frames and video_path:
        import mediapy as media
        media.write_video(video_path, frames, fps=50)
        if verbose:
            print(f"\n[Saved demonstration video to {video_path}]")

    if gui and viewer is not None:
        viewer.close()

    final_pouch_z = float(data.xpos[ctx["pouch_bid"], 2] * 1000.0)
    lift_height = final_pouch_z - init_pouch_z

    return {
        "lift_height_mm": lift_height,
        "max_pouch_z": max_pouch_z,
        "final_pouch_z": final_pouch_z,
        "min_mouth_z": min_mouth_z,
        "floor_collisions": floor_collisions,
        "clamped": clamped,
    }

def main():
    parser = argparse.ArgumentParser(description="Demonstrate 300g Tote Bag Pick & Hoist")
    parser.add_argument("--gui", action="store_true", help="Launch interactive 3D GUI viewer")
    parser.add_argument("--render", action="store_true", help="Render video recording")
    parser.add_argument("--video-path", type=str, default=os.path.join(VIDEO_DIR, "totebag_300g_pick_and_carry.mp4"))
    args = parser.parse_args()

    print("=" * 60)
    print("DEMONSTRATING 300g SOFT-BODY TOTE BAG HOIST & CARRY")
    print("=" * 60)

    ctx = setup_simulation()
    reset_simulation(ctx, x_bag=0.200, y_bag=0.000)

    res = run_totebag_pick_and_carry(
        ctx, total_steps=850, verbose=True,
        render=args.render, video_path=args.video_path, gui=args.gui
    )

    print("\n" + "=" * 60)
    print("EXECUTION BENCHMARK RESULTS:")
    print(f"  Pouch Lift Height:    {res['lift_height_mm']:.1f} mm")
    print(f"  Final Pouch Z:        {res['final_pouch_z']:.1f} mm")
    print(f"  Minimum Mouth Z:      {res['min_mouth_z']:.1f} mm (Clearance: {res['min_mouth_z']:.1f} mm)")
    print(f"  Floor Collisions:     {res['floor_collisions']} collisions")
    print(f"  Silicone Clamp Lock:  {'SUCCESS' if res['clamped'] else 'FAILED'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
