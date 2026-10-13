#!/usr/bin/env python3
"""Dedicated Lemon-Spoon Friction Verification & Benchmark Suite.

This script tests Coulomb friction between the lemon and spoon in two ways:
1. Tilt Test (--tilt or 'T' key in GUI):
   Gradually tilts the spoon forward at 1.0 deg/s to measure the exact
   critical angle where static friction breaks:
     theoretical_slip_angle = arctan(mu)
   Proves whether mu=0.10 slips at ~5.7 deg, mu=0.20 at ~11.3 deg, mu=0.50 at ~26.6 deg.

2. Automated Sweep Benchmark (--sweep):
   Runs across multiple friction coefficients [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
   and reports theoretical vs measured physical slip angles and stability.

3. Interactive 3D Viewer (--gui):
   Watch the singular robot in real time. Use W/S to manually tilt the spoon,
   press T to trigger an automated friction slip test, or press [ / ] to adjust mu live.
"""

import argparse
import math
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from infer_policy import DEFAULT_POSE
from test_spoon_lemon_balance import create_spoon_lemon_scene_xml
from test_interactive_actuator_lemon_drop import build_standing_scene_xml, setup_standing_robot


def run_friction_tilt_eval(mu: float, verbose: bool = True):
    """Gradually tilts spoon forward to measure the exact slip angle for friction mu."""
    mu_clamped = max(0.001, float(mu))
    xml_str = build_standing_scene_xml()
    # Replace friction values in XML
    import re
    xml_str = re.sub(r'friction="[0-9.]+ 0.01 0.01"', f'friction="{mu_clamped:.3f} 0.01 0.01"', xml_str)
    # Contact pairs
    xml_str = re.sub(r'geom1="lemon_geom" geom2="[^"]+" condim="6" friction="[0-9.]+ [0-9.]+',
                     f'geom1="lemon_geom" geom2="spoon_bowl_base" condim="6" friction="{mu_clamped:.3f} {mu_clamped:.3f}', xml_str)
    
    # Simpler: use create_spoon_lemon_scene_xml with custom mu
    xml_str = create_spoon_lemon_scene_xml(friction=mu_clamped)
    xml_str = xml_str.replace('gainprm="0.55" biasprm="0 -0.55"', 'gainprm="25.0" biasprm="0 -25.0"')
    xml_str = xml_str.replace('forcerange="-0.96 0.96"', 'forcerange="-5.0 5.0"')

    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    default_pose = DEFAULT_POSE[:14].copy()
    default_pose[5] = 0.2760
    default_pose[6] = 0.2760
    default_pose[7] = 0.0
    default_pose[8] = 0.0

    joint_qpos_indices = [int(model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in range(14)]
    beak_qpos = int(model.jnt_qposadr[model.actuator_trnid[14, 0]])
    beak_dof = int(model.jnt_dofadr[model.actuator_trnid[14, 0]])

    spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lemon")
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    spoon_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spoon_free")]
    lemon_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lemon_free")]

    setup_standing_robot(model, data, default_pose, joint_qpos_indices, beak_qpos, beak_dof,
                         spoon_bid, lemon_bid, jaw_bid, spoon_qadr, lemon_qadr)

    theo_angle_deg = math.degrees(math.atan(mu_clamped))
    
    # Initial bowl center
    p_s = data.xpos[spoon_bid]
    R_s = data.xmat[spoon_bid].reshape(3, 3)
    init_bowl = p_s + R_s @ np.array([0.0, 0.0052, 0.070])
    
    tilt_rate_deg_per_s = 2.0  # Slow, gentle tilt to isolate static friction
    tilt_rad_per_step = math.radians(tilt_rate_deg_per_s) * model.opt.timestep

    pitch_offset = 0.0
    slip_angle_deg = None
    drop_angle_deg = None

    for step in range(15000):  # Up to 30.0 seconds / 60 degrees tilt
        pitch_offset += tilt_rad_per_step
        # Tilt neck and head equally
        data.ctrl[5] = default_pose[5] - pitch_offset
        data.ctrl[6] = default_pose[6] - pitch_offset
        for i in range(14):
            if i not in (5, 6): data.ctrl[i] = default_pose[i]
        data.ctrl[14] = 1.550
        data.qfrc_applied[beak_dof] = -0.45

        mujoco.mj_step(model, data)

        # Measure current spoon pitch angle relative to world horizontal
        R_s = data.xmat[spoon_bid].reshape(3, 3)
        b_norm = R_s @ np.array([0.0, 1.0, 0.0])
        current_pitch_deg = math.degrees(math.asin(np.clip(-b_norm[0], -0.999, 0.999)))

        p_s = data.xpos[spoon_bid]
        bowl = p_s + R_s @ np.array([0.0, 0.0052, 0.070])
        disp_mm = float(np.linalg.norm(data.xpos[lemon_bid] - bowl)) * 1000.0

        if disp_mm > 5.0 and slip_angle_deg is None:
            slip_angle_deg = current_pitch_deg

        if disp_mm > 45.0:
            drop_angle_deg = current_pitch_deg
            break

    if verbose:
        print(f"Friction mu = {mu_clamped:5.3f} | Theoretical: {theo_angle_deg:5.1f} deg | "
              f"Physical Slip: {slip_angle_deg if slip_angle_deg else 0.0:5.1f} deg | "
              f"Fell off Rim: {drop_angle_deg if drop_angle_deg else 0.0:5.1f} deg")

    return theo_angle_deg, slip_angle_deg, drop_angle_deg


def run_sweep():
    """Runs automated sweep across 7 friction coefficients."""
    test_mus = [0.02, 0.05, 0.10, 0.15, 0.20, 0.35, 0.50]
    print("\n" + "=" * 80)
    print("COULOMB FRICTION BENCHMARK: LEMON ON STAINLESS STEEL SPOON")
    print("Formula: Critical Static Slip Angle = arctan(mu)")
    print("=" * 80)
    print(f"{'Friction (mu)':^15} | {'Theoretical Angle':^20} | {'Physical Slip Angle':^20} | {'Drop Off Rim':^15}")
    print("-" * 80)

    for mu in test_mus:
        theo, slip, drop = run_friction_tilt_eval(mu, verbose=False)
        slip_str = f"{slip:5.1f} deg" if slip is not None else "> 60 deg"
        drop_str = f"{drop:5.1f} deg" if drop is not None else "> 60 deg"
        print(f"{mu:^15.2f} | {theo:^18.1f} deg | {slip_str:^20} | {drop_str:^15}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="MicroDuck Lemon-Spoon Friction Benchmark Suite")
    parser.add_argument("--sweep", action="store_true", help="Run automated sweep benchmark across multiple frictions")
    parser.add_argument("--tilt", action="store_true", help="Run quantitative tilt slip test for specified --friction")
    parser.add_argument("--gui", action="store_true", help="Launch interactive 3D visual viewer")
    parser.add_argument("--friction", "--mu", type=float, default=0.20, help="Coulomb friction coefficient between lemon and spoon (default: 0.20)")
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
        return

    if args.tilt:
        run_friction_tilt_eval(args.friction, verbose=True)
        return

    if not args.gui:
        # Default: run sweep
        run_sweep()
        return

    # Interactive GUI Viewer Mode
    mu_clamped = max(0.001, float(args.friction))
    xml_str = create_spoon_lemon_scene_xml(friction=mu_clamped)
    xml_str = xml_str.replace('gainprm="0.55" biasprm="0 -0.55"', 'gainprm="25.0" biasprm="0 -25.0"')
    xml_str = xml_str.replace('forcerange="-0.96 0.96"', 'forcerange="-5.0 5.0"')

    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    default_pose = DEFAULT_POSE[:14].copy()
    default_pose[5] = 0.2760
    default_pose[6] = 0.2760
    default_pose[7] = 0.0
    default_pose[8] = 0.0

    joint_qpos_indices = [int(model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in range(14)]
    beak_qpos = int(model.jnt_qposadr[model.actuator_trnid[14, 0]])
    beak_dof = int(model.jnt_dofadr[model.actuator_trnid[14, 0]])

    spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lemon")
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    spoon_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spoon_free")]
    lemon_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lemon_free")]
    lemon_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lemon_geom")

    setup_standing_robot(model, data, default_pose, joint_qpos_indices, beak_qpos, beak_dof,
                         spoon_bid, lemon_bid, jaw_bid, spoon_qadr, lemon_qadr)

    current_mu = mu_clamped
    auto_tilt = False
    auto_tilt_angle = 0.0
    tilt_delta = 0.0
    user_paused = False
    reset_requested = False
    rest_local = np.array([0.0, 0.00505, 0.0712])

    def update_friction(new_mu):
        nonlocal current_mu
        current_mu = max(0.001, float(new_mu))
        crit = math.degrees(math.atan(current_mu))
        for p in range(model.npair):
            if model.pair_geom1[p] == lemon_gid or model.pair_geom2[p] == lemon_gid:
                model.pair_friction[p, 0] = current_mu
                model.pair_friction[p, 1] = current_mu
        if lemon_gid != -1:
            model.geom_friction[lemon_gid, 0] = current_mu
        print(f"\n[FRICTION] Set lemon-spoon mu = {current_mu:.2f} (Theoretical Slip Angle = {crit:.1f} deg)")

    def on_key(keycode):
        nonlocal auto_tilt, auto_tilt_angle, tilt_delta, user_paused, reset_requested, current_mu
        if keycode == 32:  # Space
            user_paused = not user_paused
        elif keycode in (ord('R'), ord('r')):
            reset_requested = True
        elif keycode in (ord('T'), ord('t')):  # T: Trigger automated tilt slip test
            auto_tilt = True
            auto_tilt_angle = 0.0
            crit = math.degrees(math.atan(current_mu))
            print(f"\n[TEST] Started Automated Tilt Test: Slowly tilting spoon forward to test mu = {current_mu:.2f} (Expected slip: {crit:.1f} deg)...")
        elif keycode in (ord('['), 91):  # [: Decrease friction
            update_friction(current_mu - 0.05)
        elif keycode in (ord(']'), 93):  # ]: Increase friction
            update_friction(current_mu + 0.05)
        elif keycode in (ord('W'), ord('w'), 265):  # W / Up: Tilt spoon pitch forward (down)
            auto_tilt = False
            tilt_delta = max(-0.35, tilt_delta - 0.012)
            data.ctrl[5] = default_pose[5] + tilt_delta
            data.ctrl[6] = default_pose[6] - tilt_delta
            print(f"\n[MANUAL TILT] Tilted forward -> delta = {tilt_delta:+.3f} rad")
        elif keycode in (ord('S'), ord('s'), 264):  # S / Down: Tilt spoon pitch backward (up)
            auto_tilt = False
            tilt_delta = min(0.20, tilt_delta + 0.012)
            data.ctrl[5] = default_pose[5] + tilt_delta
            data.ctrl[6] = default_pose[6] - tilt_delta
            print(f"\n[MANUAL TILT] Tilted backward -> delta = {tilt_delta:+.3f} rad")

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 1
    viewer.cam.distance = 0.55
    viewer.cam.elevation = -10.0
    viewer.cam.azimuth = 100.0

    print("\n" + "=" * 70)
    print("LEMON-SPOON FRICTION TESTBENCH READY (STAND & TILT)")
    print("=" * 70)
    print(f"  Current Lemon-Spoon Friction: mu = {current_mu:.2f}")
    print(f"  Theoretical Critical Slip Angle: {math.degrees(math.atan(current_mu)):.1f} deg")
    print("  Controls:")
    print("    W / Up:    Tilt spoon forward (downwards slope)")
    print("    S / Down:  Tilt spoon backward (upwards slope)")
    print("    T:         Trigger automated slow tilt test (auto-tilts until slip)")
    print("    [ / ]:     Decrease / Increase friction live by 0.05")
    print("    R:         Reset lemon back into spoon")
    print("    Space:     Pause / Resume simulation")
    print("=" * 70 + "\n")

    step = 0
    slip_detected = False
    while viewer.is_running():
        t_start = time.time()

        if reset_requested:
            reset_requested = False
            auto_tilt = False
            auto_tilt_angle = 0.0
            tilt_delta = 0.0
            slip_detected = False
            setup_standing_robot(model, data, default_pose, joint_qpos_indices, beak_qpos, beak_dof,
                                 spoon_bid, lemon_bid, jaw_bid, spoon_qadr, lemon_qadr)

        if user_paused:
            viewer.sync()
            time.sleep(0.02)
            continue

        if auto_tilt:
            auto_tilt_angle += 0.06 * model.opt.timestep
            tilt_delta = -auto_tilt_angle
            data.ctrl[5] = default_pose[5] + tilt_delta
            data.ctrl[6] = default_pose[6] - tilt_delta

        data.ctrl[14] = 1.550
        data.qfrc_applied[beak_dof] = -0.45

        mujoco.mj_step(model, data)
        step += 1

        # Spoon orientation & lemon distance
        R_s = data.xmat[spoon_bid].reshape(3, 3)
        fwd_vec = R_s[:, 2]
        spoon_pitch_deg = math.degrees(math.atan2(-fwd_vec[2], fwd_vec[0]))

        p_s = data.xpos[spoon_bid]
        current_local = R_s.T @ (data.xpos[lemon_bid] - p_s)
        disp_mm = float(np.linalg.norm(current_local - rest_local)) * 1000.0

        if disp_mm < 3.0:
            status = "SEATED [STABLE]"
        elif disp_mm < 15.0:
            status = "SLIDING [SLIP DETECTED]"
            if not slip_detected:
                slip_detected = True
                crit = math.degrees(math.atan(current_mu))
                print(f"\n>>> SLIP OCCURRED! Measured Spoon Pitch = {spoon_pitch_deg:+.1f} deg (Theoretical = {crit:.1f} deg) <<<")
        elif disp_mm < 35.0:
            status = "ON RIM [NEAR FALL]"
        else:
            status = "DROPPED [FALLEN TO FLOOR]"

        if step % 50 == 0:
            crit = math.degrees(math.atan(current_mu))
            print(f"\r[mu={current_mu:4.2f} (crit={crit:4.1f}°) | Spoon Pitch={spoon_pitch_deg:+5.1f}° | Dev={disp_mm:4.1f}mm | {status:25s}]", end="", flush=True)

        viewer.sync()
        t_elapsed = time.time() - t_start
        dt_sleep = model.opt.timestep - t_elapsed
        if dt_sleep > 0:
            time.sleep(dt_sleep)

    viewer.close()


if __name__ == "__main__":
    main()
