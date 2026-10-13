#!/usr/bin/env python3
"""Interactive Actuator Control & Lemon Drop Simulation for MicroDuck.

Features:
1. Robot stands stably and firmly upright in calibrated standing equilibrium.
2. Spoon is physically clamped in the lower jaw and upper beak.
3. Lemon rests in the concave spoon bowl with Coulomb friction (mu = 0.50, condim = 6).
4. Direct interactive keyboard control over robot actuators:
   - Pitch: W / S (or Up / Down) tilts neck and head forward / backward.
   - Roll:  A / D (or Left / Right) tilts head roll left / right.
   - Yaw:   Q / E turns head yaw left / right.
   - Beak:  O / C opens / closes beak aperture.
   - Torque: + / - increases / decreases beak clamping torque.
   - 1: Neutral level head pose reset.
   - 2: Automatic Pitch Drop test (tilts past 26.6° static friction limit).
   - 3: Automatic Roll Drop test (rolls sideways over bowl rim).
   - 4: Automatic Beak Release test (loosens jaw until spoon falls).
   - R: Reset simulation to standing state with lemon in spoon.
   - Space: Pause / Resume physics.
   - Tab: Switch camera (Close-up -> Side Profile -> Isometric -> Front).
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


STANDING_KEY_QPOS = "0.000000 -0.000000 0.124959 1.000000 0.000001 0.000013 -0.000008 0.000065 -0.087201 -0.458087 -0.004760 0.452976 0.275303 0.276662 -0.001086 -0.000020 1.561717 -0.000190 0.087218 0.458244 0.004704 -0.452974 0.042017 -0.000045 0.228806 0.492879 0.507634 0.506983 0.492287 0.113564 0.000311 0.233415 0.548665 0.457111 0.443115 0.541909"
STANDING_KEY_CTRL = "0.000000 -0.087300 -0.457900 -0.004900 0.453000 0.276000 0.276000 0.000000 0.000000 0.000000 0.087300 0.457900 0.004900 -0.453000 1.550000"


def build_standing_scene_xml():
    """Generates the standing scene with stiff position actuators on stance legs and calibrated Keyframe 0."""
    xml_str = create_spoon_lemon_scene_xml()
    # Increase actuator stiffness to firm position servos (25.0 N*m/rad) so legs hold standing stance rigidly
    xml_str = xml_str.replace('gainprm="0.55" biasprm="0 -0.55"', 'gainprm="25.0" biasprm="0 -25.0"')
    xml_str = xml_str.replace('forcerange="-0.96 0.96"', 'forcerange="-5.0 5.0"')
    # Replace default empty INIT keyframe with exact pristine standing pose (Keyframe 0)
    xml_str = xml_str.replace(
        '<key name="INIT" qpos="0 0 0.12 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"/>',
        f'<key name="INIT" qpos="{STANDING_KEY_QPOS}" ctrl="{STANDING_KEY_CTRL}"/>'
    )
    return xml_str


def setup_standing_robot(model, data, default_pose, joint_qpos_indices, beak_qpos, beak_dof,
                         spoon_bid, lemon_bid, jaw_bid, spoon_qadr, lemon_qadr,
                         target_clamp_angle=1.550, clamp_torque=0.45):
    """Firmly settles the robot into standing posture with spoon and lemon placed."""
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[0:6] = 0.0

    for i, idx in enumerate(joint_qpos_indices):
        data.qpos[idx] = default_pose[i]
        data.ctrl[i] = default_pose[i]

    data.qpos[beak_qpos] = target_clamp_angle
    data.qfrc_applied[beak_dof] = -clamp_torque
    if model.nu > 14:
        data.ctrl[14] = target_clamp_angle
    mujoco.mj_forward(model, data)

    # Place spoon in jaw cradle
    p_jaw = data.xpos[jaw_bid]
    R_jaw = data.xmat[jaw_bid].reshape(3, 3)
    HOVER_LOCAL = np.array([-0.0119, 0.0, -0.0465])
    CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

    data.qpos[spoon_qadr:spoon_qadr + 3] = p_jaw + R_jaw @ HOVER_LOCAL
    q_jaw = np.empty(4)
    mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
    q_s = np.empty(4)
    mujoco.mju_mulQuat(q_s, q_jaw, CYL_QUAT)
    data.qpos[spoon_qadr + 3:spoon_qadr + 7] = q_s
    data.qvel[spoon_qadr:spoon_qadr + 6] = 0.0
    mujoco.mj_forward(model, data)

    # Place lemon inside concave spoon bowl
    p_spoon = data.xpos[spoon_bid]
    R_spoon = data.xmat[spoon_bid].reshape(3, 3)
    data.qpos[lemon_qadr:lemon_qadr + 3] = p_spoon + R_spoon @ np.array([0.0, 0.0052, 0.070])
    data.qpos[lemon_qadr + 3:lemon_qadr + 7] = q_s
    data.qvel[lemon_qadr:lemon_qadr + 6] = 0.0
    mujoco.mj_forward(model, data)

    # Settle feet on ground
    for _ in range(150):
        data.qpos[0:3] = [0.0, 0.0, 0.125]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[0:6] = 0.0
        for i in range(14):
            data.ctrl[i] = default_pose[i]
        if model.nu > 14:
            data.ctrl[14] = target_clamp_angle
        data.qfrc_applied[beak_dof] = -clamp_torque
        data.qvel[lemon_qadr:lemon_qadr + 6] *= 0.95
        mujoco.mj_step(model, data)

    # Final zero velocity check
    data.qvel[:] = 0.0
    for i in range(14):
        data.ctrl[i] = default_pose[i]
    if model.nu > 14:
        data.ctrl[14] = target_clamp_angle
    data.qfrc_applied[beak_dof] = -clamp_torque
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description="MicroDuck Standing Actuator Lemon Drop Simulation")
    parser.add_argument("--test", action="store_true", help="Run automated test suite verifying lemon drop physics")
    parser.add_argument("--proofs", action="store_true", help="Render visual proof images of standing, slipping, and dropping")
    args = parser.parse_args()

    xml_str = build_standing_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    default_pose = DEFAULT_POSE[:14].copy()
    default_pose[5] = 0.2760  # neck pitch: level
    default_pose[6] = 0.2760  # head pitch: level
    default_pose[7] = 0.0     # head yaw
    default_pose[8] = 0.0     # head roll

    joint_qpos_indices = [int(model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in range(14)]
    beak_qpos = int(model.jnt_qposadr[model.actuator_trnid[14, 0]])
    beak_dof = int(model.jnt_dofadr[model.actuator_trnid[14, 0]])

    spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lemon")
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    spoon_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spoon_free")]
    lemon_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lemon_free")]

    _ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", str(Path(__file__).resolve().parent.parent / "docs" / "media"))

    if args.proofs:
        from PIL import Image
        renderer = mujoco.Renderer(model, width=1280, height=720)
        vopt = mujoco.MjvOption()
        vopt.geomgroup[3] = 0

        # State 1: Stable level standing
        setup_standing_robot(model, data, default_pose, joint_qpos_indices, beak_qpos, beak_dof,
                             spoon_bid, lemon_bid, jaw_bid, spoon_qadr, lemon_qadr)
        for _ in range(250):
            for i in range(14): data.ctrl[i] = default_pose[i]
            beak_err = 1.550 - data.qpos[beak_qpos]
            data.qfrc_applied[beak_dof] = float(np.clip(-0.45 + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))
            mujoco.mj_step(model, data)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat = [0.10, 0.0, 0.16]
        cam.distance = 0.52
        cam.elevation = -12.0
        cam.azimuth = 45.0
        renderer.update_scene(data, camera=cam, scene_option=vopt)
        p1 = os.path.join(_ARTIFACT_DIR, "standing_lemon_balanced_proof.png")
        Image.fromarray(renderer.render()).save(p1)

        # State 2: Head tilt slip (critical angle ~27°)
        ctrl_test = default_pose.copy()
        for step in range(500):
            t = step * 0.002
            pitch_delta = min(0.48, t * 0.50)  # ~27.5 degrees
            ctrl_test[5] = default_pose[5] - pitch_delta
            ctrl_test[6] = default_pose[6] - pitch_delta
            for i in range(14): data.ctrl[i] = ctrl_test[i]
            beak_err = 1.550 - data.qpos[beak_qpos]
            data.qfrc_applied[beak_dof] = float(np.clip(-0.45 + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=cam, scene_option=vopt)
        p2 = os.path.join(_ARTIFACT_DIR, "standing_lemon_tilt_slip_proof.png")
        Image.fromarray(renderer.render()).save(p2)

        # State 3: Lemon dropped to floor
        for step in range(1200):
            ctrl_test[5] = default_pose[5] - 0.85
            ctrl_test[6] = default_pose[6] - 0.85
            for i in range(14): data.ctrl[i] = ctrl_test[i]
            beak_err = 1.550 - data.qpos[beak_qpos]
            data.qfrc_applied[beak_dof] = float(np.clip(-0.45 + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=cam, scene_option=vopt)
        p3 = os.path.join(_ARTIFACT_DIR, "standing_lemon_dropped_floor_proof.png")
        Image.fromarray(renderer.render()).save(p3)

        print("\nVisual proofs saved:")
        print(f"  1. Balanced:  {p1}")
        print(f"  2. Slipping:  {p2}")
        print(f"  3. Dropped:   {p3}")
        return

    # Automated test suite mode
    if args.test:
        print("\n" + "=" * 70)
        print("RUNNING AUTOMATED LEMON DROP PHYSICS VERIFICATION")
        print("=" * 70)
        
        # Test 1: Baseline standing stability (tilt = 0.0°)
        setup_standing_robot(model, data, default_pose, joint_qpos_indices, beak_qpos, beak_dof,
                             spoon_bid, lemon_bid, jaw_bid, spoon_qadr, lemon_qadr)
        for _ in range(1000):  # 2.0s
            for i in range(14): data.ctrl[i] = default_pose[i]
            beak_err = 1.550 - data.qpos[beak_qpos]
            data.qfrc_applied[beak_dof] = float(np.clip(-0.45 + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))
            mujoco.mj_step(model, data)
        p_s = data.xpos[spoon_bid]
        R_s = data.xmat[spoon_bid].reshape(3, 3)
        bowl = p_s + R_s @ np.array([0.0, 0.0052, 0.070])
        disp1 = float(np.linalg.norm(data.xpos[lemon_bid] - bowl))
        print(f"Test 1 (Standing at 0° tilt):   disp = {disp1 * 1000:.2f} mm | Balanced = {disp1 < 0.005}")

        # Test 2: Pitch tilt forward past 26.6° friction limit
        setup_standing_robot(model, data, default_pose, joint_qpos_indices, beak_qpos, beak_dof,
                             spoon_bid, lemon_bid, jaw_bid, spoon_qadr, lemon_qadr)
        ctrl_test = default_pose.copy()
        dropped_pitch = False
        drop_time = None
        for step in range(2500):  # 5.0s
            t = step * 0.002
            pitch_delta = min(0.90, t * 0.30)
            ctrl_test[5] = default_pose[5] - pitch_delta
            ctrl_test[6] = default_pose[6] - pitch_delta
            for i in range(14): data.ctrl[i] = ctrl_test[i]
            beak_err = 1.550 - data.qpos[beak_qpos]
            data.qfrc_applied[beak_dof] = float(np.clip(-0.45 + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))
            mujoco.mj_step(model, data)
            p_s = data.xpos[spoon_bid]
            R_s = data.xmat[spoon_bid].reshape(3, 3)
            bowl = p_s + R_s @ np.array([0.0, 0.0052, 0.070])
            disp = float(np.linalg.norm(data.xpos[lemon_bid] - bowl))
            if disp > 0.045 and not dropped_pitch:
                dropped_pitch = True
                drop_time = t
                print(f"Test 2 (Forward Pitch Ramp):   DROPPED at t={drop_time:.2f}s (pitch_delta = {pitch_delta:.2f} rad, disp = {disp * 1000:.1f} mm)")
                break
        print(f"Test 2 Result: Dropped = {dropped_pitch}")

        # Test 3: Beak Release (loosen beak clamp)
        setup_standing_robot(model, data, default_pose, joint_qpos_indices, beak_qpos, beak_dof,
                             spoon_bid, lemon_bid, jaw_bid, spoon_qadr, lemon_qadr)
        dropped_beak = False
        for step in range(1500):
            t = step * 0.002
            data.qfrc_applied[beak_dof] = 0.0  # Zero clamping torque
            data.qpos[beak_qpos] = 2.00        # Open beak
            for i in range(14): data.ctrl[i] = default_pose[i]
            mujoco.mj_step(model, data)
            if data.xpos[spoon_bid][2] < 0.05 and not dropped_beak:
                dropped_beak = True
                print(f"Test 3 (Beak Release Open):    SPOON & LEMON FELL TO FLOOR at t={t:.2f}s (spoon_z = {data.xpos[spoon_bid][2]:.3f} m)")
                break
        print(f"Test 3 Result: Dropped = {dropped_beak}")
        print("=" * 70 + "\n")
        return

    # Interactive Native Viewer Mode
    # Calibrate each actuator's ctrlrange to its actual joint range so MuJoCo GUI sliders have perfect bounds
    for a in range(model.nu):
        jnt_id = model.actuator_trnid[a, 0]
        model.actuator_ctrlrange[a] = model.jnt_range[jnt_id]

    ACTUATOR_NAMES = [
        "left_hip_yaw",     # 0
        "left_hip_roll",    # 1
        "left_hip_pitch",   # 2
        "left_knee",        # 3
        "left_ankle",       # 4
        "neck_pitch",       # 5
        "head_pitch",       # 6
        "head_yaw",         # 7
        "head_roll",        # 8
        "right_hip_yaw",    # 9
        "right_hip_roll",   # 10
        "right_hip_pitch",  # 11
        "right_knee",       # 12
        "right_ankle",      # 13
        "beak",             # 14
    ]

    setup_standing_robot(model, data, default_pose, joint_qpos_indices, beak_qpos, beak_dof,
                         spoon_bid, lemon_bid, jaw_bid, spoon_qadr, lemon_qadr)

    clamp_torque = 0.45
    selected_actuator = 6  # Default: head_pitch
    auto_mode = None  # None, "pitch_drop", "roll_drop", "beak_release"
    auto_timer = 0.0

    user_paused = False
    reset_requested = False
    camera_mode = 0  # 0: Close-Up, 1: Side Profile, 2: Isometric, 3: Front, 4: Top-Down
    cam_preset_changed = True

    def on_key(keycode):
        nonlocal user_paused, reset_requested, selected_actuator
        nonlocal clamp_torque, auto_mode, auto_timer, camera_mode, cam_preset_changed

        if keycode == 32:  # Space
            user_paused = not user_paused
        elif keycode in (ord('R'), ord('r'), ord('1'), 8, 259):  # R, 1, or Backspace: Pristine reset
            reset_requested = True
            cam_preset_changed = True
        elif keycode in (ord('2'),):
            auto_mode = "pitch_drop"
            auto_timer = 0.0
            print("\n[DEMO] Triggered Demo 2: Automatic Forward Pitch Drop (tilts past 26.6° friction limit)")
        elif keycode in (ord('3'),):
            auto_mode = "roll_drop"
            auto_timer = 0.0
            print("\n[DEMO] Triggered Demo 3: Automatic Side Roll Drop (tilts sideways over rim)")
        elif keycode in (ord('4'),):
            auto_mode = "beak_release"
            auto_timer = 0.0
            print("\n[DEMO] Triggered Demo 4: Automatic Beak Opening & Release")
        elif keycode in (ord('['),):
            selected_actuator = (selected_actuator - 1) % model.nu
            print(f"\n[ACTUATOR] Selected: [{selected_actuator}] {ACTUATOR_NAMES[selected_actuator]} = {math.degrees(data.ctrl[selected_actuator]):+.1f}°")
        elif keycode in (ord(']'),):
            selected_actuator = (selected_actuator + 1) % model.nu
            print(f"\n[ACTUATOR] Selected: [{selected_actuator}] {ACTUATOR_NAMES[selected_actuator]} = {math.degrees(data.ctrl[selected_actuator]):+.1f}°")
        elif keycode in (ord('+'), ord('='), 265):  # + or Up: Step selected actuator positive
            auto_mode = None
            mn, mx = model.actuator_ctrlrange[selected_actuator]
            data.ctrl[selected_actuator] = float(np.clip(data.ctrl[selected_actuator] + 0.04, mn, mx))
            print(f"\n[ACTUATOR] [{selected_actuator}] {ACTUATOR_NAMES[selected_actuator]} -> {math.degrees(data.ctrl[selected_actuator]):+.1f}° ({data.ctrl[selected_actuator]:.3f} rad)")
        elif keycode in (ord('-'), ord('_'), 264):  # - or Down: Step selected actuator negative
            auto_mode = None
            mn, mx = model.actuator_ctrlrange[selected_actuator]
            data.ctrl[selected_actuator] = float(np.clip(data.ctrl[selected_actuator] - 0.04, mn, mx))
            print(f"\n[ACTUATOR] [{selected_actuator}] {ACTUATOR_NAMES[selected_actuator]} -> {math.degrees(data.ctrl[selected_actuator]):+.1f}° ({data.ctrl[selected_actuator]:.3f} rad)")
        elif keycode in (ord('W'), ord('w')):  # W: Tilt head/neck pitch down (forward)
            auto_mode = None
            data.ctrl[5] = float(np.clip(data.ctrl[5] - 0.04, model.actuator_ctrlrange[5, 0], model.actuator_ctrlrange[5, 1]))
            data.ctrl[6] = float(np.clip(data.ctrl[6] - 0.04, model.actuator_ctrlrange[6, 0], model.actuator_ctrlrange[6, 1]))
            print(f"\n[ACTUATOR] Neck & Head Pitch Down -> Neck:{math.degrees(data.ctrl[5]):+.1f}°, Head:{math.degrees(data.ctrl[6]):+.1f}°")
        elif keycode in (ord('S'), ord('s')):  # S: Tilt head/neck pitch up (backward)
            auto_mode = None
            data.ctrl[5] = float(np.clip(data.ctrl[5] + 0.04, model.actuator_ctrlrange[5, 0], model.actuator_ctrlrange[5, 1]))
            data.ctrl[6] = float(np.clip(data.ctrl[6] + 0.04, model.actuator_ctrlrange[6, 0], model.actuator_ctrlrange[6, 1]))
            print(f"\n[ACTUATOR] Neck & Head Pitch Up -> Neck:{math.degrees(data.ctrl[5]):+.1f}°, Head:{math.degrees(data.ctrl[6]):+.1f}°")
        elif keycode in (ord('A'), ord('a'), 263):  # A: Roll head left
            auto_mode = None
            data.ctrl[8] = float(np.clip(data.ctrl[8] + 0.04, model.actuator_ctrlrange[8, 0], model.actuator_ctrlrange[8, 1]))
            print(f"\n[ACTUATOR] Head Roll Left -> {math.degrees(data.ctrl[8]):+.1f}°")
        elif keycode in (ord('D'), ord('d'), 262):  # D: Roll head right
            auto_mode = None
            data.ctrl[8] = float(np.clip(data.ctrl[8] - 0.04, model.actuator_ctrlrange[8, 0], model.actuator_ctrlrange[8, 1]))
            print(f"\n[ACTUATOR] Head Roll Right -> {math.degrees(data.ctrl[8]):+.1f}°")
        elif keycode in (ord('Q'), ord('q')):       # Q: Yaw head left
            auto_mode = None
            data.ctrl[7] = float(np.clip(data.ctrl[7] + 0.05, model.actuator_ctrlrange[7, 0], model.actuator_ctrlrange[7, 1]))
            print(f"\n[ACTUATOR] Head Yaw Left -> {math.degrees(data.ctrl[7]):+.1f}°")
        elif keycode in (ord('E'), ord('e')):       # E: Yaw head right
            auto_mode = None
            data.ctrl[7] = float(np.clip(data.ctrl[7] - 0.05, model.actuator_ctrlrange[7, 0], model.actuator_ctrlrange[7, 1]))
            print(f"\n[ACTUATOR] Head Yaw Right -> {math.degrees(data.ctrl[7]):+.1f}°")
        elif keycode in (ord('O'), ord('o')):       # O: Open beak
            auto_mode = None
            data.ctrl[14] = float(np.clip(data.ctrl[14] + 0.04, 1.47, 2.05))
            print(f"\n[ACTUATOR] Beak Open Angle -> {data.ctrl[14]:.3f} rad ({math.degrees(data.ctrl[14]):.1f}°)")
        elif keycode in (ord('C'), ord('c')):       # C: Close beak
            auto_mode = None
            data.ctrl[14] = float(np.clip(data.ctrl[14] - 0.04, 1.47, 2.05))
            print(f"\n[ACTUATOR] Beak Clamp Angle -> {data.ctrl[14]:.3f} rad ({math.degrees(data.ctrl[14]):.1f}°)")
        elif keycode in (ord('K'), ord('k')):       # K: Crouch knees
            auto_mode = None
            data.ctrl[3] = float(np.clip(data.ctrl[3] + 0.06, -1.57, 1.57))
            data.ctrl[12] = float(np.clip(data.ctrl[12] - 0.06, -1.57, 1.57))
            print(f"\n[ACTUATOR] Knees Flex (Crouch) -> L:{math.degrees(data.ctrl[3]):.1f}°, R:{math.degrees(data.ctrl[12]):.1f}°")
        elif keycode in (ord('I'), ord('i')):       # I: Extend knees (Stand)
            auto_mode = None
            data.ctrl[3] = float(np.clip(data.ctrl[3] - 0.06, -1.57, 1.57))
            data.ctrl[12] = float(np.clip(data.ctrl[12] + 0.06, -1.57, 1.57))
            print(f"\n[ACTUATOR] Knees Extend (Stand) -> L:{math.degrees(data.ctrl[3]):.1f}°, R:{math.degrees(data.ctrl[12]):.1f}°")
        elif keycode in (ord('H'), ord('h')):       # H: Hips pitch forward
            auto_mode = None
            data.ctrl[2] = float(np.clip(data.ctrl[2] + 0.05, -1.57, 1.57))
            data.ctrl[11] = float(np.clip(data.ctrl[11] - 0.05, -1.57, 1.57))
            print(f"\n[ACTUATOR] Hips Pitch Forward -> L:{math.degrees(data.ctrl[2]):.1f}°, R:{math.degrees(data.ctrl[11]):.1f}°")
        elif keycode in (ord('Y'), ord('y')):       # Y: Hips pitch backward
            auto_mode = None
            data.ctrl[2] = float(np.clip(data.ctrl[2] - 0.05, -1.57, 1.57))
            data.ctrl[11] = float(np.clip(data.ctrl[11] + 0.05, -1.57, 1.57))
            print(f"\n[ACTUATOR] Hips Pitch Backward -> L:{math.degrees(data.ctrl[2]):.1f}°, R:{math.degrees(data.ctrl[11]):.1f}°")
        elif keycode in (ord('V'), ord('v'), 9, 258):  # V or Tab: Cycle camera views
            camera_mode = (camera_mode + 1) % 5
            cam_preset_changed = True
            cams = ["Spoon & Lemon Close-Up", "Side Profile", "Isometric Arena", "Front Facial", "Top-Down Bird's-Eye"]
            print(f"\n[CAMERA] Preset -> {cams[camera_mode]} (Mouse Orbit & Zoom Active)")
        elif keycode == ord('5'):
            camera_mode = 0
            cam_preset_changed = True
            print("\n[CAMERA] Preset 5 -> Spoon & Lemon Close-Up (Mouse Orbit & Zoom Active)")
        elif keycode == ord('6'):
            camera_mode = 1
            cam_preset_changed = True
            print("\n[CAMERA] Preset 6 -> Side Profile (Tilt Angle View)")
        elif keycode == ord('7'):
            camera_mode = 2
            cam_preset_changed = True
            print("\n[CAMERA] Preset 7 -> Isometric Arena View")
        elif keycode == ord('8'):
            camera_mode = 3
            cam_preset_changed = True
            print("\n[CAMERA] Preset 8 -> Front Facial View")
        elif keycode == ord('9'):
            camera_mode = 4
            cam_preset_changed = True
            print("\n[CAMERA] Preset 9 -> Top-Down Bird's-Eye View")

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.opt.geomgroup[3] = 0

    print("\n" + "=" * 76)
    print("MICRODUCK STANDING ACTUATOR CONTROL & LEMON DROP VIEWER READY")
    print("=" * 76)
    print("  GUI SLIDERS (100% REAL-TIME CONTROLLABLE IN MUJOCO WINDOW):")
    print("    Look at the left sidebar (Control tab). All 15 sliders are active:")
    print("      • neck_pitch, head_pitch, head_yaw, head_roll")
    print("      • beak (aperture & clamping)")
    print("      • left_knee, right_knee (crouch / squat)")
    print("      • left_hip_pitch, right_hip_pitch (pelvis lean)")
    print("      • left_ankle, right_ankle, hip yaw & roll")
    print("    Drag ANY slider left or right with your mouse to move it in real-time!")
    print("  KEYBOARD CONTROLS:")
    print("    [ / ]:                   Select ANY of the 15 actuators")
    print("    Up / Down (or + / -):    Adjust selected actuator angle (+/- 2.3°)")
    print("    W / S:                   Quick Head & Neck Pitch (Forward / Backward)")
    print("    A / D:                   Quick Head Roll (Left / Right)")
    print("    Q / E:                   Quick Head Yaw (Left / Right)")
    print("    O / C:                   Quick Beak Open / Close")
    print("    K / I:                   Quick Knees Flex / Extend (Squat / Stand)")
    print("    H / Y:                   Quick Hips Pitch (Lean Forward / Backward)")
    print("  PRESET DEMOS:")
    print("    1:                       Reset to Level Neutral Standing (0.0°)")
    print("    2:                       Auto Pitch Drop Demo (Tilts past 26.6° friction limit)")
    print("    3:                       Auto Roll Drop Demo (Rolls lemon over side rim)")
    print("    4:                       Auto Beak Release Demo (Loosens jaw until spoon drops)")
    print("  CAMERA PRESETS & MOUSE:")
    print("    Mouse Left-Click+Drag:   Orbit / Rotate 360° around robot")
    print("    Mouse Right-Click+Drag:  Pan / Translate view")
    print("    Scroll Wheel:            Zoom in / out smoothly")
    print("    V or Tab:                Cycle Views (Close-Up -> Side -> Iso -> Front -> Top)")
    print("    5 / 6 / 7 / 8 / 9:       Direct presets (5:Close, 6:Side, 7:Iso, 8:Front, 9:Top)")
    print("  SIMULATION CONTROLS:")
    print("    Space:                   Pause / Resume physics")
    print("    R / 1 / Backspace:       Pristine Reset (also GUI toolbar 'Reset' button)")
    print("=" * 76 + "\n")

    step = 0
    sim_dt = model.opt.timestep

    while viewer.is_running():
        t_start = time.time()

        # Check if reset requested or if native MuJoCo GUI reset button was clicked (data.time jumped back to 0 or data.ctrl[14] zeroed)
        native_gui_reset = (step > 15 and data.time < 0.010) or (data.ctrl[14] < 1.30)
        if reset_requested or native_gui_reset:
            reset_requested = False
            auto_mode = None
            auto_timer = 0.0
            clamp_torque = 0.45
            selected_actuator = 6  # head_pitch
            cam_preset_changed = True
            camera_mode = 0  # Spoon & lemon close-up view

            setup_standing_robot(model, data, default_pose, joint_qpos_indices, beak_qpos, beak_dof,
                                 spoon_bid, lemon_bid, jaw_bid, spoon_qadr, lemon_qadr,
                                 target_clamp_angle=1.550, clamp_torque=0.45)
            for i in range(14):
                data.ctrl[i] = default_pose[i]
            data.ctrl[14] = 1.550
            data.qfrc_applied[beak_dof] = -0.45
            step = 16
            print("\n[RESET] Simulation reset to pristine standing state with balanced lemon!")

        if user_paused:
            viewer.sync()
            time.sleep(0.02)
            continue

        # Handle automated demos
        if auto_mode == "pitch_drop":
            auto_timer += sim_dt
            p_val = default_pose[5] - min(0.95, auto_timer * 0.40)
            data.ctrl[5] = p_val
            data.ctrl[6] = p_val
        elif auto_mode == "roll_drop":
            auto_timer += sim_dt
            data.ctrl[8] = default_pose[8] + min(0.40, auto_timer * 0.25)
        elif auto_mode == "beak_release":
            auto_timer += sim_dt
            data.ctrl[14] = min(2.05, 1.550 + auto_timer * 0.50)

        # Active beak clamping physics tied directly to data.ctrl[14]
        target_clamp_angle = float(data.ctrl[14])
        if target_clamp_angle > 1.62:
            eff_torque = max(0.0, clamp_torque * max(0.0, (1.85 - target_clamp_angle) / 0.23))
        else:
            eff_torque = clamp_torque

        beak_err = target_clamp_angle - data.qpos[beak_qpos]
        data.qfrc_applied[beak_dof] = float(np.clip(-eff_torque + 15.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.2, 0.2))

        mujoco.mj_step(model, data)
        step += 1

        # Calculate lemon and spoon telemetry
        p_s = data.xpos[spoon_bid]
        R_s = data.xmat[spoon_bid].reshape(3, 3)
        bowl = p_s + R_s @ np.array([0.0, 0.0052, 0.070])
        lemon_pos = data.xpos[lemon_bid]
        disp_mm = float(np.linalg.norm(lemon_pos - bowl)) * 1000.0

        # Spoon orientation relative to world horizontal
        # spoon forward vector in world:
        h_fwd = R_s @ np.array([0.0, 0.0, 1.0])
        # spoon bowl normal vector in world:
        b_norm = R_s @ np.array([0.0, 1.0, 0.0])
        spoon_pitch_deg = math.degrees(math.asin(np.clip(b_norm[0], -0.999, 0.999)))
        spoon_roll_deg = math.degrees(math.asin(np.clip(b_norm[1], -0.999, 0.999)))

        # Status text
        if disp_mm < 3.0:
            status = "BALANCED (In Bowl)"
            status_color = "\033[92m"  # Green
        elif disp_mm < 45.0:
            status = "SLIPPING ON RIM"
            status_color = "\033[93m"  # Yellow
        else:
            status = "DROPPED TO FLOOR"
            status_color = "\033[91m"  # Red
        reset_color = "\033[0m"

        # Apply camera preset when requested (does NOT overwrite every frame so native mouse orbit/zoom works!)
        if cam_preset_changed:
            p_jaw = data.xpos[jaw_bid]
            if camera_mode == 0:    # Close-up of spoon and lemon
                viewer.cam.lookat = [p_s[0] + 0.03, p_s[1], p_s[2] + 0.04]
                viewer.cam.distance = 0.38
                viewer.cam.elevation = -12.0
                viewer.cam.azimuth = 32.0
            elif camera_mode == 1:  # Side profile showing tilt angle
                viewer.cam.lookat = [p_jaw[0], p_jaw[1], p_jaw[2] - 0.02]
                viewer.cam.distance = 0.55
                viewer.cam.elevation = -5.0
                viewer.cam.azimuth = 90.0
            elif camera_mode == 2:  # Isometric standing view
                viewer.cam.lookat = [0.0, 0.0, 0.12]
                viewer.cam.distance = 0.85
                viewer.cam.elevation = -18.0
                viewer.cam.azimuth = 45.0
            elif camera_mode == 3:  # Front facial view
                viewer.cam.lookat = [p_s[0] + 0.05, p_s[1], p_s[2]]
                viewer.cam.distance = 0.45
                viewer.cam.elevation = -8.0
                viewer.cam.azimuth = 180.0
            elif camera_mode == 4:  # Top-down bird's-eye view
                viewer.cam.lookat = [p_s[0], p_s[1], p_s[2]]
                viewer.cam.distance = 0.50
                viewer.cam.elevation = -88.0
                viewer.cam.azimuth = 90.0
            cam_preset_changed = False

        viewer.sync()

        # Telemetry terminal HUD every 50 steps (0.10s)
        if step % 50 == 0:
            friction_limit = 26.6  # arctan(0.50)
            tilt_warning = " [EXCEEDS 26.6° LIMIT!]" if abs(spoon_pitch_deg) > friction_limit or abs(spoon_roll_deg) > friction_limit else ""
            sel_name = ACTUATOR_NAMES[selected_actuator]
            sel_val_deg = math.degrees(data.ctrl[selected_actuator])
            hud_str = (
                f"\rSelected: [{selected_actuator:2d}] {sel_name:15s}={sel_val_deg:+5.1f}° | "
                f"Spoon Pitch:{spoon_pitch_deg:+5.1f}° | Roll:{spoon_roll_deg:+5.1f}° | "
                f"Beak:{math.degrees(target_clamp_angle):4.1f}° | "
                f"Disp:{disp_mm:4.1f}mm | {status_color}{status}{reset_color}{tilt_warning}    "
            )
            sys.stdout.write(hud_str)
            sys.stdout.flush()

        t_elapsed = time.time() - t_start
        dt_sleep = sim_dt - t_elapsed
        if dt_sleep > 0:
            time.sleep(dt_sleep)

    viewer.close()


if __name__ == "__main__":
    main()
