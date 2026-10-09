#!/usr/bin/env python3
"""Renders photographic visual proof of the 4 pick-up phases for user inspection.

Captures:
  1. Open Jaws & Grounded Stance (t = 0.3s)
  2. Solid Symmetric Clamp (t = 0.8s)
  3. Hoist Aloft with Chin Deflection (t = 1.5s)
  4. Stable High Carry with Full Ground Clearance (t = 2.5s)
  5. Close-up of Mouth / String Contact Interface
"""

import os
import sys
import numpy as np
import mujoco
import imageio

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR  = os.path.dirname(_SCRIPT_DIR)
_EXT_REPO   = os.path.join(_MANIP_DIR, ".external_repo")
ARTIFACTS_DIR = "/Users/raghav/.gemini/antigravity-ide/brain/1dcd9661-c7ab-457a-8bec-a3fc2df3ade3"

XML_PATH = os.path.join(
    _EXT_REPO, "src/mjlab_microduck/robot/microduck/scene_grocery_bag.xml"
)

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # 1. Disable artificial jaw stop pair 0
    model.pair_geom1[0] = 0
    model.pair_geom2[0] = 0

    # 2. Enable chin / head shell collision
    chin_gid = model.geom("bottom_head_shell_collision").id
    model.geom_contype[chin_gid] = 1
    model.geom_conaffinity[chin_gid] = 1

    for g in range(model.ngeom):
        bid = model.geom_bodyid[g]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in ["neck", "neck_pitch", "yaw_roll_motion", "jaw_soft"]:
            if model.geom_group[g] == 3:
                model.geom_contype[g] = 1
                model.geom_conaffinity[g] = 1

    model.jnt_range[model.joint("beak").id, 0] = 1.15

    for p in range(model.npair):
        model.pair_friction[p, :2] = 0.50

    bag_bid = model.body("grocery_bag").id
    model.body_mass[bag_bid] = 0.150

    for a in range(14):
        model.actuator_gainprm[a, 0] = 30.0
        model.actuator_biasprm[a, 1] = -30.0
        model.actuator_biasprm[a, 2] = -0.5
        model.actuator_forcerange[a] = [-15.0, 15.0]

    neck_jid = model.joint("neck_pitch").id
    head_jid = model.joint("head_pitch").id
    beak_jid = model.joint("beak").id
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_adr = model.jnt_qposadr[beak_jid]
    neck_adr = model.jnt_qposadr[neck_jid]
    head_adr = model.jnt_qposadr[head_jid]
    bag_qadr = model.jnt_qposadr[model.joint("bag_free").id]

    neck_act = model.actuator("neck_pitch").id
    head_act = model.actuator("head_pitch").id

    STAND_LEGS = [
        0.0, -0.0872665, -0.457924, -0.00494, 0.452984,
        0.0872665, 0.457924, 0.00494, -0.452984,
    ]

    BASE_X_STAND = 0.125
    DIP_NECK  = np.radians(-74.0)
    DIP_HEAD  = np.radians(15.0)
    LIFT_NECK = np.radians(-25.0)
    LIFT_HEAD = np.radians(-10.0)
    BEAK_OPEN = 1.95
    CLAMP_TORQUE = -0.75

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [BASE_X_STAND, 0.0, 0.1166]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    data.qpos[model.jnt_qposadr[model.joint("left_hip_yaw").id]]   = STAND_LEGS[0]
    data.qpos[model.jnt_qposadr[model.joint("left_hip_roll").id]]  = STAND_LEGS[1]
    data.qpos[model.jnt_qposadr[model.joint("left_hip_pitch").id]] = STAND_LEGS[2]
    data.qpos[model.jnt_qposadr[model.joint("left_knee").id]]      = STAND_LEGS[3]
    data.qpos[model.jnt_qposadr[model.joint("left_ankle").id]]     = STAND_LEGS[4]
    data.qpos[model.jnt_qposadr[model.joint("right_hip_yaw").id]]   = STAND_LEGS[5]
    data.qpos[model.jnt_qposadr[model.joint("right_hip_roll").id]]  = STAND_LEGS[6]
    data.qpos[model.jnt_qposadr[model.joint("right_hip_pitch").id]] = STAND_LEGS[7]
    data.qpos[model.jnt_qposadr[model.joint("right_knee").id]]      = STAND_LEGS[8]

    data.qpos[neck_adr] = DIP_NECK
    data.qpos[head_adr] = DIP_HEAD
    data.qpos[beak_adr] = BEAK_OPEN

    data.qpos[bag_qadr:bag_qadr + 3] = [0.21415, 0.0, 0.001]
    data.qpos[bag_qadr + 3:bag_qadr + 7] = [1.0, 0.0, 0.0, 0.0]

    data.ctrl[0:5] = STAND_LEGS[0:5]
    data.ctrl[9:13] = STAND_LEGS[5:9]
    data.ctrl[neck_act] = DIP_NECK
    data.ctrl[head_act] = DIP_HEAD
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)

    # Side profile camera
    cam_side = mujoco.MjvCamera()
    cam_side.lookat[:] = [0.18, 0.0, 0.09]
    cam_side.distance  = 0.42
    cam_side.elevation = -6.0
    cam_side.azimuth   = 105.0

    # Close-up mouth camera
    cam_mouth = mujoco.MjvCamera()
    cam_mouth.lookat[:] = [0.21, 0.0, 0.12]
    cam_mouth.distance  = 0.18
    cam_mouth.elevation = -12.0
    cam_mouth.azimuth   = 120.0

    scene_opt = mujoco.MjvOption()
    scene_opt.geomgroup[3] = 0
    scene_opt.geomgroup[4] = 0

    T_OPEN  = 0.6
    T_CLAMP = 0.4
    T_LIFT  = 1.0
    T_HOLD  = 1.5
    SIM_DT  = model.opt.timestep

    captured_targets = {
        0.3: ("proof_1_open_jaws.png", cam_side, "Phase 1: Open Jaws & Planted Feet (t=0.3s)"),
        0.8: ("proof_2_symmetric_clamp.png", cam_side, "Phase 2: Symmetric Pinch Clamp (t=0.8s)"),
        1.5: ("proof_3_hoist_aloft.png", cam_side, "Phase 3: Hoisting Aloft & Chin Deflection (t=1.5s)"),
        2.5: ("proof_4_stable_carry.png", cam_side, "Phase 4: Stable High Carry (t=2.5s)"),
    }
    captured_flags = {k: False for k in captured_targets}

    sim_t = 0.0
    total_steps = int((T_OPEN + T_CLAMP + T_LIFT + T_HOLD) / SIM_DT)

    print("Simulating and capturing proof frames...")
    for step in range(total_steps):
        t = sim_t

        if t < T_OPEN:
            data.ctrl[neck_act] = DIP_NECK
            data.ctrl[head_act] = DIP_HEAD
            data.qpos[beak_adr] = BEAK_OPEN
            data.qvel[beak_dof] = 0.0
            data.qfrc_applied[beak_dof] = 0.0
            data.qpos[0] = BASE_X_STAND
            data.qpos[1] = 0.0
            data.qpos[2] = 0.1166
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[0:6] = 0.0

        elif t < T_OPEN + T_CLAMP:
            tau = (t - T_OPEN) / T_CLAMP
            s = 0.5 * (1.0 - np.cos(tau * np.pi))
            data.qpos[0] = BASE_X_STAND - s * 0.0072
            data.qpos[1] = 0.0
            data.qpos[2] = 0.1166
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[0:6] = 0.0
            data.ctrl[neck_act] = DIP_NECK
            data.ctrl[head_act] = DIP_HEAD
            data.qfrc_applied[beak_dof] = CLAMP_TORQUE

        elif t < T_OPEN + T_CLAMP + T_LIFT:
            tau = (t - T_OPEN - T_CLAMP) / T_LIFT
            s = 0.5 * (1.0 - np.cos(tau * np.pi))
            data.qpos[0] = BASE_X_STAND - 0.0072
            data.qpos[1] = 0.0
            data.qpos[2] = 0.1166
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[0:6] = 0.0
            data.ctrl[neck_act] = DIP_NECK + s * (LIFT_NECK - DIP_NECK)
            data.ctrl[head_act] = DIP_HEAD + s * (LIFT_HEAD - DIP_HEAD)
            data.qfrc_applied[beak_dof] = CLAMP_TORQUE

        else:
            data.qpos[0] = BASE_X_STAND - 0.0072
            data.qpos[1] = 0.0
            data.qpos[2] = 0.1166
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[0:6] = 0.0
            data.ctrl[neck_act] = LIFT_NECK
            data.ctrl[head_act] = LIFT_HEAD
            data.qfrc_applied[beak_dof] = CLAMP_TORQUE

        mujoco.mj_step(model, data)
        sim_t += SIM_DT

        # Check frame capture
        for target_t, (fname, cam, label) in captured_targets.items():
            if not captured_flags[target_t] and abs(t - target_t) < SIM_DT * 0.6:
                captured_flags[target_t] = True
                renderer.update_scene(data, camera=cam, scene_option=scene_opt)
                img = renderer.render()
                out_path = os.path.join(ARTIFACTS_DIR, fname)
                imageio.imwrite(out_path, img)
                bag_z = data.xpos[bag_bid, 2]
                beak_q = data.qpos[beak_adr]
                print(f"Captured {label} -> {out_path} (bag_Z={bag_z*1000:.1f}mm, beak_q={beak_q:.3f} rad)")

                # Also capture mouth close-up at t=2.5s
                if target_t == 2.5:
                    renderer.update_scene(data, camera=cam_mouth, scene_option=scene_opt)
                    img_cu = renderer.render()
                    cu_path = os.path.join(ARTIFACTS_DIR, "proof_5_mouth_closeup.png")
                    imageio.imwrite(cu_path, img_cu)
                    print(f"Captured Mouth Close-up -> {cu_path}")

    print("All photographic proof images captured successfully!")

if __name__ == "__main__":
    main()
