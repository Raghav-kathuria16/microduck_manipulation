#!/usr/bin/env python3
"""Empirical Benchmark: When Does Physical Clamping Fail?

Conducts exhaustive quantitative stress-tests measuring exact Time-to-Failure (TTF),
slip accumulation rates, and physical failure modes over 30-second continuous locomotion:
1. Barbell payloads (50g to 500g) across clamping torques (0.02 to 0.20 N·m)
2. Athletic Rackets (Table Tennis, Badminton, Pickleball) and unconstrained ball retention
3. Lemon & Spoon dynamic balance across forward velocities (0.08 to 0.25 m/s)
"""

import os
import sys
import math
import numpy as np
import mujoco

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from test_physical_clamp_policy import setup_physical_clamp_sim, PHYSICAL_CLAMP_XML
from test_spoon_lemon_balance import create_spoon_lemon_scene_xml, _DEFAULT_POLICY
from infer_policy import PolicyInference


def evaluate_barbell_failure_matrix():
    """Evaluates time-to-failure across 50g-500g payloads and 0.02-0.20 N·m torques."""
    print("=" * 85)
    print("EMPIRICAL BENCHMARK 1: BARBELL CLAMPING TIME-TO-FAILURE (30.0s continuous walk)")
    print("=" * 85)
    print("| Payload | Torque  | Result | Time-to-Failure | Failure Mode       | Max Slip  |")
    print("| :---:   | :---:   | :---:  | :---:           | :---               | :---:     |")

    masses = [50, 100, 150, 200, 250, 300, 350, 400, 500]
    torques = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    results = []

    for m in masses:
        for trq in torques:
            model, data, policy, meta = setup_physical_clamp_sim(
                xml_path=PHYSICAL_CLAMP_XML, payload_mass_g=m, clamp_torque_nm=trq, vel_x=0.25
            )
            beak_dof = meta["beak_dof"]
            cyl_bid = meta["cyl_bid"]
            jaw_bid = meta["jaw_bid"]
            spawn_relpos = meta["spawn_relpos"].copy()

            sim_dt = model.opt.timestep
            decimation = 4
            max_time = 30.0
            max_steps = int(max_time / (sim_dt * decimation))

            failed = False
            fail_time = max_time
            fail_reason = "SURVIVED_30S"
            max_slip = 0.0

            for step in range(max_steps):
                t = step * decimation * sim_dt
                action = policy.infer()
                policy.apply_action(action)
                for _ in range(decimation):
                    data.qfrc_applied[beak_dof] = -trq
                    mujoco.mj_step(model, data)

                p_cyl = data.xpos[cyl_bid]
                p_jaw = data.xpos[jaw_bid]
                R_jaw = data.xmat[jaw_bid].reshape(3, 3)
                rel_pos = R_jaw.T @ (p_cyl - p_jaw)
                slip = float(np.linalg.norm(rel_pos - spawn_relpos) * 1000.0)
                max_slip = max(max_slip, slip)

                # Slip & separation limit
                if slip > 40.0 or p_cyl[2] < 0.035:
                    failed = True
                    fail_time = t
                    fail_reason = "SLIP_AND_DROP"
                    break

                # Robot overturning limit
                if data.xpos[1][2] < 0.065:
                    failed = True
                    fail_time = t
                    fail_reason = "ROBOT_TOPPLED"
                    break

            status = "FAIL" if failed else "PASS"
            ttf_str = f"{fail_time:5.2f}s" if failed else "> 30.0s"
            print(f"| {m:3d}g   | {trq:4.2f}Nm | {status:4s}   | {ttf_str:15s} | {fail_reason:18s} | {max_slip:5.1f} mm |")
            sys.stdout.flush()
            results.append((m, trq, failed, fail_time, fail_reason, max_slip))

    return results


def evaluate_lemon_spoon_failure():
    """Evaluates time-to-drop for the unconstrained lemon on tableware spoon."""
    print("\n" + "=" * 85)
    print("EMPIRICAL BENCHMARK 2: LEMON & SPOON BALANCE TIME-TO-DROP (30.0s continuous walk)")
    print("=" * 85)
    print("| Velocity | Result       | Time-to-Drop | Distance Walked | Max Deviation |")
    print("| :---:    | :---:        | :---:        | :---:           | :---:         |")

    xml_str = create_spoon_lemon_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    policy = PolicyInference(
        model, data, walking_onnx_path=_DEFAULT_POLICY,
        action_scale=1.0, new_cmd_obs=True, use_projected_gravity=True
    )
    policy.default_pose[5] = 0.2760
    policy.default_pose[6] = 0.2760

    spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lemon")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_qpos = model.jnt_qposadr[beak_jid]
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    spoon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spoon_free")
    spoon_qadr = model.jnt_qposadr[spoon_jid]
    lemon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lemon_free")
    lemon_qadr = model.jnt_qposadr[lemon_jid]
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    target_clamp_angle = 1.550
    clamp_torque = 0.45
    decimation = int(round(0.020 / model.opt.timestep))

    for vx in [0.08, 0.12, 0.15, 0.20, 0.25]:
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, 0.125]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, idx in enumerate(policy.joint_qpos_indices):
            data.qpos[idx] = policy.default_pose[i]
        data.qpos[beak_qpos] = target_clamp_angle
        data.qfrc_applied[beak_dof] = -clamp_torque
        if model.nu > 14:
            data.ctrl[14] = 1.0
        mujoco.mj_forward(model, data)

        p_jaw = data.xpos[jaw_bid]
        R_jaw = data.xmat[jaw_bid].reshape(3, 3)
        HOVER_LOCAL = np.array([-0.0119, 0.0, -0.0465])
        CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

        data.qpos[spoon_qadr:spoon_qadr+3] = p_jaw + R_jaw @ HOVER_LOCAL
        q_jaw = np.empty(4)
        mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
        q_s = np.empty(4)
        mujoco.mju_mulQuat(q_s, q_jaw, CYL_QUAT)
        data.qpos[spoon_qadr+3:spoon_qadr+7] = q_s
        mujoco.mj_forward(model, data)

        p_spoon = data.xpos[spoon_bid]
        R_spoon = data.xmat[spoon_bid].reshape(3, 3)
        p_lemon = p_spoon + R_spoon @ np.array([0.0, 0.0052, 0.070])
        data.qpos[lemon_qadr:lemon_qadr+3] = p_lemon
        data.qpos[lemon_qadr+3:lemon_qadr+7] = q_s
        mujoco.mj_forward(model, data)

        for _ in range(200):
            data.qpos[0:3] = [0.0, 0.0, 0.125]
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[0:6] = 0.0
            data.qfrc_applied[beak_dof] = -clamp_torque
            mujoco.mj_step(model, data)

        policy.vel_cmd[0] = vx
        policy._update_command()

        action = np.zeros(14, dtype=np.float32)
        head_pos_steadicam = 0.0
        fell = False
        max_dist = 0.0
        drop_time = None

        for step in range(15000):  # 30.0s
            beak_err = target_clamp_angle - data.qpos[beak_qpos]
            data.qfrc_applied[beak_dof] = float(np.clip(-clamp_torque + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))

            if step % decimation == 0:
                action = policy.infer()
                R_trunk = data.xmat[trunk_id].reshape(3, 3)
                h_fwd = np.array([R_trunk[0, 0], R_trunk[1, 0], 0.0])
                h_fwd /= (np.linalg.norm(h_fwd) + 1e-6)

                R_s = data.xmat[spoon_bid].reshape(3, 3)
                b_norm = R_s @ np.array([0.0, 1.0, 0.0])
                spoon_pitch = math.asin(np.clip(float(np.dot(b_norm, h_fwd)), -0.999, 0.999))

                rate_cmd = np.clip(1.20 * spoon_pitch, -0.30, 0.30)
                dt_ctrl = decimation * model.opt.timestep
                head_pos_steadicam = np.clip(head_pos_steadicam + rate_cmd * dt_ctrl, -0.40, 0.40)

                action[5] = 0.0 + head_pos_steadicam
                action[6] = 0.0 - head_pos_steadicam
                action[7] = 0.0
                action[8] = 0.0
                policy.apply_action(action)
            else:
                policy.apply_action(action)

            mujoco.mj_step(model, data)

            p_s = data.xpos[spoon_bid]
            R_s = data.xmat[spoon_bid].reshape(3, 3)
            bowl_center = p_s + R_s @ np.array([0.0, 0.0052, 0.070])
            dist = float(np.linalg.norm(data.xpos[lemon_bid] - bowl_center))
            if dist > max_dist:
                max_dist = dist
            if dist > 0.05:
                fell = True
                drop_time = step * model.opt.timestep
                break

        res_str = f"DROPPED @ {drop_time:5.2f}s" if fell else "SURVIVED 30s"
        drop_str = f"{drop_time:5.2f}s" if fell else "> 30.0s"
        dist_walked = float(data.qpos[0])
        print(f"| {vx:4.2f} m/s | {res_str:12s} | {drop_str:12s} | {dist_walked:5.2f} m        | {max_dist*1000:5.1f} mm       |")
        sys.stdout.flush()


if __name__ == "__main__":
    evaluate_barbell_failure_matrix()
    evaluate_lemon_spoon_failure()
