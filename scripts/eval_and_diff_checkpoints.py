#!/usr/bin/env python3
"""Automated Checkpoint Evaluation & Benchmarking against Baseline / Fix #1.

Evaluates a candidate policy across 300 control steps (6.0 seconds) in native MuJoCo
with a 300g barbell clamped in the beak, diffs headline metrics against the Fix #1
benchmark, and emits an automated Stage-Gate verdict.
"""

import argparse
import math
import os
import sys
import numpy as np
import mujoco

_MANIP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")
sys.path.insert(0, os.path.join(_MANIP_DIR, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))

from infer_policy import PolicyInference, DEFAULT_POSE
from eval_loadcarry_native_mujoco import compute_beak_and_weld_kinematics, CYLINDER_WELD_RELQUAT


FIX1_BENCHMARK = {
    "name": "Fix #1 Benchmark (model_149.pt)",
    "survival_steps": 300,
    "is_fallen": False,
    "dist_walked": 2.956,
    "avg_speed": 0.493,
    "pitch_deg": 2.3,
}


def evaluate_policy(
    policy_path: str,
    num_steps: int = 300,
    vel_x: float = 0.25,
    vel_y: float = 0.0,
    vel_ang: float = 0.0,
    payload_g: float = 300.0,
) -> dict:
    model_path = os.path.join(_EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_loadcarry.xml")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model XML not found at {model_path}")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy ONNX not found at {policy_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = 0.005  # 200 Hz
    data = mujoco.MjData(model)

    cyl_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    if cyl_body_id >= 0:
        model.body_mass[cyl_body_id] = payload_g / 1000.0

    beak_angle, weld_relpos = compute_beak_and_weld_kinematics(0.008, -0.068)
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "cylinder_jaw_weld")
    if weld_id >= 0:
        model.eq_data[weld_id, 3:6] = weld_relpos
        model.eq_data[weld_id, 6:10] = CYLINDER_WELD_RELQUAT

    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=policy_path,
        action_scale=1.0,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )
    policy.set_vel_cmd(vel_x, vel_y, vel_ang)
    policy.head_offset[:] = 0.0
    policy._update_command()

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = DEFAULT_POSE[i]

    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    if beak_jid >= 0:
        data.qpos[model.jnt_qposadr[beak_jid]] = beak_angle

    data.ctrl[:] = DEFAULT_POSE
    mujoco.mj_forward(model, data)

    start_x = float(data.qpos[0])
    is_fallen = False
    fall_step = None
    pitch_history = []
    neck_pitch_history = []

    for step in range(num_steps):
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            mujoco.mj_step(model, data)

        z = float(data.qpos[2])
        grav = policy.get_projected_gravity()
        pitch_deg = math.degrees(math.atan2(grav[0], -grav[2]))
        roll_deg = math.degrees(math.atan2(grav[1], -grav[2]))
        pitch_history.append(pitch_deg)

        neck_qpos = data.qpos[policy.joint_qpos_indices[5]]
        neck_pitch_history.append(math.degrees(neck_qpos))

        # Rigorous physical fall criteria:
        # 1. Torso height collapse: z < 0.095m (nominal walking height is 0.125m)
        # 2. Extreme pitch or roll tilt: |pitch| > 20° or |roll| > 20°
        # 3. Inversion / tumbling: grav[2] > -0.50
        # 4. Non-foot collision with ground (jaw, trunk, cylinder contacting floor)
        has_bad_ground_contact = False
        for c in range(data.ncon):
            con = data.contact[c]
            if con.geom1 == 0 or con.geom2 == 0:  # geom 0 is floor
                other_geom = con.geom2 if con.geom1 == 0 else con.geom1
                bid = model.geom_bodyid[other_geom]
                bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if bname not in ["ankle_left", "ankle_right"]:
                    has_bad_ground_contact = True
                    break

        if (z < 0.095 or abs(pitch_deg) > 20.0 or abs(roll_deg) > 20.0 or grav[2] > -0.50 or has_bad_ground_contact) and not is_fallen:
            is_fallen = True
            fall_step = step
            break

    survival_steps = fall_step if is_fallen else num_steps
    final_x = float(data.qpos[0])
    dist_walked = final_x - start_x
    time_elapsed = survival_steps * 0.02
    avg_speed = dist_walked / time_elapsed if time_elapsed > 0 else 0.0

    return {
        "policy_path": policy_path,
        "is_fallen": is_fallen,
        "survival_steps": survival_steps,
        "total_steps": num_steps,
        "dist_walked": dist_walked,
        "avg_speed": avg_speed,
        "final_pitch_deg": pitch_history[-1] if pitch_history else 0.0,
        "mean_pitch_deg": float(np.mean(pitch_history)) if pitch_history else 0.0,
        "rms_pitch_deg": float(np.std(pitch_history)) if pitch_history else 0.0,
        "mean_neck_deg": float(np.mean(neck_pitch_history)) if neck_pitch_history else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and Diff Checkpoint")
    default_candidate = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_residual_combined.onnx")
    parser.add_argument("--candidate", "--candidate-model", dest="candidate", default=default_candidate, help="Path to candidate ONNX policy (defaults to current best)")
    parser.add_argument("--base-model", default=None, help="Path to base ONNX policy to diff against (optional, defaults to Fix #1 benchmark)")
    parser.add_argument("--new-run-dir", default=None, help="Path to run log directory (optional)")
    parser.add_argument("--steps", type=int, default=300, help="Number of control steps to simulate")
    parser.add_argument("--vel-x", type=float, default=0.25, help="Target forward velocity in m/s")
    parser.add_argument("--vel-y", type=float, default=0.0, help="Target lateral velocity in m/s")
    parser.add_argument("--vel-ang", type=float, default=0.0, help="Target yaw angular velocity in rad/s")
    parser.add_argument("--payload", type=float, default=300.0, help="Barbell payload mass in grams")
    args = parser.parse_args()

    benchmark = FIX1_BENCHMARK.copy()
    if args.base_model and os.path.exists(args.base_model):
        base_res = evaluate_policy(
            args.base_model,
            num_steps=args.steps,
            vel_x=args.vel_x,
            vel_y=args.vel_y,
            vel_ang=args.vel_ang,
            payload_g=args.payload,
        )
        benchmark = {
            "name": os.path.basename(args.base_model),
            "survival_steps": base_res["survival_steps"],
            "is_fallen": base_res["is_fallen"],
            "dist_walked": base_res["dist_walked"],
            "avg_speed": base_res["avg_speed"],
            "pitch_deg": base_res["final_pitch_deg"],
            "rms_pitch_deg": base_res["rms_pitch_deg"],
            "mean_neck_deg": base_res["mean_neck_deg"],
        }

    results = evaluate_policy(
        args.candidate,
        num_steps=args.steps,
        vel_x=args.vel_x,
        vel_y=args.vel_y,
        vel_ang=args.vel_ang,
        payload_g=args.payload,
    )

    print("\n" + "=" * 80)
    print(f"QUANTITATIVE EVALUATION REPORT: {os.path.basename(results['policy_path'])}")
    print("=" * 80)
    bench_name = benchmark["name"][:18]
    print(f"{'Metric':<25} | {bench_name:<18} | {'Candidate (Refinement)':<22} | {'Delta':<10}")
    print("-" * 80)

    # Survival
    b_surv = f"{benchmark['survival_steps']}/{args.steps} ({benchmark['survival_steps']/args.steps*100:.0f}%)"
    c_surv = f"{results['survival_steps']}/{args.steps} ({results['survival_steps']/args.steps*100:.1f}%)"
    surv_delta = f"{results['survival_steps'] - benchmark['survival_steps']:+d}"
    print(f"{'Survival Steps':<25} | {b_surv:<18} | {c_surv:<22} | {surv_delta:<10}")

    # Distance
    b_dist = f"{benchmark['dist_walked']:.3f} m"
    c_dist = f"{results['dist_walked']:.3f} m"
    dist_delta = f"{results['dist_walked'] - benchmark['dist_walked']:+.3f} m"
    print(f"{'Distance Walked':<25} | {b_dist:<18} | {c_dist:<22} | {dist_delta:<10}")

    # Speed
    b_spd = f"{benchmark['avg_speed']:.3f} m/s"
    c_spd = f"{results['avg_speed']:.3f} m/s"
    spd_delta = f"{results['avg_speed'] - benchmark['avg_speed']:+.3f} m/s"
    print(f"{'Average Speed':<25} | {b_spd:<18} | {c_spd:<22} | {spd_delta:<10}")

    # Pitch
    b_pitch = f"{benchmark['pitch_deg']:.1f}°"
    c_pitch = f"{results['final_pitch_deg']:.1f}°"
    pitch_delta = f"{results['final_pitch_deg'] - benchmark['pitch_deg']:+.1f}°"
    print(f"{'Final Pitch':<25} | {b_pitch:<18} | {c_pitch:<22} | {pitch_delta:<10}")

    # RMS Pitch & Neck
    b_rms = f"{benchmark.get('rms_pitch_deg', 3.2):.2f}°"
    b_neck = f"{benchmark.get('mean_neck_deg', -45.2):.1f}°"
    c_rms = f"{results['rms_pitch_deg']:.2f}°"
    c_neck = f"{results['mean_neck_deg']:.1f}°"
    print(f"{'Pitch StdDev (RMS)':<25} | {b_rms:<18} | {c_rms:<22} | -")
    print(f"{'Mean Neck Pitch':<25} | {b_neck:<18} | {c_neck:<22} | -")

    print("-" * 80)

    # Automated Gate
    verdict = "ACCEPT (Stage-Gate Passed)"
    reasons = []
    if results["is_fallen"]:
        verdict = "REJECT & ROLLBACK"
        reasons.append(f"Robot fell at step {results['survival_steps']}/{args.steps}")
    elif results["avg_speed"] > args.vel_x + 0.15:
        reasons.append(f"Overspeed persist ({results['avg_speed']:.2f} > {args.vel_x + 0.15:.2f} m/s)")
    elif results["avg_speed"] < max(0.06, args.vel_x - 0.12) and args.vel_x > 0.08:
        verdict = "REJECT & ROLLBACK"
        reasons.append(f"Underspeed/stall ({results['avg_speed']:.2f} < {max(0.06, args.vel_x - 0.12):.2f} m/s)")

    print(f"VERDICT: {verdict}")
    if reasons:
        print("Notes:", "; ".join(reasons))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
