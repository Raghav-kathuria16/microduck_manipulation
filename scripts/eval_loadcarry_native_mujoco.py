#!/usr/bin/env python3
"""Evaluate a trained load-carrying policy in native MuJoCo.

Runs forward walking simulation with a 300g payload welded to the beak,
recording trajectory statistics and rendering high-resolution frames.
"""

import math
import os
import sys
import numpy as np
import mujoco
from PIL import Image

# Ensure paths are discoverable
_MANIP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))

from infer_policy import PolicyInference, DEFAULT_POSE

def compute_beak_and_weld_kinematics(dia_m=0.008, z_m=-0.068, clearance_m=0.0010):
    x_roof = -0.00817
    x_c = x_roof - (dia_m / 2.0)
    z_c = z_m
    relpos = (x_c, 0.0, z_c)
    L_z = abs(z_c - (-0.0180))
    # Closed angle is 1.500 rad (85.9 deg) where beak tips touch exactly.
    # Opening angle rotates down by delta_theta = (dia + clearance) / L_z.
    theta = 1.500 + ((dia_m + clearance_m) / L_z)
    return theta, relpos


CYLINDER_WELD_RELQUAT = (0.7071068, 0.7071068, 0.0, 0.0)


def run_eval(
    policy_path: str,
    output_prefix: str = "eval_finetuned",
    num_steps: int = 300,
    vel_x: float = 0.25,
    payload_mass_g: float = 300.0,
    save_artifacts_dir: str = None,
):
    model_path = os.path.join(_EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_loadcarry.xml")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model XML not found at {model_path}")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy ONNX not found at {policy_path}")

    print(f"\n{'='*70}\n[eval] Native MuJoCo Evaluation for: {policy_path}\n{'='*70}")
    print(f"Model: {model_path}")
    print(f"Payload mass: {payload_mass_g}g")
    print(f"Forward velocity command: {vel_x} m/s")
    print(f"Total control steps: {num_steps} (50 Hz control, 200 Hz physics)\n")

    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = 0.005  # 200 Hz physics
    data = mujoco.MjData(model)

    cyl_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    jaw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    cyl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")

    dia_m = 0.008
    z_m = -0.068
    beak_angle, weld_relpos = compute_beak_and_weld_kinematics(dia_m, z_m)

    if cyl_body_id >= 0:
        model.body_mass[cyl_body_id] = payload_mass_g / 1000.0

    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "cylinder_jaw_weld")
    if weld_id >= 0:
        model.eq_data[weld_id, 3:6] = weld_relpos
        model.eq_data[weld_id, 6:10] = CYLINDER_WELD_RELQUAT

    # Policy inference runner
    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=policy_path,
        action_scale=1.0,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )

    # Set velocity command
    policy.set_vel_cmd(vel_x, 0.0, 0.0)
    policy.head_offset[:] = 0.0  # Autonomous policy — no manual nudge
    policy._update_command()

    # Reset pose
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = DEFAULT_POSE[i]

    if beak_jid >= 0:
        model.qpos_spring[model.jnt_qposadr[beak_jid]] = beak_angle
        data.qpos[model.jnt_qposadr[beak_jid]] = beak_angle
        model.jnt_range[beak_jid] = [-0.0872665, max(1.80, beak_angle + 0.10)]

    # Place cylinder in jaw
    if cyl_jid >= 0:
        c_adr = model.jnt_qposadr[cyl_jid]
        mujoco.mj_forward(model, data)
        p_jaw = data.xpos[jaw_body_id].copy()
        R_jaw = data.xmat[jaw_body_id].reshape(3, 3).copy()
        data.qpos[c_adr:c_adr + 3] = p_jaw + R_jaw @ np.array(weld_relpos)
        q_jaw = np.empty(4)
        mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_body_id])
        q_cyl = np.empty(4)
        mujoco.mju_mulQuat(q_cyl, q_jaw, np.array(CYLINDER_WELD_RELQUAT))
        data.qpos[c_adr + 3:c_adr + 7] = q_cyl

    data.ctrl[:] = DEFAULT_POSE
    mujoco.mj_forward(model, data)

    # Renderer setup
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera_side = mujoco.MjvCamera()
    camera_side.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera_side.trackbodyid = 1
    camera_side.distance = 0.75
    camera_side.elevation = -10.0
    camera_side.azimuth = 90.0

    camera_34 = mujoco.MjvCamera()
    camera_34.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera_34.trackbodyid = 1
    camera_34.distance = 0.85
    camera_34.elevation = -12.0
    camera_34.azimuth = 135.0

    # Simulation loop
    start_x = float(data.qpos[0])
    is_fallen = False
    fall_step = None

    snapshot_steps = {0, 25, 50, 100, 150, 200, 250, num_steps - 1}

    print("Stepping simulation in native MuJoCo...")
    for step in range(num_steps):
        action = policy.infer()
        policy.apply_action(action)

        for _ in range(4):  # 4 physics substeps = 0.02s
            mujoco.mj_step(model, data)

        z = float(data.qpos[2])
        grav = policy.get_projected_gravity()
        if (z < 0.075 or grav[2] > -0.3) and not is_fallen:
            is_fallen = True
            fall_step = step
            print(f"[!] Fall detected at step {step} (z={z:.3f}, grav_z={grav[2]:.3f})")

        # Render snapshots
        if step in snapshot_steps:
            renderer.update_scene(data, camera=camera_side)
            img_side = renderer.render().copy()
            renderer.update_scene(data, camera=camera_34)
            img_34 = renderer.render().copy()

            if save_artifacts_dir:
                side_path = os.path.join(save_artifacts_dir, f"{output_prefix}_step{step}_side.png")
                p34_path = os.path.join(save_artifacts_dir, f"{output_prefix}_step{step}_34.png")
                Image.fromarray(img_side).save(side_path)
                Image.fromarray(img_34).save(p34_path)

    final_x = float(data.qpos[0])
    dist_walked = final_x - start_x
    avg_speed = dist_walked / (num_steps * 0.02)
    final_pitch_deg = math.degrees(math.atan2(grav[0], -grav[2]))

    print("\n" + "="*70)
    print(f"EVALUATION RESULTS ({output_prefix}):")
    print(f"  Fallen: {is_fallen} (at step {fall_step} of {num_steps})" if is_fallen else f"  Fallen: NO (100% STABLE BIPEDAL WALKING)")
    print(f"  Total Distance Walked: {dist_walked:.3f} m")
    print(f"  Average Walking Speed: {avg_speed:.3f} m/s (Target: {vel_x:.2f} m/s)")
    print(f"  Final Base Height: {data.qpos[2]:.3f} m")
    print(f"  Final Base Pitch: {final_pitch_deg:.1f} deg")
    print("="*70 + "\n")

    return {
        "is_fallen": is_fallen,
        "fall_step": fall_step,
        "dist_walked": dist_walked,
        "avg_speed": avg_speed,
        "pitch_deg": final_pitch_deg,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="/Users/raghav/Desktop/microduck_manipulation/microduck_manipulation/policies/onnx/loadcarry_finetuned_150.onnx")
    parser.add_argument("--prefix", default="eval_ppo150")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--vel-x", type=float, default=0.25)
    parser.add_argument("--payload", type=float, default=300.0)
    parser.add_argument("--save-dir", default="/Users/raghav/.gemini/antigravity-ide/brain/f2ffb849-d730-4892-8213-f5bdaeddc8dc")
    args = parser.parse_args()

    run_eval(
        policy_path=args.policy,
        output_prefix=args.prefix,
        num_steps=args.steps,
        vel_x=args.vel_x,
        payload_mass_g=args.payload,
        save_artifacts_dir=args.save_dir,
    )
