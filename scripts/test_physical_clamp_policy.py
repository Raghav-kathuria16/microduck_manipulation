#!/usr/bin/env python3
"""Physical Clamping Evaluation for MicroDuck with No Weld Assumption.

Features:
- Loads non-welded model: scene_loadcarry_physical_clamp.xml
- Spawns barbell cylinder inside the beak mouth cavity
- Applies constant clamping torque (default: 0.10 N·m) to the beak actuator
- Steps the walking policy (PPO Head-Up v3)
- Evaluates cylinder retention, slip, and dynamic forces
- Records high-definition MP4 video and/or launches native interactive GUI
"""

import argparse
import os
import sys
import time
import imageio
import mujoco
import numpy as np

# Ensure paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))

from infer_policy import PolicyInference, DEFAULT_POSE

PHYSICAL_CLAMP_XML = os.path.join(
    _EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_loadcarry_physical_clamp.xml"
)
DEFAULT_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_ppo_headup_v3.onnx")


def setup_physical_clamp_sim(
    xml_path=PHYSICAL_CLAMP_XML,
    policy_path=DEFAULT_POLICY,
    payload_mass_g=100.0,
    vel_x=0.25,
    clamp_torque_nm=0.10,
    dia_mm=8.0,
    z_mm=-58.0,
    x_mm=-9.6,
):
    """Sets up physical clamping simulation with no weld constraint."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 0.005  # 200 Hz physics
    data = mujoco.MjData(model)

    cyl_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    cyl_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_geom")
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    cyl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")

    if cyl_bid >= 0:
        model.body_mass[cyl_bid] = payload_mass_g / 1000.0
    if cyl_gid >= 0:
        model.geom_size[cyl_gid, 0] = (dia_mm / 1000.0) / 2.0

    dia_m = dia_mm / 1000.0
    x_m = x_mm / 1000.0
    z_m = z_mm / 1000.0
    relpos = np.array([x_m, 0.0, z_m])
    CYL_QUAT = np.array([0.7071068, 0.7071068, 0.0, 0.0])

    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=policy_path,
        action_scale=1.0,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )
    policy.set_vel_cmd(vel_x, 0.0, 0.0)
    policy.head_offset[:] = 0.0
    policy._update_command()

    # Reset robot into default standing posture
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]

    # Initial beak open angle for zero initial penetration inside mouth cavity
    if beak_jid >= 0:
        data.qpos[model.jnt_qposadr[beak_jid]] = 1.700

    # Spawn cylinder inside mouth
    if cyl_jid >= 0:
        c_adr = model.jnt_qposadr[cyl_jid]
        mujoco.mj_forward(model, data)
        p_jaw = data.xpos[jaw_bid].copy()
        R_jaw = data.xmat[jaw_bid].reshape(3, 3).copy()
        data.qpos[c_adr:c_adr + 3] = p_jaw + R_jaw @ relpos
        q_jaw = np.empty(4)
        mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
        q_cyl = np.empty(4)
        mujoco.mju_mulQuat(q_cyl, q_jaw, CYL_QUAT)
        data.qpos[c_adr + 3:c_adr + 7] = q_cyl

    data.ctrl[:len(policy.default_pose)] = policy.default_pose
    if model.nu > len(policy.default_pose):
        data.ctrl[len(policy.default_pose):] = 1.000
    mujoco.mj_forward(model, data)

    meta = {
        "cyl_bid": cyl_bid,
        "jaw_bid": jaw_bid,
        "beak_jid": beak_jid,
        "beak_dof": model.jnt_dofadr[beak_jid] if beak_jid >= 0 else -1,
        "beak_qpos": model.jnt_qposadr[beak_jid] if beak_jid >= 0 else -1,
        "spawn_relpos": relpos,
        "clamp_torque_nm": clamp_torque_nm,
        "payload_mass_g": payload_mass_g,
    }
    return model, data, policy, meta


def run_and_record(
    model,
    data,
    policy,
    meta,
    video_path,
    num_steps=300,
    fps=30,
):
    """Runs simulation, records video, and tracks physics retention metrics."""
    print(f"\n=======================================================")
    print(f" MicroDuck Physical Clamping Benchmark (NO WELD)")
    print(f" Model:        scene_loadcarry_physical_clamp.xml")
    print(f" Actuator:     Constant Clamp Torque = {meta['clamp_torque_nm']:.2f} N·m")
    print(f" Payload Mass: {meta['payload_mass_g']:.1f} g")
    print(f" Policy:       PPO Head-Up v3")
    print(f" Video Output: {video_path}")
    print(f"=======================================================\n")

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 1
    camera.distance = 0.55
    camera.elevation = -5.0
    camera.azimuth = 120.0

    frames = []
    beak_dof = meta["beak_dof"]
    beak_qpos = meta["beak_qpos"]
    cyl_bid = meta["cyl_bid"]
    jaw_bid = meta["jaw_bid"]
    spawn_relpos = meta["spawn_relpos"]
    clamp_torque = meta["clamp_torque_nm"]

    dropped = False
    drop_time = None
    drop_step = None

    print(f"Running {num_steps} policy steps (4 physics sub-steps each at 200 Hz)...")

    for step in range(num_steps):
        action = policy.infer()
        policy.apply_action(action)

        # 4 physics sub-steps = 20 ms per policy step
        for _ in range(4):
            # Constant clamping torque: negative torque closes lower beak
            data.qfrc_applied[beak_dof] = -clamp_torque
            mujoco.mj_step(model, data)

        # Track cylinder relative to mouth
        p_cyl = data.xpos[cyl_bid]
        p_jaw = data.xpos[jaw_bid]
        R_jaw = data.xmat[jaw_bid].reshape(3, 3)
        rel_pos = R_jaw.T @ (p_cyl - p_jaw)
        dist_shift = np.linalg.norm(rel_pos - spawn_relpos)

        if not dropped and dist_shift > 0.05:  # shifted > 50 mm from mouth
            dropped = True
            drop_step = step
            drop_time = data.time
            print(f"-> Cylinder dropped at t = {drop_time:.3f} s (step {drop_step})!")

        if step % 25 == 0:
            status = "CLAMPED" if not dropped else "DROPPED"
            beak_deg = np.degrees(data.qpos[beak_qpos])
            print(
                f"[{data.time:5.2f}s | Step {step:3d}] Status: {status:7s} | Beak: {beak_deg:5.1f}° | Shift: {dist_shift*1000:5.1f} mm | Duck Vx: {data.qvel[0]:5.2f} m/s"
            )

        if step % 2 == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    writer = imageio.get_writer(video_path, fps=fps, quality=9)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"\n[record] Saved MP4 video ({len(frames)} frames) to: {video_path}")

    # Summary analysis
    print(f"\n=======================================================")
    print(f" SUMMARY PERFORMANCE METRICS")
    print(f"=======================================================")
    if dropped:
        retention_ratio = (drop_step / num_steps) * 100.0
        print(f" Result:                CYLINDER DROPPED")
        print(f" Duration Clamped:      {drop_time:.3f} seconds ({drop_step} steps, {retention_ratio:.1f}% of run)")
        print(f" Root Cause Analysis:   Torque of {clamp_torque:.2f} N·m is insufficient to counter the gravitational & dynamic moment of the {meta['payload_mass_g']:.0f}g payload.")
    else:
        print(f" Result:                SUCCESSFULLY HELD ENTIRE RUN")
        print(f" Duration Clamped:      {data.time:.3f} seconds (100.0% of run)")
    print(f"=======================================================\n")
    return not dropped


def run_gui(model, data, policy, meta):
    """Runs interactive native MuJoCo viewer."""
    import mujoco.viewer
    beak_dof = meta["beak_dof"]
    clamp_torque = meta["clamp_torque_nm"]

    print("\n[gui] Launching native interactive MuJoCo viewer window...")
    print("Press SPACE to pause/resume, Double Click on robot to track, Backspace to reset, ESC to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.58
        viewer.cam.elevation = -6.0
        viewer.cam.azimuth = 120.0
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1

        prev_time = data.time
        step = 0
        while viewer.is_running():
            step_start = time.time()
            if data.time - prev_time >= 0.02:
                prev_time = data.time
                action = policy.infer()
                policy.apply_action(action)
                step += 1

            # Apply constant clamping torque
            data.qfrc_applied[beak_dof] = -clamp_torque
            mujoco.mj_step(model, data)
            viewer.sync()

            dt = time.time() - step_start
            if dt < model.opt.timestep:
                time.sleep(model.opt.timestep - dt)


def main():
    parser = argparse.ArgumentParser(description="MicroDuck Physical Clamping (No Weld) Policy Evaluation")
    parser.add_argument("--xml", default=PHYSICAL_CLAMP_XML, help="Path to physical clamp XML")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="Path to ONNX policy")
    parser.add_argument("--torque", type=float, default=0.10, help="Constant beak actuator torque in N·m (default: 0.10)")
    parser.add_argument("--mass", type=float, default=100.0, help="Barbell cylinder payload mass in grams (default: 100)")
    parser.add_argument("--vel-x", type=float, default=0.25, help="Forward walking velocity command (default: 0.25 m/s)")
    parser.add_argument("--z-mm", type=float, default=-58.0, help="Initial cylinder depth in mm (default: -58.0 for centered silicone)")
    parser.add_argument("--gui", action="store_true", help="Launch interactive native MuJoCo GUI")
    parser.add_argument("--steps", type=int, default=300, help="Number of policy steps for recording")
    parser.add_argument("--output", default="physical_clamp_0.10Nm_test.mp4", help="Output MP4 path")
    args = parser.parse_args()

    out_path = os.path.abspath(args.output)
    model, data, policy, meta = setup_physical_clamp_sim(
        xml_path=args.xml,
        policy_path=args.policy,
        payload_mass_g=args.mass,
        vel_x=args.vel_x,
        clamp_torque_nm=args.torque,
        z_mm=args.z_mm,
    )

    if args.gui:
        run_gui(model, data, policy, meta)
    else:
        run_and_record(model, data, policy, meta, video_path=out_path, num_steps=args.steps)


if __name__ == "__main__":
    main()
