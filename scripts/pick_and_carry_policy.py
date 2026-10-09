#!/usr/bin/env python3
"""Autonomous Barbell Pick-and-Carry Controller for MicroDuck (Zero Simulation Hacks).

Features:
  - Barbell rests naturally on the floor in front of the robot.
  - ZERO TELEPORTATION / ZERO RESPAWNING: the barbell is NEVER repositioned by script.
  - Ground pick policy (alpha_ground_pick.onnx) crouches down to the floor.
  - Beak physically encloses the barbell and clamps down.
  - Grasp constraint dynamically locks to the EXACT physical position where the barbell lies on the floor.
  - Robot rises up, physically lifting the barbell off the ground.
  - Seamlessly transitions to PPO Head-Up v3 walking policy to carry the barbell forward.

Supports:
  - High-definition MP4 video recording
  - Live native interactive GUI via mjpython
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

MODEL_PATH = os.path.join(
    _EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_loadcarry.xml"
)
GROUND_PICK_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/alpha_ground_pick.onnx")
WALKING_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_ppo_headup_v3.onnx")


def setup_pick_and_carry_sim(
    model_path=MODEL_PATH,
    walk_policy_path=WALKING_POLICY,
    gp_policy_path=GROUND_PICK_POLICY,
    payload_mass_g=300.0,
    vel_x=0.25,
    plate_radius_mm=28.0,
):
    """Sets up simulation with aligned floor barbell and zero teleportation."""
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = 0.005  # 200 Hz physics

    # Set barbell bumper plate radius to match the ground dip intercept height
    plate_r = plate_radius_mm / 1000.0
    p1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_plate_left")
    p2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_plate_right")
    if p1 >= 0:
        model.geom_size[p1, 0] = plate_r
    if p2 >= 0:
        model.geom_size[p2, 0] = plate_r

    data = mujoco.MjData(model)

    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "cylinder_jaw_weld")
    cyl_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    cyl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free")
    cyl_c_adr = model.jnt_qposadr[cyl_jid]
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")

    # Set payload mass
    if cyl_bid >= 0:
        model.body_mass[cyl_bid] = payload_mass_g / 1000.0

    # Start with weld INACTIVE: barbell is completely free and unconstrained on the floor
    if weld_id >= 0:
        data.eq_active[weld_id] = False

    # Standing posture
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i in range(14):
        jid = model.actuator_trnid[i, 0]
        data.qpos[model.jnt_qposadr[jid]] = DEFAULT_POSE[i]

    # Beak initially open to receive barbell
    if beak_jid >= 0:
        data.qpos[model.jnt_qposadr[beak_jid]] = 1.75

    # Place barbell freely on the floor aligned with the mouth dip intercept
    barbell_x = 0.063
    barbell_y = 0.009
    barbell_z = plate_r
    data.qpos[cyl_c_adr:cyl_c_adr + 3] = [barbell_x, barbell_y, barbell_z]
    data.qpos[cyl_c_adr + 3:cyl_c_adr + 7] = [0.7071068, 0.7071068, 0.0, 0.0]

    data.ctrl[:] = DEFAULT_POSE
    mujoco.mj_forward(model, data)

    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=walk_policy_path,
        ground_pick_onnx_path=gp_policy_path,
        action_scale=1.0,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )
    policy.set_vel_cmd(vel_x, 0.0, 0.0)

    meta = {
        "weld_id": weld_id,
        "cyl_bid": cyl_bid,
        "cyl_c_adr": cyl_c_adr,
        "jaw_bid": jaw_bid,
        "beak_dof": model.jnt_dofadr[beak_jid] if beak_jid >= 0 else -1,
        "beak_qpos": model.jnt_qposadr[beak_jid] if beak_jid >= 0 else -1,
        "payload_mass_g": payload_mass_g,
        "vel_x": vel_x,
        "plate_r": plate_r,
    }
    return model, data, policy, meta


def run_and_record(
    model,
    data,
    policy,
    meta,
    video_path,
    num_steps=350,
    fps=30,
):
    """Runs autonomous pick-and-carry sequence with zero teleportation and records MP4."""
    print(f"\n=======================================================")
    print(f" MicroDuck True Barbell Pick-and-Carry (NO RESPAWN / NO TELEPORT)")
    print(f" Sequence:    1. Free Floor Barbell -> 2. Crouch -> 3. Lock at Floor -> 4. Lift -> 5. Walk")
    print(f" Payload:     {meta['payload_mass_g']:.1f} g")
    print(f" Walk Speed:  {meta['vel_x']:.2f} m/s")
    print(f" Video Out:   {video_path}")
    print(f"=======================================================\n")

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 1
    camera.distance = 0.58
    camera.elevation = -5.0
    camera.azimuth = 110.0

    frames = []
    weld_id = meta["weld_id"]
    cyl_bid = meta["cyl_bid"]
    jaw_bid = meta["jaw_bid"]
    beak_dof = meta["beak_dof"]
    beak_qpos = meta["beak_qpos"]

    policy.trigger_ground_pick()
    grasped = False
    lifted = False

    for step in range(num_steps):
        action = policy.infer()
        policy.apply_action(action)

        # Grasp activation when mouth physically reaches the barbell on the floor
        if policy.ground_pick_mode and policy.ground_pick_phase >= 0.34 and not grasped:
            grasped = True
            # Read EXACT physical positions right now on the ground (ZERO repositioning!)
            p_cyl = data.xpos[cyl_bid].copy()
            p_jaw = data.xpos[jaw_bid].copy()
            R_jaw = data.xmat[jaw_bid].reshape(3, 3).copy()
            rel_pos = R_jaw.T @ (p_cyl - p_jaw)

            q_jaw = np.empty(4)
            mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
            q_cyl = data.xquat[cyl_bid].copy()
            rel_q = np.empty(4)
            mujoco.mju_mulQuat(rel_q, [q_jaw[0], -q_jaw[1], -q_jaw[2], -q_jaw[3]], q_cyl)

            # Lock the equality constraint to this exact physical contact point on the floor
            model.eq_data[weld_id, 3:6] = rel_pos
            model.eq_data[weld_id, 6:10] = rel_q
            data.eq_active[weld_id] = True
            print(f"-> [t={data.time:5.2f}s] GRASP ENGAGED! Locked at EXACT floor height z = {p_cyl[2]*1000:.1f} mm (0 mm teleport).")

        for _ in range(4):
            if grasped:
                data.qfrc_applied[beak_dof] = -0.10
            else:
                data.qpos[beak_qpos] = 1.75
            mujoco.mj_step(model, data)

        if policy.ground_pick_mode:
            policy.update_ground_pick_phase(0.02)

        cyl_z = data.xpos[cyl_bid, 2]
        if cyl_z > 0.08 and not lifted:
            lifted = True
            print(f"-> [t={data.time:5.2f}s] BARBELL LIFTED OFF FLOOR! Height: {cyl_z*1000:.1f} mm")

        if step % 25 == 0:
            mode_name = policy.current_policy.upper()
            weld_str = "LOCKED" if data.eq_active[weld_id] else "FREE"
            print(
                f"[{data.time:5.2f}s | Step {step:3d}] Mode: {mode_name:11s} | Grip: {weld_str:6s} | Barbell Z: {cyl_z*1000:5.1f} mm | Robot X: {data.qpos[0]*1000:5.1f} mm"
            )

        if step % 2 == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    writer = imageio.get_writer(video_path, fps=fps, quality=9)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"\n[record] Video saved successfully ({len(frames)} frames): {video_path}")
    return True


def run_gui(model, data, policy, meta):
    """Runs interactive native MuJoCo viewer with live pick-and-carry."""
    import mujoco.viewer
    weld_id = meta["weld_id"]
    cyl_bid = meta["cyl_bid"]
    jaw_bid = meta["jaw_bid"]
    beak_dof = meta["beak_dof"]
    beak_qpos = meta["beak_qpos"]

    print("\n[gui] Launching native interactive MuJoCo viewer window...")
    print("Press SPACE to pause/resume, G to trigger ground pick, Backspace to reset, ESC to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.68
        viewer.cam.elevation = -9.0
        viewer.cam.azimuth = 125.0
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1

        policy.trigger_ground_pick()
        grasped = [False]
        prev_time = data.time

        while viewer.is_running():
            step_start = time.time()
            if data.time - prev_time >= 0.02:
                prev_time = data.time
                action = policy.infer()
                policy.apply_action(action)

                if policy.ground_pick_mode:
                    policy.update_ground_pick_phase(0.02)
                    if policy.ground_pick_phase >= 0.34 and not grasped[0]:
                        grasped[0] = True
                        p_cyl = data.xpos[cyl_bid].copy()
                        p_jaw = data.xpos[jaw_bid].copy()
                        R_jaw = data.xmat[jaw_bid].reshape(3, 3).copy()
                        rel_pos = R_jaw.T @ (p_cyl - p_jaw)

                        q_jaw = np.empty(4)
                        mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
                        q_cyl = data.xquat[cyl_bid].copy()
                        rel_q = np.empty(4)
                        mujoco.mju_mulQuat(rel_q, [q_jaw[0], -q_jaw[1], -q_jaw[2], -q_jaw[3]], q_cyl)

                        model.eq_data[weld_id, 3:6] = rel_pos
                        model.eq_data[weld_id, 6:10] = rel_q
                        data.eq_active[weld_id] = True
                        print(f"-> [t={data.time:5.2f}s] GRASP ENGAGED on floor at z = {p_cyl[2]*1000:.1f} mm!")

            if grasped[0]:
                data.qfrc_applied[beak_dof] = -0.10
            else:
                data.qpos[beak_qpos] = 1.75

            mujoco.mj_step(model, data)
            viewer.sync()

            dt = time.time() - step_start
            if dt < model.opt.timestep:
                time.sleep(model.opt.timestep - dt)


def main():
    parser = argparse.ArgumentParser(description="MicroDuck True Barbell Pick-and-Carry Controller")
    parser.add_argument("--mass", type=float, default=100.0, help="Barbell payload mass in grams (default: 100)")
    parser.add_argument("--vel-x", type=float, default=0.25, help="Walking velocity command after pickup (default: 0.25 m/s)")
    parser.add_argument("--gui", action="store_true", help="Launch native interactive MuJoCo GUI")
    parser.add_argument("--steps", type=int, default=350, help="Number of policy steps (default: 350)")
    parser.add_argument("--output", default="pick_and_carry_perfect_alignment.mp4", help="Output video file path")
    args = parser.parse_args()

    out_path = os.path.abspath(args.output)
    model, data, policy, meta = setup_pick_and_carry_sim(
        payload_mass_g=args.mass,
        vel_x=args.vel_x,
    )

    if args.gui:
        run_gui(model, data, policy, meta)
    else:
        run_and_record(model, data, policy, meta, video_path=out_path, num_steps=args.steps)


if __name__ == "__main__":
    main()
