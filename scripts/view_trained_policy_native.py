#!/usr/bin/env python3
"""Interactive Native MuJoCo Viewer and Video Generator for Trained LoadCarry Policy.

Supports:
  1. Live Interactive Native MuJoCo GUI (run with mjpython on macOS)
  2. Offscreen High-Definition MP4 Video Recording (playable via QuickTime/Browser)
  3. Proper qpos0 initialization so GUI 'Reset' / Backspace preserves the standing pose!
"""

import argparse
import math
import os
import subprocess
import sys
import time
import imageio
import mujoco
import numpy as np

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


def setup_simulation(model_path, policy_path, payload_mass_g=300.0, vel_x=0.25, dia_mm=8.0, z_mm=-68.0):
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = 0.005  # 200 Hz physics
    data = mujoco.MjData(model)

    dia_m = dia_mm / 1000.0
    z_m = z_mm / 1000.0
    beak_angle, weld_relpos = compute_beak_and_weld_kinematics(dia_m, z_m)

    cyl_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    cyl_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_geom")
    jaw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    cyl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")

    if cyl_body_id >= 0:
        model.body_mass[cyl_body_id] = payload_mass_g / 1000.0

    bare_robot_mass = sum(float(model.body_mass[i]) for i in range(model.nbody) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) not in ["cylinder", "world"])
    total_sys_mass = bare_robot_mass + (payload_mass_g / 1000.0)

    print(f"[kinematics] Cyl D={dia_mm:.1f}mm, Z={z_mm:.1f}mm -> Beak Angle={beak_angle:.4f} rad ({np.degrees(beak_angle):.1f}°), relpos=({weld_relpos[0]*1000:.2f}, 0, {weld_relpos[2]*1000:.2f})mm")
    print(f"[weights] Bare Robot: {bare_robot_mass*1000:.1f}g ({bare_robot_mass:.3f}kg) | Payload: {payload_mass_g:.1f}g ({payload_mass_g/1000:.3f}kg) | Total: {total_sys_mass*1000:.1f}g ({total_sys_mass:.3f}kg)")
    print(f"[weights] Payload is {payload_mass_g/(bare_robot_mass*1000)*100:.1f}% of bare robot mass ({payload_mass_g/(total_sys_mass*1000)*100:.1f}% of total system mass)")

    if cyl_gid >= 0:
        model.geom_size[cyl_gid, 0] = dia_m / 2.0
        model.geom_rgba[cyl_gid] = [0.10, 0.10, 0.10, 1.0]

    plate_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_plate_left")
    plate_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_plate_right")
    if plate_left >= 0:
        model.geom_rgba[plate_left] = [0.12, 0.12, 0.12, 1.0]
    if plate_right >= 0:
        model.geom_rgba[plate_right] = [0.12, 0.12, 0.12, 1.0]

    # Ensure weld constraint holds the exact centered relpos with zero penetration
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "cylinder_jaw_weld")
    if weld_id >= 0:
        model.eq_data[weld_id, 3:6] = weld_relpos
        model.eq_data[weld_id, 6:10] = CYLINDER_WELD_RELQUAT

    # Ensure spring holds beak open at wide clamping angle
    if beak_jid >= 0:
        qpos_adr = model.jnt_qposadr[beak_jid]
        model.qpos_spring[qpos_adr] = beak_angle
        # Expand limit if needed
        model.jnt_range[beak_jid] = [-0.0872665, max(2.20, beak_angle + 0.20)]

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

    # Configure initial standing state
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = DEFAULT_POSE[i]

    if beak_jid >= 0:
        data.qpos[model.jnt_qposadr[beak_jid]] = beak_angle

    # Place cylinder in jaw with zero initial constraint error
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

    return model, data, policy


def record_video(model, data, policy, video_path, num_steps=350, fps=30):
    print(f"\n[record] Generating video at: {video_path}")
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 1
    camera.distance = 0.52
    camera.elevation = 2.0  # lowered camera looking level at duck side profile
    camera.azimuth = 90.0

    frames = []
    print(f"Rendering {num_steps} simulation steps...")
    for step in range(num_steps):
        action = policy.infer()
        policy.apply_action(action)

        for _ in range(4):
            mujoco.mj_step(model, data)

        if step % 2 == 0:  # 25 fps effective
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            frames.append(rgb)

    writer = imageio.get_writer(video_path, fps=fps, quality=9)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"[record] Video saved successfully ({len(frames)} frames): {video_path}")

    # Open video on macOS if running locally
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["open", video_path])
            print(f"[record] Opened video in default player.")
        except Exception as e:
            print(f"Could not open player: {e}")


def run_gui(model, data, policy, enable_web: bool = False, web_port: int = 8080):
    import mujoco.viewer
    import urllib.request
    import json
    import threading

    print("\n[gui] Launching native interactive MuJoCo viewer window...")
    print("Press SPACE to pause/resume, Double Click on robot to track, Backspace to reset, ESC to exit.\n")

    current_cmd = [0.25, 0.0, 0.0]
    stop_event = threading.Event()
    reset_pending = [False]

    cyl_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    neck_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch")

    def _post_telemetry(req):
        try:
            with urllib.request.urlopen(req, timeout=0.08) as resp:
                pass
        except Exception:
            pass

    if enable_web:
        print(f"[web] Connected to localhost teleop controller at http://localhost:{web_port}")
        last_cmd = [None, None, None]
        last_mass = [None]

        def poll_web_server():
            while not stop_event.is_set():
                try:
                    req = urllib.request.Request(f"http://localhost:{web_port}/api/command")
                    with urllib.request.urlopen(req, timeout=0.08) as response:
                        if response.status == 200:
                            cmd_data = json.loads(response.read().decode("utf-8"))

                            # Check for web UI reset request
                            if cmd_data.get("reset_requested", False):
                                reset_pending[0] = True

                            # Check for dynamic payload mass change
                            if "payload_mass" in cmd_data and cyl_body_id >= 0:
                                new_mass = float(cmd_data["payload_mass"])
                                if last_mass[0] is None or abs(new_mass - last_mass[0]) > 0.05:
                                    last_mass[0] = new_mass
                                    model.body_mass[cyl_body_id] = max(0.0001, new_mass / 1000.0)
                                    print(f"[teleop] Dynamic barbell payload mass updated to {new_mass:.1f}g ({model.body_mass[cyl_body_id]:.3f} kg)")

                            # Check for velocity commands
                            ts = float(cmd_data.get("timestamp", 0.0))
                            if ts > 0.0 or "vx" in cmd_data:
                                vx = float(cmd_data.get("vx", 0.0))
                                vy = float(cmd_data.get("vy", 0.0))
                                wz = float(cmd_data.get("wz", 0.0))
                                if [vx, vy, wz] != last_cmd:
                                    last_cmd[0], last_cmd[1], last_cmd[2] = vx, vy, wz
                                    current_cmd[0], current_cmd[1], current_cmd[2] = vx, vy, wz
                                    policy.set_vel_cmd(vx, vy, wz)
                except Exception:
                    pass
                time.sleep(0.02)

        th = threading.Thread(target=poll_web_server, daemon=True)
        th.start()

    initial_qpos = data.qpos.copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.68
        viewer.cam.elevation = -9.0
        viewer.cam.azimuth = 140.0
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1

        prev_time = data.time
        step = 0

        while viewer.is_running():
            step_start = time.time()

            # Handle reset (from web UI or native viewer reset)
            if data.time < prev_time or reset_pending[0]:
                print("[gui] Reset triggered — restoring standing pose + cylinder in beak")
                data.qpos[:] = initial_qpos.copy()
                data.ctrl[:] = DEFAULT_POSE
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                step = 0
                reset_pending[0] = False

            prev_time = data.time

            action = policy.infer()
            policy.apply_action(action)

            for _ in range(4):
                mujoco.mj_step(model, data)

            viewer.sync()

            # Send telemetry every 5 steps (10 Hz)
            if enable_web and step % 5 == 0:
                try:
                    R = data.xmat[1].reshape(3, 3)
                    pitch_deg = float(np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))))
                    roll_deg = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
                    neck_adr = model.jnt_qposadr[neck_jid] if neck_jid >= 0 else None
                    neck_deg = float(np.degrees(data.qpos[neck_adr])) if neck_adr is not None else 0.0
                    z_pos = float(data.qpos[2])
                    fwd_vel = float(data.qvel[0])
                    cur_mass_g = float(model.body_mass[cyl_body_id] * 1000.0) if cyl_body_id >= 0 else 300.0

                    telem = {
                        "z": round(z_pos, 4),
                        "pitch": round(pitch_deg, 2),
                        "roll": round(roll_deg, 2),
                        "neck": round(neck_deg, 2),
                        "step": step,
                        "mass": round(cur_mass_g, 1),
                        "speed": round(fwd_vel, 3),
                        "target_vx": round(current_cmd[0], 3),
                        "status": "Walking Stable (Env 3 v3 PPO)" if (z_pos > 0.09 and abs(pitch_deg) < 30) else "Fallen / Needs Reset",
                    }
                    telem_bytes = json.dumps(telem).encode("utf-8")
                    t_req = urllib.request.Request(
                        f"http://localhost:{web_port}/api/telemetry",
                        data=telem_bytes,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    threading.Thread(target=_post_telemetry, args=(t_req,), daemon=True).start()
                except Exception:
                    pass

            time_until_next_step = 0.02 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            step += 1

        stop_event.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_ppo_headup_v3.onnx"), help="Path to policy ONNX (defaults to Env 3 v3 policy)")
    parser.add_argument("--model", default=os.path.join(_EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_loadcarry.xml"))
    parser.add_argument("--gui", action="store_true", help="Launch native interactive MuJoCo GUI window")
    parser.add_argument("--web", action="store_true", help="Connect to localhost web controller (http://localhost:8080)")
    parser.add_argument("--web-port", type=int, default=8080, help="Port of localhost web controller")
    parser.add_argument("--video", type=str, default="/Users/raghav/.gemini/antigravity-ide/brain/f2ffb849-d730-4892-8213-f5bdaeddc8dc/trained_loadcarry_native_mujoco.mp4", help="Path to save MP4 video")
    parser.add_argument("--steps", type=int, default=350, help="Number of control steps to simulate")
    parser.add_argument("--vel-x", type=float, default=0.25, help="Forward walking speed (m/s)")
    parser.add_argument("--mass", type=float, default=300.0, help="Payload barbell mass in grams (default: 300.0g)")
    parser.add_argument("--dia", type=float, default=8.0, help="Cylinder diameter in mm (e.g. 8.0, 10.0, 12.0)")
    parser.add_argument("--depth", type=float, default=-68.0, help="Grip depth Z in mm (e.g. -68.0, -70.0, -72.0)")
    args = parser.parse_args()

    model, data, policy = setup_simulation(
        args.model,
        args.policy,
        payload_mass_g=args.mass,
        vel_x=args.vel_x,
        dia_mm=args.dia,
        z_mm=args.depth,
    )

    if args.gui:
        run_gui(model, data, policy, enable_web=args.web, web_port=args.web_port)
    else:
        record_video(model, data, policy, args.video, num_steps=args.steps)


if __name__ == "__main__":
    main()
