#!/usr/bin/env python3
"""Batch renders all showcase videos directly to docs/media/ at natural 1x speed:
1. 5-Duck Olympic Stadium Race (1x original playback, 10s competition)
2. 100g Barbell Physical Clamping & Dynamic Walk
3. 200g Barbell Physical Clamping & Dynamic Walk
4. 300g Barbell Physical Clamping & Dynamic Walk
5. Table Tennis Racket Carry (with Steadicam leveling)
6. Badminton Racket Carry (BWF official scaled spec)
7. Pickleball Paddle Carry
8. Lemon & Tableware Spoon Dynamic Balance
"""

import os
import sys
import math
import numpy as np
import mujoco
import imageio

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_DOCS_MEDIA = os.path.join(_MANIP_DIR, "docs", "media")
os.makedirs(_DOCS_MEDIA, exist_ok=True)

sys.path.insert(0, _SCRIPT_DIR)

from infer_policy import PolicyInference
from test_racket_drop_and_walk import (
    create_scaled_scene_xml,
    TableTennisPolicyInference,
    TableTennisRacketCarryPolicyInference,
    _DEFAULT_POLICY,
    _TT_POLICY,
    _TT_CARRY_POLICY,
)
from test_physical_clamp_policy import setup_physical_clamp_sim, PHYSICAL_CLAMP_XML
import test_duck_racer_lemon_balance as racer
import test_spoon_lemon_balance as spoon_sim


def render_to_mp4_and_webp(frames, output_base, fps=30):
    """Saves frame list as MP4 and WebP in docs/media/."""
    mp4_path = os.path.join(_DOCS_MEDIA, f"{output_base}.mp4")
    webp_path = os.path.join(_DOCS_MEDIA, f"{output_base}.webp")

    writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  [SAVED] {mp4_path} ({len(frames)} frames @ {fps}fps)")

    # WebP compression for GitHub display
    cmd = f'ffmpeg -y -i "{mp4_path}" -vf "fps=20,scale=540:-1:flags=lanczos" -vcodec libwebp -lossless 0 -compression_level 4 -q:v 70 -loop 0 "{webp_path}" >/dev/null 2>&1'
    os.system(cmd)
    print(f"  [SAVED] {webp_path}")


def render_duck_race_1x():
    """Renders 5-Duck Olympic Race at natural 1x speed."""
    print("\n[1/8] Rendering 5-Duck Olympic Race (1x natural playback)...")
    xml_str = racer.create_racer_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    session = racer.ort.InferenceSession(racer._DEFAULT_POLICY, providers=["CPUExecutionProvider"])
    ducks = [racer.DuckRacer(cfg["id"], cfg, model, data, session) for cfg in racer.DUCK_CONFIGS]
    racer.setup_racer_episode(ducks, model, data)

    # Physical pacing:
    # 1st: Emerald Green (0.175), 2nd: Ruby Coral Pink (0.165), 3rd: Duck Gold (0.155),
    # 4th: Amethyst Purple (0.148), 5th: Riviera Sky Blue (0.140)
    pacing = [0.165, 0.140, 0.155, 0.175, 0.148]
    for i, d in enumerate(ducks):
        d.target_vx = pacing[i]

    renderer = mujoco.Renderer(model, width=1280, height=720)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[3] = 0

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.15
    cam.elevation = -22.0
    cam.azimuth = 68.0

    duration_sec = 10.0
    fps = 30
    total_steps = int(duration_sec / model.opt.timestep)
    steps_per_frame = int(round(1.0 / (fps * model.opt.timestep)))
    decimation = 10
    smooth_x = 0.45

    frames = []
    for step in range(total_steps):
        t_sim = step * model.opt.timestep
        for d in ducks:
            d.step_clamp()

        if step % decimation == 0:
            for d in ducks:
                d.infer_and_apply(decimation, model.opt.timestep)
                d.update_telemetry(t_sim)

        mujoco.mj_step(model, data)

        if step % steps_per_frame == 0:
            xs = [d.distance for d in ducks]
            pack_mean_x = float(np.mean(xs))
            target_x = pack_mean_x + 0.45
            smooth_x = 0.95 * smooth_x + 0.05 * target_x
            cam.lookat = [smooth_x, 0.0, 0.16]

            renderer.update_scene(data, camera=cam, scene_option=vopt)
            frames.append(renderer.render())

    render_to_mp4_and_webp(frames, "duck_racer_competition_1x", fps=fps)
    render_to_mp4_and_webp(frames, "duck_racer_competition_2x", fps=fps * 2)


def render_barbell(mass_g, torque_nm, vel_x, output_name):
    """Renders Barbell carry at specified mass."""
    print(f"\nRendering {mass_g:.0f}g Barbell Carry...")
    model, data, policy, meta = setup_physical_clamp_sim(
        xml_path=PHYSICAL_CLAMP_XML,
        policy_path=os.path.join(_MANIP_DIR, "policies/onnx/payload_clamping_locomotion.onnx"),
        payload_mass_g=mass_g,
        vel_x=vel_x,
        clamp_torque_nm=torque_nm,
    )

    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 1
    camera.distance = 0.62
    camera.elevation = -8.0
    camera.azimuth = 125.0

    frames = []
    beak_dof = meta["beak_dof"]
    num_steps = 300  # 300 policy steps * 0.02s = 6.0s walk (150 frames @ 30fps = 5.0s video, 100% longer)
    fps = 30

    for step in range(num_steps):
        action = policy.infer()
        policy.apply_action(action)

        for _ in range(4):
            data.qfrc_applied[beak_dof] = -torque_nm
            mujoco.mj_step(model, data)

        if step % 2 == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    render_to_mp4_and_webp(frames, output_name, fps=fps)


def render_racket(racket_type, output_name):
    """Renders racket carry for table tennis, badminton, or pickleball."""
    print(f"\nRendering {racket_type} Carry...")
    is_tt = (racket_type == "table_tennis")
    xml_str = create_scaled_scene_xml(racket_type=racket_type, include_ball=False)
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    policy_path = _TT_CARRY_POLICY if is_tt and os.path.exists(_TT_CARRY_POLICY) else _DEFAULT_POLICY
    import onnxruntime as ort
    sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    input_dim = sess.get_inputs()[0].shape[1]

    if input_dim == 64 and is_tt:
        policy = TableTennisRacketCarryPolicyInference(model, data, onnx_path=policy_path)
    else:
        policy = PolicyInference(
            model, data,
            walking_onnx_path=policy_path,
            action_scale=1.0,
            new_cmd_obs=True,
            use_projected_gravity=True,
        )

    policy.set_vel_cmd(0.18, 0.0, 0.0)
    policy.head_offset[:] = 0.0
    policy._update_command()

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]

    racket_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "racket")
    racket_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "racket_free")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_qpos = model.jnt_qposadr[beak_jid]
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    trunk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    target_clamp_angle = 1.637 if is_tt else (1.654 if racket_type == "badminton" else 1.666)
    clamp_torque = 0.50
    data.qpos[beak_qpos] = target_clamp_angle
    data.qfrc_applied[beak_dof] = -clamp_torque
    if model.nu > 14:
        data.ctrl[14] = 1.0

    mujoco.mj_forward(model, data)
    p_jaw = data.xpos[jaw_bid]
    R_jaw = data.xmat[jaw_bid].reshape(3, 3)
    HOVER_LOCAL = np.array([-0.0120, 0.0, -0.0680])
    CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0]) if is_tt else np.array([0.7071068, 0.7071068, 0.0, 0.0])

    r_adr = model.jnt_qposadr[racket_jid]
    data.qpos[r_adr:r_adr+3] = p_jaw + R_jaw @ HOVER_LOCAL
    q_jaw = np.empty(4)
    mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
    q_r = np.empty(4)
    mujoco.mju_mulQuat(q_r, q_jaw, CYL_QUAT)
    data.qpos[r_adr+3:r_adr+7] = q_r

    for _ in range(60):
        data.qfrc_applied[beak_dof] = -clamp_torque
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, width=1280, height=720)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[3] = 0

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance = 0.72
    cam.elevation = -14.0
    cam.azimuth = 138.0

    duration_s = 3.44 if racket_type in ["badminton", "pickleball"] else 6.87
    total_steps = int(duration_s / model.opt.timestep)
    fps = 30
    steps_per_frame = int(round(1.0 / (fps * model.opt.timestep)))
    decimation = 10

    frames = []
    steadicam_pos = 0.0

    for step in range(total_steps):
        beak_err = target_clamp_angle - data.qpos[beak_qpos]
        data.qfrc_applied[beak_dof] = float(np.clip(-clamp_torque + 12.0 * beak_err - 0.5 * data.qvel[beak_dof], -1.5, 0.2))

        if step % decimation == 0:
            action = policy.infer()

            if is_tt:
                R_trunk = data.xmat[trunk_bid].reshape(3, 3)
                h_fwd = np.array([R_trunk[0, 0], R_trunk[1, 0], 0.0])
                h_fwd /= (np.linalg.norm(h_fwd) + 1e-6)
                R_r = data.xmat[racket_bid].reshape(3, 3)
                b_norm = R_r @ np.array([0.0, 1.0, 0.0])
                racket_pitch = math.asin(np.clip(float(np.dot(b_norm, h_fwd)), -0.999, 0.999))

                rate_cmd = np.clip(1.20 * racket_pitch, -0.30, 0.30)
                steadicam_pos = np.clip(steadicam_pos + rate_cmd * 0.02, -0.20, 0.20)
                action[5] += steadicam_pos
                action[6] -= steadicam_pos

            policy.apply_action(action)
            if model.nu > 14:
                data.ctrl[14] = 1.0

        mujoco.mj_step(model, data)

        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam, scene_option=vopt)
            frames.append(renderer.render())

    render_to_mp4_and_webp(frames, output_name, fps=fps)


def render_lemon_spoon():
    """Renders Lemon & Tableware Spoon dynamic balancing."""
    print("\nRendering Lemon & Spoon Balance...")
    mp4_path = os.path.join(_DOCS_MEDIA, "lemon_spoon_balance.mp4")
    webp_path = os.path.join(_DOCS_MEDIA, "lemon_spoon_balance.webp")
    spoon_sim.render_walking_video(output_path=mp4_path, duration_sec=7.0, vx=0.16)
    cmd = f'ffmpeg -y -i "{mp4_path}" -vf "fps=20,scale=540:-1:flags=lanczos" -vcodec libwebp -lossless 0 -compression_level 4 -q:v 70 -loop 0 "{webp_path}" >/dev/null 2>&1'
    os.system(cmd)
    print(f"  [SAVED] {webp_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render all showcase videos")
    parser.add_argument("--force", action="store_true", help="Force re-rendering all videos")
    args = parser.parse_args()

    print(f"Target media directory: {_DOCS_MEDIA}")
    def exists(name):
        return (not args.force) and os.path.exists(os.path.join(_DOCS_MEDIA, f"{name}.webp"))

    if not exists("duck_racer_competition_1x"):
        render_duck_race_1x()
    else:
        print("  [EXISTS] duck_racer_competition_1x")

    if not exists("barbell_100g_carry"):
        render_barbell(100.0, 0.10, 0.25, "barbell_100g_carry")
    else:
        print("  [EXISTS] barbell_100g_carry")

    if not exists("barbell_200g_carry"):
        render_barbell(200.0, 0.25, 0.20, "barbell_200g_carry")
    else:
        print("  [EXISTS] barbell_200g_carry")

    if not exists("barbell_300g_carry"):
        render_barbell(300.0, 0.50, 0.15, "barbell_300g_carry")
    else:
        print("  [EXISTS] barbell_300g_carry")

    if not exists("table_tennis_carry"):
        render_racket("table_tennis", "table_tennis_carry")
    else:
        print("  [EXISTS] table_tennis_carry")

    if not exists("badminton_carry"):
        render_racket("badminton", "badminton_carry")
    else:
        print("  [EXISTS] badminton_carry")

    if not exists("pickleball_carry"):
        render_racket("pickleball", "pickleball_carry")
    else:
        print("  [EXISTS] pickleball_carry")

    if not exists("lemon_spoon_balance"):
        render_lemon_spoon()
    else:
        print("  [EXISTS] lemon_spoon_balance")

    print("\nAll README media rendered successfully into docs/media/!")
