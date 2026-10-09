#!/usr/bin/env python3
"""Renders High-Definition Showcase Videos for all successful manipulation tasks:
1. Table Tennis Racket Carry (with active steadicam stabilization)
2. Pickleball Paddle Carry (with active steadicam stabilization)
3. Table Tennis Ball Balance (dynamic closed-loop ping-pong ball balancing)

Saves both .mp4 and .webp animations directly into the artifact directory.
"""

import os
import sys
import math
import numpy as np
import mujoco
import imageio

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(_MANIP_DIR)
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))
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

_ARTIFACT_DIR = "/Users/raghav/.gemini/antigravity-ide/brain/3cbde2a4-1f27-4d9b-b2f2-4ff527f5f996"


def render_racket_carry_video(
    racket_type="table_tennis",
    output_name="tt_racket_carry_demo.mp4",
    duration_sec=5.0,
    fps=50,
    vx=0.18,
):
    """Renders offscreen MP4 and WebP of Table Tennis or Pickleball racket carry."""
    out_mp4 = os.path.join(_ARTIFACT_DIR, output_name)
    out_webp = out_mp4.replace(".mp4", ".webp")
    print(f"\n[RENDER] Rendering {racket_type} carry demo ({duration_sec}s @ {fps}fps) -> {out_mp4}...")

    xml_str = create_scaled_scene_xml(racket_type=racket_type, include_ball=False)
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    policy_path = _TT_CARRY_POLICY if racket_type == "table_tennis" and os.path.exists(_TT_CARRY_POLICY) else _DEFAULT_POLICY
    import onnx
    m_onnx = onnx.load(policy_path)
    input_dim = m_onnx.graph.input[0].type.tensor_type.shape.dim[1].dim_value

    if input_dim == 64:
        policy = TableTennisRacketCarryPolicyInference(model, data, onnx_path=policy_path)
    else:
        policy = PolicyInference(
            model, data,
            walking_onnx_path=policy_path,
            action_scale=1.0,
            new_cmd_obs=True,
            use_projected_gravity=True,
        )

    policy.set_vel_cmd(vx, 0.0, 0.0)
    policy.head_offset[:] = 0.0
    policy._update_command()

    # Reset into standing posture
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]

    # Pre-clamp racket inside beak
    racket_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "racket")
    racket_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "racket_free")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_qpos = model.jnt_qposadr[beak_jid]
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    trunk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    target_clamp_angle = 1.630 if racket_type == "table_tennis" else 1.645
    data.qpos[beak_qpos] = target_clamp_angle
    data.qfrc_applied[beak_dof] = -0.18
    if model.nu > 14:
        data.ctrl[14] = 1.0

    mujoco.mj_forward(model, data)
    p_jaw = data.xpos[jaw_bid]
    R_jaw = data.xmat[jaw_bid].reshape(3, 3)
    HOVER_LOCAL = np.array([-0.0119, 0.0, -0.0465])
    CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

    r_adr = model.jnt_qposadr[racket_jid]
    data.qpos[r_adr:r_adr+3] = p_jaw + R_jaw @ HOVER_LOCAL
    q_jaw = np.empty(4)
    mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
    q_r = np.empty(4)
    mujoco.mju_mulQuat(q_r, q_jaw, CYL_QUAT)
    data.qpos[r_adr+3:r_adr+7] = q_r

    # Settle
    for _ in range(60):
        data.qfrc_applied[beak_dof] = -0.18
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, width=1280, height=720)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.85
    cam.elevation = -12.0
    cam.azimuth = 135.0  # Dynamic 3/4 front isometric broadcast view

    total_sim_steps = int(duration_sec / model.opt.timestep)
    sim_steps_per_frame = int(round(1.0 / (fps * model.opt.timestep)))
    decimation = 10  # 50 Hz policy (20ms)

    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8)
    smooth_x = data.xpos[trunk_bid][0] + 0.08
    smooth_z = data.xpos[trunk_bid][2] + 0.02
    steadicam_pos = 0.0

    for step in range(total_sim_steps):
        t_sim = step * model.opt.timestep
        beak_err = target_clamp_angle - data.qpos[beak_qpos]
        data.qfrc_applied[beak_dof] = float(np.clip(-0.18 + 12.0 * beak_err - 0.5 * data.qvel[beak_dof], -1.2, 0.2))

        if step % decimation == 0:
            action = policy.infer()

            # Steadicam pantograph leveling
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

        mujoco.mj_step(model, data)

        if step % sim_steps_per_frame == 0:
            p_trunk = data.xpos[trunk_bid]
            target_x = p_trunk[0] + 0.08
            target_z = p_trunk[2] + 0.02
            smooth_x = 0.94 * smooth_x + 0.06 * target_x
            smooth_z = 0.94 * smooth_z + 0.06 * target_z
            cam.lookat = [smooth_x, p_trunk[1], smooth_z]

            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            writer.append_data(img)

    writer.close()
    print(f"[SUCCESS] Saved {out_mp4}")
    cmd = f'ffmpeg -y -i "{out_mp4}" -vf "fps=25,scale=640:-1:flags=lanczos" -vcodec libwebp -lossless 0 -compression_level 4 -q:v 75 -loop 0 "{out_webp}"'
    os.system(cmd)
    print(f"[SUCCESS] Saved {out_webp}")


def render_ball_balance_video(
    output_name="pingpong_ball_balance_demo.mp4",
    duration_sec=5.0,
    fps=50,
    vx=0.15,
):
    """Renders offscreen MP4 and WebP of Table Tennis Ball Balancing on Racket."""
    out_mp4 = os.path.join(_ARTIFACT_DIR, output_name)
    out_webp = out_mp4.replace(".mp4", ".webp")
    print(f"\n[RENDER] Rendering Ball Balance demo ({duration_sec}s @ {fps}fps) -> {out_mp4}...")

    xml_str = create_scaled_scene_xml(racket_type="table_tennis", include_ball=True)
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    policy_path = _TT_POLICY if os.path.exists(_TT_POLICY) else _DEFAULT_POLICY
    policy = TableTennisPolicyInference(model, data, onnx_path=policy_path)
    policy.set_vel_cmd(vx, 0.0, 0.0)
    policy.head_offset[:] = 0.0
    policy._update_command()

    # Reset into standing posture
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]

    racket_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "racket")
    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pingpong_ball")
    racket_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "racket_free")
    ball_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pingpong_ball_free")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_qpos = model.jnt_qposadr[beak_jid]
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    trunk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    target_clamp_angle = 1.630
    data.qpos[beak_qpos] = target_clamp_angle
    data.qfrc_applied[beak_dof] = -0.18
    if model.nu > 14:
        data.ctrl[14] = 1.0

    mujoco.mj_forward(model, data)
    p_jaw = data.xpos[jaw_bid]
    R_jaw = data.xmat[jaw_bid].reshape(3, 3)
    HOVER_LOCAL = np.array([-0.0119, 0.0, -0.0465])
    CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

    r_adr = model.jnt_qposadr[racket_jid]
    data.qpos[r_adr:r_adr+3] = p_jaw + R_jaw @ HOVER_LOCAL
    q_jaw = np.empty(4)
    mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
    q_r = np.empty(4)
    mujoco.mju_mulQuat(q_r, q_jaw, CYL_QUAT)
    data.qpos[r_adr+3:r_adr+7] = q_r

    # Spawn ball gently onto the racket sweet spot
    b_adr = model.jnt_qposadr[ball_jid]
    mujoco.mj_forward(model, data)
    p_rac = data.xpos[racket_bid]
    R_rac = data.xmat[racket_bid].reshape(3, 3)
    # Sweet spot: 101.5 mm along handle, 20 mm radius ball + 1 mm clearance above rubber
    p_ball_init = p_rac + R_rac @ np.array([0.0, 0.0215, 0.1015])
    data.qpos[b_adr:b_adr+3] = p_ball_init
    data.qpos[b_adr+3:b_adr+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[model.jnt_dofadr[ball_jid]:model.jnt_dofadr[ball_jid]+6] = 0.0

    # Settle
    for _ in range(60):
        data.qfrc_applied[beak_dof] = -0.18
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, width=1280, height=720)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.72
    cam.elevation = -10.0
    cam.azimuth = 110.0  # Crisp focus on paddle face and ping-pong ball

    total_sim_steps = int(duration_sec / model.opt.timestep)
    sim_steps_per_frame = int(round(1.0 / (fps * model.opt.timestep)))
    decimation = 10  # 50 Hz policy

    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8)
    smooth_x = data.xpos[trunk_bid][0] + 0.12
    smooth_z = data.xpos[trunk_bid][2] + 0.04
    steadicam_pos = 0.0

    for step in range(total_sim_steps):
        t_sim = step * model.opt.timestep
        beak_err = target_clamp_angle - data.qpos[beak_qpos]
        data.qfrc_applied[beak_dof] = float(np.clip(-0.18 + 12.0 * beak_err - 0.5 * data.qvel[beak_dof], -1.2, 0.2))

        if step % decimation == 0:
            # Active balancing policy with ball observations
            action = policy.infer()

            # Steadicam leveling
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

        mujoco.mj_step(model, data)

        if step % sim_steps_per_frame == 0:
            p_ball = data.xpos[ball_bid]
            p_trunk = data.xpos[trunk_bid]
            target_x = 0.6 * p_ball[0] + 0.4 * p_trunk[0]
            target_z = p_ball[2] + 0.02
            smooth_x = 0.94 * smooth_x + 0.06 * target_x
            smooth_z = 0.94 * smooth_z + 0.06 * target_z
            cam.lookat = [smooth_x, p_ball[1], smooth_z]

            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            writer.append_data(img)

    writer.close()
    print(f"[SUCCESS] Saved {out_mp4}")
    cmd = f'ffmpeg -y -i "{out_mp4}" -vf "fps=25,scale=640:-1:flags=lanczos" -vcodec libwebp -lossless 0 -compression_level 4 -q:v 75 -loop 0 "{out_webp}"'
    os.system(cmd)
    print(f"[SUCCESS] Saved {out_webp}")


if __name__ == "__main__":
    render_racket_carry_video("table_tennis", "table_tennis_carry_demo.mp4")
    render_racket_carry_video("pickleball", "pickleball_carry_demo.mp4")
    render_ball_balance_video("pingpong_ball_balance_demo.mp4")
