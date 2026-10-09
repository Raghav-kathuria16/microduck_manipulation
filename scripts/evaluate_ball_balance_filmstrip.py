#!/usr/bin/env python3
"""evaluate_ball_balance_filmstrip.py

Renders multi-camera visual filmstrips and records telemetry for Problem 2:
Table Tennis Ball Balancing on the authentic 180g ITTF tournament racket.

Generates high-resolution proof images with zero LaTeX in the UI.
"""

import math
import os
import sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(_MANIP_DIR)
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))

from infer_policy import DEFAULT_POSE
from test_racket_drop_and_walk import (
    create_scaled_scene_xml,
    TableTennisPolicyInference,
    _TT_POLICY,
    _ARTIFACT_DIR,
)


def evaluate_ball_balance_telemetry(num_steps=300, vx=0.0, policy_path=None, capture_steps=None):
    """Executes a simulation rollout and captures multi-camera filmstrip frames."""
    xml_str = create_scaled_scene_xml("table_tennis", include_ball=True)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    target_policy = policy_path if policy_path is not None else _TT_POLICY
    policy = TableTennisPolicyInference(model, data, onnx_path=str(target_policy))

    racket_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "racket")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_qpos = model.jnt_qposadr[beak_jid]
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    racket_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "racket_free")
    racket_qadr = model.jnt_qposadr[racket_jid]
    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pingpong_ball")
    ball_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qadr = model.jnt_qposadr[ball_jid]

    CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])
    HOVER_LOCAL = np.array([-0.0110, 0.0, -0.0630])
    clamp_torque = 0.35

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]
    data.qpos[beak_qpos] = 1.550
    mujoco.mj_forward(model, data)

    p_jaw = data.xpos[jaw_bid]
    R_jaw = data.xmat[jaw_bid].reshape(3, 3)
    data.qpos[racket_qadr:racket_qadr+3] = p_jaw + R_jaw @ HOVER_LOCAL
    q_jaw = np.empty(4)
    mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
    q_r = np.empty(4)
    mujoco.mju_mulQuat(q_r, q_jaw, CYL_QUAT)
    data.qpos[racket_qadr+3:racket_qadr+7] = q_r
    mujoco.mj_forward(model, data)

    p_racket = data.xpos[racket_bid]
    R_racket = data.xmat[racket_bid].reshape(3, 3)
    face_center_world = p_racket + R_racket @ np.array([0.0, 0.0035 + 0.0141 + 0.002, 0.1015])
    data.qpos[ball_qadr:ball_qadr+3] = face_center_world
    data.qpos[ball_qadr+3:ball_qadr+7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)

    # 4 distinct cameras for comprehensive visual inspection
    cam_close = mujoco.MjvCamera()
    cam_close.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam_close.trackbodyid = racket_bid
    cam_close.distance = 0.32
    cam_close.elevation = -24.0
    cam_close.azimuth = 145.0

    cam_top = mujoco.MjvCamera()
    cam_top.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam_top.trackbodyid = racket_bid
    cam_top.distance = 0.38
    cam_top.elevation = -85.0
    cam_top.azimuth = 180.0

    if capture_steps is None:
        capture_steps = [int(num_steps * f) for f in [0.10, 0.35, 0.65, 0.95]]
    frames = []
    telemetry = []

    for step in range(num_steps):
        sim_time = step * 0.005
        data.qfrc_applied[beak_dof] = -clamp_torque
        if model.nu > 14:
            data.ctrl[14] = 1.470

        if step % 4 == 0:
            policy.set_vel_cmd(vx, 0.0, 0.0)
            policy._update_command()
            action = policy.infer()
            policy.apply_action(action)
            if model.nu > 14:
                data.ctrl[14] = 1.470

        mujoco.mj_step(model, data)

        p_b = data.xpos[ball_bid]
        p_r = data.xpos[racket_bid]
        R_r = data.xmat[racket_bid].reshape(3, 3)
        rel = R_r.T @ (p_b - p_r)
        dx = float(rel[0])
        dz = float(rel[2] - 0.1015)
        r_ball = math.sqrt(dx ** 2 + dz ** 2)

        normal_world = R_r @ np.array([0.0, 1.0, 0.0])
        pitch_deg = math.degrees(math.asin(np.clip(normal_world[0], -0.999, 0.999)))
        roll_deg = math.degrees(math.asin(np.clip(normal_world[1], -0.999, 0.999)))

        # Contact normal force
        fn = 0.0
        for ci in range(data.ncon):
            con = data.contact[ci]
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            if (b1 == racket_bid or b2 == racket_bid) and ("jaw" in model.geom(con.geom1).name or "jaw" in model.geom(con.geom2).name or "mouth" in model.geom(con.geom1).name or "mouth" in model.geom(con.geom2).name):
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, f)
                fn += f[0]

        telemetry.append({
            "step": step,
            "time": sim_time,
            "dx_mm": dx * 1000.0,
            "dz_mm": dz * 1000.0,
            "r_mm": r_ball * 1000.0,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "clamp_fn": fn,
            "on_blade": r_ball < 0.055,
        })

        if step in capture_steps:
            renderer.update_scene(data, camera=cam_close)
            img_close = Image.fromarray(renderer.render())
            renderer.update_scene(data, camera=cam_top)
            img_top = Image.fromarray(renderer.render())
            frames.append((sim_time, pitch_deg, roll_deg, dx * 1000.0, dz * 1000.0, fn, img_close, img_top))

    return telemetry, frames


def build_ball_balance_filmstrip_image(frames, out_path, title="TABLE TENNIS BALL BALANCING TELEMETRY & MULTI-VIEW FILMSTRIP"):
    """Combines rendered views and telemetric readouts into a clean side-by-side composite."""
    n = len(frames)
    w_single = 640
    h_single = 480
    border = 16
    header_h = 110
    total_w = n * w_single + (n + 1) * border
    total_h = 2 * h_single + header_h + border * 3

    canvas = Image.new("RGB", (total_w, total_h), color=(18, 20, 24))
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([(border, border), (total_w - border, header_h)], fill=(28, 32, 40))
    draw.text((border + 20, border + 15), title, fill=(240, 240, 245))
    subtitle = "Zero-Weld Friction Grip (80g Proportionally Scaled Racket) | 1.3g Scaled Ball | Upright Cradle Posture"
    draw.text((border + 20, border + 45), subtitle, fill=(160, 175, 200))

    # Top row: Perspective tracking camera
    # Bottom row: Top-down blade tracking camera
    for i, (t, p_deg, r_deg, dx, dz, fn, img_close, img_top) in enumerate(frames):
        x_pos = border + i * (w_single + border)
        y_top_row = header_h + border
        y_bot_row = y_top_row + h_single + border

        canvas.paste(img_close, (x_pos, y_top_row))
        canvas.paste(img_top, (x_pos, y_bot_row))

        # Overlay banner on top row
        overlay_text_1 = f"t = {t:.2f}s | Perspective View\nPitch: {p_deg:+.1f}° | Roll: {r_deg:+.1f}°\nClamp Normal: {fn:.1f} N"
        draw.rectangle([(x_pos + 10, y_top_row + 10), (x_pos + 260, y_top_row + 70)], fill=(10, 10, 15, 200))
        draw.text((x_pos + 16, y_top_row + 14), overlay_text_1, fill=(255, 255, 255))

        # Overlay banner on bottom row
        in_sweet_spot = math.sqrt(dx**2 + dz**2) <= 18.0
        status_tag = "SWEET SPOT" if in_sweet_spot else ("ON BLADE" if math.sqrt(dx**2 + dz**2) <= 55.0 else "OFF BLADE")
        overlay_text_2 = f"t = {t:.2f}s | Top-Down Blade View\ndx: {dx:+.1f}mm | dz: {dz:+.1f}mm\nStatus: [{status_tag}]"
        draw.rectangle([(x_pos + 10, y_bot_row + 10), (x_pos + 270, y_bot_row + 70)], fill=(10, 10, 15, 200))
        draw.text((x_pos + 16, y_bot_row + 14), overlay_text_2, fill=(255, 230, 100) if in_sweet_spot else (255, 255, 255))

    canvas.save(out_path)
    print(f"[filmstrip] Successfully saved to: {out_path}")


def main():
    out_img = os.path.join(_ARTIFACT_DIR, "ball_balance_current_policy_filmstrip.png")
    print(f"Running ball balance telemetry rollout...")
    telemetry, frames = evaluate_ball_balance_telemetry(num_steps=500, vx=0.18)
    build_ball_balance_filmstrip_image(frames, out_img)

    # Summary
    on_blade_steps = sum(1 for d in telemetry if d["on_blade"])
    retention_pct = (on_blade_steps / len(telemetry)) * 100.0
    print(f"Rollout finished! Ball on blade: {on_blade_steps}/{len(telemetry)} steps ({retention_pct:.1f}%)")


if __name__ == "__main__":
    main()
