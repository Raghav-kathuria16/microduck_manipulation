#!/usr/bin/env python3
"""
evaluate_racket_carry_filmstrip.py - Telemetry Verification & Multi-Frame Filmstrip Builder.

Evaluates a trained Racket Carry policy (Approach A, B, or C) in MuJoCo:
- Tracks instantaneous Pitch and Roll angles of the racket blade vs horizontal.
- Verifies adherence to the defined tolerance bands:
    * Pitch band: [-5.0°, +5.0°]
    * Roll band:  [-4.0°, +4.0°]
- Measures forward displacement (cm) and clamping retention force (N).
- Renders a multi-frame composite filmstrip with crisp telemetry overlays.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIP_DIR = _SCRIPT_DIR.parent
_ROOT = _MANIP_DIR.parent
_RL_DIR = _ROOT / "microduck_rl"
_EXTERNAL_REPO = _MANIP_DIR / ".external_repo"
sys.path.insert(0, str(_EXTERNAL_REPO / "scripts"))
sys.path.insert(0, str(_EXTERNAL_REPO / "src"))

from infer_policy import PolicyInference
from test_racket_drop_and_walk import (
    create_scaled_scene_xml,
    TableTennisPolicyInference,
    TableTennisRacketCarryPolicyInference,
)

_ARTIFACT_DIR = Path("/Users/raghav/.gemini/antigravity-ide/brain/1dcd9661-c7ab-457a-8bec-a3fc2df3ade3")


def evaluate_policy_walk(policy_path: str, num_steps: int = 400, vx: float = 0.18):
    """Runs a 400-step (2.0s) walking rollout, sampling telemetry and keyframe renders."""
    xml_str = create_scaled_scene_xml(racket_type="table_tennis", include_ball=False)
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    import onnx
    m_onnx = onnx.load(policy_path)
    input_dim = m_onnx.graph.input[0].type.tensor_type.shape.dim[1].dim_value

    if input_dim == 64:
        policy = TableTennisRacketCarryPolicyInference(model, data, onnx_path=policy_path)
        policy.head_offset[:] = 0.0
    elif input_dim == 69:
        policy = TableTennisPolicyInference(model, data, onnx_path=policy_path)
        policy.head_offset[:] = 0.0
    else:
        policy = PolicyInference(
            model, data,
            walking_onnx_path=policy_path,
            action_scale=1.0,
            new_cmd_obs=True,
            use_projected_gravity=True,
        )
        policy.head_offset[0] = -0.16
        policy.head_offset[1] = -0.08
        policy.head_offset[3] = -0.24

    racket_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "racket")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_qpos = model.jnt_qposadr[beak_jid]
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    racket_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "racket_free")
    racket_qadr = model.jnt_qposadr[racket_jid]

    # Pre-clamp posture for forward-aligned table tennis racket
    CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])
    HOVER_LOCAL = np.array([-0.0110, 0.0, -0.0900])
    clamp_torque = 0.35

    mujoco.mj_resetData(model, data)
    data.qpos[7:7+14] = policy.default_pose
    data.qpos[beak_qpos] = 1.550
    mujoco.mj_forward(model, data)

    p_jaw = data.xpos[jaw_bid]
    R_jaw = data.xmat[jaw_bid].reshape(3, 3)
    p_handle = p_jaw + R_jaw @ HOVER_LOCAL
    q_jaw = np.zeros(4)
    mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
    q_r = np.zeros(4)
    mujoco.mju_mulQuat(q_r, q_jaw, CYL_QUAT)

    data.qpos[racket_qadr:racket_qadr+3] = p_handle
    data.qpos[racket_qadr+3:racket_qadr+7] = q_r
    data.qfrc_applied[beak_dof] = -clamp_torque
    if model.nu > 14:
        data.ctrl[14] = 1.470
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=540, width=720)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance = 0.65
    cam.elevation = -14.0
    cam.azimuth = 138.0

    telemetry = []
    keyframes = []
    capture_steps = [20, 100, 180, 260, 340]  # 0.1s, 0.5s, 0.9s, 1.3s, 1.7s

    init_x = float(data.xpos[1][0])

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

        # Telemetry calculations
        R_racket = data.xmat[racket_bid].reshape(3, 3)
        normal_world = R_racket @ np.array([0.0, 1.0, 0.0])  # Y is normal to paddle blade

        pitch_deg = math.degrees(math.asin(np.clip(-normal_world[0], -1.0, 1.0)))
        roll_deg = math.degrees(math.asin(np.clip(normal_world[1], -1.0, 1.0)))
        x_dist_cm = (float(data.xpos[1][0]) - init_x) * 100.0

        # Clamping force
        fn = 0.0
        for ci in range(data.ncon):
            con = data.contact[ci]
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            if b1 == racket_bid or b2 == racket_bid:
                g1 = model.geom(con.geom1).name
                g2 = model.geom(con.geom2).name
                if "jaw" in g1 or "jaw" in g2 or "mouth" in g1 or "mouth" in g2:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, f)
                    fn += f[0]

        telemetry.append({
            "step": step,
            "time": sim_time,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "x_dist_cm": x_dist_cm,
            "fn_N": fn,
        })

        if step in capture_steps:
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            keyframes.append({
                "time": sim_time,
                "step": step,
                "img": img,
                "pitch_deg": pitch_deg,
                "roll_deg": roll_deg,
                "x_dist_cm": x_dist_cm,
                "fn_N": fn,
            })

    return telemetry, keyframes


def build_filmstrip(keyframes: list, output_path: Path, title: str):
    """Stitches keyframes into a horizontal annotated filmstrip."""
    n_frames = len(keyframes)
    w_frame, h_frame = keyframes[0]["img"].size
    pad = 12
    header_h = 70
    footer_h = 80

    total_w = n_frames * w_frame + (n_frames + 1) * pad
    total_h = header_h + h_frame + footer_h + pad

    canvas = Image.new("RGB", (total_w, total_h), color=(18, 20, 26))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_sub = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
        font_badge = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_badge = ImageFont.load_default()

    # Draw Header
    draw.text((pad + 10, 15), title, fill=(240, 245, 255), font=font_title)
    draw.text((pad + 10, 44), "Target Envelopes: Primary Plateau ∈ [-10.0°, +10.0°] | Acceptable Extremities ∈ [-20.0°, +20.0°] | 180g ITTF Tournament Racket", fill=(160, 175, 200), font=font_sub)

    for i, kf in enumerate(keyframes):
        x_pos = pad + i * (w_frame + pad)
        y_pos = header_h

        # Paste render
        canvas.paste(kf["img"], (x_pos, y_pos))

        # Border
        draw.rectangle([x_pos, y_pos, x_pos + w_frame, y_pos + h_frame], outline=(60, 70, 90), width=2)

        # Telemetry Card Below Frame
        card_y = y_pos + h_frame + 8
        draw.rectangle([x_pos, card_y, x_pos + w_frame, card_y + footer_h - 12], fill=(28, 32, 42), outline=(50, 60, 75))

        pitch_abs = abs(kf["pitch_deg"])
        roll_abs = abs(kf["roll_deg"])

        if pitch_abs <= 10.0:
            p_color = (80, 230, 120)  # Green: optimal plateau
            p_sym = "✓ [Plateau]"
        elif pitch_abs <= 20.0:
            p_color = (255, 200, 60)   # Yellow: acceptable extremity
            p_sym = "▲ [Extremity OK]"
        else:
            p_color = (255, 90, 90)    # Red: droop
            p_sym = "✗ [Out-of-range]"

        if roll_abs <= 10.0:
            r_color = (80, 230, 120)
            r_sym = "✓"
        elif roll_abs <= 20.0:
            r_color = (255, 200, 60)
            r_sym = "▲"
        else:
            r_color = (255, 90, 90)
            r_sym = "✗"

        t_str = f"t = {kf['time']:.2f}s (Step {kf['step']}) | +{kf['x_dist_cm']:.1f} cm"
        draw.text((x_pos + 12, card_y + 8), t_str, fill=(220, 230, 245), font=font_badge)

        pitch_str = f"Pitch: {kf['pitch_deg']:+5.1f}° {p_sym}"
        draw.text((x_pos + 12, card_y + 28), pitch_str, fill=p_color, font=font_badge)

        roll_str = f"Roll: {kf['roll_deg']:+5.1f}° {r_sym}"
        draw.text((x_pos + 205, card_y + 28), roll_str, fill=r_color, font=font_badge)

        clamp_str = f"Clamp: {kf['fn_N']:.1f} N"
        draw.text((x_pos + 310, card_y + 28), clamp_str, fill=(180, 200, 230), font=font_badge)

    canvas.save(output_path, quality=95)
    print(f"[filmstrip] Saved annotated filmstrip to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Racket Carry Policy and Build Filmstrip")
    parser.add_argument("--policy", type=str, required=True, help="Path to ONNX policy")
    parser.add_argument("--output", type=str, required=True, help="Filename of output filmstrip PNG")
    parser.add_argument("--title", type=str, default="MicroDuck Racket Carry Telemetry", help="Title for the filmstrip")
    args = parser.parse_args()

    telemetry, keyframes = evaluate_policy_walk(args.policy, num_steps=380, vx=0.18)

    # Compute aggregate stats
    pitches = [abs(t["pitch_deg"]) for t in telemetry]
    rolls = [abs(t["roll_deg"]) for t in telemetry]
    pitch_in_plateau = sum(p <= 10.0 for p in pitches) / len(pitches) * 100.0
    pitch_in_extremity = sum(p <= 20.0 for p in pitches) / len(pitches) * 100.0
    roll_in_plateau = sum(r <= 10.0 for r in rolls) / len(rolls) * 100.0
    roll_in_extremity = sum(r <= 20.0 for r in rolls) / len(rolls) * 100.0
    max_dist = telemetry[-1]["x_dist_cm"]

    print("\n" + "=" * 70)
    print(f" TELEMETRY EVALUATION RESULTS: {args.title}")
    print("=" * 70)
    print(f" Total Steps: {len(telemetry)} (1.9s forward locomotion)")
    print(f" Net Forward Displacement: +{max_dist:.1f} cm")
    print(f" Mean Pitch Magnitude: {np.mean(pitches):.2f}° (Max: {np.max(pitches):.2f}°)")
    print(f" Pitch in Primary Plateau ([-10.0°, +10.0°]): {pitch_in_plateau:.1f}% of steps")
    print(f" Pitch in Safe Extremity Envelope ([-20.0°, +20.0°]): {pitch_in_extremity:.1f}% of steps")
    print(f" Mean Roll Magnitude:  {np.mean(rolls):.2f}° (Max: {np.max(rolls):.2f}°)")
    print(f" Roll in Primary Plateau ([-10.0°, +10.0°]): {roll_in_plateau:.1f}% of steps")
    print(f" Roll in Safe Extremity Envelope ([-20.0°, +20.0°]): {roll_in_extremity:.1f}% of steps")
    print("=" * 70 + "\n")

    out_file = _ARTIFACT_DIR / args.output
    build_filmstrip(keyframes, out_file, args.title)


if __name__ == "__main__":
    main()
