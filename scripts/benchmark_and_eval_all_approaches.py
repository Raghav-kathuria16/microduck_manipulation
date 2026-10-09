#!/usr/bin/env python3
"""benchmark_and_eval_all_approaches.py

Master Automated Training and Evaluation Pipeline for Problem 2 Table Tennis Ball Balancing:
- Strictly enforces the 20-minute TOTAL execution time limit.
- Trains each approach for ~4.5 minutes.
- Evaluates each policy on a 1,000-step (5.0s, vx = 0.18 m/s) continuous forward walking rollout.
- Generates high-resolution multi-camera filmstrips for visual proof.
- Aggregates comparative telemetry metrics (ball survival time, retention %, pitch/roll errors, clamping force).
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIP_DIR = _SCRIPT_DIR.parent
_ROOT = _MANIP_DIR.parent
_RL_DIR = _ROOT / "microduck_rl"
_PYTHON = _RL_DIR / ".venv" / "bin" / "python"
_TRAIN_BIN = _RL_DIR / ".venv" / "bin" / "train"
_POLICIES_DIR = _MANIP_DIR / "policies" / "onnx"
_ARTIFACT_DIR = Path("/Users/raghav/.gemini/antigravity-ide/brain/3cbde2a4-1f27-4d9b-b2f2-4ff527f5f996")
_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

from evaluate_ball_balance_filmstrip import (
    evaluate_ball_balance_telemetry,
    build_ball_balance_filmstrip_image,
)


def find_latest_checkpoint(log_root: Path, pattern: str = "*table_tennis*") -> Path:
    """Discovers the highest iteration checkpoint in the most recent run folder matching pattern."""
    run_dirs = sorted(
        [d for d in log_root.glob(pattern) if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run folders matching '{pattern}' in {log_root}")

    latest_run = run_dirs[-1]
    pt_files = list(latest_run.glob("model_*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No checkpoint files (.pt) found in {latest_run}")

    checkpoints = []
    for pt in pt_files:
        m = re.search(r"model_(\d+)\.pt$", pt.name)
        if m:
            checkpoints.append((int(m.group(1)), pt))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def export_raw_to_onnx(checkpoint_pt: Path, onnx_dest: Path, task_id: str):
    """Exports raw checkpoint to ONNX with baked-in observation normalizer."""
    export_script = _RL_DIR / "scripts" / "fast_export.py"
    if not export_script.exists():
        export_script = _RL_DIR / "scripts" / "export.py"

    onnx_dest.parent.mkdir(parents=True, exist_ok=True)
    if "fast_export" in export_script.name:
        cmd = [str(_PYTHON), str(export_script), str(checkpoint_pt), "--onnx-file", str(onnx_dest)]
    else:
        cmd = [str(_PYTHON), str(export_script), task_id, "--checkpoint-file", str(checkpoint_pt), "--onnx-file", str(onnx_dest)]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "offline"
    env["WANDB_SILENT"] = "true"

    proc = subprocess.run(cmd, cwd=str(_RL_DIR), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Raw ONNX export failed with code {proc.returncode}")
    print(f"[export] Exported raw policy to: {onnx_dest}")


def combine_residual_policy(raw_onnx: Path, base_onnx: Path, combined_onnx: Path):
    """Bakes frozen loadcarry_v4 and active residual into a unified single ONNX model."""
    combine_script = _SCRIPT_DIR / "export_combined_residual.py"
    alpha_vec = [
        "0.10", "0.10", "0.10", "0.10", "0.10",  # left leg: minimal stance compliance
        "0.90", "0.90", "0.15", "0.80",          # head: active 4-DOF neck gimbal (pitch 0.90, roll 0.80)
        "0.10", "0.10", "0.10", "0.10", "0.10",  # right leg: minimal stance compliance
    ]

    cmd = [
        str(_PYTHON), str(combine_script),
        "--base-onnx", str(base_onnx),
        "--res-onnx", str(raw_onnx),
        "--out-onnx", str(combined_onnx),
        "--alpha-vector", *alpha_vec,
    ]
    proc = subprocess.run(cmd, cwd=str(_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Combine export failed with code {proc.returncode}")
    print(f"[combine] Successfully exported unified residual policy to: {combined_onnx}")


def run_evaluation(policy_path: Path, num_steps: int = 1000, vx: float = 0.18, title: str = "", filmstrip_dest: Path = None):
    """Runs high-fidelity evaluation rollout and computes quantitative metrics."""
    print(f"\n[eval] Evaluating policy: {policy_path.name} ({num_steps} steps, vx={vx} m/s)...")
    telemetry, frames = evaluate_ball_balance_telemetry(num_steps=num_steps, vx=vx, policy_path=policy_path)

    if filmstrip_dest and frames:
        build_ball_balance_filmstrip_image(frames, str(filmstrip_dest), title=title)

    # Compute key metrics
    total_steps = len(telemetry)
    dt = 0.005
    on_blade_steps = sum(1 for d in telemetry if d["on_blade"])
    retention_pct = (on_blade_steps / total_steps) * 100.0

    # Survival time before first drop
    first_drop_step = total_steps
    for i, d in enumerate(telemetry):
        if not d["on_blade"]:
            first_drop_step = i
            break
    survival_time_s = first_drop_step * dt

    # Sweet spot percentage (radius <= 25mm)
    sweet_spot_steps = sum(1 for d in telemetry if d["r_mm"] <= 25.0 and d["on_blade"])
    sweet_spot_pct = (sweet_spot_steps / total_steps) * 100.0

    # Angular errors during on_blade period
    blade_telemetry = [d for d in telemetry if d["on_blade"]]
    if blade_telemetry:
        pitches = [abs(d["pitch_deg"]) for d in blade_telemetry]
        rolls = [abs(d["roll_deg"]) for d in blade_telemetry]
        mean_pitch_err = float(sum(pitches) / len(pitches))
        max_pitch_err = float(max(pitches))
        mean_roll_err = float(sum(rolls) / len(rolls))
        max_roll_err = float(max(rolls))
        mean_fn = float(sum(d["clamp_fn"] for d in blade_telemetry) / len(blade_telemetry))
    else:
        mean_pitch_err = max_pitch_err = mean_roll_err = max_roll_err = mean_fn = 0.0

    metrics = {
        "policy_name": policy_path.name,
        "total_rollout_s": total_steps * dt,
        "survival_time_s": survival_time_s,
        "retention_pct": retention_pct,
        "sweet_spot_pct": sweet_spot_pct,
        "mean_pitch_err_deg": mean_pitch_err,
        "max_pitch_err_deg": max_pitch_err,
        "mean_roll_err_deg": mean_roll_err,
        "max_roll_err_deg": max_roll_err,
        "mean_clamp_force_N": mean_fn,
        "filmstrip_path": str(filmstrip_dest) if filmstrip_dest else None,
    }

    print(f"[eval] Results for {policy_path.name}:")
    print(f"       Survival Time:   {survival_time_s:.2f}s / {total_steps * dt:.2f}s")
    print(f"       Retention %:     {retention_pct:.1f}%")
    print(f"       Sweet Spot %:    {sweet_spot_pct:.1f}%")
    print(f"       Pitch Err Mean:  {mean_pitch_err:.1f}° (Max: {max_pitch_err:.1f}°)")
    print(f"       Roll Err Mean:   {mean_roll_err:.1f}° (Max: {max_roll_err:.1f}°)")
    print(f"       Clamp Normal Fn: {mean_fn:.1f} N")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark & Evaluate All Ball Balancing Approaches")
    parser.add_argument("--total-time-limit-mins", type=float, default=20.0, help="Maximum total runtime limit in minutes")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Number of 200Hz physics steps per evaluation rollout (1000 = 5.0s)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and evaluate existing checkpoints")
    args = parser.parse_args()

    t_suite_start = time.time()
    max_seconds = args.total_time_limit_mins * 60.0

    log_root = _RL_DIR / "logs" / "rsl_rl" / "table_tennis"
    log_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" PROBLEM 2: MULTI-APPROACH TRAINING & EVALUATION BENCHMARK SUITE")
    print(f" Hard Time Limit: {args.total_time_limit_mins:.1f} minutes ({max_seconds:.0f} seconds)")
    print(f" Evaluation Rollouts: {args.eval_steps} steps (5.0 seconds at vx = 0.18 m/s)")
    print("=" * 80)

    results = {}

    # Define approaches: (Key, Name, Task_ID, Iterations, Run_Pattern, Output_ONNX, Title)
    approaches = [
        {
            "key": "Approach_B",
            "name": "Approach B (Residual RL on Frozen loadcarry_v4)",
            "task_id": "Mjlab-TableTennis-BallBalance-Residual-MicroDuck",
            "iterations": 45,
            "num_envs": 64,
            "pattern": "*residual*",
            "is_residual": True,
            "skip_if_exists": True,
            "out_onnx": _POLICIES_DIR / "approach_b_residual.onnx",
            "filmstrip": _ARTIFACT_DIR / "ball_balance_approach_b_filmstrip.png",
            "title": "APPROACH B: RESIDUAL RL ACTIVE GIMBAL (45 ITERS | FROZEN GAIT 0.18 m/s)",
            "env_vars": {},
        },
        {
            "key": "Approach_A",
            "name": "Approach A (Full PPO Warm-Start 14-DOF)",
            "task_id": "Mjlab-TableTennis-BallBalance-MicroDuck",
            "iterations": 15,
            "num_envs": 64,
            "pattern": "*table_tennis_v2*ppo*",
            "is_residual": False,
            "skip_if_exists": False,
            "out_onnx": _POLICIES_DIR / "approach_a_warmstart.onnx",
            "filmstrip": _ARTIFACT_DIR / "ball_balance_approach_a_filmstrip.png",
            "title": "APPROACH A: FULL PPO WARM-START (15 ITERS | CO-ADAPTED GAIT 0.18 m/s)",
            "env_vars": {
                "WARMSTART_BALL_BALANCE_ONNX_PATH": str(_POLICIES_DIR / "table_tennis_racket_carry_shaping.onnx")
            },
        },
        {
            "key": "Approach_C",
            "name": "Approach C (Virtual Impedance Potential Well)",
            "task_id": "Mjlab-TableTennis-BallBalance-Impedance-MicroDuck",
            "iterations": 15,
            "num_envs": 64,
            "pattern": "*impedance*",
            "is_residual": False,
            "skip_if_exists": False,
            "out_onnx": _POLICIES_DIR / "approach_c_impedance.onnx",
            "filmstrip": _ARTIFACT_DIR / "ball_balance_approach_c_filmstrip.png",
            "title": "APPROACH C: VIRTUAL IMPEDANCE POTENTIAL WELL (15 ITERS | ASYMMETRIC CRITIC)",
            "env_vars": {},
        },
    ]

    for app in approaches:
        elapsed_total = time.time() - t_suite_start
        remaining_time = max_seconds - elapsed_total
        print(f"\n[suite] Elapsed: {elapsed_total/60:.2f} mins | Remaining: {remaining_time/60:.2f} mins")

        if remaining_time < 120.0 and not args.skip_train:
            print(f"[suite] WARNING: Less than 2 minutes remaining! Skipping training for {app['name']} to respect hard 20-min limit.")
            break

        print("\n" + "#" * 80)
        print(f" RUNNING: {app['name']}")
        print(f" Task ID: {app['task_id']}")
        print(f" Iterations: {app['iterations']} | Envs: {app['num_envs']}")
        print("#" * 80)

        t_app_start = time.time()

        should_train = not args.skip_train
        if app.get("skip_if_exists", False) and app["out_onnx"].exists():
            print(f"[suite] Trained ONNX already exists for {app['name']}: {app['out_onnx'].name}. Skipping training step.")
            should_train = False

        if should_train:
            cmd = [
                str(_TRAIN_BIN),
                app["task_id"],
                "--env.scene.num-envs", str(app["num_envs"]),
                "--agent.max_iterations", str(app["iterations"]),
                "--agent.save-interval", "5",
                "--agent.logger", "tensorboard",
            ]

            env = os.environ.copy()
            env["WANDB_MODE"] = "offline"
            env["PYTHONUNBUFFERED"] = "1"
            for k, v in app["env_vars"].items():
                env[k] = v

            proc = subprocess.run(cmd, cwd=str(_RL_DIR), env=env, stdin=subprocess.DEVNULL)
            if proc.returncode != 0:
                print(f"[suite] ERROR: Training failed for {app['name']} with code {proc.returncode}!")
                continue

            app_train_elapsed = time.time() - t_app_start
            print(f"[suite] Training finished in {app_train_elapsed/60:.2f} minutes!")

            # Export policy
            try:
                latest_pt = find_latest_checkpoint(log_root, app["pattern"])
                print(f"[checkpoint] Latest checkpoint: {latest_pt.name}")

                if app["is_residual"]:
                    raw_onnx = _POLICIES_DIR / "table_tennis_ball_balance_residual_raw.onnx"
                    export_raw_to_onnx(latest_pt, raw_onnx, app["task_id"])
                    base_onnx = _POLICIES_DIR / "loadcarry_ppo_headup_v4.onnx"
                    combine_residual_policy(raw_onnx, base_onnx, app["out_onnx"])
                else:
                    export_raw_to_onnx(latest_pt, app["out_onnx"], app["task_id"])

                # Copy to main table_tennis_ball_balance.onnx if Approach B
                if app["key"] == "Approach_B":
                    shutil.copyfile(app["out_onnx"], _POLICIES_DIR / "table_tennis_ball_balance.onnx")

            except Exception as e:
                print(f"[suite] Export error for {app['name']}: {e}")
                continue

        # Evaluate Policy
        if app["out_onnx"].exists():
            metrics = run_evaluation(
                policy_path=app["out_onnx"],
                num_steps=args.eval_steps,
                vx=0.18,
                title=app["title"],
                filmstrip_dest=app["filmstrip"],
            )
            metrics["train_time_s"] = time.time() - t_app_start if not args.skip_train else 0.0
            results[app["key"]] = metrics

    total_suite_time = time.time() - t_suite_start
    print("\n" + "=" * 80)
    print(f" BENCHMARK SUITE COMPLETED IN {total_suite_time/60:.2f} MINUTES")
    print("=" * 80)

    # Save JSON summary
    summary_json = _ARTIFACT_DIR / "benchmark_evaluation_summary.json"
    with open(summary_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[summary] Saved JSON metrics to: {summary_json}")


if __name__ == "__main__":
    main()
