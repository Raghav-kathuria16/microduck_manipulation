#!/usr/bin/env python3
"""train_ball_balance_residual_runner.py

Approach B: Residual RL on Frozen loadcarry_v4 Locomotion Engine.
1. Locomotion gait is 100% frozen to loadcarry_ppo_headup_v4.onnx (legs cannot stumble/fall).
2. Residual policy ONLY controls the 4-DOF neck gimbal (neck_pitch, head_pitch, head_yaw, head_roll).
3. Minimal additions on top of loadcarry_v4 rewards: ball centering + uphill counter-tilt + roll leveling.
4. Auto-exports unified ONNX policy to table_tennis_ball_balance.onnx.
5. Runs verification rollout and updates visual filmstrip.
"""

import argparse
import os
import re
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
_BASE_POLICY = _POLICIES_DIR / "loadcarry_ppo_headup_v4.onnx"
_RESIDUAL_RAW_ONNX = _POLICIES_DIR / "table_tennis_ball_balance_residual_raw.onnx"
_TARGET_ONNX = _POLICIES_DIR / "table_tennis_ball_balance.onnx"


def find_latest_checkpoint(log_root: Path, pattern: str = "*table_tennis*") -> Path:
    """Discovers the highest iteration checkpoint in the most recent run folder."""
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
    """Exports raw residual checkpoint to ONNX with baked-in observation normalizer."""
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
    print(f"[export] Exported raw residual policy to: {onnx_dest}")


def combine_residual_policy(raw_onnx: Path, base_onnx: Path, combined_onnx: Path):
    """Bakes frozen loadcarry_v4 and active residual into a unified single ONNX model."""
    combine_script = _SCRIPT_DIR / "export_combined_residual.py"
    alpha_vec = [
        "0.05", "0.05", "0.05", "0.05", "0.05",  # left leg: gentle compliance
        "0.30", "0.30", "0.15", "0.40",          # head: active 4-DOF neck gimbal (calibrated for 80g counterbalanced racket)
        "0.05", "0.05", "0.05", "0.05", "0.05",  # right leg: gentle compliance
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
    print(f"[combine] Successfully exported unified policy to: {combined_onnx}")


def main():
    parser = argparse.ArgumentParser(description="Train Approach B Residual Ball Balancing Pipeline")
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations (default: 50, ~4-5 mins)")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments (default: 64)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only export/evaluate")
    args = parser.parse_args()

    task_id = "Mjlab-TableTennis-BallBalance-Residual-MicroDuck"
    log_root = _RL_DIR / "logs" / "rsl_rl" / "table_tennis"
    log_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        print("=" * 80)
        print(" APPROACH B: RESIDUAL RL ON FROZEN LOADCARRY_V4 (ZERO LEG SLIP, ACTIVE GIMBAL)")
        print(f" Task ID: {task_id}")
        print(f" Frozen Base: {_BASE_POLICY}")
        print(f" Iterations: {args.iterations} | Parallel Envs: {args.num_envs}")
        print("=" * 80)

        cmd = [
            str(_TRAIN_BIN),
            task_id,
            "--env.scene.num-envs", str(args.num_envs),
            "--agent.max-iterations", str(args.iterations),
            "--agent.save-interval", "5",
            "--agent.logger", "tensorboard",
        ]

        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        env["WANDB_DISABLED"] = "true"
        env["PYTHONUNBUFFERED"] = "1"

        t_start = time.time()
        proc = subprocess.run(cmd, cwd=str(_RL_DIR), env=env, stdin=subprocess.DEVNULL)
        elapsed = time.time() - t_start

        if proc.returncode != 0:
            print(f"ERROR: Training failed with exit code {proc.returncode}!")
            sys.exit(proc.returncode)

        print(f"\n[training] Finished {args.iterations} iterations in {elapsed/60:.2f} minutes!")

    # Discover latest checkpoint
    latest_pt = find_latest_checkpoint(log_root)
    print(f"[checkpoint] Discovered latest checkpoint: {latest_pt}")

    # Step 1: Export raw residual to ONNX
    export_raw_to_onnx(latest_pt, _RESIDUAL_RAW_ONNX, task_id)

    # Step 2: Combine with frozen loadcarry_v4 into unified ONNX
    combine_residual_policy(_RESIDUAL_RAW_ONNX, _BASE_POLICY, _TARGET_ONNX)
    a_onnx = _POLICIES_DIR / "table_tennis_ball_balance_residual_a.onnx"
    import shutil
    shutil.copyfile(_TARGET_ONNX, a_onnx)
    print(f"[export] Saved dedicated Approach A policy copy to: {a_onnx}")

    # Step 3: Verification filmstrip evaluation
    print("\n" + "=" * 80)
    print(" EXECUTING VERIFICATION ROLLOUT AND RENDERING FILMSTRIP")
    print("=" * 80)
    filmstrip_script = _SCRIPT_DIR / "evaluate_ball_balance_filmstrip.py"
    subprocess.run([str(_PYTHON), str(filmstrip_script)], cwd=str(_ROOT))


if __name__ == "__main__":
    main()
