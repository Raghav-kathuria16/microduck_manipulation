#!/usr/bin/env python3
"""
train_ball_balance_runner.py - Automated Problem 2 Ball Balancing Pipeline.

1. Warm-starts from table_tennis_racket_carry_shaping.onnx (64 dims -> 69 dims).
2. Runs 35 iterations of Mjlab-TableTennis-BallBalance-MicroDuck (64 envs, ~3.5 mins).
3. Auto-exports the resulting checkpoint to microduck_manipulation/policies/onnx/table_tennis_ball_balance.onnx.
4. Executes headless evaluation rollout to measure ball retention time, drift, and forward walk.
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
_TEACHER_POLICY = _POLICIES_DIR / "table_tennis_racket_carry_shaping.onnx"
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


def export_to_onnx(checkpoint_pt: Path, onnx_dest: Path, task_id: str):
    """Exports checkpoint to ONNX with baked-in observation normalizer."""
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
        raise RuntimeError(f"Export failed with code {proc.returncode}")
    print(f"[export] Successfully exported to: {onnx_dest}")


def main():
    parser = argparse.ArgumentParser(description="Train Problem 2 Ball Balancing Pipeline")
    parser.add_argument("--iterations", type=int, default=35, help="Number of training iterations (default: 35, ~3.5 mins)")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments (default: 64)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only export/evaluate")
    args = parser.parse_args()

    task_id = "Mjlab-TableTennis-BallBalance-MicroDuck"
    log_root = _RL_DIR / "logs" / "rsl_rl" / "table_tennis"
    log_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        print("=" * 80)
        print(" PROBLEM 2: TABLE TENNIS BALL BALANCING TRAINING")
        print(f" Task ID: {task_id}")
        print(f" Warm-starting from: {_TEACHER_POLICY}")
        print(f" Iterations: {args.iterations} | Parallel Envs: {args.num_envs}")
        print("=" * 80)

        cmd = [
            str(_TRAIN_BIN),
            task_id,
            "--env.scene.num-envs", str(args.num_envs),
            "--agent.max_iterations", str(args.iterations),
            "--agent.save-interval", "10",
        ]

        env = os.environ.copy()
        env["WANDB_MODE"] = "offline"
        env["PYTHONUNBUFFERED"] = "1"
        env["WARMSTART_BALL_BALANCE_ONNX_PATH"] = str(_TEACHER_POLICY)

        t_start = time.time()
        proc = subprocess.run(cmd, cwd=str(_RL_DIR), env=env)
        elapsed = time.time() - t_start

        if proc.returncode != 0:
            print(f"ERROR: Training failed with exit code {proc.returncode}!")
            sys.exit(proc.returncode)

        print(f"\n[training] Finished {args.iterations} iterations in {elapsed/60:.2f} minutes!")

    # Find checkpoint and export
    latest_pt = find_latest_checkpoint(log_root)
    export_to_onnx(latest_pt, _TARGET_ONNX, task_id)
    b_onnx = _POLICIES_DIR / "table_tennis_ball_balance_warmstart_b.onnx"
    import shutil
    shutil.copyfile(_TARGET_ONNX, b_onnx)
    print(f"[export] Saved dedicated Approach B policy copy to: {b_onnx}")

    # Verification rollout
    print("\n" + "=" * 80)
    print(" EXECUTING VERIFICATION ROLLOUT (500 STEPS / 2.5s)")
    print("=" * 80)
    eval_cmd = [
        str(_PYTHON),
        str(_MANIP_DIR / "scripts" / "test_racket_drop_and_walk.py"),
        "--racket", "table_tennis",
        "--policy", str(_TARGET_ONNX),
        "--steps", "500",
        "--walk",
        "--vx", "0.18",
    ]
    subprocess.run(eval_cmd, cwd=str(_ROOT))


if __name__ == "__main__":
    main()
