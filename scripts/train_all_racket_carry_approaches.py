#!/usr/bin/env python3
"""
train_all_racket_carry_approaches.py - Sequential Benchmark Runner for Problem 1.

Executes all 3 contrasting approaches sequentially (zero CPU throttling):
1. Approach A: Precision Reward Shaping with Flat-Bottom Tolerance Bands
2. Approach B: Residual RL Architecture (Frozen Locomotion Engine)
3. Approach C: Virtual Impedance Potential Well (Viscoelastic Stabilization)

Auto-exports ONNX policies to microduck_manipulation/policies/onnx/
and prints ready-to-use GLFW visualization commands.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent.parent
_RL_DIR = _ROOT / "microduck_rl"
_PYTHON = _RL_DIR / ".venv" / "bin" / "python"
_TRAIN_BIN = _RL_DIR / ".venv" / "bin" / "train"
_POLICIES_DIR = _ROOT / "microduck_manipulation" / "policies" / "onnx"


APPROACHES = {
    "A": {
        "name": "Approach A (Precision Reward Shaping)",
        "task_id": "Mjlab-TableTennis-RacketCarry-Shaping-MicroDuck",
        "run_pattern": "*table_tennis_racket_carry_shaping*",
        "onnx_file": "table_tennis_racket_carry_shaping.onnx",
    },
    "B": {
        "name": "Approach B (Residual RL Architecture)",
        "task_id": "Mjlab-TableTennis-RacketCarry-Residual-MicroDuck",
        "run_pattern": "*table_tennis_racket_carry_residual*",
        "onnx_file": "table_tennis_racket_carry_residual.onnx",
    },
    "C": {
        "name": "Approach C (Virtual Impedance Potential Well)",
        "task_id": "Mjlab-TableTennis-RacketCarry-Impedance-MicroDuck",
        "run_pattern": "*table_tennis_racket_carry_impedance*",
        "onnx_file": "table_tennis_racket_carry_impedance.onnx",
    },
}


def find_latest_checkpoint(log_root: Path, pattern: str) -> Path:
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


def run_benchmark(approach_keys, iterations=50, num_envs=64):
    print("=" * 80)
    print(" PROBLEM 1: TABLE TENNIS RACKET CARRY BENCHMARK SUITE")
    print(f" Approaches to run sequentially: {approach_keys}")
    print(f" Iterations per run: {iterations} | Parallel Envs: {num_envs}")
    print("=" * 80)

    log_root = _RL_DIR / "logs" / "rsl_rl" / "table_tennis_racket_carry"
    log_root.mkdir(parents=True, exist_ok=True)

    completed_runs = {}

    for key in approach_keys:
        info = APPROACHES[key]
        print("\n" + "#" * 80)
        print(f" LAUNCHING: {info['name']}")
        print(f" Task ID: {info['task_id']}")
        print(f" Target ONNX: {info['onnx_file']}")
        print("#" * 80)

        t_start = time.time()
        cmd = [
            str(_TRAIN_BIN),
            info["task_id"],
            "--env.scene.num-envs", str(num_envs),
            "--agent.max_iterations", str(iterations),
            "--agent.save-interval", str(min(iterations, 25)),
            "--agent.logger", "tensorboard",
        ]

        env = os.environ.copy()
        env["WANDB_MODE"] = "offline"
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.run(cmd, cwd=str(_RL_DIR), env=env)
        elapsed = time.time() - t_start

        if proc.returncode != 0:
            print(f"ERROR: {info['name']} failed with exit code {proc.returncode}!")
            continue

        print(f"\n[training] {info['name']} finished in {elapsed/60:.2f} minutes!")

        # Discover checkpoint and export
        try:
            pt = find_latest_checkpoint(log_root, info["run_pattern"])
            onnx_path = _POLICIES_DIR / info["onnx_file"]
            export_to_onnx(pt, onnx_path, info["task_id"])
            completed_runs[key] = {
                "name": info["name"],
                "onnx_path": onnx_path,
                "elapsed": elapsed,
            }
        except Exception as e:
            print(f"Export warning: {e}")

    # Summary block
    print("\n" + "=" * 80)
    print(" BENCHMARK SUITE COMPLETE! SUMMARY & VISUALIZATION COMMANDS")
    print("=" * 80)
    for key, res in completed_runs.items():
        print(f"\n[{key}] {res['name']} (Completed in {res['elapsed']/60:.2f} mins):")
        print(f"    Policy: {res['onnx_path']}")
        print("    Interactive GLFW Viewer Command:")
        print(f"    ./microduck_rl/.venv/bin/mjpython microduck_manipulation/scripts/test_racket_drop_and_walk.py \\")
        print(f"        --racket table_tennis \\")
        print(f"        --policy {res['onnx_path']} \\")
        print(f"        --no-ball --gui --loop")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Sequential Benchmark Suite for Problem 1 Racket Carrying")
    parser.add_argument("--approaches", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"],
                        help="Approaches to run (default: A B C)")
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations (default: 50)")
    parser.add_argument("--envs", type=int, default=64, help="Parallel environments (default: 64)")
    args = parser.parse_args()

    run_benchmark(args.approaches, iterations=args.iterations, num_envs=args.envs)


if __name__ == "__main__":
    main()
