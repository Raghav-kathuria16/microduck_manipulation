#!/usr/bin/env python3
"""Train Microduck Steadicam Residual RL Policy (Tableware Spoon & Lemon Balancing).

Architecture:
- Base: Official walking policy (10 leg joints 100% frozen, walking gait preserved).
- Residual Policy: 4 neck joints (neck_pitch, head_pitch, head_yaw, head_roll) trained
  with PPO to eliminate 3D lateral roll wobble, sagittal pitch bobbing, and heading twitch.
- Optimization: Runs on CPU with parallel environments using Warp/MuJoCo and RSL-RL.
"""

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_RL_DIR = os.path.join(_MANIP_DIR, "microduck_rl")
if os.path.exists(_RL_DIR):
    sys.path.insert(0, os.path.join(_RL_DIR, "src"))

import mjlab.tasks.registry as reg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
import mjlab_microduck


def main():
    parser = argparse.ArgumentParser(description="Train Steadicam Residual RL Policy")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments (default: 16)")
    parser.add_argument("--iterations", type=int, default=300, help="Number of PPO iterations (default: 300)")
    parser.add_argument("--steps-per-env", type=int, default=24, help="Rollout steps per env per iteration (default: 24)")
    parser.add_argument("--log-dir", type=str, default=os.path.join(_MANIP_DIR, "logs/steadicam_residual"), help="Log directory")
    args = parser.parse_args()

    task_id = "Mjlab-Spoon-Lemon-Steadicam-Residual-MicroDuck"
    print(f"\n" + "=" * 70)
    print(f"STARTING STEADICAM RESIDUAL RL TRAINING: {task_id}")
    print(f"Envs: {args.num_envs} | Iterations: {args.iterations} | Device: CPU")
    print("=" * 70)

    env_cfg = reg.load_env_cfg(task_id)
    rl_cfg = reg.load_rl_cfg(task_id)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 5.0
    rl_cfg.num_steps_per_env = args.steps_per_env
    rl_cfg.max_iterations = args.iterations
    rl_cfg.save_interval = 50

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("[1/3] Initializing simulation environments on CPU...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    vec_env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    print(f"      Environments initialized in {time.time() - t0:.2f}s")

    print("[2/3] Initializing MicroduckOnPolicyRunner...")
    runner_cls = reg.load_runner_cls(task_id)
    agent_cfg = asdict(rl_cfg)
    agent_cfg["logger"] = "tensorboard"

    runner = runner_cls(vec_env, agent_cfg, str(log_dir), "cpu")

    print(f"[3/3] Commencing {args.iterations} PPO training iterations...")
    t_train0 = time.time()
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    t_train_total = time.time() - t_train0

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE in {t_train_total:.1f}s ({t_train_total / args.iterations:.2f}s per iteration)")
    print(f"Final checkpoint saved in: {log_dir}")
    print("=" * 70 + "\n")

    env.close()


if __name__ == "__main__":
    main()
