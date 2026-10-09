"""LeRobot Dataset exporter.

Converts raw teleop episodes (data/raw/episode_NNNN/) to LeRobotDataset format
(data/lerobot/) that is directly usable for BC training with lerobot.

LeRobotDataset v2 schema:
  data/lerobot/
    meta/
      tasks.jsonl            - one JSON per task instruction
      episodes.jsonl         - one JSON per episode
      info.json              - dataset metadata
    data/
      chunk-000/
        episode_NNNN.parquet - state + action columns per timestep
    videos/
      chunk-000/
        observation.images.beak_cam/
          episode_NNNN.mp4   - beak-cam video

Usage:
    python -m teleop.recorder.lerobot_exporter  # converts all raw episodes
    python -m teleop.recorder.lerobot_exporter --episode 3  # single episode
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys

import numpy as np

WORKSPACE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
RAW_DIR    = os.path.join(WORKSPACE_ROOT, "data", "raw")
LEROBOT_DIR = os.path.join(WORKSPACE_ROOT, "data", "lerobot")


# Column names for the parquet files
STATE_COLS  = (
    [f"observation.state.joint_pos_{i}"  for i in range(14)] +
    [f"observation.state.joint_vel_{i}"  for i in range(14)] +
    [f"observation.state.imu_quat_{i}"   for i in range(4)]  +
    [f"observation.state.imu_ang_vel_{i}" for i in range(3)] +
    [f"observation.state.imu_accel_{i}"  for i in range(3)]  +
    [f"observation.state.trunk_pose_{i}" for i in range(7)]
)
ACTION_COLS = ["action.vx", "action.wz",
               "action.neck", "action.head_pitch", "action.head_yaw", "action.head_roll",
               "action.grasp"]
PRIV_COLS   = ([f"privileged.sock_{k}" for k in ["x","y","z","qw","qx","qy","qz"]] +
               ["privileged.in_target"])


def export_episode(ep_dir: str, ep_id: int, chunk_dir: str, video_dir: str) -> dict:
    """Convert one raw episode to parquet + move video. Returns episode metadata dict."""
    states     = np.load(os.path.join(ep_dir, "states.npy"))
    actions    = np.load(os.path.join(ep_dir, "actions.npy"))
    privileged = np.load(os.path.join(ep_dir, "privileged.npy"))
    timestamps = np.load(os.path.join(ep_dir, "timestamps.npy"))
    with open(os.path.join(ep_dir, "metadata.json")) as f:
        meta = json.load(f)

    T = len(states)

    # Build parquet row by row
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed — run: pip install pandas pyarrow")
        sys.exit(1)

    rows = {}
    for i, col in enumerate(STATE_COLS):
        rows[col] = states[:, i].tolist() if i < states.shape[1] else [0.0]*T
    for i, col in enumerate(ACTION_COLS):
        rows[col] = actions[:, i].tolist() if i < actions.shape[1] else [0.0]*T
    for i, col in enumerate(PRIV_COLS):
        rows[col] = privileged[:, i].tolist() if i < privileged.shape[1] else [0.0]*T
    rows["timestamp"] = timestamps.tolist()
    rows["episode_index"] = [ep_id] * T
    rows["frame_index"]   = list(range(T))

    df = pd.DataFrame(rows)
    pq_path = os.path.join(chunk_dir, f"episode_{ep_id:04d}.parquet")
    df.to_parquet(pq_path, index=False)

    # Move video
    src_video = os.path.join(ep_dir, "video_beak.mp4")
    if os.path.exists(src_video):
        dst_video = os.path.join(video_dir, f"episode_{ep_id:04d}.mp4")
        shutil.copy2(src_video, dst_video)

    return {
        "episode_index": ep_id,
        "task":          meta.get("task_instruction", "pick_and_place"),
        "length":        T,
        "outcome":       meta.get("outcome", "unknown"),
        "duration_s":    meta.get("duration_seconds", 0.0),
    }


def export_all(specific_ep: int | None = None):
    if not os.path.exists(RAW_DIR):
        print(f"No raw data found at {RAW_DIR}")
        return

    ep_dirs = sorted(
        [d for d in os.listdir(RAW_DIR) if d.startswith("episode_")],
        key=lambda d: int(d.split("_")[1]),
    )
    if specific_ep is not None:
        ep_dirs = [d for d in ep_dirs if int(d.split("_")[1]) == specific_ep]

    if not ep_dirs:
        print("No episodes to export.")
        return

    chunk_dir = os.path.join(LEROBOT_DIR, "data", "chunk-000")
    video_dir = os.path.join(LEROBOT_DIR, "videos", "chunk-000",
                             "observation.images.beak_cam")
    meta_dir  = os.path.join(LEROBOT_DIR, "meta")
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(meta_dir,  exist_ok=True)

    all_ep_meta = []
    tasks = set()
    for ep_dir_name in ep_dirs:
        ep_id  = int(ep_dir_name.split("_")[1])
        ep_dir = os.path.join(RAW_DIR, ep_dir_name)
        print(f"  Exporting episode {ep_id}...", end=" ", flush=True)
        try:
            em = export_episode(ep_dir, ep_id, chunk_dir, video_dir)
            all_ep_meta.append(em)
            tasks.add(em["task"])
            print(f"OK ({em['length']} steps, {em['outcome']})")
        except Exception as e:
            print(f"FAILED: {e}")

    # tasks.jsonl
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for i, task in enumerate(sorted(tasks)):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")

    # episodes.jsonl
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for em in all_ep_meta:
            f.write(json.dumps(em) + "\n")

    # info.json
    info = {
        "dataset_name": "microduck_sock_pickplace_teleop",
        "total_episodes": len(all_ep_meta),
        "total_frames":   sum(e["length"] for e in all_ep_meta),
        "fps":            50,
        "robot_type":     "microduck",
        "observation_space": {
            "state":  {"shape": [51], "dtype": "float32"},
            "images": {"beak_cam": {"shape": [480, 640, 3], "dtype": "uint8"}},
        },
        "action_space": {"shape": [7], "dtype": "float32"},
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Exported {len(all_ep_meta)} episodes → {LEROBOT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export raw teleop episodes to LeRobotDataset")
    parser.add_argument("--episode", type=int, default=None, help="Export a single episode ID")
    args = parser.parse_args()
    export_all(args.episode)
