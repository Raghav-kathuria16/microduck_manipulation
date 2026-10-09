#!/usr/bin/env python3
"""Scene randomization for Microduck Apartment v1 (official robot).

Reads configs/scene_apartment_v1.yaml and applies randomized initial
conditions to a loaded MuJoCo model + data pair:
  - Robot spawn pose (trunk_base_freejoint: position + heading)
  - One interactable object (obj_*/ball_*) re-placed inside a random zone
  - Lighting preset (ambient + diffuse on headlight)

Usage as a library:
    from randomize_scene import load_scene_config, randomize_episode

Usage as a CLI:
    python scripts/randomize_scene.py            # print a random config
    python scripts/randomize_scene.py --seed 42  # reproducible
"""

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpawnPose:
    """Robot spawn pose: 2D position + heading in radians."""
    name: str
    position: list[float]
    heading: float


@dataclass
class SpawnZone:
    """Rectangular object spawn zone (floor region)."""
    name: str
    description: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        """Return a random (x, y) within this zone."""
        x = rng.uniform(self.x_min, self.x_max)
        y = rng.uniform(self.y_min, self.y_max)
        return (x, y)


@dataclass
class TargetPosition:
    """Target position (charging dock)."""
    name: str
    position: list[float]
    description: str


@dataclass
class LightingPreset:
    """Lighting preset for visual domain randomization."""
    name: str
    description: str
    ambient: list[float]
    diffuse: list[float]
    direction: list[float]


@dataclass
class SceneConfig:
    """Parsed scene configuration from YAML."""
    scene_name: str
    scene_description: str
    robot_spawn_poses: list[SpawnPose] = field(default_factory=list)
    object_spawn_zones: list[SpawnZone] = field(default_factory=list)
    target_positions: list[TargetPosition] = field(default_factory=list)
    lighting_presets: list[LightingPreset] = field(default_factory=list)


@dataclass
class EpisodeConfig:
    """Concrete configuration for a single episode after randomization."""
    robot_pose: SpawnPose
    object_zone: SpawnZone
    target: TargetPosition
    lighting: LightingPreset
    seed: int | None = None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_scene_config(config_path: str) -> SceneConfig:
    """Load and parse the scene YAML config file."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    scene_info = raw.get("scene", {})

    robot_poses = [
        SpawnPose(
            name=p["name"],
            position=p["position"],
            heading=p["heading"],
        )
        for p in raw.get("robot_spawn_poses", [])
    ]

    object_zones = [
        SpawnZone(
            name=z["name"],
            description=z.get("description", ""),
            x_min=z["bounds"]["x_min"],
            x_max=z["bounds"]["x_max"],
            y_min=z["bounds"]["y_min"],
            y_max=z["bounds"]["y_max"],
        )
        for z in raw.get("object_spawn_zones", [])
    ]

    targets = [
        TargetPosition(
            name=t["name"],
            position=t["position"],
            description=t.get("description", ""),
        )
        for t in raw.get("target_positions", [])
    ]

    lighting = [
        LightingPreset(
            name=lp["name"],
            description=lp.get("description", ""),
            ambient=lp["ambient"],
            diffuse=lp["diffuse"],
            direction=lp["direction"],
        )
        for lp in raw.get("lighting_presets", [])
    ]

    return SceneConfig(
        scene_name=scene_info.get("name", "unknown"),
        scene_description=scene_info.get("description", ""),
        robot_spawn_poses=robot_poses,
        object_spawn_zones=object_zones,
        target_positions=targets,
        lighting_presets=lighting,
    )


# ---------------------------------------------------------------------------
# Randomization
# ---------------------------------------------------------------------------

def sample_episode_config(
    config: SceneConfig,
    rng: np.random.Generator,
    lighting_name: str | None = None,
) -> EpisodeConfig:
    """Sample a random episode configuration from the scene config.

    Args:
        config: Parsed scene configuration.
        rng: NumPy random generator for reproducibility.
        lighting_name: If provided, use this specific lighting preset
            instead of choosing randomly.

    Returns:
        A concrete EpisodeConfig ready to apply.
    """
    # Robot spawn pose
    pose_idx = rng.integers(0, len(config.robot_spawn_poses))
    robot_pose = config.robot_spawn_poses[pose_idx]

    # Object spawn zone + position
    zone_idx = rng.integers(0, len(config.object_spawn_zones))
    zone = config.object_spawn_zones[zone_idx]
    obj_pos = zone.sample(rng)

    # Target (v1: always use the first/default)
    target = config.target_positions[0]

    # Lighting
    if lighting_name is not None:
        lighting = next(
            (lp for lp in config.lighting_presets if lp.name == lighting_name),
            config.lighting_presets[0],
        )
    else:
        light_idx = rng.integers(0, len(config.lighting_presets))
        lighting = config.lighting_presets[light_idx]

    return EpisodeConfig(
        robot_pose=robot_pose,
        object_zone=zone,
        target=target,
        lighting=lighting,
    )


def apply_episode_config(model, data, episode: EpisodeConfig, rng: np.random.Generator) -> None:
    """Apply an EpisodeConfig to a loaded MuJoCo model + data.

    Sets the robot root freejoint (trunk_base_freejoint) from the episode
    pose, scatters the free objects inside the episode's zone at rest
    height, zeroes velocities, and applies the lighting preset.

    Args:
        model: mujoco.MjModel
        data: mujoco.MjData
        episode: The concrete episode configuration to apply.
        rng: NumPy random generator for object scatter sampling.
    """
    import mujoco
    # --- Robot spawn pose ---
    # The robot root is trunk_base_freejoint (qpos = [x, y, z, qw, qx, qy, qz]).
    jnt_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    if jnt_root >= 0:
        adr = model.jnt_qposadr[jnt_root]
        data.qpos[adr + 0] = episode.robot_pose.position[0]
        data.qpos[adr + 1] = episode.robot_pose.position[1]
        # Keep qpos0 height (standing clearance); identity quat with yaw.
        data.qpos[adr + 2] = model.qpos0[adr + 2]
        yaw = episode.robot_pose.heading
        data.qpos[adr + 3] = np.cos(yaw / 2.0)
        data.qpos[adr + 4] = 0.0
        data.qpos[adr + 5] = 0.0
        data.qpos[adr + 6] = np.sin(yaw / 2.0)

    # Every body with a free joint besides the robot root is an object;
    # scatter them inside the episode zone with per-object rejection
    # sampling so each stays in-bounds and no two objects overlap.
    root_id = jnt_root
    zone = episode.object_zone
    placed: list[tuple[float, float]] = []
    for j in range(model.njnt):
        if j == root_id or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        bid = model.jnt_bodyid[j]
        # Rest height: max geom bottom offset among this body's geoms.
        z = 0.0
        for g in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]):
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue  # visual geom, no collision footprint
            z = max(z, float(model.geom_pos[g][2]) + float(model.geom_size[g][2]))
        # Reject-sample (x, y): inside zone + at least 9cm from all
        # previously placed objects (objects are <= 8cm wide).
        for _ in range(200):
            px, py = zone.sample(rng)
            if all((px - qx) ** 2 + (py - qy) ** 2 >= 0.09**2 for qx, qy in placed):
                break
        placed.append((px, py))
        data.qpos[model.jnt_qposadr[j] + 0] = px
        data.qpos[model.jnt_qposadr[j] + 1] = py
        data.qpos[model.jnt_qposadr[j] + 2] = z
        data.qpos[model.jnt_qposadr[j] + 3: model.jnt_qposadr[j] + 7] = [1.0, 0.0, 0.0, 0.0]

    # --- Zero all velocities ---
    data.qvel[:] = 0.0

    # --- Lighting preset ---
    # Update the headlight on the model's vis struct
    model.vis.headlight.ambient[:] = episode.lighting.ambient
    model.vis.headlight.diffuse[:] = episode.lighting.diffuse

    # Forward kinematics to update derived quantities
    mujoco.mj_forward(model, data)


def randomize_episode(
    model,
    data,
    config: SceneConfig,
    rng: np.random.Generator,
    lighting_name: str | None = None,
) -> EpisodeConfig:
    """Sample and apply a random episode configuration (convenience wrapper).

    Args:
        model: mujoco.MjModel
        data: mujoco.MjData
        config: Parsed scene configuration.
        rng: NumPy random generator.
        lighting_name: Optional specific lighting preset name.

    Returns:
        The applied EpisodeConfig.
    """
    episode = sample_episode_config(config, rng, lighting_name)
    apply_episode_config(model, data, episode, rng)
    return episode


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def format_episode(ep: EpisodeConfig) -> str:
    """Pretty-print an episode configuration."""
    lines = [
        "=" * 50,
        "Randomized Episode Configuration",
        "=" * 50,
        f"  Robot pose:  {ep.robot_pose.name}",
        f"    Position:  ({ep.robot_pose.position[0]:.2f}, {ep.robot_pose.position[1]:.2f})",
        f"    Heading:   {ep.robot_pose.heading:.2f} rad ({np.degrees(ep.robot_pose.heading):.1f} deg)",
        f"  Object zone: {ep.object_zone.name} — {ep.object_zone.description}",
        f"  Target:      {ep.target.name} at ({ep.target.position[0]:.2f}, {ep.target.position[1]:.2f})",
        f"  Lighting:    {ep.lighting.name} — {ep.lighting.description}",
        "=" * 50,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Sample a random episode config")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--lighting", type=str, default=None, help="Force a specific lighting preset")
    parser.add_argument("-n", type=int, default=1, help="Number of configs to sample")
    args = parser.parse_args()

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(workspace_root, "configs", "scene_apartment_v1.yaml")

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    config = load_scene_config(config_path)
    rng = np.random.default_rng(args.seed)

    for i in range(args.n):
        if args.n > 1:
            print(f"\n--- Sample {i + 1}/{args.n} ---")
        ep = sample_episode_config(config, rng, args.lighting)
        print(format_episode(ep))


if __name__ == "__main__":
    main()
