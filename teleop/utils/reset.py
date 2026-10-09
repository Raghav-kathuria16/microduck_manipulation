"""Episode reset: randomise spawn/sock/lighting, zero policy state."""

from __future__ import annotations
import os
import sys

import mujoco
import numpy as np

# Re-use the existing randomizer from scripts/
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")
sys.path.insert(0, _SCRIPTS)
from randomize_scene import load_scene_config, randomize_episode, EpisodeConfig  # noqa: E402


def reset_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene_config,
    rng: np.random.Generator,
    policy_runner=None,
    lighting_name: str | None = None,
) -> EpisodeConfig:
    """Reset the simulation to a new random episode.

    Steps:
    1. Randomise robot spawn pose, sock position, lighting.
    2. Call mj_forward to propagate the new qpos.
    3. If a policy_runner is supplied, zero its state (last_action, command).

    Returns the EpisodeConfig so the caller can log metadata.
    """
    ep = randomize_episode(model, data, scene_config, rng, lighting_name)

    if policy_runner is not None:
        policy_runner.reset()

    return ep
