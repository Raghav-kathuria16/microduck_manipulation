# RL Reference

This directory does **NOT** contain RL training code or checkpoints.
RL training is owned by `microduck_rl`, not this repo.

## Purpose

This folder holds only **reference pointers** to `microduck_rl` assets
that this project depends on:

- **Walking policy checkpoint** — used as the locomotion layer in the
  teleop control hierarchy (Teleop Spec §3.2). The teleop app loads this
  checkpoint at runtime to convert velocity commands into stable gait.

## Current reference

See `external/microduck_rl_pin.json` for the version-pinned reference to
`microduck_rl`. This file specifies which commit/tag and which checkpoint
path within that repo this workspace was built against.

## When to update

Update the pin when:
- `microduck_rl` releases a better walking policy checkpoint
- The robot MJCF changes (e.g., beak DOF added per FSD M0)
- Any breaking change in the action/observation space
