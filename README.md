# microduck_manipulation

Apartment scene, teleop data collection, synthetic data generation, and
behavior-cloning policy training for Microduck's sock pick-and-place task.

> **This is a separate workspace from `microduck_rl`.** It owns the data-first
> (BC → RL hybrid) path to the pick-and-place skill. The RL repo owns the
> robot model, locomotion policy, and pure-RL training.

## Status

🚧 **Project scaffolding complete. No code yet.** Scene, teleop app, and
training scripts are designed but not implemented.

## Quick Start

1. Clone this repo
2. Copy `.env.example` → `.env` and fill in your HF Hub token
3. Verify the `microduck_rl` version pin in `external/microduck_rl_pin.json`
4. Read the docs in order (see below)

## Documentation

Read in this order:

| # | Document | What it covers |
|---|---|---|
| 1 | [`docs/microduck_workspace_architecture.md`](docs/microduck_workspace_architecture.md) | Where everything lives, how it's managed |
| 2 | [`docs/microduck_teleop_data_policy_spec.md`](docs/microduck_teleop_data_policy_spec.md) | What to build first: scene, teleop, synthetic data, BC strategy |
| 3 | FSD (external) | The longer-horizon RL task spec that the BC policy feeds into |
| 4 | [`CHANGELOG.md`](CHANGELOG.md) | Accumulated design decisions and reasoning |

## Directory Structure

```
microduck_manipulation/
├── scenes/          # MuJoCo apartment scene definitions
├── teleop/          # Hierarchical teleop app (reserved)
├── data/            # Raw → processed → synthetic → manifests
├── policies/        # BC models, RL reference pointer, export staging
├── configs/         # Parametrized scene/teleop/camera/logging configs
├── docs/            # Project documentation
├── scripts/         # Operational entry points (reserved)
├── external/        # Dependency version pins (microduck_rl)
├── .env.example     # Required environment variables
├── CHANGELOG.md     # Append-only reasoning log
└── README.md        # This file
```

## Compute Environments

| Environment | Machine | Purpose |
|---|---|---|
| Local dev | MacBook M2 (CPU/MPS) | Scene authoring, teleop, data work, small BC prototyping |
| Cloud | HF Jobs / Colab (CUDA) | RL training, VLA-scale BC, large synthetic batches |

## Phased Roadmap

1. ✅ Workspace & docs scaffolded
2. ⬜ Apartment scene built and loadable
3. ⬜ Teleop hierarchy functional
4. ⬜ 20–50 clean teleop demonstrations
5. ⬜ Synthetic augmentation applied
6. ⬜ First BC policy trained
7. ⬜ (Stretch) Procedural demos + DAgger
8. ⬜ (Stretch) RL fine-tune from BC warm-start

## License

TBD
