# Behavior Cloning Policies

This directory holds BC model configurations and training artifacts,
organized by architecture.

## Subdirectories

### `act/`
Action Chunking with Transformers — smallest, fastest to prototype locally
on M2. Recommended as the first architecture to validate the data pipeline.

### `diffusion_policy/`
Diffusion Policy — slightly larger, better multi-modal action handling.
Local prototyping possible but slower than ACT.

### `smolvla/`
SmolVLA fine-tuning — vision-language-action model. Requires cloud GPU for
practical training time, but uses the same LeRobotDataset format.

## Each architecture subdirectory will contain:

```
{architecture}/
├── config.yaml          # Model hyperparameters, training config
├── checkpoints/         # Training checkpoints (git-ignored)
│   ├── local/           # CPU/MPS prototyping runs
│   └── cloud/           # Cloud GPU production runs
└── eval/                # Evaluation results, metrics, plots
```

## Naming convention

Checkpoints: `sockpickplace_{method}_{variant}_{YYYYMMDD}`

Example: `sockpickplace_act_batch03_20260915`
