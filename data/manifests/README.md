# Data Manifests

This directory contains lightweight, git-tracked index files that describe
the dataset without containing the data itself.

All actual data files (`data/raw/`, `data/processed/`, `data/synthetic/`)
are git-ignored and either stored locally as working caches or published
to the Hugging Face Hub as the canonical versioned copy.

## Files (to be created as data is collected)

- `episode_index.jsonl` — one line per episode: ID, source technique,
  outcome, scene variant, spawn zone, duration, timestamp.
- `dataset_versions.yaml` — log of each version pushed to HF Hub:
  version tag, episode count, technique breakdown, date.
- `technique_summary.yaml` — per-technique statistics: episode count,
  success rate, mean duration.

## Current Status

No data collected yet. Manifests will be generated automatically by the
data pipeline scripts (to be implemented in `scripts/`).
