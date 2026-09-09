# Changelog

All notable decisions, design changes, and milestones for the
`microduck_manipulation` project are documented here.

This is an **append-only reasoning log**, not just a list of code changes.
Each entry explains **why** something was decided, not just what changed —
this is the most valuable file for handoffs, since it captures reasoning
that the folder structure alone cannot.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased]

### 2026-09-09 — Project Initialization

**What:** Created the `microduck_manipulation` workspace with full directory
structure, documentation set, and configuration scaffolding per the
Workspace Architecture spec.

**Why:** This project is intentionally separate from `microduck_rl` because
the two have different dependency stacks (teleop/GUI/BC vs. MuJoCo Warp/PPO),
different release cadences, and different primary iteration loops (local
scene/data work vs. cloud RL training). Keeping them as separate repos with
an explicit version-pinned reference (`external/microduck_rl_pin.json`)
avoids polluting either repo with the other's concerns.

**Key decisions captured:**
- **BC-first strategy** (not RL-first): behavior cloning from teleop demos
  sidesteps the reward-design problem for the multi-phase pick-place task.
  RL is planned as a later robustness fine-tuning stage, warm-started from
  the BC policy. Full rationale in `docs/microduck_teleop_data_policy_spec.md` §5.
- **Hierarchical teleop** (not direct joint-space): human controls high-level
  commands (velocity, look-at, grasp toggle); existing walk policy + IK
  controllers fill in low-level joint detail. This matches the real robot's
  deployment-time API and keeps demo data at the right abstraction level.
- **LeRobotDataset format**: chosen for data interchange so the same dataset
  works with LeRobot's BC training pipelines, SmolVLA fine-tuning, and
  potential pi0-style training — no format conversion needed downstream.
- **Data lives on HF Hub, code lives in git**: git tracks code/configs/docs
  and lightweight manifests only; all bulky data and model weights are
  published to Hugging Face Hub repos, versioned there.
