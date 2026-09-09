# Microduck Sock Pick-and-Place — Local Teleop, Synthetic Data & Policy Strategy

This document is self-contained. It assumes no prior context beyond: Microduck
is a duck-shaped bipedal robot (legs + neck/head + articulated beak, ~15 DOF)
built by Pollen Robotics, simulated in MuJoCo, with an existing RL repo
(`microduck_rl`, using MuJoCo Warp + PPO via `mjlab`) that trains locomotion
and trick skills but does **not** yet have a pick-and-place skill. A separate
document (the FSD) already specifies the pure-RL path to that skill. This
document specifies an **alternative/complementary data-first path**: collect
human teleoperated demonstrations locally, expand them synthetically, and
train a policy via behavior cloning (BC), optionally warm-starting RL
afterward.

No code is included anywhere in this document intentionally — it is a design
and task spec, not an implementation.

---

## PART 1 — End Goal, Restated Precisely

**Target capability:** Microduck operates in a home-like environment,
locates a sock on the floor, walks to it, grasps it with its beak, walks
while carrying it to a different location (e.g., a laundry basket), releases
it there, and returns to a stable stance — first in simulation, eventually
transferred to the physical robot.

**Development strategy (three tracks that converge):**

1. **Local scene + teleoperation (Parts 2–3)** — build a MuJoCo apartment
   scene and a human-drivable control scheme so you can personally
   demonstrate the task and record it as training data.
2. **Synthetic data expansion (Part 4)** — teleop alone won't produce enough
   data or diversity; systematically generate more from what you collect.
3. **Policy learning (Part 5)** — decide between BC, RL, or a hybrid, and
   train accordingly.

**Hardware constraint that shapes everything:** MacBook M2 (Apple Silicon,
CPU + Metal/MPS, **no CUDA**).
- MuJoCo (CPU mode) runs fine locally — scene building, teleoperation, and
  data recording all happen on the M2.
- MuJoCo **Warp** (GPU-parallel) does **not** run locally — RL training
  requires a cloud GPU.
- BC model training is a spectrum: small models (ACT, small Diffusion
  Policy) can be prototyped locally on CPU/MPS slowly; VLA-scale
  (SmolVLA, pi0/pi0.5) needs cloud GPU.

---

## PART 2 — Local Apartment Scene Design

### 2.1 Purpose
This is **not** the same scene as the RL repo's flat/rough training terrains.
Its job is to be a teleoperation and data-collection environment: visually
and spatially rich enough that demonstrations generalize, but simple enough
to build, randomize, and render quickly on a laptop.

### 2.2 Scale and layout (v1)
A single-room-plus-alcove apartment segment, duck-scale:

- **Main room:** ~3.5 m × 3.0 m open floor area.
- **Alcove:** ~1.5 m × 1.5 m, connected via an open doorway.
- **Wall height:** bounds the space, supports wall-mounted cameras.

```
 ┌───────────────────────────────┬──────────────┐
 │                                │              │
 │        MAIN ROOM               │   ALCOVE      │
 │   ┌────┐                       │   ┌────────┐ │
 │   │RUG │      [duck spawn]     │   │ BASKET │ │
 │   └────┘         ●             │   │ (target)│ │
 │                                │   └────────┘ │
 │     ○ sock spawn zone (A)      │              │
 │           ○ sock spawn zone (B)│              │
 │                                │              │
 └───────────────────────────────┴──────────────┘
```

### 2.3 Object inventory

| Object | Role | Notes |
|---|---|---|
| Floor | static, textured | Vary friction/material for DR later |
| 2–4 wall segments | static, bound the space | Simple boxes |
| Rug | static, visual + friction variation | Texture contrast for vision |
| Sock (target) | free body, rigid-body approx (~40–60 g capsule) | Same definition as the RL task spec |
| Laundry basket | static container + zone volume | Visual container + success detection zone |
| 1–2 furniture pieces | static clutter | Visual/spatial diversity only in v1 |
| Sock spawn zones | invisible markers | For randomization |

### 2.4 Camera rig

1. **Top-down orthographic** — bird's-eye, primary spatial awareness.
2. **Chase camera** — third-person follow for carry phase.
3. **Beak-cam first-person** — grasp precision + training data camera.

GUI: **top-down + beak-cam side by side**, chase-cam togglable.

### 2.5 Randomization surface
Design everything as swappable/parametrized (see `configs/`):
- Sock spawn zones, target position, robot spawn pose, lighting presets.

---

## PART 3 — Teleoperation System Design

### 3.1 Why not direct joint-space teleop
~15 DOF cannot be usefully driven by a human in real time. Teleop must be
**hierarchical**.

### 3.2 Control hierarchy

| Layer | Human controls | Executor | Why |
|---|---|---|---|
| Locomotion | 2D velocity/direction (keyboard/gamepad) | Existing trained walk policy from `microduck_rl` | Natural stable gait from simple command |
| Head/neck aim | 2D cursor/stick → look-at point | IK/PD controller | Intuitive without exposing individual joints |
| Beak position | Relative nudges near sock | IK controller (local range) | Precision only when needed |
| Grasp | Button press/release | Beak actuator open/close | Matches real hardware's simple action |

### 3.3 Per-timestep logging

- Proprioceptive state (joints, IMU)
- Camera frame(s) (beak-cam minimum)
- High-level command (velocity, look-at, grasp state) — **the BC action label**
- Sock pose (privileged, for success detection)
- Episode success flag (sock in target zone + released)
- Natural-language task string per episode

### 3.4 Episode structure

- **Reset:** robot to spawn pose, sock to random zone.
- **Termination:** success, fall, drop, or timeout.
- **Retry-in-place:** abort without saving bad episodes early on.

---

## PART 4 — Synthetic Data Generation

| # | Technique | Effort | Notes |
|---|---|---|---|
| 1 | Randomized re-simulation from teleop commands | Low | Replay commands against new conditions |
| 2 | Scripted/procedural demonstrations | Medium | Heuristic controllers, zero human time per episode |
| 3 | Left-right mirroring | Low | Free 2x multiplier |
| 4 | Time-warping / speed augmentation | Low | Regularization against timing overfitting |
| 5 | State-noise injection + re-verification | Medium | Widen state basin near real demos |
| 6 | Visual domain randomization | Low–Med | Varies lighting/textures/camera for vision policies |
| 7 | Procedural apartment variants | Medium | Prevents floor-plan overfitting |
| 8 | Self-play / DAgger-style correction | Med–High | Targets BC distribution-shift; requires v1 policy first |

**Sequencing:** teleop first (20–50 episodes) → #1 and #3 → #2 → #6–#8.

---

## PART 5 — Policy Learning Strategy

### 5.1 Comparison

| | BC | RL | Hybrid (BC→RL) |
|---|---|---|---|
| Needs reward? | No | Yes (hard for this task) | Only for fine-tune |
| Needs demo data? | Yes | No | Yes for warm-start |
| Local M2 feasible? | Small models: yes, slow | No (needs CUDA) | No for RL half |
| Main failure mode | Distribution shift | Reward hacking / long training | Mitigates both |
| Time-to-first-policy | Fastest | Slowest | Medium |

### 5.2 Recommendation

1. **Start with BC.** ACT or small Diffusion Policy locally to validate pipeline.
2. **Scale up** with SmolVLA/pi0 fine-tuning on cloud GPU, same data format.
3. **Treat RL as robustness fine-tuning**, warm-started from BC.

### 5.3 Phased roadmap

| Phase | Deliverable | Depends on |
|---|---|---|
| 1 | Apartment scene built and loadable | — |
| 2 | Teleop functional, can complete one episode | Phase 1 |
| 3 | 20–50 clean teleop demos in LeRobotDataset format | Phase 2 |
| 4 | Synthetic augmentation (#1, #3) applied | Phase 3 |
| 5 | First BC policy trained | Phase 4 |
| 6 (stretch) | Procedural demos (#2) + DAgger (#8) | Phase 5 |
| 7 (stretch) | RL fine-tune from BC warm-start | Phase 5 + FSD M0–M1 |
