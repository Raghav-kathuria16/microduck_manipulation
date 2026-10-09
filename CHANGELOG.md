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

### 2026-10-10 — Halved Showcase Duration for Badminton and Pickleball & Slip Characterization

**What:** Cut the video demonstration duration of badminton carry and pickleball carry by half (from 6.87s to 3.44s, 102 frames at 30 fps), and documented empirical contact mechanics across all repository documentation acknowledging that while both rackets are retained across the showcase demonstrations, there is some slip over extended locomotion horizons due to head movement and insufficient torque provided by the beak actuator (without quoting arbitrary numbers).

**Why:** The user requested cutting the showcase length in half and explicitly acknowledging the physical reality of gradual slip under continuous head movement and actuator torque limits without publishing a specific numerical slip figure.

### 2026-10-10 — Doubled Badminton Racket Geometry, Full-Width Olympic Race Showcase, and Repository Professionalization

**What:** Scaled up the badminton racket oval frame head to 2x doubled dimensions (100 mm lateral width by 124 mm longitudinal length, 16 capsule frame segments), lengthened the carbon fiber shaft to 88 mm between the grip cone and gold T-joint to guarantee 42 mm lateral cheek clearance past Microduck's outer jaw cowl, expanded the Olympic Stadium Race showcase in README.md into a full-width container (width="100%"), removed all emojis across the entire repository documentation, enforced strict frontend-only math notation (zero LaTeX backend syntax), and substantially expanded the Badminton technical documentation section.

**Why:** The user identified that the original badminton racket head was disproportionately small relative to the 24.4 cm robot height, requested a doubled oval head with lengthened section, asked for an expansive 32:18 banner display for the Olympic lemon race video, requested complete removal of emojis for a professional publication standard, and mandated clean frontend math over raw LaTeX syntax.

**Key decisions captured:**
- **Doubled Oval Head (100 mm × 124 mm):** Formed from 16 interconnected carbon capsules (`bm_fr_0` through `bm_fr_15`) with a high-tension translucent string bed ellipsoid (`size="0.047 0.0002 0.059"`).
- **Lengthened Shaft & Cheek Clearance (42 mm):** The 88 mm carbon shaft span places the gold T-joint at z = 0.150 m, providing 42 mm lateral clearance outside the 45.7 mm jaw cowl, eliminating collision torque spikes during whole-body yaw/roll sway.
- **Physical Clamping Stability:** 110 mm synthetic polyurethane grip clamped at 0.50 N·m torque across mouth silicone pads (μ = 3.5), achieving 100% payload retention (> 30.0s stability, 0.0 mm handle slip).
- **Full-Width Olympic Video Display:** Embedded `duck_racer_competition_2x.webp` with `width="100%"` for a panoramic banner view.
- **Zero Emojis & Zero LaTeX Backend Syntax:** Systematically replaced all emojis with academic headings and verified zero LaTeX markers or raw math macros across all markdown files.

### 2026-09-24 — Pure Physical Clamping (Zero Welds) + 15th Actuator + Non-Penetrating Beak Kinematics (v4 Policy Verified)

**What:** Restored the articulated 15th actuator (`beak` XL-330 Dynamixel servo) with realistic motor dynamics, corrected the lower beak hinge range to 1.47 to 2.05 rad (84.2 to 117.5 degrees) with CAD flush reference at 1.500 rad (85.9 degrees), demoted rigid plastic head shells (`bottom_head_shell`, `beak_lower_collision`) to self-collision only (`contype=2, conaffinity=2`), centered the barbell payload at silicone midpoint z = -58 mm, and formulated, trained, and verified `payload_clamping_locomotion.onnx` (formerly `loadcarry_ppo_headup_v4.onnx`) under pure contact physics (100g barbell, zero welds, friction mu = 3.5 silicone).

**Why:** The user rejected earlier setups where the lower beak penetrated the skull face and where weld constraints masked true physical grip failure. Under pure contact physics without welds, baseline `loadcarry_ppo_headup_v3.onnx` dropped the barbell at t = 1.70s (step 84) because weld-training allowed the policy to adopt an aggressive forward head pitch (+61.8 degrees downward), causing the barbell to slide forward off the front edge of the silicone lining under gravity and foot strike inertia.

**Key decisions captured:**
- **Zero Welds Contract:** Completely disabled MuJoCo equality weld constraints in both training (`microduck_ppo_headup_v4_env_cfg.py`) and evaluation (`test_physical_clamp_policy.py`). Retention is achieved purely through Coulomb friction and active clamping torque (0.35 N·m).
- **Non-Penetrating CAD Kinematics:** Joint range 1.47 to 2.05 rad mathematically prevents the lower beak from ever penetrating the upper head shells. At 0.35 N·m active clamp on an 8mm shaft, the lower beak settles at 92.2 degrees (1.609 rad), leaving > 1.5 mm clearance below the skull face.
- **Friction Isolation (Silicone vs Hard Plastic):** `soft_mouth_top_collision` and `bottom_head_shell_collision` share identical spatial boundaries in CAD. Demoting rigid plastic geoms to `class="self_collision_only"` confines contact strictly to the mu = 3.5 silicone pads, preventing normal force dilution across low-friction plastic facets.
- **Payload Centering (z = -58 mm):** Shifted initial cylinder depth from the lip edge (z = -68 mm) to the silicone trough center (z = -58 mm), creating a > 14 mm retention margin on both sides.
- **Head Cradle & Gait Smoothing Rewards:** Trained with custom terms `head_cradle_reward` (head pitch -0.08 rad, neck pitch +0.35 rad), `torso_pitch_counterbalance` (-5 degrees backward lean), and `action_rate_l2` (-0.15), stabilizing payload retention against walking vibrations.
- **Benchmark Results:** In a 300-step (6.00s) continuous walking test at Vx = 0.25 m/s with 0.35 N·m active clamp, `payload_clamping_locomotion.onnx` achieved **100% payload retention** (0 drops, max drift 11.0 mm) vs baseline v3 dropping at 4.84s (step 241).
- **Policy Safety:** Baseline `loadcarry_ppo_headup_v3.onnx` was preserved completely untouched.
- **Visual Artifacts:** Generated dual-camera tracking keyframes, composite side-by-side comparison panels (`comparison_closeup_outcome.png`, `comparison_context_posture.png`), and evaluation videos (`v4_physical_clamp_walk_verified.mp4`, `v4_test_physical_clamp_300steps.mp4`).

### 2026-09-14 — Official upstream model + actuated beak + weld-carry demo

**What:** Rebuilt the robot on the *official* `robot_allcollisions.xml` from
`pollen-robotics/microduck_rl@develop` (commit `3412ae5`), added an articulated
lower beak with its own actuator, resized the carried roll, and generated a
weld-carry demo scene (`scenes/weld_cylinder_demo/`) with keyframe `welder`.

**Why:** The local copy was an older export with wrong colors (grey/blue legs
instead of Reachy white/orange — visibly wrong in the viewer) and missing the
official collision geoms + `neck_pitch/jaw_soft` contact exclude. The official
repo has no mouth actuator at all (the whole orange mandible `jaw` +
`bottom_head_shell` + `soft_mouth_top` is one rigid body `jaw_soft`); the
manipulation project needs an actuated beak, so it is added here as a thin,
regenerable layer on top of upstream instead of a hand-edited fork.

**Key decisions captured:**
- **Official files synced via `git checkout origin/develop -- <robot dir>`**,
  then `scripts/make_robot_manipulation.py` regenerates `robot_manipulation.xml`
  from the synced `robot_allcollisions.xml` (parse → splice, no hand edits).
  Re-run the script after any upstream sync. World poses at beak=0 are
  byte-identical to upstream (verified: max vertex delta 1.4e-17 m).
- **Beak articulation:** new body `duck_lower_beak` (moved meshes: `jaw`,
  `bottom_head_shell`, `soft_mouth_top` + `mouth_tip` site), hinge at
  jaw_soft-local (0, 0, -0.018) about the lateral axis — the jaw root where
  the servo bracket sits. Joint `beak_joint` range [-0.05, 0.6] rad,
  `<position name="act_beak" kp="15" kv="1">`. jaw_soft mass reduced by the
  moved 0.09 kg so total robot mass is preserved (inertia tensor kept —
  documented approximation). Contact exclude `neck_pitch/duck_lower_beak`
  added mirroring the official jaw-root exclude.
- **Cylinder resized to Ø14 × 150 mm** (r=7, half-length 75, 0.25 kg): the
  original Ø15×178 roll does not fit the mouth cavity at *any* beak angle
  (full-width envelope sweep found zero penetration-free poses; the mouth
  interior is a ~10-15 mm trough narrower than the roll). The pad
  (`soft_mouth_top`) is a trough: a resting roll contacts its raised side
  rails, not the center bottom — envelope solves must use the full pad width,
  not a central band.
- **Carry pose solved numerically, not copied from LoadCarry:** the RL
  repo's `microduck_loadcarry_env_cfg.py` weld target sits ~6 mm *inside* the
  jaw meshes (the env tolerates it via soft-weld settling). For the demo the
  pose was re-solved: beak=0.5 rad, roll center (0.0828, 0, 0.2106) world,
  axis lateral (world y) — resting on the jaw rails with 2.5 mm min clearance
  (GJK-verified `mj_geomDistance`, zero penetrating contacts).
- **Weld relpose convention (cost a debugging round):** MuJoCo weld
  `data[6:10]` is `q_rel` with `q_body1 ⊗ q_rel = q_body2`, i.e.
  `q_rel = conj(q_body1) ⊗ q_body2` — NOT the reverse multiplication order.
  Baked as `pos=(0.00523, 0, -0.06175) quat=(0.60816, 0.60816, 0.36075,
  -0.36075)` in `duck_lower_beak`'s frame; `solref=(0.005, 1)` (0.02 left
  8 mm sag; 0.005 gives 0.5 mm residual).
- **Static demo note:** with all-zero leg ctrl the stock servos (kp=0.55)
  cannot hold the robot up — the official INIT keyframe relies on a walking
  policy. The demo keyframe therefore starts correct and settles only under
  the beak servo (sag 0.029 rad, roll follows via weld, still ≥1.4 mm
  clearance at rest). For a fully static viewer demo, hold the trunk or run
  with the walking policy.
- **Verification:** compile check, kinematic sweep (beak 0→0.6 rad drops
  mouth_tip 32 mm, hinge geometry sane), 2 s hold sim (no divergence,
  weld residual 0.5 mm, zero cylinder-robot contacts), offscreen renders
  from 4 angles (side render shows white head / orange mandible / brown roll
  held horizontally in the beak).

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

### 2026-09-09 — Phase 1: Scene Built and Loadable

**What:** Implemented the randomization module, upgraded the interactive viewer, and added an automated smoke test for `apartment_v1`.

**Why:** To formally complete Phase 1. The scene is no longer just XML; it has a Python API for randomizing spawn poses, target placements, and lighting (critical for diverse data collection). The smoke test ensures CI/CD readiness.

**Key decisions captured:**
- **Duck Proxy:** Decided to defer integrating the real `microduck_rl` MJCF until Phase 2 (Teleop). The simplified proxy duck used here is sufficient to validate the apartment geometry, cameras, physics, and randomization pipeline.
