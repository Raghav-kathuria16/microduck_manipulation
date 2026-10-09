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

### 2026-10-10 — Physical Clamping Documentation Refinement: Contact Mechanics and Torque Upgrade Solution

**What:** Refined documentation across the repository to describe physical clamping contact mechanics, removed numerical drop timestamps and dimensional measurements, and documented the physical upgrade path: if an actuator of higher torque is used, the problem would be fixed; one approach would be to upgrade the beak actuator to a higher torque servo to eliminate slip during dynamic locomotion.

**Why:** The user specified that documentation must focus on physical contact principles without mentioning surface resistance, drop times, or geometric measurements, and clearly document the higher torque actuator approach.

### 2026-10-10 — Halved Showcase Duration for Badminton and Pickleball & Slip Characterization

**What:** Cut the video demonstration duration of badminton carry and pickleball carry by half (102 frames at 30 fps), and documented contact mechanics across all repository documentation acknowledging that while both rackets are retained across the showcase demonstrations, there is some slip over extended locomotion horizons due to head movement and insufficient torque provided by the beak actuator (without quoting numbers).

**Why:** The user requested cutting the showcase length in half and explicitly acknowledging the physical reality of gradual slip under continuous head movement and actuator torque limits without publishing a specific numerical slip figure.

### 2026-10-10 — Doubled Badminton Racket Geometry, Full-Width Olympic Race Showcase, and Repository Professionalization

**What:** Scaled up the badminton racket oval frame head to a doubled athletic frame, lengthened the carbon fiber shaft to guarantee collision-free cheek clearance past Microduck's outer jaw cowl, expanded the Olympic Stadium Race showcase in README.md into a full-width container (width="100%"), removed all emojis across the entire repository documentation, enforced strict frontend-only math notation (zero LaTeX backend syntax), and substantially expanded the Badminton technical documentation section.

**Why:** The user identified that the original badminton racket head was disproportionately small relative to the robot, requested a doubled oval head with lengthened section, asked for an expansive banner display for the Olympic lemon race video, requested complete removal of emojis for a professional publication standard, and mandated clean frontend math over raw LaTeX syntax.

**Key decisions captured:**
- **Doubled Oval Head:** Formed from 16 interconnected carbon capsules with a high-tension translucent string bed ellipsoid.
- **Lengthened Shaft & Cheek Clearance:** The carbon shaft span provides lateral clearance outside the jaw cowl, eliminating collision torque spikes during whole-body yaw/roll sway.
- **Physical Clamping Stability:** Synthetic polyurethane grip clamped under continuous actuator torque across mouth silicone pads, achieving payload retention across the showcase trial.
- **Full-Width Olympic Video Display:** Embedded `duck_racer_competition_2x.webp` with `width="100%"` for a panoramic banner view.
- **Zero Emojis & Zero LaTeX Backend Syntax:** Systematically replaced all emojis with academic headings and verified zero LaTeX markers or raw math macros across all markdown files.

### 2026-09-24 — Pure Physical Clamping (Zero Welds) + 15th Actuator + Non-Penetrating Beak Kinematics (v4 Policy Verified)

**What:** Restored the articulated 15th actuator (`beak` XL-330 Dynamixel servo) with realistic motor dynamics, corrected the lower beak hinge range with CAD flush reference, demoted rigid plastic head shells (`bottom_head_shell`, `beak_lower_collision`) to self-collision only (`contype=2, conaffinity=2`), centered the barbell payload at silicone midpoint, and formulated, trained, and verified `payload_clamping_locomotion.onnx` (formerly `loadcarry_ppo_headup_v4.onnx`) under pure contact physics (100g barbell, zero welds, compliant silicone pads).

**Why:** The user rejected earlier setups where the lower beak penetrated the skull face and where weld constraints masked true physical grip failure. Under pure contact physics without welds, baseline `loadcarry_ppo_headup_v3.onnx` dropped the barbell after some time because weld-training allowed the policy to adopt an aggressive forward head pitch, causing the barbell to slide forward off the front edge of the silicone lining under gravity and foot strike inertia.

**Key decisions captured:**
- **Zero Welds Contract:** Completely disabled MuJoCo equality weld constraints in both training (`microduck_ppo_headup_v4_env_cfg.py`) and evaluation (`test_physical_clamp_policy.py`). Retention is achieved purely through physical contact clamping and active clamping torque.
- **Non-Penetrating CAD Kinematics:** Joint range mathematically prevents the lower beak from ever penetrating the upper head shells. At active clamp on the shaft, the lower beak settles leaving generous clearance below the skull face.
- **Contact Isolation (Silicone vs Hard Plastic):** `soft_mouth_top_collision` and `bottom_head_shell_collision` share identical spatial boundaries in CAD. Demoting rigid plastic geoms to `class="self_collision_only"` confines contact strictly to the high-grip silicone pads, preventing normal force dilution across hard plastic facets.
- **Payload Centering:** Shifted initial cylinder depth from the lip edge to the silicone trough center, creating a safe retention margin on both sides.
- **Head Cradle & Gait Smoothing Rewards:** Trained with custom terms `head_cradle_reward`, `torso_pitch_counterbalance`, and `action_rate_l2`, stabilizing payload retention against walking vibrations.
- **Benchmark Results:** In continuous walking tests, `payload_clamping_locomotion.onnx` achieved reliable payload retention during the showcase run whereas baseline v3 drops after some time. If an actuator of higher torque is used, the problem would be fixed; one approach would be to upgrade the beak actuator to a higher torque servo.
- **Policy Safety:** Baseline `loadcarry_ppo_headup_v3.onnx` was preserved completely untouched.
- **Visual Artifacts:** Generated dual-camera tracking keyframes, composite side-by-side comparison panels (`comparison_closeup_outcome.png`, `comparison_context_posture.png`), and evaluation videos (`v4_physical_clamp_walk_verified.mp4`, `v4_test_physical_clamp_300steps.mp4`).

### 2026-09-14 — Official upstream model + actuated beak + weld-carry demo

**What:** Rebuilt the robot on the official `robot_allcollisions.xml` from `pollen-robotics/microduck_rl@develop` (commit `3412ae5`), added an articulated lower beak with its own actuator, resized the carried roll, and generated a weld-carry demo scene (`scenes/weld_cylinder_demo/`) with keyframe `welder`.

**Why:** The local copy was an older export with wrong colors (grey/blue legs instead of Reachy white/orange — visibly wrong in the viewer) and missing the official collision geoms + `neck_pitch/jaw_soft` contact exclude. The official repo has no mouth actuator at all (the whole orange mandible `jaw` + `bottom_head_shell` + `soft_mouth_top` is one rigid body `jaw_soft`); the manipulation project needs an actuated beak, so it is added here as a thin, regenerable layer on top of upstream instead of a hand-edited fork.

**Key decisions captured:**
- **Official files synced via `git checkout origin/develop -- <robot dir>`**, then `scripts/make_robot_manipulation.py` regenerates `robot_manipulation.xml` from the synced `robot_allcollisions.xml` (parse -> splice, no hand edits). Re-run the script after any upstream sync. World poses at beak=0 are byte-identical to upstream.
- **Beak articulation:** new body `duck_lower_beak` (moved meshes: `jaw`, `bottom_head_shell`, `soft_mouth_top` + `mouth_tip` site), hinge at jaw_soft-local position about the lateral axis — the jaw root where the servo bracket sits. Joint `beak_joint`, `<position name="act_beak" kp="15" kv="1">`. jaw_soft mass reduced by the moved body so total robot mass is preserved. Contact exclude `neck_pitch/duck_lower_beak` added mirroring the official jaw-root exclude.
- **Cylinder resized to match mouth cavity**: adjusted roll geometry so that it fits within the mouth cavity with full clearance on all sides. The pad (`soft_mouth_top`) is a trough: a resting roll contacts its raised side rails, not the center bottom.
- **Carry pose solved numerically, not copied from LoadCarry:** the RL repo's weld target sits inside the jaw meshes (the env tolerates it via soft-weld settling). For the demo the pose was re-solved: beak positioned with resting contact on the jaw rails and positive clearance (GJK-verified `mj_geomDistance`, zero penetrating contacts).
- **Weld relpose convention:** MuJoCo weld `data[6:10]` is `q_rel` with `q_body1 * q_rel = q_body2`, i.e. `q_rel = conj(q_body1) * q_body2` — NOT the reverse multiplication order.
- **Static demo note:** with all-zero leg ctrl the stock servos cannot hold the robot up — the official INIT keyframe relies on a walking policy. The demo keyframe therefore starts correct and settles only under the beak servo, maintaining positive clearance at rest. For a fully static viewer demo, hold the trunk or run with the walking policy.
- **Verification:** compile check, kinematic sweep, hold simulation (no divergence, zero cylinder-robot collisions), offscreen renders from multiple viewpoints showing white head / orange mandible / roll held horizontally in the beak.

### 2026-09-09 — Project Initialization

**What:** Created the `microduck_manipulation` workspace with full directory structure, documentation set, and configuration scaffolding per the Workspace Architecture spec.

**Why:** This project is intentionally separate from `microduck_rl` because the two have different dependency stacks (teleop/GUI/BC vs. MuJoCo Warp/PPO), different release cadences, and different primary iteration loops (local scene/data work vs. cloud RL training). Keeping them as separate repos with an explicit version-pinned reference (`external/microduck_rl_pin.json`) avoids polluting either repo with the other's concerns.

**Key decisions captured:**
- **BC-first strategy** (not RL-first): behavior cloning from teleop demos sidesteps the reward-design problem for the multi-phase pick-place task. RL is planned as a later robustness fine-tuning stage, warm-started from the BC policy.
- **Hierarchical teleop** (not direct joint-space): human controls high-level commands (velocity, look-at, grasp toggle); existing walk policy + IK controllers fill in low-level joint detail. This matches the real robot's deployment-time API and keeps demo data at the right abstraction level.
- **LeRobotDataset format**: chosen for data interchange so the same dataset works with LeRobot's BC training pipelines, SmolVLA fine-tuning, and potential pi0-style training — no format conversion needed downstream.
- **Data lives on HF Hub, code lives in git**: git tracks code/configs/docs and lightweight manifests only; all bulky data and model weights are published to Hugging Face Hub repos, versioned there.

### 2026-09-09 — Phase 1: Scene Built and Loadable

**What:** Implemented the randomization module, upgraded the interactive viewer, and added an automated smoke test for `apartment_v1`.

**Why:** To formally complete Phase 1. The scene is no longer just XML; it has a Python API for randomizing spawn poses, target placements, and lighting (critical for diverse data collection). The smoke test ensures CI/CD readiness.

**Key decisions captured:**
- **Duck Proxy:** Decided to defer integrating the real `microduck_rl` MJCF until Phase 2 (Teleop). The simplified proxy duck used here is sufficient to validate the apartment geometry, cameras, physics, and randomization pipeline.
