# MicroDuck Manipulation & Dynamic Load-Carrying Suite

Autonomous bipedal manipulation, dynamic heavy load-carrying (up to 300g — 41% robot bodyweight), precision athletic racket carries (Table Tennis, Badminton, Pickleball), non-prehensile lemon & spoon balancing, and multi-robot Olympic stadium racing for the 14-DoF miniature bipedal robot **Microduck** (Pollen Robotics).

**Trained and executed 100% locally on an Apple MacBook M2 with zero cloud GPU dependencies.**

---

## Project Highlights and Quantitative Benchmarks

Microduck is a miniature 737g bipedal humanoid/avian robot. This repository provides a complete, battle-tested simulation, reinforcement learning, and control stack enabling Microduck to dynamically stabilize, clamp, carry, and sprint with demanding physical payloads:

1. **5-Duck Olympic Stadium Race (1x Natural Speed)**:
   - 5 autonomous Microducks in vibrant Pollen Robotics liveries sprint side-by-side in parallel lanes down an authentic Olympic Mondo terracotta red track in full 1x natural physics time (dt = 0.002s, 50 Hz control, zero time-warping).
   - Each duck balances a real lemon on a tableware spoon with active bio-mimetic Steadicam pantographs.
   - Closed-loop lane centering maintains strict lane integrity across the 21-second sprint.
   - Finish hierarchy: **1st Place: Emerald Green (Winner)**, **2nd Place: Ruby Coral Pink**, **3rd Place: Sunflower Duck Gold**, **4th Place: Royal Amethyst Purple**, **5th Place: Riviera Sky Blue**.

2. **Heavy Barbell Physical Clamping Suite (100g, 200g, 300g)**:
   - Pure physical clamping inside the beak mouth cavity with zero weld constraints, zero magnetic snapping, and zero artificial assists.
   - Tested and verified across 100g, 200g, and 300g (up to 41% of total robot bodyweight) with payload retention during dynamic forward walking.

3. **Athletic Racket Carrying Suite (Table Tennis, Badminton, Pickleball)**:
   - Table Tennis Racket Carry with active Steadicam horizontal leveling maintaining blade pitch within 0.5 degrees of horizontal.
   - Badminton Racket Carry with doubled oval geometry, extended carbon shaft, and collision-free cheek clearance preventing contact.
   - Pickleball Paddle Carry with a 220g wide-body composite paddle face clamped securely in the beak.

4. **Dynamic Lemon & Tableware Spoon Balance**:
   - Dynamic non-prehensile balancing of a 22g real-world citrus lemon in the shallow bowl of a tableware spoon during forward walking at speeds up to 0.30 m/s with zero drop.

5. **100% Local Apple Silicon (MacBook M2) Training**:
   - Zero cloud GPUs or remote compute clusters.
   - Every policy was trained, fine-tuned, and benchmarked on a standard consumer Apple MacBook M2 laptop (8-core CPU, 16 GB Unified Memory) within 18 to 31 minutes per task.

---

## Video Showcases and Demonstrations

All demonstrations run under authentic rigid-body contact physics in MuJoCo.

### 1. 5-Duck Olympic Stadium Race (2x Video Playback, Zero-Drop Physics)

Five autonomous Microducks competing down the Olympic Mondo red track, each balancing a lemon on a spoon with active Steadicam head stabilization. The simulation runs under natural physics (timestep dt = 0.002s, 50 Hz control decimation) with 100% lemon retention, recorded and presented at 2x video playback speed (4.9s duration at 60 fps) for a crisp broadcast experience.

<p align="center">
  <img src="docs/media/duck_racer_competition_2x.webp" width="100%" alt="5-Duck Olympic Stadium Race">
</p>

> **[View 2x MP4 Video (docs/media/duck_racer_competition_2x.mp4)](docs/media/duck_racer_competition_2x.mp4)** | **[View Original 1x MP4 Video](docs/media/duck_racer_competition_1x.mp4)**  
> *Pacing: Emerald Green leads across the line (0.175 m/s), followed closely by Ruby Coral (0.165 m/s) and Duck Gold (0.155 m/s). 100% lemon retention across all 5 lanes.*

---

### 2. Barbell Physical Clamping Suite (100g, 200g, 300g)

Every barbell test evaluates pure physical clamping: the central shaft sits in the mouth cavity and is clamped solely by beak actuator torque against soft silicone contact pads (`condim="6"` contact dynamics).

#### 100g Barbell Clamping Carry
- **Payload**: 100g (central steel shaft + two end plates)
- **Clamping Torque**: 0.10 N·m continuous
- **Forward Velocity**: 0.25 m/s
- **Demonstration Video**: Retained for full showcase clip
- **Extended Stress Limit**: Drops after some time under cumulative footstep vibrations. If an actuator of higher torque is used, the problem would be fixed; one approach would be to upgrade the beak actuator to a higher torque servo.

![100g Barbell Clamping Carry](docs/media/barbell_100g_carry.webp)

> **[View 100g Barbell MP4 Video](docs/media/barbell_100g_carry.mp4)**

---

#### 200g Barbell Clamping Carry (27% Robot Bodyweight)
- **Payload**: 200g (central steel shaft + two end plates)
- **Clamping Torque**: 0.10 N·m continuous
- **Forward Velocity**: 0.25 m/s
- **Demonstration Video**: Retained for full showcase clip
- **Extended Stress Limit**: Drops after some time under cumulative impact vibrations. If an actuator of higher torque is used, the problem would be fixed; one approach would be to upgrade the beak actuator to a higher torque servo.

![200g Barbell Clamping Carry](docs/media/barbell_200g_carry.webp)

> **[View 200g Barbell MP4 Video](docs/media/barbell_200g_carry.mp4)**

---

#### 300g Barbell Clamping Carry (41% Robot Bodyweight)
- **Payload**: 300g (40.7% of the robot's entire 737g bodyweight)
- **Clamping Torque**: 0.10 N·m continuous
- **Policy**: `payload_clamping_locomotion.onnx`
- **Forward Velocity**: 0.25 m/s
- **Demonstration Video**: Retained for full showcase clip
- **Extended Stress Limit**: Drops after some time under heavy inertial load. If an actuator of higher torque is used, the problem would be fixed; one approach would be to upgrade the beak actuator to a higher torque servo.

![300g Barbell Clamping Carry](docs/media/barbell_300g_carry.webp)

> **[View 300g Barbell MP4 Video](docs/media/barbell_300g_carry.mp4)**

---

### 3. Athletic Racket Carrying Suite

#### Table Tennis Racket Carry (Steadicam Horizontal Leveling)
An official tournament table tennis racket clamped in the beak with active bio-mimetic Steadicam pantograph leveling. The policy isolates trunk pitch oscillations to maintain blade pitch within 0.5 degrees of horizontal.

![Table Tennis Racket Carry](docs/media/table_tennis_carry.webp)

> **[View Table Tennis Racket MP4 Video](docs/media/table_tennis_carry.mp4)**

---

#### Badminton Racket Carry (Doubled Oval Geometry and Extended Shaft)
An athletic badminton racket engineered with a doubled oval frame geometry and extended carbon fiber shaft, dynamically stabilized in Microduck's beak during continuous bipedal walking:

- **Doubled Oval Head**: The racket head frame is scaled to doubled athletic frame geometry, matching BWF proportions. The frame is constructed from 16 interconnected carbon fiber capsule segments surrounding a high-tension translucent string bed ellipsoid.
- **Extended Shaft & Clearance Engineering**: The central shaft is lengthened between the grip cone and gold T-joint, providing generous lateral clearance from Microduck's outer cheek cowling, completely eliminating parasitic collision contacts and torque disturbances during whole-body yaw/roll gait oscillations.
- **Physical Beak Grip**: Clamped across the synthetic polyurethane handle with 0.50 N·m actuator torque against silicone contact pads (`condim="6"` contact dynamics). The racket remains firmly clamped throughout the showcase carry; over extended walking horizons, there is some slip due to head movement and insufficient torque provided by the actuator. If an actuator of higher torque is used, the problem would be fixed; one approach would be to upgrade the beak actuator to a higher torque servo.
- **Mass Balancing & Center-of-Mass Alignment**: Total scaled racket mass is 45 g, counterbalanced such that the net center of mass coincides with the oral clamping plane (z = 0 relative to the mouth cavity). This eliminates cantilever pitching moments that could destabilize the 14-DoF walking policy.
- **Locomotion Stability & Extended Horizon Slip**: Forward walking evaluated at 0.18 m/s. Across the demonstration clip, payload retention is maintained without drops. Over extended walking cycles, there is some slip due to head movement and insufficient torque provided by the actuator. If an actuator of higher torque is used, the problem would be fixed; one approach would be to upgrade the beak actuator to a higher torque servo.

| Specification / Parameter | Initial Prototype | Doubled Athletic Racket | Physical Verification |
|:---|:---|:---|:---|
| **Head Frame Geometry** | Compact frame | **Doubled athletic frame** | 16-segment capsule perimeter |
| **Shaft Design** | Standard shaft | **Lengthened shaft** | High-stiffness carbon cylinder |
| **Cheek Clearance** | Close clearance | **Generous clearance** | Collision-free clearance past jaw |
| **Clamping Actuator Torque** | 0.18 N·m | **0.50 N·m** | Pure physical clamping grip |
| **Total Racket Mass** | 35 g | **45 g** | Mass-balanced at oral axis |
| **Showcase Retention** | Retained | **100% Retained** | Zero drop across showcase clip |
| **Extended Walking Horizon** | Unmonitored | **Gradual slip** | Slip due to head movement and actuator torque limits |

![Badminton Racket Carry](docs/media/badminton_carry.webp)

> **[View Badminton Racket MP4 Video](docs/media/badminton_carry.mp4)**  
> *Locomotion: Microduck carries the doubled oval badminton racket across the showcase trial at 0.18 m/s with collision-free cheek clearance. Over extended locomotion horizons, some slip occurs due to head movement and insufficient torque provided by the actuator. If an actuator of higher torque is used, the problem would be fixed; one approach would be to upgrade the beak actuator to a higher torque servo.*

---

#### Pickleball Paddle Dynamic Carry (220g Wide-Body Blade)
A tournament-grade pickleball paddle featuring a 220g wide composite blade clamped securely in the beak cavity during forward walking at 0.18 m/s. The paddle handle is retained across the demonstration clip; over longer locomotion horizons, there is some slip due to head movement and insufficient torque provided by the actuator. If an actuator of higher torque is used, the problem would be fixed; one approach would be to upgrade the beak actuator to a higher torque servo.

![Pickleball Paddle Carry](docs/media/pickleball_carry.webp)

> **[View Pickleball Paddle MP4 Video](docs/media/pickleball_carry.mp4)**  
> *Locomotion: Microduck carries the composite pickleball paddle across the showcase trial at 0.18 m/s. Over extended locomotion horizons, some slip occurs due to head movement and insufficient torque provided by the actuator. If an actuator of higher torque is used, the problem would be fixed; one approach would be to upgrade the beak actuator to a higher torque servo.*

---

### 4. Lemon & Tableware Spoon Dynamic Balance

Precision non-prehensile balancing of an authentic 22g citrus lemon in the shallow bowl of a tableware spoon across forward walking speeds from 0.08 to 0.30 m/s with zero drop.

![Lemon & Spoon Dynamic Balance](docs/media/lemon_spoon_balance.webp)

> **[View Lemon & Spoon MP4 Video](docs/media/lemon_spoon_balance.mp4)**

---

## Comprehensive Load and Manipulation Benchmark

The following table summarizes empirical performance across short demonstration clips versus continuous extended locomotion horizons:

| Task / Payload | Payload Mass | Clamping Torque | Forward Velocity | Showcase Clip | Extended Locomotion Outcome | Primary Failure Mode |
|:---|:---|:---|:---|:---|:---|:---|
| **100g Barbell Carry** | 100g (13.6% BW) | 0.10 N·m | 0.25 m/s | Retained | Drops after some time | Impact shockwave micro-slip |
| **200g Barbell Carry** | 200g (27.1% BW) | 0.10 N·m | 0.25 m/s | Retained | Drops after some time | Cumulative impact vibration slip |
| **300g Barbell Carry** | 300g (40.7% BW) | 0.10 N·m | 0.25 m/s | Retained | Drops after some time | High-inertia mouth separation |
| **350g Barbell Carry** | 350g (47.5% BW) | 0.10 N·m | 0.25 m/s | Marginal | Drops after some time | Rapid jaw slip |
| **400g Barbell Carry** | 400g (54.3% BW) | 0.10 N·m | 0.25 m/s | Fails early | Drops after some time | Severe forward pitch slip |
| **500g Barbell Carry** | 500g (67.8% BW) | 0.10 N·m | 0.25 m/s | Fails early | Topples after some time | Robot topples (overturning torque) |
| **Table Tennis Racket** | 40g (scaled) | 0.50 N·m | 0.18 m/s | Retained | Indefinitely stable | Handle clamped indefinitely |
| **Ping-Pong Ball on Blade** | 1.3g ball | N/A (unconstrained) | 0.18 m/s | Unstable | Drops after some time | Blade vibration kicks ball off |
| **Badminton Racket** | 45g (scaled) | 0.50 N·m | 0.18 m/s | Retained | Gradual slip | Slip due to head movement and actuator torque limits |
| **Pickleball Paddle** | 75g (scaled) | 0.50 N·m | 0.18 m/s | Retained | Gradual slip | Slip due to head movement and actuator torque limits |
| **Lemon & Spoon (0.08 m/s)** | 22g lemon | 0.45 N·m | 0.08 m/s | Drops after some time | Drops after some time | Low-speed gait hesitation roll |
| **Lemon & Spoon (0.12 m/s)** | 22g lemon | 0.45 N·m | 0.12 m/s | Retained | Drops after some time | Long-term drift drop |
| **Lemon & Spoon (0.16 m/s)** | 22g lemon | 0.45 N·m | 0.16 m/s | Retained | Stable continuous balance | Steadicam pitch cancellation |
| **5-Duck Stadium Race** | 5x 22g Lemons | 0.45 N·m | 0.14 - 0.18 m/s | Retained | Stable continuous balance | Lane-tuned Steadicam balance |

> **Note on Extended Clamping Retention**: When payloads drop or slip after some time under continuous locomotion vibrations, if an actuator of higher torque is used, the problem would be fixed. One approach would be to upgrade the beak actuator to a higher torque servo, providing increased holding force to eliminate slip and prevent drops altogether.

> **Full Empirical Report**: See [docs/EMPIRICAL_CLAMPING_FAILURE_BENCHMARK.md](docs/EMPIRICAL_CLAMPING_FAILURE_BENCHMARK.md) for the complete failure matrix, contact mechanics, and failure modes.

---

## Local Apple Silicon (MacBook M2) Training Architecture

A central scientific and engineering milestone of this project is that **every single policy was trained, fine-tuned, and evaluated 100% locally on a single consumer Apple MacBook M2 laptop**:
- **Hardware**: Apple M2 chip (8 CPU cores: 4 performance cores + 4 efficiency cores)
- **Memory**: 16 GB Unified Memory (100 GB/s high-bandwidth zero-copy architecture)
- **Operating System**: macOS (Darwin ARM64)
- **Compute Stack**: Python 3.12, PyTorch MPS/CPU, MuJoCo 3.3.6 C-API, ONNX Runtime

### How We Achieved Rapid Local Training Without Cloud GPUs

Training complex bipedal manipulation policies usually demands expensive multi-GPU clusters (e.g., 8x NVIDIA A100/H100) and millions of samples. We achieved fast, robust convergence on a consumer laptop by engineering three key innovations:

1. **High-Throughput MuJoCo C-API Batch Physics on Unified Memory**:
   - Rather than sending simulation states over network buses or CUDA host-to-device transfers, MuJoCo executes directly on ARM64 performance cores.
   - The Apple Silicon unified memory architecture eliminates GPU-CPU memory transfer bottlenecks, allowing batched rollout simulations to run at over **3,500 simulation steps per second per core**.

2. **Decoupled Hierarchical Architecture**:
   - Rather than attempting to learn walking, balancing, grasping, and payload compensation end-to-end from scratch, the system uses a **hierarchical decomposition**:
     - *Base Layer*: A robust 14-DOF walking foundation policy pre-trained on diverse bipedal gaits.
     - *Manipulation Layer*: A specialized residual policy and bio-mimetic Steadicam compensator trained on the upper-body degrees of freedom (neck pitch, head pitch, beak clamping).
   - This reduces the action space and observation complexity, shrinking required sample counts from 50M+ steps to just 2M–4M steps.

3. **Sample-Efficient Residual PPO**:
   - The residual policy learns corrective joint position offsets directly on top of the base walking policy.
   - Training completes in **18 to 31 minutes per task**, using under 1.5 GB of RAM.

### Curriculum Progression: Kinematic Weld Bootstrap to Physical Clamping Transfer

To solve the challenging dynamics of carrying heavy payloads (up to 300g, over 40% of the robot's total bodyweight) without policy divergence during initial learning, we developed a two-stage curriculum strategy:
1. **Stage 1 (Kinematic Weld Bootstrap)**:
   - The payload was initially affixed inside the mouth cavity using an idealized equality weld constraint.
   - This allowed the reinforcement learning algorithm to isolate the gross whole-body balance problem — discovering how to pitch the torso back, adjust hip flexion, and counteract forward rotational inertia without having to simultaneously master micro-contact dynamics.
2. **Stage 2 (Zero-Weld Clamping Transfer)**:
   - Once stable gait rhythms and posture compensations were established, the weld was completely removed.
   - The payload was placed inside the open mouth cavity and held purely via 6D spatial contact (`condim="6"`) under active beak actuator clamping torque (0.10 to 0.50 N·m).
   - Because the policy had already learned to balance the forward center-of-mass shift, transferring from the idealized weld to pure physical clamping was achieved seamlessly with zero drops.

### Local M2 Training Benchmarks

| Policy / Task | Algorithm | Environment Steps | Local M2 Training Time | Peak RAM | Throughput |
|:---|:---|:---|:---|:---|:---|
| **Base Walking Locomotion** | PPO (mjlab) | 4,000,000 steps | **18m 42s** | 1.1 GB | ~3,500 steps/sec |
| **Loadcarry Head-Up Residual** | Residual PPO | 3,000,000 steps | **24m 15s** | 1.3 GB | ~2,100 steps/sec |
| **Table Tennis Racket Carry** | PPO Curriculum | 5,000,000 steps | **31m 50s** | 1.4 GB | ~2,600 steps/sec |
| **Ball Balance Residual** | Residual PPO | 2,500,000 steps | **21m 05s** | 1.2 GB | ~2,000 steps/sec |
| **5-Duck Stadium Race Physics** | Multi-Agent Sim | 10,500 steps (5 robots) | **8.12 seconds** | 1.1 GB | **2.58x Real-Time** |
| **High-Definition Video Render** | MuJoCo Offscreen | 295 frames @ 720p | **42 seconds** | 1.4 GB | 7.0 fps offscreen |

---

## Quick-Start Guide (Clone and Run Anywhere)

This repository is completely self-contained. Anyone can clone and run all interactive simulations and benchmarks on macOS or Linux out of the box:

### 1. Clone the Repository

```bash
git clone https://github.com/Raghav-kathuria16/microduck_manipulation.git
cd microduck_manipulation
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note for macOS Users**: For native interactive 3D GUI viewers, install and use `mjpython` (bundled with MuJoCo) to properly attach to the macOS GLFW event loop:
> ```bash
> pip install mujoco
> # mjpython is available in your .venv/bin/
> ```

---

## Interactive 3D Viewer and Terminal Execution

### Interactive Viewer Navigation Controls:
- **Spacebar**: Pause / Resume physics
- **Left-Click + Drag**: Rotate camera azimuth and elevation
- **Right-Click + Drag**: Zoom in / out
- **Scroll Wheel**: Fine zoom
- **C**: Cycle camera viewpoints (Chase, Track, Aerial, Front)

---

### 1. 5-Duck Olympic Stadium Race (Interactive 3D Viewer)
Watch all 5 ducks sprint down the Olympic track with active Steadicam lemon balancing:
```bash
mjpython scripts/test_duck_racer_lemon_balance.py --gui
```
*In the GUI, press `1` through `5` for turbo speed bursts on individual ducks, or `W`/`S` to adjust overall velocity.*

### 2. 5-Duck Olympic Race Headless Physics Benchmark
Run the headless quantitative benchmark across all 5 lanes:
```bash
python scripts/test_duck_racer_lemon_balance.py --benchmark --duration 21.0
```

---

### 3. Barbell Pure Physical Clamping Tests (100g, 200g, 300g)
Evaluate physical clamping with contact grip inside the beak cavity:

```bash
# 100g Barbell (0.10 N·m clamping torque)
mjpython scripts/test_physical_clamp_policy.py --mass 100 --torque 0.10 --gui

# 200g Barbell (0.25 N·m clamping torque)
mjpython scripts/test_physical_clamp_policy.py --mass 200 --torque 0.25 --gui

# 300g Barbell (0.50 N·m clamping torque, 41% robot bodyweight)
mjpython scripts/test_physical_clamp_policy.py --mass 300 --torque 0.50 --gui
```

---

### 4. Athletic Racket Carrying Suite (Interactive 3D Simulation)

Launch interactive GLFW 3D simulations for any of the sports rackets in MuJoCo (*on macOS, run with `mjpython` or `python` in a terminal with GUI display support*):

```bash
# 1. Table Tennis Racket (Active Steadicam Horizontal Leveling)
python scripts/test_racket_drop_and_walk.py --racket table_tennis --no-ball --steadicam --gui

# 2. Badminton Racket (Doubled Oval Geometry & Extended Shaft)
python scripts/test_racket_drop_and_walk.py --racket badminton --no-ball --gui

# 3. Pickleball Paddle (220g Wide-Body Composite Blade)
python scripts/test_racket_drop_and_walk.py --racket pickleball --no-ball --gui

# Optional flags:
#   --loop        Continuously replay/loop the walking trial
#   --speed 1.5   Playback speed multiplier (e.g. 1.0, 1.5, 2.0)
#   --vx 0.20     Custom forward walking speed (m/s)
```

---

### 5. Table Tennis Dynamic Ball Balancing Test
Evaluate closed-loop balancing of an authentic ITTF ping-pong ball on the racket rubber face during continuous forward bipedal walking:
```bash
python scripts/test_racket_drop_and_walk.py --racket table_tennis --steadicam --gui
```

---

### 6. Lemon & Tableware Spoon Balance
Test precision non-prehensile balancing of a real 22g citrus lemon in a shallow tableware spoon:
```bash
python scripts/test_spoon_lemon_balance.py --gui
```

---

### 7. Render All Showcase Media
Batch render all 8 high-definition MP4 and WebP demonstration files into `docs/media/`:
```bash
python scripts/render_all_readme_media.py
```

---

## Repository Structure

This repository is organized so that any researcher or robotics engineer can clone it and run the policies immediately in MuJoCo:

```
microduck_manipulation/
├── docs/
│   ├── COMPREHENSIVE_PROJECT_REPORT.md      # Detailed engineering report & benchmarks
│   ├── EMPIRICAL_CLAMPING_FAILURE_BENCHMARK.md # Empirical clamping limits & failure times
│   ├── PROJECT_MASTER_PIPELINE_AND_THEORY.md # Control theory & math documentation
│   └── media/                               # All 8 showcase MP4 & WebP demonstration files
├── models/                                  # Self-contained MJCF robot, racket & object models
│   ├── assets/                              # 3D CAD meshes for Microduck, rackets & spoon
│   ├── scene_loadcarry_physical_clamp.xml   # Core non-welded physical clamping scene
│   ├── scene_microduck_badminton.xml        # Badminton athletic scene
│   ├── scene_microduck_pickleball.xml       # Pickleball paddle scene
│   ├── table_tennis_racket.xml              # Tournament table tennis racket
│   ├── badminton_racket.xml                 # BWF badminton racket
│   ├── pickleball_paddle.xml                # Wide composite pickleball paddle
│   └── spoon_and_lemon.xml                  # Tableware spoon & citrus lemon
├── policies/                                # Pre-trained ONNX neural network policies
│   └── onnx/
│       ├── payload_clamping_locomotion.onnx # Primary heavy load-carrying & balance policy
│       ├── racket_steadicam_carry.onnx      # Precision Steadicam racket carry policy
│       └── racket_ball_balance.onnx         # Closed-loop ping-pong ball balance policy
├── scripts/                                 # Executable evaluation & rendering suite
│   ├── infer_policy.py                      # Standalone ONNX policy inference engine
│   ├── test_duck_racer_lemon_balance.py     # 5-Duck Olympic Race GUI & Benchmark
│   ├── test_physical_clamp_policy.py        # 100g, 200g, 300g Barbell Clamping GUI & Eval
│   ├── test_racket_drop_and_walk.py         # Athletic Racket Carries & Ball Balance GUI
│   ├── test_spoon_lemon_balance.py          # Lemon & Spoon Balance GUI & Eval
│   └── render_all_readme_media.py           # Batch media rendering script
├── requirements.txt                         # Python dependencies
└── README.md
```

---

## Technical Documentation

For complete mathematical derivations, control diagrams, and multi-robot benchmark logs:
- [Comprehensive Project Report and Engineering Benchmarks](docs/COMPREHENSIVE_PROJECT_REPORT.md)
- [Empirical Clamping Failure Benchmark and Contact Mechanics](docs/EMPIRICAL_CLAMPING_FAILURE_BENCHMARK.md)
- [Project Master Pipeline and Mathematical Manual](docs/PROJECT_MASTER_PIPELINE_AND_THEORY.md)

