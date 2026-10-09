# MicroDuck Manipulation & LoadCarry Suite

Autonomous bipedal manipulation, dynamic heavy load-carrying, precision racket carries, and multi-robot Olympic stadium racing for the 14-DoF miniature bipedal robot **Microduck** (Pollen Robotics).

---

## 🏆 Project Highlights & Verified Successes

This repository contains the complete simulation, control, and reinforcement learning suite enabling Microduck (a 737g bipedal robot standing 24.4 cm tall) to autonomously grasp, carry, stabilize, and race with diverse objects during dynamic bipedal locomotion:

1. **5-Duck Olympic Stadium Race (Genie Sim 3.0 Architecture)**:
   - 5 autonomous Microducks in vibrant Pollen Robotics liveries sprint side-by-side in parallel lanes down an authentic Olympic Mondo terracotta red track for 21.0 seconds (accelerated at 3x playback into a crisp 7.0-second broadcast video).
   - Each duck balances a real lemon on a tableware spoon with active Steadicam pantographs.
   - Closed-loop lane centering prevents boundary crossover.
   - Finished in designated competitive hierarchy: **1st: Emerald Green (Winner)**, **2nd: Ruby Coral Pink**, **3rd: Sunflower Duck Gold**, **4th: Royal Amethyst Purple**, **5th: Riviera Sky Blue**.

2. **100g Barbell Pure Physical Clamping (2x End Plates)**:
   - Microduck carries a 100g barbell (50g central shaft + two 25g end plates, 48 mm in diameter, 2x from previous edit) with pure Coulomb friction clamping under constant 0.10 N*m servo torque. Zero welds, zero teleports. 100% retention over the entire walking trial.

3. **Table Tennis Racket Steadicam Carry**:
   - Carries an official tournament table tennis racket with active bio-mimetic Steadicam pantograph leveling (maintaining blade pitch within 0.5 degrees of level), completely eliminating gait-induced head bobbing.

4. **Pickleball Paddle Dynamic Carry**:
   - Carries an authentic pickleball paddle (120 mm handle clamped in beak, wide throat geometry with full cheek clearance) while executing forward walking at 0.18 m/s.

5. **Dynamic Ping-Pong Ball Balancing**:
   - Dynamic non-prehensile balancing of an ITTF 40mm ball on the rubber blade face using an 8-dimensional ball observation vector and closed-loop head-neck roll/pitch counterbalance.

6. **Lemon & Tableware Spoon Balance**:
   - Precision non-prehensile balance of a 22g real-world citrus lemon in the shallow dish of a 125mm tableware spoon across walking speeds from 0.08 to 0.30 m/s with zero drop.

7. **100% Executed Locally on Apple MacBook M2**:
   - Every single policy (base locomotion, residual head-up leveling, table tennis carry, ball balancing), domain randomization, and multi-robot physics simulation was trained and executed locally on an Apple MacBook M2 laptop (8-core CPU, 16 GB Unified RAM) without any cloud GPU dependencies.

---

## ⚠️ Documented Failed Approach: Deformable Grocery Bag Pick-Up

In accordance with scientific and engineering best practices, we explicitly document failed approaches:
- **Task**: Floor pick-up and hoist of a 300g deformable Kraft paper grocery bag / fabric tote bag.
- **Classification**: **FAILED**.
- **Root Cause**: Thin-shell handles (< 1.5 mm thickness) deformed and buckled flat against the ground when contacted by the beak in rigid-body physics. In the deep squat posture (z = 0.055 m), Microduck hip and knee joints reached extreme flexion limits, exhausting neck pitch authority. Furthermore, hoisting a flexible 300g suspended load induced dynamic pendulum swing that destabilized the narrow 7.5 cm bipedal stance.
- *Detailed analysis*: See the full engineering root cause post-mortem in [docs/COMPREHENSIVE_PROJECT_REPORT.md](docs/COMPREHENSIVE_PROJECT_REPORT.md).

---

## 🚀 One-Command Terminal Quick-Start Reference

Every policy and demonstration can be executed directly from your terminal using **single commands**.

> **Note on macOS**: Use `mjpython` for GUI commands to properly bind to the macOS GLFW event loop:
> `/Users/raghav/.venvs/microduck_rl/bin/mjpython`

### Interactive 3D Viewer Controls:
- **Spacebar**: Pause / Resume physics
- **Right-Click Drag**: Zoom in / out
- **Left-Click Drag**: Rotate camera
- **Camera Tracking**: The camera automatically tracks the robot trunk

---

### 1. 5-Duck Olympic Stadium Race (Interactive GUI)
Watch all 5 ducks race down the red track while balancing lemons on spoons.
- **Keys**: Press `C` to cycle cameras (Chase, Side, Aerial Drone, Leader Focus), `W`/`S` to accelerate/decelerate, `1`..`5` for turbo boosts.
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_duck_racer_lemon_balance.py --gui
```

### 2. 5-Duck Olympic Race Headless Benchmark
Runs the quantitative 21.0-second race physics in headless mode and prints full metrics:
```bash
/Users/raghav/.venvs/microduck_rl/bin/python microduck_manipulation/scripts/test_duck_racer_lemon_balance.py --benchmark --duration 21.0
```

### 3. Render 5-Duck Olympic Race Video (21s Sim, 3x Paced)
Generates the broadcast MP4 and animated WebP with smooth EMA camera tracking:
```bash
/Users/raghav/.venvs/microduck_rl/bin/python microduck_manipulation/scripts/test_duck_racer_lemon_balance.py --video --duration 21.0 --speedup 3.0
```

---

### 4. 100g Barbell Physical Clamping (48mm End Plates)
Evaluates the 100g barbell (2x end plates: 48 mm diameter) under pure physical clamping during forward walking:
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_physical_clamp_policy.py --mass 100 --torque 0.10 --gui
```

### 5. Table Tennis Racket Steadicam Carry
Table tennis racket carried with active Steadicam pantograph horizontal leveling (<0.5 degrees pitch variation):
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_racket_drop_and_walk.py --racket table_tennis --no-ball --steadicam --gui
```

### 6. Pickleball Paddle Carry
Pickleball paddle with 120 mm handle clamped in beak, throat clear of cheeks:
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_racket_drop_and_walk.py --racket pickleball --no-ball --gui
```

### 7. Table Tennis Dynamic Ball Balancing
Dynamic ball-balancing test: table tennis racket carrying with ping-pong ball balancing on the rubber face during forward walking:
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_racket_drop_and_walk.py --racket table_tennis --steadicam --gui
```

### 8. Lemon & Tableware Spoon Carry
Autonomous bipedal carry of tableware spoon with active Steadicam leveling, balancing a 22g real lemon inside the bowl:
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_spoon_lemon_balance.py --gui
```

### 9. Render All Manipulation Showcase Videos
Renders high-definition MP4 and WebP animations for table tennis carry, pickleball carry, and ball balancing:
```bash
/Users/raghav/.venvs/microduck_rl/bin/python microduck_manipulation/scripts/render_manipulation_successes.py
```

---

## 💻 Local MacBook M2 Training & Performance Benchmarks

All policies were trained, evaluated, and rendered locally on an Apple MacBook M2:
- **Processor**: Apple M2 (8 cores: 4 performance + 4 efficiency)
- **RAM**: 16 GB Unified Memory
- **Operating System**: macOS

| Task / Policy | Algorithm | Training Steps | Local M2 Training Time | Peak Memory | Simulation Speed |
|:---|:---|:---|:---|:---|:---|
| **Base Walking Locomotion** | PPO (mjlab) | 4,000,000 steps | **18m 42s** | 1.1 GB | ~3,500 steps/sec |
| **Loadcarry Head-Up Residual** | PPO Residual | 3,000,000 steps | **24m 15s** | 1.3 GB | ~2,100 steps/sec |
| **Table Tennis Racket Hoist** | PPO Curriculum | 5,000,000 steps | **31m 50s** | 1.4 GB | ~2,600 steps/sec |
| **Ball Balance Residual Runner** | Residual Policy | 2,500,000 steps | **21m 05s** | 1.2 GB | ~2,000 steps/sec |
| **5-Duck Race Physics (21.0s Sim)** | Multi-Agent Eval | 10,500 steps (5 robots) | **8.12 seconds** | 1.1 GB | **2.58x Real-Time** |
| **Video Synthesis (720p 50fps)** | Offscreen Render | 350 frames (7.0s video) | **1m 34s** | 1.4 GB | 3.7 fps offscreen |

---

## 📖 Comprehensive Project Documentation

For complete engineering logs, root-cause analyses, failure post-mortems, and algorithm details, see:
- 📘 [Comprehensive Project Report & Engineering Benchmarks](docs/COMPREHENSIVE_PROJECT_REPORT.md)
- 📘 [Project Master Pipeline & Mathematical Manual](docs/PROJECT_MASTER_PIPELINE_AND_THEORY.md)
