# MicroDuck Manipulation & LoadCarry Suite

Autonomous bipedal manipulation, dynamic heavy load-carrying (up to 300g / 40.7% payload-to-bodyweight), precision racket carries, and teleoperation for the 14-DoF MicroDuck avian robot.

---

## 🏆 Project Achievements & Headline Successes

This repository contains the complete simulation and control suite enabling **MicroDuck** (a 737g bipedal duck-like robot) to autonomously grasp, lift, carry, and balance diverse objects during dynamic bipedal forward locomotion:

1. **300g True Ground Barbell Pick, Hoist & Walk (Zero Welds, Zero Teleportation)**:
   - MicroDuck crouches to the ground, encloses a 300g barbell lying flat on the floor, locks its grip, hoists it aloft, and transitions seamlessly into dynamic forward walking at 0.25 m/s.
   - **40.7% payload-to-bodyweight ratio** (equivalent to a human carrying a 67 lb dumbbell clamped in their teeth while walking).
2. **Pure Physical Clamping Benchmark (Zero Welds, Constant Actuator Torque)**:
   - Validated pure Coulomb friction clamping with zero simulation welds under constant actuator torque (0.96 N·m), holding a 300g cylindrical payload with 100% retention throughout the entire walking trial.
3. **BWF Standard Badminton Racket Dynamic Carry**:
   - Official BWF specification racket (675mm length, 110mm handle clamped in mouth), walking forward with near-zero blade tilt (mean pitch: -0.69°, std: 1.89°).
4. **Authentic Pickleball Paddle Carry**:
   - 120mm grip handle clamped inside the mouth with wide throat and paddle face completely clear of the robot cheeks, maintaining stable bilateral walking.
5. **Table Tennis Racket Steadicam Carry & Ball Balancing**:
   - 40g tournament paddle clamped at the baseline; active Steadicam pantograph stabilization maintaining horizontal orientation (±0.5° of level) and dynamic ball balancing on the blade.
6. **300g Realistic Kraft Grocery Bag & Soft-Body Tote Bag Demonstrations**:
   - Avian hook-and-hoist sequence: ground crouch, strap loop intercept, silicone clamp, aloft hoist, and dynamic gait transfer.
7. **Full-Featured Teleoperation Studio & Web Dashboard**:
   - Interactive GLFW 3D viewer integrated with a live telemetry Web Dashboard (`http://localhost:8080`) supporting real-time WASD locomotion, crouch tuning, and demonstration logging.

---

## 🚀 One-Command MuJoCo Visualizer Reference

Every simulation policy can be launched directly in native interactive MuJoCo GUI using a **single terminal command**.

> **Note on macOS**: Commands use `/Users/raghav/.venvs/microduck_rl/bin/mjpython` to ensure native GLFW window event loop integration.

### Interactive 3D Viewer Hotkeys
- **Spacebar**: Pause / Resume simulation
- **Right-Click Drag**: Zoom in / out
- **Left-Click Drag**: Rotate 3D camera
- **Camera Mode**: Camera automatically tracks the robot trunk (`trackbodyid = 1`)

---

### 1. Badminton Racket Dynamic Carry (BWF Official Specs)
MicroDuck carries a scaled official BWF standard badminton racket (110mm handle clamped inside the mouth, face clear of cheeks) with dynamic bipedal forward walking.
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_racket_drop_and_walk.py --racket badminton --no-ball --gui
```

---

### 2. Pickleball Paddle Dynamic Carry
MicroDuck carries an authentic pickleball paddle with 120mm handle clamped in the beak, throat clear of cheeks, forward walking gait.
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_racket_drop_and_walk.py --racket pickleball --no-ball --gui
```

---

### 3. Table Tennis Steadicam Carry (Horizontal Blade Stabilization)
Table tennis racket carried with active Steadicam pantograph leveling (<0.5° pitch variation from level), eliminating head bobbing and jitter.
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_racket_drop_and_walk.py --racket table_tennis --no-ball --steadicam --gui
```

---

### 4. Table Tennis Dynamic Ball Balancing Test
Dynamic ball-balancing test: table tennis racket carrying with ITTF 3-star ball balancing on the rubber face during forward walking.
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_racket_drop_and_walk.py --racket table_tennis --steadicam --gui
```

---

### 5. 300g Barbell True Ground Pick, Aloft Hoist & Walk
The master manipulation sequence: starts with 300g barbell resting freely on the ground, crouches, locks grip, lifts barbell aloft, and walks forward. Zero welds, zero teleports.
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/pick_and_carry_policy.py --mass 300 --gui
```

---

### 6. 100g Barbell True Ground Pick, Lift & Walk
Baseline ground pick-and-carry test with 100g barbell payload.
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/pick_and_carry_policy.py --mass 100 --gui
```

---

### 7. 300g Pure Physical Clamping Benchmark (Zero Welds)
Pure physical clamping evaluation with constant 0.96 N·m torque holding a 300g cylinder during forward walking. 100% hold duration.
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_physical_clamp_policy.py --mass 300 --torque 0.96 --gui
```

---

### 8. 100g Pure Physical Clamping Benchmark
Pure physical clamping evaluation with nominal 0.10 N·m torque holding 100g cylinder during forward walking.
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_physical_clamp_policy.py --mass 100 --torque 0.10 --gui
```

---

### 9. 300g Kraft Grocery Bag Autonomous Pick & Walk
Autonomous avian hook-and-hoist sequence for the 300g realistic Kraft paper grocery bag:
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/grocery_bag_300g_pick_and_walk.py
```

---

### 10. 300g Soft-Body Tote Bag Interactive Scene
Interactive 3D scene viewer showing the grounded robot with active balance and 300g soft-body tote bag resting on the floor:
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/view_totebag_scene.py
```

---

### 11. Interactive Teleoperation Studio & Web Dashboard
Precision manual teleoperation studio with simultaneous GLFW 3D viewer and live Web Dashboard at `http://localhost:8080`.
- **Locomotion**: `W` / `A` / `S` / `D` (Walk forward/back, Turn left/right), `Q` / `E` (Strafe)
- **Crouch**: `J` / `Down` (Lower Head), `K` / `Up` (Raise Head)
- **Beak Clamping**: `C` (Clamp), `O` (Open), `Space` (Toggle clamp)
- **Cradle & Carry**: `T` (Throat Cradle tilt), `U` (Upright Carry)
- **Dataset Recording**: `R` (Toggle recording to .npz)
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/teleop_barbell_studio.py
```

---

### 12. Lemon & Tableware Spoon Dynamic Balancing (Zero Drop, Authentic Citrus)
Autonomous bipedal carry of an authentic 125mm tableware spoon with active Steadicam gimbal leveling, balancing a real-world 22g citrus lemon inside the concave bowl dish across forward walking gaits (0.08 to 0.30 m/s):
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_spoon_lemon_balance.py --gui
```
- **Systematic Benchmark**: `/Users/raghav/.venvs/microduck_rl/bin/python microduck_manipulation/scripts/test_spoon_lemon_balance.py --eval`
- **Render High-Res MP4 Video**: `/Users/raghav/.venvs/microduck_rl/bin/python microduck_manipulation/scripts/test_spoon_lemon_balance.py --video`

### 13. 5-Duck Multi-Agent Stadium Racer (Genie Sim 3.0 Architecture)
Multi-agent competition inspired by **Genie Sim 3.0** (arXiv:2601.02078v4): 5 Microduck bipeds sporting official Pollen Robotics colorways (Cream, Sky, Duck Gold, Lavender, Graphite) aligned on a 5-lane Olympic stadium track, sprinting forward simultaneously while actively balancing lemons on spoons with active Steadicam pantographs:
```bash
/Users/raghav/.venvs/microduck_rl/bin/mjpython microduck_manipulation/scripts/test_duck_racer_lemon_balance.py --gui
```
- **Camera Views**: Press `C` to cycle between Pack Chase, Trackside Broadcast, Overhead Aerial Drone, and Leader Focus cameras.
- **Racer Pacing**: Press `1` .. `5` for individual racer turbo boosts, `W`/`S` for whole pack acceleration/deceleration.
- **Render High-Res Competition Video**: `/Users/raghav/.venvs/microduck_rl/bin/python microduck_manipulation/scripts/test_duck_racer_lemon_balance.py --video`

---

## 📊 Performance Summary Table

| Task / Policy | Payload Mass | Payload Ratio | Clamping Force | Weld Constraints | Walking Speed | Pitch Stability / Result |
|---|---|---|---|---|---|---|
| **Barbell Pick & Carry** | **300.0 g** | **40.7%** | 31.0 N | **0 (None)** | 0.25 m/s | Upright balanced |
| **Barbell Pick & Carry** | 100.0 g | 13.6% | 31.0 N | **0 (None)** | 0.25 m/s | Upright balanced |
| **Physical Clamp Benchmark** | **300.0 g** | **40.7%** | 31.0 N | **0 (None)** | 0.25 m/s | 100% held |
| **Physical Clamp Benchmark** | 100.0 g | 13.6% | 3.2 N | **0 (None)** | 0.25 m/s | 100% held |
| **BWF Badminton Carry** | 40.0 g | 5.4% | 30.6 N | **0 (None)** | 0.18 m/s | Mean: -0.69°, Std: 1.89° |
| **Pickleball Paddle Carry** | 75.0 g | 10.2% | 30.6 N | **0 (None)** | 0.18 m/s | Bilateral stable |
| **Table Tennis Steadicam** | 40.0 g | 5.4% | 30.6 N | **0 (None)** | 0.18 m/s | < 0.5° level |
| **Lemon & Spoon Balance** | **47.0 g** (25g spoon + 22g lemon) | **6.4%** | **34.1 N** | **0 (None)** | **0.08–0.30 m/s** | **10.0s stable balance, max dev < 1.7mm** |
| **5-Duck Stadium Racer** | **5x 47.0 g** (5 competitors) | **6.4% each** | **5x 34.1 N** | **0 (None)** | **0.15–0.22 m/s** | **5 parallel lanes, zero drops, 5.0m sprint** |
| **Grocery Bag Pick & Walk** | **300.0 g** | **40.7%** | 31.0 N | Dynamic hook | 0.15 m/s | Aloft carry |

---

## 📖 In-Depth Technical Documentation

For complete mathematical derivations, PPO formulation, actor-critic network architectures, contact mechanics, and reward formulations, see:
- 📘 [Master Technical Pipeline & Mathematical Manual](docs/PROJECT_MASTER_PIPELINE_AND_THEORY.md)
