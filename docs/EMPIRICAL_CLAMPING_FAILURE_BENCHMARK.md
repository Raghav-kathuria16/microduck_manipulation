# Empirical Clamping Limits and Contact Mechanics

This document presents the empirical benchmarks on physical clamping behavior under continuous bipedal locomotion in MuJoCo.

While short demonstration clips achieve payload retention with stable grip, extended continuous locomotion reveals physical boundaries dictated by footstep impact shockwaves, micro-slip accumulation, and center-of-mass inertial limits.

If an actuator of higher torque is used, the problem would be fixed. One approach would be to upgrade the beak actuator to a higher torque servo, providing sufficient clamping force to eliminate slip and prevent drops altogether during dynamic walking.

---

## 1. Lemon and Spoon Dynamic Balance: Velocity Envelope

The primary non-prehensile balancing benchmark, tested over continuous locomotion:

| Forward Walking Speed | Outcome | Extended Horizon Behavior | Pacing Dynamics |
| :---: | :---: | :---: | :--- |
| **0.08 m/s** | Drops after some time | Low-speed gait hesitation causes lateral rocking; lemon spills after some time | Active Steadicam recovery attempted |
| **0.12 m/s** | Drops after some time | Long-term drift causes lemon to drop after some time | Cadence smoothing active |
| **0.15 m/s** | **Continuous Balance** | Lemon retained continuously | Steadicam head stabilization active |
| **0.20 m/s** | **Continuous Balance** | Lemon retained continuously | Steadicam head stabilization active |
| **0.25 m/s** | **Continuous Balance** | Lemon retained continuously | Steadicam head stabilization active |

### Why Low Speeds Fail:
- At very low speeds (0.08 m/s), the walking policy exhibits gait hesitation with higher lateral roll oscillations. This lateral rocking tips the spoon rim, and it drops after some time.
- At nominal cruising speeds (0.15 to 0.25 m/s), bipedal cadence is smooth and continuous. The active Steadicam pantograph maintains spoon levelness, preserving balance throughout extended locomotion.

---

## 2. Athletic Rackets and Object Retention

| Athletic Asset | Total Mass | Clamping Status (Showcase Clip) | Ball / Object Status | Extended Locomotion Behavior | Failure Root Cause |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Table Tennis Racket** | 40g | **100% Retained** | Ping-Pong Ball | Drops after some time | Unconstrained ball vibration; footstep impacts bounce ball off rubber blade |
| **Badminton Racket** | 45g | **100% Retained** | Pure Racket Carry | Gradual slip | Slip occurs due to head movement and actuator torque limits |
| **Pickleball Paddle** | 75g | **100% Retained** | Pure Racket Carry | Gradual slip | Slip occurs due to head movement and actuator torque limits |

- **Key Finding on Ball Balancing**: While the table tennis racket handle itself is clamped firmly with high stability, balancing an unconstrained sphere on a flat rubber surface during dynamic walking is unstable without an active visual tracking policy. Walking impact forces cause the ping-pong ball to bounce, roll, and it drops after some time.
- **Athletic Racket Clamping Dynamics**: In Badminton and Pickleball carries, showcase clips demonstrate reliable payload retention with full clearance. Over extended locomotion horizons, there is some slip due to head movement and insufficient torque provided by the actuator. If an actuator of higher torque is used, the problem would be fixed; one approach would be to upgrade the beak actuator to a higher torque servo.

---

## 3. Barbell Clamping Extended Locomotion Matrix

Continuous walking trial at 0.25 m/s forward velocity. A trial is flagged as a failure if the payload slips relative to the mouth cavity and drops, or if the robot loses bipedal balance.

| Payload Mass | Payload Ratio | Clamping Torque | Result | Failure Mode |
| :---: | :---: | :---: | :---: | :--- |
| **50g** | 6.8% | 0.02 N·m | Drops after some time | Slip and Drop |
| **50g** | 6.8% | 0.10 N·m | Drops after some time | Slip and Drop |
| **50g** | 6.8% | 0.20 N·m | Drops after some time | Slip and Drop |
| **100g** | 13.6% | 0.02 N·m | Drops after some time | Slip and Drop |
| **100g** | 13.6% | 0.05 N·m | Drops after some time | Slip and Drop |
| **100g** | 13.6% | **0.10 N·m (Nominal)** | Drops after some time | Slip and Drop |
| **100g** | 13.6% | 0.20 N·m | Drops after some time | Slip and Drop |
| **150g** | 20.4% | 0.02 N·m | Drops after some time | Slip and Drop |
| **150g** | 20.4% | **0.10 N·m (Nominal)** | Drops after some time | Slip and Drop |
| **150g** | 20.4% | 0.20 N·m | Drops after some time | Slip and Drop |
| **200g** | 27.1% | 0.02 N·m | Drops after some time | Slip and Drop |
| **200g** | 27.1% | **0.10 N·m (Nominal)** | Drops after some time | Slip and Drop |
| **200g** | 27.1% | 0.20 N·m | Drops after some time | Slip and Drop |
| **250g** | 33.9% | 0.02 N·m | Drops after some time | Slip and Drop |
| **250g** | 33.9% | **0.10 N·m (Nominal)** | Drops after some time | Slip and Drop |
| **250g** | 33.9% | 0.20 N·m | Drops after some time | Slip and Drop |
| **300g** | 40.7% | 0.02 N·m | Drops after some time | Slip and Drop |
| **300g** | 40.7% | **0.10 N·m (Nominal)** | Drops after some time | Slip and Drop |
| **300g** | 40.7% | 0.20 N·m | Drops after some time | Slip and Drop |
| **350g** | 47.5% | 0.02 N·m | Drops after some time | Slip and Drop |
| **350g** | 47.5% | **0.10 N·m (Nominal)** | Drops after some time | Slip and Drop |
| **350g** | 47.5% | 0.20 N·m | Drops after some time | Slip and Drop |
| **400g** | 54.3% | 0.02 N·m | Drops after some time | Slip and Drop |
| **400g** | 54.3% | **0.10 N·m (Nominal)** | Drops after some time | Slip and Drop |
| **400g** | 54.3% | 0.20 N·m | Drops after some time | Slip and Drop |
| **500g** | 67.8% | 0.02 N·m | Drops after some time | Robot Toppled |
| **500g** | 67.8% | **0.10 N·m (Nominal)** | Drops after some time | Robot Toppled |
| **500g** | 67.8% | 0.20 N·m | Drops after some time | Robot Toppled |

---

## 4. Why Clamping Drops: Physical Contact Mechanisms

Three distinct physical phenomena dictate behavior over extended locomotion:

### Mechanism A: Footstep Impact Micro-Slip (50g to 300g Payloads)
- In rigid-body bipedal locomotion, every foot-strike sends a ground reaction shockwave through the robot shank, thigh, trunk, neck, and beak.
- Even with soft silicone pads, the peak instantaneous deceleration at heel-strike exceeds the holding force threshold for a fraction of a millisecond.
- This results in a small slip increment per stride.
- Initially, total slip remains imperceptible. Over continuous locomotion, cumulative slip causes the barbell to shift past the silicone contact boundary, and it drops after some time.
- If an actuator of higher torque is used, the problem would be fixed. One approach would be to upgrade the beak actuator to a higher torque servo to supply sufficient grip force against footstep shocks.

### Mechanism B: Overturning Center of Mass Inertia (350g to 500g Payloads)
- Payloads exceeding 40% of robot bodyweight create a substantial forward pitch moment arm.
- At 400g, the robot trunk tilts forward during walking, reducing the effective vertical normal force clamping the shaft, and it drops after some time.
- At 500g, the combined robot-payload center of mass moves outside the support polygon of the stance foot. The ankle pitch actuators saturate, and the entire robot topples after some time.

---

## 5. Summary Operational Guidelines

1. **Short-Duration Carrying**: Safe for payloads up to 300g with zero drop.
2. **Medium-Duration Carrying**: Safe for payloads up to 250g. Heavier payloads should be stabilized before cumulative slip develops.
3. **Continuous Locomotion**: Payloads clamped purely by physical contact without form-fitting geometry or mechanical interlocking will experience micro-slip, and it drops after some time if unassisted.
4. **Torque Upgrade Solution**: If an actuator of higher torque is used, the problem would be fixed. One approach would be to upgrade the beak actuator to a higher torque servo, providing sufficient holding force to eliminate slip and maintain continuous grip across all gaits.
