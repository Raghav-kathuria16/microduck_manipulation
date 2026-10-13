# MicroDuck Manipulation & LoadCarry: Pipeline Architecture, Reward Functions & RL Theory Guide

Author: DeepMind Advanced Agentic Coding Pair Program  
Repository 1: [microduck_rl](microduck_rl) (Parallel Vectorized RL Training)  
Repository 2: [microduck_manipulation](microduck_manipulation) (Standalone Inference & Control)  

---

## 1. System Architecture & High-Level Pipeline

The MicroDuck project is structured into two cleanly separated engineering layers:

```
+---------------------------------------------------------------------------------------------------+
| TRAINING PIPELINE: microduck_rl                                                                  |
|                                                                                                   |
|  1. Robot Asset & Scene Setup (MJCF XML in src/mjlab_microduck/robot/microduck/)                 |
|     - Rigid kinematic tree (14 actuated servos + passive linkage joints)                         |
|     - Silicone mouth linings, contact geom pairs, collision filtering                             |
|                                                                                                   |
|  2. Parallel Simulation (MuJoCo-Warp / mjlab)                                                     |
|     - 4,096 parallel environments executed simultaneously                                        |
|     - Physics simulation at 200 Hz (timestep dt = 0.002 s)                                       |
|     - Policy decision decimation = 4 (control frequency at 50 Hz, dt = 0.02 s)                   |
|                                                                                                   |
|  3. Observation Assembly (mdp.py & env configs)                                                   |
|     - 61-dimensional actor observation vector (or 69-dimensional for ball balancing)              |
|     - 76-dimensional privileged critic observation vector                                        |
|     - Domain randomizations: mass scaling, motor gain noise, IMU alignment noise, sensor delays  |
|                                                                                                   |
|  4. PyTorch Actor-Critic Neural Network (rsl_rl framework)                                        |
|     - Empirical running observation normalizer (tracks mean and variance)                         |
|     - Actor MLP: [61 -> 512 -> 256 -> 128 -> 14] + Gaussian Distribution                          |
|     - Critic MLP: [76 -> 512 -> 256 -> 128 -> 1] (value baseline)                                 |
|                                                                                                   |
|  5. PPO Optimization Loop                                                                         |
|     - Rollout collection (24 steps per environment = 98,304 samples per iteration)                |
|     - Generalized Advantage Estimation (GAE)                                                      |
|     - Clipped surrogate loss, value loss clipping, entropy bonus, adaptive KL schedule            |
|                                                                                                   |
|  6. ONNX Policy Export (torch.onnx.export)                                                        |
|     - Merges running normalizer and actor MLP weights into standalone .onnx file                  |
|     - Strips training stochasticity; outputs deterministic action mean                            |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                   Exported Policy (.onnx file)
                                                  v
+---------------------------------------------------------------------------------------------------+
| INFERENCE & RUNTIME PIPELINE: microduck_manipulation                                              |
|                                                                                                   |
|  1. Standalone MuJoCo Simulation Environment                                                      |
|     - Zero PyTorch or GPU dependencies required at runtime                                        |
|     - Runs official MuJoCo C/Python bindings with real-time interactive viewer                    |
|                                                                                                   |
|  2. ONNX Runtime Engine (infer_policy.py)                                                         |
|     - Ingests sensor data from MuJoCo (IMU gyro, projected gravity, joint encoders)               |
|     - Constructs the exact 61-dimensional observation vector                                      |
|     - Evaluates .onnx model in microseconds via onnxruntime                                       |
|                                                                                                   |
|  3. Kinematic Steadicam Pantograph & Posture Control                                              |
|     - Slew-rate filtering suppresses sudden acceleration spikes                                   |
|     - Differential neck-to-head pantograph maintains perfectly level carrying posture            |
|     - Active beak clamping holds payloads securely inside the silicone mouth cradle               |
|                                                                                                   |
|  4. Low-Level Actuator Transfer & Physics Step                                                    |
|     - Target positions sent to BAM XL-330 motor model with voltage bounds                         |
|     - MuJoCo advances simulation by 4 substeps per policy tick                                    |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Where Are The Reward Functions?

All reward functions are located in the training repository:

- Core reward library: [mdp.py](src/mjlab_microduck/tasks/mdp.py)
- Pure physical clamping load-carry task: [microduck_ppo_headup_v4_env_cfg.py](src/mjlab_microduck/tasks/microduck_ppo_headup_v4_env_cfg.py)
- Base load-carry environment configuration: [microduck_loadcarry_env_cfg.py](src/mjlab_microduck/tasks/microduck_loadcarry_env_cfg.py)
- Table tennis ball balancing configuration: [microduck_table_tennis_ball_balance_env_cfg.py](src/mjlab_microduck/tasks/microduck_table_tennis_ball_balance_env_cfg.py)
- Table tennis racket carry configuration: [microduck_table_tennis_racket_carry_env_cfg.py](src/mjlab_microduck/tasks/microduck_table_tennis_racket_carry_env_cfg.py)
- Base bipedal velocity configuration: [microduck_velocity_env_cfg.py](src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)

### Detailed Breakdown of Every Reward Function

The total reward at each step is the weighted sum of individual terms:
Total Reward = Sum over all terms of ( weight_k * reward_k )

Each term is explained below in clear, accessible language without LaTeX notation:

#### 1. Forward Linear Velocity Tracking (`track_linear_velocity`)
- **Location**: Defined in `mjlab.tasks.velocity.mdp` and customized in [microduck_velocity_env_cfg.py](src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py#L599-L601).
- **Physical Purpose**: Incentivizes the robot to match the commanded forward and lateral velocity targets.
- **Mathematical Formula**:
  `linear_velocity_error = (robot_velocity_x - commanded_velocity_x)^2 + (robot_velocity_y - commanded_velocity_y)^2`  
  `reward = exp( - linear_velocity_error / (std_dev^2) )`
- **How It Works**: The exponential kernel outputs a maximum value of 1.0 when error is exactly zero. When error equals the standard deviation tolerance (for example 0.08 m/s or 0.12 m/s), reward drops to 0.368 (1/e). Farther errors smoothly approach zero without producing discontinuous gradients.

#### 2. Angular Velocity Yaw Tracking (`track_angular_velocity`)
- **Location**: Defined in `mjlab.tasks.velocity.mdp`.
- **Physical Purpose**: Rewards turning in place or along an arc matching the commanded yaw angular velocity.
- **Mathematical Formula**:
  `yaw_error = (robot_yaw_rate - commanded_yaw_rate)^2`  
  `reward = exp( - yaw_error / (std_dev^2) )`

#### 3. Upright Base Orientation (`upright` / `body_upright_gaussian`)
- **Location**: [mdp.py](src/mjlab_microduck/tasks/mdp.py#L710-L732).
- **Physical Purpose**: Keeps the robot's main torso upright relative to the gravity vector.
- **Mathematical Formula**:
  `projected_gravity_z = dot_product(robot_up_vector, world_gravity_vector)`  
  `upright_error = 1.0 + projected_gravity_z`  
  `reward = exp( - (upright_error / std_dev)^2 )`
- **How It Works**: When the torso is upright, projected gravity points straight down along the body negative z-axis, making projected_gravity_z equal to -1.0. The upright_error is zero, yielding maximum reward 1.0. If the robot tilts or tips over, error grows rapidly and reward drops to zero.

#### 4. Head Cradle Posture Reward (`head_cradle_reward`)
- **Location**: [microduck_ppo_headup_v4_env_cfg.py](src/mjlab_microduck/tasks/microduck_ppo_headup_v4_env_cfg.py#L135-L154).
- **Physical Purpose**: In physical clamping, front-heavy loads naturally tilt downward and slip out. This reward trains the robot to pitch its neck upward and head slightly back, creating an upward cradle that uses gravity to settle the payload deep into the mouth pads.
- **Mathematical Formula**:
  `neck_error = absolute_value(neck_pitch_angle - target_neck_pitch)`  
  `head_error = absolute_value(head_pitch_angle - target_head_pitch)`  
  `squared_posture_error = (neck_error / std_dev)^2 + (head_error / std_dev)^2`  
  `reward = exp( - squared_posture_error )`
- **Tuned Targets**: Target neck pitch is +0.35 radians (elevated), target head pitch is -0.08 radians (tilted backward), with a standard deviation of 0.12 radians and a heavy positive weight of +8.0.

#### 5. Payload Retention Reward (`cylinder_retained_reward`)
- **Location**: [microduck_ppo_headup_v4_env_cfg.py](src/mjlab_microduck/tasks/microduck_ppo_headup_v4_env_cfg.py#L116-L133).
- **Physical Purpose**: Actively rewards keeping the clamped payload dead-center in the mouth cavity.
- **Mathematical Formula**:
  `relative_position = rotate_by_inverse_quaternion(head_orientation, payload_position - head_position)`  
  `drift = absolute_value(relative_position_z - target_relative_z)`  
  `reward = exp( - (drift / tolerance)^2 )`
- **How It Works**: The payload position is transformed into the robot head local coordinate frame. If the payload slips or slides forward or backward, drift increases. At zero drift, the policy earns the maximum reward of +10.0.

#### 6. Torso Pitch Counterbalance Lean (`torso_pitch_counterbalance`)
- **Location**: [mdp.py](src/mjlab_microduck/tasks/mdp.py#L7454-L7467).
- **Physical Purpose**: Carrying a front-loaded cantilevered mass pulls the robot's center of mass forward over its toes. This reward encourages a slight backward torso lean (-5.0 degrees) to physically counterbalance the front mass.
- **Mathematical Formula**:
  `torso_pitch = arctan2(gravity_x, - gravity_z)`  
  `lean_error = torso_pitch - target_lean_angle`  
  `reward = exp( - (lean_error / tolerance)^2 )`

#### 7. Beak Upright Orientation (`beak_up_orientation`)
- **Location**: [mdp.py](src/mjlab_microduck/tasks/mdp.py#L7322-L7349).
- **Physical Purpose**: Prevents the beak and jaw from pitching downward toward the floor.
- **Mathematical Formula**:
  `beak_direction_world = rotate_vector(beak_orientation, local_x_axis)`  
  `angle_from_vertical = arccos(beak_direction_world_z)`  
  `error = angle_from_vertical - target_angle`  
  `reward = exp( - (error / tolerance)^2 )`

#### 8. Walking-Coupled Head Up Reward (`walking_coupled_head_up`)
- **Location**: [mdp.py](src/mjlab_microduck/tasks/mdp.py#L7416-L7438).
- **Physical Purpose**: Solves the "statue duck" failure mode where a robot achieves high head-up rewards by standing completely still.
- **Mathematical Formula**:
  `head_up_factor = exp( - (neck_pitch - target_neck_pitch)^2 / (tolerance^2) )`  
  `speed_factor = clamp( (forward_velocity - min_speed) / 0.15, min=0.0, max=1.0 )`  
  `reward = head_up_factor * speed_factor`
- **How It Works**: If forward velocity is below the minimum threshold, speed_factor is 0.0, so the robot earns zero reward regardless of how high its head is held. It only gets rewarded when simultaneously elevating the head and walking forward.

#### 9. Dynamic Foot Swing Air Time (`feet_air_time_upright`)
- **Location**: [mdp.py](src/mjlab_microduck/tasks/mdp.py#L527-L544).
- **Physical Purpose**: Incentivizes a dynamic alternating bipedal stepping gait rather than shuffling feet along the ground.
- **Mathematical Formula**:
  When a foot makes contact with the ground:  
  `reward = (air_time_duration - target_air_time) * is_upright_flag`  
  When foot is in mid-air:  
  `reward = 0.0`

#### 10. Smoothness & Actuator Protection Penalties
- **Action Rate Penalty (`action_rate_l2`)**:
  - Formula: `penalty = - sum over all joints of (action_current - action_previous)^2`
  - Purpose: Suppresses high-frequency chatter and vibrations that shake the beak and induce payload slip.
- **Joint Acceleration Penalty (`joint_accelerations_l2`)**:
  - Formula: `penalty = - sum over all joints of (joint_acceleration)^2`
  - Purpose: Protects physical Dynamixel servo gearboxes by penalizing sudden joint jerks.
- **Joint Position Limits (`dof_pos_limits`)**:
  - Formula: `penalty = - sum over all joints of max(0, joint_position - max_limit)^2 - sum of max(0, min_limit - joint_position)^2`
  - Purpose: Quadratic barrier penalty preventing actuators from slamming into mechanical end-stops.

---

## 3. Where Is The PyTorch Network? Is There One?

**Yes, there is an explicit PyTorch neural network.**

During the reinforcement learning phase, the neural network is created, parameterized, and trained in PyTorch. It is implemented via the `rsl_rl` library and wrapped by `mjlab`.

### Exact Code Locations for the PyTorch Network

1. **Top-Level Model Class**:
   - Class: `MLPModel` in `rsl_rl.models.mlp_model`
   - Defines the combined input preprocessor, observation normalizer, MLP layers, and action distribution.
2. **Core Multi-Layer Perceptron (MLP)**:
   - Class: `MLP` in `rsl_rl.modules.mlp`
   - Subclasses `torch.nn.Sequential` and instantiates the linear feedforward layers and activations.
3. **Observation Normalizer**:
   - Class: `EmpiricalNormalization` in `rsl_rl.modules.normalization`
   - Subclasses `torch.nn.Module`. Tracks empirical running mean and variance to normalize input signals on the fly.
4. **Action Distribution**:
   - Class: `GaussianDistribution` in `rsl_rl.modules.distribution`
   - Subclasses `rsl_rl.modules.distribution.Distribution`. Implements continuous Gaussian policy sampling with learned standard deviation.
5. **Warm-Starting & Layer Migration**:
   - Methods: `_warmstart_from_onnx` in [MicroduckLoadCarryOnPolicyRunner](src/mjlab_microduck/tasks/__init__.py#L35-L102) and [MicroduckTableTennisBallBalanceRunner](src/mjlab_microduck/tasks/__init__.py#L104-L180).
   - Directly loads layer weights, biases, and normalization tensors into the active PyTorch model.

### Actor Network Architecture (The Policy)

The Actor network maps robot observations to target joint positions.

```
Raw Observations s_t (61 dimensions, or 69 for ball balancing)
                 |
                 v
+--------------------------------------------------------------+
| Empirical Observation Normalizer                             |
| Formula: norm_obs = (obs - running_mean) / sqrt(var + eps)   |
| Parameters: running_mean (61-D), running_var (61-D)          |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| PyTorch Linear Layer 0: Linear(in_features=61, out=512)      |
| Weights: [512, 61], Biases: [512]                            |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| Activation Function 1: ELU (Exponential Linear Unit)         |
| Formula: f(x) = x if x > 0 else (exp(x) - 1)                 |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| PyTorch Linear Layer 2: Linear(in_features=512, out=256)     |
| Weights: [256, 512], Biases: [256]                           |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| Activation Function 3: ELU                                   |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| PyTorch Linear Layer 4: Linear(in_features=256, out=128)     |
| Weights: [128, 256], Biases: [128]                           |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| Activation Function 5: ELU                                   |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| PyTorch Linear Layer 6: Linear(in_features=128, out=14)      |
| Weights: [14, 128], Biases: [14]                             |
+--------------------------------------------------------------+
                 |
                 v
Mean Joint Action Offsets mu (14 dimensions)
                 |
                 v
+--------------------------------------------------------------+
| Gaussian Distribution Module                                 |
| Sample: action ~ Normal(mean=mu, std=learnable_std)          |
| During Evaluation / ONNX Export: action = mu (deterministic) |
+--------------------------------------------------------------+
                 |
                 v
Final Commanded Joint Targets a_t (14 dimensions)
```

#### Actor Input Observation Breakdown (61 Dimensions)
1. **Base Angular Velocity** (3 dimensions): IMU gyro rate [omega_x, omega_y, omega_z].
2. **Projected Gravity** (3 dimensions): World gravity vector projected into robot base coordinates [g_x, g_y, g_z].
3. **Relative Joint Positions** (14 dimensions): Current measured angle minus default standing angle for all 14 servos.
4. **Joint Velocities** (14 dimensions): Angular velocity for all 14 servos.
5. **Previous Actions** (14 dimensions): Action targets output by the policy on the prior timestep.
6. **Command Vector** (13 dimensions):
   - Linear and turning velocity commands: [vx, vy, yaw_rate] (3 dimensions).
   - Commanded head pose deltas: [neck_pitch, head_pitch, head_yaw, head_roll] (4 dimensions).
   - Commanded body pose offsets: [x, y, z, roll, pitch, yaw] (6 dimensions).
   - Total: 3 + 3 + 14 + 14 + 14 + 13 = 61 dimensions.

*(For table tennis ball balancing, 8 additional state features are appended: 2D ball position offset on paddle face, 2D relative ball velocity, 3D paddle normal vector, and 1D contact indicator, totaling 69 dimensions).*

### Critic Network Architecture (The Value Function)

The Critic network estimates the expected cumulative future return from state s. Because the Critic is only used during training and never deployed on physical hardware, it is provided with **privileged information**:

- **Critic Input Dimensions**: 76 dimensions.
  - Composed of the 61 actor dimensions plus 15 privileged ground truth features:
    - Base linear velocity in world frame (3 dimensions).
    - Foot ground contact forces for left and right feet (6 dimensions).
    - Foot air time duration (2 dimensions).
    - Foot height above terrain (4 dimensions).
- **Architecture**:
  - Observation normalizer: tracks 76-dimensional running statistics.
  - Layer 0: `Linear(76, 512)` -> `ELU()`
  - Layer 2: `Linear(512, 256)` -> `ELU()`
  - Layer 4: `Linear(256, 128)` -> `ELU()`
  - Layer 6: `Linear(128, 1)` -> Scalar Value Estimate `V(s)`.

### Why You Do Not See PyTorch in `microduck_manipulation`

You do not see PyTorch imports in `microduck_manipulation/scripts/infer_policy.py` because `microduck_manipulation` is the **lightweight, real-time deployment repository**.

1. At the conclusion of training in `microduck_rl`, the policy is exported using `torch.onnx.export`.
2. The export freezes the running mean and variance into constant tensors and fuses the observation normalizer with the linear layers.
3. The resulting `.onnx` file (for example `BEST_alpha_walking.onnx` or `physical_clamp_v4.onnx`) is completely self-contained.
4. In `microduck_manipulation`, the model is loaded using `onnxruntime.InferenceSession`. ONNX Runtime executes the feedforward pass in high-performance C++ binaries with zero memory overhead and zero PyTorch dependencies.

---

## 4. Proximal Policy Optimization (PPO) & Hyperparameters

### Plain-Language Formulation of PPO

PPO is an on-policy actor-critic algorithm that balances policy improvement with stability by restricting the step size taken in parameter space.

#### 1. Trajectory Rollout Collection
The current policy interacts with 4,096 parallel simulated environments for 24 steps, collecting a buffer of 98,304 transitions: `(state, action, reward, next_state, old_action_probability, value_estimate)`.

#### 2. Generalized Advantage Estimation (GAE)
We compute how much better an action was compared to the average expected return.
First, compute the single-step temporal difference residual:
`td_error_t = reward_t + gamma * critic_value(next_state) - critic_value(state)`

Then, compute the advantage as an exponentially discounted sum of future residuals:
`advantage_t = td_error_t + (gamma * lambda) * td_error_{t+1} + (gamma * lambda)^2 * td_error_{t+2} + ...`
Here:
- `gamma = 0.99` controls how far into the future the agent looks.
- `lambda = 0.95` smoothly trades off bias against variance.

Advantage values are then normalized across the batch:
`normalized_advantage = (advantage - mean_advantage) / (std_advantage + 1e-8)`

#### 3. Clipped Surrogate Objective
To prevent destructive updates when training on the same rollout data for multiple epochs, PPO calculates the probability ratio between the updated policy and the old policy:
`probability_ratio = current_policy_probability / old_policy_probability`

The clipped objective is:
`unclipped_objective = probability_ratio * normalized_advantage`  
`clipped_ratio = clamp(probability_ratio, min = 1.0 - epsilon, max = 1.0 + epsilon)`  
`clipped_objective = clipped_ratio * normalized_advantage`  
`policy_loss = - minimum(unclipped_objective, clipped_objective)`

If an action was better than average (`advantage > 0`), the optimizer increases its probability, but if the ratio exceeds `1.0 + epsilon`, the clipping removes the incentive, preventing over-confident updates.

#### 4. Clipped Value Loss
The critic is updated to minimize the difference between its predictions and empirical returns:
`value_target = normalized_advantage + old_critic_value`  
`clipped_value = clamp(current_critic_value, min = old_critic_value - epsilon_v, max = old_critic_value + epsilon_v)`  
`value_loss = 0.5 * maximum( (current_critic_value - value_target)^2, (clipped_value - value_target)^2 )`

#### 5. Entropy Regularization
To maintain exploration and prevent early collapse to stiff postures, an entropy bonus is added:
`entropy = 0.5 * sum over joints of [ log( 2 * pi * e * (std_dev^2) ) ]`

#### 6. Combined Total Loss
The complete loss minimized by the Adam optimizer across mini-batches is:
`total_loss = policy_loss + (value_loss_coefficient * value_loss) - (entropy_coefficient * entropy)`

### Master Hyperparameter Reference Table

Below are the exact hyperparameters extracted from the training configuration files:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Simulation Timestep (`dt_sim`)** | 0.002 seconds | Physics simulation runs at 200 Hz |
| **Control Decimation** | 4 sub-steps | Policy evaluates every 4 physics steps |
| **Policy Timestep (`dt_policy`)** | 0.020 seconds | Control frequency is 50 Hz |
| **Number of Environments** | 4,096 | Environments simulated in parallel on GPU |
| **Rollout Steps (`num_steps_per_env`)** | 24 steps | Steps per environment per iteration |
| **Total Batch Size per Iteration** | 98,304 transitions | 4,096 envs * 24 steps |
| **Mini-Batches (`num_mini_batches`)** | 4 | 24,576 transitions per mini-batch update |
| **Learning Epochs (`num_learning_epochs`)** | 5 epochs | Passes through the rollout buffer per iteration |
| **Discount Factor (`gamma`)** | 0.99 | Temporal credit assignment horizon |
| **GAE Parameter (`lambda`)** | 0.95 | Exponential advantage smoothing parameter |
| **PPO Clip Parameter (`clip_param`)** | 0.15 (to 0.20) | Maximum policy ratio divergence threshold |
| **Learning Rate** | 2.0e-4 (to 1.0e-3) | Adam optimizer base step size |
| **Learning Rate Schedule** | "adaptive" | Adapts based on measured KL divergence |
| **Desired KL Divergence** | 0.01 | Target KL to scale learning rate |
| **Value Loss Coefficient** | 1.0 | Relative weight of value function MSE loss |
| **Value Loss Clipping** | Enabled (`True`) | Clamps value network step size |
| **Entropy Coefficient** | 0.005 (to 0.010) | Exploration bonus weight |
| **Gradient Clipping (`max_grad_norm`)** | 1.0 | Prevents gradient explosion |
| **Initial Action Std Dev** | 0.28 radians | Gaussian exploration noise initialization |
| **Action Scale** | 0.25 | Scales normalized policy output to joint targets |
| **Actuator Model** | BAM XL-330 | Dynamixel electrical model with voltage drop |

---

## 5. Curriculum Progression & Warm-Starting Pipeline

Training a bipedal robot to walk while clamping a heavy external payload from a scratch initialization fails because the robot must simultaneously learn balance, gait coordination, and payload retention. The repository solves this through a systematic two-stage pipeline:

```
[Phase 1: Baseline Locomotion]
Trained on flat & rough terrain to follow velocity commands.
Produces: BEST_alpha_walking.onnx
        |
        v
[Phase 2: Warm-Starting]
MicroduckLoadCarryOnPolicyRunner inspects BEST_alpha_walking.onnx:
1. Copies empirical normalizer mean and variance for 61 features.
2. Copies weights and biases for Linear layers 0, 2, 4, 6.
        |
        v
[Phase 3: Payload Curriculum]
Payload mass begins light and ramps automatically in stages:
Stage 0 -> Stage 1 -> Stage 2 -> Target Mass
        |
        v
[Phase 4: Constraint Transition]
Initial stages use kinematic equality constraints to establish posture.
Final stages disable all equality constraints, switching to pure physical clamping.
```

### Physical Clamping and Actuator Torque

During dynamic walking, foot impacts produce periodic ground reaction shocks that transmit through the robot's ankles and hips up into the neck and head. 

In pure physical clamping:
1. The lower jaw is powered by an actuator generating active holding torque.
2. Contact pairs between the payload and silicone pads transmit holding forces.
3. Over extended locomotion horizons, cyclic head accelerations can induce slight payload migration. If an actuator of higher torque is used, the problem would be fixed. One approach would be to upgrade the jaw actuator to a higher torque servo to eliminate slip altogether.

---

## 6. Standalone Inference & Steadicam Pantograph Execution

In [infer_policy.py](scripts/infer_policy.py), the deployed system executes the following control loop:

1. **Sensor Acquisition**: Reads joint encoders, IMU angular velocity, and projected gravity from MuJoCo data structures.
2. **Observation Assembly**: Subtracts default standing joint angles (`DEFAULT_POSE`) to form relative coordinates and concatenates command vectors to form the 61-element array.
3. **ONNX Evaluation**: Invokes `onnxruntime.InferenceSession.run()` to obtain 14 joint offsets.
4. **Steadicam Pantograph Filtering**:
   - Applies slew-rate rate-of-change limits to prevent jerky motor demands.
   - For racket and payload carrying, the controller applies an equal and opposite differential pitch compensation between the neck and head joints. As the neck oscillates during leg swings, the head counter-rotates dynamically, keeping the mouth cradle and carrying tray level.
5. **Actuator Voltage & Torque Calculation**: Hands targets to the BAM XL-330 motor model, calculating winding voltages and motor torques.
6. **Physics Integration**: MuJoCo integrates equations of motion across 4 substeps (dt = 0.002 s) to advance world state.

---

## Summary Checklist of Code Locations

| Component | Code Location |
| :--- | :--- |
| **All Core Reward Terms** | [mdp.py](src/mjlab_microduck/tasks/mdp.py) |
| **Physical Clamping Reward & Terminations** | [microduck_ppo_headup_v4_env_cfg.py](src/mjlab_microduck/tasks/microduck_ppo_headup_v4_env_cfg.py) |
| **Base LoadCarry Config & Observations** | [microduck_loadcarry_env_cfg.py](src/mjlab_microduck/tasks/microduck_loadcarry_env_cfg.py) |
| **Table Tennis Ball Balance Rewards** | [microduck_table_tennis_ball_balance_env_cfg.py](src/mjlab_microduck/tasks/microduck_table_tennis_ball_balance_env_cfg.py) |
| **PyTorch Runner & Warm-Starter** | [__init__.py](src/mjlab_microduck/tasks/__init__.py) |
| **PyTorch MLP Architecture** | `rsl_rl.modules.mlp.MLP` in Python virtual environment |
| **PyTorch PPO Algorithm** | `rsl_rl.algorithms.ppo.PPO` in Python virtual environment |
| **Runtime ONNX Inference Engine** | [infer_policy.py](scripts/infer_policy.py) |
| **Steadicam Pantograph Test Script** | [test_spoon_lemon_balance.py](scripts/test_spoon_lemon_balance.py) |
| **Physical Clamp Policy Test Script** | [test_physical_clamp_policy.py](scripts/test_physical_clamp_policy.py) |
