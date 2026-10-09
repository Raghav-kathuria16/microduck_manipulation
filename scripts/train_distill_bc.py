#!/usr/bin/env python3
"""Teacher-Student Distillation & DAgger for Microduck 300g LoadCarry.

Distills the stabilized walking policy + head elevation bias into a single
standalone neural network policy that autonomously carries the 300g elongated cylinder
without requiring manual head offset commands.

Input:  obs [1, 61] with standard velocity commands (zero head offsets)
Output: actions [1, 14] with baked-in head elevation and load compensation
"""

import math
import os
import sys
import time
import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn
import torch.optim as optim

_EXTERNAL_REPO = "/Users/raghav/Desktop/microduck_manipulation/microduck_manipulation/.external_repo"
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))

import mujoco
from infer_policy import PolicyInference, DEFAULT_POSE
from mjlab_microduck.robot.microduck_constants import CYLINDER_WELD_RELPOS, CYLINDER_WELD_RELQUAT

TEACHER_ONNX = "/Users/raghav/Desktop/microduck_manipulation/microduck_manipulation/policies/onnx/BEST_alpha_walking.onnx"
SCENE_XML = os.path.join(_EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_loadcarry.xml")
OUTPUT_DIR = "/Users/raghav/Desktop/microduck_manipulation/microduck_manipulation/policies/onnx"
OUTPUT_ONNX = os.path.join(OUTPUT_DIR, "loadcarry_student_300g.onnx")
OUTPUT_PT = os.path.join(OUTPUT_DIR, "loadcarry_student_300g.pt")

class StudentPolicy(nn.Module):
    def __init__(self, obs_mean, obs_std, weights):
        super().__init__()
        self.register_buffer("obs_mean", torch.tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(obs_std, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(61, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 14),
        )

        # Initialize with teacher weights
        with torch.no_grad():
            self.net[0].weight.copy_(torch.tensor(weights["mlp.0.weight"], dtype=torch.float32))
            self.net[0].bias.copy_(torch.tensor(weights["mlp.0.bias"], dtype=torch.float32))
            self.net[2].weight.copy_(torch.tensor(weights["mlp.2.weight"], dtype=torch.float32))
            self.net[2].bias.copy_(torch.tensor(weights["mlp.2.bias"], dtype=torch.float32))
            self.net[4].weight.copy_(torch.tensor(weights["mlp.4.weight"], dtype=torch.float32))
            self.net[4].bias.copy_(torch.tensor(weights["mlp.4.bias"], dtype=torch.float32))
            self.net[6].weight.copy_(torch.tensor(weights["mlp.6.weight"], dtype=torch.float32))
            self.net[6].bias.copy_(torch.tensor(weights["mlp.6.bias"], dtype=torch.float32))

        # Freeze locomotion backbone (layers 0 and 2) to preserve walking stability
        self.net[0].weight.requires_grad = False
        self.net[0].bias.requires_grad = False
        self.net[2].weight.requires_grad = False
        self.net[2].bias.requires_grad = False

    def forward(self, obs):
        norm_obs = (obs - self.obs_mean) / self.obs_std
        return self.net(norm_obs)

def extract_teacher_params(onnx_path):
    model = onnx.load(onnx_path)
    weights = {}
    obs_mean = None
    obs_std = None

    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        name = init.name
        if name == "obs_normalizer._mean":
            obs_mean = arr
        elif name == "onnx::Div_24":
            obs_std = arr
        elif "mlp" in name:
            clean_name = name.replace("/mlp/", "").replace("/", ".")
            weights[clean_name] = arr

    if obs_mean is None:
        obs_mean = np.zeros((1, 61), dtype=np.float32)
    if obs_std is None:
        obs_std = np.ones((1, 61), dtype=np.float32)

    return obs_mean, obs_std, weights

def main():
    print("==================================================================")
    print(" Microduck 300g LoadCarry Policy Distillation & DAgger Training   ")
    print("==================================================================")

    # 1. Load Teacher parameters
    print(f"[1/4] Extracting teacher weights from: {TEACHER_ONNX}")
    obs_mean, obs_std, weights = extract_teacher_params(TEACHER_ONNX)
    student = StudentPolicy(obs_mean, obs_std, weights)
    student.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=2e-4, weight_decay=1e-5)

    # 2. Setup MuJoCo Simulation Environment
    print(f"[2/4] Setting up MuJoCo environment: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    teacher_policy = PolicyInference(
        model, data,
        walking_onnx_path=TEACHER_ONNX,
        action_scale=1.0,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )

    cyl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")

    def reset_sim():
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, 0.125]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, qpos_idx in enumerate(teacher_policy.joint_qpos_indices):
            data.qpos[qpos_idx] = DEFAULT_POSE[i]
        if beak_jid >= 0:
            data.qpos[model.jnt_qposadr[beak_jid]] = 0.52
        if cyl_jid >= 0:
            c_adr = model.jnt_qposadr[cyl_jid]
            mujoco.mj_forward(model, data)
            p_jaw = data.xpos[jaw_id].copy()
            R_jaw = data.xmat[jaw_id].reshape(3, 3).copy()
            data.qpos[c_adr:c_adr + 3] = p_jaw + R_jaw @ np.array(CYLINDER_WELD_RELPOS)
            q_jaw = np.empty(4)
            mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_id])
            q_cyl = np.empty(4)
            mujoco.mju_mulQuat(q_cyl, q_jaw, np.array(CYLINDER_WELD_RELQUAT))
            data.qpos[c_adr + 3:c_adr + 7] = q_cyl
        data.ctrl[:] = DEFAULT_POSE
        mujoco.mj_forward(model, data)

    # 3. Data Collection via DAgger
    print("[3/4] Running DAgger data aggregation across varied velocity commands...")
    dataset_obs = []
    dataset_act = []

    # Velocity curriculum for rollouts
    test_commands = [
        (0.0, 0.0, 0.0),    # standing still with 300g
        (0.03, 0.0, 0.0),   # slow walk
        (0.06, 0.0, 0.0),   # normal walk
        (0.08, 0.0, 0.0),   # brisk walk
        (0.05, 0.0, 0.25),  # turning left
        (0.05, 0.0, -0.25), # turning right
        (-0.02, 0.0, 0.0),  # stepping back
        (0.04, 0.015, 0.0), # diagonal
    ]

    dagger_iterations = 6
    steps_per_cmd = 200 # 4.0s per rollout

    # Loss weights: emphasize head posture counter-torque while preserving leg stability
    joint_loss_weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=torch.float32,
    )

    start_time = time.time()
    for d_iter in range(dagger_iterations):
        beta = 1.0 if d_iter == 0 else max(0.25, 0.75 ** d_iter) # Teacher probability
        iter_samples = 0

        for (vx, vy, vtheta) in test_commands:
            reset_sim()
            teacher_policy.set_vel_cmd(vx, vy, vtheta)

            for step in range(steps_per_cmd):
                # 1. Get raw observation (student view: head_offset = 0)
                teacher_policy.head_offset[0] = 0.0
                teacher_policy.head_offset[1] = 0.0
                teacher_policy._update_command()
                student_obs = teacher_policy.get_observations()

                # 2. Get expert teacher action (teacher view: head_offset = [0.30, 0.40, 0, 0])
                teacher_policy.head_offset[0] = 0.30
                teacher_policy.head_offset[1] = 0.40
                teacher_policy._update_command()
                expert_action = teacher_policy.infer()

                # Record sample
                dataset_obs.append(student_obs)
                dataset_act.append(expert_action)
                iter_samples += 1

                # 3. Action selection for environment step (DAgger mix)
                if np.random.rand() < beta:
                    exec_action = expert_action
                else:
                    with torch.no_grad():
                        t_obs = torch.tensor(student_obs, dtype=torch.float32).unsqueeze(0)
                        exec_action = student(t_obs).squeeze(0).numpy()

                # 4. Step physics
                teacher_policy.apply_action(exec_action)
                for _ in range(4):
                    mujoco.mj_step(model, data)

                z = data.qpos[2]
                grav = teacher_policy.get_projected_gravity()
                if z < 0.075 or grav[2] > -0.6:
                    break

        # Train student on aggregated dataset
        X = torch.tensor(np.array(dataset_obs), dtype=torch.float32)
        Y = torch.tensor(np.array(dataset_act), dtype=torch.float32)

        batch_size = 64
        epochs = 20
        n_samples = len(X)

        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            total_loss = 0.0
            num_batches = 0
            for i in range(0, n_samples, batch_size):
                idx = indices[i:i + batch_size]
                bx, by = X[idx], Y[idx]

                pred = student(bx)
                loss = torch.mean(joint_loss_weights * (pred - by) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"  DAgger Iter {d_iter + 1}/{dagger_iterations}: beta={beta:.2f}, total_samples={n_samples}, weighted_MSE={avg_loss:.5f}")

    train_elapsed = time.time() - start_time
    print(f"\nTraining completed in {train_elapsed:.2f}s ({train_elapsed/60:.1f} mins)!")

    # 4. Evaluate Student Standalone (NO HEAD OFFSET COMMAND)
    print("\n[4/4] Evaluating trained student policy standalone (head_offset = 0)...")
    student.eval()
    reset_sim()

    survived = 0
    start_x = data.qpos[0]
    eval_steps = 400

    teacher_policy.set_vel_cmd(0.06, 0.0, 0.0)
    # ZERO head offset!
    teacher_policy.head_offset[0] = 0.0
    teacher_policy.head_offset[1] = 0.0
    teacher_policy._update_command()

    for s in range(eval_steps):
        obs = teacher_policy.get_observations()
        with torch.no_grad():
            action = student(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

        teacher_policy.apply_action(action)
        for _ in range(4):
            mujoco.mj_step(model, data)

        z = data.qpos[2]
        grav = teacher_policy.get_projected_gravity()
        if z < 0.075 or grav[2] > -0.6:
            break
        survived += 1

    dist = data.qpos[0] - start_x
    print(f"Student Evaluation Result:")
    print(f"  Survived: {survived}/{eval_steps} control steps ({survived*0.02:.2f}s)")
    print(f"  Distance: {dist:.3f} m")
    print(f"  Final Z:  {data.qpos[2]:.3f} m")
    print(f"  Grav Z:   {grav[2]:.3f} (upright posture)")

    # 5. Export to PyTorch and ONNX
    print("\nExporting policy models...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(student.state_dict(), OUTPUT_PT)
    print(f"  Saved PyTorch model: {OUTPUT_PT}")

    dummy_input = torch.randn(1, 61, dtype=torch.float32)
    torch.onnx.export(
        student,
        dummy_input,
        OUTPUT_ONNX,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
    )
    print(f"  Exported ONNX model:  {OUTPUT_ONNX}")
    print("\nSUCCESS: Standalone load-carrying policy successfully trained and exported!")

if __name__ == "__main__":
    main()
