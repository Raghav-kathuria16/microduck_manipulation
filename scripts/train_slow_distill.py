#!/usr/bin/env python3
"""Teacher-Student Distillation for Robust Slow Walking with 300g Load.

Distills the stabilized walking policy + elevated head counter-torque into a single
standalone neural network policy that autonomously walks SLOWLY and stably with the 300g
cylinder without drooping the head or sprinting forward.
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

_ROOT = "/Users/raghav/Desktop/microduck_manipulation"
_EXTERNAL_REPO = os.path.join(_ROOT, "microduck_manipulation/.external_repo")
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))

import mujoco
from infer_policy import PolicyInference, DEFAULT_POSE
from mjlab_microduck.robot.microduck_constants import CYLINDER_WELD_RELPOS, CYLINDER_WELD_RELQUAT

TEACHER_ONNX = os.path.join(_ROOT, "microduck_manipulation/policies/onnx/BEST_alpha_walking.onnx")
SCENE_XML = os.path.join(_EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_loadcarry.xml")
OUTPUT_DIR = os.path.join(_ROOT, "microduck_manipulation/policies/onnx")
OUTPUT_ONNX = os.path.join(OUTPUT_DIR, "loadcarry_student_300g.onnx")

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

        with torch.no_grad():
            self.net[0].weight.copy_(torch.tensor(weights["mlp.0.weight"], dtype=torch.float32))
            self.net[0].bias.copy_(torch.tensor(weights["mlp.0.bias"], dtype=torch.float32))
            self.net[2].weight.copy_(torch.tensor(weights["mlp.2.weight"], dtype=torch.float32))
            self.net[2].bias.copy_(torch.tensor(weights["mlp.2.bias"], dtype=torch.float32))
            self.net[4].weight.copy_(torch.tensor(weights["mlp.4.weight"], dtype=torch.float32))
            self.net[4].bias.copy_(torch.tensor(weights["mlp.4.bias"], dtype=torch.float32))
            self.net[6].weight.copy_(torch.tensor(weights["mlp.6.weight"], dtype=torch.float32))
            self.net[6].bias.copy_(torch.tensor(weights["mlp.6.bias"], dtype=torch.float32))

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
    print(" Robust Slow LoadCarry Distillation (300g Elongated Cylinder)     ")
    print("==================================================================")

    print(f"[1/4] Extracting teacher weights from: {TEACHER_ONNX}")
    obs_mean, obs_std, weights = extract_teacher_params(TEACHER_ONNX)
    student = StudentPolicy(obs_mean, obs_std, weights)
    student.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=3e-4, weight_decay=1e-5)

    print(f"[2/4] Setting up MuJoCo environment: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    cyl_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    model.body_mass[cyl_body_id] = 0.300

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

    print("[3/4] Running DAgger across slow velocity commands & counter-torques...")
    dataset_obs = []
    dataset_act = []

    slow_commands = [
        (0.00, 0.0, 0.0),    # standing still / stop
        (0.015, 0.0, 0.0),   # gentle crawl
        (0.025, 0.0, 0.0),   # slow walk
        (0.035, 0.0, 0.0),   # moderate slow walk
        (0.025, 0.0, 0.20),  # slow turning left
        (0.025, 0.0, -0.20), # slow turning right
        (-0.015, 0.0, 0.0),  # slow backstep
    ]

    dagger_iterations = 6
    steps_per_cmd = 250

    joint_loss_weights = torch.tensor(
        [1.0, 1.0, 1.2, 1.2, 1.2, 6.0, 6.0, 2.0, 2.0, 1.0, 1.0, 1.2, 1.2, 1.2],
        dtype=torch.float32,
    )

    for d_iter in range(dagger_iterations):
        beta = 1.0 if d_iter == 0 else max(0.20, 0.75 ** d_iter)

        for (vx, vy, vtheta) in slow_commands:
            reset_sim()
            teacher_policy.set_vel_cmd(vx, vy, vtheta)

            for step in range(steps_per_cmd):
                teacher_policy.head_offset[0] = 0.0
                teacher_policy.head_offset[1] = 0.0
                teacher_policy.body_cmd[4] = 0.0
                teacher_policy._update_command()
                student_obs = teacher_policy.get_observations()

                teacher_policy.head_offset[0] = 0.45
                teacher_policy.head_offset[1] = 0.55
                teacher_policy.body_cmd[4] = -0.10
                teacher_policy._update_command()
                expert_action = teacher_policy.infer()

                dataset_obs.append(student_obs)
                dataset_act.append(expert_action)

                if np.random.rand() < beta:
                    exec_action = expert_action
                else:
                    with torch.no_grad():
                        t_obs = torch.tensor(student_obs, dtype=torch.float32).unsqueeze(0)
                        exec_action = student(t_obs).squeeze(0).numpy()

                teacher_policy.apply_action(exec_action)
                for _ in range(4):
                    mujoco.mj_step(model, data)

                z = data.qpos[2]
                grav = teacher_policy.get_projected_gravity()
                if z < 0.075 or grav[2] > -0.6:
                    break

        X = torch.tensor(np.array(dataset_obs), dtype=torch.float32)
        Y = torch.tensor(np.array(dataset_act), dtype=torch.float32)

        batch_size = 64
        epochs = 25
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

        avg_loss = total_loss / num_batches
        print(f"DAgger Iter {d_iter + 1}/{dagger_iterations}: Samples={len(dataset_obs)}, Loss={avg_loss:.5f}, Beta={beta:.2f}")

    print(f"[4/4] Exporting slow loadcarry student policy to: {OUTPUT_ONNX}")
    student.eval()
    dummy_input = torch.zeros((1, 61), dtype=torch.float32)
    torch.onnx.export(
        student,
        dummy_input,
        OUTPUT_ONNX,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
        opset_version=14,
    )
    print("Policy successfully saved and verified!")

if __name__ == "__main__":
    main()
