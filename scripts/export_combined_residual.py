#!/usr/bin/env python3
"""Export unified Residual RL policy to a single standalone ONNX model.

Combines the frozen base policy with the learned residual policy:
    action = base_policy(obs[:base_dim]) + alpha * residual_policy(obs[:res_dim])
Input: obs [batch, max_dim] (e.g. 64D for table tennis racket carry)
Output: actions [batch, 14]
"""

import os
import sys
import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn


def build_mlp_from_inits(inits):
    """Dynamically reconstructs an MLP module from ONNX initializers."""
    weight_keys = sorted(
        [k for k in inits if k.startswith("mlp.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    layers = []
    for k in weight_keys:
        w = torch.from_numpy(inits[k]).float()
        b_name = k.replace(".weight", ".bias")
        b = torch.from_numpy(inits[b_name]).float() if b_name in inits else torch.zeros(w.shape[0])
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        layers.append(nn.ELU())
    if layers:
        layers.pop()  # Remove trailing ELU
    return nn.Sequential(*layers)


def extract_normalizer(inits):
    """Extracts mean and std from ONNX initializers robustly."""
    if "obs_normalizer._mean" in inits:
        mean = torch.from_numpy(inits["obs_normalizer._mean"]).float()
    elif "mean" in inits:
        mean = torch.from_numpy(inits["mean"]).float()
    else:
        raise KeyError("Could not find mean initializer")

    div_key = next((k for k in inits if "Div" in k), None)
    if div_key:
        std = torch.clamp(torch.from_numpy(inits[div_key]).float() - 1e-4, min=1e-4)
    elif "std" in inits:
        std = torch.from_numpy(inits["std"]).float()
    elif "obs_normalizer._std" in inits:
        std = torch.from_numpy(inits["obs_normalizer._std"]).float()
    else:
        std = torch.ones_like(mean)
    return mean, std


class UnifiedResidualModel(nn.Module):
    def __init__(self, base_onnx_path, residual_onnx_path, alpha=None):
        super().__init__()
        if alpha is None:
            # 14D authority vector matching Approach B
            alpha = (
                0.10, 0.10, 0.10, 0.10, 0.10,  # left leg
                0.70, 0.70, 0.10, 0.60,        # head: neck_pitch, head_pitch, yaw, roll
                0.10, 0.10, 0.10, 0.10, 0.10,  # right leg
            )
        elif isinstance(alpha, (int, float)):
            alpha = [float(alpha)] * 14
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32).unsqueeze(0))

        # 1. Base policy
        base_model = onnx.load(base_onnx_path)
        b_inits = {init.name: np.copy(numpy_helper.to_array(init)) for init in base_model.graph.initializer}
        b_mean, b_std = extract_normalizer(b_inits)
        self.base_dim = int(b_mean.shape[-1])
        self.register_buffer("base_mean", b_mean)
        self.register_buffer("base_std", b_std)
        self.base_mlp = build_mlp_from_inits(b_inits)

        # 2. Residual policy
        res_model = onnx.load(residual_onnx_path)
        r_inits = {init.name: np.copy(numpy_helper.to_array(init)) for init in res_model.graph.initializer}
        r_mean, r_std = extract_normalizer(r_inits)
        self.res_dim = int(r_mean.shape[-1])
        self.register_buffer("res_mean", r_mean)
        self.register_buffer("res_std", r_std)
        self.res_mlp = build_mlp_from_inits(r_inits)

        self.input_dim = max(self.base_dim, self.res_dim)
        print(f"[UnifiedResidual] Base dim: {self.base_dim}, Res dim: {self.res_dim}, Total model input dim: {self.input_dim}")
        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Base forward
        b_obs = obs[:, :self.base_dim]
        b_norm = torch.clamp((b_obs - self.base_mean) / self.base_std, -5.0, 5.0)
        a_base = self.base_mlp(b_norm)

        # Residual forward
        r_obs = obs[:, :self.res_dim]
        r_norm = torch.clamp((r_obs - self.res_mean) / self.res_std, -5.0, 5.0)
        a_res = torch.clamp(self.res_mlp(r_norm), -1.0, 1.0)

        # Combined action
        return a_base + self.alpha * a_res


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export combined residual policy")
    parser.add_argument("--base-onnx", default="/Users/raghav/Desktop/microduck_manipulation/microduck_manipulation/policies/onnx/loadcarry_ppo_headup_v4.onnx")
    parser.add_argument("--res-onnx", default="/Users/raghav/Desktop/microduck_manipulation/microduck_manipulation/policies/onnx/table_tennis_racket_carry_residual.onnx")
    parser.add_argument("--out-onnx", default="/Users/raghav/Desktop/microduck_manipulation/microduck_manipulation/policies/onnx/table_tennis_racket_carry_residual_combined.onnx")
    parser.add_argument("--alpha-vector", nargs="+", type=float, default=None, help="14D authority vector")
    args = parser.parse_args()

    alpha = args.alpha_vector
    if alpha is None:
        alpha = (
            0.10, 0.10, 0.10, 0.10, 0.10,
            0.70, 0.70, 0.10, 0.60,
            0.10, 0.10, 0.10, 0.10, 0.10,
        )
    print(f"Building unified residual policy with authority vector: {alpha}")
    print(f"Base: {args.base_onnx}")
    print(f"Residual: {args.res_onnx}")
    model = UnifiedResidualModel(args.base_onnx, args.res_onnx, alpha=alpha)

    dummy_input = torch.randn(1, model.input_dim)
    with torch.no_grad():
        out = model(dummy_input)
    print("Output shape:", out.shape)

    torch.onnx.export(
        model,
        dummy_input,
        args.out_onnx,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported combined residual policy to: {args.out_onnx}")


if __name__ == "__main__":
    main()
