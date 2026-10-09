#!/usr/bin/env python3
"""Export Unified Precision Level Racket Carry Steadicam Policy to Standalone ONNX.

Architecture:
- Input: obs [batch, 64] (61D base locomotion observations + 3D racket blade normal in trunk base frame)
- Base Locomotion Subnetwork: Frozen loadcarry_ppo_headup_v4 MLP
- Steadicam Counter-Gimbal Head Layer:
    actions[:, 5] (neck_pitch) = + 1.25 * (pitch_err + 0.005)
    actions[:, 6] (head_pitch) = - 1.25 * (pitch_err + 0.005)
    actions[:, 7] (head_yaw)   = 0.0
    actions[:, 8] (head_roll)  = -0.002
    actions[:, legs]           = base_actions[:, legs]
- Output: actions [batch, 14]
"""

import os
import sys
import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn

_ROOT = "/Users/raghav/Desktop/microduck_manipulation"
sys.path.insert(0, os.path.join(_ROOT, "microduck_manipulation/scripts"))

from export_combined_residual import build_mlp_from_inits, extract_normalizer


class PrecisionLevelSteadicamPolicy(nn.Module):
    def __init__(self, base_onnx_path):
        super().__init__()
        # Load frozen base walking policy
        base_model = onnx.load(base_onnx_path)
        b_inits = {init.name: np.copy(numpy_helper.to_array(init)) for init in base_model.graph.initializer}
        b_mean, b_std = extract_normalizer(b_inits)
        self.base_dim = int(b_mean.shape[-1])
        self.register_buffer("base_mean", b_mean)
        self.register_buffer("base_std", b_std)
        self.base_mlp = build_mlp_from_inits(b_inits)

        # 64D total input (61D base obs + 3D racket normal in base frame)
        self.input_dim = 64
        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 1. Base walking locomotion
        b_obs = obs[:, :self.base_dim]
        b_norm = torch.clamp((b_obs - self.base_mean) / self.base_std, -5.0, 5.0)
        a_base = self.base_mlp(b_norm)

        # 2. Extract racket pitch error from racket normal in base frame
        # If input has >= 62 dimensions, obs[:, 61] contains normal_base[0] (forward tilt)
        batch_size = obs.shape[0]
        if obs.shape[1] >= 64:
            nx = obs[:, 61]
            pitch_err = torch.asin(torch.clamp(nx, -0.999, 0.999))
        else:
            pitch_err = torch.zeros(batch_size, device=obs.device)

        u_pitch = 1.25 * (pitch_err + 0.005)

        # 3. Construct unified actions
        # Leg actions: 0-4 (left leg) and 9-13 (right leg) pass through from base
        a_out = a_base.clone()
        a_out[:, 5] = u_pitch
        a_out[:, 6] = -u_pitch
        a_out[:, 7] = 0.0
        a_out[:, 8] = -0.002

        return a_out


def main():
    base_onnx = os.path.join(_ROOT, "microduck_manipulation/policies/onnx/loadcarry_ppo_headup_v4.onnx")
    out_onnx = os.path.join(_ROOT, "microduck_manipulation/policies/onnx/precision_level_racket_carry_steadicam.onnx")

    print(f"Loading base policy: {base_onnx}")
    model = PrecisionLevelSteadicamPolicy(base_onnx)

    dummy_input = torch.zeros(1, 64, dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Output shape: {out.shape} (Actions: {out.cpu().numpy().round(3)})")

    print(f"Exporting ONNX model to: {out_onnx}")
    torch.onnx.export(
        model,
        dummy_input,
        out_onnx,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=14,
    )

    # Validate ONNX model
    onnx_m = onnx.load(out_onnx)
    onnx.checker.check_model(onnx_m)
    print(f"Successfully validated exported ONNX model! File size: {os.path.getsize(out_onnx):,} bytes")


if __name__ == "__main__":
    main()
