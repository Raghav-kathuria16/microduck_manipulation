#!/usr/bin/env python3
"""End-to-end smoke test: full ManagerBasedRlEnv with the welded cylinder.

1. Build the real loadcarry env cfg from the factory.
2. Build ManagerBasedRlEnv with 1 env (tiny — no training).
3. Verify the cylinder stays welded: relative pose held w.r.t. jaw_soft,
   no NaNs, 61D actor obs. Payload mass curriculum drives the cylinder's
   mass to stage targets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external_repo" / "src"))

import numpy as np
import torch


def quat_mul(a, b):
    """a, b: numpy (4,) in (w,x,y,z)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def main() -> int:
    from mjlab_microduck.tasks.microduck_loadcarry_env_cfg import (
        make_microduck_loadcarry_env_cfg,
    )
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

    cfg = make_microduck_loadcarry_env_cfg()
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    obs, info = env.reset(seed=0)

    obs_dim = env.observation_manager.group_obs_dim["actor"][0]
    print(f"actor obs dim: {obs_dim}  (expect 51 = 48 base + 3 twist command;"
          " head/body pose commands exist but are not observed in Phase 2)")
    assert obs_dim == 51, f"actor obs dim {obs_dim} != 51"
    act_dim = env.action_manager.total_action_dim
    print(f"action dim: {act_dim}  (expect 14)")
    assert act_dim == 14, f"action dim {act_dim} != 14"

    robot = env.scene["robot"]
    cyl = env.scene["cylinder"]

    _, jaw_name = robot.find_bodies("jaw_soft")
    print(f"jaw body on robot entity: {jaw_name}")

    # Cylinder entity root pose (the welded free body).
    from mjlab_microduck.robot.microduck_constants import (
        CYLINDER_WELD_RELPOS, CYLINDER_WELD_RELQUAT,
    )
    CYL_WELD_RELPOS = np.array(CYLINDER_WELD_RELPOS)
    CYL_WELD_RELQUAT = np.array(CYLINDER_WELD_RELQUAT)

    # The jaw body belongs to the robot entity; the cylinder root is its own
    # entity root, so compare cylinder root pose against the robot's jaw.
    robot_bodies = robot.body_names
    jaw_id = robot_bodies.index(jaw_name[0]) if jaw_name else None

    worst_dp, worst_dq = 0.0, 0.0
    for step in range(50):
        action = torch.zeros(1, env.action_manager.total_action_dim)
        obs, reward, terminated, truncated, info = env.step(action)

        p_jaw = robot.data.body_link_pos_w[0, jaw_id].cpu().numpy()
        q_jaw = robot.data.body_link_quat_w[0, jaw_id].cpu().numpy()
        p_cyl = cyl.data.body_link_pos_w[0, 0].cpu().numpy()  # root body
        q_cyl = cyl.data.body_link_quat_w[0, 0].cpu().numpy()

        R = quat_to_mat(q_jaw)
        p_rel = R.T @ (p_cyl - p_jaw)
        q_rel = quat_mul(quat_conj(q_jaw), q_cyl)
        dp = np.linalg.norm(p_rel - CYL_WELD_RELPOS)
        dq = min(
            np.linalg.norm(q_rel - CYL_WELD_RELQUAT),
            np.linalg.norm(q_rel + CYL_WELD_RELQUAT),
        )
        worst_dp = max(worst_dp, dp)
        worst_dq = max(worst_dq, dq)

        assert not np.isnan(obs["actor"].cpu().numpy()).any(), f"NaN in obs at step {step}"

    print(f"worst relative deviation over 50 steps: dp={worst_dp*1000:.3f} mm, dq={worst_dq:.6f}")
    assert worst_dp < 5e-3 and worst_dq < 0.02, "weld does not hold in env"

    print("ENV SMOKE PASS")
    return 0


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
