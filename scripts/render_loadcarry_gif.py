#!/usr/bin/env python3
"""Generate an animated GIF of the trained loadcarry policy in native MuJoCo."""

import math
import os
import sys
import numpy as np
import mujoco
from PIL import Image
import imageio

_MANIP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))

from infer_policy import PolicyInference, DEFAULT_POSE

CYLINDER_WELD_RELPOS = (-0.013, 0.0, -0.068)
CYLINDER_WELD_RELQUAT = (0.7071068, 0.7071068, 0.0, 0.0)

def main():
    policy_path = "/Users/raghav/Desktop/microduck_manipulation/microduck_manipulation/policies/onnx/loadcarry_finetuned_150.onnx"
    model_path = os.path.join(_EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_loadcarry.xml")
    out_gif = "/Users/raghav/.gemini/antigravity-ide/brain/f2ffb849-d730-4892-8213-f5bdaeddc8dc/trained_loadcarry_native_mujoco.gif"

    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    cyl_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    jaw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    cyl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")

    if cyl_body_id >= 0:
        model.body_mass[cyl_body_id] = 0.30

    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=policy_path,
        action_scale=1.0,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )

    policy.set_vel_cmd(0.25, 0.0, 0.0)
    policy.head_offset[:] = 0.0
    policy._update_command()

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = DEFAULT_POSE[i]

    if beak_jid >= 0:
        data.qpos[model.jnt_qposadr[beak_jid]] = 0.52

    if cyl_jid >= 0:
        c_adr = model.jnt_qposadr[cyl_jid]
        mujoco.mj_forward(model, data)
        p_jaw = data.xpos[jaw_body_id].copy()
        R_jaw = data.xmat[jaw_body_id].reshape(3, 3).copy()
        data.qpos[c_adr:c_adr + 3] = p_jaw + R_jaw @ np.array(CYLINDER_WELD_RELPOS)
        q_jaw = np.empty(4)
        mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_body_id])
        q_cyl = np.empty(4)
        mujoco.mju_mulQuat(q_cyl, q_jaw, np.array(CYLINDER_WELD_RELQUAT))
        data.qpos[c_adr + 3:c_adr + 7] = q_cyl

    data.ctrl[:] = DEFAULT_POSE
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=360, width=480)
    camera_34 = mujoco.MjvCamera()
    camera_34.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera_34.trackbodyid = 1
    camera_34.distance = 0.85
    camera_34.elevation = -12.0
    camera_34.azimuth = 135.0

    frames = []
    print("Rendering 120 frames for animation...")
    for step in range(240):
        action = policy.infer()
        policy.apply_action(action)

        for _ in range(4):
            mujoco.mj_step(model, data)

        if step % 2 == 0:
            renderer.update_scene(data, camera=camera_34)
            rgb = renderer.render()
            frames.append(rgb)

    imageio.mimsave(out_gif, frames, fps=25, loop=0)
    print(f"Saved animation to {out_gif}")

if __name__ == "__main__":
    main()
