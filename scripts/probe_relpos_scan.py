"""Scan jaw_soft-local -z offset for a penetration-free welded rod pose."""
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".external_repo" / "src"))

from mjlab_microduck.robot.microduck_constants import (
    get_standup_spec, get_cylinder_spec, CYLINDER_WELD_RELPOS, CYLINDER_WELD_RELQUAT,
)
from mjlab_microduck.tasks.microduck_loadcarry_env_cfg import _weld_cylinder_to_jaw

base = mujoco.MjSpec()
frame_r = base.worldbody.add_frame()
base.attach(get_standup_spec(), prefix="robot/", frame=frame_r)
frame_c = base.worldbody.add_frame()
base.attach(get_cylinder_spec(), prefix="cylinder/", frame=frame_c)
_weld_cylinder_to_jaw(base)
model = base.compile()
data = mujoco.MjData(model)

HOME_JOINTS = {
    "left_hip_roll": -0.0873, "right_hip_roll": 0.0873,
    "left_hip_pitch": -0.4579, "right_hip_pitch": 0.4579,
    "left_knee": -0.0049, "right_knee": 0.0049,
    "left_ankle": 0.4530, "right_ankle": -0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491,
}
for name, val in HOME_JOINTS.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")
    data.qpos[model.jnt_qposadr[jid]] = val
root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/trunk_base_freejoint")
r_adr = model.jnt_qposadr[root_jid]
data.qpos[r_adr + 2] = 0.1880
data.qpos[r_adr + 3] = 1.0
mujoco.mj_forward(model, data)

jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/jaw_soft")
R = data.xmat[jaw_id].reshape(3, 3)
p_jaw = data.xpos[jaw_id]
q1 = data.xquat[jaw_id]
q2 = np.array(CYLINDER_WELD_RELQUAT)

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

cyl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder/cylinder_free")
c_adr = model.jnt_qposadr[cyl_jid]
cyl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder/cylinder")

x0, _, z0 = CYLINDER_WELD_RELPOS
print(f"base relpos: {x0:.6f}, 0, {z0:.6f}; shifting jaw-local -z (forward)")
for dz_mm in range(0, 61, 2):
    dz = dz_mm / 1000.0
    p_cyl = p_jaw + R @ np.array([x0, 0.0, z0 - dz])
    q_cyl = qmul(q1, q2)
    data.qpos[c_adr:c_adr+3] = p_cyl
    data.qpos[c_adr+3:c_adr+7] = q_cyl
    mujoco.mj_forward(model, data)
    worst = 0.0
    ncon = 0
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if "cylinder" in n1 or "cylinder" in n2:
            ncon += 1
            worst = min(worst, c.dist)
    print(f"  dz={dz_mm:3d} mm  contacts={ncon:2d}  worst={worst*1000:8.3f} mm  cyl z={data.xpos[cyl_id][2]:.4f}")
