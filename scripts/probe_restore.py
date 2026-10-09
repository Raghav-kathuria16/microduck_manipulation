"""Decisive probe for the robot-restore work.

A. Audit the REAL env path (make_microduck_loadcarry_env_cfg -> Scene.compile):
   does the weld equality carry our relpos/relquat, or defaults?
   (Audit only - fiction robot is deleted next; settle rehearsal is section C.)
B. Official-robot (robot_allcollisions.xml) FK at HOME:
   jaw_soft / mouth_tip pose, local axes, full joint+actuator inventory.
C. Spec-level weld rehearsal on the OFFICIAL robot: attach, weld to jaw_soft
   with constants derived in B, compile, settle under gravity, measure error.
"""
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".external_repo" / "src"))


def qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


print("=" * 30, "A. ENV-PATH EQ AUDIT", "=" * 30)
from mjlab.scene import Scene
from mjlab_microduck.tasks.microduck_loadcarry_env_cfg import (
    make_microduck_loadcarry_env_cfg,
)

cfg = make_microduck_loadcarry_env_cfg(play=True)
scene = Scene(cfg.scene, device="cpu")
model = scene.compile()
eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "cylinder_jaw_weld")
print(f"eq_id={eq_id}")
if eq_id >= 0:
    o1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj1id[eq_id])
    o2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj2id[eq_id])
    print(f"  obj1={o1} obj2={o2}")
    print("  FULL eq_data[0:11] =", np.array2string(model.eq_data[eq_id], precision=6))

print()
print("=" * 30, "B. OFFICIAL ROBOT FK AT HOME", "=" * 30)
OG_XML = ROOT / ".external_repo/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
rmodel = mujoco.MjModel.from_xml_path(str(OG_XML))
rdata = mujoco.MjData(rmodel)

print("-- joints (name, type, nq) --")
for j in range(rmodel.njnt):
    jname = mujoco.mj_id2name(rmodel, mujoco.mjtObj.mjOBJ_JOINT, j)
    jtype = int(rmodel.jnt_type[j])
    print(f"  [{j:2d}] {jname:28s} type={jtype}")
print(f"total joints={rmodel.njnt}  actuators={rmodel.nu}")
print("-- actuators --")
for a in range(rmodel.nu):
    print(f"  [{a:2d}] {mujoco.mj_id2name(rmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, a)}")

HOME_JOINTS = {
    "left_hip_roll": -0.0873, "right_hip_roll": 0.0873,
    "left_hip_pitch": -0.4579, "right_hip_pitch": 0.4579,
    "left_knee": -0.0049, "right_knee": 0.0049,
    "left_ankle": 0.4530, "right_ankle": -0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491,
    "head_yaw": 0.0, "head_roll": 0.0,
}
for name, val in HOME_JOINTS.items():
    jid = mujoco.mj_name2id(rmodel, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        print(f"  !! HOME joint missing in official XML: {name}")
        continue
    adr = rmodel.jnt_qposadr[jid]
    rdata.qpos[adr] = val
free_j = None
for j in range(rmodel.njnt):
    if rmodel.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        free_j = j
        break
if free_j is not None:
    adr = rmodel.jnt_qposadr[free_j]
    rdata.qpos[adr + 2] = 0.1880
    rdata.qpos[adr + 3:adr + 7] = (1.0, 0, 0, 0)
    print(f"root free joint: {mujoco.mj_id2name(rmodel, mujoco.mjtObj.mjOBJ_JOINT, free_j)}")
mujoco.mj_forward(rmodel, rdata)

js = mujoco.mj_name2id(rmodel, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
tip = mujoco.mj_name2id(rmodel, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
print(f"jaw_soft id={js}  mouth_tip id={tip}")
p_js = rdata.xpos[js].copy()
R_js = rdata.xmat[js].reshape(3, 3).copy()
print("jaw_soft  xpos =", np.array2string(p_js, precision=5))
print("jaw_soft  xmat cols (local x,y,z in world):")
for c in range(3):
    print(f"    axis{c} =", np.array2string(R_js[:, c], precision=4))
if tip >= 0:
    p_tip = rdata.site_xpos[tip].copy()
    print("mouth_tip xpos =", np.array2string(p_tip, precision=5))
else:
    print("!! mouth_tip site NOT FOUND; sites:", [
        mujoco.mj_id2name(rmodel, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(rmodel.nsite)])
    p_tip = p_js

relpos = R_js.T @ (p_tip - p_js)
lat_js = R_js.T @ np.array([0.0, 1.0, 0.0])
print("relpos (js frame)      =", np.array2string(relpos, precision=6))
print("lateral axis (js frame)=", np.array2string(lat_js, precision=6))
z = np.array([0.0, 0.0, 1.0])
v = np.cross(z, lat_js)
s = np.linalg.norm(v)
c = float(z @ lat_js)
if s < 1e-12:
    q_rel = np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1.0, 0, 0])
else:
    q_rel = np.concatenate([[1.0 + c], v / s])
    q_rel /= np.linalg.norm(q_rel)
print("q_rel (w,x,y,z)        =", np.array2string(q_rel, precision=6))
q_js = rdata.xquat[js].copy()
q_target_world = qmul(q_js, q_rel)
print("cyl world quat target  =", np.array2string(q_target_world, precision=6))
print("cyl world pos target   =", np.array2string(p_tip, precision=6))

print()
print("=" * 30, "C. SPEC-LEVEL WELD REHEARSAL (OFFICIAL ROBOT)", "=" * 30)
from mjlab_microduck.robot.microduck_constants import (
    get_standup_spec, get_cylinder_spec,
    CYLINDER_WELD_RELPOS, CYLINDER_WELD_RELQUAT,
)
relpos = np.array(CYLINDER_WELD_RELPOS)
q_rel = np.array(CYLINDER_WELD_RELQUAT)
base = mujoco.MjSpec()
frame_r = base.worldbody.add_frame()
base.attach(get_standup_spec(), prefix="robot/", frame=frame_r)
frame_c = base.worldbody.add_frame()
base.attach(get_cylinder_spec(), prefix="cylinder/", frame=frame_c)
eq = base.add_equality()
eq.type = mujoco.mjtEq.mjEQ_WELD
eq.name = "cylinder_jaw_weld"
eq.objtype = mujoco.mjtObj.mjOBJ_BODY
eq.name1 = "robot/jaw_soft"
eq.name2 = "cylinder/cylinder"
eq_data = np.zeros(11)
eq_data[3:6] = relpos
eq_data[6:10] = q_rel
eq_data[10] = 1.0
eq.data = eq_data
eq.solref = np.array([0.004, 1.0])
eq.solimp = np.array([0.95, 0.9999, 1e-5, 0.5, 2.0])
wmodel = base.compile()
wdata = mujoco.MjData(wmodel)
wid = mujoco.mj_name2id(wmodel, mujoco.mjtObj.mjOBJ_EQUALITY, "cylinder_jaw_weld")
print("rehearsal eq_data[3:11] =", np.array2string(wmodel.eq_data[wid][3:11], precision=6))
assert np.allclose(wmodel.eq_data[wid][3:6], relpos, atol=1e-9), "relpos lost at spec level!"
assert np.allclose(wmodel.eq_data[wid][6:10], q_rel, atol=1e-9), "relquat lost at spec level!"
print("  spec-level eq_data preserved OK")

for name, val in HOME_JOINTS.items():
    jid = mujoco.mj_name2id(wmodel, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")
    if jid >= 0:
        wdata.qpos[wmodel.jnt_qposadr[jid]] = val
wf = None
for j in range(wmodel.njnt):
    if wmodel.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and "robot/" in (
        mujoco.mj_id2name(wmodel, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
    ):
        wf = j
        break
if wf is not None:
    adr = wmodel.jnt_qposadr[wf]
    wdata.qpos[adr + 2] = 0.1880
    wdata.qpos[adr + 3:adr + 7] = (1.0, 0, 0, 0)
b1 = wmodel.eq_obj1id[wid]
b2 = wmodel.eq_obj2id[wid]
R1 = wdata.xmat[b1].reshape(3, 3)
p1 = wdata.xpos[b1]
q1 = wdata.xquat[b1]
cyl_jid = mujoco.mj_name2id(wmodel, mujoco.mjtObj.mjOBJ_JOINT, "cylinder/cylinder_free")
cadr = wmodel.jnt_qposadr[cyl_jid]
wdata.qpos[cadr:cadr + 3] = p1 + R1 @ relpos
wdata.qpos[cadr + 3:cadr + 7] = qmul(q1, q_rel)
mujoco.mj_forward(wmodel, wdata)
wmodel.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
for _ in range(400):
    mujoco.mj_step(wmodel, wdata)
R1 = wdata.xmat[b1].reshape(3, 3)
settled_relpos = R1.T @ (wdata.xpos[b2] - wdata.xpos[b1])
settled_relquat = qmul(qconj(wdata.xquat[b1]), wdata.xquat[b2])
err_p = np.linalg.norm(settled_relpos - relpos)
err_q = min(np.linalg.norm(settled_relquat - q_rel), np.linalg.norm(settled_relquat + q_rel))
print(f"settled relpos err = {err_p * 1000:.3f} mm   relquat err = {err_q:.5f}")
print("settled relpos =", np.array2string(settled_relpos, precision=5))
if wdata.ncon:
    min_dist = min(wdata.contact[c].dist for c in range(wdata.ncon))
    print(f"min contact dist after settle = {min_dist}")
else:
    print("no contacts after settle")
print("PROBE DONE")
