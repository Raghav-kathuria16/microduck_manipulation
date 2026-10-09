#!/usr/bin/env python3
"""Generate the weld-carry demo scene (robot + cylinder + weld + keyframe).

Run:  python scripts/make_weld_demo.py
Out:  scenes/weld_cylinder_demo/weld_cylinder_demo.xml (+ copied assets)

Contents:
  * robot_manipulation.xml (official model + articulated orange beak)
  * cylinder: r=7mm half-length 75mm (Ø14 x 150 mm newspaper roll),
    axis lateral (world y) -> held horizontally in the beak
  * weld equality: duck_lower_beak -> cylinder, relpose solved so that at
    beak=0.5 rad the roll rests on the jaw rails with >=2.5mm clearance
  * keyframe "welder": trunk z=0.12 (feet on ground), beak=0.5, cylinder
    exactly at the weld target (starts on the constraint manifold)
"""
from pathlib import Path
import shutil

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT / ".external_repo" / "src" / "mjlab_microduck" / "robot" / "microduck"
ROBOT = REPO / "robot_manipulation.xml"
OUT = ROOT / "scenes" / "weld_cylinder_demo"

BEAK_ANGLE = 0.5          # rad, open mandible
CYL_R, CYL_HALF = 0.007, 0.075
CYL_MASS = 0.25
X_C, Z_C = 0.0828, 0.2106  # world pose of cyl center at keyframe (axis lateral)


def main():
    spec = mujoco.MjSpec.from_file(str(ROBOT))
    spec.modelname = "microduck_weld_demo"
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    # floor + light
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, name="floor",
                            size=[0, 0, 0.05], rgba=[0.3, 0.3, 0.35, 1])
    spec.worldbody.add_light(pos=[0.4, 0.4, 0.6], dir=[-0.4, -0.4, -0.6],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.8, 0.8, 0.8], specular=[0.2, 0.2, 0.2])

    # cylinder body
    cyl = spec.worldbody.add_body(name="demo_cyl", pos=[X_C, 0, Z_C])
    cyl.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="demo_cyl_free")
    cyl.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, name="cyl_geom",
                 size=[CYL_R, CYL_HALF], mass=CYL_MASS,
                 rgba=[0.72, 0.53, 0.35, 1], friction=[0.8, 0.005, 0.001])
    # cylinder axis -> world y: quat Rx(90deg) = (c45, s45, 0, 0)
    c45 = np.cos(np.pi / 4)
    CYL_QUAT = [c45, c45, 0.0, 0.0]

    # weld: relpose of cylinder in duck_lower_beak frame
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name = "beak_cyl_weld"
    eq.name1 = "duck_lower_beak"
    eq.name2 = "demo_cyl"

    eq.solref = [0.005, 1.0]
    m = spec.compile()
    d = mujoco.MjData(m)
    beak_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "beak_joint")
    dlb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "duck_lower_beak")
    d.qpos[:] = m.qpos0
    d.qpos[m.jnt_qposadr[beak_jid]] = BEAK_ANGLE
    d.qpos[2] = 0.12
    mujoco.mj_forward(m, d)
    p_dlb = d.xpos[dlb].copy()
    R_dlb = d.xmat[dlb].reshape(3, 3).copy()
    rel_pos = R_dlb.T @ (np.array([X_C, 0.0, Z_C]) - p_dlb)
    # rel quat: q_rel = conj(q_dlb) ⊗ q_cyl, so that q_dlb ⊗ q_rel = q_cyl (cyl axis -> world y)
    q_dlb = np.empty(4)
    mujoco.mju_mat2Quat(q_dlb, d.xmat[dlb])
    q_cyl = np.array(CYL_QUAT)
    rel_q = np.empty(4)
    mujoco.mju_mulQuat(rel_q, [q_dlb[0], -q_dlb[1], -q_dlb[2], -q_dlb[3]], q_cyl)
    eq.data = np.array([0, 0, 0, *rel_pos, *rel_q, 1.0])

    # recompile with weld data and build the keyframe
    m = spec.compile()
    d = mujoco.MjData(m)
    beak_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "beak_joint")
    cyl_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "demo_cyl_free")
    act_beak = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_beak")
    qadr = int(m.jnt_qposadr[cyl_jid])
    qpos = m.qpos0.copy()
    qpos[2] = 0.12
    qpos[m.jnt_qposadr[beak_jid]] = BEAK_ANGLE
    qpos[qadr:qadr + 3] = [X_C, 0.0, Z_C]
    qpos[qadr + 3:qadr + 7] = CYL_QUAT
    ctrl = np.zeros(m.nu)
    ctrl[act_beak] = BEAK_ANGLE
    key = spec.add_key()
    key.qpos = qpos
    key.ctrl = ctrl
    # write
    OUT.mkdir(parents=True, exist_ok=True)
    xml = spec.to_xml()
    (OUT / "weld_cylinder_demo.xml").write_text(xml)
    assets = spec.assets if hasattr(spec, "assets") else {}
    for fname, data in (assets or {}).items():
        p = OUT / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    # assets dict may be empty (STLs referenced from source dir); copy from REPO
    if not any((OUT / "assets").glob("*.stl")):
        src_assets = REPO / "assets"
        dst_assets = OUT / "assets"
        dst_assets.mkdir(parents=True, exist_ok=True)
        for stl in src_assets.glob("*.stl"):
            shutil.copy(stl, dst_assets / stl.name)
    print("relpose (dlb frame): pos=%s quat=%s" % (np.round(rel_pos, 5), np.round(rel_q, 5)))


if __name__ == "__main__":
    main()
