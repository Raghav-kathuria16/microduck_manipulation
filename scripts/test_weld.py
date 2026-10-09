#!/usr/bin/env python3
"""LoadCarry weld smoke test (official robot).

Spec level (CPU, no warp): compiles the merged scene spec via the factory's
spec_fn and checks the weld equality, its data, and that the cylinder body's
initial pose matches the welded relpose at HOME.
"""

import sys

sys.path.insert(0, ".external_repo/src")
import mujoco as _mujoco
import mujoco
import numpy as np


def main() -> int:
    from mjlab_microduck.tasks.microduck_loadcarry_env_cfg import (
        _weld_cylinder_to_jaw,
    )

    # -- 1. Spec-level: compile merged spec with the weld ----------------------
    from mjlab_microduck.robot.microduck_constants import (
        get_cylinder_spec,
        get_standup_spec,
        CYLINDER_WELD_RELPOS,
        CYLINDER_WELD_RELQUAT,
        CYLINDER_INIT_POS,
        CYLINDER_INIT_ROT,
    )

    # Mirror mjlab Scene._add_entities: every entity attached with "<name>/" prefix.
    base = _mujoco.MjSpec()
    robot_spec = get_standup_spec()
    cyl_spec = get_cylinder_spec()
    frame_r = base.worldbody.add_frame()
    base.attach(robot_spec, prefix="robot/", frame=frame_r)
    frame_c = base.worldbody.add_frame()
    base.attach(cyl_spec, prefix="cylinder/", frame=frame_c)
    spec = base
    _weld_cylinder_to_jaw(spec)
    model = spec.compile()

    # name lookup robust: use mujoco.mj_name2id
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "cylinder_jaw_weld")
    assert eq_id >= 0, "weld equality not found in compiled model"
    assert model.eq_type[eq_id] == mujoco.mjtEq.mjEQ_WELD
    d = model.eq_data[eq_id]
    np.testing.assert_allclose(d[3:6], CYLINDER_WELD_RELPOS, atol=1e-6)
    np.testing.assert_allclose(d[6:10], CYLINDER_WELD_RELQUAT, atol=1e-6)
    print(f"  weld eq present: id={eq_id}, objtype={model.eq_objtype[eq_id]}")

    # -- 2. Welded relpose holds under gravity + no penetration ----------------
    data = mujoco.MjData(model)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mujoco.mj_forward(model, data)
    # Apply HOME joint pose (entity keyframes don't survive attach; mjlab sets
    # them from InitialStateCfg at build time).
    HOME_JOINTS = {
        "left_hip_roll": -0.0873, "right_hip_roll": 0.0873,
        "left_hip_pitch": -0.4579, "right_hip_pitch": 0.4579,
        "left_knee": -0.0049, "right_knee": 0.0049,
        "left_ankle": 0.4530, "right_ankle": -0.4530,
        "neck_pitch": 0.3491, "head_pitch": 0.3491,
    }
    for name, val in HOME_JOINTS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = val
    # Trunk root at HOME standing height (qpos layout: [x, y, z, qw, qx, qy, qz]).
    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/trunk_base_freejoint")
    r_adr = model.jnt_qposadr[root_jid]
    data.qpos[r_adr + 2] = 0.1880  # HOME root height
    data.qpos[r_adr + 3] = 1.0  # identity quat
    # Place cylinder at the welded relpose (weld holds it there afterwards).
    jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/jaw_soft")
    q2 = np.array(CYLINDER_WELD_RELQUAT)
    R = data.xmat[jaw_id].reshape(3, 3)
    p_pred = data.xpos[jaw_id] + R @ np.array(CYLINDER_WELD_RELPOS)
    q1 = data.xquat[jaw_id]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q_pred = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    cyl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder/cylinder")
    cyl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder/cylinder_joint")
    if cyl_jid >= 0:
        c_adr = model.jnt_qposadr[cyl_jid]
        data.qpos[c_adr : c_adr + 3] = p_pred
        data.qpos[c_adr + 3 : c_adr + 7] = q_pred
        mujoco.mj_forward(model, data)
    # Step with gravity. The robot has no actuators here (mjlab injects BAM
    # actuators at build time), so it will crumple — that's fine: the weld's
    # contract is the RELATIVE pose, so compare cylinder pose against the
    # jaw_soft frame AT THE SAME timestep.
    for _ in range(200):
        mujoco.mj_step(model, data)
        p_act = data.xpos[cyl_id]
        q_act = data.xquat[cyl_id]
        R_now = data.xmat[jaw_id].reshape(3, 3)
        p_rel = R_now.T @ (p_act - data.xpos[jaw_id])
        w1, x1, y1, z1 = data.xquat[jaw_id]
        qw = data.xquat[jaw_id]  # alias
        # relative quat: q_rel = q_jaw^-1 (x) q_cyl
        q_inv = np.array([w1, -x1, -y1, -z1])
        W1, X1, Y1, Z1 = q_inv
        W2, X2, Y2, Z2 = q_act
        q_rel = np.array(
            [
                W1 * W2 - X1 * X2 - Y1 * Y2 - Z1 * Z2,
                W1 * X2 + X1 * W2 + Y1 * Z2 - Z1 * Y2,
                W1 * Y2 - X1 * Z2 + Y1 * W2 + Z1 * X2,
                W1 * Z2 + X1 * Y2 - Y1 * X2 + Z1 * W2,
            ]
        )
    dp_rel = np.linalg.norm(p_rel - np.array(CYLINDER_WELD_RELPOS))
    dq_rel = min(
        np.linalg.norm(q_rel - np.array(CYLINDER_WELD_RELQUAT)),
        np.linalg.norm(q_rel + np.array(CYLINDER_WELD_RELQUAT)),
    )
    print(f"  after 200 steps settle: dp_rel={dp_rel*1000:.3f} mm, dq_rel={dq_rel:.5f} (~rad)")
    assert dp_rel < 5e-3 and dq_rel < 0.02, "weld does not hold the relative relpose"

    # -- 3. Penetration check at HOME (welded pose, settled) -------------------
    mujoco.mj_forward(model, data)
    ncon_cyl = 0
    worst = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if "cylinder" in n1 or "cylinder" in n2:
            ncon_cyl += 1
            worst = max(worst, -c.dist)
    print(f"  contacts involving cylinder after settle: {ncon_cyl}, worst pen={worst*1000:.3f} mm")
    assert ncon_cyl == 0, "cylinder penetrates robot at welded pose, HOME"

    print("LOADCARRY WELD SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
