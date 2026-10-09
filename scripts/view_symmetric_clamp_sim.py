#!/usr/bin/env python3
"""Interactive 3D Viewer for Step 1 Symmetric Clamping & Deformation.

Directly responds to user request:
"give me commmand to run in terminal to see this in sim. sum up everything done untill now"

Launch with:
    /Users/raghav/Desktop/microduck_manipulation/microduck_rl/.venv/bin/mjpython microduck_manipulation/scripts/view_symmetric_clamp_sim.py
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(_MANIP_DIR)
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")

XML_PATH = os.path.join(
    _EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_grocery_bag.xml"
)

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    beak_jid = model.joint("beak").id
    neck_jid = model.joint("neck_pitch").id
    head_jid = model.joint("head_pitch").id
    model.jnt_range[beak_jid, 0] = 1.25

    # Realistic calibrated friction 0.50
    for gname in ["geom_strap_apex", "soft_mouth_top_collision", "jaw_soft_lower_collision", "beak_lower_collision"]:
        try:
            model.geom_friction[model.geom(gname).id] = [0.50, 0.05, 0.05]
        except Exception:
            pass
    for p in range(model.npair):
        model.pair_friction[p, :2] = 0.50

    bag_bid = model.body("grocery_bag").id
    target_mass = 0.150
    model.body_mass[bag_bid] = target_mass

    target_neck = np.radians(-74.0)
    target_head = np.radians(15.0)
    base_x_0 = 0.125

    neck_adr = model.jnt_qposadr[neck_jid]
    neck_dof = model.jnt_dofadr[neck_jid]
    head_adr = model.jnt_qposadr[head_jid]
    head_dof = model.jnt_dofadr[head_jid]
    beak_adr = model.jnt_qposadr[beak_jid]
    beak_dof = model.jnt_dofadr[beak_jid]
    b_adr = model.jnt_qposadr[model.joint("bag_free").id]
    apex_gid = model.geom("geom_strap_apex").id

    def reset_state():
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [base_x_0, 0.0, 0.1166]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[neck_adr] = target_neck
        data.qpos[head_adr] = target_head
        data.qpos[beak_adr] = 2.00 # fully open
        data.qpos[b_adr:b_adr + 3] = [0.21415, 0.0, 0.001]
        data.qpos[b_adr + 3:b_adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

    reset_state()

    print("=" * 75)
    print(" MicroDuck Step 1: Symmetric Clamping & ~20% Cord Deformation Viewer")
    print("=" * 75)
    print(" Controls:")
    print("  - Mouse Left-Click + Drag: Rotate 3D Camera")
    print("  - Mouse Right-Click + Drag: Pan Camera")
    print("  - Scroll Wheel: Zoom in / out")
    print("  - Spacebar: Pause / Resume simulation")
    print("  - Backspace: Reset to open jaws")
    print("=" * 75)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Default view: crisp side/lateral profile of beak and string
        viewer.cam.lookat = [0.214, 0.0, 0.125]
        viewer.cam.distance = 0.22
        viewer.cam.elevation = -4.0
        viewer.cam.azimuth = 90.0

        # Hide red collision boxes
        viewer.opt.geomgroup[0] = 1 # floor
        viewer.opt.geomgroup[1] = 1 # visual duck
        viewer.opt.geomgroup[2] = 1 # visual bag
        viewer.opt.geomgroup[3] = 0 # hide collision boxes

        phase_step = 0
        cycle_length = 200 # loop every ~4 seconds

        while viewer.is_running():
            step_start = time.time()

            # Phase timeline:
            # 0..40: Hold fully open (inspect string centered inside open jaws)
            # 40..120: Dynamic symmetric convergence (both jaws close together)
            # 120..190: Hold 20% deformation squeeze (inspect balanced clamp)
            # 190..200: Quick reset to open jaws
            if phase_step < 40:
                bt = 2.00
                dx = 0.0
            elif phase_step < 120:
                alpha = (phase_step - 40) / 80.0
                s = 0.5 * (1.0 - np.cos(alpha * np.pi))
                bt = 2.00 - s * (2.00 - 1.48)
                dx = -s * 0.0072
            elif phase_step < 190:
                bt = 1.48
                dx = -0.0072
            else:
                reset_state()
                phase_step = 0
                continue

            data.qpos[0] = base_x_0 + dx
            data.qpos[1] = 0.0
            data.qpos[2] = 0.1166
            data.qpos[3:7] = [1, 0, 0, 0]
            data.qvel[0:6] = 0.0

            data.qfrc_applied[neck_dof] = -30.0 * (data.qpos[neck_adr] - target_neck) - 1.0 * data.qvel[neck_dof]
            data.qfrc_applied[head_dof] = -20.0 * (data.qpos[head_adr] - target_head) - 0.8 * data.qvel[head_dof]
            data.qfrc_applied[beak_dof] = -18.0 * (data.qpos[beak_adr] - bt) - 0.25 * data.qvel[beak_dof]

            mujoco.mj_step(model, data)
            viewer.sync()

            phase_step += 1

            # Regulate to real-time (50 Hz update rate)
            elapsed = time.time() - step_start
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)

if __name__ == "__main__":
    main()
