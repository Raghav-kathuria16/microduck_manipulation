#!/usr/bin/env python3
"""Evaluate Microduck Biped Clamping Spoon & Lemon Dynamic Balancing.

Architecture:
1. Spoon: Lightweight 30g stainless steel spoon clamped between upper beak and lower jaw.
2. Lemon: Real-world 55g fresh lemon (prolate ellipsoid) resting in the concave bowl.
3. Contact: Spatial contact dynamics.
4. Locomotion: Initialized with loadcarry_ppo_headup_v4 bipedal walking policy.
5. Orientation: Steadicam counter-phase pitch/roll stabilization keeps spoon level.
"""

import argparse
import math
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(_MANIP_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from infer_policy import PolicyInference, DEFAULT_POSE, TerminalInput

_BASE_SCENE_XML = os.path.join(_MANIP_DIR, "models/scene_loadcarry_physical_clamp.xml")
_PAYLOAD_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/payload_clamping_locomotion.onnx")
_DEFAULT_POLICY = _PAYLOAD_POLICY
_STEADICAM_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/spoon_lemon_steadicam_residual.onnx")
_OUTPUT_DIR = os.path.join(_MANIP_DIR, "docs/media")
_ARTIFACT_DIR = "/Users/raghav/.gemini/antigravity-ide/brain/01375344-3ce7-4a6f-afec-3cecca1c8b8d"
os.makedirs(_OUTPUT_DIR, exist_ok=True)
os.makedirs(_ARTIFACT_DIR, exist_ok=True)


def create_spoon_lemon_scene_xml(friction: float = 0.50):
    """Builds an MJCF string embedding the tableware spoon and fresh lemon with customizable friction."""
    # Clamp to positive epsilon (0.001) to avoid MuJoCo 6D spatial elliptic cone division-by-zero in solver
    friction = max(0.001, float(friction))
    with open(_BASE_SCENE_XML, "r") as f:
        base = f.read()

    assets_dir = os.path.join(_MANIP_DIR, "models/assets")
    import re
    base = re.sub(r'meshdir="[^"]*"', f'meshdir="{assets_dir}/" balanceinertia="true"', base)
    base = base.replace('elevation="-20"/>', 'elevation="-20" offwidth="1280" offheight="960"/>')

    extra_assets = """
    <!-- CAD Mesh for Tableware Spoon -->
    <mesh name="spoon_cad_mesh" file="spoon.stl"/>

    <material name="stainless_steel" rgba="0.84 0.86 0.90 1" specular="0.85" shininess="0.9" reflectance="0.3"/>
    <material name="citrus_lemon" rgba="0.98 0.88 0.12 1" specular="0.35" shininess="0.35"/>
    <material name="lemon_tip" rgba="0.72 0.82 0.18 1" specular="0.2" shininess="0.2"/>
  </asset>
"""
    base = base.replace("  </asset>", extra_assets)

    cyl_start = base.find('<body name="cylinder"')
    cyl_end = base.find('</body>\n  </worldbody>') + len('</body>')
    old_cyl = base[cyl_start:cyl_end]

    spoon_lemon_str = f"""    <!-- Tableware Spoon Body (25g stainless steel, length 125mm) -->
    <body name="spoon" pos="0.048 0.0 0.220" quat="0.7071068 0.7071068 0 0">
      <freejoint name="spoon_free"/>
      <inertial pos="0 -0.001 0.028" mass="0.025" diaginertia="0.000028 0.000024 0.000006"/>

      <!-- 100% Seamless CAD Mesh (Group 2 Visual, Zero Gaps) -->
      <geom name="spoon_vis" type="mesh" mesh="spoon_cad_mesh" material="stainless_steel" contype="0" conaffinity="0" group="2"/>

      <!-- Collision Model (group 3) -->
      <!-- Flat Handle for Beak Clamping: Length 75mm, Width 8.0mm, Thickness 2.2mm -->
      <geom name="spoon_handle" type="box" pos="0 0 0.008" size="0.0040 0.0011 0.035" material="stainless_steel" friction="0.55 0.05 0.05" contype="0" conaffinity="0" group="3"/>

      <!-- Concave Bowl Base Plate (Floor of the depression) -->
      <geom name="spoon_bowl_base" type="ellipsoid" pos="0 -0.0086 0.070" size="0.012 0.0010 0.018" material="stainless_steel" friction="{friction:.3f} 0.01 0.01" contype="0" conaffinity="0" group="3"/>

      <!-- Concave Bowl Rim Capsules (Curved perimeter wall cradling lemon) -->
      <geom name="rim_front" type="capsule" fromto="-0.007 -0.0020 0.088  0.007 -0.0020 0.088" size="0.0018" material="stainless_steel" friction="{friction:.3f} 0.01 0.01" contype="0" conaffinity="0" group="3"/>
      <geom name="rim_back"  type="capsule" fromto="-0.006 -0.0022 0.052  0.006 -0.0022 0.052" size="0.0018" material="stainless_steel" friction="{friction:.3f} 0.01 0.01" contype="0" conaffinity="0" group="3"/>
      <geom name="rim_left_m"  type="capsule" fromto="-0.0135 -0.0022 0.070 -0.007 -0.0020 0.088" size="0.0018" material="stainless_steel" friction="{friction:.3f} 0.01 0.01" contype="0" conaffinity="0" group="3"/>
      <geom name="rim_right_m" type="capsule" fromto=" 0.0135 -0.0020 0.070  0.007 -0.0020 0.088" size="0.0018" material="stainless_steel" friction="{friction:.3f} 0.01 0.01" contype="0" conaffinity="0" group="3"/>
      <geom name="rim_left_b"  type="capsule" fromto="-0.006 -0.0022 0.052 -0.0135 -0.0022 0.070" size="0.0018" material="stainless_steel" friction="{friction:.3f} 0.01 0.01" contype="0" conaffinity="0" group="3"/>
      <geom name="rim_right_b" type="capsule" fromto=" 0.006 -0.0022 0.052  0.0135 -0.0020 0.070" size="0.0018" material="stainless_steel" friction="{friction:.3f} 0.01 0.01" contype="0" conaffinity="0" group="3"/>
    </body>

    <!-- Smooth Spherical Lemon Ball (22g pure sphere, 24mm dia, r=0.012m, no tips/nubs) -->
    <body name="lemon" pos="0.050 0.065 0.220">
      <freejoint name="lemon_free"/>
      <inertial pos="0 0 0" mass="0.022" diaginertia="0.00000127 0.00000127 0.00000127"/>
      <!-- Pure smooth spherical ball with user-configurable friction -->
      <geom name="lemon_geom" type="sphere" size="0.012" material="citrus_lemon" friction="{friction:.3f} 0.01 0.01" condim="6"/>
    </body>"""

    xml = base.replace(old_cyl, spoon_lemon_str)
    xml = xml.replace('geom1="cylinder_geom"', 'geom1="spoon_handle"')
    xml = xml.replace('body1="cylinder"', 'body1="spoon"')

    # 1. Body Exclusions to prevent hinge mesh overlap
    xml = xml.replace('<exclude body1="cylinder" body2="trunk_base"/>',
                      '<exclude body1="spoon" body2="trunk_base"/>\n'
                      '    <exclude body1="neck" body2="beak_lower"/>\n'
                      '    <exclude body1="neck_pitch" body2="beak_lower"/>\n'
                      '    <exclude body1="jaw_soft" body2="beak_lower"/>\n'
                      '    <exclude body1="lemon" body2="trunk_base"/>')

    # 2. Hard limits on beak joint (prevents over-rotation)
    xml = xml.replace('<joint name="beak" pos="0 0 0" axis="0 1 0" range="1.47 2.05" ref="1.500" damping="0.05"/>',
                      '<joint name="beak" pos="0 0 0" axis="0 1 0" range="1.47 2.05" ref="1.500" damping="0.05" limited="true"/>')

    # Compliant silicone mouth lining parameters
    xml = xml.replace('solref="0.004 1.0"', 'solref="0.002 1.0"')
    xml = xml.replace('solimp="0.95 0.99 0.001 0.5 2"', 'solimp="0.98 0.995 0.0005 0.5 2"')

    # Silicone pads for beak
    roof_geom_str = """              <!-- Compliant Silicone Clamping Bed lining Upper Beak Ceiling -->
              <geom name="mouth_roof_pad" type="box" pos="-0.0080 0 -0.0650" size="0.0010 0.025 0.0125" contype="0" conaffinity="0" group="3" material="soft_mouth_top_material"/>
"""
    xml = xml.replace('              <geom name="soft_mouth_top_collision"', roof_geom_str + '              <geom name="soft_mouth_top_collision"')

    jaw_cradle_str = """                <!-- Conformal lower jaw silicone bed -->
                <geom name="jaw_cradle_bed" type="box" pos="-0.0130 0 -0.0465" size="0.0010 0.025 0.0125" contype="0" conaffinity="0" group="3" material="jaw_soft_material"/>
"""
    xml = xml.replace('                <geom name="jaw_soft_lower_collision"', jaw_cradle_str + '                <geom name="jaw_soft_lower_collision"')

    # Contact pairs with spatial contact parameters
    contact_str = f"""    <!-- Spatial contact pairs for spoon and lemon (friction = {friction:.3f}) -->
    <pair geom1="spoon_handle" geom2="mouth_roof_pad" condim="6" friction="0.55 0.55 0.05 0.05 0.05" solref="0.002 1.0" solimp="0.99 0.999 0.0001 0.5 2"/>
    <pair geom1="spoon_handle" geom2="jaw_cradle_bed" condim="6" friction="0.55 0.55 0.05 0.05 0.05" solref="0.002 1.0" solimp="0.99 0.999 0.0001 0.5 2"/>
    <pair geom1="lemon_geom" geom2="spoon_bowl_base" condim="6" friction="{friction:.3f} {friction:.3f} 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="lemon_geom" geom2="rim_front" condim="6" friction="{friction:.3f} {friction:.3f} 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="lemon_geom" geom2="rim_back" condim="6" friction="{friction:.3f} {friction:.3f} 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="lemon_geom" geom2="rim_left_m" condim="6" friction="{friction:.3f} {friction:.3f} 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="lemon_geom" geom2="rim_right_m" condim="6" friction="{friction:.3f} {friction:.3f} 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="lemon_geom" geom2="rim_left_b" condim="6" friction="{friction:.3f} {friction:.3f} 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="lemon_geom" geom2="rim_right_b" condim="6" friction="{friction:.3f} {friction:.3f} 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
"""
    xml = xml.replace('  </contact>', contact_str + '  </contact>')

    # High-traction groundplane and rubber sole contact parameters for stable bipedal stride
    xml = xml.replace('<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>',
                      '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.4 0.01 0.001"/>')
    xml = xml.replace('name="left_foot_collision" class="collision"',
                      'name="left_foot_collision" class="collision" friction="1.4 0.01 0.001"')
    xml = xml.replace('name="right_foot_collision" class="collision"',
                      'name="right_foot_collision" class="collision" friction="1.4 0.01 0.001"')

    # Visual athletics track & distance meter markers (Group 1 visual, non-colliding)
    track_markers = """    <!-- Visual Athletics Track & Distance Markers (Group 1 Visual, Non-Colliding) -->
    <body name="track_markers" pos="0 0 0.0005">
      <!-- Start Line (Bright Green, x=0.0m) -->
      <geom name="start_line" type="box" pos="0 0 0" size="0.02 0.35 0.0005" rgba="0.1 0.85 0.2 0.9" contype="0" conaffinity="0" group="1"/>
      <!-- Distance Markers every 0.25m from x=0.25 to x=2.50 -->
      <geom name="mark_025" type="box" pos="0.25 0 0" size="0.01 0.28 0.0005" rgba="1.0 1.0 1.0 0.8" contype="0" conaffinity="0" group="1"/>
      <geom name="mark_050" type="box" pos="0.50 0 0" size="0.015 0.35 0.0005" rgba="0.95 0.85 0.1 0.9" contype="0" conaffinity="0" group="1"/>
      <geom name="mark_075" type="box" pos="0.75 0 0" size="0.01 0.28 0.0005" rgba="1.0 1.0 1.0 0.8" contype="0" conaffinity="0" group="1"/>
      <geom name="mark_100" type="box" pos="1.00 0 0" size="0.015 0.35 0.0005" rgba="0.95 0.85 0.1 0.9" contype="0" conaffinity="0" group="1"/>
      <geom name="mark_125" type="box" pos="1.25 0 0" size="0.01 0.28 0.0005" rgba="1.0 1.0 1.0 0.8" contype="0" conaffinity="0" group="1"/>
      <geom name="mark_150" type="box" pos="1.50 0 0" size="0.015 0.35 0.0005" rgba="0.95 0.85 0.1 0.9" contype="0" conaffinity="0" group="1"/>
      <geom name="mark_175" type="box" pos="1.75 0 0" size="0.01 0.28 0.0005" rgba="1.0 1.0 1.0 0.8" contype="0" conaffinity="0" group="1"/>
      <geom name="mark_200" type="box" pos="2.00 0 0" size="0.015 0.35 0.0005" rgba="0.95 0.85 0.1 0.9" contype="0" conaffinity="0" group="1"/>
      <geom name="mark_250" type="box" pos="2.50 0 0" size="0.02 0.35 0.0005" rgba="0.95 0.2 0.1 0.9" contype="0" conaffinity="0" group="1"/>
      <!-- Track boundary rails -->
      <geom name="rail_left"  type="box" pos="1.25  0.35 0" size="1.3 0.006 0.0005" rgba="0.2 0.7 0.9 0.7" contype="0" conaffinity="0" group="1"/>
      <geom name="rail_right" type="box" pos="1.25 -0.35 0" size="1.3 0.006 0.0005" rgba="0.2 0.7 0.9 0.7" contype="0" conaffinity="0" group="1"/>
    </body>
"""
    xml = xml.replace('  </worldbody>', track_markers + '  </worldbody>')
    return xml


def create_policy_with_steadicam(model, data, policy_path):
    policy = PolicyInference(
        model, data,
        walking_onnx_path=policy_path,
        action_scale=1.0,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )
    policy.default_pose[5] = 0.2760
    policy.default_pose[6] = 0.2760
    policy.default_pose[7] = 0.0
    policy.default_pose[8] = 0.0

    is_rl = False
    if policy.walking_session and policy.walking_session.get_inputs()[0].shape[1] == 67:
        is_rl = True
        spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
        lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lemon")
        spoon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spoon_free")
        lemon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lemon_free")
        lemon_dof = model.jnt_dofadr[lemon_jid]
        spoon_dof = model.jnt_dofadr[spoon_jid]

        def compute_extra_obs():
            R_s = data.xmat[spoon_bid].reshape(3, 3)
            p_diff = data.xpos[lemon_bid] - data.xpos[spoon_bid]
            p_rel = R_s.T @ p_diff
            dx = float(p_rel[0])
            dz = float(p_rel[2] - 0.070)

            v_diff = data.qvel[lemon_dof:lemon_dof+3] - data.qvel[spoon_dof:spoon_dof+3]
            v_rel = R_s.T @ v_diff
            vx_rel = float(v_rel[0])
            vz_rel = float(v_rel[2])

            z_local = R_s.T @ np.array([0.0, 0.0, 1.0])
            pitch_err = float(np.arcsin(np.clip(z_local[2], -0.999, 0.999)))
            roll_err = float(np.arcsin(np.clip(z_local[0], -0.999, 0.999)))

            return np.array([dx, dz, vx_rel, vz_rel, pitch_err, roll_err], dtype=np.float32)

        policy.extra_obs_fn = compute_extra_obs
        print("[Steadicam RL] Detected 67D trained Residual Steadicam RL policy. Activating closed-loop payload feedback!")

    return policy, is_rl


def render_visual_proofs(num_settle_steps=200, policy_path=None, friction=0.50):
    """Renders high-resolution visual proof images from 4 distinct camera views:
    1. Full Robot Perspective View (showing 3D balance and lemon in spoon)
    2. Side Profile View (showing spoon held horizontally parallel to ground)
    3. Close-Up View of Beak Clamping Spoon Handle
    4. Close-Up View of Lemon Cradled in Concave Spoon Bowl
    """
    if policy_path is None:
        policy_path = _DEFAULT_POLICY
    xml_str = create_spoon_lemon_scene_xml(friction=friction)
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    policy, is_rl = create_policy_with_steadicam(model, data, policy_path)

    spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lemon")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_qpos = model.jnt_qposadr[beak_jid]
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    spoon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spoon_free")
    spoon_qadr = model.jnt_qposadr[spoon_jid]
    lemon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lemon_free")
    lemon_qadr = model.jnt_qposadr[lemon_jid]

    # Reset to standing equilibrium
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]

    # Clamp beak around spoon handle
    target_clamp_angle = 1.550  # rad (~88.8 deg, firm silicone grip on 2.2mm flat handle)
    data.qpos[beak_qpos] = target_clamp_angle
    data.qfrc_applied[beak_dof] = -0.45  # N*m clamping torque
    if model.nu > 14:
        data.ctrl[14] = 1.0
    mujoco.mj_forward(model, data)

    p_jaw = data.xpos[jaw_bid]
    R_jaw = data.xmat[jaw_bid].reshape(3, 3)
    HOVER_LOCAL = np.array([-0.0119, 0.0, -0.0465])
    CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

    data.qpos[spoon_qadr:spoon_qadr+3] = p_jaw + R_jaw @ HOVER_LOCAL
    q_jaw = np.empty(4)
    mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
    q_s = np.empty(4)
    mujoco.mju_mulQuat(q_s, q_jaw, CYL_QUAT)
    data.qpos[spoon_qadr+3:spoon_qadr+7] = q_s
    mujoco.mj_forward(model, data)

    # Place lemon directly inside concave bowl (bowl center at z=+0.070, y=-0.0022)
    p_spoon = data.xpos[spoon_bid]
    R_spoon = data.xmat[spoon_bid].reshape(3, 3)
    p_lemon = p_spoon + R_spoon @ np.array([0.0, 0.0052, 0.070])
    data.qpos[lemon_qadr:lemon_qadr+3] = p_lemon
    data.qpos[lemon_qadr+3:lemon_qadr+7] = q_s
    mujoco.mj_forward(model, data)

    # Step physics to let lemon settle in bowl and beak clamp firmly (holding robot base erect)
    print(f"Settling physics for {num_settle_steps} steps...")
    lemon_dof = model.jnt_dofadr[lemon_jid]
    for step in range(num_settle_steps):
        data.qpos[0:3] = [0.0, 0.0, 0.125]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[0:6] = 0.0
        data.qfrc_applied[beak_dof] = -0.45
        # Soft settling damping for lemon
        data.qvel[lemon_dof:lemon_dof+6] *= 0.98
        mujoco.mj_step(model, data)

    # Calculate contacts
    fn_clamp = 0.0
    fn_lemon = 0.0
    for ci in range(data.ncon):
        con = data.contact[ci]
        b1 = model.geom_bodyid[con.geom1]
        b2 = model.geom_bodyid[con.geom2]
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, ci, f)
        if (b1 == lemon_bid and b2 == spoon_bid) or (b2 == lemon_bid and b1 == spoon_bid):
            fn_lemon += f[0]
        elif (b1 == spoon_bid or b2 == spoon_bid) and (b1 == jaw_bid or b2 == jaw_bid or "mouth" in model.geom(con.geom1).name or "mouth" in model.geom(con.geom2).name or "cradle" in model.geom(con.geom1).name or "cradle" in model.geom(con.geom2).name):
            fn_clamp += f[0]

    # Calculate spoon orientation
    R_s = data.xmat[spoon_bid].reshape(3, 3)
    spoon_fwd = R_s @ np.array([0.0, 0.0, 1.0])
    spoon_norm = R_s @ np.array([0.0, 1.0, 0.0])
    spoon_pitch = math.degrees(math.asin(np.clip(spoon_fwd[2], -0.999, 0.999)))
    spoon_roll = math.degrees(math.asin(np.clip(spoon_norm[1], -0.999, 0.999)))

    print("=" * 60)
    print("Spoon & Lemon Static Equilibrium Results:")
    print(f"  Clamping Normal Force: {fn_clamp:.2f} N")
    print(f"  Lemon Contact Force:   {fn_lemon:.2f} N (Lemon Weight: {0.022*9.81:.2f} N)")
    print(f"  Spoon Pitch (Fore-Aft): {spoon_pitch:+.2f}° (Target: 0.0° level)")
    print(f"  Spoon Roll (Lateral):  {spoon_roll:+.2f}° (Target: 0.0° level)")
    print("=" * 60)

    # Render proof images with Group 3 (collision) hidden for clean CAD visual rendering
    renderer = mujoco.Renderer(model, width=1280, height=960)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[3] = 0  # Hide collision geoms

    # 1. Perspective View
    cam_persp = mujoco.MjvCamera()
    cam_persp.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam_persp.trackbodyid = 1
    cam_persp.distance = 0.65
    cam_persp.elevation = -18.0
    cam_persp.azimuth = 142.0
    renderer.update_scene(data, camera=cam_persp, scene_option=vopt)
    p_persp = os.path.join(_ARTIFACT_DIR, "spoon_lemon_perspective_proof.png")
    Image.fromarray(renderer.render()).save(p_persp)

    # 2. Side View
    cam_side = mujoco.MjvCamera()
    cam_side.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam_side.trackbodyid = 1
    cam_side.distance = 0.58
    cam_side.elevation = -4.0
    cam_side.azimuth = 90.0
    renderer.update_scene(data, camera=cam_side, scene_option=vopt)
    p_side = os.path.join(_ARTIFACT_DIR, "spoon_lemon_side_level_proof.png")
    Image.fromarray(renderer.render()).save(p_side)

    # 3. Close-Up View of Beak Clamping
    cam_beak = mujoco.MjvCamera()
    cam_beak.type = mujoco.mjtCamera.mjCAMERA_FREE
    p_beak = data.xpos[jaw_bid]
    cam_beak.lookat = [p_beak[0] + 0.02, p_beak[1], p_beak[2] + 0.02]
    cam_beak.distance = 0.22
    cam_beak.elevation = -12.0
    cam_beak.azimuth = 120.0
    renderer.update_scene(data, camera=cam_beak, scene_option=vopt)
    p_beak_img = os.path.join(_ARTIFACT_DIR, "spoon_lemon_beak_clamp_closeup.png")
    Image.fromarray(renderer.render()).save(p_beak_img)

    # 4. Close-Up View of Lemon in Concave Spoon Bowl
    cam_bowl = mujoco.MjvCamera()
    cam_bowl.type = mujoco.mjtCamera.mjCAMERA_FREE
    p_lemon = data.xpos[lemon_bid]
    cam_bowl.lookat = [p_lemon[0], p_lemon[1], p_lemon[2]]
    cam_bowl.distance = 0.18
    cam_bowl.elevation = -22.0
    cam_bowl.azimuth = 135.0
    renderer.update_scene(data, camera=cam_bowl, scene_option=vopt)
    p_bowl_img = os.path.join(_ARTIFACT_DIR, "spoon_lemon_bowl_cradle_closeup.png")
    Image.fromarray(renderer.render()).save(p_bowl_img)

    print("\nVisual proofs saved:")
    print(f"  Perspective: {p_persp}")
    print(f"  Side Level:  {p_side}")
    print(f"  Beak Clamp:  {p_beak_img}")
    print(f"  Bowl Cradle: {p_bowl_img}")
    return p_persp, p_side, p_beak_img, p_bowl_img


def render_walking_video(output_path=None, duration_sec=6.0, vx=0.30, fps=50, policy_path=None, friction=0.50):
    """Renders high-definition offscreen MP4 video of the Microduck robot
    clamping the spoon and balancing the lemon while walking forward.
    """
    import imageio
    if output_path is None:
        output_path = os.path.join(_ARTIFACT_DIR, "spoon_lemon_walking_demo.mp4")
    if policy_path is None:
        policy_path = _DEFAULT_POLICY

    print(f"\n[VIDEO] Rendering {duration_sec:.1f}s walking video at vx={vx:.2f} m/s (mu={friction:.2f}) to {output_path}...")
    xml_str = create_spoon_lemon_scene_xml(friction=friction)
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    policy, is_rl = create_policy_with_steadicam(model, data, policy_path)

    spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lemon")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_qpos = model.jnt_qposadr[beak_jid]
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    spoon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spoon_free")
    spoon_qadr = model.jnt_qposadr[spoon_jid]
    lemon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lemon_free")
    lemon_qadr = model.jnt_qposadr[lemon_jid]
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    target_clamp_angle = 1.550
    clamp_torque = 0.45

    # Reset with true ground contact initialization
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.1172]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]
    data.qpos[beak_qpos] = target_clamp_angle
    policy.set_position_targets(policy.default_pose)

    # Park spoon & lemon high in air during settling
    data.qpos[spoon_qadr:spoon_qadr+3] = [0.0, 0.0, 10.0]
    data.qpos[lemon_qadr:lemon_qadr+3] = [0.0, 0.0, 10.0]
    mujoco.mj_forward(model, data)

    # Settle duck on ground under active PD control
    for _ in range(100):
        policy.set_position_targets(policy.default_pose)
        beak_err = target_clamp_angle - data.qpos[beak_qpos]
        data.qfrc_applied[beak_dof] = float(np.clip(-clamp_torque + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))
        data.qpos[spoon_qadr:spoon_qadr+3] = [0.0, 0.0, 10.0]
        data.qpos[lemon_qadr:lemon_qadr+3] = [0.0, 0.0, 10.0]
        data.qvel[spoon_qadr:spoon_qadr+6] = 0.0
        data.qvel[lemon_qadr:lemon_qadr+6] = 0.0
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    p_jaw = data.xpos[jaw_bid]
    R_jaw = data.xmat[jaw_bid].reshape(3, 3)
    HOVER_LOCAL = np.array([-0.0119, 0.0, -0.0465])
    CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

    data.qpos[spoon_qadr:spoon_qadr+3] = p_jaw + R_jaw @ HOVER_LOCAL
    q_jaw = np.empty(4)
    mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
    q_s = np.empty(4)
    mujoco.mju_mulQuat(q_s, q_jaw, CYL_QUAT)
    data.qpos[spoon_qadr+3:spoon_qadr+7] = q_s
    mujoco.mj_forward(model, data)

    p_spoon = data.xpos[spoon_bid]
    R_spoon = data.xmat[spoon_bid].reshape(3, 3)
    p_lemon = p_spoon + R_spoon @ np.array([0.0, 0.0052, 0.070])
    data.qpos[lemon_qadr:lemon_qadr+3] = p_lemon
    data.qpos[lemon_qadr+3:lemon_qadr+7] = q_s
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    policy.vel_cmd[0] = vx
    policy._update_command()

    renderer = mujoco.Renderer(model, width=1280, height=720)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[3] = 0  # Hide collision proxy geoms

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance = 0.68
    cam.elevation = -16.0
    cam.azimuth = 138.0

    total_sim_steps = int(duration_sec / model.opt.timestep)
    sim_steps_per_frame = int(round(1.0 / (fps * model.opt.timestep)))
    decimation = int(round(0.020 / model.opt.timestep))  # 10 steps = 50 Hz

    action = np.zeros(14, dtype=np.float32)
    head_pos_steadicam = 0.0
    filtered_yaw = 0.0

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    frame_count = 0

    for step in range(total_sim_steps):
        # Beak clamp torque
        beak_err = target_clamp_angle - data.qpos[beak_qpos]
        data.qfrc_applied[beak_dof] = float(np.clip(-clamp_torque + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))

        # 50 Hz Policy & Steadicam
        if step % decimation == 0:
            if policy.vel_cmd[0] != vx:
                policy.vel_cmd[0] = vx
                policy._update_command()
            action = policy.infer()

            R_trunk = data.xmat[trunk_id].reshape(3, 3)
            h_fwd = np.array([R_trunk[0, 0], R_trunk[1, 0], 0.0])
            h_fwd /= (np.linalg.norm(h_fwd) + 1e-6)

            R_s = data.xmat[spoon_bid].reshape(3, 3)
            b_norm = R_s @ np.array([0.0, 1.0, 0.0])
            spoon_pitch = math.asin(np.clip(float(np.dot(b_norm, h_fwd)), -0.999, 0.999))

            # Active Sagittal Pitch Leveling (Dead Parallelism with Ground)
            rate_cmd = np.clip(1.20 * spoon_pitch, -0.30, 0.30)
            dt_ctrl = decimation * model.opt.timestep
            head_pos_steadicam = np.clip(head_pos_steadicam + rate_cmd * dt_ctrl, -0.40, 0.40)

            action[5] = 0.0 + head_pos_steadicam
            action[6] = 0.0 - head_pos_steadicam
            action[8] = 0.0
            if not is_rl:
                action[7] = 0.0
            policy.apply_action(action)
        else:
            policy.apply_action(action)

        mujoco.mj_step(model, data)

        if step % sim_steps_per_frame == 0:
            renderer.update_scene(data, camera=cam, scene_option=vopt)
            img = renderer.render()
            writer.append_data(img)
            frame_count += 1
            if frame_count % (fps // 2) == 0:
                p_s = data.xpos[spoon_bid]
                R_s = data.xmat[spoon_bid].reshape(3, 3)
                bowl_center = p_s + R_s @ np.array([0.0, 0.0052, 0.070])
                dist = np.linalg.norm(data.xpos[lemon_bid] - bowl_center)
                print(f"  Frame {frame_count:3d}/{int(duration_sec*fps)} (t={step*0.002:.2f}s): "
                      f"walked {data.qpos[0]:.2f}m, lemon displacement={dist*1000:.1f}mm")

    writer.close()
    print(f"[VIDEO SUCCESS] Saved {frame_count} frames to {output_path}")
    return output_path


def run_benchmark_eval(policy_path=None, friction=0.50):
    """Runs systematic quantitative evaluation of lemon balancing across multiple speeds."""
    if policy_path is None:
        policy_path = _DEFAULT_POLICY
    xml_str = create_spoon_lemon_scene_xml(friction=friction)
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    policy, is_rl = create_policy_with_steadicam(model, data, policy_path)

    spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lemon")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_qpos = model.jnt_qposadr[beak_jid]
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    spoon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spoon_free")
    spoon_qadr = model.jnt_qposadr[spoon_jid]
    lemon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lemon_free")
    lemon_qadr = model.jnt_qposadr[lemon_jid]
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    target_clamp_angle = 1.550
    clamp_torque = 0.45
    decimation = int(round(0.020 / model.opt.timestep))

    test_speeds = [0.28, 0.30, 0.32, 0.35] if is_rl else [0.08, 0.12, 0.15, 0.20, 0.25]
    print("\n" + "=" * 70)
    print("QUANTITATIVE BENCHMARK: MICRODUCK LEMON & SPOON BALANCING")
    print("=" * 70)

    for vx in test_speeds:
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, 0.1172]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, idx in enumerate(policy.joint_qpos_indices):
            data.qpos[idx] = policy.default_pose[i]
        data.qpos[beak_qpos] = target_clamp_angle
        policy.set_position_targets(policy.default_pose)

        data.qpos[spoon_qadr:spoon_qadr+3] = [0.0, 0.0, 10.0]
        data.qpos[lemon_qadr:lemon_qadr+3] = [0.0, 0.0, 10.0]
        mujoco.mj_forward(model, data)

        for _ in range(100):
            policy.set_position_targets(policy.default_pose)
            beak_err = target_clamp_angle - data.qpos[beak_qpos]
            data.qfrc_applied[beak_dof] = float(np.clip(-clamp_torque + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))
            data.qpos[spoon_qadr:spoon_qadr+3] = [0.0, 0.0, 10.0]
            data.qpos[lemon_qadr:lemon_qadr+3] = [0.0, 0.0, 10.0]
            data.qvel[spoon_qadr:spoon_qadr+6] = 0.0
            data.qvel[lemon_qadr:lemon_qadr+6] = 0.0
            mujoco.mj_step(model, data)

        mujoco.mj_forward(model, data)
        p_jaw = data.xpos[jaw_bid]
        R_jaw = data.xmat[jaw_bid].reshape(3, 3)
        HOVER_LOCAL = np.array([-0.0119, 0.0, -0.0465])
        CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

        data.qpos[spoon_qadr:spoon_qadr+3] = p_jaw + R_jaw @ HOVER_LOCAL
        q_jaw = np.empty(4)
        mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
        q_s = np.empty(4)
        mujoco.mju_mulQuat(q_s, q_jaw, CYL_QUAT)
        data.qpos[spoon_qadr+3:spoon_qadr+7] = q_s
        mujoco.mj_forward(model, data)

        p_spoon = data.xpos[spoon_bid]
        R_spoon = data.xmat[spoon_bid].reshape(3, 3)
        p_lemon = p_spoon + R_spoon @ np.array([0.0, 0.0052, 0.070])
        data.qpos[lemon_qadr:lemon_qadr+3] = p_lemon
        data.qpos[lemon_qadr+3:lemon_qadr+7] = q_s
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        policy.vel_cmd[0] = vx
        policy._update_command()

        action = np.zeros(14, dtype=np.float32)
        head_pos_steadicam = 0.0
        filtered_yaw = 0.0
        fell = False
        max_dist = 0.0

        for step in range(5000):  # 10.0 seconds
            beak_err = target_clamp_angle - data.qpos[beak_qpos]
            data.qfrc_applied[beak_dof] = float(np.clip(-clamp_torque + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))

            if step % decimation == 0:
                action = policy.infer()
                R_trunk = data.xmat[trunk_id].reshape(3, 3)
                h_fwd = np.array([R_trunk[0, 0], R_trunk[1, 0], 0.0])
                h_fwd /= (np.linalg.norm(h_fwd) + 1e-6)

                R_s = data.xmat[spoon_bid].reshape(3, 3)
                b_norm = R_s @ np.array([0.0, 1.0, 0.0])
                spoon_pitch = math.asin(np.clip(float(np.dot(b_norm, h_fwd)), -0.999, 0.999))

                rate_cmd = np.clip(1.20 * spoon_pitch, -0.30, 0.30)
                dt_ctrl = decimation * model.opt.timestep
                head_pos_steadicam = np.clip(head_pos_steadicam + rate_cmd * dt_ctrl, -0.40, 0.40)

                if is_rl:
                    action[8] = 0.0
                else:
                    action[5] = 0.0 + head_pos_steadicam
                    action[6] = 0.0 - head_pos_steadicam
                    action[7] = 0.0
                    action[8] = 0.0
                policy.apply_action(action)
            else:
                policy.apply_action(action)

            mujoco.mj_step(model, data)

            p_s = data.xpos[spoon_bid]
            R_s = data.xmat[spoon_bid].reshape(3, 3)
            bowl_center = p_s + R_s @ np.array([0.0, 0.0052, 0.070])
            dist = np.linalg.norm(data.xpos[lemon_bid] - bowl_center)
            if dist > max_dist:
                max_dist = dist
            if dist > 0.05:
                fell = True
                print(f"Speed: {vx:.2f} m/s | FAILED (Lemon fell at t={step*0.002:.2f}s, distance={data.qpos[0]:.2f}m)")
                break

        if not fell:
            print(f"Speed: {vx:.2f} m/s | PASSED: 10.0s balance, walked {data.qpos[0]:.2f}m, max bowl deviation={max_dist*1000:.1f}mm")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Microduck Spoon & Lemon Balance Test")
    parser.add_argument("--proofs", action="store_true", help="Render high-res visual proofs and exit")
    parser.add_argument("--gui", action="store_true", help="Launch native interactive GLFW MuJoCo viewer")
    parser.add_argument("--video", action="store_true", help="Render offscreen MP4 demonstration video")
    parser.add_argument("--eval", action="store_true", help="Run quantitative evaluation benchmark across speeds")
    parser.add_argument("--vx", type=float, default=0.30, help="Forward walking speed (m/s, default: 0.30)")
    parser.add_argument("--duration", type=float, default=6.0, help="Duration for video render (seconds)")
    parser.add_argument("--stand", action="store_true", help="Hold standing balance instead of walking")
    parser.add_argument("--speed", type=float, default=1.0, help="Viewer playback speed multiplier")
    parser.add_argument("--policy", type=str, default=None, help="Path to custom ONNX locomotion policy")
    parser.add_argument("--official", "--official-walk", dest="official", action="store_true", help="Use official repository baseline walking policy (pollen-robotics BEST_alpha_walking)")
    parser.add_argument("--steadicam", action="store_true", help="Use trained Steadicam Residual RL policy")
    parser.add_argument("--raw-policy", "--raw", dest="raw_policy", action="store_true", help="Run raw policy actions directly without Steadicam head pitch override")
    parser.add_argument("--friction", "--mu", type=float, default=0.50, help="Coulomb friction coefficient between lemon and spoon (default: 0.50)")
    args = parser.parse_args()

    # Resolve policy path
    if args.official:
        policy_path = _OFFICIAL_WALKING_POLICY
    elif args.steadicam:
        policy_path = _STEADICAM_POLICY
    elif args.policy is not None:
        policy_path = args.policy
    else:
        policy_path = _DEFAULT_POLICY

    if args.proofs:
        render_visual_proofs(policy_path=policy_path, friction=args.friction)
        return

    if args.video:
        render_walking_video(duration_sec=args.duration, vx=args.vx, policy_path=policy_path, friction=args.friction)
        return

    if args.eval:
        run_benchmark_eval(policy_path=policy_path, friction=args.friction)
        return

    if not args.gui:
        # Default: render proofs and show summary
        render_visual_proofs(policy_path=policy_path, friction=args.friction)
        run_benchmark_eval(policy_path=policy_path, friction=args.friction)
        return

    # Interactive GUI runner
    xml_str = create_spoon_lemon_scene_xml(friction=args.friction)
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    is_official = ("official" in policy_path.lower() or "best_alpha" in policy_path.lower())
    crit_angle = math.degrees(math.atan(args.friction))
    print(f"\n[Policy] Loading locomotion policy: {policy_path}")
    print(f"[Policy Mode] {'OFFICIAL POLLEN ROBOTICS BASELINE WALKING' if is_official else 'PAYLOAD CLAMPING'}")
    print(f"[Friction] Lemon-Spoon Coulomb Friction mu = {args.friction:.3f} (Critical Sliding Angle = {crit_angle:.1f}°)")

    policy = PolicyInference(
        model, data,
        walking_onnx_path=policy_path,
        action_scale=1.0,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )

    policy.default_pose[5] = 0.2760   # neck_pitch: level forward
    policy.default_pose[6] = 0.2760   # head_pitch: exact zero-pitch level forward
    policy.default_pose[7] = 0.0
    policy.default_pose[8] = 0.0

    spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lemon")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_qpos = model.jnt_qposadr[beak_jid]
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    spoon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spoon_free")
    spoon_qadr = model.jnt_qposadr[spoon_jid]
    lemon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lemon_free")
    lemon_qadr = model.jnt_qposadr[lemon_jid]
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    target_vx = 0.0 if args.stand else args.vx
    current_vx = target_vx
    target_clamp_angle = 1.550
    clamp_torque = 0.45
    sim_speed = args.speed
    user_paused = False
    reset_requested = False
    quit_requested = False
    head_pos_steadicam = 0.0
    cam_mode = 0  # 0: Fixed World Track, 1: Tracking Perspective, 2: Spoon Close-Up
    viewer_ref = None

    # Enable active 3-DOF Steadicam Gimbal by default to decouple sagittal pitch and coronal roll
    is_rl_steadicam = False
    if policy.walking_session and policy.walking_session.get_inputs()[0].shape[1] == 67:
        is_rl_steadicam = True
        lemon_dof = model.jnt_dofadr[lemon_jid]
        spoon_dof = model.jnt_dofadr[spoon_jid]

        def compute_extra_obs():
            R_s = data.xmat[spoon_bid].reshape(3, 3)
            p_diff = data.xpos[lemon_bid] - data.xpos[spoon_bid]
            p_rel = R_s.T @ p_diff
            dx = float(p_rel[0])
            dz = float(p_rel[2] - 0.070)

            v_diff = data.qvel[lemon_dof:lemon_dof+3] - data.qvel[spoon_dof:spoon_dof+3]
            v_rel = R_s.T @ v_diff
            vx_rel = float(v_rel[0])
            vz_rel = float(v_rel[2])

            z_local = R_s.T @ np.array([0.0, 0.0, 1.0])
            pitch_err = float(np.arcsin(np.clip(z_local[2], -0.999, 0.999)))
            roll_err = float(np.arcsin(np.clip(z_local[0], -0.999, 0.999)))

            return np.array([dx, dz, vx_rel, vz_rel, pitch_err, roll_err], dtype=np.float32)

        policy.extra_obs_fn = compute_extra_obs
        print("[Steadicam RL] Detected 67D trained Residual Steadicam RL policy. Activating closed-loop payload feedback!")

    use_steadicam = False if args.raw_policy else True
    action = np.zeros(14, dtype=np.float32)
    decimation = int(round(0.020 / model.opt.timestep))  # 50 Hz control rate
    current_friction = float(args.friction)
    lemon_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lemon_geom")

    def apply_camera(v, mode):
        if v is None:
            return
        if mode == 0:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.lookat = [0.60, 0.0, 0.15]
            v.cam.distance = 1.35
            v.cam.azimuth = 105.0
            v.cam.elevation = -14.0
            print("\n[CAMERA 0] Fixed World Track Camera (Watch robot walk across the screen!)")
        elif mode == 1:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = 1
            v.cam.distance = 0.75
            v.cam.azimuth = 138.0
            v.cam.elevation = -16.0
            print("\n[CAMERA 1] Duck Tracking Camera (Third-person chasing perspective)")
        elif mode == 2:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = 1
            v.cam.distance = 0.32
            v.cam.azimuth = 120.0
            v.cam.elevation = -10.0
            print("\n[CAMERA 2] Spoon Bowl Steadicam Close-Up (Inspect bowl cradling & grip)")

    def update_friction(mu):
        crit_deg = math.degrees(math.atan(mu))
        for p in range(model.npair):
            if model.pair_geom1[p] == lemon_gid or model.pair_geom2[p] == lemon_gid:
                model.pair_friction[p, 0] = mu
                model.pair_friction[p, 1] = mu
        if lemon_gid != -1:
            model.geom_friction[lemon_gid, 0] = mu
        print(f"\n[INTERACTIVE] Lemon-Spoon Friction -> mu = {mu:.2f} (Critical Sliding Angle = {crit_deg:.1f} deg)")

    def process_input(key):
        nonlocal user_paused, reset_requested, quit_requested, target_vx, sim_speed
        nonlocal clamp_torque, target_clamp_angle, current_friction, use_steadicam, cam_mode, viewer_ref
        if isinstance(key, int):
            k = chr(key).lower() if (32 <= key <= 126) else key
        else:
            k = str(key).lower()

        if k in (32, " ", "space"):
            user_paused = not user_paused
            print(f"\n[INTERACTIVE] {'PAUSED' if user_paused else 'RESUMED'}")
        elif k in (ord('Q'), ord('q'), 'q'):
            quit_requested = True
            print("\n[INTERACTIVE] Quitting viewer...")
        elif k in (ord('R'), ord('r'), 'r'):
            reset_requested = True
            print("\n[INTERACTIVE] Resetting episode...")
        elif k in (ord('V'), ord('v'), 'v'):
            cam_mode = (cam_mode + 1) % 3
            apply_camera(viewer_ref, cam_mode)
        elif k in (ord('T'), ord('t'), 't'):
            use_steadicam = not use_steadicam
            mode_str = "ACTIVE STEADICAM PANTOGRAPH" if use_steadicam else "RAW UNCOMPENSATED POLICY"
            print(f"\n[INTERACTIVE] Head Stabilization -> {mode_str}")
        elif k in (ord('W'), ord('w'), 265, "w", "up"):
            target_vx = min(0.35, round(target_vx + 0.05, 2))
            print(f"\n[INTERACTIVE] Commanded Forward Speed -> {target_vx:.2f} m/s")
        elif k in (ord('S'), ord('s'), 264, "s", "down"):
            target_vx = max(0.00, round(target_vx - 0.05, 2))
            print(f"\n[INTERACTIVE] Commanded Forward Speed -> {target_vx:.2f} m/s")
        elif k in (ord('['), 91, "["):
            current_friction = max(0.01, round(current_friction - 0.05, 2))
            update_friction(current_friction)
        elif k in (ord(']'), 93, "]"):
            current_friction = min(1.50, round(current_friction + 0.05, 2))
            update_friction(current_friction)
        elif k in (ord('.'), 46, "."):
            sim_speed = min(5.0, round(sim_speed + 0.25, 2))
            print(f"\n[INTERACTIVE] Playback Speed -> {sim_speed:.2f}x")
        elif k in (ord(','), 44, ","):
            sim_speed = max(0.25, round(sim_speed - 0.25, 2))
            print(f"\n[INTERACTIVE] Playback Speed -> {sim_speed:.2f}x")
        elif k in (ord('O'), ord('o'), 'o'):
            target_clamp_angle = min(2.05, round(target_clamp_angle + 0.02, 3))
            print(f"\n[INTERACTIVE] Beak Aperture -> {target_clamp_angle:.3f} rad ({math.degrees(target_clamp_angle):.1f}°)")
        elif k in (ord('C'), ord('c'), 'c'):
            target_clamp_angle = max(1.47, round(target_clamp_angle - 0.02, 3))
            print(f"\n[INTERACTIVE] Beak Aperture -> {target_clamp_angle:.3f} rad ({math.degrees(target_clamp_angle):.1f}°)")
        elif k in (ord('+'), ord('='), "+", "="):
            clamp_torque = min(1.0, round(clamp_torque + 0.05, 2))
            print(f"\n[INTERACTIVE] Clamp Torque -> {clamp_torque:.2f} N*m")
        elif k in (ord('-'), ord('_'), "-", "_"):
            clamp_torque = max(0.05, round(clamp_torque - 0.05, 2))
            print(f"\n[INTERACTIVE] Clamp Torque -> {clamp_torque:.2f} N*m")

    def reset_sim():
        nonlocal head_pos_steadicam, current_vx, step, filtered_yaw
        filtered_yaw = 0.0
        mujoco.mj_resetData(model, data)

        # 1. Place trunk at exact ground contact height so feet rest flat on the floor (no impact shock)
        data.qpos[0:3] = [0.0, 0.0, 0.1172]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, idx in enumerate(policy.joint_qpos_indices):
            data.qpos[idx] = policy.default_pose[i]
        data.qpos[beak_qpos] = target_clamp_angle
        policy.set_position_targets(policy.default_pose)

        # Park spoon & lemon high in air during duck ground contact settling
        data.qpos[spoon_qadr:spoon_qadr+3] = [0.0, 0.0, 10.0]
        data.qpos[lemon_qadr:lemon_qadr+3] = [0.0, 0.0, 10.0]
        mujoco.mj_forward(model, data)

        # 2. Settle the bipedal robot on the floor for 100 steps under active PD control
        for _ in range(100):
            policy.set_position_targets(policy.default_pose)
            beak_err = target_clamp_angle - data.qpos[beak_qpos]
            data.qfrc_applied[beak_dof] = float(np.clip(-clamp_torque + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))
            data.qpos[spoon_qadr:spoon_qadr+3] = [0.0, 0.0, 10.0]
            data.qpos[lemon_qadr:lemon_qadr+3] = [0.0, 0.0, 10.0]
            data.qvel[spoon_qadr:spoon_qadr+6] = 0.0
            data.qvel[lemon_qadr:lemon_qadr+6] = 0.0
            mujoco.mj_step(model, data)

        # 3. Position the spoon handle in the settled beak with level orientation (+1.6 deg)
        mujoco.mj_forward(model, data)
        p_jaw = data.xpos[jaw_bid]
        R_jaw = data.xmat[jaw_bid].reshape(3, 3)
        HOVER_LOCAL = np.array([-0.0119, 0.0, -0.0465])
        CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

        data.qpos[spoon_qadr:spoon_qadr+3] = p_jaw + R_jaw @ HOVER_LOCAL
        q_jaw = np.empty(4)
        mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
        q_s = np.empty(4)
        mujoco.mju_mulQuat(q_s, q_jaw, CYL_QUAT)
        data.qpos[spoon_qadr+3:spoon_qadr+7] = q_s
        mujoco.mj_forward(model, data)

        # 4. Place smooth lemon ball at the bottom of the spoon bowl
        p_spoon = data.xpos[spoon_bid]
        R_spoon = data.xmat[spoon_bid].reshape(3, 3)
        p_lemon = p_spoon + R_spoon @ np.array([0.0, 0.0052, 0.070])
        data.qpos[lemon_qadr:lemon_qadr+3] = p_lemon
        data.qpos[lemon_qadr+3:lemon_qadr+7] = q_s

        # 5. Zero all velocities and finalize forward kinematics
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        head_pos_steadicam = 0.0
        current_vx = 0.0  # Smooth acceleration ramp from stand to prevent initial drop
        step = 0

    reset_sim()

    with TerminalInput() as term, mujoco.viewer.launch_passive(model, data, key_callback=process_input) as viewer:
        viewer_ref = viewer
        apply_camera(viewer, cam_mode)
        viewer.opt.geomgroup[0] = 1
        viewer.opt.geomgroup[1] = 1
        viewer.opt.geomgroup[2] = 1
        viewer.opt.geomgroup[3] = 0

        print("\n" + "=" * 65)
        print("MICRODUCK INTERACTIVE LEMON & SPOON VIEWER (ACTIVE STRIDE READY)")
        print("=" * 65)
        print("  V:         Cycle Camera View (0: Fixed World | 1: Track Duck | 2: Spoon)")
        print("  W / Up:    Accelerate forward walking speed (+0.05 m/s)")
        print("  S / Down:  Decelerate forward walking speed (-0.05 m/s)")
        print("  T:         Toggle Active Steadicam Gimbal vs Raw Policy")
        print("  [ / ]:     Decrease / Increase lemon-spoon friction (-0.05 / +0.05)")
        print("  . / >:     Speed up playback (+0.25x)")
        print("  , / <:     Slow down playback (-0.25x)")
        print("  O / C:     Open / Close beak aperture")
        print("  + / -:     Increase / Decrease clamp torque")
        print("  Space:     Pause / Resume physics")
        print("  R:         Reset episode")
        print("  Q:         Quit viewer")
        print("=" * 65 + "\n")

        step = 0
        spoon_pitch = 0.0
        filtered_yaw = 0.0
        while viewer.is_running() and not quit_requested:
            t_start = time.time()

            # Process terminal keystrokes (concurrent with viewer GUI keys)
            for k in term.get_keys():
                process_input(k)

            if reset_requested:
                reset_requested = False
                reset_sim()

            if user_paused:
                viewer.sync()
                time.sleep(0.02)
                continue

            # Continuous silicone clamp regulation
            beak_err = target_clamp_angle - data.qpos[beak_qpos]
            data.qfrc_applied[beak_dof] = float(np.clip(-clamp_torque + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))

            # Progressive forward acceleration (avoids initial step slump / lurch)
            if current_vx < target_vx:
                current_vx = min(target_vx, current_vx + 0.003)
            elif current_vx > target_vx:
                current_vx = max(target_vx, current_vx - 0.005)

            # 50 Hz Decimated Policy & Active Steadicam Gimbal
            # Line-of-sight pure pursuit heading regulation (keeps duck walking straight on centerline Y=0)
            R_trunk = data.xmat[trunk_id].reshape(3, 3)
            yaw = math.atan2(float(R_trunk[1, 0]), float(R_trunk[0, 0]))
            y_err = float(data.xpos[trunk_id][1] - 0.0)
            target_yaw = math.atan2(-y_err, 0.60)
            yaw_err = (yaw - target_yaw + math.pi) % (2 * math.pi) - math.pi
            cmd_wz = float(np.clip(-2.5 * yaw_err, -0.45, 0.45))

            policy.vel_cmd[0] = current_vx
            policy.vel_cmd[1] = 0.0
            policy.vel_cmd[2] = cmd_wz
            policy._update_command()
            action = policy.infer()

            R_trunk = data.xmat[trunk_id].reshape(3, 3)
            h_fwd = np.array([R_trunk[0, 0], R_trunk[1, 0], 0.0])
            h_fwd /= (np.linalg.norm(h_fwd) + 1e-6)

            R_s = data.xmat[spoon_bid].reshape(3, 3)
            b_norm = R_s @ np.array([0.0, 1.0, 0.0])
            spoon_pitch = math.asin(np.clip(float(np.dot(b_norm, h_fwd)), -0.999, 0.999))

            if use_steadicam:
                # 1. Sagittal Pitch Decoupling (Differential Pantograph Integrator)
                rate_cmd = np.clip(1.20 * spoon_pitch, -0.30, 0.30)
                dt_ctrl = decimation * model.opt.timestep
                head_pos_steadicam = np.clip(head_pos_steadicam + rate_cmd * dt_ctrl, -0.40, 0.40)

                if is_rl_steadicam:
                    # Active Sagittal Pitch Leveling (Dead Parallelism with Ground) + Active Yaw Gimbal
                    action[5] = 0.0 + head_pos_steadicam
                    action[6] = 0.0 - head_pos_steadicam
                    action[8] = 0.0
                else:
                    action[5] = 0.0 + head_pos_steadicam
                    action[6] = 0.0 - head_pos_steadicam
                    action[7] = 0.0  # Head Yaw forward facing
                    action[8] = 0.0  # Coronal Roll strictly locked to 0.0 (Zero Head Tilt)

            policy.apply_action(action)

            # Step physics sub-steps at 500 Hz inside the 50 Hz control window
            for _ in range(decimation):
                beak_err = target_clamp_angle - data.qpos[beak_qpos]
                data.qfrc_applied[beak_dof] = float(np.clip(-clamp_torque + 10.0 * beak_err - 0.3 * data.qvel[beak_dof], -1.0, 0.2))
                mujoco.mj_step(model, data)
                step += 1

            # Periodic telemetry HUD log (every 0.2s = 10 control steps)
            if (step // decimation) % 10 == 0:
                p_s = data.xpos[spoon_bid]
                R_s = data.xmat[spoon_bid].reshape(3, 3)
                bowl_center = p_s + R_s @ np.array([0.0, 0.0052, 0.070])
                dist_mm = np.linalg.norm(data.xpos[lemon_bid] - bowl_center) * 1000.0
                if dist_mm < 15.0:
                    lemon_status = "BALANCED [STABLE]"
                elif dist_mm < 45.0:
                    lemon_status = "SLIDING [NEAR RIM]"
                else:
                    lemon_status = "DROPPED [FALLEN]"
                mode_tag = "STEADICAM RESIDUAL RL" if is_rl_steadicam else ("STEADICAM HEURISTIC" if use_steadicam else "RAW POLICY")
                print(f"\r[t={step*0.002:5.1f}s | x_pos={data.qpos[0]:+5.2f}m | vx={current_vx:.2f} m/s | mu={current_friction:.2f} | {mode_tag} | spoon pitch={math.degrees(spoon_pitch):+5.1f}° | bowl dev={dist_mm:4.1f} mm | {lemon_status}]", end="", flush=True)

            # Smooth real-time viewer synchronization at 50 Hz
            viewer.sync()
            t_elapsed = time.time() - t_start
            dt_sleep = ((decimation * model.opt.timestep) / sim_speed) - t_elapsed
            if dt_sleep > 0:
                time.sleep(dt_sleep)


if __name__ == "__main__":
    main()

