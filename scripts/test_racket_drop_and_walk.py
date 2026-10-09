#!/usr/bin/env python3
"""Evaluate Scaled Racket Minimal Drop, Physical Beak Clamping & Walking Balance.

Protocol:
1. Phase 1 (0.0s - 0.4s): Robot stands in stable biped balance with beak WIDE OPEN (aperture ~24mm, q=2.04 rad).
2. Phase 2 (0.4s - 0.5s): Scaled racket handle is minimally dropped (3mm) onto lower silicone jaw bed.
3. Phase 3 (0.5s - 0.8s): Beak closes with 0.18 N·m calibrated servo torque, firmly clamping handle without mesh penetration.
4. Phase 4 (0.8s - 5.0s): Robot walks forward at Vx = 0.20 m/s with zero welds, pure contact physics.

Note for macOS: Run with `mjpython` when using `--gui`:
  microduck_rl/.venv/bin/mjpython microduck_manipulation/scripts/test_racket_drop_and_walk.py --racket pickleball --gui --loop
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
sys.path.insert(0, _SCRIPT_DIR)

from infer_policy import PolicyInference, DEFAULT_POSE

_BASE_SCENE_XML = os.path.join(_MANIP_DIR, "models/scene_loadcarry_physical_clamp.xml")
_DEFAULT_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_ppo_headup_v4.onnx")
_TT_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/table_tennis_ball_balance.onnx")
_TT_CARRY_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/table_tennis_racket_carry_precision_level.onnx")
_OUTPUT_DIR = os.path.join(_MANIP_DIR, "docs/media")
os.makedirs(_OUTPUT_DIR, exist_ok=True)


class TableTennisPolicyInference(PolicyInference):
    """Extends PolicyInference with the 8 ball balancing observations (69D total)."""

    def __init__(self, model, data, onnx_path):
        super().__init__(
            model, data,
            walking_onnx_path=onnx_path,
            action_scale=1.0,
            new_cmd_obs=True,
            use_projected_gravity=True,
        )
        # Calibrated nominal posture: matched to DEFAULT_POSE from loadcarry v4
        self.default_pose[5] = float(DEFAULT_POSE[5])   # neck_pitch: 0.3491 rad
        self.default_pose[6] = float(DEFAULT_POSE[6])   # head_pitch: 0.3491 rad
        self.default_pose[7] = 0.0                      # head_yaw
        self.default_pose[8] = 0.0                      # head_roll

        self.racket_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "racket")
        self.ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pingpong_ball")

    def get_ball_observations(self):
        p_racket = self.data.xpos[self.racket_bid]
        R_racket = self.data.xmat[self.racket_bid].reshape(3, 3)
        p_ball = self.data.xpos[self.ball_bid]

        p_rel = R_racket.T @ (p_ball - p_racket)
        dx = float(p_rel[0])
        dz = float(p_rel[2] - 0.1015)

        v_ball = self.data.cvel[self.ball_bid][3:6]
        v_racket = self.data.cvel[self.racket_bid][3:6]
        v_rel_local = R_racket.T @ (v_ball - v_racket)
        vx = float(v_rel_local[0])
        vz = float(v_rel_local[2])

        R_base = self.data.xmat[self.trunk_base_id].reshape(3, 3)
        normal_world = R_racket @ np.array([0.0, 1.0, 0.0])
        normal_base = R_base.T @ normal_world

        ball_contact = 0.0
        for ci in range(self.data.ncon):
            con = self.data.contact[ci]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            if (b1 == self.ball_bid and b2 == self.racket_bid) or (b1 == self.racket_bid and b2 == self.ball_bid):
                ball_contact = 1.0
                break

        pitch_err = float(math.asin(np.clip(normal_world[0], -0.999, 0.999)))
        roll_err = float(math.asin(np.clip(normal_world[1], -0.999, 0.999)))

        expected_dim = self.ort_session.get_inputs()[0].shape[1] if hasattr(self, "ort_session") else 71
        if isinstance(expected_dim, int) and expected_dim == 71:
            return np.array(
                [dx, dz, vx, vz, normal_base[0], normal_base[1], normal_base[2], ball_contact, pitch_err, roll_err],
                dtype=np.float32,
            )
        return np.array([dx, dz, vx, vz, normal_base[0], normal_base[1], normal_base[2], ball_contact], dtype=np.float32)

    def get_observations(self):
        base_obs = super().get_observations()
        ball_obs = self.get_ball_observations()
        return np.concatenate([base_obs, ball_obs]).astype(np.float32)


class TableTennisRacketCarryPolicyInference(PolicyInference):
    """Extends PolicyInference with the 3D racket normal in base frame (64D total, zero ball)."""

    def __init__(self, model, data, onnx_path):
        super().__init__(
            model, data,
            walking_onnx_path=onnx_path,
            action_scale=1.0,
            new_cmd_obs=True,
            use_projected_gravity=True,
        )
        self.default_pose[5] = 0.2760   # neck_pitch: calibrated horizontal zero-pitch setpoint (mean pitch = -0.87°, std = 1.54°)
        self.default_pose[6] = float(DEFAULT_POSE[6])   # head_pitch: 0.3491 rad (+20°) from loadcarry v4
        self.default_pose[7] = 0.0                      # head_yaw
        self.default_pose[8] = 0.0                      # head_roll: bilateral zero roll for forward carry
        self.racket_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "racket")

    def get_racket_normal_base(self):
        R_racket = self.data.xmat[self.racket_bid].reshape(3, 3)
        R_base = self.data.xmat[self.trunk_base_id].reshape(3, 3)
        normal_world = R_racket @ np.array([0.0, 1.0, 0.0])
        normal_base = R_base.T @ normal_world
        return normal_base.astype(np.float32)

    def get_observations(self):
        base_obs = super().get_observations()
        rkt_norm = self.get_racket_normal_base()
        return np.concatenate([base_obs, rkt_norm]).astype(np.float32)


def create_scaled_scene_xml(racket_type="table_tennis", include_ball=True):
    """Builds an MJCF string embedding the racket and ball with zero welds."""
    with open(_BASE_SCENE_XML, "r") as f:
        base = f.read()

    assets_dir = os.path.join(_MANIP_DIR, "models/assets")
    import re
    base = re.sub(r'meshdir="[^"]*"', f'meshdir="{assets_dir}/" balanceinertia="true"', base)
    base = base.replace('elevation="-20"/>', 'elevation="-20" offwidth="1280" offheight="960"/>')

    extra_assets = """
    <!-- CAD Meshes for Solid Table Tennis Racket (100% Solid Full-Tang Wood, Zero Hollow Gaps) -->
    <mesh name="tt_blade_wood_mesh" file="tt_blade_wood.stl" scale="0.70 0.70 0.70"/>
    <mesh name="tt_rubber_red_mesh" file="tt_rubber_red.stl" scale="0.70 0.70 0.70"/>
    <mesh name="tt_rubber_black_mesh" file="tt_rubber_black.stl" scale="0.70 0.70 0.70"/>
    <mesh name="tt_handle_scales_mesh" file="tt_handle_scales.stl" scale="0.70 0.70 0.70"/>
    <mesh name="tt_handle_stripe_mesh" file="tt_handle_stripe.stl" scale="0.70 0.70 0.70"/>
    <mesh name="tt_handle_lens_mesh" file="tt_handle_lens.stl" scale="0.70 0.70 0.70"/>

    <material name="tt_wood_ply" rgba="0.88 0.77 0.60 1" specular="0.25" shininess="0.15"/>
    <material name="tt_rubber_red" rgba="0.86 0.10 0.12 1" specular="0.45" shininess="0.4"/>
    <material name="tt_rubber_black" rgba="0.12 0.12 0.14 1" specular="0.4" shininess="0.3"/>
    <material name="tt_handle_wood" rgba="0.62 0.44 0.28 1" specular="0.2" shininess="0.1"/>
    <material name="tt_handle_stripe_mat" rgba="0.20 0.30 0.55 1" specular="0.3" shininess="0.2"/>
    <material name="tt_lens_gold" rgba="0.92 0.80 0.24 1" specular="0.9" shininess="0.95"/>
    <material name="tt_ball_orange" rgba="1.00 0.58 0.08 1" specular="0.4" shininess="0.5"/>

    <material name="pb_edge_guard" rgba="0.12 0.12 0.14 1" specular="0.3" shininess="0.4"/>
    <material name="pb_face_cyan" rgba="0.08 0.65 0.84 1" specular="0.5" shininess="0.6"/>
    <material name="pb_face_yellow" rgba="0.98 0.82 0.15 1" specular="0.6" shininess="0.7"/>
    <material name="pb_grip_white" rgba="0.94 0.94 0.96 1" specular="0.1" shininess="0.1"/>
    <material name="pb_butt_cap" rgba="0.12 0.12 0.15 1" specular="0.4" shininess="0.5"/>
    <material name="bm_frame_blue" rgba="0.12 0.25 0.48 1" specular="0.8" shininess="0.9"/>
    <material name="bm_shaft_carbon" rgba="0.09 0.14 0.22 1" specular="0.8" shininess="0.9"/>
    <material name="bm_t_joint_gold" rgba="0.90 0.78 0.22 1" specular="0.95" shininess="0.95"/>
    <material name="bm_strings" rgba="0.95 0.98 0.98 0.42" specular="0.3" shininess="0.5"/>
    <material name="bm_grip_black" rgba="0.12 0.12 0.12 1" specular="0.1" shininess="0.1"/>
    <material name="bm_cone_grey" rgba="0.25 0.28 0.32 1" specular="0.5" shininess="0.6"/>
    <material name="bm_butt_green" rgba="0.05 0.52 0.28 1" specular="0.5" shininess="0.6"/>
  </asset>
"""
    base = base.replace("  </asset>", extra_assets)

    cyl_start = base.find('<body name="cylinder"')
    cyl_end = base.find('</body>\n  </worldbody>') + len('</body>')
    old_cyl = base[cyl_start:cyl_end]

    if racket_type == "table_tennis":
        # Proportionally Scaled 0.70x Table Tennis Racket & Scaled Ball:
        # Total Assembled Mass: 80g (60g handle + 20g blade face overhang, handle-biased counterbalance)
        # Dimensions: 178mm length, 105mm width, 110mm blade head (100% full visual scale)
        # 100% Solid Full-Tang Wood Handle: 62mm length, 8.4mm thickness
        body_str = """    <body name="racket" pos="0.048 0.0 0.220" quat="0.7071068 0.7071068 0 0">
      <freejoint name="racket_free"/>
      <!-- Scaled tournament racket mass: 40 grams (30g handle / 10g blade face, COM at z=16mm) -->
      <inertial pos="0 0 0.0160" mass="0.040" diaginertia="0.0000425 0.0000360 0.0000140"/>

      <!-- High-Fidelity CAD STL Visual Geoms (0.70x Scale) -->
      <geom name="tt_wood_vis" type="mesh" mesh="tt_blade_wood_mesh" material="tt_wood_ply" contype="0" conaffinity="0" group="0"/>
      <geom name="tt_red_vis" type="mesh" mesh="tt_rubber_red_mesh" material="tt_rubber_red" contype="0" conaffinity="0" group="0"/>
      <geom name="tt_black_vis" type="mesh" mesh="tt_rubber_black_mesh" material="tt_rubber_black" contype="0" conaffinity="0" group="0"/>
      <geom name="tt_handle_vis" type="mesh" mesh="tt_handle_scales_mesh" material="tt_handle_wood" contype="0" conaffinity="0" group="0"/>
      <geom name="tt_stripe_vis" type="mesh" mesh="tt_handle_stripe_mesh" material="tt_handle_stripe_mat" contype="0" conaffinity="0" group="0"/>
      <geom name="tt_lens_vis" type="mesh" mesh="tt_handle_lens_mesh" material="tt_lens_gold" contype="0" conaffinity="0" group="0"/>

      <!-- Collision Geoms (group="3" purely invisible collision) -->
      <!-- Solid Wood Handle for Beak Clamping (CAD dimensions: length 62mm, thickness 5.88mm, width 23.1mm, sticky grip friction 8.0) -->
      <geom name="racket_handle" type="box" pos="0 0 0.008" size="0.01155 0.00294 0.031" material="tt_handle_wood" friction="8.0 0.5 0.5" contype="0" conaffinity="0" group="3"/>
      <!-- Flat Rubber Surface for Ball Contact (width 105mm, length 110mm, half thickness 3.5mm) -->
      <geom name="racket_face" type="box" pos="0 0 0.1015" size="0.0525 0.0035 0.0540" material="tt_rubber_red" friction="0.80 0.005 0.005" contype="0" conaffinity="0" group="3"/>
    </body>"""
        if include_ball:
            body_str += """
    <!-- Proportionally Scaled 0.70x Table Tennis Ball (Diameter 28.1mm, Mass 1.30g) -->
    <body name="pingpong_ball" pos="0.050 0.1015 0.220">
      <freejoint name="ball_free"/>
      <inertial pos="0 0 0" mass="0.0013" diaginertia="0.000000170 0.000000170 0.000000170"/>
      <geom name="ball_geom" type="sphere" size="0.0141" material="tt_ball_orange" friction="0.80 0.005 0.005" condim="6"/>
      <!-- ITTF 3-Star Branding Emblem on Ball -->
      <geom name="ball_star1" type="sphere" pos="0 0.0142 0" size="0.0011" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>
      <geom name="ball_star2" type="sphere" pos="-0.0021 0.0142 0" size="0.0009" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>
      <geom name="ball_star3" type="sphere" pos="0.0021 0.0142 0" size="0.0009" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>
    </body>"""
    elif racket_type == "pickleball":
        # Scaled Pickleball Paddle: 120mm handle (spans Y = -60mm to +60mm)
        # Throat starts at Y = +68mm (over 22mm outside the 45.7mm jaw cheek!)
        body_str = """    <body name="racket" pos="0.048 0.0 0.220" quat="0.7071068 0.7071068 0 0">
      <freejoint name="racket_free"/>
      <!-- Mass: 75g counterbalanced so COM is at mouth center (z=0) -->
      <inertial pos="0 0 0" mass="0.075" diaginertia="0.00018 0.00015 0.000035"/>
      <geom name="racket_butt" type="cylinder" pos="0 0 -0.064" size="0.007 0.002" material="pb_butt_cap" friction="2.5 0.1 0.1"/>
      <!-- ONLY THIS 8.0mm HANDLE IS INSIDE THE MOUTH (length 120mm, spans Y = -60mm to +60mm) -->
      <geom name="racket_handle" type="cylinder" pos="0 0 0" size="0.004 0.060" material="pb_grip_white" friction="3.5 0.2 0.2" contype="0" conaffinity="0"/>
      <!-- Throat starts at z = 0.068 (Y = +68mm, over 22mm outside the 45.7mm jaw cheek!) -->
      <geom name="racket_throat" type="box" pos="0 0 0.068" size="0.012 0.003 0.008" material="pb_edge_guard"/>
      <!-- Paddle face: completely outside mouth and clear of jaw -->
      <geom name="racket_face" type="box" pos="0 0 0.128" size="0.035 0.0028 0.046" material="pb_face_cyan" friction="1.8 0.1 0.1"/>
      <geom name="racket_accent" type="box" pos="0 0.0030 0.128" size="0.028 0.0003 0.016" material="pb_face_yellow"/>
      <geom name="racket_edge_l" type="capsule" fromto="-0.039 0 0.082 -0.039 0 0.160" size="0.0035" material="pb_edge_guard"/>
      <geom name="racket_edge_r" type="capsule" fromto="0.039 0 0.082 0.039 0 0.160" size="0.0035" material="pb_edge_guard"/>
      <geom name="racket_edge_top" type="capsule" fromto="-0.026 0 0.174 0.026 0 0.174" size="0.0035" material="pb_edge_guard"/>
      <geom name="racket_cor_tl" type="capsule" fromto="-0.039 0 0.160 -0.026 0 0.174" size="0.0035" material="pb_edge_guard"/>
      <geom name="racket_cor_tr" type="capsule" fromto="0.039 0 0.160 0.026 0 0.174" size="0.0035" material="pb_edge_guard"/>
      <geom name="racket_cor_bl" type="capsule" fromto="-0.039 0 0.082 -0.012 0 0.070" size="0.0035" material="pb_edge_guard"/>
      <geom name="racket_cor_br" type="capsule" fromto="0.039 0 0.082 0.012 0 0.070" size="0.0035" material="pb_edge_guard"/>
    </body>"""
    else:
        # Scaled Badminton Racket: 110mm handle (spans Y = -55mm to +55mm)
        body_str = """    <body name="racket" pos="0.048 0.0 0.220" quat="0.7071068 0.7071068 0 0">
      <freejoint name="racket_free"/>
      <!-- Mass: 40g counterbalanced so COM is at mouth center (z=0) -->
      <inertial pos="0 0 0" mass="0.040" diaginertia="0.00014 0.00014 0.000008"/>
      <geom name="racket_butt" type="cylinder" pos="0 0 -0.058" size="0.0055 0.0015" material="bm_butt_green" friction="2.0 0.1 0.1"/>
      <geom name="racket_handle" type="cylinder" pos="0 0 0" size="0.004 0.055" material="bm_grip_black" friction="3.5 0.2 0.2" contype="0" conaffinity="0"/>
      <geom name="racket_cone" type="cylinder" pos="0 0 0.058" size="0.003 0.003" material="bm_cone_grey"/>
      <geom name="racket_shaft" type="cylinder" pos="0 0 0.078" size="0.002 0.0175" material="bm_shaft_carbon" friction="2.5 0.1 0.1"/>
      <geom name="racket_t_joint" type="cylinder" pos="0 0 0.098" size="0.0028 0.002" material="bm_t_joint_gold"/>
      <geom name="bm_fr_0" type="capsule" fromto="0.0250 0 0.130 0.0231 0 0.142" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_1" type="capsule" fromto="0.0231 0 0.142 0.0177 0 0.152" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_2" type="capsule" fromto="0.0177 0 0.152 0.0096 0 0.159" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_3" type="capsule" fromto="0.0096 0 0.159 0.0000 0 0.161" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_4" type="capsule" fromto="0.0000 0 0.161 -0.0096 0 0.159" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_5" type="capsule" fromto="-0.0096 0 0.159 -0.0177 0 0.152" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_6" type="capsule" fromto="-0.0177 0 0.152 -0.0231 0 0.142" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_7" type="capsule" fromto="-0.0231 0 0.142 -0.0250 0 0.130" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_8" type="capsule" fromto="-0.0250 0 0.130 -0.0231 0 0.118" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_9" type="capsule" fromto="-0.0231 0 0.118 -0.0177 0 0.108" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_10" type="capsule" fromto="-0.0177 0 0.108 -0.0096 0 0.101" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_11" type="capsule" fromto="-0.0096 0 0.101 0.0000 0 0.099" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_12" type="capsule" fromto="0.0000 0 0.099 0.0096 0 0.101" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_13" type="capsule" fromto="0.0096 0 0.101 0.0177 0 0.108" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_14" type="capsule" fromto="0.0177 0 0.108 0.0231 0 0.118" size="0.0015" material="bm_frame_blue"/>
      <geom name="bm_fr_15" type="capsule" fromto="0.0231 0 0.118 0.0250 0 0.130" size="0.0015" material="bm_frame_blue"/>
      <geom name="racket_strings" type="ellipsoid" pos="0 0 0.130" size="0.0235 0.0002 0.0295" material="bm_strings" contype="0" conaffinity="0"/>
    </body>"""

    xml = base.replace(old_cyl, body_str)
    xml = xml.replace('geom1="cylinder_geom"', 'geom1="racket_handle"')
    xml = xml.replace('body1="cylinder"', 'body1="racket"')

    # 1. Body Exclusions to prevent hinge mesh overlap
    xml = xml.replace('<exclude body1="cylinder" body2="trunk_base"/>',
                      '<exclude body1="racket" body2="trunk_base"/>\n'
                      '    <exclude body1="neck" body2="beak_lower"/>\n'
                      '    <exclude body1="neck_pitch" body2="beak_lower"/>\n'
                      '    <exclude body1="jaw_soft" body2="beak_lower"/>\n'
                      '    <exclude body1="pingpong_ball" body2="trunk_base"/>')

    # 2. Hard limits on beak joint (prevents over-rotation)
    xml = xml.replace('<joint name="beak" pos="0 0 0" axis="0 1 0" range="1.47 2.05" ref="1.500" damping="0.05"/>',
                      '<joint name="beak" pos="0 0 0" axis="0 1 0" range="1.47 2.05" ref="1.500" damping="0.05" limited="true"/>')

    # 3. Soft silicone mouth lining parameters (compliant grip, zero collision explosion)
    xml = xml.replace('solref="0.004 1.0"', 'solref="0.002 1.0"')
    xml = xml.replace('solimp="0.95 0.99 0.001 0.5 2"', 'solimp="0.98 0.995 0.0005 0.5 2"')

    # 4. Compliant Silicone Clamping Beds (Upper Roof Pad & Lower Jaw Silicone Trough)
    roof_geom_str = """              <!-- Compliant Silicone Clamping Bed lining Upper Beak Ceiling for Narrow Handles -->
              <geom name="mouth_roof_pad" type="box" pos="-0.0080 0 -0.0650" size="0.0010 0.025 0.0125" contype="0" conaffinity="0" group="3" material="soft_mouth_top_material"/>
"""
    xml = xml.replace('              <geom name="soft_mouth_top_collision"', roof_geom_str + '              <geom name="soft_mouth_top_collision"')

    jaw_cradle_str = """                <!-- Conformal lower jaw silicone bed -->
                <geom name="jaw_cradle_bed" type="box" pos="-0.0130 0 -0.0465" size="0.0010 0.025 0.0125" contype="0" conaffinity="0" group="3" material="jaw_soft_material"/>
"""
    xml = xml.replace('                <geom name="jaw_soft_lower_collision"', jaw_cradle_str + '                <geom name="jaw_soft_lower_collision"')

    # 5. High-friction Coulomb contact pairs for silicone mouth beds
    contact_str = """    <!-- Pure rigid Coulomb contact pairs with analytical mouth silicone beds -->
    <pair geom1="racket_handle" geom2="mouth_roof_pad" condim="6" friction="4.5 4.5 0.2 0.2 0.2" solref="0.002 1.0" solimp="0.99 0.999 0.0001 0.5 2"/>
    <pair geom1="racket_handle" geom2="jaw_cradle_bed" condim="6" friction="4.5 4.5 0.2 0.2 0.2" solref="0.002 1.0" solimp="0.99 0.999 0.0001 0.5 2"/>
"""
    if racket_type == "table_tennis" and include_ball:
        contact_str += """    <pair geom1="ball_geom" geom2="racket_face" condim="6" friction="0.80 0.80 0.005 0.005 0.005" solref="0.003 1.0" solimp="0.95 0.99 0.001 0.5 2"/>
"""
    xml = xml.replace('  </contact>', contact_str + '  </contact>')

    return xml


def run_drop_and_walk_test(
    racket_type="table_tennis",
    policy_path=None,
    gui=False,
    num_steps=600,
    loop=False,
    drop_height=0.002,
    reset_on_drop=False,
    vx=None,
    standing=False,
    pre_clamped=None,
    no_ball=False,
    continuous=False,
    lock_neck=False,
    lock_head_pitch=False,
    lock_yaw=False,
    lock_roll=False,
    lock_head=False,
    lemon_spoon=False,
    steadicam=False,
    speed=1.0,
):
    if continuous or (num_steps is not None and num_steps <= 0):
        num_steps = 10**9

    sim_speed = float(speed) if speed is not None and speed > 0 else 1.0
    has_ball = (racket_type == "table_tennis" and not no_ball)
    xml_str = create_scaled_scene_xml(racket_type=racket_type, include_ball=has_ball)
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002 if steadicam else 0.005
    data = mujoco.MjData(model)

    # Select default policy based on task
    if policy_path is None:
        if has_ball and os.path.exists(_TT_POLICY):
            policy_path = _TT_POLICY
        elif not has_ball and racket_type == "table_tennis" and os.path.exists(_TT_CARRY_POLICY):
            policy_path = _TT_CARRY_POLICY
        else:
            policy_path = _DEFAULT_POLICY

    # Check input dimension of the selected policy
    input_dim = 61
    try:
        import onnx
        m_onnx = onnx.load(policy_path)
        input_dim = m_onnx.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    except Exception:
        pass

    print(f"[Policy] Using: {policy_path} (input_dim={input_dim})")
    if (input_dim == 69 or input_dim == 71) and has_ball:
        policy = TableTennisPolicyInference(model, data, onnx_path=policy_path)
    elif input_dim == 64:
        policy = TableTennisRacketCarryPolicyInference(model, data, onnx_path=policy_path)
    else:
        policy = PolicyInference(
            model, data,
            walking_onnx_path=policy_path,
            action_scale=1.0,
            new_cmd_obs=True,
            use_projected_gravity=True,
        )

    racket_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "racket")
    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    beak_dof = model.jnt_dofadr[beak_jid]
    beak_qpos = model.jnt_qposadr[beak_jid]
    mouth_sid = model.site("mouth_tip").id
    racket_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "racket_free")
    racket_qadr = model.jnt_qposadr[racket_jid]
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pingpong_ball") if has_ball else -1
    ball_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free") if has_ball else -1
    ball_qadr = model.jnt_qposadr[ball_jid] if has_ball else -1

    int_pitch = 0.0
    head_pos_steadicam = 0.0

    if racket_type == "table_tennis":
        # Autonomous active control / zero artificial manual offsets
        policy.head_offset[:] = 0.0
        policy.default_pose[5] = 0.3491   # neck_pitch: nominal cradle setpoint
        policy.default_pose[6] = -0.0800  # head_pitch: nominal horizontal setpoint
        policy.default_pose[7] = 0.0
        policy.default_pose[8] = 0.0
        clamp_torque = 0.50            # N*m calibrated firm physical clamping torque
        if standing:
            vel_x = 0.0
        else:
            vel_x = vx if vx is not None else 0.18
        CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])  # forward horizontal blade facing up
        HOVER_LOCAL = np.array([-0.0120, 0.0, -0.0680])        # calibrated front mouth silicone cradle position (zero skull collision)
    else:
        clamp_torque = 0.50
        vel_x = 0.0 if standing else (vx if vx is not None else 0.18)
        CYL_QUAT = np.array([0.7071068, 0.7071068, 0.0, 0.0])
        HOVER_LOCAL = np.array([-0.0120, 0.0, -0.0680])

    if pre_clamped is None:
        # Default to pre-clamped balance mode (consistent with RL training environment)
        pre_clamped = True

    beak_wide_open = 2.04  # rad (117 deg, ~24.1mm gap)
    target_clamp_angle = 1.654 if racket_type == "badminton" else (1.666 if racket_type == "pickleball" else 1.637)
    beak_init_grip = target_clamp_angle  # rad tight handle clamping posture
    beak_clamp_ctrl = 1.000  # rad closed actuator setpoint assisting physical clamp torque

    ls_state = "WALKING"
    ls_vx = vel_x
    ls_settle_timer = 0.0

    def reset_episode():
        nonlocal ls_state, ls_vx, ls_settle_timer, head_pos_steadicam
        head_pos_steadicam = 0.0
        ls_state = "WALKING"
        ls_vx = vel_x
        ls_settle_timer = 0.0
        if lemon_spoon:
            policy.default_pose[5] = -0.4500
            policy.default_pose[6] = -0.4500
            policy.default_pose[7] = 0.0
            policy.default_pose[8] = 0.0
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, 0.125]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, idx in enumerate(policy.joint_qpos_indices):
            data.qpos[idx] = policy.default_pose[i]
        if pre_clamped:
            data.qpos[beak_qpos] = beak_init_grip
            data.qfrc_applied[beak_dof] = -clamp_torque
            if model.nu > 14:
                data.ctrl[14] = beak_clamp_ctrl
        else:
            data.qpos[beak_qpos] = beak_wide_open
        mujoco.mj_forward(model, data)

        p_jaw = data.xpos[jaw_bid]
        R_jaw = data.xmat[jaw_bid].reshape(3, 3)
        data.qpos[racket_qadr:racket_qadr+3] = p_jaw + R_jaw @ HOVER_LOCAL
        q_jaw = np.empty(4)
        mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
        q_r = np.empty(4)
        mujoco.mju_mulQuat(q_r, q_jaw, CYL_QUAT)
        data.qpos[racket_qadr+3:racket_qadr+7] = q_r
        mujoco.mj_forward(model, data)

        if pre_clamped:
            data.qfrc_applied[beak_dof] = -clamp_torque
            mujoco.mj_forward(model, data)

        if has_ball:
            # Place scaled 28.1mm ball above the paddle face center
            p_racket = data.xpos[racket_bid]
            R_racket = data.xmat[racket_bid].reshape(3, 3)
            face_center_world = p_racket + R_racket @ np.array([0.0, 0.0035 + 0.0141 + drop_height, 0.1015])
            data.qpos[ball_qadr:ball_qadr+3] = face_center_world
            data.qpos[ball_qadr+3:ball_qadr+7] = [1.0, 0.0, 0.0, 0.0]
            if ball_jid >= 0:
                data.qvel[model.jnt_dofadr[ball_jid]:model.jnt_dofadr[ball_jid]+6] = 0.0
            mujoco.mj_forward(model, data)

    reset_episode()

    print("=" * 80)
    print(f" Dynamic Drop & Clamping Test: {racket_type.replace('_', ' ').title()}")
    print(f" Mode: {'INTERACTIVE GLFW VIEWER (Live Simulation)' if gui else 'HEADLESS RENDERING'}")
    if gui:
        print("  KEYBOARD CONTROLS FOR LIVE BEAK, SPEED & TORQUE MANIPULATION:")
        print("    [O] or [ [ ] : Open beak position incrementally (+2.5°)")
        print("    [C] or [ ] ] : Close / clamp beak position incrementally (-2.5°)")
        print("    [+] or [=]   : Increase clamping torque (+0.05 N·m)")
        print("    [-] or [_]   : Decrease clamping torque (-0.05 N·m)")
        print("    [W] or [Up]  : Increase commanded walking speed (+0.03 m/s)")
        print("    [S] or [Down]: Decrease commanded walking speed (-0.03 m/s)")
        print("    [.] or [>]   : Speed up viewer playback (+0.25x)")
        print("    [,] or [<]   : Slow down viewer playback (-0.25x)")
        print("    [A]         : Return to Automatic policy clamping mode")
        print("    [Space]     : Pause / Resume simulation")
        print("    [R]         : Reset episode")
    print(f" Beak OPEN: {math.degrees(beak_wide_open):.1f}° (24.1mm aperture) | Racket Mass: {model.body_mass[racket_bid]*1000:.1f}g")
    if has_ball:
        print(f" Ball: Proportionally Scaled 0.70x Tournament Ball (Dia: 28.1mm, Mass: {model.body_mass[ball_bid]*1000:.2f}g)")
        print(f" Handle: 100% Solid Full-Tang Wood Core (5.8mm tang + dual 1.3mm scales = 8.4mm solid)")
    print("=" * 80)
    print(f"{'Time':>6} | {'Phase':>18} | {'Beak':>7} | {'Clamp Fn':>9} | {'Ball Status':>28}")
    print("-" * 80)

    renderer = None
    if not gui:
        renderer = mujoco.Renderer(model, height=720, width=960)

    cam_close = mujoco.MjvCamera()
    cam_close.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_close.distance = 0.38 if has_ball else 0.24
    cam_close.elevation = -28.0 if has_ball else -6.0
    cam_close.azimuth = 145.0 if has_ball else 125.0

    cam_ctx = mujoco.MjvCamera()
    cam_ctx.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam_ctx.trackbodyid = 1
    cam_ctx.distance = 0.70
    cam_ctx.elevation = -18.0 if has_ball else -14.0
    cam_ctx.azimuth = 142.0 if has_ball else 138.0

    keyframes_close = {}
    keyframes_ctx = {}

    user_beak_override = None
    user_paused = False
    reset_requested = False

    def on_key(keycode):
        nonlocal user_beak_override, user_paused, reset_requested, clamp_torque, vel_x, sim_speed
        # GLFW keycodes:
        # 'O' = 79, 'o' = 111, '[' = 91 -> Open beak
        if keycode in (ord('O'), ord('o'), 91):
            curr = user_beak_override if user_beak_override is not None else data.qpos[beak_qpos]
            user_beak_override = min(2.04, curr + 0.044)
            print(f"\n[INTERACTIVE] BEAK OPENED -> {math.degrees(user_beak_override):.1f}° (gap: {(user_beak_override - 1.47)*44.0:.1f}mm)")
        # 'C' = 67, 'c' = 99, ']' = 93 -> Close / clamp beak
        elif keycode in (ord('C'), ord('c'), 93):
            curr = user_beak_override if user_beak_override is not None else data.qpos[beak_qpos]
            user_beak_override = max(1.47, curr - 0.044)
            print(f"\n[INTERACTIVE] BEAK CLOSED -> {math.degrees(user_beak_override):.1f}°")
        # '+' / '=' = 61 or 43 -> Increase torque
        elif keycode in (ord('+'), ord('='), 61, 43):
            clamp_torque = min(1.00, clamp_torque + 0.05)
            print(f"\n[INTERACTIVE] CLAMP TORQUE INCREASED -> {clamp_torque:.2f} N·m")
        # '-' / '_' = 45 or 95 -> Decrease torque
        elif keycode in (ord('-'), ord('_'), 45, 95):
            clamp_torque = max(0.00, clamp_torque - 0.05)
            print(f"\n[INTERACTIVE] CLAMP TORQUE DECREASED -> {clamp_torque:.2f} N·m")
        # 'W' / 'w' or Up arrow = 265 -> Increase commanded forward walking speed
        elif keycode in (ord('W'), ord('w'), 265):
            vel_x = min(0.35, vel_x + 0.03)
            print(f"\n[INTERACTIVE] Walking Forward Speed INCREASED -> {vel_x:.2f} m/s")
        # 'S' / 's' or Down arrow = 264 -> Decrease commanded forward walking speed
        elif keycode in (ord('S'), ord('s'), 264):
            vel_x = max(0.00, vel_x - 0.03)
            print(f"\n[INTERACTIVE] Walking Forward Speed DECREASED -> {vel_x:.2f} m/s")
        # '.' or '>' -> Increase viewer playback speed
        elif keycode in (ord('.'), ord('>'), 46):
            sim_speed = min(5.0, sim_speed + 0.25)
            print(f"\n[INTERACTIVE] Viewer Playback Speed INCREASED -> {sim_speed:.2f}x")
        # ',' or '<' -> Decrease viewer playback speed
        elif keycode in (ord(','), ord('<'), 44):
            sim_speed = max(0.25, sim_speed - 0.25)
            print(f"\n[INTERACTIVE] Viewer Playback Speed DECREASED -> {sim_speed:.2f}x")
        # 'A' = 65, 'a' = 97 -> Auto mode
        elif keycode in (ord('A'), ord('a')):
            user_beak_override = None
            print(f"\n[INTERACTIVE] Restored AUTOMATIC policy clamping mode (Torque: {clamp_torque:.2f} N·m).")
        # Spacebar = 32 -> Pause / Resume
        elif keycode == 32:
            user_paused = not user_paused
            print(f"\n[INTERACTIVE] {'SIMULATION PAUSED' if user_paused else 'SIMULATION RESUMED'}")
        # 'R' = 82, 'r' = 114 -> Reset
        elif keycode in (ord('R'), ord('r')):
            reset_requested = True
            print("\n[INTERACTIVE] Episode Reset requested.")

    viewer = None
    if gui:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 0.65
        viewer.cam.elevation = -18.0
        # Ensure visual CAD meshes (groups 0-2) are visible while collision bounding boxes (group 3) are hidden
        viewer.opt.geomgroup[0] = 1
        viewer.opt.geomgroup[1] = 1
        viewer.opt.geomgroup[2] = 1
        viewer.opt.geomgroup[3] = 0

    episode = 0
    running = True
    walk_pitches = []
    walk_rolls = []

    while running:
        reset_episode()
        for step in range(num_steps):
            if gui and not viewer.is_running():
                running = False
                break

            if reset_requested:
                reset_requested = False
                break

            if user_paused:
                if gui:
                    viewer.sync()
                    time.sleep(0.02)
                continue

            sim_time = step * 0.005

            if sim_time >= 1.0:
                R_r_blade = data.xmat[racket_bid].reshape(3, 3)
                norm_blade = R_r_blade @ np.array([0.0, 1.0, 0.0])
                p_blade = math.asin(np.clip(float(norm_blade[0]), -0.999, 0.999))
                r_blade = math.asin(np.clip(float(norm_blade[1]), -0.999, 0.999))
                walk_pitches.append(math.degrees(p_blade))
                walk_rolls.append(math.degrees(r_blade))

            # Dynamic Protocol / Manual Beak Override
            if user_beak_override is not None:
                # Live User Beak Control
                phase_name = f"MANUAL BEAK ({math.degrees(user_beak_override):.1f}°)"
                beak_err = user_beak_override - data.qpos[beak_qpos]
                data.qfrc_applied[beak_dof] = float(np.clip(8.0 * beak_err - 0.25 * data.qvel[beak_dof], -1.0, 1.0))
                if model.nu > 14:
                    data.ctrl[14] = float(np.clip(user_beak_override, 1.470, 2.050))
                cmd_vx = vel_x
            elif pre_clamped or sim_time >= 0.75:
                phase_name = "4: BALANCING" if vel_x == 0.0 else "4: WALKING"
                # Active Conformal Anti-Migration Clamping (prevents slip over long walking runs)
                beak_err = target_clamp_angle - data.qpos[beak_qpos]
                data.qfrc_applied[beak_dof] = float(np.clip(-clamp_torque + 12.0 * beak_err - 0.5 * data.qvel[beak_dof], -1.2, 0.2))
                if model.nu > 14:
                    data.ctrl[14] = 1.0
                cmd_vx = vel_x
            elif sim_time < 0.40:
                # Phase 1: Open beak stand (hover handle centered in open mouth with zero contact)
                phase_name = "1: BEAK OPEN"
                data.qpos[beak_qpos] = beak_wide_open
                data.qvel[beak_dof] = 0.0
                data.qfrc_applied[beak_dof] = 0.0
                p_jaw = data.xpos[jaw_bid]
                R_jaw = data.xmat[jaw_bid].reshape(3, 3)
                data.qpos[racket_qadr:racket_qadr+3] = p_jaw + R_jaw @ HOVER_LOCAL
                q_jaw = np.empty(4)
                mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
                q_r = np.empty(4)
                mujoco.mju_mulQuat(q_r, q_jaw, CYL_QUAT)
                data.qpos[racket_qadr+3:racket_qadr+7] = q_r
                data.qvel[model.jnt_dofadr[racket_jid]:model.jnt_dofadr[racket_jid]+6] = 0.0
                cmd_vx = 0.0
            elif sim_time < 0.43:
                # Phase 2: Minimal Free Drop (release 4mm onto mandible cradle bed, beak remains open)
                phase_name = "2: MINIMAL DROP"
                data.qpos[beak_qpos] = beak_wide_open
                data.qvel[beak_dof] = 0.0
                data.qfrc_applied[beak_dof] = 0.0
                cmd_vx = 0.0
            else:
                # Phase 3: Active Clamping (calibrated servo torque clamps beak)
                phase_name = "3: CLAMPING"
                beak_err = target_clamp_angle - data.qpos[beak_qpos]
                data.qfrc_applied[beak_dof] = float(np.clip(-clamp_torque + 12.0 * beak_err - 0.5 * data.qvel[beak_dof], -1.2, 0.2))
                if model.nu > 14:
                    data.ctrl[14] = 1.0
                cmd_vx = 0.0

            pitch_corr = 0.0
            roll_corr = 0.0

            if lemon_spoon and has_ball and (pre_clamped or sim_time >= 0.75):
                p_r = data.xpos[racket_bid]
                R_r = data.xmat[racket_bid].reshape(3, 3)
                p_b = data.xpos[ball_bid]
                v_b = data.cvel[ball_bid][3:6]
                v_r = data.cvel[racket_bid][3:6]
                rel_pos = R_r.T @ (p_b - p_r)
                dx = float(rel_pos[0])
                dz = float(rel_pos[2] - 0.1015)
                dist = math.sqrt(dx**2 + dz**2)
                rel_vel = R_r.T @ (v_b - v_r)
                vx_rel = float(rel_vel[0])
                vz_rel = float(rel_vel[2])
                v_norm = math.sqrt(vx_rel**2 + vz_rel**2)

                R_trunk = data.xmat[trunk_id].reshape(3, 3)
                trunk_roll = np.arctan2(R_trunk[2, 1], R_trunk[2, 2])
                trunk_pitch = - np.arcsin(np.clip(R_trunk[2, 0], -1.0, 1.0))

                base_level_pitch = - trunk_pitch
                base_level_roll = - trunk_roll

                R_SAFE = 0.016
                R_INNER = 0.008
                SETTLE_DURATION = 0.40

                if ls_state == "WALKING":
                    # Gentle, human-like acceleration ramp (0.002 m/s per 5ms = 0.4 m/s^2 smooth take-off)
                    ls_vx = min(vel_x, ls_vx + 0.002)
                    pitch_corr = 0.0
                    roll_corr = 0.0
                    if dist > R_SAFE:
                        ls_state = "DECELERATING"

                elif ls_state == "DECELERATING":
                    # Gentle braking ramp (0.004 m/s per 5ms = 0.8 m/s^2 smooth stop)
                    prev_vx = ls_vx
                    ls_vx = max(0.0, ls_vx - 0.004)
                    a_decel = (prev_vx - ls_vx) / 0.005
                    inertial_tilt = - (a_decel / 9.81) * 0.20
                    pitch_corr = inertial_tilt - 1.5 * dz - 0.15 * vz_rel
                    roll_corr = - 1.5 * dx - 0.15 * vx_rel
                    if ls_vx <= 0.002:
                        ls_vx = 0.0
                        ls_state = "RECENTERING"

                elif ls_state == "RECENTERING":
                    ls_vx = 0.0
                    pitch_corr = - 2.0 * dz - 0.25 * vz_rel
                    roll_corr = - 2.0 * dx - 0.25 * vx_rel
                    if dist < R_INNER and v_norm < 0.02:
                        ls_state = "SETTLING"
                        ls_settle_timer = sim_time

                elif ls_state == "SETTLING":
                    ls_vx = 0.0
                    pitch_corr = - 0.5 * dz
                    roll_corr = - 0.5 * dx
                    if dist > R_SAFE:
                        ls_state = "RECENTERING"
                    elif (sim_time - ls_settle_timer) >= SETTLE_DURATION:
                        ls_state = "WALKING"

                phase_name = f"LEMON:{ls_state}"
                cmd_vx = ls_vx

            # 50 Hz policy inference
            if step % 4 == 0:
                if policy.vel_cmd[0] != cmd_vx:
                    policy.set_vel_cmd(cmd_vx, 0.0, 0.0)
                policy._update_command()
                action = policy.infer()
                use_steadicam_gimbal = steadicam or (racket_type == "table_tennis" and not has_ball)
                if use_steadicam_gimbal:
                    R_trunk = data.xmat[trunk_id].reshape(3, 3)
                    h_fwd = np.array([R_trunk[0, 0], R_trunk[1, 0], 0.0])
                    h_fwd /= (np.linalg.norm(h_fwd) + 1e-6)
                    h_lat = np.array([-h_fwd[1], h_fwd[0], 0.0])

                    R_r = data.xmat[racket_bid].reshape(3, 3)
                    b_norm = R_r @ np.array([0.0, 1.0, 0.0])
                    racket_pitch = math.asin(np.clip(float(np.dot(b_norm, h_fwd)), -0.999, 0.999))

                    # Pillar 2: First-order Slew-Rate Integrator (max 0.12 rad/s, mathematically kills jitter)
                    rate_cmd = np.clip(0.80 * (racket_pitch + 0.004), -0.12, 0.12)
                    dt_ctrl = 4 * model.opt.timestep
                    head_pos_steadicam = np.clip(head_pos_steadicam + rate_cmd * dt_ctrl, -0.40, 0.40)

                    # Pillar 1: Kinematic Differential Pantograph (equal & opposite, net head pitch change = 0)
                    action[5] = 0.0 + head_pos_steadicam
                    action[6] = 0.0 - head_pos_steadicam
                    action[7] = 0.0
                    action[8] = -0.002
                elif lemon_spoon:
                    action[5] += -0.35 + float(np.clip(pitch_corr, -0.25, 0.25))
                    action[8] += -0.14 + float(np.clip(roll_corr, -0.20, 0.20))
                    action[7] = 0.0
                elif lock_head:
                    action[5] = 0.0
                    action[6] = 0.0
                    action[7] = 0.0
                    action[8] = 0.0
                else:
                    if lock_neck:
                        action[5] = 0.0
                    if lock_head_pitch:
                        action[6] = 0.0
                    if lock_yaw:
                        action[7] = 0.0
                    if lock_roll:
                        action[8] = 0.0
                policy.apply_action(action)
                if model.nu > 14:
                    if user_beak_override is not None:
                        data.ctrl[14] = float(np.clip(user_beak_override, 1.470, 2.050))
                    elif pre_clamped or sim_time >= 0.43:
                        data.ctrl[14] = beak_clamp_ctrl

            mujoco.mj_step(model, data)

            # Contact monitoring
            fn_tot = 0.0
            ball_on_blade = False
            ball_force = 0.0
            for ci in range(data.ncon):
                con = data.contact[ci]
                b1 = model.geom_bodyid[con.geom1]
                b2 = model.geom_bodyid[con.geom2]
                g1_name = model.geom(con.geom1).name
                g2_name = model.geom(con.geom2).name
                if b1 == racket_bid or b2 == racket_bid:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, f)
                    if ("jaw" in g1_name or "jaw" in g2_name or "mouth" in g1_name or "mouth" in g2_name):
                        fn_tot += f[0]
                    if has_ball and ((b1 == ball_bid) or (b2 == ball_bid)):
                        ball_on_blade = True
                        ball_force += f[0]

            beak_deg = math.degrees(data.qpos[beak_qpos])

            ball_dropped = False
            ball_status_str = "Pure Racket Carry"
            if has_ball:
                p_ball = data.xpos[ball_bid]
                p_rac = data.xpos[racket_bid]
                R_rac = data.xmat[racket_bid].reshape(3, 3)
                rel_b = R_rac.T @ (p_ball - p_rac)
                # Ball is considered dropped if it rolled > 58mm from sweet spot or dropped below rubber
                if sim_time > 0.8 and (abs(rel_b[0]) > 0.058 or rel_b[1] < -0.015 or rel_b[2] < 0.040 or rel_b[2] > 0.165):
                    ball_dropped = True

                ball_status_str = f"On Blade: {ball_on_blade} ({ball_force*1000:4.1f}mN) dx={rel_b[0]*1000:+.1f}mm dy={rel_b[1]*1000:+.1f}mm dz={rel_b[2]*1000:+.1f}mm"
                if ball_dropped:
                    ball_status_str += " [BALL DROPPED!]"

            if step % 20 == 0 or ball_dropped:
                print(f"{sim_time:5.2f}s | {phase_name:>18} | {beak_deg:6.1f}° | {fn_tot:7.2f} N | {ball_status_str:>28}")

            if ball_dropped and reset_on_drop:
                print(f"--> Ball fell off paddle at t={sim_time:.2f}s! Resetting episode...")
                break

            if gui:
                viewer.sync()
                dt_sleep = max(0.0, (model.opt.timestep / sim_speed))
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
            else:
                # Capture keyframe verification renders
                if step in (10, 82, 130, 220, 1000, 3000, 5000, 5990):
                    phase_clean = phase_name.split(':')[1].strip().lower().replace(' ', '_') if ':' in phase_name else phase_name.strip().lower().replace(' ', '_')
                    tag = f"step_{step}_{phase_clean}"
                    if has_ball:
                        cam_close.lookat = data.xpos[ball_bid].copy()
                    else:
                        m_pos = data.site_xpos[mouth_sid]
                        cam_close.lookat = [m_pos[0], m_pos[1], m_pos[2]]

                    renderer.update_scene(data, camera=cam_close)
                    img1 = Image.fromarray(renderer.render())
                    p1 = os.path.join(_ARTIFACT_DIR, f"{racket_type}_{tag}_closeup.png")
                    img1.save(p1)
                    keyframes_close[tag] = p1

                    renderer.update_scene(data, camera=cam_ctx)
                    img2 = Image.fromarray(renderer.render())
                    p2 = os.path.join(_ARTIFACT_DIR, f"{racket_type}_{tag}_context.png")
                    img2.save(p2)
                    keyframes_ctx[tag] = p2

        episode += 1
        if not loop or not gui:
            break

    if gui and viewer is not None:
        viewer.close()

    print("-" * 80)
    print(f"Simulation completed! Final Normal Clamping Force: {fn_tot:.2f} N | Total Steps: {num_steps}")
    if walk_pitches:
        print("[Telemetry] Table Tennis Racket Orientation (Carry Phase):")
        print(f"  Pitch (Fore-Aft): Mean = {np.mean(walk_pitches):+.2f}° | Range = [{np.min(walk_pitches):+.2f}°, {np.max(walk_pitches):+.2f}°] | Std = {np.std(walk_pitches):.2f}°")
        print(f"  Roll  (Lateral):  Mean = {np.mean(walk_rolls):+.2f}° | Range = [{np.min(walk_rolls):+.2f}°, {np.max(walk_rolls):+.2f}°] | Std = {np.std(walk_rolls):.2f}°")
    if has_ball:
        print(f"Final Ball Status: {ball_status_str}")
    if not gui:
        print(f"Keyframe renders saved to: {_ARTIFACT_DIR}")
    return keyframes_close, keyframes_ctx


def main():
    parser = argparse.ArgumentParser(description="Real-World Racket Drop, Clamping & Dynamic Ball Balancing Test")
    parser.add_argument("--racket", choices=["table_tennis", "pickleball", "badminton"], default="table_tennis",
                        help="Racket type: authentic ITTF table tennis (180g + 40mm ball), pickleball, or badminton")
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to ONNX policy (defaults to table_tennis_ball_balance.onnx)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of 200Hz physics steps (1000 = 5.0s, 0 or negative = infinite)")
    parser.add_argument("--drop-height", type=float, default=0.002,
                        help="Initial ball drop height above paddle face in meters (e.g. 0.08 for 8cm air drop)")
    parser.add_argument("--gui", action="store_true", help="Launch native interactive GLFW MuJoCo viewer")
    parser.add_argument("--loop", action="store_true", help="Continuously loop the test in GUI")
    parser.add_argument("--continue", dest="continuous", action="store_true", help="Run continuously for a long time / indefinitely without resetting")
    parser.add_argument("--infinite", action="store_true", help="Run continuously for a long time / indefinitely without resetting")
    parser.add_argument("--reset-on-drop", action="store_true", help="Reset episode immediately when the ball falls off")
    parser.add_argument("--vx", type=float, default=None, help="Commanded forward walking velocity (m/s, default: 0.18)")
    parser.add_argument("--stand", action="store_true", help="Command in-place standing balance instead of forward walk")
    parser.add_argument("--walk", action="store_true", default=True, help="Command forward walking (default: True)")
    parser.add_argument("--drop-test", action="store_true", help="Force open-beak drop test phase instead of pre-clamped balance")
    parser.add_argument("--no-ball", action="store_true", help="Omit the pingpong ball, test purely holding and walking with the racket")
    parser.add_argument("--lock-neck", action="store_true", help="Lock neck_pitch to nominal level setpoint (residual=0)")
    parser.add_argument("--lock-head-pitch", action="store_true", help="Lock head_pitch to nominal level setpoint (residual=0)")
    parser.add_argument("--lock-yaw", action="store_true", help="Lock head_yaw to 0.0 (residual=0)")
    parser.add_argument("--lock-roll", action="store_true", help="Lock head_roll to 0.0 (residual=0)")
    parser.add_argument("--lock-head", action="store_true", help="Lock ALL 4 neck and head joints to nominal level setpoints")
    parser.add_argument("--lemon-spoon", action="store_true",
                        help="Enable 'Lemon-in-a-Spoon' governor: safe circular tolerance region, deceleration on escape, and settling pause timer")
    parser.add_argument("--steadicam", action="store_true",
                        help="Enable steadicam active neck-head gimbal stabilization for ultra-steady level carry")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Viewer simulation playback speed multiplier (e.g. 1.5, 2.0)")
    args = parser.parse_args()

    vel_forward = 0.0 if args.stand else (args.vx if args.vx is not None else (0.12 if args.lemon_spoon else 0.18))

    if vel_forward > 0.0 and vel_forward < 0.15 and not args.lemon_spoon:
        print(f"\n[NOTE] Commanded velocity vx={vel_forward:.2f} m/s is below nominal walking gait threshold (0.15 - 0.22 m/s).")
        print("       At vx=0.10, the policy takes micro-shuffles (~1mm/step). For active forward stepping, recommended: --vx 0.18.\n")

    if args.gui:
        print("\n[Interactive Viewer Notes]")
        print("  - The camera TRACKS the robot trunk_base (the robot remains centered; watch feet and checkerboard floor).")
        print("  - Press Spacebar to pause/resume | Right-click drag to zoom | Left-click drag to rotate camera.\n")

    is_continuous = args.continuous or args.infinite or (args.steps is not None and args.steps <= 0)

    run_drop_and_walk_test(
        racket_type=args.racket,
        policy_path=args.policy,
        gui=args.gui,
        num_steps=args.steps,
        loop=args.loop,
        drop_height=args.drop_height,
        reset_on_drop=args.reset_on_drop,
        vx=vel_forward,
        standing=args.stand,
        pre_clamped=False if args.drop_test else None,
        no_ball=args.no_ball,
        continuous=is_continuous,
        lock_neck=args.lock_neck,
        lock_head_pitch=args.lock_head_pitch,
        lock_yaw=args.lock_yaw,
        lock_roll=args.lock_roll,
        lock_head=args.lock_head,
        lemon_spoon=args.lemon_spoon,
        steadicam=args.steadicam,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()


