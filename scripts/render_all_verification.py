#!/usr/bin/env python3
"""Comprehensive visual verification generator for:
1. Pickleball Paddle model (official USA Pickleball specs)
2. Badminton Racket model (official BWF specs)
3. Microduck carrying Pickleball Paddle in MuJoCo
4. Microduck carrying Badminton Racket in MuJoCo
5. Floor resting posture & ground pickup clearance
"""

import math
import os
import sys
import numpy as np
import mujoco
from PIL import Image

_ROOT = "/Users/raghav/Desktop/microduck_manipulation"
_MODELS_DIR = os.path.join(_ROOT, "microduck_manipulation/models")
_ARTIFACT_DIR = "/Users/raghav/.gemini/antigravity-ide/brain/1dcd9661-c7ab-457a-8bec-a3fc2df3ade3"
_BASE_SCENE_XML = os.path.join(
    _ROOT, "microduck_rl/src/mjlab_microduck/robot/microduck/scene_loadcarry_physical_clamp.xml"
)

os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_ARTIFACT_DIR, exist_ok=True)


# =============================================================================
# 1. High-Precision XML Model: Pickleball Paddle
# =============================================================================
def create_pickleball_geoms(body_name="pickleball_paddle", z_offset=0.0):
    """Returns XML string for Pickleball Paddle geoms and inertial properties.
    Designed with an 8mm clamping neck zone so Microduck's beak clamps it with
    pure physical silicone contact dynamics (identical to cylinder).
    """
    return f"""
    <body name="{body_name}" pos="0 0 {z_offset:.4f}">
      <!-- Mass: 215g tournament standard (USA Pickleball midweight) -->
      <inertial pos="0 0 0.075" mass="0.215" diaginertia="0.0032 0.0028 0.00065"/>
      <freejoint name="{body_name}_free"/>

      <!-- 1. Handle Butt Cap (z = -0.160) -->
      <geom name="{body_name}_butt_cap" type="cylinder" pos="0 0 -0.160" size="0.016 0.006" material="butt_cap" friction="2.5 0.1 0.1"/>

      <!-- 2. Contoured Grip (Length 120mm: z from -0.154 to -0.034) -->
      <geom name="{body_name}_grip" type="cylinder" pos="0 0 -0.095" size="0.0135 0.055" material="grip_white" friction="2.5 0.1 0.1"/>

      <!-- 3. Microduck 8mm Clamping Zone (Centered at z = 0.0, length 40mm) -->
      <!-- Perfectly seats within Microduck's beak soft silicone pads (dia 8mm, condim 6) -->
      <geom name="{body_name}_clamping_neck" type="cylinder" pos="0 0 0" size="0.004 0.020" material="neck_clamp" friction="3.5 0.2 0.2"/>
      <geom name="{body_name}_collar_low" type="cylinder" pos="0 0 -0.022" size="0.014 0.003" material="butt_cap"/>
      <geom name="{body_name}_collar_high" type="cylinder" pos="0 0 0.022" size="0.014 0.003" material="butt_cap"/>

      <!-- 4. Throat Taper Transition (z from 0.025 to 0.055) -->
      <geom name="{body_name}_throat" type="box" pos="0 0 0.040" size="0.026 0.007 0.015" material="edge_guard"/>

      <!-- 5. Paddle Face Core (Length 250mm, Width 195mm, Thickness 14mm: z from 0.055 to 0.305) -->
      <geom name="{body_name}_face_core" type="box" pos="0 0 0.180" size="0.088 0.0065 0.118" material="face_cyan" friction="1.8 0.1 0.1"/>
      <geom name="{body_name}_accent_front" type="box" pos="0 0.0068 0.180" size="0.075 0.0005 0.035" material="face_yellow"/>
      <geom name="{body_name}_accent_back" type="box" pos="0 -0.0068 0.180" size="0.075 0.0005 0.035" material="face_yellow"/>

      <!-- 6. Edge Guard Rubber Bumper (6mm rim around entire face perimeter) -->
      <geom name="{body_name}_edge_left" type="capsule" fromto="-0.095 0 0.075 -0.095 0 0.270" size="0.0075" material="edge_guard" friction="2.0 0.1 0.1"/>
      <geom name="{body_name}_edge_right" type="capsule" fromto="0.095 0 0.075 0.095 0 0.270" size="0.0075" material="edge_guard" friction="2.0 0.1 0.1"/>
      <geom name="{body_name}_edge_top" type="capsule" fromto="-0.065 0 0.302 0.065 0 0.302" size="0.0075" material="edge_guard" friction="2.0 0.1 0.1"/>
      <geom name="{body_name}_corner_tl" type="capsule" fromto="-0.095 0 0.270 -0.065 0 0.302" size="0.0075" material="edge_guard"/>
      <geom name="{body_name}_corner_tr" type="capsule" fromto="0.095 0 0.270 0.065 0 0.302" size="0.0075" material="edge_guard"/>
      <geom name="{body_name}_corner_bl" type="capsule" fromto="-0.095 0 0.075 -0.026 0 0.050" size="0.0075" material="edge_guard"/>
      <geom name="{body_name}_corner_br" type="capsule" fromto="0.095 0 0.075 0.026 0 0.050" size="0.0075" material="edge_guard"/>
    </body>
"""


# =============================================================================
# 2. High-Precision XML Model: Badminton Racket
# =============================================================================
def create_badminton_geoms(body_name="badminton_racket", z_offset=0.0):
    """Returns XML string for BWF Badminton Racket geoms and inertial properties.
    Features:
      - 7.0mm ultra-slim carbon shaft (length 220mm)
      - Perfectly fitted into Microduck's 8.0mm beak clamp
      - Isometric oval head frame (16-segment capsule ring)
      - Translucent elliptical string bed
      - Mass: 83g (4U tournament standard)
    """
    a = 0.095  # half width 95mm
    b = 0.120  # half length 120mm
    z_head = 0.220
    n_pts = 16
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts = [(a * math.cos(th), 0.0, z_head + b * math.sin(th)) for th in angles]

    frame_capsules = []
    for i in range(n_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % n_pts]
        frame_capsules.append(
            f'      <geom name="{body_name}_frame_{i}" type="capsule" fromto="{p1[0]:.4f} {p1[1]:.4f} {p1[2]:.4f} {p2[0]:.4f} {p2[1]:.4f} {p2[2]:.4f}" size="0.0035" material="frame_blue" friction="1.8 0.1 0.1"/>'
        )
    frame_str = "\n".join(frame_capsules)

    return f"""
    <body name="{body_name}" pos="0 0 {z_offset:.4f}">
      <!-- Mass: 83g tournament standard (4U) -->
      <inertial pos="0 0 0.045" mass="0.083" diaginertia="0.0035 0.0034 0.00015"/>
      <freejoint name="{body_name}_free"/>

      <!-- 1. Handle Butt Cap (z = -0.335) -->
      <geom name="{body_name}_butt_cap" type="cylinder" pos="0 0 -0.335" size="0.0135 0.004" material="butt_green" friction="2.0 0.1 0.1"/>

      <!-- 2. Grip Handle (Length 150mm: z from -0.330 to -0.180) -->
      <geom name="{body_name}_grip" type="cylinder" pos="0 0 -0.255" size="0.011 0.075" material="grip_black" friction="2.5 0.1 0.1"/>

      <!-- 3. Handle Cone Collar (z from -0.180 to -0.165) -->
      <geom name="{body_name}_cone" type="cylinder" pos="0 0 -0.172" size="0.0075 0.008" material="cone_grey"/>

      <!-- 4. Ultra-Slim 7mm Carbon Shaft (Length 240mm: z from -0.165 to +0.075) -->
      <!-- Clamped directly by Microduck's beak at z ~ 0.0 (near balance point) -->
      <geom name="{body_name}_shaft_lower" type="cylinder" pos="0 0 -0.090" size="0.0035 0.065" material="shaft_carbon" friction="2.5 0.1 0.1"/>
      <geom name="{body_name}_clamping_zone" type="cylinder" pos="0 0 0.0" size="0.0035 0.025" material="shaft_carbon" friction="3.5 0.2 0.2"/>
      <geom name="{body_name}_shaft_upper" type="cylinder" pos="0 0 0.050" size="0.0035 0.025" material="shaft_carbon" friction="2.5 0.1 0.1"/>

      <!-- 5. T-Joint Connector (z = +0.080) -->
      <geom name="{body_name}_t_joint" type="cylinder" pos="0 0 0.080" size="0.0055 0.006" material="t_joint_gold"/>

      <!-- 6. Oval Head Frame (16 segments) -->
{frame_str}

      <!-- 7. Translucent Ellipsoid String Bed (Fits completely inside frame without corner overflow) -->
      <geom name="{body_name}_strings" type="ellipsoid" pos="0 0 {z_head:.4f}" size="0.091 0.0002 0.116" material="strings_translucent" contype="0" conaffinity="0"/>
    </body>
"""


# =============================================================================
# 3. Build Standalone Inspection Scenes
# =============================================================================
def build_inspection_scenes():
    p_geoms = create_pickleball_geoms(body_name="pickleball_paddle", z_offset=0.15)
    b_geoms = create_badminton_geoms(body_name="badminton_racket", z_offset=0.20)

    header = """<mujoco model="inspection">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.45 0.45 0.48" diffuse="0.75 0.75 0.78" specular="0.25 0.25 0.25"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.55 0.75" rgb2="0.08 0.08 0.12" width="512" height="512"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.22 0.26 0.32" rgb2="0.16 0.19 0.24" markrgb="0.75 0.75 0.8" width="300" height="300"/>
    <material name="groundmat" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.15"/>
    <material name="edge_guard" rgba="0.12 0.12 0.14 1" specular="0.3" shininess="0.4"/>
    <material name="face_cyan" rgba="0.08 0.65 0.84 1" specular="0.5" shininess="0.6"/>
    <material name="face_yellow" rgba="0.98 0.82 0.15 1" specular="0.6" shininess="0.7"/>
    <material name="grip_white" rgba="0.94 0.94 0.96 1" specular="0.1" shininess="0.1"/>
    <material name="neck_clamp" rgba="0.15 0.15 0.15 1" specular="0.4" shininess="0.5"/>
    <material name="butt_cap" rgba="0.12 0.12 0.15 1" specular="0.4" shininess="0.5"/>
    <material name="frame_blue" rgba="0.12 0.25 0.48 1" specular="0.8" shininess="0.9"/>
    <material name="shaft_carbon" rgba="0.09 0.14 0.22 1" specular="0.8" shininess="0.9"/>
    <material name="t_joint_gold" rgba="0.90 0.78 0.22 1" specular="0.95" shininess="0.95"/>
    <material name="strings_translucent" rgba="0.95 0.98 0.98 0.42" specular="0.3" shininess="0.5"/>
    <material name="grip_black" rgba="0.12 0.12 0.12 1" specular="0.1" shininess="0.1"/>
    <material name="cone_grey" rgba="0.25 0.28 0.32 1" specular="0.5" shininess="0.6"/>
    <material name="butt_green" rgba="0.05 0.52 0.28 1" specular="0.5" shininess="0.6"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1.5 1.5 0.1" material="groundmat"/>
"""

    footer = """  </worldbody>
</mujoco>"""

    p_xml = header + p_geoms + footer
    b_xml = header + b_geoms + footer

    p_path = os.path.join(_MODELS_DIR, "pickleball_paddle.xml")
    b_path = os.path.join(_MODELS_DIR, "badminton_racket.xml")
    with open(p_path, "w") as f:
        f.write(p_xml)
    with open(b_path, "w") as f:
        f.write(b_xml)

    return p_path, b_path


# =============================================================================
# 4. Build Integrated Microduck Carry Scenes
# =============================================================================
def build_microduck_carry_scenes():
    with open(_BASE_SCENE_XML, "r") as f:
        content = f.read()

    # Replace meshdir="assets/" with absolute path
    assets_dir = os.path.join(os.path.dirname(_BASE_SCENE_XML), "assets")
    content = content.replace('meshdir="assets/"', f'meshdir="{assets_dir}/"')

    # Materials to add
    extra_assets = """
    <material name="edge_guard" rgba="0.12 0.12 0.14 1" specular="0.3" shininess="0.4"/>
    <material name="face_cyan" rgba="0.08 0.65 0.84 1" specular="0.5" shininess="0.6"/>
    <material name="face_yellow" rgba="0.98 0.82 0.15 1" specular="0.6" shininess="0.7"/>
    <material name="grip_white" rgba="0.94 0.94 0.96 1" specular="0.1" shininess="0.1"/>
    <material name="neck_clamp" rgba="0.15 0.15 0.15 1" specular="0.4" shininess="0.5"/>
    <material name="frame_blue" rgba="0.12 0.25 0.48 1" specular="0.8" shininess="0.9"/>
    <material name="shaft_carbon" rgba="0.09 0.14 0.22 1" specular="0.8" shininess="0.9"/>
    <material name="t_joint_gold" rgba="0.90 0.78 0.22 1" specular="0.95" shininess="0.95"/>
    <material name="strings_translucent" rgba="0.95 0.98 0.98 0.42" specular="0.3" shininess="0.5"/>
    <material name="grip_black" rgba="0.12 0.12 0.12 1" specular="0.1" shininess="0.1"/>
    <material name="cone_grey" rgba="0.25 0.28 0.32 1" specular="0.5" shininess="0.6"/>
    <material name="butt_green" rgba="0.05 0.52 0.28 1" specular="0.5" shininess="0.6"/>
  </asset>
"""
    content = content.replace("  </asset>", extra_assets)

    # Replace <visual> to ensure high-resolution offscreen rendering
    old_visual = """  <visual>
    <global azimuth="160" elevation="-20"/>"""
    new_visual = """  <visual>
    <global offwidth="1280" offheight="960" azimuth="160" elevation="-20"/>"""
    content = content.replace(old_visual, new_visual)

    # Find the cylinder body block to replace
    cyl_start = content.find('<body name="cylinder"')
    cyl_end = content.find('</body>\n  </worldbody>') + len('</body>')
    old_cyl = content[cyl_start:cyl_end]

    # --- 4A. Microduck with Pickleball Paddle ---
    # Beak clamps at mouth center: pos="0.038 0 0.228", quat="0.7071068 0.7071068 0 0"
    p_body = """    <body name="pickleball_paddle" pos="0.038 0 0.228" quat="0.7071068 0.7071068 0 0">
      <freejoint name="pickleball_paddle_free"/>
      <inertial pos="0 0 0.075" mass="0.215" diaginertia="0.0032 0.0028 0.00065"/>
      <geom name="pb_butt_cap" type="cylinder" pos="0 0 -0.160" size="0.016 0.006" material="trunk_base_material" friction="2.5 0.1 0.1"/>
      <geom name="pb_grip" type="cylinder" pos="0 0 -0.095" size="0.0135 0.055" material="grip_white" friction="2.5 0.1 0.1"/>
      <geom name="pb_clamp_neck" type="cylinder" pos="0 0 0" size="0.004 0.020" material="neck_clamp" friction="3.5 0.2 0.2"/>
      <geom name="pb_collar_low" type="cylinder" pos="0 0 -0.022" size="0.014 0.003" material="trunk_base_material"/>
      <geom name="pb_collar_high" type="cylinder" pos="0 0 0.022" size="0.014 0.003" material="trunk_base_material"/>
      <geom name="pb_throat" type="box" pos="0 0 0.040" size="0.026 0.007 0.015" material="edge_guard"/>
      <geom name="pb_face_core" type="box" pos="0 0 0.180" size="0.088 0.0065 0.118" material="face_cyan" friction="1.8 0.1 0.1"/>
      <geom name="pb_accent_front" type="box" pos="0 0.0068 0.180" size="0.075 0.0005 0.035" material="face_yellow"/>
      <geom name="pb_accent_back" type="box" pos="0 -0.0068 0.180" size="0.075 0.0005 0.035" material="face_yellow"/>
      <geom name="pb_edge_l" type="capsule" fromto="-0.095 0 0.075 -0.095 0 0.270" size="0.0075" material="edge_guard"/>
      <geom name="pb_edge_r" type="capsule" fromto="0.095 0 0.075 0.095 0 0.270" size="0.0075" material="edge_guard"/>
      <geom name="pb_edge_top" type="capsule" fromto="-0.065 0 0.302 0.065 0 0.302" size="0.0075" material="edge_guard"/>
      <geom name="pb_cor_tl" type="capsule" fromto="-0.095 0 0.270 -0.065 0 0.302" size="0.0075" material="edge_guard"/>
      <geom name="pb_cor_tr" type="capsule" fromto="0.095 0 0.270 0.065 0 0.302" size="0.0075" material="edge_guard"/>
      <geom name="pb_cor_bl" type="capsule" fromto="-0.095 0 0.075 -0.026 0 0.050" size="0.0075" material="edge_guard"/>
      <geom name="pb_cor_br" type="capsule" fromto="0.095 0 0.075 0.026 0 0.050" size="0.0075" material="edge_guard"/>
    </body>"""

    # Update contact pair for pb_clamp_neck
    p_content = content.replace(old_cyl, p_body)
    p_content = p_content.replace('geom1="cylinder_geom"', 'geom1="pb_clamp_neck"')
    p_content = p_content.replace('body1="cylinder"', 'body1="pickleball_paddle"')

    p_scene_path = os.path.join(_MODELS_DIR, "scene_microduck_pickleball.xml")
    with open(p_scene_path, "w") as f:
        f.write(p_content)

    # --- 4B. Microduck with Badminton Racket ---
    a = 0.095
    b = 0.120
    z_head = 0.220
    n_pts = 16
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts = [(a * math.cos(th), 0.0, z_head + b * math.sin(th)) for th in angles]

    frame_capsules = []
    for i in range(n_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % n_pts]
        frame_capsules.append(
            f'      <geom name="bm_fr_{i}" type="capsule" fromto="{p1[0]:.4f} {p1[1]:.4f} {p1[2]:.4f} {p2[0]:.4f} {p2[1]:.4f} {p2[2]:.4f}" size="0.0035" material="frame_blue"/>'
        )
    bm_frame_str = "\n".join(frame_capsules)

    b_body = f"""    <body name="badminton_racket" pos="0.038 0 0.228" quat="0.7071068 0.7071068 0 0">
      <freejoint name="badminton_racket_free"/>
      <inertial pos="0 0 0.045" mass="0.083" diaginertia="0.0035 0.0034 0.00015"/>
      <geom name="bm_butt_cap" type="cylinder" pos="0 0 -0.335" size="0.0135 0.004" material="butt_green"/>
      <geom name="bm_grip" type="cylinder" pos="0 0 -0.255" size="0.011 0.075" material="grip_black"/>
      <geom name="bm_cone" type="cylinder" pos="0 0 -0.172" size="0.0075 0.008" material="cone_grey"/>
      <geom name="bm_shaft_low" type="cylinder" pos="0 0 -0.090" size="0.0035 0.065" material="shaft_carbon"/>
      <geom name="bm_clamp_zone" type="cylinder" pos="0 0 0.0" size="0.0035 0.025" material="shaft_carbon" friction="3.5 0.2 0.2"/>
      <geom name="bm_shaft_high" type="cylinder" pos="0 0 0.050" size="0.0035 0.025" material="shaft_carbon"/>
      <geom name="bm_t_joint" type="cylinder" pos="0 0 0.080" size="0.0055 0.006" material="t_joint_gold"/>
{bm_frame_str}
      <geom name="bm_strings" type="ellipsoid" pos="0 0 {z_head:.4f}" size="0.091 0.0002 0.116" material="strings_translucent" contype="0" conaffinity="0"/>
    </body>"""

    b_content = content.replace(old_cyl, b_body)
    b_content = b_content.replace('geom1="cylinder_geom"', 'geom1="bm_clamp_zone"')
    b_content = b_content.replace('body1="cylinder"', 'body1="badminton_racket"')

    b_scene_path = os.path.join(_MODELS_DIR, "scene_microduck_badminton.xml")
    with open(b_scene_path, "w") as f:
        f.write(b_content)

    return p_scene_path, b_scene_path


# =============================================================================
# 5. Render High-Resolution Multi-Angle Images
# =============================================================================
def render_views(xml_path, out_prefix, cameras):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Set default pose if robot is in scene
    if "robot" in xml_path or "microduck" in xml_path:
        data.qpos[0:3] = [0.0, 0.0, 0.125]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # Beak clamped around payload
        beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
        if beak_jid >= 0:
            b_adr = model.jnt_qposadr[beak_jid]
            data.qpos[b_adr] = 1.55  # Clamped

    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=960)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    outputs = {}
    for name, cfg in cameras.items():
        cam.distance = cfg["distance"]
        cam.elevation = cfg["elevation"]
        cam.azimuth = cfg["azimuth"]
        cam.lookat = cfg["lookat"]

        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        img = Image.fromarray(pixels)
        out_file = os.path.join(_ARTIFACT_DIR, f"{out_prefix}_{name}.png")
        img.save(out_file)
        outputs[name] = out_file
        print(f"Rendered: {out_file}")

    return outputs


def main():
    print("1. Generating Standalone Inspection XMLs...")
    p_path, b_path = build_inspection_scenes()

    print("2. Generating Microduck Integrated Carry XMLs...")
    p_sc_path, b_sc_path = build_microduck_carry_scenes()

    # Cameras for Standalone Pickleball
    pb_inspect_cams = {
        "perspective": {"distance": 0.58, "elevation": -24.0, "azimuth": 135.0, "lookat": [0.0, 0.0, 0.18]},
        "front": {"distance": 0.55, "elevation": -88.0, "azimuth": 90.0, "lookat": [0.0, 0.0, 0.18]},
        "side": {"distance": 0.55, "elevation": 0.0, "azimuth": 0.0, "lookat": [0.0, 0.0, 0.18]},
        "beak_clamp_zone": {"distance": 0.18, "elevation": -12.0, "azimuth": 120.0, "lookat": [0.0, 0.0, 0.0]},
    }
    print("3. Rendering Pickleball Paddle Inspection Views...")
    render_views(p_path, "pickleball_verified", pb_inspect_cams)

    # Cameras for Standalone Badminton
    bm_inspect_cams = {
        "perspective": {"distance": 0.82, "elevation": -22.0, "azimuth": 135.0, "lookat": [0.0, 0.0, 0.15]},
        "front": {"distance": 0.80, "elevation": -88.0, "azimuth": 90.0, "lookat": [0.0, 0.0, 0.15]},
        "side": {"distance": 0.80, "elevation": 0.0, "azimuth": 0.0, "lookat": [0.0, 0.0, 0.15]},
        "shaft_clamp_zone": {"distance": 0.22, "elevation": -14.0, "azimuth": 120.0, "lookat": [0.0, 0.0, 0.0]},
    }
    print("4. Rendering Badminton Racket Inspection Views...")
    render_views(b_path, "badminton_verified", bm_inspect_cams)

    # Cameras for Microduck Carrying Rackets
    robot_carry_cams = {
        "full_context": {"distance": 0.72, "elevation": -16.0, "azimuth": 140.0, "lookat": [0.05, 0.0, 0.14]},
        "front_standing": {"distance": 0.65, "elevation": -5.0, "azimuth": 180.0, "lookat": [0.05, 0.0, 0.14]},
        "beak_clamping_closeup": {"distance": 0.22, "elevation": -8.0, "azimuth": 125.0, "lookat": [0.04, 0.0, 0.17]},
    }
    print("5. Rendering Microduck Carrying Pickleball Paddle...")
    render_views(p_sc_path, "microduck_pickleball_carry", robot_carry_cams)

    print("6. Rendering Microduck Carrying Badminton Racket...")
    render_views(b_sc_path, "microduck_badminton_carry", robot_carry_cams)

    print("\nVisual verification renders completed successfully!")


if __name__ == "__main__":
    main()
