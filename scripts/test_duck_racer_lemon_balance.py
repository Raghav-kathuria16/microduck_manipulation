#!/usr/bin/env python3
"""Microduck 5-Duck Multi-Agent Lemon-and-Spoon Racer Simulation.

Inspired by Genie Sim 3.0 (arXiv:2601.02078v4):
- High-fidelity visual asset generation & multi-robot domain randomization.
- 5 parallel racing lanes with official Pollen Robotics colorways:
    Lane 1: Cream      (Warm Ivory / Apricot Beak)
    Lane 2: Sky        (Powder Blue / Sunset Orange Beak)
    Lane 3: Duck Gold  (Classic Sunflower Yellow / Tangerine Beak)
    Lane 4: Lavender   (Pastel Lilac / Coral Beak)
    Lane 5: Graphite   (Matte Slate Charcoal / Neon Hazard Beak)
- Real-world physics: 34 N beak silicone clamp, 25g stainless spoon, 22g citrus lemon.
- 50 Hz closed-loop locomotion policy with active Steadicam pantographs.
- Live race telemetry, lane splits, finish line detection, and dynamic camera modes.
"""

import argparse
import math
import os
import re
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image
import onnxruntime as ort

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from infer_policy import DEFAULT_POSE

_BASE_SCENE_XML = os.path.join(_MANIP_DIR, "models/scene_loadcarry_physical_clamp.xml")
_DEFAULT_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/microduck_spoon_lemon_steadicam_production.onnx")
_OUTPUT_DIR = os.path.join(_MANIP_DIR, "docs/media")
os.makedirs(_OUTPUT_DIR, exist_ok=True)
_ARTIFACT_DIR = "/Users/raghav/.gemini/antigravity-ide/brain/01375344-3ce7-4a6f-afec-3cecca1c8b8d"

# 5 Official Pollen Robotics Inspired Colorways
DUCK_CONFIGS = [
    {
        "id": 0,
        "name": "Coral",
        "lane": 1,
        "y": -0.60,
        "shell_rgba": "0.88 0.08 0.10 1",      # Vivid Ruby Coral
        "head_top_rgba": "0.88 0.08 0.10 1",   # Matching Saturated Ruby Coral
        "beak_rgba": "1.00 0.72 0.00 1",       # Canary Amber Beak
        "foot_rgba": "1.00 0.72 0.00 1",
    },
    {
        "id": 1,
        "name": "Riviera Sky",
        "lane": 2,
        "y": -0.30,
        "shell_rgba": "0.02 0.45 0.88 1",      # Electric Sapphire Riviera Cyan
        "head_top_rgba": "0.02 0.45 0.88 1",   # Matching Vivid Cyan
        "beak_rgba": "1.00 0.48 0.00 1",       # Sunset Tangerine Beak
        "foot_rgba": "1.00 0.48 0.00 1",
    },
    {
        "id": 2,
        "name": "Duck Gold",
        "lane": 3,
        "y": 0.00,
        "shell_rgba": "0.98 0.72 0.00 1",      # Brilliant Sunflower Duck Gold
        "head_top_rgba": "0.98 0.72 0.00 1",   # Matching Luminous Gold
        "beak_rgba": "1.00 0.32 0.00 1",       # Deep Tangerine Beak
        "foot_rgba": "1.00 0.32 0.00 1",
    },
    {
        "id": 3,
        "name": "Emerald",
        "lane": 4,
        "y": 0.30,
        "shell_rgba": "0.04 0.62 0.22 1",      # Vivid Emerald Racing Green
        "head_top_rgba": "0.04 0.62 0.22 1",   # Matching Rich Emerald Green
        "beak_rgba": "1.00 0.60 0.00 1",       # Golden Amber Beak
        "foot_rgba": "1.00 0.60 0.00 1",
    },
    {
        "id": 4,
        "name": "Amethyst",
        "lane": 5,
        "y": 0.60,
        "shell_rgba": "0.52 0.10 0.78 1",      # Rich Royal Amethyst Purple
        "head_top_rgba": "0.52 0.10 0.78 1",   # Matching Royal Violet
        "beak_rgba": "1.00 0.50 0.00 1",       # Coral Gold Beak
        "foot_rgba": "1.00 0.50 0.00 1",
    },
]

JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"
]


def create_racer_scene_xml():
    """Procedurally generates a complete multi-duck racing stadium scene in MJCF format."""
    with open(_BASE_SCENE_XML, "r") as f:
        base = f.read()

    assets_dir = os.path.join(_MANIP_DIR, "models/assets")
    base = re.sub(r'meshdir="[^"]*"', f'meshdir="{assets_dir}/" balanceinertia="true"', base)
    base = base.replace('elevation="-20"/>', 'elevation="-20" offwidth="1280" offheight="960"/>')

    # Add single-source CAD mesh and materials
    extra_assets = """
    <!-- CAD Mesh for Tableware Spoon -->
    <mesh name="spoon_cad_mesh" file="spoon.stl"/>

    <material name="stainless_steel" rgba="0.84 0.86 0.90 1" specular="0.85" shininess="0.9" reflectance="0.3"/>
    <material name="citrus_lemon" rgba="0.98 0.88 0.12 1" specular="0.35" shininess="0.35"/>
    <material name="lemon_tip" rgba="0.72 0.82 0.18 1" specular="0.2" shininess="0.2"/>
    <material name="start_line_mat" rgba="1.0 1.0 1.0 0.95" specular="0.4" shininess="0.5"/>
    <material name="finish_line_mat" rgba="1.0 0.84 0.0 0.95" specular="0.8" shininess="0.9"/>
    <material name="barrier_mat" rgba="0.18 0.20 0.24 1" specular="0.5" shininess="0.6"/>
    <!-- Authentic Olympic Tartan Red Running Track Material (Tokyo/Paris Mondo terracotta brick red) -->
    <material name="red_track_mat" rgba="0.22 0.05 0.04 1.0" specular="0.08" shininess="0.1" reflectance="0.005"/>
"""

    # Add custom color materials for 5 ducks
    for cfg in DUCK_CONFIGS:
        p = f"d{cfg['id']}_"
        extra_assets += f"""
    <!-- Duck {cfg['id']}: {cfg['name']} Color Palette -->
    <material name="{p}shell_mat" rgba="{cfg['shell_rgba']}" specular="0.35" shininess="0.55"/>
    <material name="{p}head_top_mat" rgba="{cfg['head_top_rgba']}" specular="0.35" shininess="0.55"/>
    <material name="{p}beak_mat" rgba="{cfg['beak_rgba']}" specular="0.3" shininess="0.5"/>
    <material name="{p}foot_mat" rgba="{cfg['foot_rgba']}" specular="0.3" shininess="0.5"/>
"""
    extra_assets += "  </asset>\n"
    base = base.replace("  </asset>", extra_assets)

    # Replace blue checker texture & groundplane material with sleek stadium slate floor
    base = re.sub(
        r'<material name="groundplane"[^/]*/>',
        '<material name="groundplane" rgba="0.14 0.16 0.18 1" specular="0.08" shininess="0.1" reflectance="0.0"/>',
        base
    )
    base = re.sub(
        r'<texture[^>]*name="groundplane"[^/]*/>',
        '<texture type="2d" colorspace="auto" name="groundplane" builtin="flat" rgb1="0.14 0.16 0.18" rgb2="0.14 0.16 0.18" width="300" height="300"/>',
        base
    )

    # Base templates
    asset_end = base.find("</asset>") + len("</asset>")
    header_assets = base[:asset_end]

    trunk_start = base.find('<body name="trunk_base"')
    trunk_end = base.find('<body name="cylinder"')
    trunk_template = base[trunk_start:trunk_end].strip()

    # Add silicone mouth pads & joint limits to trunk template
    trunk_template = trunk_template.replace(
        '<joint name="beak" pos="0 0 0" axis="0 1 0" range="1.47 2.05" ref="1.500" damping="0.05"/>',
        '<joint name="beak" pos="0 0 0" axis="0 1 0" range="1.47 2.05" ref="1.500" damping="0.05" limited="true"/>'
    )
    roof_pad = """              <geom name="mouth_roof_pad" type="box" pos="-0.0080 0 -0.0650" size="0.0010 0.025 0.0125" contype="0" conaffinity="0" group="3" material="soft_mouth_top_material"/>
"""
    trunk_template = trunk_template.replace('              <geom name="soft_mouth_top_collision"', roof_pad + '              <geom name="soft_mouth_top_collision"')
    jaw_bed = """                <geom name="jaw_cradle_bed" type="box" pos="-0.0130 0 -0.0465" size="0.0010 0.025 0.0125" contype="0" conaffinity="0" group="3" material="jaw_soft_material"/>
"""
    trunk_template = trunk_template.replace('                <geom name="jaw_soft_lower_collision"', jaw_bed + '                <geom name="jaw_soft_lower_collision"')

    spoon_lemon_template = """    <!-- Tableware Spoon Body (25g stainless steel, length 125mm) -->
    <body name="spoon" pos="0.048 0.0 0.220" quat="0.7071068 0.7071068 0 0">
      <freejoint name="spoon_free"/>
      <inertial pos="0 -0.001 0.028" mass="0.025" diaginertia="0.000028 0.000024 0.000006"/>
      <geom name="spoon_vis" type="mesh" mesh="spoon_cad_mesh" material="stainless_steel" contype="0" conaffinity="0" group="2"/>
      <geom name="spoon_handle" type="box" pos="0 0 0.008" size="0.0040 0.0011 0.035" material="stainless_steel" friction="0.55 0.05 0.05" contype="0" conaffinity="0" group="3"/>
      <geom name="spoon_bowl_base" type="ellipsoid" pos="0 -0.0086 0.070" size="0.012 0.0010 0.018" material="stainless_steel" friction="0.50 0.01 0.01" contype="0" conaffinity="0" group="3"/>
      <geom name="rim_front" type="capsule" fromto="-0.007 -0.0020 0.088  0.007 -0.0020 0.088" size="0.0018" material="stainless_steel" friction="0.50 0.01 0.01" contype="0" conaffinity="0" group="3"/>
      <geom name="rim_back"  type="capsule" fromto="-0.006 -0.0022 0.052  0.006 -0.0022 0.052" size="0.0018" material="stainless_steel" friction="0.50 0.01 0.01" contype="0" conaffinity="0" group="3"/>
      <geom name="rim_left_m"  type="capsule" fromto="-0.0135 -0.0022 0.070 -0.007 -0.0020 0.088" size="0.0018" material="stainless_steel" friction="0.50 0.01 0.01" contype="0" conaffinity="0" group="3"/>
      <geom name="rim_right_m" type="capsule" fromto=" 0.0135 -0.0020 0.070  0.007 -0.0020 0.088" size="0.0018" material="stainless_steel" friction="0.50 0.01 0.01" contype="0" conaffinity="0" group="3"/>
      <geom name="rim_left_b"  type="capsule" fromto="-0.006 -0.0022 0.052 -0.0135 -0.0022 0.070" size="0.0018" material="stainless_steel" friction="0.50 0.01 0.01" contype="0" conaffinity="0" group="3"/>
      <geom name="rim_right_b" type="capsule" fromto=" 0.006 -0.0022 0.052  0.0135 -0.0020 0.070" size="0.0018" material="stainless_steel" friction="0.50 0.01 0.01" contype="0" conaffinity="0" group="3"/>
    </body>

    <!-- Smooth Spherical Lemon Ball (22g pure sphere, 24mm dia, r=0.012m, no tips/nubs) -->
    <body name="lemon" pos="0.050 0.065 0.220">
      <freejoint name="lemon_free"/>
      <inertial pos="0 0 0" mass="0.022" diaginertia="0.00000127 0.00000127 0.00000127"/>
      <geom name="lemon_geom" type="sphere" size="0.012" material="citrus_lemon" friction="0.50 0.01 0.01" condim="6"/>
    </body>"""

    actuator_start = base.find("<actuator>")
    actuator_end = base.find("</actuator>") + len("</actuator>")
    actuator_template = base[actuator_start:actuator_end]

    sensor_start = base.find("<sensor>")
    sensor_end = base.find("</sensor>") + len("</sensor>")
    sensor_template = base[sensor_start:sensor_end]

    # Professional 5-Lane Racing Track Elements
    track_elements = """
    <!-- 5-Lane Professional Racing Arena & Stadium Track -->
    <!-- Dedicated Olympic Tartan Red Track Bed (spans from x=-4.0m to x=+8.0m, enclosed between barriers) -->
    <geom name="track_bed_red" type="box" pos="2.0 0.0 0.0005" size="6.0 0.88 0.0005" material="red_track_mat" contype="0" conaffinity="0" group="2"/>

    <!-- Lane Divider Lines (white stripes along the track extending from x=-4.0m to x=+8.0m) -->
    <geom name="lane_div_0" type="box" pos="2.0 -0.75 0.001" size="6.0 0.012 0.001" rgba="0.95 0.95 0.95 0.9" contype="0" conaffinity="0" group="2"/>
    <geom name="lane_div_1" type="box" pos="2.0 -0.45 0.001" size="6.0 0.012 0.001" rgba="0.95 0.95 0.95 0.9" contype="0" conaffinity="0" group="2"/>
    <geom name="lane_div_2" type="box" pos="2.0 -0.15 0.001" size="6.0 0.012 0.001" rgba="0.95 0.95 0.95 0.9" contype="0" conaffinity="0" group="2"/>
    <geom name="lane_div_3" type="box" pos="2.0  0.15 0.001" size="6.0 0.012 0.001" rgba="0.95 0.95 0.95 0.9" contype="0" conaffinity="0" group="2"/>
    <geom name="lane_div_4" type="box" pos="2.0  0.45 0.001" size="6.0 0.012 0.001" rgba="0.95 0.95 0.95 0.9" contype="0" conaffinity="0" group="2"/>
    <geom name="lane_div_5" type="box" pos="2.0  0.75 0.001" size="6.0 0.012 0.001" rgba="0.95 0.95 0.95 0.9" contype="0" conaffinity="0" group="2"/>

    <!-- Track Outer Barriers (Left & Right Safety Walls extending from x=-4.0m to x=+8.0m) -->
    <geom name="barrier_left"  type="box" pos="2.0  0.88 0.04" size="6.0 0.025 0.04" material="barrier_mat" contype="0" conaffinity="0" group="2"/>
    <geom name="barrier_right" type="box" pos="2.0 -0.88 0.04" size="6.0 0.025 0.04" material="barrier_mat" contype="0" conaffinity="0" group="2"/>

    <!-- Starting Line at x=0.0m (Bright White Stripe across all 5 lanes) -->
    <geom name="start_line" type="box" pos="0.0 0.0 0.002" size="0.05 0.75 0.001" material="start_line_mat" contype="0" conaffinity="0" group="2"/>

    <!-- Lane Number Start Markers (Lanes 1 to 5) -->
    <geom name="lane_start_1" type="box" pos="-0.15 -0.60 0.002" size="0.06 0.08 0.001" rgba="0.85 0.85 0.85 0.9" contype="0" conaffinity="0" group="2"/>
    <geom name="lane_start_2" type="box" pos="-0.15 -0.30 0.002" size="0.06 0.08 0.001" rgba="0.85 0.85 0.85 0.9" contype="0" conaffinity="0" group="2"/>
    <geom name="lane_start_3" type="box" pos="-0.15  0.00 0.002" size="0.06 0.08 0.001" rgba="0.85 0.85 0.85 0.9" contype="0" conaffinity="0" group="2"/>
    <geom name="lane_start_4" type="box" pos="-0.15  0.30 0.002" size="0.06 0.08 0.001" rgba="0.85 0.85 0.85 0.9" contype="0" conaffinity="0" group="2"/>
    <geom name="lane_start_5" type="box" pos="-0.15  0.60 0.002" size="0.06 0.08 0.001" rgba="0.85 0.85 0.85 0.9" contype="0" conaffinity="0" group="2"/>

    <!-- Distance Split Markers: 1m, 2m, 3m, 4m -->
    <geom name="marker_1m" type="box" pos="1.0 0.0 0.002" size="0.015 0.75 0.001" rgba="0.7 0.7 0.7 0.5" contype="0" conaffinity="0" group="2"/>
    <geom name="marker_2m" type="box" pos="2.0 0.0 0.002" size="0.015 0.75 0.001" rgba="0.7 0.7 0.7 0.5" contype="0" conaffinity="0" group="2"/>
    <geom name="marker_3m" type="box" pos="3.0 0.0 0.002" size="0.015 0.75 0.001" rgba="0.7 0.7 0.7 0.5" contype="0" conaffinity="0" group="2"/>
    <geom name="marker_4m" type="box" pos="4.0 0.0 0.002" size="0.015 0.75 0.001" rgba="0.7 0.7 0.7 0.5" contype="0" conaffinity="0" group="2"/>

    <!-- Finish Line at x=5.0m (Bright Gold Finish Ground Stripe) -->
    <geom name="finish_line" type="box" pos="5.0 0.0 0.002" size="0.08 0.75 0.001" material="finish_line_mat" contype="0" conaffinity="0" group="2"/>
    <!-- Finish Line Stadium Arch Pillars & Ribbon Tape at Microduck Height (z=0.244m) -->
    <geom name="finish_post_left"  type="cylinder" pos="5.0  0.85 0.122" size="0.025 0.122" rgba="0.95 0.75 0.1 1" contype="0" conaffinity="0" group="2"/>
    <geom name="finish_post_right" type="cylinder" pos="5.0 -0.85 0.122" size="0.025 0.122" rgba="0.95 0.75 0.1 1" contype="0" conaffinity="0" group="2"/>
    <geom name="finish_banner"     type="box"      pos="5.0  0.0  0.235" size="0.015 0.85 0.018" rgba="0.95 0.15 0.15 0.95" contype="0" conaffinity="0" group="2"/>
"""

    def prefix_block(text, prefix, y_pos):
        p_text = re.sub(r'name="([^"]+)"', f'name="{prefix}\\1"', text)
        p_text = re.sub(r'joint="([^"]+)"', f'joint="{prefix}\\1"', p_text)
        p_text = re.sub(r'geom1="([^"]+)"', f'geom1="{prefix}\\1"', p_text)
        p_text = re.sub(r'geom2="([^"]+)"', f'geom2="{prefix}\\1"', p_text)
        p_text = re.sub(r'body1="([^"]+)"', f'body1="{prefix}\\1"', p_text)
        p_text = re.sub(r'body2="([^"]+)"', f'body2="{prefix}\\1"', p_text)
        p_text = re.sub(r'site="([^"]+)"', f'site="{prefix}\\1"', p_text)
        p_text = re.sub(r'body="([^"]+)"', f'body="{prefix}\\1"', p_text)
        p_text = re.sub(r'geom="([^"]+)"', f'geom="{prefix}\\1"', p_text)

        p_text = p_text.replace('material="trunk_base_material"', f'material="{prefix}shell_mat"')
        p_text = p_text.replace('material="left_shell_material"', f'material="{prefix}shell_mat"')
        p_text = p_text.replace('material="right_shell_material"', f'material="{prefix}shell_mat"')
        p_text = p_text.replace('material="top_head_shell_material"', f'material="{prefix}head_top_mat"')
        p_text = p_text.replace('material="bottom_head_shell_material"', f'material="{prefix}beak_mat"')
        p_text = p_text.replace('material="jaw_material"', f'material="{prefix}beak_mat"')
        p_text = p_text.replace('material="foot_left_material"', f'material="{prefix}foot_mat"')
        p_text = p_text.replace('material="foot_right_material"', f'material="{prefix}foot_mat"')
        p_text = p_text.replace('material="sole_left_material"', f'material="{prefix}foot_mat"')
        p_text = p_text.replace('material="sole_right_material"', f'material="{prefix}foot_mat"')
        p_text = p_text.replace('material="ankle_left_material"', f'material="{prefix}foot_mat"')
        p_text = p_text.replace('material="ankle_right_material"', f'material="{prefix}foot_mat"')

        # Adjust position offsets
        p_text = p_text.replace('pos="0 0 0.12"', f'pos="0.0 {y_pos:.3f} 0.12"')
        p_text = p_text.replace('pos="0.048 0.0 0.220"', f'pos="0.048 {y_pos:.3f} 0.220"')
        p_text = p_text.replace('pos="0.050 0.065 0.220"', f'pos="0.050 {y_pos+0.065:.3f} 0.220"')
        return p_text

    all_bodies = [track_elements]
    all_actuators = []
    all_sensors = []
    all_contacts = []

    for cfg in DUCK_CONFIGS:
        p = f"d{cfg['id']}_"
        y = cfg["y"]

        trunk_p = prefix_block(trunk_template, p, y)
        spoon_lemon_p = prefix_block(spoon_lemon_template, p, y)
        all_bodies.append(f"<!-- === DUCK {cfg['id']}: {cfg['name']} (Lane {cfg['lane']}) === -->\n" + trunk_p + "\n" + spoon_lemon_p)

        act_p = prefix_block(actuator_template[len("<actuator>"): -len("</actuator>")], p, y)
        all_actuators.append(f"<!-- Duck {cfg['id']} Actuators -->\n" + act_p)

        sens_p = prefix_block(sensor_template[len("<sensor>"): -len("</sensor>")], p, y)
        all_sensors.append(f"<!-- Duck {cfg['id']} Sensors -->\n" + sens_p)

        # Contact pairs and exclusions for this duck
        contact_duck = f"""
    <!-- Duck {cfg['id']} Exclusions -->
    <exclude body1="{p}spoon" body2="{p}trunk_base"/>
    <exclude body1="{p}neck" body2="{p}beak_lower"/>
    <exclude body1="{p}neck_pitch" body2="{p}beak_lower"/>
    <exclude body1="{p}jaw_soft" body2="{p}beak_lower"/>
    <exclude body1="{p}lemon" body2="{p}trunk_base"/>
    <exclude body1="{p}neck_pitch" body2="{p}jaw_soft"/>
    <exclude body1="{p}yaw_roll_motion" body2="{p}beak_lower"/>

    <!-- Duck {cfg['id']} Contact Pairs -->
    <pair geom1="{p}spoon_handle" geom2="{p}mouth_roof_pad" condim="6" friction="0.55 0.55 0.05 0.05 0.05" solref="0.002 1.0" solimp="0.99 0.999 0.0001 0.5 2"/>
    <pair geom1="{p}spoon_handle" geom2="{p}jaw_cradle_bed" condim="6" friction="0.55 0.55 0.05 0.05 0.05" solref="0.002 1.0" solimp="0.99 0.999 0.0001 0.5 2"/>
    <pair geom1="{p}lemon_geom" geom2="{p}spoon_bowl_base" condim="6" friction="0.50 0.50 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="{p}lemon_geom" geom2="{p}rim_front" condim="6" friction="0.50 0.50 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="{p}lemon_geom" geom2="{p}rim_back" condim="6" friction="0.50 0.50 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="{p}lemon_geom" geom2="{p}rim_left_m" condim="6" friction="0.50 0.50 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="{p}lemon_geom" geom2="{p}rim_right_m" condim="6" friction="0.50 0.50 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="{p}lemon_geom" geom2="{p}rim_left_b" condim="6" friction="0.50 0.50 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="{p}lemon_geom" geom2="{p}rim_right_b" condim="6" friction="0.50 0.50 0.01 0.01 0.01" solref="0.010 1.0" solimp="0.98 0.995 0.0005 0.5 2"/>
    <pair geom1="{p}jaw_soft_lower_collision" geom2="{p}soft_mouth_top_collision" condim="3" friction="1.0 1.0 0.1" solref="0.004 1.0" solimp="0.95 0.99 0.001 0.5 2"/>
"""
        all_contacts.append(contact_duck)

    full_xml = f"""{header_assets}
  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.4 0.01 0.001"/>
    <light pos="2.5 0 4.5" dir="0 0 -1" type="directional"/>
    <light pos="0.0 0 3.0" dir="0 0 -1" type="directional"/>
    <light pos="5.0 0 3.0" dir="0 0 -1" type="directional"/>
{''.join(all_bodies)}
  </worldbody>

  <contact>
{''.join(all_contacts)}
  </contact>

  <equality/>

  <actuator>
{''.join(all_actuators)}
  </actuator>

  <sensor>
{''.join(all_sensors)}
  </sensor>
</mujoco>"""

    # Apply rubber sole contact grip
    full_xml = full_xml.replace('geom class="collision"', 'geom class="collision" friction="1.4 0.01 0.001"')
    return full_xml


class DuckRacer:
    """Represents an autonomous Microduck biped competitor in the race."""

    def __init__(self, duck_id, cfg, model, data, session):
        self.duck_id = duck_id
        self.cfg = cfg
        self.model = model
        self.data = data
        self.session = session
        self.prefix = f"d{duck_id}_"

        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.is_67d = (session.get_inputs()[0].shape[1] == 67)

        # Body IDs
        self.trunk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}trunk_base")
        self.jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}jaw_soft")
        self.spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}spoon")
        self.lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}lemon")

        # Joints & Qpos addresses
        self.trunk_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}trunk_base_freejoint")
        self.trunk_qadr = model.jnt_qposadr[self.trunk_jid]
        self.trunk_dof = model.jnt_dofadr[self.trunk_jid]

        self.beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}beak")
        self.beak_qpos = model.jnt_qposadr[self.beak_jid]
        self.beak_dof = model.jnt_dofadr[self.beak_jid]

        self.spoon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}spoon_free")
        self.spoon_qadr = model.jnt_qposadr[self.spoon_jid]
        self.spoon_dof = model.jnt_dofadr[self.spoon_jid]

        self.lemon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}lemon_free")
        self.lemon_qadr = model.jnt_qposadr[self.lemon_jid]
        self.lemon_dof = model.jnt_dofadr[self.lemon_jid]

        self.sensor_gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{self.prefix}imu_ang_vel")
        self.sensor_gyro_adr = model.sensor_adr[self.sensor_gyro_id]

        self.act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.prefix}{name}") for name in JOINT_NAMES]
        self.beak_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.prefix}beak")

        self.joint_qpos_indices = [int(model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in self.act_ids]
        self.joint_dof_indices = [int(model.jnt_dofadr[model.actuator_trnid[a, 0]]) for a in self.act_ids]

        self.default_pose = DEFAULT_POSE[:14].copy()
        self.default_pose[5] = 0.2760
        self.default_pose[6] = 0.2760

        self.last_action = np.zeros(14, dtype=np.float32)
        self.target_vx = 0.15
        self.current_vx = 0.0
        self.head_pos_steadicam = 0.0
        self.target_clamp_angle = 1.550
        self.clamp_torque = 0.45
        self.q_s = np.array([0.0, 0.7071068, 0.7071068, 0.0])

        self.y_int = 0.0

        # Telemetry
        self.distance = 0.0
        self.lemon_displacement = 0.0
        self.lemon_dropped = False
        self.finished = False
        self.finish_time = None
        self.rank = duck_id + 1

    def reset_posture(self):
        y0 = self.cfg["y"]
        self.data.qpos[self.trunk_qadr:self.trunk_qadr+3] = [0.0, y0, 0.125]
        self.data.qpos[self.trunk_qadr+3:self.trunk_qadr+7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.trunk_dof:self.trunk_dof+6] = 0.0

        for i, idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[idx] = self.default_pose[i]
            self.data.qvel[self.joint_dof_indices[i]] = 0.0

        for i, act_id in enumerate(self.act_ids):
            self.data.ctrl[act_id] = self.default_pose[i]

        self.data.qpos[self.beak_qpos] = self.target_clamp_angle
        self.data.qvel[self.beak_dof] = 0.0
        self.data.qfrc_applied[self.beak_dof] = -self.clamp_torque
        self.data.ctrl[self.beak_act_id] = 1.0

        self.current_vx = 0.0
        self.head_pos_steadicam = 0.0
        self.y_int = 0.0
        self.lemon_dropped = False
        self.finished = False
        self.finish_time = None

    def place_spoon(self):
        p_jaw = self.data.xpos[self.jaw_bid]
        R_jaw = self.data.xmat[self.jaw_bid].reshape(3, 3)
        HOVER_LOCAL = np.array([-0.0119, 0.0, -0.0465])
        CYL_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

        self.data.qpos[self.spoon_qadr:self.spoon_qadr+3] = p_jaw + R_jaw @ HOVER_LOCAL
        q_jaw = np.empty(4)
        mujoco.mju_mat2Quat(q_jaw, self.data.xmat[self.jaw_bid])
        q_s = np.empty(4)
        mujoco.mju_mulQuat(q_s, q_jaw, CYL_QUAT)
        self.data.qpos[self.spoon_qadr+3:self.spoon_qadr+7] = q_s
        self.data.qvel[self.spoon_dof:self.spoon_dof+6] = 0.0
        self.q_s = q_s

    def place_lemon(self):
        p_spoon = self.data.xpos[self.spoon_bid]
        R_spoon = self.data.xmat[self.spoon_bid].reshape(3, 3)
        self.data.qpos[self.lemon_qadr:self.lemon_qadr+3] = p_spoon + R_spoon @ np.array([0.0, 0.0052, 0.070])
        self.data.qpos[self.lemon_qadr+3:self.lemon_qadr+7] = self.q_s
        self.data.qvel[self.lemon_dof:self.lemon_dof+6] = 0.0

    def step_clamp(self):
        beak_err = self.target_clamp_angle - self.data.qpos[self.beak_qpos]
        self.data.qfrc_applied[self.beak_dof] = float(np.clip(-self.clamp_torque + 10.0 * beak_err - 0.3 * self.data.qvel[self.beak_dof], -1.0, 0.2))

    def get_obs(self, sim_dt_ctrl=0.02):
        base_ang_vel = self.data.sensordata[self.sensor_gyro_adr:self.sensor_gyro_adr+3].astype(np.float32)
        R_trunk = self.data.xmat[self.trunk_bid].reshape(3, 3)
        proj_grav = (R_trunk.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
        qpos_rel = np.array([self.data.qpos[idx] - self.default_pose[i] for i, idx in enumerate(self.joint_qpos_indices)], dtype=np.float32)
        qvel = np.array([self.data.qvel[idx] for idx in self.joint_dof_indices], dtype=np.float32)
        last_act = self.last_action.copy()

        # Closed-loop line-of-sight pure pursuit heading & lane centering
        yaw = math.atan2(R_trunk[1, 0], R_trunk[0, 0])
        y_err = float(self.data.xpos[self.trunk_bid][1] - self.cfg["y"])
        
        # Line-of-sight target heading points smoothly toward lane centerline
        target_yaw = math.atan2(-y_err, 0.60)
        yaw_err = (yaw - target_yaw + math.pi) % (2 * math.pi) - math.pi
        cmd_wz = float(np.clip(-2.5 * yaw_err, -0.45, 0.45))

        cmd = np.zeros(13, dtype=np.float32)
        cmd[0] = self.current_vx
        cmd[1] = 0.0
        cmd[2] = cmd_wz

        obs_base = np.concatenate([base_ang_vel, proj_grav, qpos_rel, qvel, last_act, cmd]).astype(np.float32)

        if self.is_67d:
            R_s = self.data.xmat[self.spoon_bid].reshape(3, 3)
            p_diff = self.data.xpos[self.lemon_bid] - self.data.xpos[self.spoon_bid]
            p_rel = R_s.T @ p_diff
            dx = float(p_rel[0])
            dz = float(p_rel[2] - 0.070)

            v_diff = self.data.qvel[self.lemon_dof:self.lemon_dof+3] - self.data.qvel[self.spoon_dof:self.spoon_dof+3]
            v_rel = R_s.T @ v_diff
            vx_rel = float(v_rel[0])
            vz_rel = float(v_rel[2])

            z_local = R_s.T @ np.array([0.0, 0.0, 1.0])
            pitch_err = float(np.arcsin(np.clip(z_local[2], -0.999, 0.999)))
            roll_err = float(np.arcsin(np.clip(z_local[0], -0.999, 0.999)))

            extra_obs = np.array([dx, dz, vx_rel, vz_rel, pitch_err, roll_err], dtype=np.float32)
            obs = np.concatenate([obs_base, extra_obs]).astype(np.float32)
        else:
            obs = obs_base

        return obs[np.newaxis, :]

    def infer_and_apply(self, decimation, sim_dt):
        if self.current_vx < self.target_vx:
            self.current_vx = min(self.target_vx, self.current_vx + 0.005)
        elif self.current_vx > self.target_vx:
            self.current_vx = max(self.target_vx, self.current_vx - 0.005)

        dt_ctrl = decimation * sim_dt
        obs = self.get_obs(sim_dt_ctrl=dt_ctrl)
        action = self.session.run([self.output_name], {self.input_name: obs})[0][0].astype(np.float32)
        self.last_action = action.copy()

        # Active Steadicam Pantograph:
        # Measure fore-aft tilt of the spoon bowl relative to robot heading
        R_trunk = self.data.xmat[self.trunk_bid].reshape(3, 3)
        h_fwd = np.array([R_trunk[0, 0], R_trunk[1, 0], 0.0])
        h_fwd /= (np.linalg.norm(h_fwd) + 1e-6)

        R_s = self.data.xmat[self.spoon_bid].reshape(3, 3)
        b_norm = R_s @ np.array([0.0, 1.0, 0.0])
        spoon_pitch = math.asin(np.clip(float(np.dot(b_norm, h_fwd)), -0.999, 0.999))

        rate_cmd = np.clip(1.20 * spoon_pitch, -0.30, 0.30)
        self.head_pos_steadicam = np.clip(self.head_pos_steadicam + rate_cmd * dt_ctrl, -0.40, 0.40)

        # Counter-phase neck/head pitch compensation keeps the spoon bowl level,
        # while locking head yaw (action[7]) and head roll (action[8]) to zero
        # to eliminate lateral slosh and roll tipping during walking steps.
        action[5] = 0.0 + self.head_pos_steadicam
        action[6] = 0.0 - self.head_pos_steadicam
        action[7] = 0.0
        action[8] = 0.0

        targets = self.default_pose + action
        for i, act_id in enumerate(self.act_ids):
            self.data.ctrl[act_id] = targets[i]

    def update_telemetry(self, sim_time):
        self.distance = float(self.data.qpos[self.trunk_qadr])
        p_s = self.data.xpos[self.spoon_bid]
        R_s = self.data.xmat[self.spoon_bid].reshape(3, 3)
        bowl_center = p_s + R_s @ np.array([0.0, 0.0052, 0.070])
        self.lemon_displacement = float(np.linalg.norm(self.data.xpos[self.lemon_bid] - bowl_center))

        if self.lemon_displacement > 0.045:
            self.lemon_dropped = True

        if self.distance >= 5.0 and not self.finished:
            self.finished = True
            self.finish_time = sim_time


def setup_racer_episode(ducks, model, data):
    """Resets and settles all 5 ducks firmly at the start line."""
    mujoco.mj_resetData(model, data)
    for d in ducks:
        d.reset_posture()
    mujoco.mj_forward(model, data)

    for d in ducks:
        d.place_spoon()
    mujoco.mj_forward(model, data)

    for d in ducks:
        d.place_lemon()
    mujoco.mj_forward(model, data)

    # 150-step (0.30s) standing equilibrium settling with motor ctrl held at default standing pose
    for _ in range(150):
        for d in ducks:
            d.data.qpos[d.trunk_qadr:d.trunk_qadr+3] = [0.0, d.cfg["y"], 0.125]
            d.data.qpos[d.trunk_qadr+3:d.trunk_qadr+7] = [1.0, 0.0, 0.0, 0.0]
            d.data.qvel[d.trunk_dof:d.trunk_dof+6] = 0.0
            for i, act_id in enumerate(d.act_ids):
                d.data.ctrl[act_id] = d.default_pose[i]
            d.step_clamp()
            d.data.qvel[d.lemon_dof:d.lemon_dof+6] *= 0.98
        mujoco.mj_step(model, data)


def render_racer_visual_proofs():
    """Renders high-resolution multi-view proof images of the 5-duck race."""
    xml_str = create_racer_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    session = ort.InferenceSession(_DEFAULT_POLICY, providers=["CPUExecutionProvider"])
    ducks = [DuckRacer(cfg["id"], cfg, model, data, session) for cfg in DUCK_CONFIGS]
    setup_racer_episode(ducks, model, data)

    # Simulate 1.5s (750 steps) so ducks are actively sprinting straight down the track lanes
    pacing = [0.23, 0.26, 0.25, 0.27, 0.24]
    for i, d in enumerate(ducks):
        d.target_vx = pacing[i]

    for step in range(750):
        for d in ducks:
            d.step_clamp()
        if step % 10 == 0:
            for d in ducks:
                d.infer_and_apply(10, model.opt.timestep)
                d.update_telemetry(step * model.opt.timestep)
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, width=1280, height=720)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[3] = 0  # Hide collision geoms

    pack_mean_x = float(np.mean([d.distance for d in ducks]))

    # 1. Dynamic Isometric Broadcast Perspective (Framed to exclude the back end of the track)
    cam_start = mujoco.MjvCamera()
    cam_start.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_start.lookat = [pack_mean_x + 0.45, 0.0, 0.16]
    cam_start.distance = 2.15
    cam_start.elevation = -22.0
    cam_start.azimuth = 68.0
    renderer.update_scene(data, camera=cam_start, scene_option=vopt)
    p_start = os.path.join(_ARTIFACT_DIR, "duck_racer_start_line_proof.png")
    Image.fromarray(renderer.render()).save(p_start)

    # 2. Side Broadcast Dolly View (Horizontal profile across all 5 lanes)
    cam_side = mujoco.MjvCamera()
    cam_side.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_side.lookat = [pack_mean_x + 0.20, 0.0, 0.16]
    cam_side.distance = 1.85
    cam_side.elevation = -14.0
    cam_side.azimuth = 80.0
    renderer.update_scene(data, camera=cam_side, scene_option=vopt)
    p_side = os.path.join(_ARTIFACT_DIR, "duck_racer_side_broadcast_proof.png")
    Image.fromarray(renderer.render()).save(p_side)

    # 3. Finish Line Telephoto View (Looking from finish line back at oncoming pack)
    cam_top = mujoco.MjvCamera()
    cam_top.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_top.lookat = [pack_mean_x + 0.50, 0.0, 0.16]
    cam_top.distance = 3.60
    cam_top.elevation = -12.0
    cam_top.azimuth = 180.0
    renderer.update_scene(data, camera=cam_top, scene_option=vopt)
    p_top = os.path.join(_ARTIFACT_DIR, "duck_racer_aerial_drone_proof.png")
    Image.fromarray(renderer.render()).save(p_top)

    # 4. Close-Up of Center Duck & Spoon Lemon Balance
    cam_close = mujoco.MjvCamera()
    cam_close.type = mujoco.mjtCamera.mjCAMERA_FREE
    p_d2 = data.xpos[ducks[2].trunk_bid]
    cam_close.lookat = [p_d2[0] + 0.04, p_d2[1], p_d2[2] + 0.08]
    cam_close.distance = 0.42
    cam_close.elevation = -15.0
    cam_close.azimuth = 25.0
    renderer.update_scene(data, camera=cam_close, scene_option=vopt)
    p_close = os.path.join(_ARTIFACT_DIR, "duck_racer_gold_closeup_proof.png")
    Image.fromarray(renderer.render()).save(p_close)

    print("\nVisual proofs saved:")
    print(f"  Isometric Broadcast: {p_start}")
    print(f"  Side Broadcast:      {p_side}")
    print(f"  Finish Telephoto:    {p_top}")
    print(f"  Gold Duck Close:     {p_close}")
    return p_start, p_side, p_top, p_close


def render_racer_video(output_path=None, duration_sec=21.0, fps=50):
    """Renders offscreen MP4 video of the 5-duck race with natural physics, speeding up recording 3x."""
    import imageio
    if output_path is None:
        output_path = os.path.join(_ARTIFACT_DIR, "duck_racer_competition_demo.mp4")

    raw_path = output_path.replace(".mp4", "_raw.mp4")
    print(f"\n[VIDEO] Simulating {duration_sec:.1f}s race with natural physics, recording to {output_path} (3x playback)...")
    xml_str = create_racer_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    session = ort.InferenceSession(_DEFAULT_POLICY, providers=["CPUExecutionProvider"])
    ducks = [DuckRacer(cfg["id"], cfg, model, data, session) for cfg in DUCK_CONFIGS]
    setup_racer_episode(ducks, model, data)

    # Distinct pacing speeds ensuring physical stability (all lemons 100% balanced):
    # #1 Emerald Green (Lane 4): 0.175 m/s (Winner)
    # #2 Ruby Coral Pink (Lane 1): 0.165 m/s (2nd Place)
    # #3 Duck Gold Yellow (Lane 3): 0.155 m/s (3rd Place)
    # #4 Royal Amethyst Purple (Lane 5): 0.148 m/s (4th Place)
    # #5 Riviera Sky Blue (Lane 2): 0.140 m/s (5th Place Last)
    pacing = [0.23, 0.26, 0.25, 0.27, 0.24]
    for i, d in enumerate(ducks):
        d.target_vx = pacing[i]

    renderer = mujoco.Renderer(model, width=1280, height=720)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[3] = 0

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.15
    cam.elevation = -22.0
    cam.azimuth = 68.0  # Dynamic Isometric Broadcast Steadicam tracking pack

    total_sim_steps = int(duration_sec / model.opt.timestep)
    sim_steps_per_frame = int(round(1.0 / (fps * model.opt.timestep)))
    decimation = 10
    sim_dt = model.opt.timestep

    writer = imageio.get_writer(raw_path, fps=fps, codec="libx264", quality=8)
    frame_count = 0
    smooth_lookat_x = 0.45

    for step in range(total_sim_steps):
        t_sim = step * sim_dt
        for d in ducks:
            d.step_clamp()

        if step % decimation == 0:
            for d in ducks:
                d.infer_and_apply(decimation, sim_dt)
                d.update_telemetry(t_sim)

        mujoco.mj_step(model, data)

        if step % sim_steps_per_frame == 0:
            # Dynamic Steadicam with smooth camera tracking
            xs = [d.distance for d in ducks]
            pack_mean_x = float(np.mean(xs))
            target_x = pack_mean_x + 0.45
            smooth_lookat_x = 0.95 * smooth_lookat_x + 0.05 * target_x
            cam.lookat = [smooth_lookat_x, 0.0, 0.16]

            renderer.update_scene(data, camera=cam, scene_option=vopt)
            img = renderer.render()
            writer.append_data(img)
            frame_count += 1

            if frame_count % (fps * 2) == 0:
                lead_d = max(ducks, key=lambda d: d.distance)
                print(f"  Frame {frame_count:3d}/{int(duration_sec*fps)} (t={t_sim:.2f}s): "
                      f"Leader={lead_d.cfg['name']} (x={lead_d.distance:.2f}m) | Pack Spread={max(xs)-min(xs):.2f}m")

    writer.close()
    print(f"[RAW SUCCESS] Saved {frame_count} frames to {raw_path}")

    # Speed up the video recording by 3x as requested
    print(f"[3X PACING] Applying 3x speedup to the recording via ffmpeg...")
    cmd_3x = f'ffmpeg -y -i "{raw_path}" -filter:v "setpts=0.3333333333*PTS" -r 50 "{output_path}"'
    os.system(cmd_3x)
    print(f"[VIDEO SUCCESS] 3x paced video saved to {output_path}")

    # Generate high quality WebP from the 3x paced video
    webp_path = output_path.replace(".mp4", ".webp")
    cmd_webp = f'ffmpeg -y -i "{output_path}" -vf "fps=25,scale=720:-1:flags=lanczos" -vcodec libwebp -lossless 0 -compression_level 4 -q:v 75 -loop 0 "{webp_path}"'
    os.system(cmd_webp)
    print(f"[WEBP SUCCESS] Converted to {webp_path}")

    # Cleanup raw video
    if os.path.exists(raw_path):
        os.remove(raw_path)

    return output_path


def run_benchmark(duration_sec=10.0):
    """Executes a headless 5-duck race benchmark with quantitative metrics."""
    print(f"\n[BENCHMARK] Running {duration_sec:.1f}s 5-duck competition evaluation...")
    xml_str = create_racer_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    session = ort.InferenceSession(_DEFAULT_POLICY, providers=["CPUExecutionProvider"])
    ducks = [DuckRacer(cfg["id"], cfg, model, data, session) for cfg in DUCK_CONFIGS]
    setup_racer_episode(ducks, model, data)

    pacing = [0.23, 0.26, 0.25, 0.27, 0.24]
    for i, d in enumerate(ducks):
        d.target_vx = pacing[i]

    total_sim_steps = int(duration_sec / model.opt.timestep)
    decimation = 10
    sim_dt = model.opt.timestep

    max_lat_errors = {d.duck_id: 0.0 for d in ducks}

    for step in range(total_sim_steps):
        t_sim = step * sim_dt
        for d in ducks:
            d.step_clamp()

        if step % decimation == 0:
            for d in ducks:
                d.infer_and_apply(decimation, sim_dt)
                d.update_telemetry(t_sim)
                y_err = abs(float(data.xpos[d.trunk_bid][1] - d.cfg["y"]))
                if y_err > max_lat_errors[d.duck_id]:
                    max_lat_errors[d.duck_id] = y_err

        mujoco.mj_step(model, data)

    print("\n" + "=" * 78)
    print(f"5-DUCK RACE BENCHMARK RESULTS (t = {duration_sec:.1f}s)")
    print("=" * 78)
    print(f"{'Rank':<5} | {'Lane':<4} | {'Name':<10} | {'Distance':<10} | {'Max Lat Error':<14} | {'Lemon Status':<14}")
    print("-" * 78)

    ranked = sorted(ducks, key=lambda d: d.distance, reverse=True)
    for r, d in enumerate(ranked):
        lemon_stat = "DROPPED" if d.lemon_dropped else f"BALANCED ({d.lemon_displacement*1000:.1f}mm)"
        lat_err_cm = max_lat_errors[d.duck_id] * 100.0
        print(f"#{r+1:<4} | {d.cfg['lane']:<4} | {d.cfg['name']:<10} | {d.distance:6.3f} m   | {lat_err_cm:5.2f} cm       | {lemon_stat:<14}")
    print("=" * 78)
    all_balanced = all(not d.lemon_dropped for d in ducks)
    all_in_lane = all(max_lat_errors[d.duck_id] < 0.10 for d in ducks)
    print(f"Verification: All Lemons Balanced = {all_balanced} | All In-Lane = {all_in_lane}\n")
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Microduck 5-Duck Lemon & Spoon Racer")
    parser.add_argument("--proofs", action="store_true", help="Render high-res visual proofs and exit")
    parser.add_argument("--benchmark", action="store_true", help="Run quantitative headless benchmark")
    parser.add_argument("--gui", action="store_true", help="Launch native interactive GLFW MuJoCo viewer")
    parser.add_argument("--video", action="store_true", help="Render offscreen MP4 competition video")
    parser.add_argument("--duration", type=float, default=21.0, help="Duration for video render (seconds, default: 21.0)")
    parser.add_argument("--speed", type=float, default=1.0, help="Viewer playback speed multiplier")
    args = parser.parse_args()

    if args.proofs:
        render_racer_visual_proofs()
        return

    if args.benchmark:
        run_benchmark(duration_sec=args.duration)
        return

    if args.video:
        render_racer_video(duration_sec=args.duration)
        return

    if not args.gui:
        # Default: render proofs
        render_racer_visual_proofs()
        return

    # Interactive Native GLFW Viewer
    xml_str = create_racer_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    session = ort.InferenceSession(_DEFAULT_POLICY, providers=["CPUExecutionProvider"])
    ducks = [DuckRacer(cfg["id"], cfg, model, data, session) for cfg in DUCK_CONFIGS]
    setup_racer_episode(ducks, model, data)

    # Pacing speeds
    base_pacing = [0.23, 0.26, 0.25, 0.27, 0.24]
    for i, d in enumerate(ducks):
        d.target_vx = base_pacing[i]

    sim_speed = args.speed
    user_paused = False
    reset_requested = False
    camera_mode = 0  # 0: Track Chase, 1: Side Broadcast, 2: Aerial Drone, 3: Leader Focus

    def on_key(keycode):
        nonlocal user_paused, reset_requested, sim_speed, camera_mode
        if keycode == 32:  # Space
            user_paused = not user_paused
        elif keycode in (ord('R'), ord('r')):
            reset_requested = True
        elif keycode in (ord('C'), ord('c')):
            camera_mode = (camera_mode + 1) % 4
            modes = ["Track Chase", "Side Broadcast", "Aerial Drone", "Leader Focus"]
            print(f"\n[CAMERA] Switched to View -> {modes[camera_mode]}")
        elif keycode in (ord('W'), ord('w'), 265):
            for d in ducks:
                d.target_vx = min(0.30, d.target_vx + 0.02)
            print(f"\n[INTERACTIVE] Flocked Acceleration -> All speeds increased by +0.02 m/s")
        elif keycode in (ord('S'), ord('s'), 264):
            for d in ducks:
                d.target_vx = max(0.05, d.target_vx - 0.02)
            print(f"\n[INTERACTIVE] Flocked Deceleration -> All speeds decreased by -0.02 m/s")
        elif keycode in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            idx = keycode - ord('1')
            ducks[idx].target_vx = min(0.35, ducks[idx].target_vx + 0.03)
            print(f"\n[BOOST] Racer {idx+1} ({ducks[idx].cfg['name']}) Turbo Boosted -> vx={ducks[idx].target_vx:.2f} m/s")

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 2.50
    viewer.cam.elevation = -18.0
    viewer.cam.azimuth = 15.0  # Aligned straight down the track lanes
    viewer.opt.geomgroup[3] = 0

    print("\n" + "=" * 70)
    print("MICRODUCK 5-DUCK STADIUM RACER READY (GENIE SIM 3.0 HIGH FIDELITY)")
    print("=" * 70)
    print("  Controls:")
    print("    W / Up:     Accelerate all ducks (+0.02 m/s)")
    print("    S / Down:   Decelerate all ducks (-0.02 m/s)")
    print("    1 .. 5:     Give individual turbo boost to duck 1 through 5 (+0.03 m/s)")
    print("    C:          Cycle camera: Track Chase -> Side Broadcast -> Drone -> Leader")
    print("    Space:      Pause / Resume physics")
    print("    R:          Reset race to start line")
    print("=" * 70 + "\n")

    step = 0
    decimation = 10
    sim_dt = model.opt.timestep

    while viewer.is_running():
        t_start = time.time()

        if reset_requested:
            reset_requested = False
            setup_racer_episode(ducks, model, data)
            for i, d in enumerate(ducks):
                d.target_vx = base_pacing[i]
            step = 0
            print("\n[RACE RESET] All 5 ducks reset to start line!")

        if user_paused:
            viewer.sync()
            time.sleep(0.02)
            continue

        t_sim = step * sim_dt
        for d in ducks:
            d.step_clamp()

        if step % decimation == 0:
            for d in ducks:
                d.infer_and_apply(decimation, sim_dt)
                d.update_telemetry(t_sim)

        mujoco.mj_step(model, data)
        step += 1

        # Camera management
        xs = [d.distance for d in ducks]
        pack_mean_x = float(np.mean(xs))
        leader = max(ducks, key=lambda d: d.distance)

        if camera_mode == 0:    # Track Chase
            viewer.cam.lookat = [pack_mean_x + 0.60, 0.0, 0.18]
            viewer.cam.distance = 2.50
            viewer.cam.elevation = -18.0
            viewer.cam.azimuth = 15.0
        elif camera_mode == 1:  # Side Broadcast
            viewer.cam.lookat = [pack_mean_x, 0.0, 0.18]
            viewer.cam.distance = 2.40
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 90.0
        elif camera_mode == 2:  # Aerial Drone
            viewer.cam.lookat = [pack_mean_x + 0.50, 0.0, 0.10]
            viewer.cam.distance = 3.40
            viewer.cam.elevation = -75.0
            viewer.cam.azimuth = 90.0
        elif camera_mode == 3:  # Leader Focus
            p_lead = data.xpos[leader.trunk_bid]
            viewer.cam.lookat = [p_lead[0] + 0.12, p_lead[1], p_lead[2] + 0.08]
            viewer.cam.distance = 0.70
            viewer.cam.elevation = -15.0
            viewer.cam.azimuth = 25.0

        viewer.sync()

        # Telemetry leaderboard every 50 steps (0.10s)
        if step % 50 == 0:
            ranked = sorted(ducks, key=lambda d: d.distance, reverse=True)
            leaderboard_str = f"\r[t={t_sim:5.2f}s] " + " | ".join(
                f"{r+1}:{d.cfg['name'][:5]} {d.distance:4.2f}m ({d.lemon_displacement*1000:3.1f}mm)"
                for r, d in enumerate(ranked)
            )
            sys.stdout.write(leaderboard_str)
            sys.stdout.flush()

        t_elapsed = time.time() - t_start
        dt_sleep = (sim_dt / sim_speed) - t_elapsed
        if dt_sleep > 0:
            time.sleep(dt_sleep)

    viewer.close()


if __name__ == "__main__":
    main()
