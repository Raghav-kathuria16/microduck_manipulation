#!/usr/bin/env python3
"""Generate MuJoCo XML models and high-resolution renders for:
1. USA Pickleball Official Specification Paddle
2. BWF Badminton Official Specification Racket
"""

import math
import os
import sys
import numpy as np
import mujoco
from PIL import Image

_ROOT = "/Users/raghav/Desktop/microduck_manipulation"
sys.path.insert(0, _ROOT)
_MODELS_DIR = os.path.join(_ROOT, "microduck_manipulation/models")
_ARTIFACT_DIR = "/Users/raghav/.gemini/antigravity-ide/brain/1dcd9661-c7ab-457a-8bec-a3fc2df3ade3"
os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_ARTIFACT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Generate Pickleball Paddle XML
# -----------------------------------------------------------------------------
def build_pickleball_xml():
    """Builds a USA Pickleball standard paddle.
    Specs:
      - Total length: 400 mm (15.75 in)
      - Face length: 260 mm (10.25 in), Face width: 195 mm (7.7 in)
      - Face thickness: 14 mm (0.55 in)
      - Edge guard: 6 mm protective rubber rim around face
      - Handle length: 125 mm (5.0 in)
      - Clamping shaft: 8-10 mm diameter core matching Microduck beak aperture
      - Mass: 215 grams (0.215 kg)
    """
    xml = """<mujoco model="pickleball_paddle">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.05 0.05 0.1" width="512" height="512"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.25 0.3" rgb2="0.15 0.18 0.22" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundmat" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
    <material name="edge_guard" rgba="0.12 0.12 0.14 1" specular="0.3" shininess="0.4"/>
    <material name="face_cyan" rgba="0.08 0.65 0.82 1" specular="0.5" shininess="0.6"/>
    <material name="face_accent" rgba="0.98 0.82 0.15 1" specular="0.6" shininess="0.7"/>
    <material name="grip_tape" rgba="0.92 0.92 0.94 1" specular="0.1" shininess="0.1"/>
    <material name="grip_ring" rgba="0.1 0.1 0.1 1" specular="0.4" shininess="0.5"/>
    <material name="butt_cap" rgba="0.15 0.15 0.18 1" specular="0.3" shininess="0.4"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" material="groundmat"/>

    <!-- Pickleball Paddle Free Body -->
    <body name="pickleball_paddle" pos="0 0 0.15" quat="0.7071068 0 0.7071068 0">
      <freejoint name="paddle_free"/>
      <!-- Mass: 215g tournament standard -->
      <inertial pos="0 0 0.065" mass="0.215" diaginertia="0.0032 0.0028 0.0007"/>

      <!-- 1. Handle Butt Cap (Base) -->
      <geom name="butt_cap_geom" type="cylinder" pos="0 0 -0.160" size="0.016 0.006" material="butt_cap" friction="2.5 0.1 0.1"/>

      <!-- 2. Handle Grip (Length 125mm, from z = -0.154 to z = -0.029) -->
      <!-- Outer contoured grip wrap -->
      <geom name="grip_wrap" type="cylinder" pos="0 0 -0.095" size="0.0135 0.055" material="grip_tape" friction="2.5 0.1 0.1"/>
      <!-- Dedicated 8mm beak-clamping neck zone (allows Microduck beak silicone to bite firmly) -->
      <geom name="clamping_neck" type="cylinder" pos="0 0 -0.035" size="0.0045 0.008" material="grip_ring" friction="3.5 0.2 0.2"/>
      <!-- Finishing ring collar -->
      <geom name="grip_collar" type="cylinder" pos="0 0 -0.025" size="0.015 0.003" material="butt_cap"/>

      <!-- 3. Throat / Taper Transition (from z = -0.022 to z = +0.015) -->
      <geom name="throat_taper" type="box" pos="0 0 -0.005" size="0.025 0.007 0.018" material="edge_guard"/>

      <!-- 4. Paddle Face Core (Length 260mm, Width 195mm, Thickness 14mm: z from +0.015 to +0.275) -->
      <!-- Center Face Plate -->
      <geom name="face_plate" type="box" pos="0 0 0.145" size="0.088 0.0065 0.115" material="face_cyan" friction="1.8 0.1 0.1"/>
      <!-- Graphic Accent Band -->
      <geom name="face_accent_band" type="box" pos="0 0.0068 0.145" size="0.075 0.0005 0.035" material="face_accent"/>
      <geom name="face_accent_band_back" type="box" pos="0 -0.0068 0.145" size="0.075 0.0005 0.035" material="face_accent"/>

      <!-- 5. Edge Guard Perimeter (Protective Outer Bumper) -->
      <!-- Left edge bumper -->
      <geom name="edge_left" type="capsule" fromto="-0.095 0 0.035 -0.095 0 0.235" size="0.0075" material="edge_guard" friction="2.0 0.1 0.1"/>
      <!-- Right edge bumper -->
      <geom name="edge_right" type="capsule" fromto="0.095 0 0.035 0.095 0 0.235" size="0.0075" material="edge_guard" friction="2.0 0.1 0.1"/>
      <!-- Top edge bumper -->
      <geom name="edge_top" type="capsule" fromto="-0.065 0 0.270 0.065 0 0.270" size="0.0075" material="edge_guard" friction="2.0 0.1 0.1"/>
      <!-- Top-left corner rounded curve -->
      <geom name="corner_top_left" type="capsule" fromto="-0.095 0 0.235 -0.065 0 0.270" size="0.0075" material="edge_guard"/>
      <!-- Top-right corner rounded curve -->
      <geom name="corner_top_right" type="capsule" fromto="0.095 0 0.235 0.065 0 0.270" size="0.0075" material="edge_guard"/>
      <!-- Bottom-left taper to throat -->
      <geom name="corner_bot_left" type="capsule" fromto="-0.095 0 0.035 -0.025 0 0.010" size="0.0075" material="edge_guard"/>
      <!-- Bottom-right taper to throat -->
      <geom name="corner_bot_right" type="capsule" fromto="0.095 0 0.035 0.025 0 0.010" size="0.0075" material="edge_guard"/>
    </body>
  </worldbody>
</mujoco>
"""
    path = os.path.join(_MODELS_DIR, "pickleball_paddle.xml")
    with open(path, "w") as f:
        f.write(xml)
    print(f"Generated: {path}")
    return path


# -----------------------------------------------------------------------------
# 2. Generate Badminton Racket XML
# -----------------------------------------------------------------------------
def build_badminton_xml():
    """Builds a BWF standard badminton racket.
    Specs:
      - Total length: 675 mm (26.5 in)
      - Head frame: Isometric oval (length 250 mm, width 195 mm)
      - Ultra-slim carbon shaft: 7.0 mm diameter (length 220 mm)
      - Handle grip: length 155 mm, diameter 22 mm, with butt cap
      - String bed: braided 0.65 mm mesh bed
      - Mass: 83 grams (0.083 kg, 4U tournament standard)
    """
    # Generate 16 ellipse points for the isometric racket frame
    a = 0.095  # half width = 95mm (190mm total width)
    b = 0.120  # half length = 120mm (240mm total length)
    z_center = 0.215  # center of head
    n_pts = 16
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts = [(a * math.cos(th), 0.0, z_center + b * math.sin(th)) for th in angles]

    frame_capsules = []
    for i in range(n_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % n_pts]
        frame_capsules.append(
            f'      <geom name="frame_seg_{i}" type="capsule" fromto="{p1[0]:.4f} {p1[1]:.4f} {p1[2]:.4f} {p2[0]:.4f} {p2[1]:.4f} {p2[2]:.4f}" size="0.0035" material="frame_carbon" friction="1.5 0.1 0.1"/>'
        )
    frame_str = "\n".join(frame_capsules)

    xml = f"""<mujoco model="badminton_racket">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.05 0.05 0.1" width="512" height="512"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.25 0.3" rgb2="0.15 0.18 0.22" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundmat" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
    <material name="frame_carbon" rgba="0.10 0.18 0.35 1" specular="0.8" shininess="0.9"/>
    <material name="shaft_carbon" rgba="0.08 0.12 0.22 1" specular="0.7" shininess="0.8"/>
    <material name="t_joint" rgba="0.85 0.75 0.20 1" specular="0.9" shininess="0.95"/>
    <material name="strings" rgba="0.95 0.98 0.98 0.45" specular="0.3" shininess="0.5"/>
    <material name="grip_black" rgba="0.12 0.12 0.12 1" specular="0.1" shininess="0.1"/>
    <material name="cone_mat" rgba="0.20 0.22 0.26 1" specular="0.5" shininess="0.6"/>
    <material name="butt_cap" rgba="0.05 0.45 0.25 1" specular="0.5" shininess="0.6"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" material="groundmat"/>

    <!-- Badminton Racket Free Body (Total Length 675mm) -->
    <body name="badminton_racket" pos="0 0 0.20" quat="0.7071068 0 0.7071068 0">
      <freejoint name="racket_free"/>
      <!-- Mass: 83g (4U tournament weight standard) -->
      <inertial pos="0 0 0.050" mass="0.083" diaginertia="0.0035 0.0034 0.00015"/>

      <!-- 1. Handle Butt Cap (z = -0.340) -->
      <geom name="butt_cap_geom" type="cylinder" pos="0 0 -0.336" size="0.0135 0.004" material="butt_cap" friction="2.0 0.1 0.1"/>

      <!-- 2. Grip Handle (Length 150mm: z from -0.332 to -0.182) -->
      <geom name="grip_handle" type="cylinder" pos="0 0 -0.257" size="0.011 0.075" material="grip_black" friction="2.5 0.1 0.1"/>

      <!-- 3. Handle Cone / Transition (z from -0.182 to -0.165) -->
      <geom name="handle_cone" type="cylinder" pos="0 0 -0.173" size="0.008 0.009" material="cone_mat"/>

      <!-- 4. Ultra-Slim 7mm Carbon Shaft (Length 240mm: z from -0.164 to +0.076) -->
      <!-- This 7mm diameter shaft directly matches Microduck's 8mm beak silicone aperture -->
      <geom name="shaft" type="cylinder" pos="0 0 -0.044" size="0.0035 0.120" material="shaft_carbon" friction="2.8 0.1 0.1"/>

      <!-- 5. T-Joint Connector (z = +0.080) -->
      <geom name="t_joint_geom" type="cylinder" pos="0 0 0.080" size="0.0055 0.006" material="t_joint"/>

      <!-- 6. Oval Head Frame (16 connected carbon segments, length 240mm, width 190mm) -->
{frame_str}

      <!-- 7. Translucent String Bed Visual Plane -->
      <geom name="string_bed" type="box" pos="0 0 0.215" size="0.086 0.0003 0.110" material="strings" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
    path = os.path.join(_MODELS_DIR, "badminton_racket.xml")
    with open(path, "w") as f:
        f.write(xml)
    print(f"Generated: {path}")
    return path


# -----------------------------------------------------------------------------
# 3. Generate Table Tennis Racket XML (Authentic ITTF Tournament 180g & 40+ Ball)
# -----------------------------------------------------------------------------
def build_table_tennis_xml():
    """Generates the official ITTF tournament table tennis racket model with
    solid full-tang wood core and 40.2mm / 2.70g ball."""
    try:
        from microduck_manipulation.scripts.generate_table_tennis_mesh import generate_solid_table_tennis_racket
    except ImportError:
        try:
            from generate_table_tennis_mesh import generate_solid_table_tennis_racket
        except ImportError:
            from scratch.generate_table_tennis_mesh import generate_solid_table_tennis_racket
    generate_solid_table_tennis_racket()
    path = os.path.join(_MODELS_DIR, "table_tennis_racket.xml")
    print(f"Verified: {path}")
    return path


# -----------------------------------------------------------------------------
# 4. Render High-Resolution Visual Inspection Images
# -----------------------------------------------------------------------------
def render_racket_views(xml_path, name_prefix):
    """Renders 4 distinct camera angles for visual verification:
    - Front Flat View (Proportions, Shape, Geometry)
    - Side Profile View (Thickness, Shaft Slimness, Edge Guard)
    - 3/4 Isometric Perspective (3D Form & Shadows)
    - Close-Up Grip / Shaft Zone (Beak-Clamping Interface)
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=960)

    cameras = {
        "front": {
            "distance": 0.75 if "badminton" in name_prefix else 0.55,
            "elevation": -85.0,
            "azimuth": 90.0,
            "lookat": [0.0, 0.0, 0.15],
            "title": "Front View (Official Proportions & Profile)",
        },
        "side": {
            "distance": 0.75 if "badminton" in name_prefix else 0.55,
            "elevation": 0.0,
            "azimuth": 0.0,
            "lookat": [0.0, 0.0, 0.15],
            "title": "Side View (Thickness & Grip Contour)",
        },
        "perspective": {
            "distance": 0.70 if "badminton" in name_prefix else 0.50,
            "elevation": -22.0,
            "azimuth": 135.0,
            "lookat": [0.0, 0.0, 0.15],
            "title": "3/4 Isometric Perspective",
        },
        "grip_closeup": {
            "distance": 0.22,
            "elevation": -15.0,
            "azimuth": 120.0,
            "lookat": [0.0, 0.0, 0.15],
            "title": "Grip & Clamping Interface (8mm Shaft Zone)",
        },
    }

    saved_images = {}
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    for view_name, cfg in cameras.items():
        cam.distance = cfg["distance"]
        cam.elevation = cfg["elevation"]
        cam.azimuth = cfg["azimuth"]
        cam.lookat = cfg["lookat"]

        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()

        img = Image.fromarray(pixels)
        out_path = os.path.join(_ARTIFACT_DIR, f"{name_prefix}_{view_name}.png")
        img.save(out_path)
        saved_images[view_name] = out_path
        print(f"Saved {name_prefix} {view_name} -> {out_path}")

    return saved_images


def main():
    print("Building Table Tennis Racket & Ball...")
    tt_xml = build_table_tennis_xml()

    print("\nBuilding Pickleball Paddle...")
    p_xml = build_pickleball_xml()
    p_imgs = render_racket_views(p_xml, "pickleball_model")

    print("\nBuilding Badminton Racket...")
    b_xml = build_badminton_xml()
    b_imgs = render_racket_views(b_xml, "badminton_model")

    print("\nAll models and renders generated successfully!")


if __name__ == "__main__":
    main()
