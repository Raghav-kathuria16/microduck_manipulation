#!/usr/bin/env python3
"""Generate orientation variations & floor-pickup readiness views for Pickleball & Badminton."""

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

def render_scene_cams(xml_content, out_prefix, cams, qpos_dict=None):
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
    if beak_jid >= 0:
        b_adr = model.jnt_qposadr[beak_jid]
        data.qpos[b_adr] = 1.55 if not qpos_dict or qpos_dict.get("beak_open", False) is False else 1.95

    if qpos_dict and "payload_qpos" in qpos_dict:
        name = qpos_dict["payload_name"]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
        if jid >= 0:
            adr = model.jnt_qposadr[jid]
            data.qpos[adr:adr+7] = qpos_dict["payload_qpos"]

    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=960)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    for name, cfg in cams.items():
        cam.distance = cfg["distance"]
        cam.elevation = cfg["elevation"]
        cam.azimuth = cfg["azimuth"]
        cam.lookat = cfg["lookat"]
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        img = Image.fromarray(pixels)
        out_file = os.path.join(_ARTIFACT_DIR, f"{out_prefix}_{name}.png")
        img.save(out_file)
        print(f"Saved: {out_file}")

def main():
    # Load base XML
    with open(_BASE_SCENE_XML, "r") as f:
        base_xml = f.read()

    assets_dir = os.path.join(os.path.dirname(_BASE_SCENE_XML), "assets")
    base_xml = base_xml.replace('meshdir="assets/"', f'meshdir="{assets_dir}/"')
    base_xml = base_xml.replace('elevation="-20"/>', 'elevation="-20" offwidth="1280" offheight="960"/>')

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
    base_xml = base_xml.replace("  </asset>", extra_assets)

    cyl_start = base_xml.find('<body name="cylinder"')
    cyl_end = base_xml.find('</body>\n  </worldbody>') + len('</body>')
    old_cyl = base_xml[cyl_start:cyl_end]

    # --- 1. Forward-Facing Pickleball Paddle (Grip clamped, face pointing straight forward) ---
    # Quat: identity [1, 0, 0, 0] or rotated so handle points into mouth and face points ahead (+X)
    # Beak clamps at mouth: pos ~ [0.038, 0, 0.228]. Paddle Z goes forward (+X).
    # Rotation: 90 deg around Y: quat = [cos(pi/4), 0, sin(pi/4), 0] = [0.7071, 0, 0.7071, 0]
    p_fwd_body = """    <body name="pickleball_paddle_fwd" pos="0.180 0 0.228" quat="0.7071068 0 0.7071068 0">
      <freejoint name="pickleball_paddle_fwd_free"/>
      <inertial pos="0 0 0.075" mass="0.215" diaginertia="0.0032 0.0028 0.00065"/>
      <geom name="pb_butt_cap_f" type="cylinder" pos="0 0 -0.160" size="0.016 0.006" material="trunk_base_material" friction="2.5 0.1 0.1"/>
      <geom name="pb_grip_f" type="cylinder" pos="0 0 -0.095" size="0.0135 0.055" material="grip_white" friction="2.5 0.1 0.1"/>
      <geom name="pb_clamp_neck_f" type="cylinder" pos="0 0 -0.035" size="0.004 0.020" material="neck_clamp" friction="3.5 0.2 0.2"/>
      <geom name="pb_throat_f" type="box" pos="0 0 0.040" size="0.026 0.007 0.015" material="edge_guard"/>
      <geom name="pb_face_core_f" type="box" pos="0 0 0.180" size="0.088 0.0065 0.118" material="face_cyan" friction="1.8 0.1 0.1"/>
      <geom name="pb_accent_front_f" type="box" pos="0 0.0068 0.180" size="0.075 0.0005 0.035" material="face_yellow"/>
      <geom name="pb_accent_back_f" type="box" pos="0 -0.0068 0.180" size="0.075 0.0005 0.035" material="face_yellow"/>
      <geom name="pb_edge_l_f" type="capsule" fromto="-0.095 0 0.075 -0.095 0 0.270" size="0.0075" material="edge_guard"/>
      <geom name="pb_edge_r_f" type="capsule" fromto="0.095 0 0.075 0.095 0 0.270" size="0.0075" material="edge_guard"/>
      <geom name="pb_edge_top_f" type="capsule" fromto="-0.065 0 0.302 0.065 0 0.302" size="0.0075" material="edge_guard"/>
      <geom name="pb_cor_tl_f" type="capsule" fromto="-0.095 0 0.270 -0.065 0 0.302" size="0.0075" material="edge_guard"/>
      <geom name="pb_cor_tr_f" type="capsule" fromto="0.095 0 0.270 0.065 0 0.302" size="0.0075" material="edge_guard"/>
      <geom name="pb_cor_bl_f" type="capsule" fromto="-0.095 0 0.075 -0.026 0 0.050" size="0.0075" material="edge_guard"/>
      <geom name="pb_cor_br_f" type="capsule" fromto="0.095 0 0.075 0.026 0 0.050" size="0.0075" material="edge_guard"/>
    </body>"""

    p_fwd_xml = base_xml.replace(old_cyl, p_fwd_body)
    p_fwd_xml = p_fwd_xml.replace('geom1="cylinder_geom"', 'geom1="pb_clamp_neck_f"')
    p_fwd_xml = p_fwd_xml.replace('body1="cylinder"', 'body1="pickleball_paddle_fwd"')

    fwd_cams = {
        "perspective": {"distance": 0.85, "elevation": -18.0, "azimuth": 135.0, "lookat": [0.15, 0.0, 0.16]},
        "side": {"distance": 0.80, "elevation": -2.0, "azimuth": 90.0, "lookat": [0.15, 0.0, 0.16]},
        "three_quarter_front": {"distance": 0.75, "elevation": -12.0, "azimuth": 155.0, "lookat": [0.15, 0.0, 0.16]},
    }
    print("Rendering Forward-Facing Pickleball Paddle Carry...")
    render_scene_cams(p_fwd_xml, "pickleball_forward_carry", fwd_cams)

    # --- 2. Floor-Resting Pickup Readiness (Pickleball resting on the floor in front of Microduck) ---
    # Placed on floor at X=0.18, Y=0, Z=0.015 (flat)
    p_floor_xml = p_fwd_xml.replace('pos="0.180 0 0.228"', 'pos="0.220 0 0.015"')
    floor_cams = {
        "ground_pickup_readiness": {"distance": 0.70, "elevation": -15.0, "azimuth": 130.0, "lookat": [0.12, 0.0, 0.08]},
        "ground_pickup_closeup": {"distance": 0.35, "elevation": -10.0, "azimuth": 115.0, "lookat": [0.10, 0.0, 0.06]},
    }
    print("Rendering Pickleball Floor Pickup Readiness...")
    render_scene_cams(p_floor_xml, "pickleball_floor_pickup", floor_cams, qpos_dict={"beak_open": True})

    # --- 3. Forward-Facing Badminton Racket Carry ---
    a, b, z_head = 0.095, 0.120, 0.220
    n_pts = 16
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts = [(a * math.cos(th), 0.0, z_head + b * math.sin(th)) for th in angles]
    frame_capsules = [
        f'      <geom name="bm_fr_f_{i}" type="capsule" fromto="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {pts[(i+1)%n_pts][0]:.4f} {pts[(i+1)%n_pts][1]:.4f} {pts[(i+1)%n_pts][2]:.4f}" size="0.0035" material="frame_blue"/>'
        for i, p in enumerate(pts)
    ]
    bm_frame_str = "\n".join(frame_capsules)

    b_fwd_body = f"""    <body name="badminton_racket_fwd" pos="0.320 0 0.228" quat="0.7071068 0 0.7071068 0">
      <freejoint name="badminton_racket_fwd_free"/>
      <inertial pos="0 0 0.045" mass="0.083" diaginertia="0.0035 0.0034 0.00015"/>
      <geom name="bm_butt_cap_f" type="cylinder" pos="0 0 -0.335" size="0.0135 0.004" material="butt_green"/>
      <geom name="bm_grip_f" type="cylinder" pos="0 0 -0.255" size="0.011 0.075" material="grip_black"/>
      <geom name="bm_cone_f" type="cylinder" pos="0 0 -0.172" size="0.0075 0.008" material="cone_grey"/>
      <geom name="bm_shaft_low_f" type="cylinder" pos="0 0 -0.090" size="0.0035 0.065" material="shaft_carbon"/>
      <geom name="bm_clamp_zone_f" type="cylinder" pos="0 0 -0.280" size="0.004 0.025" material="shaft_carbon" friction="3.5 0.2 0.2"/>
      <geom name="bm_shaft_high_f" type="cylinder" pos="0 0 0.050" size="0.0035 0.025" material="shaft_carbon"/>
      <geom name="bm_t_joint_f" type="cylinder" pos="0 0 0.080" size="0.0055 0.006" material="t_joint_gold"/>
{bm_frame_str}
      <geom name="bm_strings_f" type="ellipsoid" pos="0 0 {z_head:.4f}" size="0.091 0.0002 0.116" material="strings_translucent" contype="0" conaffinity="0"/>
    </body>"""

    b_fwd_xml = base_xml.replace(old_cyl, b_fwd_body)
    b_fwd_xml = b_fwd_xml.replace('geom1="cylinder_geom"', 'geom1="bm_clamp_zone_f"')
    b_fwd_xml = b_fwd_xml.replace('body1="cylinder"', 'body1="badminton_racket_fwd"')

    bm_fwd_cams = {
        "perspective": {"distance": 1.05, "elevation": -16.0, "azimuth": 135.0, "lookat": [0.25, 0.0, 0.16]},
        "side": {"distance": 1.00, "elevation": -2.0, "azimuth": 90.0, "lookat": [0.25, 0.0, 0.16]},
    }
    print("Rendering Forward-Facing Badminton Racket Carry...")
    render_scene_cams(b_fwd_xml, "badminton_forward_carry", bm_fwd_cams)

    print("All additional verification renders completed!")

if __name__ == "__main__":
    main()
