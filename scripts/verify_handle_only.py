#!/usr/bin/env python3
"""Ensure ONLY the 8.0mm handle is inside the mouth cavity for both Pickleball and Badminton."""

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

# Mouth silicone pad width is 88mm (-44mm to +44mm along Y).
# To ensure ONLY THE HANDLE is inside the mouth:
# Handle grip must be 96mm long (-48mm to +48mm), centered at Y = 0.
# The paddle face/head begins at Y > +48mm (completely outside the mouth!).
# The butt cap is at Y < -48mm (completely outside the mouth!).

def create_handle_only_xml(racket_type="pickleball", orientation="transverse"):
    with open(_BASE_SCENE_XML, "r") as f:
        base = f.read()

    assets_dir = os.path.join(os.path.dirname(_BASE_SCENE_XML), "assets")
    base = base.replace('meshdir="assets/"', f'meshdir="{assets_dir}/"')
    base = base.replace('elevation="-20"/>', 'elevation="-20" offwidth="1280" offheight="960"/>')

    extra_assets = """
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

    if orientation == "transverse":
        # Held sideways across the beak (along Y)
        # Handle length = 96mm (radius 4mm = 8.0mm diameter)
        # Mouth center is at Y=0, X=0.070, Z=0.226
        # Handle spans from local z = -0.048 to +0.048 (world y = -0.048 to +0.048)
        # Butt cap is at local z = -0.052 (world y = -0.052, OUTSIDE mouth)
        # Throat & head start at local z = +0.048 (world y = +0.048, OUTSIDE mouth!)
        quat_str = 'quat="0.7071068 0.7071068 0 0"'
        pos_str = 'pos="0.065 0.0 0.228"'

        if racket_type == "pickleball":
            body_str = f"""    <body name="racket" {pos_str} {quat_str}>
      <freejoint name="racket_free"/>
      <!-- Mass: 75g -->
      <inertial pos="0 0 0" mass="0.075" diaginertia="0.00018 0.00015 0.000035"/>
      <!-- Butt cap: completely outside mouth on the right (z = -0.064) -->
      <geom name="racket_butt" type="cylinder" pos="0 0 -0.064" size="0.007 0.002" material="pb_butt_cap" friction="2.5 0.1 0.1"/>
      <!-- ONLY THIS 8.0mm HANDLE IS INSIDE THE MOUTH (spans z = -0.060 to +0.060, length 120mm) -->
      <geom name="racket_handle" type="cylinder" pos="0 0 0" size="0.004 0.060" material="pb_grip_white" friction="3.5 0.2 0.2"/>
      <!-- Throat: completely outside mouth on the left (starts at z = +0.068, over 22mm clear of jaw cheek!) -->
      <geom name="racket_throat" type="box" pos="0 0 0.068" size="0.012 0.003 0.008" material="pb_edge_guard"/>
      <!-- Paddle face: completely outside mouth (z from +0.082 to +0.174) -->
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
            # Badminton: Handle length = 110mm inside mouth (z = -0.055 to +0.055)
            # Shaft and head start at z > +0.058 (over 12mm clear of jaw cheek!)
            body_str = f"""    <body name="racket" {pos_str} {quat_str}>
      <freejoint name="racket_free"/>
      <!-- Mass: 40g -->
      <inertial pos="0 0 0" mass="0.040" diaginertia="0.00014 0.00014 0.000008"/>
      <!-- Butt cap: outside mouth on the right (z = -0.058) -->
      <geom name="racket_butt" type="cylinder" pos="0 0 -0.058" size="0.0055 0.0015" material="bm_butt_green" friction="2.0 0.1 0.1"/>
      <!-- ONLY THIS 8.0mm HANDLE IS INSIDE THE MOUTH (spans z = -0.055 to +0.055, length 110mm) -->
      <geom name="racket_handle" type="cylinder" pos="0 0 0" size="0.004 0.055" material="bm_grip_black" friction="3.5 0.2 0.2"/>
      <!-- Shaft: completely outside mouth on the left (z from +0.058 to +0.098) -->
      <geom name="racket_cone" type="cylinder" pos="0 0 0.058" size="0.003 0.003" material="bm_cone_grey"/>
      <geom name="racket_shaft" type="cylinder" pos="0 0 0.078" size="0.002 0.0175" material="bm_shaft_carbon" friction="2.5 0.1 0.1"/>
      <geom name="racket_t_joint" type="cylinder" pos="0 0 0.098" size="0.0028 0.002" material="bm_t_joint_gold"/>
      <!-- Oval frame: completely outside mouth (z around 0.130) -->
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
    return xml


def render_handle_proof():
    for rtype in ["pickleball", "badminton"]:
        xml_str = create_handle_only_xml(racket_type=rtype)
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)

        # Standing pose
        data.qpos[0:3] = [0.0, 0.0, 0.125]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # Beak clamped firmly around 8mm handle: q_beak = 1.55 rad (closed)
        beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
        data.qpos[model.jnt_qposadr[beak_jid]] = 1.55

        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=720, width=960)

        # Verify contacts: confirm that ONLY racket_handle touches the mouth!
        r_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "racket")
        touching_geoms = set()
        for ci in range(data.ncon):
            con = data.contact[ci]
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            if b1 == r_bid or b2 == r_bid:
                g_other = con.geom2 if b1 == r_bid else con.geom1
                g_racket = con.geom1 if b1 == r_bid else con.geom2
                touching_geoms.add((model.geom(g_racket).name, model.geom(g_other).name))

        print(f"\n[{rtype.upper()}] Touching contact pairs:")
        for gr, go in touching_geoms:
            print(f"   Racket Geom: '{gr}' <--> Robot Geom: '{go}'")

        # Close-up camera focused directly into mouth cavity
        cam_close = mujoco.MjvCamera()
        cam_close.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam_close.distance = 0.22
        cam_close.elevation = -4.0
        cam_close.azimuth = 135.0
        cam_close.lookat = [0.070, 0.0, 0.228]

        # Front view checking mouth aperture span
        cam_front = mujoco.MjvCamera()
        cam_front.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam_front.distance = 0.40
        cam_front.elevation = 0.0
        cam_front.azimuth = 180.0
        cam_front.lookat = [0.070, 0.0, 0.228]

        # Context view
        cam_ctx = mujoco.MjvCamera()
        cam_ctx.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam_ctx.distance = 0.65
        cam_ctx.elevation = -14.0
        cam_ctx.azimuth = 135.0
        cam_ctx.lookat = [0.070, 0.0, 0.16]

        cams = {
            "handle_only_mouth_closeup": cam_close,
            "handle_only_front_aperture": cam_front,
            "handle_only_full_context": cam_ctx,
        }

        for cname, c in cams.items():
            renderer.update_scene(data, camera=c)
            img = Image.fromarray(renderer.render())
            p = os.path.join(_ARTIFACT_DIR, f"{rtype}_{cname}.png")
            img.save(p)
            print(f"Saved: {p}")


if __name__ == "__main__":
    render_handle_proof()
