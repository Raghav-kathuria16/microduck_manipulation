#!/usr/bin/env python3
"""Automated smoke test for the Microduck Apartment Scene.

Loads the scene, verifies geometry/bodies/cameras/actuators,
tests physics stability, tests randomization, and tests the
beak actuator.
"""

import os
import sys

import mujoco
import numpy as np

# Add the scripts directory to sys.path so we can import the randomizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from randomize_scene import load_scene_config, randomize_episode


def check(name: str, condition: bool, error_msg: str = "") -> bool:
    """Helper to print pass/fail for a check."""
    if condition:
        print(f"  ✅ {name}")
        return True
    else:
        print(f"  ❌ {name}")
        if error_msg:
            print(f"     └─ {error_msg}")
        return False


def main():
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_xml_path = os.path.join(workspace_root, "scenes", "apartment_v1", "apartment_v1.xml")
    config_path = os.path.join(workspace_root, "configs", "scene_apartment_v1.yaml")

    print("=" * 60)
    print("🦆 Microduck Manipulation — Scene Smoke Test")
    print("=" * 60)

    # -----------------------------------------------------------------------
    print("\n[1/6] Loading XML and Config")
    # -----------------------------------------------------------------------
    if not os.path.exists(scene_xml_path):
        print(f"❌ Could not find {scene_xml_path}")
        sys.exit(1)
        
    try:
        model = mujoco.MjModel.from_xml_path(scene_xml_path)
        data = mujoco.MjData(model)
        check("XML parsed successfully", True)
    except Exception as e:
        check("XML parsed successfully", False, str(e))
        sys.exit(1)
        
    try:
        config = load_scene_config(config_path)
        check("Config parsed successfully", True)
    except Exception as e:
        check("Config parsed successfully", False, str(e))
        sys.exit(1)


    # -----------------------------------------------------------------------
    print("\n[2/6] Verifying Expected Bodies & Cameras")
    # -----------------------------------------------------------------------
    expected_bodies = ["duck_base", "sock", "laundry_basket", "duck_head", "duck_lower_beak"]
    all_passed = True
    for body in expected_bodies:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        passed = check(f"Body '{body}' exists", body_id >= 0, "Missing from XML")
        all_passed = all_passed and passed

    expected_cameras = ["top_down", "chase", "beak_cam"]
    for cam in expected_cameras:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        passed = check(f"Camera '{cam}' exists", cam_id >= 0, "Missing from XML")
        all_passed = all_passed and passed
        
    if not all_passed:
        print("\n❌ Failed structural checks. Exiting.")
        sys.exit(1)


    # -----------------------------------------------------------------------
    print("\n[3/6] Verifying Actuators")
    # -----------------------------------------------------------------------
    expected_actuators = [
        "act_vel_x", "act_vel_y", "act_yaw", 
        "act_neck_yaw", "act_neck_pitch", "act_head_pitch", "act_beak"
    ]
    all_passed = True
    for act in expected_actuators:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
        passed = check(f"Actuator '{act}' exists", act_id >= 0, "Missing from XML")
        all_passed = all_passed and passed
        
    if not all_passed:
        print("\n❌ Failed actuator checks. Exiting.")
        sys.exit(1)


    # -----------------------------------------------------------------------
    print("\n[4/6] Testing Physics Stability (1000 steps)")
    # -----------------------------------------------------------------------
    mujoco.mj_resetData(model, data)
    
    # Step simulation 1000 times
    unstable = False
    for _ in range(1000):
        mujoco.mj_step(model, data)
        # Check for NaN in qpos or qvel (proxy for explosion/divergence)
        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            unstable = True
            break
            
    check("Physics remained stable (no NaN)", not unstable, "Simulation diverged")
    if unstable:
        sys.exit(1)


    # -----------------------------------------------------------------------
    print("\n[5/6] Testing Randomization (5 configs)")
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(42)
    all_passed = True
    
    jnt_sock = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "sock_joint")
    sock_qpos_adr = model.jnt_qposadr[jnt_sock]
    
    for i in range(5):
        mujoco.mj_resetData(model, data)
        ep = randomize_episode(model, data, config, rng)
        
        # Verify sock is actually where we asked it to be (in data.qpos)
        actual_x = data.qpos[sock_qpos_adr + 0]
        actual_y = data.qpos[sock_qpos_adr + 1]
        
        target_x = ep.sock_position[0]
        target_y = ep.sock_position[1]
        
        # They should be exactly equal since we just set them
        dist = np.sqrt((actual_x - target_x)**2 + (actual_y - target_y)**2)
        passed = dist < 1e-5
        
        check(f"Random config {i+1}: Sock at {target_x:.2f}, {target_y:.2f}", passed, f"Actual was {actual_x:.2f}, {actual_y:.2f}")
        all_passed = all_passed and passed
        
    if not all_passed:
         sys.exit(1)


    # -----------------------------------------------------------------------
    print("\n[6/6] Testing Beak Actuator")
    # -----------------------------------------------------------------------
    mujoco.mj_resetData(model, data)
    mujoco.mj_step(model, data)
    
    jnt_beak = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak_joint")
    beak_qpos_adr = model.jnt_qposadr[jnt_beak]
    
    initial_beak_pos = data.qpos[beak_qpos_adr]
    
    act_beak_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_beak")
    
    # Command the beak to open
    data.ctrl[act_beak_id] = 0.5
    
    # Step for a bit to let the PD controller move the joint
    for _ in range(50):
        mujoco.mj_step(model, data)
        
    final_beak_pos = data.qpos[beak_qpos_adr]
    
    passed = check("Beak actuator successfully moves beak joint", final_beak_pos > initial_beak_pos + 0.1, 
                   f"Initial: {initial_beak_pos:.3f}, Final: {final_beak_pos:.3f}")
                   
    if not passed:
        sys.exit(1)


    print("\n" + "=" * 60)
    print("✨ All checks passed! Scene is loadable and stable.")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
