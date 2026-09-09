#!/usr/bin/env python3
"""Interactive 3D viewer for the Microduck Apartment Scene.

Loads scenes/apartment_v1/apartment_v1.xml and opens the native MuJoCo viewer.
Provides interactive randomization and beak control.
"""

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

# Add scripts dir to path to import randomizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from randomize_scene import load_scene_config, randomize_episode, format_episode


def main():
    parser = argparse.ArgumentParser(description="Microduck Apartment Scene Viewer")
    parser.add_argument("--randomize", action="store_true", help="Apply a random configuration on launch")
    parser.add_argument("--lighting", type=str, default=None, help="Force a specific lighting preset")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = parser.parse_args()

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_xml_path = os.path.join(workspace_root, "scenes", "apartment_v1", "apartment_v1.xml")
    config_path = os.path.join(workspace_root, "configs", "scene_apartment_v1.yaml")

    if not os.path.exists(scene_xml_path):
        print(f"Error: Scene XML not found at {scene_xml_path}")
        sys.exit(1)

    print("=" * 60)
    print("🦆 Microduck Manipulation — Apartment v1 Scene Viewer")
    print("=" * 60)
    print(f"Loading scene: {scene_xml_path}")

    scene_config = None
    if os.path.exists(config_path):
        scene_config = load_scene_config(config_path)
        print(f"Scene Name: {scene_config.scene_name}")
        print(f"Spawn Zones: {[z.name for z in scene_config.sock_spawn_zones]}")
    else:
        print("Warning: config not found, randomization will be disabled.")

    model = mujoco.MjModel.from_xml_path(scene_xml_path)
    data = mujoco.MjData(model)
    
    rng = np.random.default_rng(args.seed)

    if args.randomize and scene_config is not None:
        ep = randomize_episode(model, data, scene_config, rng, args.lighting)
        print("\n" + format_episode(ep))

    print("\nControls inside Viewer:")
    print("  - [Tab] Cycle camera views (Free, top_down, chase, beak_cam)")
    print("  - [Space] Pause / resume simulation physics")
    print("  - [Backspace] Reset simulation")
    print("  - [Double-click body + Ctrl-drag] Apply interactive forces (drag duck or sock)")
    if scene_config is not None:
        print("  - [R] Re-randomize scene (Duck pose, sock spawn, lighting)")
    print("  - [G] Toggle beak open/closed")
    print("  - [Esc] Exit")
    print("=" * 60)

    # Find the beak actuator
    act_beak_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_beak")
    beak_open = False

    def keyboard_callback(keycode: int):
        nonlocal beak_open
        # Keycode for 'R' is 82 (or 'r' 114)
        if keycode in (82, 114) and scene_config is not None:
            # Re-randomize
            ep = randomize_episode(model, data, scene_config, rng, args.lighting)
            print("\n--- Re-randomized ---")
            print(format_episode(ep))
            
        # Keycode for 'G' is 71 (or 'g' 103)
        elif keycode in (71, 103) and act_beak_id >= 0:
            beak_open = not beak_open
            data.ctrl[act_beak_id] = 0.5 if beak_open else 0.0
            state = "OPEN" if beak_open else "CLOSED"
            print(f"Beak: {state}")

    # Launch native passive viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Rate limit to ~60 Hz simulation step display
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
