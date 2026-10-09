#!/usr/bin/env python3
"""Sync external dependencies for Microduck Manipulation.

Reads external/microduck_rl_pin.json and clones the microduck_rl
repository into the .external_repo/ directory. Creates necessary
symlinks for assets so MuJoCo can resolve them.
"""

import json
import os
import shutil
import subprocess
import sys

def main():
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pin_file = os.path.join(workspace_root, "external", "microduck_rl_pin.json")
    repo_dir = os.path.join(workspace_root, ".external_repo")
    
    if not os.path.exists(pin_file):
        print(f"Error: Pin file not found at {pin_file}")
        sys.exit(1)
        
    with open(pin_file, "r") as f:
        pin_data = json.load(f)
        
    url = pin_data.get("url")
    if not url:
        print("Error: No URL found in pin file.")
        sys.exit(1)
        
    print(f"Syncing microduck_rl from {url}...")
    
    if not os.path.exists(repo_dir):
        print(f"Cloning into {repo_dir}...")
        subprocess.run(["git", "clone", url, repo_dir], check=True)
    else:
        print(f"Repository already exists at {repo_dir}, fetching latest...")
        subprocess.run(["git", "fetch", "--all"], cwd=repo_dir, check=True)
        
    commit = pin_data.get("pinned_commit")
    tag = pin_data.get("pinned_tag")
    
    if commit:
        print(f"Checking out pinned commit: {commit}")
        subprocess.run(["git", "checkout", commit], cwd=repo_dir, check=True)
    elif tag:
        print(f"Checking out pinned tag: {tag}")
        subprocess.run(["git", "checkout", tag], cwd=repo_dir, check=True)
    else:
        print("No pinned commit or tag found. Using default branch.")
        # Ensure we're on a clean branch, e.g., main
        subprocess.run(["git", "checkout", "main"], cwd=repo_dir, check=False)
        subprocess.run(["git", "pull"], cwd=repo_dir, check=False)
        
    # Create symlink for assets so the scene can load the robot meshes
    assets_target = os.path.join(repo_dir, "src", "mjlab_microduck", "robot", "microduck", "assets")
    assets_link = os.path.join(workspace_root, "scenes", "apartment_v1", "assets")
    
    if os.path.islink(assets_link):
        os.unlink(assets_link)
    elif os.path.exists(assets_link):
        shutil.rmtree(assets_link)
        
    # Use a relative symlink
    rel_target = os.path.relpath(assets_target, start=os.path.dirname(assets_link))
    os.symlink(rel_target, assets_link)
    
    print(f"Created symlink: {assets_link} -> {rel_target}")
    
    # Also symlink all included XMLs from the robot to our scene directory
    # so we don't have to rewrite the includes inside robot_allcollisions.xml
    robot_dir = os.path.join(repo_dir, "src", "mjlab_microduck", "robot", "microduck")
    for fname in ["joints_properties.xml", "sensors.xml", "additional.xml", "allcollisions_contacts.xml"]:
        src = os.path.join(robot_dir, fname)
        dst = os.path.join(workspace_root, "scenes", "apartment_v1", fname)
        if os.path.exists(src):
            if os.path.islink(dst):
                os.unlink(dst)
            elif os.path.exists(dst):
                os.remove(dst)
            rel_src = os.path.relpath(src, start=os.path.dirname(dst))
            os.symlink(rel_src, dst)
            print(f"Created symlink for dependency: {fname}")

    print("Sync complete.")

if __name__ == "__main__":
    main()
