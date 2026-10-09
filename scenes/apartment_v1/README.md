# Microduck Apartment Scene v1

This directory contains the MuJoCo XML and assets for the `apartment_v1` scene, designed for the Microduck pick-and-place task.

## Layout

The scene represents a single-room-plus-alcove apartment segment at duck-scale (~3.5m x 3m total).

```text
 ┌───────────────────────────────┬──────────────┐
 │                                │              │
 │        MAIN ROOM               │   ALCOVE     │
 │   ┌────┐                       │   ┌────────┐ │
 │   │RUG │      [duck spawn]     │   │ BASKET │ │
 │   └────┘         ●             │   │ (target)│ │
 │                                │   └────────┘ │
 │     ○ sock spawn zone (A)      │              │
 │           ○ sock spawn zone (B)│              │
 │                                │              │
 └───────────────────────────────┴──────────────┘
```

## Inventory

| Element | Name in XML | Type | Notes |
|---------|-------------|------|-------|
| **Bodies** | `duck_base` | Robot Proxy | *See note below about the proxy model* |
| | `sock` | Target Object | ~50g rigid capsule approximation |
| | `laundry_basket` | Target Receptacle | Made of 5 box geoms |
| **Cameras** | `top_down` | Orthographic | God's-eye view for spatial awareness |
| | `chase` | Perspective | Follows the duck |
| | `beak_cam` | Perspective | Mounted to `duck_head` |
| **Actuators** | `act_vel_x`, `act_vel_y`, `act_yaw` | Velocity | Base movement |
| | `act_neck_yaw`, `act_neck_pitch` | Position | Neck joints |
| | `act_head_pitch`, `act_beak` | Position | Head and beak joints |

## How to Run

### Interactive Viewer
To view the scene and play with it interactively, use the viewer script from the repository root. Note: On macOS, you must use `mjpython` instead of `python3` to run the interactive viewer.

```bash
# Basic load
.venv/bin/mjpython scripts/view_scene.py

# Load with random duck pose, sock spawn, and lighting
.venv/bin/mjpython scripts/view_scene.py --randomize
```
**Controls:**
- **Tab**: Cycle cameras (Free, `top_down`, `chase`, `beak_cam`)
- **R**: Re-randomize the scene on the fly
- **G**: Toggle the beak open/closed
- **Double-click + Ctrl-drag**: Apply forces to objects

### Automated Smoke Test
To verify the scene geometry, physics stability, and randomization pipeline programmatically (headless):

```bash
.venv/bin/python scripts/test_scene.py
```

## Note on the Duck Proxy

The FSD and RL environments will use the true Microduck MJCF (`pollen-robotics/microduck_rl`). For this initial Phase 1 scene build, a **simplified proxy duck** (ellipsoid torso, capsule feet, simplified neck/beak joints) is used. 

This proxy is sufficient to build the apartment geometry, test cameras, validate the randomization pipeline, and ensure the sock and basket physics work. The proxy will be swapped out for the real MJCF during Phase 2 (Teleop Implementation).
