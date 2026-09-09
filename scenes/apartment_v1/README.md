# Apartment v1 — Scene Assets

This directory will contain the MuJoCo scene definition for the v1
apartment layout described in the Teleop/Data spec Part 2:

- Single room (~3.5m × 3.0m) + alcove (~1.5m × 1.5m), duck-scale
- Floor, walls, rug, laundry basket, 1–2 furniture pieces
- Sock spawn zone markers (invisible)
- Camera rig mount points

## Expected files (to be created during Phase 1 implementation)

- `apartment_v1.xml` — main MuJoCo MJCF scene file
- `assets/` — textures, meshes for furniture/basket/rug
- `includes/` — modular MJCF includes (sock body, basket body, etc.)

## Design constraints

- References the robot MJCF from `microduck_rl` (via `external/` pin),
  does NOT contain a copy of the robot model.
- All parametrizable aspects (spawn zones, lighting, etc.) are controlled
  via `configs/scene_apartment_v1.yaml`, not hard-coded in the XML.
- Scene should load and render on MuJoCo CPU mode (MacBook M2).
