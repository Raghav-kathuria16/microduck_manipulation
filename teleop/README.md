# Teleop App (Reserved)

This directory will contain the hierarchical teleoperation application
described in the Teleop/Data spec Part 3.

## Planned architecture

```
teleop/
├── app.py                  # Main entry point — launches GUI + sim
├── gui/
│   ├── viewer.py           # Multi-camera MuJoCo viewer (top-down + beak-cam)
│   └── overlay.py          # HUD: velocity, grasp state, episode status
├── controllers/
│   ├── locomotion.py       # Wraps the trained walk policy from microduck_rl
│   ├── head_aim.py         # IK/PD controller for neck+head → look-at point
│   ├── beak_fine.py        # IK controller for beak tip fine positioning
│   └── grasp.py            # Beak actuator open/close
├── input/
│   ├── keyboard.py         # Keyboard input handler
│   ├── gamepad.py          # Gamepad/controller input handler
│   └── mouse.py            # Mouse input for head aiming
├── recorder/
│   ├── episode_recorder.py # Per-timestep data logging
│   └── video_writer.py     # Camera frame capture to video
└── utils/
    ├── scene_loader.py     # Load apartment scene + configs
    └── reset.py            # Episode reset: randomize spawn/sock/lighting
```

## Design principles

1. Human controls **commands**, not joints (Teleop Spec §3.1–3.2)
2. Each controller layer is independent and composable
3. All logging follows `configs/data_logging.yaml` schema
4. Output format is LeRobotDataset (parquet + MP4)
