"""Head aim controller: arrow keys → head offset commands.

Maps arrow keys to neck_pitch / head_pitch / head_yaw increments that are
fed into the PolicyRunner's command slots [3:7] (new-cmd-obs flavor).
The policy itself produces the correct joint positions — this is a
*command*, not a direct ctrl override.

Key mapping (from configs/teleop_controls.yaml):
  ↑ / ↓   — head_pitch (look up / down)
  ← / →   — head_yaw   (look left / right)
  Z / X   — neck_pitch  (nod)
  SPACE   — reset head to centre
"""

from __future__ import annotations
import numpy as np

# Per-key step size (radians per press in discrete mode,
# radians per second in continuous held mode)
HEAD_STEP_RAD  = 0.08   # radians/press
HEAD_MAX_PITCH = 1.1    # ±rad  (training cap)
HEAD_MAX_YAW   = 1.4    # ±rad
HEAD_MAX_NECK  = 1.1    # ±rad
HEAD_MAX_ROLL  = 0.44   # ±rad  (from joint limit)


class HeadAimController:
    """Maintains a 4D head offset [neck, pitch, yaw, roll] updated by key events."""

    def __init__(self):
        # [neck_pitch_offset, head_pitch_offset, head_yaw_offset, head_roll_offset]
        self.offset = np.zeros(4, dtype=np.float32)

    def apply_key(self, key: str):
        """Apply a single discrete key press to the head offset."""
        step = HEAD_STEP_RAD
        if key in ("up", "look_up"):
            self.offset[1] = float(np.clip(self.offset[1] + step, -HEAD_MAX_PITCH, HEAD_MAX_PITCH))
        elif key in ("down", "look_down"):
            self.offset[1] = float(np.clip(self.offset[1] - step, -HEAD_MAX_PITCH, HEAD_MAX_PITCH))
        elif key in ("left", "look_left"):
            self.offset[2] = float(np.clip(self.offset[2] + step, -HEAD_MAX_YAW, HEAD_MAX_YAW))
        elif key in ("right", "look_right"):
            self.offset[2] = float(np.clip(self.offset[2] - step, -HEAD_MAX_YAW, HEAD_MAX_YAW))
        elif key == 'z':
            self.offset[0] = float(np.clip(self.offset[0] + step, -HEAD_MAX_NECK, HEAD_MAX_NECK))
        elif key == 'x':
            self.offset[0] = float(np.clip(self.offset[0] - step, -HEAD_MAX_NECK, HEAD_MAX_NECK))

    def reset(self):
        """Centre the head (Space key)."""
        self.offset[:] = 0.0

    @property
    def neck_pitch(self) -> float:
        return float(self.offset[0])

    @property
    def head_pitch(self) -> float:
        return float(self.offset[1])

    @property
    def head_yaw(self) -> float:
        return float(self.offset[2])

    @property
    def head_roll(self) -> float:
        return float(self.offset[3])
