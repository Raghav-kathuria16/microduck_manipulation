"""Locomotion controller: WASD → velocity command → walk policy.

Translates keyboard state (held keys) into a smooth [vx, wz] velocity
command that is fed into the PolicyRunner every control tick.

Key mapping (from configs/teleop_controls.yaml):
  W / ↑   — forward
  S / ↓   — backward
  A       — turn left
  D       — turn right
  SHIFT   — 2× speed multiplier
"""

from __future__ import annotations
# pyrefly: ignore [missing-import]
import numpy as np
from teleop.utils.scene_loader import VEL_FWD, VEL_BACK, VEL_ANG

# Ramp rate: how fast the command changes per second (smooth acceleration)
RAMP_RATE = 4.0   # units/sec

class LocomotionController:
    """Converts key states into a smoothly ramped [vx, vy, wz] command."""

    def __init__(self):
        # Current command (smoothed)
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.wz: float = 0.0

        # Target (from raw key state)
        self._target_vx: float = 0.0
        self._target_vy: float = 0.0
        self._target_wz: float = 0.0

    def update_from_keys(self, keys_held: set[str], shift: bool = False):
        """Update targets from the current set of held keys.

        Args:
            keys_held: set of key names ('w', 's', 'a', 'd', 'up', 'down', 'left', 'right')
            shift: True if shift modifier is held
        """
        scale = 2.0 if shift else 1.0

        # Forward / backward
        fwd  = ('w' in keys_held or 'up'   in keys_held)
        back = ('s' in keys_held or 'down' in keys_held)
        # Turn
        left  = ('a' in keys_held)
        right = ('d' in keys_held)

        target_vx = 0.0
        if fwd  and not back:
            target_vx = VEL_FWD * scale
        elif back and not fwd:
            target_vx = VEL_BACK * scale

        target_wz = 0.0
        if left  and not right:
            target_wz =  VEL_ANG
        elif right and not left:
            target_wz = -VEL_ANG

        self._target_vx = float(np.clip(target_vx, VEL_BACK * 2, VEL_FWD * 2))
        self._target_wz = float(np.clip(target_wz, -VEL_ANG, VEL_ANG))

    def tick(self, dt: float):
        """Ramp current command toward targets. Call each frame."""
        self.vx = _ramp(self.vx, self._target_vx, RAMP_RATE * dt)
        self.vy = _ramp(self.vy, self._target_vy, RAMP_RATE * dt)
        self.wz = _ramp(self.wz, self._target_wz, RAMP_RATE * dt)

    @property
    def command(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.wz)

    def is_active(self) -> bool:
        return abs(self.vx) > 0.01 or abs(self.wz) > 0.05

    def stop(self):
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_wz = 0.0


def _ramp(current: float, target: float, max_step: float) -> float:
    diff = target - current
    return current + float(np.clip(diff, -max_step, max_step))
