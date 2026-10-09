"""Grasp controller: G key toggles beak open/close.

Simple wrapper — the actual actuator write happens in PolicyRunner.toggle_beak().
This class tracks the grasp state for HUD display and logging.
"""

from __future__ import annotations


class GraspController:
    """Tracks and reports the grasp (beak) state."""

    def __init__(self):
        self.open: bool = False   # True = beak open (grasping)

    def toggle(self):
        self.open = not self.open
        return self.open

    @property
    def state_label(self) -> str:
        return "OPEN" if self.open else "CLOSED"

    @property
    def state_float(self) -> float:
        """Float action label for BC logging (1.0 = open, 0.0 = closed)."""
        return 1.0 if self.open else 0.0
