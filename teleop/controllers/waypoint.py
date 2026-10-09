"""Waypoint navigation controller: click-to-navigate like ROS 2D Nav Goal.

Given a target (x, y) in world coordinates, computes (vx, wz) to drive the
robot toward the target using a simple proportional controller:
  1. Compute heading error to target
  2. Turn in place if heading error is large
  3. Walk forward while correcting heading when roughly aligned
  4. Stop when within arrival radius
"""

from __future__ import annotations
import math
import numpy as np


# Controller gains
KP_TURN = 2.5       # rad/s per rad of heading error
KP_DIST = 1.5       # m/s per metre of distance (clamped to max)
MAX_VX  = 5      # m/s max forward speed
MAX_WZ  = 5       # rad/s max angular speed
TURN_THRESHOLD = 0.3  # rad — must be within this to start walking forward
ARRIVE_RADIUS  = 0.15 # metres — "close enough" to target


class WaypointController:
    """Computes velocity commands to navigate toward a waypoint."""

    def __init__(self):
        self.target_x: float | None = None
        self.target_y: float | None = None
        self.active: bool = False
        self.arrived: bool = False

    def set_target(self, x: float, y: float):
        """Set a new navigation target. Activates the controller."""
        self.target_x = x
        self.target_y = y
        self.active = True
        self.arrived = False

    def stop(self):
        """Immediately stop navigation."""
        self.active = False
        self.arrived = False
        self.target_x = None
        self.target_y = None

    def compute(self, robot_x: float, robot_y: float, robot_yaw: float) -> tuple[float, float]:
        """Compute (vx, wz) given the robot's current pose.

        Args:
            robot_x, robot_y: Robot position in world frame.
            robot_yaw: Robot heading in radians (from freejoint quaternion or yaw joint).

        Returns:
            (vx, wz) velocity command.
        """
        if not self.active or self.target_x is None:
            return (0.0, 0.0)

        dx = self.target_x - robot_x
        dy = self.target_y - robot_y
        distance = math.sqrt(dx * dx + dy * dy)

        # Arrived?
        if distance < ARRIVE_RADIUS:
            self.active = False
            self.arrived = True
            return (0.0, 0.0)

        # Desired heading to target
        target_heading = math.atan2(dy, dx)

        # Heading error (normalized to [-pi, pi])
        heading_error = _normalize_angle(target_heading - robot_yaw)

        # Compute angular velocity
        wz = float(np.clip(KP_TURN * heading_error, -MAX_WZ, MAX_WZ))

        # Compute forward velocity — only walk when roughly facing the target
        if abs(heading_error) > TURN_THRESHOLD:
            vx = 0.0  # turn in place first
        else:
            vx = float(np.clip(KP_DIST * distance, 0.0, MAX_VX))
            # Slow down when turning
            vx *= max(0.0, 1.0 - abs(heading_error) / TURN_THRESHOLD)

        return (vx, wz)

    @property
    def status(self) -> str:
        if self.arrived:
            return "arrived"
        elif self.active:
            return "navigating"
        else:
            return "idle"


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
