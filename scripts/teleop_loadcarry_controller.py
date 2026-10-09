#!/usr/bin/env python3
"""High-Level Directional Locomotion Controller for Load-Carrying MicroDuck.

Provides rate-limited, smooth velocity steering ([vx, vy, wz]) for:
1. Interactive Teleoperation (WASD / Arrow keys).
2. Autonomous Waypoint Navigation (Heading / P-controller).
3. Physics-preserving cantilever inertia protection (|dωz/dt| <= 1.5 rad/s^2).
"""

from __future__ import annotations
import math
import numpy as np


class HighLevelLocomotionController:
    """High-level controller converting directional inputs into smooth twist commands."""

    def __init__(
        self,
        max_vx_fwd: float = 0.16,
        max_vx_bwd: float = 0.08,
        max_vy: float = 0.10,
        max_wz: float = 0.60,
        accel_vx: float = 1.0,    # m/s^2
        accel_wz: float = 1.5,    # rad/s^2 (strictly prevents cantilever inertia tipping)
    ):
        self.max_vx_fwd = max_vx_fwd
        self.max_vx_bwd = max_vx_bwd
        self.max_vy = max_vy
        self.max_wz = max_wz
        self.accel_vx = accel_vx
        self.accel_wz = accel_wz

        # Current executed commands
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.wz: float = 0.0

        # Targets
        self.target_vx: float = 0.0
        self.target_vy: float = 0.0
        self.target_wz: float = 0.0

    def update_from_keys(self, keys_held: set[str] | list[str]):
        """Update target velocities from active keyboard state."""
        keys = set(keys_held)

        fwd = ("w" in keys or "up" in keys)
        bwd = ("s" in keys or "down" in keys)
        left = ("a" in keys or "left" in keys)
        right = ("d" in keys or "right" in keys)
        strafe_left = ("q" in keys)
        strafe_right = ("e" in keys)

        # Forward / Backward
        if fwd and not bwd:
            self.target_vx = self.max_vx_fwd
        elif bwd and not fwd:
            self.target_vx = -self.max_vx_bwd
        else:
            self.target_vx = 0.0

        # Turning (Yaw Rate)
        if left and not right:
            self.target_wz = self.max_wz
        elif right and not left:
            self.target_wz = -self.max_wz
        else:
            self.target_wz = 0.0

        # Lateral Strafe
        if strafe_left and not strafe_right:
            self.target_vy = self.max_vy
        elif strafe_right and not strafe_left:
            self.target_vy = -self.max_vy
        else:
            self.target_vy = 0.0

    def update_waypoint_heading(self, robot_x: float, robot_y: float, robot_yaw: float, target_x: float, target_y: float):
        """Autonomous steering toward a target Cartesian coordinate (xg, yg)."""
        dx = target_x - robot_x
        dy = target_y - robot_y
        dist = math.hypot(dx, dy)

        if dist < 0.10:  # Arrived
            self.target_vx = 0.0
            self.target_vy = 0.0
            self.target_wz = 0.0
            return

        target_heading = math.atan2(dy, dx)
        heading_err = math.atan2(math.sin(target_heading - robot_yaw), math.cos(target_heading - robot_yaw))

        # Proportional steering law
        kp_yaw = 2.0
        self.target_wz = float(np.clip(kp_yaw * heading_err, -self.max_wz, self.max_wz))

        # Cruise forward when heading is roughly aligned, slow down during sharp turn
        alignment = max(0.0, math.cos(heading_err))
        self.target_vx = float(np.clip(self.max_vx_fwd * alignment, 0.0, self.max_vx_fwd))
        self.target_vy = 0.0

    def step(self, dt: float = 0.02) -> np.ndarray:
        """Step rate limiter forward by dt (50 Hz control tick). Returns [vx, vy, wz]."""
        # Ramp vx
        dvx = np.clip(self.target_vx - self.vx, -self.accel_vx * dt, self.accel_vx * dt)
        self.vx += float(dvx)

        # Ramp vy
        dvy = np.clip(self.target_vy - self.vy, -self.accel_vx * dt, self.accel_vx * dt)
        self.vy += float(dvy)

        # Ramp wz with strictly bounded angular acceleration
        dwz = np.clip(self.target_wz - self.wz, -self.accel_wz * dt, self.accel_wz * dt)
        self.wz += float(dwz)

        return np.array([self.vx, self.vy, self.wz], dtype=np.float32)

    def stop(self):
        """Immediately zero targets."""
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_wz = 0.0


if __name__ == "__main__":
    controller = HighLevelLocomotionController()
    print("[HighLevelLocomotionController] Unit Test:")
    controller.update_from_keys(["w", "a"])
    for step in range(25):  # 0.5 sec at 50 Hz
        cmd = controller.step(0.02)
        if step % 5 == 0:
            print(f"  Step {step:02d} (t={step*0.02:.2f}s): vx={cmd[0]:.3f} m/s, vy={cmd[1]:.3f} m/s, wz={cmd[2]:.3f} rad/s")
    assert cmd[0] > 0.10, "vx should have ramped up"
    assert cmd[2] > 0.30, "wz should have ramped up"
    print("✓ Rate limiter & steering controller unit test PASSED!")
