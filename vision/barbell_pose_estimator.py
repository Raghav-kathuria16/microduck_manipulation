"""
Barbell Pose Estimator and Visual Nudge Controller for MicroDuck.

Provides real-time 3D pose estimation of a barbell resting on the floor from
MicroDuck's onboard head_camera, analytical back-projection, visual HUD rendering,
and closed-loop head and locomotion nudge commands for millimeter-accurate grasping.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class PoseEstimate:
    """Estimated 3D pose and image projections of the barbell."""
    pos_cam: np.ndarray        # [X_right, Y_up, Depth_forward] in meters
    pos_mouth: np.ndarray      # [dx_fwd, dy_left, dz_up] relative to mouth center in meters
    yaw_rel: float             # relative yaw angle in radians (0 = perpendicular to bill)
    pixel_center: Tuple[float, float]
    pixel_left: Tuple[float, float]
    pixel_right: Tuple[float, float]
    distance: float            # 3D distance in meters
    is_visible: bool
    status: str = "TRACKING"


class BarbellPoseEstimator:
    """Perception module for MicroDuck's head_camera to track and align with barbell."""

    def __init__(
        self,
        img_width: int = 640,
        img_height: int = 480,
        fovy_deg: float = 45.0,
        plate_baseline_m: float = 0.168,
        plate_radius_m: float = 0.028,
        shaft_radius_m: float = 0.004,
        noise_pos_std: float = 0.002,     # 2 mm Gaussian noise for sim domain randomization
        noise_yaw_std: float = 0.026,     # 1.5 deg Gaussian noise
    ):
        self.w = img_width
        self.h = img_height
        self.fovy = fovy_deg
        self.f = 0.5 * self.h / math.tan(math.radians(self.fovy) / 2.0)
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0
        self.plate_baseline = plate_baseline_m
        self.plate_radius = plate_radius_m
        self.shaft_radius = shaft_radius_m
        self.noise_pos_std = noise_pos_std
        self.noise_yaw_std = noise_yaw_std

    def project_point_to_cam(
        self,
        p_world: np.ndarray,
        cam_xpos: np.ndarray,
        cam_xmat: np.ndarray,
    ) -> Tuple[Optional[Tuple[float, float]], np.ndarray, bool]:
        """Project a 3D world point into camera frame and 2D pixel coordinates."""
        p_rel = p_world - cam_xpos
        p_cam = cam_xmat.T @ p_rel
        x_c, y_c, z_c = p_cam[0], p_cam[1], -p_cam[2]

        if z_c <= 0.01:
            return None, p_cam, False

        u = self.cx + (x_c / z_c) * self.f
        v = self.cy - (y_c / z_c) * self.f
        in_fov = (0 <= u < self.w) and (0 <= v < self.h)
        return (u, v), np.array([x_c, y_c, z_c]), in_fov

    def estimate_from_sim(
        self,
        model,
        data,
        cam_name: str = "head_camera",
        mouth_site_name: str = "mouth_tip",
        cyl_body_name: str = "cylinder",
        add_noise: bool = False,
    ) -> PoseEstimate:
        """Extract barbell pose from simulation geometry with optional sensor noise."""
        import mujoco

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        cyl_bid = model.body(cyl_body_name).id
        mouth_sid = model.site(mouth_site_name).id

        cam_xpos = data.cam_xpos[cam_id]
        cam_xmat = data.cam_xmat[cam_id].reshape(3, 3)
        mouth_xpos = data.site_xpos[mouth_sid]
        mouth_xmat = data.site_xmat[mouth_sid].reshape(3, 3)

        cyl_xpos = data.xpos[cyl_bid].copy()
        cyl_xmat = data.xmat[cyl_bid].reshape(3, 3)

        p_center_w = cyl_xpos
        p_left_w = cyl_xpos + cyl_xmat @ np.array([0, 0, 0.084])
        p_right_w = cyl_xpos + cyl_xmat @ np.array([0, 0, -0.084])

        uv_c, pc_c, vis_c = self.project_point_to_cam(p_center_w, cam_xpos, cam_xmat)
        uv_l, pc_l, vis_l = self.project_point_to_cam(p_left_w, cam_xpos, cam_xmat)
        uv_r, pc_r, vis_r = self.project_point_to_cam(p_right_w, cam_xpos, cam_xmat)

        # Transform to mouth cavity coordinate frame
        # In mouth site frame: X is forward, Y is left, Z is up
        p_rel_mouth_world = cyl_xpos - mouth_xpos
        p_mouth = mouth_xmat.T @ p_rel_mouth_world

        # Yaw orientation relative to mouth forward axis
        shaft_dir_world = cyl_xmat[:, 2] # shaft axis
        shaft_dir_mouth = mouth_xmat.T @ shaft_dir_world
        # Barbell shaft has 180-degree cylindrical symmetry, wrap to [-pi/2, pi/2]
        yaw_rel = math.atan2(shaft_dir_mouth[0], shaft_dir_mouth[1])
        while yaw_rel > math.pi / 2.0:
            yaw_rel -= math.pi
        while yaw_rel < -math.pi / 2.0:
            yaw_rel += math.pi

        if add_noise:
            pos_noise = np.random.normal(0, self.noise_pos_std, size=3)
            pc_c += pos_noise
            p_mouth += pos_noise
            yaw_rel += float(np.random.normal(0, self.noise_yaw_std))

        distance = float(np.linalg.norm(cyl_xpos - cam_xpos))

        return PoseEstimate(
            pos_cam=pc_c,
            pos_mouth=p_mouth,
            yaw_rel=yaw_rel,
            pixel_center=uv_c if uv_c is not None else (self.cx, self.cy),
            pixel_left=uv_l if uv_l is not None else (self.cx - 50, self.cy),
            pixel_right=uv_r if uv_r is not None else (self.cx + 50, self.cy),
            distance=distance,
            is_visible=vis_c or (vis_l and vis_r),
            status="TRACKING",
        )

    def compute_nudge_command(
        self,
        estimate: PoseEstimate,
        target_pregrasp_dx: float = 0.025,
        target_under_dz: float = -0.008,
    ) -> dict:
        """Compute recommended visual nudge adjustments for base and head/neck."""
        dx, dy, dz = estimate.pos_mouth[0], estimate.pos_mouth[1], estimate.pos_mouth[2]
        dpsi = estimate.yaw_rel

        # Tolerance thresholds
        align_y = abs(dy) < 0.003        # 3 mm lateral tolerance
        align_yaw = abs(dpsi) < 0.035    # ~2 degrees yaw tolerance
        align_x = abs(dx - target_pregrasp_dx) < 0.008 # within 8 mm of scoop trigger
        align_z = abs(dz - target_under_dz) < 0.005    # under-scoop height ready

        is_aligned = align_y and align_yaw and align_x and align_z

        # Closed-loop nudge gains
        # Base strafe and yaw rate for locomotion
        cmd_vy = float(np.clip(-2.5 * dy, -0.15, 0.15))
        cmd_wz = float(np.clip(-1.8 * dpsi, -0.35, 0.35))

        # Head joint offsets for fine millimeter servoing
        head_yaw_offset = float(np.clip(1.2 * math.atan2(dy, max(dx, 0.05)), -0.25, 0.25))
        head_roll_offset = float(np.clip(-0.5 * dpsi, -0.15, 0.15))

        return {
            "cmd_vy": cmd_vy,
            "cmd_wz": cmd_wz,
            "head_yaw_offset": head_yaw_offset,
            "head_roll_offset": head_roll_offset,
            "error_dx_mm": dx * 1000.0,
            "error_dy_mm": dy * 1000.0,
            "error_dz_mm": dz * 1000.0,
            "error_yaw_deg": math.degrees(dpsi),
            "is_aligned": is_aligned,
            "recommended_beak_open": 1.85, # 48 deg wide open for scoop
        }

    def draw_hud(
        self,
        rgb_image: np.ndarray,
        estimate: PoseEstimate,
        nudge: Optional[dict] = None,
        custom_status: Optional[str] = None,
    ) -> np.ndarray:
        """Render real-time HUD with tracking reticle and millimeter error telemetry."""
        img = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(img)

        status = custom_status or estimate.status
        uc, vc = int(estimate.pixel_center[0]), int(estimate.pixel_center[1])
        ul, vl = int(estimate.pixel_left[0]), int(estimate.pixel_left[1])
        ur, vr = int(estimate.pixel_right[0]), int(estimate.pixel_right[1])

        # Color coding based on status
        color_map = {
            "SEARCHING": (160, 190, 220), # Soft Blue-Gray
            "TRACKING": (0, 220, 255),    # Cyan
            "NUDGING": (255, 200, 0),     # Amber
            "SCOOPING": (255, 120, 0),    # Orange
            "CLAMPED": (0, 255, 120),     # Bright Green
            "LIFTED": (0, 255, 0),        # Solid Green
        }
        reticle_color = color_map.get(status, (0, 255, 255))

        # Draw barbell shaft axis line
        if estimate.is_visible:
            draw.line([(ul, vl), (ur, vr)], fill=reticle_color, width=3)
            # Left plate circle
            draw.ellipse([ul - 12, vl - 12, ul + 12, vl + 12], outline=reticle_color, width=2)
            # Right plate circle
            draw.ellipse([ur - 12, vr - 12, ur + 12, vr + 12], outline=reticle_color, width=2)
            # Center target crosshairs
            draw.line([(uc - 20, vc), (uc + 20, vc)], fill=reticle_color, width=2)
            draw.line([(uc, vc - 20), (uc, vc + 20)], fill=reticle_color, width=2)
            draw.rectangle([uc - 40, vc - 18, uc + 40, vc + 18], outline=reticle_color, width=1)

        # Draw Mouth Center Pre-Grasp Reference Crosshair (Image Center)
        cx, cy = int(self.cx), int(self.cy)
        draw.line([(cx - 15, cy), (cx + 15, cy)], fill=(120, 120, 120), width=1)
        draw.line([(cx, cy - 15), (cx, cy + 15)], fill=(120, 120, 120), width=1)

        # Telemetry Banner at bottom
        banner_h = 75
        draw.rectangle([0, self.h - banner_h, self.w, self.h], fill=(15, 20, 25))

        # Status badge
        draw.rectangle([10, self.h - banner_h + 10, 140, self.h - 10], fill=(30, 40, 50), outline=reticle_color, width=2)
        draw.text((25, self.h - banner_h + 25), status, fill=reticle_color)

        # Metrics text
        dx_mm = estimate.pos_mouth[0] * 1000.0
        dy_mm = estimate.pos_mouth[1] * 1000.0
        dz_mm = estimate.pos_mouth[2] * 1000.0
        yaw_deg = math.degrees(estimate.yaw_rel)
        dist_mm = estimate.distance * 1000.0

        t1 = f"DIST: {dist_mm:5.1f} mm | DX: {dx_mm:+5.1f} mm"
        t2 = f"LAT (DY): {dy_mm:+5.1f} mm | VERT (DZ): {dz_mm:+5.1f} mm | YAW: {yaw_deg:+4.1f} deg"
        draw.text((155, self.h - banner_h + 15), t1, fill=(220, 220, 220))
        draw.text((155, self.h - banner_h + 40), t2, fill=(220, 220, 220))

        if nudge is not None and nudge.get("is_aligned"):
            draw.text((490, self.h - banner_h + 25), "[ALIGNED]", fill=(0, 255, 120))

        return np.array(img)
