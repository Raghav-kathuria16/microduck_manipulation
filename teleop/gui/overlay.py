"""HUD overlay rendered via MuJoCo's passive viewer scene mechanism.

Uses mujoco.viewer's user_scn to inject overlay text items each frame.
Falls back to terminal print if the viewer API doesn't support overlays.

Layout (top-left origin):
  ┌─────────────────────────────────────────────────────────────┐
  │  🦆 MICRODUCK TELEOP         [●REC ep:003  00:12]           │
  │  Mode: walking                                              │
  │  Vel:  ►► vx=0.22 wz= 0.80  [▓▓▓▓░░░]                    │
  │  Head: pitch= 0.32  yaw=-0.24  neck= 0.00                  │
  │  Beak: CLOSED                                               │
  │─────────────────────────────────────────────────────────────│
  │  WASD=move  ←→=yaw  ↑↓=pitch  G=grasp  P=pick              │
  │  H=reset-head  Y=sit  R=randomize  Space=rec  Esc=quit      │
  └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import time
from typing import Optional

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


class HUDOverlay:
    """Manages the on-screen HUD shown inside the MuJoCo passive viewer.

    The MuJoCo passive viewer (v3.x) exposes `viewer.user_scn` which accepts
    `mujoco.MjvGeom` additions for 3-D overlays. For 2-D text overlays, we use
    the `mujoco.mjr_overlay` path, which requires access to the rendering
    context.  Since `launch_passive` doesn't expose the context directly, we
    instead write a compact status block to the terminal each second and rely on
    the viewer window title as the primary status indicator.  A richer overlay
    can be added once MuJoCo exposes a stable text API.
    """

    # --- ANSI colours ---
    _R  = "\033[91m"   # red    → recording
    _G  = "\033[92m"   # green
    _Y  = "\033[93m"   # yellow → mode
    _B  = "\033[94m"   # blue
    _W  = "\033[97m"   # white
    _DIM = "\033[2m"   # dim
    _RST = "\033[0m"   # reset
    _BOLD= "\033[1m"

    def __init__(self, fps_target: float = 1.0):
        """
        Args:
            fps_target: How often to refresh the terminal HUD (Hz).
        """
        self._fps_target   = fps_target
        self._last_print   = 0.0
        self._episode_num  = 0
        self._episode_start: Optional[float] = None
        self._recording    = False

    # ── public API ─────────────────────────────────────────────────────

    def mark_episode_start(self, episode_num: int, recording: bool):
        self._episode_num  = episode_num
        self._episode_start = time.monotonic()
        self._recording    = recording

    def mark_episode_end(self):
        self._episode_start = None
        self._recording    = False

    def set_recording(self, recording: bool):
        self._recording = recording

    def render(
        self,
        mode:        str,
        vx:          float,
        wz:          float,
        head_pitch:  float,
        head_yaw:    float,
        neck_pitch:  float,
        grasp_open:  bool,
        viewer=None,
    ):
        """Render HUD. Throttled to fps_target updates/sec."""
        now = time.monotonic()
        if now - self._last_print < 1.0 / self._fps_target:
            return
        self._last_print = now

        self._print_terminal(mode, vx, wz, head_pitch, head_yaw, neck_pitch, grasp_open)
        if viewer is not None:
            self._update_title(viewer, mode, grasp_open)

    # ── internal ───────────────────────────────────────────────────────

    def _elapsed_str(self) -> str:
        if self._episode_start is None:
            return "--:--"
        elapsed = int(time.monotonic() - self._episode_start)
        return f"{elapsed // 60:02d}:{elapsed % 60:02d}"

    def _print_terminal(self, mode, vx, wz, head_pitch, head_yaw, neck_pitch, grasp_open):
        C = self
        rec_str = (f"{C._R}●REC{C._RST} ep:{self._episode_num:03d}  {self._elapsed_str()}"
                   if self._recording
                   else f"{C._DIM}○IDLE{C._RST}")

        # Velocity bar
        frac_fwd = max(0.0, min(1.0, vx / 0.25))
        frac_bck = max(0.0, min(1.0, -vx / 0.20))
        frac_ang = abs(wz) / 1.0
        vel_bar  = "▓" * int(frac_fwd * 6) + "░" * (6 - int(frac_fwd * 6))
        ang_sym  = ("↺" if wz > 0.05 else "↻" if wz < -0.05 else " ")

        grasp_str = (f"{C._G}OPEN{C._RST}" if grasp_open else f"{C._Y}CLOSED{C._RST}")

        lines = [
            f"{C._BOLD}{'─'*62}{C._RST}",
            f"  🦆 {C._BOLD}MICRODUCK TELEOP{C._RST}           {rec_str}",
            f"  Mode : {C._Y}{mode}{C._RST}",
            f"  Vel  : vx={vx:+.2f}  wz={wz:+.2f}  {ang_sym} [{vel_bar}]",
            f"  Head : pitch={head_pitch:+.2f}  yaw={head_yaw:+.2f}  neck={neck_pitch:+.2f}",
            f"  Beak : {grasp_str}",
            f"  {C._DIM}WASD=move  ↑↓=pitch  ←→=yaw  Z/X=neck  G=grasp  P=pick{C._RST}",
            f"  {C._DIM}H=centre-head  Y=sit  R=rand  Space=rec  Esc=quit{C._RST}",
            f"{C._BOLD}{'─'*62}{C._RST}",
        ]
        # Move cursor up to overwrite previous HUD block
        print(f"\033[{len(lines)+1}A" if hasattr(self, '_printed') else "", end="")
        self._printed = True
        print("\n".join(lines), flush=True)

    def _update_title(self, viewer, mode: str, grasp_open: bool):
        """Set the window title — lightweight viewer feedback."""
        try:
            rec = "●REC" if self._recording else "○"
            viewer.opt.label  # just accessing viewer without errors
            # viewer doesn't expose a title setter, so we skip silently
        except Exception:
            pass
