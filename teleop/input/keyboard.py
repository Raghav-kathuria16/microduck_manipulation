"""Keyboard input handler for the MuJoCo passive viewer.

MuJoCo's passive viewer delivers key events via a key_callback(keycode: int).
This module translates raw keycodes into symbolic names and routes them to
the locomotion / head-aim / grasp / app-control handlers.

Key routing table:
  W=87/w=119      forward
  S=83/s=115      backward
  A=65/a=97       turn left
  D=68/d=100      turn right
  Up=265          look up       (head pitch +)
  Down=264        look down     (head pitch -)
  Left=263        look left     (head yaw +)
  Right=262       look right    (head yaw -)
  Z=90/z=122      neck pitch +
  X=88/x=120      neck pitch -
  G=71/g=103      toggle grasp
  H=72/h=104      reset head centre
  P=80/p=112      trigger ground pick (beak dip)
  Y=89/y=121      toggle sit/stand
  R=82/r=114      re-randomize scene
  Space=32        episode: start / stop recording / reset
  Esc=256         quit

GLFW key constants: https://www.glfw.org/docs/latest/group__keys.html
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import time

# GLFW keycodes used in MuJoCo's passive viewer
_KEY = {
    # WASD movement
    87: 'w', 119: 'w',
    83: 's', 115: 's',
    65: 'a',  97: 'a',
    68: 'd', 100: 'd',
    # Arrow keys (GLFW: Right=262, Left=263, Down=264, Up=265)
    265: 'up',
    264: 'down',
    263: 'left',
    262: 'right',
    # Head control
    90: 'z', 122: 'z',   # neck nod +
    88: 'x', 120: 'x',   # neck nod -
    # Actions
    71: 'g', 103: 'g',   # toggle grasp
    72: 'h', 104: 'h',   # reset head
    80: 'p', 112: 'p',   # ground pick
    89: 'y', 121: 'y',   # sit toggle
    82: 'r', 114: 'r',   # re-randomize
    340: 'shift', 344: 'shift',  # left/right shift
    # App control
    32: 'space',
    256: 'esc',
}

# Keys that affect locomotion (held = continuous command)
MOVEMENT_KEYS = {'w', 's', 'a', 'd', 'up', 'down', 'left', 'right'}
# Head-aim keys (each press = one step; not held-continuous)
HEAD_KEYS = {'up', 'down', 'left', 'right', 'z', 'x'}


@dataclass
class KeyboardState:
    """Snapshot of held key state for the locomotion controller."""
    held: set[str] = field(default_factory=set)
    shift: bool    = False
    _last_seen: dict[str, float] = field(default_factory=dict)

    def press(self, sym: str):
        self.held.add(sym)
        self._last_seen[sym] = time.time()
        if sym == 'shift':
            self.shift = True

    def release(self, sym: str):
        self.held.discard(sym)
        if sym == 'shift':
            self.shift = False

    def tick_decay(self, timeout: float = 0.25) -> bool:
        """Removes keys that haven't been refreshed recently (handles missing key-release events)."""
        now = time.time()
        # Only decay movement keys so that toggles (like G, Y, etc.) don't get cleared if they were supposed to be one-shots, 
        # but actually, one-shots are handled on press. For movement, we need them cleared.
        expired = [sym for sym in self.held if sym in MOVEMENT_KEYS and (now - self._last_seen.get(sym, 0)) > timeout]
        for sym in expired:
            self.release(sym)
        return len(expired) > 0

    @property
    def movement_held(self) -> set[str]:
        return self.held & MOVEMENT_KEYS


class KeyboardRouter:
    """Routes MuJoCo viewer keycodes to the correct handlers.

    Usage:
        router = KeyboardRouter(loco, head, grasp)
        router.set_app_callbacks(randomize=..., reset=..., record_toggle=..., quit=...)

        # then pass to viewer:
        with mujoco.viewer.launch_passive(model, data,
                                          key_callback=router.on_key) as viewer:
            ...
    """

    def __init__(
        self,
        locomotion_ctrl,
        head_ctrl,
        grasp_ctrl,
        policy_runner,
    ):
        self._loco   = locomotion_ctrl
        self._head   = head_ctrl
        self._grasp  = grasp_ctrl
        self._runner = policy_runner
        self._state  = KeyboardState()

        # App-level callbacks (set by app.py)
        self._cb_randomize:     Optional[Callable] = None
        self._cb_reset:         Optional[Callable] = None
        self._cb_record_toggle: Optional[Callable] = None
        self._cb_quit:          Optional[Callable] = None

    def set_app_callbacks(
        self,
        randomize: Optional[Callable]     = None,
        reset: Optional[Callable]         = None,
        record_toggle: Optional[Callable] = None,
        quit: Optional[Callable]          = None,
    ):
        self._cb_randomize     = randomize
        self._cb_reset         = reset
        self._cb_record_toggle = record_toggle
        self._cb_quit          = quit

    @property
    def state(self) -> KeyboardState:
        return self._state

    def on_key(self, keycode: int):
        """MuJoCo key_callback — called on every key press."""
        sym = _KEY.get(keycode)
        if sym is None:
            return

        # Update held-key state
        self._state.press(sym)

        # Route
        if sym in ('w', 's', 'a', 'd'):
            # Locomotion keys: state updated; locomotion.tick() reads them each frame
            self._loco.update_from_keys(self._state.movement_held, self._state.shift)

        elif sym == 'up':
            self._head.apply_key('up')

        elif sym == 'down':
            self._head.apply_key('down')

        elif sym == 'left':
            self._head.apply_key('left')

        elif sym == 'right':
            self._head.apply_key('right')

        elif sym == 'z':
            self._head.apply_key('z')

        elif sym == 'x':
            self._head.apply_key('x')

        elif sym == 'g':
            self._grasp.toggle()
            self._runner.toggle_beak()

        elif sym == 'h':
            self._head.reset()
            self._runner.set_head_offset(0, 0, 0, 0)

        elif sym == 'p':
            self._runner.trigger_ground_pick()

        elif sym == 'y':
            self._runner.toggle_sit()

        elif sym == 'r':
            if self._cb_randomize:
                self._cb_randomize()

        elif sym == 'space':
            if self._cb_record_toggle:
                self._cb_record_toggle()

        elif sym == 'esc':
            if self._cb_quit:
                self._cb_quit()

    def on_key_release(self, keycode: int):
        """Call on key-up events to clear held state."""
        sym = _KEY.get(keycode)
        if sym:
            self._state.release(sym)
            self._loco.update_from_keys(self._state.movement_held, self._state.shift)

    def tick(self, dt: float):
        """Per-frame update: advance locomotion ramp and push to runner."""
        # Decay keys that haven't been pressed recently (simulates key release)
        if self._state.tick_decay(timeout=0.3):
            self._loco.update_from_keys(self._state.movement_held, self._state.shift)

        self._loco.tick(dt)
        vx, vy, wz = self._loco.command
        self._runner.set_velocity_command(vx, vy, wz)
        self._runner.set_head_offset(
            self._head.neck_pitch,
            self._head.head_pitch,
            self._head.head_yaw,
            self._head.head_roll,
        )
