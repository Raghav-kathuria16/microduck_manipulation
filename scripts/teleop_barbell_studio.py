#!/usr/bin/env python3
"""Interactive Teleoperation, Alignment & Demonstration Recording Studio.

Precision manual control for MicroDuck:
  1. Locomotion & Alignment: W/A/S/D/Q/E (Walk, Turn, Strafe) - walks up and aligns to floor barbell.
  2. Manual Crouch / Descent Control: J/Down (Lower Head), K/Up (Raise Head) - finely adjust height.
  3. Edge Grasp & Clamping: 'C' to clamp at mouth edge (1.550 rad), 'O' to open (1.950 rad), 'Space' to toggle.
  4. Throat Cradle: 'T' to immediately tilt neck UPWARD (+25° to +35°), rolling the barbell deep into the throat.
  5. Stand Upright Carry: 'U' to return to upright standing with payload cradled aloft.
  6. Observation Recording: 'R' to start/stop recording states, actions, offsets & contacts to .npz.
  7. Reset: 'Backspace' sets barbell resting flat on the ground 20cm ahead (detached from beak).

Keyboard hotkeys work in BOTH the terminal window AND the 3D MuJoCo viewer window,
as well as the live Web Dashboard at http://localhost:8080.

Can be run via:
  microduck_rl/.venv/bin/mjpython microduck_manipulation/scripts/teleop_barbell_studio.py
"""

from __future__ import annotations
import argparse
import atexit
from datetime import datetime
import json
import math
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import mujoco
import mujoco.viewer
import numpy as np

# Workspace paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(_MANIP_DIR)
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))
_RL_DIR = os.path.join(_ROOT, "microduck_rl")
sys.path.insert(0, os.path.join(_RL_DIR, "scripts"))
sys.path.insert(0, _MANIP_DIR)

from infer_policy import (
    load_bam_model,
    load_mujoco_with_bam,
    PolicyInference,
    BAM_KP_FW,
    BAM_VIN_MIN,
    DEFAULT_POSE,
)
from vision.barbell_pose_estimator import BarbellPoseEstimator

PHYSICAL_CLAMP_XML = os.path.join(
    _EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_loadcarry_physical_clamp.xml"
)
DEFAULT_WALKING_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/BEST_alpha_walking.onnx")
DEFAULT_STANDING_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/BEST_alpha_stand.onnx")
DEFAULT_GROUND_PICK_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_ground_pick_headup.onnx")
DEMO_SAVE_DIR = os.path.join(_ROOT, "data/teleop_demos")

# Shared state between Simulation Thread and Web Controller Thread
teleop_lock = threading.Lock()
teleop_state = {
    # Commanded locomotion
    "cmd_vx": 0.0,
    "cmd_vy": 0.0,
    "cmd_wz": 0.0,
    "loco_decay_time": 0.0,

    # Controllable descent depth (0.0 = standing high 210mm, 1.0 = floor edge level ~34mm)
    "descent_depth": 0.0,

    # Head & Neck pitch bias
    "neck_pitch_offset": 0.0,
    "head_pitch_offset": 0.0,
    "throat_cradle": False,      # When True, tilts neck UP (+30°) to slide barbell into throat

    # Commanded beak clamping
    "beak_target": 1.950,        # 1.950 = open, 1.550 = clamp
    "beak_clamped": False,

    # Recording state
    "is_recording": False,
    "record_step_count": 0,
    "last_saved_file": "",

    # Telemetry feedback to UI
    "mode": "STANDING",
    "trunk_z_mm": 115.0,
    "mouth_z_mm": 207.0,
    "barbell_z_mm": 28.2,
    "barbell_dist_mm": 230.0,
    "dx_mm": 0.0,
    "dy_mm": 0.0,
    "dz_mm": 0.0,
    "yaw_deg": 0.0,
    "top_contacts": 0,
    "bot_contacts": 0,
    "beak_angle": 1.95,
    "beak_torque_nm": 0.0,
    "reset_requested": False,
}


def handle_command_key(k: str | int):
    """Unified handler for hotkeys from Terminal, GLFW MuJoCo Viewer, or Web UI."""
    if isinstance(k, int):
        # GLFW key codes
        if k in (ord('w'), ord('W')): k = 'w'
        elif k in (ord('s'), ord('S')): k = 's'
        elif k in (ord('a'), ord('A')): k = 'a'
        elif k in (ord('d'), ord('D')): k = 'd'
        elif k in (ord('q'), ord('Q')): k = 'q'
        elif k in (ord('e'), ord('E')): k = 'e'
        elif k in (ord('j'), ord('J'), 264): k = 'j'
        elif k in (ord('k'), ord('K'), 265): k = 'k'
        elif k in (ord('c'), ord('C')): k = 'c'
        elif k in (ord('o'), ord('O')): k = 'o'
        elif k in (ord('t'), ord('T')): k = 't'
        elif k in (ord('u'), ord('U')): k = 'u'
        elif k in (ord('r'), ord('R')): k = 'r'
        elif k in (ord('x'), ord('X')): k = 'x'
        elif k == 32: k = 'space'
        elif k in (259, ord('\b'), 127): k = 'backspace'
        else: return
    else:
        k = k.lower()
        if k in ('\r', '\n'): return
        if k in ('\x7f', '\x08'): k = 'backspace'
        if k == ' ': k = 'space'

    with teleop_lock:
        now = time.time()
        if k == 'w':
            teleop_state["cmd_vx"] = 0.28
            teleop_state["cmd_wz"] = 0.0
            teleop_state["descent_depth"] = 0.0
            teleop_state["loco_decay_time"] = now + 0.8
            print("[TELEOP] 🚶 WALKING FORWARD (vx=+0.28 m/s)")
        elif k == 's':
            teleop_state["cmd_vx"] = -0.15
            teleop_state["cmd_wz"] = 0.0
            teleop_state["descent_depth"] = 0.0
            teleop_state["loco_decay_time"] = now + 0.8
            print("[TELEOP] 🚶 WALKING BACKWARD (vx=-0.15 m/s)")
        elif k == 'a':
            teleop_state["cmd_wz"] = 0.60
            teleop_state["descent_depth"] = 0.0
            teleop_state["loco_decay_time"] = now + 0.8
            print("[TELEOP] ↩️ TURNING LEFT (wz=+0.60 rad/s)")
        elif k == 'd':
            teleop_state["cmd_wz"] = -0.60
            teleop_state["descent_depth"] = 0.0
            teleop_state["loco_decay_time"] = now + 0.8
            print("[TELEOP] ↪️ TURNING RIGHT (wz=-0.60 rad/s)")
        elif k == 'q':
            teleop_state["cmd_vy"] = 0.10
            teleop_state["loco_decay_time"] = now + 0.8
            print("[TELEOP] ⬅️ STRAFING LEFT (vy=+0.10 m/s)")
        elif k == 'e':
            teleop_state["cmd_vy"] = -0.10
            teleop_state["loco_decay_time"] = now + 0.8
            print("[TELEOP] ➡️ STRAFING RIGHT (vy=-0.10 m/s)")
        elif k == 'x':
            teleop_state["cmd_vx"] = 0.0
            teleop_state["cmd_vy"] = 0.0
            teleop_state["cmd_wz"] = 0.0
            teleop_state["loco_decay_time"] = 0.0
            print("[TELEOP] 🛑 STOPPED -> STANDING STANCE")
        elif k == 'j':
            teleop_state["cmd_vx"] = 0.0
            teleop_state["cmd_vy"] = 0.0
            teleop_state["cmd_wz"] = 0.0
            teleop_state["descent_depth"] = min(1.0, teleop_state["descent_depth"] + 0.08)
            print(f"[TELEOP] ⬇️ LOWERING HEAD (Depth: {teleop_state['descent_depth']*100:.0f}%)")
        elif k == 'k':
            teleop_state["descent_depth"] = max(0.0, teleop_state["descent_depth"] - 0.08)
            print(f"[TELEOP] ⬆️ RAISING HEAD (Depth: {teleop_state['descent_depth']*100:.0f}%)")
        elif k == 'c':
            teleop_state["beak_clamped"] = True
            teleop_state["beak_target"] = 1.550
            print("[TELEOP] 👄 BEAK CLAMPED (1.550 rad)")
        elif k == 'o':
            teleop_state["beak_clamped"] = False
            teleop_state["beak_target"] = 1.950
            print("[TELEOP] 😮 BEAK OPEN (1.950 rad)")
        elif k == 'space':
            teleop_state["beak_clamped"] = not teleop_state["beak_clamped"]
            teleop_state["beak_target"] = 1.550 if teleop_state["beak_clamped"] else 1.950
            print(f"[TELEOP] 👄 BEAK TOGGLED: {'CLAMPED' if teleop_state['beak_clamped'] else 'OPEN'}")
        elif k == 't':
            teleop_state["throat_cradle"] = not teleop_state["throat_cradle"]
            print(f"[TELEOP] 🦢 THROAT CRADLE (Tilt Neck UP): {'ON' if teleop_state['throat_cradle'] else 'OFF'}")
        elif k == 'u':
            teleop_state["descent_depth"] = 0.0
            teleop_state["throat_cradle"] = False
            print("[TELEOP] 🧍 STAND UPRIGHT CARRY")
        elif k == 'r':
            teleop_state["is_recording"] = not teleop_state["is_recording"]
            print(f"[TELEOP] 📹 RECORDING: {'STARTED 🔴' if teleop_state['is_recording'] else 'STOPPED ⏹️'}")
        elif k == 'backspace':
            teleop_state["reset_requested"] = True
            print("[TELEOP] 🔄 RESET SIMULATION: Barbell placed flat on floor 20cm ahead")


def start_terminal_listener():
    """Starts a non-blocking terminal stdin listener so keystrokes in terminal are captured."""
    if not sys.stdin.isatty():
        return
    import termios
    import tty
    import select

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings))

    def listener():
        try:
            tty.setcbreak(fd)
            while True:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch:
                        handle_command_key(ch)
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    t = threading.Thread(target=listener, daemon=True)
    t.start()


def build_web_controller_html() -> str:
    """Build modern, responsive, high-aesthetic web controller interface."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MicroDuck Precision Teleoperation & Demonstration Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card: rgba(18, 26, 43, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --accent-purple: #8b5cf6;
      --text: #f3f4f6;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body {
      background: var(--bg);
      background-image: radial-gradient(circle at 20% 20%, rgba(6, 182, 212, 0.08), transparent 40%),
                        radial-gradient(circle at 80% 80%, rgba(244, 63, 94, 0.08), transparent 40%);
      color: var(--text);
      min-height: 100vh;
      padding: 24px;
    }
    .header {
      display: flex; justify-content: space-between; align-items: center;
      background: var(--card); border: 1px solid var(--card-border);
      border-radius: 16px; padding: 18px 28px; margin-bottom: 24px;
      backdrop-filter: blur(12px);
    }
    .title h1 { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; }
    .title p { font-size: 0.82rem; color: var(--text-muted); margin-top: 4px; }
    .status-badge {
      display: flex; align-items: center; gap: 8px; font-weight: 700;
      font-size: 0.85rem; padding: 8px 16px; border-radius: 999px;
      background: rgba(16, 185, 129, 0.12); color: var(--accent-emerald);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-badge.rec {
      background: rgba(244, 63, 94, 0.2); color: var(--accent-rose);
      border-color: var(--accent-rose); animation: pulse 1.2s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }

    .grid { display: grid; grid-template-columns: 1.1fr 1.2fr 1fr; gap: 20px; }
    .card {
      background: var(--card); border: 1px solid var(--card-border);
      border-radius: 16px; padding: 22px; backdrop-filter: blur(12px);
      display: flex; flex-direction: column; gap: 16px;
    }
    .card-title {
      font-size: 1.0rem; font-weight: 700; display: flex; align-items: center; gap: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06); padding-bottom: 12px;
    }

    /* Locomotion D-Pad */
    .dpad-container { display: flex; flex-direction: column; align-items: center; gap: 8px; margin: 8px 0; }
    .dpad-row { display: flex; gap: 8px; }
    .btn-dpad {
      width: 68px; height: 52px; border-radius: 10px; border: 1px solid var(--card-border);
      background: rgba(255, 255, 255, 0.05); color: var(--text); font-weight: 700;
      font-size: 0.90rem; cursor: pointer; transition: all 0.15s;
    }
    .btn-dpad:hover { background: rgba(6, 182, 212, 0.25); border-color: var(--accent-cyan); transform: translateY(-2px); }
    .btn-dpad:active { transform: scale(0.95); background: var(--accent-cyan); color: #000; }

    /* Action Buttons */
    .btn-action {
      padding: 14px; border-radius: 12px; border: none; font-weight: 700; font-size: 0.90rem;
      cursor: pointer; transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 8px;
    }
    .btn-crouch { background: linear-gradient(135deg, #0284c7, #0369a1); color: white; }
    .btn-stand { background: rgba(255, 255, 255, 0.08); border: 1px solid var(--card-border); color: white; }
    .btn-throat { background: linear-gradient(135deg, #7c3aed, #8b5cf6); color: white; }
    .btn-throat:hover { box-shadow: 0 0 20px rgba(139, 92, 246, 0.4); }
    .btn-clamp { background: linear-gradient(135deg, #d97706, #f59e0b); color: #000; }
    .btn-open { background: rgba(255, 255, 255, 0.08); border: 1px solid var(--card-border); color: white; }
    .btn-rec { background: linear-gradient(135deg, #e11d48, #f43f5e); color: white; }
    .btn-rec:hover { box-shadow: 0 0 24px rgba(244, 63, 94, 0.5); }
    .btn-stop { background: #334155; color: white; border: 1px solid #475569; }

    /* Telemetry metrics */
    .metric-row { display: flex; justify-content: space-between; align-items: center; }
    .metric-label { font-size: 0.85rem; color: var(--text-muted); }
    .metric-val { font-family: 'JetBrains Mono', monospace; font-size: 0.92rem; font-weight: 700; color: var(--accent-cyan); }

    .key-hint {
      background: rgba(255, 255, 255, 0.06); border-radius: 6px; padding: 2px 6px;
      font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .slider-box {
      background: rgba(0, 0, 0, 0.2); border: 1px solid var(--card-border);
      border-radius: 12px; padding: 14px; display: flex; flex-direction: column; gap: 8px;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="title">
      <h1>🦆 MicroDuck Precision Teleoperation & Demonstration Studio</h1>
      <p>Manual Descent Control, Edge-of-Mouth Grasping, Throat Upward Cradle & Telemetry Recording</p>
    </div>
    <div id="statusBadge" class="status-badge">
      <span id="statusDot">●</span> <span id="statusText">STANDBY</span>
    </div>
  </div>

  <div class="grid">
    <!-- COL 1: Locomotion & Alignment -->
    <div class="card">
      <div class="card-title">
        <span>🎮 1. Locomotion & Approach</span>
      </div>
      <p style="font-size: 0.80rem; color: var(--text-muted);">
        Walk MicroDuck forward to approach the floor barbell (resting 20cm ahead). Turn/strafe to align center.
      </p>

      <div class="dpad-container">
        <button class="btn-dpad" onmousedown="sendCmd({key: 'w'})">W<br><span class="key-hint">FWD</span></button>
        <div class="dpad-row">
          <button class="btn-dpad" onmousedown="sendCmd({key: 'a'})">A<br><span class="key-hint">LEFT</span></button>
          <button class="btn-dpad" onmousedown="sendCmd({key: 's'})">S<br><span class="key-hint">BWD</span></button>
          <button class="btn-dpad" onmousedown="sendCmd({key: 'd'})">D<br><span class="key-hint">RIGHT</span></button>
        </div>
        <div class="dpad-row">
          <button class="btn-dpad" style="width: 105px;" onmousedown="sendCmd({key: 'q'})">Q <span class="key-hint">STRAFE L</span></button>
          <button class="btn-dpad" style="width: 105px;" onmousedown="sendCmd({key: 'e'})">E <span class="key-hint">STRAFE R</span></button>
        </div>
      </div>

      <div class="slider-box">
        <div class="metric-row">
          <span class="metric-label">Head/Neck Pitch Fine Aim:</span>
          <span id="neckVal" class="metric-val">0.0°</span>
        </div>
        <input type="range" id="neckSlider" min="-0.6" max="0.6" step="0.05" value="0.0" style="width: 100%;" oninput="sendHeadPitch(this.value)">
      </div>
    </div>

    <!-- COL 2: Descent Height & Edge Grasping -->
    <div class="card">
      <div class="card-title">
        <span>🦆 2. Height Control & Throat Cradle</span>
      </div>
      <p style="font-size: 0.80rem; color: var(--text-muted);">
        <b>Grasp Strategy:</b> Lower mouth to edge-of-mouth level (~34mm) → Clamp shaft → Tilt neck UP to cradle deep in throat!
      </p>

      <!-- Continuous Crouch Slider -->
      <div class="slider-box">
        <div class="metric-row">
          <span class="metric-label">Descent Depth Control:</span>
          <span id="depthVal" class="metric-val">0% (Standing 207mm)</span>
        </div>
        <input type="range" id="depthSlider" min="0" max="100" step="2" value="0" style="width: 100%;" oninput="sendDescent(this.value / 100.0)">
        <div style="display: flex; gap: 6px; margin-top: 4px;">
          <button class="btn-action btn-stand" style="padding: 6px; font-size: 0.75rem; flex: 1;" onclick="setDescentPreset(0.0)">Stand (207mm)</button>
          <button class="btn-action btn-stand" style="padding: 6px; font-size: 0.75rem; flex: 1;" onclick="setDescentPreset(0.5)">Mid (110mm)</button>
          <button class="btn-action btn-crouch" style="padding: 6px; font-size: 0.75rem; flex: 1;" onclick="setDescentPreset(1.0)">Floor Edge (34mm)</button>
        </div>
      </div>

      <!-- Edge Clamping -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
        <button class="btn-action btn-clamp" onclick="sendCmd({key: 'c'})">
          <span>👄 CLAMP AT EDGE</span> <span class="key-hint">[C]</span>
        </button>
        <button class="btn-action btn-open" onclick="sendCmd({key: 'o'})">
          <span>😮 OPEN MOUTH</span> <span class="key-hint">[O]</span>
        </button>
      </div>

      <!-- Throat Cradle (Upward Tilt) -->
      <button class="btn-action btn-throat" onclick="sendCmd({key: 't'})">
        <span>🦢 THROAT CRADLE (Tilt Neck UP)</span> <span class="key-hint">[T]</span>
      </button>

      <button class="btn-action btn-stand" onclick="sendCmd({key: 'backspace'})" style="margin-top: auto;">
        <span>🔄 RESET BARBELL (Down on Floor 20cm ahead)</span> <span class="key-hint">[Backspace]</span>
      </button>
    </div>

    <!-- COL 3: Telemetry & Dataset Recording -->
    <div class="card">
      <div class="card-title">
        <span>🔴 3. Observation Recorder</span>
      </div>
      <p style="font-size: 0.80rem; color: var(--text-muted);">
        Record complete demonstration trajectories (qpos, qvel, actions, offsets, contacts) to .npz for imitation learning.
      </p>

      <button id="btnRec" class="btn-action btn-rec" onclick="sendCmd({key: 'r'})">
        <span>🔴 START RECORDING OBSERVATIONS</span> <span class="key-hint">[R]</span>
      </button>

      <div class="slider-box">
        <div class="metric-row">
          <span class="metric-label">Recorded Steps</span>
          <span id="recStepsVal" class="metric-val" style="color: var(--accent-rose);">0 steps</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Mouth Tip Height (Z)</span>
          <span id="mouthZVal" class="metric-val">207.0 mm</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Barbell Height (Z)</span>
          <span id="barbellZVal" class="metric-val">28.2 mm</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Distance to Barbell</span>
          <span id="distVal" class="metric-val">230 mm</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Offsets [DX, DY, DZ]</span>
          <span id="offsetsVal" class="metric-val">+210, +0, -178 mm</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Palate Contacts</span>
          <span id="contactsVal" class="metric-val" style="color: var(--accent-emerald);">0 contacts</span>
        </div>
      </div>

      <div style="font-size: 0.74rem; color: var(--text-muted); line-height: 1.45;">
        <span>💡 Keyboard Hotkeys (Terminal, MuJoCo Window, or Browser):</span><br>
        • <b>W/A/S/D</b>: Walk & Turn &nbsp;|&nbsp; <b>Q/E</b>: Strafe<br>
        • <b>J / Down</b>: Lower Head &nbsp;|&nbsp; <b>K / Up</b>: Raise Head<br>
        • <b>C</b>: Clamp Edge &nbsp;|&nbsp; <b>O</b>: Open &nbsp;|&nbsp; <b>Space</b>: Toggle<br>
        • <b>T</b>: Throat Cradle (Neck Up) &nbsp;|&nbsp; <b>U</b>: Upright Stand<br>
        • <b>R</b>: Record Demo &nbsp;|&nbsp; <b>Backspace</b>: Reset on Floor
      </div>
    </div>
  </div>

  <script>
    function sendCmd(data) {
      fetch('/cmd', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      }).catch(console.error);
    }

    function sendHeadPitch(val) {
      document.getElementById('neckVal').innerText = (parseFloat(val) * 57.3).toFixed(1) + '°';
      sendCmd({ head_pitch: parseFloat(val) });
    }

    function sendDescent(depth) {
      document.getElementById('depthVal').innerText = Math.round(depth * 100) + '% (Z ≈ ' + (207 - depth * 173).toFixed(0) + 'mm)';
      sendCmd({ descent: depth });
    }

    function setDescentPreset(depth) {
      document.getElementById('depthSlider').value = depth * 100;
      sendDescent(depth);
    }

    // Keyboard Hotkey Listener in Browser
    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      const k = e.key.toLowerCase();
      if (['w', 's', 'a', 'd', 'q', 'e', 'j', 'k', 'c', 'o', 't', 'u', 'r', 'x'].includes(k)) {
        sendCmd({ key: k });
      } else if (e.key === 'ArrowDown') {
        sendCmd({ key: 'j' });
      } else if (e.key === 'ArrowUp') {
        sendCmd({ key: 'k' });
      } else if (e.key === ' ') {
        sendCmd({ key: 'space' });
      } else if (e.key === 'Backspace') {
        sendCmd({ key: 'backspace' });
      }
    });

    // Telemetry Polling Loop (20 Hz)
    setInterval(() => {
      fetch('/state')
        .then(r => r.json())
        .then(s => {
          const badge = document.getElementById('statusBadge');
          if (s.is_recording) {
            badge.className = 'status-badge rec';
            badge.innerHTML = '● RECORDING (' + s.record_step_count + ' STEPS)';
          } else {
            badge.className = 'status-badge';
            badge.innerHTML = '● ' + s.mode;
          }

          document.getElementById('recStepsVal').innerText = s.record_step_count + ' steps';
          document.getElementById('mouthZVal').innerText = s.mouth_z_mm.toFixed(1) + ' mm';
          document.getElementById('barbellZVal').innerText = s.barbell_z_mm.toFixed(1) + ' mm';
          document.getElementById('distVal').innerText = s.barbell_dist_mm.toFixed(0) + ' mm';
          document.getElementById('offsetsVal').innerText =
            (s.dx_mm > 0 ? '+' : '') + s.dx_mm.toFixed(0) + ', ' +
            (s.dy_mm > 0 ? '+' : '') + s.dy_mm.toFixed(0) + ', ' +
            (s.dz_mm > 0 ? '+' : '') + s.dz_mm.toFixed(0) + ' mm';
          document.getElementById('contactsVal').innerText = (s.top_contacts + s.bot_contacts) + ' contacts (Top: ' + s.top_contacts + ', Bot: ' + s.bot_contacts + ')';

          const btnRec = document.getElementById('btnRec');
          if (s.is_recording) {
            btnRec.className = 'btn-action btn-stop';
            btnRec.innerHTML = '⏹️ STOP RECORDING (' + s.record_step_count + ' steps) <span class="key-hint">[R]</span>';
          } else {
            btnRec.className = 'btn-action btn-rec';
            btnRec.innerHTML = '🔴 START RECORDING OBSERVATIONS <span class="key-hint">[R]</span>';
          }
        })
        .catch(console.error);
    }, 50);
  </script>
</body>
</html>
"""


class WebControllerServer(BaseHTTPRequestHandler):
    """Lightweight HTTP server handling Web Controller UI & REST endpoints."""

    def log_message(self, format, *args):
        return  # Silence access logs

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            content = build_web_controller_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/state":
            with teleop_lock:
                data = json.dumps(teleop_state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/cmd":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            cmd = json.loads(body.decode("utf-8"))

            if "key" in cmd:
                handle_command_key(cmd["key"])
            with teleop_lock:
                if "head_pitch" in cmd:
                    teleop_state["head_pitch_offset"] = float(cmd["head_pitch"])
                if "descent" in cmd:
                    teleop_state["descent_depth"] = float(np.clip(cmd["descent"], 0.0, 1.0))

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()


def start_web_server(port=8080):
    """Starts the web controller HTTP server in a daemon thread."""
    server = HTTPServer(("0.0.0.0", port), WebControllerServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def save_recording(buffer: dict, save_dir=DEMO_SAVE_DIR):
    """Saves recorded episode observations and states to a compressed NPZ file."""
    os.makedirs(save_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = os.path.join(save_dir, f"microduck_teleop_demo_{stamp}.npz")
    json_path = os.path.join(save_dir, f"microduck_teleop_demo_{stamp}.json")

    np.savez_compressed(
        npz_path,
        timesteps=np.array(buffer["timesteps"], dtype=np.float32),
        qpos=np.array(buffer["qpos"], dtype=np.float32),
        qvel=np.array(buffer["qvel"], dtype=np.float32),
        actions=np.array(buffer["actions"], dtype=np.float32),
        beak_target=np.array(buffer["beak_target"], dtype=np.float32),
        beak_q=np.array(buffer["beak_q"], dtype=np.float32),
        beak_torque=np.array(buffer["beak_torque"], dtype=np.float32),
        barbell_pos=np.array(buffer["barbell_pos"], dtype=np.float32),
        barbell_quat=np.array(buffer["barbell_quat"], dtype=np.float32),
        mouth_pos=np.array(buffer["mouth_pos"], dtype=np.float32),
        rel_offsets=np.array(buffer["rel_offsets"], dtype=np.float32),
        top_contacts=np.array(buffer["top_contacts"], dtype=np.int32),
        bot_contacts=np.array(buffer["bot_contacts"], dtype=np.int32),
        locomotion_cmd=np.array(buffer["locomotion_cmd"], dtype=np.float32),
    )

    metadata = {
        "timestamp": stamp,
        "total_steps": len(buffer["timesteps"]),
        "duration_seconds": len(buffer["timesteps"]) * 0.02,
        "initial_barbell_z": float(buffer["barbell_pos"][0][2]),
        "max_barbell_z": float(np.max([p[2] for p in buffer["barbell_pos"]])),
        "max_vertical_lift_mm": float((np.max([p[2] for p in buffer["barbell_pos"]]) - buffer["barbell_pos"][0][2]) * 1000.0),
        "peak_top_contacts": int(np.max(buffer["top_contacts"])),
        "npz_file": npz_path,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*72}")
    print(f" [RECORDING SAVED] {len(buffer['timesteps'])} steps saved successfully!")
    print(f"  NPZ Path:  {npz_path}")
    print(f"  Duration:  {metadata['duration_seconds']:.1f} s")
    print(f"  Max Lift:  {metadata['max_vertical_lift_mm']:.1f} mm")
    print(f"{'='*72}\n")
    return npz_path


def run_studio(port=8080):
    """Main execution loop for interactive teleoperation and demonstration studio."""
    bam_model = load_bam_model(kp_fw=BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, names = load_mujoco_with_bam(PHYSICAL_CLAMP_XML, bam_model, 0.005, 0.0, BAM_VIN_MIN)

    cyl_bid = model.body("cylinder").id
    cyl_jid = model.joint("cylinder_free").id
    beak_jid = model.joint("beak").id
    mouth_sid = model.site("mouth_tip").id
    cyl_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_geom")
    mouth_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "soft_mouth_top_collision")
    jaw_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "jaw_soft_lower_collision")
    jaw_soft_bid = model.body("jaw_soft").id
    beak_lower_bid = model.body("beak_lower").id
    beak_qadr = model.jnt_qposadr[beak_jid]
    beak_dof = model.jnt_dofadr[beak_jid]

    estimator = BarbellPoseEstimator(img_width=640, img_height=480, fovy_deg=45.0)

    policy = PolicyInference(
        model, data,
        walking_onnx_path=DEFAULT_WALKING_POLICY,
        standing_onnx_path=DEFAULT_STANDING_POLICY,
        ground_pick_onnx_path=DEFAULT_GROUND_PICK_POLICY,
        ground_pick_period=4.0,
        use_projected_gravity=True,
        new_cmd_obs=True,
        bam_ctrl=bam_ctrl,
    )

    def reset_sim():
        mujoco.mj_resetData(model, data)
        # Stand duck at origin
        data.qpos[0:3] = [0.0, 0.0, 0.125]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, qpos_idx in enumerate(policy.joint_qpos_indices):
            data.qpos[qpos_idx] = policy.default_pose[i]
        data.qpos[beak_qadr] = 1.950  # Start wide open

        # Reset barbell DOWN FLAT ON THE GROUND 20cm ahead of the duck (completely detached)
        if cyl_jid >= 0:
            c_adr = model.jnt_qposadr[cyl_jid]
            data.qpos[c_adr:c_adr + 3] = [0.200, 0.000, 0.0282]
            data.qpos[c_adr + 3:c_adr + 7] = [0.7071068, 0.7071068, 0.0, 0.0]
            data.qvel[model.jnt_dofadr[cyl_jid]:model.jnt_dofadr[cyl_jid] + 6] = 0.0

        bam_ctrl.reset(data.qpos)
        bam_ctrl.q_target[policy.beak_act_id] = 1.950
        mujoco.mj_forward(model, data)

        policy.current_policy = "standing"
        policy.ort_session = policy.standing_session
        policy.ground_pick_mode = False

        with teleop_lock:
            teleop_state["cmd_vx"] = 0.0
            teleop_state["cmd_vy"] = 0.0
            teleop_state["cmd_wz"] = 0.0
            teleop_state["loco_decay_time"] = 0.0
            teleop_state["descent_depth"] = 0.0
            teleop_state["throat_cradle"] = False
            teleop_state["beak_target"] = 1.950
            teleop_state["beak_clamped"] = False
            teleop_state["head_pitch_offset"] = 0.0
            teleop_state["reset_requested"] = False

    reset_sim()

    # Start Web Controller Server
    start_web_server(port)

    # Start Non-Blocking Terminal Keyboard Listener
    start_terminal_listener()

    # Observation recording buffer
    rec_buffer = {
        "timesteps": [], "qpos": [], "qvel": [], "actions": [],
        "beak_target": [], "beak_q": [], "beak_torque": [],
        "barbell_pos": [], "barbell_quat": [], "mouth_pos": [],
        "rel_offsets": [], "top_contacts": [], "bot_contacts": [],
        "locomotion_cmd": [],
    }

    print("\n" + "=" * 78)
    print(" 🦆 MicroDuck Precision Teleoperation & Demonstration Studio")
    print(f" 🌐 Web Dashboard: http://localhost:{port}")
    print(" 🎮 Direct Keyboard Hotkeys (Active in BOTH Terminal & MuJoCo Viewer):")
    print("    [W / S]       : Walk Forward / Backward (Walks up to floor barbell)")
    print("    [A / D]       : Turn Left / Right")
    print("    [Q / E]       : Strafe Left / Right")
    print("    [X]           : Stop Walking Immediately")
    print("    [J / Down]    : LOWER HEAD (Continuous Descent down to floor)")
    print("    [K / Up]      : RAISE HEAD (Continuous Ascent)")
    print("    [C]           : CLAMP AT EDGE OF MOUTH (1.550 rad)")
    print("    [T]           : THROAT CRADLE (Tilt neck UP to roll barbell deep into palate)")
    print("    [U]           : STAND UPRIGHT CARRY")
    print("    [O]           : OPEN BEAK (1.950 rad)")
    print("    [Space]       : TOGGLE BEAK CLAMP")
    print("    [R]           : TOGGLE OBSERVATION RECORDING")
    print("    [Backspace]   : RESET BARBELL (Flat on floor 20cm ahead)")
    print("=" * 78 + "\n")

    def key_callback(keycode: int):
        handle_command_key(keycode)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat = [0.12, 0.0, 0.09]
        viewer.cam.distance = 0.58
        viewer.cam.elevation = -14.0
        viewer.cam.azimuth = 135.0

        step = 0
        last_vel_cmd = (0.0, 0.0, 0.0)
        was_recording = False
        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()
            now = time.time()

            with teleop_lock:
                if teleop_state["reset_requested"]:
                    reset_sim()
                    last_vel_cmd = (0.0, 0.0, 0.0)

                # Decay walking velocity if key released for > 0.8s
                if teleop_state["loco_decay_time"] > 0 and now > teleop_state["loco_decay_time"]:
                    teleop_state["cmd_vx"] = 0.0
                    teleop_state["cmd_vy"] = 0.0
                    teleop_state["cmd_wz"] = 0.0
                    teleop_state["loco_decay_time"] = 0.0

                vx = teleop_state["cmd_vx"]
                vy = teleop_state["cmd_vy"]
                wz = teleop_state["cmd_wz"]
                depth = teleop_state["descent_depth"]
                throat_cradle = teleop_state["throat_cradle"]
                beak_target = teleop_state["beak_target"]
                head_pitch_off = teleop_state["head_pitch_offset"]
                is_rec = teleop_state["is_recording"]

            # Manage recording transitions
            if is_rec and not was_recording:
                rec_buffer = {k: [] for k in rec_buffer}
                print("\n[RECORDER] 🔴 Recording session started...")
            elif not is_rec and was_recording:
                if len(rec_buffer["timesteps"]) > 0:
                    saved_path = save_recording(rec_buffer)
                    with teleop_lock:
                        teleop_state["last_saved_file"] = saved_path

            was_recording = is_rec

            # 50 Hz Control Loop (every 4 physics steps = 20ms)
            if step % 4 == 0:
                is_moving = (abs(vx) > 0.02 or abs(vy) > 0.02 or abs(wz) > 0.05)

                if depth > 0.02 and not is_moving:
                    # Ground pick crouch descent mode
                    policy.current_policy = "ground_pick"
                    policy.ort_session = policy.ground_pick_session
                    policy.ground_pick_mode = True
                    # Map descent depth (0..1) to phi (0..0.185)
                    phi = float(np.clip(depth * 0.185, 0.0, 0.185))
                    policy.command = np.zeros(13, dtype=np.float32)
                    policy.command[0] = np.cos(2 * np.pi * phi)
                    policy.command[1] = np.sin(2 * np.pi * phi)
                else:
                    # Standing or walking mode
                    policy.ground_pick_mode = False
                    if (vx, vy, wz) != last_vel_cmd:
                        policy.set_vel_cmd(vx, vy, wz)
                        last_vel_cmd = (vx, vy, wz)

                act = policy.infer()

                # Fine head pitch aim
                if head_pitch_off != 0.0:
                    act[6] += head_pitch_off

                # Throat Cradle: Tilt neck upwards (+25° to +35°) to slide barbell deep into palate
                if throat_cradle:
                    act[5] = float(np.deg2rad(30.0) - policy.default_pose[5])
                    act[6] = float(np.deg2rad(-20.0) - policy.default_pose[6])

                policy.apply_action(act)

                # Barbell pose estimation for telemetry
                est = estimator.estimate_from_sim(model, data, add_noise=False)

                # Record observation step
                if is_rec:
                    c_pos = data.xpos[cyl_bid].copy()
                    c_adr = model.jnt_qposadr[cyl_jid]
                    c_quat = data.qpos[c_adr + 3:c_adr + 7].copy()
                    m_pos = data.site_xpos[mouth_sid].copy()

                    top_c = 0
                    bot_c = 0
                    for ci in range(data.ncon):
                        con = data.contact[ci]
                        g1, g2 = con.geom1, con.geom2
                        if cyl_geom_id in (g1, g2):
                            og = g2 if g1 == cyl_geom_id else g1
                            ob = model.geom_bodyid[og]
                            if ob == jaw_soft_bid or og == mouth_geom_id:
                                top_c += 1
                            elif ob == beak_lower_bid or og == jaw_geom_id:
                                bot_c += 1

                    rec_buffer["timesteps"].append(time.time() - start_time)
                    rec_buffer["qpos"].append(data.qpos.copy())
                    rec_buffer["qvel"].append(data.qvel.copy())
                    rec_buffer["actions"].append(act.copy())
                    rec_buffer["beak_target"].append(beak_target)
                    rec_buffer["beak_q"].append(float(data.qpos[beak_qadr]))
                    rec_buffer["beak_torque"].append(float(data.qfrc_actuator[beak_dof]))
                    rec_buffer["barbell_pos"].append(c_pos)
                    rec_buffer["barbell_quat"].append(c_quat)
                    rec_buffer["mouth_pos"].append(m_pos)
                    rec_buffer["rel_offsets"].append([est.pos_mouth[0], est.pos_mouth[1], est.pos_mouth[2], est.yaw_rel])
                    rec_buffer["top_contacts"].append(top_c)
                    rec_buffer["bot_contacts"].append(bot_c)
                    rec_buffer["locomotion_cmd"].append([vx, vy, wz])

            # Apply 15th Actuator Clamping via BAM
            bam_ctrl.q_target[policy.beak_act_id] = beak_target
            bam_ctrl.update()
            mujoco.mj_step(model, data)
            viewer.sync()

            # Update shared telemetry state every 20 steps (40ms)
            if step % 8 == 0:
                c_z = data.xpos[cyl_bid, 2] * 1000.0
                m_z = data.site_xpos[mouth_sid, 2] * 1000.0
                t_z = data.xpos[1, 2] * 1000.0
                q_b = float(data.qpos[beak_qadr])
                tau_b = float(data.qfrc_actuator[beak_dof])

                top_c = 0
                bot_c = 0
                for ci in range(data.ncon):
                    con = data.contact[ci]
                    g1, g2 = con.geom1, con.geom2
                    if cyl_geom_id in (g1, g2):
                        og = g2 if g1 == cyl_geom_id else g1
                        ob = model.geom_bodyid[og]
                        if ob == jaw_soft_bid or og == mouth_geom_id:
                            top_c += 1
                        elif ob == beak_lower_bid or og == jaw_geom_id:
                            bot_c += 1

                with teleop_lock:
                    if throat_cradle:
                        teleop_state["mode"] = "THROAT CRADLE (TILT UP)"
                    elif depth > 0.02:
                        teleop_state["mode"] = f"CROUCH DESCENT ({depth*100:.0f}%)"
                    elif is_moving:
                        teleop_state["mode"] = "WALKING"
                    else:
                        teleop_state["mode"] = "STANDING"

                    teleop_state["trunk_z_mm"] = t_z
                    teleop_state["mouth_z_mm"] = m_z
                    teleop_state["barbell_z_mm"] = c_z
                    teleop_state["barbell_dist_mm"] = est.distance * 1000.0
                    teleop_state["dx_mm"] = est.pos_mouth[0] * 1000.0
                    teleop_state["dy_mm"] = est.pos_mouth[1] * 1000.0
                    teleop_state["dz_mm"] = est.pos_mouth[2] * 1000.0
                    teleop_state["yaw_deg"] = math.degrees(est.yaw_rel)
                    teleop_state["top_contacts"] = top_c
                    teleop_state["bot_contacts"] = bot_c
                    teleop_state["beak_angle"] = q_b
                    teleop_state["beak_torque_nm"] = tau_b
                    teleop_state["record_step_count"] = len(rec_buffer["timesteps"])

            step += 1
            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="MicroDuck Precision Teleoperation & Demonstration Studio")
    parser.add_argument("--port", type=int, default=8080, help="Web Controller HTTP port (default: 8080)")
    args = parser.parse_args()

    run_studio(port=args.port)


if __name__ == "__main__":
    main()
