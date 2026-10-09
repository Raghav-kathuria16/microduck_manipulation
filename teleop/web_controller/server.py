#!/usr/bin/env python3
"""Localhost Web Controller Server for MicroDuck Load-Carrying Teleoperation.

Serves the interactive web UI and provides high-speed REST endpoints:
  - GET  /              -> Serves index.html
  - POST /api/command   -> Receives [vx, vy, wz] from the browser joystick/keys
  - GET  /api/command   -> Polled by MuJoCo simulation loops (view_trained_policy_native.py)
  - POST /api/telemetry -> Receives simulation telemetry [z, pitch, roll, neck]
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from typing import Dict, Any

_DIR = os.path.dirname(os.path.abspath(__file__))

# Shared thread-safe state buffer
_LOCK = threading.Lock()
_COMMAND_STATE: Dict[str, Any] = {
    "vx": 0.25,
    "vy": 0.0,
    "wz": 0.0,
    "target_vx": 0.25,
    "target_wz": 0.0,
    "payload_mass": 300.0,
    "reset_requested": False,
    "timestamp": 0.0,
}
_TELEMETRY_STATE: Dict[str, Any] = {
    "z": 0.125,
    "pitch": -1.9,
    "roll": 0.0,
    "neck": 12.2,
    "step": 0,
    "mass": 300.0,
    "speed": 0.25,
    "status": "Walking Stable (Env 3 v3 PPO)",
}


class TeleopRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=_DIR, **kwargs)

    def do_GET(self):
        try:
            if self.path == "/api/command":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with _LOCK:
                    data = json.dumps(_COMMAND_STATE).encode("utf-8")
                    # Consume one-shot reset request
                    _COMMAND_STATE["reset_requested"] = False
                self.wfile.write(data)
                return

            if self.path == "/api/telemetry":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with _LOCK:
                    data = json.dumps(_TELEMETRY_STATE).encode("utf-8")
                self.wfile.write(data)
                return

            # Fallback to static files
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            if self.path == "/api/command":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                try:
                    payload = json.loads(body.decode("utf-8"))
                    with _LOCK:
                        if "vx" in payload:
                            _COMMAND_STATE["vx"] = float(payload.get("vx", 0.0))
                        if "vy" in payload:
                            _COMMAND_STATE["vy"] = float(payload.get("vy", 0.0))
                        if "wz" in payload:
                            _COMMAND_STATE["wz"] = float(payload.get("wz", 0.0))
                        if "target_vx" in payload:
                            _COMMAND_STATE["target_vx"] = float(payload.get("target_vx", 0.0))
                        if "target_wz" in payload:
                            _COMMAND_STATE["target_wz"] = float(payload.get("target_wz", 0.0))
                        if "payload_mass" in payload:
                            _COMMAND_STATE["payload_mass"] = float(payload.get("payload_mass", 300.0))
                        if "reset_requested" in payload:
                            _COMMAND_STATE["reset_requested"] = bool(payload.get("reset_requested", False))
                        _COMMAND_STATE["timestamp"] = time.time()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                except Exception as e:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(f'{{"error":"{e}"}}'.encode("utf-8"))
                return

            if self.path == "/api/telemetry":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                try:
                    payload = json.loads(body.decode("utf-8"))
                    with _LOCK:
                        _TELEMETRY_STATE.update(payload)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                except Exception as e:
                    self.send_response(400)
                    self.end_headers()
                return

            self.send_response(404)
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        # Silence routine access logs to keep terminal clean
        pass


def run_server(port: int = 8080):
    server = ThreadingHTTPServer(("0.0.0.0", port), TeleopRequestHandler)
    print(f"\n" + "=" * 70)
    print(f"MICRODUCK LOCALHOST CONTROLLER READY")
    print(f"Server running at: http://localhost:{port}")
    print(f"Open in your browser on desktop, tablet, or phone!")
    print(f"=" * 70 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down teleop server.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    args = parser.parse_args()
    run_server(args.port)
