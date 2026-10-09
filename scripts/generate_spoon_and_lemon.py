#!/usr/bin/env python3
"""Procedural Generator for Authentic Tableware Spoon and Real-World Lemon Models.

Delegates to the verified seamless CAD generator in generate_realistic_spoon_lemon.py
to maintain a single source of truth for the 125mm tableware spoon and 22g lemon.
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_MODELS_DIR = os.path.join(_MANIP_DIR, "models")
_ASSETS_DIR = os.path.join(_MODELS_DIR, "assets")
_DEMO_ASSETS = os.path.join(_MANIP_DIR, "scenes/weld_cylinder_demo/assets")

sys.path.insert(0, _SCRIPT_DIR)
from generate_realistic_spoon_lemon import generate_seamless_spoon_stl

def main():
    stl_p1 = os.path.join(_ASSETS_DIR, "spoon.stl")
    stl_p2 = os.path.join(_DEMO_ASSETS, "spoon.stl")
    generate_seamless_spoon_stl(stl_p1)
    generate_seamless_spoon_stl(stl_p2)
    print("Verification complete. All spoon assets generated consistently.")

if __name__ == "__main__":
    main()
