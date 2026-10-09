#!/usr/bin/env python3
"""Benchmark: Avian Pinch-Jerk-Clamp — Perfect Grasp Metric Evaluation.

Evaluates Options 2 and 5 across 5 perturbation trials against the 5 redefined
"Perfect Grasp" metrics:

    M1  Ground Clearance:    Z_mouth ≥ 28.0 mm, 0 head-floor contacts
    M2  Throat Ingestion:    x_rel ≤ −10.0 mm (≥70% enclosed)
    M3  Symmetrical Align:   |Δyaw| ≤ 5.0°, |Δy| ≤ 3.0 mm
    M4  Avian Jerk Exec:     Head upward whip (peak_vel > 0.5) + throat lock
    M5  Coulomb Aloft Carry: Z_barbell ≥ 115 mm, Z_trunk ≥ 115 mm, 0 welds

Perturbation Trials:
    1. Nominal Centered    (X=0.105, Y=+0.0mm,  Yaw=0°)
    2. Lateral Left +10mm  (X=0.105, Y=+10.0mm, Yaw=0°)
    3. Lateral Right −10mm (X=0.105, Y=−10.0mm, Yaw=0°)
    4. Longitudinal Near   (X=0.095, Y=+0.0mm,  Yaw=0°)
    5. Longitudinal Far    (X=0.115, Y=+0.0mm,  Yaw=0°)

Output:
    - Structured JSON report to data/benchmark_avian_perfect_grasp.json
    - Formatted terminal summary table

Usage:
    microduck_rl/.venv/bin/mjpython microduck_manipulation/scripts/benchmark_avian_perfect_grasp.py
"""

from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Workspace Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(_MANIP_DIR)

# Import the Option 2 and Option 5 runner modules
sys.path.insert(0, _SCRIPT_DIR)
from option_2_avian_augmented_policy import (
    setup_simulation as setup_sim_opt2,
    reset_environment as reset_env_opt2,
    run_avian_augmented_policy,
)
from option_5_avian_state_machine import (
    setup_simulation as setup_sim_opt5,
    reset_environment as reset_env_opt5,
    run_avian_state_machine,
)

# ---------------------------------------------------------------------------
# Trial Configurations
# ---------------------------------------------------------------------------
TRIALS = [
    {"name": "Nominal_Centered",       "x": 0.105, "y":  0.000, "yaw_deg": 0.0},
    {"name": "Lateral_Left_+10mm",     "x": 0.105, "y":  0.010, "yaw_deg": 0.0},
    {"name": "Lateral_Right_-10mm",    "x": 0.105, "y": -0.010, "yaw_deg": 0.0},
    {"name": "Longitudinal_Near_-10mm","x": 0.095, "y":  0.000, "yaw_deg": 0.0},
    {"name": "Longitudinal_Far_+10mm", "x": 0.115, "y":  0.000, "yaw_deg": 0.0},
]


def format_pass_fail(passed: bool) -> str:
    """Return a colored pass/fail string for terminal output."""
    return "✅ PASS" if passed else "❌ FAIL"


def print_metric_table(results: list[dict]):
    """Print a formatted comparison table across all trials and options."""
    # Header
    print(f"\n{'='*130}")
    print(f" 🦆 AVIAN PINCH-JERK-CLAMP — PERFECT GRASP BENCHMARK RESULTS")
    print(f"{'='*130}")

    # Column widths
    col_opt  = 50
    col_tri  = 28
    col_m    = 10

    for option_name in ["Option 2 (Augmented Policy – Avian Pinch-Jerk-Clamp)",
                         "Option 5 (Analytical State Machine – Avian Pinch-Jerk-Clamp)"]:
        option_results = [r for r in results if r["option"] == option_name]
        if not option_results:
            continue

        print(f"\n┌{'─'*128}┐")
        print(f"│ {option_name:<126s} │")
        print(f"├{'─'*28}┬{'─'*18}┬{'─'*18}┬{'─'*18}┬{'─'*18}┬{'─'*18}┬{'─'*8}┤")
        print(f"│ {'Trial':26s} │ {'M1 GndClr':16s} │ {'M2 Depth':16s} │ {'M3 Align':16s} │ {'M4 Jerk':16s} │ {'M5 Carry':16s} │ {'PG':6s} │")
        print(f"├{'─'*28}┼{'─'*18}┼{'─'*18}┼{'─'*18}┼{'─'*18}┼{'─'*18}┼{'─'*8}┤")

        total_perfect = 0
        total_m_pass = [0, 0, 0, 0, 0]

        for r in option_results:
            m = r["metrics"]
            m1 = m["M1_ground_clearance"]
            m2 = m["M2_throat_depth"]
            m3 = m["M3_symmetry"]
            m4 = m["M4_avian_jerk"]
            m5 = m["M5_aloft_carry"]

            m1_str = f"{m1['min_mouth_z_mm']:5.1f}mm F={m1['floor_collisions']}"
            m2_str = f"{m2['rel_x_mm']:+6.1f}mm"
            m3_str = f"y={m3['yaw_err_deg']:4.1f}° l={m3['lateral_err_mm']:+4.1f}"
            m4_str = f"n={m4['peak_neck_vel']:.1f} h={m4['peak_head_vel']:.1f}"
            m5_str = f"b={m5['final_barbell_z_mm']:5.1f} t={m5['final_trunk_z_mm']:5.1f}"
            pg_str = "✅" if r["perfect_grasp"] else "❌"

            passes = [m1["pass"], m2["pass"], m3["pass"], m4["pass"], m5["pass"]]
            for i, p in enumerate(passes):
                total_m_pass[i] += int(p)
            total_perfect += int(r["perfect_grasp"])

            # Color pass/fail markers for each metric
            m1_mark = "✓" if m1["pass"] else "✗"
            m2_mark = "✓" if m2["pass"] else "✗"
            m3_mark = "✓" if m3["pass"] else "✗"
            m4_mark = "✓" if m4["pass"] else "✗"
            m5_mark = "✓" if m5["pass"] else "✗"

            trial_name = r.get("trial", "?")
            print(f"│ {trial_name:26s} │ {m1_mark} {m1_str:14s} │ {m2_mark} {m2_str:14s} │ {m3_mark} {m3_str:14s} │ {m4_mark} {m4_str:14s} │ {m5_mark} {m5_str:14s} │ {pg_str:6s} │")

        print(f"├{'─'*28}┼{'─'*18}┼{'─'*18}┼{'─'*18}┼{'─'*18}┼{'─'*18}┼{'─'*8}┤")
        n = len(option_results)
        print(f"│ {'PASS RATE':26s} │ {total_m_pass[0]:>2d}/{n:d}{'':<12s} │ {total_m_pass[1]:>2d}/{n:d}{'':<12s} │ {total_m_pass[2]:>2d}/{n:d}{'':<12s} │ {total_m_pass[3]:>2d}/{n:d}{'':<12s} │ {total_m_pass[4]:>2d}/{n:d}{'':<12s} │ {total_perfect:>2d}/{n:d}{'':<3s} │")
        print(f"└{'─'*28}┴{'─'*18}┴{'─'*18}┴{'─'*18}┴{'─'*18}┴{'─'*18}┴{'─'*8}┘")

    print(f"\n{'='*130}\n")


def main():
    print("\n" + "=" * 100)
    print(" 🦆 AVIAN PINCH-JERK-CLAMP — PARALLEL BENCHMARK")
    print("    Scoring against 5 Redefined 'Perfect Grasp' Metrics")
    print("    Mass: 100g Barbell | Friction: μ=3.5 Silicone | Welds: 0")
    print(f"    Trials: {len(TRIALS)} perturbation configurations × 2 options = {len(TRIALS)*2} total runs")
    print("=" * 100 + "\n")

    all_results = []

    # ── Option 2: Augmented Policy ────────────────────────────────────
    print("━" * 80)
    print(" >>> Running Option 2 (Augmented Policy – Avian Pinch-Jerk-Clamp)...")
    print("━" * 80)
    for trial in TRIALS:
        print(f"\n  Trial: {trial['name']} (X={trial['x']:.3f}, Y={trial['y']:+.3f}, Yaw={trial['yaw_deg']:.1f}°)")
        ctx = setup_sim_opt2()
        reset_env_opt2(ctx, x_cyl=trial["x"], y_cyl=trial["y"], yaw_deg=trial["yaw_deg"])
        res = run_avian_augmented_policy(ctx, total_steps=800, verbose=False)
        res["trial"] = trial["name"]
        all_results.append(res)

        m = res["metrics"]
        pg = "✅ PERFECT" if res["perfect_grasp"] else "❌ FAILED"
        print(f"    Result: {pg}")
        print(f"      M1={format_pass_fail(m['M1_ground_clearance']['pass'])} "
              f"M2={format_pass_fail(m['M2_throat_depth']['pass'])} "
              f"M3={format_pass_fail(m['M3_symmetry']['pass'])} "
              f"M4={format_pass_fail(m['M4_avian_jerk']['pass'])} "
              f"M5={format_pass_fail(m['M5_aloft_carry']['pass'])}")

    # ── Option 5: Analytical State Machine ────────────────────────────
    print("\n" + "━" * 80)
    print(" >>> Running Option 5 (Analytical State Machine – Avian Pinch-Jerk-Clamp)...")
    print("━" * 80)
    for trial in TRIALS:
        print(f"\n  Trial: {trial['name']} (X={trial['x']:.3f}, Y={trial['y']:+.3f}, Yaw={trial['yaw_deg']:.1f}°)")
        ctx = setup_sim_opt5()
        reset_env_opt5(ctx, x_cyl=trial["x"], y_cyl=trial["y"], yaw_deg=trial["yaw_deg"])
        res = run_avian_state_machine(ctx, total_steps=800, verbose=False)
        res["trial"] = trial["name"]
        all_results.append(res)

        m = res["metrics"]
        pg = "✅ PERFECT" if res["perfect_grasp"] else "❌ FAILED"
        print(f"    Result: {pg}")
        print(f"      M1={format_pass_fail(m['M1_ground_clearance']['pass'])} "
              f"M2={format_pass_fail(m['M2_throat_depth']['pass'])} "
              f"M3={format_pass_fail(m['M3_symmetry']['pass'])} "
              f"M4={format_pass_fail(m['M4_avian_jerk']['pass'])} "
              f"M5={format_pass_fail(m['M5_aloft_carry']['pass'])}")

    # ── Summary Table ─────────────────────────────────────────────────
    print_metric_table(all_results)

    # ── Save JSON Report ──────────────────────────────────────────────
    json_path = os.path.join(_ROOT, "data", "benchmark_avian_perfect_grasp.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Convert numpy types to Python native for JSON serialization
    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize(all_results), f, indent=2)

    print(f" Benchmark results saved to: {json_path}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
