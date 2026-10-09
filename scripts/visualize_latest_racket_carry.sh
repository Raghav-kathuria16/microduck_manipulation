#!/bin/bash
# ==============================================================================
# One-Command MuJoCo Visualizer for Table Tennis Racket Carry
# ==============================================================================
# Automatically locates the latest trained policy, fuses it with the walking engine,
# and launches the interactive MuJoCo 3D viewer.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$MANIP_DIR/.." && pwd)"

PYTHON="${PYTHON:-python3}"
MJPYTHON="${MJPYTHON:-mjpython}"

# Find latest training run directory
RUN_DIR=$(ls -td $ROOT/microduck_rl/logs/rsl_rl/table_tennis_racket_carry/2026* 2>/dev/null | head -n 1)

echo "================================================================="
echo " Microduck Table Tennis Racket Carry - MuJoCo 3D Visualizer"
echo "================================================================="

if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR"/*.onnx ]; then
    RAW_ONNX=$(ls "$RUN_DIR"/*.onnx | head -n 1)
    FUSED_ONNX="$ROOT/microduck_manipulation/policies/onnx/table_tennis_racket_carry_latest.onnx"
    echo "Found latest checkpoint run: $(basename "$RUN_DIR")"
    echo "Fusing residual policy with base locomotion..."
    "$PYTHON" "$ROOT/microduck_manipulation/scripts/export_combined_residual.py" \
        --res-onnx "$RAW_ONNX" \
        --out-onnx "$FUSED_ONNX" >/dev/null 2>&1
    POLICY="$FUSED_ONNX"
else
    echo "Using validated precision level steadicam policy..."
    POLICY="$ROOT/microduck_manipulation/policies/onnx/table_tennis_racket_carry_precision_level.onnx"
fi

echo "Policy: $POLICY"
echo "Controls: [Space] Pause/Play | Mouse to Rotate/Zoom | [C] Clamp | [O] Open"
echo "================================================================="

exec "$MJPYTHON" "$ROOT/microduck_manipulation/scripts/test_racket_drop_and_walk.py" \
    --racket table_tennis \
    --no-ball \
    --steadicam \
    --policy "$POLICY" \
    --gui \
    --steps 6000 \
    "$@"
