#!/usr/bin/env python3
"""Build multi-panel visual proof composite comparing the 4-phase drop, clamp & walk protocol."""

import os
from PIL import Image, ImageDraw, ImageFont

_ARTIFACT_DIR = "/Users/raghav/.gemini/antigravity-ide/brain/1dcd9661-c7ab-457a-8bec-a3fc2df3ade3"

def make_4phase_panel(racket_type, title, out_filename):
    steps = [
        ("step_10_beak_open", "PHASE 1: BEAK OPEN (22mm aperture)"),
        ("step_82_minimal_drop", "PHASE 2: MINIMAL DROP (Lands on jaw cradle)"),
        ("step_130_clamping", "PHASE 3: ACTIVE CLAMP (0.22 N·m torque)"),
        ("step_220_walking", "PHASE 4: DYNAMIC GAIT (Balance & walk)"),
    ]

    images_close = []
    images_ctx = []

    for tag, subhead in steps:
        p_close = os.path.join(_ARTIFACT_DIR, f"{racket_type}_{tag}_closeup.png")
        p_ctx = os.path.join(_ARTIFACT_DIR, f"{racket_type}_{tag}_context.png")
        im1 = Image.open(p_close).resize((480, 360))
        im2 = Image.open(p_ctx).resize((480, 360))

        # Add labels
        d1 = ImageDraw.Draw(im1)
        d1.rectangle([(0, 0), (480, 32)], fill=(15, 18, 24))
        d1.text((10, 8), f"CLOSE-UP: {subhead}", fill=(255, 215, 0))

        d2 = ImageDraw.Draw(im2)
        d2.rectangle([(0, 0), (480, 32)], fill=(15, 18, 24))
        d2.text((10, 8), f"FULL BODY: {subhead}", fill=(100, 220, 255))

        images_close.append(im1)
        images_ctx.append(im2)

    # 4 columns, 2 rows (Row 1: Close-up, Row 2: Full context)
    col_w, row_h = 480, 360
    header_h = 70
    total_w = col_w * 4 + 30
    total_h = row_h * 2 + header_h + 30

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 23, 30))
    draw = ImageDraw.Draw(canvas)

    draw.text((25, 18), title, fill=(255, 255, 255))
    draw.text((25, 42), "Authentic white silicone bump liners restored | Pure rigid Coulomb physics (Zero welds, zero tunneling)", fill=(180, 200, 220))

    for col in range(4):
        x = 10 + col * (col_w + 5)
        canvas.paste(images_close[col], (x, header_h + 10))
        canvas.paste(images_ctx[col], (x, header_h + row_h + 20))

    out_path = os.path.join(_ARTIFACT_DIR, out_filename)
    canvas.save(out_path)
    print(f"Saved: {out_path}")
    return out_path

def main():
    make_4phase_panel(
        "pickleball",
        "SCALED PICKLEBALL PADDLE: MINIMAL DROP, PHYSICAL CLAMPING & WALKING PROOF",
        "pickleball_4phase_verification_panel.png"
    )
    make_4phase_panel(
        "badminton",
        "SCALED BADMINTON RACKET: MINIMAL DROP, PHYSICAL CLAMPING & WALKING PROOF",
        "badminton_4phase_verification_panel.png"
    )

if __name__ == "__main__":
    main()
