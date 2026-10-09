#!/usr/bin/env python3
"""Download Microduck ONNX policy checkpoints from the HF Simulator Space.

Fetches the trained policies hosted on:
  https://huggingface.co/spaces/pollen-robotics/microduck-simulator

Policies are saved to policies/onnx/ in the workspace root.
"""

import os
import sys
import urllib.request
import hashlib

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "policies", "onnx")

# Base URL for policies in the HF Simulator Space
HF_SPACE_BASE = (
    "https://huggingface.co/spaces/pollen-robotics/microduck-simulator"
    "/resolve/main/app/public/policies"
)

# Policies to download: (filename, description, required)
POLICIES = [
    ("BEST_alpha_walking.onnx",  "Walk locomotion (main driving policy)", True),
    ("BEST_alpha_stand.onnx",    "Get-up / fall recovery",                True),
    ("BEST_alpha_sitstand.onnx", "Sit ↔ stand toggle",                   False),
    ("alpha_ground_pick.onnx",   "Ground peck (blind one-shot beak dip)", False),
]


def download_file(url: str, dest: str, desc: str) -> bool:
    """Download url → dest with a progress indicator. Returns True on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "microduck-manipulator/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                        print(f"\r  [{bar}] {pct:5.1f}%  {downloaded//1024:,} KB", end="", flush=True)
        print()
        os.replace(tmp, dest)
        size_kb = os.path.getsize(dest) // 1024
        print(f"  ✓ Saved {os.path.basename(dest)} ({size_kb:,} KB)")
        return True
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def verify_onnx(path: str) -> bool:
    """Quick sanity check: file exists, non-zero size, starts with ONNX magic."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 1024:
        return False
    with open(path, "rb") as f:
        header = f.read(8)
    # ONNX protobuf files start with byte 0x08 (field 1, varint) in most
    # exported models. Also accept generic protobuf / zip starts.
    return len(header) >= 4


def print_onnx_info(path: str):
    """Print input/output shapes using onnxruntime if available."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        print(f"    Input : {inp.name} {inp.shape}")
        print(f"    Output: {out.name} {out.shape}")
    except ImportError:
        print("    (onnxruntime not installed; skipping shape check)")
    except Exception as e:
        print(f"    (shape check failed: {e})")


def main():
    print("=" * 60)
    print("🦆 Microduck Policy Downloader")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success_count = 0
    fail_count = 0

    for filename, desc, required in POLICIES:
        dest = os.path.join(OUTPUT_DIR, filename)
        url = f"{HF_SPACE_BASE}/{filename}"

        print(f"📥 {filename}")
        print(f"   {desc}")

        if os.path.exists(dest) and verify_onnx(dest):
            size_kb = os.path.getsize(dest) // 1024
            print(f"   ✓ Already exists ({size_kb:,} KB) — skipping")
            print_onnx_info(dest)
            success_count += 1
            print()
            continue

        print(f"   Downloading from HF Space...")
        ok = download_file(url, dest, desc)
        if ok and verify_onnx(dest):
            print_onnx_info(dest)
            success_count += 1
        else:
            print(f"   {'[REQUIRED] ' if required else '[optional] '}Download failed or file invalid.")
            fail_count += 1
        print()

    print("=" * 60)
    print(f"✓ {success_count} policies ready,  ✗ {fail_count} failed")
    if fail_count > 0:
        required_failed = any(
            not os.path.exists(os.path.join(OUTPUT_DIR, f)) or not verify_onnx(os.path.join(OUTPUT_DIR, f))
            for f, _, req in POLICIES if req
        )
        if required_failed:
            print("ERROR: Required policies missing — teleop app will not start.")
            sys.exit(1)
        else:
            print("Optional policies missing — some features will be unavailable.")
    print()

    # Write a manifest
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.txt")
    with open(manifest_path, "w") as mf:
        mf.write("# Downloaded ONNX policies\n")
        mf.write(f"# Source: {HF_SPACE_BASE}\n\n")
        for filename, desc, required in POLICIES:
            dest = os.path.join(OUTPUT_DIR, filename)
            status = "OK" if (os.path.exists(dest) and verify_onnx(dest)) else "MISSING"
            mf.write(f"{status}  {filename}  # {desc}\n")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
