import os, struct, math
import numpy as np

_ASSETS_DIR = "/Users/raghav/Desktop/microduck_manipulation/microduck_manipulation/models/assets"
_ROBOT_ASSETS = "/Users/raghav/Desktop/microduck_manipulation/microduck_rl/src/mjlab_microduck/robot/microduck/assets"
os.makedirs(_ASSETS_DIR, exist_ok=True)
os.makedirs(_ROBOT_ASSETS, exist_ok=True)

def write_binary_stl(filepath, triangles):
    """Writes an array of triangles (N, 3, 3) to a binary STL file."""
    triangles = np.asarray(triangles, dtype=np.float32)
    num_triangles = len(triangles)
    
    # Compute normals: (v1 - v0) x (v2 - v0)
    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    normals = np.cross(v1 - v0, v2 - v0)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    normals /= lens

    with open(filepath, "wb") as f:
        header = b"Real-World Solid Table Tennis Racket (Microduck Simulation)"
        header = header.ljust(80, b"\0")
        f.write(header)
        f.write(struct.pack("<I", num_triangles))
        
        for i in range(num_triangles):
            n = normals[i]
            t = triangles[i]
            f.write(struct.pack("<3f", n[0], n[1], n[2]))
            f.write(struct.pack("<3f", t[0, 0], t[0, 1], t[0, 2]))
            f.write(struct.pack("<3f", t[1, 0], t[1, 1], t[1, 2]))
            f.write(struct.pack("<3f", t[2, 0], t[2, 1], t[2, 2]))
            f.write(struct.pack("<H", 0))

def extrude_polygon(pts, y_bottom, y_top):
    """Extrudes a 2D polygon (x, z) between y_bottom and y_top."""
    tris = []
    n = len(pts)
    center = np.mean(pts, axis=0)
    
    # Top face (+Y)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        v0 = [center[0], y_top, center[1]]
        v1 = [p1[0], y_top, p1[1]]
        v2 = [p2[0], y_top, p2[1]]
        tris.append([v0, v1, v2])
        
    # Bottom face (-Y)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        v0 = [center[0], y_bottom, center[1]]
        v1 = [p2[0], y_bottom, p2[1]]
        v2 = [p1[0], y_bottom, p1[1]]
        tris.append([v0, v1, v2])
        
    # Side walls
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        v0 = [p1[0], y_bottom, p1[1]]
        v1 = [p2[0], y_bottom, p2[1]]
        v2 = [p2[0], y_top, p2[1]]
        v3 = [p1[0], y_top, p1[1]]
        tris.append([v0, v1, v2])
        tris.append([v0, v2, v3])
        
    return tris

def generate_solid_table_tennis_racket():
    """Generates 100% solid, non-hollow, CAD-quality table tennis racket STL meshes."""
    
    # 1. Blade Head Oval Curve (z from 0.065 to 0.222)
    zc = 0.145
    L_head = 0.077  # 154mm length
    W_head = 0.075  # 150mm width
    
    num_head = 80
    head_left = []
    head_right = []
    
    for i in range(num_head + 1):
        # theta from pi (bottom) to 0 (top)
        theta = math.pi - (math.pi * i / num_head)
        z = zc + L_head * math.cos(theta)
        # Egg shape fullness
        norm_z = (z - zc) / L_head
        w = W_head * math.sin(theta)
        if norm_z > 0:
            w *= (1.0 - 0.05 * norm_z)
        else:
            w *= (1.0 - 0.03 * norm_z)
            
        # Wings blend near neck (z: 0.065 to 0.090)
        if z < 0.090:
            t = max(0.0, (z - 0.065) / (0.090 - 0.065))
            blend = 0.5 * (1.0 - math.cos(math.pi * t))
            w = 0.014 + (w - 0.014) * blend
            z = max(0.065, z)
            
        head_left.append((-w, z))
        head_right.append((w, z))

    # 2. Flared Handle Profile (z from -0.035 to +0.065)
    num_h = 30
    handle_left = []
    handle_right = []
    for i in range(num_h + 1):
        z = -0.035 + 0.100 * (i / num_h)
        if z < 0.010:
            # Butt to waist
            t = (z - (-0.035)) / (0.010 - (-0.035))
            w = 0.0165 - 0.0035 * math.sin(math.pi * 0.5 * t)
        else:
            # Waist to collar
            t = (z - 0.010) / (0.065 - 0.010)
            w = 0.0130 + 0.0010 * math.sin(math.pi * 0.5 * t)
        handle_left.append((-w, z))
        handle_right.append((w, z))

    # 3. FULL TANG FULL-BODY SOLID WOOD CORE:
    # Continuous polygon encompassing BOTH the entire handle (from butt z = -0.035) AND the blade (to tip z = 0.222)
    # This guarantees the handle is completely SOLID WOOD, NEVER HOLLOW!
    full_wood_pts = []
    # Butt bottom line
    full_wood_pts.append((-0.0165, -0.035))
    # Left handle up to neck
    for p in handle_left:
        full_wood_pts.append(p)
    # Left blade up to tip
    for p in head_left:
        full_wood_pts.append(p)
    # Right blade down from tip
    for p in reversed(head_right):
        full_wood_pts.append(p)
    # Right handle down to butt
    for p in reversed(handle_right):
        full_wood_pts.append(p)
    full_wood_pts.append((0.0165, -0.035))
    full_wood_pts = np.array(full_wood_pts)

    # Blade Wood Core (thickness 5.8mm: y from -0.0029 to +0.0029) - 100% SOLID TANG!
    tris_wood = extrude_polygon(full_wood_pts, -0.0029, +0.0029)
    p_wood = os.path.join(_ASSETS_DIR, "tt_blade_wood.stl")
    write_binary_stl(p_wood, tris_wood)
    print(f"Generated SOLID full-tang wood core: {p_wood} ({len(tris_wood)} triangles)")

    # 4. Rubber Face Sheets (Red on top, Black on bottom)
    # Sits strictly on the blade head (z >= 0.065)
    rubber_pts = []
    for p in head_left:
        rubber_pts.append(p)
    for p in reversed(head_right):
        rubber_pts.append(p)
    rubber_pts = np.array(rubber_pts)

    # Red Rubber (Top face, y from +0.0029 to +0.0050, thickness 2.1mm)
    tris_red = extrude_polygon(rubber_pts, +0.0029, +0.0050)
    p_red = os.path.join(_ASSETS_DIR, "tt_rubber_red.stl")
    write_binary_stl(p_red, tris_red)
    print(f"Generated Red Rubber sheet: {p_red} ({len(tris_red)} triangles)")

    # Black Rubber (Bottom face, y from -0.0050 to -0.0029, thickness 2.1mm)
    tris_black = extrude_polygon(rubber_pts, -0.0050, -0.0029)
    p_black = os.path.join(_ASSETS_DIR, "tt_rubber_black.stl")
    write_binary_stl(p_black, tris_black)
    print(f"Generated Black Rubber sheet: {p_black} ({len(tris_black)} triangles)")

    # 5. Handle Scales (glued seamlessly directly on top and bottom of solid wood tang)
    handle_poly = []
    handle_poly.append((-0.0165, -0.035))
    for p in handle_left:
        handle_poly.append(p)
    for p in reversed(handle_right):
        handle_poly.append(p)
    handle_poly.append((0.0165, -0.035))
    handle_poly = np.array(handle_poly)

    # Top scale: y from +0.0029 to +0.0042 (thickness 1.3mm directly bonded onto tang)
    tris_h_top = extrude_polygon(handle_poly, +0.0029, +0.0042)
    # Bottom scale: y from -0.0042 to -0.0029 (thickness 1.3mm directly bonded onto tang)
    tris_h_bot = extrude_polygon(handle_poly, -0.0042, -0.0029)
    tris_handle = tris_h_top + tris_h_bot
    p_handle = os.path.join(_ASSETS_DIR, "tt_handle_scales.stl")
    write_binary_stl(p_handle, tris_handle)
    print(f"Generated Solid Handle Scales: {p_handle} ({len(tris_handle)} triangles)")

    # 6. Two-tone Handle Stripe Inlay:
    stripe_poly = np.array([
        [-0.0035, -0.034], [0.0035, -0.034],
        [0.0030, 0.058], [-0.0030, 0.058]
    ])
    tris_s_top = extrude_polygon(stripe_poly, +0.0041, +0.0043)
    tris_s_bot = extrude_polygon(stripe_poly, -0.0043, -0.0041)
    p_stripe = os.path.join(_ASSETS_DIR, "tt_handle_stripe.stl")
    write_binary_stl(p_stripe, tris_s_top + tris_s_bot)
    print(f"Generated Handle Stripe: {p_stripe}")

    # 7. Golden Lens / Metal Emblem in handle butt:
    lens_pts = []
    for i in range(24):
        a = 2.0 * math.pi * i / 24
        lens_pts.append((0.0055 * math.sin(a), -0.026 + 0.007 * math.cos(a)))
    lens_pts = np.array(lens_pts)
    tris_lens = extrude_polygon(lens_pts, +0.0040, +0.0044)
    p_lens = os.path.join(_ASSETS_DIR, "tt_handle_lens.stl")
    write_binary_stl(p_lens, tris_lens)
    print(f"Generated Handle Lens: {p_lens}")

    # Copy all to robot assets dir as well
    os.system(f"cp {_ASSETS_DIR}/tt_*.stl {_ROBOT_ASSETS}/")
    print("Copied all generated meshes to robot assets directory.")

if __name__ == "__main__":
    generate_solid_table_tennis_racket()
