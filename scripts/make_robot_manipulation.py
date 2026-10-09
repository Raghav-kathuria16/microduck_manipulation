#!/usr/bin/env python3
"""Generate robot_manipulation.xml from the official robot_allcollisions.xml.

Adds an articulated lower beak (the orange mandible) to the official microduck
model WITHOUT touching anything else:

  * moves jaw / bottom_head_shell / soft_mouth_top (+ mouth_tip site) from the
    rigid jaw_soft body into a new hinged body `duck_lower_beak`
  * hinge at jaw_soft-local (0, 0, -0.018) about the lateral (y) axis - the jaw
    root, where the servo bracket sits in the real robot
  * adds <position name="act_beak"> so the mouth can be actuated
  * adds neck_pitch<->duck_lower_beak contact exclude (official already excludes
    neck_pitch<->jaw_soft for the same phantom jaw-root contact)
  * jaw_soft inertial mass reduced by the moved mass so total robot mass is
    preserved (tensor kept - approximation, documented in CHANGELOG)

World poses at beak=0 are byte-identical to the official model: every moved
geom gets its local z increased by 0.018 to cancel the body-frame offset.

Run:  python scripts/make_robot_manipulation.py
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / ".external_repo" / "src" / "mjlab_microduck" / "robot" / "microduck"
SRC = REPO / "robot_allcollisions.xml"
DST = REPO / "robot_manipulation.xml"

# meshes that form the lower mandible (visual + collision pairs move together)
MOVED_MESHES = ("jaw", "bottom_head_shell", "soft_mouth_top")
MOVED_MASS = 0.09  # kg, split off from jaw_soft's 0.188766


def geom_block(lines, i):
    """Return (comment_line_indices, geom_line_indices) for the geom pair starting at comment line i."""
    comment, geoms = [i], []
    j = i + 1
    while j < len(lines) and "<geom" in lines[j]:
        geoms.append(j)
        j += 1
    return comment, geoms


def main():
    text = SRC.read_text()
    lines = text.splitlines(keepends=True)

    move = []  # line indices to cut
    moved_blocks = []  # (comment_idx, [geom line texts])

    for i, ln in enumerate(lines):
        m = re.search(r"<!-- Part ([\w.]+) -->", ln)
        if not m or m.group(1) not in MOVED_MESHES:
            continue
        # only inside the jaw_soft body (the first occurrences, before bearing_roll)
        comment, geoms = geom_block(lines, i)
        moved_blocks.append((comment, geoms))
        move.extend(comment + geoms)

    # mouth_tip site (inside jaw_soft, after the geoms)
    for i, ln in enumerate(lines):
        if 'name="mouth_tip"' in ln and "<site" in ln:
            move.append(i)
            mouth_tip_line = ln
            break
    else:
        raise SystemExit("mouth_tip site not found")

    assert len(moved_blocks) == 3, f"expected 3 moved mesh pairs, got {len(moved_blocks)}"

    def shift_z(pos_str):
        """pos='x y z' -> z += 0.018 (cancel the body offset (0,0,-0.018))."""
        nums = pos_str.split()
        z = float(nums[2]) + 0.018
        return f"{nums[0]} {nums[1]} {z:g}"

    child_geoms = []
    for comment, geoms in moved_blocks:
        child_geoms.append(lines[comment[0]])
        for g in geoms:
            ln = lines[g]
            m = re.search(r'pos="([^"]+)"', ln)
            ln = ln.replace(m.group(0), f'pos="{shift_z(m.group(1))}"')
            child_geoms.append(ln)
    # mouth_tip: shift z by +0.018 as well
    m = re.search(r'pos="([^"]+)"', mouth_tip_line)
    child_site = mouth_tip_line.replace(m.group(0), f'pos="{shift_z(m.group(1))}"')

    # build the child body block (8-space indent matches jaw_soft children)
    ind = " " * 14
    child = [
        f"{ind}<!-- Articulated lower beak (orange mandible) - added by make_robot_manipulation.py -->\n",
        f'{ind}<body name="duck_lower_beak" pos="0 0 -0.018">\n',
        f'{ind}  <joint name="beak_joint" type="hinge" axis="0 1 0" range="-0.05 0.6" damping="0.2"/>\n',
        f'{ind}  <inertial pos="-0.0096 0 -0.0189" mass="{MOVED_MASS}" '
        f'diaginertia="5e-5 1.1e-4 1.1e-4"/>\n',
    ] + [ln.replace(" " * 14, ind + "  ", 1) for ln in child_geoms] + [
        child_site.replace(" " * 14, ind + "  ", 1),
        f"{ind}</body>\n",
    ]

    # splice: drop moved lines, insert child before jaw_soft's closing </body>
    jaw_soft_close = None
    for i, ln in enumerate(lines):
        if "</body>" in ln and i not in move:
            # first </body> after the last moved line that isn't itself moved
            if any(m < i for m in move) and (jaw_soft_close is None):
                # ensure this close belongs to jaw_soft: it must come after
                # mouth_tip and before bearing_roll
                after_mt = any('name="mouth_tip"' in lines[k] for k in range(max(0, i - 10), i))
                if after_mt:
                    jaw_soft_close = i
    if jaw_soft_close is None:
        raise SystemExit("could not locate jaw_soft closing tag")

    out = []
    for i, ln in enumerate(lines):
        if i in move:
            continue
        if i == jaw_soft_close:
            out.extend(child)
        out.append(ln)
    new = "".join(out)

    # patch model name
    new = new.replace('<mujoco model="microduck">', '<mujoco model="microduck_manipulation">', 1)

    # reduce jaw_soft inertial mass by the moved mass (keep com/tensor)
    new = new.replace(
        'mass="0.188766"', f'mass="{0.188766 - MOVED_MASS:g}"', 1)

    # extend the official contact exclude
    new = new.replace(
        '<exclude body1="neck_pitch" body2="jaw_soft"/>',
        '<exclude body1="neck_pitch" body2="jaw_soft"/>\n'
        '    <exclude body1="neck_pitch" body2="duck_lower_beak"/>', 1)

    # add the beak actuator after right_ankle
    new = new.replace(
        '<position class="chosen_actuator" name="right_ankle" joint="right_ankle"/>',
        '<position class="chosen_actuator" name="right_ankle" joint="right_ankle"/>\n'
        '    <!-- Beak actuator (demo): stiff position servo holding the mandible -->\n'
        '    <position name="act_beak" joint="beak_joint" kp="15" kv="1"/>', 1)

    DST.write_text(new)
    print(f"wrote {DST} ({len(new.splitlines())} lines)")


if __name__ == "__main__":
    main()
