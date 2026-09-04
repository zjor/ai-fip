# OpenSCAD CAD (wheel, rod, stand) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parametric, printable OpenSCAD models of the wheel, rod and table-clamp stand for the 2026 reaction-wheel pendulum, with headless builds, geometry tests, and a mass model fed back into the Phase 0 feasibility check.

**Architecture:** One `params.scad` holds every dimension. Each part file (`wheel.scad`, `rod.scad`, `stand.scad`) defines geometry modules and a `part` selector for headless export; `assembly.scad` places everything in the pendulum frame; `check.scad` renders interference intersections that must be empty. A stdlib-only Python tool computes volume, mass, centroid and moment of inertia from exported STLs; Python `unittest` drives OpenSCAD and asserts masses, bed fit and clearances.

**Tech Stack:** OpenSCAD 2026.06 (`/opt/homebrew/bin/openscad`, manifold backend), Python 3.12 stdlib, GNU make.

**Spec:** `docs/hardware/cad-spec.md`

## Global Constraints

- Coordinate frame (spec §2): origin at axle centre on the rod mid-plane, Z up along the upright rod, Y along the axle away from the table, X along the table edge. Every part file's *native* frame is the pendulum frame except the wheel parts, whose native axis is Z (they are rotated into place by the assembly).
- Units mm and g; densities `rho_pla = 1.25e-3`, `rho_steel = 7.85e-3` g/mm³.
- Bed 220 × 220 mm. A part fits if its print footprint `(a, b)` satisfies `max(a,b) <= 220` or `(a+b)/sqrt(2) <= 210`.
- Defaults from spec §3, with these plan-time refinements (also written back into the spec in Task 1): `pocket_n = 16` at 11.25° + k·22.5° (so the 8-bolt standard fit is every 45° and no hole lands on the lap seam), `beam_end = rod_len - 60` (bed fit), `hat = [49, 58]` mounted with the short side along X, bearing arm Y ∈ [−93.5, −46.5] with bearing centres at −90 and −50.
- Every OpenSCAD invocation: `openscad --backend=manifold -o <out.stl> [-D 'part="<name>"'] <file.scad>`; exit code must be 0 and stderr must not contain `WARNING` except the known "Current top level object is empty" for the interference check.
- No `mkdir` of `hardware/cad/build/` in git: it is ignored (Task 1 adds it to `.gitignore`).
- Commits only if the user has asked for them in this session; otherwise skip the commit steps and report the uncommitted state.

---

## File structure

```
hardware/cad/
  params.scad        every dimension and material constant
  lib.scad           helpers: plate(), arc(), at_circle(), hole(), hex_pocket(), ycyl()
  wheel.scad         quadrant(), hub(), bolts(), wheel(); part selector
  rod.scad           beam(), motor_flange(); part selector
  stand.scad         stand(), pad(); part selector
  assembly.scad      motor(), wheel_asm(), pendulum(a), axle(), table(); scene
  check.scad         interference intersections (must render empty)
  Makefile           parts, masses, check, view, test, clean
  README.md          build, print orientation, BOM, Status / Next
  tools/stl_props.py volume/mass/centroid/inertia of STL files
  tools/mass_model.py pendulum mass model from STLs + hardware masses
  tests/test_stl_props.py
  tests/test_cad.py
  build/             (ignored) rendered STLs
```

---

### Task 1: STL property tool, parameters, library, Makefile skeleton

**Files:**
- Create: `hardware/cad/tools/stl_props.py`
- Create: `hardware/cad/tests/test_stl_props.py`
- Create: `hardware/cad/params.scad`
- Create: `hardware/cad/lib.scad`
- Create: `hardware/cad/Makefile`
- Modify: `.gitignore` (add `hardware/cad/build/`)
- Modify: `docs/hardware/cad-spec.md` §3 (`pocket_n` 16), §4.1 (holes every 22.5°), §6 (masses from STL, not echoes)

**Interfaces:**
- Produces `stl_props.read_stl(path) -> list[tuple[vec3, vec3, vec3]]`, `stl_props.props(tris, axis: str) -> dict` with keys `volume_mm3, centroid, bbox_min, bbox_max, size, i_axis_mm5` (second moment about the given axis through the origin, in mm⁵ before density), and `stl_props.main()` CLI printing one line per file.
- Produces OpenSCAD modules in `lib.scad`: `plate(x0,x1,y0,y1,z0,z1)`, `arc(a0,a1,ro,ri,h)`, `at_circle(d,n,start)`, `hole(d,h)`, `hex_pocket(af,h)`, `ycyl(d,h,y0)` (cylinder along +Y from `y0`).

- [ ] **Step 1: Write the failing tests for the STL tool**

`hardware/cad/tests/test_stl_props.py`:

```python
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stl_props  # noqa: E402


def render_scad(code: str, out: pathlib.Path) -> None:
    src = out.with_suffix(".scad")
    src.write_text(code)
    result = subprocess.run(
        ["openscad", "--backend=manifold", "-o", str(out), str(src)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


class StlPropsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def test_cube_volume_and_centroid(self):
        out = self.tmp / "cube.stl"
        render_scad("cube(10);", out)
        p = stl_props.props(stl_props.read_stl(out), "z")
        self.assertAlmostEqual(p["volume_mm3"], 1000.0, places=6)
        self.assertAlmostEqual(p["centroid"][0], 5.0, places=6)
        self.assertEqual(p["size"], (10.0, 10.0, 10.0))

    def test_cylinder_inertia_about_z(self):
        out = self.tmp / "cyl.stl"
        render_scad("cylinder(r=10, h=10, center=true, $fn=360);", out)
        p = stl_props.props(stl_props.read_stl(out), "z")
        # I_z / V = r^2 / 2 for a solid cylinder
        self.assertAlmostEqual(p["i_axis_mm5"] / p["volume_mm3"], 50.0, delta=0.1)

    def test_translated_cylinder_uses_axis_through_origin(self):
        out = self.tmp / "cyl_off.stl"
        render_scad("translate([100,0,0]) cylinder(r=10, h=10, center=true, $fn=360);", out)
        p = stl_props.props(stl_props.read_stl(out), "z")
        # parallel axis: r^2/2 + d^2
        self.assertAlmostEqual(p["i_axis_mm5"] / p["volume_mm3"], 50.0 + 100.0**2, delta=1.0)

    def test_empty_stl_has_zero_volume(self):
        out = self.tmp / "empty.stl"
        out.write_text("solid OpenSCAD_Model\nendsolid OpenSCAD_Model\n")
        p = stl_props.props(stl_props.read_stl(out), "z")
        self.assertEqual(p["volume_mm3"], 0.0)

    def test_cli_prints_mass_with_density(self):
        out = self.tmp / "cube.stl"
        render_scad("cube(10);", out)
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "stl_props.py"), "--density", "1.25e-3", str(out)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mass_g=1.250", result.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd hardware/cad && python3 -m unittest tests.test_stl_props -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stl_props'`

- [ ] **Step 3: Implement the STL tool**

`hardware/cad/tools/stl_props.py`:

```python
#!/usr/bin/env python3
"""Volume, mass, centroid, bounding box and moment of inertia of closed STL meshes.

Uses the divergence theorem over signed tetrahedra (origin, triangle), so it
works for any closed, consistently oriented mesh, including several bodies.
The moment of inertia is about the chosen coordinate axis through the origin.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

Vec = tuple[float, float, float]
Tri = tuple[Vec, Vec, Vec]


def read_stl(path: str | Path) -> list[Tri]:
    data = Path(path).read_bytes()
    is_ascii = data[:5] == b"solid" and (b"facet" in data[:2000] or len(data) < 84)
    if is_ascii:
        tris: list[Tri] = []
        verts: list[Vec] = []
        for line in data.decode("ascii", "ignore").splitlines():
            parts = line.split()
            if parts and parts[0] == "vertex":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(verts) == 3:
                    tris.append((verts[0], verts[1], verts[2]))
                    verts = []
        return tris
    count = struct.unpack("<I", data[80:84])[0]
    tris = []
    for i in range(count):
        offset = 84 + i * 50
        v = struct.unpack("<12f", data[offset : offset + 48])
        tris.append(((v[3], v[4], v[5]), (v[6], v[7], v[8]), (v[9], v[10], v[11])))
    return tris


def _det(a: Vec, b: Vec, c: Vec) -> float:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def props(tris: list[Tri], axis: str = "z") -> dict:
    axis_index = "xyz".index(axis)
    others = [k for k in range(3) if k != axis_index]
    volume = 0.0
    first = [0.0, 0.0, 0.0]
    i_axis = 0.0
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for a, b, c in tris:
        v = _det(a, b, c) / 6.0
        volume += v
        for k in range(3):
            first[k] += v * (a[k] + b[k] + c[k]) / 4.0
        # For a tetrahedron with one vertex at the origin:
        # integral of x^2 dV = V/20 * (sum x_i^2 + (sum x_i)^2), i over the 3 vertices.
        s2 = sum(p[k] ** 2 for p in (a, b, c) for k in others)
        s1 = sum((a[k] + b[k] + c[k]) ** 2 for k in others)
        i_axis += v / 20.0 * (s2 + s1)
        for p in (a, b, c):
            for k in range(3):
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])
    if not tris:
        lo = hi = [0.0, 0.0, 0.0]
    centroid = tuple(f / volume for f in first) if volume else (0.0, 0.0, 0.0)
    size = tuple(round(h - l, 6) for l, h in zip(lo, hi))
    return {
        "volume_mm3": volume,
        "centroid": centroid,
        "bbox_min": tuple(lo),
        "bbox_max": tuple(hi),
        "size": size,
        "i_axis_mm5": i_axis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--density", type=float, default=1.25e-3, help="g/mm^3")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    args = parser.parse_args()
    for path in args.files:
        p = props(read_stl(path), args.axis)
        mass = p["volume_mm3"] * args.density
        inertia_kg_m2 = p["i_axis_mm5"] * args.density * 1e-9
        c = p["centroid"]
        s = p["size"]
        print(
            f"{Path(path).stem}: volume_mm3={p['volume_mm3']:.1f} mass_g={mass:.3f} "
            f"centroid=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) size=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f}) "
            f"I_{args.axis}_kg_m2={inertia_kg_m2:.6f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd hardware/cad && python3 -m unittest tests.test_stl_props -v`
Expected: 5 tests, OK. If `test_empty_stl_has_zero_volume` fails on parsing, the ASCII detection condition is wrong; the empty file is 48 bytes and must take the ASCII path.

- [ ] **Step 5: Write params.scad and lib.scad**

`hardware/cad/params.scad`:

```openscad
// All dimensions in mm, densities in g/mm^3. Source of truth: docs/hardware/cad-spec.md.
// Frame: origin at the axle centre on the rod mid-plane; Z up the upright rod;
// Y along the axle away from the table edge; X along the table edge.

bed = 220;
rho_pla = 1.25e-3;
rho_steel = 7.85e-3;

rod_len = 250;                 // axle axis to motor axis

// ---- wheel (native axis Z) ----
wheel_d = 280;
rim_radial = 8;
rim_axial = 8;
spokes = 8;                    // two per quadrant
spoke_w = 4;
spoke_t = 8;
hub_d = 60;
hub_t = 5;
hub_bc_d = 45;                 // quadrant flange screws
hub_bore_d = 10;
flange_ri = 13;                // quadrant flange sector inner radius (clears rotor screw heads)
pocket_n = 16;                 // tuning holes every 22.5 deg, first at 11.25 deg
bolt_fit = 8;                  // standard number of M6 tuning bolts
lap_len = 8;                   // rim half-lap length along the rim

// ---- motor mj5208 ----
motor_d = 63;
motor_len = 25;
rotor_bc_d = 17;
rotor_n = 3;
stator_pitch = 25;             // TO VERIFY against the mjbots 2D drawing (square assumed)
stator_n = 4;
flange_d = 70;
flange_plate_t = 6;

// ---- pivot ----
axle_d = 8;
axle_len = 120;
bearing_od = 22;
bearing_w = 7;
bearing_spacing = 40;
axle_overhang = 60;            // table edge to rod mid-plane
axle_height = 30;              // table top to axle centre
table_min = 18;
table_max = 40;

// ---- rod ----
beam_w = 20;                   // X
beam_h = 20;                   // Y
flange_t = 3;
web_t = 3;
tongue_len = 40;
beam_end = rod_len - 60;       // tongue ends here so the beam fits the bed diagonally
clamp_size = [30, 20, 30];     // X, Y, Z around the axle
battery = [35, 25, 75];        // X, Y, Z as mounted on the -Y face
hat = [49, 58];                // RPi/pi3hat hole pattern: X, Z
hat_hole = 2.8;

// ---- stand ----
jaw_w = 40;
clamp_wall = 8;
throat = 60;
arm_w = 40;

// ---- fits and hardware ----
clearance = 0.3;
M3 = 3.4;
M6 = 6.4;
M8 = 8.4;
nut_m3_af = 5.5;
nut_m3_h = 2.4;
nut_m8_af = 13;
nut_m8_h = 6.5;
$fn = 72;
```

`hardware/cad/lib.scad`:

```openscad
include <params.scad>

// Axis-aligned box from (x0,y0,z0) to (x1,y1,z1).
module plate(x0, x1, y0, y1, z0, z1) translate([x0, y0, z0]) cube([x1 - x0, y1 - y0, z1 - z0]);

// Ring sector between angles a0..a1 (deg), radii ri..ro, height h from z=0.
module arc(a0, a1, ro, ri, h) rotate([0, 0, a0]) rotate_extrude(angle = a1 - a0) translate([ri, 0]) square([ro - ri, h]);

// Children placed at n points on a circle of diameter d, first at angle start.
module at_circle(d, n, start = 0) for (i = [0 : n - 1]) rotate([0, 0, start + i * 360 / n]) translate([d / 2, 0, 0]) children();

// Through hole along Z spanning z in [-1, h+1].
module hole(d, h) translate([0, 0, -1]) cylinder(d = d, h = h + 2);

// Hex nut pocket, across-flats af, from z=0 up to h.
module hex_pocket(af, h) cylinder(d = af / cos(30) + clearance, h = h, $fn = 6);

// Cylinder along +Y starting at y0.
module ycyl(d, h, y0 = 0) translate([0, y0, 0]) rotate([-90, 0, 0]) cylinder(d = d, h = h);
```

- [ ] **Step 6: Write the Makefile skeleton and ignore the build directory**

`hardware/cad/Makefile`:

```make
OPENSCAD ?= openscad
OSFLAGS  ?= --backend=manifold
BUILD    := build
PY       ?= python3

PARTS := wheel_quadrant wheel_hub wheel_bolts beam motor_flange stand pad

.PHONY: parts masses check view test clean

parts: $(addprefix $(BUILD)/,$(addsuffix .stl,$(PARTS)))

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/wheel_quadrant.stl: wheel.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="quadrant"' -o $@ $<
$(BUILD)/wheel_hub.stl: wheel.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="hub"' -o $@ $<
$(BUILD)/wheel_bolts.stl: wheel.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="bolts"' -o $@ $<
$(BUILD)/beam.stl: rod.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="beam"' -o $@ $<
$(BUILD)/motor_flange.stl: rod.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="flange"' -o $@ $<
$(BUILD)/stand.stl: stand.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="stand"' -o $@ $<
$(BUILD)/pad.stl: stand.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="pad"' -o $@ $<
$(BUILD)/assembly.stl: assembly.scad wheel.scad rod.scad stand.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -o $@ $<
$(BUILD)/check.stl: check.scad assembly.scad wheel.scad rod.scad stand.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -o $@ $<

masses: parts
	$(PY) tools/stl_props.py --density 1.25e-3 --axis z $(BUILD)/wheel_quadrant.stl $(BUILD)/wheel_hub.stl
	$(PY) tools/stl_props.py --density 7.85e-3 --axis z $(BUILD)/wheel_bolts.stl
	$(PY) tools/stl_props.py --density 1.25e-3 --axis y $(BUILD)/beam.stl $(BUILD)/motor_flange.stl $(BUILD)/stand.stl $(BUILD)/pad.stl
	$(PY) tools/mass_model.py

check: $(BUILD)/check.stl
	$(PY) tools/stl_props.py $(BUILD)/check.stl | grep -q "volume_mm3=0.0 " && echo "check: no interference" || (echo "check: INTERFERENCE"; exit 1)

view: $(BUILD)/assembly.stl
	@echo "open $(BUILD)/assembly.stl in a viewer, or: openscad assembly.scad"

test:
	$(PY) -m unittest discover -s tests -v

clean:
	rm -rf $(BUILD)
```

Append to `.gitignore` at the repo root:

```
hardware/cad/build/
```

- [ ] **Step 7: Amend the spec for the plan-time refinements**

In `docs/hardware/cad-spec.md`:
- §3 row `pocket_n, pocket_bolt`: change `24, M6` → `16, M6` and "every 15°" → "every 22.5°, first at 11.25°".
- §4.1 "Tuning holes" bullet: "`pocket_n` axial through-holes … every 22.5° starting at 11.25°, so no hole lands on a lap seam or a spoke; the 8-bolt standard fit is every 45°."
- §6 `make masses` bullet: replace with "renders each part and computes volume, mass, centroid and moment of inertia from the STL with `tools/stl_props.py`; `tools/mass_model.py` combines them with the hardware masses into m_t, l_c, I_p and I_w."
- §4.2 first paragraph: "lap joint ending at Z = `rod_len − 60`" (was −50) and cutout centres at Z = 65, 100, 135.

- [ ] **Step 8: Smoke-test lib.scad renders**

Run:
```bash
cd hardware/cad && mkdir -p build && printf 'include <lib.scad>\nplate(0,10,0,10,0,10); translate([30,0,0]) arc(0,90,20,15,5); ycyl(5,10,50); hex_pocket(5.5,2.4);\n' > build/smoke.scad && openscad --backend=manifold -o build/smoke.stl build/smoke.scad && python3 tools/stl_props.py build/smoke.stl
```
Expected: exit 0, a line with `volume_mm3=` greater than 1000 and no WARNING on stderr.

- [ ] **Step 9: Commit (only if the user asked for commits)**

```bash
git add hardware/cad/tools/stl_props.py hardware/cad/tests/test_stl_props.py hardware/cad/params.scad hardware/cad/lib.scad hardware/cad/Makefile .gitignore docs/hardware/cad-spec.md
git commit -m "cad: STL property tool, parameters, helpers, Makefile skeleton"
```

---

### Task 2: Wheel — quadrant, hub plate, tuning bolts

**Files:**
- Create: `hardware/cad/wheel.scad`
- Create: `hardware/cad/tests/test_cad.py` (shared harness + wheel tests)

**Interfaces:**
- Consumes `lib.scad` modules and `params.scad` names from Task 1.
- Produces modules `quadrant()`, `hub()`, `bolts()`, `wheel()` in the wheel's native frame: hub plate z ∈ [0, hub_t], quadrant flanges and rim/spokes from z = hub_t upward (rim z ∈ [hub_t, hub_t + rim_axial]) in `wheel()`; individual parts render from z = 0. The wheel axis is Z. `part` ∈ {"quadrant", "hub", "bolts", "wheel"}.
- Produces test helpers `render(scad_name, part) -> (props, path)` and `fits_bed(a, b) -> bool` in `tests/test_cad.py` used by Tasks 3–6.

- [ ] **Step 1: Write the failing wheel tests**

`hardware/cad/tests/test_cad.py`:

```python
import math
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
sys.path.insert(0, str(ROOT / "tools"))
import stl_props  # noqa: E402

RHO_PLA = 1.25e-3
RHO_STEEL = 7.85e-3
BED = 220.0


def render(scad_name: str, part: str | None = None, axis: str = "z"):
    BUILD.mkdir(exist_ok=True)
    out = BUILD / f"test_{scad_name}_{part or 'scene'}.stl"
    cmd = ["openscad", "--backend=manifold", "-o", str(out)]
    if part is not None:
        cmd += ["-D", f'part="{part}"']
    cmd.append(str(ROOT / f"{scad_name}.scad"))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    warnings = [l for l in result.stderr.splitlines() if "WARNING" in l and "top level object is empty" not in l]
    if warnings:
        raise AssertionError("\n".join(warnings))
    return stl_props.props(stl_props.read_stl(out), axis), out


def fits_bed(a: float, b: float) -> bool:
    return max(a, b) <= BED or (a + b) / math.sqrt(2) <= BED - 10


class WheelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quadrant, _ = render("wheel", "quadrant")
        cls.hub, _ = render("wheel", "hub")
        cls.bolts, _ = render("wheel", "bolts")
        cls.wheel, _ = render("wheel", "wheel")

    def test_quadrant_mass_and_footprint(self):
        mass = self.quadrant["volume_mm3"] * RHO_PLA
        self.assertGreater(mass, 20.0)
        self.assertLess(mass, 40.0)
        sx, sy, sz = self.quadrant["size"]
        self.assertTrue(fits_bed(sx, sy), (sx, sy))
        self.assertAlmostEqual(sz, 8.0, delta=0.01)   # rim_axial; flange is thinner

    def test_quadrant_outer_radius(self):
        bbox_max = self.quadrant["bbox_max"]
        self.assertAlmostEqual(max(bbox_max[0], bbox_max[1]), 140.0, delta=0.5)

    def test_hub_mass(self):
        mass = self.hub["volume_mm3"] * RHO_PLA
        self.assertGreater(mass, 10.0)
        self.assertLess(mass, 25.0)
        self.assertAlmostEqual(self.hub["size"][2], 5.0, delta=0.01)

    def test_bolt_set_mass(self):
        mass = self.bolts["volume_mm3"] * RHO_STEEL
        self.assertGreater(mass, 50.0)   # 8 x (M6x20 + nut + head)
        self.assertLess(mass, 100.0)

    def test_wheel_inertia_with_standard_bolts(self):
        plastic = self.wheel["i_axis_mm5"] * RHO_PLA * 1e-9
        steel = self.bolts["i_axis_mm5"] * RHO_STEEL * 1e-9
        total = plastic + steel
        self.assertGreater(total, 0.0025, (plastic, steel))
        self.assertLess(total, 0.0035, (plastic, steel))

    def test_wheel_plastic_mass(self):
        mass = self.wheel["volume_mm3"] * RHO_PLA
        self.assertGreater(mass, 110.0)
        self.assertLess(mass, 170.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.WheelTest -v`
Expected: ERROR in `setUpClass` because `wheel.scad` does not exist (OpenSCAD exits non-zero: "Can't open input file").

- [ ] **Step 3: Implement wheel.scad**

```openscad
// Reaction wheel: four identical quadrants + hub plate, Ø280, printed rim and
// M6 tuning bolts provide the inertia. Native frame: wheel axis = Z, parts from z = 0.
include <lib.scad>

part = "quadrant";   // quadrant | hub | bolts | wheel

wr = wheel_d / 2;                 // rim outer radius
ri = wr - rim_radial;             // rim inner radius
rm = wr - rim_radial / 2;         // rim mid radius (holes, bolts)
la = lap_len / rm * 180 / PI;     // lap angle, deg
half = rim_axial / 2 - clearance / 2;

module quadrant() {
    difference() {
        union() {
            // full-height rim between the laps
            arc(la / 2, 90 - la / 2, wr, ri, rim_axial);
            // lower tongue at the 0 deg seam, upper tongue at the 90 deg seam
            arc(-la / 2, la / 2 + 0.01, wr, ri, half);
            translate([0, 0, rim_axial / 2 + clearance / 2]) arc(90 - la / 2 - 0.01, 90 + la / 2, wr, ri, half);
            // two spokes
            for (a = [22.5, 67.5]) rotate([0, 0, a]) translate([flange_ri + 2, -spoke_w / 2, 0]) cube([ri + 1 - (flange_ri + 2), spoke_w, spoke_t]);
            // hub flange sector (0.5 deg gap to the neighbours)
            arc(0.5, 89.5, hub_d / 2, flange_ri, hub_t);
        }
        // lap screw at the 0 deg seam
        translate([rm, 0, 0]) hole(M3, rim_axial);
        // flange screws
        for (a = [22.5, 67.5]) rotate([0, 0, a]) translate([hub_bc_d / 2, 0, 0]) hole(M3, hub_t);
        // tuning holes: pocket_n / 4 per quadrant
        for (i = [0 : pocket_n / 4 - 1]) rotate([0, 0, 180 / pocket_n + i * 360 / pocket_n]) translate([rm, 0, 0]) hole(M6, rim_axial);
    }
}

module hub() {
    difference() {
        cylinder(d = hub_d, h = hub_t);
        hole(hub_bore_d, hub_t);
        at_circle(rotor_bc_d, rotor_n) hole(M3, hub_t);
        // quadrant screws with trapped nuts on the motor side (z = 0)
        at_circle(hub_bc_d, 8, 22.5) {
            hole(M3, hub_t);
            translate([0, 0, -0.01]) hex_pocket(nut_m3_af, nut_m3_h);
        }
    }
}

// Steel hardware for the mass model: bolt_fit x (M6x20 bolt, head, nut) at the rim.
module bolts() {
    at_circle(2 * rm, bolt_fit, 180 / pocket_n) {
        translate([0, 0, -4]) cylinder(d = 10, h = 4);          // head
        translate([0, 0, -4]) cylinder(d = 6, h = 24);          // shank
        translate([0, 0, rim_axial]) cylinder(d = 10, h = 5, $fn = 6);  // nut
    }
}

// Assembled wheel for inertia and for the assembly file.
module wheel() {
    hub();
    translate([0, 0, hub_t]) for (a = [0 : 90 : 270]) rotate([0, 0, a]) quadrant();
}

if (part == "quadrant") quadrant();
else if (part == "hub") hub();
else if (part == "bolts") translate([0, 0, hub_t]) bolts();
else if (part == "wheel") wheel();
else assert(false, str("unknown part: ", part));
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.WheelTest -v`
Expected: 6 tests OK. If `test_wheel_inertia_with_standard_bolts` is below 0.0025, raise `bolt_fit` to 10 in `params.scad` and note it in the spec §4.1 (the design intent is to tune with bolts, not to thicken the rim). If a quadrant renders as two disconnected bodies, the tongues are not overlapping the full-height arc; check the `±0.01` angular overlaps.

- [ ] **Step 5: Render and eyeball once**

Run: `cd hardware/cad && make build/wheel_quadrant.stl build/wheel_hub.stl && openscad -D 'part="wheel"' -o build/wheel_preview.png --imgsize 1200,900 --camera 0,0,0,55,0,25,700 wheel.scad`
Expected: a PNG showing four quadrants on the hub with 16 rim holes; look at it (Read the PNG) and confirm the spokes reach the rim and the laps interleave.

- [ ] **Step 6: Commit (only if the user asked for commits)**

```bash
git add hardware/cad/wheel.scad hardware/cad/tests/test_cad.py
git commit -m "cad: wheel quadrants, hub plate, tuning bolts with inertia test"
```

---

### Task 3: Rod beam

**Files:**
- Create: `hardware/cad/rod.scad` (module `beam` and part selector; `motor_flange` is Task 4)
- Modify: `hardware/cad/tests/test_cad.py` (add `BeamTest`)

**Interfaces:**
- Produces `beam()` in the pendulum frame: axle clamp block centred on the origin, web from Z = 15 to `beam_end`, flanges from Z = −40 to `beam_end − tongue_len`, tongue (web only) Z ∈ [`beam_end − tongue_len`, `beam_end`] with two M3 holes along X at Z = `beam_end − 30` and `beam_end − 10`.
- Produces `plate`-based helper `i_flanges(z0, z1)` and `i_web(z0, z1)` reused by `motor_flange()`.

- [ ] **Step 1: Write the failing beam tests**

Append to `hardware/cad/tests/test_cad.py`:

```python
class BeamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.beam, _ = render("rod", "beam", axis="y")

    def test_mass(self):
        mass = self.beam["volume_mm3"] * RHO_PLA
        self.assertGreater(mass, 50.0)
        self.assertLess(mass, 100.0)

    def test_extents_and_bed_fit(self):
        lo, hi = self.beam["bbox_min"], self.beam["bbox_max"]
        self.assertAlmostEqual(lo[2], -40.0, delta=0.01)     # tray bottom
        self.assertAlmostEqual(hi[2], 190.0, delta=0.01)     # beam_end = rod_len - 60
        self.assertAlmostEqual(hi[1], 10.0, delta=0.01)      # +Y flange face
        self.assertAlmostEqual(lo[1], -10.0, delta=0.01)     # -Y flange face
        sx, sy, sz = self.beam["size"]
        # printed web-flat (X up): footprint is Y x Z
        self.assertTrue(fits_bed(sy, sz), (sy, sz))
        self.assertLessEqual(sx, 57.0 + 0.01)                # HAT plate width

    def test_centroid_is_low_on_the_rod(self):
        # the tray and clamp near the axle should keep the CoM below mid-span
        self.assertLess(self.beam["centroid"][2], 90.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.BeamTest -v`
Expected: ERROR, `rod.scad` missing.

- [ ] **Step 3: Implement rod.scad with the beam**

```openscad
// Pendulum rod: I-beam with axle clamp, battery plate (-Y) and HAT plate (+Y),
// tongue for the motor flange lap joint. Native frame = pendulum frame.
include <lib.scad>

part = "beam";   // beam | flange

module i_flanges(z0, z1) {
    plate(-beam_w / 2, beam_w / 2, -beam_h / 2, -beam_h / 2 + flange_t, z0, z1);
    plate(-beam_w / 2, beam_w / 2, beam_h / 2 - flange_t, beam_h / 2, z0, z1);
}

module i_web(z0, z1, x_offset = 0) {
    plate(x_offset - web_t / 2, x_offset + web_t / 2, -beam_h / 2 + flange_t, beam_h / 2 - flange_t, z0, z1);
}

module lap_holes(x_len) {
    for (z = [beam_end - 30, beam_end - 10]) translate([0, 0, z]) rotate([0, 90, 0]) cylinder(d = M3, h = x_len, center = true);
}

module beam() {
    difference() {
        union() {
            i_flanges(-40, beam_end - tongue_len);
            i_web(15, beam_end);
            // battery plate on -Y, 40 wide, Z in [-40, 40]
            plate(-battery[0] / 2 - 2.5, battery[0] / 2 + 2.5, -beam_h / 2, -beam_h / 2 + flange_t, -40, 40);
            // HAT plate on +Y, 57 x 70
            plate(-hat[0] / 2 - 4, hat[0] / 2 + 4, beam_h / 2 - flange_t, beam_h / 2, -hat[1] / 2 - 6, hat[1] / 2 + 6);
            // axle clamp block
            translate([-clamp_size[0] / 2, -clamp_size[1] / 2, -clamp_size[2] / 2]) cube(clamp_size);
        }
        // axle bore along Y, clamp slot below it, two M3 clamp screws along X
        ycyl(axle_d + clearance, clamp_size[1] + 2, -clamp_size[1] / 2 - 1);
        plate(-0.75, 0.75, -clamp_size[1] / 2 - 1, clamp_size[1] / 2 + 1, -clamp_size[2] / 2 - 1, 0);
        for (y = [-5, 5]) translate([0, y, -11]) rotate([0, 90, 0]) cylinder(d = M3, h = clamp_size[0] + 2, center = true);
        // optional cross pin through clamp and axle
        cylinder(d = 2, h = clamp_size[2] / 2 + 1);
        // web lightening cutouts, ellipse 8 (Y) x 30 (Z), through X
        for (z = [65, 100, 135]) translate([0, 0, z]) rotate([0, 90, 0]) scale([30 / 8, 1, 1]) cylinder(d = 8, h = web_t + 2, center = true);
        // strap slots 3 (X) x 12 (Z) through the battery plate
        for (x = [-17, 17], z = [-25, 25]) plate(x - 1.5, x + 1.5, -beam_h / 2 - 1, -beam_h / 2 + flange_t + 1, z - 6, z + 6);
        // HAT holes along Y
        for (x = [-hat[0] / 2, hat[0] / 2], z = [-hat[1] / 2, hat[1] / 2]) translate([x, 0, z]) ycyl(hat_hole, flange_t + 2, beam_h / 2 - flange_t - 1);
        // lap holes through the tongue
        lap_holes(web_t + 2);
    }
}

module motor_flange() { assert(false, "motor_flange: implemented in Task 4"); }

if (part == "beam") beam();
else if (part == "flange") motor_flange();
else assert(false, str("unknown part: ", part));
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.BeamTest -v`
Expected: 3 tests OK. If `hi[2]` is not 190, `beam_end` in params is wrong or the web extends past it.

- [ ] **Step 5: Preview**

Run: `cd hardware/cad && openscad -D 'part="beam"' -o build/beam_preview.png --imgsize 1200,900 --camera 0,0,75,60,0,30,600 rod.scad`
Expected: PNG with the clamp block at the bottom, the wide plates around it, three elliptical web cutouts, and a bare web tongue at the top. Read the PNG and confirm.

- [ ] **Step 6: Commit (only if the user asked for commits)**

```bash
git add hardware/cad/rod.scad hardware/cad/tests/test_cad.py
git commit -m "cad: rod beam with clamp, trays, cutouts and lap tongue"
```

---

### Task 4: Motor flange

**Files:**
- Modify: `hardware/cad/rod.scad` (replace the `motor_flange` stub)
- Modify: `hardware/cad/tests/test_cad.py` (add `MotorFlangeTest`)

**Interfaces:**
- Produces `motor_flange()` in the pendulum frame: disc Ø`flange_d` × `flange_plate_t` centred at Z = `rod_len`, Y ∈ [−3, 3]; stub with `i_flanges` and an offset `i_web(x_offset = web_t)` from Z = `beam_end − tongue_len` to Z = `rod_len − flange_d/2 + 5`; stator holes along Y on the `stator_pitch` square; Ø12 centre hole; lap holes matching `beam()`.

- [ ] **Step 1: Write the failing flange tests**

Append to `hardware/cad/tests/test_cad.py`:

```python
class MotorFlangeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flange, _ = render("rod", "flange", axis="y")

    def test_mass(self):
        mass = self.flange["volume_mm3"] * RHO_PLA
        self.assertGreater(mass, 25.0)
        self.assertLess(mass, 60.0)

    def test_extents(self):
        lo, hi = self.flange["bbox_min"], self.flange["bbox_max"]
        self.assertAlmostEqual(hi[2], 250.0 + 35.0, delta=0.01)   # disc top
        self.assertAlmostEqual(lo[2], 150.0, delta=0.01)          # stub start = beam_end - tongue_len
        self.assertAlmostEqual(hi[0], 35.0, delta=0.01)
        self.assertAlmostEqual(hi[1], 10.0, delta=0.01)           # stub flange face
        sx, sy, sz = self.flange["size"]
        self.assertTrue(fits_bed(sz, sy), (sz, sy))               # printed on edge (X up)

    def test_stub_and_disc_are_one_body_by_volume(self):
        # disc alone is ~23,000 mm3; the stub adds at least 8,000
        self.assertGreater(self.flange["volume_mm3"], 31000.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.MotorFlangeTest -v`
Expected: ERROR from the `assert(false, "motor_flange: implemented in Task 4")` stub (OpenSCAD exits non-zero).

- [ ] **Step 3: Implement motor_flange in rod.scad**

Replace the stub line with:

```openscad
module motor_flange() {
    stub_top = rod_len - flange_d / 2 + 5;
    difference() {
        union() {
            translate([0, 0, rod_len]) ycyl(flange_d, flange_plate_t, -flange_plate_t / 2);
            i_flanges(beam_end - tongue_len, stub_top);
            i_web(beam_end - tongue_len, stub_top, web_t);   // offset web lies beside the beam's tongue
        }
        // stator screws on a square, along Y, and the rotor-side centre hole
        for (x = [-1, 1], z = [-1, 1]) translate([x * stator_pitch / 2, 0, rod_len + z * stator_pitch / 2]) ycyl(M3, flange_plate_t + 2, -flange_plate_t / 2 - 1);
        translate([0, 0, rod_len]) ycyl(12, flange_plate_t + 2, -flange_plate_t / 2 - 1);
        lap_holes(3 * web_t + 2);
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.MotorFlangeTest -v`
Expected: 3 tests OK.

- [ ] **Step 5: Verify the lap joint mates**

Run:
```bash
cd hardware/cad && printf 'include <lib.scad>\nuse <rod.scad>\nintersection(){ beam(); motor_flange(); }\n' > build/lap.scad && openscad --backend=manifold -o build/lap.stl build/lap.scad; python3 tools/stl_props.py build/lap.stl
```
Expected: `volume_mm3=0.0` (the offset web must not overlap the tongue). If non-zero, `i_web(..., web_t)` is not offset by a full web thickness.

- [ ] **Step 6: Commit (only if the user asked for commits)**

```bash
git add hardware/cad/rod.scad hardware/cad/tests/test_cad.py
git commit -m "cad: motor flange with stator pattern and lap joint"
```

---

### Task 5: Stand — table clamp with bearing arm, and the screw pad

**Files:**
- Create: `hardware/cad/stand.scad`
- Modify: `hardware/cad/tests/test_cad.py` (add `StandTest`)

**Interfaces:**
- Produces `stand()` in the pendulum frame: top jaw on the table top (Z ∈ [−30, −22], Y ∈ [−120, −60]), back plate outside the edge (Y ∈ [−60, −52]), lower jaw (Z ∈ [−78, −70]) with an M8 hole and an upward-opening M8 nut pocket, bearing arm block X ∈ [−20, 20], Y ∈ [−93.5, −46.5], Z ∈ [−22, 20] with two Ø22.2 × 7 pockets and a Ø17 relief bore.
- Produces `pad()`: Ø25 × 6 disc with a Ø8.4 blind hole 4 deep, at the origin (own frame).

- [ ] **Step 1: Write the failing stand tests**

Append to `hardware/cad/tests/test_cad.py`:

```python
class StandTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stand, _ = render("stand", "stand", axis="y")
        cls.pad, _ = render("stand", "pad", axis="z")

    def test_mass(self):
        mass = self.stand["volume_mm3"] * RHO_PLA
        self.assertGreater(mass, 100.0)
        self.assertLess(mass, 300.0)

    def test_extents_and_clearances(self):
        lo, hi = self.stand["bbox_min"], self.stand["bbox_max"]
        self.assertAlmostEqual(hi[1], -46.5, delta=0.01)   # arm outboard face
        self.assertAlmostEqual(lo[1], -120.0, delta=0.01)  # jaw tip
        self.assertAlmostEqual(hi[2], 20.0, delta=0.01)    # arm top
        self.assertAlmostEqual(lo[2], -78.0, delta=0.01)   # lower jaw bottom (table_max 40)
        self.assertAlmostEqual(hi[0], 20.0, delta=0.01)
        sx, sy, sz = self.stand["size"]
        self.assertTrue(fits_bed(sy, sz), (sy, sz))        # printed on its side (X up)

    def test_pad(self):
        self.assertAlmostEqual(self.pad["size"][0], 25.0, delta=0.01)
        self.assertAlmostEqual(self.pad["size"][2], 6.0, delta=0.01)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.StandTest -v`
Expected: ERROR, `stand.scad` missing.

- [ ] **Step 3: Implement stand.scad**

```openscad
// Table-edge C-clamp with a cantilever bearing arm. Native frame = pendulum frame.
include <lib.scad>

part = "stand";   // stand | pad

edge = -axle_overhang;          // table edge plane (Y)
top = -axle_height;             // table top plane (Z)
y_out = -(axle_overhang - 10);              // outboard bearing centre (-50)
y_in = y_out - bearing_spacing;             // inboard bearing centre (-90)
arm_y0 = y_in - bearing_w / 2;              // -93.5
arm_y1 = y_out + bearing_w / 2;             // -46.5
lower_jaw_z = top - table_max - clamp_wall; // -78

module stand() {
    difference() {
        union() {
            plate(-jaw_w / 2, jaw_w / 2, edge - throat, edge, top, top + clamp_wall);                 // top jaw
            plate(-jaw_w / 2, jaw_w / 2, edge, edge + clamp_wall, lower_jaw_z, top + clamp_wall);    // back
            plate(-jaw_w / 2, jaw_w / 2, edge - throat, edge, lower_jaw_z, lower_jaw_z + clamp_wall); // lower jaw
            plate(-arm_w / 2, arm_w / 2, arm_y0, arm_y1, top + clamp_wall, 20);                      // bearing arm
        }
        // bearing pockets from each end and the relief bore between them
        ycyl(bearing_od + 0.2, bearing_w + 1, arm_y0 - 1);
        ycyl(bearing_od + 0.2, bearing_w + 1, arm_y1 - bearing_w);
        ycyl(17, arm_y1 - arm_y0 + 2, arm_y0 - 1);
        // thumb screw through the lower jaw, nut pocket opening upward into the throat
        translate([0, edge - throat / 2, lower_jaw_z - 1]) cylinder(d = M8, h = clamp_wall + 2);
        translate([0, edge - throat / 2, lower_jaw_z + clamp_wall - nut_m8_h]) hex_pocket(nut_m8_af, nut_m8_h + 0.01);
    }
}

// Pressure pad for the thumb screw (own frame, prints flat).
module pad() {
    difference() {
        cylinder(d = 25, h = 6);
        translate([0, 0, 2]) cylinder(d = M8, h = 5);
    }
}

if (part == "stand") stand();
else if (part == "pad") pad();
else assert(false, str("unknown part: ", part));
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.StandTest -v`
Expected: 3 tests OK.

- [ ] **Step 5: Preview**

Run: `cd hardware/cad && openscad -D 'part="stand"' -o build/stand_preview.png --imgsize 1200,900 --camera 0,-80,-30,60,0,140,500 stand.scad`
Expected: a C-clamp opening toward −Y with a block on top carrying two bearing bores along Y. Read the PNG.

- [ ] **Step 6: Commit (only if the user asked for commits)**

```bash
git add hardware/cad/stand.scad hardware/cad/tests/test_cad.py
git commit -m "cad: table clamp stand with bearing arm and screw pad"
```

---

### Task 6: Assembly, interference check, Makefile targets

**Files:**
- Create: `hardware/cad/assembly.scad`
- Create: `hardware/cad/check.scad`
- Modify: `hardware/cad/tests/test_cad.py` (add `AssemblyTest`)

**Interfaces:**
- Consumes `wheel()`, `bolts()`, `beam()`, `motor_flange()`, `stand()`.
- Produces `motor()`, `wheel_asm()`, `pendulum(a)`, `axle()`, `table_top()` in `assembly.scad`; `check.scad` renders `∪_a intersection(pendulum(a), stand() ∪ table_top())` for a ∈ {0, 90, 180, 270}.

- [ ] **Step 1: Write the failing assembly tests**

Append to `hardware/cad/tests/test_cad.py`:

```python
class AssemblyTest(unittest.TestCase):
    def test_assembly_renders_with_expected_height(self):
        scene, _ = render("assembly", None, axis="y")
        lo, hi = scene["bbox_min"], scene["bbox_max"]
        self.assertAlmostEqual(hi[2], 250.0 + 140.0, delta=0.5)   # wheel top when upright
        self.assertAlmostEqual(lo[2], -78.0, delta=0.5)           # lower jaw
        self.assertGreater(hi[1], 40.0)                           # wheel on the +Y side

    def test_no_interference_at_four_angles(self):
        check, _ = render("check", None, axis="y")
        self.assertEqual(check["volume_mm3"], 0.0, check["bbox_max"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.AssemblyTest -v`
Expected: ERROR, `assembly.scad` missing.

- [ ] **Step 3: Implement assembly.scad and check.scad**

`hardware/cad/assembly.scad`:

```openscad
// Full pendulum on its stand. angle = pendulum angle from upright about Y (deg).
include <lib.scad>
use <wheel.scad>
use <rod.scad>
use <stand.scad>

angle = 0;
show_table = true;

module motor() translate([0, 0, rod_len]) ycyl(motor_d, motor_len, flange_plate_t / 2);

// wheel native Z -> pendulum +Y, hub plate against the rotor face
module wheel_asm() translate([0, flange_plate_t / 2 + motor_len, rod_len]) rotate([-90, 0, 0]) {
    wheel();
    translate([0, 0, hub_t]) bolts();
}

module pendulum(a = angle) rotate([0, a, 0]) {
    beam();
    motor_flange();
    motor();
    wheel_asm();
}

module axle() ycyl(axle_d, axle_len, -105);

module table_top() translate([-200, -axle_overhang - 400, -axle_height - 25]) cube([400, 400, 25]);

stand();
axle();
pendulum();
if (show_table) %table_top();
```

`hardware/cad/check.scad`:

```openscad
// Interference check: must render EMPTY. The axle is excluded (it legitimately
// passes through the stand bearings). Run: openscad -o build/check.stl check.scad
include <lib.scad>
use <assembly.scad>
use <stand.scad>

for (a = [0, 90, 180, 270]) intersection() {
    pendulum(a);
    union() { stand(); table_top(); }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.AssemblyTest -v`
Expected: 2 tests OK. If OpenSCAD refuses to export an empty top-level object (non-zero exit), change `render()` to accept that specific case: when stderr contains "top level object is empty", write an empty STL to `out` and return zero props. If `check` reports a non-zero volume, inspect `bbox_max` in the assertion message: Y between −46 and −35 means the battery plate hits the arm; Z around −30 means the wheel hits the table when hanging.

- [ ] **Step 5: Run the full Makefile flow**

Run: `cd hardware/cad && make clean parts check test 2>&1 | tail -20`
Expected: all STLs built, `check: no interference`, and all unit tests OK (the `masses` target still fails because `tools/mass_model.py` does not exist yet; that is Task 7).

- [ ] **Step 6: Render the scene image for the README**

Run: `cd hardware/cad && openscad -o build/assembly.png --imgsize 1600,1200 --camera 40,-60,120,60,0,35,1100 assembly.scad`
Expected: PNG of the whole pendulum upright on the clamp with a translucent table. Read it and confirm the wheel is outboard, the battery side faces the table, and nothing touches.

- [ ] **Step 7: Commit (only if the user asked for commits)**

```bash
git add hardware/cad/assembly.scad hardware/cad/check.scad hardware/cad/tests/test_cad.py
git commit -m "cad: assembly scene and interference check"
```

---

### Task 7: Mass model, Phase 0 feedback, documentation

**Files:**
- Create: `hardware/cad/tools/mass_model.py`
- Create: `hardware/cad/README.md`
- Modify: `hardware/cad/tests/test_cad.py` (add `MassModelTest`)
- Modify: `software/sim/phase0.toml` (`[design]` nominal from the CAD, ±10 % ranges)
- Modify: `docs/hardware/cad-spec.md` §5 table (rendered masses)
- Modify: `docs/README.md` (link the spec), `LOG.md` (entry), `ROADMAP.md` (Phase 1 mechanics line)

**Interfaces:**
- Produces `mass_model.compute(build_dir) -> dict` with keys `m_t_kg, l_c_m, G_nm, I_p_kg_m2, I_w_kg_m2, parts` where `parts` is a list of `(name, mass_g, z_com_mm)`.

- [ ] **Step 1: Write the failing mass-model test**

Append to `hardware/cad/tests/test_cad.py`:

```python
class MassModelTest(unittest.TestCase):
    def test_pendulum_totals_are_inside_phase0_planning_range(self):
        import mass_model
        for part in ("wheel_quadrant", "wheel_hub", "wheel_bolts", "beam", "motor_flange"):
            render_part = {"wheel_quadrant": ("wheel", "quadrant"), "wheel_hub": ("wheel", "hub"),
                           "wheel_bolts": ("wheel", "bolts"), "beam": ("rod", "beam"),
                           "motor_flange": ("rod", "flange")}[part]
            _, path = render(*render_part)
            path.replace(BUILD / f"{part}.stl")
        model = mass_model.compute(BUILD)
        self.assertGreater(model["m_t_kg"], 0.6)
        self.assertLess(model["m_t_kg"], 0.9)
        self.assertGreater(model["G_nm"], 0.97)
        self.assertLess(model["G_nm"], 1.34)
        self.assertGreater(model["I_w_kg_m2"], 0.0025)
        self.assertLess(model["I_w_kg_m2"], 0.0035)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.MassModelTest -v`
Expected: `ModuleNotFoundError: No module named 'mass_model'`.

- [ ] **Step 3: Implement mass_model.py**

```python
#!/usr/bin/env python3
"""Pendulum mass model from rendered STLs plus purchased hardware.

Printed parts come from build/*.stl (pendulum frame; wheel parts in the wheel
frame at z = rod_len). Hardware masses are catalogue values. Output feeds
software/sim/phase0.toml.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stl_props  # noqa: E402

G = 9.81
RHO_PLA = 1.25e-3
RHO_STEEL = 7.85e-3
ROD_LEN_MM = 250.0

# name: (mass_g, z_com_mm). Sources: mjbots pages (motor, r4.11), typical 4S 1000 mAh
# pack, RPi 4 + pi3hat + standoffs (unsure, +-15 g), M3/M8 hardware estimate.
HARDWARE = {
    "mj5208": (193.0, ROD_LEN_MM),
    "moteus_r4.11": (14.2, ROD_LEN_MM),
    "motor_screws": (5.0, ROD_LEN_MM),
    "battery_4s": (120.0, 0.0),
    "rpi4_pi3hat": (85.0, 0.0),
    "axle_collars_clamp_screws": (60.0, 0.0),
}


def _stl(build: Path, name: str, axis: str):
    return stl_props.props(stl_props.read_stl(build / f"{name}.stl"), axis)


def compute(build: Path) -> dict:
    parts: list[tuple[str, float, float]] = []
    quadrant = _stl(build, "wheel_quadrant", "z")
    hub = _stl(build, "wheel_hub", "z")
    bolts = _stl(build, "wheel_bolts", "z")
    wheel_plastic_g = (4 * quadrant["volume_mm3"] + hub["volume_mm3"]) * RHO_PLA
    bolts_g = bolts["volume_mm3"] * RHO_STEEL
    i_w = ((4 * quadrant["i_axis_mm5"] + hub["i_axis_mm5"]) * RHO_PLA + bolts["i_axis_mm5"] * RHO_STEEL) * 1e-9
    parts.append(("wheel_plastic", wheel_plastic_g, ROD_LEN_MM))
    parts.append(("wheel_bolts", bolts_g, ROD_LEN_MM))
    for name in ("beam", "motor_flange"):
        p = _stl(build, name, "y")
        parts.append((name, p["volume_mm3"] * RHO_PLA, p["centroid"][2]))
    for name, (mass, z) in HARDWARE.items():
        parts.append((name, mass, z))

    m_t = sum(m for _, m, _ in parts) / 1000.0
    first = sum(m / 1000.0 * z / 1000.0 for _, m, z in parts)
    l_c = first / m_t
    # pendulum body inertia about the pivot: point masses at their CoM plus the
    # beam as a slender rod (m L^2 / 3 about its lower end)
    i_p = sum(m / 1000.0 * (z / 1000.0) ** 2 for name, m, z in parts if name != "beam")
    beam_mass = next(m for name, m, _ in parts if name == "beam") / 1000.0
    i_p += beam_mass * (ROD_LEN_MM / 1000.0) ** 2 / 3.0
    return {
        "m_t_kg": m_t,
        "l_c_m": l_c,
        "G_nm": m_t * G * l_c,
        "I_p_kg_m2": i_p,
        "I_w_kg_m2": i_w,
        "parts": parts,
    }


def main() -> None:
    build = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "build"
    model = compute(build)
    print("| Part | Mass g | z CoM mm |")
    print("|---|---:|---:|")
    for name, mass, z in model["parts"]:
        print(f"| {name} | {mass:.1f} | {z:.0f} |")
    print()
    print(f"m_t = {model['m_t_kg']:.3f} kg, l_c = {model['l_c_m']:.3f} m, G = {model['G_nm']:.3f} Nm")
    print(f"I_p = {model['I_p_kg_m2']:.4f} kg m2, I_w = {model['I_w_kg_m2']:.4f} kg m2")
    print()
    print("phase0.toml [design] suggestion (nominal, -10%/+10%):")
    for key, value in (("total_mass_kg", model["m_t_kg"]), ("center_of_mass_m", model["l_c_m"]),
                       ("pivot_inertia_kg_m2", model["I_p_kg_m2"]), ("flywheel_inertia_kg_m2", model["I_w_kg_m2"])):
        print(f"{key} = [{value * 0.9:.4g}, {value:.4g}, {value * 1.1:.4g}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd hardware/cad && python3 -m unittest tests.test_cad.MassModelTest -v`
Expected: OK. If `G_nm` exceeds 1.34, report it: the spec's lever is `rod_len` (do not change it silently; the user decides).

- [ ] **Step 5: Run `make masses` and copy the numbers into phase0.toml**

Run: `cd hardware/cad && make parts masses 2>&1 | tail -25`

Then edit `software/sim/phase0.toml` `[design]`: replace the four planning ranges with the printed suggestion lines, and change the block comment to:

```toml
# From hardware/cad (make masses), 2026-09-04. Nominal = CAD + catalogue hardware
# masses; -10 % / +10 % covers print density, screws and cable. Still no
# measured parts.
```

Run: `cd software/sim && poetry run phase0 | sed -n 1,30p && poetry run phase0-swingup --sweep --torque-limit 0.5 | grep -c PASS && poetry run python -m unittest discover -s tests 2>&1 | tail -2`
Expected: the report renders, every swing-up row is PASS (24 rows), tests OK. If `test_nominal_gravity_coefficient_matches_documented_estimate` fails, update its expected value to the new nominal `m_t * 9.81 * l_c` and note the change in the LOG entry.

- [ ] **Step 6: Write hardware/cad/README.md**

```markdown
# Mechanics — OpenSCAD

Parametric models of the reaction-wheel pendulum: wheel, rod, table-clamp
stand. Design and rationale: [docs/hardware/cad-spec.md](../../docs/hardware/cad-spec.md).
All dimensions live in `params.scad`.

## Build

```shell
make parts      # STLs into build/
make masses     # mass, centroid, inertia per part + pendulum mass model
make check      # interference check (must print "no interference")
make test       # geometry tests (python3 -m unittest, drives openscad)
openscad assembly.scad   # interactive view; set angle=... in the customizer
```

Requires OpenSCAD ≥ 2026.06 (manifold backend) and Python 3.12.

## Parts and print orientation

| Part | File / `part` | Qty | Orientation | Notes |
|---|---|---:|---|---|
| wheel quadrant | `wheel.scad` / `quadrant` | 4 | flat, rim down | 0.2 mm layers, 3 perimeters, 30 % infill |
| wheel hub plate | `wheel.scad` / `hub` | 1 | flat, nut pockets down | 100 % infill |
| beam | `rod.scad` / `beam` | 1 | web flat (X up), diagonal on the bed | 4 perimeters |
| motor flange | `rod.scad` / `flange` | 1 | on edge (X up), brim | 4 perimeters, 50 % infill |
| stand | `stand.scad` / `stand` | 1 | on its side (X up) | 5 perimeters, 40 % infill, PETG preferred |
| pad | `stand.scad` / `pad` | 1 | flat | |

## BOM (hardware)

| Item | Qty |
|---|---:|
| Steel shaft Ø8 × 120 mm | 1 |
| 608ZZ bearing 8 × 22 × 7 | 2 |
| Shaft collar Ø8 | 2 |
| M3 × 8 screws + nuts (rim laps 4, hub 8, clamp 2, lap joint 2) | 16 |
| M3 × 8 countersunk (rotor) | 3 |
| M3 × 10 (stator) | 4 |
| M2.5 × 12 screws + 20 mm standoffs (RPi / pi3hat) | 4 |
| M6 × 20 bolt + nut (tuning) | 8–12 |
| M8 × 60 thumb screw + M8 nut | 1 |
| Velcro strap 12 mm | 2 |
| mjbots mj5208, moteus r4.11, pi3hat, Raspberry Pi 4, 4S 1000 mAh LiPo | 1 each |

## Status / Next

- Verify the mj5208 stator hole pattern (`stator_pitch`, square assumed) against the mjbots 2D drawing before printing `motor_flange`.
- Confirm `bolt_fit` after the first wheel print by weighing the quadrants; adjust to hit I_w ≈ 0.003 kg m².
- Add cable routing (CAN from r4.11 to pi3hat, battery leads), a power switch pocket and an encoder seat at the axle end.
- Replace the RPi 4 + pi3hat mass estimate (85 g) with measured values.
```

- [ ] **Step 7: Update the spec table, docs index, LOG and ROADMAP**

- `docs/hardware/cad-spec.md` §5: replace the estimated table with the `make masses` output (part masses, z, and the totals line), keep the "two levers" paragraph.
- `docs/README.md` Hardware list: add `- [CAD specification](hardware/cad-spec.md) — wheel, rod and table-clamp stand for OpenSCAD; frame, parameters, parts, BOM, mass model`.
- `LOG.md`: prepend a `**04.09.2026 (3)**` entry (Russian, like its neighbours): CAD спецификация утверждена (A+C: вращающаяся ось в подшипниках стойки, струбцина на край стола), колесо Ø280 из 4 сегментов, обод 8×8 + 8 болтов M6 для инерции, батарея и RPi на оси; OpenSCAD-модели в `hardware/cad/` с тестами и проверкой пересечений; массовая модель из STL → `phase0.toml` (привести m_t, l_c, G, I_w из `make masses`); открыто: паттерн статора mj5208, тепловой тест.
- `ROADMAP.md` Phase 1 "Механика" line: append `→ CAD в OpenSCAD: [cad-spec.md](docs/hardware/cad-spec.md), `hardware/cad/` (2026-09-04); печать после проверки паттерна статора`.

- [ ] **Step 8: Full verification**

Run: `cd hardware/cad && make clean parts check test 2>&1 | tail -15 && cd ../../software/sim && poetry run python -m unittest discover -s tests 2>&1 | tail -2`
Expected: all STLs built, `check: no interference`, CAD tests OK (≥ 19 tests), sim tests OK.

- [ ] **Step 9: Commit (only if the user asked for commits)**

```bash
git add hardware/cad software/sim/phase0.toml docs/hardware/cad-spec.md docs/README.md LOG.md ROADMAP.md
git commit -m "cad: mass model fed back to phase0, README with BOM and print settings"
```

---

## Self-review notes

- Spec coverage: §2 frame → params/lib comments and all part frames; §3 → `params.scad`; §4.1 → Task 2; §4.2 → Task 3; §4.3 → Task 4; §4.4 → Task 5; §4.5 BOM → Task 7 README; §5 → Task 7 mass model and phase0 feedback; §6 → Makefile (Task 1) + check (Task 6); §7 → file structure; §8 out of scope untouched.
- Refinements versus the spec are listed under Global Constraints and written back in Task 1 Step 7 so the spec stays the source of truth.
- Names used across tasks: `plate`, `arc`, `at_circle`, `hole`, `hex_pocket`, `ycyl` (Task 1) → used in Tasks 2–6; `quadrant/hub/bolts/wheel` (Task 2) → Task 6; `beam/motor_flange/i_flanges/i_web/lap_holes` (Tasks 3–4) → Tasks 4, 6; `stand/pad` (Task 5) → Task 6; `render/fits_bed` (Task 2) → Tasks 3–7; `stl_props.props` keys (Task 1) → Tasks 2–7; `mass_model.compute` (Task 7).
