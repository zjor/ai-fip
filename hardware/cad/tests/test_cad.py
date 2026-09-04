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
        if "top level object is empty" in result.stderr:
            # OpenSCAD refuses to export empty geometry; that is a valid result
            # for the interference check.
            out.write_text("solid empty\nendsolid empty\n")
        else:
            raise AssertionError(result.stderr)
    warnings = [
        line for line in result.stderr.splitlines()
        if "WARNING" in line and "top level object is empty" not in line
    ]
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


class MassModelTest(unittest.TestCase):
    def test_pendulum_totals_are_inside_phase0_planning_range(self):
        import mass_model
        render_part = {
            "wheel_quadrant": ("wheel", "quadrant"), "wheel_hub": ("wheel", "hub"),
            "wheel_bolts": ("wheel", "bolts"), "beam": ("rod", "beam"),
            "motor_flange": ("rod", "flange"),
        }
        for part, args in render_part.items():
            _, path = render(*args)
            path.replace(BUILD / f"{part}.stl")
        model = mass_model.compute(BUILD)
        self.assertGreater(model["m_t_kg"], 0.6)
        self.assertLess(model["m_t_kg"], 0.9)
        self.assertGreater(model["G_nm"], 0.97)
        self.assertLess(model["G_nm"], 1.34)
        self.assertGreater(model["I_w_kg_m2"], 0.0025)
        self.assertLess(model["I_w_kg_m2"], 0.0035)


if __name__ == "__main__":
    unittest.main()
