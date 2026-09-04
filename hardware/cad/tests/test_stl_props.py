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
