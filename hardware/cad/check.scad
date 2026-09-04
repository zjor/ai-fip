// Interference check: must render EMPTY. The axle is excluded (it legitimately
// passes through the stand bearings). Run: openscad -o build/check.stl check.scad
include <lib.scad>
use <assembly.scad>
use <stand.scad>

for (a = [0, 90, 180, 270]) intersection() {
    pendulum(a);
    union() { stand(); table_top(); }
}
