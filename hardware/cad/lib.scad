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
