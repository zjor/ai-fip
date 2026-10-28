// Fit gauge for the mj5208 rotor bolt pattern: a 2 mm ring with 3x M3 on
// bolt circle `bc` (drawing: Ø16 or Ø17 by batch). Print one per candidate;
// the one that takes all three screws without force is the pattern.
include <lib.scad>

bc = 17;            // bolt circle to test
gauge_h = 2;
gauge_hole = 3.2;   // tight M3 clearance: 0.1 mm slop so Ø16 and Ø17 cannot both fit
gauge_od = bc + 10;
label_depth = 0.6;

module rotor_gauge() {
    difference() {
        cylinder(d = gauge_od, h = gauge_h);
        hole(td_bore_d, gauge_h);
        at_circle(bc, rotor_n) hole(gauge_hole, gauge_h);
        // bolt circle engraved on the top face, between two holes
        rotate(60) translate([bc / 2 + 0.5, 0, gauge_h - label_depth]) rotate(-90)
            linear_extrude(label_depth + 1) text(str(bc), size = 3.5, halign = "center", valign = "center");
    }
}

rotor_gauge();
