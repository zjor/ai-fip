// Test disc for motor characterization (PNDLM-11): Ø150 ring on three spokes,
// bolted to the mj5208 rotor. Design: docs/hardware/specs/test-disc.md.
// Native frame: disc axis = Z, rotor face at z = 0.
include <lib.scad>

part = "disc";   // disc

ro = td_d / 2;
ri = ro - td_ring_w;

// 2D web: hub + spokes, overlapping the ring; concave corners filleted
module web_2d() {
    offset(r = -td_fillet) offset(r = td_fillet) union() {
        circle(d = td_hub_d);
        for (i = [0 : td_spokes - 1]) rotate(i * 360 / td_spokes)
            translate([0, -td_spoke_w / 2]) square([ri + td_ring_w / 2, td_spoke_w]);
        difference() { circle(r = ro); circle(r = ri); }
    }
}

// ring with a rounded rectangular cross-section
module ring() rotate_extrude() translate([ri + td_edge_r, td_edge_r])
    offset(r = td_edge_r) square([td_ring_w - 2 * td_edge_r, td_ring_h - 2 * td_edge_r]);

module disc() {
    difference() {
        union() {
            ring();
            linear_extrude(td_web_h) web_2d();
        }
        hole(td_bore_d, td_ring_h);
        at_circle(rotor_bc_d, rotor_n) {
            hole(M3, td_ring_h);
            translate([0, 0, td_web_h - td_cbore_h]) cylinder(d = td_cbore_d, h = td_ring_h);
        }
    }
}

if (part == "disc") disc();
else assert(false, str("unknown part: ", part));
