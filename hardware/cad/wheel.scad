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
        translate([0, 0, -4]) cylinder(d = 10, h = 4);                  // head
        translate([0, 0, -4]) cylinder(d = 6, h = 24);                  // shank
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
