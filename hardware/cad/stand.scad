// Table-edge C-clamp with a cantilever bearing arm. Native frame = pendulum frame.
include <lib.scad>

part = "stand";   // stand | pad

edge = -axle_overhang;                      // table edge plane (Y)
top = -axle_height;                         // table top plane (Z)
y_out = -(axle_overhang - 10);              // outboard bearing centre (-50)
y_in = y_out - bearing_spacing;             // inboard bearing centre (-90)
arm_y0 = y_in - bearing_w / 2;              // -93.5
arm_y1 = y_out + bearing_w / 2;             // -46.5
lower_jaw_z = top - table_max - clamp_wall; // -78

module stand() {
    difference() {
        union() {
            plate(-jaw_w / 2, jaw_w / 2, edge - throat, edge, top, top + clamp_wall);                  // top jaw
            plate(-jaw_w / 2, jaw_w / 2, edge, edge + clamp_wall, lower_jaw_z, top + clamp_wall);     // back
            plate(-jaw_w / 2, jaw_w / 2, edge - throat, edge, lower_jaw_z, lower_jaw_z + clamp_wall); // lower jaw
            plate(-arm_w / 2, arm_w / 2, arm_y0, arm_y1, top + clamp_wall, 20);                       // bearing arm
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
