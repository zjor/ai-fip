// Rail anchor: L-foot that fixes the end of a 10x10 bottom rail of the bench
// stand to the wooden base plate (PNDLM-11 coast-down / acceleration runs).
//
// The rail lies flat on the plate. Its end face has an M3 thread in the centre.
// The vertical plate of the anchor sits against that end face; a countersunk
// M3 goes through it into the rail along the rail axis. A conical relief in the foot top
// gives the head room to slide in. The foot lies on the base plate and takes
// a wood screw. Two hyperbolic side gussets sweep from
// the plate top to the foot tip.
//
// Frame: rail end face at y = 0, rail at y < 0; base plate top at z = 0;
// X across the rail. Prints as modelled: foot flat on the bed, no supports.
// Self-contained on purpose: the bench stand is not the pendulum of params.scad.

// ---- rail ----
rail = 10;              // rail cross-section (square)
rail_hole_z = rail / 2; // M3 thread sits in the centre of the end face

// ---- anchor ----
plate_t = 3;            // vertical plate thickness (Y)
foot_t = 3;             // foot thickness (Z)
foot_len = 22;          // rail face to foot tip (Y)
gusset_t = 2;           // each side gusset (X)
gusset_k = 3;           // gusset curve: 0 = straight chamfer, larger = tighter hyperbola
gusset_n = 40;          // segments along the curve
width = rail + 2 * gusset_t;

// ---- screws ----
m3_hole = 3.4;
m3_cs_d = 6.5;          // DIN 7991 countersunk head (dk 6.0) + slop; flush with the plate
relief_d = 7;           // head relief cone at the plate face (head + 0.5)
relief_len = 14;        // cone length along +Y; sets how gently the scoop rises
ws_hole = 5.2;          // wood screw shank clearance (4 mm screw)
ws_cs_d = 8;            // wood screw head countersink; 0 = none
ws_y = 14;              // wood screw centre from the rail face

$fn = 72;

module csk(d, cs_d, h) {
    // Through hole along +Z from z=0 to z=h with a 90 deg countersink opening at z=h.
    translate([0, 0, -1]) cylinder(d = d, h = h + 2);
    if (cs_d > d) {
        cs_h = (cs_d - d) / 2;
        translate([0, 0, h - cs_h]) cylinder(d1 = d, d2 = cs_d, h = cs_h);
        translate([0, 0, h - 0.01]) cylinder(d = cs_d, h = 1);
    }
}

// Hyperbola through (plate_t, rail) and (foot_len, foot_t):
// z = foot_t + (rail - foot_t) * (1 - s) / (1 + k s), s in [0, 1] along the foot.
function gusset_pt(i) = let(s = i / gusset_n)
    [plate_t + s * (foot_len - plate_t), foot_t + (rail - foot_t) * (1 - s) / (1 + gusset_k * s)];

module gusset(x0) {
    translate([x0, 0, 0]) rotate([90, 0, 90]) linear_extrude(gusset_t)
        polygon(concat([[plate_t, foot_t]], [for (i = [0 : gusset_n]) gusset_pt(i)]));
}

module rail_anchor() {
    difference() {
        union() {
            cube([width, plate_t, rail]);                         // vertical plate
            cube([width, foot_len, foot_t]);                      // foot
            gusset(0);
            gusset(width - gusset_t);
        }
        // M3 along the rail axis, countersunk on the foot side (+Y face of the plate)
        translate([width / 2, 0, rail_hole_z]) rotate([-90, 0, 0]) csk(m3_hole, m3_cs_d, plate_t);
        // head relief: cone from the plate face narrowing toward the foot tip,
        // scoops the foot top so the head can slide in along the rail axis
        translate([width / 2, plate_t - 0.01, rail_hole_z]) rotate([-90, 0, 0])
            cylinder(d1 = relief_d, d2 = 0, h = relief_len);
        // wood screw through the foot, countersunk on top
        translate([width / 2, ws_y, 0]) csk(ws_hole, ws_cs_d, foot_t);
    }
}

translate([-width / 2, 0, 0]) rail_anchor();
