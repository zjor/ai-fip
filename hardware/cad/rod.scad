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

if (part == "beam") beam();
else if (part == "flange") motor_flange();
else assert(false, str("unknown part: ", part));
