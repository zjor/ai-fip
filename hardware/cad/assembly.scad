// Full pendulum on its stand. angle = pendulum angle from upright about Y (deg).
include <lib.scad>
use <wheel.scad>
use <rod.scad>
use <stand.scad>

angle = 0;
show_table = true;

module motor() translate([0, 0, rod_len]) ycyl(motor_d, motor_len, flange_plate_t / 2);

// wheel native Z -> pendulum +Y, hub plate against the rotor face
module wheel_asm() translate([0, flange_plate_t / 2 + motor_len, rod_len]) rotate([-90, 0, 0]) {
    wheel();
    translate([0, 0, hub_t]) bolts();
}

module pendulum(a = angle) rotate([0, a, 0]) {
    beam();
    motor_flange();
    motor();
    wheel_asm();
}

module axle() ycyl(axle_d, axle_len, -105);

module table_top() translate([-200, -axle_overhang - 400, -axle_height - 25]) cube([400, 400, 25]);

stand();
axle();
pendulum();
if (show_table) %table_top();
