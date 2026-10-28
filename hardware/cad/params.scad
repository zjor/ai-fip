// All dimensions in mm, densities in g/mm^3. Source of truth: docs/hardware/specs/cad.md.
// Frame: origin at the axle centre on the rod mid-plane; Z up the upright rod;
// Y along the axle away from the table edge; X along the table edge.

bed = 220;
rho_pla = 1.25e-3;
rho_steel = 7.85e-3;

rod_len = 250;                 // axle axis to motor axis

// ---- wheel (native axis Z) ----
wheel_d = 280;
rim_radial = 8;
rim_axial = 8;
spokes = 8;                    // two per quadrant
spoke_w = 4;
spoke_t = 8;
hub_d = 60;
hub_t = 5;
hub_bc_d = 45;                 // quadrant flange screws
hub_bore_d = 10;
flange_ri = 13;                // quadrant flange sector inner radius (clears rotor screw heads)
pocket_n = 16;                 // tuning holes every 22.5 deg, first at 11.25 deg
bolt_fit = 8;                  // standard number of M6 tuning bolts
lap_len = 8;                   // rim half-lap length along the rim

// ---- motor mj5208 ----
motor_d = 63;
motor_len = 25;
rotor_bc_d = 17;               // Ø17 confirmed with rotor_gauge.scad 2026-10-28 (drawing: Ø16 or Ø17 by batch)
rotor_n = 3;
stator_bc_d = 25;              // 4x M3 on a Ø25 bolt circle (mjbots mj5208 drawing rev.3 2025-07-26); 2nd pattern 3x M2.5 on Ø32 unused
stator_n = 4;
flange_d = 70;
flange_plate_t = 6;

// ---- test disc (PNDLM-11, docs/hardware/specs/test-disc.md) ----
td_d = 150;                    // outer diameter
td_ring_w = 14;                // ring radial width
td_ring_h = 10;                // ring axial height
td_edge_r = 2;                 // ring cross-section corner radius
td_hub_d = 40;                 // hub disc diameter
td_web_h = 8;                  // hub and spoke height
td_spokes = 3;
td_spoke_w = 10;
td_fillet = 6;                 // concave fillets at spoke junctions
td_bore_d = 6.5;               // clears the Ø5 x 2 mm shaft stub; Ø10 left too little wall to the M3 holes
td_cbore_d = 6;                // M3 socket head (Ø5.5) counterbore
td_cbore_h = 4.5;              // leaves 3.5 mm under the head: M3x8 engages 4.5 mm of rotor thread

// ---- pivot ----
axle_d = 8;
axle_len = 120;
bearing_od = 22;
bearing_w = 7;
bearing_spacing = 40;
axle_overhang = 60;            // table edge to rod mid-plane
axle_height = 30;              // table top to axle centre
table_min = 18;
table_max = 40;

// ---- rod ----
beam_w = 20;                   // X
beam_h = 20;                   // Y
flange_t = 3;
web_t = 3;
tongue_len = 40;
beam_end = rod_len - 60;       // tongue ends here so the beam fits the bed diagonally
clamp_size = [30, 20, 30];     // X, Y, Z around the axle
battery = [35, 25, 75];        // X, Y, Z as mounted on the -Y face
hat = [49, 58];                // RPi/pi3hat hole pattern: X, Z
hat_hole = 2.8;

// ---- stand ----
jaw_w = 40;
clamp_wall = 8;
throat = 60;
arm_w = 40;

// ---- fits and hardware ----
clearance = 0.3;
M3 = 3.4;
M6 = 6.4;
M8 = 8.4;
nut_m3_af = 5.5;
nut_m3_h = 2.4;
nut_m8_af = 13;
nut_m8_h = 6.5;
$fn = 72;
