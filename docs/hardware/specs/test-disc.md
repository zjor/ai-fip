# Test disc — guarded inertia for motor characterization

Status: design, 2026-10-28. YouTrack: PNDLM-11 (T-005.B).
Model: `hardware/cad/test_disc.scad`, parameters in `hardware/cad/params.scad`
(block `test disc`).

## 1. Purpose

A balanced, retained, known inertia bolted to the mj5208 rotor for the
Tier B bench runs of [motor-bench.md](motor-bench.md): coast-down
loss versus speed and torque-to-acceleration checks up to 3000 rpm. It is
**not** the pendulum wheel ([cad.md](cad.md) §4.1); it exists so
the motor's own losses (bearing, cogging, iron) can be measured before the
final wheel is printed. Windage of this disc does not transfer to the Ø280
wheel and is not a measurement target.

## 2. Requirements (as agreed 2026-10-28)

1. Outer diameter 150 mm.
2. Mounts on the rotor with the 3× M3 pattern. The unit on hand has the
   **Ø17** bolt circle, confirmed with the printed fit gauge
   (`hardware/cad/rotor_gauge.scad`, 2026-10-28; drawing allows Ø16 or Ø17).
3. Three spokes from the inner mounting disc to the outer ring.
4. Smooth transitions, no sharp edges: fillets at every spoke junction,
   rounded ring edges.
5. Mass concentrated in the outer ring.
6. No bolt pockets. The experiment does not need added inertia (§5); a plain
   ring is smoother, stronger and has no bolt imbalance. More inertia, if
   ever needed, comes from a thicker ring, not from hardware.
7. Ring height sized for 3000 rpm (§4).

## 3. Geometry

| Parameter | Value | Note |
|---|---|---|
| `td_d` | 150 | outer diameter |
| `td_ring_w`, `td_ring_h` | 14, 10 | ring radial width × axial height, cross-section rounded `td_edge_r` = 2 |
| `td_hub_d`, `td_web_h` | 40, 8 | hub disc diameter; hub and spokes share one height |
| `td_spokes`, `td_spoke_w` | 3, 10 | straight spokes, every 120° |
| `td_fillet` | 6 | concave fillet radius at hub and ring junctions |
| `rotor_bc_d`, `rotor_n` | 17, 3 | from `params.scad`, shared with the wheel hub; gauge-confirmed |
| `td_bore_d` | 6.5 | centre bore clears the Ø5 × 2 mm shaft stub; Ø10 would leave too little wall to the M3 holes |
| counterbore | Ø6 × 4.5 | M3 socket heads (Ø5.5) sink below the outer face; sized for the M3 × 8 screws on hand |

Frame: disc axis = Z, rotor face at z = 0, outer face at z = `td_web_h`
(hub) or `td_ring_h` (ring). Rotor screws at the spoke angles. Screws:
3× M3 × 8 socket head with threadlocker; 3.5 mm of plate under the head
leaves 4.5 mm of thread in the rotor, 1.5 diameters. The 2 mm gauge was
tightened with the same screws, so the rotor thread is at least 6 mm deep.

Print flat on the rotor face, PLA, 0.2 mm layers, 4 perimeters, 100 % infill.
Flat printing puts the centrifugal load in the XY plane, the strong direction
of an FDM part.

## 4. Load check at 3000 rpm (ω = 314 rad/s)

Estimates, not FEA.

| Quantity | Value |
|---|---|
| Hoop stress in the ring, ρ ω² r² at r = 75 mm | ≈ 0.7 MPa (printed PLA holds 20–30 MPa) |
| Ring mass pulling on three spokes | ≈ 75 g at r ≈ 68 mm → ≈ 500 N total, ≈ 170 N per spoke |
| Spoke tension, 10 × 8 mm section | ≈ 2 MPa |
| Tip speed | 24 m/s |
| Stored energy at I ≈ 0.00038 kg·m² | ≈ 19 J |
| Radial force from 1 g of imbalance at r = 70 mm | ≈ 7 N |

Ring height is set by stiffness and flatness, not strength: 10 mm keeps the
ring stiff against wobble with only three spokes. 8 mm would also hold.
Reserve: ≥ 10× on every stress figure above.

## 5. Expected mass and inertia

From `make masses` (PLA 1.25 g/cm³): mass ≈ 100 g, I_z ≈ 0.00038 kg·m²,
ring ≈ 90 % of the inertia. This is enough for the bench:

| Run | Lighter disc (I ≈ 0.0004) |
|---|---|
| Coast-down 3000 → 0 rpm at ~10 mNm loss | ≈ 12 s, > 1000 samples at 100 Hz |
| Acceleration to 3000 rpm at 0.1 Nm | ≈ 1.3 s |

Use 0.05–0.2 Nm steps for acceleration runs: the same current range where
K_t was measured (PNDLM-9). Coast-down with and without the disc separates
the motor rotor inertia from the disc inertia; both feed the honest model.

The geometric inertia from the STL is the primary value; the acceleration run
with the measured K_t is the independent cross-check and gives the
uncertainty the task asks for. Record:
[measurements/2026-10-28-test-disc.json](../measurements/2026-10-28-test-disc.json)
(rendered values, then printed mass and measured inertia as they arrive).
After printing, weigh the disc and scale: I_z = 0.000371 × m_printed / 98.6 g.

## 6. Retention, balance and guard

- **Retained**: bolted with three M3 screws and threadlocker. Never
  press-fit, never taped.
- **Balanced**: the model is 3-fold symmetric and the rendered centroid sits
  on the axis (checked by `make test`). After printing, static-balance on a
  horizontal shaft: the disc must rest at any angle. Remove material from the
  ring outer face with a drill, never add weight.
- **Guard**: required by PNDLM-11, not a CAD part. An enclosure from stock
  material around the whole bench: plywood or MDF box, or a sturdy cardboard
  box with a plywood lid. Cables leave through a notch; the lid is closed for
  every run; nobody stands in the disc plane. It catches the disc if the
  screws let go or a spoke snaps (19 J, 24 m/s).
- **Speed limit**: set the moteus velocity limit to ≈ 320 rad/s before the
  first spin. Ramp 500 → 1000 → 1500 → 2000 → 2500 → 3000 rpm; stop at any
  step if vibration grows or the constant-speed current jumps.

## 7. Open points

- Stator thread depth for the motor flange, same source of uncertainty.
