# CAD specification — wheel, rod, stand (OpenSCAD)

Status: approved design, 2026-09-04. Source of truth for `hardware/cad/`.
Dimensions here are defaults; `hardware/cad/params.scad` is the single place
they live in code. When the two disagree, fix the one that is wrong and say so
in [project/log.md](../../project/log.md).

## 1. Purpose and inputs

Printable mechanics for the 2026 build: a reaction-wheel pendulum on a
table-edge stand, driven by an mjbots mj5208 + moteus r4.11, sensed by the IMU
on a pi3hat riding on the pendulum, powered by a 4S pack on the pivot axis.

Decisions this spec implements (see [project/log.md](../../project/log.md), 04.09.2026):

| Decision | Value |
|---|---|
| Pivot topology | rotating 8 mm axle clamped in the rod, two 608 bearings in the stand |
| Stand | C-clamp on the desk edge, cantilever bearing arm, axle perpendicular to the edge |
| Fabrication | FDM, PLA/PETG 1.25 g/cm³, bed 220 × 220 mm, parts > bed are segmented |
| Wheel | Ø280, four identical quadrants + hub plate, printed rim provides most of the inertia, M6 bolts for tuning |
| Angle sensor | IMU on the pendulum; no encoder on the stand (axle end kept free for one) |
| Electronics on the pendulum | 4S pack and RPi + pi3hat centred on the axle |
| Targets from Phase 0 | I_w ≈ 0.003 kg·m², rod 250 mm axle-to-motor-axis, G within the 0.97–1.34 Nm planning range (section 5 estimates 1.19) |

Verified motor mounting data (mjbots product page and blog, 2026-09-04):
mj5208 Ø63 × 25 mm, 193 g; rotor 3× M3 on a Ø17 bolt circle plus 2× M3 at
12 mm spacing; stator 4× M3 at 25 mm spacing plus 3× M2.5 at 32 mm. The exact
hole geometry (square vs. circle, depth, shaft protrusion) is **to verify**
against the mjbots 2D drawing before the first motor-flange print; it is
isolated in one parameter block and one part for that reason.

## 2. Coordinate frame

Right-handed, millimetres.

- Origin: axle centre, on the rod's mid-plane.
- **Z** up along the rod when the pendulum is upright.
- **Y** along the axle, positive away from the table edge.
- **X** along the table edge.

The pendulum rotates about Y. The stand occupies Y < 0 (bearing arm) and
Y ≪ 0 (clamp on the table). The axle sits `axle_height` above the table top so
the bearing arm clears the surface; the table top is the box X ∈ ℝ,
Y ≤ −`axle_overhang`, Z ∈ [−`axle_height` − `table_thickness`, −`axle_height`].

## 3. Parameters (`params.scad`)

| Name | Default | Meaning |
|---|---:|---|
| `bed` | 220 | printable square, mm; parts longer than `bed·√2 − 15` are split |
| `rho_pla` | 1.25e-3 | g/mm³ |
| `rod_len` | 250 | axle axis to motor axis |
| `wheel_d` | 280 | rim outer diameter |
| `rim_radial` | 8 | rim radial thickness (sized for inertia per gram, not stiffness; 2026-09-04) |
| `rim_axial` | 8 | rim axial width |
| `spokes` | 8 | two per quadrant |
| `spoke_w`, `spoke_t` | 4, 8 | spoke section (tangential × axial) |
| `hub_d`, `hub_t` | 60, 5 | hub plate and quadrant flange |
| `pocket_n`, `pocket_bolt` | 16, M6 | tuning holes, every 22.5° starting at 11.25°, axial through the rim, plain (nut and head clamp the rim faces) |
| `motor_d`, `motor_len` | 63, 25 | mj5208 |
| `rotor_bc_d`, `rotor_n` | 17, 3 | rotor pattern, M3 |
| `stator_pitch`, `stator_n` | 25, 4 | stator pattern, M3, square — **to verify** |
| `axle_d`, `axle_len` | 8, 120 | steel shaft |
| `bearing` | 8 × 22 × 7 | 608 |
| `bearing_spacing` | 40 | between bearing centres |
| `axle_overhang` | 60 | table edge to rod mid-plane |
| `axle_height` | 30 | table top to axle centre |
| `table_thickness` | 18–40 | clamp range |
| `battery` | 75 × 35 × 25 | 4S 1000 mAh class, on the table side of the rod |
| `hat_holes` | 58 × 49, M2.5 | RPi/pi3hat pattern, on the outer side of the rod |
| `beam_w`, `beam_h`, `flange_t`, `web_t` | 20, 20, 3, 3 | rod I-beam section |
| `clearance` | 0.3 | print fit for lap joints and pockets |
| `M3`, `M2.5`, `M6`, `M8` | 3.4, 2.8, 6.4, 8.4 | through-hole diameters |

## 4. Parts

### 4.1 Wheel (`wheel.scad`)

Five prints: four identical **quadrants** and one **hub plate**.

- Quadrant: a 90° rim arc (r from `wheel_d/2 − rim_radial` to `wheel_d/2`,
  width `rim_axial`), two spokes at 22.5° and 67.5°, and a 90° sector of the
  hub flange (r ≤ `hub_d/2`, thickness `hub_t`). Rim ends carry a half-lap
  joint (5 mm long, half the rim width) with one M3 through-hole; flange
  sector has two M3 holes on a Ø45 circle matching the hub plate.
- Hub plate: Ø`hub_d` × `hub_t` disc with the rotor pattern (3× M3 on Ø17,
  countersunk from the motor side), a Ø10 centre bore for the rotor boss,
  and 8× M3 on Ø45 for the quadrants.
- Tuning holes: `pocket_n` axial through-holes Ø`M6` in the rim at
  r = `wheel_d/2 − rim_radial/2`, every 22.5° starting at 11.25° so no hole
  lands on a lap seam or a spoke; no recess (the M6 nut overhangs the 8 mm rim
  by 1 mm per side, which is harmless). Bolts are always fitted in
  diametrically opposite pairs; the standard fit is 8 bolts every 45°.
- Lightening: none beyond the spoke design; the quadrant's rim and spokes are
  already the minimum structure.

Estimated: rim 68 g, spokes 31 g, flanges + hub plate 34 g → **~135 g
plastic, I_w ≈ 0.0015 kg·m² empty**; each M6 × 20 bolt + nut (≈ 10 g) at
r = 136 mm adds ≈ 0.000185 kg·m². Eight bolts (80 g) reach 0.0030, so the
standard wheel is ≈ 215 g. The rim was deliberately thinned from 12 × 10 mm:
plastic at the rim gives 0.0185 kg·m² per kg, a bolt at the rim 0.0185 too,
but spokes and hub give almost nothing, so the structure is kept minimal and
the inertia is bought with bolts where it can be tuned.

The wheel bolts to the rotor. The rotor's second pattern (2× M3 at 12 mm) is
left unused.

### 4.2 Rod: beam (`rod.scad`, module `beam`)

One print, laid diagonally on the bed. From the axle clamp at Z = 0 to the
lap joint ending at Z = `rod_len − 60` (bed fit).

- Section: I-beam, `beam_w` × `beam_h`, flanges `flange_t`, web `web_t`.
- Web cutouts: elliptical holes, 8 × 30 mm, centred at Z = 65, 100, 135;
  none in the tray zone or near joints.
- Axle clamp at Z = 0: a split clamp block 30 × 30 × 20 mm around Ø`axle_d`,
  slot 1.5 mm, two M3 clamping screws across the slot. The axle is also
  cross-drilled and pinned optionally (hole provided, Ø2).
- Tray zone Z ∈ [−45, 45] (the battery is centred on the axle):
  - table side (−Y): a shelf 80 × 40 mm with two 12 mm strap slots for the
    4S pack; pack outer face at Y = −(beam_h/2 + 25).
  - outer side (+Y): four M2.5 bosses on the 58 × 49 HAT pattern, 6 mm high,
    HAT long axis along Z, bosses at Z = ±29 so the board is centred on the
    axle.
- Lap joint at the top: the beam ends in a half-thickness tongue 40 mm long
  with two M3 through-holes 25 mm apart, mating the flange plate's groove.

Estimated 60 g.

### 4.3 Rod: motor flange (`rod.scad`, module `motor_flange`)

One print. A plate Ø70 × 6 mm carrying the stator pattern (4× M3 on the
`stator_pitch` square, counterbored), a Ø12 centre hole for the rotor
shaft/magnet side, and a 40 mm groove continuing the I-beam down to overlap
the beam's tongue (two M3 through-holes). The motor sits on the +Y face; the
wheel therefore spins on the outer side, away from the table.

Estimated 25 g. Reprint this part alone once the stator pattern is verified.

### 4.4 Stand (`stand.scad`)

One print (two if `bed` forbids), PLA at ≥ 4 perimeters, solid where loaded.

- C-clamp: throat depth 60 mm, opening 18–40 mm, jaw 40 mm wide; the fixed
  jaw is the top, the moving jaw is an M8 thumb screw through a printed
  M8 nut pocket in the lower jaw, with a 25 mm pad. Clamp body wall 8 mm.
- Cantilever bearing arm: rises `axle_height` from the clamp's top jaw and
  extends along +Y so that the two 608 pockets (Ø22 +0.2, 7 deep, with a 1 mm
  shoulder and an inner Ø17 relief) sit at Y = −(`axle_overhang` − 10) − 40
  and Y = −(`axle_overhang` − 10), i.e. at −90 and −50 with the defaults. The
  arm is a 40 × 40 box beam with 6 mm walls and diagonal ribs, its underside
  ≥ 5 mm above the table top; the axle passes through both bearings with a
  printed spacer between and a shaft collar outboard of each bearing.
- The axle end at −Y protrudes 8 mm past the inner bearing and is left free
  (future encoder magnet).
- Clearance requirements: rod mid-plane at Y = 0 is `axle_overhang` from the
  table edge; the battery's outer face at Y = −35 leaves ≥ 20 mm to the edge;
  the wheel and motor are at Y > 0 and clear everything.

Estimated 180 g. Weight does not matter here; stiffness does.

### 4.5 Axle and hardware (BOM in `hardware/cad/README.md`)

| Item | Qty |
|---|---:|
| Steel shaft Ø8 × 120 mm (h6) | 1 |
| 608ZZ bearing 8 × 22 × 7 | 2 |
| Shaft collar Ø8 | 2 |
| M3 × 8 screws (wheel joints, clamp, lap joint) | 24 |
| M3 × 8 countersunk (rotor to hub plate) | 3 |
| M3 × 10 (stator to flange) | 4 |
| M2.5 × 12 + standoffs (RPi/pi3hat) | 4 |
| M6 × 20 bolt + nut (tuning) | 8–12 |
| M8 × 60 thumb screw + nut | 1 |
| Velcro strap 12 mm | 2 |

## 5. Mass model and feedback to Phase 0

`make masses` computes each printed part's volume, centroid and inertia from
its STL and combines them with catalogue hardware masses. Rendered on
2026-09-04 (`hardware/cad/README.md` has the per-part table):

| Part | Mass | z of CoM | Contribution to G (Nm) |
|---|---:|---:|---:|
| wheel plastic + 8 tuning bolts | 194 g | 250 | 0.48 |
| mj5208 + r4.11 + screws | 212 g | 250 | 0.52 |
| motor flange | 42 g | 228 | 0.09 |
| beam | 64 g | 36 | 0.02 |
| axle, collars, clamp screws (estimate) | 60 g | 0 | 0 |
| 4S pack | 120 g | 0 | 0 |
| RPi 4 + pi3hat + standoffs (estimate) | 85 g | 0 | 0 |
| **total** | **0.776 kg** | l_c = 0.146 m | **G = 1.11 Nm** |

I_p = 0.029 kg m² (beam as a slender rod, everything else as point masses),
I_w = 0.0027 kg m² with 8 bolts (10 bolts give 0.0030). This sits between the
Phase 0 nominal (0.97 Nm) and pessimistic (1.34 Nm) corners, where the r4.11
has ≥ 3.7× margin at 20° and passes torque-capped swing-up at every point. The
uncapped direct lift at 12 V reaches 71–79 % of no-load wheel speed at
I_w 0.0027, above the 70 % limit: fit 10 bolts (0.0030) or cap swing-up torque
at 0.5 Nm in software. Two levers if the printed parts
come out heavier: shorten `rod_len` (G ∝ rod length), or use an RPi Zero 2 W.
These numbers are the `[design]` block of `software/sim/phase0.toml`
(nominal, −10 % / +10 %); `poetry run phase0 && poetry run phase0-swingup
--sweep` must still pass after any change.

## 6. Build and checks (`Makefile`)

- `make parts`: renders `wheel_quadrant.stl`, `wheel_hub.stl`, `beam.stl`,
  `motor_flange.stl`, `stand.stl` headlessly
  (`openscad --backend=manifold -o <out> -D part="<name>" <file>`).
- `make masses`: renders each part and computes volume, mass, centroid and
  moment of inertia from the STL with `tools/stl_props.py`;
  `tools/mass_model.py` combines them with the hardware masses into m_t, l_c,
  I_p and I_w.
- `make check`: renders `check.scad`, which computes the intersection of the
  pendulum assembly at 0°, 90°, 180°, 270° with the stand and with the table
  box. The Makefile fails if the resulting STL contains any facet.
- `make view`: renders `assembly.stl` for visual inspection; the assembly
  file also shows the pendulum at a parametric angle.

Each part also fits the bed: `lib.scad` asserts the part's bounding box
against `bed` (diagonal allowed) at render time.

## 7. Files

```
hardware/cad/
  params.scad     all dimensions, materials, hardware sizes
  lib.scad        report(), screw holes, bed assertion, hex nut pocket
  wheel.scad      quadrant, hub_plate; part selector
  rod.scad        beam, motor_flange; part selector
  stand.scad      clamp + bearing arm
  assembly.scad   everything placed, pendulum angle parameter
  check.scad      interference checks
  Makefile        parts, masses, check, view
  README.md       how to build, BOM, print settings, project-task links
```

## 8. Out of scope

Wiring, the CAN cable routing on the rod, the power switch, a battery
charger port, cable strain relief, and an encoder seat. These are Phase 1
detailing after the first print reveals what is missing.
