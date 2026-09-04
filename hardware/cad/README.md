# Mechanics — OpenSCAD

Parametric models of the reaction-wheel pendulum: wheel, rod, table-clamp
stand. Design and rationale: [docs/hardware/cad-spec.md](../../docs/hardware/cad-spec.md).
All dimensions live in `params.scad`.

## Build

```shell
make parts      # STLs into build/
make masses     # mass, centroid, inertia per part + pendulum mass model
make check      # interference check (must print "check: no interference")
make test       # geometry tests (python3 -m unittest, drives openscad)
openscad assembly.scad   # interactive view; set angle=... in the customizer
```

Requires OpenSCAD ≥ 2026.06 (manifold backend) and Python 3.12 (stdlib only).
`make check` passes when OpenSCAD reports an empty top-level object: the
intersection of the pendulum at 0/90/180/270° with the stand and table is empty.

## Parts and print orientation

| Part | File / `part` | Qty | Mass (rendered) | Orientation | Notes |
|---|---|---:|---:|---|---|
| wheel quadrant | `wheel.scad` / `quadrant` | 4 | 27.7 g | flat, rim down (144 × 144 mm) | 0.2 mm layers, 3 perimeters, 30 % infill |
| wheel hub plate | `wheel.scad` / `hub` | 1 | 16.1 g | flat, nut pockets down | 100 % infill |
| beam | `rod.scad` / `beam` | 1 | 63.9 g | web flat (X up), diagonal on the bed (230 × 20 mm) | 4 perimeters |
| motor flange | `rod.scad` / `flange` | 1 | 41.7 g | on edge (X up), brim (135 × 20 mm) | 4 perimeters, 50 % infill |
| stand | `stand.scad` / `stand` | 1 | 151.6 g | on its side (X up) (98 × 74 mm) | 5 perimeters, 40 % infill, PETG preferred |
| pad | `stand.scad` / `pad` | 1 | 3.4 g | flat | |

`wheel.scad / bolts` is not a print: it models the eight M6 tuning bolts
(66.7 g steel) for the mass model.

## Mass model (`make masses`, 2026-09-04)

| Part | Mass g | z CoM mm |
|---|---:|---:|
| wheel plastic (4 quadrants + hub) | 127.0 | 250 |
| wheel bolts (8× M6×20 + nut) | 66.7 | 250 |
| beam | 63.9 | 36 |
| motor flange | 41.7 | 228 |
| mj5208 | 193.0 | 250 |
| moteus r4.11 + screws | 19.2 | 250 |
| 4S pack | 120.0 | 0 |
| RPi 4 + pi3hat + standoffs (estimate) | 85.0 | 0 |
| axle, collars, clamp screws (estimate) | 60.0 | 0 |
| **total** | **776 g** | l_c = 146 mm |

G = 1.11 Nm, I_p = 0.029 kg m², I_w = 0.0027 kg m². These are the
`[design]` values in `software/sim/phase0.toml`.

## BOM (hardware)

| Item | Qty |
|---|---:|
| Steel shaft Ø8 × 120 mm | 1 |
| 608ZZ bearing 8 × 22 × 7 | 2 |
| Shaft collar Ø8 | 2 |
| M3 × 8 screws + nuts (rim laps 4, hub 8, clamp 2, lap joint 2) | 16 |
| M3 × 8 countersunk (rotor) | 3 |
| M3 × 10 (stator) | 4 |
| M2.5 × 12 screws + 20 mm standoffs (RPi / pi3hat) | 4 |
| M6 × 20 bolt + nut (tuning) | 8–12 |
| M8 × 60 thumb screw + M8 nut | 1 |
| Velcro strap 12 mm | 2 |
| mjbots mj5208, moteus r4.11, pi3hat, Raspberry Pi 4, 4S 1000 mAh LiPo | 1 each |

## Status / Next

- Verify the mj5208 stator hole pattern (`stator_pitch`, square assumed) against the mjbots 2D drawing before printing `motor_flange`.
- Confirm `bolt_fit` after the first wheel print by weighing the quadrants; adjust to hit I_w ≈ 0.003 kg m² (rendered: 0.0027 with 8 bolts; 10 bolts give 0.0030).
- Add cable routing (CAN from r4.11 to pi3hat, battery leads), a power switch pocket and an encoder seat at the axle end.
- Replace the RPi 4 + pi3hat mass estimate (85 g) and the axle hardware estimate (60 g) with measured values.
