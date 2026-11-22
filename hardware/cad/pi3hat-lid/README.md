# pi3hat lid

2D reference drawing of the pi3hat r4.5 top side, for modelling a printed lid in Onshape.

- `pi3hat_r4.5_outline.dxf` — ASCII DXF R2010, mm, top view, origin at the board's bottom-left corner.
- `make_outline_dxf.py` — generates it from the mjbots Eagle board file
  (`hw/r4.5/pi3_hat.brd`, github.com/mjbots/pi3hat, commit `12ab9af`). Usage is in the script header.

Layers:

| Layer | Content |
|---|---|
| `OUTLINE` | board edge, 56.5 × 65, R3.5 corners |
| `HOLES` | 4 × Ø2.75 mounting holes, 49 × 58 pattern (RPi standard, RP-008343-DS-1); the `.brd` has the top-right hole at x = 53, the script moves it to 52.5 |
| `POWER` | J1, J2 (XT30), L1 (power inductor, tallest part) |
| `CAN` | JC1–JC5 (JST PH 3-pin) |
| `DEBUG` | JD1–JD3 (JST ZR 6-pin), pin-header pads UT, JP6–JP8, I2C |
| `LABELS` | reference designators |

Component outlines are footprint body lines (assembly layer, or silkscreen where the
footprint has none). They are not mating-plug sizes: add clearance for the plug in CAD.
