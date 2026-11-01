# Rail anchor

L-foot that fixes the end of a 10x10 bottom rail of the bench stand to the
wooden base plate, so the stand does not walk during the coast-down and
acceleration runs of [PNDLM-11](https://zjor.youtrack.cloud/issue/PNDLM-11).

The rail lies flat on the plate with no gap. Its end face carries an M3 thread
in the centre. The anchor's vertical plate sits against that end face; a
countersunk M3 goes through it into the rail along the rail axis. The foot
lies on the base plate and takes a wood screw. Two side gussets sweep in a
hyperbolic curve from the plate top to the foot tip. One anchor per rail end.

## Dimensions (defaults in `rail_anchor.scad`)

| Feature | Value |
|---|---:|
| vertical plate (W × T × H) | 14 × 3 × 10 mm, flush with the rail top |
| foot (W × L × T) | 14 × 22 × 3 mm, length from the rail face |
| gussets | 2 mm thick, both sides, hyperbola from the plate top to the foot tip (`gusset_k = 3`, 0 = straight) |
| M3 hole | Ø3.4, centre 5 mm above the plate, 90° countersink Ø6.5 on the foot side |
| head relief | cone Ø7 at the plate face, 14 mm long, coaxial with the M3; scoops the foot top to 1.5 mm at the plate |
| wood screw hole | Ø5.2, centre 14 mm from the rail face, 90° countersink Ø8 on top |
| mass (PLA, rendered) | 1.5 g |

## Screws

- M3 countersunk (DIN 7991 / ISO 10642), length 3 mm plate + rail thread depth
  minus 1 mm. Slide the head in along the relief scoop; it seats itself in the
  countersink. A socket-head M3 fits only just, so countersunk is the choice.
- Drive the M3 with an L-shaped hex key (2 mm).
- Wood screw 4 mm, countersunk head, length to suit the plate.

## Build and print

```shell
make build/rail_anchor.stl   # from hardware/cad
make test                    # RailAnchorTest checks envelope, holes and mass
```

Print as modelled: foot on the bed, no supports. 0.2 mm layers, 4 perimeters,
PETG preferred. The countersinks print as 45° overhangs.
