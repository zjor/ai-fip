"""Generate a 2D reference DXF of the pi3hat r4.5 top side from the mjbots Eagle board file.

Usage:
    git clone --depth 1 https://github.com/mjbots/pi3hat.git /tmp/pi3hat
    python make_outline_dxf.py /tmp/pi3hat/hw/r4.5/pi3_hat.brd pi3hat_r4.5_outline.dxf

Needs: ezdxf. Units: mm. Origin: board corner at bottom-left, top view.
"""
import math
import sys
import xml.etree.ElementTree as ET

import ezdxf

# reference designator -> DXF layer
PARTS = {
    "J1": "POWER", "J2": "POWER", "L1": "POWER",
    "JC1": "CAN", "JC2": "CAN", "JC3": "CAN", "JC4": "CAN", "JC5": "CAN",
    "JD1": "DEBUG", "JD2": "DEBUG", "JD3": "DEBUG",
    "UT": "DEBUG", "JP6": "DEBUG", "JP7": "DEBUG", "JP8": "DEBUG", "I2C": "DEBUG",
}
LAYERS = {  # name: ACI colour
    "OUTLINE": 7, "HOLES": 1, "POWER": 3, "CAN": 5, "DEBUG": 8, "LABELS": 2,
}
EAGLE_DOCU, EAGLE_PLACE, EAGLE_KEEPOUT = "51", "21", "25"
# The r4.5 .brd puts the top-right hole at x=53, 0.5 mm off the RPi 49 x 58 pattern
# (RP-008343-DS-1). Snap it to the Pi pattern so both short edges are 49.0.
HOLE_FIX = {(53.0, 61.5): (52.5, 61.5)}


def arc_from_chord(x1, y1, x2, y2, curve):
    """Eagle arc (chord + signed sweep in degrees, CCW positive) -> centre, radius, start, end (CCW, deg)."""
    a = math.radians(curve)
    dx, dy = x2 - x1, y2 - y1
    chord = math.hypot(dx, dy)
    r = chord / (2 * math.sin(abs(a) / 2))
    h = math.sqrt(max(r * r - (chord / 2) ** 2, 0.0))
    nx, ny = -dy / chord, dx / chord  # left normal of the chord
    s = 1 if curve > 0 else -1
    cx, cy = (x1 + x2) / 2 + s * nx * h, (y1 + y2) / 2 + s * ny * h
    t1 = math.degrees(math.atan2(y1 - cy, x1 - cx))
    t2 = math.degrees(math.atan2(y2 - cy, x2 - cx))
    return (cx, cy), r, (t1, t2) if curve > 0 else (t2, t1)


def package_wires(pkg):
    """Body outline wires: assembly (tDocu) if present, else silkscreen + keepout."""
    docu = [w for w in pkg.iter("wire") if w.get("layer") == EAGLE_DOCU]
    if docu:
        return docu
    return [w for w in pkg.iter("wire") if w.get("layer") in (EAGLE_PLACE, EAGLE_KEEPOUT)]


def main(brd, out):
    root = ET.parse(brd).getroot()
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    doc.header["$INSUNITS"] = 4
    for name, color in LAYERS.items():
        doc.layers.add(name, color=color)
    msp = doc.modelspace()

    plain = root.find(".//plain")
    for w in plain.iter("wire"):
        if w.get("layer") != "20":
            continue
        x1, y1, x2, y2 = (float(w.get(k)) for k in ("x1", "y1", "x2", "y2"))
        if w.get("curve"):
            c, r, (sa, ea) = arc_from_chord(x1, y1, x2, y2, float(w.get("curve")))
            msp.add_arc(c, r, sa, ea, dxfattribs={"layer": "OUTLINE"})
        else:
            msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": "OUTLINE"})
    for h in plain.iter("hole"):
        x, y, d = float(h.get("x")), float(h.get("y")), float(h.get("drill"))
        x, y = HOLE_FIX.get((x, y), (x, y))
        msp.add_circle((x, y), d / 2, dxfattribs={"layer": "HOLES"})
        msp.add_point((x, y), dxfattribs={"layer": "HOLES"})

    pkgs = {(lib.get("name"), p.get("name")): p for lib in root.iter("library") for p in lib.iter("package")}
    for e in root.find(".//elements"):
        name = e.get("name")
        if name not in PARTS:
            continue
        rot = e.get("rot") or "R0"
        assert not rot.startswith("M"), f"{name} is on the bottom side"
        ang = math.radians(float(rot.lstrip("R")))
        ex, ey = float(e.get("x")), float(e.get("y"))
        ca, sa = math.cos(ang), math.sin(ang)

        def tf(x, y):
            return ex + x * ca - y * sa, ey + x * sa + y * ca

        pkg = pkgs[(e.get("library"), e.get("package"))]
        wires = package_wires(pkg)
        if not wires:  # bare pin header: draw pad envelope
            pads = [(float(p.get("x")), float(p.get("y"))) for p in pkg.iter("pad")]
            xs, ys = [p[0] for p in pads], [p[1] for p in pads]
            m = 1.27
            box = [(min(xs) - m, min(ys) - m), (max(xs) + m, min(ys) - m),
                   (max(xs) + m, max(ys) + m), (min(xs) - m, max(ys) + m)]
            msp.add_lwpolyline([tf(*p) for p in box], close=True, dxfattribs={"layer": PARTS[name]})
        for w in wires:
            p1 = tf(float(w.get("x1")), float(w.get("y1")))
            p2 = tf(float(w.get("x2")), float(w.get("y2")))
            msp.add_line(p1, p2, dxfattribs={"layer": PARTS[name]})
        msp.add_text(name, height=1.2, dxfattribs={"layer": "LABELS"}).set_placement(
            (ex, ey), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)

    doc.saveas(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
