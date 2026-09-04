#!/usr/bin/env python3
"""Volume, mass, centroid, bounding box and moment of inertia of closed STL meshes.

Uses the divergence theorem over signed tetrahedra (origin, triangle), so it
works for any closed, consistently oriented mesh, including several bodies.
The moment of inertia is about the chosen coordinate axis through the origin.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

Vec = tuple[float, float, float]
Tri = tuple[Vec, Vec, Vec]


def read_stl(path: str | Path) -> list[Tri]:
    data = Path(path).read_bytes()
    is_ascii = data[:5] == b"solid" and (b"facet" in data[:2000] or len(data) < 84)
    if is_ascii:
        tris: list[Tri] = []
        verts: list[Vec] = []
        for line in data.decode("ascii", "ignore").splitlines():
            parts = line.split()
            if parts and parts[0] == "vertex":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(verts) == 3:
                    tris.append((verts[0], verts[1], verts[2]))
                    verts = []
        return tris
    count = struct.unpack("<I", data[80:84])[0]
    tris = []
    for i in range(count):
        offset = 84 + i * 50
        v = struct.unpack("<12f", data[offset : offset + 48])
        tris.append(((v[3], v[4], v[5]), (v[6], v[7], v[8]), (v[9], v[10], v[11])))
    return tris


def _det(a: Vec, b: Vec, c: Vec) -> float:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def props(tris: list[Tri], axis: str = "z") -> dict:
    axis_index = "xyz".index(axis)
    others = [k for k in range(3) if k != axis_index]
    volume = 0.0
    first = [0.0, 0.0, 0.0]
    i_axis = 0.0
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for a, b, c in tris:
        v = _det(a, b, c) / 6.0
        volume += v
        for k in range(3):
            first[k] += v * (a[k] + b[k] + c[k]) / 4.0
        # For a tetrahedron with one vertex at the origin:
        # integral of x^2 dV = V/20 * (sum x_i^2 + (sum x_i)^2), i over the 3 vertices.
        s2 = sum(p[k] ** 2 for p in (a, b, c) for k in others)
        s1 = sum((a[k] + b[k] + c[k]) ** 2 for k in others)
        i_axis += v / 20.0 * (s2 + s1)
        for p in (a, b, c):
            for k in range(3):
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])
    if not tris:
        lo = [0.0, 0.0, 0.0]
        hi = [0.0, 0.0, 0.0]
    centroid = tuple(f / volume for f in first) if volume else (0.0, 0.0, 0.0)
    size = tuple(round(h - l, 6) for l, h in zip(lo, hi))
    return {
        "volume_mm3": volume,
        "centroid": centroid,
        "bbox_min": tuple(lo),
        "bbox_max": tuple(hi),
        "size": size,
        "i_axis_mm5": i_axis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--density", type=float, default=1.25e-3, help="g/mm^3")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    args = parser.parse_args()
    for path in args.files:
        p = props(read_stl(path), args.axis)
        mass = p["volume_mm3"] * args.density
        inertia_kg_m2 = p["i_axis_mm5"] * args.density * 1e-9
        c = p["centroid"]
        s = p["size"]
        print(
            f"{Path(path).stem}: volume_mm3={p['volume_mm3']:.1f} mass_g={mass:.3f} "
            f"centroid=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) size=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f}) "
            f"I_{args.axis}_kg_m2={inertia_kg_m2:.6f}"
        )


if __name__ == "__main__":
    main()
