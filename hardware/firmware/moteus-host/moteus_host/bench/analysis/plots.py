"""Dependency-free SVG line plots for bench analysis."""

from __future__ import annotations

import html
from pathlib import Path
from typing import NamedTuple


class Series(NamedTuple):
    label: str
    color: str
    points: list[tuple[float, float]]


def write_line_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[Series],
) -> None:
    width, height = 960, 560
    left, right, top, bottom = 90, 30, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    points = [point for item in series for point in item.points]
    if not points:
        raise ValueError("plot requires at least one point")
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_min == y_max:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    y_padding = 0.05 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    output = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#222}'
        '.tick{font-size:12px}.label{font-size:14px}.title{font-size:20px;'
        'font-weight:600}.legend{font-size:13px}</style>',
        f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">'
        f'{html.escape(title)}</text>',
    ]
    for index in range(6):
        fraction = index / 5
        x = left + fraction * plot_width
        y = top + fraction * plot_height
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_max - fraction * (y_max - y_min)
        output.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                f'y2="{top + plot_height}" stroke="#e5e7eb"/>',
                f'<text class="tick" x="{x:.1f}" y="{top + plot_height + 22}" '
                f'text-anchor="middle">{x_value:.3g}</text>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
                f'y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" '
                f'text-anchor="end">{y_value:.3g}</text>',
            ]
        )
    output.extend(
        [
            f'<rect x="{left}" y="{top}" width="{plot_width}" '
            f'height="{plot_height}" fill="none" stroke="#374151"/>',
            f'<text class="label" x="{left + plot_width / 2}" y="{height - 20}" '
            f'text-anchor="middle">{html.escape(x_label)}</text>',
            f'<text class="label" x="22" y="{top + plot_height / 2}" '
            f'text-anchor="middle" transform="rotate(-90 22 '
            f'{top + plot_height / 2})">{html.escape(y_label)}</text>',
        ]
    )
    for index, item in enumerate(series):
        coordinates = " ".join(
            f"{sx(x):.2f},{sy(y):.2f}" for x, y in item.points
        )
        output.append(
            f'<polyline points="{coordinates}" fill="none" '
            f'stroke="{item.color}" stroke-width="1.7" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend_x = left + 12 + (index % 3) * 240
        legend_y = top + 20 + (index // 3) * 20
        output.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" '
                f'x2="{legend_x + 24}" y2="{legend_y}" '
                f'stroke="{item.color}" stroke-width="2.5"/>',
                f'<text class="legend" x="{legend_x + 31}" '
                f'y="{legend_y + 4}">{html.escape(item.label)}</text>',
            ]
        )
    output.append("</svg>\n")
    path.write_text("\n".join(output))
