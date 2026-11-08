"""Shared look for the article figures in docs/content/figures.

One colour code for every article:

- ``MEASURED`` (blue) — bench data. When two measured series share a plot
  (the two sides of the torque fixture), the second is ``MEASURED_2``
  (orange); the pair passes the dataviz palette validator.
- ``REFERENCE`` (gray) — requirements, firmware values, theory, models.
- ``FAIL`` (red) — a region where the system does not work.

Titles state the conclusion, subtitles say what is plotted, the footer names
the run and the data file. Fonts are sized for a phone-width article column.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ZONE = "#f0efec"

MEASURED = "#2a78d6"
MEASURED_2 = "#eb6834"
MEASURED_BAND = "#cde2fb"
REFERENCE = "#52514e"
FAIL = "#d03b3b"
FAIL_BAND = "#f6d4d4"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


HEADER_IN = 0.85  # inches reserved above the axes for title and subtitle
FOOTER_IN = 0.35  # inches reserved below the axes for the source line


def header(fig, title: str, subtitle: str) -> None:
    h = fig.get_figheight()
    fig.text(0.015, 1 - 0.12 / h, title, fontsize=15, color=INK, weight="bold", va="top")
    fig.text(0.015, 1 - 0.48 / h, subtitle, fontsize=11.5, color=INK_2, va="top")


def footer(fig, text: str) -> None:
    fig.text(0.015, 0.08 / fig.get_figheight(), text, fontsize=9.5, color=MUTED, va="bottom")


def layout(fig) -> None:
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, FOOTER_IN / h, 1, 1 - HEADER_IN / h))
