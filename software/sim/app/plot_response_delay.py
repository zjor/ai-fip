"""Plot the actuator response-delay figures for the PNDLM-8 article draft.

Usage, from software/sim:

    poetry run python -m app.plot_response_delay pulse \
        ../../docs/hardware/measurements/20261025-response-timing-0020-telemetry.csv \
        ../../docs/hardware/measurements/20261025-response-timing-0020.json \
        0 ../../docs/articles/figures/2026-11-08-response-delay-pulse.png

    poetry run python -m app.plot_response_delay scale \
        ../../docs/articles/figures/2026-11-08-response-delay-scale.png

``pulse`` (figure A): one torque pulse against time from the host request,
with every host sample, the noise thresholds and the detection times from the
derived analysis JSON. The telemetry CSV is a column extract of the git-ignored
raw ``telemetry.csv`` of run ``20261025T115034Z_response-timing`` (all 330
rows); the JSON records the raw file's SHA-256.

``scale`` (figure B): measured delay bounds next to the requirement, the
simulated LQR limit, textbook limits and the fall time, on a log time axis.
Values and sources are listed in ``SCALE_*``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from matplotlib.ticker import FixedLocator, NullLocator

from app.plot_style import (
    FAIL, FAIL_BAND, INK, INK_2, MEASURED, MEASURED_BAND, MUTED, REFERENCE, ZONE,
    footer, header, layout, plt, style,
)


def plot_pulse(telemetry: Path, analysis_json: Path, trial: int, out: str) -> None:
    analysis = json.load(open(analysis_json))
    run_id = analysis["source"]["run_id"]
    result = next(t for t in analysis["trials"] if t["trial_id"] == trial)
    rows = [r for r in csv.DictReader(open(telemetry))
            if int(r["trial_id"]) == trial]
    pulse = next(r for r in rows if r["experiment_phase"] == "pulse")
    t0 = float(pulse["request_time_s"])
    torque_cmd = float(pulse["feedforward_torque_nm"])
    timeout_ms = float(pulse["watchdog_timeout_s"]) * 1000

    def t_ms(r):
        return (float(r["response_time_s"]) - t0) * 1000

    window = [r for r in rows if -20 <= t_ms(r) <= 45]
    sig = result["signals"]
    cur, pos = sig["current_a"], sig["position_rev"]
    first = cur["first_response_ms"]
    confirm = cur["confirmation_response_ms"]
    reply = t_ms(pulse)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax in axes:
        style(ax)
        ax.axvspan(0, timeout_ms, color=MEASURED_BAND, alpha=0.5, lw=0)
        for x in (first, confirm):
            ax.axvline(x, color=MEASURED, lw=1.2, ls=":")

    # 1. Torque: command vs what the controller reports.
    ax = axes[0]
    ax.step([-20, 0, timeout_ms, 45], [0, torque_cmd * 1000, 0, 0], where="post",
            color=REFERENCE, lw=2, ls="--", label="команда хоста")
    ax.plot([t_ms(r) for r in window], [float(r["torque_nm"]) * 1000 for r in window],
            color=INK, lw=1.5, marker="o", ms=5, label="момент по данным драйвера")
    ax.set_ylabel("момент, mNm")
    ax.set_ylim(-5, torque_cmd * 1000 * 1.45)
    ax.legend(frameon=False, loc="upper right")
    ax.text(timeout_ms / 2, torque_cmd * 1000 * 1.08,
            f"импульс {timeout_ms:.0f} мс (длину задаёт watchdog)",
            ha="center", va="bottom", fontsize=10.5, color=INK_2)
    ax.text(26, torque_cmd * 1000 * 0.5,
            "ниже команды: драйвер\nсчитает момент по току,\n"
            "а к току добавлена\nкомпенсация cogging",
            ha="left", va="center", fontsize=10.5, color=INK_2)

    # 2. q-current change from the at-rest median; mode 11 is after the watchdog.
    ax = axes[1]
    y_lo, y_hi = -0.45, 0.95
    active = [r for r in window if r["mode"] != "11"]
    after = [r for r in window if r["mode"] == "11"]
    ax.axhspan(-cur["threshold"], cur["threshold"], color=ZONE, lw=0)
    ax.plot([t_ms(r) for r in active],
            [float(r["q_current_a"]) - cur["baseline"] for r in active],
            color=INK, lw=1.5, marker="o", ms=5)
    inside = [(t_ms(r), float(r["q_current_a"]) - cur["baseline"]) for r in after]
    shown = [p for p in inside if y_lo < p[1] < y_hi]
    clipped = [p for p in inside if not y_lo < p[1] < y_hi]
    if shown:
        ax.scatter(*zip(*shown), s=30, facecolor="white", edgecolor=MUTED, lw=1.2)
    if clipped:
        ax.scatter([p[0] for p in clipped], [y_lo + 0.06] * len(clipped), marker="v",
                   s=40, color=MUTED)
        low = min(p[1] for p in clipped)
        ax.text(44.5, y_lo + 0.13,
                f"после watchdog: до {low:.1f} A,\nвне шкалы, не в анализе".replace("-", "−"),
                ha="right", va="bottom", fontsize=10, color=INK_2)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("q-ток − покой, A")
    ax.text(-19.5, cur["threshold"] + 0.02, f"шум в покое: порог ±{cur['threshold']:.2f} A",
            va="bottom", fontsize=10.5, color=INK_2)
    ax.text(first - 0.6, y_hi - 0.05, f"первый замер\nвыше шума\n{first:.1f} мс",
            ha="right", va="top", fontsize=10.5, color=INK)
    ax.text(confirm + 0.6, y_hi - 0.05, f"второй\nподряд\n{confirm:.1f} мс",
            ha="left", va="top", fontsize=10.5, color=INK)
    ax.annotate(f"ответ на саму команду\n({reply:.1f} мс): состояние до неё",
                xy=(reply, float(pulse["q_current_a"]) - cur["baseline"]),
                xytext=(-19.5, -0.38), fontsize=10.5, color=INK, va="bottom",
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=1))

    # 3. Rotor angle change from the at-rest median.
    ax = axes[2]
    thr_deg = pos["threshold"] * 360
    ax.axhspan(-thr_deg, thr_deg, color=ZONE, lw=0)
    ax.plot([t_ms(r) for r in window],
            [(float(r["position_rev"]) - pos["baseline"]) * 360 for r in window],
            color=INK, lw=1.5, marker="o", ms=5)
    ax.set_ylabel("угол − покой, °")
    ax.text(-19.5, thr_deg + 0.05, f"порог ±{thr_deg:.2f}°", va="bottom", fontsize=10.5,
            color=INK_2)
    ax.text(31, 1.2, "момента уже нет,\nстрелка движется\nпо инерции", ha="left",
            va="center", fontsize=10.5, color=INK_2)
    ax.set_xlabel("время от команды хоста, мс")
    ax.set_xlim(-20, 45)

    header(fig, f"Импульс {torque_cmd * 1000:+.0f} mNm: ток выходит из шума через {first:.1f} мс",
           "Точки — замеры хоста, шаг ≈ 5.5 мс; реальный отклик привода был раньше первой "
           "точки выше шума")
    footer(fig, f"Прогон {run_id}, импульс {trial}. "
                f"Источник: {telemetry.name}, {analysis_json.name}.")
    layout(fig)
    fig.savefig(out, dpi=150)


# Sources:
#   PNDLM-9: docs/hardware/specs/motor-torque-fixture.md §9.3
#   PNDLM-8: docs/hardware/desk-stand-torque-pulse-timing.md
#   requirement, LQR sweep, t_c: docs/physics/phase-0-feasibility.md
#     ("Preliminary sensor timing budget", re-run 2026-11-08)
#   theory: λ = 1/139 ms; λ·d < 0.5 (Skogestad) and λ·d < √2 (Stépán, PD)
SCALE_MEASURED = [  # (label, upper bound ms)
    ("Измерено, PNDLM-9\n(опрос ~385 Hz)", 4.4),
    ("Измерено, PNDLM-8\n(опрос 183 Hz)", 13.0),
]
SCALE_THEORY = [  # (label, limit ms)
    ("Теория: правило\nробастности, λd < 0.5", 69.5),
    ("Теория: без\nпредсказания, λd < √2", 196.6),
]
REQUIRED_MS, SIM_HOLDS_MS, SIM_FAILS_MS = 13.9, 15.0, 20.0
FALL_MS = (139.0, 161.0)
X_MIN, X_MAX = 1.0, 400.0


def plot_scale(out: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xscale("log")
    ax.set_xlim(X_MIN, X_MAX)
    ticks = [1, 2, 5, 10, 20, 50, 100, 200]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(t) for t in ticks])
    n = len(SCALE_MEASURED) + len(SCALE_THEORY)
    ax.set_ylim(-0.7, n + 0.6)

    # Zones and lines across all rows.
    ax.axvspan(SIM_FAILS_MS, X_MAX, color=FAIL_BAND, lw=0, zorder=0)
    ax.text(SIM_FAILS_MS * 1.06, n + 0.45, f"симуляция с текущим LQR\nне удерживает (от {SIM_FAILS_MS:.0f} мс)",
            color=FAIL, fontsize=10.5, va="top")
    ax.axvspan(*FALL_MS, color=INK, alpha=0.18, lw=0, zorder=1)
    ax.text(FALL_MS[0] / 1.05, n - 0.55, f"маятник падает:\nt_c = {FALL_MS[0]:.0f}–{FALL_MS[1]:.0f} мс",
            color=INK, fontsize=10.5, va="center", ha="right")
    ax.axvline(REQUIRED_MS, color=REFERENCE, lw=1.5, ls="--", zorder=2)
    ax.axvline(SIM_HOLDS_MS, color=REFERENCE, lw=1.2, ls=":", zorder=2)
    ax.text(REQUIRED_MS * 0.95, n + 0.45,
            f"допустимо ≤ {REQUIRED_MS} мс (10 % t_c)\nсимуляция держит при {SIM_HOLDS_MS:.0f} мс",
            color=INK_2, fontsize=10.5, va="top", ha="right")

    labels = []
    y = n - 1
    for label, bound in SCALE_MEASURED:
        ax.barh(y, bound - X_MIN, left=X_MIN, height=0.5, color=MEASURED, lw=0, zorder=3)
        ax.text(bound / 1.06, y, f"≤ {bound:g} мс", va="center", ha="right",
                fontsize=11, color="white", weight="bold", zorder=4)
        labels.append((y, label))
        y -= 1
    for label, limit in SCALE_THEORY:
        ax.plot([X_MIN, limit], [y, y], color=REFERENCE, lw=1.5, zorder=3)
        ax.plot([limit], [y], marker="D", ms=8, color=REFERENCE, zorder=4)
        ax.text(limit / 1.1, y + 0.18, f"{limit:.0f} мс", va="bottom", ha="right",
                fontsize=11, color=INK, zorder=4)
        labels.append((y, label))
        y -= 1
    ax.set_yticks([p[0] for p in labels])
    ax.set_yticklabels([p[1] for p in labels], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("время, мс (логарифмическая шкала)")

    header(fig, "Привод отвечает за ≤ 4.4 мс, маятник падает за 139–161 мс",
           "Синие полосы — измеренные верхние границы задержки: настоящая задержка не больше")
    footer(fig, "Худший вариант конструкции (CAD, t_c = 139 мс). Измерено на ноутбуке + "
                "fdcanusb; теория — для линейной модели.")
    layout(fig)
    fig.savefig(out, dpi=150)


def main() -> None:
    if len(sys.argv) == 6 and sys.argv[1] == "pulse":
        plot_pulse(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    elif len(sys.argv) == 3 and sys.argv[1] == "scale":
        plot_scale(sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
