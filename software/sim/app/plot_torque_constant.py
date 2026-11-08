"""Plot the PNDLM-9 torque-constant figures for the article draft.

Usage, from software/sim:

    poetry run python -m app.plot_torque_constant constant \
        ../../docs/hardware/measurements/2026-10-26-torque-constant.json \
        ../../docs/content/figures/2026-10-26-torque-constant.png

    poetry run python -m app.plot_torque_constant extract-rise \
        ../../hardware/firmware/moteus-host/logs/bench/20261026T180300Z_torque-constant/telemetry.csv \
        ../../docs/hardware/measurements/2026-10-26-torque-rise-time-telemetry.csv

    poetry run python -m app.plot_torque_constant rise \
        ../../docs/hardware/measurements/2026-10-26-torque-rise-time-telemetry.csv \
        ../../docs/hardware/measurements/2026-10-26-torque-rise-time.json \
        ../../docs/content/figures/2026-10-26-torque-rise-time.png

``constant``: top, physical torque on the scale against commanded q-current
with the documented line (8.27/Kv) and the measured linear-region line; bottom,
each reading's deviation from the measured line through zero, with the mean
offset of each side and the ±5 mNm disturbance band centred between them.

``extract-rise``: copies the pre-hold and step rows of the git-ignored raw
``telemetry.csv`` of run ``20261026T180300Z_torque-constant`` (a column
subset) into the repository. The rise JSON records the raw file's SHA-256.

``rise``: q-current change after each torque step as a share of the step,
negative and positive steps in separate panels, with the 63 % and 95 % levels.
Start and steady levels are the analyzer's (pre-hold mean, mean of the last
half of the step), so the plot matches ``t63_ms`` / ``t95_ms`` in the JSON.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

from app.plot_style import (
    INK, INK_2, MEASURED, MEASURED_2, MEASURED_BAND, MUTED, REFERENCE, ZONE,
    footer, header, layout, plt, style,
)

G = 9.80665
RISE_COLUMNS = ("trial_id", "experiment_phase", "request_time_s", "response_time_s",
                "mode", "feedforward_torque_nm", "q_current_a", "position_rev")


def plot_constant(analysis_json: Path, out: str) -> None:
    d = json.load(open(analysis_json))
    kt_fw = d["simulator"]["kt_firmware_nm_per_a"]
    kt = d["simulator"]["kt_linear_region_per_commanded_a_nm_per_a"]
    radius_m = 0.105
    plateaus = [p for p in d["plateaus"] if p["physical_torque_nm"] is not None]
    runs = sorted({p["run_id"] for p in plateaus})
    sides = (("positive", MEASURED, "o", "+ сторона"),
             ("negative", MEASURED_2, "s", "− сторона"))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 10.5),
                                  gridspec_kw=dict(height_ratios=(1.15, 1)))
    for a in (ax, ax2):
        style(a)
        a.axvspan(0, 2, color=ZONE, lw=0, zorder=0)
        a.set_xlim(0, 6.6)
        a.set_xlabel("командный ток I_q, A")

    # Top: torque against current.
    xs = (0, 6.3)
    ax.plot(xs, [kt_fw * x * 1000 for x in xs], color=REFERENCE, lw=2, ls="--", zorder=2)
    ax.plot(xs, [kt * x * 1000 for x in xs], color=INK, lw=2, zorder=2)
    for direction, color, marker, label in sides:
        mine = [p for p in plateaus if p["direction"] == direction]
        ax.scatter([abs(p["command_a"]) for p in mine],
                   [abs(p["physical_torque_nm"]) * 1000 for p in mine],
                   s=60, color=color, edgecolor="white", lw=1.2, marker=marker,
                   label=f"{label}, показания весов", zorder=3)
    ax.text(6.33, kt_fw * 6.3 * 1000, f"по документации\n8.27/Kv: {kt_fw * 1000:.1f} mNm/A",
            color=REFERENCE, va="center", ha="left", fontsize=11)
    ax.text(6.33, kt * 6.3 * 1000 - 6, f"измерено:\n{kt * 1000:.1f} mNm/A",
            color=INK, va="center", ha="left", fontsize=11, weight="bold")
    x_gap = 5.0
    ax.annotate("", xy=(x_gap, kt * x_gap * 1000), xytext=(x_gap, kt_fw * x_gap * 1000),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(x_gap - 0.08, (kt + kt_fw) / 2 * x_gap * 1000,
            f"−{100 * (1 - kt / kt_fw):.0f} %", ha="right", va="center",
            color=INK, fontsize=12, weight="bold")
    ax.text(1.0, 160, "ниже 2 A\nрычаг садится\nна весы", ha="center", va="top",
            color=INK_2, fontsize=10.5)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(-5, 185)
    ax.set_ylabel("момент на весах, mNm")
    ax.set_title("Момент против тока", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.2, 1.0))

    # Bottom: deviation from the measured line through zero.
    means = {}
    for direction, color, marker, label in sides:
        mine = [p for p in plateaus if p["direction"] == direction]
        dev = [(abs(p["physical_torque_nm"]) - kt * abs(p["command_a"])) * 1000 for p in mine]
        ax2.scatter([abs(p["command_a"]) for p in mine], dev, s=60, color=color,
                    edgecolor="white", lw=1.2, marker=marker, zorder=3, label=label)
        linear = [v for p, v in zip(mine, dev) if abs(p["command_a"]) >= 2]
        means[direction] = statistics.fmean(linear)
        ax2.plot((2, 6.3), (means[direction],) * 2, color=color, lw=1.5, ls="--", zorder=2)
        ax2.text(6.33, means[direction], f"{label}:\n{means[direction]:+.1f} mNm".replace("-", "−"),
                 color=INK, va="center", ha="left", fontsize=10.5)
    centre = (means["positive"] + means["negative"]) / 2
    spread = abs(means["positive"] - means["negative"])
    ax2.axhspan(centre - 5, centre + 5, xmin=2 / 7.6, color=MEASURED_BAND, alpha=0.6,
                lw=0, zorder=1)
    ax2.text(3.1, centre + 1.2, "принятое\nвозмущение ±5 mNm", color=INK_2,
             fontsize=10.5, va="center")
    ax2.annotate("", xy=(5.0, means["positive"]), xytext=(5.0, means["negative"]),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax2.text(5.08, centre - 0.3, f"≈ {spread:.0f} mNm между\nсторонами, источник\nне установлен",
             color=INK, fontsize=10.5, va="center")
    ax2.axhline(0, color=REFERENCE, lw=1)
    ax2.set_xlim(0, 7.6)
    ax2.set_ylim(-16, 10)
    ax2.set_ylabel("отклонение, mNm  (1 mNm ≈ 1 г)")
    ax2.set_title(f"Отклонение от прямой {kt * 1000:.1f} mNm/A через ноль",
                  loc="left", color=INK)
    ax2.legend(frameon=False, loc="lower left")

    header(fig, f"Мотор развивает {kt * 1000:.1f} mNm на 1 A — "
                f"на {100 * (1 - kt / kt_fw):.0f} % меньше, чем по документации",
           f"mj5208 + moteus r4.11, рычаг {radius_m * 1000:.0f} мм, весы с ценой деления "
           f"0.1 г; {len(plateaus)} показаний, {len(runs)} прогонов, обе стороны")
    footer(fig, f"Прогоны {runs[0][:16]}…{runs[-1][:16]}. "
                f"Источник: {analysis_json.name}, поле plateaus.")
    layout(fig)
    fig.savefig(out, dpi=150)


def extract_rise(raw: Path, out: Path) -> None:
    with open(raw) as src, open(out, "w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=RISE_COLUMNS)
        writer.writeheader()
        for row in csv.DictReader(src):
            if row["experiment_phase"] in ("prehold", "step"):
                writer.writerow({k: row[k] for k in RISE_COLUMNS})


def plot_rise(telemetry: Path, analysis_json: Path, out: str) -> None:
    d = json.load(open(analysis_json))
    run_id = d["source"]["run_id"]
    trials = {str(r["trial_id"]): r for r in d["rise_time"]}
    rows = list(csv.DictReader(open(telemetry)))
    shade = {0.02: "#86b6ef", 0.05: "#1c5cab"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True)
    panels = ((axes[0], -1, "− ступени: работа контура тока"),
              (axes[1], +1, "+ ступени: рычаг ещё садится на весы"))
    first_reply, second_reply = [], []
    for ax, sign, title in panels:
        style(ax)
        ax.axvspan(0, 4.4, color=ZONE, lw=0, zorder=0)
        for level in (0.63, 0.95):
            ax.axhline(level, color=REFERENCE, lw=1, ls="--", zorder=1)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        seen = set()
        for tid, r in trials.items():
            if (r["torque_nm"] > 0) != (sign > 0):
                continue
            mine = [x for x in rows if x["trial_id"] == tid]
            t0 = float(next(x for x in mine if x["experiment_phase"] == "step")["request_time_s"])
            start, change = r["start_a"], r["steady_a"] - r["start_a"]
            pts = [((float(x["response_time_s"]) - t0) * 1000,
                    (float(x["q_current_a"]) - start) / change) for x in mine]
            pts = [p for p in pts if -6 <= p[0] <= 30]
            step_pts = [p for p in pts if p[0] > 0]
            first_reply.append(step_pts[0][0])
            second_reply.append(step_pts[1][0])
            level = round(abs(r["torque_nm"]), 2)
            label = f"0.01 → {level:.2f} Nm" if level not in seen else None
            seen.add(level)
            ax.plot(*zip(*pts), color=shade[level], lw=1.3, marker="o", ms=4.5,
                    alpha=0.9, label=label, zorder=3)
        ax.set_xlim(-6, 30)
        ax.set_ylim(-0.65, 1.6)
        ax.set_xlabel("время от команды ступени, мс")
        ax.set_title(title, loc="left", color=INK)
        ax.text(-5.7, 0.63, "63 %", ha="left", va="bottom", color=INK_2, fontsize=10.5)
        ax.text(-5.7, 0.95, "95 %", ha="left", va="bottom", color=INK_2, fontsize=10.5)

    f1, f2 = statistics.median(first_reply), statistics.median(second_reply)
    ax = axes[0]
    ax.annotate(f"ответ на саму команду\n({f1:.1f} мс): ещё старый ток",
                xy=(f1, -0.08), xytext=(5.5, -0.42), fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=1))
    ax.annotate(f"следующий замер\n({f2:.1f} мс): ток уже новый",
                xy=(f2, 1.02), xytext=(11, 1.4), fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=1))
    ax.text(2.2, -0.62, "ток\nустановился\nздесь", ha="center", va="bottom",
            fontsize=10, color=INK_2)
    ax.text(10, 0.25, "Оценка, не измерение:\nобмотка L/R ≈ 0.6 мс,\nконтур тока 200 Hz\n"
                      "→ 95 % за ≈ 2.4 мс", fontsize=10.5, color=INK_2, va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=MUTED, lw=0.8))
    axes[1].text(10, 0.25, "ток меняется вместе\nс положением ротора\n"
                           "(компенсация cogging)", fontsize=10.5, color=INK_2, va="center")
    axes[0].set_ylabel("доля изменения тока")
    axes[0].legend(frameon=False, loc="upper left", title="ступень момента",
                   title_fontsize=10.5)

    header(fig, "Ток устанавливается быстрее, чем хост успевает это увидеть",
           "q-ток после ступени момента, доля от изменения; 12 ступеней после удержания "
           f"±0.01 Nm; замер раз в {statistics.median(r['resolution_ms'] for r in trials.values()):.1f} мс")
    footer(fig, f"Прогон {run_id}. Источник: {telemetry.name}, {analysis_json.name}.")
    layout(fig)
    fig.savefig(out, dpi=150)


def main() -> None:
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "constant" and len(args) == 2:
        plot_constant(Path(args[0]), args[1])
    elif cmd == "extract-rise" and len(args) == 2:
        extract_rise(Path(args[0]), Path(args[1]))
    elif cmd == "rise" and len(args) == 3:
        plot_rise(Path(args[0]), Path(args[1]), args[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
