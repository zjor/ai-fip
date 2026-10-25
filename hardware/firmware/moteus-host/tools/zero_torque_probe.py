"""Diagnostic: where does the rotor move during a 0 Nm position-mode probe?

Phases, all sampled at 200 Hz:
  stopped   0.5 s  query only, controller stopped
  hold0     1.5 s  set_position(kp=kd=ilimit=0, feedforward 0 Nm, max 0.05 Nm, watchdog 0.1 s)
  silence   0.4 s  query only; the watchdog expires after 0.1 s -> mode 11 (timeout_mode 12)
  stopped2  0.5 s  after set_stop, query only
Stops immediately on |velocity| > 0.3 rev/s, |excursion| > 0.1 rev, |q| > 2 A or a fault.
Usage: poetry run python zero_torque_probe.py [--dry-run] [--out file.csv]
"""
import asyncio, csv, math, statistics, sys, time
from moteus_host.bench.servo import create_controller, result_to_state
from moteus_host.bench.experiments.common import confirm_stopped, wait_until

PERIOD = 1 / 200
PHASES = [("stopped", 0.5), ("hold0", 1.5), ("silence", 0.4), ("stopped2", 0.5)]
FIELDS = ["t_s", "phase", "mode", "fault", "position_rev", "velocity_rev_s", "q_current_a",
          "d_current_a", "torque_nm", "bus_voltage_v"]

async def main(out):
    controller, moteus = create_controller(1, timing=True)
    rows, origin = [], None
    start = time.perf_counter()
    async def sample(phase):
        if phase == "hold0":
            result = await asyncio.wait_for(controller.set_position(
                position=math.nan, velocity=0.0, kp_scale=0.0, kd_scale=0.0, ilimit_scale=0.0,
                feedforward_torque=0.0, maximum_torque=0.05, watchdog_timeout=0.1, query=True), 0.1)
        else:
            result = await asyncio.wait_for(controller.query(), 0.1)
        s = result_to_state(result, moteus)
        row = dict(t_s=round(time.perf_counter() - start, 4), phase=phase,
                   **{k: s[k] for k in FIELDS[2:]})
        rows.append(row)
        if (s["fault"] or abs(s["velocity_rev_s"]) > 0.3 or abs(s["q_current_a"]) > 2.0
                or (origin is not None and abs(s["position_rev"] - origin) > 0.1)):
            raise RuntimeError(f"abort: {row}")
        return s
    try:
        state = await confirm_stopped(controller, moteus, timeout_s=5.0)
        origin = state["position_rev"]
        print(f"origin {origin:.4f} rev", flush=True)
        for phase, duration in PHASES:
            if phase == "stopped2":
                await confirm_stopped(controller, moteus, timeout_s=0.1)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < duration:
                tick = time.perf_counter()
                await sample(phase)
                await wait_until(tick + PERIOD)
    finally:
        try:
            await confirm_stopped(controller, moteus, timeout_s=1.0)
            print("stopped and confirmed", flush=True)
        finally:
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
            for phase, _ in PHASES:
                r = [x for x in rows if x["phase"] == phase]
                if not r: continue
                print(f"{phase:9s} n={len(r):3d} modes={sorted({x['mode'] for x in r})} "
                      f"pos {r[0]['position_rev']:.4f}->{r[-1]['position_rev']:.4f} "
                      f"max|v| {max(abs(x['velocity_rev_s']) for x in r):.3f} "
                      f"q mean {statistics.fmean(x['q_current_a'] for x in r):+.3f} "
                      f"d mean {statistics.fmean(x['d_current_a'] for x in r):+.3f} "
                      f"torque mean {statistics.fmean(x['torque_nm'] for x in r):+.4f}")
            moved = [x for x in rows if origin is not None and abs(x["position_rev"] - origin) > 0.003]
            if moved:
                m = moved[0]
                print(f"first motion > 0.003 rev at t={m['t_s']} s, phase {m['phase']}, mode {m['mode']}")
            else:
                print("no motion above 0.003 rev")

if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "zero_torque_probe.csv"
    if "--dry-run" in sys.argv:
        print("dry run:", PHASES, "-> no connection"); sys.exit(0)
    asyncio.run(main(out))
