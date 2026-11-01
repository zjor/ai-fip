import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from moteus_host.bench.analysis import torque_speed
from moteus_host.bench.analysis.summarize import analyze
from moteus_host.bench.recorder import EVENT_FIELDS, TELEMETRY_FIELDS


KT = 0.0244
DT = 0.005
RPM_PER_V = 228.0
MIN_SATURATION_REV_S = 5.0
SAG_THRESHOLD_V = 0.3
SAG_BUS_V = 10.6
SATURATION_ALPHA_REV_S2 = 0.5
HOLD_S = 1.0
COAST_S = 1.0
TIMEOUT_S = 3.0
CSV_HEADER = ["trial", "torque_cmd_nm", "direction", "time_s", "speed_rev_s", "alpha_rev_s2",
              "tau_nm", "tau_se_nm", "q_current_a", "bus_v", "supply_limited"]


def _loss(v: float, a: float, b: float, c: float) -> float:
    omega = 2.0 * math.pi * abs(v)
    return a + b * omega + c * omega * omega


def _available(torque: float, v: float, no_load_rev_s: float) -> float:
    """Flat to 0.8*v_nl, then linear to zero at v_nl."""
    return torque * min(1.0, max(0.0, (no_load_rev_s - v) / (0.2 * no_load_rev_s)))


def _create_run(root: Path, *, bus_v=12.0, no_load_rev_s=40.0, torques=(0.05,), reasons=("saturated",),
                sag_from_rev_s=None, inertia=3.776e-4, a=4.43e-3, b=3.78e-5, c=9.7e-9,
                experiment="torque-speed", cap_rev_s=None) -> Path:
    """Synthetic torque-speed run in the Task 4 layout; a None reason is a step declined at GO."""
    run = root / "test_torque_speed"
    run.mkdir()
    cap = cap_rev_s if cap_rev_s is not None else 0.9 * no_load_rev_s
    rows: list[dict] = []
    events: list[dict] = []
    steps: list[dict] = []
    clock = {"t": 0.0}

    def bus(v: float) -> float:
        return SAG_BUS_V if sag_from_rev_s is not None and v >= sag_from_rev_s else bus_v

    def write(phase: str, trial: int, sign: float, v: float, tau: float, voltage: float) -> None:
        rows.append({"sample": len(rows), "host_time_s": f"{clock['t']:.6f}", "experiment_phase": phase,
                     "trial_id": trial, "command_mode": "torque" if phase in ("torque", "hold") else "stop",
                     "mode": 10, "fault": 0, "position_rev": 0.0, "velocity_rev_s": sign * v,
                     "torque_nm": sign * tau, "q_current_a": sign * tau / KT, "d_current_a": 0.0,
                     "bus_voltage_v": voltage, "controller_temperature_c": 30.0})

    def event(name: str, trial: int, note: str = "") -> None:
        events.append({"host_time_s": f"{clock['t']:.6f}", "event": name, "value": trial, "unit": "", "note": note})

    for index, (torque, reason) in enumerate(zip(torques, reasons)):
        trial = index + 1
        sign = 1.0 if torque > 0 else -1.0
        step = {"trial": trial, "torque_nm": torque, "direction": int(sign)}
        steps.append(step)
        if reason is None:
            continue
        step["reason"] = reason
        magnitude = abs(torque)
        for _ in range(3):
            write("preflight", trial, sign, 0.0, 0.0, bus_v)
            clock["t"] += DT
        event("operator_confirmed", trial)
        event("step_start", trial, f"{torque:+g} Nm")
        v, start = 0.0, clock["t"]
        while True:
            voltage = bus(v)
            available = _available(magnitude, v, no_load_rev_s)
            alpha = (available - _loss(v, a, b, c)) / (2.0 * math.pi * inertia)
            write("torque", trial, sign, v, available, voltage)
            elapsed = clock["t"] - start
            if reason == "supply-limited" and voltage < bus_v - SAG_THRESHOLD_V:
                break
            if reason == "velocity-capped" and v >= cap:
                break
            if reason == "saturated" and v > MIN_SATURATION_REV_S and abs(alpha) < SATURATION_ALPHA_REV_S2:
                break
            if elapsed >= TIMEOUT_S or elapsed > 10.0:
                break
            v += alpha * DT
            clock["t"] += DT
        clock["t"] += DT
        event("step_end", trial, reason)
        if reason == "saturated":
            event("hold_start", trial)
            hold_start = clock["t"]
            while clock["t"] - hold_start < HOLD_S:
                available = _available(magnitude, v, no_load_rev_s)
                write("hold", trial, sign, v, available, bus(v))
                v += (available - _loss(v, a, b, c)) / (2.0 * math.pi * inertia) * DT
                clock["t"] += DT
        event("coast_start", trial)
        coast_start = clock["t"]
        while clock["t"] - coast_start < COAST_S:
            write("coast", trial, sign, v, 0.0, bus_v)
            v = max(0.0, v - _loss(v, a, b, c) / (2.0 * math.pi * inertia) * DT)
            clock["t"] += DT
        event("step_complete", trial)

    with (run / "telemetry.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=TELEMETRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (run / "events.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(events)
    (run / "run.json").write_text(json.dumps({
        "run_id": "test_torque_speed", "experiment": experiment, "status": "completed",
        "sample_count": len(rows),
        "resolved_config": {
            "bus_v": bus_v,
            "records": {"inertia_kg_m2": inertia, "inertia_rel_sd": 0.026,
                        "loss_fit": {"a_nm": a, "b_nm_s": b, "c_nm_s2": c, "rmse_nm": 2e-4},
                        "source": "docs/hardware/records/test.json"},
            "sag_threshold_v": SAG_THRESHOLD_V,
            "kt_nm_per_a": 0.0278,
            "expected_noload_rpm_per_v": RPM_PER_V,
            "min_saturation_rev_s": MIN_SATURATION_REV_S,
            "steps": steps,
        },
    }))
    return run


def _equilibrium(torque: float, no_load_rev_s: float, a=4.43e-3, b=3.78e-5, c=9.7e-9) -> float:
    """v = v_nl * (1 - 0.2 * loss(v) / T), solved by fixed point."""
    v = no_load_rev_s
    for _ in range(100):
        v = no_load_rev_s * (1.0 - 0.2 * _loss(v, a, b, c) / torque)
    return v


def _summary(run: Path) -> dict:
    return json.loads(torque_speed.analyze(run).read_text())


class TorqueSpeedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_low_speed_torque_recovers_the_commanded_torque(self):
        summary = _summary(_create_run(self.root, torques=(0.05, -0.05), reasons=("saturated", "saturated")))
        for step in summary["steps"]:
            low = step["low_speed"]
            self.assertIsNotNone(low)
            self.assertGreaterEqual(low["windows"], 2)
            self.assertLess(abs(low["tau_nm"] - 0.05) / 0.05, 0.03)
            self.assertLess(abs(low["q_current_a"] - 0.05 / KT) / (0.05 / KT), 0.03)

    def test_knee_at_ten_percent_drop(self):
        summary = _summary(_create_run(self.root, torques=(0.05, -0.05), reasons=("saturated", "saturated")))
        for step in summary["steps"]:
            knee = step["knee"]
            self.assertIsNotNone(knee)
            self.assertGreater(knee["speed_se_rev_s"], 0.0)
            self.assertLessEqual(abs(knee["speed_rev_s"] - 0.82 * 40.0), 2.0 * knee["speed_se_rev_s"])
            self.assertAlmostEqual(knee["tau_nm"], 0.9 * step["low_speed"]["tau_nm"])
            self.assertGreater(knee["q_current_a"], 0.0)

    def test_no_load_from_hold(self):
        summary = _summary(_create_run(self.root, torques=(0.05, -0.05), reasons=("saturated", "saturated")))
        expected = _equilibrium(0.05, 40.0)
        for step in summary["steps"]:
            no_load = step["no_load"]
            self.assertFalse(no_load["lower_bound"])
            self.assertLess(abs(no_load["speed_rev_s"] - expected) / expected, 0.01)
            self.assertGreaterEqual(no_load["speed_sd_rev_s"], 0.0)
            self.assertEqual(no_load["bus_v"], 12.0)
            self.assertIsNotNone(no_load["loss_bias_rev_s"])
            self.assertGreater(no_load["loss_bias_rev_s"], 0.0)
        by_direction = summary["no_load_by_direction"]
        self.assertAlmostEqual(by_direction["positive"]["speed_rev_s"], by_direction["negative"]["speed_rev_s"],
                               places=2)
        self.assertAlmostEqual(by_direction["difference_rev_s"], 0.0, places=2)

    def test_velocity_capped_is_lower_bound(self):
        run = _create_run(self.root, torques=(0.05,), reasons=("velocity-capped",), cap_rev_s=35.0)
        summary = _summary(run)
        with (run / "telemetry.csv").open(newline="") as source:
            peak = max(abs(float(r["velocity_rev_s"])) for r in csv.DictReader(source)
                       if r["experiment_phase"] == "torque")
        no_load = summary["steps"][0]["no_load"]
        self.assertTrue(no_load["lower_bound"])
        self.assertEqual(no_load["speed_rev_s"], peak)
        self.assertIsNone(summary["no_load_by_direction"]["positive"])

    def test_supply_limited_windows_are_flagged_and_excluded(self):
        run = _create_run(self.root, torques=(0.05,), reasons=("timeout",), sag_from_rev_s=30.0)
        summary = _summary(run)
        step = summary["steps"][0]
        self.assertGreater(step["windows_flagged"], 0)
        self.assertIsNone(step["no_load"])
        with (run / "telemetry.csv").open(newline="") as source:
            torque_rows = [r for r in csv.DictReader(source) if r["experiment_phase"] == "torque"]
        windows = torque_speed.delivered_torque_windows(
            torque_rows,
            inertia=3.776e-4, inertia_rel_sd=0.026,
            loss_fit={"a_nm": 4.43e-3, "b_nm_s": 3.78e-5, "c_nm_s2": 9.7e-9}, bus_v=12.0, sag_threshold_v=0.3)
        flagged = [w for w in windows if w["supply_limited"]]
        self.assertEqual(len(flagged), step["windows_flagged"])
        for window in flagged:
            self.assertLess(window["bus_v"], 12.0 - 0.3)
        # every window past the sag is flagged, so no unflagged window carries the drop
        self.assertIsNone(step["knee"])
        self.assertIsNotNone(step["low_speed"])
        self.assertLess(abs(step["low_speed"]["tau_nm"] - 0.05) / 0.05, 0.03)
        self.assertIsNone(torque_speed.knee(flagged, step["low_speed"]["tau_nm"]))

    def test_no_low_speed_band_gives_null_knee(self):
        run = _create_run(self.root, torques=(0.05,), reasons=("supply-limited",), sag_from_rev_s=3.0)
        summary = _summary(run)
        step = summary["steps"][0]
        self.assertEqual(step["reason"], "supply-limited")
        self.assertIsNone(step["low_speed"])
        self.assertIsNone(step["knee"])
        self.assertIsNone(step["no_load"])
        self.assertTrue((run / "summary.json").is_file())

    def test_rejects_wrong_experiment(self):
        run = _create_run(self.root, experiment="coast-down")
        with self.assertRaises(ValueError):
            torque_speed.analyze(run)

    def test_writes_csv_and_plots(self):
        run = _create_run(self.root, torques=(0.05, -0.05, 0.08), reasons=("saturated", "saturated", None))
        summary = _summary(run)
        self.assertEqual(summary["skipped_trials"], [3])
        self.assertEqual([s["trial"] for s in summary["steps"]], [1, 2])
        self.assertEqual(summary["bus_v"], 12.0)
        self.assertEqual(summary["records"]["inertia_kg_m2"], 3.776e-4)
        envelope = summary["operating_envelope"]
        self.assertEqual(envelope["bus_voltage_v"], {"min": 12.0, "max": 12.0})
        self.assertGreater(envelope["peak_abs_q_current_a"], 0.0)
        with (run / "torque-vs-speed.csv").open(newline="") as source:
            reader = csv.reader(source)
            header = next(reader)
            body = list(reader)
        self.assertEqual(header, CSV_HEADER)
        self.assertEqual({row[0] for row in body}, {"1", "2"})
        self.assertEqual({row[1] for row in body}, {"0.05", "-0.05"})
        self.assertTrue(all(float(row[6]) > 0.0 for row in body))
        torque_plot = (run / "plots" / "torque-vs-speed.svg").read_text()
        self.assertIn("+0.05 Nm", torque_plot)
        self.assertIn("-0.05 Nm", torque_plot)
        self.assertTrue((run / "plots" / "speed-vs-time.svg").is_file())

    def test_dispatch_from_summarize(self):
        run = _create_run(self.root)
        output = analyze(run)
        self.assertEqual(output, run.resolve() / "summary.json")
        summary = json.loads(output.read_text())
        self.assertEqual(summary["source"]["experiment"], "torque-speed")
        self.assertEqual(len(summary["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
