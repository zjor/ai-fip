import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from moteus_host.bench.analysis import coast_down
from moteus_host.bench.analysis.summarize import analyze


FIELDS = (
    "sample", "host_time_s", "scheduled_time_s", "request_time_s", "response_time_s",
    "experiment_phase", "trial_id", "command_mode", "mode", "fault", "position_rev",
    "velocity_rev_s", "torque_nm", "q_current_a", "d_current_a", "bus_voltage_v",
    "controller_temperature_c",
)
INERTIA = 3.54e-4
KT = 0.0244
DT = 0.005


def _loss(v: float, a: float, b: float, c: float) -> float:
    omega = 2.0 * math.pi * abs(v)
    return a + b * omega + c * omega * omega


def _create_run(root: Path, *, inertia=INERTIA, a=0.004, b=2e-5, c=1e-7, kt=KT,
                steps=(8.33, 25.0, 50.0), accel=((0.10, 1), (0.10, -1)),
                coast_s=None, status="completed") -> Path:
    run = root / "test_coast"
    run.mkdir()
    rows: list[dict] = []
    plan: list[dict] = []
    time_s = 0.0
    sample = 0

    def emit(phase, trial, v, q, command="query"):
        nonlocal time_s, sample
        rows.append({
            "sample": sample, "host_time_s": time_s, "scheduled_time_s": time_s,
            "request_time_s": time_s, "response_time_s": time_s + 0.001,
            "experiment_phase": phase, "trial_id": trial, "command_mode": command,
            "mode": 10 if command != "query" else 0, "fault": 0, "position_rev": 0.0,
            "velocity_rev_s": v, "torque_nm": q * kt, "q_current_a": q, "d_current_a": 0.0,
            "bus_voltage_v": 24.0, "controller_temperature_c": 25.0,
        })
        time_s += DT
        sample += 1

    def coast(trial, v):
        elapsed = 0.0
        while abs(v) >= 0.2 and (coast_s is None or elapsed < coast_s):
            emit("coast", trial, v, 0.0)
            v -= math.copysign(_loss(v, a, b, c), v) / (2.0 * math.pi * inertia) * DT
            elapsed += DT

    trial = 0
    for target in steps:
        plan.append({"trial": trial, "kind": "ladder", "target_rev_s": target,
                     "torque_nm": 0.10, "direction": 1})
        v = 0.0
        while v < target:
            v = min(target, v + 20.0 * DT)
            emit("spin_up", trial, v, 0.10 / kt, "velocity")
        for _ in range(int(3.0 / DT)):
            emit("plateau", trial, target, _loss(target, a, b, c) / kt, "velocity")
        coast(trial, target)
        trial += 1
    for torque, direction in accel:
        plan.append({"trial": trial, "kind": "accel", "target_rev_s": direction * 30.0,
                     "torque_nm": direction * torque, "direction": direction})
        v = 0.0
        while abs(v) < 30.0:
            emit("accel", trial, v, direction * torque / kt, "torque")
            net = torque - _loss(v, a, b, c)
            v += direction * net / (2.0 * math.pi * inertia) * DT
        coast(trial, v)
        trial += 1

    (run / "run.json").write_text(json.dumps({
        "run_id": "test_coast", "experiment": "coast-down", "status": status,
        "sample_count": len(rows),
        "resolved_config": {
            "inertia": {"I_z_kg_m2": inertia, "source": "synthetic"},
            "kt_nm_per_a": kt, "steps": plan, "sample_rate_hz": 1.0 / DT,
        },
    }))
    with (run / "telemetry.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (run / "events.csv").write_text("host_time_s,event,value,unit,note\n")
    (run / "controller-config.txt").write_text("motor.Kv 330\n")
    return run


class CoastDownAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _summary(self, **kwargs) -> dict:
        run = _create_run(self.root, **kwargs)
        output = analyze(run)
        self.assertEqual(output, (run / "summary.json").resolve())
        return json.loads(output.read_text())

    def test_fit_recovers_loss_model(self) -> None:
        fit = self._summary()["fit"]
        self.assertAlmostEqual(fit["a_nm"], 0.004, delta=0.0004)
        self.assertAlmostEqual(fit["b_nm_s"], 2e-5, delta=3e-6)
        self.assertAlmostEqual(fit["c_nm_s2"], 1e-7, delta=1.5e-8)
        self.assertGreater(fit["n"], 6)
        for key in ("a_se", "b_se", "c_se", "rmse_nm"):
            self.assertGreaterEqual(fit[key], 0.0)

    def test_loss_map_bins_cover_ladder(self) -> None:
        run = _create_run(self.root)
        analyze(run)
        with (run / "loss-map.csv").open(newline="") as source:
            bins = list(csv.DictReader(source))
        speeds = [float(row["speed_rev_s"]) for row in bins]
        self.assertGreaterEqual(max(speeds), 45.0)
        self.assertTrue(all(int(row["n"]) >= 1 for row in bins))
        self.assertEqual(set(bins[0]), {"speed_rev_s", "n", "tau_mean_nm", "tau_sd_nm"})

    def test_plateau_cross_check(self) -> None:
        plateau = self._summary()["plateau"]
        self.assertEqual([row["trial"] for row in plateau], [0, 1, 2])
        for row in plateau:
            expected = _loss(row["speed_rev_s"], 0.004, 2e-5, 1e-7)
            self.assertAlmostEqual(row["tau_nm"], expected, delta=1e-6)

    def test_effective_inertia(self) -> None:
        inertia = self._summary()["effective_inertia"]
        self.assertEqual(len(inertia["trials"]), 2)
        self.assertAlmostEqual(inertia["mean"], INERTIA, delta=0.05 * INERTIA)
        self.assertEqual(inertia["trials"][1]["torque_nm"], -0.10)

    def test_short_coast_gives_null_fit(self) -> None:
        summary = self._summary(steps=(0.33,), accel=(), coast_s=0.1)
        self.assertIsNone(summary["fit"])
        self.assertEqual(summary["source"]["experiment"], "coast-down")

    def test_rejects_incomplete_run(self) -> None:
        run = _create_run(self.root, status="safety_stop")
        with self.assertRaisesRegex(ValueError, "completed"):
            analyze(run)

    def test_dispatch_rejects_unknown_experiment(self) -> None:
        run = _create_run(self.root)
        metadata = json.loads((run / "run.json").read_text())
        metadata["experiment"] = "thermal-hold"
        (run / "run.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "thermal-hold"):
            analyze(run)

    def test_writes_plots(self) -> None:
        run = _create_run(self.root)
        summary = json.loads(analyze(run).read_text())
        for name in summary["artifacts"]["plots"]:
            self.assertTrue((run / name).is_file(), name)

    def test_coast_points_skip_short_windows(self) -> None:
        times = [i * DT for i in range(3)]
        self.assertEqual(coast_down.coast_points(times, [1.0, 0.9, 0.8], INERTIA), [])
        times = [i * DT for i in range(100)]
        velocities = [10.0 - 2.0 * t for t in times]          # slope -2 rev/s^2
        points = coast_down.coast_points(times, velocities, INERTIA)
        self.assertTrue(points)
        for omega, tau in points:
            self.assertAlmostEqual(tau, INERTIA * 2.0 * math.pi * 2.0, delta=1e-9)
            self.assertGreater(omega, 0.0)


if __name__ == "__main__":
    unittest.main()
