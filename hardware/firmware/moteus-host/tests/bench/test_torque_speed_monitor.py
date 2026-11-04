import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from moteus_host.bench.analysis.coast_down import _linear_fit
from moteus_host.bench.config import REPO_ROOT, load_config
from moteus_host.bench.experiments import torque_speed as experiment

RATE_HZ = 200.0
BUS_V = 12.0
DERATE = 2.0


def state(v, bus=BUS_V, i_q=1.0):
    return {"velocity_rev_s": v, "q_current_a": i_q, "bus_voltage_v": bus}


class StepMonitorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config().torque_speed

    def monitor(self, cfg=None) -> experiment.StepMonitor:
        return experiment.StepMonitor(cfg or self.cfg, BUS_V, DERATE)

    def test_saturation_after_settle(self) -> None:
        # v ramps 0 -> 40 rev/s over 2 s (20 rev/s^2), then flat at 40 with +-0.02 jitter.
        flat_start = 400
        samples = []
        for i in range(1200):
            t = i / RATE_HZ
            v = 20.0 * t if i <= flat_start else 40.0 + (0.02 if i % 2 else -0.02)
            samples.append((t, v))

        # Expected sample, computed independently of the monitor: the first sample whose
        # 0.25 s window (samples with t >= now - 0.25 s, ends included) has a least-squares
        # slope below saturation_alpha_rev_s2 starts the settle timer; the timer reaches
        # saturation_settle_s = 0.5 s exactly 100 samples later at 200 Hz.
        window_s = 0.25
        first_low = None
        for i, (t, v) in enumerate(samples):
            window = [(tt, vv) for tt, vv in samples[: i + 1] if tt >= t - window_s - 1e-9]
            if len(window) < 5 or window[-1][0] - window[0][0] < 0.9 * window_s:
                continue
            slope, _ = _linear_fit([w[0] for w in window], [w[1] for w in window])
            if abs(v) > self.cfg.min_saturation_rev_s and abs(slope) < self.cfg.saturation_alpha_rev_s2:
                first_low = i
                break
        self.assertIsNotNone(first_low)
        # The window may delay detecting the flattening by at most one window length.
        self.assertGreater(first_low, flat_start)
        self.assertLessEqual(first_low, flat_start + int(window_s * RATE_HZ))
        expected = first_low + int(round(self.cfg.saturation_settle_s * RATE_HZ))

        monitor = self.monitor()
        for i, (t, v) in enumerate(samples[:expected]):
            self.assertFalse(monitor(state(v), t), f"sample {i}")
            self.assertIsNone(monitor.reason)
        t, v = samples[expected]
        self.assertTrue(monitor(state(v), t))
        self.assertEqual(monitor.reason, "saturated")
        self.assertLess(abs(monitor.alpha_rev_s2), self.cfg.saturation_alpha_rev_s2)
        # The first reason is kept and the monitor stays done.
        self.assertTrue(monitor(state(70.0, bus=5.0), 20.0))
        self.assertEqual(monitor.reason, "saturated")

    def test_no_saturation_below_min_speed(self) -> None:
        monitor = self.monitor()
        for i in range(int(3 * RATE_HZ)):
            self.assertFalse(monitor(state(2.0), i / RATE_HZ))
            self.assertIsNone(monitor.reason)
        self.assertIsNotNone(monitor.alpha_rev_s2)
        self.assertAlmostEqual(monitor.alpha_rev_s2, 0.0, places=6)
        self.assertTrue(monitor(state(2.0), self.cfg.step_timeout_s))
        self.assertEqual(monitor.reason, "timeout")

    def test_timeout_is_a_reason(self) -> None:
        monitor = self.monitor()
        self.assertFalse(monitor(state(10.0), self.cfg.step_timeout_s - 0.005))
        self.assertIsNone(monitor.reason)
        self.assertTrue(monitor(state(10.1), self.cfg.step_timeout_s))
        self.assertEqual(monitor.reason, "timeout")

    def test_sag_wins(self) -> None:
        cfg = self.cfg
        floor = load_config().safety.min_bus_voltage_v
        for bus in cfg.bus_voltages_v:
            with self.subTest(bus=bus):
                monitor = experiment.StepMonitor(cfg, bus, DERATE)
                k = round((bus - cfg.sag_stop_v) * 10)
                # k * 0.1 V is exactly --bus - sag_stop_v: not below it, so no stop.
                self.assertFalse(monitor(state(30.0, bus=round(k * 0.1, 1)), 1.0))
                self.assertIsNone(monitor.reason)
                # A lost voltage reading is no information, not a trigger.
                self.assertFalse(monitor(state(30.1, bus=None), 1.005))
                self.assertIsNone(monitor.reason)
                # one telemetry step lower, while also at the cap margin and past the timeout:
                # sag is checked first, and the reading is still at or above the safety floor
                low = round((k - 1) * 0.1, 1)
                self.assertGreaterEqual(low, floor)
                self.assertTrue(monitor(state(63.7, bus=low), cfg.step_timeout_s))
                self.assertEqual(monitor.reason, "supply-limited")

    def test_cap_margin(self) -> None:
        monitor = self.monitor()
        self.assertFalse(monitor(state(63.6), 1.0))
        self.assertFalse(monitor(state(None), 1.005))
        self.assertIsNone(monitor.reason)
        self.assertTrue(monitor(state(63.7), 1.01))
        self.assertEqual(monitor.reason, "velocity-capped")
        other = self.monitor()
        self.assertTrue(other(state(-63.7), 1.0))
        self.assertEqual(other.reason, "velocity-capped")

    def test_saturation_inside_derate_band_is_capped(self) -> None:
        # With the default 3.0 rev/s margin, 65.0 rev/s already trips the cap-margin rule.
        # A 1.0 rev/s margin (host stop at 65.7) lets the saturation rule see a plateau at
        # 65.0 inside the derate band (above 66.7 - 2.0 = 64.7), so it must end velocity-capped.
        cfg = replace(self.cfg, cap_margin_rev_s=1.0)
        monitor = self.monitor(cfg)
        end = None
        for i in range(int(2 * RATE_HZ)):
            if monitor(state(65.0 + (0.02 if i % 2 else -0.02)), i / RATE_HZ):
                end = i
                break
        self.assertIsNotNone(end)
        self.assertEqual(monitor.reason, "velocity-capped")
        # The plateau is seen once the window spans 0.9 * 0.25 s, then settles for 0.5 s.
        self.assertGreaterEqual(end / RATE_HZ, 0.9 * 0.25 + cfg.saturation_settle_s - 1e-9)
        self.assertLessEqual(end / RATE_HZ, 0.25 + cfg.saturation_settle_s + 1e-9)

    def test_plan_steps_order(self) -> None:
        steps = experiment.plan_steps(self.cfg)
        self.assertEqual(
            [(s["trial"], s["torque_nm"], s["direction"]) for s in steps],
            [(1, 0.05, 1), (2, -0.05, -1), (3, 0.10, 1), (4, -0.10, -1),
             (5, 0.15, 1), (6, -0.15, -1)],
        )
        one_way = experiment.plan_steps(
            experiment.resolve(self.cfg, torques_nm=(0.15, 0.05), direction=-1))
        self.assertEqual([(s["trial"], s["torque_nm"], s["direction"]) for s in one_way],
                         [(1, -0.05, -1), (2, -0.15, -1)])

    def test_load_records_reads_the_2026_11_01_record(self) -> None:
        records = experiment.load_records(REPO_ROOT / self.cfg.loss_record)
        self.assertAlmostEqual(records["inertia_kg_m2"], 0.0003776, delta=1e-7)
        self.assertAlmostEqual(records["inertia_rel_sd"], 0.0263, delta=1e-4)
        self.assertAlmostEqual(records["loss_fit"]["a_nm"], 0.00443, delta=1e-5)
        self.assertAlmostEqual(records["loss_fit"]["b_nm_s"], 3.778e-05, delta=1e-8)
        self.assertAlmostEqual(records["loss_fit"]["c_nm_s2"], 9.695e-09, delta=1e-11)
        self.assertAlmostEqual(records["loss_fit"]["rmse_nm"], 0.000718, delta=1e-6)
        self.assertEqual(records["source"], self.cfg.loss_record)

    def test_load_records_rejects_missing_loss_fit(self) -> None:
        record = json.loads((REPO_ROOT / self.cfg.loss_record).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            missing = dict(record)
            del missing["loss_model"]
            path.write_text(json.dumps(missing))
            with self.assertRaises(ValueError):
                experiment.load_records(path)
            bad = json.loads(json.dumps(record))
            bad["loss_model"]["pooled"]["b_nm_s"] = math.inf
            path.write_text(json.dumps(bad).replace("Infinity", "1e999"))
            with self.assertRaises(ValueError):
                experiment.load_records(path)

    def test_expected_noload(self) -> None:
        self.assertAlmostEqual(experiment.expected_noload_rev_s(self.cfg, 12.0), 45.6)
        text = experiment.describe(self.cfg, 12.0, 45.6)
        for part in ("12 V", "0.05, 0.1, 0.15", "+1", "-1", "66.7 rev/s", "4002 rpm",
                     "0.8 V", "2736 rpm", "GO"):
            self.assertIn(part, text)


if __name__ == "__main__":
    unittest.main()
