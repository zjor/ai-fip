import math
import unittest

from app.analyze_torque_constant import fit_line, parse_note, rise_time, torque_from_grams


class CoreTest(unittest.TestCase):
    def test_torque_from_grams_at_100mm(self):
        self.assertAlmostEqual(torque_from_grams(1.0, 100.0), 0.000980665)
        self.assertAlmostEqual(torque_from_grams(204.0, 100.0), 0.2000557, places=6)

    def test_fit_line_recovers_slope_and_intercept(self):
        x = [0.5, 1, 2, 4, 6, 8]
        y = [0.025 * v + 0.001 for v in x]
        fit = fit_line(x, y)
        self.assertAlmostEqual(fit["slope"], 0.025)
        self.assertAlmostEqual(fit["intercept"], 0.001)
        self.assertAlmostEqual(fit["residual_rms"], 0.0)
        self.assertAlmostEqual(fit["slope_se"], 0.0)
        self.assertEqual(fit["n"], 6)

    def test_fit_line_reports_uncertainty_for_noisy_points(self):
        fit = fit_line([1, 2, 3, 4], [1.1, 1.9, 3.1, 3.9])
        self.assertGreater(fit["slope_se"], 0.0)
        self.assertGreater(fit["residual_rms"], 0.0)

    def test_fit_line_needs_three_points(self):
        with self.assertRaises(ValueError):
            fit_line([1, 2], [1, 2])

    def test_rise_time_resolves_exponential(self):
        tau = 0.004
        samples = [dict(response_time_s=0.0025 * (i + 1),
                        q_current_a=2.0 * (1 - math.exp(-0.0025 * (i + 1) / tau)))
                   for i in range(200)]
        r = rise_time(samples, 0.0)
        self.assertEqual(r["status"], "resolved")
        self.assertAlmostEqual(r["steady_a"], 2.0, places=2)
        self.assertEqual(r["t63_ms"], 5.0)
        self.assertEqual(r["t95_ms"], 12.5)
        self.assertEqual(r["resolution_ms"], 2.5)

    def test_rise_time_within_one_sample(self):
        samples = [dict(response_time_s=0.005 * (i + 1), q_current_a=-2.0) for i in range(20)]
        r = rise_time(samples, 0.0)
        self.assertEqual(r["status"], "within_one_sample")
        self.assertEqual(r["t95_ms"], 5.0)

    def test_rise_time_not_reached(self):
        samples = [dict(response_time_s=0.005 * (i + 1), q_current_a=0.1 * i) for i in range(10)]
        self.assertEqual(rise_time(samples, 0.0, settle_fraction=0.1)["status"], "resolved")
        flat = [dict(response_time_s=0.005 * (i + 1), q_current_a=0.0) for i in range(4)]
        self.assertEqual(rise_time(flat, 0.0)["status"], "not_reached")

    def test_parse_note(self):
        self.assertEqual(parse_note("plateau=3; current_a=4.0; side=positive")["side"], "positive")
        self.assertEqual(parse_note(""), {})


import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.analyze_torque_constant import analyze, main

G_PER_NM = 1.0 / 0.000980665
TELEMETRY = ("sample", "host_time_s", "scheduled_time_s", "request_time_s", "response_time_s",
             "experiment_phase", "trial_id", "command_mode", "command_position_rev",
             "command_velocity_rev_s", "command_q_current_a", "command_d_current_a",
             "feedforward_torque_nm", "maximum_torque_nm", "velocity_limit_rev_s",
             "accel_limit_rev_s2", "watchdog_timeout_s", "kp_scale", "kd_scale", "ilimit_scale",
             "mode", "fault", "position_rev", "velocity_rev_s", "torque_nm", "q_current_a",
             "d_current_a", "bus_voltage_v", "electrical_power_w", "controller_temperature_c",
             "motor_temperature_c", "home_state", "trajectory_complete")


def write_run(directory, *, plateaus, steps=(), kt=0.025, offset_nm=0.001, radius=100.0,
              skip_reading_for=(), status="completed", experiment="torque-constant", offset_a=0.0,
              gain=1.0):
    """Synthetic run: baseline + plateau rows at 200 Hz, step rows at 400 Hz."""
    run = Path(directory) / "20260101T000000Z_torque-constant"
    run.mkdir()
    rows, events, t = [], [], 0.0
    mode = "rise_time" if steps and not plateaus else "plateau"
    if mode == "plateau":
        for side in ("positive", "negative"):
            events.append(dict(host_time_s=t, event=f"radius_{side}_mm", value=radius, unit="mm", note=""))

    def add(phase, trial, period, count, q, torque, mode_code, extra):
        nonlocal t
        for _ in range(count):
            t += period
            rows.append(dict(sample=len(rows), host_time_s=t, scheduled_time_s=t, request_time_s=t - 0.001,
                             response_time_s=t, experiment_phase=phase, trial_id=trial, mode=mode_code,
                             fault=0, position_rev=0.0, velocity_rev_s=0.0, torque_nm=torque,
                             q_current_a=q, d_current_a=0.0, bus_voltage_v=24.0,
                             controller_temperature_c=25.0, home_state=1, trajectory_complete=1, **extra))

    trial = 0
    for amps in plateaus:
        add("baseline", trial, 0.005, 10, 0.0, 0.0, 0, dict(command_mode="query"))
        add("plateau", trial, 0.005, 60, gain * amps + offset_a, kt * amps, 9,
            dict(command_mode="current", command_q_current_a=amps, command_d_current_a=0.0))
        side = "zero" if amps == 0 else "positive" if amps > 0 else "negative"
        if trial in skip_reading_for:
            events.append(dict(host_time_s=t, event="scale_reading", value=None, unit=None,
                               note=f"plateau={trial}; current_a={amps}; side={side}; status=skipped"))
        elif amps != 0:
            grams = (kt * abs(amps) + offset_nm) * G_PER_NM
            events.append(dict(host_time_s=t, event="scale_reading", value=grams, unit="g",
                               note=f"plateau={trial}; current_a={amps}; side={side}"))
        trial += 1
    for torque in steps:
        add("baseline", trial, 0.0025, 10, 0.0, 0.0, 0, dict(command_mode="query"))
        add("step", trial, 0.0025, 40, torque / kt, torque, 10,
            dict(command_mode="torque", feedforward_torque_nm=torque))
        trial += 1
    with (run / "telemetry.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=TELEMETRY)
        writer.writeheader()
        writer.writerows(rows)
    with (run / "events.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("host_time_s", "event", "value", "unit", "note"))
        writer.writeheader()
        writer.writerows(events)
    (run / "controller-config.txt").write_text("motor.Kv 296.352\nservo.default_timeout_s 0.1\n")
    (run / "run.json").write_text(json.dumps(dict(
        run_id=run.name, experiment=experiment, status=status, reason=None, sample_count=len(rows),
        resolved_config=dict(mode=mode, sample_rate_hz=200.0, rise_sample_rate_hz=400.0))))
    return run


FULL = [0, 0.5, 1, 2, 4, 6, 8] + [0.5, 1, 2, 4, 6, 8] * 2 + [-0.5, -1, -2, -4, -6, -8] * 3


class AnalyzeTest(unittest.TestCase):
    def test_recovers_kt_in_both_directions(self):
        with tempfile.TemporaryDirectory() as d:
            result = analyze(write_run(d, plateaus=FULL))
        for side in ("positive", "negative"):
            self.assertAlmostEqual(abs(result["fits"][side]["physical"]["slope"]), 0.025, places=4)
            self.assertAlmostEqual(abs(result["fits"][side]["low_current"]["slope"]), 0.025, places=4)
            self.assertAlmostEqual(abs(result["fits"][side]["reported"]["slope"]), 0.025, places=6)
        self.assertTrue(all(result["acceptance"].values()), result["acceptance"])
        self.assertAlmostEqual(result["simulator"]["kt_nm_per_a"], 0.025, places=4)
        self.assertAlmostEqual(result["simulator"]["kt_firmware_nm_per_a"], 8.27 / 296.352, places=5)
        self.assertIsNone(result["simulator"]["torque_rate_limit"])
        self.assertEqual(result["radius_mm"], dict(positive=100.0, negative=100.0))
        self.assertEqual(result["plateaus"][0]["direction"], "zero")
        self.assertIsNone(result["plateaus"][0]["physical_torque_nm"])
        self.assertEqual(len(result["source"]["sha256"]), 4)

    def test_low_current_fit_selects_by_commanded_current(self):
        with tempfile.TemporaryDirectory() as d:
            result = analyze(write_run(d, plateaus=FULL, offset_a=-0.1))
        for side in ("positive", "negative"):
            self.assertEqual(result["fits"][side]["low_current"]["n"], 6)
        self.assertIsNotNone(result["simulator"]["kt_low_current_nm_per_a"])

    def test_commanded_fit_separates_loop_gain_from_kt(self):
        # Bench 2026-10-25: 0.5 A commanded gave 0.24 A measured, so the simulator
        # needs physical torque per commanded amp as well as per measured amp.
        with tempfile.TemporaryDirectory() as d:
            result = analyze(write_run(d, plateaus=FULL, gain=0.5))
        for side in ("positive", "negative"):
            self.assertAlmostEqual(abs(result["fits"][side]["commanded"]["slope"]), 0.025, places=6)
            self.assertAlmostEqual(abs(result["fits"][side]["physical"]["slope"]), 0.05, places=6)
            self.assertEqual(result["fits"][side]["commanded"]["n"], result["fits"][side]["physical"]["n"])
        self.assertAlmostEqual(result["simulator"]["kt_per_commanded_a_nm_per_a"], 0.025, places=6)

    def test_plateau_without_reading_is_excluded_from_physical_fit(self):
        with tempfile.TemporaryDirectory() as d:
            result = analyze(write_run(d, plateaus=[0.5, 1, 2, 4, 6, 8] * 2, skip_reading_for=(3,)))
        self.assertEqual(result["fits"]["positive"]["physical"]["n"], 11)
        self.assertEqual(result["fits"]["positive"]["reported"]["n"], 12)
        self.assertIsNone(result["plateaus"][3]["physical_torque_nm"])
        self.assertIsNone(result["plateaus"][3]["scale_g"])

    def test_acceptance_flags_insufficient_magnitudes(self):
        with tempfile.TemporaryDirectory() as d:
            result = analyze(write_run(d, plateaus=[1, 2, 4] * 3))
        self.assertFalse(result["acceptance"]["four_magnitudes_per_direction"])
        self.assertFalse(result["acceptance"]["kt_symmetry_within_5pct"])
        self.assertIsNotNone(result["fits"]["positive"]["physical"])
        self.assertIsNone(result["fits"]["negative"]["physical"])
        self.assertIsNone(result["simulator"]["kt_nm_per_a"])

    def test_rise_time_steps_report_within_one_sample(self):
        with tempfile.TemporaryDirectory() as d:
            result = analyze(write_run(d, plateaus=[], steps=[0.03, -0.03]))
        self.assertEqual(result["simulator"]["torque_rate_limit"], "none within one sample")
        self.assertEqual(len(result["rise_time"]), 2)
        self.assertEqual(result["rise_time"][1]["torque_nm"], -0.03)
        self.assertEqual(result["simulator"]["resolution_ms"], 2.5)
        self.assertIsNone(result["radius_mm"])

    def test_rejects_other_experiments(self):
        with tempfile.TemporaryDirectory() as d:
            run = write_run(d, plateaus=[1, 2, 4], experiment="response-timing")
            with self.assertRaises(ValueError):
                analyze(run)

    def test_main_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            run = write_run(d, plateaus=[1, 2, 4] * 3)
            output = Path(d) / "out" / "result.json"
            with patch.object(sys, "argv", ["analyze", str(run), "--output", str(output)]):
                main()
            self.assertIn("simulator", json.loads(output.read_text()))


if __name__ == "__main__":
    unittest.main()
