import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from moteus_host.bench import cli
from moteus_host.bench.analysis import envelope

R_OHM = 0.0467
L_H = 44.8e-6
KV = 330.0
KT = 0.0244
R_MAX = 0.90
VOLTAGES = (11.0, 12.0, 13.0, 14.0, 14.8, 15.6, 16.8)
KNEE_CURRENTS = (2.0, 4.0, 6.0)
NOISE_RPM = (0.6, -0.4, 0.2, -0.7, 0.5, -0.1, 0.3, -0.5, 0.4, -0.2, 0.1, -0.6, 0.7, -0.3)
CONTROLLER_CONFIG = (f"servo.max_velocity 500\nmotor.Kv {KV:g}\nmotor.resistance_ohm {R_OHM}\n"
                     f"motor.inductance_q_H {L_H}\nmotor.poles 14\n")


def _speed(v_bus, i_q, r_max=R_MAX):
    return envelope.circle_speed_rev_s(v_bus, i_q, kv_rpm_per_v=KV, r_max=r_max, r_ohm=R_OHM, l_h=L_H)


def _write_run(root, index, v_bus, *, capped=False, experiment="torque-speed", noload_iq=0.0, measured_offset_v=None):
    """Knee and no-load points on the R_MAX circle at v_bus. measured_offset_v: the knees and no-load
    points report a measured bus v_bus + offset (the knees report none when it is None)."""
    measured = v_bus + (measured_offset_v or 0.0)
    run_id = f"2026110{index}T120000Z_torque-speed"
    path = root / run_id
    path.mkdir()
    steps = []
    trial = 0
    for direction in (1, -1):
        for i_q in KNEE_CURRENTS:
            trial += 1
            knee = {"speed_rev_s": _speed(v_bus, i_q), "speed_se_rev_s": 0.1,
                    "q_current_a": direction * i_q, "tau_nm": KT * i_q}
            if measured_offset_v is not None:
                knee["bus_v"] = measured
            steps.append({"trial": trial, "torque_cmd_nm": direction * KT * (i_q + 1.0), "direction": direction,
                          "reason": "saturated", "low_speed": None, "windows_flagged": 0,
                          "knee": knee, "no_load": None})
        trial += 1
        noise = NOISE_RPM[(2 * index + (direction < 0)) % len(NOISE_RPM)] / 60.0
        no_load = ({"speed_rev_s": _speed(v_bus, 0.0) * 0.99, "lower_bound": True} if capped else
                   {"speed_rev_s": _speed(v_bus, noload_iq) + noise, "speed_sd_rev_s": 0.05, "lower_bound": False,
                    "q_current_a": noload_iq, "bus_v": measured, "loss_bias_rev_s": 0.0})
        steps.append({"trial": trial, "torque_cmd_nm": direction * 0.1, "direction": direction,
                      "reason": "velocity_capped" if capped else "saturated", "low_speed": None,
                      "knee": None, "no_load": no_load, "windows_flagged": 0})
    summary = {"schema_version": 1, "source": {"run_id": run_id, "experiment": experiment, "status": "complete"},
               "bus_v": v_bus, "records": {}, "steps": steps, "no_load_by_direction": {},
               "operating_envelope": {}, "skipped_trials": []}
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "run.json").write_text(json.dumps({"run_id": run_id, "experiment": experiment,
                                               "resolved_config": {"kt_nm_per_a": KT, "bus_v": v_bus}}))
    (path / "controller-config.txt").write_text(CONTROLLER_CONFIG)
    return path


class EnvelopeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _runs(self, **kwargs):
        return [_write_run(self.root, index, v, **kwargs) for index, v in enumerate(VOLTAGES)]

    def _summary(self, runs):
        return json.loads(envelope.analyze(runs, output_root=self.root / "out").read_text())

    def test_circle_speed_matches_feasibility_table(self):
        check = envelope.circle_speed_rev_s(12.0, 0.0, kv_rpm_per_v=330, r_max=0.94, r_ohm=0.0467, l_h=44.8e-6)
        self.assertAlmostEqual(check, 45.6, delta=0.1)
        for v_bus, rpm in ((12.0, 2740.0), (14.8, 3380.0), (16.8, 3840.0)):
            speed = envelope.circle_speed_rev_s(v_bus, 0.0, kv_rpm_per_v=330, r_max=0.94, r_ohm=0.0467, l_h=44.8e-6)
            self.assertLess(abs(60.0 * speed - rpm) / rpm, 0.01)

    def test_r_max_for_point_inverts_circle_speed(self):
        for v_bus in (11.0, 14.8, 16.8):
            for i_q in (0.0, 2.0, 6.0):
                f = _speed(v_bus, i_q, r_max=0.87)
                r_max = envelope.r_max_for_point(v_bus, i_q, f, kv_rpm_per_v=KV, r_ohm=R_OHM, l_h=L_H)
                self.assertAlmostEqual(r_max, 0.87, delta=1e-9)

    def test_fit_recovers_r_max_from_synthetic_summaries(self):
        result = self._summary(self._runs())["r_max"]
        firmware = result["kv_firmware"]
        self.assertEqual(firmware["kv"], KV)
        self.assertAlmostEqual(firmware["mean"], 0.90, delta=0.005)
        # knees and saturated no-load points are pooled
        self.assertEqual(firmware["n"], 2 * len(VOLTAGES) * (len(KNEE_CURRENTS) + 1))
        self.assertEqual(firmware["by_group"]["knee"]["n"], 2 * len(VOLTAGES) * len(KNEE_CURRENTS))
        self.assertEqual(firmware["by_group"]["no_load"]["n"], 2 * len(VOLTAGES))
        self.assertFalse(firmware["above_one"])
        from_kt = result["kv_from_kt"]
        self.assertAlmostEqual(from_kt["kv"], 8.27 / KT, places=6)
        self.assertGreater(abs(from_kt["mean"] - 0.90), 0.01)
        self.assertFalse(from_kt["above_one"])

    def test_no_load_points_at_the_loss_current_pool_with_knees(self):
        result = self._summary(self._runs(noload_iq=0.2))["r_max"]["kv_firmware"]
        self.assertAlmostEqual(result["mean"], 0.90, delta=0.005)
        self.assertAlmostEqual(result["by_group"]["no_load"]["mean"], 0.90, delta=0.005)
        self.assertAlmostEqual(result["by_group"]["knee"]["mean"], 0.90, delta=0.005)

    def test_kt_term_only_for_kv_from_kt(self):
        summary = self._summary(self._runs(noload_iq=0.2))
        self.assertEqual(summary["inputs"]["kt_rel_sd"], 0.03)
        from_kt = summary["r_max"]["kv_from_kt"]
        self.assertGreater(from_kt["kt_term"], 0.0)
        self.assertGreaterEqual(from_kt["se"], from_kt["kt_term"])
        self.assertAlmostEqual(from_kt["se"], math.hypot(from_kt["se_scatter"], from_kt["kt_term"]))
        # r_max is about proportional to 1/Kv, so the term is about 3 % of the mean
        self.assertAlmostEqual(from_kt["kt_term"] / from_kt["mean"], 0.03, delta=0.003)
        firmware = summary["r_max"]["kv_firmware"]
        self.assertIsNone(firmware["kt_term"])
        self.assertEqual(firmware["se"], firmware["se_scatter"])

    def test_measured_bus_replaces_the_nominal(self):
        nominal = self._summary(self._runs(noload_iq=0.2))
        self.assertTrue(all(k["bus_source"] == "nominal" for k in nominal["knees"]))
        self.assertTrue(all(p["bus_source"] == "measured" for p in nominal["no_load_line"]["points"]))
        (self.root / "low").mkdir()
        low = self._summary([_write_run(self.root / "low", index, v, noload_iq=0.2, measured_offset_v=-0.3)
                             for index, v in enumerate(VOLTAGES)])
        self.assertTrue(all(k["bus_source"] == "measured" for k in low["knees"]))
        self.assertEqual(sorted({round(p["bus_v"], 6) for p in low["no_load_line"]["points"]}),
                         sorted(round(v - 0.3, 6) for v in VOLTAGES))
        self.assertEqual([r["bus_v"] for r in low["runs"]], list(VOLTAGES))
        # the same speeds at a lower measured bus need a larger fraction of it
        for name in ("kv_firmware", "kv_from_kt"):
            for group in ("knee", "no_load"):
                self.assertGreater(low["r_max"][name]["by_group"][group]["mean"],
                                   nominal["r_max"][name]["by_group"][group]["mean"])

    def test_lower_bound_points_are_listed_not_fitted(self):
        runs = self._runs(capped=True)
        summary = self._summary(runs)
        self.assertIsNone(summary["no_load_line"]["fit"])
        self.assertEqual(summary["no_load_line"]["points"], [])
        self.assertEqual(len(summary["lower_bounds"]), 2 * len(VOLTAGES))
        self.assertTrue(all(p["lower_bound"] for p in summary["lower_bounds"]))
        mixed = self._summary(runs[:3] + [_write_run(self.root, 9, 15.0)])
        self.assertEqual(len(mixed["no_load_line"]["points"]), 2)
        self.assertEqual({p["bus_v"] for p in mixed["no_load_line"]["points"]}, {15.0})
        self.assertEqual(len(mixed["lower_bounds"]), 6)

    def test_no_load_line_slope(self):
        expected = 0.5 * R_MAX * 0.85 / (60.0 / (math.sqrt(3.0) * KV)) * 60.0
        fit = self._summary(self._runs())["no_load_line"]["fit"]
        self.assertEqual(fit["n"], 2 * len(VOLTAGES))
        self.assertLess(abs(fit["slope"] - expected), 2.0 * fit["slope_se"])
        self.assertLess(abs(fit["intercept"]), 2.0 * fit["intercept_se"])

    def test_rejects_non_torque_speed_summary(self):
        runs = self._runs()[:2] + [_write_run(self.root, 8, 15.0, experiment="coast-down")]
        with self.assertRaises(ValueError):
            envelope.analyze(runs, output_root=self.root / "out")

    def test_cli_envelope_writes_summary(self):
        runs = self._runs()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli.main(["--output-root", str(self.root / "out"), "envelope", *map(str, runs)])
        self.assertEqual(code, 0)
        text = stdout.getvalue().strip()
        self.assertTrue(text.startswith("Analysis written: "))
        output = Path(text.removeprefix("Analysis written: "))
        self.assertTrue(output.name == "summary.json" and output.parent.name.endswith("_envelope"))
        self.assertEqual(output.parent.parent, self.root / "out")
        self.assertTrue((output.parent / "plots" / "envelope.svg").exists())
        self.assertEqual([r["bus_v"] for r in json.loads(output.read_text())["runs"]], list(VOLTAGES))


if __name__ == "__main__":
    unittest.main()
