import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from moteus_host.bench.analysis.summarize import analyze


FIELDS = (
    "host_time_s",
    "scheduled_time_s",
    "request_time_s",
    "response_time_s",
    "experiment_phase",
    "fault",
    "position_rev",
    "velocity_rev_s",
    "torque_nm",
    "q_current_a",
    "d_current_a",
    "bus_voltage_v",
    "controller_temperature_c",
)


class FrictionAnalysisTest(unittest.TestCase):
    def _create_run(self, root: Path, *, status: str = "completed") -> Path:
        run = root / "test_friction"
        run.mkdir()
        rows = []
        sample = 0
        for phase, direction in (("reverse", -1.0), ("forward", 1.0)):
            for index in range(100):
                position = -0.5 + (index + 0.5) / 100.0
                periodic = 0.002 * math.sin(2.0 * math.pi * 14.0 * position)
                friction = 0.01 * direction
                time_s = sample * 0.005
                rows.append(
                    {
                        "host_time_s": time_s + 0.001,
                        "scheduled_time_s": time_s,
                        "request_time_s": time_s + 0.0002,
                        "response_time_s": time_s + 0.001,
                        "experiment_phase": phase,
                        "fault": 0,
                        "position_rev": position,
                        "velocity_rev_s": 0.02 * direction,
                        "torque_nm": periodic + friction,
                        "q_current_a": 0.2 * direction,
                        "d_current_a": 0.1,
                        "bus_voltage_v": 24.2,
                        "controller_temperature_c": 26.0,
                    }
                )
                sample += 1
        metadata = {
            "run_id": "test_friction",
            "experiment": "friction",
            "status": status,
            "sample_count": len(rows),
            "resolved_config": {"sweep_center_rev": 0.0, "travel_rev": 1.0},
        }
        (run / "run.json").write_text(json.dumps(metadata))
        (run / "controller-config.txt").write_text(
            "motor.poles 14\nmotor.cogging_dq_scale 0\n"
        )
        with (run / "telemetry.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return run

    def test_writes_summary_map_and_svg_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._create_run(Path(directory))

            result = analyze(run)
            summary = json.loads(result.read_text())

            friction = summary["decomposition"]["directional_friction_nm"]
            self.assertAlmostEqual(friction["mean"], 0.01)
            self.assertEqual(
                summary["decomposition"]["dominant_spatial_harmonics"][0][
                    "cycles_per_revolution"
                ],
                14,
            )
            self.assertTrue((run / "friction-map.csv").is_file())
            self.assertTrue((run / "plots/torque-vs-position.svg").is_file())
            self.assertTrue((run / "plots/friction-decomposition.svg").is_file())
            self.assertTrue((run / "plots/velocity-vs-position.svg").is_file())

    def test_rejects_incomplete_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._create_run(Path(directory), status="safety_stop")
            with self.assertRaisesRegex(ValueError, "completed run"):
                analyze(run)
