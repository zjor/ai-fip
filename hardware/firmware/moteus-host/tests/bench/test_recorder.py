import csv
import json
import tempfile
import unittest
from pathlib import Path

from moteus_host.bench.recorder import RunRecorder, TELEMETRY_FIELDS


class RunRecorderTest(unittest.TestCase):
    def test_writes_raw_files_and_final_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder(
                output_root=Path(directory),
                experiment="test",
                resolved_config={"sample_rate_hz": 20.0},
            )
            recorder.attach_controller("conf set id.id 1\n", {"model": 4})
            recorder.write_sample({"sample": 0, "command_mode": "stop"})
            recorder.event(0.0, "note", note="hello")
            path = recorder.path
            recorder.close("completed")

            metadata = json.loads((path / "run.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["sample_count"], 1)
            self.assertTrue((path / "controller-config.txt").exists())
            with (path / "telemetry.csv").open(newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(tuple(rows[0]), TELEMETRY_FIELDS)
            self.assertEqual(rows[0]["command_mode"], "stop")
