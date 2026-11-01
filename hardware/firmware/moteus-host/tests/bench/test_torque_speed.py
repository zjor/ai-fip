import asyncio
import contextlib
import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import moteus

from moteus_host.bench import cli
from moteus_host.bench.config import load_config
from moteus_host.bench.experiments import tier_b
from moteus_host.bench.experiments import torque_speed as experiment
from moteus_host.bench.safety import SafetyStop

from .tier_b_fakes import SNAPSHOT, FakeClock, FakeController, FakeStream


class TorqueSpeedExperimentTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # the fake clock never sleeps, so a 20 s simulated coast is one long callback;
        # keep the debug loop from reporting it as slow
        asyncio.get_running_loop().slow_callback_duration = 5.0

    def config(self, safety=None, **overrides):
        config = load_config()
        values = dict(torques_nm=(0.10,), directions=(1,))
        values.update(overrides)
        config = replace(config, torque_speed=replace(config.torque_speed, **values))
        if safety:
            config = replace(config, safety=replace(config.safety, **safety))
        return config

    async def execute(self, config, *, bus_v=12.0, bus=None, inputs=("GO",) * 8, controller=None,
                      expected_error=None, stream=None, **kwargs):
        """bus_v is the fake's supply; bus is --bus (defaults to the same value)."""
        kwargs["bus_v"] = bus_v if bus is None else bus
        controller = controller or FakeController(FakeClock(), bus_v=bus_v)
        clock = controller.clock          # the experiment's perf_counter is the fake's physics clock
        stream = stream or FakeStream()
        answers = iter(inputs)
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(tier_b, "create_controller", return_value=(controller, moteus)),
                  patch.object(tier_b, "snapshot_controller", AsyncMock(return_value=(SNAPSHOT, {}))),
                  patch.object(tier_b.time, "perf_counter", clock.time),
                  patch.object(tier_b, "wait_until", clock.wait),
                  patch.object(moteus, "Stream", return_value=stream),
                  patch("builtins.input", lambda prompt="": next(answers)),
                  contextlib.redirect_stdout(io.StringIO())):
                if expected_error:
                    with self.assertRaises(expected_error):
                        await experiment.run(config, Path(directory), 1, **kwargs)
                else:
                    await experiment.run(config, Path(directory), 1, **kwargs)
            path = next(Path(directory).iterdir())
            metadata = json.loads((path / "run.json").read_text())
            with (path / "telemetry.csv").open() as source:
                rows = list(csv.DictReader(source))
            with (path / "events.csv").open() as source:
                events = list(csv.DictReader(source))
        return controller, stream, metadata, rows, events

    def phases(self, rows, trial):
        return {r["experiment_phase"] for r in rows if r["trial_id"] == str(trial)}

    def names(self, events):
        return [e["event"] for e in events]

    async def test_saturated_step_records_torque_hold_coast_and_reason(self):
        controller = FakeController(FakeClock(), bus_v=12.0, no_load_rev_s=40.0)
        controller, stream, meta, rows, events = await self.execute(
            self.config(), controller=controller, bus_v=12.0)
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(self.phases(rows, 1), {"preflight", "torque", "hold", "coast"})
        self.assertEqual(meta["resolved_config"]["steps"][0]["reason"], "saturated")
        self.assertIn("hold_start", self.names(events))
        step_end = next(e for e in events if e["event"] == "step_end")
        self.assertEqual(step_end["note"], "saturated")
        resolved = meta["resolved_config"]
        self.assertEqual(resolved["bus_v"], 12.0)
        self.assertFalse(resolved["allow_any_bus"])
        self.assertEqual(resolved["derate_rev_s"], 2.0)
        self.assertIn("inertia_kg_m2", resolved["records"])
        for key in ("sag_threshold_v", "kt_nm_per_a"):
            self.assertIn(key, resolved)
        # the hold keeps the same torque command; nothing below it is ever sent
        torques = {c["feedforward_torque"] for c in controller.commands}
        self.assertEqual(torques, {0.10})
        names = self.names(events)
        for name in ("operator_confirmed", "step_start", "step_end", "hold_start", "coast_start", "step_complete"):
            self.assertIn(name, names)
        self.assertLess(names.index("step_end"), names.index("hold_start"))
        self.assertLess(names.index("hold_start"), names.index("coast_start"))

    async def test_velocity_capped_step_has_no_hold(self):
        controller = FakeController(FakeClock(), bus_v=12.0, no_load_rev_s=80.0)
        controller, stream, meta, rows, events = await self.execute(self.config(), controller=controller)
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(meta["resolved_config"]["steps"][0]["reason"], "velocity-capped")
        self.assertNotIn("hold", self.phases(rows, 1))
        self.assertNotIn("hold_start", self.names(events))
        self.assertLessEqual(max(abs(float(r["velocity_rev_s"])) for r in rows), 66.7)

    async def test_supply_limited_step_stops_cleanly(self):
        # q = 0.18 / 0.0244 = 7.4 A: above the 5 A sag point, under the 8 A phase-current stop
        config = self.config(torques_nm=(0.18,), directions=(1, -1))
        controller = FakeController(FakeClock(), bus_v=12.0, no_load_rev_s=40.0,
                                    sag_above_a=5.0, sag_to_v=10.6)
        controller, stream, meta, rows, events = await self.execute(config, controller=controller)
        self.assertEqual(meta["status"], "completed")
        steps = meta["resolved_config"]["steps"]
        self.assertEqual([s["reason"] for s in steps], ["supply-limited", "supply-limited"])
        self.assertNotIn("hold_start", self.names(events))
        # two stops at connect, two per step (pre-step check and coast), one at exit:
        # no retries and no extra safety stops
        self.assertEqual(controller.stops, 2 * len(steps) + 3)

    async def test_stuck_rotor_times_out_cleanly(self):
        config = self.config(step_timeout_s=0.5, saturation_settle_s=0.25)
        controller = FakeController(FakeClock(), bus_v=12.0, no_load_rev_s=40.0, stuck=True)
        controller, stream, meta, rows, events = await self.execute(config, controller=controller)
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(meta["resolved_config"]["steps"][0]["reason"], "timeout")
        self.assertNotIn("hold", self.phases(rows, 1))

    async def test_bus_mismatch_refuses_before_overrides(self):
        controller, stream, meta, rows, events = await self.execute(
            self.config(), bus_v=12.0, bus=13.2, expected_error=SafetyStop)
        self.assertEqual(meta["status"], "safety_stop")
        self.assertIn("bus 12.0 V is not 13.2 ± 0.2 V; set the supply first", meta["reason"])
        self.assertFalse(any(w.startswith(b"conf set") for w in stream.writes))
        self.assertEqual(controller.commands, [])

    async def test_bus_within_granularity_passes(self):
        controller = FakeController(FakeClock(), bus_v=11.9, no_load_rev_s=40.0)
        controller, stream, meta, rows, events = await self.execute(self.config(), controller=controller, bus=12.0)
        self.assertEqual(meta["status"], "completed")
        self.assertIn(b"conf set servo.max_velocity 66.7", stream.writes)

    async def test_unlisted_bus_rejected_without_flag(self):
        create = Mock()
        with (patch.object(tier_b, "create_controller", create), tempfile.TemporaryDirectory() as directory,
              contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaises(ValueError):
                await experiment.run(self.config(), Path(directory), 1, bus_v=12.5)
            self.assertFalse(any(Path(directory).iterdir()))
        create.assert_not_called()
        controller = FakeController(FakeClock(), bus_v=12.5, no_load_rev_s=40.0)
        controller, stream, meta, rows, events = await self.execute(
            self.config(), controller=controller, bus=12.5, allow_any_bus=True)
        self.assertEqual(meta["status"], "completed")
        self.assertTrue(meta["resolved_config"]["allow_any_bus"])

    async def test_overrides_use_66_7_and_restore(self):
        controller = FakeController(FakeClock(), bus_v=12.0, no_load_rev_s=40.0)
        controller, stream, meta, rows, events = await self.execute(self.config(), controller=controller)
        self.assertIn(b"conf set servo.max_velocity 66.7", stream.writes)
        self.assertEqual(stream.settings[b"servo.max_velocity"], b"500")
        self.assertEqual(stream.settings[b"servo.timeout_mode"], b"12")
        self.assertFalse(any(b"conf write" in w for w in stream.writes))
        self.assertIn("overrides_restored", self.names(events))

    def test_ramp_warning_on_skipped_voltage(self):
        cfg = load_config().torque_speed
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (volts, status) in enumerate(((11.0, "completed"), (12.0, "completed"),
                                                     (13.2, "safety_stop"))):
                run_dir = root / f"20261101T10000{index}Z_torque-speed"
                run_dir.mkdir()
                (run_dir / "run.json").write_text(json.dumps(
                    {"status": status, "experiment": "torque-speed", "resolved_config": {"bus_v": volts}}))
            warning = experiment.ramp_warning(root, 14.8, cfg)
            self.assertIsNotNone(warning)
            self.assertIn("13.2", warning)
            self.assertIsNone(experiment.ramp_warning(root, 13.2, cfg))
            self.assertIsNone(experiment.ramp_warning(root, 12.0, cfg))
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(experiment.ramp_warning(Path(directory), 11.0, cfg))
            self.assertIn("11", experiment.ramp_warning(Path(directory), 12.0, cfg))

    def test_cli_dry_run_prints_bus_and_torques(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main(["torque-speed", "--dry-run", "--bus", "12.0"])
        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertTrue(text.startswith("Dry run (no connection): Torque-speed at 12 V"), text)
        self.assertIn("12 V", text)
        self.assertIn("0.05, 0.1, 0.15, 0.18", text)

    def test_cli_requires_bus(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["torque-speed", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
