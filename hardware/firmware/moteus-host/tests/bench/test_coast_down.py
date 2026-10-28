import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import moteus

from moteus_host.bench import cli
from moteus_host.bench.config import load_config
from moteus_host.bench.experiments import coast_down as experiment
from moteus_host.bench.safety import SafetyStop


INERTIA = 3.54e-4
KT = 0.0244
SNAPSHOT = ("servo.timeout_mode 12\nservopos.position_min -1\nservopos.position_max 1\n"
            "servo.max_velocity 500\nmotor.Kv 330\n")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    async def wait(self, deadline):
        self.now = max(self.now, deadline)


class FakeController:
    """Integrates disc speed from the last command: velocity mode ramps at accel_limit,
    torque mode accelerates by torque/(2*pi*I), stopped decays by friction."""

    def __init__(self, clock, *, velocity=0.0, friction_nm=0.004, fail_spin_up=False):
        self.clock = clock
        self.velocity = velocity
        self.friction = friction_nm
        self.fail_spin_up = fail_spin_up
        self.mode = 0
        self.command = None
        self.commands = []
        self.stops = 0
        self.last_t = clock.now

    def _integrate(self):
        dt = self.clock.now - self.last_t
        self.last_t = self.clock.now
        if dt <= 0.0:
            return
        if self.mode == 10 and self.command is not None:
            if math.isnan(self.command["position"]) and self.command.get("feedforward_torque", 0.0):
                self.velocity += self.command["feedforward_torque"] / (2 * math.pi * INERTIA) * dt
            else:
                # a failing spin-up reaches only 30 % of the target (supply current-limited)
                target = self.command["velocity"] * (0.3 if self.fail_spin_up else 1.0)
                step = self.command["accel_limit"] * dt
                self.velocity += max(-step, min(step, target - self.velocity))
        else:
            decay = self.friction / (2 * math.pi * INERTIA) * dt
            self.velocity = math.copysign(max(abs(self.velocity) - decay, 0.0), self.velocity)

    def result(self):
        self.clock.now += 0.001
        self._integrate()
        q = 0.0
        if self.mode == 10 and self.command is not None:
            q = self.command.get("feedforward_torque", 0.0) / KT or 0.5
        values = dict(MODE=self.mode, FAULT=0, POSITION=1.0, VELOCITY=self.velocity,
                      TORQUE=q * KT, Q_CURRENT=q, D_CURRENT=0.0, VOLTAGE=24.0, POWER=0.0,
                      TEMPERATURE=25.0, HOME_STATE=1, TRAJECTORY_COMPLETE=1)
        return SimpleNamespace(values={getattr(moteus.Register, k): v for k, v in values.items()})

    async def set_stop(self, **kwargs):
        self.stops += 1
        reply = self.result()
        self.mode = 0
        self.command = None
        return reply

    async def query(self):
        return self.result()

    async def set_position(self, **kwargs):
        self.commands.append(kwargs)
        self.command = kwargs
        self.mode = 10
        return self.result()


class FakeStream:
    def __init__(self):
        self.settings = {b"servo.timeout_mode": b"12", b"servopos.position_min": b"-1",
                         b"servopos.position_max": b"1", b"servo.max_velocity": b"500"}
        self.writes = []
        self.flushed = False

    async def flush_read(self):
        self.flushed = True

    async def command(self, command, allow_any_response=False):
        self.writes.append(command)
        words = command.split()
        if words[:2] == [b"conf", b"write"]:
            raise AssertionError("conf write is forbidden")
        if words[:2] == [b"conf", b"set"]:
            self.settings[words[2]] = words[3]
            return b""
        if words[:2] == [b"conf", b"get"]:
            if not allow_any_response:
                raise TimeoutError("conf get never produces OK")
            return self.settings[words[2]]
        raise AssertionError(f"unexpected stream command {command!r}")


class CoastDownExperimentTest(unittest.IsolatedAsyncioTestCase):
    def config(self, **overrides):
        config = load_config()
        return replace(config, coast_down=replace(
            config.coast_down, ladder_rpm=(60.0, 120.0), plateau_s=0.05, coast_stop_rev_s=0.2,
            spin_up_timeout_s=0.5, coast_timeout_s=30.0, accel_target_rev_s=5.0,
            accel_timeout_s=2.0, **overrides))

    async def execute(self, config, *, inputs=("GO", "GO", "GO", "GO", "GO", "GO"),
                      controller=None, interrupt_in_coast=False, expected_error=None, **kwargs):
        clock = FakeClock()
        controller = controller or FakeController(clock)
        stream = FakeStream()
        answers = iter(inputs)
        state = {"interrupted": False}

        async def wait(deadline):
            if interrupt_in_coast and controller.stops >= 1 and not state["interrupted"]:
                state["interrupted"] = True
                raise KeyboardInterrupt
            await clock.wait(deadline)

        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(experiment, "create_controller", return_value=(controller, moteus)),
                  patch.object(experiment, "snapshot_controller", AsyncMock(return_value=(SNAPSHOT, {}))),
                  patch.object(experiment.time, "perf_counter", clock.time),
                  patch.object(experiment, "wait_until", wait),
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
                events = [row["event"] for row in csv.DictReader(source)]
        return controller, stream, metadata, rows, events

    def phases(self, rows, trial):
        return {r["experiment_phase"] for r in rows if r["trial_id"] == str(trial)}

    async def test_ladder_records_phases_and_restores_overrides(self):
        controller, stream, meta, rows, events = await self.execute(self.config())
        self.assertEqual(meta["status"], "completed")
        for trial in (0, 1):
            self.assertTrue({"spin_up", "plateau", "coast"} <= self.phases(rows, trial), trial)
        self.assertEqual([s["target_rev_s"] for s in meta["resolved_config"]["steps"]], [1.0, 2.0])
        record = json.loads((experiment.REPO_ROOT / "docs/hardware/measurements/2026-10-28-test-disc.json").read_text())
        self.assertEqual(meta["resolved_config"]["inertia"]["I_z_kg_m2"], record["assembly"]["I_z_kg_m2"])
        self.assertAlmostEqual(record["assembly"]["I_z_kg_m2"], 3.54e-4, delta=1e-6)
        self.assertEqual(stream.settings[b"servo.timeout_mode"], b"12")
        self.assertEqual(stream.settings[b"servopos.position_min"], b"-1")
        self.assertEqual(stream.settings[b"servopos.position_max"], b"1")
        self.assertEqual(stream.settings[b"servo.max_velocity"], b"500")
        self.assertIn("overrides_set", events)
        self.assertIn("overrides_restored", events)
        self.assertFalse(any(b"conf write" in w for w in stream.writes))
        # every velocity command stays under the spin-up torque cap and the Tier B velocity cap
        velocity_commands = [c for c in controller.commands if not c.get("feedforward_torque")]
        self.assertTrue(velocity_commands)
        self.assertTrue(all(c["maximum_torque"] == 0.10 and c["velocity_limit"] == 55.0
                            for c in velocity_commands))
        self.assertTrue(all(c["watchdog_timeout"] == 0.1 for c in controller.commands))
        # the coast is a stop, never a braking command
        coast_rows = [r for r in rows if r["experiment_phase"] == "coast"]
        self.assertTrue(coast_rows)
        self.assertTrue(all(r["command_mode"] == "query" for r in coast_rows))
        self.assertLess(abs(float(coast_rows[-1]["velocity_rev_s"])), 0.2)

    async def test_timeout_mode_cleared_before_first_motion(self):
        controller, stream, *_ = await self.execute(self.config())
        first_set = next(i for i, w in enumerate(stream.writes) if w == b"conf set servo.timeout_mode 0")
        self.assertGreaterEqual(first_set, 0)
        # all overrides were written before the controller saw any set_position
        override_writes = [w for w in stream.writes if w.startswith(b"conf set")]
        self.assertEqual(len([w for w in override_writes[:4] if b"timeout_mode 0" in w
                              or b"position_min nan" in w or b"position_max nan" in w
                              or b"max_velocity 55" in w]), 4)

    async def test_operator_can_end_after_first_step(self):
        controller, stream, meta, rows, events = await self.execute(self.config(), inputs=("GO", "q"))
        self.assertEqual(meta["status"], "completed")
        self.assertEqual({r["trial_id"] for r in rows if r["trial_id"]}, {"0"})
        self.assertIn("operator_ended", events)

    async def test_decline_first_step_aborts(self):
        controller, stream, meta, rows, events = await self.execute(self.config(), inputs=("no",))
        self.assertEqual(meta["status"], "aborted")
        self.assertEqual(controller.commands, [])
        self.assertEqual(stream.settings[b"servo.timeout_mode"], b"12")

    async def test_spin_up_timeout_stops_and_waits_for_standstill(self):
        clock = FakeClock()
        controller = FakeController(clock, fail_spin_up=True)
        controller, stream, meta, rows, events = await self.execute(
            self.config(), controller=controller, expected_error=SafetyStop)
        self.assertEqual(meta["status"], "safety_stop")
        self.assertIn("spin-up timeout", meta["reason"])
        waits = [r for r in rows if r["experiment_phase"] == "standstill_wait"]
        self.assertTrue(waits)
        self.assertEqual(rows[-1]["experiment_phase"], "standstill_wait")
        self.assertLess(abs(float(rows[-1]["velocity_rev_s"])), 0.2)
        self.assertGreaterEqual(controller.stops, 1)
        self.assertEqual(stream.settings[b"servo.timeout_mode"], b"12")
        self.assertEqual(stream.settings[b"servo.max_velocity"], b"500")

    async def test_accelerate_mode_uses_zero_gain_torque_steps(self):
        config = self.config(accel_torques_nm=(0.05,))
        controller, stream, meta, rows, events = await self.execute(config, accelerate=True)
        self.assertEqual(meta["status"], "completed")
        steps = meta["resolved_config"]["steps"]
        self.assertEqual([s["kind"] for s in steps], ["accel", "accel"])
        self.assertEqual([s["torque_nm"] for s in steps], [0.05, -0.05])
        torques = [c for c in controller.commands if c.get("feedforward_torque")]
        self.assertTrue(torques)
        self.assertEqual({c["feedforward_torque"] for c in torques}, {0.05, -0.05})
        self.assertTrue(all(c["kp_scale"] == c["kd_scale"] == c["ilimit_scale"] == 0 for c in torques))
        self.assertTrue(all(c["maximum_torque"] == 0.05 for c in torques))
        for trial in (0, 1):
            self.assertTrue({"accel", "coast"} <= self.phases(rows, trial), trial)
        accel_rows = [r for r in rows if r["experiment_phase"] == "accel" and r["trial_id"] == "0"]
        self.assertGreaterEqual(abs(float(accel_rows[-1]["velocity_rev_s"])), 5.0 - 0.1)

    async def test_keyboard_interrupt_during_coast_still_restores(self):
        controller, stream, meta, rows, events = await self.execute(
            self.config(), interrupt_in_coast=True, expected_error=KeyboardInterrupt)
        self.assertEqual(meta["status"], "interrupted")
        self.assertGreaterEqual(controller.stops, 2)
        self.assertEqual(stream.settings[b"servo.timeout_mode"], b"12")
        self.assertIn("overrides_restored", events)
        self.assertLess(abs(float(rows[-1]["velocity_rev_s"])), 0.2)

    async def test_ladder_override_above_cap_rejected(self):
        create = Mock()
        with patch.object(experiment, "create_controller", create):
            with self.assertRaisesRegex(ValueError, "ladder"):
                await experiment.run(self.config(), Path("/nonexistent"), 1, ladder_rpm=(4000.0,))
        create.assert_not_called()

    def test_plan_steps_and_arguments(self):
        cfg = self.config().coast_down
        steps = experiment.plan_steps(cfg, accelerate=False)
        self.assertEqual([(s["trial"], s["kind"], s["target_rev_s"]) for s in steps],
                         [(0, "ladder", 1.0), (1, "ladder", 2.0)])
        steps = experiment.plan_steps(replace(cfg, direction=-1), accelerate=True)
        self.assertEqual([s["target_rev_s"] for s in steps], [5.0, -5.0] * 3)
        self.assertEqual([s["torque_nm"] for s in steps], [0.05, -0.05, 0.10, -0.10, 0.15, -0.15])
        args = experiment.velocity_arguments(-2.0, cfg, 0.1)
        self.assertTrue(math.isnan(args["position"]))
        self.assertEqual((args["velocity"], args["maximum_torque"], args["accel_limit"]), (-2.0, 0.10, 20.0))
        args = experiment.torque_arguments(-0.05, cfg, 0.1)
        self.assertEqual((args["feedforward_torque"], args["maximum_torque"], args["kp_scale"]), (-0.05, 0.05, 0.0))

    def test_cli_dry_run_prints_description(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main(["coast-down", "--dry-run", "--ladder", "20,50,100"])
        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("free coast", text)
        self.assertIn("20", text)
        self.assertIn("100", text)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["coast-down", "--dry-run", "--ladder", "20,abc"])


if __name__ == "__main__":
    unittest.main()
