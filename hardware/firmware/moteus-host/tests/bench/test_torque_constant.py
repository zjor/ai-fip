import asyncio
import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import moteus

from moteus_host.bench import cli
from moteus_host.bench.config import load_config
from moteus_host.bench.experiments import torque_constant
from moteus_host.bench.experiments.torque_constant import (
    Entry,
    format_note,
    i2t_cost,
    next_suggestion,
    parse_entry,
    parse_note,
    validate,
)
from moteus_host.bench.recorder import TELEMETRY_FIELDS
from moteus_host.bench.safety import SafetyStop


class FakeClock:
    now = 0.0

    def time(self):
        return self.now

    async def wait(self, deadline):
        self.now = max(self.now, deadline)


class FakeController:
    """Mode 10 while a torque command is fresh; q_current = gain * torque / KT + offset."""

    KT = 0.025  # matches motor.Kv 330.797 in the default test snapshot (8.27 / Kv)

    def __init__(self, clock, *, gain=1.0, offset=0.0, velocity=0.0,
                 fail_command=False, fail_stop=False, timeout_enabled=True, cancel_on_call=None,
                 calibrating_s=0.0):
        self.clock, self.gain, self.offset, self.velocity = clock, gain, offset, velocity
        self.fail_command, self.fail_stop = fail_command, fail_stop
        self.timeout_enabled, self.cancel_on_call = timeout_enabled, cancel_on_call
        self.calibrating_s = calibrating_s  # firmware mode 3 for this long after an enable
        self.currents, self.positions, self.stops = [], [], 0
        self.mode, self.q, self.expires = 0, 0.0, math.inf
        self.enabled_at = None

    def result(self):
        self.clock.now += 0.001
        if self.mode == 3 and self.clock.now - self.enabled_at >= self.calibrating_s:
            self.mode = 10
        if self.timeout_enabled and self.clock.now >= self.expires and self.mode in (9, 10):
            self.mode = 11
        q = self.q if self.mode in (9, 10) else self.offset
        values = dict(MODE=self.mode, FAULT=0, POSITION=0.0, VELOCITY=self.velocity,
                      TORQUE=self.KT * q, Q_CURRENT=q, D_CURRENT=0.0, VOLTAGE=24.0,
                      TEMPERATURE=25.0, HOME_STATE=1, TRAJECTORY_COMPLETE=1)
        return SimpleNamespace(values={getattr(moteus.Register, k): v for k, v in values.items()})

    async def set_current(self, *, d_A, q_A, query=False):
        self.currents.append((q_A, d_A))
        raise AssertionError("current mode has no firmware watchdog; never command it")

    async def set_position(self, **kwargs):
        self.positions.append(kwargs)
        if self.fail_command:
            raise TimeoutError("lost command response")
        if self.cancel_on_call is not None and len(self.positions) == self.cancel_on_call:
            raise asyncio.CancelledError()  # what asyncio.run delivers on Ctrl-C
        if self.mode == 0:
            self.mode, self.enabled_at = 3, self.clock.now  # enabling -> calibrating
        self.q = self.gain * kwargs["feedforward_torque"] / self.KT + self.offset
        self.expires = self.clock.now + kwargs["watchdog_timeout"]
        return self.result()

    @property
    def amps(self):
        """Commanded Q-current equivalents of every torque command, probe included."""
        return [round(p["feedforward_torque"] / self.KT, 3) for p in self.positions]

    async def set_stop(self, **kwargs):
        self.stops += 1
        if self.fail_stop:
            raise TimeoutError("stop response lost")
        reply = self.result()  # stale mode, like the real firmware
        self.mode = 0
        return reply

    async def query(self):
        return self.result()


class RunTest(unittest.IsolatedAsyncioTestCase):
    async def execute(self, answers, *, rise_time=False, expected_error=None,
                      snapshot="servo.default_timeout_s 0.1\nmotor.Kv 330.797\n", **overrides):
        controller_kwargs = {k: overrides.pop(k) for k in
                             ("velocity", "fail_command", "fail_stop", "timeout_enabled", "cancel_on_call",
                              "calibrating_s")
                             if k in overrides}
        config = load_config()
        defaults = dict(plateau_s=0.1, rest_s=0.05, baseline_s=0.02, rise_step_s=0.05, ramp_s=0.02)
        cfg = replace(config.torque_constant, **{**defaults, **overrides})
        config = replace(config, torque_constant=cfg)
        clock = FakeClock()
        controller = FakeController(clock, **controller_kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(torque_constant, "create_controller", return_value=(controller, moteus)),
                  patch.object(torque_constant, "snapshot_controller", AsyncMock(return_value=(snapshot, {}))),
                  patch.object(torque_constant.time, "perf_counter", clock.time),
                  patch.object(torque_constant, "wait_until", clock.wait),
                  patch("builtins.input", side_effect=list(answers))):
                if expected_error:
                    with self.assertRaises(expected_error):
                        await torque_constant.run(config, Path(directory), 1, rise_time=rise_time)
                else:
                    await torque_constant.run(config, Path(directory), 1, rise_time=rise_time)
            path = next(Path(directory).iterdir())
            metadata = json.loads((path / "run.json").read_text())
            with (path / "telemetry.csv").open() as source:
                rows = list(csv.DictReader(source))
            with (path / "events.csv").open() as source:
                events = list(csv.DictReader(source))
        return controller, metadata, rows, events

    async def test_plateau_session_records_current_and_readings(self):
        answers = ["100", "100", "checked", "LEVER", "0", "", "", "4", "204.5", "ok",
                   "-4", "203", "", "q"]
        controller, meta, rows, events = await self.execute(answers)
        self.assertEqual(meta["status"], "completed")
        self.assertTrue({0.0, 4.0, -4.0} <= set(controller.amps))  # ramp samples lie in between
        self.assertEqual(max(abs(a) for a in controller.amps), 4.0)
        self.assertEqual(controller.currents, [])
        self.assertTrue(all(p["kp_scale"] == p["kd_scale"] == p["ilimit_scale"] == 0
                            and p["watchdog_timeout"] == 0.1 and math.isnan(p["position"])
                            for p in controller.positions))
        self.assertTrue(all(abs(p["maximum_torque"] - 8.0 * controller.KT) < 1e-6
                            for p in controller.positions))
        names = [e["event"] for e in events]
        self.assertLess(names.index("watchdog_probe"), names.index("operator_confirmed"))
        self.assertEqual(next(e["value"] for e in events if e["event"] == "watchdog_probe"), "11")
        plateau_rows = [r for r in rows if r["experiment_phase"] == "plateau"]
        self.assertEqual(plateau_rows[0]["command_q_current_a"], "0.0")
        self.assertEqual(plateau_rows[0]["command_mode"], "torque")
        self.assertEqual(plateau_rows[0]["feedforward_torque_nm"], "0.0")
        four = [r for r in plateau_rows if r["command_q_current_a"] == "4.0"][-1]  # after the ramp
        self.assertAlmostEqual(float(four["feedforward_torque_nm"]), 0.1, places=5)
        self.assertEqual(four["command_d_current_a"], "0.0")
        probe = next(e for e in events if e["event"] == "watchdog_probe")
        self.assertAlmostEqual(float(parse_note(probe["note"])["firmware_kt_nm_per_a"]), 0.025, places=5)
        self.assertTrue(any(r["experiment_phase"] == "baseline" for r in rows))
        readings = [e for e in events if e["event"] == "scale_reading"]
        self.assertEqual([e["value"] for e in readings], ["", "204.5", "203.0"])
        self.assertEqual(parse_note(readings[0]["note"])["status"], "skipped")
        self.assertEqual(parse_note(readings[1]["note"])["side"], "positive")
        self.assertEqual(parse_note(readings[2]["note"])["side"], "negative")
        self.assertEqual([e["value"] for e in events if e["event"] == "radius_positive_mm"], ["100.0"])
        notes = [e["note"] for e in events if e["event"] == "operator_note"]
        self.assertEqual(notes, ["plateau=1; ok"])
        self.assertEqual(controller.stops, 1 + 1 + 1 + 3 + 1)  # connect, check, probe, plateaus, cleanup

    async def test_rejects_bad_entries_without_commanding(self):
        answers = ["100", "100", "", "LEVER", "9", "abc", "4 11", "q"]
        controller, meta, _, _ = await self.execute(answers)
        self.assertEqual(controller.amps, [0.0])  # the watchdog probe only
        self.assertEqual(meta["status"], "completed")

    async def test_budget_refuses_plateau(self):
        answers = ["100", "100", "", "LEVER", "8", "q"]
        controller, meta, _, _ = await self.execute(answers, session_i2t_budget_a2s=1.0)
        self.assertEqual(controller.amps, [0.0])
        self.assertEqual(meta["status"], "completed")

    async def test_blank_scale_reading_is_recorded_as_skipped(self):
        answers = ["100", "100", "", "LEVER", "2", "", "", "q"]
        _, _, _, events = await self.execute(answers)
        reading = next(e for e in events if e["event"] == "scale_reading")
        self.assertEqual(reading["value"], "")
        self.assertEqual(parse_note(reading["note"])["status"], "skipped")

    async def test_operator_decline_never_enables_motor(self):
        controller, meta, _, _ = await self.execute(["100", "100", "", "no"])
        self.assertEqual(controller.amps, [0.0])
        self.assertEqual(meta["status"], "aborted")

    async def test_lost_response_during_plateau_stops(self):
        controller, meta, _, _ = await self.execute(["100", "100", "", "LEVER", "4"],
                                                    fail_command=True, expected_error=TimeoutError)
        self.assertEqual(meta["status"], "safety_stop")
        self.assertEqual(controller.mode, 0)

    async def test_interrupt_during_plateau_stops_controller(self):
        # Cancelled on the third torque command: current is on, no per-plateau stop has run yet.
        answers = ["100", "100", "", "LEVER", "4"]
        controller, meta, _, _ = await self.execute(answers, cancel_on_call=3,
                                                    expected_error=asyncio.CancelledError)
        self.assertEqual(meta["status"], "failed")
        self.assertEqual(controller.mode, 0)
        self.assertEqual(controller.stops, 1 + 1 + 1 + 1)  # connect, check, probe, cleanup

    async def test_missing_firmware_watchdog_is_refused_by_probe(self):
        controller, meta, _, _ = await self.execute(["100", "100", ""], timeout_enabled=False,
                                                    expected_error=SafetyStop)
        self.assertEqual(meta["status"], "safety_stop")
        self.assertEqual(controller.amps, [0.0])
        self.assertEqual(controller.mode, 0)

    async def test_velocity_limit_stops_before_current(self):
        controller, meta, _, _ = await self.execute(["100", "100", "", "LEVER", "4"],
                                                    velocity=0.1, expected_error=SafetyStop)
        self.assertFalse([a for a in controller.amps if a != 0.0])
        self.assertEqual(meta["status"], "safety_stop")

    async def test_plateau_current_is_ramped_in(self):
        # Bench 2026-10-25: an instant 6 A step compressed the fixture 0.3 mm in 6 ms
        # and tripped the 0.05 rev/s stop; the ramp spreads that over ramp_s.
        answers = ["100", "100", "", "LEVER", "4", "86.4", "", "q"]
        controller, meta, rows, _ = await self.execute(answers, plateau_s=0.2, ramp_s=0.1)
        self.assertEqual(meta["status"], "completed")
        plateau = [r for r in rows if r["experiment_phase"] == "plateau"]
        torques = [float(r["feedforward_torque_nm"]) for r in plateau]
        self.assertLess(torques[0], 0.25 * 4.0 * controller.KT)
        self.assertTrue(all(a <= b + 1e-9 for a, b in zip(torques, torques[1:])))
        self.assertAlmostEqual(torques[-1], 4.0 * controller.KT, places=6)
        self.assertGreaterEqual(sum(1 for t in torques if abs(t - 4.0 * controller.KT) < 1e-6), len(torques) // 2)
        self.assertEqual({r["command_q_current_a"] for r in plateau}, {"4.0"})

    async def test_rise_time_steps_are_not_ramped(self):
        controller, meta, rows, _ = await self.execute(["LEVER", "0.03", "q"], rise_time=True, ramp_s=0.1)
        steps = [float(r["feedforward_torque_nm"]) for r in rows if r["experiment_phase"] == "step"]
        self.assertEqual(set(steps), {0.03})

    async def test_firmware_calibration_at_plateau_start_is_tolerated(self):
        # Real controller 2026-10-25: the second reply after an enable reported mode 3.
        answers = ["100", "100", "", "LEVER", "0.5", "14.2", "", "q"]
        controller, meta, rows, _ = await self.execute(answers, calibrating_s=0.01)
        self.assertEqual(meta["status"], "completed")
        modes = [r["mode"] for r in rows if r["experiment_phase"] == "plateau"]
        self.assertIn("3", modes)
        self.assertEqual(modes[-1], "10")

    async def test_calibration_that_never_ends_stops(self):
        controller, meta, _, _ = await self.execute(["100", "100", "", "LEVER", "0.5"],
                                                    calibrating_s=10.0, expected_error=SafetyStop)
        self.assertEqual(meta["status"], "safety_stop")
        self.assertIn("mode 3", meta["reason"])
        self.assertEqual(controller.mode, 0)

    async def test_missing_or_zero_motor_kv_is_refused(self):
        for snapshot in ("servo.default_timeout_s 0.1\n", "motor.Kv 0\n", "motor.Kv nan\n"):
            with self.subTest(snapshot=snapshot):
                controller, meta, _, _ = await self.execute(
                    ["100"], snapshot=snapshot, expected_error=SafetyStop)
                self.assertEqual(meta["status"], "safety_stop")
                self.assertEqual(controller.positions, [])

    async def test_rise_time_mode_sends_feedforward_steps(self):
        controller, meta, rows, events = await self.execute(["LEVER", "0.03", "-0.03", "q"], rise_time=True)
        self.assertEqual(meta["status"], "completed")
        steps = [p["feedforward_torque"] for p in controller.positions]
        self.assertEqual(sorted(set(steps)), [-0.03, 0.0, 0.03])  # 0.0 is the watchdog probe
        self.assertTrue(all(p["kp_scale"] == p["kd_scale"] == p["ilimit_scale"] == 0
                            for p in controller.positions))
        self.assertEqual(controller.currents, [])
        self.assertEqual(controller.amps[0], 0.0)  # the probe is a 0 Nm torque command
        self.assertFalse(any(e["event"] == "scale_reading" for e in events))
        self.assertTrue(any(r["experiment_phase"] == "step" for r in rows))
        self.assertEqual(meta["resolved_config"]["mode"], "rise_time")

    def test_cli_dry_run_does_not_connect(self):
        with patch.object(torque_constant, "create_controller") as create:
            self.assertEqual(cli.main(["torque-constant", "--dry-run"]), 0)
            self.assertEqual(cli.main(["torque-constant", "--dry-run", "--rise-time"]), 0)
            create.assert_not_called()


class ConfigTest(unittest.TestCase):
    def test_torque_constant_section_loads_from_tracked_toml(self):
        cfg = load_config().torque_constant
        self.assertEqual(cfg.max_current_a, 8.0)
        self.assertEqual(cfg.plateau_s, 3.0)
        self.assertEqual(cfg.max_rise_torque_nm, 0.05)

    def test_telemetry_fields_include_current_commands(self):
        self.assertIn("command_q_current_a", TELEMETRY_FIELDS)
        self.assertIn("command_d_current_a", TELEMETRY_FIELDS)


class HelperTest(unittest.TestCase):
    def setUp(self):
        config = load_config()
        self.cfg = config.torque_constant
        self.limits = config.safety

    def test_tracked_defaults_validate(self):
        validate(self.cfg, self.limits)

    def test_validate_rejects_unsafe_values(self):
        for change in (dict(max_current_a=9.0), dict(max_reported_torque_nm=0.5),
                       dict(max_rise_torque_nm=0.1), dict(sample_rate_hz=50.0),
                       dict(plateau_s=20.0), dict(rest_s=math.nan),
                       dict(max_velocity_rev_s=2.0), dict(rise_step_s=3.0)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(replace(self.cfg, **change), self.limits)

    def test_parse_entry_plateau_defaults_and_overrides(self):
        self.assertEqual(parse_entry("4", self.cfg, rise_time=False), Entry("plateau", 4.0, 3.0))
        self.assertEqual(parse_entry("-6 5", self.cfg, rise_time=False), Entry("plateau", -6.0, 5.0))
        self.assertEqual(parse_entry("-0", self.cfg, rise_time=False).value, 0.0)
        self.assertEqual(parse_entry(" q ", self.cfg, rise_time=False).kind, "quit")

    def test_parse_entry_rise_defaults(self):
        self.assertEqual(parse_entry("0.03", self.cfg, rise_time=True), Entry("step", 0.03, 0.5))
        self.assertEqual(parse_entry("-0.05 1", self.cfg, rise_time=True), Entry("step", -0.05, 1.0))

    def test_parse_entry_rejects_bad_input(self):
        for text in ("9", "abc", "4 11", "", "4 0", "1 2 3", "nan"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_entry(text, self.cfg, rise_time=False)
        for text in ("0.06", "0", "0.03 3"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_entry(text, self.cfg, rise_time=True)

    def test_i2t_cost(self):
        self.assertAlmostEqual(i2t_cost(Entry("plateau", 4.0, 3.0), self.cfg), 48.0)
        self.assertAlmostEqual(i2t_cost(Entry("step", 0.05, 0.5), self.cfg), (0.05 / 0.025) ** 2 * 0.5)

    def test_next_suggestion_walks_standard_sequence(self):
        self.assertEqual(next_suggestion({}, 1), 0.0)
        self.assertEqual(next_suggestion({0.0: 1}, 1), 0.5)
        self.assertEqual(next_suggestion({0.0: 1, 0.5: 3, 1.0: 2}, 1), 1.0)
        self.assertEqual(next_suggestion({0.0: 1}, -1), -0.5)
        full = {0.0: 1, **{v: 3 for v in (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)}}
        self.assertIsNone(next_suggestion(full, 1))

    def test_note_round_trip(self):
        note = format_note(plateau=3, current_a=4.0, duration_s=3.0)
        self.assertEqual(note, "plateau=3; current_a=4.0; duration_s=3.0")
        self.assertEqual(parse_note(note)["current_a"], "4.0")
        self.assertEqual(parse_note(""), {})


if __name__ == "__main__":
    unittest.main()
