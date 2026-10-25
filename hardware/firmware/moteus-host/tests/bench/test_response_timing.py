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

from moteus_host.bench.config import load_config
from moteus_host.bench.experiments import response_timing as experiment
from moteus_host.bench.safety import SafetyStop
from moteus_host.bench.servo import create_controller


class FakeClock:
    now = 0.0

    def time(self):
        return self.now

    async def wait(self, deadline):
        self.now = max(self.now, deadline)


class FakeController:
    def __init__(self, clock, velocity=0.0, fail_pulse=False, fail_stop=False):
        self.clock, self.velocity, self.fail_pulse = clock, velocity, fail_pulse
        self.commands = []
        self.mode = 0
        self.expires = math.inf
        self.stops = 0
        self.fail_stop = fail_stop

    def result(self):
        self.clock.now += .001
        if self.clock.now >= self.expires and self.mode == 10:
            self.mode = 11
        values = dict(MODE=self.mode, FAULT=0, POSITION=0.0, VELOCITY=self.velocity,
                      TORQUE=0., Q_CURRENT=0., D_CURRENT=0., VOLTAGE=24., POWER=0.,
                      TEMPERATURE=25., HOME_STATE=1, TRAJECTORY_COMPLETE=1)
        return SimpleNamespace(values={getattr(moteus.Register, k): v for k, v in values.items()})

    async def set_stop(self, **kwargs):
        self.stops += 1
        if self.fail_stop and any(c['feedforward_torque'] for c in self.commands):
            raise TimeoutError('stop response lost')
        # Real firmware: the stop reply reports the mode sampled before the
        # stop takes effect; only the next query shows mode 0.
        reply = self.result()
        self.mode = 0
        return reply

    async def query(self):
        return self.result()

    async def set_position(self, **kwargs):
        self.commands.append(kwargs)
        if kwargs['feedforward_torque'] and self.fail_pulse:
            raise TimeoutError('lost pulse response')
        self.mode = 10
        self.expires = self.clock.now + kwargs['watchdog_timeout']
        return self.result()


class FakeStream:
    def __init__(self):
        self.mode = 12
        self.writes = []
        self.flushed = False

    async def flush_read(self):
        self.flushed = True

    async def command(self, command, allow_any_response=False):
        self.writes.append(command)
        if command.startswith(b'conf set'):
            self.mode = int(command.split()[-1])
            return b''
        # Real firmware answers `conf get` with the bare value and no `OK`;
        # moteus.Stream.command then waits forever unless allow_any_response.
        if not allow_any_response:
            raise TimeoutError('conf get never produces OK')
        return str(self.mode).encode()


class ResponseTimingTest(unittest.IsolatedAsyncioTestCase):
    async def execute(self, *, confirm='PULSE', velocity=0., fail_pulse=False, fail_stop=False, expected_error=None):
        cfg = load_config()
        cfg = replace(cfg, response_timing=replace(cfg.response_timing,
                      repetitions_per_direction=1, quiet_s=.1, quiet_timeout_s=.3,
                      pulse_torque_nm=.01, pulse_s=.02, max_velocity_rev_s=.25,
                      quiet_velocity_rev_s=.01))
        clock = FakeClock()
        controller = FakeController(clock, velocity, fail_pulse, fail_stop)
        stream = FakeStream()
        snapshot = AsyncMock(return_value=(
            'servo.timeout_mode 12\nservopos.position_min -1\nservopos.position_max 1\n', {}))
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(experiment, 'create_controller', return_value=(controller, moteus)),
                  patch.object(experiment, 'snapshot_controller', snapshot),
                  patch.object(experiment.time, 'perf_counter', clock.time),
                  patch.object(experiment, 'wait_until', clock.wait),
                  patch.object(moteus, 'Stream', return_value=stream),
                  patch('builtins.input', return_value=confirm)):
                if expected_error:
                    with self.assertRaises(expected_error):
                        await experiment.run(cfg, Path(directory), 1)
                else:
                    await experiment.run(cfg, Path(directory), 1)
            path = next(Path(directory).iterdir())
            metadata = json.loads((path / 'run.json').read_text())
            with (path / 'telemetry.csv').open() as source:
                rows = list(csv.DictReader(source))
        return controller, stream, metadata, rows

    async def test_success_uses_single_pulse_each_direction_and_restores(self):
        controller, stream, meta, rows = await self.execute()
        pulses = [c for c in controller.commands if c['feedforward_torque']]
        self.assertEqual([p['feedforward_torque'] for p in pulses], [.01, -.01])
        self.assertTrue(all(p['kp_scale'] == p['kd_scale'] == p['ilimit_scale'] == 0 for p in pulses))
        self.assertEqual(meta['status'], 'completed')
        self.assertEqual(stream.mode, 12)
        self.assertFalse(any(b'conf write' in c for c in stream.writes))
        self.assertTrue(all(r['command_mode'] == 'query' for r in rows if r['experiment_phase'] == 'observe'))
        self.assertTrue(any(r['mode'] == '11' for r in rows))

    async def test_operator_decline_never_enables_motor(self):
        controller, stream, meta, _ = await self.execute(confirm='no')
        self.assertFalse(controller.commands)
        self.assertFalse(stream.writes)
        self.assertEqual(meta['status'], 'aborted')

    async def test_unquiet_baseline_does_not_fire(self):
        controller, stream, meta, _ = await self.execute(velocity=.02, expected_error=SafetyStop)
        self.assertTrue(all(c['feedforward_torque'] == 0 for c in controller.commands))
        self.assertEqual(meta['status'], 'safety_stop')
        self.assertEqual(stream.mode, 12)

    async def test_lost_pulse_response_stops_and_restores(self):
        controller, stream, meta, _ = await self.execute(fail_pulse=True, expected_error=TimeoutError)
        self.assertEqual(meta['status'], 'safety_stop')
        self.assertEqual(controller.mode, 0)
        self.assertEqual(stream.mode, 12)

    async def test_nonfinite_telemetry_aborts_before_pulse(self):
        controller, _, meta, _ = await self.execute(velocity=math.nan, expected_error=SafetyStop)
        self.assertFalse(any(c['feedforward_torque'] for c in controller.commands))
        self.assertEqual(meta['status'], 'safety_stop')

    async def test_unconfirmed_stop_keeps_torque_off_timeout(self):
        _, stream, meta, _ = await self.execute(fail_stop=True, expected_error=TimeoutError)
        self.assertEqual(meta['status'], 'cleanup_failed')
        self.assertEqual(stream.mode, 0)

    async def test_speed_limit_stops_before_pulse(self):
        controller, stream, meta, _ = await self.execute(velocity=.3, expected_error=SafetyStop)
        self.assertFalse(any(c['feedforward_torque'] for c in controller.commands))
        self.assertEqual(stream.mode, 12)
        self.assertEqual(meta['status'], 'safety_stop')

    def test_hard_limits_reject_unsafe_config(self):
        cfg = load_config()
        for change in (dict(pulse_s=.2), dict(pulse_torque_nm=.1), dict(pulse_s=math.nan),
                       dict(repetitions_per_direction=1.5), dict(max_velocity_rev_s=1.)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                experiment.validate(replace(cfg.response_timing, **change), cfg.safety)

    def test_quiet_window_rejects_missing_observations(self):
        cfg = load_config().response_timing
        rows = [dict(response_time_s=t, position_rev=0., velocity_rev_s=0., mode=10)
                for t in (0., .005, .25)]
        self.assertFalse(experiment.quiet_window(rows, cfg))

    def test_torque_packet_fits_can_fd(self):
        controller, _ = create_controller(1, timing=True)
        command = controller.make_position(**experiment.torque_arguments(.01, load_config().response_timing))
        self.assertLessEqual(len(command.data), 64)


if __name__ == '__main__':
    unittest.main()
