import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import moteus

from moteus_host.bench.config import load_config
from moteus_host.bench.experiments import friction, inspect, position_step


class FakeResult:
    def __init__(self, position: float = 0.25) -> None:
        self.values = {
            moteus.Register.MODE: 0,
            moteus.Register.FAULT: 0,
            moteus.Register.POSITION: position,
            moteus.Register.VELOCITY: 0.0,
            moteus.Register.TORQUE: 0.0,
            moteus.Register.Q_CURRENT: 0.0,
            moteus.Register.D_CURRENT: 0.0,
            moteus.Register.VOLTAGE: 24.0,
            moteus.Register.POWER: 0.0,
            moteus.Register.TEMPERATURE: 25.0,
            moteus.Register.MOTOR_TEMPERATURE: math.nan,
            moteus.Register.HOME_STATE: 1,
            moteus.Register.TRAJECTORY_COMPLETE: 1,
        }


class FakeController:
    def __init__(self) -> None:
        self.position = 0.25
        self.stop_count = 0

    async def set_stop(self) -> None:
        self.stop_count += 1

    async def query(self) -> FakeResult:
        return FakeResult(self.position)

    async def set_position(self, **kwargs: float) -> FakeResult:
        self.position = kwargs["position"]
        return FakeResult(self.position)


class ExperimentTest(unittest.IsolatedAsyncioTestCase):
    def test_friction_chooses_direction_that_fits_controller_bounds(self) -> None:
        bounds = friction.parse_position_bounds(
            "servopos.position_min -1\nservopos.position_max 1\n"
        )

        start, destination, direction = friction.plan_centered_sweep(
            0.217, 1.0, 0.0, *bounds, 0.02, 1.1
        )

        self.assertAlmostEqual(start, 0.5)
        self.assertAlmostEqual(destination, -0.5)
        self.assertEqual(direction, -1)

    def test_friction_rejects_sweep_that_cannot_fit_bounds(self) -> None:
        with self.assertRaisesRegex(friction.SafetyStop, "endpoint"):
            friction.plan_centered_sweep(0.0, 1.0, 0.0, -0.5, 0.5, 0.02, 1.1)

    def test_friction_rejects_start_outside_controller_bounds(self) -> None:
        with self.assertRaisesRegex(friction.SafetyStop, "starting position"):
            friction.plan_centered_sweep(1.002, 1.0, 0.0, -1.0, 1.0, 0.02, 1.1)

    async def test_inspect_records_stopped_sample_and_stops_controller(self) -> None:
        config = load_config()
        config = replace(
            config,
            inspect=replace(config.inspect, duration_s=0.01, sample_rate_hz=100.0),
        )
        controller = FakeController()
        snapshot = AsyncMock(return_value=("config\n", {"model": 4}))

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    inspect,
                    "create_controller",
                    return_value=(controller, moteus),
                ),
                patch.object(inspect, "snapshot_controller", snapshot),
            ):
                path = Path(await inspect.run(config, Path(directory), 1))

            metadata = json.loads((path / "run.json").read_text())
            with (path / "telemetry.csv").open(newline="") as source:
                rows = list(csv.DictReader(source))

        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["command_mode"], "stop")
        self.assertGreaterEqual(controller.stop_count, 2)

    async def test_position_step_records_motion_and_returns_to_origin(self) -> None:
        config = load_config()
        config = replace(
            config,
            position_step=replace(
                config.position_step,
                sample_rate_hz=100.0,
                settle_s=0.01,
                hold_s=0.01,
                cycles=1,
            ),
        )
        controller = FakeController()
        snapshot = AsyncMock(return_value=("config\n", {"model": 4}))

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    position_step,
                    "create_controller",
                    return_value=(controller, moteus),
                ),
                patch.object(position_step, "snapshot_controller", snapshot),
                patch("builtins.input", return_value="MOVE"),
            ):
                path = Path(await position_step.run(config, Path(directory), 1))

            metadata = json.loads((path / "run.json").read_text())
            with (path / "telemetry.csv").open(newline="") as source:
                rows = list(csv.DictReader(source))

        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(controller.position, 0.25)
        self.assertGreaterEqual(controller.stop_count, 2)

    async def test_friction_sweep_returns_to_origin(self) -> None:
        config = load_config()
        config = replace(
            config,
            friction=replace(
                config.friction,
                sample_rate_hz=100.0,
                travel_rev=0.10,
                settle_s=0.01,
                turnaround_s=0.01,
            ),
        )
        controller = FakeController()
        snapshot = AsyncMock(return_value=("config\n", {"model": 4}))

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    friction,
                    "create_controller",
                    return_value=(controller, moteus),
                ),
                patch.object(friction, "snapshot_controller", snapshot),
                patch("builtins.input", return_value="SWEEP"),
            ):
                path = Path(await friction.run(config, Path(directory), 1))

            metadata = json.loads((path / "run.json").read_text())
            with (path / "telemetry.csv").open(newline="") as source:
                rows = list(csv.DictReader(source))

        self.assertEqual(metadata["status"], "completed")
        self.assertEqual({row["experiment_phase"] for row in rows}, {
            "setup",
            "settle",
            "forward",
            "turnaround",
            "reverse",
            "final",
        })
        self.assertAlmostEqual(controller.position, 0.05)
        self.assertGreaterEqual(controller.stop_count, 2)
