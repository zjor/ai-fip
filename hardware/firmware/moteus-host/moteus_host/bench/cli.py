"""Command-line entry point for T-005 bench experiments."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from .analysis import summarize
from .config import DEFAULT_CONFIG_PATH, load_config
from .experiments import (
    coast_down, friction, inspect, position_step, response_timing, torque_constant,
)
from .safety import SafetyStop


def _ladder(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ladder must be comma-separated rpm values, got {text!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("ladder must list at least one rpm value")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a recorded moteus bench test")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="bench TOML file",
    )
    parser.add_argument("--id", type=int, help="override the configured moteus ID")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="override the configured run-data directory",
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    subparsers.add_parser("inspect", help="record stopped-state telemetry")
    subparsers.add_parser(
        "position-step", help="record conservative ±30 degree position steps"
    )
    subparsers.add_parser(
        "friction", help="record a slow one-revolution outward-and-return sweep"
    )
    timing_parser = subparsers.add_parser(
        "response-timing", help="record bounded torque pulses after a quiet baseline"
    )
    timing_parser.add_argument("--dry-run", action="store_true",
                               help="validate and print settings without connecting")
    torque_parser = subparsers.add_parser(
        "torque-constant",
        help="interactive held-rotor Q-current plateaus on the lever/scale fixture",
    )
    torque_parser.add_argument("--dry-run", action="store_true",
                               help="validate and print settings without connecting")
    torque_parser.add_argument("--rise-time", action="store_true",
                               help="feedforward torque steps for q-current rise time instead")
    coast_parser = subparsers.add_parser(
        "coast-down",
        help="speed-ladder free coast-downs of the retained test disc (Tier B, guard required)",
    )
    coast_parser.add_argument("--dry-run", action="store_true",
                              help="validate and print settings without connecting")
    coast_parser.add_argument("--accelerate", action="store_true",
                              help="feedforward torque steps from rest instead of the speed ladder")
    coast_parser.add_argument("--direction", type=int, choices=(1, -1),
                              help="override the configured ladder direction")
    coast_parser.add_argument("--ladder", type=_ladder, metavar="RPM,RPM,...",
                              help="override the configured speed ladder, e.g. 20,50,100")
    analyze_parser = subparsers.add_parser(
        "analyze", help="analyze a completed recorded run"
    )
    analyze_parser.add_argument("run_directory", type=Path)
    return parser


async def run(args: argparse.Namespace) -> str:
    if args.experiment == "analyze":
        return str(summarize.analyze(args.run_directory))
    config = load_config(args.config)
    controller_id = args.id or config.common.controller_id
    output_root = args.output_root or (
        args.config.resolve().parent / config.common.output_root
    )
    if args.experiment == "inspect":
        return await inspect.run(config, output_root, controller_id)
    if args.experiment == "position-step":
        return await position_step.run(config, output_root, controller_id)
    if args.experiment == "friction":
        return await friction.run(config, output_root, controller_id)
    if args.experiment == "response-timing":
        response_timing.validate(config.response_timing, config.safety)
        if args.dry_run:
            return response_timing.describe(config.response_timing)
        return await response_timing.run(config, output_root, controller_id)
    if args.experiment == "torque-constant":
        torque_constant.validate(config.torque_constant, config.safety)
        if args.dry_run:
            return torque_constant.describe(config.torque_constant, rise_time=args.rise_time)
        return await torque_constant.run(config, output_root, controller_id,
                                         rise_time=args.rise_time)
    if args.experiment == "coast-down":
        cfg = coast_down.resolve(config.coast_down, ladder_rpm=args.ladder, direction=args.direction)
        coast_down.validate(cfg, config.safety)
        if args.dry_run:
            return coast_down.describe(cfg, accelerate=args.accelerate)
        return await coast_down.run(config, output_root, controller_id, accelerate=args.accelerate,
                                    ladder_rpm=args.ladder, direction=args.direction)
    raise ValueError(f"unknown experiment {args.experiment}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = asyncio.run(run(args))
        label = ("Dry run (no connection)" if getattr(args, "dry_run", False)
                 else "Analysis written" if args.experiment == "analyze" else "Recorded run")
        print(f"{label}: {output}")
        return 0
    except SafetyStop as exc:
        print(f"Safety stop: {exc}")
        return 2
    except KeyboardInterrupt:
        print("Interrupted; stop command sent")
        return 130


if __name__ == "__main__":
    main()
