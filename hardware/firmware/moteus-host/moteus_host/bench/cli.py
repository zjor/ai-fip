"""Command-line entry point for T-005 bench experiments."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from .analysis import summarize
from .config import DEFAULT_CONFIG_PATH, load_config
from .experiments import friction, inspect, position_step
from .safety import SafetyStop


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
    raise ValueError(f"unknown experiment {args.experiment}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = asyncio.run(run(args))
        label = "Analysis written" if args.experiment == "analyze" else "Recorded run"
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
