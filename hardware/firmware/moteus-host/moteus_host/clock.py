"""Move an attached arrow like a twelve-tick, one-minute clock."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


TICK_SECONDS = 5.0
TICKS_PER_REVOLUTION = 12
COMMAND_PERIOD_SECONDS = 0.02
WATCHDOG_SECONDS = 0.10
MOVE_TIMEOUT_SECONDS = 10.0
POSITION_TOLERANCE_REV = 0.005
VELOCITY_TOLERANCE_REV_S = 0.02
SETTLED_SAMPLES = 5
CALIBRATION_PATH = Path(__file__).resolve().parents[1] / ".minute-clock-zero.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advance a moteus-driven arrow by 30 degrees every 5 seconds."
    )
    parser.add_argument("--id", type=int, default=1, help="moteus CAN ID (default: 1)")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="keep the motor stopped while the arrow is aligned to 12 o'clock",
    )
    parser.add_argument(
        "--clockwise-sign",
        type=int,
        choices=(-1, 1),
        help="override the saved clockwise position sign",
    )
    parser.add_argument(
        "--max-torque",
        type=float,
        default=0.10,
        help="command torque ceiling in Nm (default: 0.10)",
    )
    parser.add_argument(
        "--velocity-limit",
        type=float,
        default=0.25,
        help="motion velocity ceiling in revolutions/s (default: 0.25)",
    )
    parser.add_argument(
        "--accel-limit",
        type=float,
        default=0.50,
        help="motion acceleration ceiling in revolutions/s^2 (default: 0.50)",
    )
    return parser


def load_calibration(controller_id: int) -> dict[str, Any]:
    if not CALIBRATION_PATH.exists():
        raise RuntimeError(
            f"No clock zero stored in {CALIBRATION_PATH}. "
            "Run once with --calibrate."
        )

    try:
        document = json.loads(CALIBRATION_PATH.read_text())
        calibration = document["controllers"][str(controller_id)]
        upright_position = float(calibration["upright_position_modulo"])
        clockwise_sign = int(calibration["clockwise_sign"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Invalid clock calibration for moteus {controller_id} in "
            f"{CALIBRATION_PATH}; run with --calibrate to replace it."
        ) from exc

    if not 0.0 <= upright_position < 1.0 or clockwise_sign not in (-1, 1):
        raise RuntimeError(
            f"Invalid clock calibration for moteus {controller_id} in "
            f"{CALIBRATION_PATH}; run with --calibrate to replace it."
        )

    return {
        "upright_position_modulo": upright_position,
        "clockwise_sign": clockwise_sign,
    }


def save_calibration(
    controller_id: int, upright_position: float, clockwise_sign: int
) -> None:
    document: dict[str, Any] = {"format_version": 1, "controllers": {}}
    if CALIBRATION_PATH.exists():
        try:
            existing = json.loads(CALIBRATION_PATH.read_text())
            if (
                existing.get("format_version") == 1
                and isinstance(existing.get("controllers"), dict)
            ):
                document = existing
        except (AttributeError, json.JSONDecodeError):
            pass

    document["controllers"][str(controller_id)] = {
        "upright_position_modulo": upright_position % 1.0,
        "clockwise_sign": clockwise_sign,
    }
    temporary_path = CALIBRATION_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(document, indent=2) + "\n")
    temporary_path.replace(CALIBRATION_PATH)


def nearest_equivalent_position(current: float, position_modulo: float) -> float:
    """Return the equivalent single-turn angle nearest to current."""
    turns = math.floor(current - position_modulo + 0.5)
    return position_modulo + turns


async def set_position(controller: Any, target: float, args: argparse.Namespace) -> Any:
    return await controller.set_position(
        position=target,
        velocity=0.0,
        maximum_torque=args.max_torque,
        velocity_limit=args.velocity_limit,
        accel_limit=args.accel_limit,
        watchdog_timeout=WATCHDOG_SECONDS,
        query=True,
    )


async def move_to_upright(
    controller: Any, target: float, args: argparse.Namespace, moteus: Any
) -> None:
    deadline = time.monotonic() + MOVE_TIMEOUT_SECONDS
    settled_samples = 0

    while time.monotonic() < deadline:
        state = await set_position(controller, target, args)
        fault = state.values.get(moteus.Register.FAULT, 0)
        if fault:
            raise RuntimeError(f"moteus reported fault {fault}")

        position = state.values[moteus.Register.POSITION]
        velocity = state.values[moteus.Register.VELOCITY]
        if (
            abs(position - target) <= POSITION_TOLERANCE_REV
            and abs(velocity) <= VELOCITY_TOLERANCE_REV_S
        ):
            settled_samples += 1
            if settled_samples >= SETTLED_SAMPLES:
                return
        else:
            settled_samples = 0

        await asyncio.sleep(COMMAND_PERIOD_SECONDS)

    raise RuntimeError("Timed out while moving the arrow to 12 o'clock")


async def run_clock(args: argparse.Namespace) -> None:
    # Import here so argument parsing and help still work without hardware extras.
    import moteus

    controller = moteus.Controller(id=args.id)
    await controller.set_stop()
    try:
        if args.calibrate:
            print("Motor stopped: rotate the arrow to 12 o'clock by hand.")
            await asyncio.to_thread(input, "Press Enter when the arrow is upright...")

        state = await controller.query()
        current_position = state.values[moteus.Register.POSITION]

        if args.calibrate:
            clockwise_sign = args.clockwise_sign or 1
            start_position = current_position
            save_calibration(args.id, start_position, clockwise_sign)
            print(f"Saved clock zero in {CALIBRATION_PATH}")
        else:
            calibration = load_calibration(args.id)
            clockwise_sign = args.clockwise_sign or calibration["clockwise_sign"]
            start_position = nearest_equivalent_position(
                current_position, calibration["upright_position_modulo"]
            )
            print(
                f"Returning arrow from {current_position:.4f} to "
                f"12 o'clock at {start_position:.4f} revolutions"
            )
            await move_to_upright(controller, start_position, args, moteus)

        target_position = start_position
        tick = 0
        next_tick = time.monotonic() + TICK_SECONDS

        print(f"Arrow is upright; clockwise sign is {clockwise_sign:+d}")
        print("The first 30 degree tick will happen in 5 seconds; press Ctrl-C to stop")

        while True:
            now = time.monotonic()
            if now >= next_tick:
                # Advance from the original position so timing delays never accumulate
                # into position error. Catch up by at most the number of elapsed ticks.
                elapsed_ticks = int((now - next_tick) // TICK_SECONDS) + 1
                tick += elapsed_ticks
                target_position = (
                    start_position
                    + clockwise_sign * tick / TICKS_PER_REVOLUTION
                )
                next_tick += elapsed_ticks * TICK_SECONDS
                print(
                    f"tick {tick:>3}: target={target_position:.4f} rev "
                    f"({tick * 5}s, {tick * 30 % 360}°)"
                )

            state = await set_position(controller, target_position, args)
            fault = state.values.get(moteus.Register.FAULT, 0)
            if fault:
                raise RuntimeError(f"moteus reported fault {fault}")

            await asyncio.sleep(COMMAND_PERIOD_SECONDS)
    finally:
        await controller.set_stop()
        print("Motor stopped")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run_clock(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
