"""Slow bidirectional sweep for friction and cogging characterization."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from ..config import BenchConfig
from ..recorder import RunRecorder
from ..safety import SafetyStop, check_state, validate_friction
from ..servo import (
    command_position,
    create_controller,
    result_to_state,
    snapshot_controller,
)
from .common import await_with_progress, record_response, wait_until


def parse_position_bounds(configuration: str) -> tuple[float | None, float | None]:
    """Return configured output-position bounds, treating NaN as unbounded."""
    values: dict[str, float | None] = {}
    keys = {"servopos.position_min", "servopos.position_max"}
    for line in configuration.splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] not in keys:
            continue
        value = float(parts[1])
        values[parts[0]] = None if math.isnan(value) else value
    return values.get("servopos.position_min"), values.get("servopos.position_max")


def plan_centered_sweep(
    origin: float,
    travel: float,
    center: float,
    position_min: float | None,
    position_max: float | None,
    margin: float,
    max_setup_travel: float,
) -> tuple[float, float, int]:
    """Choose the nearest endpoint of a centered full-span sweep."""
    if position_min is not None and origin < position_min:
        raise SafetyStop(
            f"starting position {origin:.4f} rev is below controller minimum "
            f"{position_min:.4f} rev; reposition while stopped before retrying"
        )
    if position_max is not None and origin > position_max:
        raise SafetyStop(
            f"starting position {origin:.4f} rev is above controller maximum "
            f"{position_max:.4f} rev; reposition while stopped before retrying"
        )

    lower = center - travel / 2.0
    upper = center + travel / 2.0
    if position_min is not None and lower < position_min + margin:
        raise SafetyStop(
            f"sweep lower endpoint {lower:.3f} rev does not retain "
            f"{margin:.3f} rev margin from controller minimum {position_min:.3f}"
        )
    if position_max is not None and upper > position_max - margin:
        raise SafetyStop(
            f"sweep upper endpoint {upper:.3f} rev does not retain "
            f"{margin:.3f} rev margin from controller maximum {position_max:.3f}"
        )

    start, destination = min(
        ((lower, upper), (upper, lower)),
        key=lambda endpoints: abs(origin - endpoints[0]),
    )
    if abs(start - origin) > max_setup_travel:
        raise SafetyStop(
            f"nearest sweep endpoint is {abs(start - origin):.3f} rev from "
            f"the starting position; setup limit is {max_setup_travel:.3f} rev"
        )
    direction = 1 if destination > start else -1
    return start, destination, direction


async def run(config: BenchConfig, output_root: Path, controller_id: int) -> str:
    motion = config.friction
    validate_friction(motion, config.safety)
    recorder = RunRecorder(
        output_root=output_root,
        experiment="friction",
        resolved_config={
            "controller_id": controller_id,
            "sample_rate_hz": motion.sample_rate_hz,
            "travel_rev": motion.travel_rev,
            "sweep_center_rev": motion.sweep_center_rev,
            "velocity_limit_rev_s": motion.velocity_limit_rev_s,
            "accel_limit_rev_s2": motion.accel_limit_rev_s2,
            "max_torque_nm": motion.max_torque_nm,
            "settle_s": motion.settle_s,
            "turnaround_s": motion.turnaround_s,
            "move_timeout_s": motion.move_timeout_s,
            "position_bound_margin_rev": motion.position_bound_margin_rev,
            "watchdog_timeout_s": config.common.watchdog_timeout_s,
        },
    )
    controller = None
    status = "failed"
    reason = None
    try:
        controller, moteus = create_controller(controller_id)
        await controller.set_stop()
        print(
            "Connected and stopped; capturing controller config (about 30 seconds)...",
            flush=True,
        )
        controller_config, device = await await_with_progress(
            snapshot_controller(controller, moteus), "Controller config"
        )
        recorder.attach_controller(controller_config, device)
        print("Controller config captured; running motion preflight...", flush=True)

        initial_result = await controller.query()
        initial_state = result_to_state(initial_result, moteus)
        check_state(initial_state, config.safety)
        origin = initial_state["position_rev"]
        if initial_state.get("home_state") in (None, 0):
            raise RuntimeError("position is not referenced to the rotor or output")

        position_min, position_max = parse_position_bounds(controller_config)
        sweep_start, destination, direction = plan_centered_sweep(
            origin,
            motion.travel_rev,
            motion.sweep_center_rev,
            position_min,
            position_max,
            motion.position_bound_margin_rev,
            config.safety.max_friction_excursion_rev,
        )
        outbound_phase = "forward" if direction > 0 else "reverse"
        return_phase = "reverse" if direction > 0 else "forward"
        estimated_motion_s = 2.0 * motion.travel_rev / motion.velocity_limit_rev_s
        print(f"Run directory: {recorder.path}")
        print(
            f"Will stage at {sweep_start:+.3f} rev, sweep "
            f"{motion.travel_rev:.2f} revolution "
            f"{'forward' if direction > 0 else 'reverse'} and return over "
            f"about {estimated_motion_s:.0f} seconds at "
            f"≤{motion.velocity_limit_rev_s:.3f} rev/s and "
            f"≤{motion.max_torque_nm:.2f} Nm; controller position bounds are "
            f"[{position_min}, {position_max}] rev with "
            f"{motion.position_bound_margin_rev:.3f} rev endpoint margin."
        )
        confirmation = input(
            "Secure the stand, ensure 360° arrow clearance, prepare power "
            "disconnect, then type SWEEP: "
        )
        if confirmation.strip() != "SWEEP":
            status = "aborted"
            reason = "operator did not confirm motion"
            return str(recorder.path)

        recorder.event(0.0, "operator_confirmed", value="SWEEP")
        recorder.event(
            0.0,
            "sweep_plan",
            value=direction,
            note=(
                f"start={sweep_start}, destination={destination}, "
                f"bounds=[{position_min}, {position_max}]"
            ),
        )
        period = 1.0 / motion.sample_rate_hz
        start = time.perf_counter()
        sample = 0
        scheduled = 0.0

        async def sample_target(
            phase: str, target: float, safety_origin: float
        ) -> dict[str, Any]:
            nonlocal sample, scheduled
            await wait_until(start + scheduled)
            request = time.perf_counter() - start
            result = await command_position(
                controller,
                position_rev=target,
                maximum_torque_nm=motion.max_torque_nm,
                velocity_limit_rev_s=motion.velocity_limit_rev_s,
                accel_limit_rev_s2=motion.accel_limit_rev_s2,
                watchdog_timeout_s=config.common.watchdog_timeout_s,
            )
            response = time.perf_counter() - start
            state = record_response(
                recorder=recorder,
                result=result,
                moteus=moteus,
                limits=config.safety,
                sample=sample,
                scheduled_time_s=scheduled,
                request_time_s=request,
                response_time_s=response,
                phase=phase,
                origin_position_rev=safety_origin,
                max_position_excursion_rev=config.safety.max_friction_excursion_rev,
                max_abs_velocity_rev_s=config.safety.max_friction_velocity_rev_s,
                command={
                    "command_mode": "position",
                    "command_position_rev": target,
                    "command_velocity_rev_s": 0.0,
                    "feedforward_torque_nm": 0.0,
                    "maximum_torque_nm": motion.max_torque_nm,
                    "velocity_limit_rev_s": motion.velocity_limit_rev_s,
                    "accel_limit_rev_s2": motion.accel_limit_rev_s2,
                    "watchdog_timeout_s": config.common.watchdog_timeout_s,
                },
            )
            sample += 1
            scheduled += period
            return state

        async def hold(
            phase: str, target: float, duration_s: float, safety_origin: float
        ) -> None:
            print(f"Phase {phase}: holding {target:.4f} rev for {duration_s:g} s")
            recorder.event(
                time.perf_counter() - start,
                "phase_started",
                value=target,
                unit="rev",
                note=phase,
            )
            for _ in range(max(1, round(duration_s * motion.sample_rate_hz))):
                await sample_target(phase, target, safety_origin)

        async def move(phase: str, target: float, safety_origin: float) -> None:
            print(f"Phase {phase}: moving slowly to {target:.4f} rev", flush=True)
            phase_start = time.perf_counter()
            recorder.event(
                phase_start - start,
                "phase_started",
                value=target,
                unit="rev",
                note=phase,
            )
            next_report = phase_start + 5.0
            while True:
                state = await sample_target(phase, target, safety_origin)
                now = time.perf_counter()
                if now >= next_report:
                    travelled = state["position_rev"] - safety_origin
                    print(
                        f"Phase {phase}: {travelled:+.3f} rev from origin, "
                        f"torque {state['torque_nm']:+.4f} Nm",
                        flush=True,
                    )
                    next_report += 5.0
                if (
                    state.get("trajectory_complete")
                    and abs(state["position_rev"] - target) <= 0.005
                    and abs(state["velocity_rev_s"]) <= 0.01
                ):
                    return
                if now - phase_start > motion.move_timeout_s:
                    raise SafetyStop(f"{phase} movement timed out")

        await move("setup", sweep_start, origin)
        await hold("settle", sweep_start, motion.settle_s, sweep_start)
        await move(outbound_phase, destination, sweep_start)
        await hold(
            "turnaround", destination, motion.turnaround_s, sweep_start
        )
        await move(return_phase, sweep_start, sweep_start)
        await hold("final", sweep_start, motion.settle_s, sweep_start)

        status = "completed"
        return str(recorder.path)
    except SafetyStop as exc:
        status = "safety_stop"
        reason = f"SafetyStop: {exc}"
        raise
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if controller is not None:
                await controller.set_stop()
        finally:
            recorder.close(status, reason)
