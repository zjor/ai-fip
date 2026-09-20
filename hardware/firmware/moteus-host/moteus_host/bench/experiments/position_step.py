"""Conservative bidirectional position steps with full telemetry."""

from __future__ import annotations

import time
from typing import Any

from ..config import BenchConfig
from ..recorder import RunRecorder
from ..safety import SafetyStop, check_state, validate_position_step
from ..servo import (
    command_position,
    create_controller,
    result_to_state,
    snapshot_controller,
)
from .common import await_with_progress, record_response, wait_until


def target_sequence(origin: float, step: float, cycles: int) -> list[tuple[str, float]]:
    result = [("settle", origin)]
    for _ in range(cycles):
        result.extend(
            [
                ("positive", origin + step),
                ("center", origin),
                ("negative", origin - step),
                ("center", origin),
            ]
        )
    return result


async def run(config: BenchConfig, output_root: Any, controller_id: int) -> str:
    motion = config.position_step
    validate_position_step(motion, config.safety)
    recorder = RunRecorder(
        output_root=output_root,
        experiment="position-step",
        resolved_config={
            "controller_id": controller_id,
            "sample_rate_hz": motion.sample_rate_hz,
            "step_rev": motion.step_rev,
            "hold_s": motion.hold_s,
            "settle_s": motion.settle_s,
            "cycles": motion.cycles,
            "max_torque_nm": motion.max_torque_nm,
            "velocity_limit_rev_s": motion.velocity_limit_rev_s,
            "accel_limit_rev_s2": motion.accel_limit_rev_s2,
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

        print(f"Run directory: {recorder.path}")
        print(
            f"Will step around {origin:.4f} rev by ±{motion.step_rev:.4f} rev "
            f"at ≤{motion.max_torque_nm:.2f} Nm, "
            f"≤{motion.velocity_limit_rev_s:.2f} rev/s, "
            f"≤{motion.accel_limit_rev_s2:.2f} rev/s²."
        )
        confirmation = input(
            "Secure the stand, clear the arrow, prepare power disconnect, "
            "then type MOVE: "
        )
        if confirmation.strip() != "MOVE":
            status = "aborted"
            reason = "operator did not confirm motion"
            return str(recorder.path)

        recorder.event(0.0, "operator_confirmed", value="MOVE")
        period = 1.0 / motion.sample_rate_hz
        sample = 0
        scheduled = 0.0
        start = time.perf_counter()
        for phase, target in target_sequence(origin, motion.step_rev, motion.cycles):
            duration = motion.settle_s if phase == "settle" else motion.hold_s
            phase_samples = max(1, round(duration * motion.sample_rate_hz))
            print(
                f"Phase {phase}: target {target:.4f} rev for {duration:g} s",
                flush=True,
            )
            recorder.event(
                time.perf_counter() - start,
                "target_changed",
                value=target,
                unit="rev",
                note=phase,
            )
            for _ in range(phase_samples):
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
                record_response(
                    recorder=recorder,
                    result=result,
                    moteus=moteus,
                    limits=config.safety,
                    sample=sample,
                    scheduled_time_s=scheduled,
                    request_time_s=request,
                    response_time_s=response,
                    phase=phase,
                    origin_position_rev=origin,
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
