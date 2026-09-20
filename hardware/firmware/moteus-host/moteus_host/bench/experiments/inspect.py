"""Stopped-state connectivity and telemetry inspection."""

from __future__ import annotations

import time
from pathlib import Path

from ..config import BenchConfig
from ..recorder import RunRecorder
from ..safety import SafetyStop
from ..servo import create_controller, snapshot_controller
from .common import await_with_progress, record_response, wait_until


async def run(config: BenchConfig, output_root: Path, controller_id: int) -> str:
    recorder = RunRecorder(
        output_root=output_root,
        experiment="inspect",
        resolved_config={
            "controller_id": controller_id,
            "duration_s": config.inspect.duration_s,
            "sample_rate_hz": config.inspect.sample_rate_hz,
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
        print(
            f"Controller config captured; recording {config.inspect.duration_s:g} "
            "seconds of telemetry...",
            flush=True,
        )

        period = 1.0 / config.inspect.sample_rate_hz
        sample_count = max(
            1, round(config.inspect.duration_s * config.inspect.sample_rate_hz)
        )
        start = time.perf_counter()
        state = None
        for sample in range(sample_count):
            scheduled = sample * period
            await wait_until(start + scheduled)
            request = time.perf_counter() - start
            result = await controller.query()
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
                phase="stopped",
                command={"command_mode": "stop"},
            )
            samples_per_report = max(1, round(config.inspect.sample_rate_hz))
            if sample % samples_per_report == 0:
                print(
                    f"Telemetry {response:.1f}/{config.inspect.duration_s:g} s: "
                    f"{state['bus_voltage_v']:.2f} V, "
                    f"controller {state['controller_temperature_c']:.1f} °C, "
                    f"fault {state['fault']}",
                    flush=True,
                )
        assert state is not None
        print(f"Run directory: {recorder.path}")
        print(
            f"moteus {controller_id}: {state['bus_voltage_v']:.2f} V, "
            f"controller {state['controller_temperature_c']:.1f} °C, "
            f"position {state['position_rev']:.4f} rev, "
            f"home state {state['home_state']}, fault {state['fault']}"
        )
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
