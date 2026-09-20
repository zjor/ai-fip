"""Timing and sample helpers shared by bench experiments."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from typing import Any, TypeVar

from ..recorder import RunRecorder
from ..safety import check_state
from ..servo import result_to_state


T = TypeVar("T")


async def await_with_progress(
    operation: Awaitable[T], label: str, *, interval_s: float = 5.0
) -> T:
    """Await a slow operation while periodically reporting elapsed time."""
    task = asyncio.ensure_future(operation)
    start = time.perf_counter()
    while True:
        done, _ = await asyncio.wait({task}, timeout=interval_s)
        if task in done:
            return task.result()
        elapsed = time.perf_counter() - start
        print(f"{label}: still working ({elapsed:.0f} s elapsed)...", flush=True)


async def wait_until(deadline: float) -> None:
    delay = deadline - time.perf_counter()
    if delay > 0.0:
        await asyncio.sleep(delay)


def record_response(
    *,
    recorder: RunRecorder,
    result: Any,
    moteus: Any,
    limits: Any,
    sample: int,
    scheduled_time_s: float,
    request_time_s: float,
    response_time_s: float,
    phase: str,
    command: dict[str, Any],
    origin_position_rev: float | None = None,
    max_position_excursion_rev: float | None = None,
    max_abs_velocity_rev_s: float | None = None,
) -> dict[str, Any]:
    state = result_to_state(result, moteus)
    recorder.write_sample(
        {
            "sample": sample,
            "host_time_s": response_time_s,
            "scheduled_time_s": scheduled_time_s,
            "request_time_s": request_time_s,
            "response_time_s": response_time_s,
            "experiment_phase": phase,
            **command,
            **state,
        }
    )
    check_state(
        state,
        limits,
        origin_position_rev=origin_position_rev,
        max_position_excursion_rev=max_position_excursion_rev,
        max_abs_velocity_rev_s=max_abs_velocity_rev_s,
    )
    return state
