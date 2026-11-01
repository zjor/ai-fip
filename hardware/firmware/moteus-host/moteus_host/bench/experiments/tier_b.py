"""Tier B session: the connect, override, stop and restore plumbing shared by the
spinning-disc experiments (coast-down, torque-speed).

The 24 V bench supply cannot sink current, so the motor never brakes: every
deceleration is set_stop() and friction. For the run the firmware watchdog is set
to torque-off (servo.timeout_mode 0), the position bounds are opened (NaN),
servo.max_velocity derates torque above the Tier B speed and the flux brake burns
any regeneration in the windings below the supply's limit; all of them are
volatile (never conf write) and restored with a readback in __aexit__. A safety
stop, a cancelled task (what asyncio.run does on Ctrl-C) or a lost reply sends the
stop and then keeps querying until the disc is below coast_stop_rev_s, so the
operator is never left with a blind spinning disc; a second Ctrl-C during that
wait is swallowed with a reminder that only a power disconnect stops it sooner.
"""

from __future__ import annotations

import asyncio
import math
import signal
import time
from dataclasses import dataclass
from typing import Any

from ..config import BenchConfig
from ..recorder import RunRecorder
from ..safety import SafetyStop, check_state
from ..servo import create_controller, result_to_state, snapshot_controller
from .common import confirm_stopped, wait_until


CONFIRM_WORD = "GO"
CONNECT_TIMEOUT_S = 5.0
OVERRIDES = ("servo.timeout_mode", "servopos.position_min", "servopos.position_max", "servo.max_velocity")
# Regeneration above this bus voltage is dissipated in the windings instead of
# pushed into the 24 V supply, which cannot sink current. Older firmware exposes
# the threshold directly (flux_brake_min_voltage); 2025 firmware derives it as
# servo.max_voltage - servo.flux_brake_margin_voltage.
FLUX_BRAKE_MIN_VOLTAGE_V = 27.0
FLUX_BRAKE_KEYS = ("servo.flux_brake_min_voltage", "servo.flux_brake_margin_voltage")
# stopped, enabling, calibrating, calibration complete, position, timeout
ALLOWED_MODES = (0, 2, 3, 4, 10, 11)
# After an enable the firmware passes through enabling (2), calibrating (3) and
# calibration complete (4) for ~15 ms before position mode; the real controller
# reported mode 3 on the third reply (2026-10-28). Position mode is required once
# this long has passed since the first command of the step.
ENABLE_SETTLE_S = 0.05
STOP_ATTEMPTS = 3
MAX_LOST_REPLIES = 5
PROGRESS_INTERVAL_S = 1.0


@dataclass(frozen=True)
class TierBLimits:
    max_velocity_rev_s: float
    max_torque_nm: float
    response_timeout_s: float
    sample_rate_hz: float
    coast_stop_rev_s: float
    coast_settle_s: float
    coast_timeout_s: float


def torque_arguments(torque_nm: float, watchdog_s: float) -> dict[str, Any]:
    return dict(position=math.nan, velocity=0.0, kp_scale=0.0, kd_scale=0.0, ilimit_scale=0.0,
                feedforward_torque=torque_nm, maximum_torque=abs(torque_nm),
                watchdog_timeout=watchdog_s, query=True)


def flux_brake_override(settings: dict[str, str]) -> tuple[str, str]:
    """(key, value) that makes flux braking start at FLUX_BRAKE_MIN_VOLTAGE_V on this firmware."""
    if "servo.flux_brake_min_voltage" in settings:
        return "servo.flux_brake_min_voltage", f"{FLUX_BRAKE_MIN_VOLTAGE_V:g}"
    if "servo.flux_brake_margin_voltage" in settings:
        try:
            max_voltage = float(settings["servo.max_voltage"])
        except (KeyError, ValueError) as exc:
            raise SafetyStop("servo.max_voltage missing; cannot place the flux brake threshold") from exc
        margin = max(max_voltage - FLUX_BRAKE_MIN_VOLTAGE_V, 0.0)
        return "servo.flux_brake_margin_voltage", f"{margin:g}"
    raise SafetyStop("no flux brake key (" + " or ".join(FLUX_BRAKE_KEYS)
                     + ") in the controller configuration; regeneration could reach the supply")


def same(reported: str, expected: str) -> bool:
    try:
        a, b = float(reported), float(expected)
    except ValueError:
        return reported.strip() == expected.strip()
    return a == b or (math.isnan(a) and math.isnan(b))


class TierBSession:
    """Async context manager around one Tier B run on the test disc.

    __aenter__ connects, proves a stop, snapshots the configuration and opens the
    diagnostic stream; it sets no override. The caller sets the overrides, drives
    its steps through record/phase_loop/coast, and reports the outcome with
    finish(status, reason). __aexit__ stops, waits for standstill, restores the
    overrides with readback and closes the recorder; it never swallows exceptions.
    """

    def __init__(self, config: BenchConfig, controller_id: int, recorder: RunRecorder,
                 limits: TierBLimits):
        self.config = config
        self.controller_id = controller_id
        self.recorder = recorder
        self.limits = limits
        self.period = 1.0 / limits.sample_rate_hz
        self.controller = self.stream = self.moteus = None
        self.settings: dict[str, str] = {}
        self.initial_state: dict[str, Any] | None = None
        self.originals: dict[str, str] = {}
        self.flux_key: str | None = None
        self.flux_value: str | None = None
        self.overrides_set = False
        self.motion_commanded = False
        self.step_command_started = None   # perf_counter of the first command of the current step
        self.status, self.reason = "failed", None
        self.start = time.perf_counter()
        self.sample = 0

    # --- lifecycle -------------------------------------------------------

    async def __aenter__(self) -> TierBSession:
        try:
            self.controller, self.moteus = create_controller(self.controller_id, timing=True)
            await self.stopped(timeout=CONNECT_TIMEOUT_S)
            print("Stopped; capturing controller configuration...", flush=True)
            configuration, device = await asyncio.wait_for(snapshot_controller(self.controller, self.moteus), 60)
            self.recorder.attach_controller(configuration, device)
            self.settings = dict(line.split(maxsplit=1) for line in configuration.splitlines() if " " in line)
            self.flux_key, self.flux_value = flux_brake_override(self.settings)
            try:
                self.originals = {key: self.settings[key] for key in OVERRIDES + (self.flux_key,)}
            except KeyError as exc:
                raise SafetyStop(f"{exc.args[0]} missing from the controller configuration") from exc
            state = await self.stopped()
            check_state(state, self.config.safety)
            if not math.isfinite(state["position_rev"]):
                raise SafetyStop("finite rotor position required")
            self.initial_state = state
            self.stream = self.moteus.Stream(self.controller)
            await asyncio.wait_for(self.stream.flush_read(), 2)
        except BaseException as exc:
            # __aexit__ is not called when __aenter__ raises: run the same cleanup here
            await self.__aexit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                self.status = "interrupted"
            elif isinstance(exc, (SafetyStop, TimeoutError)):
                self.status = "safety_stop"
            else:
                self.status = "failed"
            self.reason = f"{type(exc).__name__}: {exc}"
        previous_handler = None
        try:
            try:
                previous_handler = signal.signal(signal.SIGINT, _second_interrupt)
            except ValueError:
                previous_handler = None   # not the main thread (tests); nothing to guard
            if self.controller is not None:
                await self._stop_with_retries()
                await self.wait_for_standstill()
            if self.overrides_set:
                for key, value in self.originals.items():
                    await self._conf(key, value)
                self.event("overrides_restored",
                           note="; ".join(f"{k}={v}" for k, v in self.originals.items()))
        except BaseException as cleanup_exc:
            self.status = "cleanup_failed"
            self.reason = f"{self.reason or ''}; cleanup: {type(cleanup_exc).__name__}: {cleanup_exc}"
            if self.motion_commanded:
                print("STOP/RESTORE NOT CONFIRMED: disconnect power; keep clear of the disc; "
                      "inspect timeout mode, position bounds, max velocity and flux brake before reuse.",
                      flush=True)
            elif self.overrides_set:
                print("OVERRIDES NOT RESTORED (motion was never commanded): inspect timeout mode, "
                      "position bounds, max velocity and flux brake before reuse.", flush=True)
            else:
                print("No reply from the controller; motion was never commanded. Check power, "
                      "the CAN cable and the controller ID.", flush=True)
            raise
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGINT, previous_handler)
            self.recorder.close(self.status, self.reason)
        return False

    def finish(self, status: str, reason: str | None = None) -> None:
        """Outcome of a run that reached the end of the caller's block: completed or aborted."""
        self.status, self.reason = status, reason

    # --- overrides -------------------------------------------------------

    async def set_overrides(self, max_velocity_rev_s: float) -> None:
        self.overrides_set = True   # mark before sending: a lost response must still trigger restoration
        await self._conf("servo.timeout_mode", 0)
        await self._conf("servopos.position_min", "nan")
        await self._conf("servopos.position_max", "nan")
        await self._conf("servo.max_velocity", f"{max_velocity_rev_s:g}")
        await self._conf(self.flux_key, self.flux_value)
        self.event("overrides_set",
                   note="; ".join(f"{k}={v}" for k, v in self.originals.items()) + "; volatile only; never conf write")

    async def _conf(self, key, value):
        await asyncio.wait_for(self.stream.command(f"conf set {key} {value}".encode()), 2)
        readback = await asyncio.wait_for(self.stream.command(f"conf get {key}".encode(), allow_any_response=True), 2)
        if not same(readback.decode(), str(value)):
            raise SafetyStop(f"{key} readback {readback.decode().strip()!r} != {value!r}")

    # --- helpers ---------------------------------------------------------

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def event(self, name, **kwargs) -> None:
        self.recorder.event(time.perf_counter() - self.start, name, **kwargs)

    def mark_step_start(self) -> None:
        self.step_command_started = None

    async def _bounded(self, awaitable, timeout=None):
        return await asyncio.wait_for(awaitable, timeout or self.limits.response_timeout_s)

    async def stopped(self, timeout=None):
        return await confirm_stopped(self.controller, self.moteus,
                                     timeout_s=timeout or self.limits.response_timeout_s)

    def _write_row(self, state, *, phase, trial, command):
        now = time.perf_counter() - self.start
        self.recorder.write_sample({"sample": self.sample, "host_time_s": now, "scheduled_time_s": now,
                                    "request_time_s": now, "response_time_s": now,
                                    "experiment_phase": phase, "trial_id": trial, **command, **state})
        self.sample += 1

    # --- sampling --------------------------------------------------------

    async def record(self, phase, trial, arguments=None, *, meta=None) -> dict:
        """One query (arguments None) or one command per sample; checks Tier B limits."""
        if arguments is None:
            result = await self._bounded(self.controller.query())
            command = {"command_mode": "query"}
        else:
            self.motion_commanded = True
            if self.step_command_started is None:
                self.step_command_started = time.perf_counter()
            result = await self._bounded(self.controller.set_position(**arguments))
            command = meta
        state = result_to_state(result, self.moteus)
        self._write_row(state, phase=phase, trial=trial, command=command)
        check_state(state, self.config.safety, max_abs_velocity_rev_s=self.limits.max_velocity_rev_s,
                    max_torque_nm=self.limits.max_torque_nm)
        if state["mode"] not in ALLOWED_MODES:
            raise SafetyStop(f"unexpected controller mode {state['mode']}")
        for field in ("velocity_rev_s", "q_current_a", "bus_voltage_v"):
            if not math.isfinite(state[field]):
                raise SafetyStop(f"non-finite {field}")
        return state

    def settled_below_stop(self):
        """done() for coasts: |v| < coast_stop_rev_s continuously for coast_settle_s."""
        since = None
        limits = self.limits

        def done(state, elapsed):
            nonlocal since
            if abs(state["velocity_rev_s"]) < limits.coast_stop_rev_s:
                since = elapsed if since is None else since
                return elapsed - since >= limits.coast_settle_s
            since = None
            return False
        return done

    async def phase_loop(self, phase, trial, arguments, meta, *, done, timeout_s, timeout_message) -> dict:
        phase_start = time.perf_counter()
        while True:
            tick = time.perf_counter()
            state = await self.record(phase, trial, arguments, meta=meta)
            settled = (arguments is not None and self.step_command_started is not None
                       and time.perf_counter() - self.step_command_started > ENABLE_SETTLE_S)
            if settled and state["mode"] != 10:
                # mode 11 = firmware watchdog fired; the disc is already coasting and
                # further commands are ignored, so these rows must not pass as data
                raise SafetyStop(f"controller left position mode during {phase}: mode {state['mode']}")
            if done(state, tick - phase_start):
                return state
            if time.perf_counter() - phase_start > timeout_s:
                raise SafetyStop(timeout_message)
            await wait_until(tick + self.period)

    async def coast(self, trial) -> dict:
        await self.stopped()
        self.event("coast_start", value=trial)
        return await self.phase_loop("coast", trial, None, None, done=self.settled_below_stop(),
                                     timeout_s=self.limits.coast_timeout_s,
                                     timeout_message=f"coast timeout after {self.limits.coast_timeout_s:g} s")

    async def _stop_with_retries(self):
        last = None
        for attempt in range(STOP_ATTEMPTS):
            try:
                return await self.stopped()
            except (TimeoutError, SafetyStop) as exc:
                last = exc
                await wait_until(time.perf_counter() + self.period)
        raise SafetyStop(f"stop not confirmed after {STOP_ATTEMPTS} attempts: {last}")

    async def wait_for_standstill(self, phase="standstill_wait", trial=None):
        """After any stop: keep the operator informed until the disc is slow for coast_settle_s."""
        wait_start = time.perf_counter()
        deadline = wait_start + self.limits.coast_timeout_s
        last_print = -math.inf
        lost = 0
        done = self.settled_below_stop()
        while True:
            tick = time.perf_counter()
            try:
                state = result_to_state(await self._bounded(self.controller.query()), self.moteus)
            except TimeoutError:
                lost += 1
                if lost >= MAX_LOST_REPLIES:
                    raise SafetyStop(f"{lost} replies lost in a row while the disc may be turning")
                await wait_until(tick + self.period)
                continue
            lost = 0
            self._write_row(state, phase=phase, trial=trial, command={"command_mode": "query"})
            speed = abs(state["velocity_rev_s"]) if state["velocity_rev_s"] is not None else math.inf
            if done(state, tick - wait_start):
                self.recorder.event(tick - self.start, "standstill", value=speed, unit="rev/s")
                return
            if speed >= self.limits.coast_stop_rev_s and tick - last_print >= PROGRESS_INTERVAL_S:
                print(f"coasting: {60 * speed:.0f} rpm", flush=True)
                last_print = tick
            if tick > deadline:
                raise SafetyStop("disc still turning after the coast timeout; keep clear until it stops")
            await wait_until(tick + self.period)


def _second_interrupt(signum, frame):
    print("still coasting; the stop was sent. Only a power disconnect stops the disc sooner.", flush=True)
