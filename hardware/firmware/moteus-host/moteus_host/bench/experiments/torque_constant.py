"""Tier-C held-rotor fixture: interactive Q-current plateaus and torque steps for K_t.

Physical torque comes from the compression scale on the ±100 mm lever
(docs/hardware/specs/motor-torque-fixture.md); moteus-reported torque is only
a comparison channel. The operator types one plateau at a time, watches the
scale while current is on, and types the reading after the controller stops.

The firmware watchdog only covers position mode (moteus
fw/bldc_servo_control.h: the timeout check runs for kPosition and
kStayWithinBounds, never for kCurrent), so every plateau is sent as a
position-mode command with zero gains and a feedforward torque of
amps x K_t_fw, where K_t_fw = 8.27 / motor.Kv is the firmware's own linear
torque constant (rotation cutoff 10000 A on this motor). The fit still uses
the measured Q-current, so the command path does not enter the result.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..config import BenchConfig, SafetyConfig, TorqueConstantConfig
from ..recorder import RunRecorder
from ..safety import SafetyStop, check_state
from ..servo import create_controller, result_to_state, snapshot_controller
from .common import confirm_stopped, wait_until


STANDARD_SEQUENCE_A = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
# fw/bldc_servo.cc kTorqueFactor: (3/2) * (1/sqrt 3) * (60 / 2 pi) ~= 8.27 Nm*rpm/V/A
FIRMWARE_TORQUE_FACTOR = 1.5 / math.sqrt(3.0) * 60.0 / (2.0 * math.pi)
REPEATS = 3  # nonzero magnitudes; the 0 A plateau once per direction
CONFIRM_PHRASE = "LEVER"
CONNECT_TIMEOUT_S = 5.0
# After an enable the firmware passes through enabling (2), calibrating (3) and
# calibration complete (4) for a few ms before the commanded mode; the real
# controller reported mode 3 on the second reply of a plateau (2026-10-25).
ENABLE_SETTLE_S = 0.05
ENABLE_MODES = (2, 3, 4)


@dataclass(frozen=True)
class Entry:
    kind: str  # "quit" | "plateau" | "step"
    value: float = 0.0  # amps for plateau, Nm for step
    duration_s: float = 0.0


def validate(cfg: TorqueConstantConfig, limits: SafetyConfig) -> None:
    for name, value in asdict(cfg).items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"torque_constant.{name} must be finite and positive")
    if cfg.max_current_a > limits.max_phase_current_a:
        raise ValueError("max_current_a exceeds the tracked phase-current limit")
    if cfg.max_reported_torque_nm > 0.30:
        raise ValueError("max_reported_torque_nm exceeds the 0.30 Nm fixture cap")
    if cfg.max_rise_torque_nm > 0.05:
        raise ValueError("max_rise_torque_nm exceeds the 0.05 Nm rise-time cap")
    for name in ("sample_rate_hz", "rise_sample_rate_hz"):
        if not 100 <= getattr(cfg, name) <= 500:
            raise ValueError(f"{name} must be within 100..500 Hz")
    if not cfg.plateau_s <= cfg.max_plateau_s <= 10:
        raise ValueError("plateau_s <= max_plateau_s <= 10 s required")
    if not cfg.ramp_s <= 1.0 and cfg.ramp_s < cfg.plateau_s:
        raise ValueError("ramp_s must be <= 1 s and shorter than plateau_s")
    if cfg.rise_step_s > 2:
        raise ValueError("rise_step_s must be <= 2 s")
    if cfg.rise_prehold_nm >= cfg.max_rise_torque_nm:
        raise ValueError("rise_prehold_nm must be below max_rise_torque_nm")
    if cfg.rise_prehold_s > 1.0:
        raise ValueError("rise_prehold_s must be <= 1 s")
    if cfg.max_velocity_rev_s > limits.max_abs_velocity_rev_s:
        raise ValueError("max_velocity_rev_s exceeds the Tier-A hard limit")
    if not cfg.max_velocity_rev_s <= cfg.settle_max_velocity_rev_s <= limits.max_abs_velocity_rev_s:
        raise ValueError("max_velocity_rev_s <= settle_max_velocity_rev_s <= Tier-A hard limit required")
    if cfg.max_excursion_rev > limits.max_position_excursion_rev:
        raise ValueError("max_excursion_rev exceeds the Tier-A hard limit")
    if cfg.response_timeout_s > 0.1:
        raise ValueError("response_timeout_s must be <= 100 ms")


def firmware_kt(settings: dict[str, str]) -> float:
    """The firmware's linear torque constant, 8.27 / motor.Kv, from a config snapshot."""
    try:
        kv = float(settings["motor.Kv"])
    except (KeyError, ValueError) as exc:
        raise SafetyStop("motor.Kv missing from the controller configuration") from exc
    if not math.isfinite(kv) or kv <= 0.0:
        raise SafetyStop(f"motor.Kv {kv} is not a positive finite value")
    return FIRMWARE_TORQUE_FACTOR / kv


def describe(cfg: TorqueConstantConfig, *, rise_time: bool) -> str:
    stops = (f"stop thresholds: speed {cfg.max_velocity_rev_s:g} rev/s "
             f"({cfg.settle_max_velocity_rev_s:g} during the first {ENABLE_SETTLE_S + cfg.ramp_s:g} s "
             f"of a plateau while the lever re-seats), excursion "
             f"{cfg.max_excursion_rev:g} rev, measured current "
             f"{cfg.max_current_a + cfg.current_overshoot_a:g} A, reported torque "
             f"{cfg.max_reported_torque_nm:g} Nm. No timeout-mode override; controller "
             "stopped between plateaus.")
    if rise_time:
        return (f"Torque-constant, rise-time mode: feedforward torque steps up to "
                f"±{cfg.max_rise_torque_nm:g} Nm for {cfg.rise_step_s:g} s (max 2 s), each from a "
                f"{cfg.rise_prehold_nm:g} Nm same-sign pre-hold of {cfg.rise_prehold_s:g} s so the "
                f"step is applied with the controller already in position mode, "
                f"{cfg.rise_sample_rate_hz:g} Hz requested, rest {cfg.rest_s:g} s, "
                f"session budget {cfg.session_i2t_budget_a2s:g} A²s. {stops}")
    sequence = ", ".join(f"{v:g}" for v in STANDARD_SEQUENCE_A)
    return (f"Torque-constant, plateau mode: Q-current plateaus typed by the operator, "
            f"standard sequence {sequence} A (x{REPEATS} nonzero, per direction), "
            f"default {cfg.plateau_s:g} s (max {cfg.max_plateau_s:g} s), ramped in over "
            f"{cfg.ramp_s:g} s, sent as position-mode feedforward torque amps x 8.27/motor.Kv "
            "with zero gains "
            "(the firmware watchdog does not cover current mode), D current 0, "
            f"{cfg.sample_rate_hz:g} Hz requested, rest {cfg.rest_s:g} s, session budget "
            f"{cfg.session_i2t_budget_a2s:g} A²s. {stops}")


def parse_entry(text: str, cfg: TorqueConstantConfig, *, rise_time: bool) -> Entry:
    words = text.strip().split()
    if words and words[0].lower() in ("q", "quit"):
        return Entry("quit")
    if not 1 <= len(words) <= 2:
        raise ValueError("type a value, optionally followed by seconds, or q")
    try:
        numbers = [float(word) for word in words]
    except ValueError as exc:
        raise ValueError(f"not a number: {text.strip()!r}") from exc
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError("values must be finite")
    value = numbers[0] + 0.0  # turns -0.0 into 0.0
    if rise_time:
        kind, unit, limit = "step", "Nm", cfg.max_rise_torque_nm
        default, max_duration = cfg.rise_step_s, 2.0
        if value == 0.0:
            raise ValueError("torque step must be nonzero")
    else:
        kind, unit, limit = "plateau", "A", cfg.max_current_a
        default, max_duration = cfg.plateau_s, cfg.max_plateau_s
    if abs(value) > limit:
        raise ValueError(f"|{value:g}| {unit} exceeds the {limit:g} {unit} limit")
    duration = numbers[1] if len(numbers) == 2 else default
    if not 0.0 < duration <= max_duration:
        raise ValueError(f"duration must be within (0, {max_duration:g}] s")
    return Entry(kind, value, duration)


def i2t_cost(entry: Entry, cfg: TorqueConstantConfig) -> float:
    if entry.kind == "plateau":
        return entry.value * entry.value * entry.duration_s
    amps = entry.value / cfg.nominal_kt_nm_per_a
    prehold = cfg.rise_prehold_nm / cfg.nominal_kt_nm_per_a
    return amps * amps * entry.duration_s + prehold * prehold * cfg.rise_prehold_s


def next_suggestion(counts: dict[float, int], sign: int) -> float | None:
    for magnitude in STANDARD_SEQUENCE_A:
        if counts.get(sign * magnitude + 0.0, 0) < (1 if magnitude == 0.0 else REPEATS):
            return sign * magnitude + 0.0
    return None


def format_note(**fields: object) -> str:
    return "; ".join(f"{key}={value}" for key, value in fields.items())


def parse_note(note: str) -> dict[str, str]:
    result = {}
    for part in note.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def _prompt_float(prompt: str, low: float, high: float) -> float:
    while True:
        text = input(prompt).strip()
        try:
            value = float(text)
        except ValueError:
            value = math.nan
        if math.isfinite(value) and low <= value <= high:
            return value
        print(f"enter a number within {low:g}..{high:g}", flush=True)


def _prompt_reading() -> float | None:
    while True:
        text = input("scale reading [g] (blank = skip): ").strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            value = math.nan
        if math.isfinite(value):
            return value
        print("enter a number in grams or leave blank", flush=True)


async def run(config: BenchConfig, output_root: Path, controller_id: int, *,
              rise_time: bool = False) -> str:
    cfg = config.torque_constant
    validate(cfg, config.safety)
    mode = "rise_time" if rise_time else "plateau"
    recorder = RunRecorder(output_root=output_root, experiment="torque-constant",
                           resolved_config={"controller_id": controller_id, "mode": mode,
                                            **asdict(cfg)})
    limits = replace(config.safety, max_torque_nm=cfg.max_reported_torque_nm,
                     max_phase_current_a=cfg.max_current_a + cfg.current_overshoot_a)
    controller = None
    status, reason = "failed", None
    start = time.perf_counter()
    sample = 0
    origin = None
    kt_fw = None
    watchdog_s = config.common.watchdog_timeout_s
    rate_hz = cfg.rise_sample_rate_hz if rise_time else cfg.sample_rate_hz
    period_s = 1.0 / rate_hz

    def elapsed() -> float:
        return time.perf_counter() - start

    async def bounded(awaitable: Any) -> Any:
        return await asyncio.wait_for(awaitable, cfg.response_timeout_s)

    async def stopped(timeout_s: float | None = None) -> dict[str, Any]:
        return await confirm_stopped(controller, moteus, timeout_s=timeout_s or cfg.response_timeout_s)

    async def record(phase: str, trial: int, entry: Entry | None = None, *,
                     enabling: bool = False, settling: bool = False,
                     fraction: float = 1.0) -> dict[str, Any]:
        """One sample; `fraction` scales the plateau torque during the ramp-in.

        The excursion origin is re-anchored on the first sample after it is cleared,
        so each plateau is judged against the rotor position at its own baseline and
        not against where the rotor sat at connect time: the enable transient and the
        operator re-seating the lever on the scale move it while the controller is off.
        """
        nonlocal sample, origin
        scheduled = elapsed()
        request = elapsed()
        if entry is None:
            result = await bounded(controller.query())
            command = {"command_mode": "query"}
            allowed_modes = (0, 9, 10, 11)
        else:
            if entry.kind == "plateau":
                torque_nm = fraction * entry.value * kt_fw
                maximum_nm = cfg.max_current_a * kt_fw
                amps = dict(command_q_current_a=entry.value, command_d_current_a=0.0)
            else:
                torque_nm, maximum_nm, amps = entry.value, cfg.max_rise_torque_nm, {}
            result = await bounded(controller.set_position(
                position=math.nan, velocity=0.0, kp_scale=0.0, kd_scale=0.0, ilimit_scale=0.0,
                feedforward_torque=torque_nm, maximum_torque=maximum_nm,
                watchdog_timeout=watchdog_s, query=True))
            command = dict(command_mode="torque", command_velocity_rev_s=0.0, **amps,
                           feedforward_torque_nm=torque_nm, maximum_torque_nm=maximum_nm,
                           watchdog_timeout_s=watchdog_s,
                           kp_scale=0.0, kd_scale=0.0, ilimit_scale=0.0)
            allowed_modes = (0, 10, 11)
        response = elapsed()
        state = result_to_state(result, moteus)
        if origin is None and math.isfinite(state["position_rev"]):
            origin = state["position_rev"]
        row = dict(sample=sample, host_time_s=response, scheduled_time_s=scheduled,
                   request_time_s=request, response_time_s=response,
                   experiment_phase=phase, trial_id=trial, **command, **state)
        recorder.write_sample(row)
        sample += 1
        check_state(state, limits, origin_position_rev=origin,
                    max_position_excursion_rev=cfg.max_excursion_rev,
                    max_abs_velocity_rev_s=(cfg.settle_max_velocity_rev_s if settling
                                            else cfg.max_velocity_rev_s))
        for field in ("position_rev", "velocity_rev_s", "q_current_a", "d_current_a",
                      "torque_nm", "controller_temperature_c", "bus_voltage_v"):
            if not math.isfinite(state[field]):
                raise SafetyStop(f"non-finite {field}")
        if enabling:
            allowed_modes = allowed_modes + ENABLE_MODES
        if state["mode"] not in allowed_modes:
            raise SafetyStop(f"unexpected controller mode {state['mode']}")
        if entry is not None and phase != "watchdog_probe" and state["mode"] == 11:
            raise SafetyStop("watchdog timeout during plateau")
        return row

    try:
        controller, moteus = create_controller(controller_id, timing=True)
        # The first transaction also opens the fdcanusb transport (~80 ms
        # measured), so it is not held to the per-sample response bound.
        await stopped(CONNECT_TIMEOUT_S)
        print("Stopped; capturing controller configuration...", flush=True)
        configuration, device = await asyncio.wait_for(
            snapshot_controller(controller, moteus), 60)
        recorder.attach_controller(configuration, device)
        settings = dict(line.split(maxsplit=1) for line in configuration.splitlines()
                        if " " in line)
        kt_fw = firmware_kt(settings)
        state = await stopped()
        check_state(state, limits)
        if not math.isfinite(state["position_rev"]):
            raise SafetyStop("finite rotor position required")
        # Prove on this controller that the firmware watchdog applies to the
        # command path used for plateaus: one 0 A (0 Nm) command with the
        # per-command watchdog, then silence for 2.5 timeouts must leave the
        # controller in its timeout state (11; 12 if the fallback mode is
        # reported instead), before any torque is ever commanded.
        probe = await record("watchdog_probe", -1, Entry("plateau", 0.0, 0.0),
                             enabling=True, settling=True)
        await wait_until(time.perf_counter() + 2.5 * watchdog_s)
        probe_mode = result_to_state(await bounded(controller.query()), moteus)["mode"]
        await stopped()
        if probe_mode not in (11, 12):
            raise SafetyStop(f"firmware watchdog not observed after a torque command (mode {probe_mode})")
        recorder.event(elapsed(), "watchdog_probe", value=probe_mode,
                       note=format_note(watchdog_timeout_s=watchdog_s,
                                        firmware_default_timeout_s=settings.get("servo.default_timeout_s"),
                                        firmware_kt_nm_per_a=round(kt_fw, 6),
                                        probe_mode_after_command=probe["mode"]))
        print(describe(cfg, rise_time=rise_time), flush=True)
        print(f"Run directory: {recorder.path}", flush=True)
        if not rise_time:
            for side in ("positive", "negative"):
                radius = _prompt_float(f"contact radius, {side} side [mm]: ", 50.0, 200.0)
                recorder.event(elapsed(), f"radius_{side}_mm", value=radius, unit="mm")
            recorder.event(elapsed(), "scale_check",
                           note=input("scale check note (known masses, resolution, auto-off): ").strip())
        phrase = input("Secure fixture, scale under the loaded contact, hands clear, "
                       f"power disconnect ready; type {CONFIRM_PHRASE}: ").strip()
        if phrase != CONFIRM_PHRASE:
            status, reason = "aborted", f"operator did not confirm {CONFIRM_PHRASE}"
            return str(recorder.path)
        recorder.event(elapsed(), "operator_confirmed", value=CONFIRM_PHRASE)

        counts: dict[float, int] = {}
        sign = 1
        i2t_used = 0.0
        last_end = None
        trial = 0
        unit = "Nm" if rise_time else "A"
        while True:
            if rise_time:
                prompt = (f"torque [Nm] (max {cfg.max_rise_torque_nm:g}), 'Nm s' for custom "
                          "seconds, q to quit: ")
            else:
                suggestion = next_suggestion(counts, sign)
                hint = (f"suggested {suggestion:+g}" if suggestion is not None
                        else "sequence complete for this sign")
                prompt = f"current [A] ({hint}), 'A s' for custom seconds, q to quit: "
            try:
                entry = parse_entry(input(prompt), cfg, rise_time=rise_time)
            except ValueError as exc:
                print(f"rejected: {exc}", flush=True)
                continue
            if entry.kind == "quit":
                break
            cost = i2t_cost(entry, cfg)
            if i2t_used + cost > cfg.session_i2t_budget_a2s:
                print(f"refused: session I²t budget {cfg.session_i2t_budget_a2s:g} A²s would be "
                      f"exceeded ({i2t_used:.0f} used, {cost:.0f} requested)", flush=True)
                continue
            if last_end is not None and time.perf_counter() - last_end < cfg.rest_s:
                print(f"resting {last_end + cfg.rest_s - time.perf_counter():.1f} s", flush=True)
                await wait_until(last_end + cfg.rest_s)

            origin = None  # re-anchor the excursion check on this plateau's baseline
            baseline_start = time.perf_counter()
            while time.perf_counter() - baseline_start < cfg.baseline_s:
                tick = time.perf_counter()
                await record("baseline", trial)
                await wait_until(tick + period_s)

            recorder.event(elapsed(), "plateau_start", value=entry.value, unit=unit,
                           note=format_note(plateau=trial, kind=entry.kind, value=entry.value,
                                            duration_s=entry.duration_s,
                                            origin_rev=round(origin, 5) if origin is not None else None))
            print(f"\a>>> {entry.kind} ON: {entry.value:+g} {unit} for {entry.duration_s:g} s"
                  " — read the scale <<<", flush=True)
            active_rows = []
            if entry.kind == "step":
                # Enable the controller on a small same-sign torque first, so the step
                # itself is applied in mode 10 and measures the current loop, not the
                # firmware enable sequence. The excursion origin stays at the baseline.
                prehold = Entry("step", math.copysign(cfg.rise_prehold_nm, entry.value),
                                cfg.rise_prehold_s)
                prehold_start = time.perf_counter()
                while time.perf_counter() - prehold_start < cfg.rise_prehold_s:
                    tick = time.perf_counter()
                    since_start = tick - prehold_start
                    await record("prehold", trial, prehold,
                                 enabling=since_start < ENABLE_SETTLE_S, settling=True)
                    await wait_until(tick + period_s)
            active_start = time.perf_counter()
            next_announce = 1.0
            while time.perf_counter() - active_start < entry.duration_s:
                tick = time.perf_counter()
                since_start = tick - active_start
                active_rows.append(await record(
                    entry.kind, trial, entry,
                    enabling=entry.kind == "plateau" and since_start < ENABLE_SETTLE_S,
                    settling=(entry.kind == "plateau"
                              and since_start < ENABLE_SETTLE_S + cfg.ramp_s),
                    fraction=min(1.0, since_start / cfg.ramp_s) if entry.kind == "plateau" else 1.0))
                remaining = entry.duration_s - (time.perf_counter() - active_start)
                if entry.duration_s - remaining >= next_announce:
                    print(f"{remaining:.0f} s left", flush=True)
                    next_announce += 1.0
                await wait_until(tick + period_s)
            await stopped()
            last_end = time.perf_counter()
            i2t_used += cost
            settled = active_rows[len(active_rows) // 5:] or active_rows
            q_values = [r["q_current_a"] for r in settled]
            recorder.event(elapsed(), "plateau_end", value=entry.value, unit=unit,
                           note=format_note(
                               plateau=trial, mean_q_current_a=round(statistics.fmean(q_values), 4),
                               std_q_current_a=round(statistics.pstdev(q_values), 4),
                               mean_torque_nm=round(statistics.fmean(
                                   r["torque_nm"] for r in settled), 5)))
            print(f"stopped; mean q-current {statistics.fmean(q_values):+.3f} A "
                  f"(budget used {i2t_used:.0f}/{cfg.session_i2t_budget_a2s:g} A²s)", flush=True)
            if not rise_time:
                counts[entry.value] = counts.get(entry.value, 0) + 1
                if entry.value != 0.0:
                    sign = 1 if entry.value > 0 else -1
                side = ("zero" if entry.value == 0.0
                        else "positive" if entry.value > 0 else "negative")
                grams = _prompt_reading()
                if grams is None:
                    recorder.event(elapsed(), "scale_reading", value=None,
                                   note=format_note(plateau=trial, current_a=entry.value,
                                                    side=side, status="skipped"))
                else:
                    recorder.event(elapsed(), "scale_reading", value=grams, unit="g",
                                   note=format_note(plateau=trial, current_a=entry.value,
                                                    side=side))
                note = input("note (optional): ").strip()
                if note:
                    recorder.event(elapsed(), "operator_note",
                                   note=format_note(plateau=trial) + "; " + note)
            trial += 1
        status = "completed"
    except BaseException as exc:
        status = "safety_stop" if isinstance(exc, (SafetyStop, TimeoutError)) else "failed"
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if controller is not None:
                await stopped()
        except BaseException as exc:
            status = "cleanup_failed"
            reason = f"{reason or ''}; cleanup: {type(exc).__name__}: {exc}"
            print("STOP NOT CONFIRMED: disconnect power.", flush=True)
            raise
        finally:
            recorder.close(status, reason)
    return str(recorder.path)
