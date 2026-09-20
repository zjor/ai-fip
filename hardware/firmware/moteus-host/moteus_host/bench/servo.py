"""Small moteus adapter with one explicit telemetry schema."""

from __future__ import annotations

from typing import Any


STATE_FIELDS = (
    "mode",
    "fault",
    "position_rev",
    "velocity_rev_s",
    "torque_nm",
    "q_current_a",
    "d_current_a",
    "bus_voltage_v",
    "electrical_power_w",
    "controller_temperature_c",
    "motor_temperature_c",
    "home_state",
    "trajectory_complete",
)


def create_controller(
    controller_id: int, *, include_motor_temperature: bool = False
) -> tuple[Any, Any]:
    import moteus

    query = moteus.QueryResolution()
    query.mode = moteus.INT8
    query.fault = moteus.INT8
    query.position = moteus.F32
    query.velocity = moteus.F32
    query.torque = moteus.F32
    # Keep the complete response inside one CAN-FD frame. INT16 still gives
    # 0.1 A, 0.1 V, 0.05 W and 0.1 °C resolution, which is sufficient here.
    query.q_current = moteus.INT16
    query.d_current = moteus.INT16
    query.voltage = moteus.INT16
    query.power = moteus.INT16
    query.temperature = moteus.INT16
    query.motor_temperature = (
        moteus.INT16 if include_motor_temperature else moteus.IGNORE
    )
    query.home_state = moteus.INT8
    query.trajectory_complete = moteus.INT8
    return moteus.Controller(id=controller_id, query_resolution=query), moteus


def result_to_state(result: Any, moteus: Any) -> dict[str, Any]:
    values = result.values
    register_fields = {
        "mode": moteus.Register.MODE,
        "fault": moteus.Register.FAULT,
        "position_rev": moteus.Register.POSITION,
        "velocity_rev_s": moteus.Register.VELOCITY,
        "torque_nm": moteus.Register.TORQUE,
        "q_current_a": moteus.Register.Q_CURRENT,
        "d_current_a": moteus.Register.D_CURRENT,
        "bus_voltage_v": moteus.Register.VOLTAGE,
        "electrical_power_w": moteus.Register.POWER,
        "controller_temperature_c": moteus.Register.TEMPERATURE,
        "motor_temperature_c": moteus.Register.MOTOR_TEMPERATURE,
        "home_state": moteus.Register.HOME_STATE,
        "trajectory_complete": moteus.Register.TRAJECTORY_COMPLETE,
    }
    return {name: values.get(register) for name, register in register_fields.items()}


async def snapshot_controller(controller: Any, moteus: Any) -> tuple[str, dict[str, Any]]:
    stream = moteus.Stream(controller)
    # `tel stop` intentionally has no normal command response. Send it without
    # waiting for `OK`, then drain any prior diagnostic telemetry.
    await stream.write_message(b"tel stop")
    await stream.flush_read()
    configuration = (await stream.command(b"conf enumerate")).decode(
        "utf-8", errors="replace"
    )
    firmware = await stream.read_data("firmware")
    device = {
        "firmware_abi_version": getattr(firmware, "version", None),
        "model": getattr(firmware, "model", None),
        "family": getattr(firmware, "family", None),
        "hardware_revision": getattr(firmware, "hwrev", None),
        "serial_number": list(getattr(firmware, "serial_number", [])),
    }
    return configuration, device


async def command_position(
    controller: Any,
    *,
    position_rev: float,
    maximum_torque_nm: float,
    velocity_limit_rev_s: float,
    accel_limit_rev_s2: float,
    watchdog_timeout_s: float,
) -> Any:
    return await controller.set_position(
        position=position_rev,
        velocity=0.0,
        maximum_torque=maximum_torque_nm,
        velocity_limit=velocity_limit_rev_s,
        accel_limit=accel_limit_rev_s2,
        watchdog_timeout=watchdog_timeout_s,
        query=True,
    )
