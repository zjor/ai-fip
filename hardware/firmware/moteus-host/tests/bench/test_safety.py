import math
import unittest

from moteus_host.bench.config import load_config
from moteus_host.bench.safety import SafetyStop, check_state


class SafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = load_config().safety
        self.state = {
            "fault": 0,
            "mode": 0,
            "bus_voltage_v": 24.0,
            "controller_temperature_c": 25.0,
            "motor_temperature_c": math.nan,
            "q_current_a": 0.0,
            "d_current_a": 0.0,
            "torque_nm": 0.0,
            "velocity_rev_s": 0.0,
            "position_rev": 0.0,
        }

    def test_accepts_invalid_unconfigured_motor_temperature(self) -> None:
        check_state(self.state, self.limits, origin_position_rev=0.0)

    def test_rejects_position_outside_tier_a_envelope(self) -> None:
        self.state["position_rev"] = 0.5
        with self.assertRaisesRegex(SafetyStop, "position excursion"):
            check_state(self.state, self.limits, origin_position_rev=0.0)

    def test_rejects_overvoltage(self) -> None:
        self.state["bus_voltage_v"] = 31.0
        with self.assertRaisesRegex(SafetyStop, "bus voltage"):
            check_state(self.state, self.limits)

    def test_friction_velocity_uses_its_experiment_specific_limit(self) -> None:
        self.state["velocity_rev_s"] = (
            0.9 * self.limits.max_friction_velocity_rev_s
        )
        check_state(
            self.state,
            self.limits,
            max_abs_velocity_rev_s=self.limits.max_friction_velocity_rev_s,
        )
        self.state["velocity_rev_s"] = (
            1.1 * self.limits.max_friction_velocity_rev_s
        )
        with self.assertRaisesRegex(SafetyStop, "velocity"):
            check_state(
                self.state,
                self.limits,
                max_abs_velocity_rev_s=self.limits.max_friction_velocity_rev_s,
            )

    def test_rejects_missing_critical_telemetry(self) -> None:
        self.state["bus_voltage_v"] = None
        with self.assertRaisesRegex(SafetyStop, "bus_voltage_v"):
            check_state(self.state, self.limits)

    def test_limit_indicator_in_control_mode_is_not_a_fault(self) -> None:
        # moteus reports 96..103 during valid control modes to say which
        # limit is active; 98 is the bus-voltage limit, i.e. saturation.
        self.state.update({"mode": 10, "fault": 98})
        check_state(self.state, self.limits)

    def test_fault_mode_with_code_stops(self) -> None:
        self.state.update({"mode": 1, "fault": 33})
        with self.assertRaises(SafetyStop) as ctx:
            check_state(self.state, self.limits)
        self.assertIn("33 (motor driver fault)", str(ctx.exception))

    def test_fault_mode_with_limit_code_stops(self) -> None:
        self.state.update({"mode": 1, "fault": 98})
        with self.assertRaises(SafetyStop):
            check_state(self.state, self.limits)

    def test_hard_fault_code_outside_fault_mode_stops(self) -> None:
        self.state.update({"mode": 10, "fault": 40})
        with self.assertRaises(SafetyStop):
            check_state(self.state, self.limits)

