import math
import unittest

from app.phase0 import (
    DEFAULT_CONFIG,
    DesignPoint,
    angular_momentum_nms,
    build_report,
    load_config,
    recoverable_angle_deg,
)
from app.phase0_control import SensorScenario, available_peak_torque_nm, simulate


class Phase0Test(unittest.TestCase):
    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG)
        self.report = build_report(self.config)

    def test_nominal_gravity_coefficient_matches_documented_estimate(self):
        expected = 0.55 * 9.81 * 0.18
        self.assertAlmostEqual(
            self.report["nominal_design"]["gravity_coefficient_nm"], expected
        )

    def test_recoverable_angle_is_capped_at_ninety_degrees(self):
        self.assertEqual(recoverable_angle_deg(2.0, 1.0, 1.5), 90.0)

    def test_c1_fails_peak_gate_at_pessimistic_corner(self):
        c1 = self.report["robust_actuators"][0]
        self.assertEqual(c1["peak_gate"], "FAIL")
        self.assertLess(c1["peak_margin_at_target_angle"], 1.5)

    def test_r4_passes_analytical_peak_gate(self):
        r4 = self.report["robust_actuators"][1]
        self.assertEqual(r4["peak_gate"], "PASS")
        self.assertGreaterEqual(r4["peak_margin_at_target_angle"], 1.5)

    def test_momentum_conversion(self):
        self.assertAlmostEqual(
            angular_momentum_nms(0.003, 60.0), 0.003 * 2.0 * math.pi
        )

    def test_time_constant_uses_unstable_upright_linearization(self):
        point = DesignPoint(0.55, 0.18, 0.025, 0.003)
        expected = math.sqrt(0.025 / (0.55 * 9.81 * 0.18))
        self.assertAlmostEqual(point.upright_time_constant_s(), expected)

    def test_gate_remains_not_ready_for_unknown_physical_inputs(self):
        self.assertEqual(self.report["status"], "NOT READY")
        for actuator in self.report["robust_actuators"]:
            self.assertEqual(actuator["motor_thermal_gate"], "UNKNOWN")
            self.assertEqual(actuator["swing_up_momentum_gate"], "UNKNOWN")

    def test_torque_speed_envelope_reaches_zero_at_no_load_speed(self):
        actuator = self.config.actuators[0]
        speed = actuator.no_load_rpm(self.config.minimum_bus_voltage_v)
        speed_rad_s = speed * 2.0 * math.pi / 60.0
        self.assertAlmostEqual(
            available_peak_torque_nm(
                actuator, self.config.minimum_bus_voltage_v, speed_rad_s
            ),
            0.0,
        )

    def test_nominal_reference_sensor_scenario_stabilizes(self):
        point = DesignPoint(0.55, 0.18, 0.025, 0.003)
        result = simulate(
            point,
            self.config.actuators[1],
            self.config.minimum_bus_voltage_v,
            SensorScenario(),
            duration_s=3.0,
        )
        self.assertTrue(result.passed, result.reason)


if __name__ == "__main__":
    unittest.main()
