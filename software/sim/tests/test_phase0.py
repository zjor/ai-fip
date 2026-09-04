import math
import unittest

from app.phase0 import (
    DEFAULT_CONFIG,
    MOTEUS_TORQUE_FACTOR,
    RAD_S_PER_RPM,
    DesignPoint,
    angular_momentum_nms,
    battery_placement,
    build_report,
    load_config,
    pessimistic_point,
    recoverable_angle_deg,
)
from app.phase0_control import SensorScenario, available_peak_torque_nm, simulate
from app.phase0_swingup import simulate_swing_up


class Phase0Test(unittest.TestCase):
    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG)
        self.report = build_report(self.config)
        self.c1, self.r4 = self.config.actuators

    def test_nominal_gravity_coefficient_matches_cad_mass_model(self):
        # hardware/cad `make masses` 2026-09-04: m_t 0.7765 kg, l_c 0.1459 m
        expected = 0.7765 * 9.81 * 0.1459
        self.assertAlmostEqual(self.report["nominal_design"]["gravity_coefficient_nm"], expected)

    def test_recoverable_angle_is_capped_at_ninety_degrees(self):
        self.assertEqual(recoverable_angle_deg(2.0, 1.0, 1.5), 90.0)

    def test_torque_constant_uses_moteus_firmware_convention(self):
        # fw/bldc_servo.cc: kTorqueFactor = (3/2) * (1/sqrt3) * (60/2pi)
        self.assertAlmostEqual(MOTEUS_TORQUE_FACTOR, 8.2699, places=3)
        self.assertAlmostEqual(self.c1.motor_kt_nm_per_a, 8.2699 / 330.0, places=5)

    def test_c1_passes_finger_poke_but_fails_preferred_angle(self):
        c1 = self.report["robust_actuators"][0]
        self.assertEqual(c1["peak_gate"], "PASS")
        self.assertEqual(c1["preferred_peak_gate"], "FAIL")
        self.assertEqual(c1["driver_continuous_proxy_gate"], "FAIL")

    def test_r4_passes_all_analytical_peak_gates(self):
        r4 = self.report["robust_actuators"][1]
        self.assertEqual(r4["peak_gate"], "PASS")
        self.assertEqual(r4["preferred_peak_gate"], "PASS")
        self.assertEqual(r4["driver_continuous_proxy_gate"], "PASS")
        self.assertAlmostEqual(r4["peak_torque_nm"], 1.7)

    def test_momentum_conversion(self):
        self.assertAlmostEqual(angular_momentum_nms(0.003, 60.0), 0.003 * 2.0 * math.pi)

    def test_time_constant_uses_unstable_upright_linearization(self):
        point = DesignPoint(0.55, 0.18, 0.025, 0.003)
        expected = math.sqrt(0.025 / (0.55 * 9.81 * 0.18))
        self.assertAlmostEqual(point.upright_time_constant_s(), expected)

    def test_gate_remains_not_ready_while_mass_model_is_planning_range(self):
        self.assertEqual(self.report["status"], "NOT READY")
        for actuator in self.report["robust_actuators"]:
            self.assertIn("unverified", actuator["motor_thermal_gate"])

    def test_no_load_speed_follows_moteus_effective_voltage(self):
        # v_per_hz = 60/(sqrt3 Kv); V_eff = 0.5 V_bus ratio (1 - margin)
        v_eff = 0.5 * 12.0 * 0.94 * 0.85
        expected_rpm = v_eff / (60.0 / (math.sqrt(3.0) * 330.0)) * 60.0
        self.assertAlmostEqual(self.r4.no_load_rpm(12.0), expected_rpm, places=6)
        self.assertLess(expected_rpm, 330.0 * 12.0)  # below the hobby Kv*V rule

    def test_torque_speed_envelope_reaches_zero_at_no_load_speed(self):
        speed_rad_s = self.c1.no_load_rpm(12.0) * RAD_S_PER_RPM
        self.assertAlmostEqual(available_peak_torque_nm(self.c1, 12.0, speed_rad_s), 0.0, places=6)
        self.assertGreater(available_peak_torque_nm(self.c1, 12.0, 0.0), 0.49)

    def test_r4_peak_torque_is_motor_limited_at_stall_and_fades_with_speed(self):
        self.assertAlmostEqual(self.r4.available_torque_nm(12.0, 0.0), 1.7)
        knee = self.r4.knee_rpm(12.0)
        self.assertGreater(knee, 500.0)
        self.assertLess(self.r4.available_torque_nm(12.0, 2.0 * knee * RAD_S_PER_RPM), 1.7)

    def test_battery_on_pivot_axis_does_not_change_gravity_torque(self):
        point = DesignPoint(0.55, 0.18, 0.025, 0.003)
        report = battery_placement(point, self.config.battery)
        self.assertAlmostEqual(
            report["on_pivot_axis"]["gravity_coefficient_nm"], point.gravity_coefficient_nm()
        )
        self.assertGreater(
            report["at_top_end"]["gravity_coefficient_nm"], point.gravity_coefficient_nm() * 1.2
        )

    def test_nominal_reference_sensor_scenario_stabilizes(self):
        point = DesignPoint(0.55, 0.18, 0.025, 0.003)
        result = simulate(point, self.r4, 12.0, SensorScenario(), duration_s=3.0)
        self.assertTrue(result.passed, result.reason)

    def test_r4_swings_up_nominal_design_at_minimum_bus_voltage(self):
        point = DesignPoint(0.55, 0.18, 0.025, 0.003)
        result = simulate_swing_up(point, self.r4, 12.0, duration_s=8.0)
        self.assertTrue(result.caught, result.reason)
        self.assertTrue(result.upright_settled, result.reason)
        self.assertLess(result.wheel_speed_fraction_of_no_load, 0.7)

    def test_pessimistic_point_is_heaviest_longest_lightest_inertia(self):
        point = pessimistic_point(self.config.design)
        self.assertEqual(point.total_mass_kg, self.config.design.total_mass_kg.high)
        self.assertEqual(point.flywheel_inertia_kg_m2, self.config.design.flywheel_inertia_kg_m2.low)


if __name__ == "__main__":
    unittest.main()
