import math
import unittest

from app.phase0 import (
    DEFAULT_CONFIG,
    MOTEUS_TORQUE_FACTOR,
    RAD_S_PER_RPM,
    DesignPoint,
    LossTorque,
    angular_momentum_nms,
    battery_placement,
    build_report,
    load_config,
    pessimistic_point,
    recoverable_angle_deg,
)
from app.phase0_control import SensorScenario, available_peak_torque_nm, dynamics, simulate
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
        # PNDLM-9 measured K_t; Kv follows from it (PNDLM-12: 339 rpm/V).
        self.assertAlmostEqual(self.c1.motor_kt_nm_per_a, 0.0244)
        self.assertAlmostEqual(self.c1.motor.kv_rpm_per_v, 338.9, delta=0.1)

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
            self.assertIn("assumption", actuator["motor_thermal_gate"])

    def test_no_load_speed_follows_moteus_effective_voltage(self):
        # v_per_hz = 60/(sqrt3 Kv); V_eff = 0.5 V_bus ratio (1 - margin)
        kv = MOTEUS_TORQUE_FACTOR / 0.0244
        v_eff = 0.5 * 12.0 * 1.0 * 0.85
        expected_rpm = v_eff / (60.0 / (math.sqrt(3.0) * kv)) * 60.0
        self.assertAlmostEqual(self.r4.no_load_rpm(12.0), expected_rpm, places=6)
        self.assertLess(expected_rpm, kv * 12.0)  # below the hobby Kv*V rule
        # PNDLM-12 measured 248.3 +/- 1.0 rpm/V of loaded bus.
        self.assertAlmostEqual(self.r4.no_load_rpm(1.0), 248.3, delta=2.0)

    def test_hot_motor_corner_matches_pndlm13(self):
        motor = self.config.motor
        self.assertAlmostEqual(motor.hot_resistance_ohm() / motor.phase_resistance_ohm, 1.236, places=3)
        self.assertAlmostEqual(motor.hot_torque_constant_nm_per_a(), 0.0227, delta=0.0001)
        # 0.20 Nm continuous holds at the pessimistic assumed R_th 6 K/W.
        self.assertGreater(self.r4.motor_thermal_continuous_torque_nm(6.0), 0.20)
        self.assertLess(self.r4.motor_thermal_continuous_torque_nm(9.0), 0.20)

    def test_torque_speed_envelope_reaches_zero_at_no_load_speed(self):
        speed_rad_s = self.c1.no_load_rpm(12.0) * RAD_S_PER_RPM
        self.assertAlmostEqual(available_peak_torque_nm(self.c1, 12.0, speed_rad_s), 0.0, places=6)
        self.assertAlmostEqual(available_peak_torque_nm(self.c1, 12.0, 0.0), 20.0 * 0.0244)

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

    def test_loss_corners_match_bench_record(self):
        losses = self.config.losses
        self.assertEqual(LossTorque.corner(losses, "none").torque_nm(300.0), 0.0)
        nominal = LossTorque.corner(losses, "nominal")
        # PNDLM-11 table: 17.26 mNm at 3000 rpm, 6.44 mNm at 500 rpm.
        self.assertAlmostEqual(nominal.torque_nm(3000 * RAD_S_PER_RPM), 0.01726, delta=1e-4)
        self.assertAlmostEqual(nominal.torque_nm(-500 * RAD_S_PER_RPM), -0.00644, delta=1e-4)
        self.assertAlmostEqual(LossTorque.corner(losses, "high").coulomb_nm, 0.007)

    def test_loss_torque_slows_wheel_and_reacts_on_body(self):
        point = DesignPoint(0.7, 0.15, 0.03, 0.0027)
        losses = LossTorque(coulomb_nm=0.005, viscous_nm_s=0.0)
        rate = dynamics(point, [0.0, 0.0, 100.0], 0.0, losses)
        self.assertAlmostEqual(rate[2], -0.005 / 0.0027, places=6)
        self.assertAlmostEqual(rate[1], 0.005 / 0.03, places=6)
        # Friction acts on the speed relative to the body.
        still = dynamics(point, [0.0, 2.0, 2.0], 0.0, losses)
        self.assertEqual(still[2], 0.0)

    def test_lqr_holds_upright_with_high_losses(self):
        point = pessimistic_point(self.config.design)
        high = LossTorque.corner(self.config.losses, "high")
        result = simulate(point, self.r4, 12.0, SensorScenario(), losses=high)
        self.assertTrue(result.passed, result.reason)


if __name__ == "__main__":
    unittest.main()
