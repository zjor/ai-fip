import math
import unittest
from dataclasses import replace

from moteus_host.bench.config import REPO_ROOT, load_config
from moteus_host.bench.safety import validate_torque_speed


class TorqueSpeedConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.cfg = self.config.torque_speed
        self.limits = self.config.safety

    def assert_rejected(self, **changes) -> None:
        with self.assertRaises(ValueError):
            validate_torque_speed(replace(self.cfg, **changes), self.limits)

    def test_tracked_defaults_match_spec_and_pass_validation(self) -> None:
        cfg = self.cfg
        self.assertEqual(cfg.bus_voltages_v, (11.0, 12.0, 13.2, 14.8, 16.0, 16.8))
        self.assertEqual(cfg.torques_nm, (0.05, 0.10, 0.15, 0.18))
        self.assertEqual(cfg.directions, (1, -1))
        self.assertEqual(cfg.max_velocity_rev_s, 66.7)
        self.assertEqual(cfg.max_torque_nm, 0.20)
        self.assertEqual(cfg.cap_margin_rev_s, 3.0)
        self.assertEqual(cfg.sag_stop_v, 1.0)
        self.assertEqual(cfg.sag_threshold_v, 0.3)
        self.assertEqual(cfg.bus_tolerance_v, 0.2)
        self.assertEqual(cfg.saturation_alpha_rev_s2, 1.0)
        self.assertEqual(cfg.saturation_settle_s, 0.5)
        self.assertEqual(cfg.min_saturation_rev_s, 5.0)
        self.assertEqual(cfg.hold_s, 1.0)
        self.assertEqual(cfg.step_timeout_s, 15.0)
        self.assertEqual(cfg.kt_nm_per_a, 0.0244)
        self.assertEqual(cfg.expected_noload_rpm_per_v, 228.0)
        self.assertEqual(cfg.response_timeout_s, 0.05)
        self.assertEqual(self.limits.max_rotating_velocity_rev_s, 70.0)
        self.assertEqual(self.limits.max_rotating_torque_nm, 0.20)
        self.assertEqual(self.config.coast_down.max_velocity_rev_s, 55.0)
        self.assertTrue((REPO_ROOT / cfg.loss_record).is_file())
        validate_torque_speed(cfg, self.limits)

    def test_rejects_velocity_cap_above_tier_b(self) -> None:
        self.assert_rejected(max_velocity_rev_s=71.0)

    def test_rejects_torque_above_tier_b(self) -> None:
        self.assert_rejected(torques_nm=(0.05, 0.25))
        self.assert_rejected(max_torque_nm=0.25)

    def test_rejects_top_torque_without_5_percent_margin(self) -> None:
        self.assert_rejected(torques_nm=(0.05, 0.20))

    def test_rejects_top_torque_above_phase_current_stop(self) -> None:
        # 0.18 Nm / 0.02 Nm/A = 9 A > 8 A
        self.assert_rejected(kt_nm_per_a=0.02)

    def test_rejects_cap_margin_below_derate_band(self) -> None:
        # must exceed servo.max_velocity_derate (2.0 rev/s)
        self.assert_rejected(cap_margin_rev_s=1.0)

    def test_rejects_sag_stop_reaching_safety_floor(self) -> None:
        # min(bus_voltages_v) - sag_stop_v must stay >= safety.min_bus_voltage_v
        self.assert_rejected(sag_stop_v=1.5)

    def test_rejects_empty_or_non_finite_lists(self) -> None:
        self.assert_rejected(bus_voltages_v=())
        self.assert_rejected(torques_nm=(math.nan,))

    def test_rejects_settle_longer_than_timeout(self) -> None:
        self.assert_rejected(saturation_settle_s=20.0)

    def test_rejects_non_positive_noload_rpm_per_v(self) -> None:
        self.assert_rejected(expected_noload_rpm_per_v=0.0)

    def test_rejects_min_saturation_above_cap(self) -> None:
        self.assert_rejected(min_saturation_rev_s=70.0)


if __name__ == "__main__":
    unittest.main()
