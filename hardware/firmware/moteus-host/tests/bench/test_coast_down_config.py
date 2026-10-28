import math
import unittest
from dataclasses import replace

from moteus_host.bench.config import REPO_ROOT, CoastDownConfig, load_config
from moteus_host.bench.safety import SafetyStop, check_state, validate_coast_down


class CoastDownConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.cfg = self.config.coast_down
        self.limits = self.config.safety

    def test_tracked_defaults_match_spec_and_pass_validation(self) -> None:
        validate_coast_down(self.cfg, self.limits)
        self.assertIsInstance(self.cfg, CoastDownConfig)
        self.assertEqual(self.cfg.ladder_rpm, (500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0))
        self.assertEqual(self.cfg.accel_torques_nm, (0.05, 0.10, 0.15))
        self.assertEqual(self.cfg.direction, 1)
        self.assertEqual(self.cfg.spin_up_torque_nm, 0.10)
        self.assertEqual(self.cfg.max_velocity_rev_s, 55.0)
        self.assertEqual(self.cfg.max_torque_nm, 0.20)
        self.assertEqual(self.cfg.kt_nm_per_a, 0.0244)
        self.assertEqual(self.cfg.coast_settle_s, 0.25)
        self.assertEqual(self.limits.max_rotating_velocity_rev_s, 55.0)
        self.assertEqual(self.limits.max_rotating_torque_nm, 0.20)
        self.assertTrue((REPO_ROOT / self.cfg.inertia_record).is_file())

    def test_rejects_ladder_above_cap(self) -> None:
        cfg = replace(self.cfg, ladder_rpm=(500.0, 4000.0))
        with self.assertRaisesRegex(ValueError, "ladder"):
            validate_coast_down(cfg, self.limits)

    def test_rejects_torque_above_tier_b(self) -> None:
        cfg = replace(self.cfg, spin_up_torque_nm=0.25)
        with self.assertRaisesRegex(ValueError, "torque"):
            validate_coast_down(cfg, self.limits)
        cfg = replace(self.cfg, max_torque_nm=0.30)
        with self.assertRaisesRegex(ValueError, "torque"):
            validate_coast_down(cfg, self.limits)

    def test_rejects_velocity_cap_above_tier_b(self) -> None:
        cfg = replace(self.cfg, max_velocity_rev_s=60.0)
        with self.assertRaisesRegex(ValueError, "velocity"):
            validate_coast_down(cfg, self.limits)

    def test_rejects_coast_stop_above_lowest_step(self) -> None:
        cfg = replace(self.cfg, ladder_rpm=(20.0,), coast_stop_rev_s=0.4)
        with self.assertRaisesRegex(ValueError, "coast_stop"):
            validate_coast_down(cfg, self.limits)

    def test_rejects_settle_longer_than_coast_timeout(self) -> None:
        cfg = replace(self.cfg, coast_settle_s=200.0)
        with self.assertRaisesRegex(ValueError, "coast_settle_s"):
            validate_coast_down(cfg, self.limits)

    def test_rejects_bad_direction(self) -> None:
        cfg = replace(self.cfg, direction=0)
        with self.assertRaisesRegex(ValueError, "direction"):
            validate_coast_down(cfg, self.limits)

    def test_rejects_empty_ladder_and_non_finite(self) -> None:
        with self.assertRaisesRegex(ValueError, "ladder"):
            validate_coast_down(replace(self.cfg, ladder_rpm=()), self.limits)
        with self.assertRaisesRegex(ValueError, "plateau_s"):
            validate_coast_down(replace(self.cfg, plateau_s=math.nan), self.limits)

    def test_check_state_torque_override(self) -> None:
        state = {
            "fault": 0, "mode": 10, "bus_voltage_v": 24.0, "controller_temperature_c": 25.0,
            "motor_temperature_c": math.nan, "q_current_a": 6.0, "d_current_a": 0.0,
            "torque_nm": 0.18, "velocity_rev_s": 0.0, "position_rev": 0.0,
        }
        check_state(state, self.limits, max_torque_nm=0.20)
        with self.assertRaisesRegex(SafetyStop, "torque"):
            check_state(state, self.limits)


if __name__ == "__main__":
    unittest.main()
