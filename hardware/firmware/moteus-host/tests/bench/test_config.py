import unittest

from moteus_host.bench.config import load_config
from moteus_host.bench.safety import validate_friction, validate_position_step


class ConfigTest(unittest.TestCase):
    def test_tracked_defaults_fit_tier_a_limits(self) -> None:
        config = load_config()
        validate_position_step(config.position_step, config.safety)
        validate_friction(config.friction, config.safety)
        self.assertEqual(config.common.controller_id, 1)
        self.assertEqual(config.position_step.cycles, 2)
