import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moteus_host.clock import (
    load_calibration,
    nearest_equivalent_position,
    save_calibration,
)


class NearestEquivalentPositionTest(unittest.TestCase):
    def test_uses_same_turn_when_nearby(self) -> None:
        self.assertAlmostEqual(nearest_equivalent_position(0.20, 0.25), 0.25)

    def test_wraps_backward_to_shortest_move(self) -> None:
        self.assertAlmostEqual(nearest_equivalent_position(0.95, 0.05), 1.05)

    def test_works_after_multiple_turns(self) -> None:
        self.assertAlmostEqual(nearest_equivalent_position(3.10, 0.90), 2.90)


class CalibrationStorageTest(unittest.TestCase):
    def test_round_trips_modulo_position_and_direction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            with patch("moteus_host.clock.CALIBRATION_PATH", path):
                save_calibration(1, 2.25, -1)
                calibration = load_calibration(1)

        self.assertEqual(calibration["upright_position_modulo"], 0.25)
        self.assertEqual(calibration["clockwise_sign"], -1)


if __name__ == "__main__":
    unittest.main()
