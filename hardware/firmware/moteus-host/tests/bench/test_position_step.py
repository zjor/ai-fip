import unittest

from moteus_host.bench.experiments.position_step import target_sequence


class TargetSequenceTest(unittest.TestCase):
    def test_is_bidirectional_and_returns_to_origin(self) -> None:
        sequence = target_sequence(2.0, 0.125, 1)
        self.assertEqual(
            sequence,
            [
                ("settle", 2.0),
                ("positive", 2.125),
                ("center", 2.0),
                ("negative", 1.875),
                ("center", 2.0),
            ],
        )
