import unittest

from app.analyze_position_step import final_entry, first_sustained, stats


class PositionStepAnalysisTest(unittest.TestCase):
    def test_isolated_spikes_do_not_count_as_motion(self):
        self.assertEqual(first_sustained([True, False, True, True, True]), 2)
        self.assertIsNone(first_sustained([False, True, True]))

    def test_first_entry_is_not_settling(self):
        self.assertEqual(final_entry([False, True, True, False, True]), 4)
        self.assertIsNone(final_entry([False, True, True, False]))
        self.assertEqual(final_entry([True, True, True]), 0)

    def test_tail_outlier_is_preserved(self):
        result = stats([2.0] * 99 + [30.0])
        self.assertEqual(result['median'], 2.0)
        self.assertAlmostEqual(result['p99'], 2.28)
        self.assertEqual(result['max'], 30.0)


if __name__ == '__main__':
    unittest.main()
