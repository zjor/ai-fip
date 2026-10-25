import unittest

from app.analyze_response_timing import detect


class ResponseTimingDetectionTest(unittest.TestCase):
    def test_detects_known_delayed_step(self):
        rows = [dict(response_time_s=i * .005, current=0. if i < 3 else .4)
                for i in range(8)]
        result = detect(rows, 'current', [0., .01, -.01], .05, 0., .0125)
        self.assertEqual(result['first_response_ms'], 15.)
        self.assertEqual(result['confirmation_response_ms'], 20.)

    def test_spike_and_sign_flip_are_not_a_response(self):
        rows = [dict(response_time_s=i * .005, current=v)
                for i, v in enumerate([0., .4, -.4, 0.])]
        self.assertEqual(detect(rows, 'current', [0., 0.], .05, 0., .0125)['status'], 'not_detected')

    def test_gap_cannot_confirm_a_fast_response(self):
        rows = [dict(response_time_s=t, current=.4) for t in (.005, .04)]
        self.assertEqual(detect(rows, 'current', [0., 0.], .05, 0., .0125)['status'], 'not_detected')

    def test_baseline_noise_can_make_small_pulse_unresolved(self):
        rows = [dict(response_time_s=t, current=.1) for t in (.005, .01)]
        result = detect(rows, 'current', [-.1, 0., .1], .05, 0., .0125)
        self.assertEqual(result['status'], 'not_detected')
        self.assertEqual(result['threshold'], .2)


if __name__ == '__main__':
    unittest.main()
