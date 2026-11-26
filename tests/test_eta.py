import unittest
from clipweb.cli import EtaEstimator


class EtaTests(unittest.TestCase):
    def test_parallel_analysis_and_sequential_forecasts(self):
        eta = EtaEstimator(('audio', 'video', 'smoking', 'render'))
        eta.start('audio', 0.0)
        eta.start('video', 0.0)
        values = {'audio': .5, 'video': .25, 'smoking': 0., 'render': 0.}
        # After 10 s: audio expects 20 s, video 40 s; unstarted stages
        # each use 40 s provisionally, and parallel analysis has 30 s left.
        self.assertEqual(eta.remaining(values, 10.0), 110.0)
        values['audio'] = 1.0
        eta.finish('audio', 20.0)
        values['video'] = 1.0
        eta.finish('video', 40.0)
        eta.start('smoking', 40.0)
        values['smoking'] = .5
        # Smoking's measured remaining time is 10 s; render still uses
        # the initial analysis estimate of 40 s.
        self.assertEqual(eta.remaining(values, 50.0), 50.0)
        values['smoking'] = 1.0
        eta.finish('smoking', 60.0)
        eta.start('render', 60.0)
        values['render'] = .5
        self.assertEqual(eta.remaining(values, 65.0), 5.0)
        values['render'] = 1.0
        self.assertEqual(eta.remaining(values, 70.0), 0.0)

    def test_unknown_until_a_stage_has_a_rate(self):
        eta = EtaEstimator(('audio', 'video', 'smoking', 'render'))
        values = dict.fromkeys(eta.stages, 0.0)
        eta.start('video', 0.0)
        self.assertIsNone(eta.remaining(values, 5.0))

    def test_skipped_stages_have_no_forecast(self):
        eta = EtaEstimator(('audio', 'video', 'smoking', 'render'))
        eta.start('video', 0.0)
        values = {'audio': 1., 'video': .5, 'smoking': 1., 'render': 0.}
        self.assertEqual(eta.remaining(values, 10.), 30.)
