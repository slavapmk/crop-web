import unittest
import numpy as np
from clipweb.cli import _speech_evidence


class SpeechEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'window_seconds': .6, 'speech_ratio': .2,
                    'lip_motion_threshold': 2.8, 'mouth_motion_ratio': .5}

    def test_short_voice_and_isolated_motion_do_not_trigger(self):
        voiced = np.array([0, 1, 0, 0, 0])
        total = np.full(5, 6)
        motion = np.array([0., 0., 3., 0., 0.])
        vad, moving = _speech_evidence(voiced, total, motion, self.cfg, .2)
        self.assertFalse(vad.any())
        self.assertFalse(moving.any())

    def test_repeated_motion_and_voice_trigger(self):
        voiced = np.full(5, 4)
        total = np.full(5, 6)
        motion = np.array([0., 3., 3., 0., 0.])
        vad, moving = _speech_evidence(voiced, total, motion, self.cfg, .2)
        self.assertTrue(vad[2])
        self.assertTrue(moving[2])

    def test_missing_audio_is_not_speech(self):
        zero = np.zeros(5)
        vad, _ = _speech_evidence(zero, zero, zero, self.cfg, .2)
        self.assertFalse(vad.any())
