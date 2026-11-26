import logging
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from clipweb import cli


class Face:
    def detectMultiScale(self, *args, **kwargs): return [(35, 15, 55, 55)]


class Voice:
    def __init__(self, *args): pass
    def is_speech(self, *args): return True


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required')
class AnalysisIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.video = root / 'stream.mp4'
        self.config = root / 'config.yaml'
        self.config.write_text('{}', encoding='utf8')
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
                        '-i', 'testsrc2=size=128x96:rate=10:duration=3', '-f', 'lavfi', '-i',
                        'sine=frequency=440:sample_rate=48000:duration=3', '-c:v', 'libx264',
                        '-c:a', 'aac', '-shortest', str(self.video)], check=True, capture_output=True)
        self.cfg = cli.load_config(self.config)
        self.cfg['webcam'] = {'x': 0, 'y': 0, 'width': 128, 'height': 96}
        self.logger = logging.getLogger('analysis-test')

    def tearDown(self): self.tmp.cleanup()

    def test_face_mode_does_not_query_voice(self):
        self.cfg['speech']['enabled'] = False
        with patch.object(cli, '_face_detector', return_value=Face()), \
             patch.object(cli.webrtcvad, 'Vad') as vad:
            vad.return_value.is_speech.side_effect = AssertionError('voice must not be analyzed')
            intervals, duration = cli.analyze_video(self.video, self.cfg, self.logger)
        vad.return_value.is_speech.assert_not_called()
        self.assertGreater(sum(end-start for start, end in intervals), 2.5)
        self.assertAlmostEqual(duration, 3.0, delta=.1)

    def test_combined_voice_and_motion_remove_speech(self):
        self.cfg['speech']['enabled'] = True
        self.cfg['speech']['method'] = 'both'
        with patch.object(cli, '_face_detector', return_value=Face()), \
             patch.object(cli.webrtcvad, 'Vad', Voice), \
             patch.object(cli, '_mouth_motion', return_value=4.):
            intervals, _ = cli.analyze_video(self.video, self.cfg, self.logger)
        self.assertLess(sum(end-start for start, end in intervals), .5)

    def test_mouth_mode_and_inverted_speech(self):
        self.cfg['speech']['enabled'] = True
        self.cfg['speech']['method'] = 'mouth'
        self.cfg['speech']['invert'] = True
        with patch.object(cli, '_face_detector', return_value=Face()), \
             patch.object(cli, '_mouth_motion', return_value=4.):
            intervals, _ = cli.analyze_video(self.video, self.cfg, self.logger)
        self.assertGreater(sum(end-start for start, end in intervals), 2.5)

    def test_face_gap_tolerance_bridges_only_short_gaps(self):
        class MissingFace:
            def __init__(self): self.count = 0
            def detectMultiScale(self, *args, **kwargs):
                self.count += 1
                return [] if self.count == 5 else [(35, 15, 55, 55)]
        self.cfg['speech']['enabled'] = False
        self.cfg['face']['missing_tolerance_seconds'] = .2
        with patch.object(cli, '_face_detector', return_value=MissingFace()):
            intervals, _ = cli.analyze_video(self.video, self.cfg, self.logger)
        self.assertEqual(len(intervals), 1)
        self.cfg['face']['missing_tolerance_seconds'] = 0
        with patch.object(cli, '_face_detector', return_value=MissingFace()):
            intervals, _ = cli.analyze_video(self.video, self.cfg, self.logger)
        self.assertEqual(len(intervals), 2)

    def test_no_audio_does_not_remove_face(self):
        silent = self.video.with_name('silent.mp4')
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(self.video),
                        '-map', '0:v:0', '-c', 'copy', str(silent)], check=True, capture_output=True)
        self.cfg['speech']['enabled'] = True
        self.cfg['speech']['method'] = 'speech'
        with patch.object(cli, '_face_detector', return_value=Face()):
            intervals, _ = cli.analyze_video(silent, self.cfg, self.logger)
        self.assertGreater(sum(end-start for start, end in intervals), 2.5)

    def test_no_face_or_speech_filter_keeps_full_video_without_detectors(self):
        self.cfg['face']['enabled'] = False
        self.cfg['speech']['enabled'] = False
        with patch.object(cli, '_face_detector', side_effect=AssertionError('face detector must not run')), \
             patch.object(cli.webrtcvad, 'Vad', side_effect=AssertionError('VAD must not run')):
            intervals, duration = cli.analyze_video(self.video, self.cfg, self.logger)
        self.assertEqual(intervals, [(0., duration)])

    def test_full_frame_uses_source_dimensions(self):
        self.assertEqual(cli._full_frame_box(self.video), {'x': 0, 'y': 0, 'width': 128, 'height': 96})

    def test_audio_only_speech_filter_works_without_face_detector(self):
        self.cfg['face']['enabled'] = False
        self.cfg['speech'].update(enabled=True, method='speech')
        self.cfg['mute']['enabled'] = True
        with patch.object(cli, '_face_detector', side_effect=AssertionError('face detector must not run')), \
             patch.object(cli.webrtcvad, 'Vad', Voice):
            intervals, _ = cli.analyze_video(self.video, self.cfg, self.logger)
        self.assertLess(sum(end-start for start, end in intervals), .5)

    def test_mouth_filter_without_face_filter_keeps_missing_faces(self):
        class NoFace:
            def detectMultiScale(self, *args, **kwargs): return []
        self.cfg['face']['enabled'] = False
        self.cfg['speech'].update(enabled=True, method='mouth')
        with patch.object(cli, '_face_detector', return_value=NoFace()):
            intervals, duration = cli.analyze_video(self.video, self.cfg, self.logger)
        self.assertEqual(intervals, [(0., duration)])

    def test_smoking_hits_create_removal_intervals(self):
        self.cfg['smoking']['min_hits'] = 2
        output = np.zeros((1, 5, 10), dtype=np.float32)
        output[0, 4, :] = .9
        with patch.object(cli, '_smoking_predictor', return_value=lambda blob: output):
            events = cli.detect_smoking(self.video, [(0., 3.)], self.cfg, Path('unused'), self.logger)
        self.assertEqual(len(events), 1)
        self.assertLessEqual(events[0][0], 0.)
        self.assertGreater(events[0][1], 2.)
