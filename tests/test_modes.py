import contextlib
import io
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from clipweb import cli


class ModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = Path.cwd()
        self.root = Path(self.tmp.name)
        os.chdir(self.root)
        (self.root / 'input').mkdir()
        (self.root / 'input' / 'sample.mp4').write_bytes(b'source')
        (self.root / 'model.onnx').write_bytes(b'model')

    def tearDown(self):
        logger = logging.getLogger('clipweb')
        for handler in logger.handlers: handler.close()
        logger.handlers.clear()
        os.chdir(self.previous)
        self.tmp.cleanup()

    def run_mode(self, mode):
        data = {'processing': {'mode': mode},
                'webcam': {'auto': True, 'samples': 8, 'x': 0, 'y': 0, 'width': 264, 'height': 168},
                'analysis': {'speech_detection': 'both', 'invert_speech': True},
                'smoking': {'model': 'model.onnx', 'enabled': True},
                'output': {'min_segment_seconds': 0.5, 'aspect_ratio': '16:10'}}
        Path('config.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')
        with patch.object(cli, 'detect_webcam', return_value={'x': 0, 'y': 0, 'width': 264, 'height': 168}) as detect, \
             patch.object(cli, 'analyze_video', return_value=([(0., 3.), (4., 8.)], 10.)) as analysis, \
             patch.object(cli, 'detect_smoking', return_value=[(1., 2.)]) as smoking, \
             patch.object(cli, 'render_episodes') as episodes, \
             patch.object(cli, 'render_video') as legacy, \
             contextlib.redirect_stderr(io.StringIO()):
            cli.main()
        return detect, analysis, smoking, episodes, legacy

    def test_episode_mode_forces_no_speech_and_separate_files(self):
        detect, analysis, smoking, episodes, legacy = self.run_mode('episodes')
        detect.assert_called_once()
        used_cfg = analysis.call_args.args[1]
        self.assertEqual(used_cfg['analysis']['speech_detection'], 'face')
        self.assertFalse(used_cfg['analysis']['invert_speech'])
        self.assertEqual((used_cfg['webcam']['width'], used_cfg['webcam']['height']), (256, 160))
        self.assertEqual(used_cfg['output']['join_gap_seconds'], 0)
        smoking.assert_called_once()
        episodes.assert_called_once()
        self.assertEqual(episodes.call_args.args[3], [(0., 1.), (2., 3.), (4., 8.)])
        legacy.assert_not_called()

    def test_legacy_preserves_original_mode_and_rectangle(self):
        detect, analysis, smoking, episodes, legacy = self.run_mode('legacy')
        detect.assert_not_called()
        used_cfg = analysis.call_args.args[1]
        self.assertEqual(used_cfg['analysis']['speech_detection'], 'both')
        self.assertTrue(used_cfg['analysis']['invert_speech'])
        self.assertEqual((used_cfg['webcam']['width'], used_cfg['webcam']['height']), (264, 168))
        episodes.assert_not_called()
        legacy.assert_called_once()

    def test_cached_analysis_and_smoking_are_reused(self):
        cfg = {'processing': {'mode': 'legacy'}, 'smoking': {'model': 'model.onnx'}}
        Path('config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
        with patch.object(cli, 'analyze_video', return_value=([(0., 3.)], 3.)) as analysis, \
             patch.object(cli, 'detect_smoking', return_value=[]) as smoking, \
             patch.object(cli, 'render_video') as render, \
             contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            cli.main()
            cli.main()
        self.assertEqual(analysis.call_count, 1)
        self.assertEqual(smoking.call_count, 1)
        self.assertEqual(render.call_count, 2)

    def test_source_directory_and_disabled_smoking(self):
        cfg = {'paths': {'source': 'input'}, 'processing': {'mode': 'legacy'},
               'smoking': {'enabled': False}}
        Path('config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
        with patch.object(cli, 'analyze_video', return_value=([(0., 3.)], 3.)), \
             patch.object(cli, 'render_video') as render, \
             contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            cli.main()
        render.assert_called_once()

    def test_empty_and_missing_source(self):
        Path('input/sample.mp4').unlink()
        Path('config.yaml').write_text('{}', encoding='utf-8')
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            cli.main()
        Path('config.yaml').write_text('paths: {source: missing.mp4}', encoding='utf-8')
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaises(SystemExit) as result:
            cli.main()
        self.assertEqual(result.exception.code, 1)

    def test_missing_model_reports_failure(self):
        cfg = {'processing': {'mode': 'legacy'}, 'smoking': {'model': 'missing.onnx'}}
        Path('config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
        with patch.object(cli, 'analyze_video', return_value=([(0., 3.)], 3.)), \
             contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaises(SystemExit) as result:
            cli.main()
        self.assertEqual(result.exception.code, 1)

    def test_missing_configuration_reports_failure(self):
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaises(SystemExit) as result:
            cli.main()
        self.assertEqual(result.exception.code, 1)
