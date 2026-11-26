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

    def run_modules(self, overrides):
        data = {'webcam': {'auto': True, 'samples': 8, 'x': 0, 'y': 0, 'width': 264, 'height': 168},
                'smoking': {'model': 'model.onnx', 'enabled': True}}
        data.update(overrides)
        Path('config.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')
        with patch.object(cli, 'detect_webcam', return_value={'x': 0, 'y': 0, 'width': 264, 'height': 168}) as detect, \
             patch.object(cli, 'analyze_video', return_value=([(0., 3.), (4., 8.)], 10.)) as analysis, \
             patch.object(cli, 'detect_smoking', return_value=[(1., 2.)]) as smoking, \
             patch.object(cli, 'render_episodes') as episodes, \
             patch.object(cli, 'render_video') as joined, \
             contextlib.redirect_stderr(io.StringIO()):
            cli.main()
        return detect, analysis, smoking, episodes, joined

    def test_default_modules_are_independent(self):
        detect, analysis, smoking, episodes, joined = self.run_modules({})
        detect.assert_called_once()
        used_cfg = analysis.call_args.args[1]
        self.assertTrue(used_cfg['face']['enabled'])
        self.assertFalse(used_cfg['speech']['enabled'])
        self.assertEqual((used_cfg['webcam']['width'], used_cfg['webcam']['height']), (256, 160))
        smoking.assert_called_once()
        episodes.assert_called_once()
        self.assertEqual(episodes.call_args.args[3], [(0., 3.), (4., 8.)])
        self.assertEqual(episodes.call_args.kwargs['blur_intervals'], [(1., 2.)])
        joined.assert_not_called()

    def test_smoking_cut_mode_removes_events_instead_of_blurring(self):
        overrides = {'smoking': {'model': 'model.onnx', 'enabled': True, 'blur': False}}
        _, _, _, episodes, _ = self.run_modules(overrides)
        self.assertEqual(episodes.call_args.args[3], [(0., 1.), (2., 3.), (4., 8.)])
        self.assertEqual(episodes.call_args.kwargs['blur_intervals'], [])

    def test_blur_works_with_concatenated_output(self):
        overrides = {'smoking': {'model': 'model.onnx', 'enabled': True, 'blur': True},
                     'output': {'split_episodes': False}}
        _, _, _, episodes, joined = self.run_modules(overrides)
        episodes.assert_not_called()
        self.assertEqual(joined.call_args.args[2], [(0., 3.), (4., 8.)])
        self.assertEqual(joined.call_args.kwargs['blur_intervals'], [(1., 2.)])

    def test_modules_can_be_switched_independently(self):
        overrides = {'webcam': {'auto': False, 'x': 0, 'y': 0, 'width': 264, 'height': 168},
                     'face': {'enabled': False}, 'speech': {'enabled': True, 'method': 'speech'},
                     'aspect': {'enabled': False}, 'smoking': {'enabled': False},
                     'output': {'split_episodes': False}}
        detect, analysis, smoking, episodes, joined = self.run_modules(overrides)
        detect.assert_not_called()
        used_cfg = analysis.call_args.args[1]
        self.assertFalse(used_cfg['face']['enabled'])
        self.assertTrue(used_cfg['speech']['enabled'])
        self.assertEqual(used_cfg['speech']['method'], 'speech')
        self.assertEqual((used_cfg['webcam']['width'], used_cfg['webcam']['height']), (264, 168))
        smoking.assert_not_called()
        episodes.assert_not_called()
        joined.assert_called_once()
        self.assertEqual(joined.call_args.args[2], [(0., 3.), (4., 8.)])

    def test_speech_and_smoking_can_run_with_separate_episodes_without_face_filter(self):
        overrides = {'face': {'enabled': False}, 'speech': {'enabled': True, 'method': 'both'},
                     'smoking': {'enabled': True, 'model': 'model.onnx'},
                     'output': {'split_episodes': True}}
        _, analysis, smoking, episodes, joined = self.run_modules(overrides)
        used_cfg = analysis.call_args.args[1]
        self.assertFalse(used_cfg['face']['enabled'])
        self.assertTrue(used_cfg['speech']['enabled'])
        smoking.assert_called_once()
        episodes.assert_called_once()
        joined.assert_not_called()

    def test_webcam_crop_can_be_disabled_independently(self):
        overrides = {'webcam': {'enabled': False, 'auto': True},
                     'aspect': {'enabled': False}, 'smoking': {'enabled': False}}
        with patch.object(cli, '_full_frame_box', return_value={'x': 0, 'y': 0, 'width': 1280, 'height': 720}) as full:
            detect, analysis, _, episodes, _ = self.run_modules(overrides)
        full.assert_called_once()
        detect.assert_not_called()
        box = analysis.call_args.args[1]['webcam']
        self.assertEqual((box['width'], box['height']), (1280, 720))
        episodes.assert_called_once()

    def test_cached_analysis_and_smoking_are_reused(self):
        cfg = {'webcam': {'auto': False}, 'output': {'split_episodes': False}, 'smoking': {'model': 'model.onnx'}}
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

    def test_existing_relative_model_cache_is_migrated(self):
        data = {'webcam': {'auto': False}, 'aspect': {'enabled': False},
                'output': {'split_episodes': False}, 'smoking': {'model': 'model.onnx'}}
        Path('config.yaml').write_text(yaml.safe_dump(data), encoding='utf8')
        cfg = cli.load_config(Path('config.yaml'))
        source = Path('input/sample.mp4').resolve()
        model = Path('model.onnx').resolve()
        out = Path('output').resolve()
        signature = cli._analysis_signature(source, cfg)
        cli._save_analysis_cache(cli._analysis_cache_path(out, source), signature, [(0., 3.)], 3.)
        old = cli._smoking_signature(signature, cfg, model, normalize_model=False)
        current = cli._smoking_signature(signature, cfg, model)
        self.assertNotEqual(old, current)
        cache = cli._smoking_cache_path(out, source)
        cli._save_analysis_cache(cache, old, [(1., 2.)], 3.)
        with patch.object(cli, 'analyze_video', side_effect=AssertionError('analysis should be cached')), \
             patch.object(cli, 'detect_smoking', side_effect=AssertionError('smoking should be cached')), \
             patch.object(cli, 'render_video') as render, \
             contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            cli.main()
        self.assertEqual(render.call_args.kwargs['blur_intervals'], [(1., 2.)])
        self.assertEqual(cli._load_analysis_cache(cache, current), ([(1., 2.)], 3.))

    def test_source_directory_and_disabled_smoking(self):
        cfg = {'paths': {'source': 'input'}, 'webcam': {'auto': False}, 'output': {'split_episodes': False},
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

    def test_empty_project_initializes_directories_idempotently(self):
        Path('input/sample.mp4').unlink()
        Path('input').rmdir()
        config = Path('config.yaml')
        config.write_text('{}', encoding='utf-8')
        original = config.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            cli.main()
            cli.main()
        self.assertEqual(config.read_bytes(), original)
        for name in ('input', 'output', 'logs'):
            self.assertTrue(Path(name).is_dir())
        self.assertEqual(len(list(Path('logs').glob('*.log'))), 2)

    def test_missing_model_reports_failure(self):
        cfg = {'webcam': {'auto': False}, 'output': {'split_episodes': False}, 'smoking': {'model': 'missing.onnx'}}
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
