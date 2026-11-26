import logging
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from clipweb import cli


class CacheAndDetectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        p = self.root / 'config.yaml'; p.write_text('{}', encoding='utf8')
        self.cfg = cli.load_config(p)
        self.source = self.root / 'video.mp4'; self.source.write_bytes(b'video')
        self.logger = logging.getLogger('cache-test')

    def tearDown(self): self.tmp.cleanup()

    def test_analysis_cache_matches_signature_and_ignores_damage(self):
        path = cli._analysis_cache_path(self.root, self.source)
        signature = cli._analysis_signature(self.source, self.cfg)
        self.assertIsNone(cli._load_analysis_cache(path, signature))
        cli._save_analysis_cache(path, signature, [(1., 2.)], 3.)
        self.assertEqual(cli._load_analysis_cache(path, signature), ([(1., 2.)], 3.))
        self.assertIsNone(cli._load_analysis_cache(path, 'different'))
        self.cfg['webcam']['x'] = 42
        self.assertNotEqual(cli._analysis_signature(self.source, self.cfg), signature)
        path.write_text('{bad', encoding='utf8')
        self.assertIsNone(cli._load_analysis_cache(path, signature))

    def test_smoking_signature_changes_when_model_or_settings_change(self):
        model = self.root / 'model'; model.write_bytes(b'model')
        old = cli._smoking_signature('base', self.cfg, model)
        self.cfg['smoking']['confidence'] = .8
        self.assertNotEqual(old, cli._smoking_signature('base', self.cfg, model))

    def test_intervals_merge_and_remove_multiple_smoking_events(self):
        merged = cli.merge_intervals([(0., 1.), (1.2, 3.), (5., 7.)], .25)
        self.assertEqual(merged, [(0., 3.), (5., 7.)])
        self.assertEqual(cli.subtract_intervals(merged, [(0.5, 1.5), (2., 6.)]),
                         [(0., .5), (1.5, 2.), (6., 7.)])

    def test_cpu_model_runs_and_directml_missing_is_explicit_or_falls_back(self):
        model = Path(__file__).resolve().parents[1] / 'models' / 'smoking_detector.onnx'
        settings = dict(self.cfg['smoking'], backend='cpu')
        predict = cli._smoking_predictor(model, settings, self.logger)
        result = predict(np.zeros((1, 3, 640, 640), dtype=np.float32))
        self.assertEqual(result.shape, (1, 5, 8400))
        runtime = types.SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'])
        with patch.dict('sys.modules', {'onnxruntime': runtime}):
            settings['backend'] = 'directml'
            with self.assertRaisesRegex(RuntimeError, 'DirectML initialization failed'):
                cli._smoking_predictor(model, settings, self.logger)
            settings['backend'] = 'auto'
            fallback = cli._smoking_predictor(model, settings, self.logger)
            self.assertEqual(fallback(np.zeros((1, 3, 640, 640), dtype=np.float32)).shape, result.shape)

    def test_config_rejects_invalid_modes_and_ratios(self):
        for text in ('face: {enabled: wrong}', 'analysis: {speech_ratio: 0}',
                     'speech: {method: bad}',
                     'output: {min_segment_seconds: -1}', 'output: {split_episodes: maybe}',
                     'face: {missing_tolerance_seconds: -1}'):
            path = self.root / 'bad.yaml'; path.write_text(text, encoding='utf8')
            with self.subTest(text=text), self.assertRaises(ValueError): cli.load_config(path)

    def test_directml_session_is_configured_and_used(self):
        class Options: pass
        class Session:
            def disable_fallback(self): self.disabled = True
            def get_providers(self): return ['DmlExecutionProvider', 'CPUExecutionProvider']
            def get_inputs(self): return [types.SimpleNamespace(name='input')]
            def run(self, names, inputs): return [inputs['input'] + 1]
        session = Session()
        runtime = types.SimpleNamespace(get_available_providers=lambda: ['DmlExecutionProvider'],
                                        SessionOptions=Options,
                                        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=1),
                                        InferenceSession=lambda *a, **kw: session)
        with patch.dict('sys.modules', {'onnxruntime': runtime}):
            settings = dict(self.cfg['smoking'], backend='directml', device_id=0)
            predictor = cli._smoking_predictor(Path('model'), settings, self.logger)
            self.assertEqual(float(predictor(np.zeros(1))[0]), 1.)
            self.assertTrue(session.disabled)

    def test_invalid_smoking_settings_and_empty_work(self):
        settings = dict(self.cfg['smoking'], backend='bad')
        with self.assertRaises(ValueError):
            cli._smoking_predictor(Path('model'), settings, self.logger)
        self.cfg['smoking']['sample_fps'] = 0
        with self.assertRaises(ValueError):
            cli.detect_smoking(self.source, [], self.cfg, Path('model'), self.logger)
        self.cfg['smoking']['sample_fps'] = 2
        self.assertEqual(cli.detect_smoking(self.source, [], self.cfg, Path('model'), self.logger), [])

    def test_missing_face_detector_and_empty_mouth_are_safe(self):
        with patch.object(cli.cv2, 'CascadeClassifier', return_value=types.SimpleNamespace(empty=lambda: True)):
            with self.assertRaisesRegex(RuntimeError, 'face detector'):
                cli._face_detector()
        gray = np.zeros((10, 10), dtype=np.uint8)
        self.assertEqual(cli._mouth_patch(gray, (30, 30, 10, 10)).shape, (24, 48))

    def test_render_empty_intervals_does_not_create_output(self):
        target = self.root / 'unused.mp4'
        cli.render_video(self.source, target, [], self.cfg, self.logger)
        self.assertFalse(target.exists())
