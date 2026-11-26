import logging
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
from clipweb.episodes import index_intervals, render_episodes
from clipweb.cli import load_config, render_video


class FakeBar:
    def __init__(self, **kwargs): self.n = 0
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, n): self.n += n


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required')
class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
                        '-i', 'testsrc2=size=128x96:rate=10:duration=2', '-c:v', 'libx264',
                        '-g', '20', '-bf', '2', str(self.source)], check=True, capture_output=True)
        self.folder = self.root / 'output' / 'source'
        self.cache = self.root / 'cache' / 'episodes.json'

    def tearDown(self): self.tmp.cleanup()

    def test_exact_zero_based_inclusive_frame_numbers(self):
        result = index_intervals(self.source, [(.2, .7), (1., 1.6)])
        self.assertEqual([(s, e) for _, _, s, e in result], [(2, 6), (10, 15)])

    def test_variable_rate_uses_decoded_frame_indices(self):
        variable = self.root / 'variable.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(self.source),
                        '-vf', r'select=not(eq(mod(n\,3)\,0))', '-fps_mode', 'vfr',
                        '-c:v', 'libx264', str(variable)], check=True, capture_output=True)
        with av.open(str(variable)) as container:
            timestamps = [float(f.pts * f.time_base) for f in container.decode(video=0)]
        self.assertTrue(any(abs((b-a) - (c-b)) > .01 for a, b, c in zip(timestamps, timestamps[1:], timestamps[2:])))
        result = index_intervals(variable, [(timestamps[2], timestamps[8])])
        self.assertEqual(result[0][2:], (2, 7))

    def test_real_render_creates_individual_numbered_clips(self):
        config = self.root / 'config.yaml'
        config.write_text('{}', encoding='utf8')
        cfg = load_config(config)
        cfg['webcam'] = dict(x=0, y=0, width=128, height=96)
        render_episodes(self.source, self.folder, self.cache, [(.2, .7), (1., 1.6)], cfg,
                        logging.getLogger('test'), render_video, None, FakeBar)
        self.assertEqual(sorted(p.name for p in self.folder.glob('*.mp4')), ['10-15.mp4', '2-6.mp4'])
        for filename, count in [('2-6.mp4', 5), ('10-15.mp4', 6)]:
            with av.open(str(self.folder / filename)) as container:
                self.assertEqual(sum(1 for _ in container.decode(video=0)), count)

    def test_outputs_are_idempotent_and_unmanaged_files_are_safe(self):
        def render(source, target, intervals, cfg, logger, show_progress):
            target.write_bytes(b'video')
        args = (self.source, self.folder, self.cache, [(.2, .7), (1., 1.6)], {},
                logging.getLogger('test'), render, None, FakeBar)
        render_episodes(*args)
        self.assertEqual(sorted(p.name for p in self.folder.glob('*.mp4')), ['10-15.mp4', '2-6.mp4'])
        render_episodes(*args)
        unrelated = self.folder / 'notes.txt'
        unrelated.write_text('keep')
        args = (self.source, self.folder, self.cache, [(.2, .7)], {},
                logging.getLogger('test'), render, None, FakeBar)
        render_episodes(*args)
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ['2-6.mp4', 'notes.txt'])
        (self.folder / '10-15.mp4').write_bytes(b'user')
        with self.assertRaises(FileExistsError):
            render_episodes(self.source, self.folder, self.cache, [(1., 1.6)], {},
                            logging.getLogger('test'), render, None, FakeBar)
        self.assertEqual((self.folder / '10-15.mp4').read_bytes(), b'user')

    def test_interrupted_render_can_be_retried(self):
        def fail(*args, **kwargs): raise RuntimeError('failure')
        args = (self.source, self.folder, self.cache, [(.2, .7)], {}, logging.getLogger('test'))
        with self.assertRaises(RuntimeError):
            render_episodes(*args, fail, None, FakeBar)
        def render(source, target, intervals, cfg, logger, show_progress): target.write_bytes(b'video')
        render_episodes(*args, render, None, FakeBar)
        self.assertEqual((self.folder / '2-6.mp4').read_bytes(), b'video')
