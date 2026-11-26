import logging
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import av
import cv2
import numpy as np

from clipweb.cli import filter_short_intervals, load_config, render_video


class FragmentTests(unittest.TestCase):
    def test_minimum_length_keeps_boundary_and_never_joins_gaps(self):
        items = [(0., .2), (1., 1.5), (3., 4.)]
        self.assertEqual(filter_short_intervals(items, .5), items[1:])
        self.assertEqual(filter_short_intervals(items, 0), items)


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required')
class RenderIntegrationTests(unittest.TestCase):
    def test_non_keyframe_segments_preserve_selected_video_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'source.mp4', root / 'output.mp4'
            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'lavfi', '-i', 'testsrc2=size=128x96:rate=30:duration=6',
                '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=6',
                '-c:v', 'libx264', '-g', '120', '-bf', '2', '-crf', '10',
                '-c:a', 'aac', '-shortest', str(source),
            ], check=True, capture_output=True)
            config_file = root / 'config.yaml'
            config_file.write_text('{}', encoding='utf8')
            cfg = load_config(config_file)
            cfg['webcam'] = dict(x=0, y=0, width=128, height=96)
            intervals = [(0.6, 1.4), (2.2, 3.2), (4.1, 5.1)]
            render_video(source, output, intervals, cfg, logging.getLogger('test'))
            with av.open(str(source)) as src, av.open(str(output)) as result:
                self.assertTrue(result.streams.audio)
                self.assertAlmostEqual(result.duration / av.time_base, 2.8, delta=.1)
                def read_at(container, timestamp):
                    stream = container.streams.video[0]
                    container.seek(int(timestamp / float(stream.time_base)), stream=stream, backward=True)
                    return next(frame.to_ndarray(format='bgr24') for frame in container.decode(video=0)
                                if frame.pts is not None and float(frame.pts * frame.time_base) + 1e-6 >= timestamp)
                position = 0.
                for start, end in intervals:
                    middle = (end - start) / 2
                    original = read_at(src, start + middle)
                    rendered = read_at(result, position + middle)
                    self.assertLess(float(np.mean(cv2.absdiff(original, rendered))), 7.)
                    position += end - start
