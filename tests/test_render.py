import logging
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import av
import cv2
import numpy as np

from clipweb.cli import filter_short_intervals, load_config, render_video
from clipweb.smoking_regions import map_regions
from clipweb.episodes import render_episodes
from clipweb.episodes import OutputPermissionError


class QuietBar:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, n): pass


class FragmentTests(unittest.TestCase):
    def test_minimum_length_keeps_boundary_and_never_joins_gaps(self):
        items = [(0., .2), (1., 1.5), (3., 4.)]
        self.assertEqual(filter_short_intervals(items, .5), items[1:])
        self.assertEqual(filter_short_intervals(items, 0), items)

    def test_smoking_regions_are_mapped_across_cuts(self):
        intervals = [(1., 2.), (4., 6.)]
        regions = [(1.2, 1.5, 10, 10, 20, 20), (4.1, 4.4, 30, 20, 12, 8),
                   (5.5, 7., 50, 40, 10, 10)]
        actual = map_regions(intervals, regions)
        expected = [(0.2, 0.5), (1.1, 1.4), (2.5, 3.)]
        self.assertEqual(len(actual), len(expected))
        for pair, reference in zip(actual, expected):
            self.assertAlmostEqual(pair[0], reference[0])
            self.assertAlmostEqual(pair[1], reference[1])



@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required')
class RenderIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows file sharing semantics required')
    def test_locked_output_is_classified_as_output_permission_error(self):
        import ctypes
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / 'source.mp4', root / 'locked.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'color=s=128x96:d=1', '-c:v', 'libx264', str(source)], check=True)
            target.touch()
            config = root / 'config.yaml'
            config.write_text('{}', encoding='utf8')
            cfg = load_config(config)
            cfg['webcam'] = dict(x=0, y=0, width=128, height=96)
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.CreateFileW.restype = ctypes.c_void_p
            handle = kernel.CreateFileW(str(target), 0x80000000, 0, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            try:
                with self.assertRaises(OutputPermissionError):
                    render_video(source, target, [(0., .5)], cfg, logging.getLogger('test'), show_progress=False)
            finally:
                kernel.CloseHandle(ctypes.c_void_p(handle))

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

    def test_ffmpeg_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
                            '-i', 'testsrc2=size=128x96:rate=10:duration=1', '-c:v', 'libx264',
                            str(source)], check=True, capture_output=True)
            config_file = root / 'config.yaml'
            config_file.write_text('{}', encoding='utf8')
            cfg = load_config(config_file)
            cfg['webcam'] = dict(x=0, y=0, width=128, height=96)
            cfg['output']['video_codec'] = 'nonexistent_codec'
            with self.assertRaisesRegex(RuntimeError, 'ffmpeg failed'):
                render_video(source, root / 'bad.mp4', [(0., .5)], cfg, logging.getLogger('test'))

    def test_unavailable_progress_timestamp_does_not_abort_render(self):
        class FakeProcess:
            def __init__(self):
                self.stdout = io.StringIO('out_time_ms=N/A\nout_time_ms=250000\nprogress=end\n')
            def wait(self): return 0
            def poll(self): return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
                            '-i', 'testsrc2=size=128x96:rate=10:duration=1', '-c:v', 'libx264',
                            str(source)], check=True, capture_output=True)
            config_file = root / 'config.yaml'
            config_file.write_text('{}', encoding='utf8')
            cfg = load_config(config_file)
            cfg['webcam'] = dict(x=0, y=0, width=128, height=96)
            process = FakeProcess()
            with patch('subprocess.Popen', return_value=process):
                render_video(source, root / 'out.mp4', [(0., .5)], cfg,
                             logging.getLogger('test'), show_progress=False)
            self.assertTrue(process.stdout.closed)

    def test_mute_removes_audio_from_joined_and_separate_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                            '-f', 'lavfi', '-i', 'testsrc2=size=128x96:rate=10:duration=3',
                            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=3',
                            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(source)],
                           check=True, capture_output=True)
            config_file = root / 'config.yaml'
            config_file.write_text('{}', encoding='utf8')
            cfg = load_config(config_file)
            cfg['webcam'] = dict(x=0, y=0, width=128, height=96)
            intervals = [(.2, 1.), (1.5, 2.5)]
            reference = root / 'with-audio.mp4'
            render_video(source, reference, intervals, cfg, logging.getLogger('test'), show_progress=False)
            cfg['mute']['enabled'] = True
            muted = root / 'muted.mp4'
            render_video(source, muted, intervals, cfg, logging.getLogger('test'), show_progress=False)
            folder = root / 'episodes'
            render_episodes(source, folder, root / 'cache' / 'episodes.json', intervals, cfg,
                            logging.getLogger('test'), render_video, None, QuietBar)
            with av.open(str(reference)) as audible, av.open(str(muted)) as silent:
                self.assertTrue(audible.streams.audio)
                self.assertFalse(silent.streams.audio)
                self.assertEqual(sum(1 for _ in audible.decode(video=0)),
                                 sum(1 for _ in silent.decode(video=0)))
            outputs = list(folder.glob('*.mp4'))
            self.assertEqual(len(outputs), 2)
            for path in outputs:
                with av.open(str(path)) as container:
                    self.assertFalse(container.streams.audio)
                    self.assertTrue(container.streams.video)

    def test_blur_is_timed_and_preserves_footage_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                            '-f', 'lavfi', '-i', 'testsrc2=size=128x96:rate=30:duration=6',
                            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=6',
                            '-c:v', 'libx264', '-g', '120', '-c:a', 'aac', '-shortest', str(source)],
                           check=True, capture_output=True)
            config_file = root / 'config.yaml'
            config_file.write_text('{}', encoding='utf8')
            cfg = load_config(config_file)
            cfg['webcam'] = dict(x=0, y=0, width=128, height=96)
            intervals = [(.6, 1.6), (3., 5.)]
            regions = [(.9, 1.3, 32, 24, 40, 30), (3.6, 4.1, 70, 40, 28, 24)]
            baseline, blurred = root / 'baseline.mp4', root / 'blurred.mp4'
            logger = logging.getLogger('test')
            render_video(source, baseline, intervals, cfg, logger, show_progress=False)
            render_video(source, blurred, intervals, cfg, logger, show_progress=False, blur_regions=regions)
            def image_at(container, timestamp):
                stream = container.streams.video[0]
                container.seek(int(timestamp / float(stream.time_base)), stream=stream, backward=True)
                return next(frame.to_ndarray(format='bgr24') for frame in container.decode(video=0)
                            if frame.pts is not None and float(frame.pts * frame.time_base) + 1e-6 >= timestamp)
            with av.open(str(baseline)) as original, av.open(str(blurred)) as result:
                self.assertTrue(result.streams.audio)
                self.assertAlmostEqual(result.duration / av.time_base, original.duration / av.time_base, delta=.05)
                for timestamp in (.1, .7, 1.25, 2.1, 2.4):
                    diff = np.mean(cv2.absdiff(image_at(original, timestamp), image_at(result, timestamp)))
                    self.assertLess(diff, 6., f'unexpected blur at {timestamp}s')
                for timestamp in (.3, .5, 1.6, 1.8):
                    diff = cv2.absdiff(image_at(original, timestamp), image_at(result, timestamp))
                    region = diff[24:54, 32:72] if timestamp < 1. else diff[40:64, 70:98]
                    self.assertGreater(np.mean(region), 8., f'blur missing at {timestamp}s')
                    self.assertLess(np.mean(diff[0:16, 0:20]), 7., f'outside region blurred at {timestamp}s')
            baseline_dir, blurred_dir = root / 'split-original', root / 'split-blurred'
            render_episodes(source, baseline_dir, root / 'cache' / 'original.json', intervals,
                            cfg, logger, render_video, None, QuietBar)
            render_episodes(source, blurred_dir, root / 'cache' / 'blurred.json', intervals,
                            cfg, logger, render_video, None, QuietBar, blur_regions=regions)
            for index, name in enumerate(('18-47.mp4', '90-149.mp4')):
                with av.open(str(baseline_dir / name)) as original, av.open(str(blurred_dir / name)) as result:
                    self.assertTrue(result.streams.audio)
                    outside, inside = (.1, .5) if index == 0 else (.2, .8)
                    self.assertLess(np.mean(cv2.absdiff(image_at(original, outside), image_at(result, outside))), 6.)
                    difference = cv2.absdiff(image_at(original, inside), image_at(result, inside))
                    area = difference[24:54, 32:72] if index == 0 else difference[40:64, 70:98]
                    self.assertGreater(np.mean(area), 8.)
            boundary = root / 'boundary.mp4'
            render_video(source, boundary, [intervals[0]], cfg, logger, show_progress=False,
                         blur_regions=[(0., 1.1, 32, 24, 40, 30)])
            with av.open(str(baseline)) as original, av.open(str(boundary)) as result:
                self.assertGreater(np.mean(cv2.absdiff(image_at(original, .1), image_at(result, .1))[24:54, 32:72]), 8.)
                self.assertLess(np.mean(cv2.absdiff(image_at(original, .8), image_at(result, .8))), 6.)

    def test_blur_command_file_handles_special_characters_in_temp_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command_dir = root / "temp's folder,part"
            command_dir.mkdir()
            source, output = root / 'source.mp4', root / 'blurred.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
                            '-i', 'testsrc2=size=128x96:rate=10:duration=2', '-c:v', 'libx264',
                            str(source)], check=True, capture_output=True)
            config_file = root / 'config.yaml'
            config_file.write_text('{}', encoding='utf8')
            cfg = load_config(config_file)
            cfg['webcam'] = dict(x=0, y=0, width=128, height=96)
            with patch('tempfile.tempdir', str(command_dir)):
                render_video(source, output, [(0., 2.)], cfg, logging.getLogger('test'),
                             show_progress=False, blur_regions=[(.5, 1.5, 32, 24, 40, 30)])
            with av.open(str(output)) as container:
                self.assertTrue(container.streams.video)
            self.assertFalse(list(command_dir.glob('cw_blur_*.txt')))
