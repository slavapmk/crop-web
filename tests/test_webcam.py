import unittest
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from clipweb.webcam import detect_webcam, infer_rectangle


class Detector:
    def __init__(self, faces):
        self.faces = faces
    def detectMultiScale(self, *args, **kwargs):
        return self.faces
    def empty(self): return False


class WebcamTests(unittest.TestCase):
    def test_finds_single_fixed_rectangle_in_top_right(self):
        frames = []
        for shade in (50, 75, 100, 125):
            image = np.zeros((720, 1280, 3), np.uint8)
            image[:168, 1016:] = shade
            frames.append(image)
        box = infer_rectangle(frames, Detector([(1134, 40, 50, 50)]))
        self.assertLessEqual(abs(box['x'] - 1016), 4)
        self.assertEqual(box['y'], 0)
        self.assertLessEqual(abs(box['height'] - 168), 4)
        self.assertEqual(box['x'] + box['width'], 1280)

    def test_no_corner_face_is_an_error(self):
        frames = [np.zeros((720, 1280, 3), np.uint8) for _ in range(3)]
        with self.assertRaisesRegex(ValueError, 'No recurring webcam'):
            infer_rectangle(frames, Detector([(580, 320, 60, 60)]))

    def test_finds_fixed_rectangle_in_bottom_right(self):
        frames = []
        for shade in (50, 75, 100, 125):
            image = np.zeros((720, 1280, 3), np.uint8)
            image[540:, 974:] = shade
            frames.append(image)
        box = infer_rectangle(frames, Detector([(1114, 566, 50, 50)]))
        self.assertLessEqual(abs(box['x'] - 974), 4)
        self.assertLessEqual(abs(box['y'] - 540), 4)
        self.assertEqual(box['x'] + box['width'], 1280)
        self.assertEqual(box['y'] + box['height'], 720)

    def test_no_frames_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'No frames'):
            infer_rectangle([], Detector([]))

    def test_too_few_confirmations_is_an_error(self):
        frames = [np.zeros((720, 1280, 3), np.uint8)]
        with self.assertRaisesRegex(ValueError, 'uncertain'):
            infer_rectangle(frames, Detector([(1134, 40, 50, 50)]))

    def test_unmarked_boundary_is_not_guessed(self):
        frames = [np.zeros((720, 1280, 3), np.uint8) for _ in range(4)]
        with self.assertRaisesRegex(ValueError, 'boundary is unclear'):
            infer_rectangle(frames, Detector([(1134, 40, 50, 50)]))

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required')
    def test_samples_recording_and_detects_corner_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'sample.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                            '-f', 'lavfi', '-i', 'color=black:size=1280x720:rate=5:duration=3',
                            '-vf', 'drawbox=x=1016:y=0:w=264:h=168:color=white:t=fill',
                            '-c:v', 'libx264', str(source)], check=True, capture_output=True)
            with patch('clipweb.webcam.cv2.CascadeClassifier', return_value=Detector([(1134, 40, 50, 50)])):
                box = detect_webcam(source, 4, logging.getLogger('webcam-test'))
            self.assertLessEqual(abs(box['x'] - 1016), 4)
            self.assertLessEqual(abs(box['height'] - 168), 4)
            with self.assertRaises(ValueError):
                detect_webcam(source, 2, logging.getLogger('webcam-test'))
