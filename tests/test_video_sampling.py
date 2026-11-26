import unittest
from fractions import Fraction
from types import SimpleNamespace
from clipweb.cli import _sample_video_frames


class FakeVideo:
    def __init__(self):
        self.start = 0
        self.seeks = 0

    def seek(self, offset, **kwargs):
        self.start = max(0, offset - 1)
        self.seeks += 1

    def decode(self, **kwargs):
        for pts in range(self.start, 13):
            yield SimpleNamespace(pts=pts, time_base=Fraction(1, 2))


class VideoSamplingTests(unittest.TestCase):
    def test_samples_match_seeks_and_handle_gaps_and_eof(self):
        stream = SimpleNamespace(time_base=Fraction(1, 2))
        times = [0.0, .1, .5, 1.0, 4.0, 4.1, 6.5, 7.0]
        video = FakeVideo()
        found = list(_sample_video_frames(video, stream, times))
        self.assertEqual([stamp for _, stamp in found], [0, .5, .5, 1, 4, 4.5, None, None])
        self.assertEqual(video.seeks, 3)

    def test_missing_timestamps_force_fresh_seek(self):
        video = FakeVideo()
        video.decode = lambda **kwargs: iter([SimpleNamespace(pts=None, time_base=1)])
        frames = list(_sample_video_frames(video, SimpleNamespace(time_base=1), [0, 1]))
        self.assertEqual([stamp for _, stamp in frames], [0, 1])
        self.assertEqual(video.seeks, 2)

