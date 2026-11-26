import unittest
import numpy as np
from clipweb.smoking_regions import build_blur_regions, map_regions, select_smoking_box


def prediction(cx, cy, width, height, confidence=.78):
    output = np.zeros((1, 5, 10), dtype=np.float32)
    output[0, :, 0] = (cx, cy, width, height, confidence)
    return output


class SmokingRegionTests(unittest.TestCase):
    def test_high_score_far_from_mouth_is_rejected(self):
        # Matches the geometry of the reported false positive: the model can
        # score a non-cigarette highly, even when it is far from the mouth.
        pred = prediction(206. * 2.5, 77.5 * 2.5 + 120, 26. * 2.5, 12. * 2.5)
        face = [(129, 31, 47, 47)]
        self.assertIsNone(select_smoking_box(pred, 256, 160, 2.5, 0, 120, .35, face, True))
        self.assertIsNotNone(select_smoking_box(pred, 256, 160, 2.5, 0, 120, .35, face, False))

    def test_near_mouth_box_is_unletterboxed(self):
        pred = prediction(155. * 2.5, 70. * 2.5 + 120, 12. * 2.5, 8. * 2.5)
        box = select_smoking_box(pred, 256, 160, 2.5, 0, 120, .35, [(129, 31, 47, 47)], True)
        self.assertEqual(box, (149., 66., 161., 74.))

    def test_large_or_invalid_boxes_are_rejected(self):
        pred = prediction(320, 320, 500, 400)
        self.assertIsNone(select_smoking_box(pred, 256, 160, 2.5, 0, 120, .35, [], False))
        self.assertIsNone(select_smoking_box(np.zeros((2, 5)), 256, 160, 2.5, 0, 120, .35, [], False))
        self.assertIsNone(select_smoking_box(np.zeros((1, 4, 4)), 256, 160, 2.5, 0, 120, .35, [], False))
        self.assertIsNone(select_smoking_box(prediction(float('nan'), 300, 20, 20), 256, 160, 2.5, 0, 120, .35, [], False))
        self.assertIsNone(select_smoking_box(prediction(300, 300, 0, 20), 256, 160, 2.5, 0, 120, .35, [], False))

    def test_adjacent_equal_regions_are_merged(self):
        hits = [(1.0, 10., 10., 20., 20.), (1.5, 10., 10., 20., 20.)]
        regions, missing = build_blur_regions([(0., 3.)], hits, 100, 100, 0, .5)
        self.assertEqual(regions, [(.5, 2., 10, 10, 10, 10)])
        self.assertEqual(missing, 0)

    def test_regions_are_local_and_have_bounded_time(self):
        regions, missing = build_blur_regions([(1., 3.), (5., 6.)],
            [(1.5, 100., 70., 110., 80.), (2., 104., 70., 114., 80.)], 256, 160, 6, .35)
        self.assertEqual(missing, 1)
        self.assertEqual(len(regions), 2)
        self.assertAlmostEqual(regions[0][0], 1.15)
        self.assertAlmostEqual(regions[0][1], 1.75)
        self.assertAlmostEqual(regions[1][0], 1.75)
        self.assertAlmostEqual(regions[1][1], 2.35)
        self.assertEqual(regions[0][2:], (94, 64, 22, 22))
        self.assertTrue(all(r[4] < 256 and r[5] < 160 for r in regions))

    def test_region_mapping_clips_to_output_intervals(self):
        source = [(1., 2.), (4., 6.)]
        regions = [(.5, 1.5, 10, 10, 20, 20), (4.5, 5., 30, 30, 10, 10)]
        self.assertEqual(map_regions(source, regions), [(0., .5, 10, 10, 20, 20),
                                                        (1.5, 2., 30, 30, 10, 10)])
