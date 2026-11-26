import unittest
from clipweb.aspect import crop_to_aspect


class AspectTests(unittest.TestCase):
    def test_center_crop_16_10_without_distortion(self):
        box = {'x': 1016, 'y': 0, 'width': 264, 'height': 168, 'auto': True}
        actual = crop_to_aspect(box, '16:10')
        self.assertEqual((actual['x'], actual['y'], actual['width'], actual['height']),
                         (1020, 4, 256, 160))
        self.assertEqual(actual['width'] * 10, actual['height'] * 16)
        self.assertTrue(actual['auto'])

    def test_tall_aspect_and_source(self):
        box = {'x': 4, 'y': 6, 'width': 300, 'height': 200}
        cropped = crop_to_aspect(box, '9:16')
        self.assertEqual(cropped['width'] * 16, cropped['height'] * 9)
        self.assertEqual(crop_to_aspect(box, 'source'), box)
        self.assertEqual(crop_to_aspect(box, None), box)

    def test_invalid_and_too_small(self):
        box = {'x': 0, 'y': 0, 'width': 10, 'height': 10}
        for value in ('abc', '0:10', '16:-1', '4:3:2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                crop_to_aspect(box, value)
        with self.assertRaises(ValueError):
            crop_to_aspect(box, '16:10')
