import unittest

from clipweb.cli import _sample_index


class SamplingTests(unittest.TestCase):
    def test_fixed_grid_has_no_drift_for_common_frame_rates(self):
        for fps in (24, 25, 30, 60):
            with self.subTest(fps=fps):
                step = 0.2
                target = 0.0
                indices = []
                for frame in range(fps * 60):
                    timestamp = frame / fps
                    if timestamp + 1e-6 < target:
                        continue
                    index = _sample_index(timestamp, step, 300)
                    indices.append(index)
                    target = (index + 1) * step
                self.assertEqual(indices, list(range(300)))


if __name__ == '__main__':
    unittest.main()
