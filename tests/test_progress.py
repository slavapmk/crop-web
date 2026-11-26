import threading
import unittest
from unittest.mock import patch
from tqdm import tqdm

from clipweb.cli import FileProgress


class FakeBar:
    format_interval = staticmethod(tqdm.format_interval)

    def __init__(self, *args, **kwargs):
        self.n = 0.0
        self.total = kwargs.get("total", 1)
        self.format_dict = {"elapsed": 1.0}

    def update(self, amount):
        self.n += amount

    def set_postfix_str(self, *args, **kwargs):
        pass

    def close(self):
        pass


class ProgressTests(unittest.TestCase):
    def test_concurrent_stages_and_repeated_updates(self):
        overall = FakeBar()
        with patch('clipweb.cli.tqdm', FakeBar):
            progress = FileProgress(overall, ('audio', 'video', 'smoking', 'render'))
            threads = [threading.Thread(target=progress.update, args=(stage, 1))
                       for stage in ('audio', 'video')]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            progress.update('audio', 0.5)
            self.assertAlmostEqual(overall.n, 0.5)
            progress.update('smoking', 1)
            progress.update('render', 2)
            self.assertAlmostEqual(overall.n, 1)
            self.assertAlmostEqual(progress.bar.n, 100)
            progress.close()

    def test_fractional_updates_never_exceed_total(self):
        overall = FakeBar()
        with patch('clipweb.cli.tqdm', FakeBar):
            progress = FileProgress(overall, ('audio', 'video', 'smoking', 'render'))
            for stage in progress.stages:
                for i in range(1, 302):
                    progress.update(stage, i / 301)
                    self.assertLessEqual(progress.bar.n, progress.bar.total)
                    self.assertLessEqual(overall.n, overall.total)
            self.assertAlmostEqual(progress.bar.n, 100)
            self.assertAlmostEqual(overall.n, 1)
