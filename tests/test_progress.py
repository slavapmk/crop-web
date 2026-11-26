import threading
import unittest
from unittest.mock import patch

from clipweb.cli import FileProgress


class FakeBar:
    def __init__(self, *args, **kwargs):
        self.n = 0.0

    def update(self, amount):
        self.n += amount

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
