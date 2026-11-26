import io
import re
import unittest
from tqdm import tqdm
from clipweb.cli import SUMMARY_BAR_FORMAT, StageBar


class ProgressFormattingTests(unittest.TestCase):
    def test_summary_hides_fractional_counts_and_eta(self):
        output = io.StringIO()
        with tqdm(total=1, desc='Overall', file=output, bar_format=SUMMARY_BAR_FORMAT,
                  mininterval=0, ascii=True) as bar:
            bar.update(0.15513757311486356)
        line = output.getvalue().split('\r')[-2]
        self.assertRegex(line, r'Overall: +16%\|[^|]{24}\| \[\d\d:\d\d\]')
        self.assertNotIn('0.155', line)
        self.assertNotIn('<', line)

    def test_stages_have_same_width_and_integer_counts(self):
        lines = []
        for label in ('Audio', 'Video', 'Smoking', 'Render'):
            output = io.StringIO()
            with StageBar(total=100, desc=label, file=output, mininterval=0, ascii=True) as bar:
                bar.update(15.5137573)
            line = output.getvalue().split('\r')[-2]
            self.assertRegex(line, r'^.{7}: +16%\|[^|]{24}\| 16/100 ')
            lines.append(line.index('|'))
        self.assertEqual(len(set(lines)), 1)

    def test_stage_caps_progress_at_total(self):
        output = io.StringIO()
        with StageBar(total=10, desc='Render', file=output, mininterval=0, ascii=True) as bar:
            bar.update(20)
            self.assertEqual(bar.n, 10)
