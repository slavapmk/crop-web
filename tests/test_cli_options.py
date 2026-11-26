import argparse
import contextlib
import io
import unittest
from pathlib import Path

import yaml

from clipweb.cli_options import OPTIONS, add_options, apply_overrides


class ConfigOptionsTests(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        add_options(self.parser)

    def test_every_example_setting_round_trips_through_its_flag(self):
        root = Path(__file__).resolve().parents[1]
        original = yaml.safe_load((root / "config.example.yaml").read_text(encoding="utf-8"))
        expected = {}

        def flatten(mapping, prefix=""):
            for key, value in mapping.items():
                name = prefix + key
                if isinstance(value, dict):
                    flatten(value, name + ".")
                else:
                    expected[name] = value

        flatten(original)
        self.assertEqual(set(expected), {key for key, _, _ in OPTIONS})
        args = []
        for key, kind, _ in OPTIONS:
            value = expected[key]
            flag = key.replace(".", "-").replace("_", "-")
            if kind == "boolean":
                args.append(("--" if value else "--no-") + flag)
            else:
                encoded = ("null" if value is None else yaml.safe_dump(
                    value, default_flow_style=True, width=10000,
                ).splitlines()[0] if isinstance(value, list) else str(value))
                args.append(f"--{flag}={encoded}")
        self.assertEqual(vars(self.parser.parse_args([])), {})
        settings = vars(self.parser.parse_args(args))
        self.assertEqual(apply_overrides({}, settings), original)

    def test_invalid_values_are_rejected(self):
        for key, kind, _ in OPTIONS:
            invalid = {"integer": "oops", "number": "nan", "YAML list": "{}",
                       "integer or auto": "oops"}.get(kind.removeprefix("optional "))
            if invalid is not None:
                flag = key.replace(".", "-").replace("_", "-")
                with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        self.parser.parse_args([f"--{flag}={invalid}"])
                    self.assertEqual(error.exception.code, 2)
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            apply_overrides([], {})
        with self.assertRaisesRegex(ValueError, "section"):
            apply_overrides({"section": 1}, {"section.key": 2})
