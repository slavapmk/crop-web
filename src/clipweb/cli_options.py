"""Command-line overrides for the project configuration."""

import argparse
import math

import yaml


OPTIONS = (
    ('paths.input', 'text', 'Directory scanned in batch mode.'),
    ('paths.output', 'text', 'Destination directory for results and applicable caches.'),
    ('paths.source', 'optional text', 'Configured source file or directory. An explicit --source takes precedence.'),
    ('processing.reanalyze', 'boolean', 'Ignore saved analysis and recompute it for this job.'),
    ('webcam.enabled', 'boolean', 'Use a webcam crop. Disable to start from the full source frame.'),
    ('webcam.auto', 'boolean', 'Detect the webcam rectangle automatically. Disable to use the four manual coordinates.'),
    ('webcam.samples', 'integer', 'Number of sampled video frames used for automatic webcam detection.'),
    ('webcam.x', 'integer', 'Left edge of the manual webcam rectangle, in source pixels.'),
    ('webcam.y', 'integer', 'Top edge of the manual webcam rectangle, in source pixels.'),
    ('webcam.width', 'integer', 'Width of the manual webcam rectangle, in source pixels.'),
    ('webcam.height', 'integer', 'Height of the manual webcam rectangle, in source pixels.'),
    ('face.enabled', 'boolean', 'Filter out intervals without a detected face.'),
    ('face.missing_tolerance_seconds', 'number', 'Bridge brief missed detections between two confirmed faces.'),
    ('speech.enabled', 'boolean', 'Apply speech filtering independently of face filtering.'),
    ('speech.method', 'text', 'both: voice and mouth motion; speech: voice only; mouth: mouth motion only.'),
    ('speech.invert', 'boolean', 'true keeps detected speech; false removes it.'),
    ('aspect.enabled', 'boolean', 'Crop the selected rectangle to the requested ratio without stretching.'),
    ('aspect.ratio', 'text', 'Target aspect ratio as width:height, for example 16:10.'),
    ('mute.enabled', 'boolean', 'Remove audio from exports while preserving audio analysis.'),
    ('analysis.sample_fps', 'number', 'Video samples per second for face and mouth analysis.'),
    ('analysis.audio_sample_rate', 'integer', 'Audio resampling rate in Hz used by WebRTC VAD.'),
    ('analysis.audio_frame_ms', 'integer', 'Audio frame duration in milliseconds used by WebRTC VAD.'),
    ('analysis.vad_aggressiveness', 'integer', 'WebRTC voice activity mode from 0 to 3.'),
    ('analysis.lip_motion_threshold', 'number', 'Minimum change in the mouth patch used to count visual speech evidence.'),
    ('analysis.mouth_motion_ratio', 'number', 'Fraction of video samples with mouth motion required in a speech window.'),
    ('analysis.speech_ratio', 'number', 'Fraction of voiced audio frames required in a speech window.'),
    ('analysis.window_seconds', 'number', 'Duration in seconds of the speech classification window.'),
    ('analysis.speech_padding_seconds', 'number', 'Seconds added around detected speech intervals.'),
    ('smoking.enabled', 'boolean', 'Enable smoking object analysis and filtering.'),
    ('smoking.blur', 'boolean', 'true blurs detected object regions; false cuts smoking intervals.'),
    ('smoking.blur_sigma', 'number', 'Gaussian blur strength for detected regions, in pixels.'),
    ('smoking.blur_near_mouth', 'boolean', 'More conservative: only blur detections near a detected mouth.'),
    ('smoking.blur_margin_pixels', 'integer', 'Extra pixels included around a detected object in the blur region.'),
    ('smoking.blur_hold_seconds', 'number', 'Maximum time on either side of a confirmed detection.'),
    ('smoking.model', 'text', 'Path to the smoking detector ONNX model.'),
    ('smoking.backend', 'text', 'Inference backend selected for the smoking detector.'),
    ('smoking.device_id', 'integer', 'GPU index for DirectML inference.'),
    ('smoking.sample_fps', 'number', 'Smoking detection samples per second.'),
    ('smoking.input_size', 'integer', 'Detector input width and height in pixels.'),
    ('smoking.confidence', 'number', 'Minimum model confidence for a detection.'),
    ('smoking.min_hits', 'integer', 'Minimum number of detections required in an event.'),
    ('smoking.max_gap_seconds', 'number', 'Maximum time between hits in one smoking event.'),
    ('smoking.padding_before_seconds', 'number', 'Seconds added before a detected smoking event.'),
    ('smoking.padding_after_seconds', 'number', 'Seconds added after a detected smoking event.'),
    ('smoking.merge_gap_seconds', 'number', 'Merge smoking events that are separated by at most this many seconds.'),
    ('smoking.threads', 'integer', 'CPU inference thread count for the detector.'),
    ('output.split_episodes', 'boolean', 'true: separate numbered episodes; false: one concatenated video.'),
    ('output.video_codec', 'text', 'Video encoder used for rendered output.'),
    ('output.preset', 'text', 'Video encoder speed and compression preset.'),
    ('output.crf', 'integer', 'Constant rate factor used for video quality.'),
    ('output.audio_codec', 'text', 'Audio encoder used for rendered output.'),
    ('output.audio_bitrate', 'text', 'Target audio bitrate such as 160k.'),
    ('output.join_gap_seconds', 'number', 'Joining gaps can include previously filtered audio/video.'),
    ('output.min_segment_seconds', 'number', 'Discard episodes shorter than one minute; 0 keeps all non-empty episodes.'),
    ('logging.console_level', 'text', 'Minimum severity written to the console.'),
    ('logging.file_level', 'text', 'Minimum severity written to the run log.'),
)


def _parse_value(value, kind):
    if kind.startswith("optional "):
        if value.lower() == "null":
            return None
        kind = kind.removeprefix("optional ")
    if kind == "text":
        return value
    if kind == "integer or auto" and value == "auto":
        return value
    if kind in {"integer", "integer or auto"}:
        return int(value)
    if kind == "number":
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Expected a finite number")
        return number
    parsed = yaml.safe_load(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a YAML list")
    return parsed


def add_options(parser):
    """Register overrides without assigning defaults to omitted flags."""
    for key, kind, description in OPTIONS:
        option = "--" + key.replace(".", "-").replace("_", "-")
        aliases = ["--" + key.split(".")[-1]] if key in {"paths.input", "paths.output"} else []
        arguments = {"dest": key, "default": argparse.SUPPRESS, "help": description}
        if kind == "boolean":
            parser.add_argument(option, *aliases, action=argparse.BooleanOptionalAction, **arguments)
        else:
            def convert(value, value_kind=kind):
                try:
                    return _parse_value(value, value_kind)
                except (ValueError, yaml.YAMLError) as error:
                    raise argparse.ArgumentTypeError(f"Expected {value_kind}: {error}") from error

            parser.add_argument(option, *aliases, type=convert, metavar="VALUE", **arguments)


def apply_overrides(config, overrides):
    """Apply CLI values to a newly loaded YAML mapping before validation."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    for key, value in overrides.items():
        *sections, name = key.split(".")
        section = config
        for part in sections:
            section = section.setdefault(part, {})
            if not isinstance(section, dict):
                raise ValueError(f"Configuration section {part} must be a mapping")
        section[name] = value
    return config
