# clipweb

[Русская версия](README.ru.md)

`clipweb` extracts webcam episodes from stream recordings. The default mode identifies one small webcam near a corner, fixes that rectangle for the entire recording, removes periods without a detected face and probable smoking, and writes every retained episode to its own MP4. Speaking and other audio do not affect selection in this mode.

## Requirements and start

Install Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and FFmpeg with `ffmpeg` on `PATH`. On Windows, DirectML can accelerate the bundled smoking model when a compatible GPU and driver are available.

```text
uv sync
uv run clipweb
```

Place supported videos in `input/`. The program creates `input/`, `output/`, and `logs/` as needed. Alternatively set `paths.source` in `config.yaml` to an absolute or config-relative video file or directory. All settings are in `config.yaml`; run the command from the project directory. Supported input extensions are MP4, MKV, MOV, AVI, WebM and M4V.

## Default episode mode

`processing.mode: episodes` is the default. The program samples `webcam.samples` points across the recording and looks for a recurrent face in a small corner region. It estimates one enclosing webcam rectangle and never moves or resizes it. The detected rectangle is logged as `x`, `y`, `width`, and `height` in source-frame pixels. `output.aspect_ratio: "16:10"` then takes the largest centered rectangle with that exact ratio inside the webcam, without scaling or stretching; the resulting fixed rectangle is also logged. Change the ratio (for example `9:16`) or set it to `source` to keep the detected dimensions. The same final rectangle is used for face and smoking detection and output. If no stable position is found, processing fails rather than silently selecting an arbitrary region. To set a known position manually, set `webcam.auto: false` and fill in those four coordinates.

The face detector samples at `analysis.sample_fps` (5 frames per second by default), so cuts are estimated on that sampling grid rather than at individual-frame precision. Intervals with no detected face are removed; `processing.face_missing_tolerance_seconds` bridges short missed detections. Longer disappearances and webcam moves can still cause detection errors. The CPU/OpenCV face detector is heuristic: head rotation, occlusion, small faces, or another face appearing in the fixed region can cause false results. Camera position and size changes are not followed; verify the rectangle and sample output when layouts differ.

The smoking detector is enabled by default. It checks the fixed webcam area only in face-present intervals, then removes probable smoking events including the configured padding. It is experimental and can mistake hands or objects for smoking or miss cigarettes. `smoking.confidence`, `min_hits`, `max_gap_seconds`, padding, and sampling rate control this trade-off. `smoking.backend: auto` uses ONNX Runtime DirectML where available and otherwise OpenCV CPU; `directml` requires DirectML and `cpu` forces CPU. `device_id` is the DXGI adapter index and may change after hardware changes. Some DirectML model operations can run on CPU. The bundled model attribution and license are in `models/`.

Results are stored in `output/<source-stem>/`. Each MP4 is named `<first>-<last>.mp4` using **zero-based inclusive indices of decoded video frames in the input file**. The program counts actual frames and timestamps rather than estimating from nominal frame rate. Source intervals are half-open, so the named last frame is the last frame before the interval end. Audio from each retained interval is included when available. Encoding uses the settings in `output`.

`processing.min_segment_seconds` discards shorter fragments after smoking removal; the default is `0`, which keeps every non-empty episode. For legacy mode the corresponding setting is `output.min_segment_seconds` (default `0.5`). Short face-detector gaps may be bridged only within the configured tolerance; smoking removals are never rejoined by this step.

Analysis results are cached in `output/.clipweb_cache/`. Source metadata, selected rectangle and relevant settings participate in cache signatures. Set `processing.reanalyze: true` to force fresh analysis. The episode manifest tracks generated files: reruns replace managed episodes, remove obsolete managed episodes after successful rendering, and refuse to overwrite an unrelated file with the same frame-range name. Files not tracked by that manifest are preserved. Do not delete the cache if you need automatic replacement of an existing episode set; if it is lost, move the previous result folder before rerunning.

## Legacy mode

Set `processing.mode: legacy` to retain the prior single-file workflow. It uses the configured `webcam.x/y/width/height` without automatic detection or the episode aspect crop, applies `analysis.speech_detection` and `analysis.invert_speech`, joins nearby selected intervals using `output.join_gap_seconds`, removes detected smoking if enabled, and writes `output/<source-stem>_clipweb.mp4`. `both` combines voice activity and mouth movement, `speech` uses voice activity, `mouth` uses mouth movement and `face` ignores speech. These are heuristics, not speaker identification: speech from a video, eating, head movement, and an open mouth can be misclassified.

## Progress and logs

The console shows overall, per-file and stage progress. Overall percentages give stages equal weights. `ETA ~` is a rough estimate based on measured rates: audio and video analysis are concurrent, while smoking detection and rendering are sequential. Before a later stage starts, its duration is assumed to match the slower analysis stage. Other files are initially assumed to take as long as the current one. Estimates may therefore change considerably. `source` indicates position in the original recording rather than wall-clock elapsed time.

`logging.console_level` and `logging.file_level` independently control console and file verbosity. Each run creates a timestamped UTF-8 log in `logs/` with fixed-width event identifiers. Logs and progress do not require external configuration files.

## Tests and limitations

Run `uv run python -m unittest discover -s tests`. To measure and enforce the 95% line-coverage minimum, run `uv run python -m coverage run --source=clipweb -m unittest discover -s tests` followed by `uv run python -m coverage report`. Tests requiring FFmpeg are skipped if it is unavailable. Detection and output correctness depend on the source, webcam geometry, codec and model; review representative episodes before processing important material. Multiple faces or large layout changes can confuse automatic detection. Timestamps without presentation timestamps or in non-monotonic order cannot be mapped reliably to frame numbers and are rejected instead of guessed. Output names are zero-based video-frame indices, not audio sample numbers.
