# clipweb

[Русская версия](README.ru.md)

`clipweb` extracts webcam episodes from stream recordings. The default configuration identifies one small webcam near a corner, fixes that rectangle for the entire recording, removes periods without a detected face, blurs the webcam during probable smoking, and writes every retained episode to its own MP4. Speaking and other audio do not affect selection by default.

## Requirements and start

Install Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and FFmpeg with `ffmpeg` on `PATH`. On Windows, DirectML can accelerate the bundled smoking model when a compatible GPU and driver are available.

```text
uv sync
uv run clipweb
```

Place supported videos in `input/`. The program creates `input/`, `output/`, and `logs/` as needed. The default `config.yaml` is tracked in the repository and remains after `git clean -ffdx`; the program does not recreate it if it is manually deleted. `git clean -ffdx` removes ignored files, including recordings, results, caches, logs and `.venv`. Alternatively set `paths.source` in `config.yaml` to an absolute or config-relative video file or directory. All settings are in `config.yaml`; run the command from the project directory. Supported input extensions are MP4, MKV, MOV, AVI, WebM and M4V.

## Independent processing modules

With `webcam.auto: true`, the program samples `webcam.samples` points across the recording and looks for a recurrent face in a small corner region. It estimates one enclosing webcam rectangle and never moves or resizes it. The detected rectangle is logged as `x`, `y`, `width`, and `height` in source-frame pixels. `aspect.enabled: true` and `aspect.ratio: "16:10"` then take the largest centered rectangle with that exact ratio inside the webcam, without scaling or stretching; the resulting fixed rectangle is also logged. Change `aspect.ratio` (for example `9:16`) or set `aspect.enabled: false` to keep the detected dimensions. The same final rectangle is used for face and smoking detection and output. If no stable position is found, processing fails rather than silently selecting an arbitrary region. To set a known position manually, set `webcam.auto: false` and fill in those four coordinates.

The face detector samples at `analysis.sample_fps` (5 frames per second by default), so cuts are estimated on that sampling grid rather than at individual-frame precision. When `face.enabled: true`, intervals with no detected face are removed; `face.missing_tolerance_seconds` bridges short missed detections. Disable `face.enabled` to retain them. Longer disappearances and webcam moves can still cause detection errors. The CPU/OpenCV face detector is heuristic: head rotation, occlusion, small faces, or another face appearing in the fixed region can cause false results. Camera position and size changes are not followed; verify the rectangle and sample output when layouts differ.

The smoking detector is enabled by default. It checks the fixed webcam area within intervals retained by the enabled face and speech filters, then handles probable smoking events including the configured padding. By default `smoking.blur: true` preserves those moments and blurs the entire output webcam frame; `smoking.blur_sigma` controls blur strength. Set `smoking.blur: false` to cut the detected events instead. The audio is unaffected unless `mute.enabled` is set. Blurring applies only to events the detector finds and does not guarantee all smoking is hidden. It is experimental and can mistake hands or objects for smoking or miss cigarettes. `smoking.confidence`, `min_hits`, `max_gap_seconds`, padding, and sampling rate control this trade-off. `smoking.backend: auto` uses ONNX Runtime DirectML where available and otherwise OpenCV CPU; `directml` requires DirectML and `cpu` forces CPU. `device_id` is the DXGI adapter index and may change after hardware changes. Some DirectML model operations can run on CPU. The bundled model attribution and license are in `models/`.

With `output.split_episodes: true`, results are stored in `output/<source-stem>/`. Each MP4 is named `<first>-<last>.mp4` using **zero-based inclusive indices of decoded video frames in the input file**. The program counts actual frames and timestamps rather than estimating from nominal frame rate. Source intervals are half-open, so the named last frame is the last frame before the interval end. Audio from each retained interval is included when available unless `mute.enabled: true` removes the audio track from exports. Encoding uses the settings in `output`.

`output.min_segment_seconds` discards shorter fragments after smoking handling; the default is `0`, which keeps every non-empty interval. Short face-detector gaps may be bridged only within the configured tolerance; in cut mode smoking removals are never rejoined by this step.

Analysis results are cached in `output/.clipweb_cache/`. Source metadata, selected rectangle and detection settings participate in cache signatures. Changing blur mode or strength reuses the smoking detection cache; a rerun renders new output files. Set `processing.reanalyze: true` to force fresh analysis. The episode manifest tracks generated files: reruns replace managed episodes, remove obsolete managed episodes after successful rendering, and refuse to overwrite an unrelated file with the same frame-range name. Files not tracked by that manifest are preserved. Do not delete the cache if you need automatic replacement of an existing episode set; if it is lost, move the previous result folder before rerunning.

## Combining modules

Each switch is independent: `webcam.enabled` enables webcam cropping (`false` uses the entire source frame), `webcam.auto` chooses automatic or manual camera coordinates when cropping is enabled; `face.enabled` removes missing-face intervals; `speech.enabled` applies the speech filter; `smoking.enabled` applies the smoking filter; `aspect.enabled` enables the fixed aspect crop; `mute.enabled` exports video without an audio track; and `output.split_episodes` chooses separate numbered files (`true`) or one concatenated `output/<source-stem>_clipweb.mp4` (`false`). For example, disable face filtering while leaving smoking enabled, or enable speech filtering while keeping separate files.

Muting affects exported files only: enabling speech filtering still analyzes the source audio. When speech filtering is enabled, `speech.method` is `both` (voice activity and mouth motion), `speech` (voice only), or `mouth` (mouth motion only). `speech.invert: true` keeps speech rather than removing it. Mouth-based methods still detect faces to find the mouth even if `face.enabled` is false, but missing faces are not removed unless the face filter is enabled. Automatic webcam detection also needs face detection to locate the camera. An enabled aspect crop still applies to the full source frame when webcam cropping is disabled; odd source dimensions are rounded down to even pixels for encoding. These heuristics do not identify the speaker: background voices, eating, head movement and an open mouth can be misclassified. `output.join_gap_seconds` joins retained intervals before smoking handling and can restore material removed by face or speech filters; it defaults to `0`.

## Progress and logs

The console shows overall, per-file and stage progress. Overall percentages give stages equal weights. `ETA ~` is a rough estimate based on measured rates: audio and video analysis are concurrent, while smoking detection and rendering are sequential. Before a later stage starts, its duration is assumed to match the slower analysis stage. Other files are initially assumed to take as long as the current one. Estimates may therefore change considerably. `source` indicates position in the original recording rather than wall-clock elapsed time.

`logging.console_level` and `logging.file_level` independently control console and file verbosity. Each run creates a timestamped UTF-8 log in `logs/` with fixed-width event identifiers. Logs and progress do not require external configuration files.

## Tests and limitations

Run `uv run python -m unittest discover -s tests`. To measure and enforce the 95% line-coverage minimum, run `uv run python -m coverage run --source=clipweb -m unittest discover -s tests` followed by `uv run python -m coverage report`. Tests requiring FFmpeg are skipped if it is unavailable. Detection and output correctness depend on the source, webcam geometry, codec and model; review representative episodes before processing important material. Multiple faces or large layout changes can confuse automatic detection. Timestamps without presentation timestamps or in non-monotonic order cannot be mapped reliably to frame numbers and are rejected instead of guessed. Output names are zero-based video-frame indices, not audio sample numbers.
