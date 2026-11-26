# clipweb

[Русская версия](README.ru.md)

`clipweb` extracts webcam episodes from stream recordings. The default configuration identifies one small webcam near a corner, fixes that rectangle for the entire recording, removes periods without a detected face, locally blurs probable smoking objects, and writes every retained episode to its own MP4. Speaking and other audio do not affect selection by default.

## Requirements and start

Install Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and FFmpeg with `ffmpeg` on `PATH`. On Windows, DirectML can accelerate the bundled smoking model when a compatible GPU and driver are available.

```text
uv sync
uv run clipweb
```

Place supported videos in `input/`. The program creates `input/`, `output/`, and `logs/` as needed. The default `config.yaml` is tracked in the repository and remains after `git clean -ffdx`; the program does not recreate it if it is manually deleted. `git clean -ffdx` removes ignored files, including recordings, results, caches, logs and `.venv`. Alternatively set `paths.source` in `config.yaml` to an absolute or config-relative video file or directory. All settings are in `config.yaml`; run the command from the project directory. Supported input extensions are MP4, MKV, MOV, AVI, WebM and M4V.

## Independent processing modules

With `webcam.auto: true`, the program samples `webcam.samples` points across the recording and looks for a recurrent face in a small corner region. It estimates one enclosing webcam rectangle and never moves or resizes it. The detected rectangle is logged as `x`, `y`, `width`, and `height` in source-frame pixels. `aspect.enabled: true` and `aspect.ratio: "16:10"` then take the largest centered rectangle with that exact ratio inside the webcam, without scaling or stretching; the resulting fixed rectangle is also logged. Change `aspect.ratio` (for example `9:16`) or set `aspect.enabled: false` to keep the detected dimensions. The same final rectangle is used for face and smoking detection and output. If no stable position is found, processing fails rather than silently selecting an arbitrary region. To set a known position manually, set `webcam.auto: false` and fill in those four coordinates.

The face detector samples at `analysis.sample_fps` (5 frames per second by default), so cuts are estimated on that sampling grid rather than at individual-frame precision. When `face.enabled: true`, intervals with no detected face are removed; `face.missing_tolerance_seconds` bridges gaps of up to 5 seconds by default when a face is detected on both sides; those seconds remain in the output even if the person was briefly absent. Disable `face.enabled` to retain them. Longer disappearances and webcam moves can still cause detection errors. The CPU/OpenCV face detector is heuristic: head rotation, occlusion, small faces, or another face appearing in the fixed region can cause false results. Camera position and size changes are not followed; verify the rectangle and sample output when layouts differ.

The smoking detector is enabled by default. It checks the fixed webcam area within intervals retained by the enabled face and speech filters, then handles probable smoking events including the configured padding. By default `smoking.blur: true` preserves those moments and blurs only the detected object region; `smoking.blur_sigma` controls blur strength. `smoking.blur_near_mouth: true` rejects candidates far from a detected face's mouth. `smoking.blur_margin_pixels` expands each box, and `smoking.blur_hold_seconds` limits how long a detection is used around a sampled frame. Set `smoking.blur: false` to cut the detected events instead. The audio is unaffected unless `mute.enabled` is set. Blurring applies only near localized detections and does not guarantee all smoking is hidden: gaps between detections, a missing face, or a cigarette held away from the mouth can leave it visible. An event with no accepted box is left unblurred and reported in the log. It is experimental and can mistake hands or objects for smoking or miss cigarettes. `smoking.confidence`, `min_hits`, `max_gap_seconds`, padding, and sampling rate control this trade-off. `smoking.backend: auto` uses ONNX Runtime DirectML where available and otherwise OpenCV CPU; `directml` requires DirectML and `cpu` forces CPU. `device_id` is the DXGI adapter index and may change after hardware changes. Some DirectML model operations can run on CPU. The bundled model attribution and license are in `models/`.

With `output.split_episodes: true`, results are stored in `output/<source-stem>/`. Each MP4 is named `<first>-<last>.mp4` using **zero-based inclusive indices of decoded video frames in the input file**. The program counts actual frames and timestamps rather than estimating from nominal frame rate. Source intervals are half-open, so the named last frame is the last frame before the interval end. Audio from each retained interval is included when available unless `mute.enabled: true` removes the audio track from exports. Encoding uses the settings in `output`.

`output.min_segment_seconds` discards shorter fragments after smoking handling; the default is `60`, so intervals shorter than one minute are not rendered. Set it to `0` to keep every non-empty interval. Short face-detector gaps may be bridged only within the configured tolerance; in cut mode smoking removals are never rejoined by this step.

Analysis results are cached in `output/.clipweb_cache/`. Source metadata, selected rectangle and detection settings participate in cache signatures. With speech filtering disabled, changing the face tolerance can reuse a matching zero-tolerance face analysis without decoding the stream again. Smoking analysis still runs again because earlier smoking caches do not cover the newly retained gaps. Changing blur mode or strength reuses the smoking detection cache; changing the near-mouth check refreshes the separate object-position cache. Older event-only caches are reused, with an additional detection pass limited to retained smoking intervals to locate objects; a rerun renders new output files. Set `processing.reanalyze: true` to force fresh analysis. The episode manifest tracks generated files: reruns replace managed episodes, remove obsolete managed episodes after successful rendering, and refuse to overwrite an unrelated file with the same frame-range name. Files not tracked by that manifest are preserved. Do not delete the cache if you need automatic replacement of an existing episode set; if it is lost, move the previous result folder before rerunning.

## Combining modules

Each switch is independent: `webcam.enabled` enables webcam cropping (`false` uses the entire source frame), `webcam.auto` chooses automatic or manual camera coordinates when cropping is enabled; `face.enabled` removes missing-face intervals; `speech.enabled` applies the speech filter; `smoking.enabled` applies the smoking filter; `aspect.enabled` enables the fixed aspect crop; `mute.enabled` exports video without an audio track; and `output.split_episodes` chooses separate numbered files (`true`) or one concatenated `output/<source-stem>_clipweb.mp4` (`false`). For example, disable face filtering while leaving smoking enabled, or enable speech filtering while keeping separate files.

Muting affects exported files only: enabling speech filtering still analyzes the source audio. When speech filtering is enabled, `speech.method` is `both` (voice activity and mouth motion), `speech` (voice only), or `mouth` (mouth motion only). `speech.invert: true` keeps speech rather than removing it. Mouth-based methods still detect faces to find the mouth even if `face.enabled` is false, but missing faces are not removed unless the face filter is enabled. Automatic webcam detection also needs face detection to locate the camera. An enabled aspect crop still applies to the full source frame when webcam cropping is disabled; odd source dimensions are rounded down to even pixels for encoding. These heuristics do not identify the speaker: background voices, eating, head movement and an open mouth can be misclassified. `output.join_gap_seconds` joins retained intervals before smoking handling and can restore material removed by face or speech filters; it defaults to `0`.

## Progress and logs

The console shows overall, per-file and stage progress. Overall percentages give stages equal weights. `ETA ~` is a rough estimate based on measured rates: audio and video analysis are concurrent, while smoking detection and rendering are sequential. Before a later stage starts, its duration is assumed to match the slower analysis stage. Other files are initially assumed to take as long as the current one. Estimates may therefore change considerably. `source` indicates position in the original recording rather than wall-clock elapsed time.

`logging.console_level` and `logging.file_level` independently control console and file verbosity. Each run creates a timestamped UTF-8 log in `logs/` with fixed-width event identifiers. Logs and progress do not require external configuration files.

## Tests and limitations

Run `uv run python -m unittest discover -s tests`. To measure and enforce the 95% line-coverage minimum, run `uv run python -m coverage run --source=clipweb -m unittest discover -s tests` followed by `uv run python -m coverage report`. Tests requiring FFmpeg are skipped if it is unavailable. Detection and output correctness depend on the source, webcam geometry, codec and model; review representative episodes before processing important material. Multiple faces or large layout changes can confuse automatic detection. Timestamps without presentation timestamps or in non-monotonic order cannot be mapped reliably to frame numbers and are rejected instead of guessed. Output names are zero-based video-frame indices, not audio sample numbers.

If Windows temporarily denies FFmpeg access to an episode temporary file, the program retries up to three times using fresh temporary files. Persistent access failures are logged and the file is not reported as successfully rendered.

## Run one job from a workflow

From the project directory:

```powershell
uv run clipweb --config D:/jobs/job-42.yaml --source D:/media/movie.mp4
```

Use [config.example.yaml](config.example.yaml) as the job configuration template. Set every processing
option and the output directory in that YAML. Relative paths inside the YAML
are resolved from its directory.
The two CLI paths are resolved from the process working directory.
Omit `--source` for the regular configured input selection. A successful run
exits with code 0. Check the expected artifact before passing it to the next
stage. The output layout is `<stem>_clipweb.mp4 or an episode directory`.

Pass the command as an argument list to `subprocess.run` with
`cwd` set to this project directory and `check=True`. Independent jobs can run
in separate processes. Use distinct output directories for jobs whose paths
could collide, and wait for each input's prior stage to finish before starting
its next stage.

## CLI arguments

Example from the project directory

```powershell
uv run clipweb --source recording.mp4 --output out/job42 --no-webcam-auto --webcam-x 976 --webcam-y 540 --webcam-width 304 --webcam-height 180 --no-output-split-episodes
```

`--config PATH` selects a YAML file (default `config.yaml`). `--source PATH` selects one video. `--help` lists options. The flags below override only the corresponding YAML values, before validation. Omitted flags keep the YAML values. The values in the last column come from the example configuration.

Boolean flags accept the positive and `--no-…` forms without a value. Pass a YAML list as one argument, for example `--paths-extensions '[".mp4", ".mkv"]'` where that flag is available. Use the literal `null` for optional values that should be cleared. For list items beginning with a dash, use the equals form, such as `--export-extra-args='["-movflags", "+faststart"]'` in film-reframe.

`--input` and `--output` are aliases for `--paths-input` and `--paths-output`.

| Flag | YAML key | Value type | Meaning | Example |
|---|---|---|---|---|
| `--paths-input` | `paths.input` | text | Directory scanned in batch mode. | `input` |
| `--paths-output` | `paths.output` | text | Destination directory for results and applicable caches. | `output` |
| `--paths-source` | `paths.source` | optional text | Configured source file or directory. An explicit --source takes precedence. | `null` |
| `--processing-reanalyze` / `--no-processing-reanalyze` | `processing.reanalyze` | boolean | Ignore saved analysis and recompute it for this job. | `false` |
| `--webcam-enabled` / `--no-webcam-enabled` | `webcam.enabled` | boolean | Use a webcam crop. Disable to start from the full source frame. | `true` |
| `--webcam-auto` / `--no-webcam-auto` | `webcam.auto` | boolean | Detect the webcam rectangle automatically. Disable to use the four manual coordinates. | `true` |
| `--webcam-samples` | `webcam.samples` | integer | Number of sampled video frames used for automatic webcam detection. | `24` |
| `--webcam-x` | `webcam.x` | integer | Left edge of the manual webcam rectangle, in source pixels. | `976` |
| `--webcam-y` | `webcam.y` | integer | Top edge of the manual webcam rectangle, in source pixels. | `540` |
| `--webcam-width` | `webcam.width` | integer | Width of the manual webcam rectangle, in source pixels. | `304` |
| `--webcam-height` | `webcam.height` | integer | Height of the manual webcam rectangle, in source pixels. | `180` |
| `--face-enabled` / `--no-face-enabled` | `face.enabled` | boolean | Filter out intervals without a detected face. | `true` |
| `--face-missing-tolerance-seconds` | `face.missing_tolerance_seconds` | number | Bridge brief missed detections between two confirmed faces. | `5.0` |
| `--speech-enabled` / `--no-speech-enabled` | `speech.enabled` | boolean | Apply speech filtering independently of face filtering. | `false` |
| `--speech-method` | `speech.method` | text | both: voice and mouth motion; speech: voice only; mouth: mouth motion only. | `both` |
| `--speech-invert` / `--no-speech-invert` | `speech.invert` | boolean | true keeps detected speech; false removes it. | `false` |
| `--aspect-enabled` / `--no-aspect-enabled` | `aspect.enabled` | boolean | Crop the selected rectangle to the requested ratio without stretching. | `true` |
| `--aspect-ratio` | `aspect.ratio` | text | Target aspect ratio as width:height, for example 16:10. | `'16:10'` |
| `--mute-enabled` / `--no-mute-enabled` | `mute.enabled` | boolean | Remove audio from exports while preserving audio analysis. | `false` |
| `--analysis-sample-fps` | `analysis.sample_fps` | number | Video samples per second for face and mouth analysis. | `5` |
| `--analysis-audio-sample-rate` | `analysis.audio_sample_rate` | integer | Audio resampling rate in Hz used by WebRTC VAD. | `16000` |
| `--analysis-audio-frame-ms` | `analysis.audio_frame_ms` | integer | Audio frame duration in milliseconds used by WebRTC VAD. | `30` |
| `--analysis-vad-aggressiveness` | `analysis.vad_aggressiveness` | integer | WebRTC voice activity mode from 0 to 3. | `2` |
| `--analysis-lip-motion-threshold` | `analysis.lip_motion_threshold` | number | Minimum change in the mouth patch used to count visual speech evidence. | `2.8` |
| `--analysis-mouth-motion-ratio` | `analysis.mouth_motion_ratio` | number | Fraction of video samples with mouth motion required in a speech window. | `0.5` |
| `--analysis-speech-ratio` | `analysis.speech_ratio` | number | Fraction of voiced audio frames required in a speech window. | `0.2` |
| `--analysis-window-seconds` | `analysis.window_seconds` | number | Duration in seconds of the speech classification window. | `0.6` |
| `--analysis-speech-padding-seconds` | `analysis.speech_padding_seconds` | number | Seconds added around detected speech intervals. | `0.35` |
| `--smoking-enabled` / `--no-smoking-enabled` | `smoking.enabled` | boolean | Enable smoking object analysis and filtering. | `true` |
| `--smoking-blur` / `--no-smoking-blur` | `smoking.blur` | boolean | true blurs detected object regions; false cuts smoking intervals. | `true` |
| `--smoking-blur-sigma` | `smoking.blur_sigma` | number | Gaussian blur strength for detected regions, in pixels. | `15.0` |
| `--smoking-blur-near-mouth` / `--no-smoking-blur-near-mouth` | `smoking.blur_near_mouth` | boolean | More conservative: only blur detections near a detected mouth. | `true` |
| `--smoking-blur-margin-pixels` | `smoking.blur_margin_pixels` | integer | Extra pixels included around a detected object in the blur region. | `6` |
| `--smoking-blur-hold-seconds` | `smoking.blur_hold_seconds` | number | Maximum time on either side of a confirmed detection. | `0.35` |
| `--smoking-model` | `smoking.model` | text | Path to the smoking detector ONNX model. | `models/smoking_detector.onnx` |
| `--smoking-backend` | `smoking.backend` | text | Inference backend selected for the smoking detector. | `auto` |
| `--smoking-device-id` | `smoking.device_id` | integer | GPU index for DirectML inference. | `0` |
| `--smoking-sample-fps` | `smoking.sample_fps` | number | Smoking detection samples per second. | `2.0` |
| `--smoking-input-size` | `smoking.input_size` | integer | Detector input width and height in pixels. | `640` |
| `--smoking-confidence` | `smoking.confidence` | number | Minimum model confidence for a detection. | `0.35` |
| `--smoking-min-hits` | `smoking.min_hits` | integer | Minimum number of detections required in an event. | `2` |
| `--smoking-max-gap-seconds` | `smoking.max_gap_seconds` | number | Maximum time between hits in one smoking event. | `2.5` |
| `--smoking-padding-before-seconds` | `smoking.padding_before_seconds` | number | Seconds added before a detected smoking event. | `1.0` |
| `--smoking-padding-after-seconds` | `smoking.padding_after_seconds` | number | Seconds added after a detected smoking event. | `2.0` |
| `--smoking-merge-gap-seconds` | `smoking.merge_gap_seconds` | number | Merge smoking events that are separated by at most this many seconds. | `4.0` |
| `--smoking-threads` | `smoking.threads` | integer | CPU inference thread count for the detector. | `2` |
| `--output-split-episodes` / `--no-output-split-episodes` | `output.split_episodes` | boolean | true: separate numbered episodes; false: one concatenated video. | `true` |
| `--output-video-codec` | `output.video_codec` | text | Video encoder used for rendered output. | `libx264` |
| `--output-preset` | `output.preset` | text | Video encoder speed and compression preset. | `veryfast` |
| `--output-crf` | `output.crf` | integer | Constant rate factor used for video quality. | `23` |
| `--output-audio-codec` | `output.audio_codec` | text | Audio encoder used for rendered output. | `aac` |
| `--output-audio-bitrate` | `output.audio_bitrate` | text | Target audio bitrate such as 160k. | `160k` |
| `--output-join-gap-seconds` | `output.join_gap_seconds` | number | Joining gaps can include previously filtered audio/video. | `0.0` |
| `--output-min-segment-seconds` | `output.min_segment_seconds` | number | Discard episodes shorter than one minute; 0 keeps all non-empty episodes. | `60.0` |
| `--logging-console-level` | `logging.console_level` | text | Minimum severity written to the console. | `INFO` |
| `--logging-file-level` | `logging.file_level` | text | Minimum severity written to the run log. | `DEBUG` |
