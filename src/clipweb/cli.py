from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import webrtcvad

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("clipweb")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    defaults = {
        "webcam": {"x": 976, "y": 540, "width": 304, "height": 180},
        "paths": {"input": "input", "output": "output", "source": None},
        "analysis": {"sample_fps": 5, "audio_sample_rate": 16000, "audio_frame_ms": 30,
                     "vad_aggressiveness": 2, "lip_motion_threshold": 2.8, "speech_ratio": .2,
                     "mouth_motion_ratio": .5,
                     "window_seconds": .6, "speech_padding_seconds": .35,
                     "face_missing_tolerance_seconds": 2,
                     "speech_detection": "both", "invert_speech": False},
        "smoking": {"enabled": True, "model": "models/smoking_detector.onnx",
                    "sample_fps": 2.0, "input_size": 640, "confidence": .35,
                    "min_hits": 2, "max_gap_seconds": 2.5,
                    "padding_before_seconds": 1.0, "padding_after_seconds": 2.0,
                    "merge_gap_seconds": 4.0, "threads": 2, "backend": "auto", "device_id": 0},
        "output": {"video_codec": "libx264", "preset": "veryfast", "crf": 23,
                   "audio_codec": "aac", "audio_bitrate": "160k", "join_gap_seconds": .25},
    }
    for section, values in defaults.items():
        config.setdefault(section, {})
        for key, value in values.items():
            config[section].setdefault(key, value)
    if config["analysis"]["speech_detection"] not in {"face", "speech", "both", "mouth"}:
        raise ValueError("analysis.speech_detection must be face, speech, both, or mouth")
    for name in ("speech_ratio", "mouth_motion_ratio"):
        if not 0 < float(config["analysis"][name]) <= 1:
            raise ValueError(f"analysis.{name} must be greater than zero and at most one")
    return config


def _face_detector() -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("OpenCV face detector could not be loaded")
    return cascade


def _mouth_patch(gray: np.ndarray, face: tuple[int, int, int, int]) -> np.ndarray:
    """Normalize the detected face position and return only the expected mouth area."""
    x, y, w, h = face
    # Restrict motion measurement to the central lower face, excluding shoulders/background.
    patch = gray[y + int(h * .62):y + int(h * .94), x + int(w * .18):x + int(w * .82)]
    if patch.size == 0:
        return np.zeros((24, 48), dtype=np.uint8)
    return cv2.resize(patch, (48, 24), interpolation=cv2.INTER_AREA)


def _mouth_motion(current: np.ndarray, previous: np.ndarray | None) -> float:
    if previous is None:
        return 0.0
    return float(np.mean(cv2.absdiff(current, previous)))


def _sample_index(timestamp: float, step: float, frame_count: int) -> int:
    return min(frame_count - 1, max(0, int((timestamp + 1e-6) / step)))


def _speech_evidence(speech_counts: np.ndarray, audio_counts: np.ndarray,
                     mouth_grid: np.ndarray, cfg: dict, step: float) -> tuple[np.ndarray, np.ndarray]:
    radius = max(1, int(round(float(cfg["window_seconds"]) / step / 2)))
    window = np.ones(2 * radius + 1)
    voiced = np.convolve(np.pad(speech_counts, (radius, radius)), window, mode="valid")
    audio_total = np.convolve(np.pad(audio_counts, (radius, radius)), window, mode="valid")
    vad_presence = (audio_total > 0) & (voiced >= float(cfg["speech_ratio"]) * audio_total)
    motion = mouth_grid >= float(cfg["lip_motion_threshold"])
    count = np.convolve(np.pad(motion.astype(float), (radius, radius)), window, mode="valid")
    moving = count >= max(1, math.ceil(float(cfg["mouth_motion_ratio"]) * len(window)))
    return vad_presence, moving


class FileProgress:
    """Track equally weighted stages, including concurrent analysis tasks."""

    def __init__(self, overall: tqdm, stages: tuple[str, ...]):
        self.overall = overall
        self.stages = {name: 0.0 for name in stages}
        self.lock = threading.RLock()
        self.value = 0.0
        self.bar = tqdm(total=100, desc="File", unit="%", position=1, leave=False, dynamic_ncols=True)

    def update(self, stage: str, fraction: float) -> None:
        with self.lock:
            self.stages[stage] = max(self.stages[stage], min(1.0, max(0.0, fraction)))
            value = sum(self.stages.values()) / len(self.stages)
            delta = value - self.value
            self.value = value
            self.bar.update(delta * 100)
            self.overall.update(delta)

    def close(self) -> None:
        self.bar.close()


class StageBar(tqdm):
    def __init__(self, *args, progress: FileProgress | None = None, stage: str = "", **kwargs):
        self.progress = progress
        self.stage = stage
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        result = super().update(n)
        if self.progress is not None and self.total:
            self.progress.update(self.stage, self.n / self.total)
        return result


def analyze_video(path: Path, cfg: dict, logger: logging.Logger, progress: FileProgress | None = None) -> tuple[list[tuple[float, float]], float]:
    a = cfg["analysis"]
    box = cfg["webcam"]
    container = av.open(str(path))
    video = container.streams.video[0]
    duration = float(container.duration / av.time_base) if container.duration else float(video.duration * video.time_base)
    if duration <= 0:
        raise ValueError(f"Cannot determine duration: {path}")
    has_audio = bool(container.streams.audio)
    container.close()
    vad = webrtcvad.Vad(int(a["vad_aggressiveness"]))
    detector = _face_detector()
    step = 1.0 / float(a["sample_fps"])
    frame_count = max(1, math.ceil(duration / step))
    speech_counts = np.zeros(frame_count, dtype=np.int32)
    audio_counts = np.zeros(frame_count, dtype=np.int32)
    face_grid = np.zeros(frame_count, dtype=bool)
    mouth_grid = np.zeros(frame_count, dtype=float)
    previous = None
    logger.info("Анализ: %s (%.1f мин., шаг %.2f с)", path.name, duration / 60, step)
    # Each decoder owns its container; PyAV containers are not shared across threads.
    stop_audio = threading.Event()
    def analyze_audio() -> None:
        if has_audio and a["speech_detection"] in {"speech", "both"}:
            with av.open(str(path)) as container:
                audio_stream = container.streams.audio[0]
                audio_stream.thread_type = "AUTO"
                resampler = av.AudioResampler(format="s16", layout="mono", rate=int(a["audio_sample_rate"]))
                sample_rate = int(a["audio_sample_rate"])
                frame_bytes = int(sample_rate * int(a["audio_frame_ms"]) / 1000) * 2
                pending = b""
                pending_start: float | None = None
                audio_total = 0
                with StageBar(total=max(1, int(duration)), desc="Audio", unit="s", position=2, leave=False, dynamic_ncols=True, progress=progress, stage="audio") as bar:
                    last_audio_second = 0
                    for af in container.decode(audio=0):
                        if stop_audio.is_set():
                            return
                        for out in resampler.resample(af):
                            data = out.to_ndarray().tobytes()
                            start = float(out.pts * out.time_base) if out.pts is not None else audio_total / sample_rate
                            audio_total += len(data) // 2
                            origin = pending_start if pending else start
                            data = pending + data
                            complete_bytes = len(data) // frame_bytes * frame_bytes
                            for offset in range(0, complete_bytes, frame_bytes):
                                stamp = origin + offset / 2 / sample_rate
                                try:
                                    spoken = vad.is_speech(data[offset:offset + frame_bytes], sample_rate)
                                except Exception:
                                    spoken = False
                                idx = min(frame_count - 1, max(0, int(stamp / step)))
                                audio_counts[idx] += 1
                                if spoken:
                                    speech_counts[idx] += 1
                            pending = data[complete_bytes:]
                            pending_start = origin + complete_bytes / 2 / sample_rate if pending else None
                            current_second = min(int(duration), max(0, int(start)))
                            bar.update(max(0, current_second - last_audio_second))
                            last_audio_second = current_second
                            bar.set_postfix(elapsed=f"{current_second/60:.1f}m", remaining=f"{max(0, duration-current_second)/60:.1f}m")
                    for out in resampler.resample(None):
                        data = pending + out.to_ndarray().tobytes()
                        complete_bytes = len(data) // frame_bytes * frame_bytes
                        origin = pending_start or 0.0
                        for offset in range(0, complete_bytes, frame_bytes):
                            idx = min(frame_count - 1, max(0, int((origin + offset / 2 / sample_rate) / step)))
                            audio_counts[idx] += 1
                            if vad.is_speech(data[offset:offset + frame_bytes], sample_rate):
                                speech_counts[idx] += 1
                        pending = data[complete_bytes:]
                        pending_start = origin + complete_bytes / 2 / sample_rate if pending else None
                    bar.update(max(0, int(duration) - last_audio_second))
        elif not has_audio and a["speech_detection"] in {"speech", "both"}:
            logger.warning("Аудиодорожка не найдена; речь можно оценить только по движению лица")
        if progress is not None:
            progress.update("audio", 1.0)
    # Analyze audio and video concurrently using independent demuxers.
    with ThreadPoolExecutor(max_workers=1) as executor:
        audio_task = executor.submit(analyze_audio)
        try:
            with av.open(str(path)) as container:
                video = container.streams.video[0]
                with StageBar(total=frame_count, desc="Video", unit="frame", position=3, leave=False, dynamic_ncols=True, progress=progress, stage="video") as bar:
                    next_sample = 0.0
                    for frame in container.decode(video=0):
                        timestamp = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
                        if timestamp + 1e-6 < next_sample:
                            continue
                        idx = _sample_index(timestamp, step, frame_count)
                        image = frame.to_ndarray(format="bgr24")
                        x, y, w, h = (int(box[k]) for k in ("x", "y", "width", "height"))
                        crop = image[max(0, y):min(image.shape[0], y+h), max(0, x):min(image.shape[1], x+w)]
                        valid = crop.size > 0 and crop.shape[0] >= 30 and crop.shape[1] >= 30
                        if valid:
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            faces = detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(40, 40))
                            present = len(faces) > 0
                            face_grid[idx] = present
                            if present:
                                # Use the largest detected face; surrounding webcam movement cannot
                                # dominate the mouth-motion score.
                                face = max(faces, key=lambda item: int(item[2]) * int(item[3]))
                                mouth = _mouth_patch(gray, tuple(map(int, face)))
                                motion = _mouth_motion(mouth, previous)
                                mouth_grid[idx] = motion
                                previous = mouth
                            else:
                                previous = None
                        # Anchor sampling to the fixed grid, not to the last decoded frame.
                        # Otherwise frame-rate rounding accumulates and leaves false face gaps.
                        next_sample = (idx + 1) * step
                        bar.update(1)
                        bar.set_postfix(elapsed=f"{timestamp/60:.1f}m", remaining=f"{max(0,duration-timestamp)/60:.1f}m")
                        if timestamp >= duration:
                            break
            audio_task.result()
        finally:
            stop_audio.set()
    if progress is not None:
        progress.update("video", 1.0)
    # Speech evidence requires voice activity coincident with mouth motion, reducing game/other voices.
    vad_presence, moving = _speech_evidence(speech_counts, audio_counts, mouth_grid, a, step)
    speech = vad_presence.copy()
    if a["speech_detection"] == "both":
        speech &= moving
    elif a["speech_detection"] == "mouth":
        speech = moving.copy()
    speech &= face_grid
    mode = a["speech_detection"]
    if mode == "face":
        keep = face_grid
    else:
        keep = face_grid & (~speech if not a["invert_speech"] else speech)
    if mode == "face" and a["invert_speech"]:
        logger.warning("invert_speech действует только в режимах speech/both")
    # Remove a small guard interval around speech so word edges are not left audible.
    padding = max(0, int(math.ceil(float(a["speech_padding_seconds"]) / step)))
    if padding and mode != "face":
        speech = np.convolve(np.pad(speech.astype(float), (padding, padding)),
                             np.ones(2 * padding + 1), mode="valid") > 0
        keep = face_grid & (speech if a["invert_speech"] else ~speech)
    logger.info("Analysis coverage: face %.1f%%, VAD %.1f%%, mouth motion %.1f%%, speech cuts %.1f%%",
                100 * np.mean(face_grid), 100 * np.mean(vad_presence),
                100 * np.mean(moving), 100 * np.mean(speech))
    # Bridge short face detector misses, but retain speech decisions as classified.
    tolerance = max(0, int(float(a["face_missing_tolerance_seconds"]) / step))
    if tolerance:
        missing = ~face_grid
        starts = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)) == -1)
        for s, e in zip(starts, ends):
            if s > 0 and e < len(face_grid) and e - s <= tolerance:
                if mode != "face" and a["invert_speech"]:
                    # In voice-only mode never bridge a face gap with silence.
                    continue
                if mode != "face":
                    # Do not bridge over possible voice when lip evidence is unavailable.
                    evidence = moving if mode == "mouth" else vad_presence
                    keep[s:e] = (keep[s-1] or keep[e]) & ~evidence[s:e]
                else:
                    keep[s:e] = keep[s-1] or keep[e]
    # Convert retained sample mask into source-time intervals; pad each edge to avoid choppy cuts.
    intervals: list[tuple[float, float]] = []
    edges = np.flatnonzero(np.diff(np.r_[False, keep, False].astype(int)) != 0)
    for s, e in zip(edges[::2], edges[1::2]):
        start, end = max(0.0, s * step), min(duration, e * step)
        if end > start:
            intervals.append((start, end))
    logger.info("Selected before smoking filter: %.1f%% (%d intervals)",
                100 * sum(end - start for start, end in intervals) / duration, len(intervals))
    return intervals, duration


def merge_intervals(intervals: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(x, y) for x, y in merged]


def _analysis_cache_path(output_dir: Path, source: Path) -> Path:
    identity = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:16]
    return output_dir / ".clipweb_cache" / f"{identity}.json"


def _analysis_signature(source: Path, cfg: dict) -> str:
    stat = source.stat()
    data = {
        "analysis_version": 3,
        "source": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "webcam": cfg["webcam"],
        "analysis": cfg["analysis"],
    }
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_analysis_cache(cache_path: Path, signature: str) -> tuple[list[tuple[float, float]], float] | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("signature") != signature:
            return None
        intervals = [(float(start), float(end)) for start, end in data["intervals"]]
        return intervals, float(data["duration"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_analysis_cache(cache_path: Path, signature: str,
                         intervals: list[tuple[float, float]], duration: float) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"signature": signature, "duration": duration, "intervals": intervals}
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_path.parent,
                                         prefix="analysis_", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(data, stream, separators=(",", ":"))
        os.replace(temp_path, cache_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _smoking_cache_path(output_dir: Path, source: Path) -> Path:
    identity = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:16]
    return output_dir / ".clipweb_cache" / f"{identity}_smoking.json"


def _smoking_signature(base_signature: str, cfg: dict, model_path: Path) -> str:
    stat = model_path.stat()
    data = {"base": base_signature, "smoking": cfg["smoking"],
            "model": str(model_path.resolve()), "model_size": stat.st_size,
            "model_mtime_ns": stat.st_mtime_ns,
            "join_gap_seconds": cfg["output"]["join_gap_seconds"]}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _smoking_predictor(model_path: Path, settings: dict, logger: logging.Logger):
    backend = settings["backend"]
    if backend not in {"auto", "cpu", "directml"}:
        raise ValueError("smoking.backend must be auto, cpu, or directml")
    if backend != "cpu":
        try:
            import onnxruntime as ort
            if "DmlExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("DirectML execution provider is unavailable")
            options = ort.SessionOptions()
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = int(settings["threads"])
            session = ort.InferenceSession(
                str(model_path), sess_options=options,
                providers=[("DmlExecutionProvider", {"device_id": int(settings["device_id"])}),
                           "CPUExecutionProvider"])
            session.disable_fallback()
            if "DmlExecutionProvider" not in session.get_providers():
                raise RuntimeError("DirectML initialization fell back to CPU")
            input_name = session.get_inputs()[0].name
            logger.info("Smoking inference: DirectML adapter %d (unsupported operations may use CPU)",
                        int(settings["device_id"]))
            return lambda blob: session.run(None, {input_name: blob})[0]
        except Exception as exc:
            if backend == "directml":
                raise RuntimeError(f"DirectML initialization failed: {exc}") from exc
            logger.warning("DirectML unavailable, using CPU: %s", exc)
    cv2.setNumThreads(int(settings["threads"]))
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    logger.info("Smoking inference: OpenCV CPU")
    def predict(blob: np.ndarray) -> np.ndarray:
        net.setInput(blob)
        return net.forward()
    return predict


def detect_smoking(source: Path, intervals: list[tuple[float, float]], cfg: dict,
                   model_path: Path, logger: logging.Logger, progress: FileProgress | None = None) -> list[tuple[float, float]]:
    """Find smoking events only inside already-retained face/silent ranges."""
    s = cfg["smoking"]
    predict = _smoking_predictor(model_path, s, logger)

    fps = float(s["sample_fps"])
    if fps <= 0:
        raise ValueError("smoking.sample_fps must be greater than zero")
    sample_times: list[float] = []
    for start, end in intervals:
        current = start
        while current < end:
            sample_times.append(current)
            current += 1.0 / fps
    if not sample_times:
        return []

    video = av.open(str(source))
    stream = video.streams.video[0]
    box = cfg["webcam"]
    x, y, w, h = (int(box[k]) for k in ("x", "y", "width", "height"))
    model_size = int(s["input_size"])
    hits: list[float] = []
    with StageBar(total=len(sample_times), desc="Smoking", unit="frame", position=2, leave=False, dynamic_ncols=True, progress=progress, stage="smoking") as bar:
        for sample_time in sample_times:
            video.seek(max(0, int(sample_time / float(stream.time_base))),
                       stream=stream, backward=True, any_frame=False)
            frame = None
            for decoded in video.decode(video=0):
                timestamp = float(decoded.pts * decoded.time_base) if decoded.pts is not None else sample_time
                if timestamp + 1e-6 >= sample_time:
                    frame = decoded
                    break
            if frame is not None:
                image = frame.to_ndarray(format="bgr24")
                crop = image[max(0, y):min(image.shape[0], y+h),
                             max(0, x):min(image.shape[1], x+w)]
                if crop.size:
                    ch, cw = crop.shape[:2]
                    scale = min(model_size / cw, model_size / ch)
                    rw, rh = int(round(cw * scale)), int(round(ch * scale))
                    resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
                    canvas = np.full((model_size, model_size, 3), 114, dtype=np.uint8)
                    left, top = (model_size-rw)//2, (model_size-rh)//2
                    canvas[top:top+rh, left:left+rw] = resized
                    blob = cv2.dnn.blobFromImage(canvas, 1/255.0, (model_size, model_size),
                                                 swapRB=True, crop=False)
                    prediction = np.asarray(predict(blob))
                    if prediction.ndim == 3 and prediction.shape[0] == 1:
                        prediction = prediction[0]
                        if prediction.shape[0] < prediction.shape[1]:
                            prediction = prediction.T
                        if prediction.shape[1] >= 5:
                            confidence = float(np.max(prediction[:, 4:]))
                            if confidence >= float(s["confidence"]):
                                hits.append(timestamp)
            bar.update(1)
            bar.set_postfix(hits=len(hits), elapsed=f"{sample_time/60:.1f}m")
    video.close()

    groups: list[list[float]] = []
    for timestamp in hits:
        if groups and timestamp - groups[-1][-1] <= float(s["max_gap_seconds"]):
            groups[-1].append(timestamp)
        else:
            groups.append([timestamp])
    events = [
        (max(0.0, group[0] - float(s["padding_before_seconds"])),
         group[-1] + float(s["padding_after_seconds"]))
        for group in groups if len(group) >= int(s["min_hits"])
    ]
    events = merge_intervals(events, float(s["merge_gap_seconds"]))
    logger.info("Найдено вероятных эпизодов курения: %d", len(events))
    return events


def subtract_intervals(base: list[tuple[float, float]], removals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in base:
        cursor = start
        for cut_start, cut_end in removals:
            if cut_end <= cursor or cut_start >= end:
                continue
            if cut_start > cursor:
                result.append((cursor, min(cut_start, end)))
            cursor = max(cursor, cut_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return [(start, end) for start, end in result if end > start]


def render_video(source: Path, target: Path, intervals: list[tuple[float, float]], cfg: dict, logger: logging.Logger, progress: FileProgress | None = None) -> None:
    if not intervals:
        logger.warning("Нет подходящих фрагментов: %s", source.name)
        return
    # Let the concat demuxer seek directly to each interval. A trim/concat filter graph
    # has to decode the entire long recording and becomes very slow with hundreds of cuts.
    import subprocess
    probe = av.open(str(source))
    has_audio = bool(probe.streams.audio)
    probe.close()
    out = cfg["output"]
    manifest_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         prefix="cw_", suffix=".ffconcat", delete=False) as stream:
            manifest_path = Path(stream.name)
            stream.write("ffconcat version 1.0\n")
            source_name = source.resolve().as_posix().replace("'", "'\\''")
            for start, end in intervals:
                stream.write(f"file '{source_name}'\ninpoint {start:.6f}\noutpoint {end:.6f}\n")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "concat", "-safe", "0", "-i", str(manifest_path),
               "-map", "0:v:0"]
        box = cfg["webcam"]
        crop = f"crop={int(box['width'])}:{int(box['height'])}:{int(box['x'])}:{int(box['y'])}"
        cmd += ["-vf", crop, "-c:v", out["video_codec"],
               "-preset", out["preset"], "-crf", str(out["crf"])]
        if has_audio:
            cmd += ["-map", "0:a:0", "-c:a", out["audio_codec"], "-b:a", out["audio_bitrate"]]
        cmd += ["-progress", "pipe:1", str(target)]
        duration = sum(end - start for start, end in intervals)
        logger.info("Сборка %s: %.1f мин. из %d фрагментов (переходы к отрезкам)", target.name, duration/60, len(intervals))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        with StageBar(total=max(0.1, duration), desc="Render", unit="s", position=2, leave=False, dynamic_ncols=True, progress=progress, stage="render") as bar:
            last = 0.0
            assert proc.stdout is not None
            for line in proc.stdout:
                if line.startswith("out_time_ms="):
                    now = int(line.partition("=")[2] or 0) / 1_000_000
                    bar.update(max(0.0, now-last)); last = now
                elif line.startswith("progress=end"):
                    bar.update(max(0.0, duration-last))
        err = proc.stderr.read() if proc.stderr else ""
        code = proc.wait()
        if code:
            raise RuntimeError(f"ffmpeg завершился с кодом {code}: {err[-2000:]}")
    finally:
        if manifest_path is not None:
            manifest_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Оставляет молчащие моменты с лицом в вебке")
    parser.add_argument("--config", default="config.yaml", help="Путь к config.yaml")
    parser.add_argument("--source", help="Обработать один файл вместо всей input-папки")
    parser.add_argument("--reanalyze", action="store_true", help="Не использовать сохранённые результаты анализа")
    args = parser.parse_args()
    logger = setup_logging()
    try:
        config_path = Path(args.config).resolve()
        cfg = load_config(config_path)
        root = config_path.parent
        source_value = args.source or cfg["paths"].get("source")
        input_dir = (root / cfg["paths"]["input"]).resolve()
        output_dir = (root / cfg["paths"]["output"]).resolve()
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        if source_value:
            one = Path(source_value)
            one = one if one.is_absolute() else (root / one).resolve()
            sources = (sorted(p for p in one.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
                       if one.is_dir() else [one])
        else:
            sources = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        if not sources:
            logger.info("В %s нет видео. Положите туда файл или задайте paths.source в config.yaml", input_dir)
            return
        had_errors = False
        overall = tqdm(total=len(sources), desc="Overall", unit="file", position=0, dynamic_ncols=True)
        for source in sources:
            progress = FileProgress(overall, ("audio", "video", "smoking", "render"))
            if not source.is_file():
                logger.error("Файл не найден: %s", source)
                had_errors = True
                for stage in progress.stages:
                    progress.update(stage, 1.0)
                progress.close()
                continue
            try:
                signature = _analysis_signature(source, cfg)
                cache_path = _analysis_cache_path(output_dir, source)
                cached = None if args.reanalyze else _load_analysis_cache(cache_path, signature)
                if cached is None:
                    intervals, duration = analyze_video(source, cfg, logger, progress)
                    _save_analysis_cache(cache_path, signature, intervals, duration)
                    logger.info("Результат анализа сохранён до сборки")
                else:
                    intervals, duration = cached
                    logger.info("Использую сохранённый анализ для %s", source.name)
                progress.update("audio", 1.0)
                progress.update("video", 1.0)
                intervals = merge_intervals(intervals, float(cfg["output"]["join_gap_seconds"]))
                if cfg["smoking"]["enabled"]:
                    model_path = Path(cfg["smoking"]["model"])
                    if not model_path.is_absolute():
                        model_path = (root / model_path).resolve()
                    if not model_path.is_file():
                        raise FileNotFoundError(f"Модель курения не найдена: {model_path}")
                    smoking_cache = _smoking_cache_path(output_dir, source)
                    smoking_signature = _smoking_signature(signature, cfg, model_path)
                    cached_smoking = None if args.reanalyze else _load_analysis_cache(smoking_cache, smoking_signature)
                    if cached_smoking is None:
                        smoking_events = detect_smoking(source, intervals, cfg, model_path, logger, progress)
                        _save_analysis_cache(smoking_cache, smoking_signature, smoking_events, duration)
                    else:
                        smoking_events, _ = cached_smoking
                        logger.info("Использую сохранённый анализ курения для %s", source.name)
                    intervals = subtract_intervals(intervals, smoking_events)
                    logger.info("Selected after smoking filter: %.1f%%",
                                100 * sum(end - start for start, end in intervals) / duration)
                    logger.info("После вырезания курения осталось фрагментов: %d", len(intervals))
                progress.update("smoking", 1.0)
                target = output_dir / f"{source.stem}_clipweb.mp4"
                render_video(source, target, intervals, cfg, logger, progress)
                logger.info("Готово: %s (сохранено %.1f%% длительности)", target.name,
                            100 * sum(e-s for s, e in intervals) / duration if intervals else 0)
            except Exception as exc:
                had_errors = True
                logger.exception("Ошибка при обработке %s: %s", source.name, exc)
            finally:
                for stage in progress.stages:
                    progress.update(stage, 1.0)
                progress.close()
        overall.close()
        if had_errors:
            sys.exit(1)
    except Exception as exc:
        logger.exception("Ошибка конфигурации/запуска: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
