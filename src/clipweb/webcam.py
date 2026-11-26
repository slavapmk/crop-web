from __future__ import annotations

import logging
from pathlib import Path

import av
import cv2
import numpy as np
from tqdm import tqdm
from clipweb.progress import STAGE_BAR_FORMAT


def _best_edge(scores: np.ndarray, low: int, high: int, ideal: float, scale: float) -> int:
    low = max(1, low)
    high = min(len(scores) - 1, high)
    if high <= low:
        raise ValueError('Cannot infer webcam boundary')
    positions = np.arange(low, high + 1)
    # Prefer an edge close to a plausible webcam aspect ratio when scores are similar.
    weighted = scores[positions] - .0003 * np.abs(positions - ideal) / max(1., scale) * 100
    position = int(positions[int(np.argmax(weighted))])
    if scores[position] < .05:
        raise ValueError('Webcam boundary is unclear; configure webcam coordinates manually')
    return position


def infer_rectangle(frames: list[np.ndarray], detector: cv2.CascadeClassifier, minimum_hits: int = 3) -> dict:
    """Find a recurrent face near a frame edge, then estimate its overlay boundaries."""
    if not frames:
        raise ValueError('No frames available for webcam detection')
    height, width = frames[0].shape[:2]
    groups: list[dict] = []
    for frame_number, image in enumerate(frames):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40),
                                          maxSize=(max(40, width // 4), max(40, height // 3)))
        for x, y, w, h in faces:
            cx, cy = x + w / 2, y + h / 2
            # A stable webcam face is normally close to an edge; this excludes
            # full-screen faces in the viewed video and game characters.
            if min(cx / width, 1 - cx / width) > .30 or min(cy / height, 1 - cy / height) > .30:
                continue
            eligible = [g for g in groups if abs(g['center'][0] - cx) < width * .10
                        and abs(g['center'][1] - cy) < height * .10]
            group = min(eligible, key=lambda g: abs(g['center'][0] - cx) + abs(g['center'][1] - cy)) if eligible else None
            if group is None:
                group = {'faces': [], 'frames': set(), 'center': (cx, cy)}
                groups.append(group)
            if frame_number not in group['frames']:
                group['frames'].add(frame_number)
                group['faces'].append((int(x), int(y), int(w), int(h), frame_number))
                group['center'] = tuple(np.median(np.array([(f[0]+f[2]/2, f[1]+f[3]/2) for f in group['faces']]), axis=0))
    if not groups:
        raise ValueError('No recurring webcam face found; set webcam.auto to false and configure webcam coordinates')
    group = max(groups, key=lambda g: len(g['frames']))
    if len(group['frames']) < minimum_hits:
        raise ValueError('Webcam position is uncertain; set webcam.auto to false and configure webcam coordinates')
    face = np.median(np.array([f[:4] for f in group['faces']]), axis=0)
    x, y, w, h = map(float, face)
    edges = np.array([cv2.Canny(cv2.cvtColor(frames[f[4]], cv2.COLOR_BGR2GRAY), 70, 140)
                      for f in group['faces']])
    xs = slice(max(0, int(x-w)), min(width, int(x+2*w)))
    vertical_scores = np.median(edges[:, :, xs].mean(axis=2), axis=0) / 255.0
    if y + h/2 < height / 2:
        top = 0 if y < height * .2 else _best_edge(vertical_scores, int(y-3*h), int(y-.2*h), y-h, h)
        bottom = _best_edge(vertical_scores, int(max(y+h, top+2*h)), int(min(height-1, top+5*h)), top+3.3*h, h)
    else:
        bottom = height if y+h > height * .8 else _best_edge(vertical_scores, int(y+h), int(y+4*h), y+2*h, h)
        top = _best_edge(vertical_scores, int(max(0, bottom-5*h)), int(min(y, bottom-2*h)), bottom-3.3*h, h)
    crop_height = bottom - top
    if crop_height < 40:
        raise ValueError('Invalid webcam height')
    horizontal_scores = np.median(edges[:, top:bottom, :].mean(axis=1), axis=0) / 255.0
    if x + w/2 > width / 2:
        right = width if x+w > width*.8 else _best_edge(horizontal_scores, int(x+w), int(x+5*w), x+2*w, w)
        left = _best_edge(horizontal_scores, int(right-2.2*crop_height), int(min(x, right-1.2*crop_height)), right-1.6*crop_height, crop_height)
    else:
        left = 0 if x < width*.2 else _best_edge(horizontal_scores, int(x-5*w), int(x), x-2*w, w)
        right = _best_edge(horizontal_scores, int(max(x+w, left+1.2*crop_height)), int(left+2.2*crop_height), left+1.6*crop_height, crop_height)
    # H.264/yuv420p crops require even coordinates and dimensions.
    left, top, right, bottom = (int(v) // 2 * 2 for v in (left, top, right, bottom))
    if right <= left or bottom <= top:
        raise ValueError('Invalid webcam rectangle')
    return {'x': left, 'y': top, 'width': right-left, 'height': bottom-top}


def detect_webcam(path: Path, sample_count: int, logger: logging.Logger) -> dict:
    if sample_count < 3:
        raise ValueError('webcam.samples must be at least 3')
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector.empty():
        raise RuntimeError('OpenCV face detector could not be loaded')
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(container.duration / av.time_base) if container.duration else float(stream.duration * stream.time_base)
        with tqdm(total=sample_count, desc='Webcam', unit='frame', position=2, leave=False,
                  dynamic_ncols=True, bar_format=STAGE_BAR_FORMAT) as bar:
            for timestamp in np.linspace(0.03 * duration, 0.97 * duration, sample_count):
                container.seek(max(0, int(timestamp / float(stream.time_base))), stream=stream, backward=True)
                for frame in container.decode(video=0):
                    if frame.pts is not None and float(frame.pts * frame.time_base) + 1e-6 >= timestamp:
                        frames.append(frame.to_ndarray(format='bgr24'))
                        break
                bar.update(1)
    box = infer_rectangle(frames, detector, max(3, sample_count // 8))
    logger.debug('Sampled %d of %d requested frames for webcam detection', len(frames), sample_count)
    logger.info('Detected webcam rectangle: x=%d y=%d width=%d height=%d', box['x'], box['y'], box['width'], box['height'])
    return box
