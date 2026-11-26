from __future__ import annotations

import math

import numpy as np


# Detections: (time, x1, y1, x2, y2); regions: (start, end, x, y, width, height).
Detection = tuple[float, float, float, float, float]
Region = tuple[float, float, int, int, int, int]


def select_smoking_box(prediction: np.ndarray, width: int, height: int, scale: float,
                       left: int, top: int, confidence: float, faces, near_mouth: bool) -> tuple[float, float, float, float] | None:
    """Decode the highest-scoring candidate that passes geometric checks."""
    result = np.asarray(prediction)
    if result.ndim != 3 or result.shape[0] != 1:
        return None
    rows = result[0]
    if rows.shape[0] < rows.shape[1]:
        rows = rows.T
    if rows.ndim != 2 or rows.shape[1] < 5:
        return None
    scores = np.max(rows[:, 4:], axis=1)
    candidates = np.flatnonzero(np.isfinite(scores) & (scores >= confidence))
    for index in candidates[np.argsort(scores[candidates])[::-1]]:
        row = rows[index]
        if not np.isfinite(row[:5]).all():
            continue
        cx, cy, bw, bh = map(float, row[:4])
        if bw <= 0 or bh <= 0:
            continue
        x1, y1 = (cx - bw / 2 - left) / scale, (cy - bh / 2 - top) / scale
        x2, y2 = (cx + bw / 2 - left) / scale, (cy + bh / 2 - top) / scale
        x1, y1 = max(0., x1), max(0., y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x2 <= x1 or y2 <= y1 or x2 - x1 > width * .4 or y2 - y1 > height * .4:
            continue
        if near_mouth:
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if not any(abs(center_x - (fx + fw * .5)) <= fw * .8 and
                       abs(center_y - (fy + fh * .82)) <= fh * .65
                       for fx, fy, fw, fh in faces):
                continue
        return x1, y1, x2, y2
    return None


def build_blur_regions(events: list[tuple[float, float]], hits: list[Detection],
                       width: int, height: int, margin: int, hold: float) -> tuple[list[Region], int]:
    """Keep blur local to confirmed samples; do not extrapolate through long gaps."""
    ordered = sorted(hits)
    regions: list[Region] = []
    missing = 0
    for event_start, event_end in events:
        relevant = [hit for hit in ordered if event_start <= hit[0] < event_end]
        if not relevant:
            missing += 1
            continue
        for i, (stamp, x1, y1, x2, y2) in enumerate(relevant):
            start = max(event_start, stamp - hold)
            end = min(event_end, stamp + hold)
            if i:
                start = max(start, (relevant[i - 1][0] + stamp) / 2)
            if i + 1 < len(relevant):
                end = min(end, (stamp + relevant[i + 1][0]) / 2)
            left = max(0, math.floor((x1 - margin) / 2) * 2)
            top = max(0, math.floor((y1 - margin) / 2) * 2)
            right = min(width // 2 * 2, math.ceil((x2 + margin) / 2) * 2)
            bottom = min(height // 2 * 2, math.ceil((y2 + margin) / 2) * 2)
            if end > start and right > left and bottom > top:
                region = (start, end, left, top, right-left, bottom-top)
                if regions and regions[-1][2:] == region[2:] and abs(regions[-1][1] - start) < 1e-6:
                    regions[-1] = (regions[-1][0], end, *region[2:])
                else:
                    regions.append(region)
    return regions, missing


def map_regions(intervals: list[tuple[float, float]], regions: list[Region]) -> list[Region]:
    """Map source-time regions to a concatenated output timeline."""
    mapped: list[Region] = []
    offset = 0.0
    for start, end in intervals:
        for region_start, region_end, x, y, width, height in regions:
            left, right = max(start, region_start), min(end, region_end)
            if left < right:
                mapped.append((offset + left - start, offset + right - start, x, y, width, height))
        offset += end - start
    return mapped
