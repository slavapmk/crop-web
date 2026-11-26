from __future__ import annotations

import math


def crop_to_aspect(box: dict, aspect: str | None) -> dict:
    """Return the largest even-aligned exact-aspect rectangle inside the box."""
    if aspect is None or aspect == 'source':
        return box.copy()
    try:
        left, right = str(aspect).split(':')
        width_ratio, height_ratio = int(left), int(right)
        if width_ratio <= 0 or height_ratio <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError('aspect.ratio must be WIDTH:HEIGHT, source, or null') from exc
    divisor = math.gcd(width_ratio, height_ratio)
    width_ratio //= divisor
    height_ratio //= divisor
    multiplier = 2 if width_ratio % 2 or height_ratio % 2 else 1
    unit_w, unit_h = width_ratio * multiplier, height_ratio * multiplier
    x, y, w, h = (int(box[key]) for key in ('x', 'y', 'width', 'height'))
    size = min(w // unit_w, h // unit_h)
    while size:
        new_w, new_h = size * unit_w, size * unit_h
        new_x = (x + (w - new_w) // 2) // 2 * 2
        new_y = (y + (h - new_h) // 2) // 2 * 2
        if new_x >= x and new_y >= y and new_x + new_w <= x + w and new_y + new_h <= y + h:
            return {**box, 'x': new_x, 'y': new_y, 'width': new_w, 'height': new_h}
        size -= 1
    raise ValueError('Webcam rectangle is too small for the requested aspect ratio')
