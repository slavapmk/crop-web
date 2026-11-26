from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

import av
from tqdm import tqdm
from clipweb.progress import STAGE_BAR_FORMAT


_FRAME_NAME = re.compile(r'^\d+-\d+\.mp4$')


def _write_manifest(path: Path, source: Path, names: set[str]) -> None:
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix='episodes_', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump({'source': str(source.resolve()), 'files': sorted(names)}, stream, ensure_ascii=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def index_intervals(source: Path, intervals: list[tuple[float, float]]) -> list[tuple[float, float, int, int]]:
    """Map half-open timestamps to zero-based, inclusive decoded frame numbers."""
    if not intervals:
        return []
    bounds = sorted((timestamp, index, edge) for index, pair in enumerate(intervals)
                    for edge, timestamp in enumerate(pair))
    indexes: list[list[int | None]] = [[None, None] for _ in intervals]
    cursor = 0
    frame_count = 0
    last_timestamp = None
    with av.open(str(source)) as container:
        video = container.streams.video[0]
        video.thread_type = 'AUTO'
        duration = float(container.duration / av.time_base) if container.duration else 0
        with tqdm(total=max(1., duration), desc='Index', unit='s', position=2, leave=False,
                  dynamic_ncols=True, bar_format=STAGE_BAR_FORMAT) as bar:
            shown = 0.0
            for frame in container.decode(video=0):
                if frame.pts is None:
                    raise ValueError('Cannot determine source frame numbers: video frame has no timestamp')
                timestamp = float(frame.pts * frame.time_base)
                if last_timestamp is not None and timestamp < last_timestamp - 1e-6:
                    raise ValueError('Cannot determine source frame numbers: non-monotonic video timestamps')
                last_timestamp = timestamp
                while cursor < len(bounds) and bounds[cursor][0] <= timestamp + 1e-6:
                    _, item, edge = bounds[cursor]
                    indexes[item][edge] = frame_count
                    cursor += 1
                frame_count += 1
                if timestamp > shown + 1:
                    bar.update(min(timestamp - shown, max(0, bar.total - bar.n)))
                    shown = timestamp
                if cursor == len(bounds):
                    break
    while cursor < len(bounds):
        _, item, edge = bounds[cursor]
        indexes[item][edge] = frame_count
        cursor += 1
    result = []
    for (start, end), (first, after_last) in zip(intervals, indexes):
        assert first is not None and after_last is not None
        if after_last > first:
            result.append((start, end, first, after_last - 1))
    return result


def render_episodes(source: Path, folder: Path, cache_path: Path, intervals: list[tuple[float, float]],
                    cfg: dict, logger: logging.Logger, render, progress, stage_bar) -> None:
    episodes = index_intervals(source, intervals)
    folder.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    old_names: set[str] = set()
    if cache_path.is_file():
        try:
            data = json.loads(cache_path.read_text(encoding='utf-8'))
            if data.get('source') == str(source.resolve()):
                old_names = {name for name in data['files'] if isinstance(name, str) and _FRAME_NAME.fullmatch(name)}
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning('Cannot read previous episode manifest: %s', cache_path)
    names = [f'{start_frame}-{end_frame}.mp4' for _, _, start_frame, end_frame in episodes]
    if len(set(names)) != len(names):
        raise ValueError('Different intervals resolve to the same frame range')
    for name in names:
        if (folder / name).exists() and name not in old_names:
            raise FileExistsError(f'Refusing to overwrite an unmanaged file: {folder / name}')
    # Record intended outputs first so interrupted renders can be retried safely.
    _write_manifest(cache_path, source, old_names.union(names))
    logger.info('Rendering %d separate episodes to %s', len(episodes), folder)
    with stage_bar(total=max(1, len(episodes)), desc='Render', unit='episode', position=2,
                   leave=False, dynamic_ncols=True, progress=progress, stage='render') as bar:
        for (start, end, _, _), name in zip(episodes, names):
            logger.debug('Rendering episode %s from %.6f to %.6f seconds', name, start, end)
            fd, temporary = tempfile.mkstemp(prefix='.clipweb_', suffix='.mp4', dir=folder)
            os.close(fd)
            temp_path = Path(temporary)
            try:
                render(source, temp_path, [(start, end)], cfg, logger, show_progress=False)
                os.replace(temp_path, folder / name)
            finally:
                temp_path.unlink(missing_ok=True)
            bar.update(1)
        if not episodes:
            bar.update(1)
    for name in old_names.difference(names):
        logger.debug('Removing obsolete generated episode %s', folder / name)
        (folder / name).unlink(missing_ok=True)
    _write_manifest(cache_path, source, set(names))
    logger.info('Finished %d episodes', len(episodes))
