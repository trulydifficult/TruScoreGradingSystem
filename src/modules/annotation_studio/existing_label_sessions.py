"""Studio-owned input pairing and crash-safe annotation batch state.

This is a local operator state file, never a training label or Git artifact.
Model inference and rendering are exclusively plugin-owned.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone


class AnnotationBatchError(ValueError):
    """An existing annotation set is missing, ambiguous or malformed."""


def default_session_path() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state"))
    return base / "TruScore" / "annotation_batch_v1.json"


def pair_selected_files(images, labels):
    """Match individually selected images against selected combined YOLO labels.

    Names are case-insensitive for matching, but duplicate case-folded names
    are rejected, rather than arbitrarily choosing the wrong card.
    """
    if not images or not labels:
        raise AnnotationBatchError("Choose label files first, then images.")
    by_stem = {}
    for value in labels:
        path = Path(value).expanduser().resolve()
        if path.suffix.lower() != ".txt" or not path.is_file():
            raise AnnotationBatchError(f"Not an existing YOLO .txt label: {path}")
        key = path.stem.casefold()
        if key in by_stem:
            raise AnnotationBatchError(f"Ambiguous duplicate label name: {path.stem}")
        by_stem[key] = path

    sources, matched, seen = [], {}, set()
    for value in images:
        path = Path(value).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
        }:
            raise AnnotationBatchError(f"Image file not found/supported: {path}")
        key = path.stem.casefold()
        if key in seen:
            raise AnnotationBatchError(f"Duplicate selected image stem: {path.stem}")
        seen.add(key)
        if key not in by_stem:
            raise AnnotationBatchError(f"No matching label selected for {path.name}")
        sources.append(str(path))
        matched[str(path)] = str(by_stem[key])
    return sources, matched


def parse_combined_yolo(path, width, height):
    """Two detections, class 0 outside and class 1 graphic; no fake masks."""
    file = Path(path)
    if width <= 0 or height <= 0 or not file.is_file():
        raise AnnotationBatchError(f"Invalid image dimensions or missing label: {file}")
    rows = {}
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise AnnotationBatchError(f"{file.name}: expected five-column YOLO detection rows")
        try:
            numbers = [float(x) for x in parts]
        except ValueError as error:
            raise AnnotationBatchError(f"{file.name}: invalid numeric label") from error
        from math import isfinite
        if not all(isfinite(value) for value in numbers):
            raise AnnotationBatchError(f"{file.name}: non-finite YOLO coordinate")
        cls_float, cx, cy, bw, bh = numbers
        if cls_float not in (0.0, 1.0) or int(cls_float) in rows:
            raise AnnotationBatchError(f"{file.name}: exactly one outer (0) and graphic (1) required")
        if not (0 < bw <= 1 and 0 < bh <= 1
                and 0 <= cx - bw/2 < cx + bw/2 <= 1
                and 0 <= cy - bh/2 < cy + bh/2 <= 1):
            raise AnnotationBatchError(f"{file.name}: normalized coordinates out of bounds")
        rows[int(cls_float)] = f"{int(cls_float)} {cx:.12g} {cy:.12g} {bw:.12g} {bh:.12g}"
    if set(rows) != {0, 1}:
        raise AnnotationBatchError(f"{file.name}: both outer (0) and graphic (1) required")
    return rows


def build_session(images, labels):
    files, matched = pair_selected_files(images, labels)
    return {
        "version": 1, "mode": "existing_yolo",
        "image_files": files, "labels": matched,
        "completed": [], "current_index": 0,
    }


def read_session(path=None):
    file = Path(path) if path is not None else default_session_path()
    if not file.is_file():
        raise AnnotationBatchError("No previous annotation session was saved.")
    try:
        state = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotationBatchError("Session record is unreadable or damaged.") from error
    if not isinstance(state, dict) or state.get("version") != 1 or state.get("mode") != "existing_yolo":
        raise AnnotationBatchError("Unsupported annotation session version or mode.")
    images = state.get("image_files")
    labels = state.get("labels")
    completed = state.get("completed")
    if not isinstance(images, list) or not isinstance(labels, dict) or not isinstance(completed, list):
        raise AnnotationBatchError("Incomplete session record.")
    paired, mapping = pair_selected_files(images, [labels.get(item) for item in images])
    if paired != images or mapping != labels:
        raise AnnotationBatchError("Saved session file pairing has changed.")
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(images)
           for i in completed):
        raise AnnotationBatchError("Invalid completed-image index in session.")
    if len(set(completed)) != len(completed):
        raise AnnotationBatchError("Duplicate completed-image indices in session.")
    return state


def next_pending(state):
    complete = set(state["completed"])
    return next((i for i in range(len(state["image_files"])) if i not in complete), None)


def save_session(state, path=None):
    """Atomic durable record: write-temp+fsync+replace+directory fsync."""
    file = Path(path) if path is not None else default_session_path()
    file.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(state, indent=2, sort_keys=True).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="._annotation_", suffix=".tmp",
            dir=str(file.parent), delete=False
        ) as fh:
            temporary = Path(fh.name)
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, file)
        temporary = None
        try:
            fd = os.open(str(file.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass  # Not all filesystems support fsync on directories.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def record_completed(state, index, path=None):
    """Persist success BEFORE updating the in-memory session."""
    if not isinstance(index, int) or index < 0 or index >= len(state["image_files"]):
        raise AnnotationBatchError("Saved image index outside batch")
    updated = {**state, "completed": sorted(set(state["completed"]) | {index})}
    updated["current_index"] = next_pending(updated)
    save_session(updated, path)
    state.clear()
    state.update(updated)
    return updated["current_index"]
