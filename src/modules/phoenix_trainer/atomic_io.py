"""Atomic local persistence helpers for Phoenix cross-process state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Atomically replace a text file with fully flushed contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """Serialize JSON completely, then atomically publish it."""
    text = json.dumps(
        payload,
        indent=indent,
        sort_keys=sort_keys,
    )
    atomic_write_text(path, text + "\n")
