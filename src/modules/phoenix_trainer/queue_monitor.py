"""Persistent Phoenix queue operations monitoring.

The queue owns operator-visible lifecycle history. Trainer workers remain isolated
processes and may exit immediately after a job, but their state transitions and
log output are mirrored into an append-only queue event stream.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class QueueMonitorStore:
    """Append-only queue event history with a bounded in-memory view."""

    def __init__(
        self,
        event_file: Path,
        session_id: str,
        memory_limit: int = 500,
    ) -> None:
        self.event_file = Path(event_file)
        self.event_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id)
        self.memory_limit = int(memory_limit)
        self.events: Deque[Dict[str, Any]] = deque(maxlen=self.memory_limit)
        self._load_tail()

    def _load_tail(self) -> None:
        if not self.event_file.exists():
            return

        lines: Deque[str] = deque(maxlen=self.memory_limit)
        with self.event_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)

        for line in lines:
            try:
                self.events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    def append(
        self,
        event_type: str,
        message: str,
        *,
        dataset: Optional[str] = None,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = {
            "timestamp": utc_now(),
            "session_id": self.session_id,
            "event_type": str(event_type),
            "dataset": dataset,
            "run_id": run_id,
            "status": status,
            "message": str(message),
            "details": json_safe(dict(details or {})),
        }
        with self.event_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        self.events.append(event)
        return event

    def recent(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        values = list(self.events)
        if limit is None:
            return values
        return values[-int(limit):]

    @staticmethod
    def format_event(event: Mapping[str, Any]) -> str:
        timestamp = str(event.get("timestamp", ""))
        display_time = timestamp
        try:
            parsed = datetime.fromisoformat(timestamp)
            display_time = parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        dataset = event.get("dataset")
        prefix = f"[{display_time}]"
        if dataset:
            prefix += f" [{dataset}]"
        return f"{prefix} {event.get('message', '')}"


class RunLogTailer:
    """Incrementally read new lines from one or more worker run.log files."""

    def __init__(self) -> None:
        self._offsets: Dict[str, int] = {}

    def reset(self, path: Path) -> None:
        self._offsets.pop(str(Path(path).resolve()), None)

    def read_new(self, path: Path) -> List[str]:
        path = Path(path)
        if not path.exists():
            return []

        key = str(path.resolve())
        offset = self._offsets.get(key, 0)
        size = path.stat().st_size

        if size < offset:
            offset = 0

        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            lines = [line.rstrip("\n") for line in handle]
            self._offsets[key] = handle.tell()

        return [line for line in lines if line.strip()]
