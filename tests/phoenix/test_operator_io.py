import json
import logging
from pathlib import Path
import threading
import time

from modules.phoenix_trainer.atomic_io import atomic_write_json
from modules.phoenix_trainer.operator_logging import PhoenixCliFilter


def test_operator_cli_filter_only_allows_explicit_records():
    filter_ = PhoenixCliFilter()

    detail = logging.LogRecord(
        name="PhoenixTrainer",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Batch 1/10",
        args=(),
        exc_info=None,
    )
    assert filter_.filter(detail) is False

    operator = logging.LogRecord(
        name="PhoenixTrainer",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Training started",
        args=(),
        exc_info=None,
    )
    operator.phoenix_cli = True
    assert filter_.filter(operator) is True


def test_atomic_json_never_exposes_partial_document(tmp_path):
    path = tmp_path / "training_status.json"
    atomic_write_json(path, {"version": 0, "payload": "initial"})

    stop = threading.Event()
    decode_errors = []

    def reader():
        while not stop.is_set():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                decode_errors.append(exc)
                stop.set()

    thread = threading.Thread(target=reader)
    thread.start()

    try:
        for version in range(1, 100):
            atomic_write_json(
                path,
                {
                    "version": version,
                    "payload": "x" * 10000,
                },
            )
    finally:
        stop.set()
        thread.join(timeout=2)

    assert decode_errors == []
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 99


def test_atomic_json_replaces_existing_file(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"status": "started"})
    atomic_write_json(path, {"status": "completed", "progress": 1.0})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "completed",
        "progress": 1.0,
    }
