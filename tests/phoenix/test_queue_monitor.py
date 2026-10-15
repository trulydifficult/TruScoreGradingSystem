import json

from modules.phoenix_trainer.queue_monitor import (
    QueueMonitorStore,
    RunLogTailer,
)


def test_queue_monitor_history_survives_restart(tmp_path):
    event_file = tmp_path / "queue_events.jsonl"

    first = QueueMonitorStore(event_file, "session-a", memory_limit=10)
    first.append(
        "dataset_dispatched",
        "Training started on dataset A.",
        dataset="dataset-A",
        status="training",
    )
    first.append(
        "dataset_completed",
        "Training complete on dataset A.",
        dataset="dataset-A",
        status="completed",
        details={"accuracy": 95.0},
    )

    second = QueueMonitorStore(event_file, "session-b", memory_limit=10)
    events = second.recent()

    assert len(events) == 2
    assert events[0]["session_id"] == "session-a"
    assert events[1]["event_type"] == "dataset_completed"
    assert events[1]["details"]["accuracy"] == 95.0


def test_run_log_tailer_only_returns_new_lines(tmp_path):
    log_file = tmp_path / "run.log"
    log_file.write_text("line one\nline two\n", encoding="utf-8")

    tailer = RunLogTailer()

    assert tailer.read_new(log_file) == ["line one", "line two"]
    assert tailer.read_new(log_file) == []

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("line three\n")

    assert tailer.read_new(log_file) == ["line three"]


def test_queue_monitor_persists_jsonl_records(tmp_path):
    event_file = tmp_path / "queue_events.jsonl"
    store = QueueMonitorStore(event_file, "session-a", memory_limit=10)

    store.append(
        "training_status",
        "TRAINING | epoch 2/5 | 40.0%",
        dataset="dataset-A",
        run_id="run-1",
        status="training",
        details={"epoch": 2, "progress": 0.4},
    )

    line = event_file.read_text(encoding="utf-8").strip()
    payload = json.loads(line)

    assert payload["dataset"] == "dataset-A"
    assert payload["run_id"] == "run-1"
    assert payload["details"]["progress"] == 0.4
