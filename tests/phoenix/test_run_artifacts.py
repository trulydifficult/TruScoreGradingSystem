import json
from pathlib import Path

from modules.phoenix_trainer.run_artifacts import RunRecorder


def test_run_recorder_creates_unique_forensic_artifacts(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "sample.txt").write_text("sample", encoding="utf-8")

    recorder = RunRecorder(
        output_root=tmp_path / "runs",
        dataset_name="demo",
        dataset_path=dataset,
        resolved_config={
            "epochs": 1,
            "batch_size": 2,
            "output_dir": "ignored",
        },
        dataset_config={"dataset_type": "corner_quality_classification"},
    )

    assert recorder.run_dir.exists()
    assert (recorder.run_dir / "training_config.json").exists()
    assert (recorder.run_dir / "dataset_manifest.json").exists()
    assert (recorder.run_dir / "environment.json").exists()
    assert (recorder.run_dir / "run.json").exists()
    assert (recorder.run_dir / "run.log").exists()

    resolved = json.loads(
        (recorder.run_dir / "training_config.json").read_text(encoding="utf-8")
    )
    assert resolved["run_id"] == recorder.run_id
    assert resolved["run_dir"] == str(recorder.run_dir)
    assert resolved["output_dir"] == str(recorder.run_dir)

    recorder.record_metrics(
        {"train_loss": 0.5, "accuracy": 75.0},
        epoch=1,
        progress=1.0,
    )
    recorder.finalize(
        status="completed",
        final_metrics={"accuracy": 75.0},
        artifacts=recorder.artifact_inventory(),
    )
    recorder.write_dataset_reference(dataset)

    manifest = json.loads(
        (recorder.run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["final_metrics"]["accuracy"] == 75.0
    assert manifest["dataset_fingerprint"]["file_count"] >= 1

    reference = json.loads(
        (dataset / "training_run.json").read_text(encoding="utf-8")
    )
    assert reference["run_id"] == recorder.run_id
    assert reference["run_dir"] == str(recorder.run_dir)

    metric_lines = (
        recorder.metrics_path.read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(metric_lines) == 1
    metric = json.loads(metric_lines[0])
    assert metric["epoch"] == 1
    assert metric["metrics"]["accuracy"] == 75.0
