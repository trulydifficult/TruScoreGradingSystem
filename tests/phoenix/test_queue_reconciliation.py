import json
from pathlib import Path

from modules.phoenix_trainer.phoenix_training_queue_standalone import (
    PhoenixTrainingQueueApp,
)


def test_load_state_reconciles_preexisting_pending_dataset(tmp_path):
    pending = tmp_path / "pending"
    pending.mkdir()
    state_file = tmp_path / "queue_state.json"

    stale = pending / "stale_dataset"
    new_dataset = pending / "new_dataset"
    new_dataset.mkdir()
    (new_dataset / "dataset_config.json").write_text(
        json.dumps(
            {
                "dataset_name": "new_dataset",
                "dataset_type": "corner_quality_classification",
                "pipeline": "Vision Transformer",
                "priority": 1,
                "hardware_hint": "gpu",
                "image_count": 16,
            }
        ),
        encoding="utf-8",
    )

    completed_path = tmp_path / "completed" / "finished_dataset"
    state_file.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "id": 1,
                        "name": "stale_dataset",
                        "path": str(stale),
                        "status": "pending",
                    },
                    {
                        "id": 2,
                        "name": "finished_dataset",
                        "path": str(completed_path),
                        "status": "completed",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    app = PhoenixTrainingQueueApp.__new__(PhoenixTrainingQueueApp)
    app.datasets = []
    app.pending_dir = pending
    app.state_file = state_file
    app.update_queue_display = lambda: None

    app.load_queue_state()

    names = {dataset["name"] for dataset in app.datasets}
    assert "new_dataset" in names
    assert "finished_dataset" in names
    assert "stale_dataset" not in names

    discovered = next(
        dataset for dataset in app.datasets if dataset["name"] == "new_dataset"
    )
    assert Path(discovered["path"]).resolve() == new_dataset.resolve()
    assert discovered["status"] == "pending"
