"""Label-pairing and durable session recovery, without Qt or GPU."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from modules.annotation_studio.existing_label_sessions import (  # noqa: E402
    AnnotationBatchError, build_session, pair_selected_files, parse_combined_yolo,
    read_session, save_session, record_completed, next_pending,
)


@pytest.fixture
def sample(tmp_path):
    photos, labels = [], []
    for idx in range(3):
        photo = tmp_path / "images" / f"card{idx:03d}.jpg"
        photo.parent.mkdir(exist_ok=True)
        photo.write_bytes(b"dummy")
        label = tmp_path / "labels" / f"Card{idx:03d}.txt"
        label.parent.mkdir(exist_ok=True)
        label.write_text("0 0.5 0.5 0.9 0.9\n1 0.5 0.5 0.8 0.8\n")
        photos.append(photo)
        labels.append(label)
    return photos, labels


def test_labels_first_case_insensitive_exact_stem_pairing(sample):
    photos, labels = sample
    state = build_session(photos, labels)
    assert len(state["labels"]) == 3
    assert list(state["labels"].values())[0].endswith("Card000.txt")
    assert set(parse_combined_yolo(labels[0], 500, 700)) == {0, 1}


def test_missing_label_and_ambiguous_stem_fail(sample, tmp_path):
    photos, labels = sample
    with pytest.raises(AnnotationBatchError, match="No matching label"):
        pair_selected_files(photos, labels[:-1])
    extra = tmp_path / "elsewhere" / "cArD000.txt"
    extra.parent.mkdir()
    extra.write_text(labels[0].read_text())
    with pytest.raises(AnnotationBatchError, match="Ambiguous"):
        pair_selected_files(photos, labels + [extra])


@pytest.mark.parametrize("bad", [
    "0 0.5 0.5 0.9 0.9\n",              # missing class 1
    "0 0.5 0.5 0.9 0.9\n0 0.5 0.5 0.8 0.8\n",
    "0 0.5 0.5 0.9 0.9\n1 0.5 0.5 NaN 0.8\n",
    "0 0.5 0.5 0.9 0.9\n1 0.5 0.5 0.8\n",
    "0 0.5 0.5 0.9 0.9\n1 0.05 0.5 0.8 0.8\n",
])
def test_invalid_labels_are_not_promoted_to_boxes(tmp_path, bad):
    file = tmp_path / "card.txt"
    file.write_text(bad)
    with pytest.raises(AnnotationBatchError):
        parse_combined_yolo(file, 500, 700)


def test_interrupted_large_batch_resumes_at_first_unsaved(sample, tmp_path):
    photos, labels = sample
    state = build_session(photos, labels)
    location = tmp_path / "session.json"
    save_session(state, location)
    assert next_pending(read_session(location)) == 0
    record_completed(state, 0, location)
    record_completed(state, 2, location)
    assert next_pending(read_session(location)) == 1
    assert read_session(location)["completed"] == [0, 2]
    record_completed(state, 1, location)
    assert next_pending(read_session(location)) is None


def test_failed_progress_persist_never_claims_saved(sample, tmp_path):
    photos, labels = sample
    state = build_session(photos, labels)
    destination = tmp_path / "not-a-directory"
    destination.write_text("block directory creation")
    with pytest.raises(OSError):
        record_completed(state, 0, destination / "session.json")
    assert state["completed"] == []


def test_session_resume_rejects_disappeared_source(sample, tmp_path):
    photos, labels = sample
    state = build_session(photos, labels)
    location = tmp_path / "session.json"
    save_session(state, location)
    labels[1].unlink()
    with pytest.raises(AnnotationBatchError):
        read_session(location)
