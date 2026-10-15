"""Self-contained Phoenix run artifacts and provenance.

Every training attempt gets an immutable run directory containing the resolved
configuration, environment snapshot, JSONL metrics, a human-readable log, and a
terminal run manifest. External experiment trackers are optional mirrors, never
the source of truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Dict, Mapping, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "run"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def dataset_structure_fingerprint(dataset_path: Path) -> Dict[str, Any]:
    """Hash dataset structure/metadata without reading every large file body."""
    dataset_path = Path(dataset_path)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0

    for path in sorted(p for p in dataset_path.rglob("*") if p.is_file()):
        rel = path.relative_to(dataset_path).as_posix()
        stat = path.stat()
        file_count += 1
        total_bytes += stat.st_size
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")

    return {
        "kind": "structure-size-mtime-sha256",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def collect_environment() -> Dict[str, Any]:
    """Capture the local runtime that materially affects a Phoenix run."""
    environment: Dict[str, Any] = {
        "captured_at": _utc_now(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "process": {
            "pid": os.getpid(),
        },
    }

    try:
        import torch

        torch_info: Dict[str, Any] = {
            "version": str(torch.__version__),
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "accelerator_available": bool(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            torch_info["device"] = {
                "index": 0,
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(props.total_memory),
            }
        environment["torch"] = torch_info
    except Exception as exc:
        environment["torch"] = {"error": str(exc)}

    try:
        import torchvision

        environment["torchvision_version"] = str(torchvision.__version__)
    except Exception as exc:
        environment["torchvision_version"] = None
        environment["torchvision_error"] = str(exc)

    try:
        import timm

        environment["timm_version"] = str(timm.__version__)
    except Exception:
        environment["timm_version"] = None

    return environment


class RunRecorder:
    """Persist one Phoenix training attempt as an immutable local run record."""

    def __init__(
        self,
        output_root: Path,
        dataset_name: str,
        dataset_path: Path,
        resolved_config: Mapping[str, Any],
        dataset_config: Mapping[str, Any],
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_id = f"{_safe_name(dataset_name)}__{stamp}"
        self.output_root = Path(output_root).expanduser().resolve()
        self.run_dir = self.output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.started_at = _utc_now()
        self.log_path = self.run_dir / "run.log"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.manifest_path = self.run_dir / "run.json"

        self.resolved_config = _json_safe(dict(resolved_config))
        self.resolved_config["output_dir"] = str(self.run_dir)
        self.resolved_config["run_id"] = self.run_id
        self.resolved_config["run_dir"] = str(self.run_dir)
        self.dataset_config = _json_safe(dict(dataset_config))
        self.dataset_path = Path(dataset_path).resolve()
        self.dataset_fingerprint = dataset_structure_fingerprint(self.dataset_path)

        self._write_json("training_config.json", self.resolved_config)
        self._write_json("dataset_manifest.json", {
            "dataset_name": dataset_name,
            "dataset_path_at_start": str(self.dataset_path),
            "dataset_config": self.dataset_config,
            "fingerprint": self.dataset_fingerprint,
        })
        self._write_json("environment.json", collect_environment())

        self._write_manifest(
            status="running",
            final_metrics=None,
            error=None,
            artifacts=None,
        )
        self.log(
            f"RUN START {self.run_id} | dataset={dataset_name} | "
            f"run_dir={self.run_dir}"
        )

    def _write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        (self.run_dir / name).write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def log(self, message: str, level: str = "INFO") -> None:
        timestamp = _utc_now()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} [{level}] {message}\n")

    def record_metrics(
        self,
        metrics: Mapping[str, Any],
        epoch: Optional[int] = None,
        progress: Optional[float] = None,
    ) -> None:
        record = {
            "timestamp": _utc_now(),
            "epoch": epoch,
            "progress": progress,
            "metrics": _json_safe(dict(metrics)),
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def write_model_config(self, trainer: Any) -> None:
        model = getattr(trainer, "model", None)
        payload = {
            "trainer_class": type(trainer).__name__,
            "model_class": type(model).__name__ if model is not None else None,
            "model_type": self.resolved_config.get("model_type"),
            "model_architecture": self.resolved_config.get("model_architecture"),
            "model_name": self.resolved_config.get("model_name"),
            "num_classes": getattr(trainer, "num_classes", None),
            "image_size": getattr(trainer, "image_size", None),
            "in_channels": getattr(trainer, "in_channels", None),
            "parameter_count": (
                sum(p.numel() for p in model.parameters())
                if model is not None and hasattr(model, "parameters")
                else None
            ),
        }
        self._write_json("model_config.json", payload)

    def artifact_inventory(self) -> Dict[str, Any]:
        """Return a compact inventory of files produced in this run directory."""
        files = []
        for path in sorted(p for p in self.run_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(self.run_dir).as_posix()
            files.append({
                "path": rel,
                "bytes": path.stat().st_size,
            })
        return {"files": files}

    def _write_manifest(
        self,
        status: str,
        final_metrics: Optional[Mapping[str, Any]],
        error: Optional[str],
        artifacts: Optional[Mapping[str, Any]],
    ) -> None:
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": status,
            "started_at": self.started_at,
            "ended_at": None if status == "running" else _utc_now(),
            "run_dir": str(self.run_dir),
            "dataset_path_at_start": str(self.dataset_path),
            "dataset_fingerprint": self.dataset_fingerprint,
            "resolved_training_config": self.resolved_config,
            "final_metrics": _json_safe(final_metrics or {}),
            "artifacts": _json_safe(dict(artifacts or {})),
            "error": error,
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def finalize(
        self,
        status: str,
        final_metrics: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
        artifacts: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.log(
            f"RUN END {self.run_id} | status={status}"
            + (f" | error={error}" if error else ""),
            "ERROR" if error else "INFO",
        )
        self._write_manifest(
            status=status,
            final_metrics=final_metrics,
            error=error,
            artifacts=artifacts,
        )

    def write_dataset_reference(self, dataset_path: Path) -> None:
        dataset_path = Path(dataset_path)
        payload = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "run_manifest": str(self.manifest_path),
            "training_config": str(self.run_dir / "training_config.json"),
            "environment": str(self.run_dir / "environment.json"),
            "metrics": str(self.metrics_path),
            "log": str(self.log_path),
        }
        (dataset_path / "training_run.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
