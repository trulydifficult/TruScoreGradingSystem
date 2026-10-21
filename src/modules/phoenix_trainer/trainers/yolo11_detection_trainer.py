"""Real Ultralytics YOLO11s bbox-detection training adapter for Phoenix.

This is an explicitly selected YOLO11s model, not a replacement for the
Mask R-CNN instance-segmentation task. No synthetic labels or metrics.
"""
from pathlib import Path
import shutil

from .base_trainer import BaseTrainer
from ..yolo_dataset import prepare_yolo_detection_data


class YOLO11DetectionTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        if self.config.get("precision", "fp32").lower() not in {"fp32", "fp16"}:
            raise ValueError("YOLO11s adapter currently supports fp32/fp16, not BF16")
        if self.config.get("resume_from"):
            raise ValueError("YOLO11s checkpoint resume is not yet wired into Phoenix")
        if self.config.get("grad_accumulation") and int(
            self.config.get("accumulation_steps", 1)
        ) > 1:
            raise ValueError("YOLO11s gradient accumulation is not implemented")
        if self.config.get("gradient_clipping"):
            raise ValueError("YOLO11s gradient clipping is not implemented")
        if self.config.get("enable_uncertainty") or self.config.get("enable_active_learning"):
            raise ValueError("Advanced training toggles are not implemented for YOLO11s")
        if self.config.get("extra_modalities") or self.config.get("fusion_mode"):
            raise ValueError("YOLO11s supports RGB detection inputs, not multimodal fusion")
        self.model = None
        self.dataset_yaml = None
        self.train_run_dir = None

    def load_dataset(self):
        self.dataset_yaml, counts = prepare_yolo_detection_data(
            self.dataset_path, self.output_dir, seed=self.seed,
        )
        self.log(
            f"Verified YOLO11s dataset: {counts['train_images']} training, "
            f"{counts['val_images']} validation images, classes={counts['class_names']}",
            "INFO",
        )
        return counts

    def build_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO11s training requires ultralytics in the Phoenix Python environment"
            ) from exc
        weight = self.config.get("model_name", "yolo11s.pt")
        if weight != "yolo11s.pt":
            raise ValueError(f"Only the explicitly supported yolo11s.pt model is accepted: {weight}")
        self.model = YOLO(weight)
        return self.model

    def train(self):
        self.load_dataset()
        self.build_model()
        epochs = int(self.config["epochs"])
        batch = int(self.config["batch_size"])
        if epochs <= 0 or batch <= 0:
            raise ValueError("Epochs and batch size must be positive")
        device = str(self.config.get("device", "cpu")).lower()
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Requested ROCm/CUDA device is unavailable")
            device = 0  # Ultralytics GPU 0; torch CUDA APIs also name ROCm devices.
        elif device != "cpu":
            raise ValueError(f"Unsupported YOLO11s device: {device}")

        def on_epoch_start(engine):
            self.current_epoch = int(engine.epoch)
            callback = self.callbacks.get("on_epoch_start")
            if callback:
                callback(self.current_epoch)

        def on_fit_epoch_end(engine):
            self.current_epoch = int(engine.epoch)
            metrics = {}
            for k, v in (getattr(engine, "metrics", None) or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics[str(k)] = float(v)
            # Alias only the actual Ultralytics mAP50-95 measurement.
            # No derived/heuristic accuracy estimates may be emitted.
            if "metrics/mAP50-95(B)" in metrics:
                metrics["mAP"] = metrics["metrics/mAP50-95(B)"]
            if metrics:
                self.update_metrics(metrics)
            self.update_progress(self.current_epoch + 1, epochs,
                                 min(1.0, (self.current_epoch+1)/epochs))

        self.model.add_callback("on_train_epoch_start", on_epoch_start)
        self.model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        self.train_run_dir = self.output_dir / "yolo11s_detection"
        if self.train_run_dir.exists():
            raise FileExistsError(f"Existing YOLO run would be overwritten: {self.train_run_dir}")

        # Ultralytics is the real optimizer/evaluation/checkpoint owner.
        # Disable framework auto-fallback: a missing model/dependency raises.
        self.model.train(
            data=str(self.dataset_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=int(self.config.get("image_size", 640)),
            lr0=float(self.config["learning_rate"]),
            optimizer=str(self.config["optimizer"]),
            project=str(self.output_dir),
            name="yolo11s_detection",
            exist_ok=False,
            device=device,
            seed=self.seed,
            deterministic=self.deterministic,
            amp=self.config.get("precision", "fp32").lower() == "fp16",
            patience=int(self.config.get("early_stopping_patience", 20)),
            workers=int(self.config.get("num_workers", 2)),
            verbose=False,
        )
        best = self.train_run_dir / "weights" / "best.pt"
        last = self.train_run_dir / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError(
                "Ultralytics returned without required real best.pt/last.pt checkpoints"
            )
        self.log(f"YOLO11s training completed with checkpoints: {best}", "SUCCESS")
        return best

    def train_epoch(self):
        raise RuntimeError("YOLO11s uses the native Ultralytics training loop; call train()")

    def validate(self):
        if self.model is None or self.dataset_yaml is None:
            raise RuntimeError("YOLO11s model/dataset not initialized")
        return self.model.val(data=str(self.dataset_yaml))

    def save_checkpoint(self, filepath):
        best = (self.train_run_dir / "weights" / "best.pt"
                if self.train_run_dir is not None else None)
        if best is None or not best.is_file():
            raise RuntimeError("No real YOLO11s checkpoint exists yet")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, filepath)
