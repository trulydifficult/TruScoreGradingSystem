"""Shared Phoenix training runtime primitives.

This module owns hardware-neutral training-loop mechanics used by conventional
PyTorch trainers. PyTorch still uses the "cuda" API/device spelling for ROCm.
Detectron2 is intentionally not forced through this runtime until its lifecycle
adapter is repaired.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
import random
from typing import Any, Dict, Mapping, Optional, Tuple


SUPPORTED_PRECISIONS = {"fp32", "fp16", "bf16"}


def normalize_precision_mode(config: Mapping[str, Any]) -> str:
    """Resolve the requested precision while preserving the legacy AMP toggle."""
    explicit = config.get("precision")
    if explicit is None:
        explicit = "fp16" if config.get("mixed_precision", False) else "fp32"

    mode = str(explicit).strip().lower()
    aliases = {
        "float32": "fp32",
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "amp": "fp16",
    }
    mode = aliases.get(mode, mode)

    if mode not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported precision '{explicit}'. "
            f"Expected one of: {sorted(SUPPORTED_PRECISIONS)}"
        )
    return mode


def accumulation_divisor(
    batch_index: int,
    total_batches: int,
    accumulation_steps: int,
) -> int:
    """Return the correct loss divisor, including a final partial group."""
    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    if batch_index < 0 or batch_index >= total_batches:
        raise ValueError("batch_index is outside the epoch")
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")

    remainder = total_batches % accumulation_steps
    partial_start = total_batches - remainder

    if remainder and batch_index >= partial_start:
        return remainder
    return accumulation_steps


def should_optimizer_step(
    batch_index: int,
    total_batches: int,
    accumulation_steps: int,
) -> bool:
    """Return True at accumulation boundaries and at the final partial batch."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    return (
        (batch_index + 1) % accumulation_steps == 0
        or batch_index + 1 == total_batches
    )


def infer_metric_mode(metric_name: str) -> str:
    """Infer whether a validation metric should be minimized or maximized."""
    name = metric_name.strip().lower()
    if "loss" in name or name in {"mae", "mse", "rmse", "error"}:
        return "min"
    return "max"


def select_monitor_metric(
    val_metrics: Mapping[str, float],
    config: Mapping[str, Any],
) -> Tuple[str, float, str]:
    """Select the validation metric and comparison direction for best-model logic."""
    requested = config.get("monitor_metric")
    if requested:
        metric_name = str(requested)
        if metric_name not in val_metrics:
            raise KeyError(
                f"Configured monitor_metric '{metric_name}' was not returned by "
                f"validation. Available: {sorted(val_metrics)}"
            )
    else:
        metric_name = next(
            (
                name
                for name in ("mAP", "accuracy", "IoU", "val_loss")
                if name in val_metrics
            ),
            "",
        )
        if not metric_name:
            if not val_metrics:
                raise ValueError("Validation returned no metrics to monitor")
            metric_name = next(iter(val_metrics))

    mode = str(config.get("monitor_mode") or infer_metric_mode(metric_name)).lower()
    if mode not in {"min", "max"}:
        raise ValueError("monitor_mode must be 'min' or 'max'")

    return metric_name, float(val_metrics[metric_name]), mode


def metric_improved(
    candidate: float,
    best: Optional[float],
    mode: str,
) -> bool:
    """Compare a candidate validation metric against the current best."""
    if best is None:
        return True
    if mode == "max":
        return candidate > best
    if mode == "min":
        return candidate < best
    raise ValueError("mode must be 'min' or 'max'")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible Phoenix runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a PyTorch DataLoader worker."""
    del worker_id
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader_generator(seed: int):
    """Create a deterministic PyTorch generator for DataLoader shuffling."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def capture_rng_state() -> Dict[str, Any]:
    """Capture RNG state required for a faithful local checkpoint resume."""
    import numpy as np
    import torch

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state previously captured by capture_rng_state."""
    if not state:
        return

    import numpy as np
    import torch

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass
class TrainingRuntimeState:
    precision: str
    accumulation_steps: int
    gradient_clipping: bool
    grad_clip_norm: float


class TrainingRuntime:
    """Shared per-step mechanics for ordinary PyTorch Phoenix trainers."""

    def __init__(
        self,
        config: Mapping[str, Any],
        device: Any,
        optimizer: Any,
    ) -> None:
        import torch

        self.config = config
        self.device = device
        self.optimizer = optimizer
        self.precision = normalize_precision_mode(config)

        accumulation_enabled = bool(config.get("grad_accumulation", False))
        requested_steps = int(config.get("accumulation_steps", 1))
        self.accumulation_steps = requested_steps if accumulation_enabled else 1
        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be at least 1")

        self.gradient_clipping = bool(config.get("gradient_clipping", False))
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
        if self.gradient_clipping and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")

        self.device_type = getattr(device, "type", str(device).split(":")[0])
        if self.precision != "fp32" and self.device_type != "cuda":
            raise RuntimeError(
                f"{self.precision} training currently requires a PyTorch GPU "
                "device; use fp32 for CPU training"
            )

        self.autocast_dtype = None
        self.scaler = None

        if self.precision == "fp16":
            self.autocast_dtype = torch.float16
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            except (AttributeError, TypeError):
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        elif self.precision == "bf16":
            self.autocast_dtype = torch.bfloat16

    def describe(self) -> TrainingRuntimeState:
        return TrainingRuntimeState(
            precision=self.precision,
            accumulation_steps=self.accumulation_steps,
            gradient_clipping=self.gradient_clipping,
            grad_clip_norm=self.grad_clip_norm,
        )

    def autocast(self):
        """Return the correct autocast context for this run."""
        import torch

        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=True,
        )

    def begin_epoch(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def backward_and_step(
        self,
        loss: Any,
        model: Any,
        batch_index: int,
        total_batches: int,
    ) -> bool:
        """Backpropagate one batch and step when the accumulation group is ready."""
        import torch

        if not bool(torch.isfinite(loss).all().item()):
            raise FloatingPointError(
                f"Non-finite training loss at batch {batch_index}: "
                f"{loss.detach().item()}"
            )

        divisor = accumulation_divisor(
            batch_index,
            total_batches,
            self.accumulation_steps,
        )
        scaled_loss = loss / divisor

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if not should_optimizer_step(
            batch_index,
            total_batches,
            self.accumulation_steps,
        ):
            return False

        if self.scaler is not None:
            if self.gradient_clipping:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.grad_clip_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.grad_clip_norm,
                )
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        return True

    def state_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "accumulation_steps": self.accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            "grad_clip_norm": self.grad_clip_norm,
            "scaler_state_dict": (
                self.scaler.state_dict() if self.scaler is not None else None
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore scaler state only when runtime-critical settings match."""
        expected = self.state_dict()
        for key in (
            "precision",
            "accumulation_steps",
            "gradient_clipping",
            "grad_clip_norm",
        ):
            if key in state and state[key] != expected[key]:
                raise RuntimeError(
                    f"Checkpoint runtime mismatch for {key}: "
                    f"checkpoint={state[key]!r}, current={expected[key]!r}"
                )

        scaler_state = state.get("scaler_state_dict")
        if scaler_state is not None:
            if self.scaler is None:
                raise RuntimeError(
                    "Checkpoint contains FP16 scaler state but the current "
                    "precision policy does not use a scaler"
                )
            self.scaler.load_state_dict(scaler_state)
