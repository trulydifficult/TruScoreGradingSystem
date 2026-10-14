"""Phoenix hardware/backend capability probing.

PyTorch intentionally exposes ROCm through the torch.cuda Python API. Phoenix
therefore keeps PyTorch's "cuda" device spelling while reporting the actual
backend (ROCm, CUDA, CPU) separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AcceleratorProbe:
    ready: bool
    backend: str
    device: str
    torch_version: Optional[str] = None
    hip_version: Optional[str] = None
    cuda_version: Optional[str] = None
    name: Optional[str] = None
    total_memory_gb: Optional[float] = None
    fp32_compute: bool = False
    fp16_compute: bool = False
    bf16_compute: bool = False
    backward_step: bool = False
    precision_errors: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def identify_torch_backend(torch_module: Any) -> str:
    """Return the actual accelerator family behind PyTorch's cuda interface."""
    version = getattr(torch_module, "version", None)
    if getattr(version, "hip", None):
        return "rocm"
    if getattr(version, "cuda", None):
        return "cuda"
    return "gpu"


def _compute_probe(torch_module: Any, device: Any, dtype: Any) -> None:
    """Execute a small real matrix operation and reject non-finite output."""
    a = torch_module.randn((128, 128), device=device, dtype=dtype)
    b = torch_module.randn((128, 128), device=device, dtype=dtype)
    result = a @ b
    if not bool(torch_module.isfinite(result).all().item()):
        raise RuntimeError("accelerator produced non-finite matrix output")
    torch_module.cuda.synchronize(device)


def _backward_probe(torch_module: Any, device: Any) -> None:
    """Verify autograd and an optimizer step on the selected accelerator."""
    model = torch_module.nn.Linear(32, 16).to(device)
    optimizer = torch_module.optim.SGD(model.parameters(), lr=1e-3)
    inputs = torch_module.randn((8, 32), device=device)

    optimizer.zero_grad(set_to_none=True)
    loss = model(inputs).square().mean()
    loss.backward()

    for parameter in model.parameters():
        if parameter.grad is not None and not bool(
            torch_module.isfinite(parameter.grad).all().item()
        ):
            raise RuntimeError("accelerator produced non-finite gradients")

    optimizer.step()
    torch_module.cuda.synchronize(device)


def probe_torch_accelerator(device_index: int = 0) -> AcceleratorProbe:
    """Probe a real PyTorch GPU path rather than trusting is_available() alone."""

    try:
        import torch
    except Exception as exc:
        return AcceleratorProbe(
            ready=False,
            backend="unavailable",
            device="cpu",
            error=f"PyTorch import failed: {exc}",
        )

    torch_version = str(getattr(torch, "__version__", "unknown"))
    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)

    try:
        if not torch.cuda.is_available():
            return AcceleratorProbe(
                ready=False,
                backend="cpu",
                device="cpu",
                torch_version=torch_version,
                hip_version=hip_version,
                cuda_version=cuda_version,
                error="PyTorch reports no GPU accelerator.",
            )

        backend = identify_torch_backend(torch)
        device = torch.device(f"cuda:{device_index}")
        name = torch.cuda.get_device_name(device_index)
        properties = torch.cuda.get_device_properties(device_index)
        total_memory_gb = round(properties.total_memory / (1024 ** 3), 2)

        probe = AcceleratorProbe(
            ready=False,
            backend=backend,
            device=str(device),
            torch_version=torch_version,
            hip_version=str(hip_version) if hip_version else None,
            cuda_version=str(cuda_version) if cuda_version else None,
            name=name,
            total_memory_gb=total_memory_gb,
        )

        # FP32 and backward/optimizer behavior are mandatory for a healthy backend.
        _compute_probe(torch, device, torch.float32)
        probe.fp32_compute = True

        _backward_probe(torch, device)
        probe.backward_step = True

        # Reduced precision is capability information, not a requirement for basic
        # readiness. Individual run configurations decide what precision they need.
        for label, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            try:
                _compute_probe(torch, device, dtype)
                setattr(probe, f"{label}_compute", True)
            except Exception as exc:
                probe.precision_errors[label] = str(exc)

        probe.ready = probe.fp32_compute and probe.backward_step
        return probe

    except Exception as exc:
        return AcceleratorProbe(
            ready=False,
            backend=(
                "rocm"
                if hip_version
                else "cuda"
                if cuda_version
                else "gpu"
            ),
            device=f"cuda:{device_index}",
            torch_version=torch_version,
            hip_version=str(hip_version) if hip_version else None,
            cuda_version=str(cuda_version) if cuda_version else None,
            error=str(exc),
        )
