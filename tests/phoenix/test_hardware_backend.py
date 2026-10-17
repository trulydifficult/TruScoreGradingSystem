from pathlib import Path
from types import SimpleNamespace
import sys

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import modules.phoenix_trainer.hardware_backend as hardware_backend
from modules.phoenix_trainer.hardware_backend import (
    build_torch_runtime_environment,
    identify_torch_backend,
)


def fake_torch(*, hip=None, cuda=None):
    return SimpleNamespace(version=SimpleNamespace(hip=hip, cuda=cuda))


def test_identifies_rocm_before_cuda_spelling():
    assert identify_torch_backend(fake_torch(hip="7.2", cuda=None)) == "rocm"


def test_identifies_cuda():
    assert identify_torch_backend(fake_torch(hip=None, cuda="13.0")) == "cuda"


def test_unknown_gpu_backend_is_explicit():
    assert identify_torch_backend(fake_torch()) == "gpu"



def test_runtime_environment_prepends_torch_and_rocm_paths(monkeypatch):
    monkeypatch.setattr(
        hardware_backend,
        "torch_runtime_library_paths",
        lambda: ["/venv/torch/lib", "/opt/rocm/lib"],
    )

    env = build_torch_runtime_environment(
        {"LD_LIBRARY_PATH": "/custom/lib:/venv/torch/lib"}
    )

    assert env["LD_LIBRARY_PATH"].split(":") == [
        "/venv/torch/lib",
        "/opt/rocm/lib",
        "/custom/lib",
    ]
