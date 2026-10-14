#!/usr/bin/env python3
"""Verify the Phoenix PyTorch accelerator backend in the active environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.phoenix_trainer.hardware_backend import probe_torch_accelerator


def main() -> int:
    probe = probe_torch_accelerator()
    print(json.dumps(probe.to_dict(), indent=2, sort_keys=True))

    if not probe.ready:
        print("\nPHOENIX BACKEND: NOT READY", file=sys.stderr)
        return 1

    print("\nPHOENIX BACKEND: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
