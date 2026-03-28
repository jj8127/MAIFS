#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


CANONICAL_PATH = Path(__file__).resolve().parents[3] / "common" / "scripts" / "rpi5_infer_cpu.py"
SPEC = importlib.util.spec_from_file_location("rpi5_infer_cpu_canonical", CANONICAL_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

ShieldCPU = MODULE.ShieldCPU
load_image = MODULE.load_image
build_parser = MODULE.build_parser
main = MODULE.main


if __name__ == "__main__":
    main()
