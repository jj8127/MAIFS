#!/usr/bin/env python3
"""루트 inference_rpi5.py를 재사용하는 RPi5 진입점."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_rpi5 import main


if __name__ == "__main__":
    main()
