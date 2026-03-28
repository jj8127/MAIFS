#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


OUT_PATH = Path(__file__).resolve().parents[2] / "assets" / "benchmark_input.png"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (224, 224), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle((16, 16, 208, 208), outline=(80, 90, 110), width=3)
    draw.rectangle((24, 24, 200, 120), fill=(214, 228, 255))
    draw.polygon([(30, 180), (90, 110), (140, 160), (195, 90), (205, 180)], fill=(255, 225, 183))
    draw.ellipse((150, 35, 185, 70), fill=(255, 166, 87))
    img.save(OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
