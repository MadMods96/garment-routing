"""Resize front and back images so the short side is 384 px and save JPEG quality 90."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def resize_image(source: Path, destination: Path, short_side: int, quality: int) -> None:
    """Write one JPEG. The short side becomes ``short_side`` pixels."""
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = short_side / min(width, height)
        resized = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        resized.save(destination, format="JPEG", quality=quality, optimize=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw"))
    parser.add_argument("--dst", type=Path, default=Path("data/resized"))
    parser.add_argument("--members", type=Path, default=Path("data/subset_members.txt"))
    parser.add_argument("--short-side", type=int, default=384)
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args(argv)
    names = [line.strip() for line in args.members.read_text(encoding="utf-8").splitlines() if line.strip()]
    written = 0
    missing = 0
    for name in names:
        source = args.src / name
        destination = args.dst / name
        if not source.is_file():
            missing += 1
            continue
        if destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime:
            continue
        resize_image(source, destination, args.short_side, args.quality)
        written += 1
    print(f"resized {written} missing {missing}")


if __name__ == "__main__":
    main()
