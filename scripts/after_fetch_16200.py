"""When the 16200 fetch finishes: resize new members, then run expanded dual eval."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _count_present(members: Path, raw: Path) -> tuple[int, int]:
    names = [line.strip() for line in members.read_text(encoding="utf-8").splitlines() if line.strip()]
    have = sum(1 for name in names if (raw / name).is_file() and (raw / name).stat().st_size > 0)
    return have, len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, default=ROOT / "data" / "subset_members_16200.txt")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--min-present-fraction", type=float, default=0.995)
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    if not args.skip_wait:
        while True:
            have, total = _count_present(args.members, args.raw)
            frac = have / total if total else 0.0
            print(f"fetch progress {have}/{total} ({frac:.1%})", flush=True)
            if total and frac >= args.min_present_fraction:
                break
            time.sleep(args.poll_seconds)

    print("resize", flush=True)
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "resize_images.py"),
            "--members",
            str(args.members),
            "--src",
            str(args.raw),
            "--dst",
            str(ROOT / "data" / "resized"),
        ],
        cwd=str(ROOT),
    )
    print("expanded dual eval", flush=True)
    subprocess.check_call(
        [sys.executable, "-u", str(ROOT / "scripts" / "eval_expanded_dual.py"), "--members", str(args.members)],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    main()
