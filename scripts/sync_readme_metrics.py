"""Refresh the README results block from reports/results.json and reports/eda_summary.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from garmentroute.reporting import sync


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "results.json")
    parser.add_argument("--eda", type=Path, default=ROOT / "reports" / "eda_summary.json")
    args = parser.parse_args()
    sync(args.readme, args.results, args.eda)


if __name__ == "__main__":
    main()
