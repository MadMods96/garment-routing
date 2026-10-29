"""Shared helpers that do not depend on the vision stack."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML document into a dict."""
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return loaded


def write_json(path: Path | str, payload: Any) -> None:
    """Write JSON with stable key order and UTF-8."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when PyTorch is installed."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_missing(value: Any) -> bool:
    """True for None and NaN, false for other values including 0 and empty checks elsewhere."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    try:
        if value is np.nan:
            return True
    except Exception:  # noqa: BLE001
        return False
    return False
