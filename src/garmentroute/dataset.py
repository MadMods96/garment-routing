"""Two-view dataset. One row is one garment, never one photograph."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from garmentroute.labels import TERNARY_TO_INDEX, USAGE_CLASS_TO_INDEX
from garmentroute.views import resolve_views

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> T.Compose:
    """Mild augmentations. Colour jitter stays small because stains are colour cues.

    Vertical flips are not used. A garment has an up.
    """
    ops: list = [T.Resize(image_size)]
    if train:
        ops.extend(
            [
                T.RandomHorizontalFlip(),
                T.RandomRotation(8),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            ]
        )
    ops.extend(
        [
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return T.Compose(ops)


class TwoViewDataset(Dataset):
    """Load the front and back image of each garment already resized on disk.

    A garment with one file missing stays in the set. The photograph that
    exists is used for both inputs, and the missing view is written to
    ``view_log`` when that path is set.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: Path,
        image_size: int,
        train: bool,
        view_log: Path | None = None,
        available: set[str] | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.frame, self.view_log = resolve_views(frame, self.image_root, available)
        self.frame = self.frame.reset_index(drop=True)
        self._write_view_log(view_log)
        self.image_size = image_size
        self.transform = build_transforms(image_size, train)

    def _write_view_log(self, path: Path | None) -> None:
        if not self.view_log:
            return
        single = [row for row in self.view_log if row["fallback"] == "single_view"]
        excluded = [row for row in self.view_log if row["fallback"] == "excluded"]
        front_only = sum(row["used_view"] == "front" for row in single)
        back_only = sum(row["used_view"] == "back" for row in single)
        print(
            f"single-view fallback: {len(single)} garments "
            f"(front only {front_only}, back only {back_only}); "
            f"excluded {len(excluded)} with neither image",
            flush=True,
        )
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in self.view_log), encoding="utf-8")

    def __len__(self) -> int:
        return len(self.frame)

    def _load(self, relative: str) -> torch.Tensor:
        path = self.image_root / relative
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return self.transform(rgb)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.frame.iloc[index]
        view = str(row["view"])
        if view == "back":
            back = self._load(str(row["back_path"]))
            front = back
        else:
            front = self._load(str(row["front_path"]))
            back = front if view == "front" else self._load(str(row["back_path"]))
        item: dict[str, torch.Tensor | str | int] = {
            "front": front,
            "back": back,
            "view": view,
            "item_id": str(row["item_id"]),
        }
        for column in ("condition", "pilling"):
            value = row[column]
            present = pd.notna(value)
            item[column] = int(value) if present else 0
            item[f"{column}_mask"] = 1.0 if present else 0.0
        stains = row["stains_3class"]
        item["stains"] = TERNARY_TO_INDEX[stains] if isinstance(stains, str) else -100
        holes = row["holes"]
        item["holes"] = TERNARY_TO_INDEX[holes] if isinstance(holes, str) else -100
        usage = row["usage_class"]
        item["usage"] = USAGE_CLASS_TO_INDEX[usage] if isinstance(usage, str) else -100
        return item
