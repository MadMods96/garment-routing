"""Train the multi-task model. Checkpoints land in ``runs/`` and are not for git."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from garmentroute.dataset import TwoViewDataset
from garmentroute.labels import TERNARY, USAGE_CLASSES, balanced_class_weights
from garmentroute.model import (
    GarmentNet,
    coral_decode,
    masked_coral_loss,
    masked_cross_entropy,
    masked_focal_cross_entropy,
)
from garmentroute.util import load_yaml, set_seed, write_json
from garmentroute.views import filter_to_subset, find_subset_members, list_image_relatives

NOMINAL = {"stains": TERNARY, "holes": TERNARY, "usage": USAGE_CLASSES}
ORDINAL = ("condition", "pilling")
HEAD_ORDER = ("usage", "condition", "pilling", "stains", "holes")
DEFAULT_LOSS_WEIGHTS = {
    "usage": 1.0,
    "condition": 0.5,
    "pilling": 0.5,
    "stains": 0.5,
    "holes": 0.5,
}


def _class_weights(
    frame: pd.DataFrame,
    column: str,
    classes: tuple[Any, ...],
    cap: float,
) -> torch.Tensor:
    typed = frame[column].dropna().astype(type(classes[0]))
    counts = {type(classes[0])(key): int(value) for key, value in typed.value_counts().items()}
    return torch.tensor(balanced_class_weights(counts, classes, cap), dtype=torch.float32)


def _ordinal_example_weight(labels: torch.Tensor, mask: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    index = (labels.clamp(min=1, max=table.shape[0]) - 1).long()
    weights = table[index]
    weights = torch.where(mask > 0, weights, torch.ones_like(weights))
    return weights


def _ordinal_table(frame: pd.DataFrame, column: str, cap: float) -> torch.Tensor:
    return _class_weights(frame, column, (1, 2, 3, 4, 5), cap)


def _loss(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    nominal_weights: dict[str, torch.Tensor],
    ordinal_tables: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
    usage_only: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parts: dict[str, torch.Tensor] = {}
    if not usage_only:
        for name in ORDINAL:
            labels = batch[name]
            mask = batch[f"{name}_mask"]
            example_weight = _ordinal_example_weight(labels, mask, ordinal_tables[name])
            parts[name] = float(loss_weights.get(name, DEFAULT_LOSS_WEIGHTS[name])) * masked_coral_loss(
                outputs[name], labels, mask, example_weight
            )
        parts["stains"] = float(loss_weights.get("stains", DEFAULT_LOSS_WEIGHTS["stains"])) * masked_cross_entropy(
            outputs["stains"], batch["stains"], nominal_weights["stains"]
        )
        holes_gamma = float(loss_weights.get("holes_focal_gamma") or 0.0)
        if holes_gamma > 0:
            parts["holes"] = float(loss_weights.get("holes", DEFAULT_LOSS_WEIGHTS["holes"])) * masked_focal_cross_entropy(
                outputs["holes"],
                batch["holes"],
                nominal_weights["holes"],
                gamma=holes_gamma,
            )
        else:
            parts["holes"] = float(loss_weights.get("holes", DEFAULT_LOSS_WEIGHTS["holes"])) * masked_cross_entropy(
                outputs["holes"], batch["holes"], nominal_weights["holes"]
            )
    parts["usage"] = float(loss_weights.get("usage", DEFAULT_LOSS_WEIGHTS["usage"])) * masked_cross_entropy(
        outputs["usage"], batch["usage"], nominal_weights["usage"]
    )
    total = outputs["usage"].sum() * 0
    for name in HEAD_ORDER:
        if name in parts:
            total = total + parts[name]
    return total, parts


def _limit_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(frame):
        return frame
    labels = frame["usage_class"] if "usage_class" in frame.columns else None
    if labels is None or labels.nunique(dropna=True) < 2:
        return frame.sample(n=n, random_state=seed)
    try:
        from sklearn.model_selection import train_test_split

        limited, _ = train_test_split(frame, train_size=n, stratify=labels, random_state=seed)
        return limited
    except ValueError:
        return frame.sample(n=n, random_state=seed)


def _pred_count_text(pred: list[int]) -> str:
    counts = [0] * len(USAGE_CLASSES)
    for value in pred:
        if 0 <= int(value) < len(USAGE_CLASSES):
            counts[int(value)] += 1
    return " ".join(f"{name}={counts[index]}" for index, name in enumerate(USAGE_CLASSES))


def _head_loss_text(parts: dict[str, float]) -> str:
    return " ".join(f"{name}={parts[name]:.4f}" for name in HEAD_ORDER if name in parts)


def _move(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _usage_macro_f1(outputs: dict[str, torch.Tensor], batch: dict) -> tuple[list[int], list[int]]:
    target = batch["usage"]
    keep = target != -100
    if int(keep.sum().item()) == 0:
        return [], []
    pred = outputs["usage"].argmax(dim=-1)
    return target[keep].detach().cpu().tolist(), pred[keep].detach().cpu().tolist()


def _condition_mae(outputs: dict[str, torch.Tensor], batch: dict) -> tuple[float, int]:
    mask = batch["condition_mask"] > 0
    count = int(mask.sum().item())
    if count == 0:
        return 0.0, 0
    pred = coral_decode(outputs["condition"]).to(dtype=torch.float32)
    truth = batch["condition"].to(dtype=torch.float32)
    error = (pred[mask] - truth[mask]).abs().sum().item()
    return float(error), count


def _loader_kwargs(config: dict, device: torch.device, shuffle: bool) -> dict:
    workers = int(config.get("num_workers", 4))
    options: dict = {
        "batch_size": int(config.get("batch_size", 32)),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return options


def train(config: dict, config_path: Path) -> Path:
    """Train and return the path of the best checkpoint, chosen on validation macro-F1."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    index_path = Path(config["index"])
    frame = pd.read_parquet(index_path)
    members = find_subset_members(
        Path(config["subset_members"]) if config.get("subset_members") else None,
        index_path.parent,
        Path(config_path).resolve().parent.parent,
        Path("/kaggle/working/package"),
    )
    if members is not None:
        before = len(frame)
        frame = filter_to_subset(frame, members)
        print(f"subset {members} rows {before} -> {len(frame)}", flush=True)
    image_root = Path(config["image_root"])
    image_size = int(config.get("image_size", 224))
    run_name = config.get("run_name") or f"{config.get('backbone', 'model')}-{config.get('fusion', 'concat')}"
    run_dir = Path(config.get("run_dir", "runs")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    available = list_image_relatives(image_root)
    print(f"images on disk {len(available)}", flush=True)

    def _with_images(split_frame: pd.DataFrame) -> pd.DataFrame:
        if split_frame.empty:
            return split_frame
        front = split_frame["front_path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
        back = split_frame["back_path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
        keep = front.isin(available) | back.isin(available)
        return split_frame.loc[keep].copy()

    train_rows = _with_images(frame[frame["split"] == "train"])
    val_rows = _with_images(frame[frame["split"] == "val"])
    test_rows = _with_images(frame[frame["split"] == "test"])
    print(
        f"rows with images train {len(train_rows)} val {len(val_rows)} test {len(test_rows)}",
        flush=True,
    )
    max_items = config.get("max_items")
    if max_items is not None:
        # Sample only among garments that already have a front or back on disk.
        train_rows = _limit_rows(train_rows, int(max_items), seed)
        val_rows = _limit_rows(val_rows, max(8, min(len(val_rows), int(max_items) // 2)), seed)
        test_rows = _limit_rows(test_rows, max(8, min(len(test_rows), int(max_items) // 2)), seed)
        print(
            f"max_items {max_items}: sampled train {len(train_rows)} val {len(val_rows)} test {len(test_rows)}",
            flush=True,
        )
    use_augment = bool(config.get("augment", True))
    train_set = TwoViewDataset(
        train_rows,
        image_root,
        image_size,
        train=use_augment,
        view_log=run_dir / "single_view_train.jsonl",
        available=available,
    )
    val_set = TwoViewDataset(
        val_rows,
        image_root,
        image_size,
        train=False,
        view_log=run_dir / "single_view_val.jsonl",
        available=available,
    )
    test_set = TwoViewDataset(test_rows, image_root, image_size, train=False, available=available)
    train_frame = train_set.frame
    val_frame = val_set.frame
    if len(train_frame) == 0 or len(val_frame) == 0:
        raise RuntimeError("train and val splits need a front or back image on disk")
    train_loader = DataLoader(train_set, **_loader_kwargs(config, device, True))
    val_loader = DataLoader(val_set, **_loader_kwargs(config, device, False))
    single_view = sum(1 for row in train_set.view_log if row["fallback"] == "single_view")
    single_view += sum(1 for row in val_set.view_log if row["fallback"] == "single_view")
    print(
        f"samples train {len(train_frame)} val {len(val_frame)} test {len(test_set)} "
        f"batches {len(train_loader)} batch_size {int(config.get('batch_size', 32))} "
        f"image_size {image_size} device {device} single_view {single_view}",
        flush=True,
    )
    print("loading model", flush=True)
    model = GarmentNet(
        backbone=config.get("backbone", "convnext_tiny"),
        pretrained=bool(config.get("pretrained", True)),
        fusion=config.get("fusion", "concat"),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)
    print("model ready", flush=True)
    usage_only = bool(config.get("usage_only", False))
    disable_class_weights = bool(config.get("disable_class_weights", False))
    weight_cap = float(config.get("class_weight_cap", 5.0))
    holes_cap = float(config.get("holes_class_weight_cap", weight_cap))
    if disable_class_weights:
        nominal_weights = {
            "stains": torch.ones(len(TERNARY), dtype=torch.float32, device=device),
            "holes": torch.ones(len(TERNARY), dtype=torch.float32, device=device),
            "usage": torch.ones(len(USAGE_CLASSES), dtype=torch.float32, device=device),
        }
        ordinal_tables = {
            name: torch.ones(5, dtype=torch.float32, device=device) for name in ORDINAL
        }
    else:
        nominal_weights = {
            "stains": _class_weights(train_frame, "stains_3class", TERNARY, weight_cap).to(device),
            "holes": _class_weights(train_frame, "holes", TERNARY, holes_cap).to(device),
            "usage": _class_weights(train_frame, "usage_class", USAGE_CLASSES, weight_cap).to(device),
        }
        ordinal_tables = {name: _ordinal_table(train_frame, name, weight_cap).to(device) for name in ORDINAL}
    usage_counts = train_frame["usage_class"].dropna().astype(str).value_counts().to_dict()
    usage_weight_list = nominal_weights["usage"].detach().cpu().tolist()
    print(
        "usage train counts "
        + " ".join(f"{name}={int(usage_counts.get(name, 0))}" for name in USAGE_CLASSES),
        flush=True,
    )
    print(
        "usage class weights "
        + " ".join(f"{name}={weight:.3f}" for name, weight in zip(USAGE_CLASSES, usage_weight_list)),
        flush=True,
    )
    backbone_lr = float(config.get("backbone_lr", config.get("lr", 1e-4)))
    head_lr = float(config.get("head_lr", 1e-3))
    backbone_ids = {id(param) for param in model.backbone.parameters()}
    head_params = [param for param in model.parameters() if id(param) not in backbone_ids]
    optimiser = torch.optim.AdamW(
        [
            {"params": list(model.backbone.parameters()), "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=float(config.get("weight_decay", 0.05)),
    )
    epochs = int(config.get("epochs", 8))
    warmup = int(config.get("warmup_epochs", 1))
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history: list[dict] = []
    best_f1 = -1.0
    stale = 0
    patience = int(config.get("early_stopping_patience", 3))
    progress_every = int(config.get("progress_every", 50))
    best_path = run_dir / "best.pt"
    loss_weights = {**DEFAULT_LOSS_WEIGHTS, **(config.get("loss_weights") or {})}
    if config.get("holes_focal_gamma") is not None:
        loss_weights["holes_focal_gamma"] = float(config["holes_focal_gamma"])
    steps_per_epoch = max(1, len(train_loader))
    global_step = 0
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup
    group_lrs = (backbone_lr, head_lr)

    constant_lr = bool(config.get("constant_lr", False))

    def set_lr(step: int) -> None:
        if constant_lr:
            for group, base_lr in zip(optimiser.param_groups, group_lrs):
                group["lr"] = base_lr
            return
        if step < warmup_steps:
            scale = (step + 1) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for group, base_lr in zip(optimiser.param_groups, group_lrs):
            group["lr"] = base_lr * scale

    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        train_heads: dict[str, float] = {}
        train_pred: list[int] = []
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, start=1):
            set_lr(global_step)
            batch = _move(batch, device)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(batch["front"], batch["back"])
                loss, parts = _loss(
                    outputs, batch, nominal_weights, ordinal_tables, loss_weights, usage_only
                )
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            batch_n = int(batch["front"].shape[0])
            running += float(loss.item()) * batch_n
            seen += batch_n
            for name, value in parts.items():
                train_heads[name] = train_heads.get(name, 0.0) + float(value.item()) * batch_n
            global_step += 1
            if batch_index % progress_every == 0 or batch_index == steps_per_epoch:
                print(
                    f"epoch {epoch} batch {batch_index}/{steps_per_epoch} "
                    f"loss {float(loss.item()):.4f} elapsed {time.perf_counter() - epoch_start:.1f}s",
                    flush=True,
                )
        model.eval()
        truth: list[int] = []
        pred: list[int] = []
        val_loss = 0.0
        val_seen = 0
        mae_total = 0.0
        mae_count = 0
        val_heads: dict[str, float] = {}
        with torch.no_grad():
            for batch in val_loader:
                batch = _move(batch, device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    outputs = model(batch["front"], batch["back"])
                    loss, parts = _loss(
                        outputs, batch, nominal_weights, ordinal_tables, loss_weights, usage_only
                    )
                batch_n = int(batch["front"].shape[0])
                val_loss += float(loss.item()) * batch_n
                val_seen += batch_n
                for name, value in parts.items():
                    val_heads[name] = val_heads.get(name, 0.0) + float(value.item()) * batch_n
                batch_truth, batch_pred = _usage_macro_f1(outputs, batch)
                truth.extend(batch_truth)
                pred.extend(batch_pred)
                error, count = _condition_mae(outputs, batch)
                mae_total += error
                mae_count += count
        macro_f1 = float(f1_score(truth, pred, average="macro", zero_division=0)) if truth else 0.0
        train_macro_f1 = None
        if usage_only:
            eval_truth: list[int] = []
            eval_pred: list[int] = []
            with torch.no_grad():
                for batch in train_loader:
                    batch = _move(batch, device)
                    with torch.amp.autocast(device.type, enabled=use_amp):
                        outputs = model(batch["front"], batch["back"])
                    batch_truth, batch_pred = _usage_macro_f1(outputs, batch)
                    eval_truth.extend(batch_truth)
                    eval_pred.extend(batch_pred)
            if eval_truth:
                train_macro_f1 = float(f1_score(eval_truth, eval_pred, average="macro", zero_division=0))
                train_pred = eval_pred
        condition_mae = mae_total / max(mae_count, 1)
        epoch_time = time.perf_counter() - epoch_start
        train_head_means = {name: train_heads[name] / max(seen, 1) for name in train_heads}
        val_head_means = {name: val_heads[name] / max(val_seen, 1) for name in val_heads}
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_loss": val_loss / max(val_seen, 1),
            "val_usage_macro_f1": macro_f1,
            "train_usage_macro_f1": train_macro_f1,
            "val_condition_mae": condition_mae,
            "val_usage_pred": {name: pred.count(index) for index, name in enumerate(USAGE_CLASSES)},
            "train_heads": train_head_means,
            "val_heads": val_head_means,
            "epoch_time": epoch_time,
            "lr_backbone": optimiser.param_groups[0]["lr"],
            "lr_head": optimiser.param_groups[1]["lr"] if len(optimiser.param_groups) > 1 else None,
        }
        history.append(row)
        print(
            f"epoch {epoch} train_loss {row['train_loss']:.4f} val_loss {row['val_loss']:.4f} "
            f"val_usage_macro_f1 {macro_f1:.4f} val_condition_mae {condition_mae:.4f} "
            f"epoch_time {epoch_time:.1f}s",
            flush=True,
        )
        if train_macro_f1 is not None:
            print(
                f"epoch {epoch} train_usage_macro_f1 {train_macro_f1:.4f} "
                f"train_usage_pred {_pred_count_text(train_pred)}",
                flush=True,
            )
        print(f"epoch {epoch} train_heads {_head_loss_text(train_head_means)}", flush=True)
        print(f"epoch {epoch} val_heads {_head_loss_text(val_head_means)}", flush=True)
        print(f"epoch {epoch} val_usage_pred {_pred_count_text(pred)}", flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "config_path": str(config_path),
            "epoch": epoch,
            "val_usage_macro_f1": macro_f1,
            "seed": seed,
        }
        torch.save(checkpoint, run_dir / f"epoch_{epoch}.pt")
        torch.save(checkpoint, run_dir / "last.pt")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            stale = 0
            torch.save(checkpoint, best_path)
        else:
            stale += 1
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        write_json(
            run_dir / "train_summary.json",
            {
                "best_val_usage_macro_f1": best_f1,
                "best_train_usage_macro_f1": max(
                    (row.get("train_usage_macro_f1") or 0.0 for row in history),
                    default=0.0,
                ),
                "seed": seed,
                "usage_only": usage_only,
                "usage_class_weights": {
                    name: float(weight) for name, weight in zip(USAGE_CLASSES, usage_weight_list)
                },
                "history": history,
            },
        )
        if stale >= patience:
            print(f"early stop after epoch {epoch} patience {patience}", flush=True)
            break
    return best_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    path = train(config, args.config)
    print(path)


if __name__ == "__main__":
    main()
