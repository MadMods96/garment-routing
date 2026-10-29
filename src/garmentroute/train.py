"""Train the multi-task model. Checkpoints land in ``runs/`` and are not for git."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from garmentroute.dataset import TwoViewDataset
from garmentroute.labels import TERNARY, USAGE_CLASSES
from garmentroute.model import GarmentNet, masked_coral_loss, masked_cross_entropy
from garmentroute.util import load_yaml, set_seed, write_json

NOMINAL = {"stains": TERNARY, "holes": TERNARY, "usage": USAGE_CLASSES}
ORDINAL = ("condition", "pilling")


def _class_weights(frame: pd.DataFrame, column: str, classes: tuple[str, ...]) -> torch.Tensor:
    counts = frame[column].dropna().astype(str).value_counts()
    present = [name for name in classes if int(counts.get(name, 0)) > 0]
    total = sum(int(counts[name]) for name in present) or 1
    weights = []
    for name in classes:
        count = int(counts.get(name, 0))
        weights.append(0.0 if count == 0 else total / (len(present) * count))
    return torch.tensor(weights, dtype=torch.float32)


def _ordinal_example_weight(labels: torch.Tensor, mask: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    index = (labels.clamp(min=1, max=table.shape[0]) - 1).long()
    weights = table[index]
    weights = torch.where(mask > 0, weights, torch.ones_like(weights))
    return weights


def _ordinal_table(frame: pd.DataFrame, column: str) -> torch.Tensor:
    counts = frame[column].dropna().astype(int).value_counts()
    present = [level for level in range(1, 6) if int(counts.get(level, 0)) > 0]
    total = sum(int(counts.get(level, 0)) for level in present) or 1
    weights = []
    for level in range(1, 6):
        count = int(counts.get(level, 0))
        weights.append(0.0 if count == 0 else total / (len(present) * count))
    return torch.tensor(weights, dtype=torch.float32)


def _loss(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    nominal_weights: dict[str, torch.Tensor],
    ordinal_tables: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
) -> torch.Tensor:
    total = outputs["usage"].sum() * 0
    for name in ORDINAL:
        labels = batch[name]
        mask = batch[f"{name}_mask"]
        example_weight = _ordinal_example_weight(labels, mask, ordinal_tables[name])
        total = total + float(loss_weights.get(name, 1.0)) * masked_coral_loss(
            outputs[name], labels, mask, example_weight
        )
    for name in NOMINAL:
        total = total + float(loss_weights.get(name, 1.0)) * masked_cross_entropy(
            outputs[name], batch[name], nominal_weights[name]
        )
    return total


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


def train(config: dict, config_path: Path) -> Path:
    """Train and return the path of the best checkpoint, chosen on validation macro-F1."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    frame = pd.read_parquet(config["index"])
    image_root = Path(config["image_root"])
    image_size = int(config.get("image_size", 224))
    run_name = config.get("run_name") or f"{config.get('backbone', 'model')}-{config.get('fusion', 'concat')}"
    run_dir = Path(config.get("run_dir", "runs")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_rows = frame[frame["split"] == "train"]
    val_rows = frame[frame["split"] == "val"]
    train_set = TwoViewDataset(
        train_rows, image_root, image_size, train=True, view_log=run_dir / "single_view_train.jsonl"
    )
    val_set = TwoViewDataset(
        val_rows, image_root, image_size, train=False, view_log=run_dir / "single_view_val.jsonl"
    )
    train_frame = train_set.frame
    val_frame = val_set.frame
    if len(train_frame) == 0 or len(val_frame) == 0:
        raise RuntimeError("train and val splits need a front or back image on disk")
    train_loader = DataLoader(
        train_set,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = GarmentNet(
        backbone=config.get("backbone", "convnext_tiny"),
        pretrained=bool(config.get("pretrained", True)),
        fusion=config.get("fusion", "concat"),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)
    nominal_weights = {
        "stains": _class_weights(train_frame, "stains_3class", TERNARY).to(device),
        "holes": _class_weights(train_frame, "holes", TERNARY).to(device),
        "usage": _class_weights(train_frame, "usage_class", USAGE_CLASSES).to(device),
    }
    ordinal_tables = {name: _ordinal_table(train_frame, name).to(device) for name in ORDINAL}
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 3e-4)),
        weight_decay=float(config.get("weight_decay", 0.05)),
    )
    epochs = int(config.get("epochs", 12))
    warmup = int(config.get("warmup_epochs", 1))
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history: list[dict] = []
    best_f1 = -1.0
    best_path = run_dir / "best.pt"
    loss_weights = config.get("loss_weights", {})
    steps_per_epoch = max(1, len(train_loader))
    global_step = 0
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup
    base_lr = float(config.get("lr", 3e-4))

    def set_lr(step: int) -> None:
        if step < warmup_steps:
            scale = (step + 1) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for group in optimiser.param_groups:
            group["lr"] = base_lr * scale

    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            set_lr(global_step)
            batch = _move(batch, device)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(batch["front"], batch["back"])
                loss = _loss(outputs, batch, nominal_weights, ordinal_tables, loss_weights)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += float(loss.item()) * batch["front"].shape[0]
            seen += int(batch["front"].shape[0])
            global_step += 1
        model.eval()
        truth: list[int] = []
        pred: list[int] = []
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _move(batch, device)
                outputs = model(batch["front"], batch["back"])
                loss = _loss(outputs, batch, nominal_weights, ordinal_tables, loss_weights)
                val_loss += float(loss.item()) * batch["front"].shape[0]
                val_seen += int(batch["front"].shape[0])
                batch_truth, batch_pred = _usage_macro_f1(outputs, batch)
                truth.extend(batch_truth)
                pred.extend(batch_pred)
        macro_f1 = float(f1_score(truth, pred, average="macro", zero_division=0)) if truth else 0.0
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_loss": val_loss / max(val_seen, 1),
            "val_usage_macro_f1": macro_f1,
            "lr": optimiser.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch} train_loss {row['train_loss']:.4f} val_loss {row['val_loss']:.4f} "
            f"val_usage_macro_f1 {macro_f1:.4f}",
            flush=True,
        )
        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "config_path": str(config_path),
            "epoch": epoch,
            "val_usage_macro_f1": macro_f1,
            "seed": seed,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if macro_f1 >= best_f1:
            best_f1 = macro_f1
            torch.save(checkpoint, best_path)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    write_json(run_dir / "train_summary.json", {"best_val_usage_macro_f1": best_f1, "seed": seed, "history": history})
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
