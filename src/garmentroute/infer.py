"""Load a checkpoint and turn two photographs plus an optional passport into a decision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from garmentroute.dataset import build_transforms
from garmentroute.labels import TERNARY, USAGE_CLASSES
from garmentroute.model import GarmentNet, coral_decode
from garmentroute.passport import coerce_passport
from garmentroute.router import GarmentSignals, decide, load_router_config
from garmentroute.uncertainty import apply_temperature


def load_model(checkpoint_path: Path, device: torch.device | None = None) -> tuple[GarmentNet, dict, torch.device]:
    """Load a training checkpoint. The backbone weights come from the checkpoint, not a fresh download."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GarmentNet(
        backbone=config.get("backbone", "convnext_tiny"),
        pretrained=False,
        fusion=config.get("fusion", "concat"),
        dropout=float(config.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, config, device


def _tensor(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    transform = build_transforms(image_size, train=False)
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


def assess(
    model: GarmentNet,
    config: dict,
    front: Image.Image,
    back: Image.Image,
    passport: dict[str, Any] | None,
    router_config_path: Path,
    device: torch.device,
    temperature: float = 1.0,
    abstain_threshold: float | None = None,
) -> dict[str, Any]:
    """Predict condition and damage, then ask the router for a route.

    Fibre composition is taken only from the passport. The photograph is not
    treated as a material test.
    """
    image_size = int(config.get("image_size", 224))
    with torch.no_grad():
        outputs = model(_tensor(front, image_size, device), _tensor(back, image_size, device))
    usage_logits = outputs["usage"].detach().float().cpu().numpy()
    scaled = apply_temperature(usage_logits, temperature)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    usage_index = int(probabilities[0].argmax())
    condition = int(coral_decode(outputs["condition"]).item())
    pilling = int(coral_decode(outputs["pilling"]).item())
    stains = TERNARY[int(outputs["stains"].argmax(dim=-1).item())]
    holes = TERNARY[int(outputs["holes"].argmax(dim=-1).item())]
    probs = {label: float(probabilities[0, index]) for index, label in enumerate(USAGE_CLASSES)}
    document = coerce_passport(passport)
    signals = GarmentSignals(
        condition=condition,
        pilling=pilling,
        stains=stains,
        holes=holes,
        usage_confidence=float(probabilities[0, usage_index]),
        usage_probs=probs,
    )
    router_config = load_router_config(router_config_path)
    decision = decide(signals, document, router_config, abstain_threshold=abstain_threshold)
    return {
        "condition": condition,
        "pilling": pilling,
        "stains": stains,
        "holes": holes,
        "usage_class": USAGE_CLASSES[usage_index],
        "usage_probabilities": probs,
        "route": decision.route,
        "confidence": decision.confidence,
        "reasons": decision.reasons,
        "alternatives": decision.alternatives,
        "passport": document,
        "temperature": temperature,
    }
