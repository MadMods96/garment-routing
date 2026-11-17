"""Load a checkpoint and turn two photographs plus an optional passport into a decision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from garmentroute.dataset import build_transforms
from garmentroute.labels import TERNARY, USAGE_CLASSES
from garmentroute.model import GarmentNet, coral_decode, decode_ternary
from garmentroute.passport import coerce_passport
from garmentroute.router import GarmentSignals, decide, load_router_config
from garmentroute.uncertainty import apply_temperature

DEMO_FORMAT = "garmentroute_demo_v1"


def load_model(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[GarmentNet, dict, torch.device, dict[str, Any]]:
    """Load a training checkpoint or a demo pack.

    The backbone weights come from the file, not a fresh download. Floating
    weights stored as float16 are promoted to float32 on the inference device.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config") or {})
    meta = {
        "temperature": float(checkpoint.get("temperature", 1.0)),
        "abstain_threshold": checkpoint.get("abstain_threshold"),
        "holes_major_threshold": checkpoint.get("holes_major_threshold"),
        "coral_decode": checkpoint.get("coral_decode"),
        "format": checkpoint.get("format"),
    }
    if meta["abstain_threshold"] is not None:
        meta["abstain_threshold"] = float(meta["abstain_threshold"])
    if meta["holes_major_threshold"] is not None:
        meta["holes_major_threshold"] = float(meta["holes_major_threshold"])
    model = GarmentNet(
        backbone=config.get("backbone", "convnext_tiny"),
        pretrained=False,
        fusion=config.get("fusion", "concat"),
        dropout=float(config.get("dropout", 0.2)),
    )
    state = checkpoint["model"]
    promoted = {
        key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
        for key, value in state.items()
    }
    model.load_state_dict(promoted)
    model.to(device)
    model.eval()
    return model, config, device, meta


def export_demo_pack(
    checkpoint_path: Path,
    output_path: Path,
    temperature: float,
    abstain_threshold: float | None,
    holes_major_threshold: float | None = None,
    coral_decode_method: str | None = None,
) -> Path:
    """Write weights-only float16 weights plus calibration scalars. No optimizer state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config") or {})
    keep = ("backbone", "fusion", "image_size", "dropout", "seed")
    slim_config = {key: config[key] for key in keep if key in config}
    slim_config.setdefault("backbone", "convnext_tiny")
    slim_config.setdefault("fusion", "front")
    slim_config.setdefault("image_size", 224)
    slim_config.setdefault("dropout", 0.2)
    state = {
        key: value.detach().half().cpu() if torch.is_tensor(value) and value.is_floating_point() else value
        for key, value in checkpoint["model"].items()
    }
    pack = {
        "format": DEMO_FORMAT,
        "model": state,
        "config": slim_config,
        "temperature": float(temperature),
        "abstain_threshold": None if abstain_threshold is None else float(abstain_threshold),
        "holes_major_threshold": None if holes_major_threshold is None else float(holes_major_threshold),
        "coral_decode": coral_decode_method,
        "dtype": "float16",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, output_path)
    return output_path


def _tensor(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    transform = build_transforms(image_size, train=False)
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


def ood_warning(front: Image.Image, usage_probs: dict[str, float]) -> str | None:
    """Flag photos that do not look like the tiled sorting-table training setup.

    Plain studio / seamless grey backgrounds fail because they lack detectable
    grid lines (strong horizontal and vertical edge bands), not merely low
    average gradient energy.
    """
    max_prob = max(usage_probs.values()) if usage_probs else 0.0
    if max_prob < 0.35:
        return "Low max usage probability — photo may be out of distribution."
    gray = front.convert("L").resize((96, 96))
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    border_mean = float(border.mean())
    # Training photos sit on light tiles; a dark border is atypical.
    if border_mean < 0.35:
        return "Background looks unlike the light tiled sorting table used in training."
    gx = np.abs(np.diff(arr, axis=1))
    gy = np.abs(np.diff(arr, axis=0))
    # Require actual grid lines: enough strong edges in both orientations.
    line_x = float((gx > 0.07).mean())
    line_y = float((gy > 0.07).mean())
    if line_x < 0.025 or line_y < 0.025:
        return (
            "No tiled sorting-table grid lines detected — "
            "plain / studio backgrounds are out of distribution."
        )
    return None


def assess(
    model: GarmentNet,
    config: dict,
    front: Image.Image,
    back: Image.Image | None,
    passport: dict[str, Any] | None,
    router_config_path: Path,
    device: torch.device,
    temperature: float = 1.0,
    abstain_threshold: float | None = None,
    holes_major_threshold: float | None = None,
    coral_decode_method: str | None = None,
) -> dict[str, Any]:
    """Predict condition and damage, then ask the router for a route.

    Fibre composition is taken only from the passport. The photograph is not
    treated as a material test. If ``back`` is missing, the front image is used
    for both views (front-only fusion ignores the second stream).
    """
    image_size = int(config.get("image_size", 224))
    back_image = front if back is None else back
    with torch.no_grad():
        outputs = model(_tensor(front, image_size, device), _tensor(back_image, image_size, device))
    usage_logits = outputs["usage"].detach().float().cpu().numpy()
    scaled = apply_temperature(usage_logits, temperature)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    usage_index = int(probabilities[0].argmax())
    condition = int(coral_decode(outputs["condition"], method=coral_decode_method).item())
    pilling = int(coral_decode(outputs["pilling"], method=coral_decode_method).item())
    stains_idx = int(decode_ternary(outputs["stains"]).item())
    holes_idx = int(decode_ternary(outputs["holes"], major_threshold=holes_major_threshold).item())
    stains = TERNARY[stains_idx]
    holes = TERNARY[holes_idx]
    stains_probs = torch.softmax(outputs["stains"].float(), dim=-1)[0].detach().cpu().numpy()
    holes_probs = torch.softmax(outputs["holes"].float(), dim=-1)[0].detach().cpu().numpy()
    probs = {label: float(probabilities[0, index]) for index, label in enumerate(USAGE_CLASSES)}
    document = coerce_passport(passport)
    usage_confidence = float(probabilities[0, usage_index])
    signals = GarmentSignals(
        condition=condition,
        pilling=pilling,
        stains=stains,
        holes=holes,
        usage_confidence=usage_confidence,
        usage_probs=probs,
        usage_class=USAGE_CLASSES[usage_index],
        holes_confidence=float(holes_probs[holes_idx]),
        stains_confidence=float(stains_probs[stains_idx]),
    )
    router_config = load_router_config(router_config_path)
    decision = decide(signals, document, router_config, abstain_threshold=abstain_threshold)
    send_to_human = decision.route == "HUMAN_REVIEW" or (
        abstain_threshold is not None and usage_confidence < float(abstain_threshold)
    )
    rule_route = decision.rule_route or (
        max(decision.scores, key=decision.scores.get) if decision.scores else "HUMAN_REVIEW"
    )
    warning = ood_warning(front.convert("RGB"), probs)
    return {
        "condition": condition,
        "pilling": pilling,
        "stains": stains,
        "holes": holes,
        "usage_class": USAGE_CLASSES[usage_index],
        "usage_probabilities": probs,
        "usage_confidence": usage_confidence,
        "holes_confidence": float(holes_probs[holes_idx]),
        "stains_confidence": float(stains_probs[stains_idx]),
        "route": decision.route,
        "rule_route": rule_route,
        "confidence": decision.confidence,
        "reasons": list(dict.fromkeys(decision.reasons)),
        "alternatives": decision.alternatives,
        "passport": document,
        "temperature": temperature,
        "abstain_threshold": abstain_threshold,
        "send_to_human_review": bool(send_to_human),
        "ood_warning": warning,
    }
