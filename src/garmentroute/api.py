"""FastAPI service. ``POST /assess`` takes a front image, a back image, and an optional passport."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="Garment routing", version="0.1.0")
_RUNTIME: dict | None = None


def _runtime() -> dict:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    checkpoint = os.environ.get("GARMENT_CHECKPOINT", "")
    if not checkpoint or not Path(checkpoint).is_file():
        raise HTTPException(status_code=503, detail="Set GARMENT_CHECKPOINT to a trained checkpoint. This process will not invent predictions.")
    from garmentroute.infer import load_model

    model, config, device = load_model(Path(checkpoint))
    _RUNTIME = {
        "model": model,
        "config": config,
        "device": device,
        "router": Path(os.environ.get("GARMENT_ROUTER", "configs/router.yaml")),
        "temperature": float(os.environ.get("GARMENT_TEMPERATURE", "1")),
        "threshold": float(os.environ["GARMENT_ABSTAIN"]) if os.environ.get("GARMENT_ABSTAIN") else None,
    }
    return _RUNTIME


@app.get("/health")
def health() -> dict:
    checkpoint = os.environ.get("GARMENT_CHECKPOINT", "")
    return {"status": "ok", "checkpoint_configured": bool(checkpoint and Path(checkpoint).is_file())}


@app.post("/assess")
async def assess(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    passport: str | None = Form(None),
) -> dict:
    """Return condition, damage, route, reasons, and confidence for one garment."""
    runtime = _runtime()
    try:
        front_image = Image.open(front.file)
        back_image = Image.open(back.file)
    except Exception as exc:  # noqa: BLE001 - bad uploads are a client error
        raise HTTPException(status_code=400, detail=f"could not read images: {exc}") from exc
    document = None
    if passport:
        try:
            document = json.loads(passport)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="passport is not valid JSON") from exc
    from garmentroute.infer import assess as predict

    return predict(
        runtime["model"],
        runtime["config"],
        front_image,
        back_image,
        document,
        runtime["router"],
        runtime["device"],
        temperature=runtime["temperature"],
        abstain_threshold=runtime["threshold"],
    )
