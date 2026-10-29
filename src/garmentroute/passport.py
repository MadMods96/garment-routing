"""Mock digital product passport.

The schema is inspired by the direction of the EU Ecodesign for Sustainable
Products Regulation. It is not an official DPP standard, and this project does
not claim to implement one.
"""

from __future__ import annotations

from typing import Any

SCHEMA_ID = "garmentroute.mock-dpp.v1"
SCHEMA_NOTE = (
    "Mock schema inspired by the EU ESPR digital product passport direction. "
    "Not an official standard."
)


def empty_passport(product_id: str | None = None) -> dict[str, Any]:
    """Return a passport with the required keys and no invented fibre or origin."""
    return {
        "schema": SCHEMA_ID,
        "schema_note": SCHEMA_NOTE,
        "product_id": product_id,
        "garment_type": None,
        "fibre_composition": None,
        "brand": None,
        "repair_history": [],
        "manufacturing_origin": None,
    }


def passport_from_annotation(
    *,
    product_id: str,
    garment_type: str | None,
    fibre_composition: str | None,
    brand: str | None,
    repair_history: list[str] | None = None,
    manufacturing_origin: str | None = None,
) -> dict[str, Any]:
    """Build a mock passport from dataset fields.

    ``fibre_composition`` is the dataset material string, which was mostly read
    by an NIR scanner rather than predicted from the photograph.
    """
    document = empty_passport(product_id)
    document["garment_type"] = garment_type or None
    document["fibre_composition"] = fibre_composition or None
    document["brand"] = brand or None
    document["repair_history"] = list(repair_history or [])
    document["manufacturing_origin"] = manufacturing_origin or None
    return document


def coerce_passport(payload: Any, product_id: str | None = None) -> dict[str, Any]:
    """Accept a partial JSON object and fill the mock schema around it."""
    if payload is None:
        return empty_passport(product_id)
    if not isinstance(payload, dict):
        raise TypeError("passport must be a JSON object")
    document = empty_passport(payload.get("product_id") or product_id)
    for key in ("garment_type", "fibre_composition", "brand", "manufacturing_origin"):
        value = payload.get(key)
        document[key] = value if isinstance(value, str) and value.strip() else document[key]
    history = payload.get("repair_history") or []
    if not isinstance(history, list) or not all(isinstance(item, str) for item in history):
        raise TypeError("repair_history must be a list of strings")
    document["repair_history"] = history
    document["schema"] = SCHEMA_ID
    document["schema_note"] = SCHEMA_NOTE
    return document
