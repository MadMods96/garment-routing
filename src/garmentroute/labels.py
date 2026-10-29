"""Label normalisation for the Clothing Dataset for Second-Hand Fashion.

The published options are documented on the Zenodo record. Two annotation
protocols appear in the files: stains were Yes/No before 17 November 2022, and
holes and smell were added after that date. Missing holes or smell stay missing.
They are never filled with "None".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

USAGE_LABELS: tuple[str, ...] = (
    "Reuse",
    "Export",
    "Recycle",
    "Repair",
    "Remake",
    "Energy recovery",
)
USAGE_CLASSES: tuple[str, ...] = ("Reuse", "Export", "Recycle", "Other")
TERNARY: tuple[str, ...] = ("None", "Minor", "Major")
ORDINAL_LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)

USAGE_CLASS_TO_INDEX: dict[str, int] = {label: index for index, label in enumerate(USAGE_CLASSES)}
TERNARY_TO_INDEX: dict[str, int] = {label: index for index, label in enumerate(TERNARY)}

_USAGE_ALIASES: dict[str, str] = {
    "reuse": "Reuse",
    "export": "Export",
    "recycle": "Recycle",
    "rcycle": "Recycle",
    "repair": "Repair",
    "remake": "Remake",
    "energy recovery": "Energy recovery",
}
_TRAIN_CLASS: dict[str, str] = {
    "Reuse": "Reuse",
    "Export": "Export",
    "Recycle": "Recycle",
    "Repair": "Other",
    "Remake": "Other",
    "Energy recovery": "Other",
}
_TERNARY_ALIASES: dict[str, str] = {
    "none": "None",
    "minor": "Minor",
    "major": "Major",
}
_GOLD_RE = re.compile(r"\bTest(?:\s*[-_:]?\s*(\d+))?\b")
_GOLD_RE_I = re.compile(_GOLD_RE.pattern, re.IGNORECASE)
_ITEM_RE = re.compile(r"labels_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})")


@dataclass(frozen=True)
class StainNorm:
    """Normalised stain fields.

    ``stains_3class`` is None for legacy Yes/No labels so those rows can be
    masked out of the three-class head. ``has_stain`` keeps them for a binary
    metric. Yes is stored as ``Minor_or_Major`` and is not treated as Minor.
    """

    stains: str | None
    stains_3class: str | None
    stains_legacy: bool
    has_stain: bool | None


def as_text(value: Any) -> str:
    """Flatten a JSON value to a stripped string. Lists are joined with spaces."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(as_text(item) for item in value).strip()
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def canonical_usage(value: Any) -> str | None:
    """Map a raw usage string onto the six published labels."""
    text = as_text(value)
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return _USAGE_ALIASES.get(text.lower())


def usage_class(usage: str | None) -> str | None:
    """Map a canonical usage label onto the four training classes.

    Repair, Remake, and Energy recovery become Other. Their raw counts stay
    available on the ``usage`` column for EDA.
    """
    if usage is None:
        return None
    return _TRAIN_CLASS.get(usage)


def canonical_ternary(value: Any) -> str | None:
    """Map holes or smell onto None/Minor/Major. Missing and unknown stay missing."""
    text = as_text(value)
    if not text:
        return None
    return _TERNARY_ALIASES.get(text.lower())


def normalise_stains(value: Any) -> StainNorm:
    """Normalise a stains value from either annotation protocol."""
    text = as_text(value)
    if not text:
        return StainNorm(None, None, False, None)
    key = text.lower()
    if key == "yes":
        return StainNorm("Minor_or_Major", None, True, True)
    if key == "no":
        return StainNorm("No", None, True, False)
    ternary = _TERNARY_ALIASES.get(key)
    if ternary is None:
        return StainNorm(None, None, False, None)
    has_stain = ternary != "None"
    return StainNorm(ternary, ternary, False, has_stain)


def parse_ordinal(value: Any) -> int | None:
    """Parse a 1–5 condition or pilling score. Values outside that range are missing."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and value != value:
        return None
    text = as_text(value)
    if not text:
        return None
    try:
        number = int(float(text))
    except ValueError:
        return None
    if number not in ORDINAL_LEVELS:
        return None
    return number


def parse_gold_id(comment: Any) -> str | None:
    """Return the gold-set item number from a comment such as ``Test 92``.

    The marker is case-sensitive. A lowercase ``test 92`` does not yield an id.
    """
    match = _GOLD_RE.search(as_text(comment))
    if match is None or match.group(1) is None:
        return None
    return match.group(1)


def loose_gold_number(comment: Any) -> str | None:
    """Item number from ``Test 92`` or ``test 92``.

    This does not assign ``gold_id``. It only recognises the same wording with
    either capitalisation, so a census can see which lowercase comments refer
    to a garment the case-sensitive parser already numbered.
    """
    match = _GOLD_RE_I.search(as_text(comment))
    if match is None or match.group(1) is None:
        return None
    return str(int(match.group(1)))


def is_gold_comment(comment: Any) -> bool:
    """True when the comment contains the dataset's ``Test`` marker."""
    return _GOLD_RE.search(as_text(comment)) is not None


def item_id_from_filename(filename: str) -> str | None:
    """Extract the datetime item id from ``labels_YYYY_MM_DD_HH_MM_SS.json``."""
    match = _ITEM_RE.search(filename.replace("\\", "/"))
    if match is None:
        return None
    return match.group(1)


def empty_damage() -> dict[str, str | None]:
    """Damage-location fields present on later station1 and station2 forms."""
    keys = (
        "damageimage",
        "damageloc",
        "damage",
        "damage2image",
        "damage2loc",
        "damage2",
        "damage3image",
        "damage3loc",
        "damage3",
    )
    return {key: None for key in keys}


def _optional_text(raw: dict[str, Any], key: str) -> str | None:
    if key not in raw:
        return None
    text = as_text(raw.get(key))
    return text or None


def normalise_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn one annotation object into flat label columns.

    Paths, station, and the split are added by the index builder. This function
    only interprets the JSON object.
    """
    stains = normalise_stains(raw.get("stains"))
    usage = canonical_usage(raw.get("usage"))
    comment = _optional_text(raw, "comment")
    if comment is None:
        comment = _optional_text(raw, "comments")
    damage = empty_damage()
    for key in damage:
        damage[key] = _optional_text(raw, key)
    holes_present = "holes" in raw and as_text(raw.get("holes")) != ""
    smell_present = "smell" in raw and as_text(raw.get("smell")) != ""
    return {
        "usage": usage,
        "usage_raw": _optional_text(raw, "usage"),
        "usage_class": usage_class(usage),
        "condition": parse_ordinal(raw.get("condition")),
        "pilling": parse_ordinal(raw.get("pilling")),
        "stains": stains.stains,
        "stains_3class": stains.stains_3class,
        "stains_legacy": stains.stains_legacy,
        "has_stain": stains.has_stain,
        "holes": canonical_ternary(raw.get("holes")) if holes_present else None,
        "smell": canonical_ternary(raw.get("smell")) if smell_present else None,
        "material": _optional_text(raw, "material"),
        "type": _optional_text(raw, "type"),
        "brand": _optional_text(raw, "brand"),
        "trend": _optional_text(raw, "trend"),
        "price": _optional_text(raw, "price"),
        "comment": comment,
        "annotator": _optional_text(raw, "annotator"),
        "gold_id": parse_gold_id(comment),
        "is_gold_comment": is_gold_comment(comment),
        **damage,
    }
