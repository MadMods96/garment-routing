"""Build one row per garment from local annotation JSON files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from garmentroute.labels import item_id_from_filename, normalise_record

SPLIT_COLUMNS_META = ("seed", "val_fraction", "test_fraction", "stratified")


def _load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return None, f"read error: {exc}"
    except OSError as exc:
        return None, f"read error: {exc}"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"json decode error: {exc}"


def _objects_from_payload(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(payload, dict):
        return [payload], None
    if isinstance(payload, list):
        objects = [item for item in payload if isinstance(item, dict)]
        if not objects:
            return [], "json list contained no annotation objects"
        return objects, None
    return [], f"json top-level type was {type(payload).__name__}"


def _station(relative: Path) -> tuple[str, bool]:
    parts = relative.parts
    if not parts:
        return "unknown", False
    if parts[0] == "test100":
        station = parts[1] if len(parts) > 1 else "unknown"
        return station, True
    return parts[0], False


def _image_name(json_path: Path, item_id: str, view: str) -> str:
    return str(json_path.with_name(f"{view}_{item_id}.jpg")).replace("\\", "/")


def iter_rows(
    root: Path,
    archive_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Walk ``root`` and return garment rows plus a skip log.

    Anything under a ``json_errors`` directory is skipped. Gold-set files live
    under ``test100`` in dataset version 3; a ``Test`` comment also marks gold.
    """
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    seen: dict[str, int] = {}
    root = root.resolve()
    json_paths = sorted(path for path in root.rglob("*.json") if path.is_file())
    for path in json_paths:
        relative = path.relative_to(root)
        relative_posix = relative.as_posix()
        if "json_errors" in relative.parts:
            skipped.append({"path": relative_posix, "reason": "inside json_errors"})
            continue
        item_id = item_id_from_filename(path.name)
        if item_id is None:
            skipped.append({"path": relative_posix, "reason": "filename has no item id"})
            continue
        payload, error = _load_json(path)
        if error is not None or payload is None:
            skipped.append({"path": relative_posix, "reason": error or "empty json"})
            continue
        objects, shape_error = _objects_from_payload(payload)
        if shape_error is not None:
            skipped.append({"path": relative_posix, "reason": shape_error})
            continue
        station, in_test100 = _station(relative)
        for offset, raw in enumerate(objects):
            record = normalise_record(raw)
            comment_gold = bool(record.pop("is_gold_comment"))
            row_id = item_id if offset == 0 else f"{item_id}__ann{offset}"
            if row_id in seen:
                seen[row_id] += 1
                row_id = f"{row_id}__{station}__{seen[row_id]}"
            else:
                seen[row_id] = 0
            front = _image_name(relative, item_id, "front")
            back = _image_name(relative, item_id, "back")
            front_local = root / front
            back_local = root / back
            date_parts = item_id.split("_")
            date = None
            if len(date_parts) >= 3 and all(part.isdigit() for part in date_parts[:3]):
                date = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"
            rows.append(
                {
                    "item_id": row_id,
                    "source_item_id": item_id,
                    "json_path": relative_posix,
                    "station": station,
                    "month_folder": relative.parent.name,
                    "date": date,
                    "front_path": front,
                    "back_path": back,
                    "front_local": front_local.is_file(),
                    "back_local": back_local.is_file(),
                    "front_in_archive": None if archive_names is None else front in archive_names,
                    "back_in_archive": None if archive_names is None else back in archive_names,
                    "is_gold": bool(in_test100 or comment_gold),
                    **record,
                }
            )
    return rows, skipped


def _can_stratify(labels: list[str], test_size: float) -> bool:
    counts = Counter(labels)
    if not counts or min(counts.values()) < 2:
        return False
    n_test = int(round(len(labels) * test_size))
    n_train = len(labels) - n_test
    return n_test >= 1 and n_train >= 1


def assign_splits(
    frame: pd.DataFrame,
    seed: int,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign splits without putting one garment in two of them.

    Each row is one garment. Station3 is the cross-site test. Gold rows are
    held out. Station1 and station2 are split 70/15/15 on ``usage_class`` when
    every class has at least two rows. Thresholds must be tuned on val only.
    """
    out = frame.copy()
    out["split"] = "unused"
    if "duplicate_of" not in out.columns:
        out["duplicate_of"] = pd.NA
    out.loc[out["duplicate_of"].notna(), "split"] = "duplicate"
    out.loc[out["split"].eq("unused") & out["is_gold"].astype(bool), "split"] = "gold"
    out.loc[out["split"].eq("unused") & out["station"].eq("station3"), "split"] = "cross_site"
    pool_mask = (
        out["split"].eq("unused")
        & out["station"].isin(["station1", "station2"])
        & out["usage_class"].notna()
    )
    pool = out.loc[pool_mask].copy()
    stratified = False
    if len(pool) > 0:
        labels = pool["usage_class"].astype(str).tolist()
        ids = pool["item_id"].tolist()
        try:
            if _can_stratify(labels, test_fraction):
                rest_ids, test_ids = train_test_split(
                    ids,
                    test_size=test_fraction,
                    stratify=labels,
                    random_state=seed,
                )
                rest_labels = pool.set_index("item_id").loc[rest_ids, "usage_class"].astype(str).tolist()
                relative_val = val_fraction / max(1e-9, 1.0 - test_fraction)
                if _can_stratify(rest_labels, relative_val):
                    train_ids, val_ids = train_test_split(
                        rest_ids,
                        test_size=relative_val,
                        stratify=rest_labels,
                        random_state=seed,
                    )
                    stratified = True
                else:
                    train_ids, val_ids = train_test_split(
                        rest_ids,
                        test_size=relative_val,
                        random_state=seed,
                    )
            else:
                rest_ids, test_ids = train_test_split(
                    ids,
                    test_size=test_fraction,
                    random_state=seed,
                )
                relative_val = val_fraction / max(1e-9, 1.0 - test_fraction)
                train_ids, val_ids = train_test_split(
                    rest_ids,
                    test_size=relative_val,
                    random_state=seed,
                )
        except ValueError:
            rest_ids, test_ids = train_test_split(ids, test_size=test_fraction, random_state=seed)
            relative_val = val_fraction / max(1e-9, 1.0 - test_fraction)
            train_ids, val_ids = train_test_split(rest_ids, test_size=relative_val, random_state=seed)
            stratified = False
        out.loc[out["item_id"].isin(train_ids), "split"] = "train"
        out.loc[out["item_id"].isin(val_ids), "split"] = "val"
        out.loc[out["item_id"].isin(test_ids), "split"] = "test"
    meta = {
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "stratified": stratified,
        "split_counts": {str(key): int(value) for key, value in out["split"].value_counts().items()},
    }
    return out, meta


def _phash_int(path: Path) -> int:
    import imagehash
    from PIL import Image

    with Image.open(path) as image:
        digest = imagehash.phash(image.convert("RGB"))
    return int(str(digest), 16)


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class _BKNode:
    __slots__ = ("digest", "item_id", "children")

    def __init__(self, digest: int, item_id: str) -> None:
        self.digest = digest
        self.item_id = item_id
        self.children: dict[int, _BKNode] = {}


def _bk_add(root: _BKNode | None, digest: int, item_id: str) -> _BKNode:
    if root is None:
        return _BKNode(digest, item_id)
    node = root
    while True:
        distance = _hamming(digest, node.digest)
        child = node.children.get(distance)
        if child is None:
            node.children[distance] = _BKNode(digest, item_id)
            return root
        node = child


def _bk_query(root: _BKNode | None, digest: int, max_distance: int) -> list[str]:
    if root is None:
        return []
    found: list[str] = []
    stack = [root]
    while stack:
        node = stack.pop()
        distance = _hamming(digest, node.digest)
        if distance <= max_distance:
            found.append(node.item_id)
        low = distance - max_distance
        high = distance + max_distance
        for edge, child in node.children.items():
            if low <= edge <= high:
                stack.append(child)
    return found


def _cluster_item_ids(records: list[tuple[str, int]], threshold: int) -> list[list[str]]:
    """Group item ids whose perceptual hashes fall within ``threshold`` bits."""
    root: _BKNode | None = None
    for item_id, digest in records:
        root = _bk_add(root, digest, item_id)
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        while parent.get(item, item) != item:
            parent[item] = parent.get(parent[item], parent[item])
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for item_id, digest in records:
        for other in _bk_query(root, digest, threshold):
            if other != item_id:
                union(item_id, other)
    clusters: dict[str, list[str]] = {}
    for item_id, _digest in records:
        clusters.setdefault(find(item_id), []).append(item_id)
    return list(clusters.values())


def mark_near_duplicates(
    frame: pd.DataFrame,
    image_root: Path,
    threshold: int = 4,
) -> pd.DataFrame:
    """Mark near-duplicate front images. Gold rows are left untouched.

    ``duplicate_of`` is the item id that is kept. Rows whose front image is not
    on disk are not hashed and are not marked. Re-run this after images exist.
    """
    out = frame.copy()
    if "duplicate_of" not in out.columns:
        out["duplicate_of"] = pd.NA
    if "phash" not in out.columns:
        out["phash"] = pd.NA
    candidates = out.index[~out["is_gold"].astype(bool)].tolist()
    hashes: list[tuple[Any, int, str]] = []
    for index in candidates:
        relative = out.at[index, "front_path"]
        path = image_root / str(relative)
        if not path.is_file():
            continue
        digest = _phash_int(path)
        hashes.append((index, digest, str(out.at[index, "item_id"])))
        out.at[index, "phash"] = f"{digest:016x}"
    id_to_index = {item_id: index for index, _, item_id in hashes}
    label_cols = ["condition", "pilling", "stains_3class", "holes", "smell", "usage", "material"]
    for members in _cluster_item_ids([(item_id, digest) for _, digest, item_id in hashes], threshold):
        if len(members) < 2:
            continue
        ranked = sorted(members, key=lambda item: _keep_score(out.loc[id_to_index[item]], label_cols), reverse=True)
        keeper = ranked[0]
        for item in ranked[1:]:
            out.at[id_to_index[item], "duplicate_of"] = keeper
    return out


_SPLIT_KEEP_RANK = {"train": 0, "val": 1, "test": 2, "cross_site": 3, "unused": 4}


def resolve_near_duplicates(members: list[dict[str, Any]], threshold: int) -> list[dict[str, Any]]:
    """Drop a row only when a kept row is within ``threshold`` bits.

    Gold rows are never dropped. Rows are considered in keep order: gold, then
    train, val, test, and station 3, and within a split the more complete row
    first. A test row matched to a training row is marked ``train_test``.
    """
    ordered = sorted(members, key=lambda member: member["score"], reverse=True)
    ordered.sort(key=lambda member: (0 if member["is_gold"] else 1, _SPLIT_KEEP_RANK.get(str(member["split"]), 9)))
    kept: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for member in ordered:
        neighbors = [
            other for other in kept if _hamming(int(member["digest"]), int(other["digest"])) <= threshold
        ]
        if member["is_gold"] or not neighbors:
            kept.append(member)
            continue
        drops.append(_removal(member, neighbors[0]))
    return drops


def _removal(member: dict[str, Any], keeper: dict[str, Any]) -> dict[str, Any]:
    removed_split = str(member["split"])
    kept_split = str(keeper["split"])
    if removed_split == kept_split:
        reason = "within_split"
    elif removed_split == "test" and kept_split == "train":
        reason = "train_test"
    else:
        reason = "cross_split"
    return {
        "removed_item_id": str(member["item_id"]),
        "kept_item_id": str(keeper["item_id"]),
        "removed_split": removed_split,
        "kept_split": kept_split,
        "hamming": _hamming(int(member["digest"]), int(keeper["digest"])),
        "reason": reason,
    }


def mark_split_safe_duplicates(
    frame: pd.DataFrame,
    image_root: Path,
    threshold: int = 4,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Mark near-duplicate fronts without leaving one photo in two splits.

    The existing split of a kept row stays as it is. A dropped row moves to
    split ``duplicate``. Gold rows are hashed so a training photo of a gold
    garment can be dropped, and the gold row itself is kept.
    """
    out = frame.copy()
    if "duplicate_of" not in out.columns:
        out["duplicate_of"] = pd.NA
    if "phash" not in out.columns:
        out["phash"] = pd.NA
    label_cols = ["condition", "pilling", "stains_3class", "holes", "smell", "usage", "material"]
    hashed: list[dict[str, Any]] = []
    for index in out.index:
        relative = out.at[index, "front_path"]
        if relative is None or (isinstance(relative, float) and pd.isna(relative)):
            continue
        path = image_root / str(relative)
        if not path.is_file():
            continue
        digest = _phash_int(path)
        item_id = str(out.at[index, "item_id"])
        out.at[index, "phash"] = f"{digest:016x}"
        split = str(out.at[index, "split"]) if "split" in out.columns else "unused"
        hashed.append(
            {
                "index": index,
                "item_id": item_id,
                "digest": digest,
                "split": split,
                "is_gold": bool(out.at[index, "is_gold"]),
                "score": _keep_score(out.loc[index], label_cols),
            }
        )
    by_id = {member["item_id"]: member for member in hashed}
    removals = resolve_near_duplicates(hashed, threshold)
    for removal in removals:
        index = by_id[removal["removed_item_id"]]["index"]
        out.at[index, "duplicate_of"] = removal["kept_item_id"]
        if "split" in out.columns:
            out.at[index, "split"] = "duplicate"
    return out, removals


def cross_split_near_duplicates(frame: pd.DataFrame, threshold: int = 4) -> list[dict[str, str]]:
    """Kept rows whose fronts still match across two splits.

    Two gold rows of the same photograph are allowed. Every other kept pair
    within ``threshold`` is reported.
    """
    if "phash" not in frame.columns or "duplicate_of" not in frame.columns:
        return []
    kept = frame[frame["duplicate_of"].isna() & frame["phash"].notna()]
    records: list[tuple[str, int]] = []
    meta: dict[str, tuple[str, bool]] = {}
    for index in kept.index:
        text = str(kept.at[index, "phash"])
        if not text or text == "nan":
            continue
        item_id = str(kept.at[index, "item_id"])
        records.append((item_id, int(text, 16)))
        split = str(kept.at[index, "split"]) if "split" in kept.columns else "unused"
        meta[item_id] = (split, bool(kept.at[index, "is_gold"]))
    violations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    root: _BKNode | None = None
    for item_id, digest in records:
        root = _bk_add(root, digest, item_id)
    for item_id, digest in records:
        for other in _bk_query(root, digest, threshold):
            if other == item_id:
                continue
            pair = tuple(sorted((item_id, other)))
            if pair in seen:
                continue
            seen.add(pair)
            left_split, left_gold = meta[item_id]
            right_split, right_gold = meta[other]
            if left_gold and right_gold and left_split == right_split:
                continue
            if left_split == right_split and (left_gold or right_gold):
                continue
            violations.append(
                {
                    "left_item_id": pair[0],
                    "right_item_id": pair[1],
                    "left_split": meta[pair[0]][0],
                    "right_split": meta[pair[1]][0],
                }
            )
    return violations


def _keep_score(row: pd.Series, columns: list[str]) -> tuple[int, str, str]:
    filled = 0
    for column in columns:
        value = row.get(column)
        if value is not None and pd.notna(value):
            filled += 1
    date = str(row.get("date") or "")
    return filled, date, str(row.get("item_id"))


def _allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Spread ``total`` across groups in proportion to ``sizes``, without overflowing a group."""
    available = {key: int(value) for key, value in sizes.items() if int(value) > 0}
    if total <= 0 or not available:
        return {key: 0 for key in sizes}
    if total >= sum(available.values()):
        return {key: available.get(key, 0) for key in sizes}
    weight = sum(available.values())
    raw = {key: total * value / weight for key, value in available.items()}
    floors = {key: min(int(raw[key]), available[key]) for key in available}
    left = total - sum(floors.values())
    order = sorted(available, key=lambda key: (raw[key] - int(raw[key]), available[key]), reverse=True)
    while left > 0:
        progressed = False
        for key in order:
            if left <= 0:
                break
            if floors[key] < available[key]:
                floors[key] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    return {key: floors.get(key, 0) for key in sizes}


def _stratified_take(pool: pd.DataFrame, n: int, seed: int, column: str) -> pd.DataFrame:
    """Take ``n`` rows, stratified on ``column`` when every class can support it."""
    if n <= 0 or pool.empty:
        return pool.iloc[0:0]
    if n >= len(pool):
        return pool
    if column not in pool.columns:
        return pool.sample(n=n, random_state=seed)
    labels = pool[column].astype(str)
    fraction = n / len(pool)
    if _can_stratify(labels.tolist(), fraction) or _can_stratify(labels.tolist(), 1 - fraction):
        try:
            taken, _rest = train_test_split(pool, train_size=n, stratify=labels, random_state=seed)
            return taken
        except ValueError:
            pass
    return pool.sample(n=n, random_state=seed)


def _station3_sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Keep every non-Reuse row that fits, then fill with Reuse.

    Non-Reuse is over-represented relative to station 3's own mix. If the
    non-Reuse rows themselves exceed ``n``, those classes are stratified and
    Reuse is left out.
    """
    if n <= 0 or pool.empty:
        return pool.iloc[0:0]
    if n >= len(pool):
        return pool
    label_column = "usage" if "usage" in pool.columns else "usage_class"
    labels = pool[label_column]
    non_reuse = pool[labels.ne("Reuse")]
    reuse = pool[labels.eq("Reuse")]
    if len(non_reuse) >= n:
        return _stratified_take(non_reuse, n, seed, label_column)
    filler = _stratified_take(reuse, n - len(non_reuse), seed, label_column)
    return pd.concat([non_reuse, filler], axis=0)


def _wargon_sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified sample from the Wargön train, validation, and test splits."""
    if n <= 0 or pool.empty:
        return pool.iloc[0:0]
    column = "usage_class" if "usage_class" in pool.columns else "usage"
    if "split" not in pool.columns:
        return _stratified_take(pool, n, seed, column)
    sizes = {split: int(pool["split"].eq(split).sum()) for split in ("train", "val", "test")}
    parts = []
    for offset, (split, count) in enumerate(_allocate(sizes, n).items()):
        if count <= 0:
            continue
        parts.append(_stratified_take(pool[pool["split"].eq(split)], count, seed + offset, column))
    if not parts:
        return pool.iloc[0:0]
    return pd.concat(parts, axis=0)


def choose_subset(
    frame: pd.DataFrame,
    target: int,
    seed: int,
    station3_n: int = 1500,
) -> pd.DataFrame:
    """About ``target`` rows: all gold, ``station3_n`` from Myrorna, the rest from Wargön.

    Wargön rows are drawn separately from train, validation, and test, stratified
    on ``usage_class``. The station 3 draw keeps every non-Reuse garment that
    fits in ``station3_n`` and fills the rest with Reuse. Gold rows are never
    dropped to hit the target.
    """
    eligible = frame[frame["duplicate_of"].isna()].copy() if "duplicate_of" in frame.columns else frame.copy()
    gold = eligible[eligible["is_gold"].astype(bool)]
    station3_mask = eligible["station"].eq("station3") & ~eligible["is_gold"].astype(bool)
    if "split" in eligible.columns:
        station3_mask = station3_mask & eligible["split"].eq("cross_site")
    station3 = _station3_sample(eligible[station3_mask], station3_n, seed)
    pool_mask = eligible["station"].isin(["station1", "station2"]) & ~eligible["is_gold"].astype(bool)
    if "split" in eligible.columns:
        pool_mask = pool_mask & eligible["split"].isin(["train", "val", "test"])
    if "usage_class" in eligible.columns:
        pool_mask = pool_mask & eligible["usage_class"].notna()
    need = target - len(gold) - len(station3)
    filler = _wargon_sample(eligible[pool_mask], need, seed)
    return pd.concat([gold, station3, filler], axis=0)


def subset_members(subset: pd.DataFrame) -> list[str]:
    """Archive-relative front and back paths for the image download."""
    names: list[str] = []
    for column in ("front_path", "back_path"):
        names.extend(subset[column].dropna().astype(str).tolist())
    return names
