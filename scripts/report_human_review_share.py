"""Score front-224 on test under the current model-first router; keep the 72.1% comparison row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from garmentroute.evaluate import predict_checkpoint, router_from_predictions
from garmentroute.util import load_yaml, write_json
from garmentroute.views import filter_to_subset, find_subset_members

OLD_DISAGREEMENT_POLICY = {
    "policy": "escalate_on_soft_rule_disagreement",
    "human_review_share": 0.721,
    "abstained": 428,
    "n": 594,
    "note": (
        "Previous policy sent soft rule-vs-model disagreements to HUMAN_REVIEW. "
        "Kept as a comparison row only."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "convnext_tiny_front" / "best.pt")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "index.parquet")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "resized")
    parser.add_argument("--router", type=Path, default=ROOT / "configs" / "router.yaml")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    frame = pd.read_parquet(args.index)
    members = find_subset_members(ROOT / "data", ROOT)
    if members is not None:
        frame = filter_to_subset(frame, members)
    test = frame[frame["split"] == "test"]
    print(f"predict test n={len(test)}", flush=True)
    predictions = predict_checkpoint(args.checkpoint, test, args.image_root, args.batch_size, mc_passes=0) or []
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    temperature = float(ckpt.get("temperature") or 1.0)
    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results.is_file() else {}
    thr = (results.get("front") or {}).get("abstain_threshold")
    if isinstance(thr, dict):
        thr = thr.get("threshold")
    if thr is None:
        thr = ckpt.get("abstain_threshold")
    if thr is not None:
        thr = float(thr)
    router = router_from_predictions(test, predictions, load_yaml(args.router), temperature, thr)
    share = router.get("human_review_share")
    selective = router.get("selective") or {}
    all_g = router.get("all_garments_model_usage") or {}
    print(
        f"test HUMAN_REVIEW share={share:.3f} ({router.get('abstained')}/{router.get('n')})",
        flush=True,
    )
    print(
        f"selective n={selective.get('n_decided')} coverage={selective.get('coverage')} "
        f"acc={selective.get('accuracy')} macro_f1={selective.get('macro_f1')}",
        flush=True,
    )
    print(
        f"all_garments_model_usage acc={all_g.get('accuracy')} macro_f1={all_g.get('macro_f1')}",
        flush=True,
    )
    if share is not None and share > 0.5:
        print("More than half of test garments go to HUMAN_REVIEW.", flush=True)
    curve = router.get("coverage_accuracy") or []
    if curve:
        mid = curve[len(curve) // 2]
        print(
            f"coverage_accuracy example coverage={mid.get('coverage')} accuracy={mid.get('accuracy')}",
            flush=True,
        )

    front = results.setdefault("front", {})
    splits = front.setdefault("splits", {}).setdefault("test", {})
    splits["router_policy_comparison"] = {
        "escalate_on_soft_disagreement": OLD_DISAGREEMENT_POLICY,
        "model_first_hard_rule_safety_only": {
            "policy": router.get("policy"),
            "human_review_share": share,
            "abstained": router.get("abstained"),
            "n": router.get("n"),
            "selective": selective,
            "all_garments_model_usage": all_g,
            "coverage_accuracy": curve,
            "human_review_share_note": router.get("human_review_share_note"),
            "temperature": temperature,
            "abstain_threshold": thr,
        },
    }
    splits["router_on_predicted_attributes_current"] = splits["router_policy_comparison"][
        "model_first_hard_rule_safety_only"
    ]
    write_json(args.results, results)
    print(f"wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
