"""Cross-site gold labels against within-Wargön labels on the same garments."""

import pandas as pd

from garmentroute.metrics import site_label_agreement
from garmentroute.reporting import render_results


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item_id": "a1", "gold_id": "1", "station": "station1", "usage": "Reuse", "condition": 4},
            {"item_id": "a2", "gold_id": "1", "station": "station1", "usage": "Export", "condition": 4},
            {"item_id": "a3", "gold_id": "1", "station": "station3", "usage": "Reuse", "condition": 4},
            {"item_id": "b1", "gold_id": "2", "station": "station2", "usage": "Recycle", "condition": 2},
            {"item_id": "b2", "gold_id": "2", "station": "station1", "usage": "Recycle", "condition": 1},
            {"item_id": "b3", "gold_id": "2", "station": "station3", "usage": "Export", "condition": 2},
            {"item_id": "c1", "gold_id": "3", "station": "station1", "usage": "Reuse", "condition": 5},
            {"item_id": "d1", "gold_id": "4", "station": "station3", "usage": "Reuse", "condition": 5},
        ]
    )


def test_site_agreement_keeps_dual_site_garments_and_splits_view_from_route():
    report = site_label_agreement(_frame())
    assert report["n_garments"] == 2
    assert report["cross_site"]["usage"]["n_pairs"] == 4
    assert report["within_wargon"]["usage"]["n_pairs"] == 2
    assert report["cross_site"]["usage"]["raw_agreement"] == 0.25
    assert report["within_wargon"]["usage"]["raw_agreement"] == 0.5
    matrix = report["cross_site"]["usage"]["confusion_matrix"]
    labels = matrix["labels"]
    cells = matrix["matrix"]
    assert cells[labels.index("Reuse")][labels.index("Reuse")] == 1
    assert cells[labels.index("Export")][labels.index("Reuse")] == 1
    assert cells[labels.index("Recycle")][labels.index("Export")] == 2
    assert report["cross_site"]["condition"]["mean_difference_myrorna_minus_wargon"] == 0.25
    assert report["cross_site"]["condition"]["cohen_kappa_quadratic"] is not None
    assert report["within_wargon"]["condition"]["cohen_kappa_quadratic"] is not None
    view = report["view_versus_route"]
    assert view["n_pairs"] == 4
    assert view["same_view_same_route"] == 1
    assert view["same_view_different_route"] == 2
    assert view["different_view_same_route"] == 0
    assert view["different_view_different_route"] == 1


def test_readme_quotes_the_site_kappas():
    results = {
        "dataset": "clothing_v3",
        "smoke": False,
        "seed": 1,
        "split_meta": {"stratified": True},
        "majority": {},
        "router": {"on_expert_attributes": {}},
        "front": None,
        "fusion": None,
        "gold": {
            "expert_vs_expert_usage": {"cohen_kappa": 0.2, "raw_agreement": 0.5, "n_pairs": 1, "n_items": 1, "n_items_with_pairs": 1},
            "expert_vs_expert_condition": {"cohen_kappa": 0.1, "cohen_kappa_quadratic": 0.3},
            "site_agreement": site_label_agreement(_frame()),
        },
    }
    text = render_results(results, {"n_rows": 1, "n_gold": 1})
    assert "Usage Cohen's kappa, cross-site" in text
    assert "within Wargön" in text
    assert "Same condition and a different route" in text
    assert "Different condition: the sites give the garment different condition scores." in text
    assert "0.250" in text
    assert "0.99" not in text
