"""CORAL targets stay on the 1..5 label scale used by the dataset."""

import torch

from garmentroute.model import coral_decode, coral_expected_level, coral_targets


def test_coral_targets_are_one_indexed():
    targets = coral_targets(torch.tensor([1, 2, 3, 4, 5]), num_classes=5)
    assert targets.tolist() == [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]


def test_coral_decode_keeps_one_to_five_and_avoids_threshold_collapse():
    # All cumulative probs above 0.5 would hard-count to 5; the expected level
    # stays near the middle when the mass is spread across thresholds.
    logits = torch.tensor([[0.8, 0.6, 0.4, 0.2]], dtype=torch.float32)
    hard_count = 1 + int((logits.sigmoid() > 0.5).sum().item())
    decoded = int(coral_decode(logits).item())
    expected = float(coral_expected_level(logits).item())
    assert hard_count == 5
    assert decoded == round(expected)
    assert decoded < hard_count
    assert 1 <= decoded <= 5


def test_coral_decode_extremes():
    assert int(coral_decode(torch.full((1, 4), -20.0)).item()) == 1
    assert int(coral_decode(torch.full((1, 4), 20.0)).item()) == 5
