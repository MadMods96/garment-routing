"""Shared-backbone multi-task model for front and back garment photos."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from garmentroute.labels import TERNARY, USAGE_CLASSES


class CoralHead(nn.Module):
    """CORAL-style ordinal head: one shared direction and one bias per threshold."""

    def __init__(self, in_features: int, num_classes: int = 5) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("ordinal head needs at least two levels")
        self.num_classes = num_classes
        self.weight = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.weight(features) + self.bias


def coral_targets(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Binary targets P(y > k) for k = 1 .. num_classes-1. ``labels`` are 1-indexed."""
    levels = torch.arange(1, num_classes, device=labels.device)
    return (labels.unsqueeze(1) > levels).to(dtype=torch.float32)


def coral_level_probs(logits: torch.Tensor) -> torch.Tensor:
    """Class probabilities on the 1..K scale from cumulative CORAL logits."""
    probs = logits.sigmoid()
    # P(Y=1) = 1 - P(Y>1); P(Y=k) = P(Y>k-1) - P(Y>k); P(Y=K) = P(Y>K-1)
    greater = torch.cat([torch.ones_like(probs[:, :1]), probs, torch.zeros_like(probs[:, :1])], dim=1)
    return (greater[:, :-1] - greater[:, 1:]).clamp(min=0)


def coral_expected_level(logits: torch.Tensor) -> torch.Tensor:
    """Expected level from the cumulative probabilities, still on the 1..K scale."""
    level_probs = coral_level_probs(logits)
    levels = torch.arange(1, level_probs.shape[1] + 1, device=logits.device, dtype=level_probs.dtype)
    return (level_probs * levels).sum(dim=-1)


def coral_decode_expected(logits: torch.Tensor) -> torch.Tensor:
    """Round the expected level back onto 1 .. K."""
    maximum = logits.shape[-1] + 1
    return coral_expected_level(logits).round().clamp(1, maximum).to(dtype=torch.long)


def coral_decode_threshold(logits: torch.Tensor) -> torch.Tensor:
    """Count cumulative probabilities above 0.5 (can collapse to extremes)."""
    return (1 + (logits.sigmoid() > 0.5).sum(dim=-1)).to(dtype=torch.long)


def coral_decode_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over implied class probabilities on the 1 .. K scale."""
    return (coral_level_probs(logits).argmax(dim=-1) + 1).to(dtype=torch.long)


CORAL_DECODE_METHODS = {
    "expected": coral_decode_expected,
    "threshold": coral_decode_threshold,
    "argmax_levels": coral_decode_argmax,
}

# Chosen by quadratic weighted kappa on val (see front.condition_decode in results).
CORAL_DECODE_DEFAULT = "expected"


def coral_decode(logits: torch.Tensor, method: str | None = None) -> torch.Tensor:
    """Decode cumulative logits to a level in 1 .. K.

    ``method`` is ``expected`` (default), ``threshold``, or ``argmax_levels``.
    The default is set from the val quadratic-kappa bake-off.
    """
    name = method or CORAL_DECODE_DEFAULT
    decoder = CORAL_DECODE_METHODS.get(name)
    if decoder is None:
        raise ValueError(f"unknown coral decode method: {name}")
    return decoder(logits)


class GarmentNet(nn.Module):
    """One pretrained backbone, two views, five task heads.

    ``fusion`` is ``front``, ``mean``, or ``concat``. Dropout stays in the graph
    so Monte Carlo dropout can enable it at inference without touching the
    normalisation layers.
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        fusion: str = "concat",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if fusion not in {"front", "mean", "concat"}:
            raise ValueError("fusion must be front, mean, or concat")
        import timm

        self.fusion = fusion
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feature_dim = int(self.backbone.num_features)
        head_dim = feature_dim * 2 if fusion == "concat" else feature_dim
        self.dropout = nn.Dropout(dropout)
        self.condition = CoralHead(head_dim, 5)
        self.pilling = CoralHead(head_dim, 5)
        self.stains = nn.Linear(head_dim, len(TERNARY))
        self.holes = nn.Linear(head_dim, len(TERNARY))
        self.usage = nn.Linear(head_dim, len(USAGE_CLASSES))

    def encode(self, front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
        front_features = self.backbone(front)
        if self.fusion == "front":
            return front_features
        back_features = self.backbone(back)
        if self.fusion == "mean":
            return 0.5 * (front_features + back_features)
        return torch.cat([front_features, back_features], dim=1)

    def forward(self, front: torch.Tensor, back: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(self.encode(front, back))
        return {
            "condition": self.condition(features),
            "pilling": self.pilling(features),
            "stains": self.stains(features),
            "holes": self.holes(features),
            "usage": self.usage(features),
        }

    def mc_dropout_mode(self) -> None:
        """Eval-mode batch/layer norm, train-mode dropout, for MC dropout."""
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


def masked_coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    example_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean binary cross-entropy on cumulative targets, ignoring masked rows."""
    if int(mask.sum().item()) == 0:
        return logits.sum() * 0
    target = coral_targets(labels.clamp(min=1, max=5), num_classes=logits.shape[1] + 1)
    per_example = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=1)
    if example_weight is not None:
        per_example = per_example * example_weight
    weight = mask.to(per_example.dtype)
    return (per_example * weight).sum() / weight.sum().clamp(min=1)


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy that is zero when every target in the batch is ignored."""
    if int((target != -100).sum().item()) == 0:
        return logits.sum() * 0
    return F.cross_entropy(logits, target, weight=class_weight, ignore_index=-100)


def decode_ternary(
    logits: torch.Tensor,
    major_threshold: float | None = None,
) -> torch.Tensor:
    """Decode 3-way logits. Optional Major gate: P(Major) >= t → Major, else argmax of None/Minor."""
    probs = torch.softmax(logits.float(), dim=-1)
    major_index = len(TERNARY) - 1
    pred = probs.argmax(dim=-1)
    if major_threshold is None:
        return pred
    major_hit = probs[:, major_index] >= float(major_threshold)
    other = probs[:, :major_index].argmax(dim=-1)
    return torch.where(major_hit, torch.full_like(pred, major_index), other)


def masked_focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weight: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Class-weighted focal CE; zero when every target in the batch is ignored."""
    if int((target != -100).sum().item()) == 0:
        return logits.sum() * 0
    log_probs = F.log_softmax(logits, dim=-1)
    gather_target = target.clamp(min=0)
    log_pt = log_probs.gather(1, gather_target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    focal = (1.0 - pt).clamp(min=0.0).pow(float(gamma))
    loss = -focal * log_pt
    if class_weight is not None:
        weight = class_weight.gather(0, gather_target.clamp(max=class_weight.numel() - 1))
        loss = loss * weight
    keep = target != -100
    return loss[keep].mean()
