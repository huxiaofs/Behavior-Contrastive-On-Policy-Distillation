"""Metrics used by the manuscript's frozen offline protocol."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence


def normalize_interest(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    return re.sub(r"[\s\W_]+", "", text)


def canonical_set(values: Iterable[str], cutoff: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_interest(value)
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
        if cutoff is not None and len(result) >= cutoff:
            break
    return result


def exact_match_metrics(
    predictions: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
    futures: Sequence[Sequence[str]],
    cutoff: int = 8,
) -> dict[str, float]:
    """Macro Ref-R@K, Ref-P@K, and user-level Future Exact Hit@K."""

    if not (len(predictions) == len(references) == len(futures)):
        raise ValueError("predictions, references, and futures must align by user")
    recall, precision, future_hit = [], [], []
    for pred, ref, future in zip(predictions, references, futures):
        pred_set = set(canonical_set(pred, cutoff))
        ref_set = set(canonical_set(ref))
        future_set = set(canonical_set(future))
        overlap = len(pred_set & ref_set)
        recall.append(overlap / len(ref_set) if ref_set else 0.0)
        precision.append(overlap / len(pred_set) if pred_set else 0.0)
        future_hit.append(float(bool(pred_set & future_set)))
    n = len(predictions)
    if n == 0:
        return {"reference_recall": 0.0, "reference_precision": 0.0, "future_exact_hit": 0.0}
    return {
        "reference_recall": sum(recall) / n,
        "reference_precision": sum(precision) / n,
        "future_exact_hit": sum(future_hit) / n,
    }


def selective_evidence_lift(
    full_scores: Sequence[float],
    target_deleted_scores: Sequence[float],
    control_deleted_scores: Sequence[float],
) -> dict[str, float]:
    if not (len(full_scores) == len(target_deleted_scores) == len(control_deleted_scores)):
        raise ValueError("score vectors must have equal length")
    if not full_scores:
        return {"target_response": 0.0, "control_response": 0.0, "evidence_lift": 0.0}
    target = [f - t for f, t in zip(full_scores, target_deleted_scores)]
    control = [f - c for f, c in zip(full_scores, control_deleted_scores)]
    n = len(target)
    return {
        "target_response": sum(target) / n,
        "control_response": sum(control) / n,
        "evidence_lift": sum(t - c for t, c in zip(target, control)) / n,
    }
