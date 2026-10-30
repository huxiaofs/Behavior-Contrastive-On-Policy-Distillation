"""Permutation-invariant behavior-contrastive rollout rewards."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CandidateScore:
    interest: str
    target_value: float = 0.0
    control_value: float = 0.0
    valid: bool = False


@dataclass(frozen=True)
class RolloutReward:
    teacher_value: float
    behavior_score: float
    normalized_teacher_value: float
    normalized_behavior_score: float
    reward: float
    advantage: float


def behavior_contrast(score: CandidateScore) -> float:
    """Eq. 8: v(z, H_control) - v(z, H_target-deleted)."""

    if not score.valid:
        return 0.0
    return float(score.control_value) - float(score.target_value)


def set_behavior_score(scores: Sequence[CandidateScore]) -> float:
    """Eq. 9, retaining invalid candidates in the set denominator."""

    if not scores:
        return 0.0
    return sum(behavior_contrast(score) for score in scores) / len(scores)


def standardize(values: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    """Population standardization within one user's rollout group."""

    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = sqrt(variance)
    if std < epsilon:
        return [0.0] * len(values)
    return [(value - mean) / (std + epsilon) for value in values]


def build_group_rewards(
    teacher_values: Sequence[float],
    candidate_groups: Sequence[Sequence[CandidateScore]],
    lambda_bc: float = 0.1,
    epsilon: float = 1e-8,
) -> list[RolloutReward]:
    """Eq. 10--12 for all rollouts sampled for one user.

    Teacher values and behavior scores are standardized separately.  Only the
    positive part of the normalized behavior score calibrates the rollout.
    The combined reward is standardized once more to obtain the group-relative
    advantage used by the sequence-level OPD objective.
    """

    if len(teacher_values) != len(candidate_groups):
        raise ValueError("teacher_values and candidate_groups must have equal length")
    if not teacher_values:
        return []

    values = [float(value) for value in teacher_values]
    behavior = [set_behavior_score(scores) for scores in candidate_groups]
    value_z = standardize(values, epsilon)
    behavior_z = standardize(behavior, epsilon)
    rewards = [
        value + lambda_bc * max(contrast, 0.0)
        for value, contrast in zip(value_z, behavior_z)
    ]
    advantages = standardize(rewards, epsilon)
    return [
        RolloutReward(v, b, vz, bz, reward, advantage)
        for v, b, vz, bz, reward, advantage in zip(
            values, behavior, value_z, behavior_z, rewards, advantages
        )
    ]


def group_by_user(rows: Iterable[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["user_id"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(row["rollout_id"]))
    return groups

