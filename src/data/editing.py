"""Deterministic target and matched-control history editing.

Candidate--event relevance is assumed to come from a fixed canonical mapping or
a frozen high-confidence semantic matcher.  This module performs only the
deterministic matching and editing step described in Section 4.2 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class BehaviorEvent:
    event_id: str
    timestamp: float
    source: str
    action: str
    text: str = ""


@dataclass(frozen=True)
class CandidateEdit:
    interest: str
    target_event_ids: tuple[str, ...]
    control_event_ids: tuple[str, ...]
    valid: bool
    reason: str = "ok"


def _profile(event: BehaviorEvent) -> tuple[str, str]:
    return event.source, event.action


def build_matched_edit(
    events: Sequence[BehaviorEvent],
    interest: str,
    evidence_event_ids: Iterable[str],
    related_event_ids: Iterable[str] = (),
    max_control_gap_days: float | None = None,
) -> CandidateEdit:
    """Match every evidence event to one unrelated event of the same profile.

    Controls are selected without replacement by minimum temporal distance.
    Events related to the candidate, including target evidence, are ineligible.
    The complete pair is invalid if any target event cannot be matched.
    """

    by_id: Mapping[str, BehaviorEvent] = {event.event_id: event for event in events}
    target_ids = tuple(dict.fromkeys(str(x) for x in evidence_event_ids))
    if not target_ids:
        return CandidateEdit(interest, (), (), False, "no_evidence")
    if any(event_id not in by_id for event_id in target_ids):
        return CandidateEdit(interest, target_ids, (), False, "missing_evidence_event")

    excluded = set(target_ids) | {str(x) for x in related_event_ids}
    available = {event.event_id for event in events if event.event_id not in excluded}

    # Scarce profiles are matched first to avoid a greedy ordering artifact.
    ordered_targets = sorted(
        (by_id[event_id] for event_id in target_ids),
        key=lambda event: (
            sum(_profile(candidate) == _profile(event) for candidate in events if candidate.event_id in available),
            event.timestamp,
            event.event_id,
        ),
    )
    controls: list[str] = []
    max_gap_seconds = None
    if max_control_gap_days is not None and max_control_gap_days > 0:
        max_gap_seconds = max_control_gap_days * 86400.0

    for target in ordered_targets:
        candidates = [
            by_id[event_id]
            for event_id in available
            if _profile(by_id[event_id]) == _profile(target)
        ]
        if max_gap_seconds is not None:
            candidates = [
                event
                for event in candidates
                if abs(event.timestamp - target.timestamp) <= max_gap_seconds
            ]
        if not candidates:
            return CandidateEdit(
                interest, target_ids, tuple(controls), False, "control_match_failed"
            )
        chosen = min(
            candidates,
            key=lambda event: (abs(event.timestamp - target.timestamp), event.event_id),
        )
        controls.append(chosen.event_id)
        available.remove(chosen.event_id)

    return CandidateEdit(interest, target_ids, tuple(controls), True)


def remove_events(
    events: Sequence[BehaviorEvent], event_ids: Iterable[str]
) -> list[BehaviorEvent]:
    """Return a history with the selected event identifiers removed."""

    removed = {str(x) for x in event_ids}
    return [event for event in events if event.event_id not in removed]

