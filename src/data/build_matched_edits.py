#!/usr/bin/env python3
"""Build target-deleted and matched-control-deleted histories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .editing import BehaviorEvent, build_matched_edit, remove_events


def render_history(events: list[BehaviorEvent]) -> str:
    ordered = sorted(events, key=lambda event: (event.timestamp, event.event_id))
    return "\n".join(
        f"[{event.timestamp:g}] [{event.source}/{event.action}] {event.text}"
        for event in ordered
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-control-gap-days", type=float, default=90.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"candidates": 0, "valid": 0, "invalid": 0}

    with open(args.input, encoding="utf-8") as source, output.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            events = [BehaviorEvent(**event) for event in row.get("events", [])]
            for candidate in row.get("candidates", []):
                edit = build_matched_edit(
                    events=events,
                    interest=str(candidate["interest"]),
                    evidence_event_ids=candidate.get("evidence_event_ids", []),
                    related_event_ids=candidate.get("related_event_ids", []),
                    max_control_gap_days=args.max_control_gap_days,
                )
                record = {
                    "user_id": str(row["user_id"]),
                    **asdict(edit),
                    "full_history": render_history(events),
                    "target_deleted_history": render_history(
                        remove_events(events, edit.target_event_ids)
                    ),
                    "control_deleted_history": render_history(
                        remove_events(events, edit.control_event_ids)
                    ),
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary["candidates"] += 1
                summary["valid" if edit.valid else "invalid"] += 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

