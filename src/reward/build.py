#!/usr/bin/env python3
"""Construct paper-aligned BC-OPD rewards for student rollout groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bcopd import CandidateScore, build_group_rewards, group_by_user


def parse_score(item: dict) -> CandidateScore:
    return CandidateScore(
        interest=str(item["interest"]),
        target_value=float(item.get("target_value", 0.0)),
        control_value=float(item.get("control_value", 0.0)),
        valid=bool(item.get("valid", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambda-bc", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    groups = group_by_user(rows)
    output_rows: list[dict] = []
    valid_candidates = total_candidates = 0
    for user_id in sorted(groups):
        group = groups[user_id]
        teacher_values = [float(row["teacher_set_value"]) for row in group]
        candidate_groups = [
            [parse_score(item) for item in row.get("candidate_scores", [])]
            for row in group
        ]
        results = build_group_rewards(
            teacher_values,
            candidate_groups,
            lambda_bc=args.lambda_bc,
            epsilon=args.epsilon,
        )
        for row, scores, result in zip(group, candidate_groups, results):
            output_rows.append(
                {
                    **row,
                    "teacher_value_z": result.normalized_teacher_value,
                    "behavior_score": result.behavior_score,
                    "behavior_score_z": result.normalized_behavior_score,
                    "reward": result.reward,
                    "advantage": result.advantage,
                }
            )
            valid_candidates += sum(score.valid for score in scores)
            total_candidates += len(scores)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "users": len(groups),
                "rollouts": len(output_rows),
                "lambda_bc": args.lambda_bc,
                "valid_candidate_fraction": (
                    valid_candidates / total_candidates if total_candidates else 0.0
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

