#!/usr/bin/env python3
"""Selective Evidence Lift with user-cluster bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from .metrics import selective_evidence_lift


def cluster_bootstrap(rows: list[dict], n_boot: int, seed: int) -> list[float]:
    by_user: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_user[str(row["user_id"])].append(
            (float(row["full_score"]) - float(row["target_deleted_score"]))
            - (float(row["full_score"]) - float(row["control_deleted_score"]))
        )
    users = sorted(by_user)
    if not users:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sampled = rng.choice(users, size=len(users), replace=True)
        values = [value for user in sampled for value in by_user[user]]
        draws[index] = np.mean(values)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    metrics = selective_evidence_lift(
        [float(row["full_score"]) for row in rows],
        [float(row["target_deleted_score"]) for row in rows],
        [float(row["control_deleted_score"]) for row in rows],
    )
    metrics["evidence_lift_ci95"] = cluster_bootstrap(
        rows, args.bootstrap_resamples, args.seed
    )
    metrics["pairs"] = len(rows)
    metrics["users"] = len({str(row["user_id"]) for row in rows})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
