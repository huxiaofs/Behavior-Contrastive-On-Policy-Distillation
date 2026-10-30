#!/usr/bin/env python3
"""Summarize blind semantic-judge outputs by system."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    metrics = defaultdict(lambda: defaultdict(list))
    with open(args.input, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            judgment = row["judgment"]
            for system, candidate_ids in row["system_candidate_ids"].items():
                valid_ids = [cid for cid in candidate_ids if cid in judgment]
                items = [judgment[cid] for cid in valid_ids]
                supports = [max(0, min(2, int(item.get("support", 0)))) for item in items]
                clusters = [
                    str(item.get("cluster", cid))
                    for item, cid in zip(items, valid_ids)
                ]
                n = len(supports)
                duplicate_pairs = sum(
                    clusters[i] == clusters[j]
                    for i in range(n)
                    for j in range(i + 1, n)
                )
                total_pairs = n * (n - 1) / 2
                metrics[system]["judge_support"].append(
                    sum(supports) / (2 * n) if n else 0.0
                )
                metrics[system]["strong_precision"].append(
                    sum(value == 2 for value in supports) / n if n else 0.0
                )
                metrics[system]["strong_hit"].append(float(any(value == 2 for value in supports)))
                metrics[system]["redundancy"].append(
                    duplicate_pairs / total_pairs if total_pairs else 0.0
                )

    result = {
        system: {
            name: sum(values) / len(values) if values else 0.0
            for name, values in system_metrics.items()
        }
        for system, system_metrics in metrics.items()
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
