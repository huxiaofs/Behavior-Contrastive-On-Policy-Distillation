#!/usr/bin/env python3
"""Evaluate the three exact-match metrics reported in the main table."""

from __future__ import annotations

import argparse
import json

from .metrics import exact_match_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--cutoff", type=int, default=8)
    args = parser.parse_args()

    predictions, references, futures = [], [], []
    with open(args.input, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions.append(row.get("prediction", []))
            references.append(row.get("reference", []))
            futures.append(row.get("future", []))
    print(
        json.dumps(
            exact_match_metrics(predictions, references, futures, args.cutoff),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
