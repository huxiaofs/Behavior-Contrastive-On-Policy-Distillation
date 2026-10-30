#!/usr/bin/env python3
"""Blind semantic evaluation with the official Gemini REST API."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

SYSTEM = """You are an independent evaluator for prospective-interest discovery. Judge every candidate only from the user's history and future behavior. Do not infer the source system. Assign support=2 for clear user-specific support, support=1 for plausible but weak support, and support=0 for no support. Assign the same cluster to candidates that are semantic substitutes. Return JSON only."""


def normalize(text: str) -> str:
    return "".join(str(text).strip().lower().split())


def parse_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("judge output must be a JSON object")
    return value


def call_gemini(prompt: str, model: str, api_key: str) -> dict:
    import requests

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": SYSTEM + "\n\n" + prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
    }
    delay = 1.0
    for attempt in range(5):
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=body,
            timeout=300,
        )
        if response.ok:
            data = response.json()
            parts = data["candidates"][0]["content"]["parts"]
            return parse_json("".join(part.get("text", "") for part in parts))
        if attempt == 4:
            raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:500]}")
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def build_prompt(row: dict, seed: int) -> tuple[str, dict[str, str], dict[str, list[str]]]:
    canonical: dict[str, str] = {}
    membership: dict[str, list[str]] = {}
    for system, interests in row["systems"].items():
        membership[system] = []
        for interest in interests:
            key = normalize(interest)
            canonical.setdefault(key, str(interest))
            membership[system].append(key)
    interests = list(canonical.values())
    random.Random(seed).shuffle(interests)
    id_to_interest = {f"C{index + 1:02d}": interest for index, interest in enumerate(interests)}
    id_by_key = {normalize(interest): cid for cid, interest in id_to_interest.items()}
    system_ids = {
        system: [id_by_key[key] for key in keys if key in id_by_key]
        for system, keys in membership.items()
    }
    future = "\n".join(f"- {event}" for event in row.get("future_events", []))
    candidates = "\n".join(f"- {cid}: {interest}" for cid, interest in id_to_interest.items())
    prompt = f"""User history:
{row.get('history', '')}

Future behavior:
{future}

Candidate prospective interests:
{candidates}

Return one entry per candidate using this schema:
{{"C01": {{"support": 0, "cluster": "G01"}}}}"""
    return prompt, id_to_interest, system_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("set GEMINI_API_KEY before running semantic evaluation")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(str(json.loads(line)["user_id"]))

    with open(args.input, encoding="utf-8") as source, output.open("a", encoding="utf-8") as sink:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid in completed:
                continue
            prompt, id_to_interest, system_ids = build_prompt(row, args.seed + index)
            judgment = call_gemini(prompt, args.model, api_key)
            sink.write(
                json.dumps(
                    {
                        "user_id": uid,
                        "candidates": id_to_interest,
                        "system_candidate_ids": system_ids,
                        "judgment": judgment,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()


if __name__ == "__main__":
    main()
