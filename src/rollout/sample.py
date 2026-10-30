#!/usr/bin/env python3
"""Sample prospective-interest sets from the current student policy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM = """You discover prospective user interests for recommendation. Infer directions that are not explicitly expressed in the history but may be accepted after exploration. Return at most eight concise interests as a JSON array and no other text."""


def normalize(text: str) -> str:
    return "".join(str(text).strip().lower().split())


def parse_interests(text: str, cutoff: int) -> list[str]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate)
    if fenced:
        candidate = fenced.group(1)
    values: list[str] = []
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            parsed = parsed.get("interests", [])
        if isinstance(parsed, list):
            values = [str(item) for item in parsed]
    except json.JSONDecodeError:
        for line in text.splitlines():
            if "兴趣扩展标签" in line or "prospective" in line.lower():
                value = line.split("：", 1)[-1].split(":", 1)[-1]
                values.extend(re.split(r"[,，、;；]", value))

    result, seen = [], set()
    for value in values:
        value = value.strip().strip('"\'')
        key = normalize(value)
        if value and key not in seen:
            result.append(value)
            seen.add(key)
        if len(result) >= cutoff:
            break
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-prompt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_path = config["student_model"]
    group_size = int(config.get("rollout_group_size", 4))
    cutoff = int(config.get("max_interests", 8))
    system = (
        Path(args.system_prompt).read_text(encoding="utf-8")
        if args.system_prompt
        else SYSTEM
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as source, output.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            history = str(row["history"])
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": history},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=float(config.get("temperature", 0.9)),
                    max_new_tokens=int(config.get("max_new_tokens", 256)),
                    num_return_sequences=group_size,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            for rollout_id, sequence in enumerate(generated):
                response = tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
                interests = parse_interests(response, cutoff)
                sink.write(
                    json.dumps(
                        {
                            "user_id": str(row["user_id"]),
                            "rollout_id": rollout_id,
                            "user_full": history,
                            "response": response,
                            "interests": interests,
                            "status": "ok" if interests else "parse_error",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()

