#!/usr/bin/env python3
"""Score rollout sets and behavior-edited candidates with a frozen teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


SET_SYSTEM = """You evaluate a complete set of prospective interests for recommendation. A good set is plausible for the current user, prospective rather than already expressed, and non-redundant under a finite exploration budget. Reply with exactly one of the two labels requested."""

CANDIDATE_SYSTEM = """You evaluate whether one prospective interest is worth exploring for the current user. A worthy direction should have user-specific behavioral support; a direction suggested only by generic world knowledge is insufficient. Reply with exactly one of the two labels requested."""


class VerbalizerScorer:
    def __init__(self, model_path: str, device: str, positive: str, negative: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.device = device
        self.labels = {
            1: self.tokenizer(positive, add_special_tokens=False)["input_ids"],
            -1: self.tokenizer(negative, add_special_tokens=False)["input_ids"],
        }

    def prompt(self, system: str, user: str) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    @torch.inference_mode()
    def score(self, prompt_ids: list[int]) -> float:
        total = 0.0
        for sign, label_ids in self.labels.items():
            sequence = torch.tensor([prompt_ids + label_ids], device=self.device)
            start = len(prompt_ids) - 1
            positions = torch.arange(start, start + len(label_ids), device=self.device)
            labels = sequence[0, len(prompt_ids) :]
            # Qwen exposes the backbone and LM head separately. Projecting only
            # the label positions avoids materializing vocabulary logits for a
            # long behavior history. The fallback keeps other causal LMs usable.
            backbone = getattr(self.model, "model", None)
            lm_head = getattr(self.model, "lm_head", None)
            if backbone is not None and lm_head is not None:
                hidden = backbone(
                    input_ids=sequence, use_cache=False, return_dict=True
                ).last_hidden_state[0, positions]
                logits = lm_head(hidden).float()
            else:
                logits = self.model(input_ids=sequence, use_cache=False).logits[0, positions].float()
            logp = F.log_softmax(logits, dim=-1).gather(
                1, labels[:, None]
            ).sum()
            total += sign * float(logp.item())
        return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--edits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--positive-label", default="值得")
    parser.add_argument("--negative-label", default="不值得")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    scorer = VerbalizerScorer(
        config["teacher_model"], args.device, args.positive_label, args.negative_label
    )
    edits = {}
    with open(args.edits, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                edits[(str(row["user_id"]), str(row["interest"]))] = row

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.rollouts, encoding="utf-8") as source, output.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row["user_id"])
            interests = list(dict.fromkeys(str(x) for x in row.get("interests", [])))
            set_user = (
                f"User history:\n{row['user_full']}\n\n"
                f"Candidate interest set:\n{json.dumps(interests, ensure_ascii=False)}"
            )
            teacher_set_value = scorer.score(scorer.prompt(SET_SYSTEM, set_user))
            candidate_scores = []
            for interest in interests:
                edit = edits.get((uid, interest))
                if not edit or not edit.get("valid"):
                    candidate_scores.append({"interest": interest, "valid": False})
                    continue
                target_user = (
                    f"User history:\n{edit['target_deleted_history']}\n\n"
                    f"Candidate prospective interest: {interest}"
                )
                control_user = (
                    f"User history:\n{edit['control_deleted_history']}\n\n"
                    f"Candidate prospective interest: {interest}"
                )
                candidate_scores.append(
                    {
                        "interest": interest,
                        "valid": True,
                        "target_value": scorer.score(
                            scorer.prompt(CANDIDATE_SYSTEM, target_user)
                        ),
                        "control_value": scorer.score(
                            scorer.prompt(CANDIDATE_SYSTEM, control_user)
                        ),
                    }
                )
            sink.write(
                json.dumps(
                    {
                        **row,
                        "teacher_set_value": teacher_set_value,
                        "candidate_scores": candidate_scores,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()


if __name__ == "__main__":
    main()
