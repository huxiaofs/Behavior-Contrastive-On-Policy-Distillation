# -*- coding: utf-8 -*-
"""Sequence-level clipped on-policy distillation for unordered interest sets.

Each student rollout receives one group-normalized set reward.  The same scalar
advantage is applied to the complete response.  A length-normalized sequence
probability ratio is computed against the frozen policy that generated the
rollout.  This preserves set-level supervision without inventing an
order-sensitive token reward.
"""

import glob
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn.functional as F


def _assistant_mask(input_ids, tokenizer):
    start = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    if not end:
        end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    ids = input_ids.tolist()
    mask = [0] * len(ids)
    cursor = 0
    while cursor <= len(ids) - len(start):
        if ids[cursor:cursor + len(start)] != start:
            cursor += 1
            continue
        content_start = cursor + len(start)
        stop = content_start
        while stop <= len(ids) - len(end) and ids[stop:stop + len(end)] != end:
            stop += 1
        if stop > len(ids) - len(end):
            stop = len(ids)
        for idx in range(content_start, stop):
            mask[idx] = 1
        cursor = stop + len(end)
    if mask:
        mask[-1] = 0
    return torch.tensor(mask, dtype=torch.bool)


class GroupOPDLoader:
    """Deterministically shard the same rollout pool across distributed ranks."""

    def __init__(self, jsonl_path, reference_dir, tokenizer, system, advantage_key,
                 device, rank=0, world_size=1, seed=0, max_length=30000,
                 schedule_mode="legacy", kl_reference_dir=None):
        if advantage_key not in ("value_advantage", "ba_advantage", "advantage"):
            raise ValueError(f"Unsupported OPD advantage key: {advantage_key}")
        if schedule_mode not in ("legacy", "full_padded"):
            raise ValueError(f"Unsupported OPD schedule mode: {schedule_mode}")
        with open(jsonl_path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        references = {}
        for path in sorted(glob.glob(os.path.join(reference_dir, "rank-*.reference.jsonl"))):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (str(row["user_id"]), int(row["rollout_id"]))
                    if key in references:
                        raise ValueError(f"Duplicate OPD reference key: {key}")
                    references[key] = row
        if len(references) != len(rows):
            raise ValueError(
                f"OPD rollout/reference mismatch: {len(rows)} vs {len(references)}"
            )
        kl_references = references
        if kl_reference_dir:
            kl_references = {}
            for path in sorted(glob.glob(os.path.join(kl_reference_dir, "rank-*.reference.jsonl"))):
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        key = (str(item["user_id"]), int(item["rollout_id"]))
                        if key in kl_references:
                            raise ValueError(f"Duplicate KL reference key: {key}")
                        kl_references[key] = item
            if len(kl_references) != len(rows):
                raise ValueError(
                    f"KL-reference rollout mismatch: {len(rows)} vs {len(kl_references)}"
                )
        # A GRPO-style global batch should contain complete prompt groups so
        # the normalized advantages sum to zero at every optimizer step.  With
        # six ranks and 2--4 valid rollouts per user, exact six-way batches are
        # formed as 4+2, 3+3, or 2+2+2 groups.  Both reward arms receive the
        # identical schedule.
        rng = random.Random(seed)
        by_user = defaultdict(list)
        for row in rows:
            by_user[str(row["user_id"])].append(row)
        by_size = defaultdict(list)
        for user_rows in by_user.values():
            user_rows.sort(key=lambda item: int(item["rollout_id"]))
            by_size[len(user_rows)].append(user_rows)
        for groups in by_size.values():
            rng.shuffle(groups)

        global_batches = []
        if world_size == 6 and set(by_size).issubset({2, 3, 4}):
            if schedule_mode == "legacy":
                while by_size[4] and by_size[2]:
                    global_batches.append(by_size[4].pop() + by_size[2].pop())
                while len(by_size[3]) >= 2:
                    global_batches.append(by_size[3].pop() + by_size[3].pop())
                while len(by_size[2]) >= 3:
                    global_batches.append(
                        by_size[2].pop() + by_size[2].pop() + by_size[2].pop()
                    )
            else:
                # Preserve every complete user group.  A four-rollout group is
                # padded to six ranks with two zero-weight graph participants.
                # The four active samples receive weight 6/4 so DDP's six-rank
                # averaging equals the mean gradient over the real group.
                while by_size[4]:
                    group = by_size[4].pop()
                    active = [dict(row, _opd_sample_weight=1.5) for row in group]
                    padding = [
                        dict(group[0], _opd_sample_weight=0.0, _opd_dummy=True),
                        dict(group[1], _opd_sample_weight=0.0, _opd_dummy=True),
                    ]
                    global_batches.append(active + padding)
                while len(by_size[3]) >= 2:
                    rows_3 = by_size[3].pop() + by_size[3].pop()
                    global_batches.append([
                        dict(row, _opd_sample_weight=1.0) for row in rows_3
                    ])
                while len(by_size[2]) >= 3:
                    rows_2 = by_size[2].pop() + by_size[2].pop() + by_size[2].pop()
                    global_batches.append([
                        dict(row, _opd_sample_weight=1.0) for row in rows_2
                    ])
            rng.shuffle(global_batches)
            if not global_batches:
                raise ValueError("Unable to form any complete six-way OPD group batch")
            self.rows = [batch[rank] for batch in global_batches]
        else:
            rng.shuffle(rows)
            self.rows = rows[rank::world_size]
            global_batches = None
        self.references = references
        self.kl_references = kl_references
        self.tokenizer = tokenizer
        self.system = system
        self.advantage_key = advantage_key
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.max_length = max_length
        self.global_size = len(rows)
        self.scheduled_steps = len(self.rows)
        self.complete_group_batches = global_batches is not None
        self.schedule_mode = schedule_mode
        if global_batches is not None:
            active = [
                row for batch in global_batches for row in batch
                if float(row.get("_opd_sample_weight", 1.0)) > 0
            ]
            self.scheduled_active_rows = len(active)
            self.scheduled_active_users = len({str(row["user_id"]) for row in active})
            self.scheduled_dummy_slots = sum(
                float(row.get("_opd_sample_weight", 1.0)) == 0
                for batch in global_batches for row in batch
            )
        else:
            self.scheduled_active_rows = len(rows)
            self.scheduled_active_users = len({str(row["user_id"]) for row in rows})
            self.scheduled_dummy_slots = 0
        self.idx = 0

    def __len__(self):
        return len(self.rows)

    def next_batch(self):
        if self.idx >= len(self.rows):
            raise StopIteration(
                f"OPD rank {self.rank} exhausted after {len(self.rows)} rows"
            )
        row = self.rows[self.idx]
        self.idx += 1
        key = (str(row["user_id"]), int(row["rollout_id"]))
        reference = self.references[key]
        kl_reference = self.kl_references[key]
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": row["user_full"]},
            {"role": "assistant", "content": row["response"]},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ) + (self.tokenizer.pad_token or "")
        input_ids = self.tokenizer(
            text, return_tensors="pt", padding=False, truncation=False,
            add_special_tokens=False,
        )["input_ids"][0]
        if input_ids.numel() > self.max_length:
            raise ValueError(
                f"OPD sequence {key} has {input_ids.numel()} tokens > {self.max_length}"
            )
        mask = _assistant_mask(input_ids, self.tokenizer)
        keep = (mask[1:] == 1).nonzero(as_tuple=False).squeeze(1)
        old_logps = torch.tensor(reference["old_logps"], dtype=torch.float32)
        reference_logps = torch.tensor(kl_reference["old_logps"], dtype=torch.float32)
        if keep.numel() != old_logps.numel():
            raise ValueError(
                f"OPD token alignment failed for {key}: {keep.numel()} vs "
                f"{old_logps.numel()}"
            )
        if keep.numel() != reference_logps.numel():
            raise ValueError(
                f"OPD KL-reference alignment failed for {key}: {keep.numel()} vs "
                f"{reference_logps.numel()}"
            )
        return {
            "input_ids": input_ids.to(self.device),
            "target_positions": keep.to(self.device),
            "target_labels": input_ids[keep + 1].to(self.device),
            "old_logps": old_logps.to(self.device),
            "reference_logps": reference_logps.to(self.device),
            "advantage": torch.tensor(
                0.0 if row.get("_opd_dummy") else float(row[self.advantage_key]),
                dtype=torch.float32, device=self.device
            ),
            "sample_weight": torch.tensor(
                float(row.get("_opd_sample_weight", 1.0)),
                dtype=torch.float32, device=self.device,
            ),
            "user_id": key[0],
            "rollout_id": key[1],
            "sequence_length": int(input_ids.numel()),
        }


def compute_group_opd_forward_backward(model, entry, opd_lambda=0.1,
                                       clip_epsilon=0.2, kl_beta=0.02):
    """Apply a response-level PPO/GRPO objective to one sampled interest set."""
    ids = entry["input_ids"]
    keep = entry["target_positions"]
    labels = entry["target_labels"]
    old_logps = entry["old_logps"]
    advantage = entry["advantage"]
    sample_weight = entry.get(
        "sample_weight", torch.tensor(1.0, dtype=torch.float32, device=ids.device)
    )

    out = model(
        input_ids=ids.unsqueeze(0),
        attention_mask=None,
        labels=None,
        use_cache=False,
    )
    hidden = out.logits[0]
    logits = model.lm_head(hidden[keep]).float()
    current_logps = F.log_softmax(logits, dim=-1).gather(
        1, labels.unsqueeze(1)
    ).squeeze(1)
    if current_logps.shape != old_logps.shape:
        raise ValueError(
            f"OPD current/reference shape mismatch: {current_logps.shape} vs {old_logps.shape}"
        )

    token_log_ratio = (current_logps - old_logps).clamp(-20.0, 20.0)
    log_ratio = token_log_ratio.mean()
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped_objective = ratio * advantage
    clipped_objective = clipped_ratio * advantage
    policy_loss = -torch.minimum(unclipped_objective, clipped_objective)

    # Non-negative sampled KL estimator with the frozen SFT reference policy.
    reference_logps = entry.get("reference_logps", old_logps)
    reverse_log_ratio = (reference_logps - current_logps).clamp(-20.0, 20.0)
    approx_kl = (reverse_log_ratio.exp() - reverse_log_ratio - 1.0).mean()
    unweighted_loss = policy_loss + kl_beta * approx_kl
    loss = sample_weight * unweighted_loss
    (opd_lambda * loss).backward()

    with torch.no_grad():
        clipped = ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float()
        return loss.detach(), {
            "loss_sum": loss.detach(),
            "policy_loss_sum": (sample_weight * policy_loss).detach(),
            "kl_sum": (sample_weight * approx_kl).detach(),
            "ratio_sum": (sample_weight * ratio).detach(),
            "abs_log_ratio_sum": (sample_weight * log_ratio.abs()).detach(),
            "clip_fraction_sum": (sample_weight * clipped).detach(),
            "advantage_sum": (sample_weight * advantage).detach(),
            "abs_advantage_sum": (sample_weight * advantage.abs()).detach(),
            "token_count": sample_weight * torch.tensor(
                float(current_logps.numel()), device=ids.device
            ),
            "sequence_tokens": sample_weight * torch.tensor(
                float(entry["sequence_length"]), device=ids.device
            ),
            "count": sample_weight.detach(),
        }
