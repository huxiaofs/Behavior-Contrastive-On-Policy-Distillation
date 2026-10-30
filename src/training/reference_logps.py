#!/usr/bin/env python3
"""Precompute behavior-policy token log-probabilities for clipped group OPD."""

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def assistant_mask(input_ids, tokenizer):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=30000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    with open(args.system_prompt, encoding="utf-8") as handle:
        system_prompt = handle.read()
    with open(args.rollouts, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    local_rows = rows[rank::world_size]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"rank-{rank:02d}.reference.jsonl")

    with open(output_path, "w", encoding="utf-8") as output:
        for local_idx, row in enumerate(local_rows):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["user_full"]},
                {"role": "assistant", "content": row["response"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ) + (tokenizer.pad_token or "")
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            if ids.numel() > args.max_length:
                raise ValueError(
                    f"row {row['user_id']}/{row['rollout_id']} has {ids.numel()} tokens"
                )
            mask = assistant_mask(ids, tokenizer)
            keep = (mask[1:] == 1).nonzero(as_tuple=False).squeeze(1)
            if keep.numel() == 0:
                raise ValueError(f"no assistant tokens for {row['user_id']}/{row['rollout_id']}")
            labels = ids[keep + 1].to(device)
            with torch.inference_mode():
                result = model(
                    input_ids=ids.unsqueeze(0).to(device),
                    use_cache=False,
                    logits_to_keep=keep.to(device),
                )
                logits = result.logits[0].float()
                logps = F.log_softmax(logits, dim=-1).gather(
                    1, labels.unsqueeze(1)
                ).squeeze(1)
            record = {
                "global_index": rank + local_idx * world_size,
                "user_id": row["user_id"],
                "rollout_id": row["rollout_id"],
                "token_count": int(logps.numel()),
                "sequence_length": int(ids.numel()),
                "old_logps": logps.cpu().tolist(),
                "mean_old_logp": float(logps.mean().item()),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if local_idx % 10 == 0:
                print(
                    f"rank={rank} row={local_idx + 1}/{len(local_rows)} "
                    f"seq={ids.numel()} target={logps.numel()}",
                    flush=True,
                )
    dist.barrier()
    if rank == 0:
        with open(os.path.join(args.output_dir, "COMPLETE"), "w", encoding="utf-8") as handle:
            handle.write(f"world_size={world_size} rows={len(rows)}\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
