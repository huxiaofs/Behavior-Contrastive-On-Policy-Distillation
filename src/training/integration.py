"""Minimal integration hooks for interleaved SFT and BC-OPD training.

The host trainer remains responsible for the ordinary supervised forward pass,
optimizer, scheduler, distributed wrapping, and checkpointing.
"""

from __future__ import annotations

import argparse

from .group_opd import GroupOPDLoader, compute_group_opd_forward_backward


def add_bcopd_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--opd-data")
    parser.add_argument("--opd-reference-dir")
    parser.add_argument("--opd-kl-reference-dir")
    parser.add_argument("--opd-system-prompt")
    parser.add_argument("--opd-advantage-key", default="advantage")
    parser.add_argument("--opd-lambda", type=float, default=0.1)
    parser.add_argument("--opd-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--opd-kl-beta", type=float, default=0.02)
    parser.add_argument("--opd-seed", type=int, default=20261012)


def build_loader(args, tokenizer, device, rank: int, world_size: int):
    if not args.opd_data:
        return None
    if not args.opd_reference_dir or not args.opd_system_prompt:
        raise ValueError("OPD data requires reference log-probabilities and a system prompt")
    with open(args.opd_system_prompt, encoding="utf-8") as handle:
        system = handle.read()
    return GroupOPDLoader(
        args.opd_data,
        args.opd_reference_dir,
        tokenizer,
        system,
        args.opd_advantage_key,
        device=device,
        rank=rank,
        world_size=world_size,
        seed=args.opd_seed,
        max_length=min(args.max_length, 30000),
        schedule_mode="full_padded",
        kl_reference_dir=args.opd_kl_reference_dir,
    )


def backward_bcopd(model, loader, args):
    """Call after the SFT backward pass and before global gradient clipping."""

    if loader is None:
        return None, {}
    entry = loader.next_batch()
    return compute_group_opd_forward_backward(
        model,
        entry,
        opd_lambda=args.opd_lambda,
        clip_epsilon=args.opd_clip_epsilon,
        kl_beta=args.opd_kl_beta,
    )
