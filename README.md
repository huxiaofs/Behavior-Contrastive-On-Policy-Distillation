# Behavior-Contrastive On-Policy Distillation

Code for **Behavior-Contrastive On-Policy Distillation (BC-OPD)**, a framework
for prospective-interest discovery that combines set-level on-policy
distillation with behavior-contrastive reward calibration.

The student samples complete, unordered interest sets. A frozen teacher scores
each set, while matched history edits test whether every candidate responds to
its supporting user behavior. The resulting set value and behavioral-grounding
signal are combined into a group-relative reward for on-policy optimization.

## Setup

```bash
pip install -r requirements.txt
```

Edit model paths and hyperparameters in `config/bcopd.json`. No raw user data,
model checkpoints, or API credentials are included.

## Layout

```text
config/          # paper configuration and model paths
data/example/    # anonymized inputs for a CPU-only reward demo
src/data/        # target and matched-control history editing
src/rollout/     # student rollout sampling and interest parsing
src/teacher/     # frozen-teacher set and candidate scoring
src/reward/      # BC-OPD reward construction
src/training/    # sequence-level OPD objective and trainer hooks
src/evaluation/  # exact-match, evidence-response, and semantic evaluation
tests/           # unit tests for the central method invariants
```

## Usage

Run all commands from the repository root.

```bash
# 1. Sample G interest-set rollouts from the current student policy
python -m src.rollout.sample \
  --config config/bcopd.json \
  --input /path/to/user_histories.jsonl \
  --output output/rollouts.jsonl

# 2. Construct target and matched-control history edits
python -m src.data.build_matched_edits \
  --input /path/to/mapped_histories.jsonl \
  --output output/edit_pairs.jsonl \
  --max-control-gap-days 90

# 3. Score complete sets and edited candidates with the frozen teacher
python -m src.teacher.score \
  --config config/bcopd.json \
  --rollouts output/rollouts.jsonl \
  --edits output/edit_pairs.jsonl \
  --output output/teacher_scored_rollouts.jsonl

# 4. Build the group-normalized one-sided BC-OPD reward
python -m src.reward.build \
  --input output/teacher_scored_rollouts.jsonl \
  --output output/bcopd_rollouts.jsonl \
  --lambda-bc 0.1

# 5. Cache rollout-policy and frozen-SFT token log probabilities
torchrun --nproc_per_node=6 -m src.training.reference_logps \
  --model /path/to/rollout_policy \
  --rollouts output/bcopd_rollouts.jsonl \
  --system-prompt data/system_prompt.txt \
  --output-dir output/old_policy_logps

torchrun --nproc_per_node=6 -m src.training.reference_logps \
  --model /path/to/sft_reference \
  --rollouts output/bcopd_rollouts.jsonl \
  --system-prompt data/system_prompt.txt \
  --output-dir output/sft_reference_logps
```

Integrate `src.training.group_opd` into an existing supervised trainer using
the minimal hooks in `src.training.integration`. The total objective is

```text
L = L_SFT + lambda_OPD * L_OPD + beta_KL * KL(policy || SFT reference).
```

## CPU-only reward demo

```bash
python -m src.data.build_matched_edits \
  --input data/example/mapped_histories.jsonl \
  --output output/example_edits.jsonl

python -m src.reward.build \
  --input data/example/teacher_scored_rollouts.jsonl \
  --output output/example_rewards.jsonl \
  --lambda-bc 0.1
```

## Evaluation

```bash
python -m src.evaluation.exact_match \
  --input /path/to/exact_match_eval.jsonl --cutoff 8

python -m src.evaluation.evidence \
  --input /path/to/evidence_eval.jsonl \
  --bootstrap-resamples 10000

python -m src.evaluation.semantic_judge \
  --input /path/to/semantic_eval.jsonl \
  --output output/semantic_judgments.jsonl \
  --model gemini-2.5-pro
```

The semantic judge uses the official Gemini REST API and reads
`GEMINI_API_KEY` from the environment. It is blind to method identity.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover matched-control construction, permutation invariance, invalid
candidate handling, one-sided reward calibration, sequence-level clipping, and
the manuscript's exact-match and evidence-response metrics.

Additional implementation notes and the paper-to-code map are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## License

This project is released under the [Apache License 2.0](LICENSE).
