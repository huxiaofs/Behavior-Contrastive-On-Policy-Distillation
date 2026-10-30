# Reproducibility Notes

## Paper-to-code map

| Manuscript component | Implementation |
|---|---|
| Student set rollouts | `src/rollout/sample.py` |
| Candidate canonicalization and deduplication | `src/rollout/sample.py::parse_interests` |
| Target and matched-control edits | `src/data/editing.py`, `src/data/build_matched_edits.py` |
| Teacher set value and edited candidate values | `src/teacher/score.py` |
| Candidate contrast | `src/reward/bcopd.py::behavior_contrast` |
| Permutation-invariant set contrast | `src/reward/bcopd.py::set_behavior_score` |
| Separate within-user standardization | `src/reward/bcopd.py::standardize` |
| One-sided calibrated reward | `src/reward/bcopd.py::build_group_rewards` |
| Length-normalized sequence ratio | `src/training/objective.py::sequence_log_ratio` |
| Clipped set-level OPD objective | `src/training/objective.py`, `src/training/group_opd.py` |
| Frozen-SFT KL anchor | `src/training/group_opd.py` |
| Exact-match evaluation | `src/evaluation/exact_match.py` |
| Selective Evidence Lift | `src/evaluation/evidence.py` |
| Blind semantic evaluation | `src/evaluation/semantic_judge.py` |

## Data interfaces

### User histories for rollout sampling

```json
{"user_id": "u1", "history": "serialized user behavior history"}
```

### Mapped histories for behavior editing

Candidate--event links are produced by a fixed canonical dictionary or a frozen
high-confidence semantic matcher. Broad parent-category relations should be
excluded before this file is written.

```json
{
  "user_id": "u1",
  "events": [
    {
      "event_id": "e1",
      "timestamp": 1710000000,
      "source": "search",
      "action": "query",
      "text": "sports watch"
    }
  ],
  "candidates": [
    {
      "interest": "wearable accessories",
      "evidence_event_ids": ["e1"],
      "related_event_ids": ["e1"]
    }
  ]
}
```

`related_event_ids` contains every event that remains semantically related to
the candidate and is therefore ineligible as a matched control. The matching
code preserves event count, behavior source, action type, and temporal
proximity. Incomplete pairs are marked invalid and receive zero calibration
weight.

### Teacher-scored rollouts

```json
{
  "user_id": "u1",
  "rollout_id": 0,
  "response": "...",
  "interests": ["wearable accessories", "running nutrition"],
  "teacher_set_value": 1.7,
  "candidate_scores": [
    {
      "interest": "wearable accessories",
      "valid": true,
      "target_value": 0.2,
      "control_value": 1.1
    },
    {"interest": "running nutrition", "valid": false}
  ]
}
```

Invalid candidates stay in the set denominator. A rollout with no valid pair
receives only its teacher set value.

## Training protocol

1. Warm-start the student with supervised fine-tuning.
2. Freeze the current student as the rollout policy and sample `G=4` sets per user.
3. Evaluate the complete sets and their candidate-level history edits with one frozen teacher.
4. Normalize teacher value and behavior contrast separately inside each user's rollout group.
5. Add the positive part of normalized behavior contrast with `lambda_BC=0.1`.
6. Normalize the combined reward into a group-relative advantage.
7. Cache token log probabilities under both the rollout policy and frozen SFT reference.
8. Interleave one OPD backward pass with the supervised stream and refresh rollouts in the next round.

The default paper configuration is in `config/bcopd.json`. The model-specific
trainer remains responsible for distributed wrapping, the SFT data loader,
optimizer, gradient clipping, scheduler, and checkpointing.

## Determinism

- Canonicalization, matched-control selection, reward construction, and offline
  metrics are deterministic given their inputs and seed.
- Student rollouts are stochastic and should record model revision, decoding
  parameters, random seed, and library versions.
- Confidence intervals resample users rather than individual interests.
- The semantic judge merges and shuffles candidates before calling the API, so
  method identity is not present in the judge prompt.

## Public-release boundary

This repository contains the complete method logic and anonymized data schemas.
Raw industrial logs, platform dictionaries, model checkpoints, production
retrieval services, and online A/B infrastructure are excluded. The
candidate--event mapping interface is public, while the underlying platform
ontology must be supplied by each deployment.

