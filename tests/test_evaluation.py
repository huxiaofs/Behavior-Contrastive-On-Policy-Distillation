import unittest

from src.evaluation.metrics import exact_match_metrics, selective_evidence_lift
from src.evaluation.semantic_judge import build_prompt


class EvaluationTests(unittest.TestCase):
    def test_exact_match_metrics(self):
        metrics = exact_match_metrics(
            predictions=[["A", "B"], []],
            references=[["A", "C"], ["D"]],
            futures=[["B"], ["D"]],
            cutoff=8,
        )
        self.assertAlmostEqual(metrics["reference_recall"], 0.25)
        self.assertAlmostEqual(metrics["reference_precision"], 0.25)
        self.assertAlmostEqual(metrics["future_exact_hit"], 0.5)

    def test_selective_evidence_lift(self):
        metrics = selective_evidence_lift([1.0], [0.2], [0.9])
        self.assertAlmostEqual(metrics["evidence_lift"], 0.7)

    def test_judge_prompt_is_blind_to_system_identity(self):
        prompt, _, membership = build_prompt(
            {
                "history": "running",
                "future_events": ["sports watch"],
                "systems": {"set_opd": ["watch"], "bc_opd": ["nutrition"]},
            },
            seed=0,
        )
        self.assertNotIn("set_opd", prompt)
        self.assertNotIn("bc_opd", prompt)
        self.assertEqual(set(membership), {"set_opd", "bc_opd"})


if __name__ == "__main__":
    unittest.main()

