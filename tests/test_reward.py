import unittest

from src.reward.bcopd import CandidateScore, build_group_rewards, set_behavior_score


class RewardTests(unittest.TestCase):
    def test_invalid_candidate_stays_in_denominator(self):
        scores = [
            CandidateScore("a", target_value=0.0, control_value=2.0, valid=True),
            CandidateScore("b", valid=False),
        ]
        self.assertEqual(set_behavior_score(scores), 1.0)

    def test_permutation_invariance(self):
        scores = [
            CandidateScore("a", 0.0, 2.0, True),
            CandidateScore("b", 1.0, 0.0, True),
            CandidateScore("c", valid=False),
        ]
        self.assertAlmostEqual(
            set_behavior_score(scores), set_behavior_score(list(reversed(scores)))
        )

    def test_one_sided_calibration(self):
        groups = [
            [CandidateScore("a", 0.0, 2.0, True)],
            [CandidateScore("a", 2.0, 0.0, True)],
        ]
        result = build_group_rewards([1.0, 1.0], groups, lambda_bc=0.1)
        self.assertGreater(result[0].reward, result[1].reward)
        self.assertEqual(result[1].reward, 0.0)

    def test_no_valid_contrast_falls_back_to_teacher(self):
        groups = [[CandidateScore("a", valid=False)] for _ in range(2)]
        result = build_group_rewards([0.0, 2.0], groups, lambda_bc=100.0)
        self.assertEqual([item.normalized_behavior_score for item in result], [0.0, 0.0])
        self.assertLess(result[0].advantage, result[1].advantage)


if __name__ == "__main__":
    unittest.main()

