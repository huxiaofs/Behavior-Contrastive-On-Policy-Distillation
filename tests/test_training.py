import unittest

import torch

from src.training.objective import clipped_sequence_opd_loss, sequence_log_ratio


class TrainingTests(unittest.TestCase):
    def test_length_normalized_sequence_ratio(self):
        old = torch.tensor([-2.0, -3.0])
        current = torch.tensor([-1.0, -2.0])
        self.assertAlmostEqual(sequence_log_ratio(current, old).item(), 1.0)

    def test_sequence_clipping(self):
        old = torch.tensor([-2.0, -3.0])
        current = torch.tensor([-1.0, -2.0])
        loss, stats = clipped_sequence_opd_loss(current, old, 1.0, clip_epsilon=0.2)
        self.assertAlmostEqual(stats["clipped_ratio"].item(), 1.2, places=6)
        self.assertAlmostEqual(loss.item(), -1.2, places=6)


if __name__ == "__main__":
    unittest.main()

