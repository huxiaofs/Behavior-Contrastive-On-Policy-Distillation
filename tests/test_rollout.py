import unittest

from src.rollout.sample import parse_interests


class RolloutTests(unittest.TestCase):
    def test_json_output(self):
        self.assertEqual(parse_interests('["camping", "photography"]', 8), ["camping", "photography"])

    def test_deduplication_and_cutoff(self):
        text = "兴趣扩展标签：露营、露营、户外摄影、越野跑"
        self.assertEqual(parse_interests(text, 2), ["露营", "户外摄影"])


if __name__ == "__main__":
    unittest.main()

