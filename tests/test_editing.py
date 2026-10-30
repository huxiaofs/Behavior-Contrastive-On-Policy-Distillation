import unittest

from src.data.editing import BehaviorEvent, build_matched_edit


class EditingTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            BehaviorEvent("target", 100.0, "search", "query", "sports watch"),
            BehaviorEvent("near", 110.0, "search", "query", "running shoes"),
            BehaviorEvent("far", 500.0, "search", "query", "camera"),
            BehaviorEvent("wrong_action", 101.0, "search", "click", "watch"),
        ]

    def test_nearest_profile_matched_control(self):
        edit = build_matched_edit(
            self.events, "sports watch", ["target"], related_event_ids=["target"]
        )
        self.assertTrue(edit.valid)
        self.assertEqual(edit.control_event_ids, ("near",))

    def test_related_events_are_excluded(self):
        edit = build_matched_edit(
            self.events,
            "sports watch",
            ["target"],
            related_event_ids=["target", "near", "far"],
        )
        self.assertFalse(edit.valid)
        self.assertEqual(edit.reason, "control_match_failed")

    def test_empty_evidence_is_invalid(self):
        edit = build_matched_edit(self.events, "sports watch", [])
        self.assertFalse(edit.valid)
        self.assertEqual(edit.reason, "no_evidence")


if __name__ == "__main__":
    unittest.main()

