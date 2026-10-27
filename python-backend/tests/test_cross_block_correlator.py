"""Unit tests for the isolated cross-block correlation MVP."""

import unittest

from app.services.cross_block_correlator import build_cross_block_report


def event(block_id, line_id, time, message, event_id="E7", component="dfs.DataNode$DataXceiver"):
    return {
        "block_id": block_id,
        "line_id": line_id,
        "log_date": "081109",
        "log_time": time,
        "event_id": event_id,
        "component": component,
        "message": message,
    }


class CrossBlockCorrelationTests(unittest.TestCase):
    def test_matches_same_specific_failure_in_another_block_within_window(self):
        target = [event("blk_target", 10, "204053", "writeBlock received exception: Could not read from stream")]
        related = [event("blk_other", 20, "204100", "writeBlock received exception: Could not read from stream")]
        report = build_cross_block_report("blk_target", target, related)
        self.assertEqual(report["status"], "correlations_found")
        self.assertEqual(report["total_matching_blocks"], 1)
        self.assertEqual(report["candidates"][0]["related_block_id"], "blk_other")
        self.assertEqual(report["candidates"][0]["matched_signatures"], ["stream_read_failure"])

    def test_excludes_target_block_from_cross_block_results(self):
        target = [event("blk_target", 10, "204053", "Could not read from stream")]
        related = [event("blk_target", 11, "204054", "Could not read from stream")]
        report = build_cross_block_report("blk_target", target, related)
        self.assertEqual(report["status"], "no_matching_cross_block_events")

    def test_excludes_events_outside_window(self):
        target = [event("blk_target", 10, "204053", "Could not read from stream")]
        related = [event("blk_other", 20, "204600", "Could not read from stream")]
        report = build_cross_block_report("blk_target", target, related, time_window_seconds=300)
        self.assertEqual(report["total_matching_blocks"], 0)

    def test_excludes_different_failure_signature(self):
        target = [event("blk_target", 10, "204053", "Could not read from stream")]
        related = [event("blk_other", 20, "204054", "Connection reset by peer")]
        report = build_cross_block_report("blk_target", target, related)
        self.assertEqual(report["total_matching_blocks"], 0)

    def test_abstains_when_target_has_no_specific_failure_signature(self):
        target = [event("blk_target", 10, "204053", "Receiving block blk_target", event_id="E5")]
        related = [event("blk_other", 20, "204054", "Could not read from stream")]
        report = build_cross_block_report("blk_target", target, related)
        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["candidates"], [])

    def test_ignores_related_events_with_unparseable_timestamps(self):
        target = [event("blk_target", 10, "204053", "Could not read from stream")]
        related = [event("blk_other", 20, "badtime", "Could not read from stream")]
        report = build_cross_block_report("blk_target", target, related)
        self.assertEqual(report["total_matching_blocks"], 0)

    def test_invalid_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 3600"):
            build_cross_block_report("blk_target", [], [], time_window_seconds=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
