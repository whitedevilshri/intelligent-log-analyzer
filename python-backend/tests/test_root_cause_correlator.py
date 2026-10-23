"""Offline unit tests for the root-cause correlator.

Run from python-backend with:
    python -m unittest discover -s tests -v

These tests mock PostgreSQL access and do not write to the database.
"""

import unittest
from unittest.mock import patch

from app.services import root_cause_correlator as correlator


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class FakeConnection:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params=None):
        return FakeCursor(self._rows)


def profile_for_tests():
    """Small deterministic training profile for rule tests."""
    return {
        "E5": {
            "anomaly_blocks": 200,
            "normal_blocks": 4881,
            "anomaly_rate": 1.0,
            "normal_rate": 1.0,
            "prevalence_difference": 0.0,
        },
        "E7": {
            "anomaly_blocks": 44,
            "normal_blocks": 0,
            "anomaly_rate": 0.22,
            "normal_rate": 0.0,
            "prevalence_difference": 0.22,
        },
        "E9": {
            "anomaly_blocks": 116,
            "normal_blocks": 4880,
            "anomaly_rate": 0.58,
            "normal_rate": 0.9998,
            "prevalence_difference": -0.4198,
        },
        "E11": {
            "anomaly_blocks": 116,
            "normal_blocks": 4880,
            "anomaly_rate": 0.58,
            "normal_rate": 0.9998,
            "prevalence_difference": -0.4198,
        },
        "E14": {
            "anomaly_blocks": 1,
            "normal_blocks": 0,
            "anomaly_rate": 0.005,
            "normal_rate": 0.0,
            "prevalence_difference": 0.005,
        },
    }


def row(line_id, log_time, event_id, content, component="dfs.TestComponent"):
    """Match the row shape returned by the production SQL query."""
    return (
        line_id,
        "081109",
        log_time,
        component,
        "INFO",
        event_id,
        f"template for {event_id}",
        content,
    )


class ParseAndClassificationTests(unittest.TestCase):
    def test_parse_timestamp_valid_and_invalid(self):
        parsed = correlator.parse_timestamp("081109", "203500")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.strftime("%Y-%m-%d %H:%M:%S"), "2009-08-11 20:35:00")
        self.assertIsNone(correlator.parse_timestamp("bad", "date"))

    def test_classify_common_exception_signatures(self):
        cases = {
            "writeBlock received exception java.io.EOFException": "unexpected_end_of_stream",
            "Connection reset by peer": "connection_reset",
            "SocketTimeoutException occurred": "socket_timeout",
            "Could not read from stream": "stream_read_failure",
            "InterruptedIOException": "interrupted_io",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(correlator.classify_exception(message), expected)


class AnalyzeBlockTests(unittest.TestCase):
    def analyze_with_rows(self, block_id, rows):
        with patch.object(correlator, "load_training_event_profile", return_value=profile_for_tests()), \
             patch.object(correlator, "get_connection", return_value=FakeConnection(rows)):
            return correlator.analyze_block(block_id)

    def test_normal_trace_has_no_anchor_or_root_cause(self):
        rows = [
            row(1, "203500", "E5", "Receiving block blk_normal"),
            row(2, "203501", "E11", "PacketResponder terminating"),
            row(3, "203502", "E26", "blockMap updated"),
            row(4, "203503", "E9", "Received block"),
        ]
        report = self.analyze_with_rows("blk_normal", rows)

        self.assertEqual(report["root_cause_status"], "insufficient_evidence")
        self.assertIsNone(report["anchor_event"])
        self.assertEqual(report["root_cause_candidates"], [])
        self.assertEqual(report["diagnostic_findings"], [])
        self.assertEqual(report["processing_evidence"]["missing_markers_anywhere"], [])
        self.assertEqual(report["event_ordering"], "timestamp_then_line_id")

    def test_exception_trace_explains_rejected_preceding_events(self):
        rows = [
            row(10, "203500", "E5", "Receiving block blk_anomaly"),
            row(11, "203501", "E14", "Exception: Connection reset by peer"),
            row(12, "203502", "E7", "writeBlock received exception java.io.EOFException"),
            row(13, "203503", "E7", "writeBlock received exception java.io.EOFException"),
            row(14, "203504", "E11", "PacketResponder terminating"),
            row(15, "203505", "E26", "blockMap updated"),
            row(16, "203506", "E9", "Received block"),
        ]
        report = self.analyze_with_rows("blk_anomaly", rows)

        self.assertEqual(report["anchor_event"]["event_id"], "E7")
        self.assertEqual(report["event_ordering"], "timestamp_then_line_id")
        self.assertEqual(report["root_cause_status"], "insufficient_evidence")
        self.assertEqual(report["root_cause_candidates"], [])
        self.assertGreaterEqual(len(report["candidate_evaluations"]), 1)

        rejection_sets = [set(item["rejection_reasons"]) for item in report["candidate_evaluations"]]
        self.assertTrue(any("below_minimum_prevalence_difference" in reasons for reasons in rejection_sets))
        self.assertTrue(any("below_minimum_anomaly_support" in reasons for reasons in rejection_sets))
        self.assertTrue(any("same_event_as_anchor" in reasons for reasons in rejection_sets))
        self.assertEqual(
            report["candidate_screening_summary"]["eligible_candidate_instances"],
            0,
        )

    def test_events_are_sorted_by_timestamp_when_available(self):
        # The source rows deliberately arrive in LineId order that disagrees
        # with clock order. The E7 at 20:35:02 should be selected as anchor.
        rows = [
            row(100, "203503", "E9", "Received block"),
            row(101, "203502", "E7", "writeBlock received exception java.io.EOFException"),
            row(102, "203501", "E5", "Receiving block"),
        ]
        report = self.analyze_with_rows("blk_order", rows)

        self.assertEqual(report["event_ordering"], "timestamp_then_line_id")
        self.assertEqual(report["anchor_event"]["event_id"], "E7")
        self.assertEqual(report["anchor_event"]["line_id"], 101)
        self.assertEqual(len(report["preceding_context"]), 1)
        self.assertEqual(report["preceding_context"][0]["event_id"], "E5")

    def test_empty_block_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "No structured log events found"):
            self.analyze_with_rows("blk_missing", [])

    def test_empty_block_id_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "BlockId must not be empty"):
            correlator.analyze_block(" ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
