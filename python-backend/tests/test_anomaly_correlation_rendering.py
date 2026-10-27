"""Regression tests for visible event correlation in the standalone HTML report."""
import unittest

from ml.reporting.anomaly_html_report import (
    _render_candidate_evaluations,
    _render_event_timeline,
)


class AnomalyCorrelationRenderingTests(unittest.TestCase):
    def test_timeline_displays_preceding_event_and_selected_anchor(self):
        rendered = _render_event_timeline(
            [
                {
                    "line_id": 41,
                    "log_date": "2026-10-01",
                    "log_time": "12:00:00",
                    "event_id": "E5",
                    "component": "dfs.DataNode",
                    "message": "receiving block",
                    "seconds_before_anchor": 5.0,
                    "training_prevalence_difference": 0.02,
                    "role": "context_only",
                    "candidate_rejection_reasons": ["below_minimum_anomaly_support"],
                }
            ],
            {
                "line_id": 42,
                "log_date": "2026-10-01",
                "log_time": "12:00:05",
                "event_id": "E7",
                "component": "dfs.DataNode$DataXceiver",
                "message": "writeBlock failed",
            },
            "timestamp_then_line_id",
        )
        self.assertIn("E5", rendered)
        self.assertIn("E7", rendered)
        self.assertIn("2026-10-01 12:00:00", rendered)
        self.assertIn("5.000 s", rendered)
        self.assertIn("below_minimum_anomaly_support", rendered)
        self.assertIn("cross-block correlation is not implemented", rendered.lower())

    def test_timeline_escapes_untrusted_log_messages(self):
        rendered = _render_event_timeline(
            [],
            {"line_id": 7, "event_id": "E7", "message": "<script>alert(1)</script>"},
            "line_id_fallback",
        )
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertNotIn("<script>", rendered)

    def test_candidate_screening_renders_rejection_reasons_and_counts(self):
        rendered = _render_candidate_evaluations(
            [
                {
                    "line_id": 40,
                    "event_id": "E5",
                    "component": "dfs.DataNode",
                    "message": "receiving block",
                    "seconds_before_anchor": 4.0,
                    "training_anomaly_blocks": 1,
                    "training_normal_blocks": 50,
                    "prevalence_difference": -0.01,
                    "eligible_as_candidate": False,
                    "rejection_reasons": ["below_minimum_anomaly_support"],
                }
            ],
            {
                "preceding_event_instances_evaluated": 1,
                "eligible_candidate_instances": 0,
                "rejected_candidate_instances": 1,
                "rejection_reason_counts": {"below_minimum_anomaly_support": 1},
                "minimum_anomaly_support": 5,
                "note": "Counts are per preceding event instance.",
            },
        )
        self.assertIn("Candidate screening and rejection reasons", rendered)
        self.assertIn("below_minimum_anomaly_support", rendered)
        self.assertIn("Rejected", rendered)
        self.assertIn("Events screened", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
