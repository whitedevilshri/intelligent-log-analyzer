"""HTML rendering tests for cross-block associations."""
import json
import unittest

from ml.reporting.anomaly_html_report import render_html_report


class CrossBlockHtmlRenderingTests(unittest.TestCase):
    def render(self, correlation):
        summary = {
            "model_name": "test-model",
            "generated_at": "test",
            "threshold": -0.1,
            "report_scope": {
                "labeled_trace_count": 1,
                "predicted_anomaly_count": 1,
                "predicted_normal_count": 0,
                "anomaly_diagnostic_rows": 1,
                "anomaly_diagnostic_errors": 0,
            },
            "database_counts": {"unlabeled_traces": 0},
            "diagnostic_status_counts": {"insufficient_evidence": 1},
        }
        row = {
            "block_id": "blk_target",
            "actual_label": "Anomaly",
            "predicted_label": "Anomaly",
            "anomaly_score": 0.2,
            "threshold": -0.1,
            "is_correct": True,
            "root_cause_status": "insufficient_evidence",
            "diagnostic_summary": "No root cause determined.",
            "combined_log_text": "sample trace",
            "cross_block_correlations_json": json.dumps(correlation),
        }
        return render_html_report(summary, [row])

    def test_renders_candidate_and_matched_signature(self):
        html = self.render({
            "status": "correlations_found",
            "reason": "Same signature within the time window.",
            "time_window_seconds": 300,
            "candidates": [{
                "related_block_id": "blk_related",
                "nearest_time_gap_seconds": 11.0,
                "matched_signatures": ["stream_read_failure"],
                "matched_event_pairs": [{
                    "target_event": {"event_id": "E7", "line_id": 12, "message": "Could not read from stream"},
                    "related_event": {"event_id": "E7", "line_id": 34, "message": "Could not read from stream"},
                }],
            }],
        })
        self.assertIn("Cross-block associations", html)
        self.assertIn("blk_related", html)
        self.assertIn("stream_read_failure", html)
        self.assertIn("11.0 s", html)
        self.assertIn("not a confirmed cause", html)

    def test_renders_empty_result_reason(self):
        html = self.render({
            "status": "no_matching_cross_block_events",
            "reason": "No related signatures found.",
            "time_window_seconds": 300,
            "candidates": [],
        })
        self.assertIn("no_matching_cross_block_events", html)
        self.assertIn("No related signatures found.", html)
        self.assertIn("No matching cross-block events", html)

    def test_escapes_untrusted_event_text(self):
        html = self.render({
            "status": "correlations_found",
            "reason": "Matched.",
            "time_window_seconds": 300,
            "candidates": [{
                "related_block_id": "blk_related",
                "nearest_time_gap_seconds": 2.0,
                "matched_signatures": ["stream_read_failure"],
                "matched_event_pairs": [{
                    "target_event": {"event_id": "E7", "line_id": 1, "message": "safe"},
                    "related_event": {"event_id": "E7", "line_id": 2, "message": "<script>alert(1)</script>"},
                }],
            }],
        })
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
