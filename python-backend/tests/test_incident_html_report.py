import unittest
from ml.reporting.anomaly_html_report import render_html_report


class IncidentHtmlReportTests(unittest.TestCase):
    def test_renders_incident_summary(self):
        summary = {
            "model_name": "test-model",
            "generated_at": "test",
            "threshold": -0.1,
            "report_scope": {
                "labeled_trace_count": 1,
                "predicted_anomaly_count": 0,
                "predicted_normal_count": 1,
                "anomaly_diagnostic_rows": 0,
                "anomaly_diagnostic_errors": 0,
            },
            "database_counts": {"unlabeled_traces": 0},
            "incident_grouping": {
                "incident_count": 1,
                "unique_event_pair_count": 1,
                "skipped_pairs_missing_timestamps": 0,
                "cluster_overlap_pair_count": 1,
                "clusters_with_overlaps": 2,
                "candidate_coverage": {
                    "cross_block_reports_collected": 2,
                    "candidate_rows_returned_total": 50,
                    "candidate_rows_omitted_total": 0,
                    "complete_for_collected_reports": True,
                },
                "incidents": [{
                    "incident_id": "INC-001",
                    "failure_signature": "stream_read_failure",
                    "block_ids": ["blk_A", "blk_B"],
                    "correlation_edge_count": 1,
                    "matched_event_pair_count": 1,
                    "time_range": {
                        "start": "2020-01-01T00:00:00+00:00",
                        "end": "2020-01-01T00:00:08+00:00",
                        "duration_seconds": 8.0,
                    },
                }],
            },
        }

        rendered = render_html_report(summary, [])
        self.assertIn("Potential Incident Clusters", rendered)
        self.assertIn("INC-001", rendered)
        self.assertIn("stream_read_failure", rendered)
        self.assertIn("blk_A, blk_B", rendered)
        self.assertIn("8.0 s", rendered)
        self.assertIn("Relationship midpoint range (UTC)", rendered)
        self.assertIn("Observed event range (UTC)", rendered)
        self.assertIn("not proof of a shared root cause", rendered)
        self.assertIn("Overlapping same-signature cluster pairs: 1", rendered)
        self.assertIn("Candidate rows returned: 50; omitted: 0", rendered)

    def test_escapes_incident_values(self):
        summary = {
            "report_scope": {},
            "database_counts": {},
            "incident_grouping": {
                "incidents": [{
                    "incident_id": "<script>",
                    "failure_signature": "safe",
                    "block_ids": [],
                    "time_range": {},
                }],
            },
        }
        rendered = render_html_report(summary, [])
        self.assertNotIn("<code><script></code>", rendered)
        self.assertIn("&lt;script&gt;", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
