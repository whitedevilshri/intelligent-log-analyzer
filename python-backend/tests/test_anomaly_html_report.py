"""Unit tests for the standalone HTML anomaly report renderer."""

import unittest
from html.parser import HTMLParser

from ml.reporting.anomaly_html_report import render_html_report


class _TagCounter(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.append(tag)


def sample_summary():
    return {
        "generated_at": "2026-10-26T12:00:00+05:30",
        "model_name": "R1_masked_tfidf_isolation_forest",
        "threshold": -0.1543060505094772,
        "database_counts": {"unlabeled_traces": 1},
        "report_scope": {
            "labeled_trace_count": 7940,
            "predicted_anomaly_count": 151,
            "predicted_normal_count": 7789,
            "anomaly_diagnostic_errors": 0,
        },
        "diagnostic_status_counts": {"insufficient_evidence": 1},
        "root_cause_candidates_reported_total": 0,
        "full_labeled_dataset_confusion_matrix_descriptive_only": {
            "true_positive": 147,
            "false_negative": 166,
            "false_positive": 4,
            "true_negative": 7623,
        },
        "held_out_test_evaluation": {
            "test_samples": 1588,
            "threshold_source": "validation set",
            "roc_auc": 0.682956,
            "pr_auc_average_precision": 0.45262,
            "precision": 1.0,
            "recall": 0.380952,
            "f1": 0.551724,
            "tn": 1525,
            "fp": 0,
            "fn": 39,
            "tp": 24,
        },
    }


def sample_diagnostic(**overrides):
    record = {
        "block_id": "blk_example_123",
        "actual_label": "Anomaly",
        "predicted_label": "Anomaly",
        "anomaly_score": -0.047323,
        "threshold": -0.154306,
        "is_correct": True,
        "root_cause_status": "insufficient_evidence",
        "diagnostic_summary": (
            "Selected anomaly anchor: E7. Observed failure signature(s): "
            "stream_read_failure. Root cause remains undetermined."
        ),
        "combined_log_text": "writeBlock received exception: Could not read from stream",
        "anchor_event_json": '{"event_id":"E7","line_id":65631}',
        "diagnostic_findings_json": '[{"code":"observed_exceptions"}]',
        "failure_evidence_json": '{"failure_events":[{"signature":"stream_read_failure"}]}',
        "processing_evidence_json": '{"missing_markers_anywhere":["E11","E26","E9"]}',
        "precursor_events_json": "[]",
        "root_cause_candidates_json": "[]",
        "diagnostic_error": "",
    }
    record.update(overrides)
    return record


class AnomalyHtmlReportTests(unittest.TestCase):
    def test_renders_scope_metrics_and_both_confusion_matrices(self):
        rendered = render_html_report(sample_summary(), [sample_diagnostic()])

        self.assertIn("Intelligent Log Analyzer", rendered)
        self.assertIn("7,940", rendered)
        self.assertIn("151", rendered)
        self.assertIn("0.683", rendered)
        self.assertIn("0.453", rendered)
        self.assertIn("38.1%", rendered)
        self.assertIn("1,525", rendered)
        self.assertIn("7,623", rendered)
        self.assertIn("descriptive only", rendered)
        self.assertIn("Print / Save as PDF", rendered)

    def test_renders_diagnostic_evidence_and_root_cause_uncertainty(self):
        rendered = render_html_report(sample_summary(), [sample_diagnostic()])

        self.assertIn("blk_example_123", rendered)
        self.assertIn("stream_read_failure", rendered)
        self.assertIn("missing_markers_anywhere", rendered)
        self.assertIn("insufficient_evidence", rendered)
        self.assertIn("No root causes confirmed", rendered)
        self.assertIn("not proof of causality", rendered)

    def test_untrusted_log_content_is_html_escaped(self):
        payload = '<script>alert("x")</script><img src=x onerror="boom">'
        rendered = render_html_report(
            sample_summary(),
            [sample_diagnostic(combined_log_text=payload, diagnostic_summary=payload)],
        )

        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;", rendered)
        self.assertIn("&lt;img src=x onerror=&quot;boom&quot;&gt;", rendered)
        self.assertNotIn('<script>alert("x")</script>', rendered)
        self.assertNotIn('<img src=x onerror="boom">', rendered)

    def test_empty_anomaly_list_renders_without_crashing(self):
        rendered = render_html_report(sample_summary(), [])
        self.assertIn("No predicted anomalies found", rendered)
        self.assertIn("Showing 0 traces", rendered)

    def test_output_is_parseable_html_and_has_search_filter(self):
        rendered = render_html_report(sample_summary(), [sample_diagnostic()])
        parser = _TagCounter()
        parser.feed(rendered)
        parser.close()

        self.assertIn("html", parser.tags)
        self.assertIn("table", parser.tags)
        self.assertIn("details", parser.tags)
        self.assertIn("reportSearch", rendered)
        self.assertIn("filterRows", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
