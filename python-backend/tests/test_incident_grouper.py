
import unittest

from app.services.incident_grouper import build_incident_groups


def make_report(source, related, source_time, related_time,
                source_line, related_line,
                signature="stream_read_failure"):
    return {
        "block_id": source,
        "status": "correlations_found",
        "candidates": [{
            "related_block_id": related,
            "matched_signatures": [signature],
            "matched_event_pairs": [{
                "target_event": {
                    "timestamp": source_time,
                    "line_id": source_line,
                },
                "related_event": {
                    "timestamp": related_time,
                    "line_id": related_line,
                    "signature": signature,
                },
            }],
        }],
    }


class IncidentGrouperTests(unittest.TestCase):

    def test_groups_connected_blocks(self):
        reports = [
            make_report(
                "A", "B",
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:00:08Z", 1, 2,
            ),
            make_report(
                "B", "C",
                "2020-01-01T00:00:10Z",
                "2020-01-01T00:00:18Z", 3, 4,
            ),
        ]

        result = build_incident_groups(reports)

        self.assertEqual(result["incident_count"], 1)
        incident = result["incidents"][0]
        self.assertEqual(incident["block_ids"], ["A", "B", "C"])
        self.assertEqual(incident["correlation_edge_count"], 2)
        self.assertEqual(incident["failure_signature"],
                         "stream_read_failure")

    def test_deduplicates_reciprocal_reports(self):
        reports = [
            make_report(
                "A", "B",
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:00:08Z", 1, 2,
            ),
            make_report(
                "B", "A",
                "2020-01-01T00:00:08Z",
                "2020-01-01T00:00:00Z", 2, 1,
            ),
        ]

        result = build_incident_groups(reports)

        self.assertEqual(result["unique_event_pair_count"], 1)
        self.assertEqual(result["incident_count"], 1)
        self.assertEqual(
            result["incidents"][0]["matched_event_pair_count"], 1
        )

    def test_does_not_merge_different_signatures(self):
        reports = [
            make_report(
                "A", "B",
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:00:08Z", 1, 2,
                "stream_read_failure",
            ),
            make_report(
                "B", "C",
                "2020-01-01T00:00:10Z",
                "2020-01-01T00:00:18Z", 3, 4,
                "connection_reset",
            ),
        ]

        result = build_incident_groups(reports)

        self.assertEqual(result["incident_count"], 2)

    def test_separates_relationships_outside_window(self):
        reports = [
            make_report(
                "A", "B",
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:00:08Z", 1, 2,
            ),
            make_report(
                "B", "C",
                "2020-01-01T00:06:40Z",
                "2020-01-01T00:06:48Z", 3, 4,
            ),
        ]

        result = build_incident_groups(
            reports, incident_window_seconds=300
        )

        self.assertEqual(result["incident_count"], 2)

    def test_skips_pairs_with_missing_timestamps(self):
        reports = [
            make_report(
                "A", "B", None,
                "2020-01-01T00:00:08Z", 1, 2,
            ),
        ]

        result = build_incident_groups(reports)

        self.assertEqual(result["incident_count"], 0)
        self.assertEqual(
            result["skipped_pairs_missing_timestamps"], 1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
