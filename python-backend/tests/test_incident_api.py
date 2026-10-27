import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.api.routes.root_cause import (
    IncidentGroupingRequest,
    get_incident_groups,
)


def report_for(source, related):
    events = {
        "A": {
            "timestamp": "2020-01-01T00:00:00Z",
            "line_id": 1,
            "signature": "stream_read_failure",
        },
        "B": {
            "timestamp": "2020-01-01T00:00:08Z",
            "line_id": 2,
            "signature": "stream_read_failure",
        },
    }

    return {
        "block_id": source,
        "status": "correlations_found",
        "candidates": [{
            "related_block_id": related,
            "matched_signatures": ["stream_read_failure"],
            "matched_event_pairs": [{
                "target_event": events[source],
                "related_event": events[related],
            }],
        }],
    }


class IncidentApiTests(unittest.TestCase):

    @patch("app.api.routes.root_cause.correlate_block")
    def test_groups_blocks_and_deduplicates_ids(self, mock_correlate):
        mock_correlate.side_effect = lambda block_id, **kwargs: (
            report_for("A", "B") if block_id == "A"
            else report_for("B", "A")
        )

        result = get_incident_groups(
            IncidentGroupingRequest(block_ids=["A", "A", "B"])
        )

        self.assertEqual(result["requested_block_count"], 3)
        self.assertEqual(result["unique_block_count"], 2)
        self.assertEqual(result["processed_block_count"], 2)
        self.assertEqual(result["incident_count"], 1)
        self.assertEqual(result["unique_event_pair_count"], 1)
        self.assertEqual(mock_correlate.call_count, 2)

    @patch("app.api.routes.root_cause.correlate_block")
    def test_reports_per_block_lookup_errors(self, mock_correlate):
        def fake_correlate(block_id, **kwargs):
            if block_id == "missing":
                raise ValueError("No structured log events found.")
            return {
                "block_id": block_id,
                "status": "insufficient_evidence",
                "candidates": [],
            }

        mock_correlate.side_effect = fake_correlate
        result = get_incident_groups(
            IncidentGroupingRequest(block_ids=["missing", "A"])
        )

        self.assertEqual(result["processed_block_count"], 1)
        self.assertEqual(result["incident_count"], 0)
        self.assertEqual(len(result["query_errors"]), 1)
        self.assertEqual(result["query_errors"][0]["block_id"], "missing")

    @patch("app.api.routes.root_cause.correlate_block")
    def test_rejects_blank_block_ids(self, mock_correlate):
        with self.assertRaises(HTTPException) as context:
            get_incident_groups(
                IncidentGroupingRequest(block_ids=["   "])
            )

        self.assertEqual(context.exception.status_code, 422)
        mock_correlate.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
