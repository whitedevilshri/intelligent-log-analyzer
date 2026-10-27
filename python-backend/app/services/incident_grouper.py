
"""Group cross-block failure associations into potential incidents."""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse a timestamp and normalize it to UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(
                value.strip().replace("Z", "+00:00")
            )
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def build_incident_groups(
    correlation_reports: Iterable[Mapping[str, Any]],
    *,
    incident_window_seconds: int = 300,
) -> dict[str, Any]:
    """
    Group related blocks when they share a failure signature,
    connected block relationships, and a bounded time range.

    These groups represent potential associations, not confirmed causes.
    """
    if not 1 <= int(incident_window_seconds) <= 3600:
        raise ValueError(
            "incident_window_seconds must be between 1 and 3600."
        )

    # Deduplicate reciprocal reports using both event identities.
    unique_pairs: dict[tuple[Any, ...], dict[str, Any]] = {}
    skipped_missing_timestamps = 0

    for report in correlation_reports:
        source_block = str(report.get("block_id", "")).strip()
        if not source_block:
            continue

        candidates = report.get("candidates", [])
        if not isinstance(candidates, list):
            continue

        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue

            related_block = str(
                candidate.get("related_block_id", "")
            ).strip()

            if not related_block or related_block == source_block:
                continue

            signatures = candidate.get("matched_signatures", [])
            signatures = (
                signatures if isinstance(signatures, list) else []
            )

            pairs = candidate.get("matched_event_pairs", [])
            if not isinstance(pairs, list):
                continue

            for pair in pairs:
                if not isinstance(pair, Mapping):
                    continue

                target = pair.get("target_event", {})
                related = pair.get("related_event", {})

                if not isinstance(target, Mapping):
                    target = {}
                if not isinstance(related, Mapping):
                    related = {}

                signature = (
                    related.get("signature")
                    or target.get("signature")
                )

                if not signature and len(signatures) == 1:
                    signature = signatures[0]

                if not signature:
                    continue

                target_time = _parse_timestamp(
                    target.get("timestamp")
                )
                related_time = _parse_timestamp(
                    related.get("timestamp")
                )

                if target_time is None or related_time is None:
                    skipped_missing_timestamps += 1
                    continue

                endpoints = [
                    (
                        source_block,
                        str(target.get("line_id") or ""),
                        target_time.isoformat(),
                    ),
                    (
                        related_block,
                        str(related.get("line_id") or ""),
                        related_time.isoformat(),
                    ),
                ]
                endpoints.sort()

                # The same event pair reported in both directions is one pair.
                key = (str(signature), *endpoints)

                unique_pairs[key] = {
                    "signature": str(signature),
                    "block_a": min(source_block, related_block),
                    "block_b": max(source_block, related_block),
                    "start": min(target_time, related_time),
                    "end": max(target_time, related_time),
                    "gap_seconds": abs(
                        (related_time - target_time).total_seconds()
                    ),
                }

    # Assign each unique pair to a deterministic UTC bucket using its midpoint.
    # This partitions relationship evidence rather than generating many
    # nearly-identical rolling windows.
    by_bucket: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    window = int(incident_window_seconds)

    for raw_pair in unique_pairs.values():
        pair = dict(raw_pair)
        pair["relationship_time"] = (
            pair["start"] + (pair["end"] - pair["start"]) / 2
        )
        bucket_index = int(pair["relationship_time"].timestamp() // window)
        by_bucket[(pair["signature"], bucket_index)].append(pair)

    clusters: list[dict[str, Any]] = []

    for (signature, bucket_index), pairs in sorted(by_bucket.items()):
        pairs.sort(
            key=lambda pair: (
                pair["relationship_time"],
                pair["start"],
                pair["end"],
                pair["block_a"],
                pair["block_b"],
            )
        )

        # Each event pair is an edge between two BlockIds. Compute connected
        # components within this signature and time bucket.
        block_to_edges: dict[str, list[int]] = defaultdict(list)
        for edge_index, pair in enumerate(pairs):
            block_to_edges[pair["block_a"]].append(edge_index)
            block_to_edges[pair["block_b"]].append(edge_index)

        unseen = set(range(len(pairs)))
        bucket_start = datetime.fromtimestamp(
            bucket_index * window, tz=timezone.utc
        )
        bucket_end = datetime.fromtimestamp(
            (bucket_index + 1) * window, tz=timezone.utc
        )

        while unseen:
            seed = min(unseen)
            pending = [seed]
            component_indices: list[int] = []
            component_blocks: set[str] = set()

            while pending:
                edge_index = pending.pop()
                if edge_index not in unseen:
                    continue

                unseen.remove(edge_index)
                component_indices.append(edge_index)
                pair = pairs[edge_index]

                for block_id in (pair["block_a"], pair["block_b"]):
                    if block_id not in component_blocks:
                        component_blocks.add(block_id)
                        pending.extend(
                            neighbour for neighbour in block_to_edges[block_id]
                            if neighbour in unseen
                        )

            component_pairs = [pairs[i] for i in sorted(component_indices)]
            clusters.append({
                "signature": signature,
                "bucket_index": bucket_index,
                "window_bucket_start": bucket_start,
                "window_bucket_end": bucket_end,
                "block_ids": component_blocks,
                "start": min(p["relationship_time"] for p in component_pairs),
                "end": max(p["relationship_time"] for p in component_pairs),
                "observed_start": min(p["start"] for p in component_pairs),
                "observed_end": max(p["end"] for p in component_pairs),
                "pairs": component_pairs,
            })

    clusters.sort(
        key=lambda cluster: (
            cluster["start"],
            cluster["signature"],
            cluster["bucket_index"],
            sorted(cluster["block_ids"]),
        )
    )

    incidents = []
    for index, cluster in enumerate(clusters, start=1):
        relationships: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for pair in cluster["pairs"]:
            relationships[(pair["block_a"], pair["block_b"])].append(pair)

        relationship_rows = []
        for (block_a, block_b), event_pairs in sorted(relationships.items()):
            relationship_rows.append({
                "block_a": block_a,
                "block_b": block_b,
                "matched_event_pair_count": len(event_pairs),
                "nearest_time_gap_seconds": round(
                    min(pair["gap_seconds"] for pair in event_pairs), 3
                ),
            })

        incidents.append({
            "incident_id": f"INC-{index:03d}",
            "failure_signature": cluster["signature"],
            "block_ids": sorted(cluster["block_ids"]),
            "block_count": len(cluster["block_ids"]),
            "correlation_edge_count": len(relationship_rows),
            "matched_event_pair_count": len(cluster["pairs"]),
            # Relationship midpoint range is contained in one fixed window.
            "time_range": {
                "start": cluster["start"].isoformat(),
                "end": cluster["end"].isoformat(),
                "duration_seconds": round(
                    (cluster["end"] - cluster["start"]).total_seconds(), 3
                ),
            },
            # The full event endpoints can span across a bucket boundary.
            "observed_event_range": {
                "start": cluster["observed_start"].isoformat(),
                "end": cluster["observed_end"].isoformat(),
                "duration_seconds": round(
                    (cluster["observed_end"] - cluster["observed_start"]).total_seconds(),
                    3,
                ),
            },
            "window_bucket": {
                "start": cluster["window_bucket_start"].isoformat(),
                "end": cluster["window_bucket_end"].isoformat(),
            },
            "relationships": relationship_rows,
            "overlapping_incident_ids": [],
            "interpretation": (
                "Potential incident association only; not a confirmed common cause."
            ),
        })

    # Report clusters whose full observed event ranges overlap and whose
    # same-signature membership shares at least one BlockId.
    overlap_pair_count = 0
    overlapping_ids: dict[str, set[str]] = defaultdict(set)

    for index, left in enumerate(incidents):
        left_blocks = set(left["block_ids"])
        left_range = left["observed_event_range"]
        left_start = datetime.fromisoformat(left_range["start"])
        left_end = datetime.fromisoformat(left_range["end"])

        for right in incidents[index + 1:]:
            if left["failure_signature"] != right["failure_signature"]:
                continue
            if not left_blocks.intersection(right["block_ids"]):
                continue

            right_range = right["observed_event_range"]
            right_start = datetime.fromisoformat(right_range["start"])
            right_end = datetime.fromisoformat(right_range["end"])

            if min(left_end, right_end) < max(left_start, right_start):
                continue

            overlap_pair_count += 1
            overlapping_ids[left["incident_id"]].add(right["incident_id"])
            overlapping_ids[right["incident_id"]].add(left["incident_id"])

    for incident in incidents:
        incident["overlapping_incident_ids"] = sorted(
            overlapping_ids.get(incident["incident_id"], set())
        )

    return {
        "analysis_type": "cross_block_incident_grouping",
        "incident_window_seconds": window,
        "window_strategy": "fixed_utc_buckets_by_event_pair_midpoint",
        "incident_count": len(incidents),
        "cluster_count": len(incidents),
        "cluster_overlap_pair_count": overlap_pair_count,
        "clusters_with_overlaps": len(overlapping_ids),
        "incidents": incidents,
        "unique_event_pair_count": len(unique_pairs),
        "skipped_pairs_missing_timestamps": skipped_missing_timestamps,
        "method_notes": [
            "Only matching classified failure signatures are grouped.",
            "Each unique event pair is assigned to one fixed UTC bucket by its midpoint.",
            "Connected BlockId components are calculated separately inside each bucket.",
            "The relationship midpoint range stays within one configured bucket.",
            "The full observed event range may cross bucket boundaries.",
            "Blocks can recur in different time buckets; clusters are not proof of independent incidents.",
            "Reciprocal reports of the same event pair are deduplicated.",
            "Incident associations are not proof of a shared root cause.",
        ],
    }
