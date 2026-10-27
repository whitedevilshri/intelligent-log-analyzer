
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

    # Group event-pair relationships by signature and temporal proximity.
    by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in unique_pairs.values():
        by_signature[pair["signature"]].append(pair)

    clusters: list[dict[str, Any]] = []
    window = int(incident_window_seconds)

    for signature, pairs in sorted(by_signature.items()):
        signature_clusters: list[dict[str, Any]] = []

        pairs.sort(
            key=lambda pair: (
                pair["start"],
                pair["end"],
                pair["block_a"],
                pair["block_b"],
            )
        )

        for pair in pairs:
            pair_blocks = {pair["block_a"], pair["block_b"]}

            compatible = []
            for cluster in signature_clusters:
                overlaps_blocks = bool(
                    cluster["block_ids"] & pair_blocks
                )
                combined_start = min(cluster["start"], pair["start"])
                combined_end = max(cluster["end"], pair["end"])
                fits_window = (
                    combined_end - combined_start
                ).total_seconds() <= window

                if overlaps_blocks and fits_window:
                    compatible.append(cluster)

            if not compatible:
                signature_clusters.append({
                    "signature": signature,
                    "block_ids": pair_blocks.copy(),
                    "start": pair["start"],
                    "end": pair["end"],
                    "pairs": [pair],
                })
                continue

            # Prefer the closest compatible cluster.
            compatible.sort(
                key=lambda cluster: (
                    min(
                        abs((pair["start"] - cluster["end"]).total_seconds()),
                        abs((cluster["start"] - pair["end"]).total_seconds()),
                    ),
                    cluster["start"],
                )
            )

            chosen = compatible[0]
            chosen["pairs"].append(pair)
            chosen["block_ids"].update(pair_blocks)
            chosen["start"] = min(chosen["start"], pair["start"])
            chosen["end"] = max(chosen["end"], pair["end"])

            # Merge additional touching clusters only if the overall
            # time span still fits the configured incident window.
            for other in compatible[1:]:
                if other not in signature_clusters:
                    continue

                merged_start = min(chosen["start"], other["start"])
                merged_end = max(chosen["end"], other["end"])

                if (merged_end - merged_start).total_seconds() > window:
                    continue

                chosen["pairs"].extend(other["pairs"])
                chosen["block_ids"].update(other["block_ids"])
                chosen["start"] = merged_start
                chosen["end"] = merged_end
                signature_clusters.remove(other)

        clusters.extend(signature_clusters)

    clusters.sort(
        key=lambda cluster: (
            cluster["start"],
            cluster["signature"],
            sorted(cluster["block_ids"]),
        )
    )

    incidents = []

    for index, cluster in enumerate(clusters, start=1):
        relationships: dict[tuple[str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )

        for pair in cluster["pairs"]:
            relationships[
                (pair["block_a"], pair["block_b"])
            ].append(pair)

        relationship_rows = []
        for (block_a, block_b), event_pairs in sorted(
            relationships.items()
        ):
            relationship_rows.append({
                "block_a": block_a,
                "block_b": block_b,
                "matched_event_pair_count": len(event_pairs),
                "nearest_time_gap_seconds": round(
                    min(p["gap_seconds"] for p in event_pairs), 3
                ),
            })

        incidents.append({
            "incident_id": f"INC-{index:03d}",
            "failure_signature": cluster["signature"],
            "block_ids": sorted(cluster["block_ids"]),
            "block_count": len(cluster["block_ids"]),
            "correlation_edge_count": len(relationship_rows),
            "matched_event_pair_count": len(cluster["pairs"]),
            "time_range": {
                "start": cluster["start"].isoformat(),
                "end": cluster["end"].isoformat(),
                "duration_seconds": round(
                    (cluster["end"] - cluster["start"]).total_seconds(),
                    3,
                ),
            },
            "relationships": relationship_rows,
            "interpretation": (
                "Potential incident association only; "
                "not a confirmed common cause."
            ),
        })


    overlap_pair_count = 0
    overlapping_ids: dict[str, set[str]] = defaultdict(set)

    for index, left in enumerate(incidents):
        left_blocks = set(left.get("block_ids", []))
        left_range = left.get("time_range", {})
        left_start = datetime.fromisoformat(left_range["start"])
        left_end = datetime.fromisoformat(left_range["end"])

        for right in incidents[index + 1:]:
            if left.get("failure_signature") != right.get("failure_signature"):
                continue

            common_blocks = left_blocks.intersection(right.get("block_ids", []))
            if not common_blocks:
                continue

            right_range = right.get("time_range", {})
            right_start = datetime.fromisoformat(right_range["start"])
            right_end = datetime.fromisoformat(right_range["end"])

            overlap_start = max(left_start, right_start)
            overlap_end = min(left_end, right_end)
            if overlap_end < overlap_start:
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
        "incident_count": len(incidents),
        "cluster_count": len(incidents),
        "cluster_overlap_pair_count": overlap_pair_count,
        "clusters_with_overlaps": len(overlapping_ids),
        "incidents": incidents,
        "unique_event_pair_count": len(unique_pairs),
        "skipped_pairs_missing_timestamps": skipped_missing_timestamps,
        "method_notes": [
            "Only matching classified failure signatures are grouped.",
            "Groups are time-bounded correlation clusters; clusters may overlap in time and BlockId membership.",
            "Reciprocal reports of the same event pair are deduplicated.",
            "Incident associations are not proof of a shared root cause.",
        ],
    }
