"""Render a self-contained, interactive, print-friendly HTML anomaly report."""

from __future__ import annotations

import html
import json
from typing import Any


def _parse_json(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _number(value: Any, places: int = 3) -> str:
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "N/A"


def _count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def _percent(value: Any, places: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{places}f}%"
    except (TypeError, ValueError):
        return "N/A"


def _evidence(title: str, value: Any) -> str:
    if value in (None, "", [], {}):
        text = "No evidence recorded."
    else:
        text = html.escape(json.dumps(value, ensure_ascii=False, indent=2))
    return (
        '<section class="evidence-block">'
        f'<h4>{html.escape(title)}</h4><pre>{text}</pre>'
        '</section>'
    )


def render_html_report(
    summary: dict[str, Any], diagnostic_rows: list[dict[str, Any]]
) -> str:
    """Render the aggregate summary and every predicted-anomaly diagnostic row."""
    scope = summary.get("report_scope", {})
    db_counts = summary.get("database_counts", {})
    held_out = summary.get("held_out_test_evaluation") or {}
    descriptive = summary.get("full_labeled_dataset_confusion_matrix_descriptive_only", {})
    trace_count = int(scope.get("labeled_trace_count", 0))
    anomaly_count = int(scope.get("predicted_anomaly_count", len(diagnostic_rows)))
    normal_count = int(scope.get("predicted_normal_count", max(0, trace_count - anomaly_count)))
    error_count = int(scope.get("anomaly_diagnostic_errors", len(summary.get("diagnostic_errors", []))))
    threshold = summary.get("threshold", held_out.get("threshold", "N/A"))
    generated_at = html.escape(str(summary.get("generated_at", "Unknown")))
    model = html.escape(str(summary.get("model_name", "R1 model")))
    anomaly_rate = anomaly_count / trace_count if trace_count else 0.0
    candidate_count = int(summary.get("root_cause_candidates_reported_total", 0))
    diagnostic_statuses = summary.get("diagnostic_status_counts", {})
    incident_grouping = summary.get("incident_grouping", {})
    incident_grouping_html = _render_incident_grouping(incident_grouping)

    evidence_terms = (
        "Observed failure signature", "Selected anomaly anchor",
        "Reference processing markers", "Failure event line ID",
    )
    evidence_summary_count = sum(
        any(term in str(row.get("diagnostic_summary", "")) for term in evidence_terms)
        for row in diagnostic_rows
    )

    status_html = "".join(
        '<span class="status-chip">'
        f'{html.escape(str(status))} <strong>{int(count)}</strong>'
        '</span>'
        for status, count in sorted(diagnostic_statuses.items())
    ) or '<span class="muted">No diagnostic statuses recorded.</span>'

    if held_out:
        metric_html = f"""
        <div class="metric-grid">
          <div class="metric"><span>ROC-AUC</span><strong>{_number(held_out.get('roc_auc'))}</strong></div>
          <div class="metric"><span>Average precision (PR-AUC)</span><strong>{_number(held_out.get('pr_auc_average_precision'))}</strong></div>
          <div class="metric"><span>Precision</span><strong>{_percent(held_out.get('precision'))}</strong></div>
          <div class="metric"><span>Recall</span><strong>{_percent(held_out.get('recall'))}</strong></div>
          <div class="metric"><span>F1 score</span><strong>{_number(held_out.get('f1'))}</strong></div>
          <div class="metric"><span>Test samples</span><strong>{html.escape(_count(held_out.get('test_samples', 'N/A')))}</strong></div>
        </div>
        <p class="subtle-note">Official held-out test results. Threshold source: {html.escape(str(held_out.get('threshold_source', 'validation set')))}. The test set was not used to tune the threshold.</p>
        <h3>Held-out test confusion matrix</h3>
        <table class="matrix"><thead><tr><th>Actual / Predicted</th><th>Normal</th><th>Anomaly</th></tr></thead>
          <tbody><tr><th>Normal</th><td>{html.escape(_count(held_out.get('tn', 'N/A')))}</td><td>{html.escape(_count(held_out.get('fp', 'N/A')))}</td></tr>
          <tr><th>Anomaly</th><td>{html.escape(_count(held_out.get('fn', 'N/A')))}</td><td>{html.escape(_count(held_out.get('tp', 'N/A')))}</td></tr></tbody>
        </table>
        """
    else:
        metric_html = '<p class="warning-callout">Held-out test metrics were not available when this report was generated.</p>'

    anomaly_records = sorted(
        diagnostic_rows,
        key=lambda row: _score_for_sort(row.get("anomaly_score")),
        reverse=True,
    )
    table_rows = []
    for rank, row in enumerate(anomaly_records, start=1):
        block_id = str(row.get("block_id", ""))
        actual = str(row.get("actual_label", "Unknown"))
        predicted = str(row.get("predicted_label", "Anomaly"))
        status = str(row.get("root_cause_status", "unknown"))
        diagnosis = str(row.get("diagnostic_summary", ""))
        log_text = str(row.get("combined_log_text", ""))
        score = _number(row.get("anomaly_score"), 5)
        threshold_text = _number(row.get("threshold"), 5)
        correct = str(row.get("is_correct", "")).lower() in {"true", "1"}
        classification_text = "Correct classification" if correct else "Prediction differs from actual label"
        classification_class = "badge-positive" if correct else "badge-review"
        status_class = "badge-review" if status == "insufficient_evidence" or status == "diagnostic_error" else "badge-neutral"
        search = " ".join((block_id, actual, predicted, status, diagnosis, log_text)).lower()
        failure = _parse_json(row.get("failure_evidence_json"), {})
        processing = _parse_json(row.get("processing_evidence_json"), {})
        findings = _parse_json(row.get("diagnostic_findings_json"), [])
        anchor = _parse_json(row.get("anchor_event_json"), None)
        precursors = _parse_json(row.get("precursor_events_json"), [])
        candidates = _parse_json(row.get("root_cause_candidates_json"), [])
        candidate_evaluations = _parse_json(row.get("candidate_evaluations_json"), [])
        candidate_screening_summary = _parse_json(row.get("candidate_screening_summary_json"), {})
        preceding_context = _parse_json(row.get("preceding_context_json"), [])
        anchor_selection = _parse_json(row.get("anchor_selection_json"), {})
        method_notes = _parse_json(row.get("method_notes_json"), [])
        event_ordering = str(row.get("event_ordering", "not_recorded"))
        search += " " + str(preceding_context).lower() + " " + str(candidate_evaluations).lower()
        cross_block = _parse_json(row.get("cross_block_correlations_json"), {})
        search += " " + str(cross_block).lower()
        cross_block_html = _render_cross_block_correlations(cross_block)
        error = str(row.get("diagnostic_error", "") or "")
        error_html = (
            f'<p class="error-note"><strong>Diagnostic error:</strong> {html.escape(error)}</p>'
            if error else ""
        )
        timeline_html = _render_event_timeline(preceding_context, anchor, event_ordering)
        screening_html = _render_candidate_evaluations(candidate_evaluations, candidate_screening_summary)
        anchor_selection_html = _evidence("Anchor selection checks", anchor_selection)
        method_notes_html = _evidence("Diagnostic method notes", method_notes)
        table_rows.append(f"""
        <tr class="anomaly-row" data-search="{html.escape(search, quote=True)}">
          <td class="rank-cell">{rank}</td>
          <td><code>{html.escape(block_id)}</code><div class="small-muted">Actual: {html.escape(actual)}</div></td>
          <td><strong>{score}</strong><div class="small-muted">Threshold {threshold_text}</div></td>
          <td><span class="badge {status_class}">{html.escape(status)}</span></td>
          <td><span class="badge {classification_class}">{html.escape(classification_text)}</span></td>
          <td><p class="diagnostic-summary">{html.escape(diagnosis or 'No diagnostic summary returned.')}</p>
            <details class="trace-details"><summary>Inspect log and evidence</summary>
              {error_html}
              <section class="evidence-block"><h4>Combined log text</h4><pre class="log-text">{html.escape(log_text or 'No trace text available.')}</pre></section>
              {timeline_html}
              {screening_html}
              {anchor_selection_html}
              {method_notes_html}
              {_evidence('Selected anchor', anchor)}
              {_evidence('Diagnostic findings', findings)}
              {_evidence('Failure evidence', failure)}
              {_evidence('Processing evidence', processing)}
              {_evidence('Precursor events', precursors)}
              {cross_block_html}
              {_evidence('Root-cause candidates', candidates)}
            </details>
          </td>
        </tr>
        """)

    if not table_rows:
        table_rows.append('<tr><td colspan="6">No predicted anomalies found.</td></tr>')

    descriptive_matrix = f"""
      <h3>Full labeled-dataset comparison <span class="subtle-note">(descriptive only)</span></h3>
      <table class="matrix"><thead><tr><th>Actual / Predicted</th><th>Normal</th><th>Anomaly</th></tr></thead>
        <tbody><tr><th>Normal</th><td>{html.escape(_count(descriptive.get('true_negative', 0)))}</td><td>{html.escape(_count(descriptive.get('false_positive', 0)))}</td></tr>
        <tr><th>Anomaly</th><td>{html.escape(_count(descriptive.get('false_negative', 0)))}</td><td>{html.escape(_count(descriptive.get('true_positive', 0)))}</td></tr></tbody>
      </table>
      <p class="subtle-note">These counts describe stored predictions over the labeled dataset. They are not an independent held-out evaluation.</p>
    """
    if candidate_count:
        causal_note = (
            f"{candidate_count} potential candidate(s) were reported. "
            "Candidates are diagnostic associations, not confirmed causes."
        )
    else:
        causal_note = (
            "No root causes confirmed. The current diagnostic rules produced no supported candidates. "
            "Failure signatures, missing markers, and event order are investigation clues—not proof of causality. "
            "insufficient_evidence means the engine abstained, not that no underlying cause exists."
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light"><title>Intelligent Log Analyzer — Anomaly Report</title>
<style>
:root{{--ink:#172b4d;--muted:#61718a;--paper:#f4f7fb;--line:#dde5ef;--panel:#fff;--navy:#132b49;--blue:#2864a8;--amber:#8c5600;--amber-bg:#fff6e6;--green:#176448;--green-bg:#e8f7ef;--shadow:0 10px 30px rgba(19,43,73,.07)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}.container{{width:min(1240px,calc(100% - 40px));margin:auto;padding:28px 0 56px}}
.hero{{background:linear-gradient(125deg,#102943,#1e527b);color:white;border-radius:24px;padding:34px 38px;box-shadow:var(--shadow)}}.eyebrow{{text-transform:uppercase;letter-spacing:.15em;font-size:11px;font-weight:800;color:#b9d7f2}}h1{{margin:8px 0;font-size:clamp(30px,4vw,46px);letter-spacing:-.04em;line-height:1.1}}.hero p{{margin:0;color:#dceafa;max-width:820px}}.hero-meta{{margin-top:20px;padding-top:14px;border-top:1px solid #ffffff33;color:#d1e0f2;font-size:12px;display:flex;gap:20px;flex-wrap:wrap}}
.toolbar{{display:flex;justify-content:flex-end;margin:16px 0 24px}}button{{cursor:pointer;background:var(--navy);color:white;border:0;border-radius:10px;padding:10px 15px;font-weight:700}}.report-section{{margin-top:28px}}.section-heading{{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-bottom:14px}}h2{{font-size:23px;letter-spacing:-.025em;margin:0}}h3{{font-size:16px;margin:20px 0 10px}}.section-heading p{{margin:0;color:var(--muted);font-size:13px}}
.card,.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}}.card{{padding:20px}}.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}}.kpi{{padding:19px;min-width:0}}.label{{font-size:12px;color:var(--muted);font-weight:700}}.value{{font-weight:800;letter-spacing:-.04em;font-size:clamp(25px,3vw,36px);margin:5px 0 0}}.hint,.small-muted,.subtle-note{{color:var(--muted);font-size:12px}}.metric-grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}}.metric{{border:1px solid var(--line);padding:14px;border-radius:13px;background:#fbfdff}}.metric span{{display:block;color:var(--muted);font-size:12px;min-height:36px}}.metric strong{{font-size:25px;letter-spacing:-.03em}}
.warning-callout{{color:#744800;border:1px solid #f0d79f;background:var(--amber-bg);padding:16px 18px;border-radius:13px;margin:16px 0}}.matrix{{border-collapse:collapse;width:min(520px,100%);font-variant-numeric:tabular-nums}}.matrix th,.matrix td{{text-align:center;padding:11px 14px;border:1px solid var(--line)}}.matrix th{{background:#edf3f9;font-size:12px}}.matrix td{{background:white;font-size:16px;font-weight:800}}.status-row{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}}.status-chip{{display:inline-flex;gap:6px;background:#eff3f8;border-radius:100px;padding:6px 10px;color:#3d526b;font-size:12px}}
.table-toolbar{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}}input[type=search]{{flex:1;min-width:260px;padding:12px 14px;border-radius:10px;border:1px solid #ccd7e5;background:white;font:inherit}}.table-count{{align-self:center;font-size:12px;color:var(--muted)}}.table-wrap{{overflow-x:auto;border-radius:14px;border:1px solid var(--line);background:white}}table.anomaly-table{{border-collapse:collapse;width:100%;min-width:1030px}}.anomaly-table th{{position:sticky;top:0;text-align:left;background:#edf3f9;color:#43566e;font-size:11px;text-transform:uppercase;letter-spacing:.06em;padding:12px 10px;border-bottom:1px solid var(--line)}}.anomaly-table td{{padding:13px 10px;border-bottom:1px solid #e8edf4;vertical-align:top}}.rank-cell{{color:var(--muted);font-weight:700}}code{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;overflow-wrap:anywhere}}.badge{{display:inline-block;border-radius:999px;padding:4px 8px;font-size:11px;font-weight:800;line-height:1.4}}.badge-review{{color:var(--amber);background:#fff0d2}}.badge-positive{{color:var(--green);background:var(--green-bg)}}.badge-neutral{{color:#465a72;background:#eef2f6}}.diagnostic-summary{{max-width:600px;margin:0 0 8px;font-size:13px}}details.trace-details{{margin-top:8px}}details.trace-details summary{{cursor:pointer;font-size:12px;color:var(--blue);font-weight:800;padding:4px 0}}.evidence-block{{border:1px solid var(--line);border-radius:10px;padding:12px;margin-top:10px;background:#fbfdff}}.evidence-block h4{{margin:0 0 8px;font-size:13px}}.evidence-block pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;color:#263b52}}.log-text{{max-height:420px;overflow:auto}}.evidence-table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:12px}}.evidence-table th,.evidence-table td{{border:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}}.evidence-table th{{background:#edf3f9;color:#43566e;font-size:10px;text-transform:uppercase;letter-spacing:.04em}}.evidence-table td.message-cell{{min-width:260px;white-space:pre-wrap;overflow-wrap:anywhere}}.evidence-table .anchor-row{{background:#fff6e6}}.screening-summary{{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}}.screening-summary span{{background:#eff3f8;border-radius:8px;padding:6px 9px;font-size:11px}}.cross-block-table td{{min-width:130px;white-space:normal;overflow-wrap:anywhere}}.cross-block-table td:nth-child(4){{min-width:300px}}.error-note{{background:#fff0f0;color:#8d2525;padding:10px;border-radius:8px}}.footer{{margin-top:30px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:15px}}
@media(max-width:900px){{.kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}.metric-grid{{grid-template-columns:repeat(3,minmax(0,1fr))}}.hero{{padding:26px}}}}@media(max-width:560px){{.container{{width:calc(100% - 22px)}}.kpis,.metric-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.card{{padding:14px}}}}
@media print{{body{{background:white}}.container{{width:100%;padding:0}}.hero{{border-radius:0;box-shadow:none;print-color-adjust:exact}}.toolbar,.table-toolbar{{display:none!important}}.card,.kpi{{box-shadow:none;break-inside:avoid}}.table-wrap{{overflow:visible}}table.anomaly-table{{min-width:0;font-size:9px}}.anomaly-table th,.anomaly-table td{{padding:6px 4px}}details.trace-details summary{{color:#111}}details.trace-details:not([open])>*{{display:none!important}}.evidence-block{{break-inside:avoid}}}}
</style></head><body><div class="container">
<header class="hero"><div class="eyebrow">Log observability · Model-assisted triage</div><h1>Intelligent Log Analyzer</h1>
<p>Anomaly Detection &amp; Diagnostic Report. Prioritize traces using anomaly scores, then review recorded evidence. Correlations are investigation clues, not confirmed causal explanations.</p>
<div class="hero-meta"><span><strong>Model:</strong> {model}</span><span><strong>Generated:</strong> {generated_at}</span><span><strong>Threshold:</strong> {html.escape(_number(threshold, 8))}</span></div></header>
<div class="toolbar"><button type="button" onclick="window.print()">Print / Save as PDF</button></div>
{incident_grouping_html}
<section class="report-section"><div class="section-heading"><h2>Executive summary</h2><p>Scope: labeled traces only</p></div>
<div class="kpis">
<div class="kpi"><div class="label">Traces analyzed</div><div class="value">{trace_count:,}</div><div class="hint">{int(db_counts.get('unlabeled_traces', 0)):,} unlabeled trace excluded</div></div>
<div class="kpi"><div class="label">Predicted anomalies</div><div class="value">{anomaly_count:,}</div><div class="hint">{_percent(anomaly_rate)} of labeled traces</div></div>
<div class="kpi"><div class="label">Predicted normal</div><div class="value">{normal_count:,}</div><div class="hint">Model's normal classification</div></div>
<div class="kpi"><div class="label">Diagnostic errors</div><div class="value">{error_count:,}</div><div class="hint">Of {len(diagnostic_rows):,} diagnostic requests</div></div>
</div><div class="warning-callout"><strong>How to read this report:</strong> predictions are classifications; diagnostic findings are observed patterns. Missing markers do not prove a failed operation.</div></section>
<section class="report-section"><div class="section-heading"><h2>Model performance</h2><p>Official held-out test set · {html.escape(_count(held_out.get('test_samples', 'N/A')))} samples</p></div><div class="card">{metric_html}
<div class="warning-callout"><strong>Interpretation:</strong> test precision is {_percent(held_out.get('precision'))}; recall is {_percent(held_out.get('recall'))}. The model misses a substantial proportion of held-out anomalies at this threshold.</div>{descriptive_matrix}</div></section>
<section class="report-section"><div class="section-heading"><h2>Diagnostic overview</h2><p>Evidence for predicted anomalies</p></div><div class="card"><div class="status-row">{status_html}</div>
<p><strong>Summaries with specific observed evidence:</strong> {evidence_summary_count} of {len(diagnostic_rows)}.</p><p><strong>Root-cause candidates reported:</strong> {candidate_count}.</p><div class="warning-callout">{html.escape(causal_note)}</div></div></section>
<section class="report-section"><div class="section-heading"><h2>Anomaly investigation queue</h2><p>{len(diagnostic_rows):,} predicted anomalies · highest score first</p></div>
<div class="table-toolbar"><input id="reportSearch" type="search" placeholder="Search BlockId, evidence, or log text…" aria-label="Search anomaly report"><span class="table-count" id="visibleCount">Showing {len(diagnostic_rows)} traces</span></div>
<div class="table-wrap"><table class="anomaly-table"><thead><tr><th>#</th><th>Block ID / actual label</th><th>Score / threshold</th><th>Diagnostic status</th><th>Classification</th><th>Summary &amp; evidence</th></tr></thead><tbody id="anomalyRows">{''.join(table_rows) or '<tr><td colspan="6">No predicted anomalies found.</td></tr>'}</tbody></table></div>
<p class="subtle-note">A higher anomaly score means the R1 model considers a trace more anomalous. It is not a calibrated probability of failure.</p></section>
<footer class="footer">Generated from existing PostgreSQL predictions and saved R1 evaluation artifacts. This report does not retrain the model or alter database records. See <code>final_anomaly_report.csv</code> for all labeled traces, and <code>predicted_anomalies_diagnostics.csv</code> for full diagnostic fields.</footer>
</div><script>(()=>{{const input=document.getElementById('reportSearch');const rows=Array.from(document.querySelectorAll('.anomaly-row'));const counter=document.getElementById('visibleCount');function filterRows(){{const q=input.value.trim().toLowerCase();let visible=0;for(const row of rows){{const match=!q||(row.dataset.search||'').includes(q);row.hidden=!match;if(match)visible++;}}counter.textContent=`Showing ${{visible}} of ${{rows.length}} traces`;}}input.addEventListener('input',filterRows);}})();</script></body></html>"""



def _display_html_value(value: Any) -> str:
    """Render arbitrary diagnostic values safely for an HTML table cell."""
    if value is None or value == "":
        return "—"
    if isinstance(value, (list, tuple, set)):
        value = ", ".join(str(item) for item in value) if value else "—"
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return html.escape(str(value))


def _event_timestamp(event: dict[str, Any]) -> str:
    date = event.get("log_date")
    time = event.get("log_time")
    if date and time:
        return f"{date} {time}"
    return "—"


def _gap_text(value: Any) -> str:
    try:
        return f"{float(value):.3f} s"
    except (TypeError, ValueError):
        return "—"


def _render_event_timeline(
    preceding_context: Any, anchor: Any, event_ordering: Any
) -> str:
    """Render same-BlockId context followed by the selected anchor event."""
    contexts = preceding_context if isinstance(preceding_context, list) else []
    anchor = anchor if isinstance(anchor, dict) else None
    rows: list[str] = []

    for event in contexts:
        if not isinstance(event, dict):
            continue
        role = event.get("role", "context_only")
        role_label = "Candidate passed screen" if role == "ranked_candidate" else "Preceding context"
        reasons = event.get("candidate_rejection_reasons") or []
        screening = (
            "Passed candidate screen"
            if role == "ranked_candidate"
            else (", ".join(str(reason) for reason in reasons) if reasons else "Context only; not ranked")
        )
        rows.append(
            "<tr>"
            f"<td>{_display_html_value(role_label)}</td>"
            f"<td>{_display_html_value(_event_timestamp(event))}</td>"
            f"<td><code>{_display_html_value(event.get('line_id'))}</code></td>"
            f"<td>{_display_html_value(event.get('event_id'))}</td>"
            f"<td>{_display_html_value(event.get('component'))}</td>"
            f"<td>{_gap_text(event.get('seconds_before_anchor'))}</td>"
            f"<td>{_display_html_value(event.get('training_prevalence_difference'))}</td>"
            f"<td>{_display_html_value(screening)}</td>"
            f"<td class=\"message-cell\">{_display_html_value(event.get('message'))}</td>"
            "</tr>"
        )

    if anchor:
        rows.append(
            "<tr class=\"anchor-row\">"
            "<td>Selected failure / anchor</td>"
            f"<td>{_display_html_value(_event_timestamp(anchor))}</td>"
            f"<td><code>{_display_html_value(anchor.get('line_id'))}</code></td>"
            f"<td>{_display_html_value(anchor.get('event_id'))}</td>"
            f"<td>{_display_html_value(anchor.get('component'))}</td>"
            "<td>0.000 s</td><td>—</td><td>Selected anchor</td>"
            f"<td class=\"message-cell\">{_display_html_value(anchor.get('message') or anchor.get('template'))}</td>"
            "</tr>"
        )

    ordering_text = _display_html_value(event_ordering or "not recorded")
    if not rows:
        return (
            '<section class="evidence-block"><h4>Correlated event timeline</h4>'
            '<p>No ordered context was returned for this trace. Regenerate the report '
            'after refreshing diagnostics if the source is expected to include it.</p>'
            '<p class="small-muted">This report currently shows same-BlockId context only; '
            'cross-block correlation is not implemented.</p></section>'
        )

    return f"""
      <section class="evidence-block">
        <h4>Correlated event timeline</h4>
        <p class="small-muted">Ordering: {ordering_text}. Events shown before the anchor belong to this BlockId.
        Temporal proximity and training association are investigation clues, not proof of causality.
        Cross-block correlation is not implemented.</p>
        <div class="table-wrap"><table class="evidence-table">
          <thead><tr><th>Role</th><th>Timestamp</th><th>Line ID</th><th>Event ID</th><th>Component</th>
          <th>Seconds before anchor</th><th>Training prevalence difference</th><th>Screening</th><th>Observed message</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table></div>
      </section>
    """


def _render_candidate_evaluations(evaluations: Any, summary: Any) -> str:
    """Show why each preceding event passed or failed candidate screening."""
    evaluations = evaluations if isinstance(evaluations, list) else []
    summary = summary if isinstance(summary, dict) else {}
    chips = []
    for label, key in (
        ("Events screened", "preceding_event_instances_evaluated"),
        ("Eligible", "eligible_candidate_instances"),
        ("Rejected", "rejected_candidate_instances"),
        ("Minimum anomaly support", "minimum_anomaly_support"),
        ("Minimum prevalence difference", "minimum_prevalence_difference"),
        ("Maximum time gap (s)", "maximum_time_gap_seconds"),
    ):
        if key in summary:
            chips.append(f"<span><strong>{html.escape(label)}:</strong> {_display_html_value(summary.get(key))}</span>")

    reason_counts = summary.get("rejection_reason_counts")
    reason_html = ""
    if isinstance(reason_counts, dict) and reason_counts:
        reason_items = ", ".join(
            f"{html.escape(str(reason))}: {_display_html_value(count)}"
            for reason, count in sorted(reason_counts.items())
        )
        reason_html = f'<p class="small-muted"><strong>Rejection reason counts:</strong> {reason_items}</p>'

    rows: list[str] = []
    for item in evaluations:
        if not isinstance(item, dict):
            continue
        eligible = bool(item.get("eligible_as_candidate", False))
        verdict = "Eligible for ranking" if eligible else "Rejected"
        reasons = item.get("rejection_reasons") or []
        reason_text = ", ".join(str(reason) for reason in reasons) if reasons else ("—" if eligible else "No reason returned")
        rows.append(
            "<tr>"
            f"<td><code>{_display_html_value(item.get('line_id'))}</code></td>"
            f"<td>{_display_html_value(item.get('event_id'))}</td>"
            f"<td>{_display_html_value(item.get('component'))}</td>"
            f"<td>{_gap_text(item.get('seconds_before_anchor'))}</td>"
            f"<td>{_display_html_value(item.get('training_anomaly_blocks'))}</td>"
            f"<td>{_display_html_value(item.get('training_normal_blocks'))}</td>"
            f"<td>{_display_html_value(item.get('prevalence_difference'))}</td>"
            f"<td>{_display_html_value(verdict)}</td>"
            f"<td>{_display_html_value(reason_text)}</td>"
            f"<td class=\"message-cell\">{_display_html_value(item.get('message'))}</td>"
            "</tr>"
        )

    if not rows:
        body = '<p>No candidate instances were screened. This can happen when no eligible anchor event is found.</p>'
    else:
        body = f"""
          <div class="table-wrap"><table class="evidence-table">
            <thead><tr><th>Line ID</th><th>Event ID</th><th>Component</th><th>Seconds before anchor</th>
            <th>Training anomaly blocks</th><th>Training normal blocks</th><th>Prevalence difference</th>
            <th>Decision</th><th>Rejection reasons</th><th>Observed message</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table></div>
        """

    summary_note = summary.get("note")
    note_html = f'<p class="small-muted">{_display_html_value(summary_note)}</p>' if summary_note else ""
    return f"""
      <section class="evidence-block">
        <h4>Candidate screening and rejection reasons</h4>
        <p class="small-muted">An eligible event is only a candidate for ranking, not a confirmed cause. Rejection counts may count multiple reasons for one event.</p>
        <div class="screening-summary">{''.join(chips)}</div>
        {reason_html}{body}{note_html}
      </section>
    """



def _render_cross_block_correlations(value: Any) -> str:
    """Render cross-BlockId failure-signature associations without implying causality."""
    report = value if isinstance(value, dict) else {}
    status = str(report.get("status", "not_computed"))
    reason = str(report.get("reason", "Cross-block analysis was not recorded."))
    window = report.get("time_window_seconds", "N/A")
    candidates = report.get("candidates", [])
    candidates = candidates if isinstance(candidates, list) else []

    rows = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        pairs = candidate.get("matched_event_pairs", [])
        pairs = pairs if isinstance(pairs, list) else []
        pair_text = []
        for pair in pairs[:3]:
            if not isinstance(pair, dict):
                continue
            target = pair.get("target_event") or {}
            related = pair.get("related_event") or {}
            target_id = target.get("event_id", "?")
            related_id = related.get("event_id", "?")
            target_line = target.get("line_id", "?")
            related_line = related.get("line_id", "?")
            target_message = str(target.get("message", ""))
            related_message = str(related.get("message", ""))
            pair_text.append(
                f"<div><strong>{html.escape(str(target_id))} line {html.escape(str(target_line))}</strong>: "
                f"{html.escape(target_message)}<br>↳ <strong>{html.escape(str(related_id))} line {html.escape(str(related_line))}</strong>: "
                f"{html.escape(related_message)}</div>"
            )
        signatures = candidate.get("matched_signatures", [])
        signature_text = ", ".join(str(s) for s in signatures) if signatures else "—"
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(candidate.get('related_block_id', '')))}</code></td>"
            f"<td>{html.escape(str(candidate.get('nearest_time_gap_seconds', 'N/A')))} s</td>"
            f"<td>{html.escape(signature_text)}</td>"
            f"<td>{''.join(pair_text) or 'Matched event details unavailable.'}</td>"
            "<td>Potential temporal association; not a confirmed cause.</td>"
            "</tr>"
        )

    if rows:
        body = f"""
          <div class="table-wrap"><table class="evidence-table cross-block-table">
            <thead><tr><th>Related Block ID</th><th>Nearest gap</th><th>Matched signature(s)</th><th>Observed event pair(s)</th><th>Interpretation</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table></div>
        """
    else:
        body = '<p>No matching cross-block events were returned for this trace.</p>'

    return f"""
      <section class="evidence-block cross-block-evidence">
        <h4>Cross-block associations</h4>
        <p class="small-muted"><strong>Status:</strong> {html.escape(status)} · <strong>Window:</strong> ±{html.escape(str(window))} seconds</p>
        <p>{html.escape(reason)}</p>
        <p class="small-muted">Matches require the same specific classified failure signature in a different BlockId within the time window. These are investigation clues, not proof of a shared incident or root cause.</p>
        {body}
      </section>
    """


def _score_for_sort(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _render_incident_grouping(value: Any) -> str:
    """Render grouped cross-block associations without claiming causality."""
    report = value if isinstance(value, dict) else {}
    incidents = report.get("incidents", [])
    if not isinstance(incidents, list):
        incidents = []
    rows = []
    for incident in incidents:
        if not isinstance(incident, dict):
            continue
        time_range = incident.get("time_range", {})
        if not isinstance(time_range, dict):
            time_range = {}
        blocks = incident.get("block_ids", [])
        if not isinstance(blocks, list):
            blocks = []
        block_text = ", ".join(str(x) for x in blocks) or "N/A"
        start = html.escape(str(time_range.get("start", "N/A")))
        end = html.escape(str(time_range.get("end", "N/A")))
        duration = html.escape(str(time_range.get("duration_seconds", "N/A")))
        edges = html.escape(str(incident.get("correlation_edge_count", 0)))
        pairs = html.escape(str(incident.get("matched_event_pair_count", 0)))
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(incident.get('incident_id', 'N/A')))}</code></td>"
            f"<td>{html.escape(str(incident.get('failure_signature', 'N/A')))}</td>"
            f"<td>{html.escape(block_text)}</td>"
            f"<td>{start} &ndash; {end}</td>"
            f"<td>{duration} s</td>"
            f"<td>{edges} relationship(s) / {pairs} event pair(s)</td>"
            "</tr>"
        )
    body = "".join(rows) or (
        '<tr><td colspan="6">No incident groups could be formed from the available evidence.</td></tr>'
    )
    unique_pairs = html.escape(str(report.get("unique_event_pair_count", 0)))
    skipped = html.escape(str(report.get("skipped_pairs_missing_timestamps", 0)))
    return (
        '<section class="report-section" id="potential-incidents">'
        '<div class="section-heading"><h2>Potential Incidents</h2>'
        f"<p>{len(rows)} grouped incident(s)</p></div>"
        '<p class="small-muted">Grouped from the cross-block associations collected for this report. '
        f"Unique event pairs: {unique_pairs}. Pairs skipped for missing timestamps: {skipped}. "
        "These groups are investigation leads, not proof of a shared root cause.</p>"
        '<div class="table-wrap"><table class="evidence-table">'
        '<thead><tr><th>Incident</th><th>Failure signature</th><th>Involved blocks</th>'
        '<th>Time range (UTC)</th><th>Duration</th><th>Relationships / event pairs</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div></section>"
    )
