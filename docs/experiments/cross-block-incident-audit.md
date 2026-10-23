# Cross-Block Incident Correlation Experiment

## Objective
Investigate whether events from other blocks shortly before E7 failures
provide repeatable evidence for root-cause analysis.

## Data and safeguards
- Training traces: 5,081
- Validation traces: 1,271
- Structured events analysed: 83,949
- Held-out test split: not read
- Production API, model, and database: unchanged

## Findings
- 59 E7 log entries were observed.
- The 30-second grouping produced 4 episodes.
- The 60- and 120-second groupings produced 3 episodes each.
- Only 1 pre-incident window was eligible under the audit clearance rule.
- Control windows were excluded because of nearby E7 activity or because
  they contained no events belonging to the selected training/validation
  blocks.
- The initial IP correlation was affected by overlapping control windows.
  The observed IP activity also included ordinary block-transfer messages.

## Conclusion
The current data and window coverage are insufficient to establish a
repeatable cross-block precursor or a reliable root cause.

No cross-block root-cause rule will be added based on this experiment.
The IP reference 10.251.106.10 remains an unvalidated lead.

## Limitations
The existing split distributes blocks rather than independent time
episodes. Training and validation contain blocks from the same large E7
episode, so they do not provide independent evidence of generalization.
Empty windows in this audit mean no events from the selected
training/validation blocks were available; they do not prove the full
system had no activity during those times.

## Decision
Retain the R1 anomaly detector unchanged. Continue improving
trace-local diagnostic evidence and ensure the API distinguishes observed
errors, processing-marker evidence, and unconfirmed root-cause hypotheses.
