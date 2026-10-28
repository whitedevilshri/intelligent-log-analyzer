# Intelligent Log Analyzer

A Java and Python application for ingesting structured HDFS logs, grouping events into traces, detecting anomalous traces with machine learning, and providing diagnostic evidence for investigation.

The root-cause component reports evidence and correlations. It does not guarantee that a reported event is the actual cause of a failure.

## Technology Stack

- Java 25 and Maven — structured log ingestion
- Python and FastAPI — backend REST APIs
- PostgreSQL — persistent log events, traces, and predictions
- scikit-learn — masked TF-IDF and Isolation Forest anomaly detection
- JUnit — Java unit tests
- Python `unittest` — backend tests

## Architecture

1. Java reads a structured CSV and validates its required fields.
2. Java extracts BlockIds and sends records to FastAPI in batches of up to 500 records.
3. FastAPI persists structured events and their BlockId relationships in PostgreSQL.
4. Trace construction groups log content by BlockId in memory.
5. The labeling workflow matches constructed traces with HDFS anomaly labels and writes a labeled CSV.
6. Loader scripts can populate PostgreSQL from the labeled dataset and the existing R1 model.
7. The R1 API classifies an individual submitted trace and saves its prediction.
8. The diagnostic endpoint returns event evidence and candidate evaluations when supported by the available evidence.

Trace construction, CSV labeling, bulk database loading, and individual API prediction are separate operations. Calling the trace builder does not automatically persist traces or generate predictions.

## Requirements

- PostgreSQL
- Python 3.14.3 (the current development environment)
- JDK 25
- Apache Maven
- A compatible structured HDFS log CSV
- The matching HDFS anomaly-label CSV for the labeling workflow

Input datasets and generated data artifacts are not all included in Git.

## Configuration

From the project root, create your local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and configure the PostgreSQL connection:

```dotenv
DB_HOST=localhost
DB_PORT=5432
DB_NAME=intelligent_log_analyzer
DB_USER=postgres
DB_PASSWORD=replace_with_your_password
```

Never commit `.env` or real database credentials.

## PostgreSQL Setup

First create a PostgreSQL database named `intelligent_log_analyzer`.

Using pgAdmin or `psql`, execute these schema files against that database, in this order:

1. `python-backend/app/db/event_schema.sql`
2. `python-backend/app/db/schema.sql`

The first creates structured event tables and BlockId relationships. The second creates trace and prediction tables. These scripts use `CREATE TABLE IF NOT EXISTS`; they initialize a compatible schema and are not migrations for arbitrary older schemas.

The application does not automatically initialize the database schema at startup.

## Python Backend

From the project root, create and activate the virtual environment, then install the pinned dependencies:

```powershell
cd python-backend
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

Start FastAPI from the `python-backend` directory:

```powershell
python -m uvicorn app.main:app --reload
```

Useful URLs:

- Health: `http://127.0.0.1:8000/health`
- Swagger documentation: `http://127.0.0.1:8000/docs`
- OpenAPI specification: `http://127.0.0.1:8000/openapi.json`

## Java Ingestion

Run Java tests from the `ingestion-java` directory:

```powershell
cd ingestion-java
mvn test
```

Build the application and generate its dependency classpath:

```powershell
mvn -DskipTests package dependency:build-classpath "-Dmdep.outputFile=target\classpath.txt"
```

Run the ingestion application with a compatible structured CSV:

```powershell
$classpath = (Get-Content .\target\classpath.txt -Raw).Trim()

java -cp "target\classes;$classpath" `
    com.intelligentloganalyzer.ingestion.App `
    "C:\path\to\HDFS_100k.log_structured.csv"
```

The input CSV must contain these columns:

`LineId, Date, Time, Pid, Level, Component, Content, EventId, EventTemplate`

Each record must contain exactly one distinct BlockId in its `Content`.

The default ingestion endpoint is `http://127.0.0.1:8000/api/v1/ingestion/batches`. It can be overridden using the Java system property `intelligent.log.analyzer.api-url` or the environment variable `INTELLIGENT_LOG_ANALYZER_API_URL`.

**Warning:** running ingestion against the full dataset can update existing LineIds and replace their BlockId mappings. Only do so when a dataset refresh is intended.

## Data and Trace Workflow

The following operations are separate:

1. `POST /api/v1/ingestion/batches` persists a batch of structured log records.
2. `POST /api/v1/traces/build` groups existing records into traces in memory.
3. `POST /api/v1/traces/label` matches traces against `data/raw/anomaly_label.csv` and writes `data/processed/HDFS_trace_labeled_dataset.csv`.
4. From the `python-backend` directory, `python -m app.db.load_traces` loads the labeled CSV into PostgreSQL.
5. From the `python-backend` directory, `python -m app.db.load_predictions` rebuilds R1 predictions for every trace currently in PostgreSQL.

Run loader commands only with the virtual environment activated and the PostgreSQL connection configured. These operations change data:

- The trace-labeling endpoint overwrites its generated CSV output.
- The trace loader updates existing rows whose `block_id` matches a CSV row; it does not remove database rows absent from the CSV.
- The prediction loader deletes existing predictions for the R1 model and recreates them in a transaction.

The individual trace endpoint, `POST /api/v1/predictions/trace`, runs R1 and persists a new API-submitted trace and its prediction.

## Anomaly Report Generation

The reporting utility generates reproducible anomaly reports from the existing PostgreSQL traces and R1 predictions. It does not retrain the model or modify database records.

Run from the `python-backend` directory with the virtual environment activated and PostgreSQL configured:

`python .\ml\reporting\generate_anomaly_report.py`

The utility validates prediction coverage and checks the score threshold against the saved R1 validation metadata. It refuses to overwrite existing report files unless `--force` is provided.

Generated files are saved under the repository's `reports/` directory:

- `final_anomaly_report.csv` — one row for each labeled trace, including actual and predicted labels, anomaly score, threshold, and correctness.
- `predicted_anomalies_diagnostics.csv` — diagnostic evidence for traces predicted as anomalous.
- `anomaly_report_summary.json` — dataset counts, descriptive confusion matrix, diagnostic status counts, and the separate held-out test metrics.

For the current report, 7,940 labeled traces were processed, including 151 predicted anomalies. All 151 root-cause diagnostic reports returned `insufficient_evidence`; no root-cause candidates were produced. Cross-block correlation is a separate analysis that can identify related events in other blocks. Such associations are investigation leads, not proof of root cause. Insufficient diagnostic evidence does not mean an anomaly has no underlying cause.

The full-dataset confusion matrix is descriptive and must not be presented as an independent test evaluation. Use the separately recorded held-out test metrics when reporting model performance.

The detailed CSVs contain dataset-derived log text and are generated locally rather than tracked in Git. The reporting script and compact JSON summary can be version-controlled.

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/v1/ingestion/batches` | Persist an ingestion batch |
| GET | `/api/v1/ingestion/stats` | Read ingestion statistics |
| POST | `/api/v1/traces/build` | Build traces in memory |
| POST | `/api/v1/traces/label` | Write the labeled trace CSV |
| POST | `/api/v1/predictions/trace` | Predict and save one submitted trace |
| GET | `/api/v1/predictions/history` | Read prediction history |
| GET | `/api/v1/dashboard/summary` | Read dashboard summary statistics |
| GET | `/api/v1/root-cause/blocks/{block_id}` | Retrieve diagnostic evidence |
| GET | `/api/v1/root-cause/blocks/{block_id}/cross-block-correlations` | Retrieve cross-block correlation evidence |
| POST | `/api/v1/root-cause/incidents` | Group cross-block associations into potential incident clusters |

## Automated Tests

Run Python tests from `python-backend`:

```powershell
python -m unittest discover -s tests -v
```

Run Java tests from `ingestion-java`:

```powershell
mvn test
```

At the latest verified checkpoint, 41 Python tests and 16 Java tests passed. Rerun both suites after making changes.

## R1 Model

Production model: `R1_masked_tfidf_isolation_forest`.

The serving pipeline applies the same text masking and preprocessing used during training. It uses the saved TF-IDF vectorizer, trained Isolation Forest model, and validation-selected anomaly-score threshold. The anomaly score is calculated as the negative Isolation Forest decision function, so higher scores indicate greater anomaly likelihood.

### Recorded validation results

- Samples: 1,271
- ROC-AUC: 0.7937
- Average precision (PR-AUC): 0.5630
- Precision: 0.9615
- Recall: 0.5000
- F1-score: 0.6579

### Recorded held-out test results

- Samples: 1,588
- Threshold source: validation set
- Threshold: -0.15430605
- ROC-AUC: 0.6830
- Average precision (PR-AUC): 0.4526
- Precision: 1.0000
- Recall: 0.3810
- F1-score: 0.5517
- True negatives: 1,525
- False positives: 0
- False negatives: 39
- True positives: 24

These are the saved evaluation results for the current R1 experiment. At the selected threshold, the model misses a substantial proportion of anomalies. Do not describe its recall as high.

The trained R1 model, vectorizer, metadata, and threshold metrics are version-controlled. Raw datasets, split files, and generated feature matrices are excluded from Git; reproducing the full evaluation requires the corresponding data and features.

Do not rerun the training script merely to reproduce predictions. Training overwrites the saved model and validation metrics.

## Diagnostic Limitations

The root-cause correlator evaluates event associations, event sequences, failure signatures, and processing-marker evidence. It can return `insufficient_evidence` with no eligible anchor and no root-cause candidates. Common events are not necessarily useful discriminators, and observed sequence relationships do not prove causality.

Describe results as diagnostic evidence for investigation rather than confirmed root causes.

## Git and Generated Files

- Keep `.env` and other secrets out of Git.
- Python virtual environments, Java build outputs, datasets, and generated feature matrices are ignored.
- Keep the R1 model, vectorizer, metadata, and threshold metrics together.
- Use the held-out test set only for final evaluation, not threshold tuning.

## Current Project Status

The Java ingestion client, FastAPI endpoints, PostgreSQL storage, R1 scoring pipeline, and diagnostic endpoint are implemented. Integration and automated tests have been exercised in the current development environment.

The root-cause component remains evidence-oriented and may abstain when the data do not support a candidate. The system should be presented with that limitation clearly stated.
