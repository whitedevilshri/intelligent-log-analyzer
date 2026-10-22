# Intelligent Log Analyzer

An intelligent log analysis system that ingests structured logs, detects
anomalous traces, stores log data, and provides diagnostic evidence to help
investigate possible root causes.

## Technology Stack

- Java and Maven for log ingestion
- Python and FastAPI for the backend APIs
- PostgreSQL for persistent storage
- scikit-learn for machine-learning-based anomaly detection

## Architecture

1. **Log Ingestion:** Java reads structured log data and sends it to the
   backend in batches.
2. **Database:** PostgreSQL stores structured events, block relationships,
   traces, predictions, and related history.
3. **Anomaly Detection:** The Python backend uses the R1 masked TF-IDF and
   Isolation Forest model to classify traces.
4. **Root-Cause Analysis:** The correlator examines errors, event sequences,
   processing markers, and historical event associations to find diagnostic
   evidence.
5. **REST APIs:** FastAPI exposes endpoints for ingestion, predictions,
   prediction history, dashboard statistics, and root-cause analysis.

## Current Status

- Java log ingestion implemented and tested.
- PostgreSQL integration implemented.
- FastAPI endpoints implemented and tested.
- R1 anomaly detection integrated.
- Root-cause correlation is still under development and evaluation.

The root-cause component reports diagnostic evidence, not proven causality.
It may return `insufficient_evidence` when there is not enough support for
a reliable candidate.

## Configuration

Copy `.env.example` to `.env` and configure the PostgreSQL connection
settings for your local environment.

Do not commit `.env` or real database credentials.

## Model Artifacts

The R1 masked model, its TF-IDF vectorizer, and its metrics file are stored
under `data/processed/`. They must remain together because the backend loads
all three for prediction.

Datasets and generated feature matrices are excluded from Git.

## Development Notes

- Keep the R1 anomaly-detection model unchanged while experimenting with
  root-cause correlation.
- Use the training and validation splits for development.
- Reserve the held-out test split for final evaluation.
- Treat event correlations as evidence to investigate, not proof of cause.
