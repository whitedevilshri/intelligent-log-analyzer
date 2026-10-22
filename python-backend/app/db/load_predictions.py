
from app.db.database import get_connection
from app.services.anomaly_detector import load_artifacts
from ml.features.build_masked_tfidf import mask_text, preprocess_text

MODEL_NAME = "R1_masked_tfidf_isolation_forest"


def main():
    conn = get_connection()

    try:
        traces = conn.execute(
            """
            SELECT id, combined_log_text
            FROM traces
            ORDER BY id
            """
        ).fetchall()

        if not traces:
            raise RuntimeError("No traces found. Load the trace dataset first.")

        # Load the same model, vectorizer, and threshold used by the API.
        model, vectorizer, threshold = load_artifacts()

        trace_ids = []
        processed_texts = []

        for trace_id, log_text in traces:
            masked_text, _ = mask_text(log_text)
            processed_text = preprocess_text(masked_text)

            if not processed_text.strip():
                raise ValueError(f"Trace {trace_id} is empty after preprocessing.")

            trace_ids.append(trace_id)
            processed_texts.append(processed_text)

        # Predict in a batch using the existing R1 pipeline.
        features = vectorizer.transform(processed_texts)
        scores = -model.decision_function(features)

        records = [
            (
                trace_id,
                "Anomaly" if float(score) >= threshold else "Normal",
                float(score),
                float(threshold),
                MODEL_NAME,
            )
            for trace_id, score in zip(trace_ids, scores)
        ]

        # Replace previous results from this model to avoid duplicates
        # when this script is run again.
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM anomaly_predictions WHERE model_name = %s",
                (MODEL_NAME,),
            )

            cur.executemany(
                """
                INSERT INTO anomaly_predictions
                    (trace_id, prediction, anomaly_score, threshold, model_name)
                VALUES (%s, %s, %s, %s, %s)
                """,
                records,
            )

        conn.commit()

        print(f"Traces processed: {len(records)}")
        print("Prediction counts:")

        counts = conn.execute(
            """
            SELECT prediction, COUNT(*)
            FROM anomaly_predictions
            WHERE model_name = %s
            GROUP BY prediction
            ORDER BY prediction
            """,
            (MODEL_NAME,),
        ).fetchall()

        for prediction, count in counts:
            print(f"  {prediction}: {count}")

        print("R1 predictions saved successfully.")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
