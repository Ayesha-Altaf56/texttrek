
import pandas as pd
from src.answering import answer_query


def evaluate_answers(data, engine, threshold=0.15):
    test_data = data[data["split"] == "test"].copy()
    records = []

    for _, row in test_data.iterrows():
        query = str(row["query"])

        response = answer_query(
            engine,
            query,
            method="tfidf",
            threshold=threshold
        )

        abstained = response["status"] == "abstained"
        expected_abstention = not (
            str(row["answerable"]).strip().lower()
            in ("true", "1", "yes")
        )

        records.append({
            "qid": row["qid"],
            "language": row["language"],
            "expected_answerable": not expected_abstention,
            "abstained": abstained,
            "abstention_correct": (
                abstained == expected_abstention
            ),
            "has_source": bool(response.get("source")),
            "score": response.get("score", 0.0),
            "answer": response.get("answer", "")
        })

    return pd.DataFrame(records)


def summarize_answers(results):
    return pd.Series({
        "Test queries": len(results),
        "Correct abstention decisions (%)": (
            results["abstention_correct"].mean() * 100
        ),
        "Queries with source (%)": (
            results["has_source"].mean() * 100
        ),
        "Abstained queries": int(results["abstained"].sum())
    })
