
import time
import pandas as pd

from src.retrieval import RetrievalEngine
from src.semantic_retrieval import SemanticRetriever
from src.hybrid_retrieval import HybridRetriever


def evaluate_retrieval(data, k=5):
    train_data = data[data["split"] == "train_dev"].copy()
    test_data = data[data["split"] == "test"].copy()

    # Exclude missing intent IDs from the retrieval relevance index.
    valid_train = train_data.dropna(subset=["intent_id"]).copy()

    keyword_engine = RetrievalEngine(train_data)
    semantic_engine = SemanticRetriever(train_data)
    hybrid_engine = HybridRetriever(keyword_engine, semantic_engine)

    engines = {
        "tfidf": keyword_engine,
        "bm25": keyword_engine,
        "semantic": semantic_engine,
        "hybrid": hybrid_engine
    }

    records = []

    for _, query_row in test_data.iterrows():
        query = str(query_row["query"])
        intent = query_row["intent_id"]

        # Relevance is defined by matching intent IDs.
        if pd.isna(intent):
            relevant = set()
        else:
            relevant = set(
                valid_train.loc[
                    valid_train["intent_id"] == intent, "qid"
                ].astype(str)
            )

        for method, engine in engines.items():
            start = time.perf_counter()

            if method in ("tfidf", "bm25"):
                retrieved_results = engine.search(
                    query, method=method, top_k=k
                )
            else:
                retrieved_results = engine.search(
                    query, top_k=k
                )

            latency_ms = (time.perf_counter() - start) * 1000
            retrieved = (
                retrieved_results["qid"].astype(str).tolist()
            )

            if not relevant:
                recall = float("nan")
                reciprocal_rank = float("nan")
            else:
                hits = sum(item in relevant for item in retrieved)
                recall = len(set(retrieved) & relevant) / len(relevant)

                reciprocal_rank = 0.0
                for rank, item in enumerate(retrieved, start=1):
                    if item in relevant:
                        reciprocal_rank = 1 / rank
                        break

            records.append({
                "query_id": str(query_row["qid"]),
                "language": query_row["language"],
                "method": method,
                "recall@5": recall,
                "mrr@5": reciprocal_rank,
                "latency_ms": latency_ms
            })

    return pd.DataFrame(records)


def summarize_results(results):
    summary = results.groupby("method").agg(
        queries_total=("query_id", "count"),
        queries_with_relevance=("recall@5", "count"),
        recall_at_5=("recall@5", "mean"),
        mrr_at_5=("mrr@5", "mean"),
        average_latency_ms=("latency_ms", "mean")
    ).reset_index()

    return summary.sort_values(
        by="recall_at_5",
        ascending=False
    )
