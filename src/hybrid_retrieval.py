
import pandas as pd


class HybridRetriever:
    def __init__(
        self,
        keyword_engine,
        semantic_engine,
        tfidf_weight=0.25,
        bm25_weight=0.25,
        semantic_weight=0.50,
    ):
        self.keyword_engine = keyword_engine
        self.semantic_engine = semantic_engine

        weights = [tfidf_weight, bm25_weight, semantic_weight]

        if any(weight < 0 for weight in weights) or sum(weights) == 0:
            raise ValueError("Weights must be non-negative and not all zero.")

        total = sum(weights)
        self.weights = [weight / total for weight in weights]

    def search(self, query, top_k=5):
        # Retrieve scores for every dataset record.
        tfidf_results = self.keyword_engine.search(
            query, method="tfidf", top_k=len(self.keyword_engine.df)
        )

        bm25_results = self.keyword_engine.search(
            query, method="bm25", top_k=len(self.keyword_engine.df)
        )

        semantic_results = self.semantic_engine.search(
            query, top_k=len(self.semantic_engine.df)
        )

        # Align scores by record ID.
        tfidf_scores = tfidf_results.set_index("qid")["retrieval_score"]
        bm25_scores = bm25_results.set_index("qid")["retrieval_score"]
        semantic_scores = semantic_results.set_index("qid")["retrieval_score"]

        # Normalize each method's scores to [0, 1].
        def normalize_scores(scores):
            minimum = scores.min()
            maximum = scores.max()

            if maximum == minimum:
                return scores * 0

            return (scores - minimum) / (maximum - minimum)

        tfidf_scores = normalize_scores(tfidf_scores)
        bm25_scores = normalize_scores(bm25_scores)
        semantic_scores = normalize_scores(semantic_scores)

        combined = pd.DataFrame({
            "tfidf_score": tfidf_scores,
            "bm25_score": bm25_scores,
            "semantic_score": semantic_scores,
        }).fillna(0)

        combined["hybrid_score"] = (
            self.weights[0] * combined["tfidf_score"]
            + self.weights[1] * combined["bm25_score"]
            + self.weights[2] * combined["semantic_score"]
        )

        # Preserve original dataset fields.
        results = self.keyword_engine.df.copy()
        results = results.set_index("qid", drop=False)

        results["tfidf_score"] = combined["tfidf_score"]
        results["bm25_score"] = combined["bm25_score"]
        results["semantic_score"] = combined["semantic_score"]
        results["retrieval_score"] = combined["hybrid_score"]

        results = results.sort_values(
            "retrieval_score", ascending=False
        )

        return results.head(top_k).reset_index(drop=True)
