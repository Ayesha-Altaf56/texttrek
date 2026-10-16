
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval import normalize_text


# Multilingual model that runs locally after download.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SemanticRetriever:
    def __init__(self, df):
        self.df = df.reset_index(drop=True).copy()

        # Build searchable documents from questions and evidence.
        self.documents = (
            self.df["query"].fillna("").astype(str)
            + " "
            + self.df["gold_evidence_en"].fillna("").astype(str)
            + " "
            + self.df["gold_evidence_ur"].fillna("").astype(str)
        ).tolist()

        print("Loading multilingual model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Encoding dataset...")
        self.document_embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        print("Semantic search is ready.")

    def search(self, query, top_k=5):
        if not str(query).strip():
            return self.df.head(0).copy()

        query_embedding = self.model.encode(
            [str(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores = cosine_similarity(
            query_embedding,
            self.document_embeddings,
        )[0]

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = self.df.iloc[ranked_indices].copy()
        results["retrieval_score"] = [
            float(scores[index]) for index in ranked_indices
        ]

        return results.reset_index(drop=True)
