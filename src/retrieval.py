
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


# Locate the project folder and dataset.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.txt"


def normalize_text(text):
    """Normalize text while preserving Unicode characters."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data():
    """Load JSON Lines or JSON arrays while skipping plain-text headings."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place the dataset at data/dataset.txt."
        )

    text = DATA_PATH.read_text(encoding="utf-8-sig").strip()

    if not text:
        raise ValueError("The dataset file is empty.")

    # Remove optional Markdown code fences.
    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("```"):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # First try parsing the whole file as JSON.
    try:
        data = json.loads(text)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            raise ValueError("The dataset must contain JSON objects.")

    except json.JSONDecodeError:
        # Otherwise parse one JSON object per line.
        records = []

        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()

            # Ignore headings and other lines that aren't JSON objects.
            if not line.startswith("{"):
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: "
                    f"{line[:150]!r}"
                ) from error

            if isinstance(record, dict):
                records.append(record)

    if not records:
        raise ValueError(
            "No JSON records found. Check the dataset format and filename."
        )

    df = pd.DataFrame(records)

    required_columns = [
        "qid",
        "query",
        "gold_evidence_en",
        "gold_evidence_ur",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Prevent missing evidence fields from breaking retrieval.
    for column in [
        "query",
        "gold_evidence_en",
        "gold_evidence_ur",
    ]:
        df[column] = df[column].fillna("").astype(str)

    return df


class RetrievalEngine:
    """Baseline search engine using TF-IDF and BM25."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True).copy()

        # Build searchable documents from questions and evidence.
        self.df["search_text"] = (
            self.df["query"]
            + " "
            + self.df["gold_evidence_en"]
            + " "
            + self.df["gold_evidence_ur"]
        )

        self.documents = [
            normalize_text(document)
            for document in self.df["search_text"]
        ]

        # Character n-grams help with spelling variations.
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.documents
        )

        # BM25 uses word tokens as its baseline.
        self.tokenized_documents = [
            document.split() for document in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

    def search(self, query, method="tfidf", top_k=5):
        """Return the highest-ranked matching dataset records."""
        if method not in ("tfidf", "bm25"):
            raise ValueError("Choose method='tfidf' or method='bm25'.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        normalized_query = normalize_text(query)

        if not normalized_query:
            return self.df.head(0).copy()

        if method == "tfidf":
            query_vector = self.vectorizer.transform(
                [normalized_query]
            )

            scores = cosine_similarity(
                query_vector,
                self.tfidf_matrix,
            ).ravel()

        else:
            scores = self.bm25.get_scores(
                normalized_query.split()
            )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = self.df.iloc[ranked_indices].copy()

        results["retrieval_score"] = [
            float(scores[index])
            for index in ranked_indices
        ]

        return results.reset_index(drop=True)
