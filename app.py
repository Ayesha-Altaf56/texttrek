
import re
import streamlit as st

from src.retrieval import load_data, RetrievalEngine
from src.answering import is_urdu_script
from src.semantic_retrieval import SemanticRetriever
from src.hybrid_retrieval import HybridRetriever


st.set_page_config(
    page_title="Multilingual E-Sahulat AI",
    page_icon="🔎",
    layout="wide"
)


@st.cache_resource
def load_keyword_system():
    data = load_data()
    engine = RetrievalEngine(data)
    return data, engine


@st.cache_resource
def load_semantic_system():
    data, _ = load_keyword_system()
    return SemanticRetriever(data)


def is_answerable(value):
    return str(value).strip().lower() not in (
        "false", "0", "no", "none", ""
    )


def main():
    st.title("Multilingual E-Sahulat AI")
    st.write(
        "An evidence-grounded assistant for finding information "
        "from the provided Pakistan public-service dataset."
    )

    st.info(
        "This prototype uses local retrieval and dataset evidence. "
        "It does not call paid AI APIs."
    )

    try:
        data, keyword_engine = load_keyword_system()
    except Exception as error:
        st.error(f"Could not load the dataset: {error}")
        st.stop()

    st.caption(f"Dataset records loaded: {len(data)}")

    query = st.text_area(
        "Ask your question / اپنا سوال لکھیں",
        placeholder="Example: Pakistan ka passport kon banwa sakta hai?",
        height=100
    )

    method = st.selectbox(
        "Choose retrieval method",
        [
            "Hybrid",
            "TF-IDF",
            "BM25",
            "Semantic"
        ]
    )

    top_k = st.slider(
        "Number of evidence results",
        min_value=1,
        max_value=5,
        value=3
    )

    default_thresholds = {
        "TF-IDF": 0.15,
        "BM25": 0.50,
        "Semantic": 0.30,
        "Hybrid": 0.20
    }

    threshold = st.slider(
        "Abstention threshold (provisional)",
        min_value=0.0,
        max_value=1.0,
        value=default_thresholds[method],
        step=0.05,
        help=(
            "These are initial thresholds, not calibrated values. "
            "Tune them using validation data before reporting results."
        )
    )

    if st.button("Find Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question first.")
            st.stop()

        try:
            if method == "TF-IDF":
                results = keyword_engine.search(
                    query, method="tfidf", top_k=top_k
                )

            elif method == "BM25":
                results = keyword_engine.search(
                    query, method="bm25", top_k=top_k
                )

            elif method == "Semantic":
                semantic_engine = load_semantic_system()
                results = semantic_engine.search(query, top_k=top_k)

            else:
                semantic_engine = load_semantic_system()
                hybrid_engine = HybridRetriever(
                    keyword_engine=keyword_engine,
                    semantic_engine=semantic_engine
                )
                results = hybrid_engine.search(query, top_k=top_k)

        except Exception as error:
            st.error(f"Search failed: {error}")
            st.stop()

        if results.empty:
            st.warning(
                "I could not find supporting evidence. "
                "Please try rephrasing your question."
            )
            st.stop()

        best = results.iloc[0]
        score = float(best["retrieval_score"])

        st.subheader("Answer")

        if not is_answerable(best.get("answerable", True)):
            st.warning(
                "The dataset marks this question as unanswerable. "
                "I will not provide an unsupported answer."
            )

        elif score < threshold:
            st.warning(
                "I could not find sufficiently strong evidence. "
                "Please rephrase your question or consult an official source."
            )

        else:
            if is_urdu_script(query):
                answer = best.get("gold_evidence_ur", "")
            else:
                answer = best.get("gold_evidence_en", "")

            if not isinstance(answer, str) or not answer.strip():
                answer = best.get("gold_evidence_en", "")

            if isinstance(answer, str) and answer.strip():
                st.write(answer)
                st.caption(
                    "Displayed text is retrieved dataset evidence, "
                    "not a newly generated answer."
                )
            else:
                st.warning(
                    "A matching record was found, but it has no "
                    "usable answer evidence."
                )

        st.caption(f"Top retrieval score: {score:.4f}")

        st.subheader("Retrieved Evidence")

        for rank, (_, result) in enumerate(results.iterrows(), start=1):
            with st.expander(
                f"{rank}. {result.get('qid', 'Record')} — "
                f"Score: {float(result['retrieval_score']):.4f}",
                expanded=(rank == 1)
            ):
                st.write("**Matched query:**")
                st.write(result.get("query", "Not available"))

                english_evidence = result.get("gold_evidence_en", "")
                urdu_evidence = result.get("gold_evidence_ur", "")

                if isinstance(english_evidence, str) and english_evidence.strip():
                    st.write("**English evidence:**")
                    st.write(english_evidence)

                if isinstance(urdu_evidence, str) and urdu_evidence.strip():
                    st.write("**Urdu evidence:**")
                    st.write(urdu_evidence)

                source = result.get("source_url", "")
                if isinstance(source, str) and source.startswith(("https://", "http://")):
                    st.markdown(f"[View source]({source})")

                st.caption(
                    f"Intent: {result.get('intent_id', 'Not available')} | "
                    f"Topic: {result.get('topic', 'Not available')}"
                )

        st.caption(
            "Research note: retrieval scores and abstention thresholds "
            "must be evaluated on a separate validation set."
        )


if __name__ == "__main__":
    main()
