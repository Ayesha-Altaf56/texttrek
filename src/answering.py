
import re


def is_urdu_script(text):
    """Check whether the query contains Urdu/Arabic script."""
    return bool(re.search(r"[\u0600-\u06FF]", str(text)))

def is_roman_urdu(text):
    """Check whether the query is written in Roman Urdu."""
    words = set(str(text).lower().split())

    roman_urdu_words = {
        "kya", "kon", "kaun", "hasil", "sakta", "sakti",
        "hai", "hain", "kaise", "kahan", "kis", "ko",
        "ke", "ki", "aam", "kar", "karta", "karti",
        "chahiye", "batao", "karo", "mera", "meri",
        "passport"
    }

    return len(words.intersection(roman_urdu_words)) >= 2

def answer_query(engine, query, method="tfidf", threshold=0.3418):
    """
    Return retrieved evidence with a source citation.
    Abstain when the score is below the provisional threshold.
    """
    results = engine.search(query, method=method, top_k=1)

    if results.empty:
        return {
            "status": "abstained",
            "answer": "I could not find sufficient evidence.",
            "source": None,
            "score": 0.0,
        }

    result = results.iloc[0]
    score = float(result["retrieval_score"])

    if score < threshold:
        return {
            "status": "abstained",
            "answer": "I could not find sufficient evidence.",
            "source": None,
            "score": score,
        }

    # Do not use the retrieved record's answerable field to decide
    # whether the original query is answerable. It may be unrelated.
    if is_urdu_script(query):
        evidence = result.get("gold_evidence_ur", "")
    else:
        evidence = result.get("gold_evidence_en", "")

    source = result.get("source_url", None)

    if not isinstance(evidence, str) or not evidence.strip():
        return {
            "status": "abstained",
            "answer": "No usable evidence was found.",
            "source": None,
            "score": score,
        }

    if not isinstance(source, str) or not source.strip():
        return {
            "status": "abstained",
            "answer": "No verifiable source was found.",
            "source": None,
            "score": score,
        }

    return {
        "status": "answered",
        "answer": evidence.strip(),
        "source": source,
        "score": score,
        "qid": result.get("qid", None),
    }
