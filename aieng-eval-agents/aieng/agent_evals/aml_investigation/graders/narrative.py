"""Narrative-focused graders for AML investigation outputs."""

from typing import Any

from langfuse.experiment import Evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ._common import get_field, normalize_pattern, normalize_transaction_ids


def narrative_consistency_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Check whether the generated narrative fields are internally consistent."""
    del input, metadata, kwargs

    narrative_text = get_field(output, "narrative_text") or ""
    filing_recommendation = get_field(output, "filing_recommendation")

    predicted_is_laundering = get_field(output, "is_laundering")
    predicted_pattern = normalize_pattern(get_field(output, "pattern_type"))
    predicted_ids = normalize_transaction_ids(get_field(output, "flagged_transaction_ids"))

    expected_is_laundering = get_field(expected_output, "is_laundering")
    expected_pattern = normalize_pattern(get_field(expected_output, "pattern_type"))
    expected_ids = normalize_transaction_ids(get_field(expected_output, "attempt_transaction_ids"))

    narrative_text_lower = narrative_text.lower()

    mentions_pattern = False
    if predicted_pattern:
        mentions_pattern = predicted_pattern.lower() in narrative_text_lower

    mentions_any_flagged_id = any(str(txn_id) in narrative_text for txn_id in predicted_ids)

    recommendation_consistent = (
        filing_recommendation is None
        or bool(filing_recommendation) == bool(predicted_is_laundering)
    )

    id_overlap = len(predicted_ids & expected_ids) / max(1, len(expected_ids))

    return [
        Evaluation(
            name="narrative_present",
            value=1.0 if narrative_text.strip() else 0.0,
        ),
        Evaluation(
            name="narrative_recommendation_consistent",
            value=1.0 if recommendation_consistent else 0.0,
            metadata={
                "filing_recommendation": filing_recommendation,
                "predicted_is_laundering": predicted_is_laundering,
            },
        ),
        Evaluation(
            name="narrative_mentions_pattern",
            value=1.0 if mentions_pattern else 0.0,
            metadata={"pattern_type": predicted_pattern},
        ),
        Evaluation(
            name="narrative_mentions_flagged_ids",
            value=1.0 if mentions_any_flagged_id or not predicted_ids else 0.0,
            metadata={"flagged_ids_count": len(predicted_ids)},
        ),
        Evaluation(
            name="narrative_case_decision_correct",
            value=1.0 if predicted_is_laundering == expected_is_laundering else 0.0,
        ),
        Evaluation(
            name="narrative_pattern_correct",
            value=1.0 if predicted_pattern == expected_pattern else 0.0,
        ),
        Evaluation(
            name="narrative_id_overlap",
            value=float(id_overlap),
            metadata={
                "intersection_count": len(predicted_ids & expected_ids),
                "expected_count": len(expected_ids),
                "predicted_count": len(predicted_ids),
            },
        ),
    ]


def narrative_similarity_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Compare generated narrative text against a reference narrative in metadata."""
    del input, expected_output, kwargs

    candidate = get_field(output, "narrative_text") or ""
    reference = (metadata or {}).get("reference_narrative", "")

    if not reference.strip():
        return [Evaluation(name="narrative_similarity_missing_reference", value=0.0)]

    if not candidate.strip():
        return [
            Evaluation(
                name="narrative_tfidf_cosine_similarity",
                value=0.0,
                metadata={"has_reference": True},
            )
        ]

    vec = TfidfVectorizer().fit_transform([reference, candidate])
    score = float(cosine_similarity(vec[0:1], vec[1:2])[0, 0])

    return [
        Evaluation(
            name="narrative_tfidf_cosine_similarity",
            value=score,
            metadata={"has_reference": True},
        )
    ]


__all__ = [
    "narrative_consistency_grader",
    "narrative_similarity_grader",
]