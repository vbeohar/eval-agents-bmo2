"""Trace-level graders for narrative workflow validation."""

from typing import Any

from langfuse.experiment import Evaluation


def trace_narrative_workflow_grader(
    *,
    trace: Any,
    item_result: Any,
    **kwargs: Any,
) -> list[Evaluation]:
    """Evaluate whether the trace appears to include evidence collection before narrative generation."""
    del kwargs

    observations = getattr(trace, "observations", []) or []

    sql_like_count = 0
    for obs in observations:
        raw_input = getattr(obs, "input", None)

        if isinstance(raw_input, dict):
            query = raw_input.get("query")
            if isinstance(query, str) and "select" in query.lower():
                sql_like_count += 1
        elif isinstance(raw_input, str):
            if "select" in raw_input.lower():
                sql_like_count += 1

    output = getattr(item_result, "output", None) or {}
    narrative_text = output.get("narrative_text", "") if isinstance(output, dict) else ""

    has_narrative = bool(narrative_text.strip())
    has_evidence_collection = sql_like_count > 0

    return [
        Evaluation(
            name="trace_narrative_has_evidence_collection",
            value=1.0 if (has_evidence_collection or not has_narrative) else 0.0,
            metadata={"sql_query_count": sql_like_count},
        ),
        Evaluation(
            name="trace_narrative_sql_query_count",
            value=float(sql_like_count),
        ),
        Evaluation(
            name="trace_narrative_present",
            value=1.0 if has_narrative else 0.0,
        ),
    ]


__all__ = ["trace_narrative_workflow_grader"]