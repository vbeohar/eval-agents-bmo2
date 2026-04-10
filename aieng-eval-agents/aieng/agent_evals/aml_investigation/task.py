"""Task function for AML investigation experiment execution.

This module provides a Langfuse-compatible task callable that executes the AML
investigation agent on one dataset item and returns a structured analyst output.

The task is designed for use with the evaluation harness and Langfuse
``run_experiment`` APIs. It handles:

- Input normalization for both dict and dataset item objects.
- Running the ADK agent through a shared ``Runner``.
- Extracting and validating final model output.
- Returning consistent ``dict`` results for evaluator consumption.
- Logging execution metadata and output to Langfuse spans.

Examples
--------
>>> import asyncio
>>> from aieng.agent_evals.aml_investigation.task import AmlInvestigationTask
>>> task = AmlInvestigationTask()
>>> sample_item = {
...     "input": {
...         "case_id": "case-001",
...         "seed_transaction_id": "txn-001",
...         "seed_timestamp": "2022-09-01T12:00:00",
...         "window_start": "2022-09-01T00:00:00",
...         "trigger_label": "RANDOM_REVIEW",
...     }
... }
>>> _ = asyncio.run(task(item=sample_item))

>>> from aieng.agent_evals.evaluation.experiment import run_experiment
>>> result = run_experiment(
...     dataset_name="aml_eval_dataset",
...     name="AML Investigation Evaluation",
...     task=AmlInvestigationTask(),
...     evaluators=[...],
... )
"""

from __future__ import annotations

import getpass
import json
import logging
import uuid
from typing import Any

from aieng.agent_evals.aml_investigation.agent import create_aml_investigation_agent
from aieng.agent_evals.aml_investigation.data import AnalystOutput
from aieng.agent_evals.db_manager import DbManager
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langfuse import Langfuse
from langfuse.experiment import ExperimentItem

logger = logging.getLogger(__name__)


class AmlInvestigationTask:
    """Langfuse-compatible task wrapper for AML case investigations.

    This class implements the callable protocol expected by Langfuse
    experiments: ``__call__(*, item, **kwargs)``.

    A single task instance owns:

    - One AML investigation agent.
    - One ADK runner used to execute agent calls.
    - One Langfuse client for span-level tracing.

    Parameters
    ----------
    agent : LlmAgent | None, optional
        Pre-configured AML investigation agent to use. If ``None``, the default
        factory ``create_aml_investigation_agent()`` is used.

    Examples
    --------
    >>> task = AmlInvestigationTask()
    >>> isinstance(task, AmlInvestigationTask)
    True

    >>> custom_agent = create_aml_investigation_agent(name="aml_custom")
    >>> task = AmlInvestigationTask(agent=custom_agent)
    """

    def __init__(self, *, agent: LlmAgent | None = None) -> None:
        self._agent = agent or create_aml_investigation_agent()
        self._runner = Runner(
            app_name="aml_investigation",
            agent=self._agent,
            session_service=InMemorySessionService(),
            auto_create_session=True,
        )
        self._langfuse = Langfuse()

    @staticmethod
    def _extract_item_input(item: ExperimentItem) -> dict[str, Any]:
        """Extract the input payload from either a dict-like or object-like item."""
        if isinstance(item, dict):
            return item.get("input", {})
        return getattr(item, "input", {}) or {}

    @staticmethod
    def _extract_item_metadata(item: ExperimentItem) -> dict[str, Any]:
        """Extract metadata from either a dict-like or object-like item."""
        if isinstance(item, dict):
            return item.get("metadata", {}) or {}
        return getattr(item, "metadata", {}) or {}

    @staticmethod
    def _derive_case_id(item_input: dict[str, Any], metadata: dict[str, Any]) -> str:
        """Best-effort case_id derivation for logs and tracing."""
        return str(
            item_input.get("case_id")
            or metadata.get("case_id")
            or metadata.get("id")
            or "unknown"
        )

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        """Try to parse a JSON object from model output.

        The method first attempts direct JSON decoding. If that fails, it tries
        to extract the substring between the first '{' and the last '}'.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(text[start : end + 1])

    async def __call__(self, *, item: ExperimentItem, **kwargs: Any) -> dict[str, Any] | None:
        """Run one AML investigation case and return structured output.

        Parameters
        ----------
        item : ExperimentItem
            One Langfuse experiment item. This can be either:

            - A dict-like local item with an ``"input"`` key.
            - A Langfuse dataset item object with an ``input`` attribute.

            The input payload is serialized to JSON and passed as the user
            message to the agent.

        **kwargs : Any
            Additional keyword arguments forwarded by Langfuse. They are
            accepted for protocol compatibility and ignored by this task.

        Returns
        -------
        dict[str, Any] | None
            Parsed analyst output as a dictionary if a valid final response was
            produced, otherwise ``None``.

        Notes
        -----
        The method first attempts strict schema parsing with
        ``AnalystOutput.model_validate_json``. If that fails, it falls back to a
        direct JSON parse and validates the resulting object.
        """
        item_input = self._extract_item_input(item)
        metadata = self._extract_item_metadata(item)
        case_id = self._derive_case_id(item_input, metadata)

        serialized_input = json.dumps(item_input, ensure_ascii=False, indent=2)
        message = types.Content(parts=[types.Part(text=serialized_input)], role="user")

        session_id = str(uuid.uuid4())
        user_id = getpass.getuser()

        final_text: str | None = None

        with self._langfuse.start_as_current_span(
            name="aml-investigation-case",
            input=item_input,
            metadata={
                "case_id": case_id,
                "dataset_item_metadata": metadata,
                "agent_name": getattr(self._agent, "name", None),
                "agent_model": getattr(self._agent, "model", None),
                "session_id": session_id,
                "user_id": user_id,
            },
        ) as span:
            try:
                async for event in self._runner.run_async(
                    session_id=session_id,
                    user_id=user_id,
                    new_message=message,
                ):
                    if event.is_final_response() and event.content and event.content.parts:
                        final_text = "".join(
                            part.text or "" for part in event.content.parts if part.text
                        )

                if not final_text:
                    logger.warning("No analyst output produced for case_id=%s", case_id)
                    span.update(output={"status": "no_output", "case_id": case_id})
                    return None

                try:
                    parsed = AnalystOutput.model_validate_json(final_text.strip()).model_dump()
                except Exception:
                    parsed_json = self._extract_json_object(final_text)
                    parsed = AnalystOutput.model_validate(parsed_json).model_dump()

                span.update(output=parsed)
                return parsed

            except Exception as exc:
                logger.exception("AML investigation task failed for case_id=%s", case_id)
                span.update(
                    output={
                        "status": "error",
                        "case_id": case_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "raw_output": final_text,
                    }
                )
                raise

    async def close(self) -> None:
        """Close runner and database connections used by this task instance."""
        await self._runner.close()
        DbManager().aml_db().close()