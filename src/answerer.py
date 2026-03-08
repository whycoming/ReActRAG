"""
Final answer synthesis.

Takes the question, retrieved context from the ReAct search, and any
extra metadata to produce a concise, factual answer via a single LLM call.
The system prompt is driven entirely by AnswerSchema — no hardcoded prompts.
"""

from __future__ import annotations

import config as cfg
from src.llm_client import LLMClient
from src.schema import AnswerSchema
from src.searcher import SearchResult


class Answerer:
    """
    Generates the final answer from retrieved context.
    Fully schema-driven: system prompt and per-context hints come from AnswerSchema.
    """

    def __init__(
        self,
        answer_schema: AnswerSchema,
        client: LLMClient | None = None,
    ):
        self.answer_schema = answer_schema
        self.client = client or LLMClient()
        self.model = cfg.get_model("answerer")

    def answer(
        self,
        question: str,
        search_result: SearchResult,
        extra_context: dict | None = None,
    ) -> str:
        """
        Produce the final answer.

        Args:
            question:      The original question.
            search_result: SearchResult from ReActSearcher.search().
            extra_context: Optional metadata from the query schema (e.g. question_type).
                           Used to select per_context_hints from the AnswerSchema.

        Returns:
            Final answer string.
        """
        # System prompt rendered with per-context hints for this specific query
        system_prompt = self.answer_schema.render_system_prompt(extra_context or {})

        user_message = self._build_message(question, search_result, extra_context)

        response = self.client.chat(
            messages=[{"role": "user", "content": user_message}],
            system=system_prompt,
            model=self.model,
            temperature=cfg.ANSWERER_TEMPERATURE,
            max_tokens=1024,
        )

        return response.content.strip()

    def _build_message(
        self,
        question: str,
        search_result: SearchResult,
        extra_context: dict | None,
    ) -> str:
        extra_str = ""
        if extra_context:
            parts = [f"  {k}: {v}" for k, v in extra_context.items()]
            extra_str = "\nQuestion metadata:\n" + "\n".join(parts) + "\n"

        confidence_note = ""
        if search_result.confidence == "low":
            confidence_note = "\nNote: The search agent reported low confidence — the context may be incomplete.\n"
        elif search_result.confidence == "medium":
            confidence_note = "\nNote: The search agent reported partial information found.\n"

        trace_summary = ""
        if search_result.reasoning_trace:
            steps = [
                f"  Step {t['step']}: {t['tool']}({self._fmt_input(t['input'])})"
                for t in search_result.reasoning_trace[:8]
            ]
            trace_summary = "\nSearch steps taken:\n" + "\n".join(steps) + "\n"

        return f"""QUESTION: {question}
{extra_str}{confidence_note}
RETRIEVED CONTEXT FROM DOCUMENT:
{search_result.retrieved_context}
{trace_summary}
Provide the final answer now."""

    @staticmethod
    def _fmt_input(inp: dict) -> str:
        if "query" in inp:
            return f'query="{inp["query"][:50]}"'
        if "chunk_ids" in inp:
            return f"chunk_ids={inp['chunk_ids']}"
        if "expression" in inp:
            return f'expr="{inp["expression"]}"'
        return str(inp)[:60]
