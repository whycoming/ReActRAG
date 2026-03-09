"""
ReAct search agent.

Runs a Reason+Act loop using the LLM with tool use. The agent:
1. Receives the question + document index overview as the initial message.
2. Iteratively calls tools (search_index, read_chunks, list_sections, calculate).
3. Calls `finish` when it has gathered sufficient context, or when max steps reached.

Works with all supported providers:
- Anthropic / Qwen: native tool_use
- DeepSeek: JSON-in-text fallback (LLMClient handles transparently)
"""

from __future__ import annotations
import re
from dataclasses import dataclass

import config as cfg
from src.llm_client import LLMClient
from src.schema import IndexSchema, SearchSchema
from src.tools import ToolRegistry, _score_section


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    retrieved_context: str          # final context passed to answerer
    confidence: str                 # "high" | "medium" | "low"
    chunks_read: list[str]          # chunk IDs that were actually read
    react_steps: int                # number of tool-call steps taken
    reasoning_trace: list[dict]     # list of {step, tool, input, output}
    finished_normally: bool         # True if agent called finish, False if max steps hit


# ---------------------------------------------------------------------------
# ReActSearcher
# ---------------------------------------------------------------------------

class ReActSearcher:
    """
    Schema-driven ReAct agent for document search.

    The agent is configured entirely by SearchSchema (system prompt, tools,
    max_steps, top_k). IndexSchema informs search scoring. No hardcoded prompts.
    """

    def __init__(
        self,
        index_schema: IndexSchema,
        search_schema: SearchSchema,
        client: LLMClient | None = None,
    ):
        self.index_schema = index_schema
        self.search_schema = search_schema
        self.client = client or LLMClient()
        self.model = cfg.get_model("searcher")

    def search(
        self,
        question: str,
        doc_index: dict,
        extra_context: dict | None = None,
    ) -> SearchResult:
        """
        Run the ReAct loop for a question against a document index.

        Args:
            question:      The question to answer.
            doc_index:     Pre-built index dict (from DocumentIndexer).
            extra_context: Optional additional metadata (e.g. question_type).

        Returns:
            SearchResult with retrieved context and trace.
        """
        registry = ToolRegistry(doc_index, self.index_schema, self.search_schema)
        tools = registry.get_tool_definitions()

        # Build tool descriptions string for system prompt rendering
        tool_descriptions = self._build_tool_descriptions(tools)
        system_prompt = self.search_schema.render_system_prompt(tool_descriptions)

        initial_message = self._build_initial_message(question, doc_index, extra_context)
        messages = [{"role": "user", "content": initial_message}]

        reasoning_trace: list[dict] = []
        chunks_read: list[str] = []
        finished_context = ""
        finished_confidence = "low"
        finished_chunks: list[str] = []
        finished_normally = False
        response = None

        for step in range(self.search_schema.max_steps):
            response = self.client.chat(
                messages=messages,
                system=system_prompt,
                model=self.model,
                tools=tools,
                temperature=cfg.SEARCHER_TEMPERATURE,
                max_tokens=4096,
            )

            # Append assistant response to message history
            if self.client.provider == "anthropic":
                messages.append({"role": "assistant", "content": response.raw.content})
            else:
                asst_content: list = []
                if response.content:
                    asst_content.append({"type": "text", "text": response.content})
                for tc in response.tool_calls:
                    asst_content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                messages.append({"role": "assistant", "content": asst_content})

            # end_turn with no tool calls — agent finished without calling finish tool
            if response.stop_reason == "end_turn" and not response.tool_calls:
                if response.content:
                    finished_context = response.content
                    finished_confidence = "medium"
                break

            if not response.tool_calls:
                break

            # Process tool calls
            tool_results = []
            for tc in response.tool_calls:
                if tc.name == "finish":
                    finished_context = tc.input.get("context", "")
                    finished_confidence = tc.input.get("confidence", "low")
                    finished_chunks = tc.input.get("chunks_read", [])
                    finished_normally = True
                    reasoning_trace.append({
                        "step": step + 1,
                        "tool": "finish",
                        "input": tc.input,
                        "output": "Search complete.",
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": "Search complete. Preparing final answer.",
                    })
                else:
                    observation = registry.execute(tc.name, tc.input)
                    if tc.name == "read_chunks":
                        chunks_read.extend(tc.input.get("chunk_ids", []))
                    reasoning_trace.append({
                        "step": step + 1,
                        "tool": tc.name,
                        "input": tc.input,
                        "output": observation[:500],
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": observation,
                    })

            messages.append({"role": "user", "content": tool_results})

            if finished_normally:
                break

        if not finished_context:
            last_content = (response.content if response is not None else "") or ""
            finished_context = last_content or "No relevant information found."
            finished_confidence = "low"

        all_chunks = list(dict.fromkeys(chunks_read + finished_chunks))

        return SearchResult(
            retrieved_context=finished_context,
            confidence=finished_confidence,
            chunks_read=all_chunks,
            react_steps=len(reasoning_trace),
            reasoning_trace=reasoning_trace,
            finished_normally=finished_normally,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tool_descriptions(tools: list[dict]) -> str:
        """Build a human-readable tool list for injection into the system prompt."""
        lines = []
        for t in tools:
            lines.append(f"- {t['name']}: {t['description']}")
        return "\n".join(lines)

    def _build_initial_message(
        self,
        question: str,
        doc_index: dict,
        extra_context: dict | None,
    ) -> str:
        doc_name = doc_index.get("doc_name", "unknown")
        doc_summary = doc_index.get("doc_summary", "")
        total_chunks = doc_index.get("total_chunks", 0)

        sections = doc_index.get("sections", [])

        # Sort sections by relevance to the question so the agent sees the most
        # promising chunks first, reducing the number of search_index calls needed.
        query_terms = re.findall(r"\w+", question)
        if query_terms and sections:
            sections = sorted(
                sections,
                key=lambda s: _score_section(s, query_terms, self.index_schema, self.search_schema),
                reverse=True,
            )
        preview_sections = sections[:20]
        toc_lines = [
            f"{'chunk_id':<12} {'pages':<10} {'has_nums':<10} section_title",
            "-" * 70,
        ]
        for sec in preview_sections:
            cid = sec.get("chunk_id", "?")
            pages = sec.get("page_range", [])
            page_str = f"{pages[0]}-{pages[1]}" if len(pages) == 2 else "?"
            has_nums = "yes" if sec.get("has_numeric_data") else "no"
            title = sec.get("section_title", "untitled")
            toc_lines.append(f"{cid:<12} {page_str:<10} {has_nums:<10} {title}")

        if len(sections) > 20:
            toc_lines.append(f"... and {len(sections) - 20} more sections (use list_sections to see all)")

        toc_str = "\n".join(toc_lines)

        extra_str = ""
        if extra_context:
            extra_parts = [f"{k}: {v}" for k, v in extra_context.items()]
            extra_str = "\nAdditional context:\n" + "\n".join(extra_parts)

        return f"""QUESTION: {question}

DOCUMENT: {doc_name}
Summary: {doc_summary}
Total chunks: {total_chunks}
{extra_str}

INITIAL INDEX OVERVIEW (first {len(preview_sections)} sections):
{toc_str}

Begin your search. Use the available tools to find the information needed to answer the question."""
