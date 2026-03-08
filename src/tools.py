"""
Tools available to the ReAct search agent.

- search_index:   Keyword-scored search over index sections (schema-aware weights)
- read_chunks:    Read actual chunk text content
- list_sections:  Full table of contents of the index
- calculate:      Safe math expression evaluator
- finish:         Signal end of search with gathered context

The ToolRegistry builds Anthropic-format tool definitions that work with
all supported providers (via LLMClient's format conversion).
"""

from __future__ import annotations
import ast
import math
import operator
import re
from typing import Any

import config as cfg
from src.schema import IndexSchema, SearchSchema


# ---------------------------------------------------------------------------
# Schema-aware keyword scoring
# ---------------------------------------------------------------------------

def _score_section(section: dict, query_terms: list[str],
                   index_schema: IndexSchema,
                   search_schema: SearchSchema | None = None) -> float:
    """
    Score a section against query terms using:
    1. Field weights from IndexSchema (base scoring)
    2. Extra boost rules from SearchSchema (domain-specific overrides)
    Higher = more relevant.
    """
    score = 0.0
    query_lower = {t.lower() for t in query_terms}

    # Base scoring: IndexSchema field weights
    for field_def in index_schema.scored_fields():
        value = section.get(field_def.name)
        if value is None:
            continue

        if isinstance(value, list):
            field_tokens = set()
            for item in value:
                field_tokens.update(re.findall(r"\w+", str(item).lower()))
        else:
            field_tokens = set(re.findall(r"\w+", str(value).lower()))

        hits = len(query_lower & field_tokens)
        score += hits * field_def.search_weight

    # Extra scoring: SearchSchema.extra_score_rules
    if search_schema:
        for rule in search_schema.extra_score_rules:
            if rule.match_keyword.lower() in query_lower:
                field_val = section.get(rule.boost_field)
                if field_val:  # truthy: non-empty string/list, True boolean
                    score += rule.boost

    return score


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_index(
    doc_index: dict,
    query: str,
    index_schema: IndexSchema,
    search_schema: SearchSchema | None = None,
    top_k: int = cfg.TOP_K_CHUNKS,
) -> str:
    """
    Search the document index for sections relevant to the query.
    Uses IndexSchema field weights + SearchSchema extra_score_rules.
    Returns a formatted table of the top_k results.
    """
    query_terms = re.findall(r"\w+", query)
    sections = doc_index.get("sections", [])

    scored = []
    for sec in sections:
        score = _score_section(sec, query_terms, index_schema, search_schema)
        scored.append((score, sec))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top or top[0][0] == 0:
        return "No relevant sections found for that query. Try different search terms."

    lines = [f"Top {len(top)} sections for query: '{query}'\n"]
    lines.append(f"{'chunk_id':<12} {'pages':<10} {'score':<7} title + hints")
    lines.append("-" * 80)
    for score, sec in top:
        chunk_id = sec.get("chunk_id", "?")
        pages = sec.get("page_range", [])
        page_str = f"{pages[0]}-{pages[1]}" if len(pages) == 2 else "?"
        title = sec.get("section_title", sec.get("title", "untitled"))
        hints = sec.get("relevance_hints", "")
        lines.append(f"{chunk_id:<12} {page_str:<10} {score:<7.1f} {title}")
        if hints:
            lines.append(f"{'':12} {'':10} {'':7} → {hints}")

    return "\n".join(lines)


def read_chunks(
    doc_index: dict,
    chunk_ids: list[str],
    max_chunks: int = 5,
) -> str:
    """
    Read the actual text content of specified chunks.
    Limited to max_chunks per call to prevent context blowout.
    """
    chunk_ids = chunk_ids[:max_chunks]
    chunk_texts: dict = doc_index.get("_chunk_texts", {})
    sections_by_id = {s["chunk_id"]: s for s in doc_index.get("sections", [])}

    if not chunk_texts:
        return "ERROR: No chunk text stored in index. Re-index with current version."

    parts = []
    for cid in chunk_ids:
        text = chunk_texts.get(cid)
        if text is None:
            parts.append(f"[{cid}]: NOT FOUND")
            continue
        sec = sections_by_id.get(cid, {})
        pages = sec.get("page_range", [])
        page_str = f"pages {pages[0]}-{pages[1]}" if len(pages) == 2 else ""
        header = f"=== {cid} ({page_str}) ==="
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts) if parts else "No chunks found for the given IDs."


def list_sections(doc_index: dict) -> str:
    """
    Return a full table of contents for the document index.
    """
    sections = doc_index.get("sections", [])
    if not sections:
        return "Index has no sections."

    doc_name = doc_index.get("doc_name", "?")
    summary = doc_index.get("doc_summary", "")
    lines = [
        f"Document: {doc_name}",
        f"Summary: {summary}",
        f"Total chunks: {len(sections)}",
        "",
        f"{'chunk_id':<12} {'pages':<10} {'has_nums':<10} section_title",
        "-" * 70,
    ]
    for sec in sections:
        chunk_id = sec.get("chunk_id", "?")
        pages = sec.get("page_range", [])
        page_str = f"{pages[0]}-{pages[1]}" if len(pages) == 2 else "?"
        has_nums = "yes" if sec.get("has_numeric_data") else "no"
        title = sec.get("section_title", sec.get("title", "untitled"))
        lines.append(f"{chunk_id:<12} {page_str:<10} {has_nums:<10} {title}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Safe calculator
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "percent": lambda a, b: (a / b * 100) if b != 0 else float("nan"),
    "growth_rate": lambda old, new: ((new - old) / abs(old) * 100) if old != 0 else float("nan"),
    "average": lambda *vals: sum(vals) / len(vals) if vals else float("nan"),
    "sum": sum,
    "min": min,
    "max": max,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _SAFE_OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        return _SAFE_OPS[op_type](_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        fname = node.func.id
        if fname not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unknown function: {fname}. "
                             f"Available: {list(_SAFE_FUNCTIONS)}")
        args = [_safe_eval(a) for a in node.args]
        return _SAFE_FUNCTIONS[fname](*args)
    elif isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    else:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supported: +, -, *, /, **, (, ), numbers
    Functions: percent(a,b), growth_rate(old,new), average(*vals),
               sum(*vals), min(*vals), max(*vals), abs, round, sqrt, log, log10
    """
    expression = expression.strip()
    # Remove trailing commas or semicolons that LLMs sometimes add
    expression = expression.rstrip(",;")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        # Format result
        if isinstance(result, float):
            if math.isnan(result):
                return "ERROR: Division by zero or undefined result"
            if result == int(result) and abs(result) < 1e12:
                return str(int(result))
            return f"{result:.4f}"
        return str(result)
    except ZeroDivisionError:
        return "ERROR: Division by zero"
    except (ValueError, TypeError, SyntaxError) as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# ToolRegistry — builds Anthropic-format tool definitions
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Builds the tool definitions list passed to the LLM.
    Holds references to doc_index, IndexSchema, and SearchSchema so tool functions
    can be called with the correct context.
    Only tools listed in search_schema.enabled_tools are included.
    """

    def __init__(self, doc_index: dict, index_schema: IndexSchema,
                 search_schema: SearchSchema | None = None):
        self.doc_index = doc_index
        self.index_schema = index_schema
        self.search_schema = search_schema
        self._enabled = set(
            search_schema.enabled_tools if search_schema else [
                "search_index", "read_chunks", "list_sections", "calculate", "finish"
            ]
        )

    def get_tool_definitions(self) -> list[dict]:
        """Return Anthropic-format tool definitions for enabled tools only."""
        all_tools = [
            {
                "name": "search_index",
                "description": (
                    "Search the document index for sections relevant to a query. "
                    "Returns ranked section IDs, page ranges, and relevance hints. "
                    "Use this first to identify which chunks to read."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free-text search query describing what you are looking for.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "read_chunks",
                "description": (
                    "Read the actual text content of specific document chunks. "
                    "Use the chunk_ids returned by search_index or list_sections. "
                    "Maximum 5 chunks per call."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of chunk IDs to read (e.g. ['chunk_0019', 'chunk_0020']).",
                        },
                    },
                    "required": ["chunk_ids"],
                },
            },
            {
                "name": "list_sections",
                "description": (
                    "Get the full table of contents for the document. "
                    "Shows all section titles, page ranges, and whether they contain numeric data. "
                    "Use for an overview before searching, or when unsure where to look."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "calculate",
                "description": (
                    "Evaluate a mathematical expression. "
                    "Supported: +, -, *, /, **, parentheses, numbers. "
                    "Built-in functions: percent(a, b) → a/b*100, "
                    "growth_rate(old, new) → (new-old)/old*100, "
                    "average(v1, v2, ...), sum(...), min(...), max(...), "
                    "abs, round, sqrt, log, log10."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate, e.g. 'percent(167, 3991)' or '(100 + 200) / 3'.",
                        },
                    },
                    "required": ["expression"],
                },
            },
            {
                "name": "finish",
                "description": (
                    "Signal that you have gathered sufficient information. "
                    "Call this when you are ready to answer, or when you cannot find "
                    "the information after thorough searching."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "All relevant text, numbers, and findings extracted from the document.",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Your confidence that the context is sufficient to answer the question.",
                        },
                        "chunks_read": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of chunk IDs you read during the search.",
                        },
                    },
                    "required": ["context", "confidence"],
                },
            },
        ]
        # Filter to only enabled tools (preserves order of all_tools)
        return [t for t in all_tools if t["name"] in self._enabled]

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch a tool call and return the observation string."""
        if tool_name == "search_index":
            return search_index(
                self.doc_index,
                query=tool_input.get("query", ""),
                index_schema=self.index_schema,
                search_schema=self.search_schema,
                top_k=tool_input.get("top_k",
                    self.search_schema.top_k if self.search_schema else cfg.TOP_K_CHUNKS),
            )
        elif tool_name == "read_chunks":
            return read_chunks(
                self.doc_index,
                chunk_ids=tool_input.get("chunk_ids", []),
            )
        elif tool_name == "list_sections":
            return list_sections(self.doc_index)
        elif tool_name == "calculate":
            return calculate(tool_input.get("expression", ""))
        elif tool_name == "finish":
            # finish is handled by the searcher loop; return acknowledgement
            return "Search complete."
        else:
            return f"ERROR: Unknown tool '{tool_name}'"
