"""
Schema definitions for GraphIndex.
All schemas share a common BaseSchema interface and are loaded from YAML.

Schema hierarchy:
  BaseSchema (ABC)
    ├── IndexSchema   — defines what fields LLM extracts per chunk + doc-level fields
    ├── QuerySchema   — maps input dataset fields to pipeline field names
    ├── SearchSchema  — configures ReAct agent: prompt template, tools, scoring rules
    ├── AnswerSchema  — configures answer generation: prompt + per-context hints
    └── JudgeSchema   — configures evaluation: judge prompt template + tolerance rules

PipelineSchema — composes all five schemas from a single YAML file
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


# ---------------------------------------------------------------------------
# BaseSchema — common interface for all schemas
# ---------------------------------------------------------------------------

class BaseSchema(ABC):
    """Abstract base class for all GraphIndex schemas."""

    @classmethod
    @abstractmethod
    def from_yaml(cls, path: str | Path) -> "BaseSchema":
        """Load the schema from a YAML file."""
        ...

    def to_prompt_section(self) -> str:
        """Return a text block that can be injected into an LLM prompt."""
        return ""


# ---------------------------------------------------------------------------
# IndexSchema — chunk-level field extraction + doc-level metadata
# ---------------------------------------------------------------------------

@dataclass
class IndexFieldDef:
    name: str
    description: str
    type: str           # "string" | "list[string]" | "boolean"
    search_weight: int  # 0 = extracted but not used for keyword scoring

    def is_list(self) -> bool:
        return self.type.startswith("list")

    def is_bool(self) -> bool:
        return self.type == "boolean"

    def json_type_hint(self) -> str:
        if self.is_bool():
            return "true|false"
        if self.is_list():
            return '["<item1>", "<item2>", ...]'
        return '"<value>"'


@dataclass
class IndexSchema(BaseSchema):
    fields: list[IndexFieldDef]
    doc_fields: list[IndexFieldDef] = field(default_factory=list)

    def scored_fields(self) -> list[IndexFieldDef]:
        """Fields used for keyword scoring in search_index (weight > 0)."""
        return [f for f in self.fields if f.search_weight > 0]

    def to_json_schema_str(self) -> str:
        """Generate the JSON schema block injected into the chunk indexer prompt."""
        lines = ["{"]
        for i, f in enumerate(self.fields):
            comma = "," if i < len(self.fields) - 1 else ""
            lines.append(f'  "{f.name}": {f.json_type_hint()}{comma}')
        lines.append("}")
        return "\n".join(lines)

    def to_field_descriptions(self) -> str:
        """Human-readable chunk field descriptions for the indexer prompt."""
        return "\n".join(
            f'- "{f.name}" ({f.type}): {f.description}' for f in self.fields
        )

    def to_doc_json_schema_str(self) -> str:
        """JSON schema block for doc-level metadata extraction prompt."""
        if not self.doc_fields:
            return '{"doc_summary": "<2-sentence summary of the document>"}'
        lines = ["{"]
        for i, f in enumerate(self.doc_fields):
            comma = "," if i < len(self.doc_fields) - 1 else ""
            lines.append(f'  "{f.name}": {f.json_type_hint()}{comma}')
        lines.append("}")
        return "\n".join(lines)

    def to_doc_field_descriptions(self) -> str:
        """Human-readable doc-level field descriptions."""
        if not self.doc_fields:
            return '- "doc_summary" (string): 2-sentence summary of the document'
        return "\n".join(
            f'- "{f.name}" ({f.type}): {f.description}' for f in self.doc_fields
        )

    def to_prompt_section(self) -> str:
        return self.to_field_descriptions()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IndexSchema":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        def parse_fields(items: list) -> list[IndexFieldDef]:
            result = []
            for item in items:
                result.append(IndexFieldDef(
                    name=item["name"],
                    description=item["description"],
                    type=item.get("type", "string"),
                    search_weight=item.get("search_weight", 1),
                ))
            return result

        fields = parse_fields(data.get("fields", []))
        if not fields:
            raise ValueError(f"Index schema at {path} has no 'fields' defined.")
        doc_fields = parse_fields(data.get("doc_fields", []))
        return cls(fields=fields, doc_fields=doc_fields)


def load_index_schema(path: str | Path) -> IndexSchema:
    return IndexSchema.from_yaml(path)


# ---------------------------------------------------------------------------
# QuerySchema — input dataset field mapping
# ---------------------------------------------------------------------------

@dataclass
class QuerySchema(BaseSchema):
    question_field: str
    doc_field: str
    answer_field: str | None
    id_field: str | None
    extra_context_fields: list[str]

    def extract_question(self, record: dict) -> str:
        val = record.get(self.question_field)
        if val is None:
            raise KeyError(
                f"Question field '{self.question_field}' not found in record: {list(record.keys())}"
            )
        return str(val)

    def extract_doc_name(self, record: dict) -> str:
        val = record.get(self.doc_field)
        if val is None:
            raise KeyError(
                f"Doc field '{self.doc_field}' not found in record: {list(record.keys())}"
            )
        return str(val).removesuffix(".md")

    def extract_answer(self, record: dict) -> str | None:
        if self.answer_field is None:
            return None
        return record.get(self.answer_field)

    def extract_id(self, record: dict) -> str | None:
        if self.id_field is None:
            return None
        return record.get(self.id_field)

    def extract_extra_context(self, record: dict) -> dict[str, Any]:
        return {f: record[f] for f in self.extra_context_fields if f in record}

    def to_prompt_section(self) -> str:
        return (f"question_field={self.question_field}, "
                f"doc_field={self.doc_field}, "
                f"extra_context={self.extra_context_fields}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QuerySchema":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(
            question_field=data["question_field"],
            doc_field=data["doc_field"],
            answer_field=data.get("answer_field"),
            id_field=data.get("id_field"),
            extra_context_fields=data.get("extra_context_fields", []),
        )


def load_query_schema(path: str | Path) -> QuerySchema:
    return QuerySchema.from_yaml(path)


# ---------------------------------------------------------------------------
# SearchSchema — ReAct agent configuration
# ---------------------------------------------------------------------------

@dataclass
class ExtraScoreRule:
    match_keyword: str
    boost_field: str
    boost: float


@dataclass
class SearchSchema(BaseSchema):
    enabled_tools: list[str]
    system_prompt_template: str   # may contain {tool_descriptions}
    max_steps: int
    top_k: int
    extra_score_rules: list[ExtraScoreRule]

    def render_system_prompt(self, tool_descriptions: str = "") -> str:
        """Fill {tool_descriptions} placeholder in the system prompt template."""
        try:
            return self.system_prompt_template.format(
                tool_descriptions=tool_descriptions
            )
        except KeyError:
            # Template has other placeholders we don't fill — return as-is
            return self.system_prompt_template.replace(
                "{tool_descriptions}", tool_descriptions
            )

    def to_prompt_section(self) -> str:
        return self.render_system_prompt()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SearchSchema":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        rules = []
        for r in data.get("extra_score_rules", []):
            rules.append(ExtraScoreRule(
                match_keyword=r["match_keyword"],
                boost_field=r["boost_field"],
                boost=float(r.get("boost", 0)),
            ))
        return cls(
            enabled_tools=data.get("enabled_tools", [
                "search_index", "read_chunks", "list_sections", "calculate", "finish"
            ]),
            system_prompt_template=data["system_prompt_template"],
            max_steps=int(data.get("max_steps", 12)),
            top_k=int(data.get("top_k", 5)),
            extra_score_rules=rules,
        )


def load_search_schema(path: str | Path) -> SearchSchema:
    return SearchSchema.from_yaml(path)


# ---------------------------------------------------------------------------
# AnswerSchema — answer generation configuration
# ---------------------------------------------------------------------------

@dataclass
class AnswerSchema(BaseSchema):
    system_prompt: str
    # per_context_hints: {field_name: {field_value: hint_text}}
    per_context_hints: dict[str, dict[str, str]]

    def render_system_prompt(self, extra_context: dict) -> str:
        """Append per-context hints matching the current query's extra_context."""
        prompt = self.system_prompt
        for field_name, hints in self.per_context_hints.items():
            value = extra_context.get(field_name)
            if value is not None and str(value) in hints:
                prompt += f"\n\nAdditional instruction: {hints[str(value)]}"
        return prompt

    def to_prompt_section(self) -> str:
        return self.system_prompt

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnswerSchema":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(
            system_prompt=data["system_prompt"],
            per_context_hints=data.get("per_context_hints", {}),
        )


def load_answer_schema(path: str | Path) -> AnswerSchema:
    return AnswerSchema.from_yaml(path)


# ---------------------------------------------------------------------------
# JudgeSchema — evaluation scoring configuration
# ---------------------------------------------------------------------------

@dataclass
class JudgeSchema(BaseSchema):
    system_prompt: str
    prompt_template: str          # placeholders: {question}, {expected}, {predicted}, {numeric_tolerance_pct}
    numeric_tolerance_pct: float
    correct_prefix: str
    incorrect_prefix: str
    reasoning_prefix: str

    def render_prompt(self, question: str, expected: str, predicted: str) -> str:
        """Render the judge prompt with actual values."""
        return self.prompt_template.format(
            question=question,
            expected=expected,
            predicted=predicted,
            numeric_tolerance_pct=self.numeric_tolerance_pct,
        )

    def parse_response(self, text: str) -> tuple[bool, str]:
        """
        Parse LLM judge response into (is_correct, reasoning).
        Expects format:
          CORRECT or INCORRECT
          Reasoning: <one sentence>
        """
        text = text.strip()
        lines = text.split("\n")
        first_line = lines[0].strip().upper() if lines else ""
        is_correct = (
            first_line.startswith(self.correct_prefix.upper())
            and not first_line.startswith(self.incorrect_prefix.upper())
        )
        reasoning = ""
        for line in lines:
            if line.lower().startswith(self.reasoning_prefix.lower()):
                reasoning = line[len(self.reasoning_prefix):].strip()
                break
        return is_correct, reasoning or text[:200]

    def to_prompt_section(self) -> str:
        return self.system_prompt

    @classmethod
    def from_yaml(cls, path: str | Path) -> "JudgeSchema":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(
            system_prompt=data["system_prompt"],
            prompt_template=data["prompt_template"],
            numeric_tolerance_pct=float(data.get("numeric_tolerance_pct", 2.0)),
            correct_prefix=data.get("correct_prefix", "CORRECT"),
            incorrect_prefix=data.get("incorrect_prefix", "INCORRECT"),
            reasoning_prefix=data.get("reasoning_prefix", "Reasoning:"),
        )


def load_judge_schema(path: str | Path) -> JudgeSchema:
    return JudgeSchema.from_yaml(path)


# ---------------------------------------------------------------------------
# PipelineSchema — unified entry point composing all five schemas
# ---------------------------------------------------------------------------

@dataclass
class PipelineSchema:
    """
    Composes all five schemas from a single pipeline YAML file.
    Child schema paths are resolved relative to the pipeline YAML's directory.

    pipeline.yaml format:
      index_schema:  default_index_schema.yaml
      query_schema:  default_query_schema.yaml
      search_schema: default_search_schema.yaml
      answer_schema: default_answer_schema.yaml
      judge_schema:  default_judge_schema.yaml
    """
    index: IndexSchema
    query: QuerySchema
    search: SearchSchema
    answer: AnswerSchema
    judge: JudgeSchema

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineSchema":
        path = Path(path)
        base_dir = path.parent
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        def resolve(key: str) -> Path:
            val = data.get(key)
            if val is None:
                raise ValueError(f"Pipeline schema missing required key: '{key}'")
            p = Path(val)
            return p if p.is_absolute() else base_dir / p

        return cls(
            index=IndexSchema.from_yaml(resolve("index_schema")),
            query=QuerySchema.from_yaml(resolve("query_schema")),
            search=SearchSchema.from_yaml(resolve("search_schema")),
            answer=AnswerSchema.from_yaml(resolve("answer_schema")),
            judge=JudgeSchema.from_yaml(resolve("judge_schema")),
        )


def load_pipeline_schema(path: str | Path) -> PipelineSchema:
    return PipelineSchema.from_yaml(path)
