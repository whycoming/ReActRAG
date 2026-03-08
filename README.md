# ReActRAG

**Schema-driven RAG powered by a ReAct reasoning agent — no vectors, no embeddings.**

ReActRAG indexes documents with an LLM, searches them through an iterative Reason+Act loop, and generates grounded answers — every stage driven by YAML schemas you define. Swap documents, datasets, or domains without touching Python code.

Inspired by [PageIndex](https://github.com/VectifyAI/PageIndex).

---

## Results

Evaluated on [FinanceBench](https://github.com/patronus-ai/financebench) open-source split (financial document QA benchmark).

**Provider: Qwen** (`qwen-turbo` for indexing/judging, `qwen-plus` for search/answering)

| Metric | Value |
|---|---|
| **Overall accuracy** | **92.7% (51 / 55)** |
| metrics-generated | 100.0% (11 / 11) |
| domain-relevant | 90.0% (18 / 20) |
| novel-generated | 91.7% (22 / 24) |
| Avg ReAct steps | 5.0 |
| Avg latency | 34.3 s / question |
| Errors | 0 |

**Dataset:** 55 questions across 21 documents, extracted from `financebench_open_source.jsonl` by filtering for documents with pre-built indexes.

**Command used:**
```powershell
python main.py eval `
  --eval-file financebench_output/eval/financebench_indexed.jsonl `
  --query-schema schemas/financebench_query_schema.yaml `
  --skip-index
```

---

## Why ReActRAG?

Most RAG systems embed chunks and retrieve by cosine similarity. This breaks down when:

- The question requires **multi-step reasoning** across multiple sections
- The relevant chunk **doesn't share vocabulary** with the query
- You need to **verify a calculation** before trusting a number
- You want to **customize retrieval behavior** without retraining an embedding model

ReActRAG replaces the embedding layer with an LLM that understands what each chunk *means*, and replaces similarity search with a ReAct agent that *reasons iteratively* about where to look next.

---

## How It Works

```
Document (Markdown)
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  Indexer  (cheap LLM)                   │
  │  Chunk → extract structured metadata    │
  │  fields defined by index_schema.yaml    │
  └──────────────┬──────────────────────────┘
                 │  indexes/{doc}.json  (cached)
                 ▼
  ┌─────────────────────────────────────────┐
  │  ReAct Searcher  (capable LLM)          │
  │  Reason+Act loop:                       │
  │    search_index → read_chunks →         │
  │    calculate → finish                   │
  │  Agent behavior from search_schema.yaml │
  └──────────────┬──────────────────────────┘
                 │  retrieved context
                 ▼
  ┌─────────────────────────────────────────┐
  │  Answerer  (capable LLM)                │
  │  Grounded answer from retrieved text    │
  │  Instructions from answer_schema.yaml   │
  └──────────────┬──────────────────────────┘
                 │  predicted answer
                 ▼
  ┌─────────────────────────────────────────┐
  │  LLM Judge  (cheap LLM)                 │
  │  Scores predicted vs. expected          │
  │  Criteria from judge_schema.yaml        │
  └─────────────────────────────────────────┘
```

**Indexes are cached** — LLM indexing cost is paid once per document. All subsequent queries load `indexes/{doc}.json` directly.

---

## Features

- **Vectorless retrieval** — schema-aware keyword scoring + LLM reasoning, zero embeddings
- **ReAct agent loop** — iterative tool calls: `search_index`, `read_chunks`, `calculate`, `finish`
- **Five configurable schemas** — every prompt, field weight, tool list, and eval criterion lives in YAML
- **Multi-provider** — Anthropic (Claude), Qwen (DashScope), DeepSeek; swap with one env var
- **Offline index cache** — index once, query many times; `--force` to rebuild
- **LLM judge evaluation** — uniform scoring for numeric, narrative, and yes/no answers
- **Arbitrary dataset format** — `query_schema.yaml` maps any JSONL field names to the pipeline
- **Full reasoning traces** — every search step saved to results JSONL for offline audit

---

## Document Preparation

ReActRAG consumes documents in Markdown format, with pages delimited by `## Page N` headers. The Markdown files in `financebench_output/markdown/` were converted from PDF using [Docling](https://github.com/docling-project/docling):

```bash
pip install docling
docling --to md --output financebench_output/markdown/ your_document.pdf
```

Docling handles layout-aware PDF parsing (tables, multi-column text, headers) and produces clean, structured Markdown that the indexer can reliably chunk and annotate.

---

## Installation

```bash
git clone <repo>
cd reactrag
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
# Choose provider: anthropic | qwen | deepseek
LLM_PROVIDER=qwen

DASHSCOPE_API_KEY=sk-xxxx   # Qwen (DashScope)
ANTHROPIC_API_KEY=sk-xxxx   # Anthropic Claude
DEEPSEEK_API_KEY=sk-xxxx    # DeepSeek
```

---

## Quick Start

```powershell
# 1. Index a document (cached to indexes/AMD_2022_10K.json)
python main.py index --doc AMD_2022_10K

# 2. Ask a question
python main.py ask --doc AMD_2022_10K --question "What was AMD's net revenue in FY2022?"

# 3. Run evaluation on a JSONL benchmark
python main.py eval --eval-file my_eval.jsonl
```

---

## CLI Reference

All subcommands accept schema override flags. Run `python main.py <cmd> --help` for full options.

### `index` — Build document indexes

```powershell
# Index a single document
python main.py index --doc AMD_2022_10K

# Index all .md files in a directory
python main.py index --docs-from financebench_output/eval/mini_markdown/

# Force rebuild even if cached
python main.py index --doc AMD_2022_10K --force
```

### `ask` — Ask a single question

```powershell
python main.py ask --doc AMD_2022_10K --question "What is the FY2022 D&A margin?"

# Print the full ReAct reasoning trace
python main.py ask --doc AMD_2022_10K --question "..." --verbose
```

### `eval` — Run benchmark evaluation

```powershell
# Built-in split
python main.py eval --split validation

# Custom JSONL file, limit to 10 questions
python main.py eval --eval-file my_eval.jsonl --max 10

# Skip re-indexing (reuse cached indexes)
python main.py eval --eval-file my_eval.jsonl --skip-index

# Load all schemas from a single pipeline file
python main.py eval --split validation --pipeline-schema schemas/default_pipeline.yaml
```

Results are written to `results/eval_{name}_{timestamp}.jsonl` with full reasoning traces.

---

## Schema System

ReActRAG has five YAML schemas. All have working defaults in `schemas/` — override only what you need.

### `index_schema.yaml` — What the indexer extracts per chunk

```yaml
fields:
  - name: section_title
    description: "Descriptive title for this chunk"
    type: string
    search_weight: 3          # higher = more influential in keyword scoring

  - name: has_numeric_data
    description: "Whether this chunk contains numbers or measurements"
    type: boolean
    search_weight: 0          # used as filter flag, not text scoring

# Document-level metadata (extracted once per document)
doc_fields:
  - name: doc_summary
    description: "2-sentence summary of the document"
    type: string
```

Add domain-specific fields (e.g. `legal_citations`, `api_endpoints`) to improve search precision.

### `query_schema.yaml` — Map your dataset's field names

```yaml
question_field: "question"       # field containing the query text
doc_field:       "expected_md"   # field identifying the source document
answer_field:    "answer"        # ground truth (for eval)
id_field:        "record_id"     # unique ID for result tracking
extra_context_fields:
  - "question_type"              # passed to the answerer as extra context
```

Supports any JSONL format — just remap the field names.

### `search_schema.yaml` — ReAct agent behavior

```yaml
enabled_tools: [search_index, read_chunks, list_sections, calculate, finish]
max_steps: 12
top_k: 5

system_prompt_template: |
  You are an expert analyst...
  {tool_descriptions}       # auto-filled with enabled tool descriptions
  ...

extra_score_rules:
  - match_keyword: "margin"
    boost_field: "has_numeric_data"
    boost: 5
```

### `answer_schema.yaml` — How the answer is generated

```yaml
system_prompt: |
  You are an expert analyst. Give a precise, factual answer...

per_context_hints:           # append extra instructions based on query metadata
  question_type:
    metrics-generated: "Show the full calculation chain."
    domain-relevant:   "Start with Yes. or No."
```

### `judge_schema.yaml` — Evaluation scoring criteria

```yaml
numeric_tolerance_pct: 2    # allow ±2% error for numeric answers
prompt_template: |
  Question: {question}
  Expected: {expected}
  Predicted: {predicted}
  Allow ±{numeric_tolerance_pct}% tolerance...
```

### `pipeline.yaml` — Load all five schemas at once

```yaml
index_schema:  default_index_schema.yaml
query_schema:  default_query_schema.yaml
search_schema: default_search_schema.yaml
answer_schema: default_answer_schema.yaml
judge_schema:  default_judge_schema.yaml
```

```powershell
python main.py eval --pipeline-schema schemas/my_domain_pipeline.yaml
```

---

## Supported Providers

| Provider | Env var | Indexer | Searcher | Answerer | Judge |
|---|---|---|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | claude-haiku-4-5 | claude-sonnet-4-6 | claude-sonnet-4-6 | claude-haiku-4-5 |
| Qwen | `DASHSCOPE_API_KEY` | qwen-turbo | qwen-plus | qwen-plus | qwen-turbo |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat | deepseek-chat | deepseek-chat | deepseek-chat |

Switch provider: set `LLM_PROVIDER=qwen` (or `anthropic` / `deepseek`) in `.env` or shell.

Model names are configured per-role in `config.py → MODELS`.

---

## Project Structure

```
reactrag/
├── main.py                        # CLI entry point
├── config.py                      # paths, model names, chunking params
├── schemas/
│   ├── default_index_schema.yaml  # chunk + doc-level field definitions
│   ├── default_query_schema.yaml  # dataset field mapping
│   ├── default_search_schema.yaml # ReAct agent config + scoring rules
│   ├── default_answer_schema.yaml # answerer prompt + per-context hints
│   ├── default_judge_schema.yaml  # judge prompt + tolerance rules
│   └── default_pipeline.yaml     # composes all five schemas
├── src/
│   ├── schema.py                  # BaseSchema + all five schema classes
│   ├── llm_client.py              # unified client (Anthropic / Qwen / DeepSeek)
│   ├── indexer.py                 # token chunker + LLM metadata extractor
│   ├── searcher.py                # ReAct agent loop
│   ├── tools.py                   # search_index, read_chunks, calculate, finish
│   ├── answerer.py                # final answer generation
│   └── evaluator.py              # end-to-end eval + LLM judge
├── indexes/                       # cached document indexes (JSON)
└── results/                       # eval output JSONL with traces
```

---

## Domain Customization

**To adapt to a new domain** (legal, medical, technical, etc.):

1. Add domain-specific fields to `index_schema.yaml`:
   ```yaml
   fields:
     - name: legal_citations
       description: "Statute references or case numbers in this chunk"
       type: "list[string]"
       search_weight: 3
   ```

2. Tune search scoring in `search_schema.yaml`:
   ```yaml
   extra_score_rules:
     - match_keyword: "statute"
       boost_field: "legal_citations"
       boost: 10
   ```

3. Adjust answerer instructions in `answer_schema.yaml`:
   ```yaml
   per_context_hints:
     doc_type:
       contract: "Cite the specific clause number when referencing obligations."
   ```

4. Tighten or loosen the judge in `judge_schema.yaml`:
   ```yaml
   numeric_tolerance_pct: 0   # strict exact match
   ```

No Python code changes required.

---

## Evaluation Output

Each result JSONL record contains:

```json
{
  "record_id": "...",
  "question": "...",
  "doc_name": "...",
  "predicted": "...",
  "expected": "...",
  "is_correct": true,
  "judge_reasoning": "...",
  "chunks_read": ["chunk_0027", "chunk_0031"],
  "react_steps": 5,
  "confidence": "high",
  "latency_seconds": 39.78,
  "reasoning_trace": [
    {"step": 1, "tool": "search_index", "input": {"query": "..."}, "output": "..."},
    {"step": 2, "tool": "read_chunks",  "input": {"chunk_ids": [...]}, "output": "..."},
    {"step": 3, "tool": "finish",       "input": {"context": "...", "confidence": "high"}, "output": "..."}
  ],
  "error": null
}
```

The `reasoning_trace` field lets you replay and audit every search decision offline.

---

## Roadmap

### Schema Auto-Generation (next)

Currently, users write schemas by hand. The planned next step removes this friction entirely:

> **Given a document sample and a natural-language description of your use case, the LLM generates all five schemas automatically.**

The workflow would become:

```
User describes use case in plain text
          ↓
LLM drafts index_schema, search_schema, answer_schema,
    judge_schema, and query_schema
          ↓
User reviews / tweaks the generated YAML
          ↓
Pipeline runs — no code written
```

This closes the loop on schema-driven RAG: the same LLM capabilities that power retrieval and answering would also be used to *configure the system itself*. A domain expert with no coding experience could fully deploy ReActRAG by describing their documents in plain English.
