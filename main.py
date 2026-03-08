"""
GraphIndex CLI entry point.

Commands:
  index   — index one or more documents
  ask     — ask a single question against a document
  eval    — run evaluation against a JSONL benchmark

Examples:
  # Index all documents in the mini eval set
  python main.py index --docs-from financebench_output/eval/mini_markdown/

  # Index a single document
  python main.py index --doc AMD_2015_10K

  # Ask a single question
  python main.py ask --doc AMD_2015_10K \\
    --question "What is the FY2015 D&A % margin for AMD?"

  # Run validation evaluation
  python main.py eval --split validation

  # Run with a custom query schema (any dataset format)
  python main.py eval --eval-file my_dataset.jsonl \\
    --query-schema schemas/my_query_schema.yaml

  # Load all schemas from a single pipeline file
  python main.py eval --split validation \\
    --pipeline-schema schemas/default_pipeline.yaml

  # Use Qwen instead of Anthropic
  LLM_PROVIDER=qwen python main.py eval --split test
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Make sure project root is on path when running as script
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import config as cfg
from src.schema import (
    load_index_schema, load_query_schema,
    load_search_schema, load_answer_schema, load_judge_schema,
    load_pipeline_schema,
)
from src.llm_client import LLMClient
from src.indexer import DocumentIndexer
from src.searcher import ReActSearcher
from src.answerer import Answerer
from src.evaluator import Evaluator


def build_components(
    index_schema_path: str | None = None,
    query_schema_path: str | None = None,
    search_schema_path: str | None = None,
    answer_schema_path: str | None = None,
    judge_schema_path: str | None = None,
    pipeline_schema_path: str | None = None,
):
    """Load schemas and build pipeline components."""
    # --pipeline-schema overrides individual schema paths
    if pipeline_schema_path:
        ps = load_pipeline_schema(pipeline_schema_path)
        index_schema  = ps.index
        query_schema  = ps.query
        search_schema = ps.search
        answer_schema = ps.answer
        judge_schema  = ps.judge
    else:
        index_schema  = load_index_schema(index_schema_path or cfg.DEFAULT_INDEX_SCHEMA)
        query_schema  = load_query_schema(query_schema_path or cfg.DEFAULT_QUERY_SCHEMA)
        search_schema = load_search_schema(search_schema_path or cfg.DEFAULT_SEARCH_SCHEMA)
        answer_schema = load_answer_schema(answer_schema_path or cfg.DEFAULT_ANSWER_SCHEMA)
        judge_schema  = load_judge_schema(judge_schema_path or cfg.DEFAULT_JUDGE_SCHEMA)

    client = LLMClient()
    print(f"Provider: {cfg.LLM_PROVIDER}")

    indexer  = DocumentIndexer(index_schema, client=client)
    searcher = ReActSearcher(index_schema, search_schema, client=client)
    answerer = Answerer(answer_schema, client=client)

    return index_schema, query_schema, search_schema, answer_schema, judge_schema, \
           indexer, searcher, answerer, client


# ---------------------------------------------------------------------------
# Command: index
# ---------------------------------------------------------------------------

def cmd_index(args):
    index_schema, _, _, _, _, indexer, _, _, _ = build_components(
        index_schema_path=args.index_schema,
        pipeline_schema_path=args.pipeline_schema,
    )

    doc_names = []

    if args.doc:
        doc_names = [args.doc]
    elif args.docs_from:
        docs_dir = Path(args.docs_from)
        if not docs_dir.exists():
            print(f"ERROR: Directory not found: {docs_dir}")
            sys.exit(1)
        doc_names = [p.stem for p in sorted(docs_dir.glob("*.md"))]
        print(f"Found {len(doc_names)} documents in {docs_dir}")
    elif args.all:
        md_dir = cfg.MARKDOWN_DIR
        doc_names = [p.stem for p in sorted(md_dir.glob("*.md"))]
        print(f"Found {len(doc_names)} documents in {md_dir}")
    else:
        print("ERROR: Specify --doc, --docs-from, or --all")
        sys.exit(1)

    print(f"\nIndexing {len(doc_names)} document(s) with provider={cfg.LLM_PROVIDER}")
    for name in doc_names:
        print(f"\n{'─'*50}")
        print(f"Document: {name}")
        try:
            idx = indexer.index_document(name, force=args.force)
            print(f"  ✓ {idx['total_chunks']} chunks indexed → indexes/{name}.json")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\nIndexing complete.")


# ---------------------------------------------------------------------------
# Command: ask
# ---------------------------------------------------------------------------

def cmd_ask(args):
    index_schema, query_schema, search_schema, answer_schema, judge_schema, \
        indexer, searcher, answerer, _ = build_components(
        index_schema_path=args.index_schema,
        query_schema_path=args.query_schema,
        search_schema_path=args.search_schema,
        answer_schema_path=args.answer_schema,
        judge_schema_path=args.judge_schema,
        pipeline_schema_path=args.pipeline_schema,
    )

    doc_name = args.doc.removesuffix(".md")
    question = args.question

    print(f"\nDocument: {doc_name}")
    print(f"Question: {question}\n")

    # Load or build index
    doc_index = indexer.load_index(doc_name)
    if doc_index is None:
        print("Index not found. Indexing now...")
        doc_index = indexer.index_document(doc_name)

    # Search
    print("Searching...\n")
    search_result = searcher.search(
        question=question,
        doc_index=doc_index,
        extra_context={"user_query": True},
    )

    if args.verbose:
        print("\n── Reasoning Trace ──")
        for t in search_result.reasoning_trace:
            print(f"  Step {t['step']}: {t['tool']}({_fmt_input(t['input'])})")
            print(f"    → {t['output'][:200]}")
        print()

    print(f"Chunks read: {search_result.chunks_read}")
    print(f"ReAct steps: {search_result.react_steps}")
    print(f"Confidence:  {search_result.confidence}")

    # Answer
    print("\nGenerating answer...")
    final_answer = answerer.answer(
        question=question,
        search_result=search_result,
    )

    print(f"\n{'='*50}")
    print(f"ANSWER: {final_answer}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Command: eval
# ---------------------------------------------------------------------------

def cmd_eval(args):
    index_schema, query_schema, search_schema, answer_schema, judge_schema, \
        indexer, searcher, answerer, client = build_components(
        index_schema_path=args.index_schema,
        query_schema_path=args.query_schema,
        search_schema_path=args.search_schema,
        answer_schema_path=args.answer_schema,
        judge_schema_path=args.judge_schema,
        pipeline_schema_path=args.pipeline_schema,
    )

    # Resolve eval file
    if args.eval_file:
        eval_path = Path(args.eval_file)
    elif args.split == "test":
        eval_path = cfg.EVAL_DIR / "test_small.jsonl"
    elif args.split == "validation":
        eval_path = cfg.EVAL_DIR / "validation_small.jsonl"
    else:
        print(f"ERROR: Unknown split '{args.split}'. Use 'test' or 'validation', or provide --eval-file.")
        sys.exit(1)

    if not eval_path.exists():
        print(f"ERROR: Eval file not found: {eval_path}")
        sys.exit(1)

    evaluator = Evaluator(
        index_schema=index_schema,
        query_schema=query_schema,
        judge_schema=judge_schema,
        indexer=indexer,
        searcher=searcher,
        answerer=answerer,
        judge_client=client,
    )

    evaluator.evaluate(
        eval_jsonl_path=eval_path,
        max_questions=args.max,
        skip_index=args.skip_index,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_input(inp: dict) -> str:
    if "query" in inp:
        return f'query="{inp["query"][:50]}"'
    if "chunk_ids" in inp:
        return f"chunk_ids={inp['chunk_ids']}"
    if "expression" in inp:
        return f'expr="{inp["expression"]}"'
    return str(inp)[:60]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_schema_args(parser):
    """Add common schema arguments to a (sub)parser."""
    parser.add_argument(
        "--index-schema", default=None, metavar="PATH",
        help="Path to index schema YAML (default: schemas/default_index_schema.yaml)",
    )
    parser.add_argument(
        "--query-schema", default=None, metavar="PATH",
        help="Path to query schema YAML (default: schemas/default_query_schema.yaml)",
    )
    parser.add_argument(
        "--search-schema", default=None, metavar="PATH",
        help="Path to search/ReAct schema YAML (default: schemas/default_search_schema.yaml)",
    )
    parser.add_argument(
        "--answer-schema", default=None, metavar="PATH",
        help="Path to answer schema YAML (default: schemas/default_answer_schema.yaml)",
    )
    parser.add_argument(
        "--judge-schema", default=None, metavar="PATH",
        help="Path to judge schema YAML (default: schemas/default_judge_schema.yaml)",
    )
    parser.add_argument(
        "--pipeline-schema", default=None, metavar="PATH",
        help="Load all schemas from a single pipeline YAML (overrides individual schema flags)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="GraphIndex — LLM-native RAG with arbitrary schema support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── index ──────────────────────────────────────────────────────────
    p_index = subparsers.add_parser("index", help="Index documents")
    _add_schema_args(p_index)
    group = p_index.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc", metavar="DOC_NAME",
                       help="Index a single document (without .md extension)")
    group.add_argument("--docs-from", metavar="DIR",
                       help="Index all .md files in a directory")
    group.add_argument("--all", action="store_true",
                       help="Index all documents in MARKDOWN_DIR")
    p_index.add_argument("--force", action="store_true",
                         help="Re-index even if cached index exists")
    p_index.set_defaults(func=cmd_index)

    # ── ask ────────────────────────────────────────────────────────────
    p_ask = subparsers.add_parser("ask", help="Ask a single question")
    _add_schema_args(p_ask)
    p_ask.add_argument("--doc", required=True, metavar="DOC_NAME",
                       help="Document name (with or without .md)")
    p_ask.add_argument("--question", required=True, metavar="TEXT",
                       help="Question to answer")
    p_ask.add_argument("--verbose", "-v", action="store_true",
                       help="Print full reasoning trace")
    p_ask.set_defaults(func=cmd_ask)

    # ── eval ───────────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("eval", help="Run evaluation")
    _add_schema_args(p_eval)
    p_eval.add_argument("--split", default="validation",
                        choices=["test", "validation"],
                        help="Built-in eval split to use (default: validation)")
    p_eval.add_argument("--eval-file", metavar="PATH",
                        help="Path to a custom JSONL eval file (overrides --split)")
    p_eval.add_argument("--max", type=int, default=None, metavar="N",
                        help="Only evaluate first N questions")
    p_eval.add_argument("--skip-index", action="store_true",
                        help="Use existing indexes; skip indexing step")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
