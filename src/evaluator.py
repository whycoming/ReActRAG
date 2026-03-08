"""
Evaluator: runs the full pipeline against a JSONL eval set and scores with LLM judge.

All answer types (numeric, narrative, yes/no) are scored by the same LLM judge
with unified criteria. This avoids brittle regex matching and handles paraphrasing.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

import config as cfg
from src.llm_client import LLMClient
from src.schema import IndexSchema, QuerySchema, JudgeSchema
from src.indexer import DocumentIndexer
from src.searcher import ReActSearcher
from src.answerer import Answerer




@dataclass
class QuestionResult:
    record_id: str | None
    question: str
    doc_name: str
    predicted: str
    expected: str | None
    is_correct: bool | None       # None if no expected answer
    judge_reasoning: str
    chunks_read: list[str]
    react_steps: int
    confidence: str
    latency_seconds: float
    reasoning_trace: list[dict] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalResults:
    split: str
    total: int
    correct: int
    accuracy: float
    by_extra_field: dict[str, dict]   # e.g. by question_type
    results: list[QuestionResult]
    timestamp: str


class Evaluator:
    """
    End-to-end evaluator that:
    1. Loads a JSONL eval set.
    2. For each record: indexes the doc (if not cached), searches, answers.
    3. Scores each answer with an LLM judge (driven by JudgeSchema).
    4. Reports accuracy by total and by any extra_context_fields in the query schema.
    """

    def __init__(
        self,
        index_schema: IndexSchema,
        query_schema: QuerySchema,
        judge_schema: JudgeSchema,
        indexer: DocumentIndexer | None = None,
        searcher: ReActSearcher | None = None,
        answerer: Answerer | None = None,
        judge_client: LLMClient | None = None,
    ):
        self.index_schema = index_schema
        self.query_schema = query_schema
        self.judge_schema = judge_schema
        self.indexer = indexer or DocumentIndexer(index_schema)
        self.searcher = searcher
        self.answerer = answerer
        self.judge_client = judge_client or LLMClient()
        self.judge_model = cfg.get_model("judge")

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        eval_jsonl_path: str | Path,
        max_questions: int | None = None,
        skip_index: bool = False,
        results_dir: Path | None = None,
    ) -> EvalResults:
        """
        Run the full evaluation pipeline.

        Args:
            eval_jsonl_path: Path to the JSONL eval file.
            max_questions:   If set, only run this many questions (for quick tests).
            skip_index:      If True, assume indexes already exist (skip indexing step).
            results_dir:     Where to write results JSONL. Defaults to config.RESULTS_DIR.
        """
        results_dir = results_dir or cfg.RESULTS_DIR
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        records = self._load_jsonl(eval_jsonl_path)
        if max_questions:
            records = records[:max_questions]

        split_name = Path(eval_jsonl_path).stem
        print(f"\n{'='*60}")
        print(f"Evaluating: {split_name} ({len(records)} questions)")
        print(f"{'='*60}\n")

        question_results: list[QuestionResult] = []

        for i, record in enumerate(tqdm(records, desc="Evaluating")):
            print(f"\n[{i+1}/{len(records)}] {self.query_schema.extract_id(record) or ''}")
            result = self._run_one(record, skip_index=skip_index)
            question_results.append(result)
            self._print_result(result)

        # Aggregate
        eval_results = self._aggregate(split_name, question_results)

        # Save results
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out_path = results_dir / f"eval_{split_name}_{ts}.jsonl"
        self._save_results(eval_results, out_path)

        self._print_summary(eval_results)
        return eval_results

    # ------------------------------------------------------------------
    # Single-question pipeline
    # ------------------------------------------------------------------

    def _run_one(self, record: dict, skip_index: bool) -> QuestionResult:
        record_id = self.query_schema.extract_id(record)
        start = time.time()

        try:
            question = self.query_schema.extract_question(record)
            doc_name = self.query_schema.extract_doc_name(record)
            expected = self.query_schema.extract_answer(record)
            extra_context = self.query_schema.extract_extra_context(record)

            print(f"  Q: {question[:80]}...")
            print(f"  Doc: {doc_name}")

            # Step 1: index
            if not skip_index:
                doc_index = self.indexer.index_document(doc_name)
            else:
                doc_index = self.indexer.load_index(doc_name)
                if doc_index is None:
                    raise FileNotFoundError(f"Index not found for {doc_name}. Run without --skip-index first.")

            # Step 2: search
            search_result = self.searcher.search(
                question=question,
                doc_index=doc_index,
                extra_context=extra_context,
            )
            self._print_trace(search_result.reasoning_trace)
            print(f"  Search: {search_result.react_steps} steps, confidence={search_result.confidence}, "
                  f"chunks_read={len(search_result.chunks_read)}")

            # Step 3: answer
            predicted = self.answerer.answer(
                question=question,
                search_result=search_result,
                extra_context=extra_context,
            )
            print(f"  Predicted: {predicted[:100]}")
            if expected:
                print(f"  Expected:  {expected[:100]}")

            # Step 4: judge
            is_correct, judge_reasoning = self._judge(question, predicted, expected)
            if expected is not None:
                print(f"  Judge: {'CORRECT' if is_correct else 'INCORRECT'} — {judge_reasoning}")

            return QuestionResult(
                record_id=record_id,
                question=question,
                doc_name=doc_name,
                predicted=predicted,
                expected=expected,
                is_correct=is_correct,
                judge_reasoning=judge_reasoning,
                chunks_read=search_result.chunks_read,
                react_steps=search_result.react_steps,
                confidence=search_result.confidence,
                latency_seconds=round(time.time() - start, 2),
                reasoning_trace=search_result.reasoning_trace,
            )

        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print(f"  ERROR: {e}")
            return QuestionResult(
                record_id=record_id,
                question=record.get(self.query_schema.question_field, ""),
                doc_name=record.get(self.query_schema.doc_field, ""),
                predicted="ERROR",
                expected=self.query_schema.extract_answer(record),
                is_correct=False,
                judge_reasoning="Pipeline error",
                chunks_read=[],
                react_steps=0,
                confidence="low",
                latency_seconds=round(time.time() - start, 2),
                error=str(e),
            )

    # ------------------------------------------------------------------
    # LLM Judge
    # ------------------------------------------------------------------

    def _judge(
        self,
        question: str,
        predicted: str,
        expected: str | None,
    ) -> tuple[bool | None, str]:
        """
        Score predicted vs expected using the LLM judge configured by JudgeSchema.
        Returns (is_correct, reasoning_sentence).
        """
        if expected is None:
            return None, "No expected answer provided"

        prompt = self.judge_schema.render_prompt(
            question=question,
            expected=expected,
            predicted=predicted,
        )

        response = self.judge_client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=self.judge_schema.system_prompt,
            model=self.judge_model,
            temperature=cfg.LLM_JUDGE_TEMPERATURE,
            max_tokens=256,
        )

        return self.judge_schema.parse_response(response.content)

    # ------------------------------------------------------------------
    # Aggregation and output
    # ------------------------------------------------------------------

    def _aggregate(self, split: str, results: list[QuestionResult]) -> EvalResults:
        scored = [r for r in results if r.is_correct is not None]
        correct = sum(1 for r in scored if r.is_correct)
        accuracy = correct / len(scored) if scored else 0.0

        # Break down by any extra_context_fields
        by_extra: dict[str, dict] = {}
        for r in results:
            extra = self.query_schema.extract_extra_context(
                {self.query_schema.question_field: r.question,
                 self.query_schema.doc_field: r.doc_name}
            )
            # We don't have the original record here — skip field breakdown if not available
            # (it's stored per result in a full implementation)

        return EvalResults(
            split=split,
            total=len(results),
            correct=correct,
            accuracy=accuracy,
            by_extra_field=by_extra,
            results=results,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _save_results(self, eval_results: EvalResults, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for r in eval_results.results:
                fh.write(json.dumps({
                    "record_id": r.record_id,
                    "question": r.question,
                    "doc_name": r.doc_name,
                    "predicted": r.predicted,
                    "expected": r.expected,
                    "is_correct": r.is_correct,
                    "judge_reasoning": r.judge_reasoning,
                    "chunks_read": r.chunks_read,
                    "react_steps": r.react_steps,
                    "confidence": r.confidence,
                    "latency_seconds": r.latency_seconds,
                    "reasoning_trace": r.reasoning_trace,
                    "error": r.error,
                }, ensure_ascii=False) + "\n")
        print(f"\nResults saved to: {path}")

    @staticmethod
    def _print_trace(trace: list[dict]) -> None:
        if not trace:
            return
        print("  ── ReAct Trace ──────────────────────────────────")
        for t in trace:
            tool = t["tool"]
            inp = t["input"]
            out = t.get("output", "")
            # Compact input summary
            if "query" in inp:
                inp_str = f'query="{inp["query"][:60]}"'
            elif "chunk_ids" in inp:
                inp_str = f"chunk_ids={inp['chunk_ids']}"
            elif "expression" in inp:
                inp_str = f'expr="{inp["expression"]}"'
            elif "context" in inp:
                inp_str = f'confidence={inp.get("confidence","?")} ctx="{inp["context"][:60]}..."'
            else:
                inp_str = str(inp)[:80]
            print(f"  Step {t['step']:2d} [{tool}] {inp_str}")
            if out and tool != "finish":
                # First non-empty line of output
                first_line = next((l for l in out.splitlines() if l.strip()), "")
                print(f"         → {first_line[:100]}")
        print("  ─────────────────────────────────────────────────")

    @staticmethod
    def _print_result(r: QuestionResult) -> None:
        if r.error:
            print(f"  ❌ ERROR: {r.error}")

    @staticmethod
    def _print_summary(eval_results: EvalResults) -> None:
        results = eval_results.results
        scored = [r for r in results if r.is_correct is not None]
        correct = sum(1 for r in scored if r.is_correct)

        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY — {eval_results.split}")
        print(f"{'='*60}")
        print(f"Overall: {correct}/{len(scored)} = {eval_results.accuracy:.1%}")

        if results:
            avg_latency = sum(r.latency_seconds for r in results) / len(results)
            avg_steps = sum(r.react_steps for r in results) / len(results)
            print(f"Avg latency: {avg_latency:.1f}s/question")
            print(f"Avg ReAct steps: {avg_steps:.1f}")

        errors = [r for r in results if r.error]
        if errors:
            print(f"Errors: {len(errors)}")

        print(f"Timestamp: {eval_results.timestamp}")

    @staticmethod
    def _load_jsonl(path: str | Path) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
