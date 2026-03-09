"""
Evaluator: runs the full pipeline against a JSONL eval set and scores with LLM judge.

All answer types (numeric, narrative, yes/no) are scored by the same LLM judge
with unified criteria. This avoids brittle regex matching and handles paraphrasing.
"""

from __future__ import annotations
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        fixed_doc: str | None = None,
        search_all_docs: bool = False,
    ):
        self.index_schema = index_schema
        self.query_schema = query_schema
        self.judge_schema = judge_schema
        self.indexer = indexer or DocumentIndexer(index_schema)
        self.searcher = searcher
        self.answerer = answerer
        self.judge_client = judge_client or LLMClient()
        self.judge_model = cfg.get_model("judge")
        self.fixed_doc = fixed_doc
        self.search_all_docs = search_all_docs

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

        # Step A: index all unique documents sequentially (avoids concurrent write conflicts)
        if not skip_index:
            unique_docs: set[str] = set()
            for record in records:
                try:
                    doc_name = self.query_schema.extract_doc_name(record, override_doc=self.fixed_doc)
                    unique_docs.add(doc_name)
                except (ValueError, KeyError):
                    pass  # all-docs mode or missing field — handled per-question
            if unique_docs:
                print(f"Pre-indexing {len(unique_docs)} document(s)...")
                for doc_name in sorted(unique_docs):
                    self.indexer.index_document(doc_name)
                print()

        # Step B: run search + answer + judge in parallel
        parallelism = cfg.EVAL_PARALLELISM
        ordered_results: list[QuestionResult | None] = [None] * len(records)

        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {
                pool.submit(
                    self._run_one, record, True, self.fixed_doc
                ): i
                for i, record in enumerate(records)
            }
            for future in tqdm(as_completed(futures), total=len(records), desc="Evaluating"):
                i = futures[future]
                ordered_results[i] = future.result()

        question_results: list[QuestionResult] = []
        for i, result in enumerate(ordered_results):
            print(f"\n[{i+1}/{len(records)}] {self.query_schema.extract_id(records[i]) or ''}")
            self._print_result(result)
            question_results.append(result)

        # Aggregate and save
        eval_results = self._aggregate(split_name, question_results)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out_path = results_dir / f"eval_{split_name}_{ts}.jsonl"
        self._save_results(eval_results, out_path)
        self._print_summary(eval_results)
        return eval_results

    # ------------------------------------------------------------------
    # Single-question pipeline
    # ------------------------------------------------------------------

    def _run_one(self, record: dict, skip_index: bool, fixed_doc: str | None = None) -> QuestionResult:
        record_id = self.query_schema.extract_id(record)
        start = time.time()

        try:
            question = self.query_schema.extract_question(record)
            extra_context = self.query_schema.extract_extra_context(record)
            expected = self.query_schema.extract_answer(record)

            try:
                doc_name = self.query_schema.extract_doc_name(record, override_doc=fixed_doc)
                presearched = False
            except ValueError:
                if self.search_all_docs:
                    print(f"  No doc_field — searching all indexed documents...")
                    doc_name, search_result = self._search_all_docs(question, extra_context)
                    presearched = True
                else:
                    raise

            print(f"  Q: {question[:80]}...")
            print(f"  Doc: {doc_name}")

            # Step 1: index
            if not presearched:
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
                doc_name=record.get(self.query_schema.doc_field, "") if self.query_schema.doc_field else (fixed_doc or ""),
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
    # All-docs brute-force search
    # ------------------------------------------------------------------

    def _search_all_docs(self, question: str, extra_context: dict):
        """Search every indexed document and return (doc_name, best_search_result).

        If the search schema has a doc_prescreening_prompt, first runs a cheap LLM
        call to rank documents by their metadata (primary_subject, time_period,
        doc_summary), then runs full ReAct only on the top-N candidates.
        """
        CONF_RANK = {"high": 2, "medium": 1, "low": 0}
        index_files = sorted(cfg.INDEX_DIR.glob("*.json"))
        if not index_files:
            raise FileNotFoundError(f"No index files found in {cfg.INDEX_DIR}. Run indexing first.")

        # Load all doc-level metadata (no LLM call, just JSON reads)
        all_docs: list[tuple] = []
        for path in index_files:
            with open(path, encoding="utf-8") as fh:
                doc_index = json.load(fh)
            all_docs.append((path, doc_index))

        # Keyword pre-sort: rank docs by how well their metadata matches the question.
        # This runs before LLM pre-screening so the LLM sees the most relevant docs first,
        # and the fallback path (first top_n) also picks the right candidates.
        keyword_scores = {
            d.get("doc_name", p.stem): self._score_doc_meta(question, d)
            for p, d in all_docs
        }

        # Semantic scoring: single LLM batch call on doc_summary
        print(f"  Semantic scoring {len(all_docs)} doc summaries ...", flush=True)
        semantic_scores = self._semantic_score_summaries(question, all_docs)

        def _combined_score(item: tuple) -> float:
            path, doc_index = item
            name = doc_index.get("doc_name", path.stem)
            return keyword_scores.get(name, 0.0) + semantic_scores.get(name, 0.0)

        all_docs.sort(key=_combined_score, reverse=True)

        # Pre-screening: use doc-level metadata to shortlist candidates
        search_schema = self.searcher.search_schema
        if search_schema.doc_prescreening_prompt:
            candidate_names = self._prescreen_docs(question, all_docs)
            all_docs = [(p, d) for p, d in all_docs
                        if d.get("doc_name", p.stem) in candidate_names]
            print(f"  Pre-screened: {[d.get('doc_name', p.stem) for p, d in all_docs]}")
        else:
            print(f"  No pre-screening — will try all {len(all_docs)} documents")

        best_doc: str | None = None
        best_result = None

        for i, (path, doc_index) in enumerate(all_docs, 1):
            doc_name = doc_index.get("doc_name", path.stem)
            print(f"  [{i}/{len(all_docs)}] Trying: {doc_name} ...", flush=True)
            result = self.searcher.search(question, doc_index, extra_context)
            if best_result is None or \
               CONF_RANK.get(result.confidence, 0) > CONF_RANK.get(best_result.confidence, 0):
                best_doc, best_result = doc_name, result
            if best_result.confidence == "high":
                print(f"  → High-confidence match found: {best_doc}")
                break

        return best_doc, best_result

    @staticmethod
    def _score_doc_meta(question: str, doc_index: dict) -> float:
        """Keyword score of doc_name and doc_meta fields against the question.

        Year-aware: extracts 4-digit years from compound tokens so that
        "FY2022" in a question matches "2022" in "3M_2022_10K", and
        "2023Q2" in a doc_name matches "2023" in "Q2 FY2023" queries.

        Weights:
        - doc_name (company + year encoded in filename): 3x
        - doc_meta fields (time_period, primary_subject, doc_type): 2x
        """
        _year_pat = re.compile(r"\d{4}")

        def _expand(text: str) -> set[str]:
            tokens = set(re.findall(r"\w+", text.lower()))
            for t in list(tokens):
                tokens.update(_year_pat.findall(t))
            return tokens

        query_terms = _expand(question)
        if not query_terms:
            return 0.0

        score = 0.0
        score += len(query_terms & _expand(doc_index.get("doc_name", ""))) * 3.0
        for value in doc_index.get("doc_meta", {}).values():
            score += len(query_terms & _expand(str(value))) * 2.0
        return score

    def _semantic_score_summaries(self, question: str, all_docs: list[tuple]) -> dict[str, float]:
        """Single LLM batch call: score each document's summary for relevance to the question.

        Returns a dict mapping doc_name → float score (0–10).
        Falls back to all-zeros on parse failure (keyword score still applies).
        """
        lines = []
        for i, (path, doc_index) in enumerate(all_docs, 1):
            doc_name = doc_index.get("doc_name", path.stem)
            summary = (doc_index.get("doc_summary") or "")[:300]
            lines.append(f"{i}. {doc_name}: {summary}")

        prompt = (
            "You are a document relevance ranker.\n"
            f"Question: {question}\n\n"
            "Rate each document's summary for how likely it contains the answer, "
            "on a scale of 0 (irrelevant) to 10 (highly relevant).\n\n"
            "Documents:\n"
            + "\n".join(lines)
            + "\n\nReturn ONLY a JSON object mapping each document name to its integer score. "
            'Example: {"3M_2022_10K": 9, "3M_2019_10K": 1}'
        )

        response = self.judge_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.judge_model,
            temperature=0,
            max_tokens=1024,
        )

        match = re.search(r"\{.*?\}", response.content, re.DOTALL)
        if match:
            try:
                scores = json.loads(match.group())
                return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
            except (json.JSONDecodeError, ValueError):
                pass
        print("  WARNING: could not parse semantic scoring response, using keyword scores only")
        return {}

    def _prescreen_docs(self, question: str, all_docs: list[tuple]) -> list[str]:
        """Use a cheap LLM call to shortlist candidate documents by metadata."""
        top_n = self.searcher.search_schema.doc_prescreening_top_n
        prompt_template = self.searcher.search_schema.doc_prescreening_prompt

        doc_lines = []
        for i, (path, doc_index) in enumerate(all_docs, 1):
            doc_name = doc_index.get("doc_name", path.stem)
            meta = doc_index.get("doc_meta", {})
            subject  = meta.get("primary_subject", "")
            period   = meta.get("time_period", "")
            doc_type = meta.get("doc_type", "")
            summary  = (doc_index.get("doc_summary") or meta.get("doc_summary", ""))[:200]
            doc_lines.append(
                f"{i:2d}. {doc_name} | {subject} | {period} | {doc_type}\n"
                f"    {summary}"
            )
        doc_list_str = "\n".join(doc_lines)

        prompt = prompt_template.format(
            question=question,
            top_n=top_n,
            doc_list=doc_list_str,
        )
        print(f"  Pre-screening {len(all_docs)} docs → top {top_n} ...", flush=True)
        response = self.judge_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.judge_model,
            temperature=0,
            max_tokens=512,
        )

        # Parse JSON array from response
        text = response.content
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                names = json.loads(match.group())
                if isinstance(names, list) and names:
                    return [str(n) for n in names]
            except json.JSONDecodeError:
                pass
        # Fallback: return top_n by original order
        print("  WARNING: could not parse pre-screening response, using first top_n docs")
        return [d.get("doc_name", p.stem) for p, d in all_docs[:top_n]]

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
