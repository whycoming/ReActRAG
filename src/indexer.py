"""
Document indexer: token-based chunking with overlap + LLM-driven metadata extraction.

Pipeline per document:
  markdown text
    → parse pages (split on ## Page N)
    → build chunks (greedy, CHUNK_TOKENS limit, OVERLAP_TOKENS overlap)
    → index each chunk via LLM (cheap model, structured JSON output)
    → build doc-level metadata via LLM (1 call on first pages)
    → save to indexes/{doc_name}.json
"""

from __future__ import annotations
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

import config as cfg
from src.llm_client import LLMClient
from src.schema import IndexSchema


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Chunk:
    def __init__(self, chunk_id: str, chunk_index: int, text: str,
                 page_range: tuple[int, int]):
        self.chunk_id = chunk_id
        self.chunk_index = chunk_index
        self.text = text
        self.page_range = page_range  # (start_page, end_page) inclusive


# ---------------------------------------------------------------------------
# DocumentIndexer
# ---------------------------------------------------------------------------

class DocumentIndexer:
    """
    Indexes a single markdown document into a structured JSON index.
    The JSON schema for each chunk is determined by the provided IndexSchema.
    """

    def __init__(
        self,
        index_schema: IndexSchema,
        markdown_dir: Path | None = None,
        index_dir: Path | None = None,
        client: LLMClient | None = None,
    ):
        self.index_schema = index_schema
        self.markdown_dir = markdown_dir or cfg.MARKDOWN_DIR
        self.index_dir = index_dir or cfg.INDEX_DIR
        self.client = client or LLMClient()
        self.model = cfg.get_model("indexer")
        self.index_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_document(self, doc_name: str, force: bool = False) -> dict:
        """
        Index a document and save to indexes/{doc_name}.json.
        Returns the index dict. Uses cached file if it exists (unless force=True).
        """
        index_path = self.index_dir / f"{doc_name}.json"

        if not force and index_path.exists():
            with open(index_path, "r", encoding="utf-8") as fh:
                return json.load(fh)

        md_path = self._resolve_markdown_path(doc_name)
        text = md_path.read_text(encoding="utf-8")

        pages = self._parse_pages(text)
        chunks = self._build_chunks(pages)

        print(f"  Indexing {doc_name}: {len(pages)} pages → {len(chunks)} chunks")

        # Index each chunk
        sections = []
        for chunk in tqdm(chunks, desc="  chunks", leave=False):
            section = self._index_chunk(chunk)
            section["chunk_id"] = chunk.chunk_id
            section["chunk_index"] = chunk.chunk_index
            section["page_range"] = list(chunk.page_range)
            sections.append(section)

        # Build doc-level metadata
        first_pages_text = "\n".join(
            pages[p] for p in sorted(pages)[:5] if p in pages
        )
        doc_meta = self._build_doc_metadata(doc_name, first_pages_text)

        index = {
            "doc_name": doc_name,
            "source_file": md_path.name,
            "total_pages": max(pages.keys()) if pages else 0,
            "total_chunks": len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "doc_summary": doc_meta.get("doc_summary", ""),
            "doc_meta": doc_meta,
            "sections": sections,
            # Store chunk texts for retrieval (key: chunk_id → text)
            "_chunk_texts": {c.chunk_id: c.text for c in chunks},
        }

        with open(index_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False, indent=2)

        return index

    def load_index(self, doc_name: str) -> dict | None:
        index_path = self.index_dir / f"{doc_name}.json"
        if not index_path.exists():
            return None
        with open(index_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def index_all(self, doc_names: list[str], force: bool = False) -> list[dict]:
        results = []
        for doc_name in doc_names:
            print(f"\nIndexing: {doc_name}")
            try:
                idx = self.index_document(doc_name, force=force)
                results.append(idx)
            except Exception as e:
                print(f"  ERROR indexing {doc_name}: {e}")
        return results

    # ------------------------------------------------------------------
    # Page parsing
    # ------------------------------------------------------------------

    def _resolve_markdown_path(self, doc_name: str) -> Path:
        """Find the markdown file for a doc_name across known directories."""
        candidates = [
            cfg.MINI_MARKDOWN_DIR / f"{doc_name}.md",
            self.markdown_dir / f"{doc_name}.md",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Markdown file for '{doc_name}' not found. Searched:\n"
            + "\n".join(f"  {p}" for p in candidates)
        )

    def _parse_pages(self, text: str) -> dict[int, str]:
        """
        Split markdown on '## Page N' delimiters.
        Returns {page_num: page_content}.
        If no page markers found, treats the whole doc as page 1.
        """
        pages: dict[int, str] = {}
        # Match lines like "## Page 12" or "## Page 12\n"
        pattern = re.compile(r"^##\s+Page\s+(\d+)", re.MULTILINE)
        matches = list(pattern.finditer(text))

        if not matches:
            pages[1] = text.strip()
            return pages

        for i, m in enumerate(matches):
            page_num = int(m.group(1))
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            pages[page_num] = content

        return pages

    # ------------------------------------------------------------------
    # Token-based chunking with overlap
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Fast token count estimation (no tokenizer required)."""
        return max(1, int(len(text) / cfg.APPROX_CHARS_PER_TOKEN))

    def _tail_text(self, text: str, max_tokens: int) -> str:
        """Return the last ~max_tokens worth of text from a string."""
        max_chars = int(max_tokens * cfg.APPROX_CHARS_PER_TOKEN)
        if len(text) <= max_chars:
            return text
        # Try to break at a word boundary
        tail = text[-max_chars:]
        newline_pos = tail.find("\n")
        if newline_pos > 0:
            tail = tail[newline_pos:]
        return tail

    def _build_chunks(self, pages: dict[int, str]) -> list[Chunk]:
        """
        Greedy chunk builder:
        - Accumulate pages until adding the next would exceed CHUNK_TOKENS.
        - Prepend the previous chunk's tail (OVERLAP_TOKENS) to each new chunk.
        - Pages larger than CHUNK_TOKENS are kept as single-page chunks.
        """
        chunks: list[Chunk] = []
        sorted_pages = sorted(pages.items())

        buffer_text = ""
        buffer_tokens = 0
        buffer_start_page = None
        buffer_end_page = None
        overlap_tail = ""

        def flush(start: int, end: int, text: str) -> None:
            if not text.strip():
                return
            idx = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"chunk_{idx:04d}",
                chunk_index=idx,
                text=overlap_tail + text,
                page_range=(start, end),
            ))

        for page_num, page_text in sorted_pages:
            page_tokens = self._estimate_tokens(page_text)

            if page_tokens > cfg.CHUNK_TOKENS:
                # Oversized single page — flush current buffer first, then this page alone
                if buffer_text:
                    flush(buffer_start_page, buffer_end_page, buffer_text)
                    overlap_tail = self._tail_text(buffer_text, cfg.OVERLAP_TOKENS)
                    buffer_text = ""
                    buffer_tokens = 0
                    buffer_start_page = None

                flush(page_num, page_num, page_text)
                overlap_tail = self._tail_text(page_text, cfg.OVERLAP_TOKENS)
                continue

            if buffer_tokens + page_tokens > cfg.CHUNK_TOKENS and buffer_text:
                # Flush current buffer
                flush(buffer_start_page, buffer_end_page, buffer_text)
                overlap_tail = self._tail_text(buffer_text, cfg.OVERLAP_TOKENS)
                buffer_text = page_text
                buffer_tokens = page_tokens
                buffer_start_page = page_num
                buffer_end_page = page_num
            else:
                buffer_text = (buffer_text + "\n" + page_text).strip()
                buffer_tokens += page_tokens
                if buffer_start_page is None:
                    buffer_start_page = page_num
                buffer_end_page = page_num

        if buffer_text:
            flush(buffer_start_page, buffer_end_page, buffer_text)

        return chunks

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _index_chunk(self, chunk: Chunk) -> dict:
        """
        Call the cheap indexer LLM to extract structured metadata for a chunk.
        The JSON schema is generated from the IndexSchema, making this
        fully schema-agnostic.
        """
        schema_str = self.index_schema.to_json_schema_str()
        field_descs = self.index_schema.to_field_descriptions()

        prompt = f"""Analyze the following document chunk and extract structured metadata.

--- CHUNK (pages {chunk.page_range[0]}–{chunk.page_range[1]}) ---
{chunk.text[:6000]}
--- END CHUNK ---

Field descriptions:
{field_descs}

Respond with ONLY a JSON object matching this exact schema (no markdown fences, no explanation):
{schema_str}

Rules:
- Be specific and factual. Only include information actually present in the chunk.
- For list fields, return an empty list [] if nothing relevant is present.
- For the relevance_hints field: be precise — include key numeric values or facts if present."""

        response = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=cfg.INDEXER_TEMPERATURE,
            max_tokens=2048,
        )

        return self._parse_json_response(response.content, chunk.chunk_id)

    def _build_doc_metadata(self, doc_name: str, first_pages_text: str) -> dict:
        """
        One LLM call to extract document-level metadata from the first pages.
        The JSON schema is generated from IndexSchema.doc_fields — fully schema-driven.
        Falls back to a minimal default schema if no doc_fields are defined.
        """
        schema_str = self.index_schema.to_doc_json_schema_str()
        field_descs = self.index_schema.to_doc_field_descriptions()

        prompt = f"""Analyze the beginning of this document and extract document-level metadata.

Document name: {doc_name}

--- FIRST PAGES ---
{first_pages_text[:4000]}
--- END ---

Field descriptions:
{field_descs}

Respond with ONLY a JSON object matching this exact schema (no markdown fences, no explanation):
{schema_str}

Rules:
- Be factual. Only include information visible in the provided text.
- For string fields with no applicable value, use null."""

        response = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=cfg.INDEXER_TEMPERATURE,
            max_tokens=512,
        )

        try:
            return self._parse_json_response(response.content, "doc_meta")
        except Exception:
            # Minimal fallback
            return {"doc_summary": ""}

    @staticmethod
    def _parse_json_response(text: str, context: str) -> dict:
        """
        Parse JSON from LLM response. Strips markdown fences if present.
        """
        # Strip markdown code fences
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Attempt to extract the first JSON object
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            print(f"  WARNING: Failed to parse JSON for {context}: {e}")
            print(f"  Raw: {text[:200]}")
            return {}
