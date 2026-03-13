"""
Document indexer: token-based chunking with overlap + LLM-driven metadata extraction.

Pipeline per document:
  markdown text
    → parse pages (split on ## Page N)
    → build chunks (greedy token-based OR agentic LLM-boundary)
    → index each chunk via LLM (cheap model, structured JSON output)
    → build doc-level metadata from ALL chunk metadata (not just first pages)
    → save to indexes/graphindex.db (SQLite)
"""

from __future__ import annotations
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

import config as cfg
from src.llm_client import LLMClient
from src.schema import IndexSchema


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_name     TEXT PRIMARY KEY,
    source_file  TEXT,
    total_pages  INTEGER,
    total_chunks INTEGER,
    indexed_at   TEXT,
    doc_summary  TEXT,
    doc_meta     TEXT   -- JSON blob
);

CREATE TABLE IF NOT EXISTS sections (
    doc_name    TEXT,
    chunk_id    TEXT,
    chunk_index INTEGER,
    page_start  INTEGER,
    page_end    INTEGER,
    section     TEXT,   -- JSON blob of all LLM-extracted fields
    PRIMARY KEY (doc_name, chunk_id)
);

CREATE TABLE IF NOT EXISTS chunk_texts (
    doc_name TEXT,
    chunk_id TEXT,
    text     TEXT,
    PRIMARY KEY (doc_name, chunk_id)
);
"""


def _open_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite index database and ensure schema exists."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_DB_SCHEMA)
    conn.commit()
    return conn


def _index_to_db(conn: sqlite3.Connection, index: dict) -> None:
    """Write a complete index dict into the database (upsert)."""
    doc_name = index["doc_name"]

    conn.execute(
        "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?)",
        (
            doc_name,
            index.get("source_file", ""),
            index.get("total_pages", 0),
            index.get("total_chunks", 0),
            index.get("indexed_at", ""),
            index.get("doc_summary", ""),
            json.dumps(index.get("doc_meta", {}), ensure_ascii=False),
        ),
    )

    # Delete old sections/texts for this doc before re-inserting
    conn.execute("DELETE FROM sections WHERE doc_name=?", (doc_name,))
    conn.execute("DELETE FROM chunk_texts WHERE doc_name=?", (doc_name,))

    for sec in index.get("sections", []):
        chunk_id = sec["chunk_id"]
        conn.execute(
            "INSERT INTO sections VALUES (?,?,?,?,?,?)",
            (
                doc_name,
                chunk_id,
                sec.get("chunk_index", 0),
                sec.get("page_range", [0, 0])[0],
                sec.get("page_range", [0, 0])[-1],
                json.dumps(sec, ensure_ascii=False),
            ),
        )

    for chunk_id, text in index.get("_chunk_texts", {}).items():
        conn.execute(
            "INSERT INTO chunk_texts VALUES (?,?,?)",
            (doc_name, chunk_id, text),
        )

    conn.commit()


def _db_to_index(conn: sqlite3.Connection, doc_name: str) -> dict | None:
    """Reconstruct a full index dict from the database for one document."""
    row = conn.execute(
        "SELECT source_file, total_pages, total_chunks, indexed_at, doc_summary, doc_meta "
        "FROM documents WHERE doc_name=?",
        (doc_name,),
    ).fetchone()
    if row is None:
        return None

    source_file, total_pages, total_chunks, indexed_at, doc_summary, doc_meta_json = row
    doc_meta = json.loads(doc_meta_json) if doc_meta_json else {}

    sections = []
    for (sec_json,) in conn.execute(
        "SELECT section FROM sections WHERE doc_name=? ORDER BY chunk_index",
        (doc_name,),
    ):
        sections.append(json.loads(sec_json))

    chunk_texts = {
        cid: text
        for cid, text in conn.execute(
            "SELECT chunk_id, text FROM chunk_texts WHERE doc_name=?", (doc_name,)
        )
    }

    return {
        "doc_name": doc_name,
        "source_file": source_file,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "indexed_at": indexed_at,
        "doc_summary": doc_summary,
        "doc_meta": doc_meta,
        "sections": sections,
        "_chunk_texts": chunk_texts,
    }


def _db_list_docs(conn: sqlite3.Connection) -> list[str]:
    """Return all indexed doc_names."""
    return [r[0] for r in conn.execute("SELECT doc_name FROM documents ORDER BY doc_name")]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Chunk:
    def __init__(self, chunk_id: str, chunk_index: int, text: str,
                 page_range: tuple[int, int], summary: str = ""):
        self.chunk_id = chunk_id
        self.chunk_index = chunk_index
        self.text = text
        self.page_range = page_range  # (start_page, end_page) inclusive
        self.summary = summary        # optional: LLM-generated summary (agentic mode)


class Segment:
    """A structural unit of text extracted from a page (header, table, or paragraph)."""
    def __init__(self, text: str, page_num: int, seg_type: str):
        self.text = text
        self.page_num = page_num
        self.seg_type = seg_type  # "header" | "table" | "paragraph"


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
        self._db_path = self.index_dir / "graphindex.db"
        self._conn = _open_db(self._db_path)
        self._db_lock = __import__("threading").Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_document(self, doc_name: str, force: bool = False) -> dict:
        """
        Index a document and persist to the SQLite database.
        Returns the index dict. Uses cached entry if it exists (unless force=True).
        """
        if not force:
            with self._db_lock:
                cached = _db_to_index(self._conn, doc_name)
            if cached is not None:
                return cached

        md_path = self._resolve_markdown_path(doc_name)
        text = md_path.read_text(encoding="utf-8")

        pages = self._parse_pages(text)
        chunks = self._agentic_build_chunks(pages) if cfg.AGENTIC_CHUNKING else self._build_chunks(pages)

        print(f"  Indexing {doc_name}: {len(pages)} pages → {len(chunks)} chunks")

        # Index each chunk
        sections = []
        for chunk in tqdm(chunks, desc="  chunks", leave=False):
            section = self._index_chunk(chunk)
            section["chunk_id"] = chunk.chunk_id
            section["chunk_index"] = chunk.chunk_index
            section["page_range"] = list(chunk.page_range)
            sections.append(section)

        # Build doc-level metadata from all indexed chunk sections
        doc_meta = self._build_doc_metadata(doc_name, sections)

        index = {
            "doc_name": doc_name,
            "source_file": md_path.name,
            "total_pages": max(pages.keys()) if pages else 0,
            "total_chunks": len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "doc_summary": doc_meta.get("doc_summary", ""),
            "doc_meta": doc_meta,
            "sections": sections,
            # Store chunk texts with context prefix for richer read_chunks output
            "_chunk_texts": {
                sec["chunk_id"]: self._make_context_prefix(doc_name, doc_meta, sec, chunk) + chunk.text
                for sec, chunk in zip(sections, chunks)
            },
        }

        with self._db_lock:
            _index_to_db(self._conn, index)
        return index

    def load_index(self, doc_name: str) -> dict | None:
        with self._db_lock:
            return _db_to_index(self._conn, doc_name)

    def list_indexed_docs(self) -> list[str]:
        """Return all doc names currently in the database."""
        with self._db_lock:
            return _db_list_docs(self._conn)

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
    # Agentic chunking
    # ------------------------------------------------------------------

    def _extract_segments(self, pages: dict[int, str]) -> list[Segment]:
        """
        Rule-based structural segmentation: split each page into headers,
        table blocks, and paragraphs. Preserves page number per segment.
        """
        segments: list[Segment] = []
        header_pat = re.compile(r"^#{1,4}\s+.+", re.MULTILINE)

        for page_num, page_text in sorted(pages.items()):
            lines = page_text.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]

                # Header line → its own segment
                if header_pat.match(line):
                    segments.append(Segment(line, page_num, "header"))
                    i += 1
                    continue

                # Table block: collect consecutive lines starting with |
                if line.strip().startswith("|"):
                    table_lines = []
                    while i < len(lines) and lines[i].strip().startswith("|"):
                        table_lines.append(lines[i])
                        i += 1
                    segments.append(Segment("\n".join(table_lines), page_num, "table"))
                    continue

                # Paragraph: collect until blank line or header/table
                para_lines = []
                while i < len(lines):
                    l = lines[i]
                    if not l.strip():
                        i += 1
                        break
                    if header_pat.match(l) or l.strip().startswith("|"):
                        break
                    para_lines.append(l)
                    i += 1
                if para_lines:
                    segments.append(Segment("\n".join(para_lines), page_num, "paragraph"))

        return segments

    def _agent_should_split(self, running_summary: str, segment_text: str) -> bool:
        """
        Ask the LLM whether the next segment should start a new chunk.
        Returns True = new_chunk, False = continue.
        """
        prompt = (
            "You are chunking a financial document for a RAG retrieval system.\n\n"
            f"Current chunk summary: {running_summary}\n\n"
            "Next segment to evaluate:\n"
            "---\n"
            f"{segment_text[:600]}\n"
            "---\n\n"
            "Should this segment START A NEW CHUNK or CONTINUE the current chunk?\n"
            "Answer with ONLY one word: \"continue\" or \"new_chunk\"\n\n"
            "Rules:\n"
            "- new_chunk: topic clearly changes (new financial statement, new fiscal year, "
            "new business segment, new footnote section, new MD&A topic)\n"
            "- continue: segment elaborates on the same topic, or is a table/data that "
            "belongs with the preceding header\n"
            "- Tables must stay with their header context"
        )
        response = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0,
            max_tokens=8,
        )
        return "new_chunk" in response.content.lower()

    def _update_running_summary(self, chunk_text: str) -> str:
        """Generate a one-sentence summary of a completed chunk for use as decision context."""
        prompt = (
            "Summarize in ONE sentence what this financial document chunk covers "
            "(company, time period, topic — e.g. 'AMD FY2022 revenue breakdown by segment'):\n\n"
            f"{chunk_text[:1000]}"
        )
        response = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0,
            max_tokens=80,
        )
        return response.content.strip()

    def _agentic_build_chunks(self, pages: dict[int, str]) -> list[Chunk]:
        """
        Agentic chunking: structural segmentation + LLM boundary decisions.

        Algorithm:
        1. Extract structural segments (headers, tables, paragraphs) from all pages.
        2. For each segment, ask LLM whether to continue current chunk or start a new one.
           - Skip LLM call if current chunk is below AGENTIC_MIN_TOKENS (too small to split).
           - Always flush if current chunk would exceed AGENTIC_MAX_TOKENS.
        3. On flush, generate a one-sentence summary for the next decision's context.
        """
        segments = self._extract_segments(pages)
        chunks: list[Chunk] = []

        current_segs: list[Segment] = []
        current_tokens = 0
        running_summary = ""

        def flush() -> None:
            nonlocal current_segs, current_tokens, running_summary
            if not current_segs:
                return
            text = "\n".join(s.text for s in current_segs).strip()
            if not text:
                current_segs = []
                current_tokens = 0
                return
            start_page = current_segs[0].page_num
            end_page = current_segs[-1].page_num
            idx = len(chunks)
            # Generate summary before appending so it describes this chunk
            summary = self._update_running_summary(text)
            chunks.append(Chunk(
                chunk_id=f"chunk_{idx:04d}",
                chunk_index=idx,
                text=text,
                page_range=(start_page, end_page),
                summary=summary,
            ))
            running_summary = summary
            current_segs = []
            current_tokens = 0

        for seg in segments:
            seg_tokens = self._estimate_tokens(seg.text)

            # Hard cap: flush before adding if it would overflow
            if current_tokens + seg_tokens > cfg.AGENTIC_MAX_TOKENS and current_segs:
                flush()

            # LLM boundary decision (only when chunk is large enough to be worth splitting)
            if current_segs and current_tokens >= cfg.AGENTIC_MIN_TOKENS:
                if self._agent_should_split(running_summary, seg.text):
                    flush()

            current_segs.append(seg)
            current_tokens += seg_tokens

        flush()
        return chunks

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    @staticmethod
    def _make_context_prefix(doc_name: str, doc_meta: dict, section: dict, chunk: "Chunk") -> str:
        """
        Build a one-line context prefix for a chunk using already-available metadata.
        No extra LLM call — pure string assembly.
        Prefers chunk.summary (agentic mode) over section_title for the topic description.
        Example: "[Context: AMD — FY2022 — AMD_2022_10K — Consolidated Statements of Operations]\n"
        """
        # Year from doc_name is more reliable than LLM-extracted time_period for multi-year docs
        import re as _re
        year_match = _re.search(r"\b(20\d{2})\b", doc_name)
        period = year_match.group(1) if year_match else doc_meta.get("time_period", "")

        topic = chunk.summary or section.get("section_title", "")
        parts = [
            doc_meta.get("primary_subject", ""),
            period,
            doc_name,
            topic,
        ]
        parts = [p for p in parts if p]
        if not parts:
            return ""
        return f"[Context: {' — '.join(parts)}]\n"

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

    def _build_doc_metadata(self, doc_name: str, sections: list[dict]) -> dict:
        """
        Derive document-level metadata from the full set of indexed chunk sections.
        Aggregates key_topics, entities, and section_titles across ALL chunks so that
        time_period and primary_subject reflect the entire document, not just the first pages.
        Falls back to a minimal default if no doc_fields are defined.
        """
        schema_str = self.index_schema.to_doc_json_schema_str()
        field_descs = self.index_schema.to_doc_field_descriptions()

        # Collect a compact digest of all chunk metadata (capped to avoid huge prompts)
        digest_lines: list[str] = []
        for sec in sections:
            title = sec.get("section_title", "")
            topics = sec.get("key_topics", [])
            entities = sec.get("entities", [])
            hints = sec.get("relevance_hints", "")
            parts = []
            if title:
                parts.append(f"Title: {title}")
            if topics:
                parts.append(f"Topics: {', '.join(topics)}")
            if entities:
                parts.append(f"Entities: {', '.join(entities)}")
            if hints:
                parts.append(f"Hints: {hints}")
            if parts:
                digest_lines.append(" | ".join(parts))

        # Cap digest at ~4000 chars to stay within prompt budget
        digest = "\n".join(digest_lines)
        if len(digest) > 4000:
            digest = digest[:4000] + "\n... (truncated)"

        prompt = f"""Analyze the metadata digest of ALL chunks in this document and extract document-level metadata.

Document name: {doc_name}

--- CHUNK METADATA DIGEST (all {len(sections)} chunks) ---
{digest}
--- END ---

Field descriptions:
{field_descs}

Respond with ONLY a JSON object matching this exact schema (no markdown fences, no explanation):
{schema_str}

Rules:
- Infer primary_subject and time_period from the full digest, not just the first few chunks.
- For time_period, list ALL fiscal years or periods present (e.g. "FY2021, FY2022" if both appear).
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
