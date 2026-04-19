"""
LeanAI · Phase 4c — Project Indexer
The Copilot killer.

Point LeanAI at any directory. It:
  1. Walks every code file recursively
  2. Chunks each file into semantic units (functions, classes, blocks)
  3. Embeds each chunk into ChromaDB vector store
  4. Enables semantic search across your entire codebase

Then you can ask:
  "find all functions that touch the database"
  "what does the auth middleware do"
  "show me how errors are handled"
  "find code similar to this function"

Supports: .py .js .ts .go .rs .java .cpp .c .cs .rb .php .sh .sql .md

GitHub Copilot only sees your current file.
LeanAI sees your entire project — forever, offline, free.
"""

import os
import re
import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# File types to index
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".cpp", ".c",
    ".cs", ".rb", ".php", ".sh", ".bash",
    ".sql", ".md", ".yaml", ".yml", ".toml",
    ".json", ".html", ".css", ".dockerfile",
}

# Files and dirs to always skip
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".env", "dist", "build", ".next", ".nuxt", "target",
    ".pytest_cache", ".mypy_cache", "coverage", ".idea",
    ".vscode", "eggs", ".eggs",
}

SKIP_FILES = {
    ".gitignore", ".env", "package-lock.json",
    "yarn.lock", "poetry.lock", "Pipfile.lock",
}

# Chunk size limits
MAX_CHUNK_CHARS  = 1500
MIN_CHUNK_CHARS  = 20
MAX_FILE_SIZE_MB = 2.0


@dataclass
class CodeChunk:
    """A semantic unit of code from a file."""
    id: str
    file_path: str
    relative_path: str
    language: str
    chunk_type: str       # "function" | "class" | "block" | "whole_file"
    name: str             # function/class name if detected
    content: str
    start_line: int
    end_line: int
    project_root: str
    indexed_at: float = field(default_factory=time.time)


@dataclass
class IndexStats:
    project_root: str
    total_files: int
    indexed_files: int
    skipped_files: int
    total_chunks: int
    languages: dict
    index_time_s: float
    last_updated: float


class ProjectIndexer:
    """
    Indexes an entire codebase into ChromaDB for semantic search.
    """

    COLLECTION_NAME = "leanai_project_index"

    def __init__(
        self,
        storage_path: str = "~/.leanai/project_index",
        embedder=None,
        auto_load_embedder: bool = True,
        brain=None,
    ):
        """
        storage_path:        where ChromaDB persists its collection
        embedder:            optional pre-loaded SentenceTransformer-like
                             object (must expose `.encode(str) -> vector`).
                             If provided, takes priority over auto_load.
        auto_load_embedder:  if True AND no embedder is provided, load
                             all-MiniLM-L6-v2 on construction. Without
                             an embedder the indexer silently degrades
                             to keyword-search with flat relevance
                             scores — a confusing UX. Default on.
        brain:               optional ProjectBrain instance. When provided,
                             Python chunking uses the AST graph (function
                             boundaries, docstrings, call relationships)
                             instead of regex splitting. Substantially
                             improves semantic retrieval quality.
                             Can be set later via set_brain().
        """
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._embedder   = embedder
        self._collection = None
        self._chroma     = None
        self._stats_file = self.storage_path / "stats.json"
        self._file_hashes: dict = {}
        self._hashes_file = self.storage_path / "file_hashes.json"
        self._brain      = brain
        self._load_hashes()
        self._init_collection()
        # M6 fix — auto-load the same embedder the memory store uses,
        # so that search() actually runs semantic retrieval (not the
        # flat-0.5 keyword fallback). Load AFTER collection init so we
        # know whether we'll need it.
        if self._embedder is None and auto_load_embedder:
            self._auto_load_embedder()
        # Detect legacy index built without a real embedder. If the
        # existing collection was populated with simple (TF-IDF) vectors
        # but we now have a real embedder, the vectors are incompatible
        # and search results will be noise. Flag this for rebuild.
        self._embeddings_are_stale = self._detect_stale_embeddings()

    def set_brain(self, brain):
        """Wire a ProjectBrain into the indexer AFTER construction.
        Enables AST-grounded chunking for Python files on the next
        index_project() call. If brain changes structure (e.g. a
        project re-scan happens), call this again.
        """
        self._brain = brain
        # Invalidate the hybrid retriever so it rebuilds with the new brain
        self._hybrid_retriever = None
        # Re-detect staleness now that we have the brain — the AST-
        # signature check needs the brain to know we WANT AST chunks.
        # If an old regex-chunked index exists, this flips
        # _embeddings_are_stale to True so the next index_project()
        # triggers a clean rebuild.
        self._embeddings_are_stale = self._detect_stale_embeddings()

    def set_git_intel(self, git_intel):
        """Wire GitIntel for rerank 'recently modified' boost."""
        self._git_intel = git_intel
        self._hybrid_retriever = None

    def _get_hybrid(self):
        """Lazy-build HybridRetriever on first use. Rebuilds when brain
        changes (e.g. /brain . re-scan).
        """
        if getattr(self, '_hybrid_retriever', None) is not None:
            return self._hybrid_retriever
        if getattr(self, '_brain', None) is None:
            return None
        try:
            from core.retrieval import HybridRetriever
            self._hybrid_retriever = HybridRetriever(
                indexer=self,
                brain=self._brain,
                git_intel=getattr(self, '_git_intel', None),
            )
            return self._hybrid_retriever
        except Exception as e:
            print(f"[Indexer] HybridRetriever unavailable ({e}) — "
                  f"using semantic-only search")
            self._hybrid_retriever = None
            return None

    def _auto_load_embedder(self):
        """Try to load sentence-transformers all-MiniLM-L6-v2.
        Mirrors vector_memory._load_embedder. Shares weights via HF cache
        so no second download if memory already loaded it.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print(f"[Indexer] Semantic embedder loaded (all-MiniLM-L6-v2)")
        except Exception as e:
            print(f"[Indexer] Semantic embedder unavailable ({e}) — "
                  f"search will use keyword fallback with flat scores")
            self._embedder = None

    def _detect_stale_embeddings(self) -> bool:
        """Peek at one stored embedding/document to see if the index is
        stale for any of these reasons:

        1. Stored embedding dimensions don't match what the current
           embedder produces (legacy TF-IDF fallback from pre-M6).
        2. A Python chunk exists but its content DOESN'T contain the
           AST-chunker signature '# LeanAI chunk ·' — meaning the index
           was built with the old regex chunker and needs a rebuild to
           unlock the M6 AST-grounded retrieval quality.

        Returns True if stale. False if fresh or unknown.
        """
        if not self._collection:
            return False
        try:
            count = self._collection.count()
        except Exception:
            return False
        if count == 0:
            return False

        # Check 1 — embedding dimension mismatch
        if self._embedder:
            try:
                sample = self._collection.get(include=["embeddings"], limit=1)
                embs = sample.get("embeddings")
                # ChromaDB returns embeddings as numpy arrays; use len()
                # + None-check with explicit 'is None' so we don't trip
                # numpy's ambiguous-truth-value error.
                if embs is not None and len(embs) > 0 and embs[0] is not None:
                    stored_dim = len(embs[0])
                    probe_vec = self._embedder.encode("dimension probe").tolist()
                    if stored_dim != len(probe_vec):
                        print(f"[Indexer] Stale: embedding dim mismatch "
                              f"(stored={stored_dim}, current={len(probe_vec)})")
                        return True
            except Exception as e:
                # Note the error but continue — we can still do check 2
                pass

        # Check 2 — AST chunker signature absent from Python chunks
        # If any Python chunk lacks the '# LeanAI chunk ·' header,
        # it was built with the old regex chunker. The AST chunker
        # prefixes EVERY chunk with that signature.
        if self._brain is not None:
            try:
                py_sample = self._collection.get(
                    where={"language": "python"},
                    include=["documents"], limit=3)
                py_docs = py_sample.get("documents") or []
                if py_docs:
                    has_ast_signature = any(
                        '# LeanAI chunk ·' in d for d in py_docs
                    )
                    if not has_ast_signature:
                        print("[Indexer] Stale: Python chunks lack AST-chunker "
                              "signature — rebuilding for M6 quality")
                        return True
            except Exception:
                pass

        return False

    # ══════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════

    def index_project(self, project_root: str, force: bool = False) -> IndexStats:
        """
        Index an entire project directory.
        Only re-indexes files that have changed since last run.

        Args:
            project_root: Path to the project directory
            force: Re-index all files even if unchanged
        """
        root = Path(project_root).resolve()
        if not root.exists():
            raise ValueError(f"Project root does not exist: {root}")

        # M6 fix — if we loaded a real embedder but the existing collection
        # was built with the legacy TF-IDF fallback, nuke it and start
        # fresh. Without this, new embeddings can't sit alongside old
        # incompatible-dim vectors and search returns garbage.
        if self._embeddings_are_stale:
            print(f"[Indexer] Stale index detected (built without semantic "
                  f"embedder) — rebuilding for accurate search...")
            try:
                self._chroma.delete_collection(self.COLLECTION_NAME)
                self._collection = self._chroma.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                self._file_hashes = {}
                self._save_hashes()
                self._embeddings_are_stale = False
                force = True  # must re-chunk everything
            except Exception as e:
                print(f"[Indexer] Rebuild failed ({e}) — keeping legacy index")

        print(f"[Indexer] Scanning: {root}")
        start_time = time.time()

        files = self._walk_project(root)
        print(f"[Indexer] Found {len(files)} code files")

        indexed = skipped = 0
        all_chunks = []
        languages: dict = {}

        for file_path in files:
            lang = self._detect_language(file_path)
            languages[lang] = languages.get(lang, 0) + 1

            # Check if file changed
            file_hash = self._hash_file(file_path)
            rel_path  = str(file_path.relative_to(root))

            if not force and self._file_hashes.get(rel_path) == file_hash:
                skipped += 1
                continue

            # Chunk the file
            chunks = self._chunk_file(file_path, root, lang)
            if chunks:
                all_chunks.extend(chunks)
                self._file_hashes[rel_path] = file_hash
                indexed += 1

        # Store chunks in vector store
        if all_chunks:
            self._store_chunks(all_chunks)

        self._save_hashes()

        elapsed = time.time() - start_time
        stats = IndexStats(
            project_root=str(root),
            total_files=len(files),
            indexed_files=indexed,
            skipped_files=skipped,
            total_chunks=self.count(),
            languages=languages,
            index_time_s=round(elapsed, 2),
            last_updated=time.time(),
        )
        self._save_stats(stats)

        print(f"[Indexer] Done: {indexed} indexed, {skipped} unchanged, "
              f"{self.count()} total chunks, {elapsed:.1f}s")
        return stats

    def search(self, query: str, top_k: int = 5, language: str = None) -> list:
        """
        Retrieve chunks matching a query.

        Routing:
          - If a ProjectBrain is wired AND the HybridRetriever loads
            successfully → route through hybrid retrieval (semantic +
            BM25 + graph + RRF fusion + rerank). Best quality.
          - Otherwise → fall back to semantic-only search via
            _semantic_search(). Same behavior as before M6.
          - If neither embedder nor collection → keyword fallback with
            flat 0.5 relevance. Unchanged legacy behavior.

        Returns list of dicts with keys: content, relative_path, name,
        chunk_type, start_line, end_line, relevance.
        """
        # M6 fix — one-time diagnostic so it's obvious what search mode
        # is active. `_diag_logged` guards against spamming every call.
        if not getattr(self, '_diag_logged', False):
            self._diag_logged = True
            if self._collection and self._embedder:
                if getattr(self, '_brain', None) is not None:
                    mode = "hybrid (semantic + BM25 + graph)"
                else:
                    mode = "semantic (embeddings only)"
            else:
                mode = "keyword fallback (flat 0.5 relevance)"
            try:
                chunk_count = self._collection.count() if self._collection else 0
            except Exception:
                chunk_count = 0
            print(f"[Indexer] Search mode: {mode} · {chunk_count} chunks indexed")

        if not self._collection or not self._embedder:
            return self._keyword_search(query, top_k)

        # Try hybrid retrieval first. If it fails or isn't available,
        # fall through to semantic-only (legacy) search.
        hybrid = self._get_hybrid()
        if hybrid is not None:
            try:
                return hybrid.search_legacy(query, top_k=top_k)
            except Exception as e:
                # Never block search on hybrid failure
                print(f"[Indexer] Hybrid retrieval failed ({e}) — "
                      f"using semantic-only")

        return self._semantic_search(query, top_k, language)

    def _semantic_search(self, query: str, top_k: int = 5, language: str = None) -> list:
        """Semantic-only search — the original indexer.search behavior,
        preserved for fallback and as the primitive that HybridRetriever
        calls internally (via its own indexer.search path to avoid
        recursion, this method sits under the hybrid layer).
        """
        try:
            count = self._collection.count()
            if count == 0:
                return []

            embedding = self._embedder.encode(query).tolist()
            where = {"language": language} if language else None

            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, count),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            chunks = []
            for i, doc in enumerate(results["documents"][0]):
                meta     = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                if distance < 0.9:
                    chunks.append({
                        "content":       doc,
                        "file_path":     meta.get("file_path", ""),
                        "relative_path": meta.get("relative_path", ""),
                        "language":      meta.get("language", ""),
                        "chunk_type":    meta.get("chunk_type", ""),
                        "name":          meta.get("name", ""),
                        "start_line":    meta.get("start_line", 0),
                        "end_line":      meta.get("end_line", 0),
                        "relevance":     round(1 - distance, 3),
                    })
            return chunks

        except Exception as e:
            print(f"[Indexer] Search error: {e}")
            return self._keyword_search(query, top_k)

    def get_file(self, relative_path: str) -> Optional[str]:
        """Get all indexed content for a specific file."""
        if not self._collection:
            return None
        try:
            results = self._collection.get(
                where={"relative_path": relative_path},
                include=["documents"],
            )
            if results["documents"]:
                return "\n\n".join(results["documents"])
        except Exception:
            pass
        return None

    def count(self) -> int:
        if self._collection:
            try:
                return self._collection.count()
            except Exception:
                pass
        return 0

    def clear(self):
        """Remove all indexed data."""
        if self._chroma and self._collection:
            try:
                self._chroma.delete_collection(self.COLLECTION_NAME)
                self._collection = None
                self._file_hashes = {}
                self._save_hashes()
                self._init_collection()
                print("[Indexer] Index cleared.")
            except Exception as e:
                print(f"[Indexer] Clear error: {e}")

    def format_search_results(self, chunks: list, max_chars: int = 2000) -> str:
        """Format search results for injection into AI prompt."""
        if not chunks:
            return ""

        parts = []
        total_chars = 0

        for chunk in chunks:
            rel_path = chunk.get("relative_path", "unknown")
            name     = chunk.get("name", "")
            lang     = chunk.get("language", "")
            line     = chunk.get("start_line", 0)
            content  = chunk.get("content", "")
            relevance = chunk.get("relevance", 0)

            header = f"# {rel_path}"
            if name:
                header += f" — {name}"
            if line:
                header += f" (line {line})"
            header += f" [{relevance:.0%} relevant]"

            entry = f"{header}\n```{lang}\n{content[:600]}\n```"

            if total_chars + len(entry) > max_chars:
                break

            parts.append(entry)
            total_chars += len(entry)

        return "\n\n".join(parts)

    def stats(self) -> dict:
        stats_file = self._stats_file
        if stats_file.exists():
            try:
                return json.loads(stats_file.read_text())
            except Exception:
                pass
        return {
            "total_chunks": self.count(),
            "indexed_files": len(self._file_hashes),
        }

    # ══════════════════════════════════════════════════
    # File Walking
    # ══════════════════════════════════════════════════

    def _walk_project(self, root: Path) -> list:
        """Walk directory tree and return list of indexable files."""
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden and excluded directories
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS and not d.startswith(".")
            ]

            for filename in filenames:
                if filename in SKIP_FILES:
                    continue
                file_path = Path(dirpath) / filename
                ext = file_path.suffix.lower()

                # Check extension
                if ext not in SUPPORTED_EXTENSIONS and filename.lower() != "dockerfile":
                    continue

                # Check file size
                try:
                    size_mb = file_path.stat().st_size / 1e6
                    if size_mb > MAX_FILE_SIZE_MB:
                        continue
                except Exception:
                    continue

                files.append(file_path)

        return sorted(files)

    # ══════════════════════════════════════════════════
    # File Chunking
    # ══════════════════════════════════════════════════

    def _chunk_file(self, file_path: Path, root: Path, language: str) -> list:
        """
        Split a file into semantic chunks.
        For Python/JS: split by function/class definitions.
        For others: split by logical blocks or size.
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        if not content.strip():
            return []

        rel_path = str(file_path.relative_to(root))
        chunks   = []

        if language == "python":
            chunks = self._chunk_python(content, file_path, rel_path, root)
        elif language in ("javascript", "typescript"):
            chunks = self._chunk_js(content, file_path, rel_path, root)
        else:
            chunks = self._chunk_generic(content, file_path, rel_path, root, language)

        return [c for c in chunks if len(c.content) >= MIN_CHUNK_CHARS]

    def _chunk_python(self, content: str, file_path: Path, rel_path: str, root: Path) -> list:
        """Chunk a Python file. If a ProjectBrain is wired, use its AST
        graph for qualified-name boundaries, docstring extraction, and
        call relationships — substantially better semantic retrieval
        than regex boundaries. Falls back to the regex chunker if no
        brain is available or analysis is incomplete.
        """
        if self._brain is not None:
            try:
                ast_chunks = self._chunk_python_ast(content, file_path, rel_path, root)
                # Only use AST chunks if we got a meaningful amount — a
                # file with zero parseable functions (module-level code
                # only) should fall back to the generic chunker.
                if ast_chunks:
                    return ast_chunks
            except Exception as e:
                # Any failure → safe fallback. Never block indexing.
                pass
        return self._chunk_python_regex(content, file_path, rel_path, root)

    def _chunk_python_ast(self, content: str, file_path: Path, rel_path: str, root: Path) -> list:
        """
        AST-grounded Python chunking — the M6 quality lead.

        Each chunk is a single function, method, class (header+body),
        or module-level block. The chunk TEXT is enriched with:
          - TYPE (function|method|class|module), CLASS, NAME, LINES, FILE
          - DOCSTRING (parsed, cleaned)
          - CALLS: qualified names of functions this one calls
          - CALLED_BY: qualified names of functions that call this one
          - SOURCE: the raw code

        This structured header makes the embedding vector represent
        "a function named X in class Y that does Z and is called by W"
        — which matches natural-language queries ("how does caching
        work") far better than the raw code alone.

        Returns list of CodeChunk instances. Never returns empty; falls
        back to the caller's regex path via exception if brain info
        looks incomplete (e.g. file not yet analyzed).
        """
        analysis = self._brain._file_analyses.get(rel_path)
        if not analysis or analysis.error:
            # Brain doesn't know this file yet (new scan pending). Let
            # the caller fall back to regex.
            raise ValueError("brain has no analysis for this file")

        lines = content.split("\n")
        total_lines = len(lines)
        chunks = []

        # Build called_by index once for this file by scanning the brain's
        # dependency graph. For each function in this file, find who
        # calls it across the project.
        called_by_index: dict = {}
        try:
            graph = self._brain.graph
            # Build reverse adjacency (target -> list of source qnames)
            for src_id, targets in graph._adjacency.items():
                src_qname = src_id.rsplit(':', 1)[-1] if ':' in src_id else src_id
                src_file = src_id.rsplit(':', 1)[0] if ':' in src_id else ''
                for tgt_id in targets:
                    if tgt_id.startswith('__'):
                        continue
                    # Only record cross-file callers — same-file calls are
                    # already visible from the chunk itself.
                    if src_file and src_file != str(file_path):
                        tgt_qname = tgt_id.rsplit(':', 1)[-1] if ':' in tgt_id else tgt_id
                        called_by_index.setdefault(tgt_qname, set()).add(src_qname)
        except Exception:
            called_by_index = {}

        # Track which line ranges are covered by function/method chunks,
        # so the remainder can be collected as module-level code.
        covered_ranges = []

        # ─── Functions & methods ───────────────────────────────────
        for func in analysis.functions:
            ls = max(0, func.line_start - 1)
            le = min(total_lines, func.line_end)
            if le <= ls:
                continue
            covered_ranges.append((ls, le))

            source = "\n".join(lines[ls:le])
            # Header facts for the embedder
            kind = "method" if func.is_method else "function"
            qname = func.qualified_name
            doc = (func.docstring or "").strip()
            if len(doc) > 300:
                doc = doc[:300] + "..."

            # Outgoing calls (qualified where possible)
            calls_list = []
            seen = set()
            for c in func.calls[:12]:  # cap for header size
                if c not in seen and c:
                    seen.add(c)
                    calls_list.append(c)
            calls_str = ", ".join(calls_list) if calls_list else "(none)"

            # Incoming callers (from graph index)
            incoming = called_by_index.get(qname, set())
            if not incoming and func.is_method:
                # Try class-method lookup too
                incoming = called_by_index.get(func.name, set())
            callers_list = sorted(incoming)[:8]
            callers_str = ", ".join(callers_list) if callers_list else "(none outside this file)"

            header_lines = [
                f"# LeanAI chunk · {kind} · {qname}",
                f"# File: {rel_path} · Lines {func.line_start}-{func.line_end}",
            ]
            if func.class_name:
                header_lines.append(f"# Class: {func.class_name}")
            if doc:
                header_lines.append(f"# Docstring: {doc}")
            header_lines.append(f"# Calls: {calls_str}")
            header_lines.append(f"# Called by: {callers_str}")
            header_lines.append(f"# Complexity: {func.complexity}")
            if func.decorators:
                header_lines.append(f"# Decorators: {', '.join(func.decorators[:6])}")
            header = "\n".join(header_lines)

            chunk_content = header + "\n\n" + source
            # Budget check: MiniLM truncates at ~256 tokens (~1000 chars).
            # If the source is much larger, keep the header + first 800
            # chars of the source so the semantic signal stays in-window.
            if len(chunk_content) > 1600:
                truncated_source = source[:1200] + "\n# ... [truncated for embedding]"
                chunk_content = header + "\n\n" + truncated_source

            chunks.append(CodeChunk(
                id=self._make_id(rel_path, func.line_start, qname),
                file_path=str(file_path),
                relative_path=rel_path,
                language="python",
                chunk_type=kind,
                name=qname,
                content=chunk_content,
                start_line=func.line_start,
                end_line=func.line_end,
                project_root=str(root),
            ))

        # ─── Classes (with method bodies NOT re-included) ──────────
        # Represent each class with its header, docstring, bases, and
        # method-list — gives semantic coverage for queries like
        # "where is the SessionStore class defined".
        for cls in analysis.classes:
            ls = max(0, cls.line_start - 1)
            le = min(total_lines, cls.line_end)
            if le <= ls:
                continue

            # Class-header text: take the line range but ONLY the header
            # + non-method code. We skip the method bodies because they
            # already have their own chunks.
            class_source_lines = lines[ls:min(ls + 30, le)]  # first ~30 lines
            class_header_source = "\n".join(class_source_lines)

            doc = (cls.docstring or "").strip()
            if len(doc) > 300:
                doc = doc[:300] + "..."
            bases_str = ", ".join(cls.bases) if cls.bases else "(none)"
            methods_str = ", ".join(cls.methods[:20]) if cls.methods else "(none)"

            header_lines = [
                f"# LeanAI chunk · class · {cls.name}",
                f"# File: {rel_path} · Lines {cls.line_start}-{cls.line_end}",
                f"# Bases: {bases_str}",
                f"# Methods: {methods_str}",
            ]
            if doc:
                header_lines.append(f"# Docstring: {doc}")
            if cls.decorators:
                header_lines.append(f"# Decorators: {', '.join(cls.decorators[:6])}")
            header = "\n".join(header_lines)

            chunks.append(CodeChunk(
                id=self._make_id(rel_path, cls.line_start, cls.name + "__class"),
                file_path=str(file_path),
                relative_path=rel_path,
                language="python",
                chunk_type="class",
                name=cls.name,
                content=header + "\n\n" + class_header_source,
                start_line=cls.line_start,
                end_line=cls.line_end,
                project_root=str(root),
            ))

        # ─── Module-level code ─────────────────────────────────────
        # Imports + top-level constants + module docstring + anything
        # between function definitions. Collect contiguous uncovered
        # ranges and emit them as module chunks.
        covered_ranges.sort()
        uncovered_blocks = []
        cur = 0
        for (rs, re_) in covered_ranges:
            if rs > cur:
                uncovered_blocks.append((cur, rs))
            cur = max(cur, re_)
        if cur < total_lines:
            uncovered_blocks.append((cur, total_lines))

        for (bs, be) in uncovered_blocks:
            # Strip pure-whitespace blocks
            text = "\n".join(lines[bs:be]).strip()
            if len(text) < MIN_CHUNK_CHARS:
                continue

            # Identify what this module chunk roughly contains
            has_imports = any(l.strip().startswith(('import ', 'from '))
                              for l in lines[bs:be])
            has_const = any(re.match(r'^[A-Z_][A-Z0-9_]+\s*=', l)
                            for l in lines[bs:be])
            tags = []
            if has_imports:
                tags.append("imports")
            if has_const:
                tags.append("constants")
            if not tags and bs == 0:
                tags.append("module-docstring-or-header")
            if not tags:
                tags.append("module-level-code")
            tag_str = " + ".join(tags)

            header = (
                f"# LeanAI chunk · module · {rel_path}\n"
                f"# Lines {bs+1}-{be}\n"
                f"# Contains: {tag_str}"
            )
            # Cap module chunks to avoid huge single chunks
            if len(text) > 1200:
                text_clipped = text[:1200] + "\n# ... [truncated]"
            else:
                text_clipped = text

            chunks.append(CodeChunk(
                id=self._make_id(rel_path, bs + 1, f"module_block_{bs}"),
                file_path=str(file_path),
                relative_path=rel_path,
                language="python",
                chunk_type="module",
                name=f"{Path(rel_path).stem}:module",
                content=header + "\n\n" + text_clipped,
                start_line=bs + 1,
                end_line=be,
                project_root=str(root),
            ))

        return chunks

    def _chunk_python_regex(self, content: str, file_path: Path, rel_path: str, root: Path) -> list:
        """Legacy regex-boundary Python chunker. Used when no brain
        is wired OR when AST chunking fails for some reason. Preserves
        pre-M6 behavior exactly.
        """
        chunks = []
        lines  = content.split("\n")

        # Find top-level def/class boundaries
        boundaries = []  # (start_line, name, type)
        for i, line in enumerate(lines):
            m = re.match(r"^(def|class)\s+(\w+)", line)
            if m:
                boundaries.append((i, m.group(2), m.group(1)))

        if not boundaries:
            return self._chunk_generic(content, file_path, rel_path, root, "python")

        # Add end boundary
        boundaries.append((len(lines), "", ""))

        for i, (start, name, chunk_type) in enumerate(boundaries[:-1]):
            end = boundaries[i + 1][0]
            chunk_lines = lines[start:end]
            chunk_content = "\n".join(chunk_lines).strip()

            if len(chunk_content) < MIN_CHUNK_CHARS:
                continue

            chunk_id = self._make_id(rel_path, start, name)
            chunks.append(CodeChunk(
                id=chunk_id,
                file_path=str(file_path),
                relative_path=rel_path,
                language="python",
                chunk_type="function" if chunk_type == "def" else "class",
                name=name,
                content=chunk_content[:MAX_CHUNK_CHARS],
                start_line=start + 1,
                end_line=end,
                project_root=str(root),
            ))

        return chunks

    def _chunk_js(self, content: str, file_path: Path, rel_path: str, root: Path) -> list:
        """Split JS/TS file by function definitions."""
        chunks = []
        lines  = content.split("\n")

        boundaries = []
        patterns = [
            r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)",
            r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(",
            r"^(?:export\s+)?class\s+(\w+)",
            r"^\s{0,2}(\w+)\s*[=:]\s*(?:async\s+)?\(",
        ]
        combined = re.compile("|".join(f"(?:{p})" for p in patterns))

        for i, line in enumerate(lines):
            m = combined.match(line)
            if m:
                name = next((g for g in m.groups() if g), "anonymous")
                boundaries.append((i, name))

        if not boundaries:
            return self._chunk_generic(content, file_path, rel_path, root, "javascript")

        boundaries.append((len(lines), ""))

        for i, (start, name) in enumerate(boundaries[:-1]):
            end = boundaries[i + 1][0]
            chunk_content = "\n".join(lines[start:end]).strip()
            if len(chunk_content) < MIN_CHUNK_CHARS:
                continue
            chunk_id = self._make_id(rel_path, start, name)
            chunks.append(CodeChunk(
                id=chunk_id,
                file_path=str(file_path),
                relative_path=rel_path,
                language="javascript",
                chunk_type="function",
                name=name,
                content=chunk_content[:MAX_CHUNK_CHARS],
                start_line=start + 1,
                end_line=end,
                project_root=str(root),
            ))

        return chunks

    def _chunk_generic(self, content: str, file_path: Path,
                        rel_path: str, root: Path, language: str) -> list:
        """Split any file into fixed-size chunks."""
        chunks = []
        lines  = content.split("\n")
        chunk_size = 50  # lines per chunk

        for i in range(0, len(lines), chunk_size):
            chunk_lines   = lines[i:i + chunk_size]
            chunk_content = "\n".join(chunk_lines).strip()
            if len(chunk_content) < MIN_CHUNK_CHARS:
                continue
            chunk_id = self._make_id(rel_path, i, "")
            chunks.append(CodeChunk(
                id=chunk_id,
                file_path=str(file_path),
                relative_path=rel_path,
                language=language,
                chunk_type="block",
                name="",
                content=chunk_content[:MAX_CHUNK_CHARS],
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines)),
                project_root=str(root),
            ))

        return chunks

    # ══════════════════════════════════════════════════
    # Storage
    # ══════════════════════════════════════════════════

    def _init_collection(self):
        try:
            import chromadb
            from chromadb.config import Settings
            self._chroma = chromadb.PersistentClient(
                path=str(self.storage_path),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._chroma.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[Indexer] Collection ready — {self._collection.count()} chunks")
        except Exception as e:
            print(f"[Indexer] ChromaDB unavailable: {e}")

    def _store_chunks(self, chunks: list):
        """Store chunks in ChromaDB with or without embeddings."""
        if not self._collection:
            return

        ids       = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.content)
            metadatas.append({
                "file_path":     chunk.file_path,
                "relative_path": chunk.relative_path,
                "language":      chunk.language,
                "chunk_type":    chunk.chunk_type,
                "name":          chunk.name,
                "start_line":    chunk.start_line,
                "end_line":      chunk.end_line,
                "project_root":  chunk.project_root,
            })
            if self._embedder:
                try:
                    emb = self._embedder.encode(chunk.content).tolist()
                    embeddings.append(emb)
                except Exception:
                    embeddings.append(None)

        # Store in batches of 100
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids  = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            try:
                batch_embs = (embeddings[i:i + batch_size]
                              if embeddings and all(
                                  e is not None for e in embeddings[i:i + batch_size])
                              else None)
                try:
                    if batch_embs:
                        self._collection.upsert(
                            ids=batch_ids, documents=batch_docs,
                            metadatas=batch_meta, embeddings=batch_embs)
                    else:
                        # Generate simple TF-IDF-style embeddings to avoid
                        # ChromaDB trying to download its own model
                        simple_embs = self._simple_embeddings(batch_docs)
                        self._collection.upsert(
                            ids=batch_ids, documents=batch_docs,
                            metadatas=batch_meta, embeddings=simple_embs)
                except Exception as e:
                    print(f"[Indexer] Store error: {e}")
            except Exception as e:
                print(f"[Indexer] Store batch error: {e}")

    def _simple_embeddings(self, texts: list) -> list:
        """
        Generate simple character-frequency embeddings.
        Used when no sentence-transformer is available.
        Dimension: 384 (matches all-MiniLM-L6-v2 for compatibility).
        """
        import math
        dim = 384
        result = []
        for text in texts:
            vec = [0.0] * dim
            text_lower = text.lower()
            # Character frequency features
            for i, char in enumerate(text_lower[:dim]):
                vec[i % dim] += ord(char) / 1000.0
            # Word features
            words = text_lower.split()
            for word in words[:50]:
                h = hash(word) % dim
                vec[h] += 1.0
            # Normalize
            magnitude = math.sqrt(sum(x*x for x in vec)) or 1.0
            vec = [x / magnitude for x in vec]
            result.append(vec)
        return result

    def _keyword_search(self, query: str, top_k: int) -> list:
        """Fallback keyword search when vector search unavailable."""
        if not self._collection:
            return []
        try:
            results = self._collection.get(include=["documents", "metadatas"])
            query_words = set(query.lower().split())
            scored = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                doc_words = set(doc.lower().split())
                overlap = len(query_words & doc_words)
                if overlap > 0:
                    scored.append((overlap, doc, meta))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [{"content": d, **m, "relevance": 0.5}
                    for _, d, m in scored[:top_k]]
        except Exception:
            return []

    # ══════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════

    def _detect_language(self, file_path: Path) -> str:
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".tsx": "typescript", ".jsx": "javascript", ".go": "go",
            ".rs": "rust", ".java": "java", ".cpp": "cpp", ".c": "c",
            ".cs": "csharp", ".rb": "ruby", ".php": "php",
            ".sh": "bash", ".bash": "bash", ".sql": "sql",
            ".md": "markdown", ".yaml": "yaml", ".yml": "yaml",
            ".toml": "toml", ".json": "json", ".html": "html",
            ".css": "css",
        }
        return ext_map.get(file_path.suffix.lower(), "text")

    def _hash_file(self, file_path: Path) -> str:
        try:
            content = file_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def _make_id(self, rel_path: str, line: int, name: str) -> str:
        raw = f"{rel_path}:{line}:{name}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _save_hashes(self):
        self._hashes_file.write_text(json.dumps(self._file_hashes, indent=2))

    def _load_hashes(self):
        if self._hashes_file.exists():
            try:
                self._file_hashes = json.loads(self._hashes_file.read_text())
            except Exception:
                self._file_hashes = {}

    def _save_stats(self, stats: IndexStats):
        self._stats_file.write_text(json.dumps(asdict(stats), indent=2))
