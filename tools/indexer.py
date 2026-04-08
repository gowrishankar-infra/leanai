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
    ):
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._embedder   = embedder
        self._collection = None
        self._chroma     = None
        self._stats_file = self.storage_path / "stats.json"
        self._file_hashes: dict = {}
        self._hashes_file = self.storage_path / "file_hashes.json"
        self._load_hashes()
        self._init_collection()

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
        Semantic search across the indexed codebase.

        Returns list of CodeChunk-like dicts, sorted by relevance.
        """
        if not self._collection or not self._embedder:
            return self._keyword_search(query, top_k)

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
        """Split Python file by function and class definitions."""
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
