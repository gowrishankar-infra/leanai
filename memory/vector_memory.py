"""
LeanAI · Phase 2 — Vector Episodic Memory
Replaces Phase 1's keyword search with true semantic vector similarity.

Phase 1: "fox dog" finds entries containing those exact words
Phase 2: "canine animals" finds entries about dogs and foxes
         even if those words never appear — because the MEANING is similar

How it works:
  1. Every conversation turn is embedded into a 384-dim vector
  2. Stored in ChromaDB (runs fully local, no cloud)
  3. Queries are also embedded, then nearest vectors are returned
  4. Nearest = most semantically similar, not just keyword matching

This is what makes the AI feel like it actually remembers context,
not just ctrl+F searching through old conversations.
"""

import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class MemoryEntry:
    id: str
    content: str
    role: str               # "user" | "assistant" | "fact" | "procedure"
    timestamp: float
    importance: float       # 0.0-1.0 — higher = retrieved more often
    access_count: int = 0
    tags: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class VectorEpisodicMemory:
    """
    Semantic vector memory using ChromaDB + sentence-transformers.
    Falls back to keyword search if ChromaDB unavailable.
    """

    COLLECTION_NAME = "leanai_episodic"
    EMBED_MODEL     = "all-MiniLM-L6-v2"  # 80MB, runs on CPU, 384-dim vectors
    MAX_RESULTS     = 5

    def __init__(self, storage_path: str = "~/.leanai/vector_memory"):
        self.path = Path(storage_path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self._chroma  = None
        self._embedder = None
        self._collection = None
        self._fallback: dict = {}
        self._init_backend()

    def _init_backend(self):
        """Try ChromaDB + sentence-transformers, fall back to keyword search."""
        try:
            import chromadb
            from chromadb.config import Settings
            self._chroma = chromadb.PersistentClient(
                path=str(self.path),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._chroma.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._load_embedder()
            print(f"[Memory] Vector store ready — {self._collection.count()} entries")
        except Exception as e:
            print(f"[Memory] ChromaDB unavailable ({e}) — using keyword fallback")
            self._load_fallback()

    def _load_embedder(self):
        """Load sentence-transformer embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.EMBED_MODEL)
            print(f"[Memory] Embedder loaded: {self.EMBED_MODEL}")
        except Exception as e:
            print(f"[Memory] Embedder unavailable: {e}")

    def store(
        self,
        content: str,
        role: str = "conversation",
        importance: float = 0.5,
        tags: list = None,
        metadata: dict = None,
    ) -> str:
        """Store a memory entry. Returns its ID."""
        entry_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:16]
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            role=role,
            timestamp=time.time(),
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
        )

        if self._collection and self._embedder:
            self._store_vector(entry)
        else:
            self._store_fallback(entry)

        return entry_id

    def search(self, query: str, top_k: int = None) -> list[MemoryEntry]:
        """Find most semantically relevant memories for a query."""
        k = top_k or self.MAX_RESULTS

        if self._collection and self._embedder:
            return self._search_vector(query, k)
        else:
            return self._search_fallback(query, k)

    def store_fact(self, fact: str, importance: float = 0.8) -> str:
        """Store a standalone fact with high importance."""
        return self.store(fact, role="fact", importance=importance,
                          tags=["fact"])

    def store_exchange(self, user_msg: str, ai_response: str):
        """Store a complete conversation exchange."""
        combined = f"User: {user_msg}\nAssistant: {ai_response[:400]}"
        importance = self._estimate_importance(user_msg, ai_response)
        self.store(combined, role="conversation", importance=importance,
                   tags=["exchange"])

    def count(self) -> int:
        if self._collection:
            return self._collection.count()
        return len(self._fallback)

    def recent(self, n: int = 5) -> list[MemoryEntry]:
        """Return n most recently stored entries."""
        if self._fallback:
            entries = sorted(self._fallback.values(),
                             key=lambda e: e.timestamp, reverse=True)
            return entries[:n]
        return []

    # ── Vector backend ─────────────────────────────────────────────────

    def _store_vector(self, entry: MemoryEntry):
        try:
            embedding = self._embedder.encode(entry.content).tolist()
            self._collection.upsert(
                ids=[entry.id],
                embeddings=[embedding],
                documents=[entry.content],
                metadatas=[{
                    "role": entry.role,
                    "timestamp": entry.timestamp,
                    "importance": entry.importance,
                    "tags": json.dumps(entry.tags),
                }],
            )
        except Exception as e:
            self._store_fallback(entry)

    def _search_vector(self, query: str, k: int) -> list[MemoryEntry]:
        try:
            if self._collection.count() == 0:
                return []
            query_embedding = self._embedder.encode(query).tolist()
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            entries = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                # Only return if similarity > threshold (distance < 0.8)
                if distance < 0.8:
                    entry = MemoryEntry(
                        id=results["ids"][0][i],
                        content=doc,
                        role=meta.get("role", "conversation"),
                        timestamp=meta.get("timestamp", 0),
                        importance=meta.get("importance", 0.5),
                        tags=json.loads(meta.get("tags", "[]")),
                    )
                    entries.append(entry)
            return entries
        except Exception as e:
            return self._search_fallback(query, k)

    # ── Keyword fallback ───────────────────────────────────────────────

    def _store_fallback(self, entry: MemoryEntry):
        self._fallback[entry.id] = entry
        self._save_fallback()

    def _search_fallback(self, query: str, k: int) -> list[MemoryEntry]:
        query_words = set(query.lower().split())
        scored = []
        for entry in self._fallback.values():
            entry_words = set(entry.content.lower().split())
            overlap = len(query_words & entry_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append((score * entry.importance, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    def _load_fallback(self):
        fb_file = self.path / "fallback.json"
        if fb_file.exists():
            try:
                data = json.loads(fb_file.read_text())
                self._fallback = {k: MemoryEntry(**v) for k, v in data.items()}
            except Exception:
                self._fallback = {}

    def _save_fallback(self):
        fb_file = self.path / "fallback.json"
        from dataclasses import asdict
        data = {k: asdict(v) for k, v in self._fallback.items()}
        fb_file.write_text(json.dumps(data, indent=2))

    def _estimate_importance(self, user_msg: str, ai_response: str) -> float:
        """Estimate how important this exchange is to remember."""
        score = 0.4  # base

        # Long responses tend to be more informative
        if len(ai_response) > 300:
            score += 0.1

        # Questions are important
        if "?" in user_msg:
            score += 0.1

        # Personal information is very important
        personal_signals = [
            "my name", "i am", "i work", "i live", "i have",
            "i like", "i want", "i need", "remember",
        ]
        if any(s in user_msg.lower() for s in personal_signals):
            score += 0.3

        return min(1.0, score)

    @property
    def backend(self) -> str:
        if self._collection and self._embedder:
            return "chromadb+vectors"
        return "keyword_fallback"
