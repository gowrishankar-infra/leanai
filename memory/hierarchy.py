"""
LeanAI · Hierarchical Memory System
4 layers — exactly like human memory architecture.

L1 · Working memory    — current context window (fast, limited, volatile)
L2 · Episodic memory   — what happened (vector search, persistent)
L3 · Semantic memory   — what things mean (knowledge graph, persistent)
L4 · Procedural memory — how to do things (solved programs, persistent)

No other edge AI has this. Most have nothing. Some bolt on a vector store.
This is a full memory architecture that mirrors human cognition.
"""

import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class MemoryEntry:
    id: str
    content: str
    layer: str                    # "episodic" | "semantic" | "procedural"
    timestamp: float
    access_count: int = 0
    tags: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkingMemory:
    """
    L1: In-context working memory.
    Holds current conversation + active retrieved memories.
    Cleared between sessions (unless explicitly persisted).
    """
    messages: list = field(default_factory=list)
    active_context: list = field(default_factory=list)
    max_tokens: int = 4096
    current_tokens: int = 0

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        # Rough token estimate: 1 token ≈ 4 chars
        self.current_tokens += len(content) // 4

    def get_context_window(self, max_tokens: Optional[int] = None) -> list:
        """Return messages that fit within token budget."""
        limit = max_tokens or self.max_tokens
        result = []
        tokens_used = 0
        for msg in reversed(self.messages):
            msg_tokens = len(msg["content"]) // 4
            if tokens_used + msg_tokens > limit:
                break
            result.insert(0, msg)
            tokens_used += msg_tokens
        return result

    def clear(self):
        self.messages.clear()
        self.active_context.clear()
        self.current_tokens = 0


class EpisodicMemory:
    """
    L2: Episodic memory — what happened, when, in what context.
    Stores conversation turns and retrieved by semantic similarity.
    
    Phase 0: JSON file store with keyword search (no GPU needed)
    Phase 2: Replace with ChromaDB vector store for true semantic search
    """

    def __init__(self, storage_path: str = "~/.leanai/episodic"):
        self.path = Path(storage_path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, MemoryEntry] = {}
        self._load()

    def store(self, content: str, tags: list = None, metadata: dict = None) -> str:
        """Store a new episodic memory. Returns its ID."""
        entry_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:12]
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            layer="episodic",
            timestamp=time.time(),
            tags=tags or [],
            metadata=metadata or {},
        )
        self._index[entry_id] = entry
        self._save()
        return entry_id

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """
        Phase 0: keyword-based search.
        Phase 2: replace with vector similarity search.
        """
        query_words = set(query.lower().split())
        scored = []
        for entry in self._index.values():
            entry_words = set(entry.content.lower().split())
            overlap = len(query_words & entry_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:top_k]]

        # Update access counts
        for entry in results:
            entry.access_count += 1

        return results

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return n most recent memories."""
        sorted_entries = sorted(
            self._index.values(),
            key=lambda e: e.timestamp,
            reverse=True,
        )
        return sorted_entries[:n]

    def _save(self):
        index_file = self.path / "index.json"
        data = {k: asdict(v) for k, v in self._index.items()}
        with open(index_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        index_file = self.path / "index.json"
        if index_file.exists():
            with open(index_file) as f:
                data = json.load(f)
            self._index = {k: MemoryEntry(**v) for k, v in data.items()}


class SemanticMemory:
    """
    L3: Semantic memory — what things mean and how they relate.
    A persistent knowledge graph of entities and relationships.
    
    Phase 0: NetworkX in-memory graph, JSON persisted
    Phase 2: Replace with full graph database (Neo4j / ArangoDB)
    """

    def __init__(self, storage_path: str = "~/.leanai/semantic"):
        self.path = Path(storage_path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self._entities: dict = {}    # entity_name → {properties}
        self._relations: list = []   # (entity_a, relation, entity_b)
        self._load()

    def add_entity(self, name: str, entity_type: str, properties: dict = None):
        """Add or update a known entity."""
        self._entities[name.lower()] = {
            "name": name,
            "type": entity_type,
            "properties": properties or {},
            "added_at": time.time(),
        }
        self._save()

    def add_relation(self, entity_a: str, relation: str, entity_b: str):
        """Add a relationship between two entities."""
        rel = (entity_a.lower(), relation, entity_b.lower())
        if rel not in self._relations:
            self._relations.append(rel)
            self._save()

    def query_entity(self, name: str) -> Optional[dict]:
        """Look up what we know about an entity."""
        return self._entities.get(name.lower())

    def query_relations(self, entity: str) -> list[tuple]:
        """Get all known relationships for an entity."""
        e = entity.lower()
        return [r for r in self._relations if r[0] == e or r[2] == e]

    def get_context_for_query(self, query: str) -> str:
        """
        Given a query, retrieve relevant semantic facts as a context string.
        Injected into the model's prompt to ground its knowledge.
        """
        query_words = query.lower().split()
        relevant_facts = []

        for word in query_words:
            entity = self.query_entity(word)
            if entity:
                props = entity.get("properties", {})
                prop_str = ", ".join(f"{k}: {v}" for k, v in props.items())
                relevant_facts.append(f"{entity['name']} ({entity['type']}): {prop_str}")

            for subj, rel, obj in self.query_relations(word):
                relevant_facts.append(f"{subj} {rel} {obj}")

        if not relevant_facts:
            return ""

        return "Known facts:\n" + "\n".join(f"• {f}" for f in relevant_facts[:10])

    def _save(self):
        with open(self.path / "entities.json", "w") as f:
            json.dump(self._entities, f, indent=2)
        with open(self.path / "relations.json", "w") as f:
            json.dump(self._relations, f, indent=2)

    def _load(self):
        ent_file = self.path / "entities.json"
        rel_file = self.path / "relations.json"
        if ent_file.exists():
            with open(ent_file) as f:
                self._entities = json.load(f)
        if rel_file.exists():
            with open(rel_file) as f:
                self._relations = json.load(f)


class ProceduralMemory:
    """
    L4: Procedural memory — how to do things.
    Stores verified solutions to tasks so they never need to be solved again.
    
    When the AI solves a problem it hasn't seen before:
      1. Solve it (expensive)
      2. Verify the solution (verifier)
      3. Store the solution template (free)
      4. Next time: retrieve and adapt (near-instant)
    """

    def __init__(self, storage_path: str = "~/.leanai/procedural"):
        self.path = Path(storage_path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self._procedures: dict = {}
        self._load()

    def store_solution(
        self,
        task_signature: str,
        solution: str,
        verified: bool = False,
        tags: list = None,
    ) -> str:
        """Store a verified solution template."""
        proc_id = hashlib.md5(task_signature.encode()).hexdigest()[:12]
        self._procedures[proc_id] = {
            "signature": task_signature,
            "solution": solution,
            "verified": verified,
            "tags": tags or [],
            "use_count": 0,
            "stored_at": time.time(),
        }
        self._save()
        return proc_id

    def find_similar(self, task: str, threshold: float = 0.5) -> Optional[dict]:
        """Find a stored procedure similar to the current task."""
        task_words = set(task.lower().split())
        best_score = 0.0
        best_proc = None

        for proc in self._procedures.values():
            sig_words = set(proc["signature"].lower().split())
            overlap = len(task_words & sig_words)
            score = overlap / max(len(task_words | sig_words), 1)
            if score > best_score:
                best_score = score
                best_proc = proc

        if best_score >= threshold:
            best_proc["use_count"] += 1
            self._save()
            return best_proc
        return None

    def _save(self):
        with open(self.path / "procedures.json", "w") as f:
            json.dump(self._procedures, f, indent=2)

    def _load(self):
        proc_file = self.path / "procedures.json"
        if proc_file.exists():
            with open(proc_file) as f:
                self._procedures = json.load(f)


class HierarchicalMemory:
    """
    Unified interface to all 4 memory layers.
    This is what the engine talks to — it never touches individual layers directly.
    """

    def __init__(self, base_path: str = "~/.leanai"):
        self.working    = WorkingMemory()
        self.episodic   = EpisodicMemory(f"{base_path}/episodic")
        self.semantic   = SemanticMemory(f"{base_path}/semantic")
        self.procedural = ProceduralMemory(f"{base_path}/procedural")

    def prepare_context(self, query: str) -> str:
        """
        Build a rich context string for the current query by pulling from all layers.
        Injected into the model prompt.
        """
        parts = []

        # L2: relevant past episodes
        past = self.episodic.search(query, top_k=3)
        if past:
            past_text = "\n".join(f"• {e.content[:200]}" for e in past)
            parts.append(f"Relevant past context:\n{past_text}")

        # L3: semantic facts
        facts = self.semantic.get_context_for_query(query)
        if facts:
            parts.append(facts)

        # L4: known solution template
        proc = self.procedural.find_similar(query)
        if proc and proc.get("verified"):
            parts.append(f"Known solution template:\n{proc['solution'][:400]}")

        return "\n\n".join(parts)

    def record_exchange(self, user_msg: str, ai_response: str):
        """Record a conversation turn across all relevant layers."""
        self.working.add_message("user", user_msg)
        self.working.add_message("assistant", ai_response)

        # Store in episodic memory
        self.episodic.store(
            content=f"User: {user_msg}\nAssistant: {ai_response[:300]}",
            tags=["conversation"],
        )

    def stats(self) -> dict:
        return {
            "working_tokens": self.working.current_tokens,
            "working_messages": len(self.working.messages),
            "episodic_entries": len(self.episodic._index),
            "semantic_entities": len(self.semantic._entities),
            "procedural_solutions": len(self.procedural._procedures),
        }
