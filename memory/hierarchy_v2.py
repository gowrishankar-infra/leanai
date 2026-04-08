"""
LeanAI · Phase 2 — Upgraded Hierarchical Memory
Replaces Phase 1 keyword search with vector similarity.
Adds causal world model layer.

L1 · Working memory    — current context (unchanged)
L2 · Episodic memory   — ChromaDB vector search (UPGRADED)
L3 · Semantic memory   — world model with entity extraction (UPGRADED)
L4 · Procedural memory — verified solution cache (unchanged)
"""

import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from memory.vector_memory import VectorEpisodicMemory
from world.world_model import WorldModel


@dataclass
class WorkingMemory:
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
        self.current_tokens += len(content) // 4

    def get_context_window(self, max_tokens: Optional[int] = None) -> list:
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


class ProceduralMemory:
    """L4: Verified solution cache — unchanged from Phase 1."""

    def __init__(self, storage_path: str = "~/.leanai/procedural"):
        self.path = Path(storage_path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self._procedures: dict = {}
        self._load()

    def store_solution(self, task_signature: str, solution: str,
                       verified: bool = False, tags: list = None) -> str:
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


class HierarchicalMemoryV2:
    """
    Phase 2 unified memory interface.
    All 4 layers — working, episodic (vector), semantic (world model), procedural.
    """

    def __init__(self, base_path: str = "~/.leanai"):
        self.working    = WorkingMemory()
        self.episodic   = VectorEpisodicMemory(f"{base_path}/vector_memory")
        self.world      = WorldModel(f"{base_path}/world")
        self.procedural = ProceduralMemory(f"{base_path}/procedural")

    def prepare_context(self, query: str) -> str:
        """
        Build rich context from all memory layers.
        This is what gets injected into every prompt.
        """
        parts = []

        # L3: World model — user profile + relevant entities
        world_ctx = self.world.get_context_for_query(query)
        if world_ctx:
            parts.append(world_ctx)

        # L2: Episodic — semantically similar past exchanges
        past = self.episodic.search(query, top_k=3)
        if past:
            past_lines = []
            for e in past:
                # Only include if content is meaningfully different from query
                if len(e.content) > 20:
                    past_lines.append(f"  • {e.content[:180]}")
            if past_lines:
                parts.append("Relevant past context:\n" + "\n".join(past_lines))

        # L4: Procedural — known solution template
        proc = self.procedural.find_similar(query)
        if proc and proc.get("verified"):
            parts.append(f"Known solution:\n  {proc['solution'][:300]}")

        return "\n\n".join(parts)

    def record_exchange(self, user_msg: str, ai_response: str):
        """Record a full exchange across all relevant layers."""
        # L1: Working memory
        self.working.add_message("user", user_msg)
        self.working.add_message("assistant", ai_response)

        # L2: Episodic vector store
        self.episodic.store_exchange(user_msg, ai_response)

        # L3: World model learning
        self.world.learn_from_exchange(user_msg, ai_response)

    def remember_fact(self, fact: str):
        """Explicitly store an important fact."""
        self.episodic.store_fact(fact)
        self.world.learn_fact(fact)

    def answer_from_memory(self, query: str) -> Optional[str]:
        """Try to answer directly from memory without using the model."""
        # Try world model first
        answer = self.world.answer_about_user(query)
        if answer:
            return answer
        return None

    def stats(self) -> dict:
        world_stats = self.world.stats()
        return {
            "working_tokens": self.working.current_tokens,
            "working_messages": len(self.working.messages),
            "episodic_entries": self.episodic.count(),
            "episodic_backend": self.episodic.backend,
            "semantic_entities": world_stats["entities"],
            "semantic_relations": world_stats["relations"],
            "user_profile_fields": world_stats["user_profile_fields"],
            "procedural_solutions": len(self.procedural._procedures),
            "world_entity_types": world_stats["entity_types"],
        }
