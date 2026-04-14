"""
LeanAI — Training Data Pipeline
Converts collected interaction pairs into fine-tuning datasets.

Flow:
  1. Collect: Every query/response pair is stored (already built)
  2. Curate: Filter by quality — only verified, high-confidence pairs
  3. Format: Convert to instruction-tuning format (ShareGPT/Alpaca)
  4. Deduplicate: Remove near-duplicate examples
  5. Export: Save as JSONL ready for QLoRA training

The key insight: not all pairs are equal. Code that passed verification
is worth 10x more than a generic chat response for fine-tuning.
"""

import os
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class TrainingExample:
    """A single training example for fine-tuning."""
    instruction: str          # what the user asked
    response: str             # what the model answered
    quality_score: float      # 0-1, how good this example is
    source: str = ""          # where this came from (chat, tdd, swarm, etc.)
    verified: bool = False    # was the code verified working?
    timestamp: float = 0.0
    tags: List[str] = field(default_factory=list)

    def to_sharegpt(self) -> dict:
        """Convert to ShareGPT format (used by most fine-tuning tools)."""
        return {
            "conversations": [
                {"from": "human", "value": self.instruction},
                {"from": "gpt", "value": self.response},
            ]
        }

    def to_alpaca(self) -> dict:
        """Convert to Alpaca format."""
        return {
            "instruction": self.instruction,
            "input": "",
            "output": self.response,
        }

    def to_chatml(self) -> str:
        """Convert to ChatML format (what Qwen uses)."""
        return (
            f"<|im_start|>user\n{self.instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{self.response}<|im_end|>"
        )

    @property
    def content_hash(self) -> str:
        """Hash for deduplication."""
        text = f"{self.instruction.lower().strip()}|{self.response[:100].lower().strip()}"
        return hashlib.md5(text.encode()).hexdigest()


@dataclass
class DatasetStats:
    """Statistics about a curated dataset."""
    total_pairs: int = 0
    high_quality: int = 0
    verified_code: int = 0
    average_quality: float = 0.0
    sources: Dict[str, int] = field(default_factory=dict)
    total_tokens_estimate: int = 0

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.total_pairs} total, {self.high_quality} high-quality",
            f"Verified code: {self.verified_code}",
            f"Avg quality: {self.average_quality:.0%}",
            f"Est. tokens: {self.total_tokens_estimate:,}",
        ]
        if self.sources:
            src = ", ".join(f"{k}:{v}" for k, v in self.sources.items())
            lines.append(f"Sources: {src}")
        return "\n".join(lines)


class TrainingDataPipeline:
    """
    Curates and manages fine-tuning datasets from user interactions.
    
    Usage:
        pipeline = TrainingDataPipeline()
        
        # Add examples from interactions
        pipeline.add_example(
            instruction="write a decorator with retry logic",
            response="def retry(max_retries=3): ...",
            quality_score=0.95,
            source="tdd",
            verified=True,
        )
        
        # Curate high-quality dataset
        dataset = pipeline.curate(min_quality=0.7)
        
        # Export for training
        pipeline.export_sharegpt("training_data.jsonl")
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or str(Path(os.environ.get('LEANAI_HOME', str(Path.home() / '.leanai'))) / "finetune" / "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self._examples: List[TrainingExample] = []
        self._seen_hashes: set = set()
        self._load()

    def add_example(
        self,
        instruction: str,
        response: str,
        quality_score: float = 0.5,
        source: str = "chat",
        verified: bool = False,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Add a training example. Returns True if added, False if duplicate.
        """
        # Skip truly empty content
        if len(instruction.strip()) < 5 or len(response.strip()) < 10:
            return False
        # Truncate very long responses
        if len(response) > 5000:
            response = response[:5000]

        example = TrainingExample(
            instruction=instruction,
            response=response,
            quality_score=quality_score,
            source=source,
            verified=verified,
            timestamp=time.time(),
            tags=tags or [],
        )

        # Deduplicate
        h = example.content_hash
        if h in self._seen_hashes:
            return False
        self._seen_hashes.add(h)

        # Boost quality score for verified code
        if verified:
            example.quality_score = max(example.quality_score, 0.85)

        self._examples.append(example)

        # Auto-save periodically
        if len(self._examples) % 20 == 0:
            self.save()

        return True

    def add_from_session_store(self, session_store) -> int:
        """Import examples from the session store (Phase 7e)."""
        added = 0
        for session in session_store._sessions.values():
            for exchange in session.exchanges:
                if self.add_example(
                    instruction=exchange.query,
                    response=exchange.response,
                    quality_score=exchange.confidence / 100 if exchange.confidence > 1 else exchange.confidence,
                    source=exchange.tier or "chat",
                    verified=exchange.code_generated,
                    tags=exchange.topics,
                ):
                    added += 1
        return added

    def add_from_training_exports(self, exports_dir: Optional[str] = None) -> int:
        """Import from existing training export JSONL files."""
        exports_dir = exports_dir or str(Path(os.environ.get('LEANAI_HOME', str(Path.home() / '.leanai'))) / "training_exports")
        if not os.path.isdir(exports_dir):
            return 0

        added = 0
        for filename in os.listdir(exports_dir):
            if not filename.endswith(".jsonl"):
                continue
            filepath = os.path.join(exports_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if self.add_example(
                                instruction=data.get("instruction", ""),
                                response=data.get("response", ""),
                                quality_score=data.get("quality_score", 0.5),
                                source="export",
                                verified=data.get("verified", False),
                            ):
                                added += 1
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
        return added

    def curate(self, min_quality: float = 0.7, min_examples: int = 50) -> List[TrainingExample]:
        """
        Curate high-quality examples for fine-tuning.
        Prioritizes verified code and high-confidence responses.
        """
        # Filter by quality
        candidates = [e for e in self._examples if e.quality_score >= min_quality]

        # Sort by quality (highest first), then by verified status
        candidates.sort(key=lambda e: (e.verified, e.quality_score), reverse=True)

        # If not enough high-quality, lower the threshold
        if len(candidates) < min_examples:
            lower_threshold = min_quality * 0.7
            candidates = [e for e in self._examples if e.quality_score >= lower_threshold]
            candidates.sort(key=lambda e: (e.verified, e.quality_score), reverse=True)

        return candidates

    def get_stats(self) -> DatasetStats:
        """Get statistics about the current dataset."""
        if not self._examples:
            return DatasetStats()

        sources = {}
        verified = 0
        total_quality = 0
        total_tokens = 0

        for e in self._examples:
            sources[e.source] = sources.get(e.source, 0) + 1
            if e.verified:
                verified += 1
            total_quality += e.quality_score
            total_tokens += len(e.instruction.split()) + len(e.response.split())

        high_quality = len([e for e in self._examples if e.quality_score >= 0.7])

        return DatasetStats(
            total_pairs=len(self._examples),
            high_quality=high_quality,
            verified_code=verified,
            average_quality=total_quality / len(self._examples),
            sources=sources,
            total_tokens_estimate=total_tokens,
        )

    # ── Export formats ────────────────────────────────────────────

    def export_sharegpt(self, filename: Optional[str] = None, min_quality: float = 0.7) -> str:
        """Export as ShareGPT JSONL (compatible with most fine-tuning tools)."""
        filename = filename or f"sharegpt_{int(time.time())}.jsonl"
        filepath = os.path.join(self.data_dir, filename)
        examples = self.curate(min_quality)

        with open(filepath, "w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(e.to_sharegpt()) + "\n")

        return filepath

    def export_alpaca(self, filename: Optional[str] = None, min_quality: float = 0.7) -> str:
        """Export as Alpaca JSON."""
        filename = filename or f"alpaca_{int(time.time())}.json"
        filepath = os.path.join(self.data_dir, filename)
        examples = self.curate(min_quality)

        data = [e.to_alpaca() for e in examples]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return filepath

    def export_chatml(self, filename: Optional[str] = None, min_quality: float = 0.7) -> str:
        """Export as ChatML text (Qwen native format)."""
        filename = filename or f"chatml_{int(time.time())}.txt"
        filepath = os.path.join(self.data_dir, filename)
        examples = self.curate(min_quality)

        with open(filepath, "w", encoding="utf-8") as f:
            for e in examples:
                f.write(e.to_chatml() + "\n\n")

        return filepath

    # ── Persistence ───────────────────────────────────────────────

    def save(self):
        """Save all examples to disk."""
        filepath = os.path.join(self.data_dir, "training_examples.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            for e in self._examples:
                data = {
                    "instruction": e.instruction,
                    "response": e.response,
                    "quality_score": e.quality_score,
                    "source": e.source,
                    "verified": e.verified,
                    "timestamp": e.timestamp,
                    "tags": e.tags,
                }
                f.write(json.dumps(data) + "\n")

    def _load(self):
        """Load examples from disk."""
        filepath = os.path.join(self.data_dir, "training_examples.jsonl")
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        example = TrainingExample(
                            instruction=data["instruction"],
                            response=data["response"],
                            quality_score=data.get("quality_score", 0.5),
                            source=data.get("source", ""),
                            verified=data.get("verified", False),
                            timestamp=data.get("timestamp", 0),
                            tags=data.get("tags", []),
                        )
                        h = example.content_hash
                        if h not in self._seen_hashes:
                            self._seen_hashes.add(h)
                            self._examples.append(example)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            pass

    @property
    def count(self) -> int:
        return len(self._examples)
