"""
LeanAI · Phase 3 — Continuous LoRA Training Loop
Runs in a background thread. Trains the model automatically.

How it works:
  1. Every N minutes, checks if enough high-quality pairs exist
  2. If yes, exports training data in instruction format
  3. Runs LoRA fine-tuning (when full training available)
  4. Saves the adapter
  5. Engine picks up the new adapter on next load

Phase 3 implements the full training data pipeline and scheduling.
Actual LoRA weight updates require the full model in fp16/bf16 format
(not GGUF) — this is the Phase 3 foundation, Phase 4 completes it
with the quantization-aware training loop.

What ships now:
  ✓ Training scheduler
  ✓ Data export pipeline
  ✓ Adapter management
  ✓ Background thread
  ✓ Training history tracking
  ✓ Self-play batch generation on schedule
"""

import json
import time
import threading
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from training.quality_filter import QualityFilter
from training.self_improve import TrainingDataStore, FeedbackSignal
from training.self_play_v2 import EnhancedSelfPlayEngine


@dataclass
class TrainingRun:
    id: str
    started_at: float
    completed_at: Optional[float]
    pairs_used: int
    adapter_path: Optional[str]
    status: str           # "running" | "complete" | "failed" | "skipped"
    notes: str = ""


@dataclass
class TrainingConfig:
    # Scheduling
    check_interval_minutes: int = 30
    min_pairs_to_train: int = 50
    min_new_pairs_since_last: int = 20

    # LoRA hyperparameters (used when full trainer available)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation: int = 4
    max_steps: int = 200
    warmup_steps: int = 20
    save_steps: int = 50

    # Self-play
    self_play_every_n_checks: int = 3
    self_play_batch_size: int = 20


class ContinualTrainer:
    """
    Background training loop.
    Runs continuously, trains when enough data accumulates.
    """

    def __init__(
        self,
        store: TrainingDataStore,
        config: Optional[TrainingConfig] = None,
        base_path: str = "~/.leanai",
    ):
        self.store   = store
        self.config  = config or TrainingConfig()
        self.filter  = QualityFilter()
        self.self_play = EnhancedSelfPlayEngine()

        self.base_path    = Path(base_path).expanduser()
        self.adapters_dir = self.base_path / "adapters"
        self.exports_dir  = self.base_path / "training_exports"
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        self._history: list[TrainingRun] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._check_count = 0
        self._last_pair_count = 0

        self._load_history()
        print(f"[Trainer] Initialized. {len(self._history)} training runs in history.")

    # ══════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════

    def start(self):
        """Start the background training loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="LeanAI-Trainer",
        )
        self._thread.start()
        print(f"[Trainer] Background loop started. "
              f"Checks every {self.config.check_interval_minutes} min.")

    def stop(self):
        """Stop the background loop gracefully."""
        self._running = False
        print("[Trainer] Background loop stopping...")

    def run_now(self) -> TrainingRun:
        """Manually trigger a training check right now."""
        return self._training_cycle()

    def generate_self_play(self, n: int = None) -> int:
        """Generate and store self-play training pairs."""
        n = n or self.config.self_play_batch_size
        pairs = self.self_play.generate_batch(n)
        added = 0
        for sp in pairs:
            self.store.add_pair(
                instruction=sp.problem,
                response=sp.solution,
                feedback=FeedbackSignal.EXCELLENT if sp.verified else FeedbackSignal.GOOD,
                confidence=0.95 if sp.verified else 0.7,
                verified=sp.verified,
                latency_ms=0,
                tier_used="self_play",
                tags=[sp.domain, sp.subdomain, "phase3_selfplay"],
            )
            added += 1
        print(f"[Trainer] Generated {added} self-play pairs "
              f"({sum(1 for p in pairs if p.verified)} verified)")
        return added

    def export_training_data(
        self,
        output_path: Optional[str] = None,
        min_quality: float = 0.65,
    ) -> str:
        """
        Export high-quality pairs in instruction-tuning JSONL format.
        Compatible with standard fine-tuning frameworks.
        """
        all_pairs = list(self.store._pairs.values())
        accepted, rejected = self.filter.filter_batch(all_pairs)

        if not accepted:
            print("[Trainer] No pairs passed quality filter.")
            return ""

        path = output_path or str(
            self.exports_dir / f"training_{int(time.time())}.jsonl"
        )

        with open(path, "w", encoding="utf-8") as f:
            for pair in accepted:
                record = {
                    "instruction": pair.instruction,
                    "output": pair.response,
                    "quality": round(self.filter.assess(pair).score, 3),
                    "verified": pair.verified,
                    "domain": pair.tier_used,
                    "tags": pair.tags,
                }
                f.write(json.dumps(record) + "\n")

        print(f"[Trainer] Exported {len(accepted)} pairs → {path}")
        print(f"[Trainer] Rejected {len(rejected)} pairs (below quality threshold)")
        return path

    def status(self) -> dict:
        all_pairs = list(self.store._pairs.values())
        filter_stats = self.filter.stats(all_pairs)
        last_run = self._history[-1] if self._history else None

        return {
            "running": self._running,
            "total_pairs": len(all_pairs),
            "quality_filter": filter_stats,
            "training_runs": len(self._history),
            "last_run": asdict(last_run) if last_run else None,
            "config": {
                "check_interval_min": self.config.check_interval_minutes,
                "min_pairs": self.config.min_pairs_to_train,
                "lora_rank": self.config.lora_rank,
                "learning_rate": self.config.learning_rate,
            },
            "adapters": [str(p) for p in self.adapters_dir.glob("*.json")],
        }

    # ══════════════════════════════════════════════════
    # Background loop
    # ══════════════════════════════════════════════════

    def _loop(self):
        """Main background loop."""
        while self._running:
            try:
                self._check_count += 1

                # Generate self-play data periodically
                if self._check_count % self.config.self_play_every_n_checks == 0:
                    self.generate_self_play()

                # Run training cycle
                self._training_cycle()

            except Exception as e:
                print(f"[Trainer] Error in loop: {e}")

            # Wait for next check
            interval = self.config.check_interval_minutes * 60
            for _ in range(interval):
                if not self._running:
                    break
                time.sleep(1)

    def _training_cycle(self) -> TrainingRun:
        """
        One training cycle: check data → export → train → save.
        """
        run_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        run = TrainingRun(
            id=run_id,
            started_at=time.time(),
            completed_at=None,
            pairs_used=0,
            adapter_path=None,
            status="running",
        )

        all_pairs = list(self.store._pairs.values())
        accepted, _ = self.filter.filter_batch(all_pairs)
        new_pairs = len(all_pairs) - self._last_pair_count

        # Check if we have enough to train
        if len(accepted) < self.config.min_pairs_to_train:
            run.status = "skipped"
            run.notes = (
                f"Only {len(accepted)} quality pairs "
                f"(need {self.config.min_pairs_to_train})"
            )
            run.completed_at = time.time()
            self._history.append(run)
            self._save_history()
            print(f"[Trainer] Skipped — {run.notes}")
            return run

        if new_pairs < self.config.min_new_pairs_since_last:
            run.status = "skipped"
            run.notes = f"Only {new_pairs} new pairs since last run"
            run.completed_at = time.time()
            self._history.append(run)
            self._save_history()
            print(f"[Trainer] Skipped — {run.notes}")
            return run

        # Export training data
        export_path = self.export_training_data()
        if not export_path:
            run.status = "failed"
            run.notes = "Export failed"
            run.completed_at = time.time()
            self._history.append(run)
            return run

        run.pairs_used = len(accepted)
        self._last_pair_count = len(all_pairs)

        # Attempt LoRA training
        adapter_path = self._attempt_lora_training(export_path, run_id)

        if adapter_path:
            run.adapter_path = adapter_path
            run.status = "complete"
            run.notes = f"LoRA adapter saved: {adapter_path}"
            print(f"[Trainer] Training complete. Adapter: {adapter_path}")
        else:
            # Save training manifest even without weights
            # (full training requires fp16 model + GPU or extended CPU time)
            manifest = {
                "run_id": run_id,
                "timestamp": time.time(),
                "pairs_used": len(accepted),
                "export_path": export_path,
                "config": asdict(self.config),
                "status": "data_ready",
                "note": (
                    "Training data exported and ready. "
                    "Full LoRA training requires: pip install peft transformers "
                    "and a fp16 model (not GGUF). "
                    "Run: python train_lora.py to execute."
                ),
            }
            manifest_path = self.adapters_dir / f"manifest_{run_id}.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))
            run.adapter_path = str(manifest_path)
            run.status = "complete"
            run.notes = "Training data ready. Manifest saved."

        run.completed_at = time.time()
        self._history.append(run)
        self._save_history()
        return run

    def _attempt_lora_training(self, export_path: str, run_id: str) -> Optional[str]:
        """
        Try to run actual LoRA training.
        Requires peft + transformers + fp16 model.
        Returns adapter path if successful, None if requirements not met.
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
            import torch

            print("[Trainer] PEFT available — attempting LoRA training...")

            # This would load the base model in fp16 and apply LoRA
            # For now, returns None as GGUF models need conversion first
            # Full implementation in Phase 4
            print("[Trainer] Note: GGUF → fp16 conversion needed for training.")
            print("[Trainer] Training manifest saved. Use train_lora.py for full training.")
            return None

        except ImportError:
            # PEFT not installed — save manifest only
            return None

    # ══════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════

    def _save_history(self):
        history_file = self.base_path / "training_history.json"
        data = [asdict(r) for r in self._history[-50:]]  # keep last 50
        history_file.write_text(json.dumps(data, indent=2))

    def _load_history(self):
        history_file = self.base_path / "training_history.json"
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text())
                self._history = [TrainingRun(**r) for r in data]
                self._last_pair_count = (
                    self._history[-1].pairs_used if self._history else 0
                )
            except Exception:
                self._history = []
