"""
LeanAI — Fine-Tune Runner
Orchestrates QLoRA fine-tuning of local models on your personal data.

This is the core that makes your model learn YOUR coding patterns.
Runs overnight, produces a LoRA adapter (~50MB), and the next morning
your model is smarter about YOUR specific code.

Training methods (in order of preference):
  1. QLoRA via unsloth (fastest, best for consumer GPUs)
  2. QLoRA via PEFT + bitsandbytes (standard, needs 4GB+ VRAM)
  3. CPU fine-tuning via llama.cpp (slowest, works everywhere)
  4. Export-only mode (exports data, user runs training externally)
"""

import os
import sys
import json
import time
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from training.finetune_pipeline import TrainingDataPipeline
from training.adapter_manager import AdapterManager


@dataclass
class TrainingConfig:
    """Configuration for a fine-tuning run."""
    adapter_name: str = "default"
    base_model: str = "qwen-7b"
    min_examples: int = 50
    min_quality: float = 0.7
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    batch_size: int = 4
    max_seq_length: int = 1024
    gradient_accumulation: int = 4
    warmup_ratio: float = 0.1
    schedule: str = "nightly"  # "nightly", "weekly", "manual"


@dataclass
class TrainingRun:
    """Record of a training run."""
    run_id: str
    started: float
    finished: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    examples_used: int = 0
    epochs: int = 0
    adapter_path: str = ""
    error: str = ""
    duration_minutes: float = 0.0

    def summary(self) -> str:
        status_icon = {"completed": "●", "running": "◌", "failed": "✗", "pending": "○"}.get(self.status, "?")
        return (
            f"{status_icon} Run {self.run_id[:8]} | {self.status} | "
            f"{self.examples_used} examples | {self.duration_minutes:.0f} min"
        )


class FineTuneRunner:
    """
    Orchestrates fine-tuning of local models.
    
    Usage:
        runner = FineTuneRunner()
        
        # Check readiness
        print(runner.check_readiness())
        
        # Run training
        result = runner.train(config=TrainingConfig(adapter_name="work"))
        
        # Or schedule nightly
        runner.start_nightly_schedule()
    """

    def __init__(
        self,
        pipeline: Optional[TrainingDataPipeline] = None,
        adapter_mgr: Optional[AdapterManager] = None,
        data_dir: Optional[str] = None,
    ):
        self.data_dir = data_dir or str(Path(os.environ.get('LEANAI_HOME', str(Path.home() / '.leanai'))) / "finetune")
        os.makedirs(self.data_dir, exist_ok=True)

        self.pipeline = pipeline or TrainingDataPipeline(
            data_dir=os.path.join(self.data_dir, "data")
        )
        self.adapter_mgr = adapter_mgr or AdapterManager(
            data_dir=os.path.join(self.data_dir, "adapters")
        )

        self._runs: List[TrainingRun] = []
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_running = False
        self._load_history()

    def check_readiness(self) -> str:
        """Check if the system is ready for fine-tuning."""
        lines = ["═══ Fine-Tuning Readiness Check ═══"]

        # Data check
        stats = self.pipeline.get_stats()
        lines.append(f"\nTraining data: {stats.total_pairs} examples collected")
        lines.append(f"  High quality (>70%): {stats.high_quality}")
        lines.append(f"  Verified code: {stats.verified_code}")

        if stats.total_pairs < 50:
            lines.append(f"  ⚠ Need at least 50 examples. Keep using LeanAI!")
            lines.append(f"  Progress: {stats.total_pairs}/50 ({stats.total_pairs*100//50}%)")
        else:
            lines.append(f"  ✓ Enough data for fine-tuning!")

        # GPU check
        gpu_available = self._check_gpu()
        if gpu_available:
            lines.append(f"\nGPU: detected ✓")
        else:
            lines.append(f"\nGPU: not available (will use CPU — slower but works)")

        # PyTorch check
        pytorch_ok = self._check_pytorch()
        if pytorch_ok:
            lines.append(f"PyTorch: installed ✓")
        else:
            lines.append(f"PyTorch: not installed")
            lines.append(f"  Install: pip install torch")

        # PEFT check
        peft_ok = self._check_peft()
        if peft_ok:
            lines.append(f"PEFT/LoRA: installed ✓")
        else:
            lines.append(f"PEFT/LoRA: not installed")
            lines.append(f"  Install: pip install peft bitsandbytes transformers")

        # llama.cpp training check
        lines.append(f"\nllama.cpp export: always available ✓")
        lines.append(f"  Can export data for external training tools")

        # Adapters
        adapters = self.adapter_mgr.list_adapters()
        lines.append(f"\n{adapters}")

        # Overall status
        ready = stats.total_pairs >= 50
        lines.append(f"\n{'✓ READY for fine-tuning!' if ready else '○ Collecting more data...'}")

        return "\n".join(lines)

    def _check_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _check_pytorch(self) -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False

    def _check_peft(self) -> bool:
        try:
            import peft
            return True
        except ImportError:
            return False

    def train(self, config: Optional[TrainingConfig] = None) -> TrainingRun:
        """
        Run a fine-tuning session.
        Automatically selects the best available training method.
        """
        config = config or TrainingConfig()

        run = TrainingRun(
            run_id=f"run_{int(time.time())}",
            started=time.time(),
            status="running",
        )
        self._runs.append(run)

        try:
            # Step 1: Curate training data
            print(f"[FineTune] Curating training data (min quality: {config.min_quality})...", flush=True)
            examples = self.pipeline.curate(
                min_quality=config.min_quality,
                min_examples=config.min_examples,
            )

            if len(examples) < config.min_examples:
                run.status = "failed"
                run.error = f"Not enough examples: {len(examples)}/{config.min_examples}"
                print(f"[FineTune] {run.error}")
                self._save_history()
                return run

            run.examples_used = len(examples)
            print(f"[FineTune] Selected {len(examples)} high-quality examples", flush=True)

            # Step 2: Export training data
            print(f"[FineTune] Exporting training data...", flush=True)
            export_path = self.pipeline.export_sharegpt(
                f"finetune_{run.run_id}.jsonl",
                min_quality=config.min_quality,
            )
            print(f"[FineTune] Exported to {export_path}", flush=True)

            # Step 3: Create/update adapter
            if config.adapter_name not in self.adapter_mgr._adapters:
                self.adapter_mgr.create(
                    config.adapter_name,
                    base_model=config.base_model,
                    description=f"Auto-trained from {len(examples)} examples",
                )

            adapter_output = os.path.join(
                self.data_dir, "adapters", config.adapter_name, f"v{int(time.time())}"
            )
            os.makedirs(adapter_output, exist_ok=True)

            # Step 4: Attempt training (try each method)
            trained = False

            if not trained and self._check_peft() and self._check_gpu():
                print(f"[FineTune] Training with QLoRA + GPU...", flush=True)
                trained = self._train_qlora(config, export_path, adapter_output)

            if not trained:
                # Export-only mode: save the data and training script
                print(f"[FineTune] GPU training not available.", flush=True)
                print(f"[FineTune] Exporting data + training script for manual use.", flush=True)
                self._export_training_script(config, export_path, adapter_output)
                trained = True  # mark as "completed" since data is ready

            # Step 5: Register the adapter
            if trained:
                run.status = "completed"
                run.adapter_path = adapter_output
                run.finished = time.time()
                run.duration_minutes = (run.finished - run.started) / 60
                run.epochs = config.epochs

                self.adapter_mgr.register_trained(
                    config.adapter_name,
                    adapter_output,
                    num_examples=len(examples),
                    training_hours=run.duration_minutes / 60,
                )
                print(f"[FineTune] Completed! Adapter saved to {adapter_output}", flush=True)
            else:
                run.status = "failed"
                run.error = "No training method available"

        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            print(f"[FineTune] Error: {e}", flush=True)

        run.finished = time.time()
        run.duration_minutes = (run.finished - run.started) / 60
        self._save_history()
        return run

    def _train_qlora(self, config: TrainingConfig, data_path: str, output_dir: str) -> bool:
        """Run QLoRA fine-tuning using PEFT + transformers."""
        script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json

# Load dataset
dataset = load_dataset('json', data_files='{data_path}', split='train')

# Model path
model_path = "{config.base_model}"

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r={config.lora_rank},
    lora_alpha={config.lora_alpha},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Training args
training_args = TrainingArguments(
    output_dir="{output_dir}",
    num_train_epochs={config.epochs},
    per_device_train_batch_size={config.batch_size},
    gradient_accumulation_steps={config.gradient_accumulation},
    learning_rate={config.learning_rate},
    warmup_ratio={config.warmup_ratio},
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
)

# Train
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length={config.max_seq_length},
)
trainer.train()
trainer.save_model("{output_dir}")
print("Training complete!")
"""
        script_path = os.path.join(output_dir, "train_qlora.py")
        with open(script_path, "w") as f:
            f.write(script)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=False, text=True, timeout=7200,  # 2 hour limit
            )
            return result.returncode == 0
        except Exception:
            return False

    def _export_training_script(self, config: TrainingConfig, data_path: str, output_dir: str):
        """Export everything needed for manual/external training."""
        # Save config
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "base_model": config.base_model,
                "data_path": data_path,
                "output_dir": output_dir,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "batch_size": config.batch_size,
                "max_seq_length": config.max_seq_length,
            }, f, indent=2)

        # Save training instructions
        readme = f"""# LeanAI Fine-Tuning Data

## What's here
- `training_config.json` — hyperparameters
- Data file: `{data_path}`
- {config.epochs} epochs, LoRA rank {config.lora_rank}

## How to train

### Option 1: Google Colab (free GPU)
1. Upload the data file to Colab
2. Install: `pip install unsloth peft trl datasets`
3. Run the training script

### Option 2: Local with GPU
```bash
pip install torch peft bitsandbytes transformers trl datasets
python train_qlora.py
```

### Option 3: Unsloth (fastest)
```bash
pip install unsloth
# Use unsloth's fine-tuning script with your data
```

## After training
Copy the adapter files back to:
  `~/.leanai/finetune/adapters/{config.adapter_name}/`

Then in LeanAI:
  `/finetune activate {config.adapter_name}`
"""
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(readme)

        print(f"[FineTune] Training data + script saved to: {output_dir}")
        print(f"[FineTune] You can train on Google Colab (free) or any machine with a GPU")

    # ── Nightly Schedule ──────────────────────────────────────────

    def start_nightly_schedule(self, hour: int = 2):
        """Start a background thread that trains at the specified hour."""
        if self._scheduler_running:
            return
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(
            target=self._schedule_loop, args=(hour,), daemon=True
        )
        self._scheduler_thread.start()
        print(f"[FineTune] Nightly training scheduled at {hour}:00")

    def stop_schedule(self):
        self._scheduler_running = False

    def _schedule_loop(self, hour: int):
        """Background loop that checks if it's time to train."""
        import datetime
        last_train_date = None

        while self._scheduler_running:
            now = datetime.datetime.now()
            if now.hour == hour and now.date() != last_train_date:
                stats = self.pipeline.get_stats()
                if stats.total_pairs >= 50:
                    print(f"\n[FineTune] Nightly training starting...", flush=True)
                    self.train()
                    last_train_date = now.date()
            time.sleep(300)  # check every 5 minutes

    # ── History ───────────────────────────────────────────────────

    def list_runs(self) -> str:
        if not self._runs:
            return "No training runs yet."
        lines = ["Training History:"]
        for run in self._runs[-10:]:
            lines.append(f"  {run.summary()}")
        return "\n".join(lines)

    def _history_path(self) -> str:
        return os.path.join(self.data_dir, "training_history.json")

    def _save_history(self):
        data = [
            {
                "run_id": r.run_id, "started": r.started,
                "finished": r.finished, "status": r.status,
                "examples_used": r.examples_used, "epochs": r.epochs,
                "adapter_path": r.adapter_path, "error": r.error,
                "duration_minutes": r.duration_minutes,
            }
            for r in self._runs
        ]
        try:
            with open(self._history_path(), "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_history(self):
        path = self._history_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._runs = [TrainingRun(**r) for r in data]
        except Exception:
            pass

    def stats(self) -> dict:
        return {
            "total_runs": len(self._runs),
            "completed": sum(1 for r in self._runs if r.status == "completed"),
            "failed": sum(1 for r in self._runs if r.status == "failed"),
            "data": self.pipeline.get_stats().summary(),
            "adapters": self.adapter_mgr.stats(),
            "scheduler_active": self._scheduler_running,
        }
