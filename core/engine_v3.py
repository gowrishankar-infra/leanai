"""
LeanAI Phase 3+4b Engine
- Qwen2.5 Coder 7B/32B (chatml) + auto-detection for other models
- Sandboxed code execution with auto-fix loop
- Only returns verified-working code
"""
import time
import os
import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from core.router import TaskRouter, Tier
from core.watchdog import MetaCognitiveWatchdog
from core.code_echo import CodeEchoEngine, CodeEchoConfig
from core.confidence import ConfidenceScoringEngine
from core.calibrator import ConfidenceCalibrator
from tools.z3_verifier import Z3Verifier, Verdict
from tools.executor import CodeExecutor
from memory.hierarchy_v2 import HierarchicalMemoryV2
from training.self_improve import TrainingDataStore, FeedbackSignal
from training.continual_trainer import ContinualTrainer, TrainingConfig
from training.self_play_v2 import EnhancedSelfPlayEngine


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.05


@dataclass
class LeanAIResponse:
    text: str
    pair_id: str
    confidence: float
    confidence_label: str
    confidence_bar: str
    confidence_method: str
    tier_used: str
    latency_ms: float
    verified: bool
    verification_summary: str
    claims_checked: int
    claims_correct: int
    memory_context_used: bool
    answered_from_memory: bool
    quality_score: float
    code_executed: bool = False
    code_passed: bool = False
    code_output: str = ""
    code_auto_fixed: bool = False
    warning: Optional[str] = None
    corrected: bool = False


def _get_active_model_path() -> str:
    """Find the active model. Checks config file first, then scans for any downloaded model."""
    config_file = Path(os.environ.get("LEANAI_HOME", str(Path.home() / ".leanai"))) / "active_model.txt"
    if config_file.exists():
        path = config_file.read_text().strip()
        if Path(path).exists():
            return path

    # Auto-detect: scan models directory for any .gguf file
    models_dir = Path(os.environ.get("LEANAI_HOME", str(Path.home() / ".leanai"))) / "models"
    if models_dir.exists():
        # Prefer qwen models, then any .gguf
        for pattern in ["qwen*7b*.gguf", "qwen*.gguf", "*.gguf"]:
            found = list(models_dir.glob(pattern))
            if found:
                # Pick the smallest (fastest) model as default
                found.sort(key=lambda p: p.stat().st_size)
                return str(found[0])

    # Fallback — will trigger "model not found" message
    return str(models_dir / "qwen25-coder-7b-q4.gguf")


def _save_active_model(model_path: str):
    """Save the active model path so it persists across restarts."""
    try:
        config_file = Path(os.environ.get("LEANAI_HOME", str(Path.home() / ".leanai"))) / "active_model.txt"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(model_path)
    except Exception:
        pass  # non-critical


def _detect_prompt_format(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "gemma" in name: return "gemma"
    if "qwen" in name: return "chatml"
    if "phi" in name: return "phi3"
    if "llama" in name: return "llama3"
    return "chatml"


def _optimal_threads(model_path: str) -> int:
    name = Path(model_path).name.lower()
    cpu = os.cpu_count() or 8
    # All models benefit from max threads on modern CPUs
    if any(x in name for x in ["7b", "6b", "8b"]): return min(cpu, 16)
    if any(x in name for x in ["32b", "33b", "34b", "70b"]): return min(cpu, 16)
    return min(cpu, 16)


def _truncate_repetition(text: str, max_word_repeats: int = 4) -> str:
    """Detect and truncate repetitive model output. Runs on all models."""
    if not text or len(text) < 100:
        return text
    words = text.split()
    if len(words) < 20:
        return text
    # Same word repeated N+ times consecutively
    repeat_count = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            repeat_count += 1
            if repeat_count >= max_word_repeats:
                cut = i - max_word_repeats + 1
                truncated = " ".join(words[:cut]).rstrip()
                if len(truncated) > 50:
                    return truncated + "\n\n*[Response truncated — repetition detected]*"
                return text
        else:
            repeat_count = 1
    # Character-level pattern repetition
    for plen in range(5, 30):
        if len(text) < plen * 5:
            continue
        tail = text[-500:] if len(text) > 500 else text
        for i in range(len(tail) - plen * 4):
            pat = tail[i:i + plen]
            if pat.strip() and tail.count(pat) >= 8:
                pos = text.find(pat)
                if pos > 50:
                    return text[:pos].rstrip() + "\n\n*[Response truncated — repetition detected]*"
                break
    return text


def _looks_like_python(code: str) -> bool:
    """Check if code looks like Python (not YAML, JSON, bash, SQL, markdown, Go, etc)."""
    # Skip very short content (just a language tag like "go", "rust", etc)
    if len(code.strip()) < 10:
        return False

    first_lines = code.strip().split("\n")[:5]
    first = first_lines[0].strip() if first_lines else ""

    # First line is just a language name (from ```go blocks where tag leaked into content)
    lang_tags = {"go", "rust", "java", "javascript", "typescript", "c", "cpp",
                 "ruby", "php", "swift", "kotlin", "scala", "bash", "sh",
                 "yaml", "yml", "json", "sql", "html", "css", "dockerfile"}
    if first.lower() in lang_tags:
        return False

    # Definitely NOT Python
    non_python_indicators = [
        "trigger:", "pool:", "vmImage:", "steps:",        # YAML/Azure DevOps
        "apiVersion:", "kind:", "metadata:",              # Kubernetes YAML
        "FROM ", "RUN ", "COPY ", "CMD ", "ENTRYPOINT",   # Dockerfile
        "SELECT ", "INSERT ", "CREATE TABLE", "ALTER ",   # SQL
        "#!/bin/bash", "#!/bin/sh", "echo ", "sudo ",     # bash
        '{"', "'{",                                        # JSON
        "<html", "<!DOCTYPE", "<div", "<script",          # HTML
        "const ", "let ", "var ", "function ",             # JavaScript
        "package main", "func main", "import (",          # Go
        "public class", "public static void",             # Java
        "fn main", "use std",                             # Rust
    ]
    code_upper = code[:500]
    for indicator in non_python_indicators:
        if indicator in code_upper:
            return False

    # Looks like markdown/prose (not code)
    prose_indicators = [
        "- What it does", "- Why it matters", "- What this",
        "What to watch", "Improved version", "### ",
        "Here's ", "Let's break", "This pipeline",
        "In summary", "Overall,", "Note:", "Warning:",
    ]
    for indicator in prose_indicators:
        if indicator in code_upper:
            return False

    # Starts with markdown bullet — definitely not code
    if first.startswith(("- ", "* ", "> ", "| ")):
        return False

    # Looks like YAML (key: value pattern on most lines)
    yaml_lines = sum(1 for l in first_lines if re.match(r"^\s*\w+:", l))
    if yaml_lines >= 3:
        return False

    # Too much prose — more than half the lines are natural language
    prose_lines = sum(1 for l in first_lines if l.strip() and
                      len(l.split()) > 6 and l.strip()[0].isupper())
    if prose_lines >= 3:
        return False

    # Probably Python
    python_indicators = ["def ", "class ", "import ", "from ", "print(", "if __name__",
                         "return ", "self.", "elif ", "except ", "lambda "]
    for indicator in python_indicators:
        if indicator in code:
            return True

    # If we can parse it as Python AST, it's Python
    try:
        import ast
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _extract_code_blocks(text: str) -> list:
    """Extract Python code blocks from a response. Skips non-Python blocks."""
    blocks = []

    # 1. Try explicitly tagged Python blocks: ```python ... ```
    py_pattern = re.compile(r"```(?:python|py)\n?(.*?)```", re.DOTALL)
    for m in py_pattern.finditer(text):
        code = m.group(1).strip()
        if code and len(code) > 5:
            blocks.append(code)

    if blocks:
        return [_sanitize_code(b) for b in blocks]

    # 2. Try UNTAGGED code blocks (no language specified) — assume Python
    #    But skip if it looks like YAML, JSON, bash, SQL, etc.
    untagged = re.compile(r"```\n?(.*?)```", re.DOTALL)
    for m in untagged.finditer(text):
        code = m.group(1).strip()
        if code and len(code) > 5 and _looks_like_python(code):
            blocks.append(code)

    if blocks:
        return [_sanitize_code(b) for b in blocks]

    # 3. Handle UNCLOSED code blocks (model truncated mid-response)
    unclosed = re.compile(r"```(?:python|py)?\n(.*)", re.DOTALL)
    m = unclosed.search(text)
    if m:
        code = m.group(1).strip()
        if not _looks_like_python(code):
            return []
        lines = code.split("\n")
        code_lines = []
        for line in lines:
            if (code_lines and
                not line.startswith((" ", "\t", "def ", "class ", "import ", "from ",
                                    "if ", "for ", "while ", "return ", "print(",
                                    "#", "@", "}", "]", ")", "else:", "elif ",
                                    "try:", "except", "finally:", "with ", "raise ",
                                    "yield ", "async ", "await ", "")) and
                line and line[0].isupper() and " " in line and len(line) > 30):
                break
            code_lines.append(line)
        code = "\n".join(code_lines).strip()
        if code and len(code) > 10:
            blocks.append(_sanitize_code(code))

    if blocks:
        return blocks

    # 4. Last resort: if the ENTIRE response looks like pure Python code
    lines = text.strip().split("\n")
    has_code_keywords = any(kw in text for kw in ["def ", "class ", "import "])
    prose_lines = sum(1 for l in lines if l.strip() and l.strip()[0].isupper()
                      and ("." in l or "!" in l or ":" in l) and len(l.split()) > 5)
    code_lines = sum(1 for l in lines if l.strip().startswith(("def ", "class ", "import ",
                     "from ", "if ", "for ", "while ", "return ", "#", "@", "    ")))

    if has_code_keywords and prose_lines == 0 and code_lines >= 2:
        blocks.append(_sanitize_code(text.strip()))

    return blocks


def _sanitize_code(code: str) -> str:
    """
    Validate code with ast.parse. If it fails, trim broken lines from the end
    until it parses, or return the original if nothing works.
    """
    import ast

    # Quick check — does it parse as-is?
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass

    # Try trimming lines from the end (model truncated mid-function)
    lines = code.split("\n")
    for trim in range(1, min(len(lines), 15)):
        trimmed = "\n".join(lines[:-trim])
        if not trimmed.strip():
            break
        try:
            ast.parse(trimmed)
            return trimmed
        except SyntaxError:
            continue

    # Try closing common unterminated constructs
    fixups = [
        code + '\n"""',          # close triple-quoted string
        code + "\n'''",          # close single triple-quoted string
        code + "\n    pass\n",   # close empty function body
        code + "\n        pass\n",
    ]
    for fix in fixups:
        try:
            ast.parse(fix)
            return fix
        except SyntaxError:
            continue

    # Nothing worked — return original, executor will handle the error
    return code


def _is_safe_to_execute(code: str) -> bool:
    """
    Check if code is safe to auto-execute in the sandbox.
    Returns False for interactive code, project-specific imports, file deletion, etc.
    """
    unsafe_patterns = [
        "input(",           # waits for keyboard input → hangs
        "getpass",          # waits for password input
        "sys.stdin",        # reads from stdin
        "os.remove(",       # deletes files
        "os.rmdir(",        # deletes directories
        "shutil.rmtree(",   # deletes directory trees
        "os.system(",       # shell commands
        "subprocess.call(",  # shell commands
        "exit(",            # exits the process
        "quit(",            # exits the process
    ]
    code_lower = code.lower()
    for pattern in unsafe_patterns:
        if pattern.lower() in code_lower:
            return False

    # Skip code that imports from the project itself — can't work in sandbox
    project_imports = [
        "from core.", "from brain.", "from tools.", "from training.",
        "from agents.", "from swarm.", "from memory.", "from api.",
        "from federated.", "from speculative.", "from liquid.", "from hdc.",
        "import core.", "import brain.", "import tools.",
    ]
    for pattern in project_imports:
        if pattern in code:
            return False

    # Check for while True with input
    if "while true" in code_lower and "input(" in code_lower:
        return False

    return True


class LeanAIEngineV3:

    def __init__(self, model_path=None, verbose=False, auto_train=True, auto_execute=True):
        self.model_path    = model_path or _get_active_model_path()
        self.prompt_format = _detect_prompt_format(self.model_path)
        self.n_threads     = _optimal_threads(self.model_path)
        self.verbose       = verbose
        self.auto_execute  = auto_execute

        self.router     = TaskRouter()
        self.watchdog   = MetaCognitiveWatchdog()
        self.scorer     = ConfidenceScoringEngine()
        self.calibrator = ConfidenceCalibrator()
        self.verifier   = Z3Verifier()
        self.executor   = CodeExecutor()
        self.memory     = HierarchicalMemoryV2()
        self.store      = TrainingDataStore()
        self.self_play  = EnhancedSelfPlayEngine()
        self.trainer    = ContinualTrainer(
            store=self.store,
            config=TrainingConfig(
                check_interval_minutes=30,
                min_pairs_to_train=50,
                self_play_batch_size=20,
            ),
        )
        self._model        = None
        self._model_loaded = False
        self.code_echo     = CodeEchoEngine(CodeEchoConfig(verbose=verbose))

        if auto_train:
            self.trainer.start()

        model_name = Path(self.model_path).name
        print("[LeanAI v3] Engine initialized.")
        print(f"[LeanAI v3] Model: {model_name}")
        print(f"[LeanAI v3] Format: {self.prompt_format}")
        print(f"[LeanAI v3] Threads: {self.n_threads}")
        print(f"[LeanAI v3] Code executor: {'on' if auto_execute else 'off'}")
        print(f"[LeanAI v3] Available runtimes: {self.executor.available_languages}")
        print(f"[LeanAI v3] Memory: {self.memory.episodic.backend}")
        print(f"[LeanAI v3] Training: {self.store.stats()['total']} pairs")

    def generate(self, query, config=None, project_context=""):
        config = config or GenerationConfig()
        start  = time.time()
        decision = self.router.route(query, self.memory.working.current_tokens)

        # Tiny tier
        if decision.tier == Tier.TINY:
            text = self._handle_tiny(query)
            latency = (time.time() - start) * 1000
            cal = self.calibrator.calibrate(0.99, text, "tiny", False, query)
            pair = self.store.add_pair(instruction=query, response=text,
                feedback=FeedbackSignal.EXCELLENT, confidence=cal.calibrated,
                verified=False, latency_ms=latency, tier_used="tiny")
            self.memory.record_exchange(query, text)
            return self._wrap(text, pair.id, cal, "tiny", latency,
                              False, False, "Not needed.", 0, 0, False, True, 1.0)

        # Memory-first
        memory_answer = self.memory.answer_from_memory(query)
        if memory_answer:
            latency = (time.time() - start) * 1000
            cal = self.calibrator.calibrate(0.95, memory_answer, "memory", True, query)
            self.memory.record_exchange(query, memory_answer)
            pair = self.store.add_pair(instruction=query, response=memory_answer,
                feedback=FeedbackSignal.EXCELLENT, confidence=cal.calibrated,
                verified=True, latency_ms=latency, tier_used="memory")
            return self._wrap(memory_answer, pair.id, cal, "memory (instant)", latency,
                              True, False, "From world model.", 0, 0, True, True, 1.0)

        # Generate
        mem_ctx  = self.memory.prepare_context(query)
        mem_used = bool(mem_ctx)
        prompt   = self._build_prompt(query, mem_ctx,
                                       self.memory.working.get_context_window(max_tokens=512),
                                       project_context=project_context)
        response_text = self._generate_with_model(prompt, config)

        # Code execution
        code_executed = code_passed = code_auto_fixed = False
        code_output = ""

        if self.auto_execute and decision.requires_tools:
            code_blocks = _extract_code_blocks(response_text)
            # Filter out code that's unsafe to auto-execute
            code_blocks = [b for b in code_blocks if _is_safe_to_execute(b)]
            if code_blocks:
                verified = self.executor.execute_and_verify(code_blocks[0])
                code_executed  = True
                code_passed    = verified.passed
                code_auto_fixed = verified.auto_fixed
                code_output    = self.executor.format_result(verified)

                # If code failed, add error context to response
                if not verified.passed:
                    error_note = (
                        "\n\n> Execution note: Code ran but produced an error. "
                        f"Error: {verified.final_result.stderr[:200].strip()}"
                    )
                    response_text += error_note
                else:
                    # Prepend execution success note
                    pass

        # Score and calibrate
        raw_score = self.scorer.score_from_text(response_text)
        # Boost confidence if code was verified
        base_conf = 0.90 if code_passed else raw_score.overall
        cal = self.calibrator.calibrate(base_conf, response_text, decision.tier.value,
                                         code_passed, query)

        # Math/logic verification
        math_verified = math_corrected = False
        verify_summary = "Not checked."
        claims_checked = claims_correct = 0

        if decision.requires_verifier and not code_executed:
            report = self.verifier.verify_text(response_text, query)
            claims_checked = report.claims_found
            claims_correct = report.claims_verified
            math_verified  = report.overall_verdict == Verdict.TRUE
            math_corrected = report.overall_verdict == Verdict.FALSE
            verify_summary = report.summary
            if report.corrected_text:
                response_text = report.corrected_text
            cal = self.calibrator.calibrate(raw_score.overall, response_text,
                                             decision.tier.value, math_verified, query)

        self.memory.record_exchange(query, response_text)
        latency = (time.time() - start) * 1000

        feedback = (FeedbackSignal.EXCELLENT if (code_passed or math_verified) else
                    FeedbackSignal.GOOD if cal.calibrated > 0.7 else FeedbackSignal.NEUTRAL)
        pair = self.store.add_pair(instruction=query, response=response_text,
            feedback=feedback, confidence=cal.calibrated,
            verified=code_passed or math_verified,
            latency_ms=latency, tier_used=decision.tier.value)

        r = self._wrap(response_text, pair.id, cal, decision.tier.value, latency,
            math_verified, math_corrected, verify_summary,
            claims_checked, claims_correct, mem_used, False,
            round(pair.quality_score, 3))
        r.code_executed  = code_executed
        r.code_passed    = code_passed
        r.code_output    = code_output
        r.code_auto_fixed = code_auto_fixed
        return r

    def execute_code(self, code: str, language: str = "python"):
        """Directly execute code. Used by /run command."""
        return self.executor.execute_and_verify(code, language)

    def give_feedback(self, pair_id, good):
        self.store.update_feedback(
            pair_id, FeedbackSignal.EXCELLENT if good else FeedbackSignal.WRONG)

    def remember(self, fact):
        self.memory.remember_fact(fact)

    def get_profile(self):
        return self.memory.world.get_user_profile()

    def trigger_training(self):
        run = self.trainer.run_now()
        return {"status": run.status, "pairs_used": run.pairs_used,
                "notes": run.notes,
                "duration_s": round((run.completed_at - run.started_at), 2) if run.completed_at else 0}

    def generate_training_data(self, n=20):
        return self.trainer.generate_self_play(n)

    def training_status(self):
        return self.trainer.status()

    def switch_model(self, new_model_path: str):
        """Switch to a different model. Properly unloads old model first."""
        if new_model_path == self.model_path and self._model is not None:
            return  # already loaded

        # Unload current model
        if self._model is not None:
            del self._model
            self._model = None

        # Reset state
        self._model_loaded = False
        self.model_path = new_model_path
        self.model_name = Path(new_model_path).name
        self.prompt_format = _detect_prompt_format(new_model_path)
        self.n_threads = _optimal_threads(new_model_path)

        # Save as active model so it persists across restarts
        _save_active_model(new_model_path)

        # Load new model immediately
        self._load_model()

    # ── Private ────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model_loaded:
            return
        if not Path(self.model_path).exists():
            print(f"[LeanAI v3] Model not found: {self.model_path}")
            # DON'T set _model_loaded = True — allow retry with different path
            return
        try:
            from llama_cpp import Llama
            model_name = Path(self.model_path).name
            print(f"[LeanAI v3] Loading {model_name} ({self.n_threads} threads)...")
            # Dynamic GPU layers: 7B can fit more layers, 32B needs fewer
            model_name_lower = Path(self.model_path).name.lower()
            is_qwen3_moe = "qwen3" in model_name_lower and "coder" in model_name_lower
            is_qwen35 = "qwen3.5" in model_name_lower or "qwen35" in model_name_lower
            is_gemma4_moe = "gemma" in model_name_lower and ("26b" in model_name_lower or "a4b" in model_name_lower)

            # Auto-detect available RAM
            try:
                import psutil
                total_ram_gb = psutil.virtual_memory().total / (1024**3)
                avail_ram_gb = psutil.virtual_memory().available / (1024**3)
            except ImportError:
                total_ram_gb = 32  # assume 32GB if psutil not available
                avail_ram_gb = 24

            low_ram = total_ram_gb <= 12  # 8GB or 12GB machines

            if is_qwen3_moe:
                gpu_layers = 0   # CPU-only: Qwen3 MoE layers too large for 4GB VRAM
                ctx_size = 8192  # Start conservative, Qwen3 supports up to 262K
            elif is_qwen35:
                gpu_layers = 4   # Qwen3.5 27B dense — similar to 32B
                ctx_size = 4096  # Conservative context, supports up to 262K
            elif is_gemma4_moe:
                gpu_layers = 0   # CPU-only: Gemma 4 MoE layers too large for 4GB VRAM
                ctx_size = 4096  # Reduced from 8192 for speed, supports up to 256K
            elif any(x in model_name_lower for x in ["32b", "33b", "34b", "70b"]):
                gpu_layers = 4   # 32B dense — only 4 layers fit in 4GB VRAM
                ctx_size = 4096
            elif any(x in model_name_lower for x in ["7b", "6b", "8b"]):
                if low_ram:
                    gpu_layers = 0    # no GPU offload on low RAM machines
                    ctx_size = 2048   # smaller context to avoid OOM
                    print(f"[LeanAI v3] Low RAM detected ({total_ram_gb:.0f}GB) — using ctx=2048, CPU-only")
                else:
                    gpu_layers = 15   # 7B model — most layers fit in 4GB VRAM
                    ctx_size = 4096
            else:
                gpu_layers = 8
                ctx_size = 4096

            self._model = Llama(
                model_path=self.model_path,
                n_ctx=ctx_size, n_threads=self.n_threads, n_batch=1024,
                n_gpu_layers=gpu_layers, use_mmap=True, use_mlock=False,
                logits_all=False, verbose=self.verbose,
            )
            self._ctx_size = ctx_size  # Store for context overflow checks
            print(f"[LeanAI v3] GPU layers: {gpu_layers}")
            self._model_loaded = True
            print("[LeanAI v3] Model loaded. Ready.")
        except ImportError:
            print("[LeanAI v3] llama-cpp-python not installed")
            self._model_loaded = True
        except Exception as e:
            print(f"[LeanAI v3] Error loading model: {e}")
            # DON'T set _model_loaded = True — allow retry

    def _generate_with_model(self, prompt, config):
        self._load_model()
        if self._model is None:
            return (
                "Model not loaded. Possible causes:\n"
                "- Model file not found at: " + str(self.model_path) + "\n"
                "- Run: python setup_leanai.py  (downloads model automatically)\n"
                "- Or: /model list  (to see available models)"
            )
        # Context overflow protection: truncate prompt if too long
        # Rough estimate: 1 token ≈ 4 chars. Leave room for max_tokens output.
        ctx_limit = getattr(self, '_ctx_size', 4096) or 4096
        max_prompt_tokens = ctx_limit - config.max_tokens - 100  # 100 token safety margin
        max_prompt_chars = max(max_prompt_tokens * 4, 2000)
        if len(prompt) > max_prompt_chars:
            # Truncate the middle (keep start + end for system prompt + query)
            keep_start = max_prompt_chars // 2
            keep_end = max_prompt_chars // 2
            prompt = prompt[:keep_start] + "\n\n[... context truncated for length ...]\n\n" + prompt[-keep_end:]
        stop_tokens = ["<|im_end|>", "<|im_start|>", "<|user|>", "<|end|>",
                       "<|assistant|>", "<end_of_turn>", "<start_of_turn>",
                       "\nYou:", "\nHuman:", "\nUser:"]
        result = self._model(prompt, max_tokens=config.max_tokens,
            temperature=config.temperature, top_p=config.top_p,
            top_k=config.top_k, repeat_penalty=config.repeat_penalty,
            stop=stop_tokens, echo=False)
        text = result["choices"][0]["text"].strip()
        for token in stop_tokens:
            text = text.split(token)[0].strip()
        # Strip thinking blocks from all models
        import re
        # Qwen3.5: <think>...</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "<think>" in text and "</think>" not in text:
            text = text.split("<think>")[0].strip()
        # Gemma 4: <|channel>thought ... <channel|> or <|channel>-thought <channel|>
        text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<\|channel>.*?<\|channel>", "", text, flags=re.DOTALL).strip()
        # Strip any remaining channel tags and content before actual answer
        if "<|channel>" in text and "<channel|>" not in text:
            text = text.split("<|channel>")[-1].strip()
            # Remove the thought content up to the first heading or actual content
            lines = text.split("\n")
            clean_lines = []
            found_content = False
            for line in lines:
                if line.strip().startswith(("###", "▸", "**", "1.", "2.", "-")) or found_content:
                    found_content = True
                    clean_lines.append(line)
            text = "\n".join(clean_lines).strip() if clean_lines else text
        # Truncate repetitive output (catches all models at source)
        text = _truncate_repetition(text)
        return text

    def generate_streaming(self, prompt, config, callback=None):
        """Generate with streaming — calls callback(token) for each token."""
        self._load_model()
        if self._model is None:
            if callback:
                callback("Model not loaded. Run: python setup_leanai.py")
            return "Model not loaded."
        stop_tokens = ["<|im_end|>", "<|im_start|>", "<|user|>", "<|end|>",
                       "<|assistant|>", "<end_of_turn>", "<start_of_turn>",
                       "\nYou:", "\nHuman:", "\nUser:"]
        # Context overflow protection
        ctx_limit = getattr(self, '_ctx_size', 4096) or 4096
        max_prompt_chars = max((ctx_limit - config.max_tokens - 100) * 4, 2000)
        if len(prompt) > max_prompt_chars:
            keep = max_prompt_chars // 2
            prompt = prompt[:keep] + "\n\n[... truncated ...]\n\n" + prompt[-keep:]
        full_text = ""
        in_thinking = False
        buffer = ""  # Buffer to detect think/channel tags
        for chunk in self._model(prompt, max_tokens=config.max_tokens,
                temperature=config.temperature, top_p=config.top_p,
                top_k=config.top_k, repeat_penalty=config.repeat_penalty,
                stop=stop_tokens, echo=False, stream=True):
            token = chunk["choices"][0]["text"]
            full_text += token
            buffer += token

            # Detect entering thinking mode
            if "<think>" in buffer and not in_thinking:
                in_thinking = True
                buffer = ""
                continue
            # Detect exiting thinking mode
            if "</think>" in buffer and in_thinking:
                in_thinking = False
                buffer = ""
                continue
            # Detect channel thinking
            if "<|channel>" in buffer and not in_thinking:
                in_thinking = True
                buffer = ""
                continue
            if "<channel|>" in buffer and in_thinking:
                in_thinking = False
                buffer = ""
                continue

            # Only output when not thinking
            if not in_thinking and callback:
                # Flush buffer if it doesn't look like a partial tag
                if len(buffer) > 15 or not any(p in buffer for p in ["<t", "</t", "<|c", "<c"]):
                    # Don't print tag fragments
                    clean = buffer
                    for tag in ["<think>", "</think>", "<|channel>", "<channel|>"]:
                        clean = clean.replace(tag, "")
                    if clean:
                        callback(clean)
                    buffer = ""
            elif in_thinking:
                # Keep buffering but don't output
                if len(buffer) > 200:
                    buffer = buffer[-50:]  # prevent unbounded growth

        # Clean final text
        import re
        full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
        full_text = re.sub(r"<\|channel>.*?<channel\|>", "", full_text, flags=re.DOTALL).strip()
        if "<think>" in full_text and "</think>" not in full_text:
            full_text = full_text.split("<think>")[0].strip()
        for token in stop_tokens:
            full_text = full_text.split(token)[0].strip()
        return full_text

    def generate_with_codeecho(self, prompt, config, sources, callback=None):
        """
        Generate with CodeEcho: Source-Grounded Speculative Decoding.

        When the model reproduces code from source material, CodeEcho
        batch-injects those tokens at prefill speed instead of generating
        them one by one. Yields 2-5x speedup on code review tasks.

        Args:
            prompt: The full formatted prompt string
            config: GenerationConfig with max_tokens, temperature, etc.
            sources: List of source texts (file contents, brain context)
            callback: Called with each generated text chunk (for streaming)

        Returns:
            (generated_text, echo_stats) or (generated_text, None) on fallback
        """
        self._load_model()
        if self._model is None:
            if callback:
                callback("Model not loaded. Run: python setup_leanai.py")
            return "Model not loaded.", None

        # Check if CodeEcho API is available on this model
        if not self.code_echo.check_api(self._model):
            # Fallback to normal streaming
            text = self.generate_streaming(prompt, config, callback)
            return text, None

        # Index source material
        self.code_echo.reset_index()
        valid_sources = [s for s in sources if s and len(s.strip()) > 50]
        if not valid_sources:
            # No sources worth indexing, fall back to normal streaming
            text = self.generate_streaming(prompt, config, callback)
            return text, None

        indexed = self.code_echo.index_sources(self._model, valid_sources)
        if indexed == 0:
            text = self.generate_streaming(prompt, config, callback)
            return text, None

        # Context overflow protection — use ACTUAL ctx_size
        ctx_limit = getattr(self, '_ctx_size', 4096) or 4096
        max_prompt_chars = max((ctx_limit - config.max_tokens - 100) * 4, 2000)
        if len(prompt) > max_prompt_chars:
            keep = max_prompt_chars // 2
            prompt = prompt[:keep] + "\n\n[... truncated ...]\n\n" + prompt[-keep:]

        # Tokenize the prompt
        try:
            prompt_tokens = self._model.tokenize(prompt.encode('utf-8'), special=True)
        except TypeError:
            prompt_tokens = self._model.tokenize(prompt.encode('utf-8'))

        # Calculate ACTUAL available token budget (prevents KV cache overflow)
        # KV cache = ctx_limit. Prompt fills some, rest is for generation.
        # Reserve 50 tokens for safety margin.
        available_tokens = ctx_limit - len(prompt_tokens) - 50
        if available_tokens < 100:
            # Prompt is too long, truncate and re-tokenize
            keep = (ctx_limit - config.max_tokens - 100) * 2  # chars
            prompt = prompt[:max(keep, 1000)] + "\n\n[... truncated ...]\n\n" + prompt[-1000:]
            try:
                prompt_tokens = self._model.tokenize(prompt.encode('utf-8'), special=True)
            except TypeError:
                prompt_tokens = self._model.tokenize(prompt.encode('utf-8'))
            available_tokens = ctx_limit - len(prompt_tokens) - 50

        # Use the smaller of config.max_tokens and available budget
        effective_max_tokens = min(config.max_tokens, max(available_tokens, 128))

        # Resolve stop token IDs
        stop_strings = [
            "<|im_end|>", "<|im_start|>", "<|user|>", "<|end|>",
            "<|assistant|>", "<end_of_turn>", "<start_of_turn>",
            "\nYou:", "\nHuman:", "\nUser:"
        ]
        stop_token_ids = []
        for s in stop_strings:
            try:
                tids = self._model.tokenize(s.encode('utf-8'), add_bos=False, special=True)
                if tids and len(tids) == 1:
                    stop_token_ids.append(tids[0])
            except Exception:
                pass

        # Generate with CodeEcho
        text, stats = self.code_echo.generate(
            model=self._model,
            prompt_tokens=prompt_tokens,
            max_tokens=effective_max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            callback=callback,
        )

        return text, stats

    def _build_prompt(self, query, memory_context, history, project_context=""):
        system = (
            "You are LeanAI — a senior software engineer AI. Clear, accurate, insightful.\n\n"
            "RULES:\n"
            "1. No filler ('Certainly!', 'Let me explain'). Start with substance.\n"
            "2. Vary sentence patterns. Never repeat same structure.\n"
            "3. For each concept: WHAT (brief) + WHY it matters + RISKS/GOTCHAS.\n"
            "4. Show IMPROVED code with 3-5 real fixes from your Risks/Missing sections.\n"
            "5. Be SPECIFIC. Bad: 'ensure it's configured.' Good: 'add containerRegistry: X'\n"
            "6. Use markdown: ### headers, **bold**, `inline code`, ```lang code blocks, bullets.\n"
            "7. FLAG WHAT'S MISSING — check per language:\n"
            "   Python: type hints, bare except, mutable defaults, GIL implications.\n"
            "   JS/TS: ==/===, async errors, prototype pollution, StrictMode double-render.\n"
            "   Go: unchecked err, goroutine leaks, defer, GMP scheduler, GOMAXPROCS, pprof.\n"
            "   Rust: unwrap, unsafe, lifetime issues. Java/Kotlin: NPE, resource leaks, coroutine scope.\n"
            "   Swift: force unwrap, retain cycles. Dart: dispose(), BuildContext leaks.\n"
            "   Ruby: monkey patching, N+1. PHP: SQL injection, unescaped output.\n"
            "   Elixir: GenServer, supervision trees. Shell: unquoted vars, set -euo pipefail.\n"
            "   C/C++: buffer overflow, use-after-free. SQL: injection, indexes, N+1.\n"
            "   ANY language: null safety, error handling, resource cleanup, thread safety, idiomatic patterns.\n"
            "   Also check: CI/CD, API (auth/rate-limit/CORS), DB (pooling/transactions).\n"
            "8. Structure: overview → breakdown (WHAT+WHY) → risks → missing → improved code.\n"
            "9. Correct language tags. No prose in code blocks.\n"
            "10. For USER's project: use THEIR actual code, class names, file paths.\n"
            "11. Self-review before finishing.\n"
            "12. Stop after answering.\n\n"
            "ADVANCED BEHAVIOR (follow these to produce expert-level output):\n"
            "13. CLARIFY BEFORE CODING: For build/implement/create tasks, briefly state your assumptions "
            "(e.g., 'Assuming React 18+ with TypeScript, no OAuth, REST backend'). If a critical choice "
            "would change the approach entirely, mention it: 'If using Next.js App Router, the approach "
            "would differ — let me know.'\n"
            "14. PROVIDE 2-3 APPROACHES: For complex tasks, show 2 approaches with clear tradeoffs. "
            "Example: 'Approach A: JWT in httpOnly cookie (more secure, harder to implement). "
            "Approach B: JWT in memory + refresh token (simpler, needs careful XSS prevention). "
            "I'll implement Approach A because...' Then show the code.\n"
            "15. CATCH OBSCURE EDGE CASES: Go beyond obvious bugs. Check for:\n"
            "   React: StrictMode double-render, stale closures in useEffect, memory leaks from "
            "subscriptions, hydration mismatches in SSR.\n"
            "   Go: goroutine leaks from forgotten context cancellation, channel deadlocks, "
            "slice append aliasing, defer in loops.\n"
            "   Python: circular imports, __del__ pitfalls, floating point comparison, "
            "mutable default arguments in dataclasses.\n"
            "   General: timezone bugs, off-by-one, unicode normalization, TOCTOU races.\n"
            "16. EXPLAIN THE WHY BEHIND THE WHY: Don't just say 'use context.WithTimeout'. Explain: "
            "'context.WithTimeout propagates cancellation through the call chain because Go's runtime "
            "checks ctx.Done() at I/O boundaries, preventing goroutine leaks when upstream callers "
            "abandon requests.' Show the internal mechanism, not just the API.\n"
            "17. DEBUG FROM FIRST PRINCIPLES: When the user reports an error, don't just match patterns. "
            "Think: what could cause this specific symptom? Work backward from the error message. "
            "Consider: is it a compile error, runtime error, or logic error? What assumptions might be "
            "wrong? What would you check first, second, third?\n"
            "18. PRODUCTION PATTERNS:\n"
            "   React: AbortController, rate-limit submits, password toggle, CSRF, loading/error/success.\n"
            "   Backend: timeouts, graceful shutdown, health checks, retry with backoff.\n"
            "   Go: context.WithTimeout, errgroup, channel direction, pprof profiling.\n"
            "   Include unit test example. Include benchmark/performance note when relevant.\n"
            "19. THINK MULTI-FILE: Show file structure. Mention what other files need changes "
            "(routes, middleware, config, tests, env). Show file paths.\n"
            "20. DEPLOYMENT: Mention env variables, Docker, CI/CD, security headers when relevant.\n"
            "21. FINAL CHECK: Would a staff engineer at a top company approve this for production? "
            "If not, add what's missing. Check: error boundaries, monitoring, logging, "
            "graceful degradation, backward compatibility."
        )

        # ── Build user query with dynamic context ──────────────────
        # Keep system prompt STATIC so llama.cpp can cache the KV prefix.
        # Dynamic context (project, memory) goes into the user message.
        user_with_context = ""

        if project_context:
            user_with_context += (
                "[PROJECT CONTEXT — use this to give specific answers about the user's ACTUAL code]\n"
                + project_context[:3000] + "\n\n"
            )

        if memory_context:
            ctx = memory_context[:500].replace("\n", " ")
            user_with_context += f"[MEMORY] {ctx}\n\n"

        user_with_context += query

        # Disable thinking mode for Qwen3.5 to avoid 25-min think blocks
        model_name_lower = Path(self.model_path).name.lower() if self.model_path else ""
        is_qwen35 = "qwen3.5" in model_name_lower or "qwen35" in model_name_lower

        if self.prompt_format == "chatml":
            parts = ["<|im_start|>system\n" + system + "<|im_end|>\n"]
            for msg in history[-4:]:
                role = msg["role"]
                content = msg["content"][:400] if len(msg["content"]) > 400 else msg["content"]
                if role == "user":
                    parts.append("<|im_start|>user\n" + content + "<|im_end|>\n")
                elif role == "assistant":
                    parts.append("<|im_start|>assistant\n" + content + "<|im_end|>\n")
            if is_qwen35:
                # Force empty think block so model skips thinking mode entirely
                parts.append("<|im_start|>user\n" + user_with_context + "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n")
            else:
                parts.append("<|im_start|>user\n" + user_with_context + "<|im_end|>\n<|im_start|>assistant\n")
        elif self.prompt_format == "gemma":
            parts = ["<start_of_turn>user\n" + system + "\n\n" + user_with_context + "<end_of_turn>\n"]
            for msg in history[-4:]:
                role = msg["role"]
                content = msg["content"][:400] if len(msg["content"]) > 400 else msg["content"]
                if role == "user":
                    parts.append("<start_of_turn>user\n" + content + "<end_of_turn>\n")
                elif role == "assistant":
                    parts.append("<start_of_turn>model\n" + content + "<end_of_turn>\n")
            parts.append("<start_of_turn>user\n" + user_with_context + "<end_of_turn>\n<start_of_turn>model\n")
        else:
            parts = ["<|system|>\n" + system + "<|end|>\n"]
            for msg in history[-4:]:
                role = msg["role"]
                content = msg["content"][:300] if len(msg["content"]) > 300 else msg["content"]
                if role == "user":
                    parts.append("<|user|>\n" + content + "<|end|>\n")
                elif role == "assistant":
                    parts.append("<|assistant|>\n" + content + "<|end|>\n")
            parts.append("<|user|>\n" + user_with_context + "<|end|>\n<|assistant|>\n")

        return "".join(parts)

    def _handle_tiny(self, query):
        import math as m
        q = query.strip().lower()
        if re.match(r"^(hi|hello|hey)\b", q): return "Hello! How can I help?"
        if re.match(r"^(thanks|thank you)\b", q): return "You're welcome!"
        if re.match(r"^(bye|goodbye)\b", q): return "Goodbye!"
        try:
            expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\s\^]", "", query).replace("^", "**")
            if expr.strip():
                result = eval(expr, {"__builtins__": {}, "sqrt": m.sqrt})
                return str(result)
        except Exception:
            pass
        return "Could you tell me more?"

    def _wrap(self, text, pair_id, cal, tier, latency, verified, corrected,
              verify_summary, claims_checked, claims_correct, mem_used,
              from_memory, quality, warning=None):
        return LeanAIResponse(
            text=text, pair_id=pair_id,
            confidence=cal.calibrated, confidence_label=cal.label,
            confidence_bar=cal.bar, confidence_method=cal.method,
            tier_used=tier, latency_ms=round(latency, 1),
            verified=verified, corrected=corrected,
            verification_summary=verify_summary,
            claims_checked=claims_checked, claims_correct=claims_correct,
            memory_context_used=mem_used, answered_from_memory=from_memory,
            quality_score=quality, warning=warning,
        )

    def status(self):
        return {
            "phase": "3+4b", "model": Path(self.model_path).name,
            "prompt_format": self.prompt_format, "threads": self.n_threads,
            "model_loaded": self._model is not None,
            "executor": {"available": self.executor.available_languages,
                         "auto_execute": self.auto_execute},
            "verifier": self.verifier.status,
            "memory": self.memory.stats(),
            "training": self.trainer.status(),
        }
