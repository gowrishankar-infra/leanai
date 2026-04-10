"""
LeanAI Phase 3+4b Engine
- Qwen2.5 Coder 7B (chatml) + Phi-3 (phi3) auto-detection
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
    config_file = Path.home() / ".leanai" / "active_model.txt"
    if config_file.exists():
        path = config_file.read_text().strip()
        if Path(path).exists():
            return path
    return str(Path.home() / ".leanai" / "models" / "phi3-mini-q4.gguf")


def _detect_prompt_format(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "qwen" in name: return "chatml"
    if "phi" in name: return "phi3"
    if "llama" in name: return "llama3"
    return "chatml"


def _optimal_threads(model_path: str) -> int:
    name = Path(model_path).name.lower()
    cpu = os.cpu_count() or 8
    if any(x in name for x in ["7b", "6b", "8b"]): return min(cpu, 16)
    if any(x in name for x in ["33b", "34b"]): return min(cpu, 8)
    return min(cpu, 8)


def _looks_like_python(code: str) -> bool:
    """Check if code looks like Python (not YAML, JSON, bash, SQL, markdown, etc)."""
    first_lines = code.strip().split("\n")[:5]
    first = first_lines[0].strip() if first_lines else ""

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

    # ── Private ────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model_loaded:
            return
        if not Path(self.model_path).exists():
            print(f"[LeanAI v3] Model not found: {self.model_path}")
            self._model_loaded = True
            return
        try:
            from llama_cpp import Llama
            model_name = Path(self.model_path).name
            print(f"[LeanAI v3] Loading {model_name} ({self.n_threads} threads)...")
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=4096, n_threads=self.n_threads, n_batch=1024,
                n_gpu_layers=0, use_mmap=True, use_mlock=False,
                logits_all=False, verbose=self.verbose,
            )
            self._model_loaded = True
            print("[LeanAI v3] Model loaded. Ready.")
        except ImportError:
            self._model_loaded = True

    def _generate_with_model(self, prompt, config):
        self._load_model()
        if self._model is None:
            return "Demo mode — run: python setup.py --download-model --model qwen25-coder"
        stop_tokens = ["<|im_end|>", "<|im_start|>", "<|user|>", "<|end|>",
                       "<|assistant|>", "\nYou:", "\nHuman:", "\nUser:"]
        result = self._model(prompt, max_tokens=config.max_tokens,
            temperature=config.temperature, top_p=config.top_p,
            top_k=config.top_k, repeat_penalty=config.repeat_penalty,
            stop=stop_tokens, echo=False)
        text = result["choices"][0]["text"].strip()
        for token in stop_tokens:
            text = text.split(token)[0].strip()
        return text

    def _build_prompt(self, query, memory_context, history, project_context=""):
        system = (
            "You are LeanAI — a senior software engineer AI assistant running locally. "
            "You give clear, accurate, and deeply insightful answers.\n\n"
            "RESPONSE QUALITY RULES:\n"
            "1. NO FILLER. Never start with 'Certainly!', 'Of course!', 'Great question!', "
            "'Let me explain'. Start directly with the substance.\n"
            "2. VARY YOUR LANGUAGE. Never use the same sentence pattern twice. "
            "Bad: 'This specifies X. This defines Y. This sets Z.' "
            "Good: 'The trigger fires on main branch pushes. Under the hood, the pool "
            "selects an Ubuntu runner. The first step compiles via Maven.'\n"
            "3. EXPLAIN WHY, NOT JUST WHAT. For every concept, explain:\n"
            "   - WHAT it does (briefly)\n"
            "   - WHY it matters or WHY it's done this way\n"
            "   - What RISKS or GOTCHAS exist\n"
            "   Example: Instead of 'tags: latest assigns the latest tag' say "
            "'tags: latest labels this image as the most recent build. However, relying "
            "solely on latest makes rollbacks difficult — consider using $(Build.BuildId) "
            "or semantic versioning for production.'\n"
            "4. SHOW IMPROVED CODE. When explaining code or config, after your explanation, "
            "show an improved/corrected version with your recommendations applied. "
            "Use a section header like '### Improved version' followed by a proper code block.\n"
            "5. SPECIFIC SUGGESTIONS, NOT GENERIC. "
            "Bad: 'Ensure the registry is correctly configured.' "
            "Good: 'Add containerRegistry: myDockerHub as an input, and create a Docker "
            "Registry service connection in Project Settings > Service Connections.'\n"
            "6. FORMAT WITH MARKDOWN:\n"
            "   - Use ### headers to separate sections\n"
            "   - Use **bold** for key terms on first mention\n"
            "   - Use `inline code` for file names, commands, config keys\n"
            "   - Use ```language for code blocks with correct language tag\n"
            "   - Use bullet points for lists of items\n"
            "   - Use > blockquotes for warnings or important notes\n"
            "7. FLAG WHAT'S MISSING. If the code/config is incomplete or has risks, "
            "explicitly call them out in a '### What\\'s missing' or '### Risks' section. "
            "Be specific about what could break and how to fix it.\n"
            "8. KEEP IT STRUCTURED. Use this pattern for explanations:\n"
            "   - Brief overview (2-3 sentences)\n"
            "   - Section-by-section breakdown\n"
            "   - What to watch out for (specific risks)\n"
            "   - Improved version (if applicable)\n"
            "9. CODE BLOCKS must use the correct language tag: ```python, ```yaml, ```bash, "
            "```javascript, ```sql, ```dockerfile, etc. Never put prose inside code blocks. "
            "Never leak HTML or formatting artifacts into code.\n"
            "10. If the question is about the USER'S project (context below), give answers "
            "about THEIR actual code with real class names and file paths — never generic examples.\n"
            "11. Stop after answering. No follow-up questions unless genuinely needed."
        )

        # Inject project context (from smart context — brain, git, sessions)
        if project_context:
            system += (
                "\n\nIMPORTANT — The user's ACTUAL project context is below. "
                "Use it to give specific answers about THEIR code, not generic examples:\n"
                + project_context[:1500]
            )

        # Inject memory context
        if memory_context:
            ctx = memory_context[:500].replace("\n", " ")
            system += "\nMemory: " + ctx

        if self.prompt_format == "chatml":
            parts = ["<|im_start|>system\n" + system + "<|im_end|>\n"]
            for msg in history[-4:]:
                role = msg["role"]
                content = msg["content"][:400] if len(msg["content"]) > 400 else msg["content"]
                if role == "user":
                    parts.append("<|im_start|>user\n" + content + "<|im_end|>\n")
                elif role == "assistant":
                    parts.append("<|im_start|>assistant\n" + content + "<|im_end|>\n")
            parts.append("<|im_start|>user\n" + query + "<|im_end|>\n<|im_start|>assistant\n")
        else:
            parts = ["<|system|>\n" + system + "<|end|>\n"]
            for msg in history[-4:]:
                role = msg["role"]
                content = msg["content"][:300] if len(msg["content"]) > 300 else msg["content"]
                if role == "user":
                    parts.append("<|user|>\n" + content + "<|end|>\n")
                elif role == "assistant":
                    parts.append("<|assistant|>\n" + content + "<|end|>\n")
            parts.append("<|user|>\n" + query + "<|end|>\n<|assistant|>\n")

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
