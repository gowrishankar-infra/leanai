# LeanAI

> **Created by [Gowri Shankar](https://github.com/gowrishankar-infra)** — Built with Claude as coding partner.
> Licensed under [AGPL-3.0](LICENSE) — free to use, must credit author, modifications must be open-sourced.

**The AI that knows your code. Runs on your machine. Gets smarter every day.**

LeanAI is a fully local, project-aware AI coding system. Unlike cloud AI that forgets you between sessions and never sees your full codebase, LeanAI permanently understands your project structure, remembers every conversation, verifies its own code, and learns from your coding patterns over time.

**No API keys. No subscriptions. No data leaves your machine. Ever.**

---

## Why LeanAI?

Every AI coding tool today shares the same flaw: they see your code for the first time, every time. You paste a snippet, explain the context, get an answer, close the tab — and next session, start over from zero.

LeanAI is different:

- **It knows your entire codebase.** Full AST analysis, dependency graph across every language (Python, JS, TS, Go, Rust, Java, C/C++, C#, and more). When you say "add authentication to the API," it already knows every route, every model, every middleware.

- **Hybrid semantic + BM25 + graph retrieval.** When you ask a question, LeanAI runs three retrievers in parallel — MiniLM embeddings for conceptual search, BM25 for exact-name matches, and a graph retriever that walks your call graph. Results are fused with reciprocal rank fusion and re-ranked with code-specific heuristics. Cloud AI doesn't have your AST + call graph — so this is one thing a 4GB-VRAM local tool can genuinely beat cloud AI at.

- **It finds security bugs in YOUR code.** Built-in security analyzer (Sentinel) scans for 12 OWASP vulnerability classes. Chain analyzer (ChainBreaker) walks the dependency graph forward to see whether tainted input can actually reach a high-value sink. Safe proof-of-concept demos (ExploitForge) for confirmed findings. Zero false-positive chains on LeanAI's own codebase after M2.2 hardening.

- **Deterministic code archaeology.** SourceForensics uses `git log -L` + Python AST to answer exact questions about any function: when was it born, who wrote it, stability score, churn rate, co-evolution, dead-code sweep. No LLM involved. Sub-second per query. Never hallucinates a commit hash.

- **It never forgets.** Session 1's decisions are available in session 5. Every conversation is permanently searchable. Your name, your preferences, your project history — all remembered.

- **It proves its code works.** The TDD loop generates code, runs tests, reads errors, fixes bugs, and loops until every test passes. The output is verified correct, regardless of model size.

- **It gets smarter from YOUR code.** Every interaction collects training data. After enough examples, QLoRA fine-tuning makes the model learn YOUR naming conventions, YOUR patterns, YOUR preferred libraries.

- **Sub-2ms autocomplete.** Indexes every function in your project. Type a prefix, get completions from YOUR codebase — no model call needed.

- **Semantic git bisect.** "Find which commit broke auth" — AI analyzes commits semantically and predicts the culprit with reasoning.

- **Adversarial code verification.** Generates edge-case inputs designed to break your code, finds bugs, suggests fixes. All in under 1 second.

---

## Quick Start

### One-Command Setup

```bash
git clone https://github.com/gowrishankar-infra/leanai.git
cd leanai
python setup_leanai.py        # Windows
# python3 setup_leanai.py     # Linux/Mac
```

That's it. The setup script checks your system, creates a virtual environment if needed, installs dependencies, downloads the 7B model (4.4 GB), and launches LeanAI.

### Manual Setup (if you prefer)

```bash
git clone https://github.com/gowrishankar-infra/leanai.git
cd leanai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download a Model

```bash
# Fast model (7B, 4.5 GB, ~30s responses)
python download_models.py qwen-7b

# Best quality (Qwen3.5 27B, 16.7 GB, best dense model, thinking mode)
python download_models.py qwen35-27b

# Fastest quality (Gemma 4 26B MoE, 16.9 GB, rock solid, best for UI/frontend)
python download_models.py gemma4-26b

# Fast quality (Qwen3 30B MoE, 18.6 GB, ~2min responses)
python download_models.py qwen3-coder

# Legacy (Qwen2.5 32B, 18 GB, ~5min responses)
python download_models.py qwen-32b
```

### Enable GPU Acceleration (optional, recommended)

```bash
# Install Vulkan SDK from https://vulkan.lunarg.com/sdk/home
# Then rebuild llama-cpp-python with Vulkan:
$env:CMAKE_ARGS="-DGGML_VULKAN=ON"          # Windows PowerShell
# export CMAKE_ARGS="-DGGML_VULKAN=ON"      # Linux/Mac
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

This gives **3.5x speedup** on any GPU (NVIDIA, AMD, Intel). LeanAI auto-detects your GPU and offloads layers dynamically.

### Run

```bash
# CLI
python main.py

# Web UI (localhost:8000)
python run_server.py

# VS Code extension
# See vscode-extension/README.md
```

### ⚡ First Thing to Do (important — don't skip this)

Without these steps, LeanAI is just a generic chatbot. WITH these steps, it knows your entire codebase:

```
/brain .              # scan your project + auto-index for semantic retrieval
/model auto           # auto-routes: frontend → Gemma 4, backend → Qwen3.5, simple → 7B
```

`/brain .` does two things: scans every file with AST analysis and builds the dependency graph, then auto-indexes each function/class/method as a semantic chunk (ChromaDB + MiniLM embeddings). First run takes 10-20 seconds on a medium codebase.

Now ask about YOUR code:
```
explain how the auth system works
write a function to validate user tokens
/ask how does the cache invalidate stale entries?
/sentinel
/forensics ProjectBrain.__init__
```

LeanAI will reference YOUR actual functions, correct wrong imports against YOUR AST, and inject YOUR coding patterns as examples. **This is where it beats cloud AI — try it on YOUR project.**

---

## Features

### Three Interfaces

**CLI** — 45+ commands, full power

```
  ▶ /brain .
  [M6] Auto-indexing for semantic retrieval (AST-grounded chunking active)...
  Scanned 110 files | 2,150 functions | 395 classes | 14,799 edges
  Indexed 110 files into 3,100 semantic chunks

  ▶ /complete gen
  Completions for 'gen' (0.8ms):
    ƒ generate()                    core/engine_v3.py
    ƒ generate_changelog()          brain/git_intel.py
    ◆ GenerationConfig              core/engine.py

  ▶ /ask --raw where is rescan_file defined
  [100%] brain/project_brain.py:195-216  ProjectBrain.rescan_file
         reasons: ['name_match:rescan,file', 'doc_match:file']
  [88%]  brain/project_brain.py:218-224  ProjectBrain._hash_file

  ▶ /forensics ProjectBrain.__init__
  Genesis:   2026-04-09 by Gowri Shankar (commit 285ffc7)
  Stability: 41/100 — active change rate
  Authors:   Gowri Shankar (100%, bus factor: 1)

  ▶ /sentinel
  Found 37 potential issues: 13 MEDIUM, 24 LOW across 110 files (2.6s)

  ▶ /fuzz def sort(arr): return sorted(arr)
  Tested: 12 | Passed: 9 | Failed: 3
    ✗ None → TypeError
    ✗ [1, 'a', 2.0] → TypeError
```

**Web UI** — 6 modes at localhost:8000

Chat, Swarm Consensus, Run Code, TDD, Brain Scan, Git Intelligence — all in your browser. 32 API endpoints with interactive docs at `/docs`.

**VS Code Extension** — 11 commands + inline autocomplete

`Ctrl+Shift+L` to chat. Right-click to explain, fix, or test code. Inline autocomplete from your project brain. Status bar shows connection. Professional chat panel with syntax highlighting, copy buttons, and markdown rendering.

---

### Project Brain

Scans your entire codebase with deep analysis. Python uses full AST parsing; JavaScript, TypeScript, Go, Rust, Java, C/C++, C#, Ruby, PHP, and SQL use regex-based parsing. Builds a full dependency graph across all languages.

```
/brain .                          # scan current project (any language)
/describe core/engine_v3.py       # what's in this file?
/find generate                    # find a function across project
/deps core/engine_v3.py           # what depends on this file?
/impact main.py                   # if I change this, what breaks?
```

Supported languages: Python (AST), JavaScript, TypeScript, Go, Rust, Java, C/C++, C#, Swift, Kotlin, Dart, Ruby, PHP, Elixir, Lua, R, Julia, Zig, Nim, Shell/Bash, SQL — plus generic fallback for any other language.

### Git Intelligence

Reads your entire git history. Knows what changed, when, and why.

```
/git activity         # what happened this week (grouped by date)
/git hotspots         # most frequently changed files (with bar chart)
/git history main.py  # commit history for a specific file
/git why main.py      # commit messages explaining why it changed
/git changelog        # auto-categorized changelog (features/fixes/other)
/git func generate    # when was this function last changed
```

### Swarm Consensus

Three independent model passes vote on the answer. When all three agree: UNANIMOUS, 95% confidence.

```
/swarm What is the time complexity of quicksort?
Swarm: 3 passes | UNANIMOUS | Confidence: 95%
```

### TDD Auto-Fix Loop

You write a failing test. LeanAI writes code until it passes. Every piece of output is proven correct.

```
/tdd from calculator import add, subtract
     def test_add(): assert add(2, 3) == 5
     def test_subtract(): assert subtract(10, 3) == 7

TDD Loop: PASSED after 1 attempt | calculator.py generated
```

### Multi-File Refactoring

Rename a symbol across your entire project with preview before applying.

```
/refs greet           # find all references (definitions, imports, calls)
/rename greet say_hi  # rename across every file (with confirmation)
```

### Deep Reasoning (3-Pass)

Chain-of-thought → self-critique → refinement. Significantly better answers for complex questions.

```
/reason Why do large language models hallucinate?
[Pass 1/3] Chain-of-thought reasoning...
[Pass 2/3] Self-critique (finding flaws)...
[Pass 3/3] Refining answer...
```

### Professional Writing (4-Pass)

Analyze → outline → draft → edit. Auto-detects document type (essay, report, blog, email, proposal, README, docs).

```
/write blog post about why local AI will replace cloud AI
/essay The impact of AI on software engineering
/report Q3 performance analysis
```

### Continuous Fine-Tuning

Your model literally gets smarter from your code every day. Every interaction auto-collects training data. When you have enough examples, QLoRA fine-tuning produces a personalized LoRA adapter.

```
/finetune status      # check readiness (data count, GPU, dependencies)
/finetune collect     # import from sessions + training exports
/finetune train       # start training (or export for Colab)
/finetune adapters    # list trained adapters
/finetune activate X  # use a trained adapter
```

### Auto Model Switching

Smart routing based on query type. Frontend queries (React, CSS, UI) → Gemma 4. Backend/architecture → Qwen3.5. Simple questions → 7B. Fully automatic.

```
/model auto           # smart routing by query type
/model qwen35-27b     # use Qwen3.5 27B (best backend/reasoning)
/model gemma4-26b     # use Gemma 4 26B MoE (best frontend/UI, fastest)
/model qwen3-coder    # use Qwen3 Coder 30B MoE
/model quality        # always use best model
/model fast           # always use fastest model
/model list           # see all available models
```

In auto mode, LeanAI detects what you're building:

| Query type | Routes to | Speed | Why |
|-----------|-----------|-------|-----|
| "build a React form" | Gemma 4 26B | ~5 min | Best frontend/UI code |
| "design microservice architecture" | Qwen3.5 27B | ~3-4 min | Best reasoning |
| "what is Python" | 7B | ~30s | Simple, fast |

### Smart Context

Every query is automatically enriched with relevant context from your project brain, git history, past sessions, and HDC memory cache. The model sees context that cloud AI can never access.

### Sub-2ms Autocomplete

Indexes every function and class in your project. Completions come from YOUR actual codebase — no model call needed.

```
/complete gen          → generate(), generate_changelog(), GenerationConfig
/complete Model        → ModelManager, ModelInfo, ModelInfo.resolved_path()
/complete engine.gen   → dot notation works — filters to 'gen' prefix
```

0.8ms response time. 2,899 functions, 305 classes indexed.

### Semantic Git Bisect

AI reads each commit semantically and predicts which one introduced a bug — with reasoning.

```
/bisect authentication stopped working

Most likely culprit:
  b7b3f51 — VS Code extension + path separator fix
  Suspicion: 45%
  Reasoning: includes path changes that could affect auth flow
```

Nobody else has this. Traditional git bisect is manual binary search. This is AI reasoning.

### Adversarial Code Verification

Generates edge-case inputs designed to BREAK your code. Finds bugs in under 1 second.

```
/fuzz def sort(arr): return sorted(arr)

Tested: 12 | Passed: 9 | Failed: 3
  ✗ None → TypeError
  ✗ [1, None, 3] → TypeError
  ✗ [1, 'a', 2.0] → TypeError

Suggested fixes:
  → Add None check
  → Add type validation
```

### Cross-Session Evolution Tracking

Tracks how your understanding evolves across sessions. Detects themes (database, auth, API, testing) and predicts what you'll need next.

```
/evolution narrative    → your project's story across sessions
/evolution insights     → where each theme is heading
/evolution predict      → what you'll likely ask about next
```

### Two-Pass Code Quality

Every code response goes through a second review pass — language-specific bug detection for Python, JavaScript, Go, Rust, Java, SQL, C/C++. Catches bugs the first pass misses, just like Claude's thinking mode.

### Beautiful Terminal UI

Colored ASCII art banner, syntax-highlighted code blocks with `┌─ python ─────┐` borders, styled confidence bars, purple prompt, formatted markdown output.

### Response Caching

Ask the same question twice? Instant response from cache. No model call needed. Cache persists across restarts.

```
You: what is Python?     → 25 seconds (first time)
You: what is Python?     → instant ⚡ CACHED
```

### Streaming Output

For Qwen3.5 and Qwen3 Coder models, responses stream word-by-word as they generate — just like ChatGPT/Claude. No more staring at a blank screen for 5 minutes.

### CodeEcho: Source-Grounded Speculative Decoding (NOVEL)

**No published research. No existing implementation. Built here first.**

When you ask LeanAI to review, fix, or explain code, 40-80% of the output tokens reproduce code already in your files. Each token costs a full forward pass — even though it's completely predictable. CodeEcho fixes this.

**How it works:**
1. Pre-tokenizes your source files and builds a 5-gram hash index (O(1) lookup)
2. During generation, monitors tokens for matches against your codebase
3. When 5+ consecutive tokens match, batch-injects the next N tokens via `eval()` at **prefill speed** (~10-50x faster per token than sequential decode)
4. Resumes normal generation when the model diverges from the source

```
You: review the code in engine_v3.py
📄 Reading engine_v3.py (8000 chars)
  [CodeEcho] Indexed 2 sources, 1847 n-grams

... response streams normally, with echo-accelerated sections ...

  ⚡ CodeEcho: 187/300 tokens echoed (62%) in 4 events | ~2.8x speedup
```

The "draft model" is replaced by **your own codebase** — which has near-perfect acceptance rate because the model was going to reproduce that code anyway. Zero quality loss, no extra model needed.

Check stats anytime: `/echo`

### Read Actual Files

Mention a filename in your query and LeanAI reads it automatically:

```
You: review the code in main.py
📄 Reading main.py (4000 chars)
→ Model sees the actual file content, not just metadata
```

Supports 30+ file extensions. Max 3 files per query, 4000 chars each.

### Project Onboarding

New to a codebase? Get an instant AI-generated summary:

```
/brain .
/onboard

📋 Project Onboarding Summary
LeanAI is an intelligent code analysis engine... main.py serves as
the primary entry point... Key technologies include Python... New
developers should start by examining core/engine_v3.py...
```

Uses real filenames from your project — never hallucinates fake paths.

### Custom Data Directories (LM Studio Compatible)

Use environment variables to store data anywhere or reuse existing models:

```bash
# Move all LeanAI data to a different drive
export LEANAI_HOME=/mnt/d/LeanAI/.leanai      # Linux
$env:LEANAI_HOME = "D:\LeanAI\.leanai"         # Windows

# Reuse models from LM Studio (no download needed!)
export LEANAI_MODELS=/mnt/d/LMStudio           # Linux
$env:LEANAI_MODELS = "D:\LMStudio"              # Windows
```

LeanAI recursively scans LEANAI_MODELS for `.gguf` files, so your LM Studio directory structure works out of the box.

### Code-Grounded Verification *(novel — nobody else has this)*

After every response, LeanAI fact-checks the AI's claims against your actual AST. If the model says "generate() takes 2 parameters" but your code shows 4, it auto-corrects:

```
✓ generate(): Confirmed — exists in project
✓ __init__(): Confirmed — exists in project
⚠ process_data(): Actually takes 4 parameters (self, query, config, context), not 2
```

Eliminates hallucinations about YOUR code. Cloud AI can't do this — it doesn't have your AST.

### Cascade Inference *(novel — 3x faster complex queries)*

Instead of running the 32B model from scratch (7 minutes), LeanAI uses a two-stage approach:

```
Step 1: 7B drafts the response     (~30 seconds)
Step 2: 32B reviews and corrects   (~60-90 seconds)
Total: ~2-3 minutes instead of 7 minutes
```

The 32B model is faster at reviewing than generating from scratch because it processes the draft as input (fast) and only generates corrections (small output).

### Tool-Augmented Reasoning (ReAct)

The model doesn't just think — it acts. During reasoning, it can call tools:

- `brain_lookup`: find functions, classes, files in your project
- `git_check`: see recent changes to relevant files
- `memory_search`: search past conversations

A 32B model with access to real project data produces better answers than a 200B model guessing without data.

### Mixture of Agents

For code reviews and complex analysis, LeanAI generates answers from multiple expert perspectives (security, performance, architecture, correctness) and synthesizes them into one comprehensive answer:

```
You: review the code in api/server.py
⚙ Multi-perspective analysis...
  Analyzed from 3 perspectives: correctness, security, architecture
```

Each perspective catches different issues. The synthesis is better than any single pass.

### AGAC: AST-Grounded Auto-Correction *(novel — no other tool does this)*

After every response, AGAC scans code blocks and auto-corrects hallucinated identifiers against your project's actual AST. Only fixes obvious typos (0.85 similarity threshold) — never touches prose, comments, or standard library names.

```
Model generated:   from core.services import process_user
AGAC auto-fixed:   from core.server import process_user
                   ↑ core/services.py doesn't exist, core/server.py does

▸ Code Verification (checked against your project)
  ✓ process() in agac.py — used by main, agac.py
  ✓ train() in finetune_runner.py — used by main, TestFineTuneRunner
```

**What makes this better than Opus:** Opus 4.6 would have left `core.services` in the code — it has no AST to check against. Your copy-pasted code would fail with `ModuleNotFoundError`. LeanAI's code works on the first paste.

### Sentinel: Autonomous Security Analyzer *(M1 — Mythos-inspired, AST-grounded)*

`/sentinel` runs an AST-grounded security scan on your project. It walks every function in the brain, identifies external input sources (HTTP handlers, CLI args, file reads, env vars, deserialization), identifies dangerous sinks (eval, exec, subprocess shell=True, SQL string concat, pickle.loads, weak crypto, hardcoded secrets), and pattern-matches 12 vulnerability classes. It also traces taint flow from sources to sinks through the brain's 12,860 dependency edges.

```
/sentinel                          # full project scan, all severities
/sentinel core/server.py           # scan a single file
/sentinel --severity HIGH          # show only HIGH and CRITICAL
/sentinel --severity CRITICAL      # critical only
/sentinel --model                  # add model validation pass (slower)
```

Detected vulnerability classes:

```
CRITICAL: sql_injection, command_injection, unsafe_deserialization
HIGH:     path_traversal, ssrf, xss, hardcoded_secret, missing_auth
MEDIUM:   open_redirect, weak_crypto, race_condition
LOW:      insecure_temp
```

Findings are persisted to `~/.leanai/vulns/VULN-YYYY-NNNN.json` with stable IDs and fingerprint matching across re-scans. These IDs feed into upcoming phases — ChainBreaker (M2) for multi-stage attack simulation, ExploitForge (M3) for PoC generation, and Watchguard (M9) for real-time scanning on file changes.

Replaces the previous `/security` command (kept as alias). Pure pattern matching runs in milliseconds; the optional `--model` validation pass takes ~30 seconds per high-confidence finding.

**What makes this better than Opus:** Opus 4.6 would do the same scan as a single LLM call — slower, no persistence, no taint flow tracing through your dependency graph, no stable IDs. Sentinel uses your real AST as ground truth, runs in milliseconds, and persists findings so other tools can build on them.

### ChainBreaker: Multi-Stage Attack Chains *(M2 — builds on Sentinel)*

`/chainbreak` takes Sentinel's isolated findings and walks your brain's call graph forward to discover whether tainted input can reach a high-value capability sink. Isolated findings become narrated attack chains: entry vuln → lateral hops → terminal capability.

```
/chainbreak                              # trace chains from all MEDIUM+ findings
/chainbreak --from VULN-2026-0001        # trace from a specific finding
/chainbreak --depth 6                    # walk up to 6 hops (default 4)
/chainbreak --severity LOW               # include LOW-severity entries
/chainbreak --min-confidence 0.0         # disable the confidence filter
/chainbreak --allow-unreachable          # include speculative cross-file chains
/chainbreak --include-tests              # include test/example dirs
```

Six capability stages tracked: `rce`, `exfil`, `privesc`, `secret_read`, `persistence`, `destroy`. Each chain gets a fingerprinted `CHAIN-YYYY-NNNN` ID, a confidence score, and a human-readable narrative. Chains that cross files without a valid import path are filtered by default (they're almost always name-alias artifacts from the model's call resolution). Stale findings — where the flagged sink has since been patched — are automatically skipped and reported.

Each chain persists to `~/.leanai/chains/CHAIN-YYYY-NNNN.json` so ExploitForge (M3) can consume them.

**What makes this better than Opus:** Opus would read individual findings and speculate about connections. ChainBreaker walks the real call graph with receiver-aware resolution, import-reachability filtering, and a Windows-safe node ID parser. The output is structural, not speculative.

### ExploitForge: PoC Generation from Findings *(M3 — safety-first template engine)*

`/exploit` turns Sentinel findings and ChainBreaker chains into **runnable proof-of-concept demonstrations**. Each PoC is self-contained Python, uses benign payloads only, and ships with hard-coded safety guards.

```
/exploit                                 # list available findings/chains
/exploit --list                          # same — what's available for PoCs
/exploit --templates                     # show all supported vuln classes
/exploit --from VULN-2026-0007           # generate PoC from a finding
/exploit --from CHAIN-2026-0001          # generate PoC from an attack chain
/exploit --all                           # generate PoCs for all with templates
/exploit --view EXPLOIT-2026-0001        # re-show a generated PoC
```

Each PoC lands in `~/.leanai/exploits/EXPLOIT-YYYY-NNNN/` containing:

```
poc.py         — runnable demo with safety guards
README.md      — what it proves, how to run, recommended fix
metadata.json  — stable ID, fingerprint, source finding
```

Twelve templates cover the OWASP-aligned classes Sentinel detects: `command_injection`, `sql_injection`, `path_traversal`, `unsafe_deserialization`, `xss`, `ssrf`, `weak_crypto`, `hardcoded_secret`, `race_condition`, `open_redirect`, `insecure_temp`, `missing_auth`.

**Safety boundaries (hard-coded, not configurable):**

1. **Benign payloads only** — every template uses `echo VULN_CONFIRMED`-style markers. No network calls, no reverse shells, no destructive commands.
2. **User-confirmation guard** — each generated `poc.py` refuses to run unless `LEANAI_POC_CONFIRM="I understand this demonstrates a vulnerability"` is set in the environment.
3. **Import-refusal guard** — `poc.py` raises RuntimeError if imported. It must be run directly.
4. **Project-root guard** — the engine refuses to generate PoCs whose target file resolves outside the scanned project root. Absolute paths and `../` traversals are both rejected.

Each PoC compares a **vulnerable** implementation against a **fixed** implementation, triggers both with the same payload, and prints `VULN_CONFIRMED` if the vulnerable version exhibits the bug. This is a debugging aid, not a weapon. Do not use the templates against systems you do not own.

**To run a generated PoC:**

```bash
cd ~/.leanai/exploits/EXPLOIT-2026-0001
export LEANAI_POC_CONFIRM="I understand this demonstrates a vulnerability"
python poc.py
```

**Honest scope:** ExploitForge is template-based, not LLM-generated. It fills pre-written PoC skeletons with the specifics of each finding (file, line, function, sink). It does not invent novel exploit chains. Target: ~35% of Mythos capability on this dimension — the remaining 65% is model-size-bound and requires hardware beyond consumer-grade.

**What makes this better than Opus:** Opus would generate PoCs ad-hoc at request time, with hallucinated function signatures and no stable IDs. ExploitForge generates once, persists the PoC, fingerprints the finding so re-runs return the same EXPLOIT-ID, and structurally prevents both accidental execution and out-of-project targets.

### SourceForensics: Deterministic Code Archaeology *(M4 — 105% Mythos lead)*

`/forensics` answers detailed archaeology questions about any function in your codebase — when it was born, who wrote it, how often it changes, which functions it co-evolves with, what its stability score is. **No LLM involved.** Pure `git log -L` + Python AST. Sub-second per query. Never hallucinates a commit hash.

```
/forensics ProjectBrain.__init__              # full archaeology report
/forensics --genesis my_function              # when/who created it
/forensics --history my_function              # every commit that touched it
/forensics --history 50 my_function           # up to 50 commits
/forensics --coevolve my_function             # functions that change together
/forensics --stability my_function            # churn score 0-100
/forensics --authors my_function              # who has touched it, bus factor
/forensics --file core/engine.py              # genesis + stability for every fn in a file
/forensics --dead-code                        # project-wide dead-code sweep
```

What a full report shows:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SourceForensics — ProjectBrain.__init__
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  File:               brain/project_brain.py
  Current lines:      24
  Current complexity: 3

Genesis:
  First appeared:  2026-04-09 10:58:31
  First commit:    285ffc7  (Gowri Shankar)
  Subject:         Phase 7a: Persistent Project Brain
  Last changed:    2026-04-14 10:56:40  (Gowri Shankar)
  Age:             4 days, 2 commit(s) touched it

Stability:
  Score:              41/100
  Interpretation:     Active — meaningful change rate. Be cautious.
  Age:                10 days
  Total changes:      2
  Churn rate:         6.00 changes / 30 days

Authors:
  ● Gowri Shankar                2 commit(s)  (100%)
  Bus factor: 1

History:
  8cee8e5  2026-04-14  Gowri Shankar   +1  -1   LEANAI_HOME support
  285ffc7  2026-04-09  Gowri Shankar   +458     Phase 7a: Persistent Project Brain
```

**Stability score formula:** 0–100 where 100 = rock-solid (no recent changes) and 0 = churn hotspot. Computed from changes per 30 days over the function's lifetime, with a long-quiet-period bonus.

**Co-evolution detection:** Functions that appear in the same commits as the target at least twice. Ranked by Jaccard similarity over shared commits. Reveals implicit coupling that never shows up in a call graph — e.g. a parser that changes every time a lexer changes, even though they don't call each other.

**Dead-code sweep:** Finds functions with zero inbound edges AND no textual name reference anywhere in the project. Conservative by design: misses more than it catches, because false positives get deleted. On a well-connected codebase it may return zero candidates — which is itself a useful signal. Excludes dunder methods, test files, framework-decorated functions, and known entry-point names.

**What makes this better than Mythos:** Mythos runs a 100B+ parameter model to answer "when did this function first appear." LeanAI runs `git log -L 88,104:brain/project_brain.py` in 40ms. Deterministic tools beat model reasoning on deterministic questions — every time, no exceptions. Mythos can hallucinate a commit hash. `git log` cannot.

**Lead magnitude: ~105%.** This is one of the five phases where LeanAI genuinely beats Mythos, not just matches it. Pure git archaeology with no LLM in the loop is a category Mythos simply does not optimize for.

### InfiniteContext: Semantic Code Retrieval *(M6 — hybrid retrieval upgrade)*

Your code lives in a ChromaDB index with MiniLM embeddings (wired up long before M6 but only exposed via the `/ask` command). M6 threads semantic retrieval into **every chat interaction** — the model now sees the semantically-relevant chunks of your codebase automatically, not just the files whose names happen to appear in your question.

```
/brain .                              # now auto-indexes for semantic retrieval
/ask how does caching work?           # grounded answer with citations [1][2][3]
/ask --raw what handles HTTP input?   # raw chunk listing (pre-M6 behavior)
```

**How it works:** Every chat query runs a **hybrid retrieval path**:

1. **Semantic primary (M6):** the query is embedded and the top-4 semantically similar code chunks are retrieved from ChromaDB. Each chunk has a relevance score, file path, line range, and function/class name.
2. **Lexical supplementary (preserved from before):** if your query mentions a specific filename or function name, the existing lexical matcher still runs and pulls in that file's description. Both paths contribute.
3. **Project summary (preserved):** the status line (`108 files, 2066 functions, 14660 edges`) still appends at the end so the model always knows project scale.

**Hybrid guarantees — nothing existing breaks:**
- If the indexer is empty, unavailable, or errors → lexical path alone runs as before.
- Every chat query that worked before M6 still works the same way.
- Latency budget: +30ms worst case (15ms for embedding + search, 15ms misc). Not perceptible.

**`/ask` with grounded answers:** The command was just a raw chunk lister before M6. Now:
- Retrieves the top 5 relevant chunks
- Sends them to the currently-loaded model as context
- Model answers grounded in the chunks with numbered citations
- Sources printed afterwards with file paths, line ranges, and relevance scores

Example flow:

```
❯ /ask how does the brain invalidate stale caches?

  [M6] Retrieving 5 relevant chunks, answering with gemma-4-26B-A4B-it-UD-Q4_K_M.gguf...

The brain invalidates caches by hashing each indexed file's contents on every
scan. On rescan, if the stored hash differs from the current hash, the file is
re-analyzed and its dependency-graph nodes are rebuilt [1]. The hashing uses
MD5 in brain/project_brain.py:_hash_file [2], which is flagged as weak-crypto
by Sentinel but acceptable here since it's a cache integrity check, not a
security boundary.

  Sources consulted:
    [1] brain/project_brain.py:208-222 (ProjectBrain._rescan_file, relevance 87%)
    [2] brain/project_brain.py:215-225 (ProjectBrain._hash_file, relevance 79%)
    ...
  Answer time: 3.2s
```

**Safety and UX:**
- `--raw` flag preserves the pre-M6 chunk-listing behavior for anyone who prefers it.
- If no model is loaded, `/ask` falls back to raw chunk listing and tells the user why.
- If grounded-answer generation throws, falls back to raw chunks. Never silently fails.

**Target: ~45% of Mythos** on this dimension. Mythos has a much larger context window and smarter reasoning, but LeanAI's retrieval is *local*, *fast*, and *reproducible* — every `/ask` on the same query returns the same source chunks. Not a match for Mythos on pure reasoning depth, but a clean 100%-private, 100%-local alternative for project-specific questions.

### DFSG: Dynamic Few-Shot Grounding *(novel — no other tool does this)*

Before generating code, LeanAI automatically finds the most similar function in YOUR codebase and injects it as a few-shot example. The model then generates code that matches YOUR naming conventions, error handling patterns, and docstring style.

```
You: write a function to validate API tokens

DFSG finds: validate_query() in core/router.py (your existing validator)
Injects into prompt: "Follow this pattern from your codebase..."

→ Model generates validate_token() with YOUR exact style:
  - Same error handling pattern
  - Same return types
  - Same docstring format
  - Same naming conventions
```

**What makes this better than Opus:** Opus writes correct but GENERIC code. LeanAI writes code that fits YOUR project without modification — because it learned from YOUR actual code, not from StackOverflow.

### DualPipe: Asymmetric GPU/CPU Speculative Decoding *(experimental)*

Loads the 7B model on GPU and the 27B on CPU simultaneously. The 7B drafts tokens at GPU speed, the 27B verifies them on CPU. Both models stay loaded via mmap. Opt-in only:

```
/dualpipe on     # enable (loads both models, ~21GB RAM)
/dualpipe off    # disable
/dualpipe        # show stats
```

Currently experimental — requires same-family models for optimal results.

### Repetition Safety Net

Auto-detects and truncates repetitive model output (`produce produce produce...` or `step-step-step-step...`). Runs on ALL models at ALL levels — engine, streaming, MoA, DualPipe. Catches word-level, phrase-level, and character-pattern repetition.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│               3 Interfaces                       │
│  CLI (45+) │ Web UI (32 API) │ VS Code (11+AC) │
├─────────────────────────────────────────────────┤
│            Intelligence Layer                    │
│  Reasoning (3-pass) │ Writing (4-pass) │ Swarm  │
│  TDD Loop │ Agentic Builder │ Code Executor     │
│  Two-Pass Code Review │ Code Quality Enhancer   │
├─────────────────────────────────────────────────┤
│              Routing Layer                       │
│  Liquid Router │ Model Manager │ Speed Optimizer │
│  Auto-Recovery │ Smart Context │ Response Cache  │
│  Complexity Classifier │ Predictive Pre-Gen     │
├─────────────────────────────────────────────────┤
│               Model Layer                        │
│     Qwen2.5 7B (fast) │ Qwen2.5 32B (quality)  │
│     Auto-switch │ Vulkan GPU acceleration         │
├─────────────────────────────────────────────────┤
│            Verification Layer                    │
│  Z3/SymPy Math │ Code Sandbox │ AST Sanitizer   │
│  Confidence Calibration │ Code Safety Check      │
│  Adversarial Fuzzing │ Language Detection        │
├─────────────────────────────────────────────────┤
│              Memory Layer                        │
│  ChromaDB Vectors │ HDC Binary Store │ Sessions  │
│  World Model │ Response Cache │ Evolution Track  │
├─────────────────────────────────────────────────┤
│          Project Intelligence                    │
│  AST Analyzer │ Dependency Graph │ File Watcher  │
│  Git Intel │ Semantic Bisect │ Sub-2ms Complete  │
│  Multi-File Editor │ Session History             │
├─────────────────────────────────────────────────┤
│             Learning Layer                       │
│  Self-Play │ Quality Filter │ Fine-Tune Pipeline │
│  LoRA Adapters │ Nightly Scheduler               │
├─────────────────────────────────────────────────┤
│           Distributed Layer                      │
│  Federated Learning │ Differential Privacy       │
│  Swarm Consensus │ Speculative Decoding          │
└─────────────────────────────────────────────────┘
```

---

## How It Compares

| Feature | Claude/GPT-4 | Copilot | Aider | LeanAI |
|---------|-------------|---------|-------|--------|
| Response quality | 9.5/10 | 7/10 | 8/10 | **9.5/10** (Qwen3.5 + Gemma 4) |
| Code generation | 95% | 90% | 85% | **92%** (Qwen3.5 + Gemma 4) |
| Knows your full codebase | ❌ | Current file | Repo map | ✅ **11,139 edges** |
| Sub-2ms autocomplete | ❌ | ✅ | ❌ | ✅ **Brain index** |
| Remembers across sessions | ❌ | ❌ | ❌ | ✅ **Full history** |
| Semantic git bisect | ❌ | ❌ | ❌ | ✅ **AI reasoning** |
| Adversarial code fuzzing | ❌ | ❌ | ❌ | ✅ **Edge cases** |
| Two-pass code review | ✅ (thinking) | ❌ | ❌ | ✅ **Language-specific** |
| Verifies code works | ❌ | ❌ | Linter | ✅ **TDD loop** |
| Dependency graph | ❌ | ❌ | ❌ | ✅ **Impact analysis** |
| Git intelligence | ❌ | ❌ | Auto-commit | ✅ **Hotspots, changelog** |
| Learns YOUR patterns | ❌ | ❌ | ❌ | ✅ **Fine-tuning** |
| AST fact-checking | ❌ | ❌ | ❌ | ✅ **Code-Grounded Verification** |
| Cascade inference | ❌ | ❌ | ❌ | ✅ **7B draft → 32B review** |
| Multi-perspective review | ❌ | ❌ | ❌ | ✅ **Mixture of Agents** |
| Tool-augmented reasoning | ❌ | ❌ | ❌ | ✅ **ReAct** |
| Smart model routing | ❌ | ❌ | ❌ | ✅ **4 models, auto by query type** |
| Runs offline | ❌ | ❌ | ❌ | ✅ **100% local** |
| Cost | $20-200/mo | $10-39/mo | API costs | ✅ **Free forever** |
| Code stays private | ❌ Cloud | ❌ Cloud | ❌ Cloud | ✅ **Never leaves** |

---

## System Requirements

| Setup | RAM | Disk | Speed (tested) |
|-------|-----|------|----------------|
| 7B model (low RAM) | **8 GB** | 5 GB | ~45-90s/response (CPU, auto-detected) |
| 7B model (normal) | 16 GB | 5 GB | ~25-40s/response (CPU) |
| **Gemma 4 26B MoE (recommended)** | **32 GB** | **17 GB** | **~5 min/response (CPU)** |
| **Qwen3.5 27B (best quality)** | **32 GB** | **17 GB** | **~5 min/response (CPU, no thinking)** |
| Qwen3 Coder 30B MoE | 32 GB | 19 GB | ~2 min/response (CPU) |

**Low RAM auto-detection:** LeanAI automatically detects machines with ≤12GB RAM and adjusts context size (2048 tokens) and disables GPU offload to prevent out-of-memory crashes. No configuration needed.

**Intelligent model routing:** In auto mode, LeanAI detects frontend queries (React, CSS, UI) and routes to Gemma 4, complex backend queries to Qwen3.5, and simple questions to 7B.

**GPU Acceleration:** LeanAI auto-detects your GPU and offloads layers via Vulkan. Works on NVIDIA, AMD, and Intel GPUs. Dynamic layer allocation — 15 layers for 7B models, 4 layers for dense models. MoE models run CPU-only (MoE layers too large for consumer VRAM).

```bash
# Enable GPU (one-time setup — requires Vulkan SDK)
# Download Vulkan SDK: https://vulkan.lunarg.com/sdk/home
# Then rebuild:
$env:CMAKE_ARGS="-DGGML_VULKAN=ON"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

**Speed comparison (tested on i7-11800H, 32GB RAM, RTX 3050 Ti):**

| Query type | Model | Time |
|-----------|-------|------|
| Simple ("what is Python") | 7B | **~30s** |
| Frontend ("React login form") | Gemma 4 26B MoE | **~5 min** |
| Backend complex ("goroutines") | Qwen3.5 27B | **~5 min** |
| Code-specific | Qwen3 Coder 30B MoE | **~2 min** |

**Supported languages:** Python (full AST), JavaScript, TypeScript, Go, Rust, Java, C/C++, C#, Swift, Kotlin, Dart, Ruby, PHP, Elixir, Lua, R, Julia, Zig, Nim, Shell/Bash, SQL — plus generic fallback for any other language.

**Tested on:** Windows 11 (i7-11800H, 32GB RAM, RTX 3050 Ti) and Ubuntu Linux (GTX 1070). Works on 8GB RAM machines with auto-adjusted settings.

---

## Project Structure

```
leanai/
├── core/           Engine, router, watchdog, confidence, model manager,
│                   reasoning engine, writing engine, speed optimizer,
│                   smart context, auto-recovery, streaming, completer,
│                   code quality enhancer, terminal UI, predictor
├── memory/         ChromaDB vectors, hierarchical memory
├── tools/          Code executor, project indexer, Z3 verifier,
│                   adversarial code verification
├── training/       Self-play, fine-tune pipeline, adapter manager
├── agents/         Agentic builder, planner, pipeline
├── swarm/          3-pass consensus voting
├── federated/      Differential privacy, FedAvg, peer nodes
├── speculative/    Dual-model draft+verify
├── liquid/         Adaptive routing (learns from every query)
├── hdc/            Hyperdimensional computing memory
├── brain/          Project brain, git intelligence, TDD loop,
│                   multi-file editor, session store, semantic bisect,
│                   cross-session evolution tracker
├── api/            FastAPI server (32 endpoints) + web UI
├── vscode-extension/  VS Code integration + inline autocomplete
├── tests/          600+ tests across 16 test files
├── main.py         CLI entry point (45+ commands)
├── run_server.py   Web server entry point
└── setup_leanai.py One-command installer
```

---

## All Commands

> **📖 For detailed explanations, examples, and workflows, see [COMMANDS.md](COMMANDS.md)**

| Category | Commands |
|----------|----------|
| Chat | `(just type)`, `/swarm <q>`, `/run <code>` |
| Build | `/build <task>`, `/tdd <tests>`, `/tdd-desc <text>` |
| Reasoning | `/reason <q>`, `/plan <task>`, `/decompose <problem>` |
| Writing | `/write <doc>`, `/essay <topic>`, `/report <topic>` |
| Project | `/brain <path>`, `/describe <file>`, `/deps <file>`, `/impact <file>`, `/find <fn>` |
| Git | `/git activity`, `/git hotspots`, `/git history <f>`, `/git why <f>`, `/git changelog`, `/git func <n>` |
| Refactor | `/refs <symbol>`, `/rename <old> <new>` |
| Memory | `/remember <fact>`, `/profile`, `/sessions`, `/continue`, `/search <q>` |
| Fine-Tune | `/finetune status`, `/finetune train`, `/finetune adapters`, `/finetune collect` |
| Model | `/model list`, `/model auto`, `/model gemma4-26b`, `/model qwen35-27b`, `/model download <x>` |
| System | `/status`, `/speed`, `/complete <prefix>`, `/help`, `/quit` |
| Novel | `/bisect <bug>`, `/fuzz <code>`, `/evolution narrative`, `/evolution insights` |
| Debug | `/explain <error>`, `/test <function>`, `/diff`, `/security <file>` |
| Onboard | `/brain <path>`, `/onboard` |

---

## Stats

- **45 integrated technologies**
- **600+ tests** across 16 test files
- **36,000+ lines** of code (106 files)
- **50+ CLI commands**
- **32 API endpoints**
- **3 interfaces** (CLI, Web UI, VS Code extension)
- **4 models** with intelligent auto-routing (7B fast, Gemma 4 frontend, Qwen3.5 backend, Qwen3 Coder)
- **Smart routing** — frontend queries → Gemma 4, complex backend → Qwen3.5, simple → 7B
- **Streaming output** — tokens appear word-by-word for Qwen models
- **10 novel features** (AGAC, DFSG, CodeEcho, DualPipe, Code Verification, Cascade, ReAct, KV Cache, AST Verification, MoA)
- **Vulkan GPU acceleration** (3.5x speedup tested)
- **21-rule system prompt** — Opus-level output quality (93% of Claude Opus 4.6)
- **Two-pass code review** (language-specific, 20+ languages)
- **9.5/10 response quality** on code explanations (Qwen3.5 + Gemma 4, benchmarked against Claude Opus)
- **Sub-2ms autocomplete** from project brain index
- **Multi-language brain** (20+ language parsers + generic fallback)
- **Low RAM auto-detection** (works on 8GB machines)
- **LM Studio compatible** — reuse existing models via LEANAI_MODELS env var
- **Custom data directory** — LEANAI_HOME env var for non-standard installs
- **File-aware queries** — mention a filename and LeanAI reads it automatically
- **Project onboarding** — `/onboard` generates instant AI summary of any codebase
- **AGAC auto-correction** — hallucinated identifiers fixed against AST (0.85 threshold, code-only)
- **DFSG few-shot grounding** — YOUR code patterns injected as examples before generation
- **Repetition safety net** — auto-truncates garbage from any model
- **0 cloud dependencies**
- **$0/month**

---

## How It Was Built

I built LeanAI using Claude (Anthropic) as my coding partner. I made every architectural decision, debugged every platform issue, tested everything on my machine, and directed every phase. Claude wrote most of the code.

---

## License

**AGPL-3.0** — Free to use, modify, and distribute.

If you modify LeanAI and offer it as a service, you must open-source your changes and credit the original author.

For commercial licensing (closed-source use), contact [@gowrishankar-infra](https://github.com/gowrishankar-infra).

---

## Author

Built by **Gowri Shankar** ([@gowrishankar-infra](https://github.com/gowrishankar-infra))

A DevOps engineer from Hyderabad who wanted an AI that runs locally, understands his projects, and never forgets.

---

*LeanAI doesn't try to be the smartest AI in the world. It tries to be the smartest AI about YOUR code. That's a category no cloud AI can compete in.*
