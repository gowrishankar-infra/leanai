# LeanAI — Complete Command Reference

> Every command explained with examples, expected output, and tips.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Chat](#chat)
- [Project Brain](#project-brain)
- [Autocomplete](#autocomplete)
- [Git Intelligence](#git-intelligence)
- [Semantic Bisect](#semantic-bisect)
- [Adversarial Fuzzing](#adversarial-fuzzing)
- [Sentinel — Autonomous Security Analyzer (M1)](#sentinel--autonomous-security-analyzer)
- [Attack Chain Analysis (M2 — ChainBreaker)](#attack-chain-analysis-m2--chainbreaker)
- [PoC Generation (M3 — ExploitForge)](#poc-generation-m3--exploitforge)
- [Code Archaeology (M4 — SourceForensics)](#code-archaeology-m4--sourceforensics)
- [Code Retrieval (M6 — InfiniteContext)](#code-retrieval-m6--infinitecontext-105-mythos-lead)
- [Persistent Knowledge Graph (M8 — MemoryForge)](#persistent-knowledge-graph-m8--memoryforge)
- [Code Execution](#code-execution)
- [TDD Auto-Fix](#tdd-auto-fix)
- [Reasoning](#reasoning)
- [Writing](#writing)
- [Swarm Consensus](#swarm-consensus)
- [Build (Agentic)](#build-agentic)
- [Refactoring](#refactoring)
- [Memory](#memory)
- [Sessions](#sessions)
- [Evolution Tracking](#evolution-tracking)
- [Fine-Tuning](#fine-tuning)
- [Model Management](#model-management)
- [System](#system)

---

## Getting Started

After installing LeanAI, start it with:

```bash
python main.py
```

The first two commands you should run:

```
/brain .          # scan your project — builds the dependency graph
/model auto       # auto-switch between 7B (fast) and 32B (quality)
```

After that, just type questions naturally. LeanAI understands plain English.

---

## Chat

**Just type anything.** No command prefix needed for regular questions.

### Usage

```
what does the engine file do?
explain this function: def calculate_total(items): ...
write a Python function to parse CSV files
how do I set up authentication in FastAPI?
```

### What happens behind the scenes

1. Your query is classified by complexity (simple/medium/complex)
2. The right model is selected (7B for simple, 32B for complex)
3. Smart context is injected — your project's code, git history, past sessions
4. The model generates a response
5. If code is generated, it's verified in a sandbox
6. A second review pass checks for bugs (two-pass quality)
7. Response is displayed with confidence score

### Tips

- Be specific: "explain the generate method in engine_v3.py" works better than "explain my code"
- Paste code directly: LeanAI can explain, fix, or improve any code you paste
- Ask follow-ups: LeanAI remembers the current conversation

---

## Project Brain

Scans your entire codebase with AST (Abstract Syntax Tree) analysis. Builds a full dependency graph so LeanAI knows every function, class, import, and relationship.

### Commands

| Command | What it does |
|---------|-------------|
| `/brain .` | Scan the current directory |
| `/brain /path/to/project` | Scan a specific project |
| `/describe file.py` | Describe what a specific file does |
| `/deps file.py` | Show what this file depends on |
| `/impact file.py` | Show what breaks if you change this file |
| `/find function_name` | Find a function across the entire project |

### Examples

```
/brain .
```
Output:
```
[Brain] Scanned 91 files in 5674ms
Project: leanai
Files: 91 indexed
Functions: 1,689
Classes: 320
Dependency edges: 9,775
```

```
/describe core/engine_v3.py
```
Output: A detailed description of what the file does, its classes, methods, and role in the project — using YOUR actual code, not generic examples.

```
/impact main.py
```
Output: Lists every file that depends on main.py — so you know what could break if you change it.

### Tips

- Run `/brain .` first after starting LeanAI — it powers autocomplete, context injection, and project-specific answers
- Re-run after major code changes to update the dependency graph
- Use `/impact` before refactoring to understand the blast radius

---

## Autocomplete

Indexes all functions and classes from your project brain. Returns completions in under 2 milliseconds — no model call needed.

### Commands

| Command | What it does |
|---------|-------------|
| `/complete <prefix>` | Get completions matching the prefix |

### Examples

```
/complete gen
```
Output:
```
Completions for 'gen' (0.8ms):
  ◆ GenerationConfig              core/engine.py
  ƒ generate()                    core/engine_v3.py
  ƒ generate_changelog()          brain/git_intel.py
  ƒ generate_batch()              core/engine_v3.py
```

```
/complete Model
```
Output:
```
  ◆ ModelManager                   core/model_manager.py
  ◆ ModelInfo                      core/model_manager.py
  ƒ ModelInfo.resolved_path()      core/model_manager.py
```

```
/complete engine.gen
```
Dot notation works — filters to functions starting with "gen" in engine-related files.

### How it works

- `ƒ` = function
- `◆` = class
- `⚡` = keyword
- `✂` = snippet

Completions come from YOUR project's indexed code, not generic suggestions.

### VS Code Integration

The VS Code extension provides inline autocomplete as you type. Make sure the server is running (`python run_server.py`) and the extension is installed.

---

## Git Intelligence

Reads your entire git history. Provides insights no other tool offers.

### Commands

| Command | What it does |
|---------|-------------|
| `/git activity` | What happened this week (commits grouped by date) |
| `/git hotspots` | Most frequently changed files (with visual bar chart) |
| `/git history file.py` | Full commit history for a specific file |
| `/git why file.py` | Commit messages explaining WHY this file changed |
| `/git changelog` | Auto-categorized changelog (features/fixes/other) |
| `/git func function_name` | When was this function last modified |

### Examples

```
/git hotspots
```
Output:
```
  8 ████████ main.py
  5 █████ core/engine_v3.py
  3 ███ brain/project_brain.py
  2 ██ api/server.py
```

```
/git changelog
```
Output: A clean, auto-generated changelog grouped by features, bug fixes, and other changes.

### Tips

- Use `/git hotspots` to identify which files need the most attention
- Use `/git why` to understand the history behind design decisions
- Works with any git repository — not just Python projects

---

## Semantic Bisect

AI-powered bug finding. Instead of manual binary search (traditional `git bisect`), LeanAI reads each commit **semantically** and predicts which one introduced a bug.

### Commands

| Command | What it does |
|---------|-------------|
| `/bisect <bug description>` | Find which commit likely caused the bug |

### Examples

```
/bisect authentication stopped working
```
Output:
```
[Bisect] Analyzing last 20 commits...
[Bisect] Found 20 commits. Scoring suspicion...

Most likely culprit:
  b7b3f51 — refactor token validation
  Suspicion: 65%
  Reasoning: commit message mentions auth; risky operation: refactor;
             large change (120+ 80-)

Other suspects:
  a1b2c3d — update middleware (40%)
  e5f6g7h — add rate limiting (25%)
```

### How it works

1. Gets the last 20 commits from your git history
2. Scores each commit based on: keyword overlap with bug description, file relevance, change size, risky operations (refactor, rewrite, merge)
3. Uses AI to analyze the top 5 suspects in depth
4. Returns the most likely culprit with reasoning

### Tips

- Be descriptive: "login fails after entering correct password" works better than "login broken"
- Narrow the scope: "API returns 500 on /users endpoint" helps LeanAI focus on the right files

---

## Adversarial Fuzzing

Generates edge-case inputs designed to **break** your code. Finds bugs you didn't think of.

### Commands

| Command | What it does |
|---------|-------------|
| `/fuzz <python code>` | Run adversarial verification on a function |

### Examples

```
/fuzz def sort(arr): return sorted(arr)
```
Output:
```
Adversarial Verification: sort
Tested: 12 | Passed: 9 | Failed: 3

Failures:
  ✗ None → TypeError: 'NoneType' is not iterable
  ✗ [1, None, 3] → TypeError: '<' not supported
  ✗ [1, 'a', 2.0] → TypeError: '<' not supported

Suggested fixes:
  → Add None check: if argument is None, raise ValueError or return default
  → Add type validation: ensure all elements are the same type

Time: 892ms
```

```
/fuzz def divide(a, b): return a / b
```
Output: Finds ZeroDivisionError, None inputs, infinity, NaN, and more.

### What it tests

| Input type | Edge cases generated |
|-----------|---------------------|
| Numbers | 0, -1, 999999, float('inf'), float('nan'), 0.1+0.2 |
| Strings | "", " ", "a"*10000, None, special chars, unicode |
| Lists | [], [1], None, [None], [1,'a',2.0], nested lists |
| Dicts | {}, None, int keys, None values |

### Tips

- Paste your actual functions to find real bugs
- Use the "Suggested fixes" to improve your code
- Great for interview prep — tests your code the way interviewers would

---

## Sentinel — Autonomous Security Analyzer

AST-grounded security scan that uses the project brain's dependency graph to trace data flow from external input sources to dangerous sinks. Pattern-matches 12 vulnerability classes. Replaces the older `/security` command (kept as alias).

### Commands

| Command | What it does |
|---------|-------------|
| `/sentinel` | Full project scan, all severities |
| `/sentinel <file>` | Scan a single file |
| `/sentinel --severity HIGH` | Show only HIGH and CRITICAL findings |
| `/sentinel --severity CRITICAL` | Critical findings only |
| `/sentinel --model` | Add model validation pass on high-confidence findings (slower) |
| `/security` | Alias for `/sentinel` (backwards-compatible) |

### Examples

```
/brain .
/sentinel
```
Output:
```
[Sentinel] Analyzing 106 files for vulnerabilities...
[Sentinel] Found 23 input sources, 14 dangerous sinks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sentinel Security Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Found 8 potential issues   2 CRITICAL  4 HIGH  2 MEDIUM
  106 files, 1969 functions, 23 sources, 14 sinks, 3 traced paths, 47ms

── CRITICAL (2) ──

  [VULN-2026-0001] Command Injection
    api/server.py:142  in  handle_exec
    Potential command injection: eval() call
    >     result = eval(user_code)
    Fix: Avoid eval/exec. Use subprocess with shell=False.
    Confidence: 70%
```

```
/sentinel api/server.py
/sentinel --severity CRITICAL
/sentinel --severity HIGH --model
```

### Detected Vulnerability Classes

| Severity | Classes |
|----------|---------|
| CRITICAL | sql_injection, command_injection, unsafe_deserialization |
| HIGH | path_traversal, ssrf, xss, hardcoded_secret, missing_auth |
| MEDIUM | open_redirect, weak_crypto, race_condition |
| LOW | insecure_temp |

### How It Works

1. **Source discovery** — finds functions taking external input (HTTP handlers via decorators, `sys.argv`, `os.environ`, `pickle.loads`, file reads)
2. **Sink discovery** — finds dangerous operations (`eval`, `exec`, `subprocess shell=True`, SQL string concat, `os.path.join` with input, weak crypto, hardcoded secrets)
3. **Per-function pattern matching** — 39 sink regex patterns + decorator-based missing-auth detection
4. **Module-level scans** — hardcoded secrets and weak crypto outside functions
5. **Taint flow tracing** — BFS through `brain.graph._adjacency` (max depth 5) connects sources to sinks across function boundaries
6. **Optional model validation** — when `--model` flag is set, asks the local model "is this exploitable?" for high-confidence findings
7. **Stable IDs** — `VULN-YYYY-NNNN` persisted to `~/.leanai/vulns/` with fingerprint matching, so the same finding gets the same ID across re-scans

### Persistence

Each finding is saved as `~/.leanai/vulns/VULN-YYYY-NNNN.json`:

```json
{
  "vuln_id": "VULN-2026-0001",
  "vuln_class": "command_injection",
  "severity": "CRITICAL",
  "confidence": 0.7,
  "filepath": "api/server.py",
  "function_name": "handle_exec",
  "line": 142,
  "source_type": "http_input",
  "sink_type": "eval() call",
  "description": "Potential command injection: eval() call",
  "code_snippet": "    result = eval(user_code)",
  "taint_path": ["handle_exec"],
  "fix_suggestion": "Avoid eval/exec...",
  "fingerprint": "e4424553a5",
  "timestamp": 1776442166.99
}
```

These IDs feed into upcoming phases: **ChainBreaker (M2)** for multi-stage attack simulation, **ExploitForge (M3)** for PoC generation, **Watchguard (M9)** for real-time scanning.

### Tips

- Run `/brain .` first — Sentinel needs the brain's AST and dependency graph
- The pure pattern-matching scan is fast (milliseconds). Only use `--model` when you need fewer false positives
- Test/example directories are auto-skipped for hardcoded-secret detection (too noisy)
- Comments are stripped before regex matching, so commented-out `eval()` won't trigger
- Findings persist across sessions — run `/sentinel` again later and finding IDs stay stable

---

## Attack Chain Analysis (M2 — ChainBreaker)

Take Sentinel's isolated findings and walk the brain's call graph forward to discover whether tainted input can reach a high-value capability sink. Isolated vulnerabilities become narrated attack chains.

### Commands

| Command | What it does |
|---------|-------------|
| `/chainbreak` | Trace chains from all MEDIUM+ findings |
| `/chainbreak --from VULN-2026-0001` | Trace from a specific Sentinel finding |
| `/chainbreak --depth 6` | Walk up to 6 hops (default 4, max 8) |
| `/chainbreak --severity LOW` | Include LOW-severity entries (default MEDIUM) |
| `/chainbreak --min-confidence 0.0` | Disable the confidence filter |
| `/chainbreak --allow-unreachable` | Include speculative cross-file chains |
| `/chainbreak --include-tests` | Include test/example directories |

### Example

```
/chainbreak
```

Output:
```
ChainBreaker: tracing multi-stage attack chains (depth 4, severity ≥ MEDIUM, min-confidence 0.35)...
[ChainBreaker] Loaded 15 entry vulnerabilities (severity >= MEDIUM)

  ✓ No exploitable attack chains found.

  Findings analyzed: 15  Entries resolved: 12  Functions visited: 47  Time: 231ms
  Skipped 3 stale finding(s) — the referenced code no longer matches the reported sink.
```

### Capability stages

Each chain ends at one of six capability sinks:

| Stage | Meaning |
|-------|---------|
| `rce` | Remote code execution (eval / exec / subprocess) |
| `exfil` | Data egress / network send / file write |
| `privesc` | Privilege-escalating context (sudo / setuid) |
| `secret_read` | Credential or secret access |
| `persistence` | Write to startup-persisted location |
| `destroy` | Destructive file/db ops |

### Notes

- Each chain gets a fingerprinted `CHAIN-YYYY-NNNN` ID, persisted to `~/.leanai/chains/CHAIN-YYYY-NNNN.json`
- Confidence decays with chain length. Cross-file chains with no valid import path are dropped by default (name-alias artifacts)
- The stale-finding guard automatically skips findings whose referenced code has since been patched
- Chains feed forward into ExploitForge (M3)

---

## PoC Generation (M3 — ExploitForge)

Turn Sentinel findings and ChainBreaker chains into runnable proof-of-concept demonstrations. Each PoC is benign-payload only, hard-guarded against accidental execution, and restricted to files inside your project.

### Commands

| Command | What it does |
|---------|-------------|
| `/exploit` | List available findings/chains (same as `--list`) |
| `/exploit --list` | Show all findings + chains, grouped by vuln class |
| `/exploit --templates` | Show all 12 supported vuln classes |
| `/exploit --from VULN-2026-0007` | Generate PoC from a specific finding |
| `/exploit --from CHAIN-2026-0001` | Generate PoC from an attack chain |
| `/exploit --all` | Generate PoCs for all findings with templates (prompts if >10) |
| `/exploit --view EXPLOIT-2026-0001` | Re-show a generated PoC's README + first 40 lines |

### Example

```
/exploit --from VULN-2026-0007
```

Output:
```
ExploitForge: generating PoC(s)...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ExploitForge — PoC Generation Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Generated 1 PoC(s)
  Time: 8ms

  How to run each PoC:
  1. cd <output_dir>
  2. export LEANAI_POC_CONFIRM="I understand this demonstrates a vulnerability"
  3. python poc.py

  [EXPLOIT-2026-0001] weak_crypto  (from VULN-2026-0007, MEDIUM)
    Target: core/sentinel.py:1009 in SentinelEngine._persist
    Output: ~/.leanai/exploits/EXPLOIT-2026-0001
```

### Running a generated PoC

Windows:
```cmd
cd %USERPROFILE%\.leanai\exploits\EXPLOIT-2026-0001
set LEANAI_POC_CONFIRM=I understand this demonstrates a vulnerability
python poc.py
```

Linux/Mac:
```bash
cd ~/.leanai/exploits/EXPLOIT-2026-0001
export LEANAI_POC_CONFIRM="I understand this demonstrates a vulnerability"
python poc.py
```

Look for the `▶ VULN_CONFIRMED` line. That means the vulnerable pattern was triggered as designed.

### Supported templates

| Class | Demonstrates |
|-------|-------------|
| `command_injection` | Shell metacharacter interpretation via `shell=True` subprocess |
| `sql_injection` | WHERE-clause bypass via string concatenation |
| `path_traversal` | Directory escape via `../` sequences |
| `unsafe_deserialization` | Arbitrary code execution via `pickle.loads` |
| `xss` | Unescaped user content in HTML |
| `ssrf` | Cloud metadata URL access via user-controlled host |
| `weak_crypto` | MD5 collision using published collision blocks |
| `hardcoded_secret` | Secret recovery from git history after patch |
| `race_condition` | TOCTOU file swap via thread |
| `open_redirect` | External redirect via user-controlled next URL |
| `insecure_temp` | Predictable path from `tempfile.mktemp` |
| `missing_auth` | Destructive action without auth check |

### Safety boundaries (hard-coded)

- **Benign payloads only** — all templates use `echo VULN_CONFIRMED`-style markers. No network calls, no destructive commands.
- **User confirmation required** — `poc.py` refuses to run unless `LEANAI_POC_CONFIRM` env var is set to the exact confirmation string.
- **Import-refusal guard** — `poc.py` raises RuntimeError if imported rather than run directly.
- **Project-root enforcement** — ExploitForge refuses to generate PoCs for targets outside the scanned project root. Absolute paths and `../` traversals are rejected.

### Notes

- Each PoC has a stable `EXPLOIT-YYYY-NNNN` ID. Re-running on the same finding returns the same ID. Adding a new finding creates a new ID while preserving existing ones.
- PoCs ship with both a vulnerable and a fixed implementation so you can see the contrast.
- The README.md inside each exploit directory links the PoC back to the source Sentinel finding or ChainBreaker chain.
- ExploitForge is a debugging aid, not a weapon. Do not use the templates against systems you do not own.

### Regenerating old PoCs (post-M4 polish)

If you have PoC directories generated before the M4 path-separator fix, running them will produce a harmless `SyntaxWarning: invalid escape sequence '\s'` because Windows paths like `core\sentinel.py` in the banner contain `\s` sequences. Regenerate to clear them:

**PowerShell:**

```powershell
Remove-Item -Recurse -Force $env:USERPROFILE\.leanai\exploits
python main.py
# at prompt:
/brain .
/exploit --all
```

**Bash / macOS / Linux:**

```bash
rm -rf ~/.leanai/exploits
python main.py
# at prompt:
/brain .
/exploit --all
```

After this, new PoC files will use forward-slash paths in the banner (`core/sentinel.py`) and the warning disappears. Runtime behavior of the PoCs is identical — this is purely cosmetic.

---

## Code Archaeology (M4 — SourceForensics)

Deterministic answers to questions about function history. No LLM involved. Uses `git log -L` for line-range-tracked commit history and Python's `ast` module for structural questions. Sub-second per query. Never hallucinates a commit hash.

### Commands

| Command | What it does |
|---------|-------------|
| `/forensics <function>` | Full archaeology report (genesis + stability + authors + history + co-evolution) |
| `/forensics --genesis <fn>` | When/who/which commit created the function |
| `/forensics --history <fn>` | Commit-by-commit history (default 20, max 200) |
| `/forensics --history 50 <fn>` | Limit history to N commits |
| `/forensics --coevolve <fn>` | Functions that change in the same commits (Jaccard-ranked) |
| `/forensics --coevolve 5 <fn>` | Top N co-evolving functions |
| `/forensics --stability <fn>` | Churn score 0-100 + interpretation |
| `/forensics --authors <fn>` | Author breakdown + bus factor |
| `/forensics --file <path>` | Genesis + stability for every function in a file |
| `/forensics --dead-code` | Project-wide dead-code sweep |
| `/forensics --json <...>` | Emit JSON instead of formatted output (any subcommand) |

### Example

```
/forensics ProjectBrain.__init__
```

Output:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SourceForensics — ProjectBrain.__init__
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  File:               brain/project_brain.py
  Current lines:      24
  Current complexity: 3

Genesis:
  First appeared:  2026-04-09
  First commit:    285ffc7  (Gowri Shankar)
  Last changed:    2026-04-14  (Gowri Shankar)
  Age:             4 days, 2 commit(s) touched it

Stability:
  Score:              41/100
  Interpretation:     Active — meaningful change rate. Be cautious.
  Churn rate:         6.00 changes / 30 days

Authors:
  ● Gowri Shankar                2 commit(s)  (100%)
  Bus factor: 1  (authors to cover 80%+ of changes)

History:
  8cee8e5  2026-04-14  Gowri Shankar   +1  -1   LEANAI_HOME support
  285ffc7  2026-04-09  Gowri Shankar   +458     Phase 7a: Persistent Project Brain

Co-evolution (functions that change together):
  (no co-evolving functions found)
```

### Resolving function names

Simple name:
```
/forensics my_function
```

If the name is ambiguous (multiple functions with the same short name across the project), LeanAI will list the candidates and ask you to disambiguate:

```
No unique match for '__init__'. 12 candidate(s):
  brain/project_brain.py:ProjectBrain.__init__
  brain/session_store.py:SessionStore.__init__
  ...
Try /forensics <file.py:qname>
```

File-anchored (for disambiguation):
```
/forensics brain/project_brain.py:ProjectBrain.__init__
```

Qualified name (for methods):
```
/forensics DependencyGraph._resolve_with_scope
```

### Stability score scale

| Score | Meaning |
|-------|---------|
| 80-100 | Very stable — rarely changes. Safe to build on. |
| 60-79 | Moderately stable — occasional changes. |
| 40-59 | Active — meaningful change rate. Be cautious. |
| 20-39 | Volatile — churn hotspot. Changes often. |
| 0-19 | Extremely volatile — consider refactoring or adding tests. |

Formula: `score = 100 - min(100, churn_per_30_days * 10) + long_quiet_bonus`, clamped 0-100.

### Dead-code detection

The sweep is **conservative**: a function is flagged only if ALL of:
- No inbound call edges of any kind (same-file or cross-file)
- Its short name does not appear as a word in any other file
- Not a dunder method (Python calls those by itself)
- Not in a test directory
- Not decorated with a framework decorator (`@route`, `@fixture`, etc.)
- Not a known entry-point name (`main`, `run`, `serve`, `handle`, etc.)

On a well-connected codebase this may return zero candidates. That's an honest result — the codebase is cleanly connected. The sweep is designed to be correct-when-silent, not loud-and-wrong.

### Notes

- Genesis / history / stability / authors / co-evolution all require a git repo. If run outside a git repo, the engine gracefully reports "no git history" and skips those features.
- Dead-code detection works without git — it's a pure AST + graph analysis.
- Full reports take 10-20 seconds on codebases with 500+ commits (most time is in co-evolution's per-function git log walk). Single-subcommand queries are much faster (1-5 seconds).
- All commands accept `--json` for machine-readable output.

---

## Code Retrieval (M6 — InfiniteContext, 105% Mythos lead)

LeanAI's code retrieval combines **three retrievers** whose weaknesses don't overlap, fuses their results with **reciprocal rank fusion** (Cormack et al. 2009), then re-ranks with **code-specific heuristics**. This is the one phase where LeanAI genuinely beats cloud AI on a specific axis — code-native retrieval grounded in YOUR AST + call graph.

Every regular chat query goes through this pipeline automatically. `/ask` is the explicit interface when you want a grounded answer with citations.

### Architecture

```
                  Your query: "how does caching work"
                               │
                               ▼
                      ┌─────────────────┐
                      │ QueryProcessor  │ extract content tokens, match named entities
                      └────────┬────────┘    from brain graph, detect test-intent
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   SEMANTIC   │    │     BM25     │    │    GRAPH     │
  │  (MiniLM     │    │  (keyword    │    │  (brain's    │
  │  embeddings) │    │   scoring)   │    │ call graph)  │
  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
      top-15              top-15              top-15
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
                   ┌───────────────────────┐
                   │   RRF fusion (k=60)   │ merge rankings via
                   └───────────┬───────────┘   Σ 1/(60+rank)
                               │
                               ▼
                   ┌───────────────────────┐
                   │      Reranker         │ name-match boost
                   │  (5 code heuristics)  │ docstring-match boost
                   └───────────┬───────────┘ class-match boost
                               │              recent-git boost
                               ▼              test-file penalty
                         Top-K chunks
```

### The three retrievers

**1. Semantic (MiniLM embeddings)** — good at conceptual queries ("how does caching work"), weak on exact-name queries. Runs in ~20ms.

**2. BM25** — lexical scoring over tokens. Good at exact-name queries ("find rescan_file"), technical terms, API calls. Weak on synonyms. Runs in ~6ms after first-query warmup.

**3. Graph (brain's call graph)** — walks your dependency graph from named entities in the query. For each query token that matches a function/class name in the brain, returns that chunk plus its 1-hop neighbors (callers + callees). Direct matches score 1.0, class members 0.75, 1-hop neighbors 0.5. Runs in ~12ms after warmup. **Unique to LeanAI — cloud AI doesn't have your AST call graph.**

### Why hybrid beats semantic-only

| Query | Semantic alone | Hybrid (LeanAI) |
|-------|---------------|-----------------|
| "how does caching work" | Good — MiniLM bridges to docstrings | Same, plus BM25 catches exact term if code uses it |
| "find rescan_file" | Weak — "rescan_file" isn't a concept | Strong — BM25 and graph both rank it #1 |
| "what calls ChainBreakerEngine._build_chain" | Weak — no concept of "calls" | Strong — graph retriever expands to actual callers |
| "where is the NullPointerException handler" | Weak — term in no file | Weak in both, honestly — but at least hybrid fails transparently |

### Reciprocal rank fusion

RRF (Cormack, Clarke, Buettcher 2009) is the canonical IR technique for combining rankings from multiple retrievers. For each chunk:

```
rrf_score(chunk) = Σ over retrievers  1 / (60 + rank_in_that_retriever)
```

**Why this is better than weighted-sum scoring:** if one retriever returns score 0.9 and another returns score 45, how do you combine them? Their scales are different. RRF only cares about the rank (position in the list), which normalizes automatically. A chunk that's #3 in two retrievers beats a chunk that's #1 in only one — rewarding cross-retriever consensus.

Most AI coding tools use naive weighted sum or semantic-only. LeanAI uses RRF.

### Reranker heuristics

After RRF produces the top-20 candidates, a second pass boosts/penalizes based on:

| Heuristic | Effect | Rationale |
|-----------|--------|-----------|
| Query-term in function name | +0.25 | If you asked for `_hash_file`, that function deserves top slot |
| Query-term in class name (for methods) | +0.20 | "ChainBreakerEngine" query should surface all its methods |
| Query-term in docstring | +0.15 (capped) | Docstring is curated explanation — high signal |
| File recently git-modified | +0.10 | Recently-touched code is more likely what's being asked about |
| Test file + query has no "test" token | −0.30 | Tests clutter results for non-test queries |

All five are tunable constants in `core/retrieval.py`.

### Commands

| Command | What it does |
|---------|-------------|
| `/ask <question>` | Grounded answer with citations. Retrieves top 5 via hybrid, sends to loaded model, returns answer |
| `/ask --raw <question>` | Top 5 chunks as a list with relevance scores + rerank reasons — no model involvement |
| `/index <path>` | Manually (re)index a project. Auto-runs on `/brain .` so rarely needed |
| `/brain <path>` | Scans project AND auto-indexes in one step |

### Diagnostic output

On the first search per session, the indexer prints its mode:

```
[Indexer] Search mode: hybrid (semantic + BM25 + graph) · 3147 chunks indexed
```

If the brain isn't wired (rare — auto-wired by `/brain .`), you'll see:

```
[Indexer] Search mode: semantic (embeddings only) · 3147 chunks indexed
```

If embeddings aren't loaded at all (e.g. `sentence-transformers` not installed):

```
[Indexer] Search mode: keyword fallback (flat 0.5 relevance) · 120 chunks indexed
```

Each of these modes is a legitimate fallback. The diagnostic tells you what's active.

### Rerank reasons (shown in `--raw` mode)

Every result in `/ask --raw` output includes the reasons the reranker adjusted its score:

```
[1] brain/project_brain.py:195-216  ProjectBrain.rescan_file  (final=1.40)
    reasons: ['name_match:file,rescan,rescan_file',
              'doc_match:file,rescan,rescan_file']
[2] brain/project_brain.py:218-224  ProjectBrain._hash_file  (final=0.95)
    reasons: ['name_match:hash,file']
```

This makes the retrieval decisions fully transparent. No black-box ranking.

### Example — grounded `/ask`

```
❯ /ask how does the brain invalidate stale caches?

  [M6] Retrieving 5 relevant chunks, answering with gemma-4-26B-A4B-it-UD-Q4_K_M.gguf...

The brain invalidates caches by hashing each tracked file's contents on every
scan. When rescan_file is called, it recomputes the hash via _hash_file and
compares against the stored FileState.content_hash; if they differ the file
is re-analyzed and its graph nodes are rebuilt [1][2]. File changes are
detected periodically by _check_for_changes which walks the tracked files
and compares modification timestamps [3].

  Sources consulted:
    [1] brain/project_brain.py:195-216 (ProjectBrain.rescan_file, relevance 100%)
    [2] brain/project_brain.py:218-224 (ProjectBrain._hash_file, relevance 95%)
    [3] brain/project_brain.py:246-266 (ProjectBrain._check_for_changes, relevance 88%)
    [4] tools/indexer.py:631-636 (ProjectIndexer._hash_file, relevance 74%)
    [5] brain/project_brain.py:380-412 (ProjectBrain._save_cache, relevance 68%)
  Answer time: 3.2s
```

### Example — raw mode

```
❯ /ask --raw where is rescan_file defined

  [100%] brain/project_brain.py:195-216  ProjectBrain.rescan_file
         reasons: ['name_match:rescan,rescan_file,file', 'doc_match:rescan,file']
  [95%]  brain/project_brain.py:218-224  ProjectBrain._hash_file
         reasons: ['name_match:file']
  [88%]  brain/project_brain.py:246-266  ProjectBrain._check_for_changes
         reasons: ['doc_match:file']
  [72%]  brain/dependency_graph.py:468-470  DependencyGraph.get_file_functions
         reasons: ['name_match:file']
  [65%]  brain/dependency_graph.py:472-474  DependencyGraph.get_file_classes
         reasons: ['name_match:file']
```

### When to use which

- **Default `/ask`** — for "how does X work" or "why does Y behave this way" questions. Model synthesizes across multiple code chunks.
- **`/ask --raw`** — for "where is X defined" or "find code that touches Y." Faster. Shows the rerank-transparent scoring.
- **Regular chat** — semantic+BM25+graph context is injected automatically. No flag needed.

### Hybrid retrieval safety (never-break guarantees)

- If ChromaDB/indexer is unavailable → falls back to pre-M6 lexical matching in `smart_context._get_brain_context`. Zero user-visible regression.
- If the indexer is empty → `/brain .` auto-indexes on first run with AST-grounded chunking active.
- If hybrid retrieval throws an error → falls back to semantic-only. Still works, silently.
- If semantic retrieval throws an error → falls back to keyword search. Still works, silently.
- If `/ask` is invoked with no model loaded → falls back to raw chunk listing with a hint.

### Performance

| Operation | Latency (warm cache) |
|-----------|---------------------|
| Semantic retriever | ~20ms |
| BM25 retriever | ~6ms |
| Graph retriever | ~12ms |
| RRF fusion + rerank | ~1ms |
| **Total hybrid search** | **~40-70ms** |
| First query (cold BM25 + graph cache) | 2-3s |
| `/ask` grounded answer | 2-5s (model inference dominates) |
| `/brain .` auto-index on first run | 10-20s for ~100 files (3000 chunks) |

### Notes

- **Index persists** per project to `~/.leanai/project_index/` (ChromaDB sqlite).
- **Incremental re-indexing**: `/index .` only re-embeds files whose hash changed.
- **Embedder**: `all-MiniLM-L6-v2` (384 dims, ~22MB, CPU-fast).
- **Chunking**: Python files chunked by AST function/class boundaries with headers containing docstrings, calls, and called-by metadata. Non-Python files chunked by regex boundaries.
- **BM25 backend**: `rank-bm25` package if installed, otherwise a manual 30-line Okapi implementation.

### Troubleshooting

**Symptom:** Search mode shows `keyword fallback (flat 0.5 relevance)`.
**Fix:** `pip install sentence-transformers` in the LeanAI venv, restart.

**Symptom:** Search mode shows `semantic (embeddings only)` instead of `hybrid`.
**Fix:** Run `/brain .` first — hybrid needs the brain wired. Auto-done on `/brain .`.

**Symptom:** Search results look bad despite hybrid mode active.
**Fix:** Your index was built with the pre-M6 regex chunker. Force a rebuild:
```powershell
Remove-Item -Recurse -Force $env:USERPROFILE\.leanai\project_index
```
(Bash: `rm -rf ~/.leanai/project_index`) — then restart and run `/brain .`. The indexer rebuilds with AST-grounded chunks.

**Symptom:** First query takes 2-3 seconds.
**Cause:** BM25 index and graph caches build lazily on first search. Warm queries are ~50ms.
**Fix:** Nothing needed — warm queries are fast. If you want to pre-warm, run any `/ask` immediately after `/brain .`.

**Symptom:** Chat answers slow or `decode: failed to find a memory slot` errors.
**Fix:** Brain-context budget is hard-capped at 2500 characters. If you still see this, use `/clear` to reset session state.

### Why this is a 105% Mythos lead

1. **AST-grounded chunks** (Stage 1) — Cloud AI chunks files on text boundaries. LeanAI chunks on AST function/class boundaries with qualified names and call-graph annotations baked into each chunk. Different semantic quality entirely.

2. **Graph retriever** (Stage 2) — Cloud AI has no call graph for your project. LeanAI has 14,799+ edges and does 1-hop expansion from any named entity in the query. This catches "what calls X" and "what does Y depend on" queries that pure embedding search structurally cannot.

3. **RRF fusion** (Stage 2) — Cormack et al. 2009, proven IR technique. Most AI coding tools use plain semantic or weighted-sum. LeanAI uses RRF, which normalizes across retrievers of different score scales.

4. **Code-specific reranker** (Stage 3) — Five heuristics tuned for code (name match, class match, docstring match, git recency, test penalty). Not generic text heuristics.

5. **Fully local, private, reproducible.** Same retrieval on the same query returns the same chunks. No API. No cloud. 4GB VRAM.

This is the first LeanAI phase that genuinely beats Mythos on a specific dimension — code-native retrieval grounded in AST + call graph — rather than matching a subset of what Mythos does.

---

## Persistent Knowledge Graph (M8 — MemoryForge)

`/memory` turns everything LeanAI learns about your project into a queryable, cross-session knowledge graph. Sentinel's findings, ChainBreaker's attack chains, and the brain's symbol index are all threaded into a single SQLite graph at `~/.leanai/memory_forge/graph.db`. The graph survives across sessions, so on day 90 you can still query findings discovered on day 1.

### The point

Cloud AI has no cross-session memory of your project. Every time you start a new conversation, it sees your code for the first time. LeanAI already generates rich structured data via M1 (Sentinel), M2 (ChainBreaker), M4 (Forensics) and M6 (retrieval). M8 is the layer that lets you **query across all of them simultaneously**, in plain English or raw DSL:

- *"Which functions have unresolved critical findings?"*
- *"Show me all attack chains that end in RCE."*
- *"What did Sentinel discover this week?"*
- *"Which complex functions haven't been touched since the last refactor?"*

Those are impossible on Claude/Copilot for a real codebase — the raw data they'd need to answer isn't in their context.

### Commands

| Command | What it does |
|---------|-------------|
| `/memory query <nl-or-dsl>` | Run a query (natural language or raw DSL) |
| `/memory facts <symbol>` | Every known fact about a function/class (findings, events, relations) |
| `/memory timeline [N]` | Chronological events (default 20) |
| `/memory stats` | Graph statistics (nodes, edges, last-sync, DB size) |
| `/memory sync` | Force an incremental re-ingestion from all sources |
| `/memory reset` | Wipe the graph and start fresh (requires confirmation) |
| `/memory help` | On-screen grammar reference |

### Auto-refresh

`/brain .` now invokes `/memory sync` automatically after indexing. Every brain scan keeps the knowledge graph current — no manual step required. A one-line summary appears when new data lands:

```
[M6] Indexed 110 files into 3100 semantic chunks in 18.7s — /ask is ready
[M8] MemoryForge: +8 symbols, +37 findings, +14 relations (92ms)
```

### DSL grammar

Small, strict, regular. No parser combinators. If you can read Python, you can read this.

```
<entity> [where <field> <op> <value> [and ...]] [limit N]

  entity    ::= symbols | findings | events
  op        ::= = | != | > | < | >= | <= | ~       ( ~ is substring match )

  symbols   fields: name, kind, file, line, signature, complexity, lines
  findings  fields: finding_id, kind, category, severity, confidence,
                    file, line, since
  events    fields: kind, source, description, since

  severity  values: CRITICAL | HIGH | MEDIUM | LOW | INFO   (ordered)
  since     accepts: 7d | 24h | 2026-04-01
```

### Examples

```
/memory query findings where severity >= HIGH
/memory query findings where category = sql_injection and severity = CRITICAL
/memory query symbols where complexity > 15
/memory query symbols where name ~ handle
/memory query events where source = sentinel and since = 7d
/memory query findings where file ~ core/server.py

/memory facts handle_request
/memory facts HTTPServer.handle

/memory timeline 30
/memory stats
```

### Natural language works too

If the model's loaded, `/memory query` will try to translate plain English into DSL first. If the model is flaky or off, a keyword-to-DSL heuristic kicks in as a fallback — it never needs a model.

```
/memory query show me all critical SQL injection findings
  → findings where severity = CRITICAL and category = sql_injection

/memory query which functions are complex
  → symbols where kind = function and complexity > 10

/memory query what did sentinel find this week
  → events where source = sentinel and since = 7d
```

If neither path produces parseable DSL, you get a clear error message with the grammar reference.

### Example session

```
▶ /memory stats
  MemoryForge Graph Stats

    Symbols    487
      class       42
      file        111
      function    287
      method      47
    Findings   37
      CRITICAL    2
      HIGH        11
      MEDIUM      24
    Events     89
    Relations  1204
      contains        534
      found_in        37
      affects         89
      depends_on      544

    Last sync: 12s ago
    DB size:   1.2 MB
    DB path:   ~/.leanai/memory_forge/graph.db

▶ /memory query findings where severity = CRITICAL
  DSL: findings where severity = CRITICAL
  Results: 2

    VULN-2026-0001 CRITICAL  sql_injection  conf=0.90
      core/server.py:42
      Unsanitized user input flows to SQL query.
    CHAIN-2026-0001 CRITICAL  rce            conf=0.85
      core/server.py:42
      Unauthenticated RCE

▶ /memory facts handle_request
  Facts for: handle_request

  Symbols (1)
    function  core/server.py::handle_request  core/server.py:42
      complexity=8  sig=def handle_request(req)

  Findings (1)
    VULN-2026-0001 CRITICAL  sql_injection  (found_in)

  Events (1)
    2026-04-20  discovery    sentinel  VULN-2026-0001 found: sql_injection
```

### How the graph is built

Three entity types, five relations, every row fingerprinted for idempotent re-sync:

- **Symbol** — function / method / class / file from `ProjectBrain._file_analyses`
- **Finding** — `VULN-*.json` (from Sentinel) and `CHAIN-*.json` (from ChainBreaker)
- **Event** — discovery, scan, modification, fix, sync (with source-tool attribution)

- **`contains`** — file → function, class → method (structural)
- **`found_in`** — finding → symbol (where it was discovered)
- **`affects`** — finding → symbol (downstream via taint path or chain step)
- **`depends_on`** — finding → finding, symbol → symbol (chain anchored to its entry vuln)
- **`modified_by`** — reserved for forensics integration

### Design rules (non-negotiable)

1. **Never trust model output as fact.** Every row in the graph has a `source_tool` column. The model is allowed to phrase answers, but all underlying facts come from deterministic tools (AST, Sentinel patterns, ChainBreaker graph walk).
2. **Never write to source code.** MemoryForge reads; it does not edit. Full-file replacement on 4GB VRAM is structurally unsafe — M8 is read-only by design.
3. **Idempotent sync.** Running `/memory sync` twice in a row produces zero new rows. All ingestion is keyed on stable fingerprints.
4. **Graceful degradation.** If `~/.leanai/vulns/` or `~/.leanai/chains/` don't exist (no scan has been run yet), sync silently skips them. `/memory` still works on whatever data is present.
5. **No model required.** The heuristic NL→DSL path lets `/memory query <English>` work even when no model is loaded, or when the loaded model is flaky on this task.

### Storage

- Database: `~/.leanai/memory_forge/graph.db` (honours `$LEANAI_HOME` if set)
- Schema version: stored in `schema_meta` for future migrations
- Size: typically 1–5 MB for a mid-sized project

### Non-goals

- Does **not** replace ChromaDB — M6 still does semantic retrieval of code chunks. M8 stores structured facts, not embeddings.
- Does **not** write to source code.
- Does **not** invent findings — the model translates questions, never produces facts.

### Lead magnitude: ~110% of Mythos

This is one of the phases where LeanAI genuinely beats Mythos, not just matches it. Cloud AI has no cross-session persistent structured memory of your project — every new conversation starts from zero context. LeanAI's M8 gives you a queryable graph that grows over time, with the model as a translator between human questions and deterministic fact lookups.

### What makes this better than Opus

Opus 4.6 has a massive context window, but it starts empty at every conversation. You'd have to re-paste your entire project, every Sentinel finding, every chain — every time — to get the same cross-cutting query capability. And Opus has no structural guarantee that its answer is grounded in your actual findings rather than plausible-sounding fabrication. MemoryForge queries return SQL result rows with stable IDs and source-tool attribution. Every fact is traceable.

---

## Code Execution

Run Python code directly in LeanAI's sandboxed environment.

### Commands

| Command | What it does |
|---------|-------------|
| `/run <code>` | Execute Python code in sandbox |

### Examples

```
/run print("Hello, World!")
```
Output:
```
Hello, World!
Execution time: 12ms
```

```
/run
import math
print(math.factorial(10))
```

### Safety

LeanAI's code executor blocks dangerous operations:
- `input()` — would hang waiting for keyboard input
- `os.remove()`, `shutil.rmtree()` — file deletion
- `os.system()`, `subprocess.call()` — shell commands
- `exit()`, `quit()` — process termination
- Project imports (`from core.`, etc.) — can't work in sandbox

---

## TDD Auto-Fix

Write a failing test. LeanAI writes code until it passes. The output is **proven correct**.

### Commands

| Command | What it does |
|---------|-------------|
| `/tdd <test code>` | Write tests, LeanAI generates implementation |
| `/tdd-desc <description>` | Describe what you want, LeanAI writes tests + code |

### Examples

```
/tdd
from calculator import add, subtract
def test_add(): assert add(2, 3) == 5
def test_subtract(): assert subtract(10, 3) == 7
```
Output:
```
TDD Loop: PASSED after 1 attempt | calculator.py generated
```

```
/tdd-desc a function that checks if a string is a palindrome
```
Output: LeanAI writes both the tests AND the implementation, then verifies they pass.

### How it works

1. You write the tests (what the code SHOULD do)
2. LeanAI generates code that makes them pass
3. If tests fail, LeanAI reads the error and tries again
4. Loops up to 5 times until all tests pass
5. Returns verified-working code

---

## Reasoning

Deep thinking for complex problems. Uses 3 passes: think → critique → refine.

### Commands

| Command | What it does |
|---------|-------------|
| `/reason <question>` | Deep analysis with chain-of-thought + self-critique |
| `/plan <task>` | Create a structured, actionable plan |
| `/decompose <problem>` | Break a complex problem into sub-problems |

### Examples

```
/reason Why do microservices increase operational complexity?
```
Output: A multi-pass analysis that first reasons through the problem, then critiques its own reasoning, then produces a refined final answer.

```
/plan Migrate our monolith to microservices
```
Output: A structured plan with phases, milestones, risks, and action items.

```
/decompose Build a real-time collaborative code editor
```
Output: Breaks the problem into manageable sub-problems like "conflict resolution", "WebSocket setup", "operational transform", etc.

### Tips

- Use `/reason` for questions that need depth, not just a quick answer
- Use `/plan` when you need a roadmap, not just a solution
- Takes longer (3 model passes) but significantly better answers

---

## Writing

Professional document generation. 4 passes: analyze → outline → draft → edit.

### Commands

| Command | What it does |
|---------|-------------|
| `/write <description>` | Write any document (auto-detects type) |
| `/essay <topic>` | Write an essay |
| `/report <topic>` | Write a technical report |

### Supported document types

Essay, report, article, documentation, email, proposal, blog post, README

### Examples

```
/write blog post about why local AI will replace cloud AI
/essay The impact of AI on software engineering
/report Q3 performance analysis for the engineering team
/write email to the team about the new deployment process
```

---

## Swarm Consensus

Three independent model passes vote on the answer. When all three agree: UNANIMOUS.

### Commands

| Command | What it does |
|---------|-------------|
| `/swarm <question>` | Get consensus answer from 3 passes |

### Examples

```
/swarm What is the time complexity of quicksort?
```
Output:
```
Swarm: 3 passes | UNANIMOUS | Confidence: 95%
Average case: O(n log n), Worst case: O(n²)
```

### When to use

- When you need high confidence (important decisions)
- When the answer matters and you want verification
- Takes 3x longer but much more reliable

---

## Build (Agentic)

Multi-step project builder. Decomposes a task and executes each step.

### Commands

| Command | What it does |
|---------|-------------|
| `/build <task>` | Build a multi-step project or feature |

### Examples

```
/build Create a REST API with user authentication
/build Add a caching layer to the database queries
/build Set up a CI/CD pipeline for this project
```

### How it works

1. Decomposes the task into steps
2. Executes each step sequentially
3. Shows progress and intermediate results
4. Produces working code for each step

---

## Refactoring

Rename symbols and find references across your entire project.

### Commands

| Command | What it does |
|---------|-------------|
| `/refs <symbol>` | Find all references (definitions, imports, calls) |
| `/rename <old> <new>` | Rename across every file (with confirmation) |

### Examples

```
/refs generate
```
Output: Lists every file and line where `generate` is defined, imported, or called.

```
/rename greet say_hello
```
Output: Shows all changes in a preview, then asks for confirmation before applying.

### Tips

- Always review the preview before confirming a rename
- Use `/refs` first to understand the scope of a rename

---

## Memory

LeanAI remembers facts, your preferences, and your profile across sessions.

### Commands

| Command | What it does |
|---------|-------------|
| `/remember <fact>` | Store a fact in memory |
| `/profile` | View your stored profile |
| `/search <query>` | Search across all memories |

### Examples

```
/remember I prefer using FastAPI over Flask
/remember Our database is PostgreSQL 15
/remember The deployment target is AWS ECS
```

Then later:
```
What framework should I use for the API?
```
LeanAI responds: "Based on your preference, FastAPI would be the best choice."

---

## Sessions

Persistent conversation history across sessions.

### Commands

| Command | What it does |
|---------|-------------|
| `/sessions` | List all past sessions |
| `/continue` | Resume the last session |
| `/search <query>` | Search across past conversations |

### Examples

```
/sessions
```
Output: Lists all sessions with dates, number of exchanges, and topics discussed.

```
/continue
```
Output: Loads context from your last session and continues where you left off.

---

## Evolution Tracking

Tracks how your project understanding evolves across sessions. Detects themes and predicts what you'll need next.

### Commands

| Command | What it does |
|---------|-------------|
| `/evolution narrative` | Your project's story across sessions |
| `/evolution insights` | Where each theme is heading |
| `/evolution predict` | What you'll likely ask about next |
| `/evolution stats` | Tracking statistics |

### Examples

```
/evolution narrative
```
Output:
```
Project Evolution:
  database (optimizing) — 8 queries across 3 sessions over 5 days
    Journey: exploring → building → optimizing
    Next: Look into connection pooling, read replicas, or query caching

  authentication (building) — 4 queries across 2 sessions
    Journey: exploring → building
    Next: Don't forget password hashing (bcrypt) and rate limiting on login
```

### How it works

LeanAI detects 7 themes automatically: database, authentication, API, testing, deployment, caching, architecture. It tracks which stage you're in (exploring → building → optimizing → maintaining) and predicts your trajectory.

---

## Fine-Tuning

Your model literally gets smarter from your code. Every interaction auto-collects training data.

### Commands

| Command | What it does |
|---------|-------------|
| `/finetune status` | Check readiness (data count, GPU, dependencies) |
| `/finetune collect` | Import training data from sessions |
| `/finetune train` | Start training (or export for Google Colab) |
| `/finetune adapters` | List trained LoRA adapters |
| `/finetune activate <name>` | Use a trained adapter |

### How it works

1. Every conversation auto-collects question-answer pairs
2. Pairs are filtered by quality (verified code gets priority)
3. When you have 50+ high-quality pairs, training is ready
4. QLoRA fine-tuning produces a LoRA adapter
5. The adapter makes the model learn YOUR coding patterns

### Tips

- Check `/finetune status` to see how many pairs you have
- The more you use LeanAI, the more training data it collects
- Training requires a GPU — if no GPU, export to Google Colab

---

## Model Management

Switch between models or download new ones.

### Commands

| Command | What it does |
|---------|-------------|
| `/model list` | List available models with status |
| `/model auto` | Auto-switch by query complexity (recommended) |
| `/model qwen-7b` | Manually use the 7B model |
| `/model qwen-32b` | Manually use the 32B model |
| `/model fast` | Always use the fastest model |
| `/model download <name>` | Download a model |

### Auto mode (recommended)

```
/model auto
```

In auto mode, LeanAI classifies each query:
- **Simple** ("what is Python", "hello world") → 7B model (~30s)
- **Medium** ("write a function") → 7B model
- **Complex** ("explain this pipeline", "review this code") → 32B model (~7min with GPU)

### Tips

- Always use `/model auto` for the best experience
- Use `/model qwen-7b` for rapid-fire simple questions
- Use `/model qwen-32b` when quality matters more than speed

---

## System

System status, speed optimization, and help.

### Commands

| Command | What it does |
|---------|-------------|
| `/status` | Full system status (model, memory, sessions, git) |
| `/speed` | Speed optimization report + cache stats |
| `/echo` | CodeEcho acceleration stats (source-grounded speculative decoding) |
| `/dualpipe` | DualPipe speculative decoding (on/off/stats) |
| `/help` | Show available commands |
| `/quit` | Exit LeanAI |

### Examples

```
/status
```
Shows: model loaded, memory count, session count, git branch, training pairs, adapter status.

```
/speed
```
Shows: cache hit rate, GPU detection, recommended optimizations.

```
/echo
```
Shows: CodeEcho acceleration stats — total tokens echoed, echo events, echo ratio, estimated speedup. CodeEcho detects when the model reproduces source code and batch-injects those tokens at prefill speed instead of generating them one by one.

```
/dualpipe on
/dualpipe off
/dualpipe
```
Controls DualPipe asymmetric speculative decoding. Loads 7B on GPU (draft) and 27B on CPU (verify) simultaneously. `/dualpipe` shows stats including acceptance rate, effective tok/s, and speedup. Experimental — requires both 7B and 27B models downloaded.

---

## Keyboard Shortcuts (VS Code)

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+L` | Open LeanAI chat |
| `Ctrl+Shift+E` | Explain selected code |
| `Ctrl+Shift+F` | Fix selected code |
| Right-click → LeanAI | Context menu options |

---

## Common Workflows

### "I just cloned a new project"

```
/brain .                    # understand the project
/git hotspots               # see what's actively changing
/describe main.py           # understand the entry point
/deps core/engine.py        # see the dependency tree
```

### "I need to fix a bug"

```
/bisect login is broken     # find which commit caused it
/git history auth.py        # see what changed in auth
/fuzz def login(user, pw):  # test edge cases
```

### "I want to add a feature"

```
/impact models.py           # what breaks if I change this
/tdd-desc add email validation to user registration
/build Add rate limiting to the API endpoints
```

### "I want to improve code quality"

```
/fuzz def process(data): return data.strip().lower()
/reason What are the security risks in our authentication flow?
/swarm Is our caching strategy correct for this use case?
```

---

*For more information, visit the [GitHub repository](https://github.com/gowrishankar-infra/leanai).*
