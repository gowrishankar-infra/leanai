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
- [Sentinel — Autonomous Security Analyzer](#sentinel--autonomous-security-analyzer)
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
- Chains feed forward into ExploitForge (M3), AutoFix (M5), and Aegis (M7)

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
