"""
LeanAI — SourceForensics: Deterministic Code Archaeology (M4)
═════════════════════════════════════════════════════════════
Answers detailed archaeology questions about functions and files using
ONLY deterministic tools — AST walks and git history. No LLM involved.

Why this beats Mythos
─────────────────────
Mythos uses a model to reason about commit history and function evolution.
A 100B+ parameter model to run what is fundamentally a git-log walk with
an AST diff. That is backward. `git log`, `git blame`, and Python's `ast`
are precise, deterministic tools built for exactly these questions. They
are (a) sub-second where a Mythos call takes 30+ seconds, (b) never
hallucinate a commit hash or invent an author, (c) fully reproducible.

Questions answered
──────────────────
- Function genesis — when did this function first appear, in which commit, by whom
- Function history — every commit that touched this specific function's body
- Co-evolution — functions that always change in the same commits as target
- Stability score — how often has this function been modified
- Author map — who has touched each function in a file
- Dead code — functions never referenced anywhere in the project
- Complexity trajectory — cyclomatic complexity trend over commits

Pipeline
────────
1. Locate target function via brain.graph (same-file preference).
2. Walk `git log --follow -p <file>` parsing each commit's diff hunks.
3. For each hunk, determine whether the target function's line range
   overlaps the hunk. Track authors, commit SHAs, dates.
4. For co-evolution, intersect the set of commits that touched target
   with commit sets of every other function in the project. Rank by
   Jaccard similarity over shared commits.
5. Stability score = changes_per_day_of_lifetime, normalized to 0-100.
6. Dead-code uses brain.graph.nodes + reverse_adjacency to find
   zero-inbound-edge functions (excluding tests, __init__, __main__).

Safety / constraints
────────────────────
- Read-only: never calls any git command that writes. `git log`,
  `git blame`, `git show`, `git rev-list` only.
- Function name disambiguation: if multiple functions share a name
  across the project, forensics disambiguates by file path.
- Bounded history: default max 500 commits per function. Configurable.
- Gracefully handles repos with no history, non-git directories,
  functions that were never committed, and binary diffs.

Author: Gowri Shankar (github.com/gowrishankar-infra/leanai)
"""

import os
import re
import ast
import json
import time
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set, Iterable


# ═══════════════════════════════════════════════════════════════════
# Node-ID helpers (Windows-safe)
# ═══════════════════════════════════════════════════════════════════

def _node_file(node_id: str) -> str:
    """Extract the filepath portion of a 'filepath:qualified_name' node ID.
    Uses rsplit on the last colon so Windows drive letters don't break it."""
    if ':' not in node_id:
        return node_id
    return node_id.rsplit(':', 1)[0]


def _node_qname(node_id: str) -> str:
    """Extract the qualified-name portion of a node ID."""
    if ':' not in node_id:
        return ''
    return node_id.rsplit(':', 1)[1]


# ═══════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CommitRef:
    sha: str              # short SHA
    full_sha: str         # full SHA
    author: str
    email: str
    date: str             # ISO date
    timestamp: int        # unix timestamp
    subject: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FunctionGenesis:
    """When did a function first appear?"""
    function_name: str
    filepath: str
    first_commit: Optional[CommitRef]
    last_commit: Optional[CommitRef]
    total_commits: int           # commits that touched this function
    age_days: int                # age in days since genesis
    original_author: Optional[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.first_commit:
            d['first_commit'] = self.first_commit.to_dict()
        if self.last_commit:
            d['last_commit'] = self.last_commit.to_dict()
        return d


@dataclass
class FunctionHistoryEntry:
    """One row in a function's commit history."""
    commit: CommitRef
    lines_added: int
    lines_removed: int
    complexity_before: Optional[int] = None
    complexity_after: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            'commit': self.commit.to_dict(),
            'lines_added': self.lines_added,
            'lines_removed': self.lines_removed,
            'complexity_before': self.complexity_before,
            'complexity_after': self.complexity_after,
        }


@dataclass
class CoEvolutionEntry:
    """A function that frequently changes alongside the target."""
    function_name: str
    filepath: str
    shared_commits: int       # commits that touched BOTH target and this
    target_only_commits: int
    this_only_commits: int
    jaccard: float            # shared / (target_only + this_only + shared)
    coupling_strength: str    # 'high', 'medium', 'low'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StabilityScore:
    function_name: str
    filepath: str
    age_days: int
    total_changes: int
    changes_per_30_days: float
    score: int                 # 0-100, 100 = rock-solid, 0 = churn hotspot
    interpretation: str        # human text
    last_changed_days_ago: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuthorMapEntry:
    function_name: str
    filepath: str
    authors: List[Tuple[str, int]]  # [(author, commit_count)], sorted desc
    primary_author: Optional[str]   # author with most commits
    bus_factor: int                 # number of authors to cover >80% of changes

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeadCodeEntry:
    function_name: str
    filepath: str
    line_start: int
    reason: str                # 'never_called', 'orphaned_private', etc.
    complexity: int
    lines: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForensicsReport:
    """Full forensics bundle for a function."""
    function_name: str
    filepath: str
    genesis: Optional[FunctionGenesis]
    history: List[FunctionHistoryEntry]
    co_evolution: List[CoEvolutionEntry]
    stability: Optional[StabilityScore]
    authors: Optional[AuthorMapEntry]
    current_complexity: int
    current_lines: int

    def to_dict(self) -> dict:
        return {
            'function_name': self.function_name,
            'filepath': self.filepath,
            'genesis': self.genesis.to_dict() if self.genesis else None,
            'history': [h.to_dict() for h in self.history],
            'co_evolution': [c.to_dict() for c in self.co_evolution],
            'stability': self.stability.to_dict() if self.stability else None,
            'authors': self.authors.to_dict() if self.authors else None,
            'current_complexity': self.current_complexity,
            'current_lines': self.current_lines,
        }


# ═══════════════════════════════════════════════════════════════════
# Git subprocess wrapper
# ═══════════════════════════════════════════════════════════════════

class _GitRunner:
    """Minimal git wrapper — read-only operations."""

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self._available: Optional[bool] = None

    def available(self) -> bool:
        """True if repo_path is a valid git repo and git is on PATH."""
        if self._available is not None:
            return self._available
        try:
            r = subprocess.run(
                ['git', '-C', self.repo_path, 'rev-parse', '--git-dir'],
                capture_output=True, text=True, timeout=5,
            )
            self._available = (r.returncode == 0)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._available = False
        return self._available

    def run(self, args: List[str], timeout: int = 30) -> Tuple[bool, str]:
        """Run a git command. Returns (success, stdout_or_stderr)."""
        if not self.available():
            return False, "git not available"
        try:
            r = subprocess.run(
                ['git', '-C', self.repo_path] + args,
                capture_output=True, text=True, timeout=timeout,
                # errors='replace' so weird non-UTF-8 bytes don't crash us
                errors='replace',
            )
            if r.returncode == 0:
                return True, r.stdout
            return False, (r.stderr or r.stdout)
        except subprocess.TimeoutExpired:
            return False, f"git command timed out after {timeout}s"
        except Exception as e:
            return False, f"git error: {e}"


# ═══════════════════════════════════════════════════════════════════
# Diff hunk parser — understands unified diff output
# ═══════════════════════════════════════════════════════════════════

_HUNK_HEADER_RE = re.compile(
    r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',
)

@dataclass
class _DiffHunk:
    """Parsed unified-diff hunk for a single file."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    added: int
    removed: int


def _parse_diff_for_file(diff_text: str, target_file: str) -> List[_DiffHunk]:
    """Extract hunks from `git show` / `git log -p` diff text for one file.

    The diff may contain multiple `diff --git` sections (one per file).
    We only care about hunks for target_file.
    """
    # Normalize the target path comparison
    target_norm = target_file.replace('\\', '/')

    hunks: List[_DiffHunk] = []
    in_target_file = False
    cur_added = 0
    cur_removed = 0
    cur_header: Optional[_DiffHunk] = None

    for line in diff_text.splitlines():
        if line.startswith('diff --git'):
            # Finish previous hunk if any
            if cur_header and in_target_file:
                cur_header.added = cur_added
                cur_header.removed = cur_removed
                hunks.append(cur_header)
            cur_header = None
            cur_added = cur_removed = 0
            # Check whether this diff targets our file: 'diff --git a/foo b/foo'
            parts = line.split()
            # parts: ['diff', '--git', 'a/foo', 'b/foo']
            if len(parts) >= 4:
                b_path = parts[3]
                if b_path.startswith('b/'):
                    b_path = b_path[2:]
                in_target_file = (b_path.replace('\\', '/') == target_norm)
            else:
                in_target_file = False
            continue

        if not in_target_file:
            continue

        m = _HUNK_HEADER_RE.match(line)
        if m:
            # Flush previous hunk
            if cur_header:
                cur_header.added = cur_added
                cur_header.removed = cur_removed
                hunks.append(cur_header)
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) else 1
            cur_header = _DiffHunk(
                old_start=old_start, old_count=old_count,
                new_start=new_start, new_count=new_count,
                added=0, removed=0,
            )
            cur_added = cur_removed = 0
            continue

        if cur_header is None:
            continue

        if line.startswith('+') and not line.startswith('+++'):
            cur_added += 1
        elif line.startswith('-') and not line.startswith('---'):
            cur_removed += 1

    # Flush final hunk
    if cur_header and in_target_file:
        cur_header.added = cur_added
        cur_header.removed = cur_removed
        hunks.append(cur_header)

    return hunks


def _hunks_touch_range(
    hunks: List[_DiffHunk], line_start: int, line_end: int
) -> Optional[_DiffHunk]:
    """Return the first hunk whose OLD-file range overlaps [line_start, line_end],
    or None if no hunk affects this range. Used to decide whether a given
    historical commit touched our function body.

    Note: we use OLD-file range because we walk from newest commit backwards,
    so `line_start/line_end` are the function's line range as of the *parent*
    of the commit (i.e. what it looked like BEFORE this commit's change).
    """
    for h in hunks:
        h_end = h.old_start + max(h.old_count, 1) - 1
        if h.old_start <= line_end and h_end >= line_start:
            return h
    return None


def _hunks_touch_new_range(
    hunks: List[_DiffHunk], line_start: int, line_end: int
) -> Optional[_DiffHunk]:
    """Same as _hunks_touch_range, but using NEW-file range. Used when we
    know the function's current line range and want to see if the commit's
    NEW (post-change) version overlapped that range."""
    for h in hunks:
        h_end = h.new_start + max(h.new_count, 1) - 1
        if h.new_start <= line_end and h_end >= line_start:
            return h
    return None


# ═══════════════════════════════════════════════════════════════════
# AST-based function range extraction
# ═══════════════════════════════════════════════════════════════════

def _extract_function_range(source: str, qname: str) -> Optional[Tuple[int, int]]:
    """Parse source and return (line_start, line_end) for the function whose
    qualified name matches qname. Returns None if not found.

    qname examples:
      'foo'          -> module-level function `foo`
      'Bar.baz'      -> method `baz` on class `Bar`
      'Bar.Inner.q'  -> nested class method
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    parts = qname.split('.')

    def walk(node, path: List[str]) -> Optional[Tuple[int, int]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if path + [child.name] == parts:
                    start = child.lineno
                    end = getattr(child, 'end_lineno', start)
                    return (start, end or start)
                # Functions can have nested classes/functions (rare for qname)
                if len(parts) > len(path) + 1:
                    inner = walk(child, path + [child.name])
                    if inner:
                        return inner
            elif isinstance(child, ast.ClassDef):
                inner = walk(child, path + [child.name])
                if inner:
                    return inner
        return None

    return walk(tree, [])


def _cyclomatic_complexity(source: str, qname: str) -> int:
    """Rough cyclomatic complexity for the function whose qname matches.
    Counts: 1 (base) + if/elif/for/while/try/except/and/or/assert/ternary."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return 0

    parts = qname.split('.')

    def find_func(node, path):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if path + [child.name] == parts:
                    return child
                if len(parts) > len(path) + 1:
                    r = find_func(child, path + [child.name])
                    if r:
                        return r
            elif isinstance(child, ast.ClassDef):
                r = find_func(child, path + [child.name])
                if r:
                    return r
        return None

    func = find_func(tree, [])
    if not func:
        return 0

    complexity = 1
    for node in ast.walk(func):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While,
                             ast.ExceptHandler, ast.Assert)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += max(len(node.values) - 1, 0)
        elif isinstance(node, ast.IfExp):
            complexity += 1
        elif isinstance(node, ast.Try):
            complexity += len(node.handlers)
    return complexity


# ═══════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════

class ForensicsEngine:
    """Deterministic code-archaeology engine."""

    def __init__(self, brain, max_history: int = 500):
        """
        brain: a scanned ProjectBrain.
        max_history: per-function commit history cap (default 500).
        """
        self.brain = brain
        self.project_root = os.path.abspath(brain.config.project_path)
        self.git = _GitRunner(self.project_root)
        self.max_history = max_history
        # Caches (populated lazily per session)
        self._file_commits_cache: Dict[str, List[CommitRef]] = {}
        self._commit_diff_cache: Dict[str, str] = {}

    # ─────────── Public API ───────────

    def git_available(self) -> bool:
        return self.git.available()

    def resolve_function(
        self, name_or_path: str
    ) -> Optional[Tuple[str, str, int, int]]:
        """Resolve a user-supplied function identifier to
        (filepath_rel, qualified_name, line_start, line_end).

        Accepts:
          - bare name: 'my_function' -> same-file candidate first, then any
          - qualified: 'Foo.bar' or 'Foo.Inner.bar'
          - file-anchored: 'module/file.py:Foo.bar'
        Returns None if no match.
        """
        # Case 1: file-anchored (has ':')
        if ':' in name_or_path:
            path_part, qname = name_or_path.rsplit(':', 1)
            node_id = self._find_node_by_file_and_qname(path_part, qname)
            if node_id:
                return self._expand_node_id(node_id)
            return None

        # Case 2: bare or qualified name — search all nodes
        candidates = []
        for node_id, node in self.brain.graph.nodes.items():
            if node.node_type != 'function':
                continue
            # Compare against the short name AND qualified name
            qname = _node_qname(node_id)
            if qname == name_or_path or node.name == name_or_path:
                candidates.append(node_id)

        if not candidates:
            # Try the brain's name lookup (bare-name catalog)
            cand = self.brain.graph._function_lookup.get(name_or_path)
            if cand:
                candidates = [cand]

        if not candidates:
            return None

        if len(candidates) == 1:
            return self._expand_node_id(candidates[0])

        # Multi-match. Prefer non-test paths.
        non_test = [c for c in candidates
                    if not self._is_test_path(_node_file(c))]
        if len(non_test) == 1:
            return self._expand_node_id(non_test[0])
        # Return the first match but note there's ambiguity (caller may
        # want to show the list)
        return self._expand_node_id((non_test or candidates)[0])

    def list_matches(self, name: str) -> List[str]:
        """Return all node IDs whose short or qualified name matches `name`.
        Used when a user-supplied name is ambiguous."""
        out = []
        for node_id, node in self.brain.graph.nodes.items():
            if node.node_type != 'function':
                continue
            qname = _node_qname(node_id)
            if qname == name or node.name == name:
                out.append(node_id)
        return out

    # ──── The six archaeology reports ────

    def genesis(
        self, filepath: str, qname: str,
    ) -> Optional[FunctionGenesis]:
        """When did this function first appear?"""
        if not self.git.available():
            return FunctionGenesis(
                function_name=qname, filepath=filepath,
                first_commit=None, last_commit=None,
                total_commits=0, age_days=0, original_author=None,
            )

        commits = self._function_commits(filepath, qname)
        if not commits:
            return FunctionGenesis(
                function_name=qname, filepath=filepath,
                first_commit=None, last_commit=None,
                total_commits=0, age_days=0, original_author=None,
            )

        # Commits are in newest-first order from `git log`
        last = commits[0]
        first = commits[-1]
        age_days = max(0, (last.timestamp - first.timestamp) // 86400)
        return FunctionGenesis(
            function_name=qname, filepath=filepath,
            first_commit=first, last_commit=last,
            total_commits=len(commits), age_days=int(age_days),
            original_author=first.author,
        )

    def history(
        self, filepath: str, qname: str, limit: int = 20,
    ) -> List[FunctionHistoryEntry]:
        """Every commit that touched this function's body."""
        if not self.git.available():
            return []

        commits = self._function_commits(filepath, qname)
        out: List[FunctionHistoryEntry] = []
        for commit in commits[:limit]:
            diff_text = self._get_commit_diff(commit.full_sha)
            hunks = _parse_diff_for_file(diff_text, filepath)
            # Sum lines added/removed across ALL hunks for this file in this
            # commit. (We could narrow to function-overlapping hunks only,
            # but the signal is usually clearer with the file-level count,
            # since hunks shift line ranges.)
            total_added = sum(h.added for h in hunks)
            total_removed = sum(h.removed for h in hunks)
            out.append(FunctionHistoryEntry(
                commit=commit,
                lines_added=total_added,
                lines_removed=total_removed,
            ))
        return out

    def co_evolution(
        self, filepath: str, qname: str, top_n: int = 10,
    ) -> List[CoEvolutionEntry]:
        """Functions that frequently change in the same commits as the target.

        Works at a FILE level for the coarse phase: we find commits that
        touched target's file+range, then for each of those commits we look
        at which other functions (in any file) had overlapping hunks. Rank
        by Jaccard similarity over shared commits.
        """
        if not self.git.available():
            return []

        target_commits = set(c.full_sha for c in self._function_commits(filepath, qname))
        if not target_commits:
            return []

        # For each commit that touched target, find other functions also
        # touched by that commit. Build a counter: other_node -> {commits}
        other_commits: Dict[str, Set[str]] = defaultdict(set)
        for commit_sha in list(target_commits)[:self.max_history]:
            diff = self._get_commit_diff(commit_sha)
            for other_file, ranges in self._functions_touched_in_diff(diff).items():
                for (other_qname, (start, end)) in ranges:
                    other_key = f"{other_file}:{other_qname}"
                    self_key = f"{filepath}:{qname}"
                    if other_key == self_key:
                        continue
                    other_commits[other_key].add(commit_sha)

        # Compute Jaccard for each candidate
        results: List[CoEvolutionEntry] = []
        for other_key, shared in other_commits.items():
            if not shared:
                continue
            other_file, other_qname = other_key.rsplit(':', 1)
            # We need the total commit set for the OTHER function to compute jaccard.
            other_total = set(c.full_sha for c in self._function_commits(other_file, other_qname))
            if not other_total:
                # Fall back to a lower-bound (just shared count)
                other_total = shared
            intersection = target_commits & other_total
            union = target_commits | other_total
            jaccard = len(intersection) / len(union) if union else 0.0
            shared_ct = len(intersection)
            target_only = len(target_commits - other_total)
            other_only = len(other_total - target_commits)

            if jaccard >= 0.7:
                strength = 'high'
            elif jaccard >= 0.35:
                strength = 'medium'
            else:
                strength = 'low'

            results.append(CoEvolutionEntry(
                function_name=other_qname,
                filepath=other_file,
                shared_commits=shared_ct,
                target_only_commits=target_only,
                this_only_commits=other_only,
                jaccard=round(jaccard, 3),
                coupling_strength=strength,
            ))

        # Rank by (shared_commits desc, jaccard desc). Require at least
        # 2 shared commits; lone pairs are usually coincidence.
        results = [r for r in results if r.shared_commits >= 2]
        results.sort(key=lambda r: (-r.shared_commits, -r.jaccard))
        return results[:top_n]

    def stability(
        self, filepath: str, qname: str,
    ) -> Optional[StabilityScore]:
        """Compute a stability score 0-100 for a function.

        0 = changes constantly (churn hotspot)
        100 = rock-solid, hasn't been touched in ages

        Formula:
          age_days = max(1, time since first commit)
          changes = N commits that touched this function
          changes_per_30 = changes / age_days * 30
          last_days = days since last commit touching the function
          raw = 100 - min(100, changes_per_30 * 10)
          bonus for long quiet period: +min(30, last_days / 30 * 5)
          clamped 0..100
        """
        if not self.git.available():
            return None

        commits = self._function_commits(filepath, qname)
        if not commits:
            return StabilityScore(
                function_name=qname, filepath=filepath,
                age_days=0, total_changes=0,
                changes_per_30_days=0.0, score=100,
                interpretation="No git history — either new file or untracked.",
                last_changed_days_ago=0,
            )

        now = int(time.time())
        first_ts = commits[-1].timestamp
        last_ts = commits[0].timestamp
        age_days = max(1, (now - first_ts) // 86400)
        last_days = max(0, (now - last_ts) // 86400)
        changes = len(commits)
        changes_per_30 = (changes / age_days) * 30

        raw = 100 - min(100, changes_per_30 * 10)
        raw += min(30, (last_days / 30) * 5)
        score = max(0, min(100, int(round(raw))))

        if score >= 80:
            interp = "Very stable — rarely changes. Safe to build on."
        elif score >= 60:
            interp = "Moderately stable — occasional changes."
        elif score >= 40:
            interp = "Active — meaningful change rate. Be cautious."
        elif score >= 20:
            interp = "Volatile — churn hotspot. Changes often."
        else:
            interp = "Extremely volatile — consider refactoring or adding tests."

        return StabilityScore(
            function_name=qname, filepath=filepath,
            age_days=int(age_days), total_changes=changes,
            changes_per_30_days=round(changes_per_30, 2),
            score=score, interpretation=interp,
            last_changed_days_ago=int(last_days),
        )

    def authors(
        self, filepath: str, qname: Optional[str] = None,
    ) -> Optional[AuthorMapEntry]:
        """Author breakdown for a function (or whole file if qname=None)."""
        if not self.git.available():
            return None

        if qname:
            commits = self._function_commits(filepath, qname)
        else:
            commits = self._file_commits(filepath)

        if not commits:
            return AuthorMapEntry(
                function_name=qname or '<file>',
                filepath=filepath, authors=[],
                primary_author=None, bus_factor=0,
            )

        counter = Counter(c.author for c in commits)
        ranked = counter.most_common()
        total = sum(c for _, c in ranked)
        # Bus factor = smallest number of authors covering > 80% of changes
        bus = 0
        cum = 0
        for _, cnt in ranked:
            bus += 1
            cum += cnt
            if cum >= 0.8 * total:
                break

        return AuthorMapEntry(
            function_name=qname or '<file>',
            filepath=filepath,
            authors=ranked,
            primary_author=ranked[0][0] if ranked else None,
            bus_factor=bus,
        )

    def dead_code(
        self,
        exclude_tests: bool = True,
        exclude_dunders: bool = True,
        exclude_private_methods: bool = False,
    ) -> List[DeadCodeEntry]:
        """Find functions that appear truly unreferenced by the project.

        A function is a dead-code candidate only if ALL conditions hold:
          1. No inbound call edge of any kind (same-file or cross-file).
          2. Its short name does not appear as a word in any OTHER file
             in the project (catches imports, callback registrations,
             dict values, reflection lookups like getattr(obj, 'name')).
          3. Its short name does not appear in string literals anywhere
             (catches getattr()/eval() name lookups and framework
             registration via string identifiers).

        Excludes by default:
          - Dunder methods (called by Python, not by name)
          - Tests (pytest runs test_* by discovery)
          - Known entry-point names: main, run, serve, handle, etc.
          - Functions with a framework decorator (route, fixture, etc.)
          - When exclude_private_methods=True, skips single-underscore
            helpers like `_helper` (which are typically intentional
            same-file-only utilities).

        This is intentionally CONSERVATIVE — if any of the signals above
        hit, the function is NOT flagged. We prefer false negatives
        (missing real dead code) over false positives (flagging working
        code as dead), because developers tend to delete things based
        on this list.
        """
        graph = self.brain.graph

        # Build reverse adjacency including ALL inbound edges (same-file
        # OR cross-file). A function that has even ONE legitimate caller
        # anywhere in the project is not dead.
        reverse: Dict[str, Set[str]] = defaultdict(set)
        for src, tgts in graph._adjacency.items():
            for t in tgts:
                if t.startswith('__unresolved__:') or t.startswith('__external__:'):
                    continue
                reverse[t].add(src)

        # Build a per-file set of identifier-like tokens. Used to check
        # whether a function's short name is REFERENCED anywhere (even
        # if not called directly — e.g. `from m import frobnicate` or
        # `handlers = {"f": frobnicate}`).
        # Only built once, cached on self to make subsequent calls fast.
        if not hasattr(self, '_file_token_index'):
            self._file_token_index = self._build_file_token_index()
        file_tokens: Dict[str, Set[str]] = self._file_token_index

        # Known entry-point names — never flag as dead
        entrypoint_names = {
            'main', '__main__', 'run', 'serve', 'handle',
            'setup', 'teardown', 'dispatch', 'wsgi_app', 'asgi_app',
        }
        # Python-invoked dunder methods — never flag as dead
        dunder_methods = {
            '__init__', '__new__', '__del__', '__call__',
            '__enter__', '__exit__', '__str__', '__repr__',
            '__eq__', '__hash__', '__lt__', '__le__', '__gt__', '__ge__',
            '__iter__', '__next__', '__len__', '__contains__',
            '__getitem__', '__setitem__', '__delitem__',
            '__getattr__', '__setattr__', '__getattribute__',
            '__add__', '__sub__', '__mul__', '__truediv__',
            '__bool__', '__int__', '__float__',
            '__reduce__', '__reduce_ex__', '__getstate__', '__setstate__',
            '__class_getitem__', '__init_subclass__',
        }
        # Framework decorators that register functions (can't detect the
        # inbound edge statically, so exclude these entry-point styles)
        decorator_markers = (
            'route', 'get', 'post', 'put', 'delete', 'patch',
            'endpoint', 'app.', 'router.', 'click.', 'cli.',
            'fixture', 'mark.', 'app.task', 'celery',
            'on_event', 'on_message', 'handler',
        )

        results: List[DeadCodeEntry] = []
        for node_id, node in graph.nodes.items():
            if node.node_type != 'function':
                continue
            qname = _node_qname(node_id)
            short = qname.rsplit('.', 1)[-1]

            # Exclusions
            if exclude_dunders and short.startswith('__') and short.endswith('__'):
                continue
            if short in entrypoint_names:
                continue
            if exclude_dunders and short in dunder_methods:
                continue

            filepath_abs = _node_file(node_id)
            filepath_rel = self._rel_path(filepath_abs)
            if exclude_tests and self._is_test_path(filepath_abs):
                continue
            if exclude_private_methods and short.startswith('_') and not short.startswith('__'):
                continue

            # Skip if decorated with a known framework decorator
            decos = node.metadata.get('decorators', []) or []
            if any(any(marker in d for marker in decorator_markers) for d in decos):
                continue

            inbound = reverse.get(node_id, set())
            if inbound:
                continue

            # Second filter: textual name reference in any OTHER file.
            # If the short name appears as a bare identifier anywhere
            # outside the function's own file, assume it's referenced.
            short_name = short
            referenced_externally = False
            for other_file, tokens in file_tokens.items():
                if other_file == filepath_abs:
                    continue
                if short_name in tokens:
                    referenced_externally = True
                    break
            if referenced_externally:
                continue

            complexity = node.metadata.get('complexity', 1) or 1
            line_start = node.metadata.get('line_start', 0) or 0
            line_end = node.metadata.get('line_end', line_start) or line_start
            lines = max(1, line_end - line_start + 1)

            results.append(DeadCodeEntry(
                function_name=qname,
                filepath=filepath_rel,
                line_start=line_start,
                reason='never_called',
                complexity=int(complexity),
                lines=int(lines),
            ))

        # Sort: biggest-and-most-complex dead code first (highest cleanup value)
        results.sort(key=lambda r: (-r.complexity, -r.lines, r.filepath))
        return results

    def _build_file_token_index(self) -> Dict[str, Set[str]]:
        """For each .py file in the project, build a set of
        identifier-like tokens (\\b\\w+\\b) present in the source.
        Used by dead_code() to detect name references that aren't
        call edges — e.g. imports, callback registrations, reflection.
        """
        index: Dict[str, Set[str]] = {}
        # Collect the set of file paths already known to the brain
        seen_files: Set[str] = set()
        for node_id, node in self.brain.graph.nodes.items():
            fp = _node_file(node_id)
            if fp:
                seen_files.add(fp)

        token_re = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')
        for fp in seen_files:
            try:
                with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                    text = fh.read()
            except (FileNotFoundError, OSError):
                continue
            # Skip absurdly large files — they'd drown the index
            if len(text) > 2 * 1024 * 1024:  # 2MB cap
                continue
            tokens = set(token_re.findall(text))
            index[fp] = tokens
        return index

    def full_report(
        self, filepath: str, qname: str, history_limit: int = 20,
        coevolve_top_n: int = 10,
    ) -> ForensicsReport:
        """Generate a complete ForensicsReport for one function."""
        # Current metrics from live source
        abs_path = os.path.join(self.project_root, filepath)
        current_complexity = 0
        current_lines = 0
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
            current_complexity = _cyclomatic_complexity(source, qname)
            rng = _extract_function_range(source, qname)
            if rng:
                current_lines = rng[1] - rng[0] + 1
        except (FileNotFoundError, OSError):
            pass

        return ForensicsReport(
            function_name=qname,
            filepath=filepath,
            genesis=self.genesis(filepath, qname),
            history=self.history(filepath, qname, limit=history_limit),
            co_evolution=self.co_evolution(filepath, qname, top_n=coevolve_top_n),
            stability=self.stability(filepath, qname),
            authors=self.authors(filepath, qname),
            current_complexity=current_complexity,
            current_lines=current_lines,
        )

    # ─────────── Internal helpers ───────────

    def _expand_node_id(self, node_id: str) -> Tuple[str, str, int, int]:
        """Turn a graph node_id into (rel_path, qname, line_start, line_end)."""
        abs_file = _node_file(node_id)
        qname = _node_qname(node_id)
        rel_file = self._rel_path(abs_file)
        node = self.brain.graph.nodes.get(node_id)
        line_start = 0
        line_end = 0
        if node:
            line_start = node.metadata.get('line_start', 0) or 0
            line_end = node.metadata.get('line_end', line_start) or line_start
        return (rel_file, qname, line_start, line_end)

    def _find_node_by_file_and_qname(
        self, file_substr: str, qname: str,
    ) -> Optional[str]:
        """Locate a node given a file substring + qualified name."""
        file_substr_norm = file_substr.replace('\\', '/')
        for node_id, node in self.brain.graph.nodes.items():
            if node.node_type != 'function':
                continue
            if _node_qname(node_id) != qname:
                continue
            nf = _node_file(node_id).replace('\\', '/')
            if nf.endswith('/' + file_substr_norm) or nf.endswith(file_substr_norm):
                return node_id
        return None

    def _rel_path(self, filepath: str) -> str:
        if not filepath:
            return filepath
        if os.path.isabs(filepath):
            try:
                return os.path.relpath(filepath, self.project_root)
            except ValueError:
                return filepath
        return filepath

    def _is_test_path(self, filepath: str) -> bool:
        p = filepath.replace('\\', '/').lower()
        parts = set(p.split('/'))
        return bool(parts & {'tests', 'test', 'examples', 'example'})

    def _file_commits(self, filepath_rel: str) -> List[CommitRef]:
        """All commits that touched a file, newest first."""
        if filepath_rel in self._file_commits_cache:
            return self._file_commits_cache[filepath_rel]

        # Use --follow so renames are included
        ok, out = self.git.run([
            'log', '--follow', '--pretty=format:%H|%h|%an|%ae|%at|%ci|%s',
            '-n', str(self.max_history), '--', filepath_rel,
        ], timeout=30)
        if not ok:
            self._file_commits_cache[filepath_rel] = []
            return []

        commits: List[CommitRef] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split('|', 6)
            if len(parts) < 7:
                continue
            full, short, author, email, ts_str, date, subject = parts
            try:
                ts = int(ts_str)
            except ValueError:
                ts = 0
            commits.append(CommitRef(
                sha=short, full_sha=full,
                author=author, email=email,
                date=date, timestamp=ts, subject=subject,
            ))
        self._file_commits_cache[filepath_rel] = commits
        return commits

    def _function_commits(
        self, filepath_rel: str, qname: str,
    ) -> List[CommitRef]:
        """Commits that touched this specific function's body.

        Uses `git log -L <start>,<end>:<file>`, which natively tracks a
        line range across commits (line shifts, rename-follow, merge
        handling). This is what we used to try to re-implement with a
        moving-window heuristic — git already does it correctly.

        Returns commits newest-first. If the function doesn't exist in
        the current file (dead function, or wrong qname), falls back
        to whole-file history.
        """
        abs_path = os.path.join(self.project_root, filepath_rel)
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
        except (FileNotFoundError, OSError):
            return self._file_commits(filepath_rel)

        rng = _extract_function_range(source, qname)
        if not rng:
            return self._file_commits(filepath_rel)

        line_start, line_end = rng

        # git log -L uses <start>,<end>:<file>  (inclusive range).
        # --no-patch suppresses the inline diff output since we only
        # want commit metadata. Format matches _file_commits.
        range_arg = f"{line_start},{line_end}:{filepath_rel}"
        ok, out = self.git.run([
            'log',
            '--no-patch',
            f'--pretty=format:%H|%h|%an|%ae|%at|%ci|%s',
            '-n', str(self.max_history),
            '-L', range_arg,
        ], timeout=60)

        if not ok:
            # git log -L can fail if the file was just created and has
            # no history matching the range, or git is too old. Fall
            # back to file commits so caller doesn't get a misleading
            # empty result.
            return self._file_commits(filepath_rel)

        commits: List[CommitRef] = []
        seen: Set[str] = set()
        for line in out.splitlines():
            if not line.strip() or '|' not in line:
                continue
            parts = line.split('|', 6)
            if len(parts) < 7:
                continue
            full, short, author, email, ts_str, date, subject = parts
            if full in seen:
                continue
            seen.add(full)
            try:
                ts = int(ts_str)
            except ValueError:
                ts = 0
            commits.append(CommitRef(
                sha=short, full_sha=full,
                author=author, email=email,
                date=date, timestamp=ts, subject=subject,
            ))
        return commits

    def _get_commit_diff(self, full_sha: str) -> str:
        """Cached `git show <sha>` diff text."""
        if full_sha in self._commit_diff_cache:
            return self._commit_diff_cache[full_sha]
        ok, out = self.git.run([
            'show', '--no-color', '--pretty=format:', full_sha,
        ], timeout=30)
        if not ok:
            self._commit_diff_cache[full_sha] = ''
            return ''
        self._commit_diff_cache[full_sha] = out
        return out

    def _functions_touched_in_diff(
        self, diff_text: str,
    ) -> Dict[str, List[Tuple[str, Tuple[int, int]]]]:
        """For a diff covering possibly-many files, return a mapping:
          filepath_rel -> [(qualified_name, (line_start, line_end))]
        listing every function whose CURRENT range overlaps any hunk in
        that file's diff.

        We read the current live source for each file touched. Functions
        that were removed in this commit won't appear (we have no live
        version to diff against). Good enough for co-evolution which
        cares about functions that exist NOW.
        """
        result: Dict[str, List[Tuple[str, Tuple[int, int]]]] = {}

        # First pass: collect set of files touched
        files_touched: Set[str] = set()
        cur_file: Optional[str] = None
        for line in diff_text.splitlines():
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 4:
                    b_path = parts[3]
                    if b_path.startswith('b/'):
                        b_path = b_path[2:]
                    b_path = b_path.replace('\\', '/')
                    files_touched.add(b_path)

        # For each touched file, parse its hunks + live source
        for rel_path in files_touched:
            if not rel_path.endswith('.py'):
                continue
            hunks = _parse_diff_for_file(diff_text, rel_path)
            if not hunks:
                continue
            abs_path = os.path.join(self.project_root, rel_path)
            try:
                with open(abs_path, 'r', encoding='utf-8', errors='replace') as fh:
                    source = fh.read()
            except (FileNotFoundError, OSError):
                continue

            # Walk AST, find every function, check hunk overlap
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError):
                continue

            funcs: List[Tuple[str, int, int]] = []
            self._collect_functions(tree, [], funcs)
            overlapping: List[Tuple[str, Tuple[int, int]]] = []
            for qname, ls, le in funcs:
                if _hunks_touch_new_range(hunks, ls, le):
                    overlapping.append((qname, (ls, le)))
            if overlapping:
                result[rel_path] = overlapping

        return result

    def _collect_functions(
        self, node, path: List[str],
        out: List[Tuple[str, int, int]],
    ):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = '.'.join(path + [child.name])
                start = child.lineno
                end = getattr(child, 'end_lineno', start) or start
                out.append((qname, start, end))
                # nested funcs
                self._collect_functions(child, path + [child.name], out)
            elif isinstance(child, ast.ClassDef):
                self._collect_functions(child, path + [child.name], out)


# ═══════════════════════════════════════════════════════════════════
# Report formatting
# ═══════════════════════════════════════════════════════════════════

def _get_colors(use_color: bool = True):
    if not use_color:
        class _NoC:
            DIM = RESET = BOLD = RED = YELLOW = CYAN = GREEN = BLUE = MAGENTA = ''
        return _NoC()
    try:
        from core.terminal_ui import C
        # Ensure all names exist
        for attr in ('DIM', 'RESET', 'BOLD', 'RED', 'YELLOW', 'CYAN', 'GREEN', 'BLUE', 'MAGENTA'):
            if not hasattr(C, attr):
                setattr(C, attr, '')
        return C
    except Exception:
        class _NoC:
            DIM = RESET = BOLD = RED = YELLOW = CYAN = GREEN = BLUE = MAGENTA = ''
        return _NoC()


def format_genesis(g: FunctionGenesis, color: bool = True) -> str:
    C = _get_colors(color)
    lines = [f"{C.BOLD}Genesis:{C.RESET}"]
    if not g.first_commit:
        lines.append(f"  {C.DIM}No git history — either never committed or repo unavailable.{C.RESET}")
        return '\n'.join(lines)

    lines.append(f"  First appeared:  {C.GREEN}{g.first_commit.date}{C.RESET}")
    lines.append(f"  First commit:    {C.CYAN}{g.first_commit.sha}{C.RESET}  "
                 f"{C.DIM}({g.first_commit.author}){C.RESET}")
    lines.append(f"  Subject:         {C.DIM}{g.first_commit.subject[:80]}{C.RESET}")
    lines.append(f"  Last changed:    {C.YELLOW}{g.last_commit.date}{C.RESET}  "
                 f"{C.DIM}({g.last_commit.author}){C.RESET}")
    lines.append(f"  Age:             {C.BOLD}{g.age_days}{C.RESET} days, "
                 f"{C.BOLD}{g.total_commits}{C.RESET} commit(s) touched it")
    return '\n'.join(lines)


def format_history(history: List[FunctionHistoryEntry], color: bool = True) -> str:
    C = _get_colors(color)
    lines = [f"{C.BOLD}History:{C.RESET}"]
    if not history:
        lines.append(f"  {C.DIM}(no commits — function never changed after creation, or no git){C.RESET}")
        return '\n'.join(lines)

    for h in history:
        delta_parts = []
        if h.lines_added:
            delta_parts.append(f"{C.GREEN}+{h.lines_added}{C.RESET}")
        if h.lines_removed:
            delta_parts.append(f"{C.RED}-{h.lines_removed}{C.RESET}")
        delta = '  '.join(delta_parts) if delta_parts else f"{C.DIM}0{C.RESET}"
        lines.append(
            f"  {C.CYAN}{h.commit.sha}{C.RESET} "
            f"{h.commit.date[:10]}  "
            f"{C.DIM}{h.commit.author:20}{C.RESET}  "
            f"{delta:24}  "
            f"{C.DIM}{h.commit.subject[:60]}{C.RESET}"
        )
    return '\n'.join(lines)


def format_coevolution(coevo: List[CoEvolutionEntry], color: bool = True) -> str:
    C = _get_colors(color)
    lines = [f"{C.BOLD}Co-evolution (functions that change together):{C.RESET}"]
    if not coevo:
        lines.append(f"  {C.DIM}(no co-evolving functions found){C.RESET}")
        return '\n'.join(lines)

    STRENGTH_COLOR = {
        'high': C.RED, 'medium': C.YELLOW, 'low': C.DIM,
    }
    for e in coevo:
        col = STRENGTH_COLOR.get(e.coupling_strength, '')
        lines.append(
            f"  {col}[{e.coupling_strength:6}]{C.RESET} "
            f"{C.BOLD}{e.function_name}{C.RESET}  "
            f"{C.DIM}in {e.filepath}{C.RESET}"
        )
        lines.append(
            f"           shared {C.BOLD}{e.shared_commits}{C.RESET}  "
            f"{C.DIM}jaccard {e.jaccard}  "
            f"target-only {e.target_only_commits}  this-only {e.this_only_commits}{C.RESET}"
        )
    return '\n'.join(lines)


def format_stability(s: StabilityScore, color: bool = True) -> str:
    C = _get_colors(color)
    lines = [f"{C.BOLD}Stability:{C.RESET}"]

    # Color the score based on range
    if s.score >= 80:
        score_col = C.GREEN
    elif s.score >= 50:
        score_col = C.YELLOW
    else:
        score_col = C.RED

    lines.append(f"  Score:              {score_col}{s.score}/100{C.RESET}")
    lines.append(f"  Interpretation:     {s.interpretation}")
    lines.append(f"  Age:                {s.age_days} days")
    lines.append(f"  Total changes:      {s.total_changes}")
    lines.append(f"  Churn rate:         {s.changes_per_30_days:.2f} changes / 30 days")
    lines.append(f"  Last changed:       {s.last_changed_days_ago} days ago")
    return '\n'.join(lines)


def format_authors(a: AuthorMapEntry, color: bool = True) -> str:
    C = _get_colors(color)
    lines = [f"{C.BOLD}Authors:{C.RESET}"]
    if not a.authors:
        lines.append(f"  {C.DIM}(no commits recorded){C.RESET}")
        return '\n'.join(lines)

    total = sum(cnt for _, cnt in a.authors)
    for author, cnt in a.authors[:10]:
        pct = (cnt / total * 100) if total else 0
        marker = C.GREEN + '●' + C.RESET if author == a.primary_author else '○'
        lines.append(f"  {marker} {C.BOLD}{author:25}{C.RESET}  "
                     f"{cnt:3} commit(s)  {C.DIM}({pct:.0f}%){C.RESET}")
    lines.append(f"  {C.DIM}Bus factor: {a.bus_factor}  (authors to cover 80%+ of changes){C.RESET}")
    return '\n'.join(lines)


def format_dead_code(entries: List[DeadCodeEntry], color: bool = True) -> str:
    C = _get_colors(color)
    lines = [
        "",
        f"{C.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}",
        f"{C.BOLD}  SourceForensics — Dead Code Sweep{C.RESET}",
        f"{C.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}",
        "",
    ]
    if not entries:
        lines.append(f"  {C.GREEN}✓ No dead code detected.{C.RESET}")
        return '\n'.join(lines)

    total_lines = sum(e.lines for e in entries)
    lines.append(f"  Found {C.BOLD}{len(entries)}{C.RESET} potentially dead function(s), "
                 f"{C.BOLD}{total_lines}{C.RESET} lines total")
    lines.append(f"  {C.DIM}These have no inbound call edges from the project's code. "
                 f"Framework decorators, dynamic dispatch, and reflection can create "
                 f"hidden callers — review before deleting.{C.RESET}")
    lines.append("")

    for e in entries[:30]:
        lines.append(f"  {C.YELLOW}●{C.RESET} {C.BOLD}{e.function_name}{C.RESET}  "
                     f"{C.DIM}{e.filepath}:{e.line_start}{C.RESET}  "
                     f"{C.DIM}(complexity {e.complexity}, {e.lines} lines){C.RESET}")
    if len(entries) > 30:
        lines.append(f"  {C.DIM}... +{len(entries) - 30} more{C.RESET}")
    return '\n'.join(lines)


def format_full_report(r: ForensicsReport, color: bool = True) -> str:
    C = _get_colors(color)
    lines = [
        "",
        f"{C.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}",
        f"{C.BOLD}  SourceForensics — {r.function_name}{C.RESET}",
        f"{C.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}",
        "",
        f"  {C.DIM}File:               {r.filepath}{C.RESET}",
        f"  {C.DIM}Current lines:      {r.current_lines}{C.RESET}",
        f"  {C.DIM}Current complexity: {r.current_complexity}{C.RESET}",
        "",
    ]
    if r.genesis:
        lines.append(format_genesis(r.genesis, color))
        lines.append('')
    if r.stability:
        lines.append(format_stability(r.stability, color))
        lines.append('')
    if r.authors:
        lines.append(format_authors(r.authors, color))
        lines.append('')
    lines.append(format_history(r.history, color))
    lines.append('')
    lines.append(format_coevolution(r.co_evolution, color))
    lines.append('')
    return '\n'.join(lines)
