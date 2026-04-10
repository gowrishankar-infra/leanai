"""
LeanAI — Semantic Git Bisect
AI-powered bug finding that understands code behavior, not just line diffs.

Traditional git bisect: binary search, run test, narrow down.
Semantic git bisect: read each commit's changes, understand what they DO,
predict which one likely introduced the bug.

Example:
  /bisect "authentication stopped working after last week's changes"
  
  LeanAI reads the last 20 commits, understands each one semantically:
    commit a1b2c3: "Added rate limiting to API" → modifies middleware
    commit d4e5f6: "Refactored auth token validation" → HIGH SUSPICION
    commit g7h8i9: "Updated README" → no code changes
  
  Result: "Most likely culprit: d4e5f6 — refactored auth token validation.
           This commit modified the token verification logic which directly
           affects authentication flow."

Nobody has this. Git bisect is manual binary search. This is AI reasoning.
"""

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable


@dataclass
class CommitAnalysis:
    """Semantic analysis of a single commit."""
    hash: str
    short_hash: str
    message: str
    author: str
    date: str
    files_changed: List[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0
    diff_summary: str = ""
    semantic_summary: str = ""
    suspicion_score: float = 0.0  # 0-1, how likely this commit caused the bug
    reasoning: str = ""


@dataclass
class BisectResult:
    """Result of a semantic bisect operation."""
    bug_description: str
    commits_analyzed: int
    most_likely: Optional[CommitAnalysis] = None
    top_suspects: List[CommitAnalysis] = field(default_factory=list)
    analysis_time_ms: float = 0.0
    reasoning: str = ""

    def summary(self) -> str:
        if not self.most_likely:
            return "No likely culprit found."
        lines = [
            f"Bug: {self.bug_description}",
            f"Analyzed: {self.commits_analyzed} commits",
            f"",
            f"Most likely culprit:",
            f"  {self.most_likely.short_hash} — {self.most_likely.message}",
            f"  Author: {self.most_likely.author} | Date: {self.most_likely.date}",
            f"  Files: {', '.join(self.most_likely.files_changed[:5])}",
            f"  Suspicion: {self.most_likely.suspicion_score:.0%}",
            f"  Reasoning: {self.most_likely.reasoning}",
        ]
        if len(self.top_suspects) > 1:
            lines.append(f"\nOther suspects:")
            for s in self.top_suspects[1:3]:
                lines.append(f"  {s.short_hash} — {s.message} ({s.suspicion_score:.0%})")
        lines.append(f"\nAnalysis time: {self.analysis_time_ms:.0f}ms")
        return "\n".join(lines)


class SemanticGitBisect:
    """
    AI-powered semantic git bisect.
    
    Usage:
        bisect = SemanticGitBisect(repo_path=".", model_fn=my_model)
        
        result = bisect.find_bug(
            "authentication stopped working",
            num_commits=20,
        )
        print(result.summary())
    """

    def __init__(self, repo_path: str = ".", model_fn: Optional[Callable] = None):
        self.repo_path = repo_path
        self.model_fn = model_fn

    def _git(self, *args) -> str:
        """Run a git command and return output."""
        try:
            result = subprocess.run(
                ["git"] + list(args),
                capture_output=True, text=True, cwd=self.repo_path,
                timeout=30,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _get_recent_commits(self, num_commits: int = 20) -> List[CommitAnalysis]:
        """Get recent commits with metadata."""
        log = self._git(
            "log", f"-{num_commits}",
            "--format=%H|%h|%s|%an|%ad",
            "--date=short",
        )
        if not log:
            return []

        commits = []
        for line in log.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|", 4)
            if len(parts) < 5:
                continue

            commit = CommitAnalysis(
                hash=parts[0],
                short_hash=parts[1],
                message=parts[2],
                author=parts[3],
                date=parts[4],
            )

            # Get changed files and stats
            stat = self._git("diff", "--stat", "--name-only", f"{commit.hash}~1..{commit.hash}")
            if stat:
                commit.files_changed = [f for f in stat.split("\n") if f.strip() and not f.startswith(" ")]

            numstat = self._git("diff", "--numstat", f"{commit.hash}~1..{commit.hash}")
            if numstat:
                for nline in numstat.split("\n"):
                    nparts = nline.split("\t")
                    if len(nparts) >= 2:
                        try:
                            commit.insertions += int(nparts[0]) if nparts[0] != "-" else 0
                            commit.deletions += int(nparts[1]) if nparts[1] != "-" else 0
                        except ValueError:
                            pass

            # Get a brief diff summary (first 500 chars)
            diff = self._git("diff", f"{commit.hash}~1..{commit.hash}", "--stat")
            commit.diff_summary = diff[:500] if diff else ""

            commits.append(commit)

        return commits

    def find_bug(self, bug_description: str, num_commits: int = 20,
                 verbose: bool = False) -> BisectResult:
        """
        Find which commit most likely introduced a bug.
        
        Args:
            bug_description: description of the bug
            num_commits: how many recent commits to analyze
            verbose: print progress
        
        Returns:
            BisectResult with the most likely culprit
        """
        start = time.time()

        if verbose:
            print(f"[Bisect] Analyzing last {num_commits} commits...", flush=True)

        commits = self._get_recent_commits(num_commits)
        if not commits:
            return BisectResult(
                bug_description=bug_description,
                commits_analyzed=0,
                reasoning="No commits found in this repository.",
            )

        if verbose:
            print(f"[Bisect] Found {len(commits)} commits. Scoring suspicion...", flush=True)

        # Score each commit's suspicion level
        for commit in commits:
            commit.suspicion_score = self._score_suspicion(commit, bug_description)
            commit.reasoning = self._explain_suspicion(commit, bug_description)

        # If model is available, use AI for deeper analysis on top suspects
        if self.model_fn:
            # Sort by suspicion and analyze top 5 with AI
            commits.sort(key=lambda c: c.suspicion_score, reverse=True)
            top = commits[:5]

            if verbose:
                print(f"[Bisect] AI analyzing top {len(top)} suspects...", flush=True)

            self._ai_analyze(top, bug_description)

        # Sort by final suspicion score
        commits.sort(key=lambda c: c.suspicion_score, reverse=True)

        elapsed = (time.time() - start) * 1000

        return BisectResult(
            bug_description=bug_description,
            commits_analyzed=len(commits),
            most_likely=commits[0] if commits else None,
            top_suspects=commits[:5],
            analysis_time_ms=elapsed,
            reasoning=commits[0].reasoning if commits else "No commits analyzed.",
        )

    def _score_suspicion(self, commit: CommitAnalysis, bug_description: str) -> float:
        """
        Score how suspicious a commit is based on heuristics.
        Returns 0-1 suspicion score.
        """
        score = 0.0
        bug_lower = bug_description.lower()
        msg_lower = commit.message.lower()

        # 1. Keyword overlap between bug description and commit message
        bug_words = set(bug_lower.split())
        msg_words = set(msg_lower.split())
        overlap = len(bug_words & msg_words)
        if overlap > 0:
            score += min(overlap * 0.15, 0.45)

        # 2. File relevance — does the commit touch files related to the bug?
        bug_file_hints = self._extract_file_hints(bug_description)
        for f in commit.files_changed:
            f_lower = f.lower()
            for hint in bug_file_hints:
                if hint in f_lower:
                    score += 0.2
                    break

        # 3. Size of change — larger changes are more suspicious
        total_changes = commit.insertions + commit.deletions
        if total_changes > 100:
            score += 0.15
        elif total_changes > 50:
            score += 0.10
        elif total_changes > 20:
            score += 0.05

        # 4. Risky commit message keywords
        risky_words = {"refactor", "rewrite", "overhaul", "redesign", "migration",
                       "replace", "remove", "delete", "breaking", "major",
                       "merge", "conflict", "hotfix", "hack", "workaround", "temp"}
        if any(w in msg_lower for w in risky_words):
            score += 0.15

        # 5. Touches config/env/infrastructure files
        config_files = {"config", "env", ".yml", ".yaml", ".toml", "settings",
                        "dockerfile", "requirements", "package.json"}
        if any(any(cf in f.lower() for cf in config_files) for f in commit.files_changed):
            score += 0.05

        # 6. Skip obvious non-suspects
        skip_words = {"readme", "docs", "comment", "typo", "formatting", "lint", "style"}
        if any(w in msg_lower for w in skip_words) and total_changes < 20:
            score *= 0.3

        return min(score, 1.0)

    def _extract_file_hints(self, bug_description: str) -> List[str]:
        """Extract likely file/module names from bug description."""
        hints = []
        words = bug_description.lower().split()
        # Look for words that look like filenames or modules
        code_words = {"auth", "login", "database", "api", "route", "model",
                      "config", "server", "client", "test", "deploy", "build",
                      "cache", "session", "memory", "engine", "router",
                      "handler", "middleware", "controller", "service",
                      "user", "admin", "payment", "email", "notification"}
        for word in words:
            clean = word.strip(".,!?()[]{}\"'")
            if clean in code_words or "." in clean:
                hints.append(clean)
        return hints

    def _explain_suspicion(self, commit: CommitAnalysis, bug_description: str) -> str:
        """Generate a human-readable explanation of why this commit is suspicious."""
        reasons = []
        bug_lower = bug_description.lower()
        msg_lower = commit.message.lower()

        bug_words = set(bug_lower.split())
        msg_words = set(msg_lower.split())
        overlap = bug_words & msg_words - {"the", "a", "an", "is", "in", "to", "of", "and"}
        if overlap:
            reasons.append(f"commit message mentions: {', '.join(list(overlap)[:3])}")

        total = commit.insertions + commit.deletions
        if total > 50:
            reasons.append(f"large change ({commit.insertions}+ {commit.deletions}-)")

        risky = {"refactor", "rewrite", "overhaul", "replace", "remove", "merge"}
        found_risky = [w for w in risky if w in msg_lower]
        if found_risky:
            reasons.append(f"risky operation: {', '.join(found_risky)}")

        if not reasons:
            reasons.append("low suspicion — no direct indicators")

        return "; ".join(reasons)

    def _ai_analyze(self, commits: List[CommitAnalysis], bug_description: str):
        """Use the AI model for deeper analysis of top suspects."""
        if not self.model_fn:
            return

        commit_summaries = []
        for c in commits:
            files = ", ".join(c.files_changed[:5])
            commit_summaries.append(
                f"  {c.short_hash}: \"{c.message}\" "
                f"({c.insertions}+ {c.deletions}-) files: {files}"
            )

        prompt = (
            f"Bug report: \"{bug_description}\"\n\n"
            f"These commits are suspects:\n"
            + "\n".join(commit_summaries) + "\n\n"
            f"Which commit most likely caused this bug? "
            f"Rank them from most to least suspicious. "
            f"For each, explain WHY in one sentence."
        )

        try:
            system = "You are a senior developer doing a code review to find which commit introduced a bug."
            response = self.model_fn(system, prompt)

            # Try to extract rankings from the response
            for i, commit in enumerate(commits):
                # If the commit hash appears early in the response, boost its score
                if commit.short_hash in response[:200]:
                    commit.suspicion_score = min(commit.suspicion_score + 0.3, 1.0)
                elif commit.short_hash in response:
                    commit.suspicion_score = min(commit.suspicion_score + 0.1, 1.0)

                # Extract reasoning if mentioned
                if commit.short_hash in response:
                    idx = response.index(commit.short_hash)
                    snippet = response[idx:idx+200]
                    # Find the sentence containing the hash
                    for sent in snippet.split("."):
                        if commit.short_hash in sent or commit.message[:20] in sent:
                            commit.reasoning = sent.strip()[:150]
                            break
        except Exception:
            pass  # AI analysis is optional, heuristics still work
