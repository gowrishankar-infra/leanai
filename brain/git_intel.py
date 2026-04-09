"""
LeanAI Phase 7b — Git-Aware Intelligence
Reads your git history and provides deep understanding of code evolution.

Capabilities:
  - "What did I work on this week?" — summarizes recent commits
  - "When was this function last changed?" — finds the commit
  - "Why did we change database.py?" — finds commit messages explaining changes
  - Generate commit messages from staged diffs
  - Generate PR/changelog descriptions from commit ranges
  - Track file change frequency (hotspots)
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


@dataclass
class GitCommit:
    """A single git commit."""
    hash: str
    short_hash: str
    author: str
    date: str
    timestamp: float
    message: str
    files_changed: List[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0

    def to_dict(self) -> dict:
        return {
            "hash": self.hash, "short_hash": self.short_hash,
            "author": self.author, "date": self.date,
            "message": self.message, "files_changed": self.files_changed,
            "insertions": self.insertions, "deletions": self.deletions,
        }

    @property
    def summary(self) -> str:
        files = f" ({len(self.files_changed)} files)" if self.files_changed else ""
        return f"[{self.short_hash}] {self.message}{files}"


@dataclass
class FileDiff:
    """Diff information for a single file."""
    filepath: str
    status: str  # "modified", "added", "deleted", "renamed"
    insertions: int = 0
    deletions: int = 0
    diff_text: str = ""


@dataclass
class GitStatus:
    """Current working tree status."""
    staged: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    untracked: List[str] = field(default_factory=list)
    branch: str = ""
    ahead: int = 0
    behind: int = 0


class GitIntel:
    """
    Git-aware intelligence layer.
    
    Usage:
        git = GitIntel("/path/to/repo")
        
        # Query history
        print(git.recent_activity(days=7))
        print(git.file_history("src/api.py", limit=5))
        print(git.function_last_changed("handle_request"))
        print(git.why_changed("database.py"))
        
        # Generate
        print(git.generate_commit_message())
        print(git.generate_changelog(since="v1.0"))
        
        # Analysis
        print(git.hotspots(top_n=10))
        print(git.contributor_stats())
    """

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        if not os.path.isdir(os.path.join(self.repo_path, ".git")):
            # Not a fatal error — just won't have git features
            self._git_available = False
        else:
            self._git_available = True

    def _run_git(self, args: List[str], timeout: int = 30) -> Tuple[bool, str]:
        """Run a git command and return (success, output)."""
        if not self._git_available:
            return False, "Not a git repository"
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                return False, result.stderr.strip()
            return True, output
        except FileNotFoundError:
            return False, "git not found"
        except subprocess.TimeoutExpired:
            return False, "git command timed out"
        except Exception as e:
            return False, str(e)

    @property
    def is_available(self) -> bool:
        return self._git_available

    # ── Status ────────────────────────────────────────────────────

    def status(self) -> GitStatus:
        """Get current working tree status."""
        result = GitStatus()

        ok, branch = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if ok:
            result.branch = branch

        ok, output = self._run_git(["status", "--porcelain"])
        if ok and output:
            for line in output.split("\n"):
                if not line.strip():
                    continue
                status_code = line[:2]
                filepath = line[3:].strip()
                if status_code[0] in "MADRC":
                    result.staged.append(filepath)
                if status_code[1] == "M":
                    result.modified.append(filepath)
                if status_code == "??":
                    result.untracked.append(filepath)

        return result

    # ── History ───────────────────────────────────────────────────

    def get_commits(self, limit: int = 20, since: Optional[str] = None,
                    path: Optional[str] = None) -> List[GitCommit]:
        """Get recent commits with details."""
        args = [
            "log", f"--max-count={limit}",
            "--format=%H|%h|%an|%aI|%at|%s",
            "--no-merges",
        ]
        if since:
            args.append(f"--since={since}")
        if path:
            args.extend(["--", path])

        ok, output = self._run_git(args)
        if not ok or not output:
            return []

        commits = []
        for line in output.split("\n"):
            parts = line.split("|", 5)
            if len(parts) < 6:
                continue
            commit = GitCommit(
                hash=parts[0],
                short_hash=parts[1],
                author=parts[2],
                date=parts[3],
                timestamp=float(parts[4]),
                message=parts[5],
            )

            # Get files changed in this commit
            ok2, files_output = self._run_git([
                "diff-tree", "--no-commit-id", "--name-only", "-r", commit.hash
            ])
            if ok2 and files_output:
                commit.files_changed = [f for f in files_output.split("\n") if f.strip()]

            # Get stats
            ok3, stats = self._run_git([
                "diff-tree", "--no-commit-id", "--shortstat", "-r", commit.hash
            ])
            if ok3 and stats:
                ins_match = re.search(r"(\d+) insertion", stats)
                del_match = re.search(r"(\d+) deletion", stats)
                if ins_match:
                    commit.insertions = int(ins_match.group(1))
                if del_match:
                    commit.deletions = int(del_match.group(1))

            commits.append(commit)

        return commits

    def recent_activity(self, days: int = 7) -> str:
        """Summarize recent git activity."""
        since = f"{days} days ago"
        commits = self.get_commits(limit=50, since=since)
        if not commits:
            return f"No commits in the last {days} days."

        lines = [f"Activity in the last {days} days: {len(commits)} commits"]

        # Group by date
        by_date: Dict[str, List[GitCommit]] = {}
        for c in commits:
            date = c.date[:10]  # YYYY-MM-DD
            by_date.setdefault(date, []).append(c)

        for date in sorted(by_date.keys(), reverse=True):
            day_commits = by_date[date]
            lines.append(f"\n  {date} ({len(day_commits)} commits):")
            for c in day_commits:
                lines.append(f"    {c.summary}")

        # Stats
        total_files = set()
        total_ins = sum(c.insertions for c in commits)
        total_del = sum(c.deletions for c in commits)
        for c in commits:
            total_files.update(c.files_changed)

        lines.append(f"\nTotal: {len(total_files)} files touched, +{total_ins}/-{total_del} lines")
        return "\n".join(lines)

    def file_history(self, filepath: str, limit: int = 10) -> str:
        """Get the commit history for a specific file."""
        commits = self.get_commits(limit=limit, path=filepath)
        if not commits:
            return f"No history found for {filepath}"
        lines = [f"History for {filepath} ({len(commits)} commits):"]
        for c in commits:
            lines.append(f"  {c.date[:10]} [{c.short_hash}] {c.message}")
        return "\n".join(lines)

    def why_changed(self, filepath: str, limit: int = 5) -> str:
        """Find commit messages that explain why a file was changed."""
        commits = self.get_commits(limit=limit, path=filepath)
        if not commits:
            return f"No changes found for {filepath}"
        lines = [f"Why {filepath} was changed:"]
        for c in commits:
            lines.append(f"  [{c.date[:10]}] {c.message}")
        return "\n".join(lines)

    def function_last_changed(self, function_name: str) -> str:
        """Find when a function was last modified using git log -S."""
        ok, output = self._run_git([
            "log", "--max-count=5",
            "--format=%h|%aI|%s",
            f"-S{function_name}", "--all",
        ])
        if not ok or not output:
            return f"No changes found for function '{function_name}'"
        lines = [f"Changes involving '{function_name}':"]
        for line in output.split("\n"):
            parts = line.split("|", 2)
            if len(parts) >= 3:
                lines.append(f"  [{parts[1][:10]}] {parts[0]} — {parts[2]}")
        return "\n".join(lines)

    # ── Generation ────────────────────────────────────────────────

    def get_staged_diff(self) -> str:
        """Get the diff of staged changes."""
        ok, output = self._run_git(["diff", "--cached", "--stat"])
        if not ok or not output:
            return ""
        return output

    def get_staged_diff_full(self) -> str:
        """Get the full diff of staged changes."""
        ok, output = self._run_git(["diff", "--cached"])
        if not ok:
            return ""
        # Truncate very long diffs
        if len(output) > 5000:
            output = output[:5000] + "\n... (truncated)"
        return output

    def get_unstaged_diff(self) -> str:
        """Get diff of unstaged changes."""
        ok, output = self._run_git(["diff", "--stat"])
        if not ok:
            return ""
        return output

    def generate_commit_message_context(self) -> str:
        """
        Generate context for an AI to write a commit message.
        Returns the staged diff summary + recent commits for style matching.
        """
        staged = self.get_staged_diff()
        if not staged:
            return "No staged changes."

        # Get recent commit messages for style reference
        recent = self.get_commits(limit=5)
        style_examples = [c.message for c in recent[:3]]

        parts = [
            "Staged changes:",
            staged,
            "",
            "Full diff (key changes):",
            self.get_staged_diff_full()[:3000],
        ]
        if style_examples:
            parts.extend(["", "Recent commit message style:", *style_examples])

        return "\n".join(parts)

    def generate_changelog(self, since: Optional[str] = None, limit: int = 20) -> str:
        """Generate a changelog from commit history."""
        commits = self.get_commits(limit=limit, since=since)
        if not commits:
            return "No commits found."

        lines = ["# Changelog", ""]

        # Categorize commits by prefix
        features = []
        fixes = []
        others = []
        for c in commits:
            msg = c.message.lower()
            if any(kw in msg for kw in ["feat", "add", "new", "implement"]):
                features.append(c)
            elif any(kw in msg for kw in ["fix", "bug", "patch", "repair"]):
                fixes.append(c)
            else:
                others.append(c)

        if features:
            lines.append("## Features")
            for c in features:
                lines.append(f"- {c.message} ({c.short_hash})")
            lines.append("")

        if fixes:
            lines.append("## Bug Fixes")
            for c in fixes:
                lines.append(f"- {c.message} ({c.short_hash})")
            lines.append("")

        if others:
            lines.append("## Other Changes")
            for c in others:
                lines.append(f"- {c.message} ({c.short_hash})")

        return "\n".join(lines)

    # ── Analysis ──────────────────────────────────────────────────

    def hotspots(self, top_n: int = 10, since: Optional[str] = None) -> str:
        """Find files that change most frequently (change hotspots)."""
        commits = self.get_commits(limit=100, since=since)
        file_counts: Dict[str, int] = {}
        for c in commits:
            for f in c.files_changed:
                file_counts[f] = file_counts.get(f, 0) + 1

        sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
        lines = [f"Change hotspots (top {top_n}):"]
        for filepath, count in sorted_files[:top_n]:
            bar = "█" * min(count, 20)
            lines.append(f"  {count:3d} {bar} {filepath}")
        return "\n".join(lines)

    def contributor_stats(self) -> str:
        """Get contributor statistics."""
        ok, output = self._run_git(["shortlog", "-sn", "--all", "--no-merges"])
        if not ok or not output:
            return "No contributor data."
        lines = ["Contributors:"]
        for line in output.split("\n")[:10]:
            line = line.strip()
            if line:
                lines.append(f"  {line}")
        return "\n".join(lines)

    def current_branch(self) -> str:
        """Get current branch name."""
        ok, output = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        return output if ok else "unknown"

    def branches(self) -> List[str]:
        """List all local branches."""
        ok, output = self._run_git(["branch", "--format=%(refname:short)"])
        if not ok or not output:
            return []
        return [b.strip() for b in output.split("\n") if b.strip()]

    # ── Context for AI ────────────────────────────────────────────

    def get_context_for_query(self, query: str) -> str:
        """Build git context for an AI query about the project history."""
        query_lower = query.lower()
        parts = []

        parts.append(f"Branch: {self.current_branch()}")

        if any(kw in query_lower for kw in ["recent", "today", "week", "yesterday", "last"]):
            days = 1 if "today" in query_lower or "yesterday" in query_lower else 7
            parts.append(self.recent_activity(days=days))

        if any(kw in query_lower for kw in ["why", "changed", "modify", "history"]):
            # Try to extract a filename from the query
            for word in query.split():
                if "." in word and not word.startswith("."):
                    parts.append(self.file_history(word, limit=5))
                    break

        if any(kw in query_lower for kw in ["hotspot", "frequent", "most changed"]):
            parts.append(self.hotspots())

        if any(kw in query_lower for kw in ["commit", "staged", "changes"]):
            parts.append(self.generate_commit_message_context())

        if not parts[1:]:  # if no specific context matched, give recent activity
            parts.append(self.recent_activity(days=7))

        return "\n".join(parts)

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Get git repository statistics."""
        ok, count = self._run_git(["rev-list", "--count", "HEAD"])
        total_commits = int(count) if ok and count.isdigit() else 0

        ok, branch = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        current_branch = branch if ok else "unknown"

        branches = self.branches()

        return {
            "available": self._git_available,
            "branch": current_branch,
            "total_commits": total_commits,
            "branches": len(branches),
        }
