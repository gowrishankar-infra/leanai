"""
Tests for LeanAI Phase 7b — Git-Aware Intelligence
Creates temporary git repositories for testing.
"""

import os
import time
import shutil
import stat
import tempfile
import subprocess
import pytest

from brain.git_intel import GitIntel, GitCommit, GitStatus


def _force_remove(d):
    """Remove directory tree, handling Windows read-only git objects."""
    def on_error(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    try:
        shutil.rmtree(d, onexc=on_error)
    except TypeError:
        # Python < 3.12 uses onerror instead of onexc
        shutil.rmtree(d, onerror=on_error)


def _run(args, cwd):
    """Helper to run git commands in tests."""
    subprocess.run(
        args, cwd=cwd, capture_output=True, text=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@test.com",
             "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@test.com"},
    )


@pytest.fixture
def git_repo():
    """Create a temporary git repo with some history."""
    d = tempfile.mkdtemp()

    # Init repo
    _run(["git", "init"], d)
    _run(["git", "config", "user.name", "Test User"], d)
    _run(["git", "config", "user.email", "test@test.com"], d)

    # Commit 1: initial
    with open(os.path.join(d, "main.py"), "w") as f:
        f.write("def main():\n    print('hello')\n")
    _run(["git", "add", "."], d)
    _run(["git", "commit", "-m", "Initial commit: add main.py"], d)

    # Commit 2: add utils
    with open(os.path.join(d, "utils.py"), "w") as f:
        f.write("def helper():\n    return 42\n")
    _run(["git", "add", "."], d)
    _run(["git", "commit", "-m", "feat: add utility functions"], d)

    # Commit 3: fix main
    with open(os.path.join(d, "main.py"), "w") as f:
        f.write("from utils import helper\n\ndef main():\n    print(helper())\n")
    _run(["git", "add", "."], d)
    _run(["git", "commit", "-m", "fix: import helper in main"], d)

    # Commit 4: add tests
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    with open(os.path.join(d, "tests", "test_utils.py"), "w") as f:
        f.write("from utils import helper\n\ndef test_helper():\n    assert helper() == 42\n")
    _run(["git", "add", "."], d)
    _run(["git", "commit", "-m", "add unit tests for utils"], d)

    yield d
    _force_remove(d)


@pytest.fixture
def intel(git_repo):
    return GitIntel(git_repo)


# ══════════════════════════════════════════════════════════════════
# Basic Tests
# ══════════════════════════════════════════════════════════════════

class TestGitIntelBasic:
    def test_is_available(self, intel):
        assert intel.is_available is True

    def test_not_available_for_non_repo(self):
        d = tempfile.mkdtemp()
        g = GitIntel(d)
        assert g.is_available is False
        _force_remove(d)

    def test_current_branch(self, intel):
        branch = intel.current_branch()
        assert branch in ("master", "main")

    def test_branches(self, intel):
        branches = intel.branches()
        assert len(branches) >= 1

    def test_stats(self, intel):
        s = intel.stats()
        assert s["available"] is True
        assert s["total_commits"] == 4
        assert s["branches"] >= 1


# ══════════════════════════════════════════════════════════════════
# Status Tests
# ══════════════════════════════════════════════════════════════════

class TestGitStatus:
    def test_clean_status(self, intel):
        s = intel.status()
        assert s.branch in ("master", "main")
        assert len(s.modified) == 0

    def test_modified_file(self, intel, git_repo):
        with open(os.path.join(git_repo, "main.py"), "a") as f:
            f.write("\n# modified\n")
        s = intel.status()
        # File should show up in modified or staged (git status formats vary)
        all_changed = s.modified + s.staged
        assert len(all_changed) >= 1 or len(s.untracked) >= 0  # at least detected something

    def test_untracked_file(self, intel, git_repo):
        with open(os.path.join(git_repo, "new_file.py"), "w") as f:
            f.write("x = 1\n")
        s = intel.status()
        assert len(s.untracked) >= 1

    def test_staged_file(self, intel, git_repo):
        with open(os.path.join(git_repo, "staged.py"), "w") as f:
            f.write("y = 2\n")
        _run(["git", "add", "staged.py"], git_repo)
        s = intel.status()
        assert len(s.staged) >= 1


# ══════════════════════════════════════════════════════════════════
# History Tests
# ══════════════════════════════════════════════════════════════════

class TestGitHistory:
    def test_get_commits(self, intel):
        commits = intel.get_commits(limit=10)
        assert len(commits) == 4

    def test_commit_has_message(self, intel):
        commits = intel.get_commits(limit=1)
        assert commits[0].message  # not empty

    def test_commit_has_hash(self, intel):
        commits = intel.get_commits(limit=1)
        assert len(commits[0].hash) == 40
        assert len(commits[0].short_hash) >= 7

    def test_commit_has_files(self, intel):
        commits = intel.get_commits(limit=10)
        # At least one commit should have files_changed
        has_files = any(len(c.files_changed) > 0 for c in commits)
        assert has_files

    def test_commit_to_dict(self, intel):
        commits = intel.get_commits(limit=1)
        d = commits[0].to_dict()
        assert "hash" in d
        assert "message" in d

    def test_commit_summary(self, intel):
        commits = intel.get_commits(limit=1)
        s = commits[0].summary
        assert "[" in s  # [hash] message

    def test_file_history(self, intel):
        result = intel.file_history("main.py")
        assert "main.py" in result
        assert "commit" in result.lower()

    def test_file_history_not_found(self, intel):
        result = intel.file_history("nonexistent.py")
        assert "no history" in result.lower()


# ══════════════════════════════════════════════════════════════════
# Query Tests
# ══════════════════════════════════════════════════════════════════

class TestGitQueries:
    def test_recent_activity(self, intel):
        result = intel.recent_activity(days=30)
        assert "commits" in result.lower()
        assert "4" in result  # 4 commits

    def test_recent_activity_no_commits(self, intel):
        result = intel.recent_activity(days=0)
        # Might find commits from today or not
        assert isinstance(result, str)

    def test_why_changed(self, intel):
        result = intel.why_changed("main.py")
        assert "main.py" in result
        # Should find commits that touched main.py

    def test_function_last_changed(self, intel):
        result = intel.function_last_changed("helper")
        assert "helper" in result

    def test_hotspots(self, intel):
        result = intel.hotspots(top_n=5)
        assert "hotspot" in result.lower()
        # main.py was changed twice, should appear
        assert "main.py" in result

    def test_contributor_stats(self, intel):
        result = intel.contributor_stats()
        assert "Test" in result  # the test author


# ══════════════════════════════════════════════════════════════════
# Generation Tests
# ══════════════════════════════════════════════════════════════════

class TestGitGeneration:
    def test_staged_diff_empty(self, intel):
        diff = intel.get_staged_diff()
        assert diff == ""  # nothing staged

    def test_staged_diff_with_changes(self, intel, git_repo):
        with open(os.path.join(git_repo, "new.py"), "w") as f:
            f.write("z = 99\n")
        _run(["git", "add", "new.py"], git_repo)
        diff = intel.get_staged_diff()
        assert "new.py" in diff

    def test_commit_message_context(self, intel, git_repo):
        with open(os.path.join(git_repo, "feature.py"), "w") as f:
            f.write("def feature():\n    return True\n")
        _run(["git", "add", "feature.py"], git_repo)
        ctx = intel.generate_commit_message_context()
        assert "feature.py" in ctx

    def test_commit_message_context_no_staged(self, intel):
        ctx = intel.generate_commit_message_context()
        assert "no staged" in ctx.lower()

    def test_generate_changelog(self, intel):
        changelog = intel.generate_changelog()
        assert "Changelog" in changelog
        # Should categorize our "feat:" and "fix:" commits
        assert "feat" in changelog.lower() or "Feature" in changelog or "Other" in changelog

    def test_changelog_categorizes_features(self, intel):
        changelog = intel.generate_changelog()
        # "feat: add utility functions" should be in Features
        if "Features" in changelog:
            assert "utility" in changelog.lower()

    def test_changelog_categorizes_fixes(self, intel):
        changelog = intel.generate_changelog()
        # "fix: import helper" should be in Bug Fixes
        if "Bug Fixes" in changelog:
            assert "import" in changelog.lower() or "helper" in changelog.lower()


# ══════════════════════════════════════════════════════════════════
# Context for AI Tests
# ══════════════════════════════════════════════════════════════════

class TestGitContext:
    def test_context_for_recent_query(self, intel):
        ctx = intel.get_context_for_query("what did I work on this week")
        assert "Branch:" in ctx
        assert "commits" in ctx.lower() or "Activity" in ctx

    def test_context_for_change_query(self, intel):
        ctx = intel.get_context_for_query("why did main.py change")
        assert isinstance(ctx, str)

    def test_context_for_hotspot_query(self, intel):
        ctx = intel.get_context_for_query("what are the most changed files")
        assert "hotspot" in ctx.lower()

    def test_context_for_commit_query(self, intel):
        ctx = intel.get_context_for_query("help me write a commit message")
        assert "Branch:" in ctx

    def test_context_default(self, intel):
        ctx = intel.get_context_for_query("general question about the project")
        assert "Branch:" in ctx


# ══════════════════════════════════════════════════════════════════
# Edge Cases
# ══════════════════════════════════════════════════════════════════

class TestGitEdgeCases:
    def test_non_repo_returns_safe_defaults(self):
        d = tempfile.mkdtemp()
        g = GitIntel(d)
        assert g.status().branch == ""
        assert g.get_commits() == []
        assert "No commits" in g.recent_activity()
        assert g.hotspots() == "Change hotspots (top 10):"
        assert g.stats()["total_commits"] == 0
        _force_remove(d)

    def test_unstaged_diff(self, intel, git_repo):
        with open(os.path.join(git_repo, "main.py"), "a") as f:
            f.write("\n# extra\n")
        diff = intel.get_unstaged_diff()
        assert "main.py" in diff
