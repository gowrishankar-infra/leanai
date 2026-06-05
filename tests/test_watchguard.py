"""
Tests for LeanAI M9 — Watchguard

All tests use a temp directory and a fake brain + fake memory_forge that
record calls. No real watchdog Observer is started in most tests — we
test the dispatcher (_handle_events_batch) and the PathFilter directly.
This makes tests deterministic and platform-independent.

One integration test (test_real_observer_fires) starts a real Observer
if watchdog is importable; it's skipped otherwise.
"""

from __future__ import annotations

import os
import sys
import time
import json
import queue
import threading
import tempfile
import shutil
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import dataclass, field

import pytest

# Make core/ importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.watchguard import (
    Watchguard,
    WatchguardStatus,
    BatchResult,
    PathFilter,
    watchguard_pause,
    format_status,
    WATCHED_EXTENSIONS,
    HARDCODED_IGNORE_DIRS,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory."""
    proj = tmp_path / "project"
    proj.mkdir()
    # Create a representative layout
    (proj / "core").mkdir()
    (proj / "tests").mkdir()
    (proj / ".git").mkdir()
    (proj / ".venv").mkdir()
    (proj / "__pycache__").mkdir()
    (proj / "node_modules").mkdir()
    (proj / "core" / "server.py").write_text("def handle(): pass\n")
    (proj / "tests" / "test_server.py").write_text("def test(): pass\n")
    (proj / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (proj / "README.md").write_text("# test\n")
    # .gitignore
    (proj / ".gitignore").write_text("*.log\nbuild/\nsecrets.py\n")
    return proj


@pytest.fixture
def tmp_home(tmp_path):
    """Temporary LEANAI_HOME."""
    home = tmp_path / "leanai_home"
    home.mkdir()
    return home


@dataclass
class FakeBrain:
    """Records every rescan_file call."""
    rescans: List[str] = field(default_factory=list)
    raise_on: Optional[str] = None   # substring that triggers a raise
    _file_analyses: dict = field(default_factory=dict)

    def rescan_file(self, path: str):
        if self.raise_on and self.raise_on in path:
            raise RuntimeError(f"fake rescan failure for {path}")
        self.rescans.append(path)


@dataclass
class FakeMemoryForge:
    """Records sync() calls and returns fake stats."""
    syncs: int = 0
    raise_on_sync: bool = False
    symbol_count: int = 100
    finding_count: int = 0
    relation_count: int = 200
    set_brain_calls: int = 0

    def set_brain(self, brain):
        self.set_brain_calls += 1

    def sync(self, verbose: bool = False):
        self.syncs += 1
        if self.raise_on_sync:
            raise RuntimeError("fake sync failure")
        # Simulate gaining a symbol on every sync so BatchResult deltas
        # come out non-zero
        self.symbol_count += 5
        self.relation_count += 3

    def stats(self):
        return {
            "symbols": self.symbol_count,
            "findings": self.finding_count,
            "relations": self.relation_count,
        }


@pytest.fixture
def wg_factory(tmp_project, tmp_home):
    """Factory: make_wg() returns (Watchguard, brain, mf) tuples."""
    def _make():
        brain = FakeBrain()
        mf = FakeMemoryForge()
        wg = Watchguard(
            project_path=str(tmp_project),
            brain=brain,
            memory_forge=mf,
            leanai_home=str(tmp_home),
            debounce_sec=0.1,   # fast for tests
            log_path=str(tmp_home / "watchguard.log"),
        )
        return wg, brain, mf
    return _make


# ═══════════════════════════════════════════════════════════════════
# PathFilter tests
# ═══════════════════════════════════════════════════════════════════

class TestPathFilter:
    def test_watches_python_files(self, tmp_project, tmp_home):
        pf = PathFilter(str(tmp_project), str(tmp_home))
        assert pf.should_watch(str(tmp_project / "core" / "server.py"))

    def test_ignores_markdown_files(self, tmp_project, tmp_home):
        pf = PathFilter(str(tmp_project), str(tmp_home))
        assert not pf.should_watch(str(tmp_project / "README.md"))

    def test_ignores_git_directory(self, tmp_project, tmp_home):
        pf = PathFilter(str(tmp_project), str(tmp_home))
        (tmp_project / ".git" / "hook.py").write_text("pass")
        assert not pf.should_watch(str(tmp_project / ".git" / "hook.py"))

    def test_ignores_venv_directory(self, tmp_project, tmp_home):
        pf = PathFilter(str(tmp_project), str(tmp_home))
        (tmp_project / ".venv" / "lib.py").write_text("pass")
        assert not pf.should_watch(str(tmp_project / ".venv" / "lib.py"))

    def test_ignores_pycache(self, tmp_project, tmp_home):
        pf = PathFilter(str(tmp_project), str(tmp_home))
        (tmp_project / "__pycache__" / "mod.cpython-313.pyc").write_text("")
        assert not pf.should_watch(
            str(tmp_project / "__pycache__" / "mod.py")
        )

    def test_ignores_leanai_home(self, tmp_project, tmp_home):
        """Self-trigger prevention — the critical one."""
        pf = PathFilter(str(tmp_project), str(tmp_home))
        (tmp_home / "something.py").write_text("pass")
        assert not pf.should_watch(str(tmp_home / "something.py"))

    def test_respects_gitignore_suffix(self, tmp_project, tmp_home):
        pf = PathFilter(str(tmp_project), str(tmp_home))
        # .gitignore has "*.log" — but .log isn't in WATCHED_EXTENSIONS
        # so this is moot; more interesting is "secrets.py"
        (tmp_project / "secrets.py").write_text("API_KEY='x'")
        assert not pf.should_watch(str(tmp_project / "secrets.py"))

    def test_rejects_paths_outside_project(self, tmp_project, tmp_home, tmp_path):
        pf = PathFilter(str(tmp_project), str(tmp_home))
        outside = tmp_path / "elsewhere.py"
        outside.write_text("pass")
        assert not pf.should_watch(str(outside))


# ═══════════════════════════════════════════════════════════════════
# Dispatch tests (no observer, direct _handle_events_batch calls)
# ═══════════════════════════════════════════════════════════════════

class TestDispatch:
    def test_file_edit_triggers_rescan(self, wg_factory, tmp_project):
        wg, brain, mf = wg_factory()
        path = str(tmp_project / "core" / "server.py")
        result = wg._handle_events_batch([path])
        assert path in brain.rescans
        assert result.files_processed == 1
        assert result.files_failed == 0
        assert mf.syncs == 1

    def test_file_delete_triggers_rescan(self, wg_factory, tmp_project):
        """Deletion → rescan_file is called, brain handles pop internally."""
        wg, brain, mf = wg_factory()
        # Pretend the file was deleted — path doesn't need to exist
        path = str(tmp_project / "core" / "deleted.py")
        result = wg._handle_events_batch([path])
        assert path in brain.rescans
        assert result.files_processed == 1

    def test_batch_of_same_file_deduped_upstream(self, wg_factory, tmp_project):
        """Dedupe happens in _collect_ready_events. If caller passes
        duplicates to _handle_events_batch, they DO get processed multiple
        times — which is intentional, callers decide."""
        wg, brain, mf = wg_factory()
        path = str(tmp_project / "core" / "server.py")
        wg._handle_events_batch([path, path, path])
        assert brain.rescans.count(path) == 3

    def test_failure_in_rescan_is_caught(self, wg_factory, tmp_project):
        wg, brain, mf = wg_factory()
        brain.raise_on = "core/server.py"   # specific — only one file matches
        # Also create the second file so paths are realistic
        result = wg._handle_events_batch([
            str(tmp_project / "core" / "server.py"),
            str(tmp_project / "tests" / "test_server.py"),
        ])
        assert result.files_failed == 1
        # The other file still processed
        assert result.files_processed == 1

    def test_failure_in_memory_forge_sync_caught(self, wg_factory, tmp_project):
        wg, brain, mf = wg_factory()
        mf.raise_on_sync = True
        path = str(tmp_project / "core" / "server.py")
        result = wg._handle_events_batch([path])
        # rescan_file still ran
        assert result.files_processed == 1
        # error captured
        assert result.error is not None
        assert "memory_forge sync failed" in result.error

    def test_empty_batch_is_noop(self, wg_factory):
        wg, brain, mf = wg_factory()
        result = wg._handle_events_batch([])
        assert result.files_processed == 0
        assert mf.syncs == 0
        assert not brain.rescans

    def test_batch_computes_deltas(self, wg_factory, tmp_project):
        wg, brain, mf = wg_factory()
        path = str(tmp_project / "core" / "server.py")
        result = wg._handle_events_batch([path])
        # FakeMemoryForge.sync adds +5 symbols, +3 relations each call
        assert result.symbols_delta == 5
        assert result.relations_delta == 3


# ═══════════════════════════════════════════════════════════════════
# Start / stop / idempotency tests
# ═══════════════════════════════════════════════════════════════════

class TestLifecycle:
    def test_stop_before_start_is_noop(self, wg_factory):
        wg, _, _ = wg_factory()
        # Must not raise
        wg.stop()
        assert not wg.running

    def test_start_without_watchdog_fails_gracefully(
        self, wg_factory, monkeypatch
    ):
        """If watchdog is missing, start() returns False and records the
        error. Simulates the import failing by inserting a broken module."""
        wg, _, _ = wg_factory()
        # Block watchdog import by inserting a dummy that raises on attr access
        class _BrokenObserver:
            def __init__(self, *a, **kw):
                raise RuntimeError("simulated missing watchdog")
        # Easier: monkeypatch the internal class lookup
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "watchdog.observers" or name == "watchdog":
                raise ImportError("watchdog not installed (simulated)")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        ok = wg.start()
        assert ok is False
        assert not wg.running
        assert wg.status().last_error is not None


class TestPauseResume:
    def test_pause_resume_flags(self, wg_factory):
        wg, _, _ = wg_factory()
        assert not wg.paused
        wg.pause()
        assert wg.paused
        wg.resume()
        assert not wg.paused

    def test_pause_context_manager(self, wg_factory):
        wg, _, _ = wg_factory()
        # Force it into the "running" state manually by setting a fake
        # worker thread — context manager only pauses if running
        fake_thread = threading.Thread(target=lambda: time.sleep(0.2))
        fake_thread.start()
        wg._worker_thread = fake_thread

        assert not wg.paused
        with watchguard_pause(wg):
            assert wg.paused
        assert not wg.paused

        fake_thread.join()

    def test_pause_context_manager_safe_for_none(self):
        """Must be safe to pass None — happens when watchguard isn't
        instantiated yet."""
        with watchguard_pause(None):
            pass   # must not raise


# ═══════════════════════════════════════════════════════════════════
# Output drain tests
# ═══════════════════════════════════════════════════════════════════

class TestOutputDrain:
    def test_drain_empty_queue(self, wg_factory):
        wg, _, _ = wg_factory()
        assert wg.drain_pending_output() == []

    def test_summary_queued_after_batch(self, wg_factory, tmp_project):
        wg, _, _ = wg_factory()
        wg._process_batch([str(tmp_project / "core" / "server.py")])
        out = wg.drain_pending_output()
        assert len(out) == 1
        assert "[Watchguard]" in out[0]
        assert "rescanned" in out[0]

    def test_drain_clears_queue(self, wg_factory, tmp_project):
        wg, _, _ = wg_factory()
        wg._process_batch([str(tmp_project / "core" / "server.py")])
        wg.drain_pending_output()
        # Second drain is empty
        assert wg.drain_pending_output() == []


# ═══════════════════════════════════════════════════════════════════
# Status / formatting tests
# ═══════════════════════════════════════════════════════════════════

class TestStatus:
    def test_initial_status(self, wg_factory):
        wg, _, _ = wg_factory()
        s = wg.status()
        assert s.running is False
        assert s.paused is False
        assert s.batches_processed == 0
        assert s.last_error is None

    def test_status_counts_processed_batches(self, wg_factory, tmp_project):
        wg, _, _ = wg_factory()
        wg._process_batch([str(tmp_project / "core" / "server.py")])
        wg._process_batch([str(tmp_project / "tests" / "test_server.py")])
        s = wg.status()
        assert s.batches_processed == 2
        assert s.files_updated_total == 2

    def test_status_records_last_error(self, wg_factory, tmp_project):
        wg, _, mf = wg_factory()
        mf.raise_on_sync = True
        wg._process_batch([str(tmp_project / "core" / "server.py")])
        s = wg.status()
        assert s.batches_failed == 1

    def test_format_status_includes_state(self, wg_factory):
        wg, _, _ = wg_factory()
        s = wg.status()
        out = format_status(s, color=False)
        assert "Watchguard" in out
        assert "stopped" in out


# ═══════════════════════════════════════════════════════════════════
# BatchResult formatting
# ═══════════════════════════════════════════════════════════════════

class TestBatchResultSummary:
    def test_summary_single_file(self):
        r = BatchResult(files_processed=1, symbols_delta=3, elapsed_ms=120)
        s = r.summary_line()
        assert "1 file rescanned" in s
        assert "+3s" in s

    def test_summary_multiple_files(self):
        r = BatchResult(files_processed=3, symbols_delta=5, relations_delta=7, elapsed_ms=200)
        s = r.summary_line()
        assert "3 files rescanned" in s

    def test_summary_truncated_if_too_long(self):
        r = BatchResult(files_processed=999999, symbols_delta=99999,
                        findings_delta=9999, relations_delta=99999,
                        files_failed=999, elapsed_ms=99999)
        s = r.summary_line()
        assert len(s) <= 80

    def test_summary_includes_failures(self):
        r = BatchResult(files_processed=2, files_failed=1, elapsed_ms=50)
        s = r.summary_line()
        assert "1 failed" in s


# ═══════════════════════════════════════════════════════════════════
# Real watchdog integration (opportunistic, skipped if unavailable)
# ═══════════════════════════════════════════════════════════════════

class TestRealObserver:
    def test_real_observer_fires_on_file_write(self, wg_factory, tmp_project):
        """Start a real Observer, write a file, confirm the worker
        produced a summary. Skipped if watchdog isn't installed."""
        try:
            import watchdog.observers    # noqa: F401
        except ImportError:
            pytest.skip("watchdog not installed")

        wg, brain, mf = wg_factory()
        ok = wg.start()
        if not ok:
            pytest.skip("observer couldn't start on this platform")
        try:
            # Give watchdog a moment to settle
            time.sleep(0.2)
            target = tmp_project / "core" / "new_file.py"
            target.write_text("def hello(): pass\n")

            # Wait up to 5 seconds for the batch to settle + process
            deadline = time.time() + 5.0
            got_summary = False
            while time.time() < deadline:
                out = wg.drain_pending_output()
                if out:
                    got_summary = True
                    break
                time.sleep(0.1)

            assert got_summary, \
                "Watchguard should have produced a summary line"
            assert any("new_file.py" in p for p in brain.rescans), \
                f"brain.rescan_file should have been called; got {brain.rescans}"
        finally:
            wg.stop()
