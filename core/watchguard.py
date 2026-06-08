"""
LeanAI M9 — Watchguard
======================

Real-time file watcher that keeps the ProjectBrain index and MemoryForge
graph current with the filesystem.

Design goals (from M9 design doc, signed off):
  - Off by default; user opts in with /watchguard start
  - Triggers brain.rescan_file + memory_forge.sync only (NOT Sentinel)
  - One-line summary per batch, printed between prompts only
  - Silent failure; last error exposed via /watchguard status
  - Auto-pauses during /build, /sentinel, /brain to avoid mid-edit reads
  - Ignores .git/, .venv/, ~/.leanai/, and .gitignore paths
  - Self-trigger prevention: never watches ~/.leanai/ directory

Threading model
---------------
One watchdog Observer thread (owned by watchdog library), plus one
worker thread owned by Watchguard. All file events land in a single
thread-safe queue; the worker drains the queue with a debounce.
Dispatching to brain/memory_forge happens on the worker thread — never
on the observer thread — because brain.rescan_file holds locks.

Output gating
-------------
Summaries are not printed directly. They are appended to an in-memory
queue (drain_pending_output). The main loop reads this queue before
drawing each prompt, so no print() ever lands during input(). This
matches the keyboard-fix pattern established in M8.1.2 and is the
single non-negotiable rule: violating it re-introduces the keyboard
lockup bug.
"""

from __future__ import annotations

import os
import sys
import time
import json
import queue
import threading
import traceback
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple


# ── Module-level constants ────────────────────────────────────────────

# Paths that are ALWAYS ignored, regardless of .gitignore content.
# Absolute-path-rooted — computed at runtime against the project root
# and the user's home, so we never walk into them.
HARDCODED_IGNORE_DIRS = {
    ".git", ".venv", "venv", "env",
    "__pycache__", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", ".tox",
    ".leanai",                 # self-trigger prevention
    ".vscode", ".idea",
}

# File extensions Watchguard cares about. Everything else is dropped
# at the observer-filter layer to keep the queue small.
WATCHED_EXTENSIONS = {".py", ".pyi", ".js", ".jsx", ".ts", ".tsx"}

# Debounce / drain tuning.
DEFAULT_DEBOUNCE_SEC = 2.0
WORKER_POLL_INTERVAL = 0.5       # how often the worker wakes
SOFT_EDGE_MS = 200               # treat events this close to the
                                 # debounce boundary as "ready"

# Log rotation.
LOG_MAX_BYTES = 1024 * 1024      # 1 MB
LOG_BACKUP_COUNT = 3


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class BatchResult:
    """Outcome of processing one debounced batch of file events."""
    files_processed: int = 0
    files_failed: int = 0
    symbols_delta: int = 0
    findings_delta: int = 0
    relations_delta: int = 0
    elapsed_ms: int = 0
    error: Optional[str] = None
    incr_summary: str = ""   # M10: incremental Sentinel fragment, e.g. "sec +1/-2"

    def summary_line(self) -> str:
        """One-line summary, max 80 chars, suitable for printing."""
        parts = [f"{self.files_processed} file{'s' if self.files_processed != 1 else ''} rescanned"]
        if self.symbols_delta:
            parts.append(f"+{self.symbols_delta}s")
        if self.findings_delta:
            parts.append(f"+{self.findings_delta}f")
        if self.relations_delta:
            parts.append(f"+{self.relations_delta}r")
        if self.incr_summary:
            parts.append(self.incr_summary)
        if self.files_failed:
            parts.append(f"{self.files_failed} failed")
        core = " · ".join(parts)
        line = f"[Watchguard] {core} ({self.elapsed_ms}ms)"
        if len(line) > 80:
            line = line[:77] + "..."
        return line


@dataclass
class WatchguardStatus:
    """Point-in-time snapshot of Watchguard state."""
    running: bool = False
    paused: bool = False
    started_at: Optional[float] = None
    batches_processed: int = 0
    batches_failed: int = 0
    files_updated_total: int = 0
    last_batch_at: Optional[float] = None
    last_batch_summary: Optional[str] = None
    last_error: Optional[str] = None
    last_error_at: Optional[float] = None
    queue_depth: int = 0
    pending_output_count: int = 0
    # M10: incremental Sentinel
    incremental_enabled: bool = False
    incr_files_scanned: int = 0
    incr_findings_added: int = 0
    incr_findings_resolved: int = 0
    incr_last_error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── .gitignore loading ────────────────────────────────────────────────

def _load_gitignore_patterns(project_root: str) -> List[str]:
    """Load .gitignore patterns as a flat list of simple substrings.

    We don't implement full gitignore grammar (it's surprisingly rich:
    anchors, negation, double-star, etc.). For Watchguard purposes a
    conservative substring match is enough — if the file path contains
    the pattern, ignore it. Combined with HARDCODED_IGNORE_DIRS, this
    catches ~all of what users expect while staying predictable.

    If the project has no .gitignore, returns [].
    """
    gitignore = os.path.join(project_root, ".gitignore")
    if not os.path.isfile(gitignore):
        return []
    patterns: List[str] = []
    try:
        with open(gitignore, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Strip leading "/" — we're doing substring match anyway
                line = line.lstrip("/")
                # Strip trailing "/" — we match on paths without trailing slash
                line = line.rstrip("/")
                # Skip negations (!) and wildcards we can't handle
                if line.startswith("!"):
                    continue
                # A pattern like "*.log" becomes a suffix check. We just
                # store it as ".log" and check `path.endswith(suffix)`.
                if line.startswith("*."):
                    patterns.append(line[1:])   # e.g. ".log"
                else:
                    patterns.append(line)
    except Exception:
        # Unreadable .gitignore should not crash Watchguard
        pass
    return patterns


# ── Path filter ───────────────────────────────────────────────────────

class PathFilter:
    """Decides whether a file path is relevant to Watchguard.

    Extracted so it can be unit-tested without an Observer.
    """

    def __init__(self, project_root: str, leanai_home: str):
        self.project_root = os.path.abspath(project_root)
        self.leanai_home = os.path.abspath(leanai_home)
        self.gitignore_patterns = _load_gitignore_patterns(self.project_root)

    def should_watch(self, filepath: str) -> bool:
        """True if the path is under project root, has a watched
        extension, and does not sit in any ignored directory or match
        a .gitignore pattern."""
        if not filepath:
            return False
        try:
            abs_path = os.path.abspath(filepath)
        except Exception:
            return False

        # Must be under project root
        try:
            rel = os.path.relpath(abs_path, self.project_root)
        except ValueError:
            # On Windows, cross-drive paths raise — definitely not ours
            return False
        if rel.startswith(".."):
            return False

        # Never watch anything in LeanAI's own home — self-trigger guard
        if abs_path.startswith(self.leanai_home + os.sep):
            return False
        if abs_path == self.leanai_home:
            return False

        # Must be a watched extension
        _, ext = os.path.splitext(abs_path)
        if ext.lower() not in WATCHED_EXTENSIONS:
            return False

        # Must not be in a hardcoded-ignore directory segment
        rel_norm = rel.replace("\\", "/")
        parts = rel_norm.split("/")
        for part in parts[:-1]:  # exclude the filename itself
            if part in HARDCODED_IGNORE_DIRS:
                return False

        # Must not match a gitignore pattern (conservative substring test)
        for pat in self.gitignore_patterns:
            if pat.startswith("."):
                # suffix pattern like ".log"
                if rel_norm.endswith(pat):
                    return False
            elif pat in rel_norm:
                return False

        return True


# ── Event queue entry ─────────────────────────────────────────────────

@dataclass(order=True)
class _QueuedEvent:
    """One file-change notification. Ordered by timestamp so the worker
    can peek at "oldest pending" cheaply."""
    timestamp: float
    filepath: str = field(compare=False)
    kind: str = field(default="modify", compare=False)  # "modify"|"delete"


# ── Logger setup ──────────────────────────────────────────────────────

def _make_logger(log_path: str) -> logging.Logger:
    """Create a rotating-file logger scoped to Watchguard."""
    logger = logging.getLogger(f"leanai.watchguard.{id(log_path)}")
    logger.setLevel(logging.INFO)
    # Remove any previous handlers so tests don't double-log
    for h in list(logger.handlers):
        logger.removeHandler(h)
    try:
        from logging.handlers import RotatingFileHandler
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handler = RotatingFileHandler(
            log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    except Exception:
        # If we can't open the log file, fall back to a null handler —
        # we must never crash because of log issues.
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


# ── Main Watchguard class ─────────────────────────────────────────────

class Watchguard:
    """File-system observer that keeps brain + MemoryForge current.

    Public API:
      start() / stop()       — control the observer thread
      pause() / resume()     — temporarily halt processing without
                               losing events
      status()               — WatchguardStatus snapshot
      drain_pending_output() — main loop calls this before each prompt

    Test API (underscore-prefixed — stable within the file, subject to
    change if M9.1+ revises internals):
      _handle_events_batch(paths) — dispatch a list of file events as
                                    if they had just settled out of
                                    the debounce window. Bypasses both
                                    the observer and the queue.
    """

    # ── Construction ─────────────────────────────────────────────────

    def __init__(
        self,
        project_path: str,
        brain: Any,
        memory_forge: Any,
        *,
        sentinel: Any = None,
        leanai_home: Optional[str] = None,
        debounce_sec: float = DEFAULT_DEBOUNCE_SEC,
        log_path: Optional[str] = None,
        output_queue: Optional[queue.Queue] = None,
        at_prompt_event: Optional[threading.Event] = None,
    ):
        self.project_path = os.path.abspath(project_path)
        self.brain = brain
        self.memory_forge = memory_forge
        # M10: optional SentinelEngine for incremental on-save scanning.
        # Off by default; opt in with enable_incremental(True). If no engine
        # is injected here, one is built lazily from self.brain on first use.
        self.sentinel = sentinel
        self._incremental_enabled = False
        self._incremental_cross_file = False   # M11: 1-hop cross-file taint
        self._incremental = None        # cached IncrementalSentinel
        self.incremental_max_files = 25  # batch cap; >this -> skip + suggest /sentinel

        if leanai_home is None:
            leanai_home = os.environ.get(
                "LEANAI_HOME",
                os.path.join(str(Path.home()), ".leanai"),
            )
        self.leanai_home = os.path.abspath(leanai_home)

        self.debounce_sec = float(debounce_sec)

        # Log file — default to ~/.leanai/watchguard.log
        if log_path is None:
            log_path = os.path.join(self.leanai_home, "watchguard.log")
        self.log_path = log_path
        self._logger = _make_logger(self.log_path)

        # Path filter (lazy-refreshed on start so .gitignore updates pick up)
        self._path_filter: Optional[PathFilter] = None

        # Threading plumbing
        self._event_queue: "queue.PriorityQueue[_QueuedEvent]" = queue.PriorityQueue()
        self._output_queue: "queue.Queue[str]" = (
            output_queue if output_queue is not None else queue.Queue()
        )
        self._at_prompt_event = at_prompt_event    # optional: set by main.py
        self._worker_thread: Optional[threading.Thread] = None
        self._observer: Optional[Any] = None       # watchdog.observers.Observer
        self._stop_flag = threading.Event()
        self._paused_flag = threading.Event()
        self._state_lock = threading.RLock()

        # Status counters
        self._status = WatchguardStatus()

    # ── Properties / status ──────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    @property
    def paused(self) -> bool:
        return self._paused_flag.is_set()

    @property
    def incremental_enabled(self) -> bool:
        """M10: whether incremental Sentinel runs on each batch."""
        return self._incremental_enabled

    @property
    def incremental_cross_file(self) -> bool:
        """M11: whether incremental scans pull in 1-hop import neighbours."""
        return self._incremental_cross_file

    def enable_incremental(self, enabled: bool, *, cross_file=None) -> bool:
        """Turn incremental Sentinel on/off. Returns the EFFECTIVE state.

        Enabling requires a usable SentinelEngine. If none was injected at
        construction, one is built lazily from self.brain. If the brain is
        not available yet, enabling fails and this returns False.

        cross_file (M11): None leaves the mode unchanged; True/False sets
        whether 1-hop import neighbours are scanned together for cross-file
        taint. Changing it rebuilds the cached scanner."""
        if cross_file is not None:
            new_cf = bool(cross_file)
            if new_cf != self._incremental_cross_file:
                self._incremental_cross_file = new_cf
                self._incremental = None   # force rebuild with new mode
        if not enabled:
            self._incremental_enabled = False
            return False
        if self._get_incremental() is None:
            self._incremental_enabled = False
            return False
        self._incremental_enabled = True
        return True

    def _get_incremental(self):
        """Return a cached IncrementalSentinel, building it (and a
        SentinelEngine, if needed) on first use. Returns None if it cannot
        be constructed (e.g. no brain yet). Never raises."""
        if self._incremental is not None:
            return self._incremental
        try:
            from core.sentinel_incremental import IncrementalSentinel
            sentinel = self.sentinel
            if sentinel is None:
                if self.brain is None:
                    return None
                from core.sentinel import SentinelEngine
                sentinel = SentinelEngine(self.brain)
                self.sentinel = sentinel
            self._incremental = IncrementalSentinel(
                sentinel, self.memory_forge,
                max_files=self.incremental_max_files,
                cross_file=self._incremental_cross_file,
            )
            return self._incremental
        except Exception as e:
            self._record_error(f"incremental init failed: {e}")
            return None

    def status(self) -> WatchguardStatus:
        """Snapshot of current status. Safe to call from any thread."""
        with self._state_lock:
            snap = WatchguardStatus(
                running=self.running,
                paused=self.paused,
                started_at=self._status.started_at,
                batches_processed=self._status.batches_processed,
                batches_failed=self._status.batches_failed,
                files_updated_total=self._status.files_updated_total,
                last_batch_at=self._status.last_batch_at,
                last_batch_summary=self._status.last_batch_summary,
                last_error=self._status.last_error,
                last_error_at=self._status.last_error_at,
                queue_depth=self._event_queue.qsize(),
                pending_output_count=self._output_queue.qsize(),
                incremental_enabled=self._incremental_enabled,
                incr_files_scanned=self._status.incr_files_scanned,
                incr_findings_added=self._status.incr_findings_added,
                incr_findings_resolved=self._status.incr_findings_resolved,
                incr_last_error=self._status.incr_last_error,
            )
        return snap

    # ── Start / stop ─────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the observer + worker. Idempotent. Returns True on
        successful start, False if watchdog unavailable or already up."""
        with self._state_lock:
            if self.running:
                return True
            try:
                # Lazy import — watchdog is only needed when start()
                # is called, so the module loads on machines without
                # it installed.
                from watchdog.observers import Observer
                from watchdog.events import FileSystemEventHandler
            except ImportError as e:
                self._record_error(f"watchdog not installed: {e}")
                return False

            self._path_filter = PathFilter(self.project_path, self.leanai_home)

            # Build a per-start handler class that knows how to route
            # events to our queue.
            wg = self

            class _Handler(FileSystemEventHandler):
                def on_created(self, event):
                    if event.is_directory:
                        return
                    wg._enqueue(event.src_path, "modify")

                def on_modified(self, event):
                    if event.is_directory:
                        return
                    wg._enqueue(event.src_path, "modify")

                def on_deleted(self, event):
                    if event.is_directory:
                        return
                    wg._enqueue(event.src_path, "delete")

                def on_moved(self, event):
                    if event.is_directory:
                        return
                    # Treat move as delete(old) + modify(new)
                    wg._enqueue(event.src_path, "delete")
                    wg._enqueue(event.dest_path, "modify")

            try:
                self._observer = Observer()
                self._observer.schedule(
                    _Handler(), self.project_path, recursive=True,
                )
                self._observer.start()
            except Exception as e:
                self._record_error(f"observer start failed: {e}")
                self._observer = None
                return False

            self._stop_flag.clear()
            self._paused_flag.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="watchguard-worker",
                daemon=True,
            )
            self._worker_thread.start()

            self._status.started_at = time.time()
            self._logger.info(
                "Watchguard started (project=%s, debounce=%.1fs)",
                self.project_path, self.debounce_sec,
            )
            return True

    def stop(self) -> None:
        """Stop the observer + worker. Idempotent. Blocks up to 3s
        waiting for threads to settle."""
        with self._state_lock:
            if not self.running and self._observer is None:
                return
            self._stop_flag.set()

            # Stop the watchdog Observer
            obs = self._observer
            if obs is not None:
                try:
                    obs.stop()
                    obs.join(timeout=2.0)
                except Exception as e:
                    self._logger.warning("observer stop error: %s", e)
                self._observer = None

            # Wait for worker
            t = self._worker_thread
            if t is not None and t.is_alive():
                t.join(timeout=3.0)
            self._worker_thread = None
            self._logger.info("Watchguard stopped")

    # ── Pause / resume ───────────────────────────────────────────────

    def pause(self) -> None:
        """Pause batch processing. Events still enqueue; the worker
        waits until resume() is called."""
        self._paused_flag.set()

    def resume(self) -> None:
        """Resume batch processing."""
        self._paused_flag.clear()

    # ── Output drain ─────────────────────────────────────────────────

    def drain_pending_output(self) -> List[str]:
        """Return all pending summary lines and clear the queue.
        Called by the main loop before drawing each prompt. Safe to
        call even if Watchguard was never started."""
        out: List[str] = []
        try:
            while True:
                out.append(self._output_queue.get_nowait())
        except queue.Empty:
            pass
        return out

    # ── Internals: enqueueing ────────────────────────────────────────

    def _enqueue(self, filepath: str, kind: str) -> None:
        """Called from the watchdog observer thread. Cheap — just a
        filter check and a put. Heavy work happens on the worker."""
        try:
            if self._path_filter is None:
                return
            if not self._path_filter.should_watch(filepath):
                return
            ev = _QueuedEvent(
                timestamp=time.time(), filepath=filepath, kind=kind,
            )
            self._event_queue.put(ev)
        except Exception as e:
            # Observer thread must NEVER raise — would kill watchdog
            try:
                self._logger.warning("enqueue error: %s", e)
            except Exception:
                pass

    # ── Internals: worker loop ───────────────────────────────────────

    def _worker_loop(self) -> None:
        """Drain the event queue into debounced batches."""
        try:
            while not self._stop_flag.is_set():
                self._stop_flag.wait(WORKER_POLL_INTERVAL)
                if self._stop_flag.is_set():
                    break
                if self._paused_flag.is_set():
                    continue

                # Pull everything ready out of the queue
                batch_paths = self._collect_ready_events()
                if not batch_paths:
                    continue

                self._process_batch(list(batch_paths))
        except Exception as e:
            # Last-ditch safety — worker must never leak an exception
            tb = traceback.format_exc()
            self._record_error(f"worker crashed: {e}\n{tb}")

    def _collect_ready_events(self) -> List[str]:
        """Pop events that are older than (debounce - soft_edge) seconds.
        Dedupe by path, keeping the latest kind for each."""
        now = time.time()
        threshold = self.debounce_sec - (SOFT_EDGE_MS / 1000.0)
        # Peek the oldest; if it's not ready yet, do nothing
        try:
            oldest = self._event_queue.queue[0]   # thread-safe enough for a peek
        except IndexError:
            return []
        if now - oldest.timestamp < threshold:
            return []

        # Pull everything currently in the queue. By now we know at
        # least the oldest is ready; any later events are effectively
        # "settled with it" for batching purposes.
        seen: dict = {}   # path → kind
        while True:
            try:
                ev = self._event_queue.get_nowait()
            except queue.Empty:
                break
            # Latest-wins for kind
            seen[ev.filepath] = ev.kind
        return list(seen.keys())

    def _process_batch(self, paths: List[str]) -> None:
        """Run brain.rescan_file + memory_forge.sync for a batch.
        Records timing, counts, errors. Enqueues the summary line."""
        result = self._handle_events_batch(paths)
        summary = result.summary_line()
        with self._state_lock:
            self._status.batches_processed += 1
            if result.error:
                self._status.batches_failed += 1
            self._status.files_updated_total += result.files_processed
            self._status.last_batch_at = time.time()
            self._status.last_batch_summary = summary
        # Queue for the main loop to print between prompts
        try:
            self._output_queue.put_nowait(summary)
        except queue.Full:
            pass
        self._logger.info("batch: %s", summary)

    # ── Public test hook (and the real dispatcher) ───────────────────

    def _handle_events_batch(self, paths: List[str]) -> BatchResult:
        """Dispatch a batch of paths. Returns a BatchResult.

        Exposed for tests — they can call this directly and skip the
        observer + debounce, making tests deterministic and platform-
        independent.

        Contract: never raises. All exceptions are caught, logged, and
        summarized into BatchResult.error."""
        result = BatchResult()
        t0 = time.time()

        if not paths:
            result.elapsed_ms = int((time.time() - t0) * 1000)
            return result

        # Snapshot memory_forge stats before/after to compute deltas
        sym_before, fnd_before, rel_before = self._snapshot_counts()

        for path in paths:
            try:
                # brain.rescan_file handles both "file exists, reparse"
                # and "file gone, evict from index" internally
                self.brain.rescan_file(path)
                result.files_processed += 1
            except Exception as e:
                result.files_failed += 1
                self._logger.warning("rescan_file failed for %s: %s", path, e)

        # Sync MemoryForge once per batch, not once per file —
        # cheaper and gives a clean delta count
        if result.files_processed > 0:
            # M10: incremental Sentinel runs BEFORE the sync, so the sync
            # ingests any new/updated VULN-*.json the per-file scans write.
            # Resolved findings are pruned directly (delete JSON +
            # forget_finding). Off by default; no-op unless opted in.
            if self._incremental_enabled:
                try:
                    incr = self._get_incremental()
                    if incr is not None:
                        ir = incr.process_batch(paths)
                        with self._state_lock:
                            self._status.incr_files_scanned += ir.files_scanned
                            self._status.incr_findings_added += ir.findings_added
                            self._status.incr_findings_resolved += ir.findings_resolved
                            if ir.error:
                                self._status.incr_last_error = ir.error
                        frag = ir.summary_line()
                        if frag:
                            result.incr_summary = frag
                        if ir.error:
                            self._logger.warning(
                                "incremental sentinel: %s", ir.error)
                except Exception as e:
                    # Incremental must never break the core batch.
                    self._logger.warning(
                        "incremental sentinel crashed: %s", e)
                    with self._state_lock:
                        self._status.incr_last_error = str(e)

            try:
                self.memory_forge.set_brain(self.brain)
                self.memory_forge.sync(verbose=False)
            except Exception as e:
                result.error = f"memory_forge sync failed: {e}"
                self._logger.warning("memory_forge sync failed: %s", e)

        sym_after, fnd_after, rel_after = self._snapshot_counts()
        result.symbols_delta = sym_after - sym_before
        result.findings_delta = fnd_after - fnd_before
        result.relations_delta = rel_after - rel_before
        result.elapsed_ms = int((time.time() - t0) * 1000)
        return result

    def _snapshot_counts(self) -> Tuple[int, int, int]:
        """Best-effort read of current (symbols, findings, relations)
        counts from memory_forge. Never raises."""
        try:
            st = self.memory_forge.stats()
            return (
                int(st.get("symbols", 0)),
                int(st.get("findings", 0)),
                int(st.get("relations", 0)),
            )
        except Exception:
            return (0, 0, 0)

    def _record_error(self, msg: str) -> None:
        with self._state_lock:
            self._status.last_error = msg
            self._status.last_error_at = time.time()
        try:
            self._logger.error(msg)
        except Exception:
            pass


# ── Pause context manager for main.py use ─────────────────────────────

class _PauseContext:
    """Context manager that pauses a Watchguard for the duration of
    a block. Safe if watchguard is None.

        with watchguard_pause(wg):
            run_sentinel()
    """

    def __init__(self, wg: Optional[Watchguard]):
        self.wg = wg
        self._was_running = False

    def __enter__(self):
        if self.wg is not None and self.wg.running:
            self._was_running = True
            self.wg.pause()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._was_running and self.wg is not None:
            try:
                self.wg.resume()
            except Exception:
                pass
        return False  # never swallow exceptions


def watchguard_pause(wg: Optional[Watchguard]) -> _PauseContext:
    """Return a context manager that pauses `wg` during its scope."""
    return _PauseContext(wg)


# ── Output formatting for /watchguard status ──────────────────────────

def format_status(status: WatchguardStatus, color: bool = True) -> str:
    """Render a WatchguardStatus as a human-readable block for the
    /watchguard status command."""
    if color:
        BOLD = "\033[1m"
        DIM = "\033[2m"
        GREEN = "\033[32m"
        RED = "\033[31m"
        YELLOW = "\033[33m"
        RESET = "\033[0m"
    else:
        BOLD = DIM = GREEN = RED = YELLOW = RESET = ""

    if status.running:
        state = f"{GREEN}running{RESET}"
        if status.paused:
            state = f"{YELLOW}paused{RESET}"
    else:
        state = f"{DIM}stopped{RESET}"

    lines = [f"{BOLD}Watchguard{RESET} — {state}"]
    lines.append("")

    if status.started_at:
        uptime = time.time() - status.started_at
        lines.append(f"  Uptime          {_fmt_duration(uptime)}")

    lines.append(f"  Batches         {status.batches_processed} processed, {status.batches_failed} failed")
    lines.append(f"  Files updated   {status.files_updated_total}")

    if status.last_batch_at:
        age = time.time() - status.last_batch_at
        lines.append(f"  Last batch      {_fmt_duration(age)} ago")
    if status.last_batch_summary:
        lines.append(f"                  {DIM}{status.last_batch_summary}{RESET}")

    lines.append(f"  Queue depth     {status.queue_depth}")
    lines.append(f"  Pending output  {status.pending_output_count}")

    # M10: incremental Sentinel
    incr_state = f"{GREEN}on{RESET}" if status.incremental_enabled else f"{DIM}off{RESET}"
    lines.append("")
    lines.append(f"  Incremental     {incr_state}  {DIM}(per-file scan on save){RESET}")
    if status.incremental_enabled or status.incr_files_scanned:
        lines.append(
            f"                  {DIM}{status.incr_files_scanned} scanned, "
            f"+{status.incr_findings_added} new, "
            f"-{status.incr_findings_resolved} resolved{RESET}"
        )
        lines.append(
            f"                  {DIM}note: same-file flows only; "
            f"run /sentinel for cross-file taint{RESET}"
        )
    if status.incr_last_error:
        err = status.incr_last_error
        if len(err) > 120:
            err = err[:120] + "..."
        lines.append(f"                  {RED}last incr error:{RESET} {DIM}{err}{RESET}")

    if status.last_error:
        age = (time.time() - status.last_error_at) if status.last_error_at else 0
        lines.append("")
        lines.append(f"  {RED}Last error{RESET}      {_fmt_duration(age)} ago")
        # Truncate long tracebacks
        err = status.last_error
        if len(err) > 200:
            err = err[:200] + "..."
        for l in err.splitlines()[:3]:
            lines.append(f"    {DIM}{l}{RESET}")

    return "\n".join(lines)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m {int(seconds % 60)}s"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h {int((seconds % 3600) / 60)}m"
    return f"{int(seconds / 86400)}d {int((seconds % 86400) / 3600)}h"


__all__ = [
    "Watchguard",
    "WatchguardStatus",
    "BatchResult",
    "PathFilter",
    "watchguard_pause",
    "format_status",
    "WATCHED_EXTENSIONS",
    "HARDCODED_IGNORE_DIRS",
]
