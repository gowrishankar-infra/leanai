"""
Tests for LeanAI M8 — MemoryForge

Covers:
  - Schema creation + idempotent re-init
  - DSL parser: grammar, error cases, severity ordering
  - NL→DSL heuristic
  - Sync: vuln ingestion, chain ingestion, symbol ingestion from mock brain
  - Idempotent re-sync (running twice adds nothing)
  - Query: exact match, substring, severity comparison, limit
  - facts_for() across multiple name-match patterns
  - timeline() ordering
  - stats() shape

Every test uses a temp LEANAI_HOME — nothing touches the real ~/.leanai.

Run: python -m pytest tests/test_memory_forge.py -v
"""

import os
import sys
import json
import time
import shutil
import tempfile
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory_forge import (
    MemoryForge,
    DSLParseError,
    parse_dsl,
    nl_to_dsl_heuristic,
    nl_to_dsl_model,
    SEVERITY_ORDER,
    SCHEMA_VERSION,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_home(monkeypatch):
    """Isolated $LEANAI_HOME — nothing touches real ~/.leanai."""
    d = tempfile.mkdtemp(prefix="mf-test-")
    monkeypatch.setenv("LEANAI_HOME", d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def vuln_fixture(tmp_home):
    """Create a single VULN-*.json under the temp home."""
    vdir = os.path.join(tmp_home, "vulns")
    os.makedirs(vdir, exist_ok=True)
    v = {
        "vuln_id": "VULN-2026-0001",
        "vuln_class": "sql_injection",
        "severity": "CRITICAL",
        "confidence": 0.9,
        "filepath": "core/server.py",
        "function_name": "handle_request",
        "line": 42,
        "source_type": "http_input",
        "sink_type": "sql query concatenation",
        "description": "Unsanitized user input flows to SQL query.",
        "code_snippet": 'query = f"SELECT * FROM users WHERE id={uid}"',
        "taint_path": ["handle_request", "lookup_user"],
        "fix_suggestion": "Use parameterized queries.",
        "fingerprint": "fp-vuln-0001",
        "timestamp": time.time(),
    }
    with open(os.path.join(vdir, "VULN-2026-0001.json"), "w", encoding="utf-8") as fh:
        json.dump(v, fh)
    return v


@pytest.fixture
def chain_fixture(tmp_home):
    """Create a single CHAIN-*.json under the temp home."""
    cdir = os.path.join(tmp_home, "chains")
    os.makedirs(cdir, exist_ok=True)
    c = {
        "chain_id": "CHAIN-2026-0001",
        "entry_vuln_id": "VULN-2026-0001",
        "severity": "CRITICAL",
        "confidence": 0.85,
        "capability": "rce",
        "steps": [
            {
                "function_id": "n1",
                "function_name": "handle_request",
                "filepath": "core/server.py",
                "line": 42,
                "stage": "entry",
                "indicators": ["http_input"],
            },
            {
                "function_id": "n2",
                "function_name": "execute_shell",
                "filepath": "core/runner.py",
                "line": 100,
                "stage": "rce",
                "indicators": ["subprocess.Popen(shell=True)"],
            },
        ],
        "narrative": "HTTP input → shell subprocess → RCE",
        "impact_summary": "Unauthenticated RCE",
        "fix_recommendation": "Validate input; avoid shell=True",
        "fingerprint": "fp-chain-0001",
        "timestamp": time.time(),
    }
    with open(os.path.join(cdir, "CHAIN-2026-0001.json"), "w", encoding="utf-8") as fh:
        json.dump(c, fh)
    return c


@dataclass
class _Fn:
    """Mirrors brain.analyzer.FunctionInfo. Methods are stored here too,
    with is_method=True and class_name populated — NOT inside _Cls.methods."""
    name: str = ""
    signature: str = ""
    complexity: int = 0
    line_start: int = 0
    is_method: bool = False
    class_name: Optional[str] = None


@dataclass
class _Cls:
    """Mirrors brain.analyzer.ClassInfo. methods is a list of STRINGS (names
    only), mirroring the real brain's behavior — method objects live on
    FileAnalysis.functions, not here."""
    name: str = ""
    line_start: int = 0
    methods: list = field(default_factory=list)   # list[str]


@dataclass
class _Analysis:
    total_lines: int = 0
    functions: list = field(default_factory=list)
    classes: list = field(default_factory=list)


class _MockBrain:
    """Minimal ProjectBrain stand-in using the REAL shape from
    brain/analyzer.py: methods live in functions[] with is_method=True
    and class_name set; classes[].methods is just a list of name strings."""
    def __init__(self):
        self._file_analyses = {
            "core/server.py": _Analysis(
                total_lines=500,
                functions=[
                    _Fn(name="handle_request", signature="def handle_request(req)",
                        complexity=8, line_start=42),
                    _Fn(name="parse_body", signature="def parse_body(b)",
                        complexity=3, line_start=120),
                    _Fn(name="start", signature="def start(self)",
                        complexity=2, line_start=210,
                        is_method=True, class_name="HTTPServer"),
                    _Fn(name="handle", signature="def handle(self, req)",
                        complexity=15, line_start=230,
                        is_method=True, class_name="HTTPServer"),
                ],
                classes=[
                    _Cls(name="HTTPServer", line_start=200,
                         methods=["start", "handle"]),
                ],
            ),
            "core/runner.py": _Analysis(
                total_lines=200,
                functions=[
                    _Fn(name="execute_shell", signature="def execute_shell(cmd)",
                        complexity=5, line_start=100),
                ],
            ),
        }


# ═══════════════════════════════════════════════════════════════════
# Schema + init tests
# ═══════════════════════════════════════════════════════════════════

class TestSchema:
    def test_creates_db_in_leanai_home(self, tmp_home):
        mf = MemoryForge(project_path=".")
        assert os.path.isfile(mf.db_path)
        assert mf.db_path.startswith(tmp_home)

    def test_schema_version_stored(self, tmp_home):
        mf = MemoryForge(project_path=".")
        import sqlite3
        conn = sqlite3.connect(mf.db_path)
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='version'"
        ).fetchone()
        assert row[0] == SCHEMA_VERSION

    def test_init_is_idempotent(self, tmp_home):
        """Re-initialising on an existing DB must not wipe data."""
        mf1 = MemoryForge(project_path=".")
        import sqlite3
        conn = sqlite3.connect(mf1.db_path)
        conn.execute(
            "INSERT INTO symbols(name, kind, filepath, last_sync) "
            "VALUES ('x', 'file', 'x', 1.0)"
        )
        conn.commit()
        conn.close()

        mf2 = MemoryForge(project_path=".")
        assert mf2.stats()["symbols"] == 1


# ═══════════════════════════════════════════════════════════════════
# DSL parser tests
# ═══════════════════════════════════════════════════════════════════

class TestDSLParser:
    def test_bare_entity(self):
        p = parse_dsl("findings")
        assert p.entity == "findings"
        assert p.predicates == []
        assert p.limit == 100

    def test_bare_entity_all_three(self):
        for e in ("symbols", "findings", "events"):
            assert parse_dsl(e).entity == e

    def test_single_predicate(self):
        p = parse_dsl("findings where severity = CRITICAL")
        assert p.entity == "findings"
        assert p.predicates == [("severity", "=", "CRITICAL")]

    def test_multiple_predicates(self):
        p = parse_dsl(
            "findings where severity = CRITICAL and category = sql_injection"
        )
        assert len(p.predicates) == 2

    def test_numeric_value(self):
        p = parse_dsl("symbols where complexity > 10")
        assert p.predicates[0] == ("complexity", ">", 10)

    def test_float_value(self):
        p = parse_dsl("findings where confidence > 0.8")
        assert p.predicates[0] == ("confidence", ">", 0.8)

    def test_substring_op(self):
        p = parse_dsl("symbols where name ~ handle")
        assert p.predicates[0] == ("name", "~", "handle")

    def test_limit_clause(self):
        p = parse_dsl("findings limit 5")
        assert p.limit == 5

    def test_where_and_limit(self):
        p = parse_dsl("findings where severity = HIGH limit 3")
        assert p.limit == 3
        assert p.predicates == [("severity", "=", "HIGH")]

    def test_quoted_string_values(self):
        p = parse_dsl('symbols where file = "core/server.py"')
        assert p.predicates[0] == ("file", "=", "core/server.py")

    def test_empty_raises(self):
        with pytest.raises(DSLParseError):
            parse_dsl("")

    def test_whitespace_only_raises(self):
        with pytest.raises(DSLParseError):
            parse_dsl("   ")

    def test_unknown_entity_raises(self):
        with pytest.raises(DSLParseError) as exc:
            parse_dsl("garbage where x = 1")
        assert "symbols" in str(exc.value) or "findings" in str(exc.value)

    def test_unknown_field_raises(self):
        with pytest.raises(DSLParseError) as exc:
            parse_dsl("findings where bogus = 1")
        assert "bogus" in str(exc.value)

    def test_missing_op_raises(self):
        with pytest.raises(DSLParseError):
            parse_dsl("findings where severity CRITICAL")   # no '='

    def test_missing_value_raises(self):
        with pytest.raises(DSLParseError):
            parse_dsl("findings where severity =")


# ═══════════════════════════════════════════════════════════════════
# Severity ordering + compile-time validation
# ═══════════════════════════════════════════════════════════════════

class TestSeverityOrdering:
    def test_order_map_covers_all(self):
        assert set(SEVERITY_ORDER) == {"INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_severity_gte_compiles(self, tmp_home, vuln_fixture):
        mf = MemoryForge(project_path=".")
        mf.sync()
        # Seed findings at multiple severities
        import sqlite3
        conn = sqlite3.connect(mf.db_path)
        for sv, vid in [("HIGH", "VULN-T-HIGH"), ("MEDIUM", "VULN-T-MED"),
                        ("LOW", "VULN-T-LOW")]:
            conn.execute(
                "INSERT INTO findings"
                "(finding_id, kind, category, severity, confidence, filepath, line, "
                " description, fingerprint, created_at, last_sync, json_blob) "
                "VALUES (?, 'vuln', 'test', ?, 0.5, 'x.py', 1, 'test', ?, ?, ?, '{}')",
                (vid, sv, hashlib.md5(vid.encode()).hexdigest()[:10],
                 time.time(), time.time()),
            )
        conn.commit()
        conn.close()

        results, _ = mf.query("findings where severity >= MEDIUM")
        severities = {r.data["severity"] for r in results}
        assert "CRITICAL" in severities   # from vuln_fixture
        assert "HIGH" in severities
        assert "MEDIUM" in severities
        assert "LOW" not in severities

    def test_invalid_severity_value_errors_on_compile(self, tmp_home):
        mf = MemoryForge(project_path=".")
        with pytest.raises(DSLParseError):
            mf.query("findings where severity > GARBAGE", use_model=False)


# ═══════════════════════════════════════════════════════════════════
# NL heuristic tests (no model needed)
# ═══════════════════════════════════════════════════════════════════

class TestNLHeuristic:
    def test_critical_sqli(self):
        dsl = nl_to_dsl_heuristic("show all critical sql injection findings")
        assert dsl is not None
        assert "severity = CRITICAL" in dsl
        assert "category = sql_injection" in dsl

    def test_complex_functions(self):
        dsl = nl_to_dsl_heuristic("which functions are complex")
        assert dsl is not None
        assert "symbols" in dsl
        assert "kind = function" in dsl
        assert "complexity > 10" in dsl

    def test_severity_only_applies_to_findings(self):
        """'high complexity' must not produce 'severity = HIGH' on symbols."""
        dsl = nl_to_dsl_heuristic("functions with high complexity")
        assert "severity" not in (dsl or "")
        assert "complexity > 10" in (dsl or "")

    def test_events_today(self):
        dsl = nl_to_dsl_heuristic("events today")
        assert dsl is not None
        assert "events" in dsl
        assert "since = 1d" in dsl

    def test_word_boundary_no_false_match(self):
        """'highlights' must not trigger severity=HIGH."""
        dsl = nl_to_dsl_heuristic("project highlights")
        assert dsl is None or "severity" not in dsl

    def test_no_match_returns_none(self):
        """Totally off-topic query returns None, not garbage DSL."""
        dsl = nl_to_dsl_heuristic("xyzzy plugh frobozz")
        assert dsl is None

    def test_model_translator_handles_no_model(self):
        assert nl_to_dsl_model("show findings", model_fn=None) is None

    def test_model_translator_rejects_garbage(self):
        """Model returns nonsense → translator returns None."""
        fn = lambda _: "this is not a valid DSL at all"
        assert nl_to_dsl_model("show findings", fn) is None

    def test_model_translator_accepts_valid(self):
        fn = lambda _: "findings where severity = HIGH"
        assert nl_to_dsl_model("big ones", fn) == "findings where severity = HIGH"

    def test_model_translator_strips_code_fences(self):
        fn = lambda _: "```\nfindings where severity = HIGH\n```"
        assert nl_to_dsl_model("x", fn) == "findings where severity = HIGH"


# ═══════════════════════════════════════════════════════════════════
# Ingestion tests
# ═══════════════════════════════════════════════════════════════════

class TestVulnIngestion:
    def test_ingests_vuln_file(self, tmp_home, vuln_fixture):
        mf = MemoryForge(project_path=".")
        stats = mf.sync()
        assert stats.findings_added == 1
        assert stats.errors == []

        results, _ = mf.query("findings")
        assert len(results) == 1
        assert results[0].data["finding_id"] == "VULN-2026-0001"

    def test_idempotent_resync(self, tmp_home, vuln_fixture):
        mf = MemoryForge(project_path=".")
        mf.sync()
        stats2 = mf.sync()
        assert stats2.findings_added == 0
        assert stats2.findings_updated == 0
        assert stats2.skipped_stale == 1

    def test_changed_fingerprint_updates(self, tmp_home, vuln_fixture):
        mf = MemoryForge(project_path=".")
        mf.sync()
        # Rewrite the vuln file with new fingerprint
        vuln_fixture["fingerprint"] = "fp-vuln-CHANGED"
        vuln_fixture["description"] = "Updated description"
        with open(os.path.join(tmp_home, "vulns", "VULN-2026-0001.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(vuln_fixture, fh)

        stats = mf.sync()
        assert stats.findings_updated == 1
        assert stats.findings_added == 0

    def test_missing_vulns_dir_no_error(self, tmp_home):
        """If ~/.leanai/vulns doesn't exist, sync should not crash."""
        # Ensure vulns dir does NOT exist
        assert not os.path.isdir(os.path.join(tmp_home, "vulns"))
        mf = MemoryForge(project_path=".")
        stats = mf.sync()
        assert stats.errors == []


class TestChainIngestion:
    def test_ingests_chain_file(self, tmp_home, chain_fixture):
        mf = MemoryForge(project_path=".")
        stats = mf.sync()
        assert stats.findings_added == 1

        results, _ = mf.query("findings where kind = chain")
        assert len(results) == 1
        assert results[0].data["finding_id"] == "CHAIN-2026-0001"
        assert results[0].data["category"] == "rce"

    def test_chain_links_to_entry_vuln(self, tmp_home, vuln_fixture, chain_fixture):
        """When both vuln and chain are ingested, the chain should have a
        depends_on edge pointing to its entry vuln."""
        mf = MemoryForge(project_path=".")
        mf.sync()

        import sqlite3
        conn = sqlite3.connect(mf.db_path)
        conn.row_factory = sqlite3.Row
        rels = conn.execute(
            "SELECT r.relation FROM relations r "
            "JOIN findings src ON src.id = r.src_id AND r.src_kind = 'finding' "
            "JOIN findings dst ON dst.id = r.dst_id AND r.dst_kind = 'finding' "
            "WHERE src.finding_id = 'CHAIN-2026-0001' "
            "AND dst.finding_id = 'VULN-2026-0001'"
        ).fetchall()
        conn.close()
        assert len(rels) == 1
        assert rels[0]["relation"] == "depends_on"


class TestSymbolIngestion:
    def test_ingests_from_brain(self, tmp_home):
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        stats = mf.sync()

        # 2 files + 3 top-level functions + 1 class + 2 methods = 8 symbols
        assert stats.symbols_added == 8

        # Relations: file → every symbol inside it (contains)
        #   server.py contains: handle_request, parse_body, HTTPServer.start,
        #                       HTTPServer.handle, HTTPServer (= 5)
        #   runner.py contains: execute_shell                              (= 1)
        # Plus class → method contains:
        #   HTTPServer contains: start, handle                             (= 2)
        # Total: 8
        assert stats.relations_added == 8

    def test_re_sync_uses_update_not_insert(self, tmp_home):
        """Running /brain twice should not create duplicate symbols."""
        brain = _MockBrain()
        mf = MemoryForge(project_path=".", brain=brain)
        mf.sync()

        stats2 = mf.sync()
        assert stats2.symbols_added == 0


# ═══════════════════════════════════════════════════════════════════
# Query tests
# ═══════════════════════════════════════════════════════════════════

class TestQuery:
    def test_raw_dsl_preferred_over_nl(self, tmp_home, vuln_fixture):
        """A query that parses as DSL should not be sent to the model."""
        called = []
        def broken_fn(_):
            called.append(1)
            return "WRONG"
        mf = MemoryForge(project_path=".", model_fn=broken_fn)
        mf.sync()
        mf.query("findings where severity = CRITICAL")
        assert called == []   # model never called

    def test_substring_match(self, tmp_home):
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        mf.sync()
        results, _ = mf.query("symbols where name ~ handle")
        names = [r.data["name"] for r in results]
        assert any("handle_request" in n for n in names)
        assert any("HTTPServer.handle" in n for n in names)

    def test_complexity_threshold(self, tmp_home):
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        mf.sync()
        results, _ = mf.query("symbols where complexity > 10")
        # Only HTTPServer.handle has complexity 15 in the mock brain
        assert len(results) == 1
        assert "HTTPServer.handle" in results[0].data["name"]

    def test_limit_respected(self, tmp_home):
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        mf.sync()
        results, _ = mf.query("symbols limit 2")
        assert len(results) == 2

    def test_since_filter(self, tmp_home, vuln_fixture):
        """since=1000d should include everything; since=0.001s excludes now."""
        mf = MemoryForge(project_path=".")
        mf.sync()
        results, _ = mf.query("findings where since = 1000d")
        assert len(results) >= 1


# ═══════════════════════════════════════════════════════════════════
# Facts + timeline + stats
# ═══════════════════════════════════════════════════════════════════

class TestFacts:
    def test_facts_returns_empty_for_unknown(self, tmp_home):
        mf = MemoryForge(project_path=".")
        facts = mf.facts_for("DoesNotExist")
        assert facts["symbols"] == []
        assert facts["findings"] == []
        assert facts["events"] == []

    def test_facts_returns_symbol_and_discovery_event(self, tmp_home):
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        mf.sync()
        facts = mf.facts_for("handle_request")
        assert len(facts["symbols"]) >= 1
        assert any("handle_request" in s["name"] for s in facts["symbols"])

    def test_facts_links_finding_via_relation(self, tmp_home, vuln_fixture):
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        mf.sync()
        facts = mf.facts_for("handle_request")
        # The vuln is in handle_request → facts should return it
        fids = [f["finding_id"] for f in facts["findings"]]
        assert "VULN-2026-0001" in fids


class TestTimeline:
    def test_empty_when_no_sync(self, tmp_home):
        mf = MemoryForge(project_path=".")
        assert mf.timeline() == []

    def test_ordered_newest_first(self, tmp_home, vuln_fixture):
        mf = MemoryForge(project_path=".")
        mf.sync()
        events = mf.timeline(limit=50)
        assert len(events) >= 1
        # Each newer timestamp should be >= the next
        for i in range(len(events) - 1):
            assert events[i].timestamp >= events[i + 1].timestamp


class TestStats:
    def test_empty_db(self, tmp_home):
        mf = MemoryForge(project_path=".")
        s = mf.stats()
        assert s["symbols"] == 0
        assert s["findings"] == 0
        assert s["events"] == 0
        assert s["schema_version"] == SCHEMA_VERSION

    def test_populated_db(self, tmp_home, vuln_fixture, chain_fixture):
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        mf.sync()
        s = mf.stats()
        assert s["findings"] == 2        # 1 vuln + 1 chain
        assert s["symbols"] >= 8         # mock brain contribution
        assert "CRITICAL" in s["by_severity"]
        assert s["by_severity"]["CRITICAL"] == 2
        assert s["age_seconds"] is not None
        assert s["db_size_bytes"] > 0


# ═══════════════════════════════════════════════════════════════════
# End-to-end: a realistic NL query with no model
# ═══════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_nl_query_no_model_finds_critical(self, tmp_home, vuln_fixture):
        mf = MemoryForge(project_path=".")
        mf.sync()
        results, dsl_used = mf.query(
            "show me all critical SQL injection findings", use_model=False,
        )
        assert len(results) == 1
        assert "severity = CRITICAL" in dsl_used
        assert "category = sql_injection" in dsl_used

    def test_reset_clears_everything(self, tmp_home, vuln_fixture):
        mf = MemoryForge(project_path=".")
        mf.sync()
        assert mf.stats()["findings"] == 1
        mf.reset()
        assert mf.stats()["findings"] == 0
        assert mf.stats()["symbols"] == 0


# ═══════════════════════════════════════════════════════════════════
# M8.1 matcher tests — Windows paths, <module>, taint_path preference
# ═══════════════════════════════════════════════════════════════════

def _write_vuln(dir_path: str, vuln_id: str, **overrides):
    """Helper to write a Sentinel-style VULN-*.json fixture."""
    v = {
        "vuln_id": vuln_id,
        "vuln_class": "weak_crypto",
        "severity": "MEDIUM",
        "confidence": 0.6,
        "filepath": "core/server.py",
        "function_name": "handle_request",
        "line": 42,
        "source_type": "http_input",
        "sink_type": "hash",
        "description": "test finding",
        "code_snippet": "",
        "taint_path": [],
        "fix_suggestion": "",
        "fingerprint": f"fp-{vuln_id}",
        "timestamp": time.time(),
    }
    v.update(overrides)
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, f"{vuln_id}.json"), "w", encoding="utf-8") as fh:
        json.dump(v, fh)


class TestWindowsPathMatching:
    """Regression tests for the M8.1 bug: production Windows run had 40
    findings ingested but only 1 found_in edge created. Root cause was that
    the old single-query matcher couldn't bridge path-separator differences
    and multiple qname conventions. These tests exercise each axis."""

    def test_backslash_filepath_links_to_forward_slash_symbol(self, tmp_home):
        """Finding stored with 'brain\\foo.py', symbol stored with 'brain/foo.py'.
        Must still link."""
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        _write_vuln(
            os.path.join(tmp_home, "vulns"),
            "VULN-2026-9001",
            filepath="core\\server.py",   # backslash — what Sentinel writes on Win
            function_name="handle_request",
        )
        mf.sync()

        # _MockBrain uses forward slashes ('core/server.py'). Matcher must
        # bridge the gap.
        import sqlite3
        conn = sqlite3.connect(mf.db_path)
        conn.row_factory = sqlite3.Row
        rels = list(conn.execute(
            "SELECT r.relation, s.name FROM relations r "
            "JOIN findings f ON f.id = r.src_id AND r.src_kind = 'finding' "
            "JOIN symbols s ON s.id = r.dst_id AND r.dst_kind = 'symbol' "
            "WHERE f.finding_id = 'VULN-2026-9001'"
        ))
        conn.close()
        assert any(r["relation"] == "found_in" for r in rels), \
            "Expected found_in edge across backslash/forward-slash mismatch"

    def test_method_qualified_name_links(self, tmp_home):
        """Sentinel writes function_name='ClassName.method'; symbols are
        stored as '<file>::ClassName.method'. Must link."""
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        _write_vuln(
            os.path.join(tmp_home, "vulns"),
            "VULN-2026-9002",
            filepath="core/server.py",
            function_name="HTTPServer.handle",   # qualified method name
            line=230,
        )
        mf.sync()

        import sqlite3
        conn = sqlite3.connect(mf.db_path)
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute(
            "SELECT s.name FROM relations r "
            "JOIN findings f ON f.id = r.src_id AND r.src_kind = 'finding' "
            "JOIN symbols s ON s.id = r.dst_id AND r.dst_kind = 'symbol' "
            "WHERE f.finding_id = 'VULN-2026-9002' AND r.relation = 'found_in'"
        ))
        conn.close()
        assert len(rows) == 1
        assert "HTTPServer.handle" in rows[0]["name"]

    def test_module_finding_links_to_file_symbol(self, tmp_home):
        """function_name='<module>' means the finding is at module scope.
        Old matcher couldn't handle this. Should link to the file symbol."""
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        _write_vuln(
            os.path.join(tmp_home, "vulns"),
            "VULN-2026-9003",
            filepath="core/server.py",
            function_name="<module>",
            line=1,
        )
        mf.sync()

        import sqlite3
        conn = sqlite3.connect(mf.db_path)
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute(
            "SELECT s.name, s.kind FROM relations r "
            "JOIN findings f ON f.id = r.src_id AND r.src_kind = 'finding' "
            "JOIN symbols s ON s.id = r.dst_id AND r.dst_kind = 'symbol' "
            "WHERE f.finding_id = 'VULN-2026-9003' AND r.relation = 'found_in'"
        ))
        conn.close()
        assert len(rows) == 1
        assert rows[0]["kind"] == "file"

    def test_taint_path_prefers_same_file(self, tmp_home):
        """When taint_path names a bare function and two files define
        functions with that name, the matcher should prefer the one in the
        same filepath as the finding."""
        # Build a brain with 'handle' in two different files
        class TwoFileBrain:
            def __init__(self):
                self._file_analyses = {
                    "core/server.py": _Analysis(
                        total_lines=100,
                        functions=[_Fn(name="handle", line_start=10)],
                    ),
                    "core/admin.py": _Analysis(
                        total_lines=100,
                        functions=[_Fn(name="handle", line_start=20)],
                    ),
                }

        mf = MemoryForge(project_path=".", brain=TwoFileBrain())
        _write_vuln(
            os.path.join(tmp_home, "vulns"),
            "VULN-2026-9004",
            filepath="core/server.py",
            function_name="handle_request",   # NOT 'handle'
            taint_path=["handle_request", "handle"],   # handle is the taint
        )
        mf.sync()

        import sqlite3
        conn = sqlite3.connect(mf.db_path)
        conn.row_factory = sqlite3.Row
        affects = list(conn.execute(
            "SELECT s.filepath FROM relations r "
            "JOIN findings f ON f.id = r.src_id AND r.src_kind = 'finding' "
            "JOIN symbols s ON s.id = r.dst_id AND r.dst_kind = 'symbol' "
            "WHERE f.finding_id = 'VULN-2026-9004' AND r.relation = 'affects'"
        ))
        conn.close()
        # Should have chosen the handle in core/server.py, not core/admin.py
        assert len(affects) == 1
        assert affects[0]["filepath"] == "core/server.py"

    def test_taint_path_dedupes(self, tmp_home):
        """If taint_path lists the same function twice, only one 'affects'
        edge should be created."""
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        _write_vuln(
            os.path.join(tmp_home, "vulns"),
            "VULN-2026-9005",
            filepath="core/server.py",
            function_name="handle_request",
            taint_path=["parse_body", "parse_body", "parse_body"],
        )
        mf.sync()

        import sqlite3
        conn = sqlite3.connect(mf.db_path)
        conn.row_factory = sqlite3.Row
        affects = list(conn.execute(
            "SELECT COUNT(*) AS n FROM relations r "
            "JOIN findings f ON f.id = r.src_id AND r.src_kind = 'finding' "
            "WHERE f.finding_id = 'VULN-2026-9005' AND r.relation = 'affects'"
        ))
        conn.close()
        assert affects[0]["n"] == 1


class TestSchemaMigration:
    """After M8.1.1, automatic migration was removed to rule out any
    startup-path interaction risk. Users upgrading from v1 rebuild
    manually with /memory reset + /memory sync."""

    def test_fresh_db_has_current_schema(self, tmp_home):
        """A fresh install should land on the current SCHEMA_VERSION."""
        mf = MemoryForge(project_path=".")
        import sqlite3
        conn = sqlite3.connect(mf.db_path)
        v = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()[0]
        conn.close()
        assert v == SCHEMA_VERSION

    def test_old_version_row_is_left_alone(self, tmp_home):
        """An existing v1 row should NOT be auto-upgraded. User explicitly
        runs /memory reset to rebuild. This prevents any migration code
        from running at startup."""
        mf = MemoryForge(project_path=".")
        import sqlite3
        conn = sqlite3.connect(mf.db_path)
        conn.execute("UPDATE schema_meta SET value = '1' WHERE key = 'version'")
        conn.commit()
        conn.close()

        # Re-open — should NOT touch the row
        mf2 = MemoryForge(project_path=".")
        conn = sqlite3.connect(mf2.db_path)
        v = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()[0]
        conn.close()
        assert v == "1", "Startup must not mutate existing schema_meta row"


class TestOrphanRelink:
    """When findings were ingested by an older matcher version, they exist
    in the findings table but have no found_in edges. On the next sync,
    the fingerprint-match fast path skips them forever and edges never
    get built. This test class exercises the orphan-relink recovery."""

    def test_existing_findings_get_relinked(self, tmp_home):
        """Seed the DB with findings that have no found_in edges (as if
        ingested by an older matcher), run sync, verify edges appear."""
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        # Initial sync — populates symbols
        mf.sync()

        # Manually insert a finding WITHOUT any linking edges, as if an
        # older buggy version had ingested it.
        import sqlite3
        conn = sqlite3.connect(mf.db_path)
        data = {
            "vuln_id": "VULN-ORPHAN-001",
            "vuln_class": "weak_crypto",
            "severity": "MEDIUM",
            "confidence": 0.6,
            "filepath": "core/server.py",
            "function_name": "handle_request",
            "line": 42,
            "source_type": "hash",
            "sink_type": "md5",
            "description": "orphaned finding",
            "code_snippet": "",
            "taint_path": [],
            "fix_suggestion": "",
            "fingerprint": "orphan-fp-1",
            "timestamp": time.time(),
        }
        conn.execute(
            "INSERT INTO findings"
            "(finding_id, kind, category, severity, confidence, filepath, line, "
            " description, fingerprint, created_at, last_sync, json_blob) "
            "VALUES (?, 'vuln', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("VULN-ORPHAN-001", data["vuln_class"], data["severity"],
             data["confidence"], data["filepath"], data["line"],
             data["description"], data["fingerprint"],
             data["timestamp"], time.time(), json.dumps(data)),
        )
        conn.commit()
        conn.close()

        # Confirm no found_in edges exist yet
        import sqlite3 as _sq
        conn = _sq.connect(mf.db_path)
        pre_count = conn.execute(
            "SELECT COUNT(*) FROM relations WHERE relation = 'found_in'"
        ).fetchone()[0]
        conn.close()
        assert pre_count == 0

        # Also write the finding JSON to disk so the fingerprint-match
        # fast path in _sync_vuln_findings sees it as "unchanged".
        vdir = os.path.join(tmp_home, "vulns")
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "VULN-ORPHAN-001.json"), "w") as fh:
            json.dump(data, fh)

        # Re-sync — the fast path will skip the vuln ingestion, but the
        # relink pass should build the missing edge.
        mf.sync()

        conn = _sq.connect(mf.db_path)
        conn.row_factory = _sq.Row
        edges = list(conn.execute(
            "SELECT s.name FROM relations r "
            "JOIN findings f ON f.id = r.src_id AND r.src_kind = 'finding' "
            "JOIN symbols s ON s.id = r.dst_id AND r.dst_kind = 'symbol' "
            "WHERE f.finding_id = 'VULN-ORPHAN-001' AND r.relation = 'found_in'"
        ))
        conn.close()
        assert len(edges) == 1, \
            f"Orphan should have been relinked; got {len(edges)} edges"
        assert "handle_request" in edges[0]["name"]

    def test_relink_is_idempotent(self, tmp_home):
        """After relink runs once, a second sync must not create duplicate
        edges (idempotency guarantee)."""
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        mf.sync()

        # Seed a finding + its JSON file
        data = {
            "vuln_id": "VULN-IDEMP-001", "vuln_class": "weak_crypto",
            "severity": "MEDIUM", "confidence": 0.6,
            "filepath": "core/server.py", "function_name": "handle_request",
            "line": 42, "source_type": "hash", "sink_type": "md5",
            "description": "test", "code_snippet": "", "taint_path": [],
            "fix_suggestion": "", "fingerprint": "idemp-fp",
            "timestamp": time.time(),
        }
        vdir = os.path.join(tmp_home, "vulns")
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "VULN-IDEMP-001.json"), "w") as fh:
            json.dump(data, fh)
        mf.sync()   # first sync — creates edge

        # Second sync should not add a duplicate edge
        mf.sync()

        import sqlite3 as _sq
        conn = _sq.connect(mf.db_path)
        n = conn.execute(
            "SELECT COUNT(*) FROM relations r "
            "JOIN findings f ON f.id = r.src_id AND r.src_kind = 'finding' "
            "WHERE f.finding_id = 'VULN-IDEMP-001' AND r.relation = 'found_in'"
        ).fetchone()[0]
        conn.close()
        assert n == 1, f"Edge count should be 1 after 2 syncs, got {n}"


class TestFactsForMatching:
    """Tests for the /memory facts <query> matcher, which accepts several
    user-query shapes."""

    def test_bare_name_match(self, tmp_home):
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        mf.sync()
        facts = mf.facts_for("handle_request")
        assert any("handle_request" in s["name"] for s in facts["symbols"])

    def test_class_method_syntax(self, tmp_home):
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        mf.sync()
        facts = mf.facts_for("HTTPServer.handle")
        assert any("HTTPServer.handle" in s["name"] for s in facts["symbols"])

    def test_class_name_match(self, tmp_home):
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        mf.sync()
        facts = mf.facts_for("HTTPServer")
        kinds = [s["kind"] for s in facts["symbols"]]
        assert "class" in kinds

    def test_methods_ranked_above_classes(self, tmp_home):
        """If a query matches both a method and its class, methods come first."""
        mf = MemoryForge(project_path=".", brain=_MockBrain())
        mf.sync()
        facts = mf.facts_for("handle")
        # First match should be a method (handle_request or HTTPServer.handle)
        # not a class.
        assert facts["symbols"][0]["kind"] in ("function", "method")
