"""
LeanAI — M8 MemoryForge
========================

A persistent, queryable knowledge graph of everything LeanAI has learned
about your project. Threads Sentinel findings, ChainBreaker chains,
forensics events, and brain-derived symbols into a single SQLite graph
that survives across sessions.

Design rules (these are non-negotiable — see session journal end-M7):
  1. NEVER trust model output as fact. Every row in this graph has a
     `source_tool` column. The model is allowed to PHRASE answers, but
     the underlying facts all come from deterministic tools (AST, git
     log, Sentinel patterns, ChainBreaker graph walk).
  2. NEVER write to source code. MemoryForge reads; it does not edit.
  3. Idempotent sync. Running /memory sync twice in a row should produce
     zero new rows. All ingestion is keyed on stable fingerprints.
  4. Graceful degradation. If a source directory doesn't exist
     (no /sentinel has been run yet), skip it silently. Don't crash.
  5. The NL→DSL step is optional. If the model refuses or produces
     garbage, the user can still run raw DSL queries. Fallback is a
     keyword-to-DSL heuristic that never needs a model.

Schema (sqlite at $LEANAI_HOME/memory_forge/graph.db):

  symbols      — functions, methods, classes, files (from brain)
  findings     — VULN-* from Sentinel, CHAIN-* from ChainBreaker
  events       — discovery / scan / modification timeline
  relations    — 5 types: contains / affects / depends_on / modified_by / found_in
  sync_state   — per-source last-sync timestamp + fingerprint set
  schema_meta  — schema version for future migrations

Public API:
  mf = MemoryForge(project_path=".")            # create / open
  mf.set_brain(brain)                           # optional, enables symbol sync
  mf.set_model_fn(model_fn)                     # optional, enables NL→DSL
  stats = mf.sync(verbose=False)                # ingest everything new
  results = mf.query(dsl_or_nl_string)          # run a query
  facts = mf.facts_for("ProjectBrain.scan")     # all known facts about a symbol
  events = mf.timeline(limit=20)                # chronological events
  stats = mf.stats()                            # graph stats

DSL grammar (tiny, regular, no parser combinators needed):

  <entity> [where <predicate> [and <predicate> ...]] [limit N]

  entity   := symbols | findings | events
  predicate:= <field> <op> <value>
  op       := = | != | > | < | >= | <= | ~    (~ is substring match)
  value    := quoted-string | unquoted-word | number | severity-keyword

  Examples:
    findings where severity >= MEDIUM
    findings where category = sql_injection and severity = CRITICAL
    symbols where kind = function and complexity > 15
    events where source = sentinel since = 2026-04-01 limit 10
    findings where file ~ core/server.py

Author: Gowri Shankar — MemoryForge shipped as M8 of the LeanAI roadmap.
License: AGPL-3.0 (same as the rest of LeanAI).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

SCHEMA_VERSION = "2"

SEVERITY_ORDER = {
    "INFO": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

VALID_RELATIONS = {
    "contains",      # file → function (structural)
    "found_in",      # finding → symbol (location)
    "affects",       # finding → symbol (downstream impact via taint/chain)
    "depends_on",    # symbol → symbol (from brain's call graph)
    "modified_by",   # symbol → event (forensics / git)
}

VALID_SYMBOL_KINDS = {"file", "function", "method", "class", "module"}
VALID_FINDING_KINDS = {"vuln", "chain"}
VALID_EVENT_KINDS = {"discovery", "scan", "modification", "fix", "sync"}


# ═══════════════════════════════════════════════════════════════════
# Data classes — return shapes for query results
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SymbolRow:
    id: int
    name: str
    kind: str
    filepath: str
    line: int
    signature: str
    complexity: int
    lines: int
    last_sync: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FindingRow:
    id: int
    finding_id: str          # VULN-2026-0001 / CHAIN-2026-0001
    kind: str                # vuln | chain
    category: str            # sql_injection / rce / exfil / ...
    severity: str
    confidence: float
    filepath: str
    line: int
    description: str
    fingerprint: str
    created_at: float
    last_sync: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EventRow:
    id: int
    kind: str                # discovery | scan | modification | fix | sync
    source_tool: str         # sentinel | chainbreaker | brain | forensics | user
    timestamp: float
    description: str
    symbol_id: Optional[int]
    finding_id: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SyncStats:
    symbols_added: int = 0
    symbols_updated: int = 0
    findings_added: int = 0
    findings_updated: int = 0
    events_added: int = 0
    relations_added: int = 0
    skipped_stale: int = 0
    errors: List[str] = field(default_factory=list)
    time_ms: int = 0

    @property
    def changes(self) -> int:
        return (self.symbols_added + self.symbols_updated +
                self.findings_added + self.findings_updated +
                self.events_added + self.relations_added)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['changes'] = self.changes
        return d


@dataclass
class QueryResult:
    """Wraps one row of any query output, with its entity kind."""
    entity: str               # 'symbol' | 'finding' | 'event'
    data: dict

    def to_dict(self) -> dict:
        return {"entity": self.entity, "data": self.data}


# ═══════════════════════════════════════════════════════════════════
# DSL parser — tiny, strict, no dependencies
# ═══════════════════════════════════════════════════════════════════

# Fields that map to SQL columns per entity
_ENTITY_FIELDS: Dict[str, Dict[str, str]] = {
    "symbols": {
        "name": "name",
        "kind": "kind",
        "file": "filepath",
        "filepath": "filepath",
        "line": "line",
        "signature": "signature",
        "complexity": "complexity",
        "lines": "lines",
    },
    "findings": {
        "id": "finding_id",
        "finding_id": "finding_id",
        "kind": "kind",
        "category": "category",
        "severity": "severity",
        "confidence": "confidence",
        "file": "filepath",
        "filepath": "filepath",
        "line": "line",
        "description": "description",
        "since": "_since",   # virtual — maps to created_at >= value
    },
    "events": {
        "kind": "kind",
        "source": "source_tool",
        "source_tool": "source_tool",
        "description": "description",
        "since": "_since",
    },
}

_VALID_OPS = {"=", "!=", ">", "<", ">=", "<=", "~"}

# Tokens: quoted-string, word, number, operators, keywords
_TOKEN_RE = re.compile(
    r'"([^"]*)"|\'([^\']*)\'|(>=|<=|!=|=|>|<|~)|(\d+(?:\.\d+)?)|([A-Za-z_][A-Za-z0-9_./-]*)'
)


def _tokenize(s: str) -> List[Tuple[str, str]]:
    """Lex the DSL into (kind, value) tuples. Unknown chars are skipped."""
    tokens: List[Tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(s):
        qs1, qs2, op, num, word = m.groups()
        if qs1 is not None:
            tokens.append(("STR", qs1))
        elif qs2 is not None:
            tokens.append(("STR", qs2))
        elif op is not None:
            tokens.append(("OP", op))
        elif num is not None:
            tokens.append(("NUM", num))
        elif word is not None:
            tokens.append(("WORD", word))
    return tokens


class DSLParseError(ValueError):
    """Raised when a DSL query can't be parsed."""


@dataclass
class ParsedQuery:
    entity: str                   # 'symbols' | 'findings' | 'events'
    predicates: List[Tuple[str, str, Any]]  # (field, op, value)
    limit: int = 100


def parse_dsl(dsl: str) -> ParsedQuery:
    """
    Parse a DSL query string. Strict — raises DSLParseError on any malformed
    input. The NL→DSL model is expected to produce well-formed DSL; user
    typos fall back to error messages that tell them the grammar.
    """
    if not dsl or not dsl.strip():
        raise DSLParseError("Empty query")

    tokens = _tokenize(dsl)
    if not tokens:
        raise DSLParseError("No tokens in query")

    # First token must be an entity
    first_kind, first_val = tokens[0]
    if first_kind != "WORD" or first_val.lower() not in _ENTITY_FIELDS:
        raise DSLParseError(
            f"Query must start with one of: {sorted(_ENTITY_FIELDS.keys())} "
            f"(got {first_val!r})"
        )
    entity = first_val.lower()
    idx = 1
    predicates: List[Tuple[str, str, Any]] = []
    limit = 100

    # Optional 'where <predicates>' then optional 'limit N'
    while idx < len(tokens):
        kind, val = tokens[idx]

        if kind == "WORD" and val.lower() == "where":
            idx += 1
            # parse one or more predicates separated by 'and'
            while idx < len(tokens):
                # field
                if idx >= len(tokens) or tokens[idx][0] != "WORD":
                    raise DSLParseError(
                        f"Expected field name at position {idx} "
                        f"(got {tokens[idx] if idx < len(tokens) else 'EOF'})"
                    )
                field_name = tokens[idx][1].lower()
                idx += 1

                # op
                if idx >= len(tokens) or tokens[idx][0] != "OP":
                    raise DSLParseError(
                        f"Expected operator after {field_name!r} "
                        f"(one of: {_VALID_OPS})"
                    )
                op = tokens[idx][1]
                idx += 1

                # value
                if idx >= len(tokens):
                    raise DSLParseError(f"Expected value after {field_name} {op}")
                vkind, vval = tokens[idx]
                if vkind == "NUM":
                    value: Any = float(vval) if "." in vval else int(vval)
                else:
                    value = vval
                idx += 1

                # Field must be valid for this entity
                if field_name not in _ENTITY_FIELDS[entity]:
                    raise DSLParseError(
                        f"Unknown field {field_name!r} for {entity}. "
                        f"Valid: {sorted(_ENTITY_FIELDS[entity].keys())}"
                    )

                predicates.append((field_name, op, value))

                # 'and' continues, anything else ends the predicate list
                if idx < len(tokens) and tokens[idx][0] == "WORD" and tokens[idx][1].lower() == "and":
                    idx += 1
                    continue
                break
            continue

        if kind == "WORD" and val.lower() == "limit":
            idx += 1
            if idx >= len(tokens) or tokens[idx][0] != "NUM":
                raise DSLParseError("'limit' must be followed by a number")
            limit = int(tokens[idx][1])
            idx += 1
            continue

        # Unknown trailing token — skip it rather than fail, so a stray
        # word from the NL layer doesn't kill the whole query.
        idx += 1

    return ParsedQuery(entity=entity, predicates=predicates, limit=limit)


def _compile_predicates(
    entity: str, predicates: List[Tuple[str, str, Any]]
) -> Tuple[str, List[Any]]:
    """
    Convert parsed predicates into a SQL WHERE fragment + parameter list.
    Returns ('', []) if there are no predicates.
    """
    if not predicates:
        return "", []

    clauses: List[str] = []
    params: List[Any] = []
    field_map = _ENTITY_FIELDS[entity]

    for field_name, op, value in predicates:
        col = field_map[field_name]

        # Virtual 'since' field → timestamp on created_at (findings) or
        # timestamp (events).
        if col == "_since":
            ts_col = "created_at" if entity == "findings" else "timestamp"
            ts = _parse_since(value)
            clauses.append(f"{ts_col} >= ?")
            params.append(ts)
            continue

        # Severity ordered comparison — stored as string, compared via CASE
        if field_name == "severity" and op in {">", "<", ">=", "<="}:
            sv = str(value).upper()
            if sv not in SEVERITY_ORDER:
                raise DSLParseError(
                    f"Unknown severity {value!r}. Valid: {sorted(SEVERITY_ORDER)}"
                )
            ord_val = SEVERITY_ORDER[sv]
            clauses.append(
                "(CASE severity "
                + " ".join(f"WHEN '{k}' THEN {v}" for k, v in SEVERITY_ORDER.items())
                + f" ELSE -1 END) {op} ?"
            )
            params.append(ord_val)
            continue

        # Severity equality → uppercase normalization
        if field_name == "severity" and op in {"=", "!="}:
            sv = str(value).upper()
            clauses.append(f"severity {op} ?")
            params.append(sv)
            continue

        # Substring match
        if op == "~":
            clauses.append(f"{col} LIKE ?")
            params.append(f"%{value}%")
            continue

        # Generic comparison
        if op not in _VALID_OPS:
            raise DSLParseError(f"Unknown operator {op!r}")
        clauses.append(f"{col} {op} ?")
        params.append(value)

    return "WHERE " + " AND ".join(clauses), params


def _parse_since(value: Any) -> float:
    """Parse a since= value into a unix timestamp. Accepts ISO date, ISO
    datetime, unix timestamp, or relative ('7d', '24h', '30m')."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    # Relative: 7d, 24h, 30m
    m = re.match(r"^(\d+)([dhm])$", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        secs = {"d": 86400, "h": 3600, "m": 60}[unit]
        return time.time() - n * secs
    # ISO date or datetime
    try:
        # date-only
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            import datetime as _dt
            return _dt.datetime.strptime(s, "%Y-%m-%d").timestamp()
        # datetime
        if "T" in s or " " in s:
            import datetime as _dt
            s_norm = s.replace("T", " ").split(".")[0]
            return _dt.datetime.strptime(s_norm, "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception as e:
        raise DSLParseError(f"Could not parse since={value!r}: {e}")
    # Fallback: try as unix
    try:
        return float(s)
    except ValueError:
        raise DSLParseError(f"Could not parse since={value!r} as date or timestamp")


# ═══════════════════════════════════════════════════════════════════
# NL → DSL heuristic (no model needed)
# ═══════════════════════════════════════════════════════════════════

_NL_SEVERITY_KEYWORDS = {
    "critical": "CRITICAL",
    "crit": "CRITICAL",
    "high": "HIGH",
    "medium": "MEDIUM",
    "med": "MEDIUM",
    "low": "LOW",
}

_NL_CATEGORY_KEYWORDS = {
    "sql": "sql_injection",
    "sqli": "sql_injection",
    "injection": "sql_injection",
    "command": "command_injection",
    "rce": "rce",
    "xss": "xss",
    "ssrf": "ssrf",
    "path": "path_traversal",
    "traversal": "path_traversal",
    "deserialization": "unsafe_deserialization",
    "secret": "hardcoded_secret",
    "crypto": "weak_crypto",
    "race": "race_condition",
    "redirect": "open_redirect",
    "temp": "insecure_temp",
    "auth": "missing_auth",
    "exfil": "exfil",
    "exfiltration": "exfil",
    "persistence": "persistence",
    "destroy": "destroy",
    "privesc": "privesc",
}


def nl_to_dsl_heuristic(nl: str) -> Optional[str]:
    """
    Heuristic NL→DSL. Handles the common cases without ever calling a model.
    Returns None if it can't confidently generate a DSL query.
    """
    q = nl.lower().strip()
    if not q:
        return None

    # Detect entity
    if any(w in q for w in ("finding", "vuln", "vulnerability", "chain", "attack")):
        entity = "findings"
    elif any(w in q for w in ("event", "timeline", "history", "when", "happened")):
        entity = "events"
    elif any(w in q for w in ("symbol", "function", "method", "class", "file")):
        entity = "symbols"
    else:
        # If we can't tell, findings is the most common intent
        entity = "findings"

    preds: List[str] = []

    # Severity — only applies to findings. Symbols have no severity
    # field, and applying this to symbols produces queries that fail
    # at compile time. "functions with high complexity" is NOT a severity
    # query; skip it here and let the complexity branch below handle it.
    if entity == "findings":
        for kw, sv in _NL_SEVERITY_KEYWORDS.items():
            # word-boundary match so 'highlights' doesn't match 'high'
            if re.search(rf"\b{kw}\b", q):
                preds.append(f"severity = {sv}")
                break

    # Category — also only applies to findings
    if entity == "findings":
        for kw, cat in _NL_CATEGORY_KEYWORDS.items():
            if re.search(rf"\b{kw}\b", q):
                preds.append(f"category = {cat}")
                break

    # Recency
    if any(w in q for w in ("today", "recent", "recently", "new")):
        preds.append("since = 1d")
    elif "yesterday" in q:
        preds.append("since = 2d")
    elif "week" in q or "7 day" in q:
        preds.append("since = 7d")
    elif "month" in q or "30 day" in q:
        preds.append("since = 30d")

    # Symbol kind + complexity hints
    if entity == "symbols":
        for kw, kind in (("function", "function"), ("method", "method"),
                         ("class", "class"), ("file", "file")):
            if re.search(rf"\b{kw}s?\b", q):
                preds.append(f"kind = {kind}")
                break
        # "high complexity" / "complex" → complexity > 10
        if re.search(r"\bcomplex(?:ity)?\b", q) and re.search(r"\b(high|very|too|over)\b", q):
            preds.append("complexity > 10")
        elif re.search(r"\bcomplex(?:ity)?\b", q):
            # bare "complex functions" is still a "high complexity" intent
            preds.append("complexity > 10")

    # Source tool for events
    if entity == "events":
        for tool in ("sentinel", "chainbreaker", "brain", "forensics"):
            if tool in q:
                preds.append(f"source = {tool}")
                break

    # If we extracted absolutely nothing, bail out — the caller will tell
    # the user to write DSL directly instead of us guessing wrong.
    if not preds and entity == "findings":
        # But for the bare question "show me findings" we can at least
        # default to "findings" — an unfiltered query is still useful.
        if any(w in q for w in ("all", "show", "list", "what", "any")):
            return f"{entity} limit 20"
        return None

    if not preds:
        return f"{entity} limit 20"

    return f"{entity} where " + " and ".join(preds) + " limit 50"


def nl_to_dsl_model(
    nl: str, model_fn: Optional[Callable[[str], str]]
) -> Optional[str]:
    """
    Ask the model to translate NL→DSL. Returns a DSL string if the model
    produces something parseable, otherwise None. Never raises.

    The model at 27B on 4GB VRAM is flaky here (journal risk #M8). The
    caller MUST try the heuristic as a second line of defense — this
    function's job is just "if the model gets it right, use it."
    """
    if not model_fn:
        return None

    prompt = (
        "Translate this natural-language question into LeanAI MemoryForge DSL.\n"
        "Output ONLY the DSL string, nothing else. No code fences, no explanation.\n\n"
        "DSL grammar:\n"
        "  <entity> [where <field> <op> <value> [and ...]] [limit N]\n"
        "  entity: symbols | findings | events\n"
        "  op: = != > < >= <= ~\n"
        "  symbols fields: name kind file line complexity lines\n"
        "  findings fields: kind category severity confidence file line since\n"
        "  events fields: kind source description since\n"
        "  severity values: CRITICAL HIGH MEDIUM LOW INFO (ordered)\n"
        "  since accepts: 7d 24h ISO-date\n\n"
        "Examples:\n"
        "  Q: show all critical SQL injection findings\n"
        "  A: findings where severity = CRITICAL and category = sql_injection\n\n"
        "  Q: which functions have high complexity\n"
        "  A: symbols where kind = function and complexity > 10\n\n"
        "  Q: what did sentinel find in the last day\n"
        "  A: events where source = sentinel and since = 1d\n\n"
        f"Q: {nl}\n"
        "A:"
    )
    try:
        raw = model_fn(prompt)
    except Exception:
        return None
    if not raw:
        return None

    # Strip code fences — models often wrap output in ``` blocks.
    # Handle three shapes:
    #   "```\n<dsl>\n```"              (fenced block on separate lines)
    #   "```dsl\n<dsl>\n```"           (fenced with lang hint)
    #   "<dsl>"                        (bare)
    text = raw.strip()
    if text.startswith("```"):
        # Drop the opening fence line (everything up to the first newline)
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        # Drop a closing fence if present
        close = text.rfind("```")
        if close != -1:
            text = text[:close]
        text = text.strip()

    # Take the first non-empty line
    first_line = ""
    for line in text.split("\n"):
        s = line.strip().strip("`").strip()
        if s:
            first_line = s
            break
    if not first_line:
        return None
    if first_line.lower().startswith("a:"):
        first_line = first_line[2:].strip()

    # Validate by attempting to parse
    try:
        parse_dsl(first_line)
        return first_line
    except DSLParseError:
        return None


# ═══════════════════════════════════════════════════════════════════
# MemoryForge engine
# ═══════════════════════════════════════════════════════════════════

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    filepath TEXT NOT NULL,
    line INTEGER NOT NULL DEFAULT 0,
    signature TEXT NOT NULL DEFAULT '',
    complexity INTEGER NOT NULL DEFAULT 0,
    lines INTEGER NOT NULL DEFAULT 0,
    last_sync REAL NOT NULL,
    UNIQUE(name, filepath)
);
CREATE INDEX IF NOT EXISTS idx_sym_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_sym_file ON symbols(filepath);
CREATE INDEX IF NOT EXISTS idx_sym_kind ON symbols(kind);

CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    filepath TEXT NOT NULL,
    line INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    fingerprint TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    last_sync REAL NOT NULL,
    json_blob TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_fnd_kind ON findings(kind);
CREATE INDEX IF NOT EXISTS idx_fnd_sev ON findings(severity);
CREATE INDEX IF NOT EXISTS idx_fnd_file ON findings(filepath);
CREATE INDEX IF NOT EXISTS idx_fnd_cat ON findings(category);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    source_tool TEXT NOT NULL,
    timestamp REAL NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    symbol_id INTEGER,
    finding_id INTEGER,
    fingerprint TEXT NOT NULL DEFAULT '',
    UNIQUE(fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_evt_time ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_evt_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_evt_src ON events(source_tool);

CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src_kind TEXT NOT NULL,
    src_id INTEGER NOT NULL,
    dst_kind TEXT NOT NULL,
    dst_id INTEGER NOT NULL,
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    last_sync REAL NOT NULL,
    UNIQUE(src_kind, src_id, dst_kind, dst_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_rel_src ON relations(src_kind, src_id);
CREATE INDEX IF NOT EXISTS idx_rel_dst ON relations(dst_kind, dst_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relations(relation);

CREATE TABLE IF NOT EXISTS sync_state (
    source TEXT PRIMARY KEY,
    last_sync REAL NOT NULL,
    fingerprints TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _leanai_home() -> str:
    return os.environ.get(
        "LEANAI_HOME",
        os.path.join(str(Path.home()), ".leanai"),
    )


class MemoryForge:
    """
    The M8 MemoryForge engine. See module docstring for design rules.

    Thread-safety: the SQLite connection is created per-method using
    check_same_thread=True (default), so each method is safe to call
    from the main thread. The background daemon threads in main.py
    do NOT call into MemoryForge; all access is from the command loop.
    """

    def __init__(
        self,
        project_path: str = ".",
        db_path: Optional[str] = None,
        brain: Optional[Any] = None,
        model_fn: Optional[Callable[[str], str]] = None,
        vuln_dir: Optional[str] = None,
        chain_dir: Optional[str] = None,
    ):
        self.project_path = os.path.abspath(project_path)
        home = _leanai_home()
        self.db_dir = os.path.join(home, "memory_forge")
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = db_path or os.path.join(self.db_dir, "graph.db")
        self.vuln_dir = vuln_dir or os.path.join(home, "vulns")
        self.chain_dir = chain_dir or os.path.join(home, "chains")

        self._brain = brain
        self._model_fn = model_fn

        self._ensure_schema()

    # ── Wiring ─────────────────────────────────────────────────────

    def set_brain(self, brain: Any) -> None:
        """Wire in a ProjectBrain for symbol ingestion."""
        self._brain = brain

    def set_model_fn(self, model_fn: Optional[Callable[[str], str]]) -> None:
        """Wire in a model function for NL→DSL translation."""
        self._model_fn = model_fn

    # ── Schema management ─────────────────────────────────────────

    def _ensure_schema(self) -> None:
        # Minimal schema setup. No auto-migration runs here — migration code
        # was removed in M8.1.1 after a Windows console interaction issue
        # was traced to this startup path. Users upgrading from v1 can
        # rebuild the graph manually with /memory reset + /memory sync.
        with self._conn() as c:
            c.executescript(_SCHEMA_SQL)
            row = c.execute(
                "SELECT value FROM schema_meta WHERE key = 'version'"
            ).fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO schema_meta(key, value) VALUES ('version', ?)",
                    (SCHEMA_VERSION,),
                )
            # If an older version row exists, leave it alone. The improved
            # matcher in this file will still produce correct edges for any
            # NEW findings synced from this point forward. Old cached bad
            # edges persist until the user runs /memory reset.

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ── Sync: ingest all known sources ────────────────────────────

    def sync(self, verbose: bool = False) -> SyncStats:
        """
        Incremental ingestion from all sources. Safe to run repeatedly —
        the UNIQUE constraints + fingerprint matching make this idempotent.
        """
        start = time.time()
        stats = SyncStats()
        now = time.time()

        # Path normalization pass. Older syncs may have stored symbols and
        # findings with backslash paths (from earlier versions of this
        # file). Rewrite them to forward slash in-place so matching works.
        # This is a no-op after the first sync post-upgrade.
        try:
            self._normalize_paths_if_needed()
        except Exception as e:
            stats.errors.append(f"path-normalize: {e}")

        # Symbols from brain (optional — brain might not be scanned yet)
        if self._brain is not None:
            try:
                self._sync_symbols(stats, now, verbose=verbose)
            except Exception as e:
                stats.errors.append(f"symbol sync: {e}")

        # Relink orphaned findings. A finding is "orphaned" if it has no
        # found_in edge to any symbol — usually because it was ingested by
        # an older version of this code (before the matcher was correct)
        # and the fingerprint-match fast path in vuln-sync skips it every
        # time. Run the matcher explicitly for these findings. This is
        # idempotent: subsequent syncs find no orphans and do nothing.
        try:
            self._relink_orphan_findings(stats, now)
        except Exception as e:
            stats.errors.append(f"relink: {e}")

        # Sentinel findings
        try:
            self._sync_vuln_findings(stats, now, verbose=verbose)
        except Exception as e:
            stats.errors.append(f"vuln sync: {e}")

        # ChainBreaker chains
        try:
            self._sync_chain_findings(stats, now, verbose=verbose)
        except Exception as e:
            stats.errors.append(f"chain sync: {e}")

        # Record the sync itself as an event
        try:
            with self._conn() as c:
                fp = hashlib.md5(f"sync:{now}".encode()).hexdigest()[:12]
                c.execute(
                    "INSERT OR IGNORE INTO events"
                    "(kind, source_tool, timestamp, description, fingerprint) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("sync", "memory_forge", now,
                     f"+{stats.symbols_added}s/+{stats.findings_added}f", fp),
                )
                # Update sync_state
                c.execute(
                    "INSERT OR REPLACE INTO sync_state(source, last_sync, fingerprints) "
                    "VALUES (?, ?, ?)",
                    ("memory_forge", now, "[]"),
                )
        except Exception as e:
            stats.errors.append(f"sync-event: {e}")

        stats.time_ms = int((time.time() - start) * 1000)
        return stats

    def _normalize_paths_if_needed(self) -> None:
        """One-time rewrite of backslash paths to forward slash in existing
        rows. Runs at the start of every sync; no-op after first run since
        the UPDATE matches zero rows once everything is normalized.

        This migrates graphs that were built by earlier versions of this
        file, which stored Windows paths as-is with backslashes. Without
        this, a post-upgrade sync would create NEW forward-slash symbols
        next to the old backslash ones, and matching would still fail.
        """
        with self._conn() as c:
            # Symbols: filepath column AND name column (qnames embed the path)
            c.execute(
                "UPDATE symbols SET filepath = REPLACE(filepath, '\\', '/') "
                "WHERE filepath LIKE '%\\%'"
            )
            c.execute(
                "UPDATE symbols SET name = REPLACE(name, '\\', '/') "
                "WHERE name LIKE '%\\%'"
            )
            # Findings: filepath column
            c.execute(
                "UPDATE findings SET filepath = REPLACE(filepath, '\\', '/') "
                "WHERE filepath LIKE '%\\%'"
            )

    def _sync_symbols(self, stats: SyncStats, now: float, verbose: bool) -> None:
        """Ingest function/class/method/file symbols from ProjectBrain."""
        brain = self._brain
        analyses = getattr(brain, "_file_analyses", None)
        if not analyses:
            return

        with self._conn() as c:
            # Existing symbols for this project (keyed on name + filepath)
            existing = {
                (r["name"], r["filepath"]): r["id"]
                for r in c.execute("SELECT id, name, filepath FROM symbols")
            }

            for rel_path_raw, analysis in analyses.items():
                # Normalize path to forward slashes. This is THE fix for
                # finding→symbol matching on Windows: if every path in the
                # graph uses '/', and findings are also normalized before
                # matching, there is no ambiguity. No four-strategy matcher,
                # no path-variant enumeration, just equality.
                rel_path = rel_path_raw.replace("\\", "/")
                # File-level symbol
                file_key = (rel_path, rel_path)
                file_lines = getattr(analysis, "total_lines", 0)
                if file_key in existing:
                    c.execute(
                        "UPDATE symbols SET lines=?, last_sync=? WHERE id=?",
                        (file_lines, now, existing[file_key]),
                    )
                    stats.symbols_updated += 1
                    file_sym_id = existing[file_key]
                else:
                    cur = c.execute(
                        "INSERT INTO symbols"
                        "(name, kind, filepath, line, signature, complexity, lines, last_sync) "
                        "VALUES (?, 'file', ?, 0, '', 0, ?, ?)",
                        (rel_path, rel_path, file_lines, now),
                    )
                    file_sym_id = cur.lastrowid
                    stats.symbols_added += 1
                    existing[file_key] = file_sym_id
                    self._add_event(
                        c, "discovery", "brain", now,
                        f"File discovered: {rel_path}",
                        symbol_id=file_sym_id, fingerprint=f"disc:file:{rel_path}",
                    )

                # Functions and methods. The brain's ClassInfo.methods is
                # just a list of names, not method objects — all the actual
                # callable info lives on FunctionInfo, with is_method=True
                # and class_name set for methods. We detect the kind here
                # and build the qname accordingly, so a method like
                # ProjectBrain.__init__ gets stored as
                # "brain/project_brain.py::ProjectBrain.__init__" rather
                # than the bare "brain/project_brain.py::__init__".
                for fn in getattr(analysis, "functions", []) or []:
                    fn_name = getattr(fn, "name", None)
                    if not fn_name:
                        continue
                    is_method = bool(getattr(fn, "is_method", False))
                    cls_name = getattr(fn, "class_name", None) or ""
                    if is_method and cls_name:
                        qname = f"{rel_path}::{cls_name}.{fn_name}"
                        kind = "method"
                    else:
                        qname = f"{rel_path}::{fn_name}"
                        kind = "function"
                    key = (qname, rel_path)
                    sig = getattr(fn, "signature", "") or ""
                    cplx = getattr(fn, "complexity", 0) or 0
                    # brain/analyzer uses line_start; older code may use line
                    line = (getattr(fn, "line_start", 0)
                            or getattr(fn, "line", 0) or 0)
                    if key in existing:
                        c.execute(
                            "UPDATE symbols SET kind=?, signature=?, complexity=?, "
                            "line=?, last_sync=? WHERE id=?",
                            (kind, sig, cplx, line, now, existing[key]),
                        )
                        stats.symbols_updated += 1
                        sid = existing[key]
                    else:
                        cur = c.execute(
                            "INSERT INTO symbols"
                            "(name, kind, filepath, line, signature, complexity, lines, last_sync) "
                            "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                            (qname, kind, rel_path, line, sig, cplx, now),
                        )
                        sid = cur.lastrowid
                        stats.symbols_added += 1
                        existing[key] = sid
                    self._add_relation(
                        c, "symbol", file_sym_id, "symbol", sid, "contains", now, stats
                    )

                # Classes
                for cls in getattr(analysis, "classes", []) or []:
                    cls_name = getattr(cls, "name", None)
                    if not cls_name:
                        continue
                    qname = f"{rel_path}::{cls_name}"
                    key = (qname, rel_path)
                    line = (getattr(cls, "line_start", 0)
                            or getattr(cls, "line", 0) or 0)
                    if key in existing:
                        c.execute(
                            "UPDATE symbols SET line=?, last_sync=? WHERE id=?",
                            (line, now, existing[key]),
                        )
                        stats.symbols_updated += 1
                        cid = existing[key]
                    else:
                        cur = c.execute(
                            "INSERT INTO symbols"
                            "(name, kind, filepath, line, signature, complexity, lines, last_sync) "
                            "VALUES (?, 'class', ?, ?, '', 0, 0, ?)",
                            (qname, rel_path, line, now),
                        )
                        cid = cur.lastrowid
                        stats.symbols_added += 1
                        existing[key] = cid
                    self._add_relation(
                        c, "symbol", file_sym_id, "symbol", cid, "contains", now, stats
                    )
                    # Note: the class → method "contains" relations get
                    # built in a second pass below, once all methods and
                    # classes in this file have been inserted.

                # Second pass: class → method containment edges. Runs after
                # all symbols in this file are inserted, so we can match
                # methods to their parent class by qname prefix.
                class_syms = {
                    getattr(cls, "name", ""): existing.get(
                        (f"{rel_path}::{getattr(cls, 'name', '')}", rel_path)
                    )
                    for cls in (getattr(analysis, "classes", []) or [])
                    if getattr(cls, "name", None)
                }
                for fn in getattr(analysis, "functions", []) or []:
                    if not getattr(fn, "is_method", False):
                        continue
                    cname = getattr(fn, "class_name", None)
                    fname = getattr(fn, "name", None)
                    if not (cname and fname):
                        continue
                    cid = class_syms.get(cname)
                    mid = existing.get(
                        (f"{rel_path}::{cname}.{fname}", rel_path)
                    )
                    if cid is not None and mid is not None:
                        self._add_relation(
                            c, "symbol", cid, "symbol", mid,
                            "contains", now, stats,
                        )

    def _relink_orphan_findings(self, stats: SyncStats, now: float) -> None:
        """Find findings that have no found_in edge and link them with the
        current matcher.

        A finding is orphaned if:
          - It exists in the findings table
          - No relation row has (src_kind='finding', src_id=this finding, relation='found_in')

        This happens on graphs that were populated by old matcher versions.
        Running once per sync is cheap — after the first pass, the query
        returns zero rows, so this is effectively a no-op.
        """
        with self._conn() as c:
            orphans = list(c.execute(
                "SELECT f.id, f.kind, f.finding_id, f.json_blob FROM findings f "
                "WHERE NOT EXISTS ("
                "  SELECT 1 FROM relations r "
                "  WHERE r.src_kind = 'finding' AND r.src_id = f.id "
                "  AND r.relation = 'found_in'"
                ")"
            ))
            if not orphans:
                return

            for row in orphans:
                try:
                    data = json.loads(row["json_blob"])
                except Exception:
                    continue
                # Normalize paths in the stored blob — older versions may
                # have written backslash paths into json_blob even after
                # rewriting the filepath column.
                if "filepath" in data:
                    data["filepath"] = str(data["filepath"]).replace("\\", "/")
                for step in data.get("steps", []) or []:
                    if "filepath" in step:
                        step["filepath"] = str(step["filepath"]).replace("\\", "/")

                if row["kind"] == "vuln":
                    self._link_finding_to_symbols(c, row["id"], data, now, stats)
                elif row["kind"] == "chain":
                    self._link_chain_to_symbols(c, row["id"], data, now, stats)

    def _sync_vuln_findings(self, stats: SyncStats, now: float, verbose: bool) -> None:
        """Ingest Sentinel VULN-*.json files."""
        if not os.path.isdir(self.vuln_dir):
            return
        pattern = os.path.join(self.vuln_dir, "VULN-*.json")
        for fpath in glob.glob(pattern):
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                stats.errors.append(f"load {fpath}: {e}")
                continue

            vid = data.get("vuln_id") or os.path.basename(fpath).replace(".json", "")
            fingerprint = data.get("fingerprint", "")
            created_at = float(data.get("timestamp", now))

            # Normalize filepath to forward slashes so it matches the
            # normalized paths used in symbols. This is the key to the
            # matcher working on Windows.
            if "filepath" in data:
                data["filepath"] = str(data["filepath"]).replace("\\", "/")

            with self._conn() as c:
                existing = c.execute(
                    "SELECT id, fingerprint FROM findings WHERE finding_id = ?",
                    (vid,),
                ).fetchone()

                if existing and existing["fingerprint"] == fingerprint:
                    # No change — fast path
                    c.execute(
                        "UPDATE findings SET last_sync = ? WHERE id = ?",
                        (now, existing["id"]),
                    )
                    stats.skipped_stale += 1
                    continue

                if existing:
                    c.execute(
                        "UPDATE findings SET "
                        "category=?, severity=?, confidence=?, filepath=?, line=?, "
                        "description=?, fingerprint=?, created_at=?, last_sync=?, json_blob=? "
                        "WHERE id=?",
                        (
                            data.get("vuln_class", "unknown"),
                            str(data.get("severity", "INFO")).upper(),
                            float(data.get("confidence", 0.0)),
                            data.get("filepath", ""),
                            int(data.get("line", 0) or 0),
                            data.get("description", ""),
                            fingerprint,
                            created_at,
                            now,
                            json.dumps(data),
                            existing["id"],
                        ),
                    )
                    stats.findings_updated += 1
                    fid = existing["id"]
                    ev_kind = "fix"    # finding changed = likely updated or reopened
                    ev_desc = f"{vid} updated: {data.get('vuln_class', '')}"
                else:
                    cur = c.execute(
                        "INSERT INTO findings"
                        "(finding_id, kind, category, severity, confidence, filepath, line, "
                        " description, fingerprint, created_at, last_sync, json_blob) "
                        "VALUES (?, 'vuln', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            vid,
                            data.get("vuln_class", "unknown"),
                            str(data.get("severity", "INFO")).upper(),
                            float(data.get("confidence", 0.0)),
                            data.get("filepath", ""),
                            int(data.get("line", 0) or 0),
                            data.get("description", ""),
                            fingerprint,
                            created_at,
                            now,
                            json.dumps(data),
                        ),
                    )
                    fid = cur.lastrowid
                    stats.findings_added += 1
                    ev_kind = "discovery"
                    ev_desc = f"{vid} found: {data.get('vuln_class', '')} " \
                              f"in {data.get('filepath', '')}"

                # Link finding → symbol (found_in)
                self._link_finding_to_symbols(c, fid, data, now, stats)

                # Record the event
                self._add_event(
                    c, ev_kind, "sentinel", created_at, ev_desc,
                    finding_id=fid,
                    fingerprint=f"{ev_kind}:{vid}:{fingerprint}",
                )
                stats.events_added += 1

    def _sync_chain_findings(self, stats: SyncStats, now: float, verbose: bool) -> None:
        """Ingest ChainBreaker CHAIN-*.json files."""
        if not os.path.isdir(self.chain_dir):
            return
        pattern = os.path.join(self.chain_dir, "CHAIN-*.json")
        for fpath in glob.glob(pattern):
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                stats.errors.append(f"load {fpath}: {e}")
                continue

            cid = data.get("chain_id") or os.path.basename(fpath).replace(".json", "")
            fingerprint = data.get("fingerprint", "")
            created_at = float(data.get("timestamp", now))

            # Normalize every filepath in the chain to forward slashes
            # (step filepaths too) so matching against normalized symbols
            # works reliably on Windows.
            for step in data.get("steps", []) or []:
                if "filepath" in step:
                    step["filepath"] = str(step["filepath"]).replace("\\", "/")

            # Derive primary location from first step (after normalization)
            steps = data.get("steps", []) or []
            primary_file = steps[0].get("filepath", "") if steps else ""
            primary_line = int(steps[0].get("line", 0)) if steps else 0

            with self._conn() as c:
                existing = c.execute(
                    "SELECT id, fingerprint FROM findings WHERE finding_id = ?",
                    (cid,),
                ).fetchone()

                if existing and existing["fingerprint"] == fingerprint:
                    c.execute(
                        "UPDATE findings SET last_sync = ? WHERE id = ?",
                        (now, existing["id"]),
                    )
                    stats.skipped_stale += 1
                    continue

                if existing:
                    c.execute(
                        "UPDATE findings SET "
                        "category=?, severity=?, confidence=?, filepath=?, line=?, "
                        "description=?, fingerprint=?, created_at=?, last_sync=?, json_blob=? "
                        "WHERE id=?",
                        (
                            data.get("capability", "unknown"),
                            str(data.get("severity", "INFO")).upper(),
                            float(data.get("confidence", 0.0)),
                            primary_file,
                            primary_line,
                            data.get("impact_summary") or data.get("narrative", ""),
                            fingerprint,
                            created_at,
                            now,
                            json.dumps(data),
                            existing["id"],
                        ),
                    )
                    stats.findings_updated += 1
                    fid = existing["id"]
                    ev_kind = "fix"
                    ev_desc = f"{cid} chain updated"
                else:
                    cur = c.execute(
                        "INSERT INTO findings"
                        "(finding_id, kind, category, severity, confidence, filepath, line, "
                        " description, fingerprint, created_at, last_sync, json_blob) "
                        "VALUES (?, 'chain', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            cid,
                            data.get("capability", "unknown"),
                            str(data.get("severity", "INFO")).upper(),
                            float(data.get("confidence", 0.0)),
                            primary_file,
                            primary_line,
                            data.get("impact_summary") or data.get("narrative", ""),
                            fingerprint,
                            created_at,
                            now,
                            json.dumps(data),
                        ),
                    )
                    fid = cur.lastrowid
                    stats.findings_added += 1
                    ev_kind = "discovery"
                    ev_desc = f"{cid} attack chain: " \
                              f"{data.get('capability', '')} " \
                              f"via {len(steps)} step(s)"

                # Link to entry vuln + each step's function
                self._link_chain_to_symbols(c, fid, data, now, stats)

                self._add_event(
                    c, ev_kind, "chainbreaker", created_at, ev_desc,
                    finding_id=fid,
                    fingerprint=f"{ev_kind}:{cid}:{fingerprint}",
                )
                stats.events_added += 1

    def _link_chain_to_symbols(
        self, c: sqlite3.Connection, fid: int, data: dict, now: float,
        stats: SyncStats,
    ) -> None:
        """Link a chain finding to its entry vuln (depends_on) and each of
        its step functions (found_in for entry, affects for subsequent)."""
        # depends_on → entry vuln
        entry_vuln = data.get("entry_vuln_id")
        if entry_vuln:
            ev_row = c.execute(
                "SELECT id FROM findings WHERE finding_id = ?",
                (entry_vuln,),
            ).fetchone()
            if ev_row:
                self._add_relation(
                    c, "finding", fid, "finding", ev_row["id"],
                    "depends_on", now, stats,
                )

        # found_in / affects → step functions
        for step in data.get("steps", []) or []:
            fn_name = step.get("function_name", "")
            fpath = step.get("filepath", "")
            if not fn_name or not fpath:
                continue
            line = int(step.get("line", 0) or 0)
            sym_id = self._find_symbol(c, fpath, fn_name, line)
            if sym_id is not None:
                rel = "found_in" if step.get("stage") == "entry" else "affects"
                self._add_relation(
                    c, "finding", fid, "symbol", sym_id,
                    rel, now, stats,
                )

    def _link_finding_to_symbols(
        self, c: sqlite3.Connection, fid: int, data: dict, now: float,
        stats: SyncStats,
    ) -> None:
        """Link a vuln finding to the symbols it touches (found_in + affects).

        Matching is best-effort and uses four progressively-looser strategies.
        Accepts the first that produces a match, so that when any strategy
        works, we record the edge. Previously a single compound query missed
        ~97% of findings in production because path separators and qualifier
        conventions don't round-trip cleanly across OSes and across Sentinel
        versions. The production symptom was: 40 findings ingested, only
        1 found_in edge recorded. With this matcher, the same data should
        produce ~40 edges.
        """
        fpath = data.get("filepath", "")
        fn_name = data.get("function_name", "")
        line = int(data.get("line", 0) or 0)

        if fpath and fn_name and fn_name != "<module>":
            sym_id = self._find_symbol(c, fpath, fn_name, line)
            if sym_id is not None:
                self._add_relation(
                    c, "finding", fid, "symbol", sym_id,
                    "found_in", now, stats,
                )
        elif fpath and fn_name == "<module>":
            # Module-level finding — link to the file symbol instead.
            file_id = self._find_file_symbol(c, fpath)
            if file_id is not None:
                self._add_relation(
                    c, "finding", fid, "symbol", file_id,
                    "found_in", now, stats,
                )

        # Taint path → 'affects' edges on intermediate functions.
        # taint_path contains bare function names (no class prefix, no file
        # prefix). Skip the entry function (already 'found_in') and dedupe
        # so one taint function with many matches doesn't create 10 edges.
        taint_path = data.get("taint_path") or []
        seen: set = set()
        for tfn in taint_path:
            if not tfn or tfn == fn_name or tfn in seen:
                continue
            seen.add(tfn)
            sym_id = self._find_symbol_by_bare_name(c, tfn, prefer_filepath=fpath)
            if sym_id is not None:
                self._add_relation(
                    c, "finding", fid, "symbol", sym_id,
                    "affects", now, stats,
                )

    # --- Symbol-matching helpers (Windows-aware, multi-strategy) ---

    @staticmethod
    def _norm(p: str) -> str:
        """Normalize a path to forward slashes. After M8.1.2 every path
        in the graph uses forward slash, so matching is just equality."""
        return str(p).replace("\\", "/") if p else ""

    def _find_file_symbol(
        self, c: sqlite3.Connection, fpath: str,
    ) -> Optional[int]:
        """Find the 'file' kind symbol for a given filepath."""
        row = c.execute(
            "SELECT id FROM symbols WHERE kind = 'file' AND filepath = ? LIMIT 1",
            (self._norm(fpath),),
        ).fetchone()
        return row["id"] if row else None

    def _find_symbol(
        self, c: sqlite3.Connection, fpath: str, fn_name: str, line: int = 0,
    ) -> Optional[int]:
        """Simple symbol lookup. Because every path in the graph (both
        sides of the match) is forward-slash-normalized, this is just
        three equality/suffix queries in order of specificity:

          1. Exact qname: '<normalized-filepath>::<fn_name>'
          2. Same file, name ends in <fn_name> after '::' or '.'
          3. Anywhere in project, name ends in <fn_name>
             (fallback for findings whose filepath doesn't exactly match
             any symbol's filepath — e.g. findings from an old scan)

        Prefers method > function > class > file when ties happen.
        """
        if not fn_name:
            return None
        fpath_norm = self._norm(fpath)

        # Strategy 1: exact qname match
        qname = f"{fpath_norm}::{fn_name}"
        row = c.execute(
            "SELECT id FROM symbols WHERE name = ? "
            "ORDER BY CASE kind WHEN 'method' THEN 1 WHEN 'function' THEN 2 "
            "WHEN 'class' THEN 3 ELSE 4 END LIMIT 1",
            (qname,),
        ).fetchone()
        if row:
            return row["id"]

        # Strategy 2: same file, suffix match on fn_name
        row = c.execute(
            "SELECT id FROM symbols WHERE filepath = ? AND ("
            "  name = ? OR name LIKE ? OR name LIKE ?"
            ") ORDER BY CASE kind WHEN 'method' THEN 1 WHEN 'function' THEN 2 "
            "WHEN 'class' THEN 3 ELSE 4 END LIMIT 1",
            (fpath_norm, fn_name, f"%::{fn_name}", f"%.{fn_name}"),
        ).fetchone()
        if row:
            return row["id"]

        # Strategy 3: project-wide suffix match (ambiguous but useful)
        basename = os.path.basename(fpath_norm)
        rows = list(c.execute(
            "SELECT id, filepath FROM symbols "
            "WHERE (name = ? OR name LIKE ? OR name LIKE ?) "
            "AND kind IN ('function', 'method')",
            (fn_name, f"%::{fn_name}", f"%.{fn_name}"),
        ))
        if rows:
            # Prefer same-basename match if available
            if basename:
                for r in rows:
                    if os.path.basename(r["filepath"]) == basename:
                        return r["id"]
            return rows[0]["id"]

        return None

    def _find_symbol_by_bare_name(
        self, c: sqlite3.Connection, bare_name: str, prefer_filepath: str = "",
    ) -> Optional[int]:
        """Look up a symbol by a bare function/method name (no file, no
        class prefix). Used for taint_path entries."""
        rows = list(c.execute(
            "SELECT id, filepath FROM symbols "
            "WHERE (name = ? OR name LIKE ? OR name LIKE ?) "
            "AND kind IN ('function', 'method') LIMIT 20",
            (bare_name, f"%::{bare_name}", f"%.{bare_name}"),
        ))
        if not rows:
            return None
        if prefer_filepath:
            prefer = self._norm(prefer_filepath)
            for r in rows:
                if r["filepath"] == prefer:
                    return r["id"]
        return rows[0]["id"]

    def _add_event(
        self, c: sqlite3.Connection, kind: str, source_tool: str,
        timestamp: float, description: str,
        symbol_id: Optional[int] = None, finding_id: Optional[int] = None,
        fingerprint: str = "",
    ) -> Optional[int]:
        if not fingerprint:
            fingerprint = hashlib.md5(
                f"{kind}:{source_tool}:{timestamp}:{description}".encode()
            ).hexdigest()[:12]
        cur = c.execute(
            "INSERT OR IGNORE INTO events"
            "(kind, source_tool, timestamp, description, symbol_id, finding_id, fingerprint) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kind, source_tool, timestamp, description, symbol_id, finding_id, fingerprint),
        )
        return cur.lastrowid

    def _add_relation(
        self, c: sqlite3.Connection,
        src_kind: str, src_id: int,
        dst_kind: str, dst_id: int,
        relation: str, now: float,
        stats: SyncStats,
        weight: float = 1.0,
    ) -> None:
        if relation not in VALID_RELATIONS:
            return
        cur = c.execute(
            "INSERT OR IGNORE INTO relations"
            "(src_kind, src_id, dst_kind, dst_id, relation, weight, last_sync) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (src_kind, src_id, dst_kind, dst_id, relation, weight, now),
        )
        if cur.rowcount > 0:
            stats.relations_added += 1

    # ── Query API ──────────────────────────────────────────────────

    def query(
        self, dsl_or_nl: str, use_model: bool = True,
    ) -> Tuple[List[QueryResult], str]:
        """
        Run a query. Returns (results, dsl_used).

        The dsl_used string tells the caller what DSL actually ran —
        useful for showing the user "I interpreted your question as X."

        Routing:
          1. If dsl_or_nl parses as valid DSL, run it directly.
          2. Else, if use_model and we have a model_fn, try model translation.
          3. Else, try the NL heuristic.
          4. If all fail, raise DSLParseError.
        """
        raw = (dsl_or_nl or "").strip()
        if not raw:
            raise DSLParseError("Empty query")

        # Try raw parse first — if the user typed DSL, honour it.
        try:
            parsed = parse_dsl(raw)
            return self._run_parsed(parsed), raw
        except DSLParseError:
            pass

        # Try model translation
        if use_model and self._model_fn is not None:
            dsl = nl_to_dsl_model(raw, self._model_fn)
            if dsl:
                try:
                    parsed = parse_dsl(dsl)
                    return self._run_parsed(parsed), dsl
                except DSLParseError:
                    pass

        # Fall back to heuristic
        dsl = nl_to_dsl_heuristic(raw)
        if dsl:
            try:
                parsed = parse_dsl(dsl)
                return self._run_parsed(parsed), dsl
            except DSLParseError:
                pass

        raise DSLParseError(
            f"Could not interpret query: {raw!r}\n"
            f"Try raw DSL. Grammar: '<entity> [where <field> <op> <value> "
            f"[and ...]] [limit N]'\n"
            f"Entities: symbols | findings | events"
        )

    def _run_parsed(self, parsed: ParsedQuery) -> List[QueryResult]:
        where_sql, params = _compile_predicates(parsed.entity, parsed.predicates)
        if parsed.entity == "symbols":
            sql = (
                "SELECT id, name, kind, filepath, line, signature, complexity, lines, last_sync "
                f"FROM symbols {where_sql} ORDER BY filepath, line LIMIT ?"
            )
            entity_kind = "symbol"
        elif parsed.entity == "findings":
            sql = (
                "SELECT id, finding_id, kind, category, severity, confidence, filepath, "
                "line, description, fingerprint, created_at, last_sync "
                f"FROM findings {where_sql} "
                "ORDER BY "
                "(CASE severity WHEN 'CRITICAL' THEN 4 WHEN 'HIGH' THEN 3 "
                "WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 1 ELSE 0 END) DESC, "
                "created_at DESC LIMIT ?"
            )
            entity_kind = "finding"
        else:   # events
            sql = (
                "SELECT id, kind, source_tool, timestamp, description, symbol_id, finding_id "
                f"FROM events {where_sql} ORDER BY timestamp DESC LIMIT ?"
            )
            entity_kind = "event"
        params = list(params) + [parsed.limit]

        out: List[QueryResult] = []
        with self._conn() as c:
            for r in c.execute(sql, params):
                out.append(QueryResult(entity=entity_kind, data=dict(r)))
        return out

    # ── Facts: all info about a specific symbol ───────────────────

    def facts_for(self, symbol_query: str) -> Dict[str, Any]:
        """
        Return everything MemoryForge knows about a symbol:
          - matching symbol rows
          - findings found_in or affects it
          - events that touched it
          - relations (contains, depends_on) for context

        Matches on exact name, suffix (`SomeClass.method`), or bare name.
        """
        q = symbol_query.strip()
        out: Dict[str, Any] = {
            "query": q,
            "symbols": [],
            "findings": [],
            "events": [],
            "relations": [],
        }
        if not q:
            return out

        with self._conn() as c:
            # Multi-pattern match: accepts user queries in any of these shapes:
            #   'handle_request'                        (bare fn name)
            #   'HTTPServer.handle'                     (class.method)
            #   'core/server.py::handle_request'        (full qname, forward slash)
            #   'core\\server.py::handle_request'       (full qname, backslash)
            #   'core/server.py'                        (file name — returns file + members)
            #   'HTTPServer'                            (class name)
            patterns = [
                q,                  # exact full qname OR bare name
                f"%::{q}",          # matches qualified version of a bare name
                f"%.{q}",           # matches class.method pattern
                f"%{q}",            # last-resort substring-at-end match
            ]
            # Also try path-separator variants in case user typed it the
            # non-native way for their OS.
            if "/" in q or "\\" in q:
                fwd = q.replace("\\", "/")
                bwd = q.replace("/", "\\")
                patterns.extend([fwd, bwd])

            # Deduplicate while preserving order
            seen_p: set = set()
            uniq_patterns = [p for p in patterns if not (p in seen_p or seen_p.add(p))]

            where_clause = " OR ".join(["name = ?" if "%" not in p else "name LIKE ?"
                                        for p in uniq_patterns])
            sym_rows = list(c.execute(
                "SELECT id, name, kind, filepath, line, signature, complexity, "
                "lines, last_sync FROM symbols WHERE " + where_clause + " "
                "ORDER BY CASE kind WHEN 'method' THEN 1 WHEN 'function' THEN 2 "
                "WHEN 'class' THEN 3 WHEN 'file' THEN 4 ELSE 5 END, name "
                "LIMIT 20",
                uniq_patterns,
            ))
            out["symbols"] = [dict(r) for r in sym_rows]
            if not sym_rows:
                return out

            sym_ids = [r["id"] for r in sym_rows]
            placeholders = ",".join("?" * len(sym_ids))

            # Findings that mention this symbol (found_in or affects)
            fnd_rows = list(c.execute(
                f"SELECT DISTINCT f.id, f.finding_id, f.kind, f.category, f.severity, "
                f"f.confidence, f.filepath, f.line, f.description, f.fingerprint, "
                f"f.created_at, f.last_sync, r.relation "
                f"FROM findings f JOIN relations r "
                f"ON r.src_kind = 'finding' AND r.src_id = f.id "
                f"AND r.dst_kind = 'symbol' AND r.dst_id IN ({placeholders}) "
                f"ORDER BY (CASE f.severity WHEN 'CRITICAL' THEN 4 WHEN 'HIGH' THEN 3 "
                f"WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 1 ELSE 0 END) DESC",
                sym_ids,
            ))
            out["findings"] = [dict(r) for r in fnd_rows]

            # Events referencing this symbol directly
            ev_rows = list(c.execute(
                f"SELECT id, kind, source_tool, timestamp, description, symbol_id, finding_id "
                f"FROM events WHERE symbol_id IN ({placeholders}) "
                f"ORDER BY timestamp DESC LIMIT 30",
                sym_ids,
            ))
            out["events"] = [dict(r) for r in ev_rows]

            # Relations involving this symbol (both directions)
            rel_rows = list(c.execute(
                f"SELECT src_kind, src_id, dst_kind, dst_id, relation, weight "
                f"FROM relations "
                f"WHERE (src_kind = 'symbol' AND src_id IN ({placeholders})) "
                f"   OR (dst_kind = 'symbol' AND dst_id IN ({placeholders})) "
                f"LIMIT 100",
                sym_ids + sym_ids,
            ))
            out["relations"] = [dict(r) for r in rel_rows]

        return out

    # ── Timeline ──────────────────────────────────────────────────

    def timeline(self, limit: int = 20) -> List[EventRow]:
        """Return the most recent events, newest first."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, kind, source_tool, timestamp, description, symbol_id, finding_id "
                "FROM events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            EventRow(
                id=r["id"], kind=r["kind"], source_tool=r["source_tool"],
                timestamp=r["timestamp"], description=r["description"],
                symbol_id=r["symbol_id"], finding_id=r["finding_id"],
            )
            for r in rows
        ]

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Graph statistics for /memory stats."""
        with self._conn() as c:
            n_sym = c.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()["n"]
            n_fnd = c.execute("SELECT COUNT(*) AS n FROM findings").fetchone()["n"]
            n_evt = c.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
            n_rel = c.execute("SELECT COUNT(*) AS n FROM relations").fetchone()["n"]

            by_sev = {
                r["severity"]: r["n"]
                for r in c.execute(
                    "SELECT severity, COUNT(*) AS n FROM findings GROUP BY severity"
                )
            }
            by_kind = {
                r["kind"]: r["n"]
                for r in c.execute(
                    "SELECT kind, COUNT(*) AS n FROM symbols GROUP BY kind"
                )
            }
            by_rel = {
                r["relation"]: r["n"]
                for r in c.execute(
                    "SELECT relation, COUNT(*) AS n FROM relations GROUP BY relation"
                )
            }

            last_sync_row = c.execute(
                "SELECT MAX(timestamp) AS t FROM events WHERE kind = 'sync'"
            ).fetchone()
            last_sync = last_sync_row["t"] if last_sync_row and last_sync_row["t"] else 0.0

        db_size = 0
        try:
            db_size = os.path.getsize(self.db_path)
        except OSError:
            pass

        return {
            "symbols": n_sym,
            "findings": n_fnd,
            "events": n_evt,
            "relations": n_rel,
            "by_severity": by_sev,
            "by_symbol_kind": by_kind,
            "by_relation": by_rel,
            "last_sync": last_sync,
            "age_seconds": int(time.time() - last_sync) if last_sync else None,
            "db_path": self.db_path,
            "db_size_bytes": db_size,
            "schema_version": SCHEMA_VERSION,
        }

    # ── Maintenance ───────────────────────────────────────────────

    def reset(self) -> None:
        """Drop and recreate the whole graph. Destructive — user should
        be warned by the caller. Leaves sync_state empty."""
        with self._conn() as c:
            for tbl in ("relations", "events", "findings", "symbols",
                        "sync_state", "schema_meta"):
                c.execute(f"DROP TABLE IF EXISTS {tbl}")
        self._ensure_schema()


# ═══════════════════════════════════════════════════════════════════
# Formatting helpers — called from main.py's /memory handler
# ═══════════════════════════════════════════════════════════════════

def format_query_results(
    results: List[QueryResult], dsl_used: str, color: bool = True,
) -> str:
    """Pretty-print query results for the terminal."""
    if color:
        try:
            from core.terminal_ui import C
            DIM = C.DIM
            RESET = C.RESET
            BOLD = getattr(C, "BOLD", "\033[1m")
            RED = getattr(C, "RED", "\033[31m")
            YELLOW = getattr(C, "YELLOW", "\033[33m")
            CYAN = getattr(C, "CYAN", "\033[36m")
            GREEN = getattr(C, "GREEN", "\033[32m")
            MAG = getattr(C, "MAGENTA", "\033[35m")
        except Exception:
            DIM = RESET = BOLD = RED = YELLOW = CYAN = GREEN = MAG = ""
    else:
        DIM = RESET = BOLD = RED = YELLOW = CYAN = GREEN = MAG = ""

    lines: List[str] = []
    lines.append(f"{DIM}DSL: {dsl_used}{RESET}")
    lines.append(f"{DIM}Results: {len(results)}{RESET}")
    lines.append("")

    if not results:
        lines.append(f"  {DIM}(no matches){RESET}")
        return "\n".join(lines)

    def sev_color(sv: str) -> str:
        return {"CRITICAL": RED, "HIGH": RED, "MEDIUM": YELLOW,
                "LOW": CYAN, "INFO": DIM}.get(sv, "")

    for r in results:
        d = r.data
        if r.entity == "finding":
            sv = d.get("severity", "")
            lines.append(
                f"  {sev_color(sv)}{BOLD}{d.get('finding_id', '')}{RESET} "
                f"{sev_color(sv)}{sv}{RESET}  "
                f"{CYAN}{d.get('category', '')}{RESET}  "
                f"conf={d.get('confidence', 0):.2f}"
            )
            fp = d.get("filepath", "")
            ln = d.get("line", 0)
            if fp:
                lines.append(f"    {DIM}{fp}:{ln}{RESET}")
            desc = d.get("description", "")
            if desc:
                desc = desc.replace("\n", " ")
                if len(desc) > 120:
                    desc = desc[:120] + "…"
                lines.append(f"    {desc}")
        elif r.entity == "symbol":
            lines.append(
                f"  {GREEN}{d.get('kind', ''):<8}{RESET} "
                f"{BOLD}{d.get('name', '')}{RESET}  "
                f"{DIM}{d.get('filepath', '')}:{d.get('line', 0)}{RESET}"
            )
            cplx = d.get("complexity", 0)
            ln_ct = d.get("lines", 0)
            if cplx or ln_ct:
                extras = []
                if cplx:
                    extras.append(f"complexity={cplx}")
                if ln_ct:
                    extras.append(f"lines={ln_ct}")
                lines.append(f"    {DIM}" + "  ".join(extras) + RESET)
        else:   # event
            ts = d.get("timestamp", 0)
            import datetime as _dt
            dt_str = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") \
                if ts else "-"
            lines.append(
                f"  {DIM}{dt_str}{RESET}  "
                f"{MAG}{d.get('kind', ''):<12}{RESET}  "
                f"{CYAN}{d.get('source_tool', '')}{RESET}  "
                f"{d.get('description', '')}"
            )

    return "\n".join(lines)


def format_facts(facts: Dict[str, Any], color: bool = True) -> str:
    """Pretty-print /memory facts output."""
    if color:
        try:
            from core.terminal_ui import C
            DIM = C.DIM
            RESET = C.RESET
            BOLD = getattr(C, "BOLD", "\033[1m")
            RED = getattr(C, "RED", "\033[31m")
            YELLOW = getattr(C, "YELLOW", "\033[33m")
            CYAN = getattr(C, "CYAN", "\033[36m")
            GREEN = getattr(C, "GREEN", "\033[32m")
        except Exception:
            DIM = RESET = BOLD = RED = YELLOW = CYAN = GREEN = ""
    else:
        DIM = RESET = BOLD = RED = YELLOW = CYAN = GREEN = ""

    lines: List[str] = []
    lines.append(f"{BOLD}Facts for: {facts['query']}{RESET}")
    lines.append("")

    syms = facts.get("symbols", [])
    if not syms:
        lines.append(f"  {DIM}(no matching symbols — try /memory sync){RESET}")
        return "\n".join(lines)

    lines.append(f"{CYAN}Symbols ({len(syms)}){RESET}")
    for s in syms[:10]:
        lines.append(
            f"  {GREEN}{s.get('kind', ''):<8}{RESET} "
            f"{BOLD}{s.get('name', '')}{RESET}  "
            f"{DIM}{s.get('filepath', '')}:{s.get('line', 0)}{RESET}"
        )
        extras = []
        if s.get("complexity"):
            extras.append(f"complexity={s['complexity']}")
        if s.get("signature"):
            sig = s["signature"]
            if len(sig) > 80:
                sig = sig[:80] + "…"
            extras.append(f"sig={sig}")
        if extras:
            lines.append(f"    {DIM}" + "  ".join(extras) + RESET)
    lines.append("")

    fnds = facts.get("findings", [])
    lines.append(f"{CYAN}Findings ({len(fnds)}){RESET}")
    if not fnds:
        lines.append(f"  {DIM}(none){RESET}")
    else:
        for f in fnds[:15]:
            sv = f.get("severity", "")
            sev_color = {"CRITICAL": RED, "HIGH": RED, "MEDIUM": YELLOW,
                         "LOW": CYAN, "INFO": DIM}.get(sv, "")
            rel = f.get("relation", "")
            lines.append(
                f"  {sev_color}{f.get('finding_id', '')}{RESET} "
                f"{sev_color}{sv}{RESET}  "
                f"{f.get('category', '')}  "
                f"{DIM}({rel}){RESET}"
            )
    lines.append("")

    evs = facts.get("events", [])
    lines.append(f"{CYAN}Events ({len(evs)}){RESET}")
    if not evs:
        lines.append(f"  {DIM}(none){RESET}")
    else:
        import datetime as _dt
        for e in evs[:10]:
            ts = e.get("timestamp", 0)
            dt_str = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") \
                if ts else "-"
            lines.append(
                f"  {DIM}{dt_str}{RESET}  "
                f"{e.get('kind', ''):<12}  "
                f"{DIM}{e.get('source_tool', '')}{RESET}  "
                f"{e.get('description', '')}"
            )

    rels = facts.get("relations", [])
    if rels:
        lines.append("")
        lines.append(f"{CYAN}Relations ({len(rels)}){RESET}")
        by_rel: Dict[str, int] = {}
        for r in rels:
            by_rel[r["relation"]] = by_rel.get(r["relation"], 0) + 1
        for k, v in sorted(by_rel.items(), key=lambda x: -x[1]):
            lines.append(f"  {k:<14} {v}")

    return "\n".join(lines)


def format_timeline(events: List[EventRow], color: bool = True) -> str:
    """Pretty-print /memory timeline output."""
    if color:
        try:
            from core.terminal_ui import C
            DIM = C.DIM
            RESET = C.RESET
            BOLD = getattr(C, "BOLD", "\033[1m")
            CYAN = getattr(C, "CYAN", "\033[36m")
            MAG = getattr(C, "MAGENTA", "\033[35m")
        except Exception:
            DIM = RESET = BOLD = CYAN = MAG = ""
    else:
        DIM = RESET = BOLD = CYAN = MAG = ""

    if not events:
        return f"  {DIM}(no events yet — run /memory sync){RESET}"

    lines: List[str] = []
    lines.append(f"{BOLD}Timeline — {len(events)} event(s){RESET}")
    lines.append("")
    import datetime as _dt
    for e in events:
        dt_str = _dt.datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d %H:%M:%S") \
            if e.timestamp else "-"
        lines.append(
            f"  {DIM}{dt_str}{RESET}  "
            f"{MAG}{e.kind:<12}{RESET}  "
            f"{CYAN}{e.source_tool:<14}{RESET}  "
            f"{e.description}"
        )
    return "\n".join(lines)


def format_stats(stats: Dict[str, Any], color: bool = True) -> str:
    """Pretty-print /memory stats output."""
    if color:
        try:
            from core.terminal_ui import C
            DIM = C.DIM
            RESET = C.RESET
            BOLD = getattr(C, "BOLD", "\033[1m")
            CYAN = getattr(C, "CYAN", "\033[36m")
            GREEN = getattr(C, "GREEN", "\033[32m")
        except Exception:
            DIM = RESET = BOLD = CYAN = GREEN = ""
    else:
        DIM = RESET = BOLD = CYAN = GREEN = ""

    lines: List[str] = []
    lines.append(f"{BOLD}MemoryForge Graph Stats{RESET}")
    lines.append("")
    lines.append(f"  {CYAN}Symbols{RESET}    {stats['symbols']}")
    for k, v in sorted(stats.get("by_symbol_kind", {}).items()):
        lines.append(f"    {DIM}{k:<10}{RESET} {v}")
    lines.append(f"  {CYAN}Findings{RESET}   {stats['findings']}")
    for k, v in sorted(stats.get("by_severity", {}).items(),
                       key=lambda x: -SEVERITY_ORDER.get(x[0], -1)):
        lines.append(f"    {DIM}{k:<10}{RESET} {v}")
    lines.append(f"  {CYAN}Events{RESET}     {stats['events']}")
    lines.append(f"  {CYAN}Relations{RESET}  {stats['relations']}")
    for k, v in sorted(stats.get("by_relation", {}).items(), key=lambda x: -x[1]):
        lines.append(f"    {DIM}{k:<14}{RESET} {v}")
    lines.append("")

    age = stats.get("age_seconds")
    if age is None:
        age_str = f"{DIM}never synced{RESET}"
    elif age < 60:
        age_str = f"{GREEN}{age}s ago{RESET}"
    elif age < 3600:
        age_str = f"{GREEN}{age // 60}m ago{RESET}"
    elif age < 86400:
        age_str = f"{age // 3600}h ago"
    else:
        age_str = f"{DIM}{age // 86400}d ago{RESET}"
    lines.append(f"  {DIM}Last sync:{RESET} {age_str}")

    size_kb = stats.get("db_size_bytes", 0) / 1024.0
    lines.append(f"  {DIM}DB size:{RESET}   {size_kb:.1f} KB")
    lines.append(f"  {DIM}DB path:{RESET}   {stats['db_path']}")
    return "\n".join(lines)


__all__ = [
    "MemoryForge",
    "SyncStats",
    "SymbolRow",
    "FindingRow",
    "EventRow",
    "QueryResult",
    "ParsedQuery",
    "DSLParseError",
    "parse_dsl",
    "nl_to_dsl_heuristic",
    "nl_to_dsl_model",
    "format_query_results",
    "format_facts",
    "format_timeline",
    "format_stats",
    "SEVERITY_ORDER",
    "VALID_RELATIONS",
    "SCHEMA_VERSION",
]
