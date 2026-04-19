"""
LeanAI — ChainBreaker: Multi-Stage Attack Simulation
═════════════════════════════════════════════════════
M2 follow-on to Sentinel (M1).

Sentinel finds isolated vulnerabilities. ChainBreaker turns them into
ATTACK NARRATIVES by walking the brain's call graph forward from each
finding to discover whether the tainted input can reach a high-value
"capability" sink:

  rce          — remote code execution (eval / exec / subprocess)
  exfil        — data egress / network send / clipboard / file write
  privesc      — privilege-escalating context (sudo / setuid / role-bearing)
  secret_read  — credential or secret access (env vars, keys, /etc/shadow)
  persistence  — write to startup-persisted location
  destroy      — destructive file/db ops (rm -rf, DROP TABLE, etc.)

Pipeline
────────
  1. Load Sentinel's persisted findings (~/.leanai/vulns/VULN-YYYY-NNNN.json)
  2. Resolve each finding to a brain graph node (handles abs OR rel path IDs)
  3. BFS forward through brain.graph._adjacency (default depth 4)
  4. For every visited function, classify which "capability stage" it
     represents using sink-pattern matching on its source text
  5. A "chain" = entry vuln → 0+ lateral steps → capability stage
  6. Emit chains with auto-generated narrative + stable CHAIN-YYYY-NNNN IDs
  7. Persist to ~/.leanai/chains/CHAIN-YYYY-NNNN.json so ExploitForge (M3),
     AutoFix (M5), and Aegis (M7) can consume them

Strict rules
────────────
  - Never reports a chain unless an actual capability sink is reached
  - Confidence decays with chain length (longer = lower)
  - Skips test/example dirs by default (configurable via include_tests=True)
  - Stable CHAIN-YYYY-NNNN IDs preserved across re-runs via fingerprint
  - Robust to Sentinel's filepath format (relative or absolute)

Reads Sentinel's Vulnerability JSON format directly. Does not modify it.

Author: Gowri Shankar (github.com/gowrishankar-infra/leanai)
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


# ═══════════════════════════════════════════════════════════════════
# Windows-safe helpers (M2.2)
# ═══════════════════════════════════════════════════════════════════
# Node IDs have the form "filepath:qualified_name". On Windows,
# filepath contains a drive-letter colon, so split(":", 1) truncates to
# just "C". Function names can't contain ":" so rsplit works everywhere.

def _node_file(node_id: str) -> str:
    """Return filepath portion of 'filepath:qualified_name'. Windows-safe."""
    if ':' not in node_id:
        return node_id
    return node_id.rsplit(':', 1)[0]


# ═══════════════════════════════════════════════════════════════════
# Severity (mirrors core.sentinel.Severity to avoid hard import coupling)
# ═══════════════════════════════════════════════════════════════════

class Severity(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0

    def __str__(self):
        return self.name


# ═══════════════════════════════════════════════════════════════════
# Capability-stage detection patterns
# ═══════════════════════════════════════════════════════════════════
#
# Each stage maps to a list of (regex, indicator_label) tuples.
# A function is classified as "exhibiting" a stage if any pattern matches
# its source (after string/comment stripping to avoid false positives).
#
# Stages are ranked by impact_weight — used when a function exhibits more
# than one stage, the highest-weight stage wins for chain classification.
# ═══════════════════════════════════════════════════════════════════

STAGE_PATTERNS = {
    'rce': [
        (r'(?<![\w.])eval\s*\(',                                "eval() call"),
        (r'(?<![\w.])exec\s*\(',                                "exec() call"),
        (r'\bos\.system\s*\(',                                  "os.system() call"),
        (r'\bsubprocess\.\w+\s*\([^)]*shell\s*=\s*True',        "subprocess shell=True"),
        (r'\bcommands\.(?:getoutput|getstatusoutput)\s*\(',     "legacy commands module"),
        (r'\bimportlib\.import_module\s*\(',                    "dynamic import"),
        (r'\bcompile\s*\([^)]*[\'"]exec[\'"]',                   "compile(..., 'exec')"),
        (r'\b__import__\s*\(',                                  "__import__() call"),
    ],
    'exfil': [
        (r'\brequests\.(?:get|post|put|delete|patch|head)\s*\(',"HTTP request"),
        (r'\burllib\.request\.urlopen\s*\(',                    "urlopen()"),
        (r'\bhttpx\.\w+\s*\(',                                  "httpx call"),
        (r'\bsocket\.socket\s*\(',                              "raw socket"),
        (r'\.sendall\s*\(',                                     "socket.sendall"),
        (r'\bsmtplib\.',                                        "smtplib (email send)"),
        (r'\bftplib\.',                                         "ftplib (FTP send)"),
        (r'\bpyperclip\.copy\s*\(',                             "clipboard write"),
    ],
    'privesc': [
        (r'\bos\.setuid\s*\(',                                  "os.setuid"),
        (r'\bos\.seteuid\s*\(',                                 "os.seteuid"),
        (r'\bos\.setgid\s*\(',                                  "os.setgid"),
        (r'\bsudo\s+',                                          "sudo invocation"),
        (r'\.is_superuser\b',                                   "is_superuser check"),
        (r'\.is_admin\b',                                       "is_admin check"),
        (r'role\s*=\s*[\'"]admin',                              "role=admin assignment"),
        (r'\brunas\b',                                          "runas elevation"),
    ],
    'secret_read': [
        (r'\bos\.environ\s*\[\s*[\'"][A-Z_]*(?:KEY|SECRET|TOKEN|PASSWORD|PWD)',
                                                                "env-var secret read"),
        (r'\bos\.getenv\s*\(\s*[\'"][A-Z_]*(?:KEY|SECRET|TOKEN|PASSWORD|PWD)',
                                                                "getenv secret"),
        (r'open\s*\(\s*[\'"][^\'"]*\.pem[\'"]',                  ".pem file read"),
        (r'open\s*\(\s*[\'"]/etc/shadow',                        "/etc/shadow read"),
        (r'open\s*\(\s*[\'"][^\'"]*id_rsa',                      "SSH key read"),
        (r'open\s*\(\s*[\'"][^\'"]*\.env[\'"]',                  ".env file read"),
        (r'\bkeyring\.get_password\s*\(',                       "keyring lookup"),
    ],
    'persistence': [
        (r'open\s*\([^)]*[\'"][^\'"]*authorized_keys',           "SSH authorized_keys write"),
        (r'open\s*\([^)]*[\'"][^\'"]*\.bashrc',                  ".bashrc write"),
        (r'open\s*\([^)]*[\'"][^\'"]*\.profile',                 ".profile write"),
        (r'open\s*\([^)]*[\'"][^\'"]*rc\.local',                 "rc.local write"),
        (r'\bcrontab\b',                                        "crontab manipulation"),
        (r'CurrentVersion\\\\Run',                              "Windows registry Run key"),
        (r'systemctl\s+(?:enable|start)',                       "systemd service control"),
    ],
    'destroy': [
        (r'\bshutil\.rmtree\s*\(',                              "shutil.rmtree"),
        (r'\bos\.remove\s*\([^)]*\+',                           "os.remove with concat"),
        (r'\bos\.unlink\s*\([^)]*\+',                           "os.unlink with concat"),
        (r'\.execute\s*\(\s*[\'"][^\'"]*DROP\s+TABLE',           "SQL DROP TABLE"),
        (r'\.execute\s*\(\s*[\'"][^\'"]*TRUNCATE',               "SQL TRUNCATE"),
        (r'\brm\s+-rf\b',                                       "shell rm -rf"),
    ],
}

# Higher = more impactful. Used to pick the "primary" stage when a
# single function exhibits multiple, and to sort chains by severity.
STAGE_IMPACT_WEIGHT = {
    'rce':         10,
    'destroy':      9,
    'persistence':  8,
    'secret_read':  7,
    'privesc':      6,
    'exfil':        5,
    'lateral':      0,
    'entry':        0,
}

STAGE_DESCRIPTIONS = {
    'rce':         "Remote code execution",
    'exfil':       "Data exfiltration / network egress",
    'privesc':     "Privilege escalation context",
    'secret_read': "Credential / secret access",
    'persistence': "Persistence mechanism",
    'destroy':     "Destructive operation",
    'lateral':     "Lateral movement (intermediate hop)",
    'entry':       "Entry point",
}

# Comment markers per language
COMMENT_PREFIXES = ('#', '//', '/*', '*', '"""', "'''")


# ═══════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AttackStep:
    """One node in an attack chain."""
    function_id: str            # graph node id
    function_name: str          # qualified name
    filepath: str               # file containing this function (rel-path preferred)
    line: int                   # line where the indicator was matched (or func start)
    stage: str                  # 'entry' / 'lateral' / 'rce' / 'exfil' / etc.
    indicators: List[str] = field(default_factory=list)  # human-readable matches

    def to_dict(self):
        return asdict(self)


@dataclass
class AttackChain:
    """A multi-step path from a vulnerability entry to a capability sink."""
    chain_id: str               # CHAIN-YYYY-NNNN
    entry_vuln_id: str          # VULN-YYYY-NNNN that seeded this chain
    severity: Severity          # max severity along the chain
    confidence: float           # 0.0 - 1.0 (decays with chain length)
    capability: str             # primary terminal stage: 'rce' / 'exfil' / etc.
    steps: List[AttackStep]
    narrative: str              # auto-generated attack story
    impact_summary: str         # one-line worst-case description
    fix_recommendation: str

    def to_dict(self):
        d = asdict(self)
        d['severity'] = str(self.severity)
        return d


@dataclass
class ChainBreakerStats:
    findings_loaded: int = 0
    entries_resolved: int = 0
    entries_unresolved: int = 0
    functions_visited: int = 0
    chains_found: int = 0
    filtered_low_confidence: int = 0   # chains dropped below min_confidence (M2.1)
    stale_findings_skipped: int = 0    # pre-patch fingerprints no longer valid (M2.2)
    by_capability: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════

class ChainBreakerEngine:
    """
    Multi-stage attack-chain analyzer.

    Consumes Sentinel's persisted findings + the brain's dependency graph.
    Emits AttackChain objects that name the entry vuln, the lateral hops,
    the terminal capability sink, and a human-readable narrative.
    """

    def __init__(
        self,
        brain,
        project_root: str = None,
        vuln_dir: str = None,
        chain_dir: str = None,
        include_tests: bool = False,
        allow_unreachable: bool = False,
        strict_entry_check: bool = True,
    ):
        """
        brain:               ProjectBrain (must already be scanned)
        project_root:        project root (defaults to brain.config.project_path)
        vuln_dir:            where Sentinel persists findings (default ~/.leanai/vulns)
        chain_dir:           where to persist chains (default ~/.leanai/chains)
        include_tests:       include test/example dirs as chain participants
        allow_unreachable:   (M2.2) keep chains where entry file does NOT
                             transitively import terminal file. Default False —
                             these are almost always name-alias artifacts.
        strict_entry_check:  (M2.2) before using a Sentinel finding as an
                             entry, re-read its file at the reported line and
                             confirm the dangerous sink pattern still appears.
                             Kills stale-fingerprint entries (finding points
                             at code that was already patched).
        """
        self.brain = brain
        self.project_root = project_root or brain.config.project_path

        leanai_home = os.environ.get(
            'LEANAI_HOME', os.path.join(str(Path.home()), '.leanai')
        )
        self.vuln_dir = vuln_dir or os.path.join(leanai_home, 'vulns')
        self.chain_dir = chain_dir or os.path.join(leanai_home, 'chains')
        os.makedirs(self.chain_dir, exist_ok=True)

        self.include_tests = include_tests
        self.allow_unreachable = allow_unreachable
        self.strict_entry_check = strict_entry_check

        # Caches
        self._source_cache: Dict[str, str] = {}            # rel_path -> text
        self._function_source_cache: Dict[str, str] = {}   # node_id -> source text
        self._stage_cache: Dict[str, Tuple[str, List[str], int]] = {}
        # node_id -> (primary_stage, indicators, line_offset)

    # ─────────── Public API ───────────

    def analyze(
        self,
        from_vuln_id: Optional[str] = None,
        max_depth: int = 4,
        severity_floor: Severity = Severity.MEDIUM,
        min_confidence: float = 0.35,
        verbose: bool = True,
    ) -> Tuple[List[AttackChain], ChainBreakerStats]:
        """
        Run multi-stage attack analysis.

        from_vuln_id:    if set, only chain from this single VULN-YYYY-NNNN
        max_depth:       BFS depth from each entry (default 4 hops)
        severity_floor:  drop entry vulns below this severity (default MEDIUM)
        min_confidence:  drop chains whose computed confidence is below this
                         threshold (default 0.35 — filters out likely aliases
                         from name-only call resolution).
        verbose:         print progress

        Returns (chains, stats).
        """
        start = time.time()
        stats = ChainBreakerStats()

        findings = self._load_findings(from_vuln_id, severity_floor)
        stats.findings_loaded = len(findings)

        if verbose:
            if from_vuln_id:
                print(f"[ChainBreaker] Tracing chains from {from_vuln_id}...")
            else:
                print(f"[ChainBreaker] Loaded {len(findings)} entry vulnerabilities "
                      f"(severity >= {severity_floor})")

        all_chains: List[AttackChain] = []
        visited_globally: Set[str] = set()

        for finding in findings:
            # M2.2 stale-finding guard — confirm the sink pattern still
            # appears at the referenced line before treating this as a
            # live entry. Protects against pre-patch fingerprints surviving
            # after the source was fixed.
            if self.strict_entry_check and self._is_stale_finding(finding):
                stats.stale_findings_skipped += 1
                continue

            entry_node_id = self._resolve_vuln_to_node(finding)
            if not entry_node_id:
                stats.entries_unresolved += 1
                continue
            stats.entries_resolved += 1

            chains = self._trace_chains(entry_node_id, finding, max_depth, visited_globally)
            all_chains.extend(chains)

        stats.functions_visited = len(visited_globally)

        # Apply minimum-confidence filter BEFORE deduplication / persistence so
        # low-signal chains never get a CHAIN-ID and never hit the report.
        pre_filter_count = len(all_chains)
        all_chains = [c for c in all_chains if c.confidence >= min_confidence]
        stats.filtered_low_confidence = pre_filter_count - len(all_chains)
        all_chains = self._deduplicate(all_chains)
        all_chains = self._assign_ids(all_chains)
        all_chains.sort(key=lambda c: (-c.severity.value, -c.confidence, c.chain_id))

        for c in all_chains:
            self._persist(c)
            stats.by_capability[c.capability] = stats.by_capability.get(c.capability, 0) + 1
            stats.by_severity[str(c.severity)] = stats.by_severity.get(str(c.severity), 0) + 1

        stats.chains_found = len(all_chains)
        stats.time_ms = (time.time() - start) * 1000

        return all_chains, stats

    # ─────────── Sentinel integration ───────────

    def _load_findings(
        self, from_vuln_id: Optional[str], severity_floor: Severity
    ) -> List[dict]:
        """Read Sentinel's persisted Vulnerability JSON files."""
        findings = []
        if not os.path.isdir(self.vuln_dir):
            return findings

        for fname in os.listdir(self.vuln_dir):
            if not (fname.startswith('VULN-') and fname.endswith('.json')):
                continue
            if from_vuln_id and fname[:-5] != from_vuln_id:
                continue
            try:
                with open(os.path.join(self.vuln_dir, fname), 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
            except Exception:
                continue

            sev_str = str(data.get('severity', 'MEDIUM'))
            try:
                sev = Severity[sev_str]
            except KeyError:
                sev = Severity.MEDIUM
            if sev.value < severity_floor.value:
                continue

            findings.append(data)

        # Sort by severity (CRITICAL first) so chain IDs are stable & meaningful
        findings.sort(key=lambda d: (
            -Severity[str(d.get('severity', 'MEDIUM'))].value
            if str(d.get('severity', 'MEDIUM')) in Severity.__members__ else 0,
            d.get('vuln_id', ''),
        ))
        return findings

    def _resolve_vuln_to_node(self, finding: dict) -> Optional[str]:
        """
        Resolve a Sentinel finding to a brain graph node id.

        Tries (in order):
          1. {vuln.filepath}:{vuln.function_name}              [Sentinel-style relative]
          2. {abs_project_root}/{vuln.filepath}:{function_name}[brain abs-path style]
          3. graph._function_lookup[vuln.function_name]        [name-only fallback]
        """
        vuln_path = finding.get('filepath', '')
        func_name = finding.get('function_name', '')
        if not func_name or func_name == '<module>':
            return None

        # 1. Sentinel-style id
        candidate = f"{vuln_path}:{func_name}"
        if candidate in self.brain.graph._adjacency:
            return candidate

        # 2. Brain absolute-path id
        if not os.path.isabs(vuln_path):
            abs_path = os.path.normpath(os.path.join(self.project_root, vuln_path))
            candidate = f"{abs_path}:{func_name}"
            if candidate in self.brain.graph._adjacency:
                return candidate

        # 3. Name-only lookup
        return self.brain.graph._function_lookup.get(func_name)

    # ─────────── M2.2 Stale-finding guard ───────────

    # Keywords that should still appear in the live source if a Sentinel
    # finding is still valid. Keyed by vuln_class; each entry is a list of
    # regex patterns — if ANY pattern matches the live source in the
    # vicinity of finding['line'], the finding is considered fresh.
    _FRESHNESS_PATTERNS: Dict[str, List[str]] = {
        'command_injection': [
            r'\bos\.system\s*\(',
            r'\bsubprocess\.\w+\s*\([^)]*shell\s*=\s*True',
            r'(?<![\w.])eval\s*\(',
            r'(?<![\w.])exec\s*\(',
            r'\bcommands\.(?:getoutput|getstatusoutput)\s*\(',
        ],
        'sql_injection': [
            r'\.execute\s*\(\s*[fF]["\']',
            r'\.executemany\s*\(\s*[fF]["\']',
            r'\.raw\s*\(\s*[fF]["\']',
        ],
        'unsafe_deserialization': [
            r'\bpickle\.loads?\s*\(',
            r'\byaml\.load\s*\(',
            r'\bmarshal\.loads?\s*\(',
        ],
        'path_traversal': [
            r'open\s*\(\s*[^)]*\+',
            r'open\s*\(\s*[fF]["\']',
            r'os\.path\.join\s*\([^)]*(?:request\.|argv)',
        ],
        'ssrf': [
            r'requests\.\w+\s*\([^)]*\+',
            r'urllib\.request\.urlopen\s*\([^)]*\+',
        ],
        'xss': [
            r'render_template_string\s*\([^)]*\+',
            r'\.innerHTML\s*=',
        ],
        'open_redirect': [
            r'redirect\s*\([^)]*request\.',
        ],
        'weak_crypto': [
            r'\bhashlib\.md5\b',
            r'\bhashlib\.sha1\b',
            r'\bDES\.new\b',
            r'\bRC4\.new\b',
        ],
        'race_condition': [
            r'os\.path\.exists\(',
            r'os\.access\(',
        ],
        'insecure_temp': [
            r'\btempfile\.mktemp\s*\(',
        ],
        'hardcoded_secret': [
            # Hardcoded secrets are rare to patch without actually removing
            # the constant, so most stale cases here are genuine moves.
            r'(?i)(?:password|secret|api[_-]?key|token)\s*=\s*["\']',
        ],
    }

    # Window around finding['line'] to search for the sink pattern. The
    # finding records the line of the dangerous call; the patch might
    # reformat the enclosing code by a handful of lines.
    _STALENESS_LINE_WINDOW = 4

    def _is_stale_finding(self, finding: dict) -> bool:
        """
        Return True if the Sentinel finding no longer matches the live
        source. A stale finding is one whose fingerprint persisted from
        a prior scan but whose code was subsequently patched.

        Strategy:
          - If the finding describes a TAINT SOURCE (http_input, cli_arg,
            env_var, etc.) rather than a sink, the vuln_class patterns
            wouldn't match at the source line anyway — skip the staleness
            check and treat it as fresh. Sources can only become stale if
            the file is gone entirely.
          - Otherwise, read the file at finding['filepath'], look at a
            small window around finding['line'], and ask: does at least ONE
            pattern for this vuln_class still appear in that window?
              - yes → fresh (not stale)
              - no  → stale; skip it
          - If the file can't be read → stale (finding points at nothing).
          - If vuln_class is unknown → fresh (don't over-penalize).
        """
        path_rel = finding.get('filepath', '')
        line = int(finding.get('line', 0) or 0)
        vuln_class = finding.get('vuln_class', '')
        source_type = finding.get('source_type', '')
        if not path_rel or not vuln_class:
            return False

        # If this is a source-based finding, the pattern to check isn't the
        # vuln_class sink pattern — it would be the source pattern (http
        # handler decorator, argparse, env var read, etc.), which is more
        # likely to remain across patches. Easiest: require the file to
        # still exist, otherwise treat as fresh.
        taint_source_types = {
            'http_input', 'cli_arg', 'env_var', 'file_read',
            'deserialization_input',
        }
        # 'traced:*' sources are also source-based (e.g. 'traced:cli_arg')
        is_source_based = (source_type in taint_source_types or
                           source_type.startswith('traced:'))

        try:
            text = self._read_file(path_rel)
        except Exception:
            return True  # file gone — stale

        if is_source_based:
            return False  # source lives in file that exists — fresh

        # Sink-based finding — look for the pattern near the reported line
        patterns = self._FRESHNESS_PATTERNS.get(vuln_class)
        if not patterns or not line:
            return False  # unknown class or no line info — treat as fresh

        src_lines = text.splitlines()
        if not src_lines:
            return True

        start = max(0, line - 1 - self._STALENESS_LINE_WINDOW)
        end = min(len(src_lines), line + self._STALENESS_LINE_WINDOW)
        window = '\n'.join(src_lines[start:end])
        # Strip strings/comments so we don't match an `os.system(` that
        # lives in a docstring of the patched function (M1.1 correctness).
        cleaned = self._strip_strings_and_comments(window)

        for pat in patterns:
            if re.search(pat, cleaned):
                return False  # pattern still present — finding is fresh

        # No live pattern found in the window — stale.
        return True

    # ─────────── M2.1 Chain quality scoring ───────────

    def _chain_quality_signals(
        self,
        path: List[str],
        entry_finding: dict,
    ) -> Tuple[float, bool, int]:
        """
        Compute three signals describing how trustworthy a chain is:

          1. receiver_fraction (0..1) — fraction of internal hops whose
             underlying call edge carried a non-empty receiver hint.
             Higher = more precise call resolution along the path.

          2. import_reachable (bool) — can the entry file reach the
             terminal file through the imports graph (transitively)?
             False is a strong "probably a name collision" signal.

          3. same_class_edges (int) — number of hops where the edge was
             a 'self' / 'cls' call within the same class. This is the
             most reliable kind of internal edge we have.
        """
        # Receiver fraction across hop edges (path[i] -> path[i+1])
        hop_edges_with_receiver = 0
        hop_edges_total = 0
        same_class_edges = 0

        # Build a fast lookup from (src, tgt) -> list of edges (there may be
        # parallel edges from the same function; any with a receiver counts)
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            hop_edges_total += 1
            found_receiver = False
            found_same_class = False
            for e in self.brain.graph.edges:
                if e.source != src or e.target != tgt:
                    continue
                if e.edge_type != "calls":
                    continue
                recv = e.metadata.get("receiver", "") if e.metadata else ""
                src_class = e.metadata.get("source_class", "") if e.metadata else ""
                if recv:
                    found_receiver = True
                if recv in ("self", "cls") and src_class:
                    # Confirm target really is in the same class
                    tgt_node = self.brain.graph.nodes.get(tgt)
                    # tgt_node.filepath is a raw path (no qualified name),
                    # but `src` is a node ID so we use _node_file on src only.
                    if tgt_node and tgt_node.metadata.get("class_name") == src_class \
                            and tgt_node.filepath == _node_file(src):
                        found_same_class = True
                # Don't break — there might be multiple call sites with
                # the same src/tgt and we want the STRONGEST one to count
            if found_receiver:
                hop_edges_with_receiver += 1
            if found_same_class:
                same_class_edges += 1

        receiver_fraction = (
            hop_edges_with_receiver / hop_edges_total if hop_edges_total else 0.0
        )

        # Import reachability between entry file and terminal file
        entry_file_rel = entry_finding.get("filepath", "")
        terminal_node = self.brain.graph.nodes.get(path[-1])
        terminal_file_abs = terminal_node.filepath if terminal_node else ""
        terminal_file_rel = self._rel_path(terminal_file_abs)
        import_reachable = self._is_import_reachable(entry_file_rel, terminal_file_rel)

        return receiver_fraction, import_reachable, same_class_edges

    def _is_import_reachable(self, from_file_rel: str, to_file_rel: str) -> bool:
        """
        True if `from_file_rel` transitively imports anything that resolves
        to (or lives in the same package as) `to_file_rel`.

        Heuristic — the brain tracks import *module names*, not resolved
        target files. We consider a reach if:
          - same file, OR
          - to_file_rel's module path (relative, dots for separators,
            minus .py) appears among from_file's imports, OR
          - to_file_rel's top-level package matches one of from_file's imports
        """
        if not from_file_rel or not to_file_rel:
            # Can't verify; give benefit of the doubt (don't over-penalize)
            return True
        if from_file_rel == to_file_rel:
            return True

        graph = self.brain.graph

        # Resolve from_file_rel to the graph's stored filepath key
        from_file_key = self._find_file_key(from_file_rel)
        to_file_key = self._find_file_key(to_file_rel)
        if not from_file_key or not to_file_key:
            return True  # can't verify; don't penalize

        # Build module paths for the target file
        to_module_candidates = self._module_name_candidates(to_file_rel)

        # BFS from from_file_key via imports to see if we reach any candidate
        seen: Set[str] = set()
        queue: List[str] = [from_file_key]
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            imports_here = graph._file_imports.get(cur, set())
            # Direct match against module-name candidates
            if imports_here & to_module_candidates:
                return True
            # Walk to any project file whose module name is imported here
            for other_key in graph._file_imports:
                if other_key in seen or other_key == cur:
                    continue
                other_rel = self._rel_path(other_key)
                other_mods = self._module_name_candidates(other_rel)
                if other_mods & imports_here:
                    queue.append(other_key)

        return False

    def _find_file_key(self, rel_path: str) -> Optional[str]:
        """Find the graph's stored key (abs or rel) for a relative path."""
        # Try the rel path first (some brains store rel)
        if rel_path in self.brain.graph._file_imports:
            return rel_path
        # Try absolute
        abs_path = os.path.normpath(os.path.join(self.project_root, rel_path))
        if abs_path in self.brain.graph._file_imports:
            return abs_path
        # Try scanning all keys for a suffix match
        rp = rel_path.replace('\\', '/')
        for key in self.brain.graph._file_imports:
            if key.replace('\\', '/').endswith('/' + rp) or key.replace('\\', '/').endswith(rp):
                return key
        return None

    def _module_name_candidates(self, rel_path: str) -> Set[str]:
        """Convert a relative file path into possible import module names.

        E.g. 'brain/tdd_loop.py' → {'brain.tdd_loop', 'tdd_loop', 'brain'}
        """
        if not rel_path:
            return set()
        norm = rel_path.replace('\\', '/')
        if norm.endswith('.py'):
            norm = norm[:-3]
        if norm.endswith('/__init__'):
            norm = norm[:-len('/__init__')]
        parts = [p for p in norm.split('/') if p and p != '.']
        candidates: Set[str] = set()
        if parts:
            candidates.add('.'.join(parts))          # brain.tdd_loop
            candidates.add(parts[-1])                # tdd_loop
            candidates.add(parts[0])                 # brain
            # Any tail subsequence (for deep packages)
            for i in range(len(parts)):
                candidates.add('.'.join(parts[i:]))
        return candidates

    # ─────────── Stage classification ───────────

    def _classify_function(self, node_id: str) -> Tuple[str, List[str], int]:
        """
        Determine the strongest "capability stage" exhibited by a function.

        Returns (primary_stage, indicators, line_offset_of_first_match)
        primary_stage is 'lateral' if no capability indicators found.
        """
        if node_id in self._stage_cache:
            return self._stage_cache[node_id]

        func_source, line_start = self._get_function_source(node_id)
        if not func_source:
            result = ('lateral', [], 0)
            self._stage_cache[node_id] = result
            return result

        cleaned = self._strip_strings_and_comments(func_source)

        best_stage = 'lateral'
        best_weight = -1
        all_indicators: List[str] = []
        first_match_line_offset = 0

        for stage, patterns in STAGE_PATTERNS.items():
            stage_indicators: List[str] = []
            stage_first_offset = -1
            for pat, label in patterns:
                m = re.search(pat, cleaned)
                if m:
                    stage_indicators.append(label)
                    if stage_first_offset < 0:
                        stage_first_offset = cleaned[:m.start()].count('\n')
            if stage_indicators:
                weight = STAGE_IMPACT_WEIGHT.get(stage, 1)
                all_indicators.extend(f"{stage}: {ind}" for ind in stage_indicators)
                if weight > best_weight:
                    best_stage = stage
                    best_weight = weight
                    first_match_line_offset = stage_first_offset

        result = (best_stage, all_indicators, first_match_line_offset)
        self._stage_cache[node_id] = result
        return result

    # ─────────── BFS chain construction ───────────

    def _trace_chains(
        self,
        entry_node_id: str,
        entry_finding: dict,
        max_depth: int,
        visited_globally: Set[str],
    ) -> List[AttackChain]:
        """
        BFS from entry. Each path that reaches a function with a non-'lateral'
        stage becomes an AttackChain.

        We do NOT prune on first capability hit — we keep walking so that
        "rce that ALSO triggers persistence" gets recorded as a chain ending
        at the deeper, more impactful stage. Each (entry, terminal) pair
        emits one chain.
        """
        if entry_node_id not in self.brain.graph._adjacency:
            return []

        chains: List[AttackChain] = []
        seen_terminals: Set[str] = set()    # terminal_node_id per entry
        queue: List[Tuple[str, List[str]]] = [(entry_node_id, [entry_node_id])]
        visited_globally.add(entry_node_id)

        while queue:
            current, path = queue.pop(0)
            depth = len(path) - 1

            if depth >= max_depth:
                continue

            for target in self.brain.graph._adjacency.get(current, []):
                # Skip unresolved imports / cross-language refs
                if target.startswith('__'):
                    continue
                # Skip if this target would create a cycle in this path
                if target in path:
                    continue
                # Skip test/example unless explicitly included
                if not self.include_tests and self._is_test_path(target):
                    continue

                visited_globally.add(target)
                new_path = path + [target]

                # Classify this newly visited node
                stage, indicators, line_offset = self._classify_function(target)
                if stage != 'lateral' and target not in seen_terminals:
                    seen_terminals.add(target)
                    chain = self._build_chain(
                        entry_finding, new_path, terminal_stage=stage,
                        terminal_indicators=indicators, terminal_line_offset=line_offset,
                    )
                    if chain:
                        chains.append(chain)

                # Continue BFS regardless — a deeper node may exhibit a stronger stage
                queue.append((target, new_path))

        return chains

    def _build_chain(
        self,
        entry_finding: dict,
        path: List[str],
        terminal_stage: str,
        terminal_indicators: List[str],
        terminal_line_offset: int,
    ) -> Optional[AttackChain]:
        """Construct an AttackChain from a path of node IDs."""
        if len(path) < 2:
            return None

        steps: List[AttackStep] = []

        # Step 0: the entry vuln (use Sentinel's metadata directly)
        entry_node = self.brain.graph.nodes.get(path[0])
        entry_filepath = entry_finding.get('filepath', '')
        entry_step = AttackStep(
            function_id=path[0],
            function_name=entry_finding.get('function_name', entry_node.name if entry_node else '<unknown>'),
            filepath=entry_filepath,
            line=int(entry_finding.get('line', 0) or 0),
            stage='entry',
            indicators=[
                f"{entry_finding.get('vuln_class', 'vulnerability')}: "
                f"{entry_finding.get('sink_type', '')}"
            ],
        )
        steps.append(entry_step)

        # Lateral steps (everything between entry and terminal)
        for nid in path[1:-1]:
            node = self.brain.graph.nodes.get(nid)
            if not node:
                continue
            steps.append(AttackStep(
                function_id=nid,
                function_name=node.name,
                filepath=self._rel_path(node.filepath),
                line=int(node.metadata.get('line_start', 0) or 0),
                stage='lateral',
                indicators=[],
            ))

        # Terminal step: the capability hit
        terminal_id = path[-1]
        terminal_node = self.brain.graph.nodes.get(terminal_id)
        if not terminal_node:
            return None
        terminal_func_line = int(terminal_node.metadata.get('line_start', 0) or 0)
        steps.append(AttackStep(
            function_id=terminal_id,
            function_name=terminal_node.name,
            filepath=self._rel_path(terminal_node.filepath),
            line=terminal_func_line + terminal_line_offset,
            stage=terminal_stage,
            indicators=terminal_indicators,
        ))

        # Severity = max(entry severity, terminal capability severity-tier)
        entry_sev_str = str(entry_finding.get('severity', 'MEDIUM'))
        try:
            entry_sev = Severity[entry_sev_str]
        except KeyError:
            entry_sev = Severity.MEDIUM
        terminal_sev = self._terminal_stage_severity(terminal_stage)
        chain_sev = entry_sev if entry_sev.value >= terminal_sev.value else terminal_sev

        # ── M2.1 Confidence formula ──
        # Base decay is stricter than before because name-only call resolution
        # is noisy. A raw chain gets a pessimistic base; we add positive
        # signals when evidence is strong.
        #
        # Signals computed across the path:
        #   - receiver_fraction: fraction of internal edges whose raw call
        #     had a non-empty receiver hint (e.g. `self.foo.bar`, not bare
        #     `bar()`). More receivers = more precise resolution.
        #   - import_reachable:  True if the entry file transitively imports
        #     the terminal file (or they are the same file).
        #   - same_class_edges:  intermediate hops where both endpoints live
        #     in the same file AND the edge carried a 'self' receiver.
        #
        # Confidence = base_conf * entry_conf_adjust + bonuses, clamped [0.2, 0.98]
        chain_len = len(steps) - 1  # hops, not nodes
        base_conf = max(0.25, 0.75 - 0.15 * chain_len)

        receiver_fraction, import_reachable, same_class_edges = \
            self._chain_quality_signals(path, entry_finding)

        # ── M2.2 HARD import-reachability filter ──
        # If the chain crosses files AND the entry file does NOT transitively
        # import the terminal file, the chain is almost certainly a
        # name-alias artifact (two unrelated functions sharing a name). Drop
        # it entirely rather than surface it at low confidence — users have
        # told us they don't trust chains that "might be real".
        #
        # Users can opt back in via self.allow_unreachable=True when they
        # want to explore speculative chains.
        entry_file = entry_finding.get("filepath", "")
        terminal_node = self.brain.graph.nodes.get(path[-1])
        terminal_file = self._rel_path(terminal_node.filepath) if terminal_node else ""
        cross_file = bool(entry_file and terminal_file and
                          self._normalize_path(entry_file) != self._normalize_path(terminal_file))
        if cross_file and not import_reachable and not self.allow_unreachable:
            return None

        entry_conf_mul = float(entry_finding.get('confidence', 0.7) or 0.7)
        # Entry confidence moves the needle less than before — 0.7..1.0 range
        entry_adjust = 0.7 + 0.3 * entry_conf_mul

        confidence = base_conf * entry_adjust

        # Quality bonuses (only raise confidence when we have real evidence)
        if import_reachable:
            confidence += 0.10     # file-level reachability check passes
        if receiver_fraction >= 0.67:
            confidence += 0.10     # most edges have receiver hints
        elif receiver_fraction >= 0.34:
            confidence += 0.05     # at least a third do
        if same_class_edges >= 1:
            confidence += 0.05     # intra-class `self.foo()` pattern

        # Quality penalty if we got here via --allow-unreachable (soft mode)
        if cross_file and not import_reachable:
            confidence -= 0.25     # loud signal this is speculative

        confidence = round(max(0.20, min(0.98, confidence)), 2)

        narrative = self._build_narrative(steps, terminal_stage)
        impact = self._impact_summary(terminal_stage, steps[-1])
        fix_rec = self._fix_recommendation(entry_finding, terminal_stage)

        return AttackChain(
            chain_id='',  # filled by _assign_ids
            entry_vuln_id=entry_finding.get('vuln_id', 'VULN-?'),
            severity=chain_sev,
            confidence=confidence,
            capability=terminal_stage,
            steps=steps,
            narrative=narrative,
            impact_summary=impact,
            fix_recommendation=fix_rec,
        )

    # ─────────── Narrative + impact + fix generation ───────────

    def _build_narrative(self, steps: List[AttackStep], terminal_stage: str) -> str:
        """One-paragraph attack story written from the steps."""
        if not steps:
            return ""
        entry = steps[0]
        terminal = steps[-1]
        lateral = steps[1:-1]

        parts = [
            f"Attacker-controlled input enters via {entry.function_name} "
            f"({entry.filepath}:{entry.line}; {entry.indicators[0] if entry.indicators else 'tainted source'})."
        ]
        if lateral:
            hops = ' → '.join(s.function_name for s in lateral)
            parts.append(f"It propagates through the call chain: {hops}.")
        parts.append(
            f"Eventually it reaches {terminal.function_name} "
            f"({terminal.filepath}:{terminal.line}), where "
            f"{STAGE_DESCRIPTIONS.get(terminal_stage, terminal_stage)} is possible "
            f"({', '.join(terminal.indicators[:2]) if terminal.indicators else 'capability sink'})."
        )
        return ' '.join(parts)

    def _impact_summary(self, terminal_stage: str, terminal_step: AttackStep) -> str:
        """One-line worst-case summary."""
        impact_map = {
            'rce':         "Arbitrary code execution as the LeanAI process user.",
            'exfil':       "Sensitive data may be sent to an attacker-controlled endpoint.",
            'privesc':     "Attacker may execute downstream operations with elevated privileges.",
            'secret_read': "Secrets, credentials, or private keys may be exposed.",
            'persistence': "Attacker may install a persistent backdoor that survives reboot.",
            'destroy':     "Files, tables, or data may be irreversibly deleted.",
        }
        return impact_map.get(terminal_stage, "Capability sink reachable from untrusted input.")

    def _fix_recommendation(self, entry_finding: dict, terminal_stage: str) -> str:
        """Combine the entry vuln's fix with capability-specific advice."""
        entry_fix = entry_finding.get('fix_suggestion', '')
        cap_fix = {
            'rce':         "Validate or sandbox the executed payload; never pass untrusted data to eval/exec/subprocess.",
            'exfil':       "Add an allowlist of egress destinations; redact sensitive fields before logging or sending.",
            'privesc':     "Drop privileges before processing untrusted input; never escalate inside a request handler.",
            'secret_read': "Move secret reads behind explicit authorization; do not let a request path reach .env / .pem files.",
            'persistence': "Refuse writes to startup-persistence locations from any code path reachable by user input.",
            'destroy':     "Add confirmation gates; never let user input flow into rmtree / unlink / DROP.",
        }
        return f"Patch the entry: {entry_fix}  Then defense-in-depth: {cap_fix.get(terminal_stage, '')}"

    def _terminal_stage_severity(self, stage: str) -> Severity:
        """Map terminal stage to its inherent severity tier."""
        if stage in ('rce', 'destroy'):
            return Severity.CRITICAL
        if stage in ('persistence', 'secret_read', 'privesc'):
            return Severity.HIGH
        if stage == 'exfil':
            return Severity.MEDIUM
        return Severity.LOW

    # ─────────── Helpers ───────────

    def _rel_path(self, filepath: str) -> str:
        """Return rel-path form of a node's filepath (graph stores absolute)."""
        if not filepath:
            return filepath
        if os.path.isabs(filepath):
            try:
                return os.path.relpath(filepath, self.project_root)
            except ValueError:
                return filepath
        return filepath

    def _normalize_path(self, filepath: str) -> str:
        """Normalize a filepath for comparison: lowercase (for Windows
        case-insensitivity), forward-slash separators, strip trailing slash."""
        if not filepath:
            return ''
        return filepath.replace('\\', '/').rstrip('/').lower()

    def _is_test_path(self, node_id: str) -> bool:
        """True if this node lives in a test/example directory."""
        path = _node_file(node_id).lower().replace('\\', '/')
        parts = set(path.split('/'))
        return bool(parts & {'tests', 'test', 'examples', 'example'})

    def _get_function_source(self, node_id: str) -> Tuple[str, int]:
        """
        Return (source_text, line_start) for a function node, or ('', 0) if not resolvable.
        """
        if node_id in self._function_source_cache:
            return self._function_source_cache[node_id], 0

        node = self.brain.graph.nodes.get(node_id)
        if not node:
            return '', 0
        line_start = int(node.metadata.get('line_start', 0) or 0)
        line_end = int(node.metadata.get('line_end', 0) or 0)
        if not line_start:
            return '', 0
        if not line_end or line_end < line_start:
            line_end = line_start + 60

        rel = self._rel_path(node.filepath)
        try:
            text = self._read_file(rel) if not os.path.isabs(node.filepath) \
                else self._read_abs(node.filepath)
        except Exception:
            return '', 0

        lines = text.splitlines()
        start = max(0, line_start - 1)
        end = min(len(lines), line_end)
        src = '\n'.join(lines[start:end])
        self._function_source_cache[node_id] = src
        return src, line_start

    def _read_file(self, rel_path: str) -> str:
        if rel_path in self._source_cache:
            return self._source_cache[rel_path]
        full = os.path.join(self.project_root, rel_path)
        with open(full, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        self._source_cache[rel_path] = text
        return text

    def _read_abs(self, abs_path: str) -> str:
        if abs_path in self._source_cache:
            return self._source_cache[abs_path]
        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        self._source_cache[abs_path] = text
        return text

    def _strip_strings_and_comments(self, source: str) -> str:
        """
        Replace string literals and comments with spaces (preserves line numbers).
        Same approach Sentinel uses — eliminates false positives from
        dangerous patterns that appear inside strings or docstrings.
        """
        out = []
        i = 0
        n = len(source)
        while i < n:
            c = source[i]

            # Comment to EOL
            if c == '#':
                end = source.find('\n', i)
                if end == -1:
                    end = n
                out.append(' ' * (end - i))
                i = end
                continue

            # Triple-quoted
            if c in ('"', "'") and i + 2 < n and source[i:i + 3] in ('"""', "'''"):
                quote = source[i:i + 3]
                out.append('   ')
                i += 3
                end = source.find(quote, i)
                if end == -1:
                    chunk = source[i:n]
                    out.append(''.join(' ' if ch != '\n' else '\n' for ch in chunk))
                    i = n
                else:
                    chunk = source[i:end]
                    out.append(''.join(' ' if ch != '\n' else '\n' for ch in chunk))
                    out.append('   ')
                    i = end + 3
                continue

            # Single-line string
            if c in ('"', "'"):
                quote = c
                out.append(' ')
                i += 1
                while i < n:
                    if source[i] == '\\' and i + 1 < n:
                        out.append('  ')
                        i += 2
                        continue
                    if source[i] == quote:
                        out.append(' ')
                        i += 1
                        break
                    if source[i] == '\n':
                        out.append('\n')
                        i += 1
                        break
                    out.append(' ')
                    i += 1
                continue

            out.append(c)
            i += 1
        return ''.join(out)

    # ─────────── ID assignment + persistence ───────────

    def _deduplicate(self, chains: List[AttackChain]) -> List[AttackChain]:
        """Keep one chain per (entry_vuln_id, terminal_node_id) pair."""
        seen = set()
        out = []
        for c in chains:
            key = (c.entry_vuln_id, c.steps[-1].function_id if c.steps else '')
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _assign_ids(self, chains: List[AttackChain]) -> List[AttackChain]:
        """Assign stable CHAIN-YYYY-NNNN IDs preserved across runs via fingerprint."""
        existing_by_fp: Dict[str, str] = {}
        try:
            for fname in os.listdir(self.chain_dir):
                if not (fname.startswith('CHAIN-') and fname.endswith('.json')):
                    continue
                try:
                    with open(os.path.join(self.chain_dir, fname), 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                    fp = data.get('fingerprint')
                    if fp:
                        existing_by_fp[fp] = fname[:-5]
                except Exception:
                    pass
        except FileNotFoundError:
            pass

        max_num = 0
        for cid in existing_by_fp.values():
            try:
                n = int(cid.split('-')[-1])
                if n > max_num:
                    max_num = n
            except (ValueError, IndexError):
                pass
        next_num = max_num + 1
        year = time.strftime('%Y')

        for c in chains:
            fp = self._fingerprint(c)
            if fp in existing_by_fp:
                c.chain_id = existing_by_fp[fp]
            else:
                c.chain_id = f"CHAIN-{year}-{next_num:04d}"
                next_num += 1
        return chains

    def _fingerprint(self, c: AttackChain) -> str:
        terminal = c.steps[-1] if c.steps else None
        seed = (
            f"{c.entry_vuln_id}|"
            f"{terminal.function_id if terminal else ''}|"
            f"{c.capability}"
        )
        return hashlib.md5(seed.encode(), usedforsecurity=False).hexdigest()[:10]

    def _persist(self, chain: AttackChain) -> None:
        data = chain.to_dict()
        data['fingerprint'] = self._fingerprint(chain)
        data['timestamp'] = time.time()
        fpath = os.path.join(self.chain_dir, chain.chain_id + '.json')
        try:
            with open(fpath, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Formatting helper (used by main.py)
# ═══════════════════════════════════════════════════════════════════

def format_chains_report(
    chains: List[AttackChain], stats: ChainBreakerStats, color: bool = True
) -> str:
    """Human-readable report. Mirrors Sentinel's format_findings_report style."""
    if color:
        try:
            from core.terminal_ui import C
            DIM    = C.DIM
            RESET  = C.RESET
            BOLD   = getattr(C, 'BOLD', '\033[1m')
            RED    = getattr(C, 'RED', '\033[31m')
            YELLOW = getattr(C, 'YELLOW', '\033[33m')
            CYAN   = getattr(C, 'CYAN', '\033[36m')
            GREEN  = getattr(C, 'GREEN', '\033[32m')
            MAG    = getattr(C, 'MAGENTA', '\033[35m')
        except Exception:
            DIM = RESET = BOLD = RED = YELLOW = CYAN = GREEN = MAG = ''
    else:
        DIM = RESET = BOLD = RED = YELLOW = CYAN = GREEN = MAG = ''

    SEV_COLORS = {
        'CRITICAL': RED, 'HIGH': RED, 'MEDIUM': YELLOW,
        'LOW': CYAN, 'INFO': DIM,
    }
    STAGE_COLORS = {
        'entry':       MAG,
        'lateral':     DIM,
        'rce':         RED,
        'destroy':     RED,
        'persistence': YELLOW,
        'secret_read': YELLOW,
        'privesc':     YELLOW,
        'exfil':       CYAN,
    }

    lines = [
        "",
        f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}",
        f"{BOLD}  ChainBreaker Multi-Stage Attack Analysis{RESET}",
        f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}",
        "",
    ]

    if not chains:
        lines.append(f"  {GREEN}✓ No exploitable attack chains found.{RESET}")
        lines.append("")
        lines.append(
            f"  {DIM}Findings analyzed: {stats.findings_loaded}  "
            f"Entries resolved: {stats.entries_resolved}  "
            f"Functions visited: {stats.functions_visited}  "
            f"Time: {stats.time_ms:.0f}ms{RESET}"
        )
        if stats.entries_unresolved:
            lines.append(
                f"  {DIM}Note: {stats.entries_unresolved} finding(s) could not be "
                f"resolved to a graph node (likely module-level findings).{RESET}"
            )
        if stats.stale_findings_skipped:
            lines.append(
                f"  {DIM}Skipped {stats.stale_findings_skipped} stale finding(s) — "
                f"the referenced code no longer matches the reported sink "
                f"(likely patched since the finding was stored).{RESET}"
            )
        if stats.filtered_low_confidence:
            lines.append(
                f"  {DIM}Filtered {stats.filtered_low_confidence} low-confidence "
                f"chain(s) below the threshold (likely name-only aliasing).{RESET}"
            )
        return '\n'.join(lines)

    sev_summary = '  '.join(
        f"{SEV_COLORS.get(s, '')}{c} {s}{RESET}"
        for s, c in sorted(stats.by_severity.items(), key=lambda x: -Severity[x[0]].value)
    )
    cap_summary = ', '.join(f"{c} {cap}" for cap, c in stats.by_capability.items())

    lines.append(f"  Found {BOLD}{len(chains)}{RESET} attack chain(s)   {sev_summary}")
    lines.append(
        f"  {DIM}{stats.findings_loaded} entry findings, "
        f"{stats.entries_resolved} resolved, "
        f"{stats.functions_visited} functions traversed, "
        f"{stats.time_ms:.0f}ms{RESET}"
    )
    if stats.filtered_low_confidence:
        lines.append(
            f"  {DIM}Filtered {stats.filtered_low_confidence} low-confidence "
            f"chain(s) — use --min-confidence to see them.{RESET}"
        )
    if stats.stale_findings_skipped:
        lines.append(
            f"  {DIM}Skipped {stats.stale_findings_skipped} stale finding(s) "
            f"(code was patched since the finding was stored).{RESET}"
        )
    if cap_summary:
        lines.append(f"  {DIM}Capabilities reached: {cap_summary}{RESET}")
    lines.append("")

    # Group by severity
    by_sev: Dict[str, List[AttackChain]] = {}
    for c in chains:
        by_sev.setdefault(str(c.severity), []).append(c)

    for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
        if sev not in by_sev:
            continue
        col = SEV_COLORS.get(sev, '')
        lines.append(f"{col}{BOLD}── {sev} ({len(by_sev[sev])}) ──{RESET}")
        for c in by_sev[sev]:
            lines.append("")
            lines.append(
                f"  {col}[{c.chain_id}]{RESET} "
                f"{BOLD}{c.capability.upper()}{RESET} chain "
                f"{DIM}(from {c.entry_vuln_id}, {c.confidence:.0%} confidence){RESET}"
            )
            for i, step in enumerate(c.steps, 1):
                stage_col = STAGE_COLORS.get(step.stage, '')
                marker = f"{stage_col}[{step.stage}]{RESET}"
                line_str = f":{step.line}" if step.line else ""
                lines.append(
                    f"    {DIM}{i}.{RESET} {marker} "
                    f"{step.function_name}  "
                    f"{DIM}{step.filepath}{line_str}{RESET}"
                )
                if step.indicators:
                    for ind in step.indicators[:2]:
                        lines.append(f"        {DIM}└─ {ind}{RESET}")
            lines.append(f"    {GREEN}Impact:{RESET} {c.impact_summary}")
            lines.append(f"    {DIM}Story:{RESET}  {c.narrative}")
            lines.append(f"    {GREEN}Fix:{RESET}    {c.fix_recommendation}")
        lines.append("")

    return '\n'.join(lines)
