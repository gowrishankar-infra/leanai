"""
LeanAI — Sentinel: Autonomous Security Analyzer
═══════════════════════════════════════════════
Mythos-inspired, locally-grounded security analysis.

Pipeline:
  1. Walk every function in the brain's AST
  2. Identify external input sources (HTTP handlers, CLI args, file reads,
     env vars, deserialization, user input)
  3. Identify dangerous sinks (eval, exec, subprocess shell=True, SQL
     string concat, os.path.join with input, pickle.loads, weak crypto,
     hardcoded secrets)
  4. Pattern-match 12 vulnerability classes against function source
  5. Trace data flow from sources to sinks via the brain's call graph
  6. (Optional) Ask the model "is this exploitable?" for high-confidence
     candidates
  7. Report with severity, location, taint path, fix suggestion
  8. Persist with stable IDs (VULN-YYYY-NNNN) to ~/.leanai/vulns/ for
     ChainBreaker (M2) and ExploitForge (M3) integration

Strict rules:
  - Never reports without an actual code pattern match or reachable path
  - Conservative confidence scoring per detector
  - Skips test/example files for hardcoded-secret detection
  - Skips comment lines for all detectors
  - Deduplicates findings (same file + line + class)
  - Stable IDs preserved across scans via fingerprint matching

Author: Gowri Shankar (github.com/gowrishankar-infra/leanai)
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════════════════════
# Severity / Vulnerability dataclasses
# ═══════════════════════════════════════════════════════════════════

class Severity(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0

    def __str__(self):
        return self.name


@dataclass
class Vulnerability:
    vuln_id: str           # VULN-2026-0001
    vuln_class: str        # sql_injection, path_traversal, etc.
    severity: Severity
    confidence: float      # 0.0 - 1.0
    filepath: str
    function_name: str
    line: int
    source_type: str       # http_input, cli_arg, file_read, env_var, etc.
    sink_type: str         # human-readable sink description
    description: str
    code_snippet: str
    taint_path: List[str]  # function names traversed (length > 1 = real flow)
    fix_suggestion: str

    def to_dict(self):
        d = asdict(self)
        d['severity'] = str(self.severity)
        return d


@dataclass
class SentinelStats:
    files_scanned: int = 0
    functions_analyzed: int = 0
    sources_found: int = 0
    sinks_found: int = 0
    taint_paths_traced: int = 0
    vulnerabilities_found: int = 0
    time_ms: float = 0.0
    by_severity: Dict[str, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Detection patterns
# ═══════════════════════════════════════════════════════════════════

# Source patterns — code patterns that take in external/untrusted data
SOURCE_PATTERNS = {
    'http_input': [
        r'\brequest\.(?:args|form|json|values|data|files|cookies)\b',
        r'\bawait\s+request\.(?:json|form|body)\(\)',
        r'\bbody\s*[:=]\s*await',
    ],
    'cli_arg': [
        r'\bargparse\b',
        r'\bsys\.argv\b',
        r'(?<![\w.])input\s*\(',
    ],
    'env_var': [
        r'\bos\.environ\[',
        r'\bos\.getenv\s*\(',
    ],
    'file_read': [
        r'\.read\s*\(',
        r'(?<![\w.])open\s*\(',
    ],
    'deserialization_input': [
        r'\bpickle\.loads?\s*\(',
        r'\byaml\.load\s*\(',
        r'\bjson\.loads?\s*\(',
    ],
}

# Sink patterns — dangerous operations
# Format: { vuln_class: [ (regex, human_readable_description), ... ] }
SINK_PATTERNS = {
    'sql_injection': [
        (r'\.execute\s*\(\s*[fF]["\']', "SQL execute() with f-string"),
        (r'\.execute\s*\(\s*["\'][^"\']*["\']\s*[%+]', "SQL execute() with string concat/format"),
        (r'\.executemany\s*\(\s*[fF]["\']', "SQL executemany() with f-string"),
        (r'\.raw\s*\(\s*[fF]["\']', "Django .raw() with f-string"),
    ],
    'command_injection': [
        (r'\bos\.system\s*\(', "os.system() call"),
        (r'\bsubprocess\.(?:call|run|Popen|check_output|check_call)\s*\([^)]*shell\s*=\s*True', "subprocess with shell=True"),
        (r'(?<![\w.])eval\s*\(', "eval() call"),
        (r'(?<![\w.])exec\s*\(', "exec() call"),
        (r'\bcommands\.(?:getoutput|getstatusoutput)\s*\(', "commands module (legacy, unsafe)"),
    ],
    'path_traversal': [
        (r'open\s*\(\s*[^)]*\+\s*[^)]*\)', "open() with concatenated path"),
        (r'open\s*\(\s*[fF]["\']', "open() with f-string path"),
        (r'os\.path\.join\s*\([^)]*request\.', "os.path.join with HTTP input"),
        (r'os\.path\.join\s*\([^)]*\bargv\b', "os.path.join with sys.argv"),
        (r'Path\s*\([^)]*\+\s*[^)]*\)', "Path() with concatenated input"),
    ],
    'unsafe_deserialization': [
        (r'\bpickle\.loads?\s*\(', "pickle.loads (RCE risk if untrusted)"),
        (r'\byaml\.load\s*\(\s*[^,)]+\s*\)(?!\s*,\s*Loader)', "yaml.load without safe Loader"),
        (r'\bshelve\.open\s*\(', "shelve.open uses pickle internally"),
        (r'\bmarshal\.loads?\s*\(', "marshal.loads (RCE risk)"),
    ],
    'xss': [
        (r'render_template_string\s*\([^)]*\+', "render_template_string with concat"),
        (r'\bMarkup\s*\([^)]*\+', "Markup() with concat"),
        (r'\.innerHTML\s*=', "innerHTML assignment (JS)"),
    ],
    'ssrf': [
        (r'requests\.(?:get|post|put|delete|patch|head)\s*\([^)]*\+', "HTTP request with concatenated URL"),
        (r'urllib\.request\.urlopen\s*\([^)]*\+', "urlopen with concatenated URL"),
        (r'httpx\.\w+\s*\([^)]*\+', "httpx call with concatenated URL"),
    ],
    'open_redirect': [
        (r'redirect\s*\([^)]*request\.', "redirect with HTTP input"),
    ],
    'weak_crypto': [
        (r'\bhashlib\.md5\b', "MD5 hash (cryptographically broken)"),
        (r'\bhashlib\.sha1\b', "SHA1 hash (cryptographically weak)"),
        (r'\bDES\.new\b', "DES cipher (broken)"),
        (r'\bRC4\.new\b', "RC4 cipher (broken)"),
        (r'(?<![\w.])random\.(?:random|randint|choice)\s*\([^)]*\)\s*#?\s*(?:token|secret|key|password|salt|nonce|iv)', "random module used in crypto context"),
    ],
    'hardcoded_secret': [
        (r'(?i)(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|private[_-]?key|auth[_-]?token|client[_-]?secret)\s*=\s*["\'][a-zA-Z0-9_\-+/=]{12,}["\']', "Hardcoded credential"),
        (r'AKIA[0-9A-Z]{16}', "AWS access key ID"),
        (r'sk-[a-zA-Z0-9_\-]{20,}', "OpenAI/Anthropic-style API key"),
        (r'ghp_[a-zA-Z0-9]{36}', "GitHub personal access token"),
        (r'xox[baprs]-[0-9a-zA-Z\-]{10,}', "Slack token"),
        (r'-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----', "Embedded private key"),
    ],
    'race_condition': [
        (r'os\.path\.exists\([^)]+\)[^\n]*\n[^\n]*open\s*\(', "TOCTOU: exists()-then-open()"),
        (r'os\.access\([^)]+\)[^\n]*\n[^\n]*open\s*\(', "TOCTOU: access()-then-open()"),
    ],
    'insecure_temp': [
        (r'\btempfile\.mktemp\s*\(', "tempfile.mktemp is insecure (race condition)"),
    ],
    # missing_auth handled separately via decorator analysis (not regex)
}

# Severity per vuln class
DEFAULT_SEVERITY = {
    'sql_injection': Severity.CRITICAL,
    'command_injection': Severity.CRITICAL,
    'unsafe_deserialization': Severity.CRITICAL,
    'path_traversal': Severity.HIGH,
    'ssrf': Severity.HIGH,
    'xss': Severity.HIGH,
    'hardcoded_secret': Severity.HIGH,
    'open_redirect': Severity.MEDIUM,
    'weak_crypto': Severity.MEDIUM,
    'race_condition': Severity.MEDIUM,
    'insecure_temp': Severity.LOW,
    'missing_auth': Severity.HIGH,
}

FIX_SUGGESTIONS = {
    'sql_injection': "Use parameterized queries: cursor.execute('SELECT ... WHERE id = %s', (value,))",
    'command_injection': "Avoid eval/exec. Use subprocess with shell=False and a list of args. Validate inputs against allowlist.",
    'unsafe_deserialization': "Replace pickle with json. Use yaml.safe_load() instead of yaml.load(). Never deserialize untrusted data.",
    'path_traversal': "Sanitize: filename = os.path.basename(filename). Validate against allowed directory: Path(filename).resolve().is_relative_to(safe_dir).",
    'ssrf': "Validate URLs against an allowlist of permitted hosts/schemes. Block private IP ranges (10.*, 172.16-31.*, 192.168.*, 169.254.*).",
    'xss': "Use template engines with auto-escaping (Jinja2 default). Never concatenate user input into HTML or use Markup() on it.",
    'open_redirect': "Validate redirect targets against an allowlist of permitted URLs/paths.",
    'weak_crypto': "Use hashlib.sha256 or higher for hashing. Use the secrets module for crypto-grade randomness. Use AES-GCM for symmetric encryption.",
    'hardcoded_secret': "Move to environment variables or a secrets manager (HashiCorp Vault, AWS Secrets Manager). Rotate the exposed credential immediately.",
    'race_condition': "Use atomic operations: try/except around open() instead of exists()-then-open(). Use os.O_EXCL for file creation.",
    'insecure_temp': "Use tempfile.NamedTemporaryFile() or tempfile.mkstemp() instead of mktemp().",
    'missing_auth': "Add an authentication decorator (e.g. @login_required, @require_auth) or middleware to the route handler.",
}

# Comment markers per language (skip lines starting with these)
COMMENT_PREFIXES = ('#', '//', '/*', '*', '"""', "'''")


# ═══════════════════════════════════════════════════════════════════
# Sentinel Engine
# ═══════════════════════════════════════════════════════════════════

class SentinelEngine:
    """
    AST-grounded autonomous security analyzer.

    Uses the project brain's dependency graph to trace data flow from
    external input sources to dangerous sinks, pattern-matching against
    12+ vulnerability classes.
    """

    def __init__(self, brain, project_root: str = None, model_fn=None, vuln_dir: str = None):
        """
        brain:        ProjectBrain instance (must be scanned first)
        project_root: project root path (defaults to brain.config.project_path)
        model_fn:     optional callable(prompt: str) -> str for model validation
        vuln_dir:     directory to persist vulnerability findings
        """
        self.brain = brain
        self.project_root = project_root or brain.config.project_path
        self.model_fn = model_fn
        self.vuln_dir = vuln_dir or os.path.join(
            os.environ.get('LEANAI_HOME', os.path.join(str(Path.home()), '.leanai')),
            'vulns'
        )
        os.makedirs(self.vuln_dir, exist_ok=True)

        self._source_cache: Dict[str, str] = {}
        self._function_source_cache: Dict[str, str] = {}

    # ─────────── Public API ───────────

    def scan(self, target: str = None, severity_floor: Severity = Severity.LOW,
             use_model: bool = False, verbose: bool = True) -> Tuple[List[Vulnerability], SentinelStats]:
        """
        Run a security scan.

        target:         specific filepath to scan, or None for full project
        severity_floor: minimum severity to report
        use_model:      ask the model to validate high-confidence findings
        verbose:        print progress

        Returns (findings, stats)
        """
        start = time.time()
        stats = SentinelStats()
        findings: List[Vulnerability] = []

        # Choose targets
        if target:
            rel_target = self._resolve_to_rel(target)
            if rel_target not in self.brain._file_analyses:
                if verbose:
                    print(f"[Sentinel] File not in brain index: {target}")
                return findings, stats
            file_analyses = {rel_target: self.brain._file_analyses[rel_target]}
        else:
            file_analyses = self.brain._file_analyses

        if verbose:
            print(f"[Sentinel] Analyzing {len(file_analyses)} files for vulnerabilities...")

        # Pass 1: identify entry points (sources)
        entry_points = self._find_entry_points(file_analyses)
        stats.sources_found = len(entry_points)

        # Pass 2: identify dangerous sinks
        sinks = self._find_sinks(file_analyses)
        stats.sinks_found = len(sinks)

        if verbose:
            print(f"[Sentinel] Found {len(entry_points)} input sources, {len(sinks)} dangerous sinks")

        # Pass 3: per-function pattern matching (catches direct vulns)
        for rel_path, analysis in file_analyses.items():
            stats.files_scanned += 1
            for func in analysis.functions:
                stats.functions_analyzed += 1
                func_findings = self._analyze_function(rel_path, func)
                for f in func_findings:
                    if f.severity.value >= severity_floor.value:
                        findings.append(f)

        # Pass 4: file-level scans (hardcoded secrets, module-level)
        for rel_path in file_analyses.keys():
            file_findings = self._analyze_file_level(rel_path)
            for f in file_findings:
                if f.severity.value >= severity_floor.value:
                    findings.append(f)

        # Pass 5: taint flow tracing — connect sources to sinks
        for source_func_id in entry_points:
            paths = self._trace_taint_flow(source_func_id, sinks, max_depth=5)
            for path, sink_info in paths:
                stats.taint_paths_traced += 1
                vuln = self._build_taint_vulnerability(source_func_id, path, sink_info, entry_points)
                if vuln and vuln.severity.value >= severity_floor.value:
                    findings.append(vuln)

        # Deduplicate
        findings = self._deduplicate(findings)

        # Pass 6 (optional): model-augmented validation
        if use_model and self.model_fn:
            if verbose:
                print(f"[Sentinel] Validating {len(findings)} findings with model...")
            findings = self._validate_with_model(findings)

        # Assign stable IDs and persist
        findings = self._assign_ids(findings)
        self._persist(findings)

        # Final stats
        stats.vulnerabilities_found = len(findings)
        stats.time_ms = (time.time() - start) * 1000
        for f in findings:
            sev_name = str(f.severity)
            stats.by_severity[sev_name] = stats.by_severity.get(sev_name, 0) + 1

        return findings, stats

    # ─────────── Source / Sink discovery ───────────

    def _find_entry_points(self, file_analyses) -> Dict[str, dict]:
        """Find functions that take external/untrusted input."""
        entry_points = {}

        for rel_path, analysis in file_analyses.items():
            try:
                source = self._read_file(rel_path)
            except Exception:
                continue

            for func in analysis.functions:
                func_source = self._extract_function_source(source, func)
                if not func_source:
                    continue

                input_source_type = None

                # Check decorators for HTTP handler patterns
                for decorator in func.decorators:
                    dec_lower = decorator.lower()
                    if any(p in dec_lower for p in ['route', '.get', '.post', '.put', '.delete', '.patch', 'app.', 'router.', 'api.']):
                        input_source_type = 'http_input'
                        break

                # Check function body for input source patterns
                if not input_source_type:
                    for source_type, patterns in SOURCE_PATTERNS.items():
                        for pat in patterns:
                            if re.search(pat, func_source):
                                input_source_type = source_type
                                break
                        if input_source_type:
                            break

                if input_source_type:
                    func_id = self._get_func_id(rel_path, func)
                    entry_points[func_id] = {
                        'rel_path': rel_path,
                        'function': func,
                        'source_type': input_source_type,
                    }

        return entry_points

    def _find_sinks(self, file_analyses) -> Dict[str, dict]:
        """Find functions that contain dangerous operations."""
        sinks = {}

        for rel_path, analysis in file_analyses.items():
            try:
                source = self._read_file(rel_path)
            except Exception:
                continue

            for func in analysis.functions:
                func_source = self._extract_function_source(source, func)
                if not func_source:
                    continue

                # Strip strings/docstrings/comments before matching to avoid
                # false positives from strings that contain dangerous patterns
                # (e.g. "os.system(" in a forbidden-patterns list, or eval()
                # mentioned in a docstring).
                cleaned = self._strip_strings_and_comments(func_source)
                func_sinks = []
                for vuln_class, patterns in SINK_PATTERNS.items():
                    for pat, desc in patterns:
                        for match in re.finditer(pat, cleaned):
                            # Skip safely-flagged MD5 calls
                            if vuln_class == 'weak_crypto' and self._is_safe_md5(func_source, match.start()):
                                continue
                            func_sinks.append((vuln_class, desc))
                            break  # one match per pattern is enough for sink presence

                if func_sinks:
                    func_id = self._get_func_id(rel_path, func)
                    sinks[func_id] = {
                        'rel_path': rel_path,
                        'function': func,
                        'sink_types': func_sinks,
                    }

        return sinks

    # ─────────── Per-function analysis ───────────

    def _analyze_function(self, rel_path, func) -> List[Vulnerability]:
        """Pattern-match within a single function for direct vulnerabilities."""
        findings = []
        try:
            source = self._read_file(rel_path)
        except Exception:
            return findings

        func_source = self._extract_function_source(source, func)
        if not func_source:
            return findings

        cleaned = self._strip_strings_and_comments(func_source)

        # Missing auth check (HTTP routes without auth decorators)
        is_http_route = any(
            any(p in d.lower() for p in ['route', '.get', '.post', '.put', '.delete', '.patch'])
            for d in func.decorators
        )
        if is_http_route:
            has_auth = any(
                any(a in d.lower() for a in ['auth', 'login_required', 'requires_auth', 'permission', 'token_required', 'jwt_required'])
                for d in func.decorators
            )
            if not has_auth:
                findings.append(Vulnerability(
                    vuln_id='',
                    vuln_class='missing_auth',
                    severity=DEFAULT_SEVERITY['missing_auth'],
                    confidence=0.6,
                    filepath=rel_path,
                    function_name=func.qualified_name,
                    line=func.line_start,
                    source_type='http_input',
                    sink_type='unauthenticated_endpoint',
                    description=f"HTTP handler '{func.qualified_name}' has no authentication decorator",
                    code_snippet=self._snippet_lines(func_source, 0, 3),
                    taint_path=[func.qualified_name],
                    fix_suggestion=FIX_SUGGESTIONS['missing_auth'],
                ))

        # Match each sink pattern with context-aware classification
        for vuln_class, patterns in SINK_PATTERNS.items():
            for pat, desc in patterns:
                for match in re.finditer(pat, cleaned):
                    line_offset = cleaned[:match.start()].count('\n')
                    vuln_line = func.line_start + line_offset
                    snippet = self._snippet_lines(func_source, line_offset, 2)

                    # ─── Context analysis (Mythos-style) ───
                    # Look at the ORIGINAL source (with strings intact) for context
                    severity = DEFAULT_SEVERITY.get(vuln_class, Severity.MEDIUM)
                    confidence = 0.6  # default for direct match without taint flow
                    description = f"Potential {vuln_class.replace('_', ' ')}: {desc}"
                    skip = False

                    # 1. Sandboxed eval/exec → downgrade or skip
                    if vuln_class == 'command_injection' and ('eval' in desc or 'exec' in desc):
                        sandbox_kind = self._detect_sandbox(func_source, match.start())
                        if sandbox_kind:
                            severity = Severity.LOW
                            confidence = 0.4
                            description = f"{vuln_class.replace('_', ' ').capitalize()}: {desc} (mitigated: {sandbox_kind})"

                    # 2. MD5 with usedforsecurity=False → skip entirely
                    if vuln_class == 'weak_crypto' and 'MD5' in desc:
                        if self._is_safe_md5(func_source, match.start()):
                            skip = True
                        else:
                            # Check if used in obvious non-security context (cache key / id / fingerprint)
                            md5_context = self._md5_context(func.qualified_name, func_source)
                            if md5_context == 'non_security':
                                severity = Severity.LOW
                                confidence = 0.4
                                description = f"{desc} — appears used as cache/ID hash, not crypto"

                    # 3. SHA1 same logic as MD5
                    if vuln_class == 'weak_crypto' and 'SHA1' in desc:
                        sha_context = self._md5_context(func.qualified_name, func_source)  # same heuristic
                        if sha_context == 'non_security':
                            severity = Severity.LOW
                            confidence = 0.4

                    if skip:
                        continue

                    findings.append(Vulnerability(
                        vuln_id='',
                        vuln_class=vuln_class,
                        severity=severity,
                        confidence=confidence,
                        filepath=rel_path,
                        function_name=func.qualified_name,
                        line=vuln_line,
                        source_type='unknown',
                        sink_type=desc,
                        description=description,
                        code_snippet=snippet,
                        taint_path=[func.qualified_name],
                        fix_suggestion=FIX_SUGGESTIONS.get(vuln_class, ""),
                    ))

        return findings

    def _analyze_file_level(self, rel_path) -> List[Vulnerability]:
        """Module-level scans (hardcoded secrets, weak crypto)."""
        findings = []
        try:
            source = self._read_file(rel_path)
        except Exception:
            return findings

        # Strip strings/comments before pattern matching (avoid false positives
        # from regex pattern strings that themselves contain dangerous patterns)
        cleaned_source = self._strip_strings_and_comments(source)

        # Skip test/example files for noisy classes
        is_test_or_example = any(
            t in rel_path.lower().replace('\\', '/').split('/')
            for t in ('tests', 'test', 'examples', 'example')
        )

        for vuln_class in ['hardcoded_secret', 'weak_crypto']:
            if vuln_class == 'hardcoded_secret' and is_test_or_example:
                continue
            # weak_crypto in tests is also noisy — likely test fixtures
            if vuln_class == 'weak_crypto' and is_test_or_example:
                continue

            for pat, desc in SINK_PATTERNS.get(vuln_class, []):
                # For hardcoded_secret, search original source (string contents matter)
                # For weak_crypto, search cleaned source (skip strings)
                search_in = source if vuln_class == 'hardcoded_secret' else cleaned_source
                for match in re.finditer(pat, search_in):
                    line = search_in[:match.start()].count('\n') + 1
                    src_lines = source.splitlines()
                    line_text = src_lines[line - 1] if line <= len(src_lines) else ""

                    # Skip if comment
                    if line_text.lstrip().startswith(COMMENT_PREFIXES):
                        continue

                    severity = DEFAULT_SEVERITY.get(vuln_class, Severity.MEDIUM)
                    confidence = 0.8
                    description = f"{vuln_class.replace('_', ' ').title()}: {desc}"

                    # MD5/SHA1 with usedforsecurity=False → skip
                    if vuln_class == 'weak_crypto' and ('MD5' in desc or 'SHA1' in desc):
                        if self._is_safe_md5(source, match.start()):
                            continue

                    findings.append(Vulnerability(
                        vuln_id='',
                        vuln_class=vuln_class,
                        severity=severity,
                        confidence=confidence,
                        filepath=rel_path,
                        function_name='<module>',
                        line=line,
                        source_type='source_code',
                        sink_type=desc,
                        description=description,
                        code_snippet=line_text.strip()[:140],
                        taint_path=['<module>'],
                        fix_suggestion=FIX_SUGGESTIONS.get(vuln_class, ""),
                    ))

        return findings

    # ─────────── Taint flow tracing ───────────

    def _trace_taint_flow(self, source_func_id, sinks, max_depth=5) -> List[Tuple[List[str], dict]]:
        """BFS from a source function through the call graph to reach sinks."""
        results = []
        if source_func_id not in self.brain.graph._adjacency:
            return results

        visited = set()
        queue = [(source_func_id, [source_func_id])]

        while queue:
            current, path = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if len(path) > max_depth:
                continue

            # Check if current is a sink
            if current in sinks and len(path) > 1:
                name_path = []
                for node_id in path:
                    node = self.brain.graph.nodes.get(node_id)
                    if node:
                        name_path.append(node.name)
                    else:
                        name_path.append(node_id.split(':')[-1])
                results.append((name_path, sinks[current]))

            # Continue BFS
            for target in self.brain.graph._adjacency.get(current, []):
                if target not in visited:
                    queue.append((target, path + [target]))

        return results

    def _build_taint_vulnerability(self, source_func_id, path, sink_info, entry_points) -> Optional[Vulnerability]:
        """Build a Vulnerability from a traced taint path."""
        if not sink_info or not sink_info.get('sink_types'):
            return None

        sink_func = sink_info['function']
        rel_path = sink_info['rel_path']
        vuln_class, desc = sink_info['sink_types'][0]

        source_info = entry_points.get(source_func_id, {})
        source_type = source_info.get('source_type', 'unknown')

        return Vulnerability(
            vuln_id='',
            vuln_class=vuln_class,
            severity=DEFAULT_SEVERITY.get(vuln_class, Severity.MEDIUM),
            confidence=0.85,
            filepath=rel_path,
            function_name=sink_func.qualified_name,
            line=sink_func.line_start,
            source_type=f'traced:{source_type}',
            sink_type=desc,
            description=(
                f"Taint flow ({source_type}): untrusted input reaches "
                f"{desc} via {' → '.join(path)}"
            ),
            code_snippet=f"sink at {rel_path}:{sink_func.line_start} in {sink_func.qualified_name}",
            taint_path=path,
            fix_suggestion=FIX_SUGGESTIONS.get(vuln_class, ""),
        )

    # ─────────── Model validation ───────────

    def _validate_with_model(self, findings) -> List[Vulnerability]:
        """For each high-confidence finding, ask the model: is this real?"""
        if not self.model_fn:
            return findings

        validated = []
        for f in findings:
            if f.confidence < 0.7:
                validated.append(f)
                continue

            prompt = (
                f"Security audit. Is this a real, exploitable vulnerability? "
                f"Answer with: YES, NO, or UNCLEAR. Then one short sentence why.\n\n"
                f"Class: {f.vuln_class}\n"
                f"File: {f.filepath}:{f.line}\n"
                f"Function: {f.function_name}\n"
                f"Code:\n{f.code_snippet}\n"
            )
            try:
                response = (self.model_fn(prompt) or "")[:200].strip().lower()
                if response.startswith('no') or 'false positive' in response:
                    f.confidence *= 0.4
                elif response.startswith('yes'):
                    f.confidence = min(1.0, f.confidence * 1.15)
            except Exception:
                pass

            validated.append(f)

        # Drop very low confidence
        return [f for f in validated if f.confidence >= 0.4]

    # ─────────── Helpers ───────────

    def _read_file(self, rel_path: str) -> str:
        if rel_path in self._source_cache:
            return self._source_cache[rel_path]
        full = os.path.join(self.project_root, rel_path)
        with open(full, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        self._source_cache[rel_path] = text
        return text

    def _extract_function_source(self, source: str, func) -> str:
        """Extract a function's source text using its line numbers."""
        cache_key = f"{func.filepath}:{func.qualified_name}:{func.line_start}"
        if cache_key in self._function_source_cache:
            return self._function_source_cache[cache_key]

        lines = source.splitlines()
        start = max(0, func.line_start - 1)
        end = min(len(lines), func.line_end if func.line_end else func.line_start + 50)
        text = '\n'.join(lines[start:end])

        self._function_source_cache[cache_key] = text
        return text

    def _strip_strings_and_comments(self, source: str) -> str:
        """
        Replace all string literals (single/double/triple-quoted) and comments
        with spaces, preserving line numbers and column positions so regex
        matches map back to original line numbers correctly.

        This eliminates false positives from:
          - Dangerous patterns mentioned in docstrings (e.g. eval() in docs)
          - Strings stored in lists like FORBIDDEN = ["os.system(", "eval("]
          - URLs and example code in comments

        This is the Mythos-like move: don't flag what's clearly text, only
        flag what's actually executed.
        """
        out = []
        i = 0
        n = len(source)
        while i < n:
            c = source[i]

            # Comment to end-of-line
            if c == '#':
                end = source.find('\n', i)
                if end == -1:
                    end = n
                out.append(' ' * (end - i))
                i = end
                continue

            # Triple-quoted string (must check before single-quote)
            if c in ('"', "'") and i + 2 < n and source[i:i + 3] in ('"""', "'''"):
                quote = source[i:i + 3]
                out.append('   ')  # the opening triple
                i += 3
                end = source.find(quote, i)
                if end == -1:
                    # Unterminated — blank to EOF, preserve newlines
                    chunk = source[i:n]
                    out.append(''.join(' ' if ch != '\n' else '\n' for ch in chunk))
                    i = n
                else:
                    chunk = source[i:end]
                    out.append(''.join(' ' if ch != '\n' else '\n' for ch in chunk))
                    out.append('   ')  # the closing triple
                    i = end + 3
                continue

            # Single-line string (handles escape sequences)
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
                        # Unterminated string — preserve newline
                        out.append('\n')
                        i += 1
                        break
                    out.append(' ')
                    i += 1
                continue

            out.append(c)
            i += 1
        return ''.join(out)

    def _extract_call_args(self, source: str, match_pos: int, max_lookahead: int = 600) -> str:
        """
        Given a position in source where a function-call match started,
        return the text of the call's arguments (everything between
        the matching parens). Used by sandbox / safe-md5 detection.
        """
        # Find the '(' starting at or after match_pos
        i = match_pos
        end_search = min(n_total := len(source), match_pos + 80)
        while i < end_search and source[i] != '(':
            i += 1
        if i >= end_search:
            return ''
        # Now scan to the matching ')'
        start = i + 1
        depth = 0
        in_str = None
        escape = False
        j = i
        scan_end = min(n_total, match_pos + max_lookahead)
        while j < scan_end:
            ch = source[j]
            if escape:
                escape = False
                j += 1
                continue
            if in_str:
                if ch == '\\':
                    escape = True
                elif ch == in_str:
                    in_str = None
                j += 1
                continue
            if ch in ('"', "'"):
                in_str = ch
                j += 1
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return source[start:j]
            j += 1
        return source[start:scan_end]

    def _detect_sandbox(self, source: str, match_pos: int) -> Optional[str]:
        """
        Check if an eval/exec call is sandboxed.

        Sandbox markers:
          - {"__builtins__": {}} or {"__builtins__": None}
          - safe_globals / restricted_globals as a kwarg
          - explicit dict with no __builtins__

        Returns a description like 'restricted globals' or None.
        """
        args = self._extract_call_args(source, match_pos)
        if not args:
            return None
        # Empty/restricted builtins
        if re.search(r'__builtins__\s*[":]\s*\{\s*\}', args):
            return 'empty __builtins__'
        if re.search(r'__builtins__\s*[":]\s*None', args):
            return 'None __builtins__'
        if re.search(r'\bsafe_globals\b', args):
            return 'safe_globals'
        if re.search(r'\brestricted_globals\b', args):
            return 'restricted_globals'
        # An explicit dict literal as second arg suggests restricted env
        if re.search(r',\s*\{[^}]*"__builtins__"', args):
            return 'restricted globals dict'
        return None

    def _is_safe_md5(self, source: str, match_pos: int) -> bool:
        """Check if a hashlib.md5/sha1 call has usedforsecurity=False kwarg."""
        args = self._extract_call_args(source, match_pos)
        if not args:
            return False
        return bool(re.search(r'usedforsecurity\s*=\s*False', args))

    def _md5_context(self, func_name: str, func_source: str) -> str:
        """
        Determine if MD5 is being used in a security-sensitive context.

        Returns:
          'high'         — likely security use (password, auth, signature near call)
          'non_security' — clearly cache/ID/fingerprint use
          'unknown'      — can't tell; default behavior
        """
        fn_lower = func_name.lower()
        # Function names that scream "hashing for non-security purposes"
        non_sec_names = (
            '_hash', 'make_id', '_make_id', '_id', 'cache', 'fingerprint',
            'content_hash', 'generate_id', '_generate_id', 'dedupe', 'dedup',
            '_make_pair', '_keyword_hash', 'store',
        )
        if any(n in fn_lower for n in non_sec_names):
            return 'non_security'
        # Function names that scream security
        sec_names = ('password', 'auth', 'token', 'signature', 'sign_', '_sign', 'verify_', 'hmac')
        if any(n in fn_lower for n in sec_names):
            return 'high'
        # Look for context words near the function body
        body_lower = func_source.lower()
        if any(w in body_lower for w in ('password', 'authenticate', 'jwt', 'signature', 'hmac', 'crypto')):
            return 'high'
        return 'unknown'

    def _snippet_lines(self, text: str, line_offset: int, context: int) -> str:
        lines = text.splitlines()
        start = max(0, line_offset - context)
        end = min(len(lines), line_offset + context + 1)
        out = []
        for i in range(start, end):
            marker = '>' if i == line_offset else ' '
            out.append(f"{marker} {lines[i].rstrip()[:140]}")
        return '\n'.join(out)

    def _get_func_id(self, rel_path: str, func) -> str:
        """Build the brain's function node id (matches DependencyGraph format)."""
        return f"{rel_path}:{func.qualified_name}"

    def _resolve_to_rel(self, target: str) -> str:
        """Resolve a target to a brain-keyed relative path."""
        if os.path.isabs(target):
            try:
                return os.path.relpath(target, self.project_root)
            except ValueError:
                pass
        for rel_path in self.brain._file_analyses.keys():
            if rel_path == target:
                return rel_path
            if rel_path.replace('\\', '/').endswith(target.replace('\\', '/')):
                return rel_path
            if target.lower() in rel_path.lower():
                return rel_path
        return target

    def _deduplicate(self, findings: List[Vulnerability]) -> List[Vulnerability]:
        """Remove duplicates (same file + line + class)."""
        seen = set()
        out = []
        for f in findings:
            key = (f.filepath, f.line, f.vuln_class)
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    def _assign_ids(self, findings: List[Vulnerability]) -> List[Vulnerability]:
        """Assign stable IDs (VULN-YYYY-NNNN), preserving across scans via fingerprint."""
        existing_by_fingerprint = {}
        try:
            for fname in os.listdir(self.vuln_dir):
                if fname.startswith('VULN-') and fname.endswith('.json'):
                    fpath = os.path.join(self.vuln_dir, fname)
                    try:
                        with open(fpath, 'r') as fh:
                            data = json.load(fh)
                        if 'fingerprint' in data:
                            existing_by_fingerprint[data['fingerprint']] = fname[:-5]
                    except Exception:
                        pass
        except FileNotFoundError:
            pass

        # Determine next number
        max_num = 0
        for vid in existing_by_fingerprint.values():
            try:
                n = int(vid.split('-')[-1])
                if n > max_num:
                    max_num = n
            except (ValueError, IndexError):
                pass
        next_num = max_num + 1

        year = time.strftime('%Y')

        # Sort for stable ordering
        findings.sort(key=lambda f: (-f.severity.value, f.filepath, f.line))

        for f in findings:
            fingerprint = hashlib.md5(
                f"{f.filepath}:{f.line}:{f.vuln_class}".encode()
            ).hexdigest()[:10]

            if fingerprint in existing_by_fingerprint:
                f.vuln_id = existing_by_fingerprint[fingerprint]
            else:
                f.vuln_id = f"VULN-{year}-{next_num:04d}"
                next_num += 1

        return findings

    def _persist(self, findings: List[Vulnerability]):
        """Save each finding to disk for ChainBreaker/ExploitForge integration."""
        for f in findings:
            data = f.to_dict()
            data['fingerprint'] = hashlib.md5(
                f"{f.filepath}:{f.line}:{f.vuln_class}".encode()
            ).hexdigest()[:10]
            data['timestamp'] = time.time()

            fpath = os.path.join(self.vuln_dir, f.vuln_id + '.json')
            try:
                with open(fpath, 'w', encoding='utf-8') as fh:
                    json.dump(data, fh, indent=2)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════
# Formatting helper (called from main.py)
# ═══════════════════════════════════════════════════════════════════

def format_findings_report(findings: List[Vulnerability], stats: SentinelStats, color: bool = True) -> str:
    """Format findings into a human-readable terminal report."""
    if color:
        try:
            from core.terminal_ui import C
            DIM = C.DIM
            RESET = C.RESET
            BOLD = getattr(C, 'BOLD', '\033[1m')
            RED = getattr(C, 'RED', '\033[31m')
            YELLOW = getattr(C, 'YELLOW', '\033[33m')
            CYAN = getattr(C, 'CYAN', '\033[36m')
            GREEN = getattr(C, 'GREEN', '\033[32m')
        except Exception:
            DIM = RESET = BOLD = RED = YELLOW = CYAN = GREEN = ''
    else:
        DIM = RESET = BOLD = RED = YELLOW = CYAN = GREEN = ''

    SEV_COLORS = {
        'CRITICAL': RED,
        'HIGH': RED,
        'MEDIUM': YELLOW,
        'LOW': CYAN,
        'INFO': DIM,
    }

    lines = []
    lines.append("")
    lines.append(f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    lines.append(f"{BOLD}  Sentinel Security Analysis{RESET}")
    lines.append(f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    lines.append("")

    if not findings:
        lines.append(f"  {GREEN}✓ No vulnerabilities found.{RESET}")
        lines.append("")
        lines.append(f"  {DIM}Files scanned: {stats.files_scanned}")
        lines.append(f"  Functions analyzed: {stats.functions_analyzed}")
        lines.append(f"  Sources identified: {stats.sources_found}")
        lines.append(f"  Sinks identified: {stats.sinks_found}")
        lines.append(f"  Time: {stats.time_ms:.0f}ms{RESET}")
        return '\n'.join(lines)

    sev_summary = '  '.join(
        f"{SEV_COLORS.get(s, '')}{c} {s}{RESET}"
        for s, c in sorted(stats.by_severity.items(), key=lambda x: -Severity[x[0]].value)
    )
    lines.append(f"  Found {BOLD}{len(findings)}{RESET} potential issues   {sev_summary}")
    lines.append(f"  {DIM}{stats.files_scanned} files, {stats.functions_analyzed} functions, "
                 f"{stats.sources_found} sources, {stats.sinks_found} sinks, "
                 f"{stats.taint_paths_traced} traced paths, {stats.time_ms:.0f}ms{RESET}")
    lines.append("")

    # Group by severity
    by_sev = {}
    for f in findings:
        by_sev.setdefault(str(f.severity), []).append(f)

    for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
        if sev not in by_sev:
            continue
        c = SEV_COLORS.get(sev, '')
        lines.append(f"{c}{BOLD}── {sev} ({len(by_sev[sev])}) ──{RESET}")
        for f in by_sev[sev]:
            lines.append("")
            lines.append(f"  {c}[{f.vuln_id}] {f.vuln_class.replace('_', ' ').title()}{RESET}")
            lines.append(f"    {DIM}{f.filepath}:{f.line}  in  {f.function_name}{RESET}")
            lines.append(f"    {f.description}")
            if f.code_snippet:
                for sline in f.code_snippet.splitlines()[:5]:
                    lines.append(f"    {DIM}{sline}{RESET}")
            if f.taint_path and len(f.taint_path) > 1:
                lines.append(f"    {DIM}Taint path: {' → '.join(f.taint_path)}{RESET}")
            lines.append(f"    {GREEN}Fix:{RESET} {f.fix_suggestion}")
            lines.append(f"    {DIM}Confidence: {f.confidence:.0%}{RESET}")
        lines.append("")

    return '\n'.join(lines)
