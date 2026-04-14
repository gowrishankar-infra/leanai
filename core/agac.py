"""
LeanAI — AGAC: AST-Grounded Auto-Correction
+ PARE: Project-Aware Response Enrichment
═══════════════════════════════════════════════

AGAC auto-corrects ONLY clear typos/case errors in code blocks.
It does NOT:
  - Touch prose or comments
  - "Improve" the model's new code suggestions
  - Correct standard library or framework names
  - Force-match loosely similar names

STRICT RULES:
  1. Only correct inside ```code blocks``` — never in prose
  2. Only correct if similarity >= 85% (obvious typos only)
  3. Never correct names the model is INVENTING (new methods/classes)
  4. Never correct standard library, framework, or common names
  5. Never correct inside comments (# lines)

PARE enriches responses with verified project references:
  - File paths for mentioned functions/classes
  - "Used by" references from dependency graph

Author: Gowri Shankar (github.com/gowrishankar-infra/leanai)
"""

import re
import os
import difflib
from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass, field


@dataclass
class AGACStats:
    identifiers_scanned: int = 0
    corrections_made: int = 0
    functions_corrected: int = 0
    classes_corrected: int = 0
    imports_corrected: int = 0
    enrichments_added: int = 0
    time_ms: float = 0.0

    def summary(self) -> str:
        if self.corrections_made == 0 and self.enrichments_added == 0:
            return ""
        parts = []
        if self.corrections_made > 0:
            parts.append(f"{self.corrections_made} auto-corrected")
        if self.enrichments_added > 0:
            parts.append(f"{self.enrichments_added} enriched")
        return f"AGAC: {', '.join(parts)} ({self.time_ms:.0f}ms)"


class AGACEngine:
    """
    AST-Grounded Auto-Correction + Project-Aware Response Enrichment.

    ONLY corrects obvious typos in code blocks. Never touches prose,
    comments, standard library names, or new code the model is inventing.
    """

    # Minimum similarity to consider a correction (0.85 = only obvious typos)
    FUNC_THRESHOLD = 0.85
    CLASS_THRESHOLD = 0.85
    IMPORT_THRESHOLD = 0.80

    def __init__(self, brain=None):
        self.brain = brain
        self._all_functions: Dict[str, dict] = {}
        self._all_classes: Dict[str, dict] = {}
        self._all_files: Set[str] = set()
        self._func_names_lower: Dict[str, str] = {}
        self._class_names_lower: Dict[str, str] = {}
        self._indexed = False
        self._cumulative = AGACStats()

    def update_brain(self, brain):
        self.brain = brain
        self._indexed = False

    def _build_index(self):
        if self._indexed or not self.brain:
            return

        self._all_functions.clear()
        self._all_classes.clear()
        self._all_files.clear()
        self._func_names_lower.clear()
        self._class_names_lower.clear()

        # Index functions
        if hasattr(self.brain, 'graph') and hasattr(self.brain.graph, '_function_lookup'):
            for func_name, node_id in self.brain.graph._function_lookup.items():
                node = self.brain.graph.nodes.get(node_id)
                if node:
                    self._all_functions[func_name] = {
                        "file": node.filepath or "",
                        "args": node.metadata.get("args", []),
                        "node_id": node_id,
                    }
                    self._func_names_lower[func_name.lower()] = func_name
                else:
                    self._all_functions[func_name] = {"file": "", "args": [], "node_id": node_id}
                    self._func_names_lower[func_name.lower()] = func_name

        # Index classes
        if hasattr(self.brain, 'graph'):
            for node_id, node in self.brain.graph.nodes.items():
                if node.node_type == "class":
                    name = node.name
                    self._all_classes[name] = {
                        "file": node.filepath or "",
                        "node_id": node_id,
                    }
                    self._class_names_lower[name.lower()] = name

        # Index files
        if hasattr(self.brain, '_file_analyses'):
            for path in self.brain._file_analyses:
                self._all_files.add(os.path.basename(path))

        self._indexed = True

    def process(self, response: str, query: str = "") -> Tuple[str, AGACStats]:
        """Process a response: auto-correct code blocks + enrich."""
        import time
        start = time.time()
        stats = AGACStats()

        if not self.brain:
            return response, stats

        self._build_index()

        if not self._all_functions and not self._all_classes:
            return response, stats

        # Step 1: AGAC — Auto-correct ONLY inside code blocks
        response, stats = self._auto_correct_code_blocks(response, stats)

        # Step 2: PARE — Enrich with verified project references
        response, stats = self._enrich_response(response, stats)

        stats.time_ms = (time.time() - start) * 1000

        # Update cumulative
        self._cumulative.identifiers_scanned += stats.identifiers_scanned
        self._cumulative.corrections_made += stats.corrections_made
        self._cumulative.functions_corrected += stats.functions_corrected
        self._cumulative.classes_corrected += stats.classes_corrected
        self._cumulative.imports_corrected += stats.imports_corrected
        self._cumulative.enrichments_added += stats.enrichments_added
        self._cumulative.time_ms += stats.time_ms

        return response, stats

    # ── AGAC: Auto-Correct Code Blocks ONLY ────────────────────────

    def _auto_correct_code_blocks(self, text: str, stats: AGACStats) -> Tuple[str, AGACStats]:
        """Find code blocks and auto-correct OBVIOUS typos inside them."""
        corrections_log = []

        # ONLY correct inside ```code blocks``` — NEVER in prose text
        def correct_block(match):
            lang_tag = match.group(1) or ""
            code = match.group(2)
            corrected, block_corrections = self._correct_code(code, stats)
            corrections_log.extend(block_corrections)
            return f"```{lang_tag}\n{corrected}\n```"

        pattern = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)
        text = pattern.sub(correct_block, text)

        # NO inline code correction — prose text is NEVER touched

        # Add corrections footer only if real corrections were made
        if corrections_log:
            footer = "\n\n> **Auto-corrected** (verified against your project's AST):\n"
            for old, new, reason in corrections_log[:5]:
                footer += f">   `{old}` → `{new}` — {reason}\n"
            text = text.rstrip() + footer

        stats.corrections_made = len(corrections_log)
        return text, stats

    def _correct_code(self, code: str, stats: AGACStats) -> Tuple[str, List[Tuple[str, str, str]]]:
        """Correct ONLY obvious typos in a code block."""
        corrections = []

        # Split code into lines to skip comments
        lines = code.split('\n')
        code_lines_only = []
        for line in lines:
            stripped = line.lstrip()
            # Skip comment lines entirely
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('*'):
                code_lines_only.append(None)  # placeholder
            else:
                code_lines_only.append(line)

        # Rejoin non-comment lines for pattern matching
        code_for_matching = '\n'.join(
            line if line is not None else '# AGAC_SKIP'
            for line in code_lines_only
        )

        # ── Fix function calls (STRICT: only exact case-insensitive or >85% match) ──
        func_call_pattern = re.compile(r'\b([a-zA-Z_]\w{3,})\s*\(')
        already_corrected = set()

        for match in func_call_pattern.finditer(code_for_matching):
            name = match.group(1)
            if name in already_corrected:
                continue
            stats.identifiers_scanned += 1

            # Skip if name is in ANY skip list
            if name.lower() in _SKIP_NAMES_LOWER:
                continue

            # Skip if it already exists in the project (correct as-is)
            if name in self._all_functions:
                continue

            # Skip if this looks like the model is INVENTING a new function
            # (contains "new", "custom", "my", or starts with underscore + novel name)
            if self._is_likely_new_code(name):
                continue

            # Only correct if there's a VERY close match (>85% = typo/case error)
            best = self._fuzzy_match_function(name, self.FUNC_THRESHOLD)
            if best and best != name:
                # Extra safety: don't correct if the match is a completely different word
                # (e.g., "Model" → "mode" is a different concept)
                if len(best) < len(name) * 0.6 or len(name) < len(best) * 0.6:
                    continue  # too different in length = different concept

                code = code.replace(name + "(", best + "(", 1)
                corrections.append((name + "()", best + "()", "function name corrected"))
                stats.functions_corrected += 1
                already_corrected.add(name)

        # ── Fix class references (STRICT: only >85% match, skip common words) ──
        class_pattern = re.compile(r'\b([A-Z][a-zA-Z0-9]{3,})\b')
        for match in class_pattern.finditer(code_for_matching):
            name = match.group(1)
            if name in already_corrected:
                continue
            stats.identifiers_scanned += 1

            if name in _SKIP_CLASSES or name.lower() in _SKIP_ENGLISH_WORDS:
                continue

            if name in self._all_classes:
                continue

            if self._is_likely_new_code(name):
                continue

            best = self._fuzzy_match_class(name, self.CLASS_THRESHOLD)
            if best and best != name:
                if len(best) < len(name) * 0.6 or len(name) < len(best) * 0.6:
                    continue
                code = re.sub(r'\b' + re.escape(name) + r'\b', best, code, count=1)
                corrections.append((name, best, "class name corrected"))
                stats.classes_corrected += 1
                already_corrected.add(name)

        # ── Fix import paths (only for project-internal imports) ──────
        import_pattern = re.compile(r'from\s+(core|brain|tools|training|agents|api|memory|hdc|liquid|federated|speculative|swarm)\.([\w.]+)\s+import')
        for match in import_pattern.finditer(code):
            pkg = match.group(1)
            module = match.group(2)
            expected_file = module.split(".")[-1] + ".py"
            stats.identifiers_scanned += 1

            if expected_file in self._all_files:
                continue  # correct import

            # Try to find close match
            best_file = self._fuzzy_match_file(expected_file)
            if best_file and best_file != expected_file:
                corrected_module = module.replace(
                    module.split(".")[-1], best_file.replace(".py", "")
                )
                old_import = f"{pkg}.{module}"
                new_import = f"{pkg}.{corrected_module}"
                code = code.replace(old_import, new_import, 1)
                corrections.append((old_import, new_import, "import path corrected"))
                stats.imports_corrected += 1

        return code, corrections

    def _is_likely_new_code(self, name: str) -> bool:
        """Check if the model is inventing new code (don't correct these)."""
        lower = name.lower()

        # Prefixes that suggest new/custom code
        new_prefixes = ['new_', 'custom_', 'my_', 'create_', 'build_',
                        'setup_', 'init_', 'test_']
        for prefix in new_prefixes:
            if lower.startswith(prefix):
                return True

        # If the name doesn't closely match ANYTHING in the project,
        # the model is probably inventing it on purpose
        if lower not in self._func_names_lower and lower not in self._class_names_lower:
            # Check if even a loose match exists
            best_func = self._closest_string(name, list(self._all_functions.keys()), 0.70)
            best_cls = self._closest_string(name, list(self._all_classes.keys()), 0.70)
            if not best_func and not best_cls:
                return True  # nothing even close — model is inventing

        return False

    # ── PARE: Project-Aware Response Enrichment ────────────────────

    def _enrich_response(self, text: str, stats: AGACStats) -> Tuple[str, AGACStats]:
        """Add verified project references — file paths and 'used by' info."""
        if not self.brain or not hasattr(self.brain, 'graph'):
            return text, stats

        # Find function/class names that ACTUALLY exist in the project
        # and are mentioned in the response
        mentioned = set()
        for name in self._all_functions:
            if len(name) > 4 and name in text:
                mentioned.add(name)

        if not mentioned or len(mentioned) > 10:
            return text, stats

        see_also = []
        for func_name in list(mentioned)[:4]:
            info = self._all_functions.get(func_name)
            if not info:
                continue

            node_id = info.get("node_id", "")
            file_path = info.get("file", "")
            if not file_path:
                continue

            # Only show enrichment for functions that have dependents
            try:
                dependents = self.brain.graph.get_dependents(node_id)
                if dependents and 1 <= len(dependents) <= 5:
                    dep_names = []
                    for dep_id in dependents[:3]:
                        dep_node = self.brain.graph.nodes.get(dep_id)
                        if dep_node and dep_node.name != func_name:
                            dep_names.append(dep_node.name)
                    if dep_names:
                        short_path = os.path.basename(file_path)
                        see_also.append(
                            f"`{func_name}()` in `{short_path}` — "
                            f"used by {', '.join(f'`{d}`' for d in dep_names)}"
                        )
                        stats.enrichments_added += 1
            except Exception:
                pass

        if see_also:
            section = "\n\n▸ **Code Verification** (checked against your project)\n"
            for ref in see_also:
                section += f"  ✓ {ref}\n"
            text = text.rstrip() + section

        return text, stats

    # ── Fuzzy Matching (STRICT) ────────────────────────────────────

    def _fuzzy_match_function(self, name: str, threshold: float = 0.85) -> Optional[str]:
        if not self._all_functions:
            return None
        # Exact case-insensitive first
        lower = name.lower()
        if lower in self._func_names_lower:
            actual = self._func_names_lower[lower]
            if actual != name:  # different case
                return actual
            return None  # same name, no correction needed
        return self._closest_string(name, list(self._all_functions.keys()), threshold)

    def _fuzzy_match_class(self, name: str, threshold: float = 0.85) -> Optional[str]:
        if not self._all_classes:
            return None
        lower = name.lower()
        if lower in self._class_names_lower:
            actual = self._class_names_lower[lower]
            if actual != name:
                return actual
            return None
        return self._closest_string(name, list(self._all_classes.keys()), threshold)

    def _fuzzy_match_file(self, filename: str) -> Optional[str]:
        if not self._all_files:
            return None
        return self._closest_string(filename, list(self._all_files), self.IMPORT_THRESHOLD)

    @staticmethod
    def _closest_string(target: str, candidates: List[str],
                        threshold: float = 0.85) -> Optional[str]:
        if not candidates:
            return None
        best_score = 0.0
        best_match = None
        target_lower = target.lower()

        for candidate in candidates:
            # Skip very different lengths (different concept, not a typo)
            len_ratio = len(candidate) / max(len(target), 1)
            if len_ratio < 0.7 or len_ratio > 1.4:
                continue

            score = difflib.SequenceMatcher(
                None, target_lower, candidate.lower()
            ).ratio()

            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score >= threshold and best_match != target:
            return best_match
        return None

    # ── Public API ──────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "identifiers_scanned": self._cumulative.identifiers_scanned,
            "corrections_made": self._cumulative.corrections_made,
            "functions_corrected": self._cumulative.functions_corrected,
            "classes_corrected": self._cumulative.classes_corrected,
            "imports_corrected": self._cumulative.imports_corrected,
            "enrichments_added": self._cumulative.enrichments_added,
            "time_ms": f"{self._cumulative.time_ms:.0f}",
        }


# ── Skip Lists ─────────────────────────────────────────────────────

# Standard library, builtins, framework names — NEVER correct these
_SKIP_NAMES_LOWER = frozenset({
    # Python builtins
    "print", "len", "range", "type", "str", "int", "float", "bool", "list",
    "dict", "set", "tuple", "sorted", "reversed", "enumerate", "zip", "map",
    "filter", "any", "all", "sum", "min", "max", "abs", "round", "open",
    "input", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "delattr", "super", "property", "staticmethod", "classmethod", "iter",
    "next", "hash", "id", "repr", "format", "vars", "dir", "callable",
    "eval", "exec", "compile", "object", "bytes", "bytearray", "memoryview",
    "frozenset", "complex", "bin", "hex", "oct", "ord", "chr", "ascii",
    # Common standard library
    "join", "split", "strip", "replace", "append", "extend", "insert",
    "remove", "pop", "get", "keys", "values", "items", "update", "copy",
    "encode", "decode", "read", "write", "close", "flush", "seek",
    "exists", "mkdir", "makedirs", "walk", "glob", "match", "search",
    "dumps", "loads", "dump", "load", "parse", "format_exc",
    "sleep", "time", "datetime", "path", "basename", "dirname",
    "empty_cache", "no_grad", "zero_grad", "backward", "forward",
    "fit", "predict", "transform", "compile",
    # Common patterns
    "init", "__init__", "__str__", "__repr__", "__call__", "__enter__", "__exit__",
    "main", "setup", "run", "start", "stop", "reset", "clear",
    "connect", "disconnect", "send", "receive", "process", "handle",
    "test", "assertEqual", "assertTrue", "assertRaises", "setUp", "tearDown",
    "configure", "initialize", "shutdown", "destroy", "dispose",
    "validate", "serialize", "deserialize", "register", "unregister",
    # JS/TS/React
    "require", "export", "import", "console", "log", "warn", "error",
    "useState", "useEffect", "useCallback", "useMemo", "useRef",
    "fetch", "then", "catch", "finally", "async", "await",
    "addEventListener", "removeEventListener", "querySelector",
    "createElement", "appendChild", "setAttribute", "render",
    # Model/AI framework
    "model", "mode", "load", "save", "train", "eval", "generate",
    "tokenize", "detokenize", "sample", "logits",
    "download", "upload", "install", "uninstall",
    # Common method names that exist everywhere
    "get_model", "set_model", "load_model", "save_model", "list_models",
    "add_model", "remove_model", "switch_model", "select_model",
    "unload", "reload", "refresh", "check", "verify", "status",
})

_SKIP_CLASSES = frozenset({
    # Python standard
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "RuntimeError", "IOError", "OSError", "FileNotFoundError",
    "NotImplementedError", "AttributeError", "ImportError",
    "StopIteration", "GeneratorExit", "SystemExit",
    "None", "True", "False", "Any", "Optional", "List", "Dict", "Set",
    "Tuple", "Union", "Callable", "Type", "Sequence", "Iterator",
    "Path", "Thread", "Process", "Lock", "Event", "Queue",
    # Common framework classes — these are NOT project-specific
    "App", "Request", "Response", "Router", "View",
    "Controller", "Service", "Repository", "Factory", "Builder",
    "Config", "Settings", "Logger", "Handler", "Middleware",
    "Component", "Provider", "Module", "Plugin", "Manager",
    "Model", "Schema", "Form", "Field", "Column", "Table",
    "Client", "Server", "Connection", "Session", "Pool",
    "Cache", "Store", "Buffer", "Stream", "Channel",
    "Task", "Worker", "Scheduler", "Executor", "Timer",
    "Parser", "Lexer", "Token", "Node", "Tree", "Graph",
    "Base", "Abstract", "Interface", "Mixin", "Meta",
    # JS/TS
    "Promise", "Error", "Array", "Object", "String", "Number",
    "Boolean", "Map", "Date", "RegExp", "JSON", "Math",
    "Buffer", "EventEmitter", "Observable",
})

# English words that look like class names but aren't identifiers
_SKIP_ENGLISH_WORDS = frozenset({
    "updated", "implementation", "production", "prediction",
    "completion", "configuration", "application", "information",
    "description", "observation", "preparation", "registration",
    "notification", "verification", "authorization", "authentication",
    "initialization", "optimization", "serialization", "transformation",
    "documentation", "demonstration", "communication", "integration",
    "migration", "operation", "collection", "connection", "execution",
    "exception", "extension", "expression", "instruction", "construction",
    "destruction", "extraction", "injection", "selection", "detection",
    "protection", "reflection", "inspection", "correction", "direction",
    "function", "section", "action", "option", "position", "condition",
    "version", "session", "mission", "permission", "submission",
    "important", "different", "available", "following", "required",
    "specific", "existing", "standard", "critical", "improved",
    "approach", "pattern", "example", "summary", "overview",
    "robust", "simple", "basic", "common", "custom", "default",
})
