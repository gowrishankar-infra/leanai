"""
LeanAI Phase 7a — Dependency Graph
Tracks relationships between files, functions, and classes in a project.

Relationships tracked:
  - File imports File (A.py imports from B.py)
  - Function calls Function (func_a() calls func_b())
  - Class inherits Class (ClassA extends ClassB)
  - File defines Function/Class
  
Enables queries like:
  - "What files depend on database.py?"
  - "If I change this function, what breaks?"
  - "Show me the call chain for handle_request()"
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path

from brain.analyzer import FileAnalysis, FunctionInfo, ClassInfo


# ═══════════════════════════════════════════════════════════════════
# Node-ID helpers (M2.2 — Windows-safe)
# ═══════════════════════════════════════════════════════════════════
# Node IDs have the form "filepath:qualified_name". On Windows,
# filepath contains a drive-letter colon ("C:\..."), so split(":", 1)
# truncates to just "C". Function-qualified names cannot contain ":",
# so rsplit(":", 1) always gives the right answer.

def _node_file(node_id: str) -> str:
    """Return the filepath portion of a 'filepath:qualified_name' node ID.
    Safe on Windows (drive-letter colon) and POSIX paths."""
    if ':' not in node_id:
        return node_id
    return node_id.rsplit(':', 1)[0]


def _top_package(filepath: str) -> str:
    """Return the top-level directory (package name) of a filepath,
    or the filename stem if the file is at the project root.
    E.g. 'brain/tdd_loop.py' → 'brain',
         '/abs/leanai/brain/tdd_loop.py' → 'brain' (takes immediate parent
         when absolute; we strip back to first non-drive component).
    For files at the root ('setup_leanai.py') → 'setup_leanai'.
    """
    if not filepath:
        return ''
    norm = filepath.replace('\\', '/').strip('/')
    # Drop drive letter if any
    if len(norm) >= 2 and norm[1] == ':':
        norm = norm[2:].lstrip('/')
    parts = [p for p in norm.split('/') if p and p != '.']
    if not parts:
        return ''
    if len(parts) == 1:
        # Single file at root — use its stem
        return parts[0].rsplit('.', 1)[0]
    return parts[0]


@dataclass
class GraphNode:
    """A node in the dependency graph."""
    id: str            # unique identifier (filepath, or filepath:function_name)
    node_type: str     # "file", "function", "class"
    name: str          # display name
    filepath: str      # source file
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A directed edge in the dependency graph."""
    source: str        # source node id
    target: str        # target node id
    edge_type: str     # "imports", "calls", "inherits", "defines"
    metadata: dict = field(default_factory=dict)


class DependencyGraph:
    """
    Directed graph of project dependencies.
    
    Usage:
        graph = DependencyGraph()
        graph.add_file_analysis(analysis)  # from CodeAnalyzer
        
        # Query
        deps = graph.get_dependents("utils.py")  # who depends on utils.py?
        chain = graph.get_call_chain("handle_request")  # call tree
        impact = graph.impact_analysis("database.py")  # what breaks if I change this?
    """

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._adjacency: Dict[str, List[str]] = {}      # node -> [targets]
        self._reverse_adj: Dict[str, List[str]] = {}     # node -> [sources]
        self._file_to_functions: Dict[str, List[str]] = {}
        self._file_to_classes: Dict[str, List[str]] = {}
        self._function_lookup: Dict[str, str] = {}       # func_name -> node_id
        # ── M2.1 additions for call-site disambiguation ──
        # Per-file set of imported module names — lets us recognize calls like
        # `subprocess.run(...)` as external (no internal edge).
        self._file_imports: Dict[str, Set[str]] = {}     # filepath -> set of imported names
        # name-only lookup: name -> list of node_ids (not just one).
        # Used when receiver is unknown but we still want a ranked resolution.
        self._name_candidates: Dict[str, List[str]] = {}
        # (filepath, class_name) -> list of function node ids defined on that class
        self._class_methods: Dict[Tuple[str, str], List[str]] = {}

    def _add_node(self, node: GraphNode):
        self.nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []
        if node.id not in self._reverse_adj:
            self._reverse_adj[node.id] = []

    def _add_edge(self, edge: GraphEdge):
        self.edges.append(edge)
        if edge.source not in self._adjacency:
            self._adjacency[edge.source] = []
        if edge.target not in self._reverse_adj:
            self._reverse_adj[edge.target] = []
        self._adjacency[edge.source].append(edge.target)
        self._reverse_adj[edge.target].append(edge.source)

    def add_file_analysis(self, analysis: FileAnalysis):
        """Add a complete file analysis to the graph."""
        filepath = analysis.filepath

        # Add file node
        file_node = GraphNode(
            id=filepath, node_type="file", name=os.path.basename(filepath),
            filepath=filepath, metadata={"lines": analysis.total_lines},
        )
        self._add_node(file_node)
        self._file_to_functions[filepath] = []
        self._file_to_classes[filepath] = []

        # Record imports for this file — used later to recognize external
        # calls and to enforce cross-file import-reachability (M2.1).
        imported: Set[str] = set()
        for imp in analysis.imports:
            # For `import x.y.z` → record 'x', 'y', 'z', 'x.y.z'
            parts = imp.module.split('.')
            for i in range(len(parts)):
                imported.add('.'.join(parts[:i + 1]))
            imported.add(imp.module)
            # For `from x import a, b` → record 'a' and 'b' as names
            if imp.is_from:
                for name in imp.names:
                    imported.add(name)
                    if imp.alias:
                        imported.add(imp.alias)
            else:
                if imp.alias:
                    imported.add(imp.alias)
        self._file_imports[filepath] = imported

        # Add function nodes
        for func in analysis.functions:
            func_id = f"{filepath}:{func.qualified_name}"
            func_node = GraphNode(
                id=func_id, node_type="function", name=func.qualified_name,
                filepath=filepath, metadata={
                    "args": func.args, "return_type": func.return_type,
                    "docstring": func.docstring, "complexity": func.complexity,
                    "line_start": func.line_start, "line_end": func.line_end,
                    "class_name": func.class_name,
                },
            )
            self._add_node(func_node)
            self._file_to_functions[filepath].append(func_id)
            self._function_lookup[func.name] = func_id
            if func.qualified_name != func.name:
                self._function_lookup[func.qualified_name] = func_id

            # Track ALL candidates per name (not just first-wins).
            self._name_candidates.setdefault(func.name, []).append(func_id)
            if func.qualified_name != func.name:
                self._name_candidates.setdefault(func.qualified_name, []).append(func_id)

            # Index class methods
            if func.class_name:
                self._class_methods.setdefault((filepath, func.class_name), []).append(func_id)

            # Edge: file defines function
            self._add_edge(GraphEdge(
                source=filepath, target=func_id, edge_type="defines",
            ))

            # Edges: function calls other functions.
            # Prefer call_sites (M2.1, structured) when available; fall back
            # to func.calls for old cached analyses without call_sites.
            call_sites = getattr(func, 'call_sites', None)
            if call_sites:
                for receiver, method in call_sites:
                    # Encode receiver in the unresolved target so resolve_references
                    # can make an informed choice:
                    #   __unresolved__:METHOD|RECEIVER
                    # Empty receiver is bare call. Non-empty receiver gets
                    # used during resolution to pick the right candidate (or
                    # to drop the edge entirely if it's clearly external).
                    call_id = f"__unresolved__:{method}|{receiver}"
                    self._add_edge(GraphEdge(
                        source=func_id, target=call_id, edge_type="calls",
                        metadata={"raw_name": method, "receiver": receiver,
                                  "source_file": filepath,
                                  "source_class": func.class_name or ""},
                    ))
            else:
                for call_name in func.calls:
                    call_id = f"__unresolved__:{call_name}|"
                    self._add_edge(GraphEdge(
                        source=func_id, target=call_id, edge_type="calls",
                        metadata={"raw_name": call_name, "receiver": "",
                                  "source_file": filepath,
                                  "source_class": func.class_name or ""},
                    ))

        # Add class nodes
        for cls in analysis.classes:
            cls_id = f"{filepath}:{cls.name}"
            cls_node = GraphNode(
                id=cls_id, node_type="class", name=cls.name,
                filepath=filepath, metadata={
                    "bases": cls.bases, "methods": cls.methods,
                    "docstring": cls.docstring,
                },
            )
            self._add_node(cls_node)
            self._file_to_classes[filepath].append(cls_id)

            # Edge: file defines class
            self._add_edge(GraphEdge(
                source=filepath, target=cls_id, edge_type="defines",
            ))

            # Edges: class inherits from base classes
            for base in cls.bases:
                base_id = f"__unresolved__:{base}"
                self._add_edge(GraphEdge(
                    source=cls_id, target=base_id, edge_type="inherits",
                ))

        # Add import edges
        for imp in analysis.imports:
            # Try to resolve import to a project file
            target_id = f"__import__:{imp.module}"
            self._add_edge(GraphEdge(
                source=filepath, target=target_id, edge_type="imports",
                metadata={"names": imp.names, "is_from": imp.is_from},
            ))

    def resolve_references(self):
        """
        After all files are added, resolve unresolved function/class references.
        Replaces __unresolved__:name|receiver with actual node IDs.

        Receiver-aware resolution (M2.1):
          - If receiver matches a stdlib / imported module name in the
            calling file, DROP the edge (it's external — e.g. subprocess.run).
          - If receiver is 'self' and the caller is inside a class, prefer
            methods defined on that same class in the same file.
          - If receiver is 'cls', same as self.
          - If receiver is a name that matches the caller's file's imports,
            DROP the edge (external lib usage).
          - If receiver is empty (bare call), use the old name-only lookup.
          - Tiebreaker when multiple candidates exist: prefer same-file,
            then same-directory, then first match.
        """
        resolved_edges = []
        for edge in self.edges:
            if edge.edge_type == "inherits" and edge.target.startswith("__unresolved__:"):
                # inherits edges are still name-only, handle them the old way
                name = edge.target.split(":", 1)[1].split("|", 1)[0]
                if name in self._function_lookup:
                    edge.target = self._function_lookup[name]
                resolved_edges.append(edge)
                continue

            if not edge.target.startswith("__unresolved__:"):
                resolved_edges.append(edge)
                continue

            # Split "name|receiver" (new format); handle old format "name" gracefully
            payload = edge.target.split(":", 1)[1]
            if "|" in payload:
                method, receiver = payload.split("|", 1)
            else:
                method, receiver = payload, ""

            source_file = edge.metadata.get("source_file", "")
            source_class = edge.metadata.get("source_class", "")
            file_imports = self._file_imports.get(source_file, set())

            resolved_id: Optional[str] = None
            drop_edge = False

            # Decide based on receiver
            if receiver and receiver != "self" and receiver != "cls":
                # Non-trivial receiver. Is the root of the receiver an
                # imported module / external name?
                receiver_root = receiver.split(".", 1)[0]
                if self._is_external(receiver_root, file_imports):
                    # External call (e.g. subprocess.run, os.path.join).
                    # Record as external, NOT as an internal edge.
                    edge.target = f"__external__:{receiver}.{method}"
                    drop_edge = False
                    resolved_edges.append(edge)
                    continue
                # Receiver is internal (e.g. self.planner) — try scoped lookup
                # using the last attribute in the receiver as a hint.
                # For "self.planner.build_plan_prompt" method="build_plan_prompt"
                # we can try to find it anywhere.
                resolved_id = self._resolve_with_scope(method, receiver, source_file, source_class)
            elif receiver in ("self", "cls"):
                # Same-class resolution
                if source_class:
                    candidates = self._class_methods.get((source_file, source_class), [])
                    for cid in candidates:
                        node = self.nodes.get(cid)
                        if node and node.name.endswith(f".{method}"):
                            resolved_id = cid
                            break
                        if node and node.name == method:
                            resolved_id = cid
                            break
                # Fallback to same-file
                if not resolved_id:
                    for fid in self._file_to_functions.get(source_file, []):
                        node = self.nodes.get(fid)
                        if node and (node.name == method or node.name.endswith(f".{method}")):
                            resolved_id = fid
                            break
            else:
                # Bare call, no receiver. Could be a builtin, a same-module
                # function, or a `from X import Y` name. Check imports first.
                if self._is_external(method, file_imports):
                    edge.target = f"__external__:{method}"
                    resolved_edges.append(edge)
                    continue
                # Same-file preferred, then name-wide candidates
                resolved_id = self._resolve_with_scope(method, "", source_file, source_class)

            if resolved_id:
                edge.target = resolved_id
            else:
                # Couldn't resolve — mark as unresolved but make target stable
                edge.target = f"__unresolved__:{method}"

            resolved_edges.append(edge)

        # Rebuild adjacency
        self.edges = resolved_edges
        self._adjacency = {}
        self._reverse_adj = {}
        for node_id in self.nodes:
            self._adjacency[node_id] = []
            self._reverse_adj[node_id] = []
        for edge in self.edges:
            if edge.source in self._adjacency:
                self._adjacency[edge.source].append(edge.target)
            if edge.target in self._reverse_adj:
                self._reverse_adj[edge.target].append(edge.source)

    # Common Python stdlib / 3rd-party modules we always treat as external
    # (even if the user didn't explicitly `import` them — defensive default).
    _STDLIB_ROOTS: Set[str] = {
        'os', 'sys', 're', 'json', 'time', 'math', 'pathlib', 'pickle',
        'subprocess', 'shutil', 'shlex', 'socket', 'threading', 'asyncio',
        'hashlib', 'hmac', 'base64', 'urllib', 'http', 'email', 'smtplib',
        'ftplib', 'sqlite3', 'csv', 'xml', 'html', 'datetime', 'collections',
        'itertools', 'functools', 'typing', 'dataclasses', 'enum', 'abc',
        'logging', 'tempfile', 'io', 'ctypes', 'platform', 'ast', 'inspect',
        'traceback', 'warnings', 'atexit', 'signal', 'argparse', 'random',
        'secrets', 'string', 'textwrap', 'unicodedata', 'struct', 'array',
        'queue', 'concurrent', 'multiprocessing', 'copy', 'weakref',
        'contextlib', 'glob', 'fnmatch', 'linecache', 'fileinput',
        # Common 3rd-party
        'requests', 'httpx', 'urllib3', 'numpy', 'pandas', 'torch',
        'transformers', 'chromadb', 'z3', 'llama_cpp', 'pyperclip',
        'yaml', 'toml', 'click', 'typer', 'flask', 'fastapi', 'django',
        'pytest', 'unittest',
    }

    def _is_external(self, name: str, file_imports: Set[str]) -> bool:
        """True if `name` looks like an external module/symbol, not internal."""
        if name in file_imports:
            return True
        if name in self._STDLIB_ROOTS:
            return True
        return False

    def _resolve_with_scope(
        self, method: str, receiver: str, source_file: str, source_class: str
    ) -> Optional[str]:
        """
        Pick the best internal candidate for a call. Ranking:
          1) same file, same class (if receiver is 'self')
          2) same file, any class
          3) same directory
          4) any project-wide candidate with a preference for rarer names
             (skip if >5 candidates — too ambiguous to guess)
        """
        candidates = list(self._name_candidates.get(method, []))
        if not candidates:
            return None

        # Also try qualified name if method contains a dot
        if '.' in method:
            more = self._name_candidates.get(method, [])
            for m in more:
                if m not in candidates:
                    candidates.append(m)

        if len(candidates) == 1:
            return candidates[0]

        # Same-file preference
        same_file = [c for c in candidates if _node_file(c) == source_file]
        if same_file:
            return same_file[0]

        # Same-directory preference
        src_dir = os.path.dirname(source_file)
        same_dir = [c for c in candidates if os.path.dirname(_node_file(c)) == src_dir]
        if same_dir:
            return same_dir[0]

        # If there are many candidates project-wide and we can't disambiguate,
        # refuse to resolve (don't invent a false edge).
        if len(candidates) > 5:
            return None

        # Last resort: if we'd resolve to a file in a different top-level
        # package than the source, refuse rather than guess. This prevents
        # aliased resolutions (e.g. setup_leanai.py:main trying to call a
        # same-named function in a completely unrelated package).
        src_top = _top_package(source_file)
        cand_tops = {_top_package(_node_file(c)) for c in candidates}
        if src_top and cand_tops - {src_top}:
            # Multiple candidates across packages — ambiguous. Refuse.
            if len([c for c in candidates if _top_package(_node_file(c)) == src_top]) != 1:
                return None
            # Exactly one in the same top package — use it
            for c in candidates:
                if _top_package(_node_file(c)) == src_top:
                    return c

        return candidates[0]

    # ── Query methods ─────────────────────────────────────────────

    def get_dependents(self, node_id: str) -> List[str]:
        """Get all nodes that depend on (point to) this node."""
        return self._reverse_adj.get(node_id, [])

    def get_dependencies(self, node_id: str) -> List[str]:
        """Get all nodes that this node depends on (points to)."""
        return self._adjacency.get(node_id, [])

    def get_file_functions(self, filepath: str) -> List[str]:
        """Get all function node IDs defined in a file."""
        return self._file_to_functions.get(filepath, [])

    def get_file_classes(self, filepath: str) -> List[str]:
        """Get all class node IDs defined in a file."""
        return self._file_to_classes.get(filepath, [])

    def impact_analysis(self, filepath: str, max_depth: int = 3) -> Dict[str, int]:
        """
        Analyze impact of changing a file.
        Returns dict of {affected_node: distance} reachable within max_depth.
        """
        affected = {}
        queue = [(filepath, 0)]
        visited = {filepath}

        while queue:
            current, depth = queue.pop(0)
            if depth > max_depth:
                continue
            dependents = self.get_dependents(current)
            for dep in dependents:
                if dep not in visited and not dep.startswith("__"):
                    visited.add(dep)
                    affected[dep] = depth + 1
                    queue.append((dep, depth + 1))

        return affected

    def get_call_chain(self, function_name: str, max_depth: int = 5) -> List[Tuple[str, int]]:
        """
        Get the call chain starting from a function.
        Returns list of (function_id, depth) tuples.
        """
        start = self._function_lookup.get(function_name)
        if not start:
            return []

        chain = []
        queue = [(start, 0)]
        visited = {start}

        while queue:
            current, depth = queue.pop(0)
            if depth > max_depth:
                continue
            chain.append((current, depth))
            for target in self._adjacency.get(current, []):
                if target not in visited and not target.startswith("__"):
                    visited.add(target)
                    queue.append((target, depth + 1))

        return chain

    def find_function(self, name: str) -> Optional[GraphNode]:
        """Find a function node by name."""
        node_id = self._function_lookup.get(name)
        if node_id:
            return self.nodes.get(node_id)
        return None

    # ── Stats ─────────────────────────────────────────────────────

    @property
    def num_files(self) -> int:
        return sum(1 for n in self.nodes.values() if n.node_type == "file")

    @property
    def num_functions(self) -> int:
        return sum(1 for n in self.nodes.values() if n.node_type == "function")

    @property
    def num_classes(self) -> int:
        return sum(1 for n in self.nodes.values() if n.node_type == "class")

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def stats(self) -> dict:
        edge_types = {}
        for e in self.edges:
            edge_types[e.edge_type] = edge_types.get(e.edge_type, 0) + 1
        return {
            "files": self.num_files,
            "functions": self.num_functions,
            "classes": self.num_classes,
            "edges": self.num_edges,
            "edge_types": edge_types,
        }

    def summary(self) -> str:
        s = self.stats()
        return (
            f"Dependency graph: {s['files']} files, {s['functions']} functions, "
            f"{s['classes']} classes, {s['edges']} edges"
        )
