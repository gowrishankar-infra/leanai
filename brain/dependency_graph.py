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

        # Add function nodes
        for func in analysis.functions:
            func_id = f"{filepath}:{func.qualified_name}"
            func_node = GraphNode(
                id=func_id, node_type="function", name=func.qualified_name,
                filepath=filepath, metadata={
                    "args": func.args, "return_type": func.return_type,
                    "docstring": func.docstring, "complexity": func.complexity,
                    "line_start": func.line_start, "line_end": func.line_end,
                },
            )
            self._add_node(func_node)
            self._file_to_functions[filepath].append(func_id)
            self._function_lookup[func.name] = func_id
            if func.qualified_name != func.name:
                self._function_lookup[func.qualified_name] = func_id

            # Edge: file defines function
            self._add_edge(GraphEdge(
                source=filepath, target=func_id, edge_type="defines",
            ))

            # Edges: function calls other functions
            for call_name in func.calls:
                # We'll resolve these after all files are added
                call_id = f"__unresolved__:{call_name}"
                self._add_edge(GraphEdge(
                    source=func_id, target=call_id, edge_type="calls",
                    metadata={"raw_name": call_name},
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
        Replaces __unresolved__:name with actual node IDs where possible.
        """
        resolved_edges = []
        for edge in self.edges:
            if edge.target.startswith("__unresolved__:"):
                name = edge.target.split(":", 1)[1]
                if name in self._function_lookup:
                    edge.target = self._function_lookup[name]
                # else leave as unresolved (external dependency)
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
