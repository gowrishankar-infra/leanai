"""
LeanAI Phase 7a — Code Analyzer
Uses Python's AST module to extract deep structure from source files:
  - Function definitions with signatures, docstrings, line numbers
  - Class definitions with methods and inheritance
  - Import statements (what this file depends on)
  - Function calls (what this file uses)
  - Global variables and constants
  
This is the foundation of the Project Brain's understanding.
"""

import ast
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple


@dataclass
class FunctionInfo:
    """Information about a function or method."""
    name: str
    filepath: str
    line_start: int
    line_end: int
    args: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_method: bool = False
    class_name: Optional[str] = None
    calls: List[str] = field(default_factory=list)  # functions this one calls
    complexity: int = 1  # cyclomatic complexity estimate

    @property
    def qualified_name(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name

    def to_dict(self) -> dict:
        return {
            "name": self.name, "filepath": self.filepath,
            "line_start": self.line_start, "line_end": self.line_end,
            "args": self.args, "return_type": self.return_type,
            "docstring": self.docstring, "decorators": self.decorators,
            "is_method": self.is_method, "class_name": self.class_name,
            "calls": self.calls, "complexity": self.complexity,
            "qualified_name": self.qualified_name,
        }


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    filepath: str
    line_start: int
    line_end: int
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "filepath": self.filepath,
            "line_start": self.line_start, "line_end": self.line_end,
            "bases": self.bases, "methods": self.methods,
            "docstring": self.docstring, "decorators": self.decorators,
        }


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str                      # the module being imported
    names: List[str] = field(default_factory=list)  # specific names imported
    alias: Optional[str] = None      # import X as alias
    is_from: bool = False            # True if "from X import Y"
    line: int = 0

    def to_dict(self) -> dict:
        return {
            "module": self.module, "names": self.names,
            "alias": self.alias, "is_from": self.is_from, "line": self.line,
        }


@dataclass
class FileAnalysis:
    """Complete analysis of a single source file."""
    filepath: str
    language: str = "python"
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    global_vars: List[str] = field(default_factory=list)
    total_lines: int = 0
    docstring: Optional[str] = None  # module-level docstring
    error: Optional[str] = None      # parse error if any

    @property
    def function_names(self) -> List[str]:
        return [f.qualified_name for f in self.functions]

    @property
    def class_names(self) -> List[str]:
        return [c.name for c in self.classes]

    @property
    def import_modules(self) -> List[str]:
        return [i.module for i in self.imports]

    def to_dict(self) -> dict:
        return {
            "filepath": self.filepath, "language": self.language,
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
            "imports": [i.to_dict() for i in self.imports],
            "global_vars": self.global_vars,
            "total_lines": self.total_lines,
            "docstring": self.docstring,
            "error": self.error,
        }

    def summary(self) -> str:
        parts = [self.filepath]
        if self.functions:
            parts.append(f"{len(self.functions)} functions")
        if self.classes:
            parts.append(f"{len(self.classes)} classes")
        if self.imports:
            parts.append(f"{len(self.imports)} imports")
        parts.append(f"{self.total_lines} lines")
        return " | ".join(parts)


class _CallCollector(ast.NodeVisitor):
    """Collects function/method calls from an AST subtree."""

    def __init__(self):
        self.calls: List[str] = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        self.generic_visit(node)


class _ComplexityCounter(ast.NodeVisitor):
    """Estimates cyclomatic complexity of a function."""

    def __init__(self):
        self.complexity = 1  # base

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self.complexity += 1
        self.generic_visit(node)


def _get_end_line(node, default: int) -> int:
    """Get the end line of an AST node."""
    return getattr(node, "end_lineno", default)


def _get_decorators(node) -> List[str]:
    """Extract decorator names from a function/class node."""
    decorators = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            decorators.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            decorators.append(dec.attr)
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                decorators.append(dec.func.id)
            elif isinstance(dec.func, ast.Attribute):
                decorators.append(dec.func.attr)
    return decorators


def _get_args(node: ast.FunctionDef) -> List[str]:
    """Extract argument names from a function definition."""
    args = []
    for arg in node.args.args:
        name = arg.arg
        if arg.annotation:
            try:
                ann = ast.dump(arg.annotation)
                # Simplify common types
                if isinstance(arg.annotation, ast.Name):
                    ann = arg.annotation.id
                elif isinstance(arg.annotation, ast.Constant):
                    ann = str(arg.annotation.value)
                name = f"{name}: {ann}"
            except Exception:
                pass
        args.append(name)
    return args


def _get_return_type(node: ast.FunctionDef) -> Optional[str]:
    """Extract return type annotation."""
    if node.returns:
        if isinstance(node.returns, ast.Name):
            return node.returns.id
        elif isinstance(node.returns, ast.Constant):
            return str(node.returns.value)
        try:
            return ast.unparse(node.returns)
        except Exception:
            return None
    return None


def analyze_python_file(filepath: str, source: Optional[str] = None) -> FileAnalysis:
    """
    Analyze a Python source file and extract its structure.
    
    Args:
        filepath: path to the file
        source: source code (if None, reads from filepath)
    
    Returns:
        FileAnalysis with functions, classes, imports, etc.
    """
    result = FileAnalysis(filepath=filepath)

    if source is None:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except (FileNotFoundError, IOError) as e:
            result.error = str(e)
            return result

    result.total_lines = source.count("\n") + 1

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        result.error = f"SyntaxError: {e}"
        return result

    # Module docstring
    result.docstring = ast.get_docstring(tree)

    for node in ast.walk(tree):
        # ── Functions ──────────────────────────────────────────
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Determine if it's a method
            is_method = False
            class_name = None
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    if node in ast.walk(parent) and node is not parent:
                        for item in parent.body:
                            if item is node:
                                is_method = True
                                class_name = parent.name
                                break

            # Collect calls
            call_collector = _CallCollector()
            call_collector.visit(node)

            # Estimate complexity
            cc = _ComplexityCounter()
            cc.visit(node)

            func = FunctionInfo(
                name=node.name,
                filepath=filepath,
                line_start=node.lineno,
                line_end=_get_end_line(node, node.lineno),
                args=_get_args(node),
                return_type=_get_return_type(node),
                docstring=ast.get_docstring(node),
                decorators=_get_decorators(node),
                is_method=is_method,
                class_name=class_name,
                calls=call_collector.calls,
                complexity=cc.complexity,
            )
            result.functions.append(func)

        # ── Classes ────────────────────────────────────────────
        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.unparse(base) if hasattr(ast, "unparse") else base.attr)

            methods = [
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]

            cls = ClassInfo(
                name=node.name,
                filepath=filepath,
                line_start=node.lineno,
                line_end=_get_end_line(node, node.lineno),
                bases=bases,
                methods=methods,
                docstring=ast.get_docstring(node),
                decorators=_get_decorators(node),
            )
            result.classes.append(cls)

        # ── Imports ────────────────────────────────────────────
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imp = ImportInfo(
                    module=alias.name,
                    alias=alias.asname,
                    is_from=False,
                    line=node.lineno,
                )
                result.imports.append(imp)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imp = ImportInfo(
                module=module,
                names=names,
                is_from=True,
                line=node.lineno,
            )
            result.imports.append(imp)

        # ── Global assignments ─────────────────────────────────
        elif isinstance(node, ast.Assign) and hasattr(node, "lineno"):
            # Only top-level assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        result.global_vars.append(target.id)

    return result


def analyze_file(filepath: str, source: Optional[str] = None) -> FileAnalysis:
    """Analyze a source file. Currently supports Python; extensible."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".py", ".pyw"):
        return analyze_python_file(filepath, source)
    # For non-Python files, return basic info
    result = FileAnalysis(filepath=filepath, language=ext.lstrip("."))
    if source is None:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except (FileNotFoundError, IOError) as e:
            result.error = str(e)
            return result
    result.total_lines = source.count("\n") + 1
    return result
