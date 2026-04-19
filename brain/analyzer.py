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
    calls: List[str] = field(default_factory=list)  # functions this one calls (names only)
    # Structured calls: list of (receiver, method) tuples. Populated by
    # _CallCollector for Python files. Enables the dependency graph to
    # disambiguate calls like `subprocess.run(...)` from `self.run(...)`.
    # (M2.1 fix for name-only collision false positives.)
    call_sites: List[tuple] = field(default_factory=list)
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
            "calls": self.calls, "call_sites": self.call_sites,
            "complexity": self.complexity,
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
    """
    Collects function/method calls from an AST subtree.

    Two shapes of data are produced:

    - self.calls: List[str]
        Flat list of just the method/function names. Kept for backwards
        compatibility with any existing code that reads func.calls.

    - self.call_sites: List[Tuple[str, str]]
        Structured (receiver, method) tuples. Used by the dependency graph
        to avoid name-only collisions that create false edges (M2.1 fix).

        Receiver encoding:
          ""           — bare call,          e.g.  isinstance(x, str)
          "self"       — instance method,    e.g.  self.run(...)
          "cls"        — classmethod,        e.g.  cls.build(...)
          "name"       — attribute on name,  e.g.  subprocess.run(...)
          "a.b"        — deeper chain,       e.g.  self.planner.build(...)
          "_expr_"     — non-trivial base,   e.g.  obj().run() → ("_expr_", "run")

        The graph uses this receiver to:
          1) skip resolution when receiver is a known external module
          2) prefer same-class resolution when receiver is "self"
          3) prefer same-file resolution as a tiebreaker
    """

    def __init__(self):
        self.calls: List[str] = []
        self.call_sites: List[Tuple[str, str]] = []

    # Render an ast.Attribute/ast.Name chain into a string receiver like "a.b.c"
    def _render_receiver(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._render_receiver(node.value)
            if base and base != "_expr_":
                return f"{base}.{node.attr}"
            return "_expr_"
        # Calls, subscripts, literals, etc. — non-trivial receiver
        return "_expr_"

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            # Bare call: foo(...)
            name = node.func.id
            self.calls.append(name)
            self.call_sites.append(("", name))
        elif isinstance(node.func, ast.Attribute):
            # Attribute call: receiver.method(...)
            method = node.func.attr
            receiver = self._render_receiver(node.func.value)
            self.calls.append(method)
            self.call_sites.append((receiver, method))
        # else: node.func is a Call / Subscript / etc — skip, we don't track these
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
                call_sites=list(call_collector.call_sites),
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
    """Analyze a source file. Python uses AST; all other languages use regex parsing."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".py", ".pyw"):
        return analyze_python_file(filepath, source)

    # All code files get regex-based parsing
    CODE_EXTENSIONS = {
        # Web
        ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte",
        # Systems
        ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".cc", ".cxx", ".hh",
        # JVM
        ".java", ".kt", ".kts", ".scala", ".groovy", ".clj",
        # .NET
        ".cs", ".fs", ".vb",
        # Mobile
        ".swift", ".m", ".mm", ".dart",
        # Scripting
        ".rb", ".php", ".pl", ".pm", ".lua", ".r", ".R", ".jl",
        # Shell
        ".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".bat", ".cmd",
        # Data/Config
        ".sql", ".proto", ".graphql", ".gql",
        # Functional
        ".hs", ".ex", ".exs", ".erl", ".elm", ".ml", ".mli", ".f90", ".f95",
        # Other
        ".zig", ".nim", ".v", ".cr", ".d", ".ada", ".pas", ".cob", ".cobol",
        ".lisp", ".cl", ".rkt", ".scm", ".tcl", ".awk", ".sed",
    }

    if ext in CODE_EXTENSIONS:
        return _analyze_with_regex(filepath, source, ext.lstrip("."))

    # Non-code files — basic info only
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


def _analyze_with_regex(filepath: str, source: Optional[str] = None,
                        lang: str = "") -> FileAnalysis:
    """Regex-based analysis for non-Python languages."""
    result = FileAnalysis(filepath=filepath, language=lang)

    if source is None:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except (FileNotFoundError, IOError) as e:
            result.error = str(e)
            return result

    result.total_lines = source.count("\n") + 1
    lines = source.split("\n")

    # ── Language-specific patterns ────────────────────────────
    if lang in ("js", "jsx", "ts", "tsx", "mjs", "cjs", "vue", "svelte"):
        _parse_javascript(source, lines, filepath, result)
    elif lang == "go":
        _parse_go(source, lines, filepath, result)
    elif lang in ("rs",):
        _parse_rust(source, lines, filepath, result)
    elif lang in ("java", "kt", "kts", "scala", "groovy"):
        _parse_java(source, lines, filepath, result)
    elif lang in ("c", "cpp", "h", "hpp", "cc", "cxx", "hh", "cs", "m", "mm"):
        _parse_c_cpp(source, lines, filepath, result)
    elif lang == "sql":
        _parse_sql(source, lines, filepath, result)
    elif lang in ("rb",):
        _parse_ruby(source, lines, filepath, result)
    elif lang in ("php",):
        _parse_php(source, lines, filepath, result)
    elif lang in ("swift",):
        _parse_swift(source, lines, filepath, result)
    elif lang in ("dart",):
        _parse_dart(source, lines, filepath, result)
    elif lang in ("ex", "exs"):
        _parse_elixir(source, lines, filepath, result)
    elif lang in ("sh", "bash", "zsh", "fish"):
        _parse_shell(source, lines, filepath, result)
    elif lang in ("lua",):
        _parse_lua(source, lines, filepath, result)
    elif lang in ("r", "R"):
        _parse_r(source, lines, filepath, result)
    elif lang in ("jl",):
        _parse_julia(source, lines, filepath, result)
    elif lang in ("zig",):
        _parse_zig(source, lines, filepath, result)
    elif lang in ("nim",):
        _parse_nim(source, lines, filepath, result)
    else:
        # Generic fallback — catches common patterns across all languages
        _parse_generic(source, lines, filepath, result)

    return result


def _parse_javascript(source, lines, filepath, result):
    """Parse JavaScript/TypeScript functions and classes."""
    # Functions: function name(), const name = () =>, async function name()
    patterns = [
        r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)",
        r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>",
        r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function\s*\(([^)]*)\)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, source):
            name = match.group(1)
            args = [a.strip().split(":")[0].strip() for a in match.group(2).split(",") if a.strip()]
            line_num = source[:match.start()].count("\n") + 1
            result.functions.append(FunctionInfo(
                name=name, filepath=filepath, line_start=line_num,
                line_end=line_num + 5, args=args,
            ))

    # Methods inside classes
    for match in re.finditer(r"(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*\{", source):
        name = match.group(1)
        if name not in ("if", "for", "while", "switch", "catch", "function"):
            args = [a.strip().split(":")[0].strip() for a in match.group(2).split(",") if a.strip()]
            line_num = source[:match.start()].count("\n") + 1
            # Avoid duplicates
            if not any(f.name == name and f.line_start == line_num for f in result.functions):
                result.functions.append(FunctionInfo(
                    name=name, filepath=filepath, line_start=line_num,
                    line_end=line_num + 5, args=args, is_method=True,
                ))

    # Classes
    for match in re.finditer(r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?", source):
        line_num = source[:match.start()].count("\n") + 1
        bases = [match.group(2)] if match.group(2) else []
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num,
            line_end=line_num + 10, bases=bases,
        ))

    # Imports
    for match in re.finditer(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]", source):
        result.imports.append(ImportInfo(module=match.group(1)))
    for match in re.finditer(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", source):
        result.imports.append(ImportInfo(module=match.group(1)))


def _parse_go(source, lines, filepath, result):
    """Parse Go functions, methods, structs, interfaces."""
    # Functions: func name(args) returntype
    for match in re.finditer(r"func\s+(\w+)\s*\(([^)]*)\)(?:\s*\(?([^)]*)\)?)?", source):
        name = match.group(1)
        args = [a.strip().split(" ")[0] for a in match.group(2).split(",") if a.strip()]
        ret = match.group(3) if match.group(3) else None
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num,
            line_end=line_num + 10, args=args, return_type=ret,
        ))

    # Methods: func (receiver Type) name(args)
    for match in re.finditer(r"func\s+\(\s*\w+\s+\*?(\w+)\s*\)\s+(\w+)\s*\(([^)]*)\)", source):
        class_name = match.group(1)
        name = match.group(2)
        args = [a.strip().split(" ")[0] for a in match.group(3).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num,
            line_end=line_num + 10, args=args, is_method=True, class_name=class_name,
        ))

    # Structs
    for match in re.finditer(r"type\s+(\w+)\s+struct\s*\{", source):
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10,
        ))

    # Interfaces
    for match in re.finditer(r"type\s+(\w+)\s+interface\s*\{", source):
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10,
        ))

    # Imports
    for match in re.finditer(r'"([^"]+)"', source[:500]):
        result.imports.append(ImportInfo(module=match.group(1)))


def _parse_rust(source, lines, filepath, result):
    """Parse Rust functions, structs, enums, traits, impls."""
    # Functions
    for match in re.finditer(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*(\S+))?", source):
        name = match.group(1)
        args = [a.strip().split(":")[0].strip() for a in match.group(2).split(",") if a.strip()]
        ret = match.group(3) if match.group(3) else None
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num,
            line_end=line_num + 10, args=args, return_type=ret,
        ))

    # Structs
    for match in re.finditer(r"(?:pub\s+)?struct\s+(\w+)", source):
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10,
        ))

    # Enums
    for match in re.finditer(r"(?:pub\s+)?enum\s+(\w+)", source):
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10,
        ))

    # Traits
    for match in re.finditer(r"(?:pub\s+)?trait\s+(\w+)", source):
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10,
        ))

    # Use imports
    for match in re.finditer(r"use\s+([\w:]+)", source):
        result.imports.append(ImportInfo(module=match.group(1)))


def _parse_java(source, lines, filepath, result):
    """Parse Java/Kotlin classes, methods, interfaces."""
    # Classes and interfaces
    for match in re.finditer(r"(?:public|private|protected)?\s*(?:abstract\s+)?(?:class|interface)\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?", source):
        name = match.group(1)
        bases = []
        if match.group(2):
            bases.append(match.group(2))
        if match.group(3):
            bases.extend([b.strip() for b in match.group(3).split(",")])
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 20, bases=bases,
        ))

    # Methods
    for match in re.finditer(r"(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(\w+)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+\w+)?\s*\{", source):
        ret_type = match.group(1)
        name = match.group(2)
        if name in ("if", "for", "while", "switch", "catch", "class"):
            continue
        args = [a.strip().split(" ")[-1] for a in match.group(3).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num,
            line_end=line_num + 10, args=args, return_type=ret_type, is_method=True,
        ))

    # Imports
    for match in re.finditer(r"import\s+([\w.]+);", source):
        result.imports.append(ImportInfo(module=match.group(1)))


def _parse_c_cpp(source, lines, filepath, result):
    """Parse C/C++/C# functions, classes, structs."""
    # Functions (C-style)
    for match in re.finditer(r"(?:static\s+)?(?:inline\s+)?(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{", source):
        ret_type = match.group(1)
        name = match.group(2)
        if name in ("if", "for", "while", "switch", "catch", "return", "sizeof"):
            continue
        args = [a.strip().split(" ")[-1].replace("*", "").replace("&", "") for a in match.group(3).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num,
            line_end=line_num + 10, args=args, return_type=ret_type,
        ))

    # Classes/structs
    for match in re.finditer(r"(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|private|protected)?\s*(\w+))?", source):
        bases = [match.group(2)] if match.group(2) else []
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10, bases=bases,
        ))

    # Includes
    for match in re.finditer(r"#include\s*[<\"]([^>\"]+)[>\"]", source):
        result.imports.append(ImportInfo(module=match.group(1)))


def _parse_sql(source, lines, filepath, result):
    """Parse SQL procedures, functions, tables."""
    # Procedures/functions
    for match in re.finditer(r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:PROCEDURE|FUNCTION)\s+(\w+)\s*\(([^)]*)\)", source, re.IGNORECASE):
        name = match.group(1)
        args = [a.strip().split(" ")[0] for a in match.group(2).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 10, args=args,
        ))

    # Tables
    for match in re.finditer(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", source, re.IGNORECASE):
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10,
        ))


def _parse_ruby(source, lines, filepath, result):
    """Parse Ruby classes, modules, methods."""
    for match in re.finditer(r"def\s+(?:self\.)?(\w+)(?:\(([^)]*)\))?", source):
        name = match.group(1)
        args = [a.strip() for a in (match.group(2) or "").split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 5, args=args,
        ))

    for match in re.finditer(r"(?:class|module)\s+(\w+)(?:\s*<\s*(\w+))?", source):
        bases = [match.group(2)] if match.group(2) else []
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10, bases=bases,
        ))


def _parse_php(source, lines, filepath, result):
    """Parse PHP classes, functions, methods."""
    for match in re.finditer(r"(?:public|private|protected)?\s*(?:static\s+)?function\s+(\w+)\s*\(([^)]*)\)", source):
        name = match.group(1)
        args = [a.strip().split("=")[0].strip().split(" ")[-1].replace("$", "") for a in match.group(2).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 10, args=args,
        ))

    for match in re.finditer(r"class\s+(\w+)(?:\s+extends\s+(\w+))?", source):
        bases = [match.group(2)] if match.group(2) else []
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10, bases=bases,
        ))


def _parse_swift(source, lines, filepath, result):
    """Parse Swift functions, classes, structs, protocols, enums."""
    for match in re.finditer(r"(?:public\s+|private\s+|internal\s+)?(?:static\s+)?func\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\S+))?", source):
        name = match.group(1)
        args = [a.strip().split(":")[0].strip() for a in match.group(2).split(",") if a.strip()]
        ret = match.group(3) if match.group(3) else None
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 10, args=args, return_type=ret,
        ))

    for match in re.finditer(r"(?:class|struct|protocol|enum)\s+(\w+)(?:\s*:\s*(\w+))?", source):
        bases = [match.group(2)] if match.group(2) else []
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10, bases=bases,
        ))


def _parse_dart(source, lines, filepath, result):
    """Parse Dart functions, classes, mixins."""
    for match in re.finditer(r"(?:Future<\w+>\s+|void\s+|int\s+|String\s+|bool\s+|double\s+|dynamic\s+|(\w+)\s+)?(\w+)\s*\(([^)]*)\)\s*(?:async\s*)?\{", source):
        name = match.group(2)
        if name in ("if", "for", "while", "switch", "catch", "class"):
            continue
        args = [a.strip().split(" ")[-1] for a in match.group(3).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 10, args=args,
        ))

    for match in re.finditer(r"(?:abstract\s+)?(?:class|mixin)\s+(\w+)(?:\s+extends\s+(\w+))?", source):
        bases = [match.group(2)] if match.group(2) else []
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10, bases=bases,
        ))


def _parse_elixir(source, lines, filepath, result):
    """Parse Elixir modules, functions."""
    for match in re.finditer(r"def[p]?\s+(\w+)\s*\(([^)]*)\)", source):
        name = match.group(1)
        args = [a.strip() for a in match.group(2).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 5, args=args,
        ))

    for match in re.finditer(r"defmodule\s+([\w.]+)", source):
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10,
        ))


def _parse_shell(source, lines, filepath, result):
    """Parse Shell/Bash functions."""
    for match in re.finditer(r"(?:function\s+)?(\w+)\s*\(\s*\)\s*\{", source):
        name = match.group(1)
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 5,
        ))


def _parse_lua(source, lines, filepath, result):
    """Parse Lua functions."""
    for match in re.finditer(r"(?:local\s+)?function\s+([\w.:]+)\s*\(([^)]*)\)", source):
        name = match.group(1)
        args = [a.strip() for a in match.group(2).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 5, args=args,
        ))


def _parse_r(source, lines, filepath, result):
    """Parse R functions."""
    for match in re.finditer(r"(\w+)\s*<-\s*function\s*\(([^)]*)\)", source):
        name = match.group(1)
        args = [a.strip().split("=")[0].strip() for a in match.group(2).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 5, args=args,
        ))


def _parse_julia(source, lines, filepath, result):
    """Parse Julia functions, structs."""
    for match in re.finditer(r"function\s+(\w+)\s*\(([^)]*)\)", source):
        name = match.group(1)
        args = [a.strip().split("::")[0].strip() for a in match.group(2).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 5, args=args,
        ))

    for match in re.finditer(r"(?:mutable\s+)?struct\s+(\w+)", source):
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10,
        ))


def _parse_zig(source, lines, filepath, result):
    """Parse Zig functions, structs."""
    for match in re.finditer(r"(?:pub\s+)?fn\s+(\w+)\s*\(([^)]*)\)", source):
        name = match.group(1)
        args = [a.strip().split(":")[0].strip() for a in match.group(2).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 5, args=args,
        ))

    for match in re.finditer(r"const\s+(\w+)\s*=\s*(?:packed\s+)?struct", source):
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10,
        ))


def _parse_nim(source, lines, filepath, result):
    """Parse Nim procedures, types."""
    for match in re.finditer(r"(?:proc|func|method)\s+(\w+)\s*\(([^)]*)\)", source):
        name = match.group(1)
        args = [a.strip().split(":")[0].strip() for a in match.group(2).split(",") if a.strip()]
        line_num = source[:match.start()].count("\n") + 1
        result.functions.append(FunctionInfo(
            name=name, filepath=filepath, line_start=line_num, line_end=line_num + 5, args=args,
        ))

    for match in re.finditer(r"type\s+(\w+)\s*=\s*(?:object|ref object|enum)", source):
        line_num = source[:match.start()].count("\n") + 1
        result.classes.append(ClassInfo(
            name=match.group(1), filepath=filepath, line_start=line_num, line_end=line_num + 10,
        ))


def _parse_generic(source, lines, filepath, result):
    """
    Generic fallback parser for any language not specifically handled.
    Uses universal patterns that work across most programming languages.
    """
    # Universal function patterns (covers most C-family and functional languages)
    func_patterns = [
        # C-style: type name(args) {
        r"(?:public|private|protected|static|inline|virtual|async|export)?\s*(?:\w+\s+)?(\w+)\s*\(([^)]*)\)\s*[{:]",
        # function keyword: function name(args)
        r"function\s+(\w+)\s*\(([^)]*)\)",
        # def keyword: def name(args)
        r"def\s+(\w+)\s*\(([^)]*)\)",
        # fn keyword: fn name(args)
        r"fn\s+(\w+)\s*\(([^)]*)\)",
        # proc/func keyword: proc name(args)
        r"(?:proc|func|method|sub)\s+(\w+)\s*\(([^)]*)\)",
    ]

    skip_words = {"if", "for", "while", "switch", "catch", "return", "else",
                  "case", "try", "except", "finally", "with", "elif",
                  "sizeof", "typeof", "instanceof", "new", "delete"}

    seen_funcs = set()
    for pattern in func_patterns:
        for match in re.finditer(pattern, source):
            name = match.group(1)
            if name.lower() in skip_words or len(name) < 2:
                continue
            if name in seen_funcs:
                continue
            seen_funcs.add(name)
            args_str = match.group(2) if match.lastindex >= 2 else ""
            args = [a.strip().split(" ")[-1].split(":")[0].strip()
                    for a in args_str.split(",") if a.strip()] if args_str else []
            line_num = source[:match.start()].count("\n") + 1
            result.functions.append(FunctionInfo(
                name=name, filepath=filepath, line_start=line_num, line_end=line_num + 5, args=args,
            ))

    # Universal class/struct/type patterns
    class_patterns = [
        r"(?:class|struct|interface|trait|protocol|enum|union|type)\s+(\w+)",
        r"(?:module|namespace)\s+(\w+)",
    ]

    seen_classes = set()
    for pattern in class_patterns:
        for match in re.finditer(pattern, source):
            name = match.group(1)
            if name in seen_classes or len(name) < 2:
                continue
            if name[0].islower() and name not in ("main",):
                continue  # Most class names start with uppercase
            seen_classes.add(name)
            line_num = source[:match.start()].count("\n") + 1
            result.classes.append(ClassInfo(
                name=name, filepath=filepath, line_start=line_num, line_end=line_num + 10,
            ))
