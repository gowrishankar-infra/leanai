"""
LeanAI — Autocomplete Engine
Provides sub-200ms code completions using the project brain's indexed data.

How it works (no model call needed for most completions):
  1. User types "engine.gen" in VS Code
  2. VS Code sends prefix to LeanAI's /complete endpoint
  3. Completer searches the brain's 1,540+ indexed functions
  4. Returns matching completions from YOUR actual codebase
  5. Total time: <50ms

For unknown code (not in project), falls back to:
  - Common Python/JS/Go patterns (builtins, stdlib)
  - Language keyword completions

This is how Tabby and Continue work — they DON'T use the big model
for autocomplete. They use a fast lookup + tiny model.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class Completion:
    """A single completion suggestion."""
    text: str               # the completion text
    label: str              # display label (e.g. "generate(query, config)")
    kind: str = "function"  # function, class, variable, keyword, snippet
    detail: str = ""        # extra info (e.g. "engine_v3.py")
    sort_order: int = 0     # lower = higher priority
    insert_text: str = ""   # what to actually insert (may differ from label)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "kind": self.kind,
            "detail": self.detail,
            "sortOrder": self.sort_order,
            "insertText": self.insert_text or self.text,
        }


# ── Common language completions (no brain needed) ─────────────

PYTHON_BUILTINS = [
    "print(", "len(", "range(", "str(", "int(", "float(", "list(",
    "dict(", "set(", "tuple(", "type(", "isinstance(", "hasattr(",
    "getattr(", "setattr(", "enumerate(", "zip(", "map(", "filter(",
    "sorted(", "reversed(", "any(", "all(", "min(", "max(", "sum(",
    "abs(", "round(", "open(", "input(", "super(", "property(",
    "classmethod(", "staticmethod(", "dataclass(",
]

PYTHON_KEYWORDS = [
    "def ", "class ", "if ", "elif ", "else:", "for ", "while ",
    "try:", "except ", "finally:", "with ", "as ", "import ",
    "from ", "return ", "yield ", "raise ", "pass", "break",
    "continue", "lambda ", "async ", "await ", "assert ",
    "global ", "nonlocal ",
]

PYTHON_SNIPPETS = {
    "def": "def ${1:name}(${2:args}):\n    ${3:pass}",
    "class": "class ${1:Name}:\n    def __init__(self${2:}):\n        ${3:pass}",
    "if": "if ${1:condition}:\n    ${2:pass}",
    "for": "for ${1:item} in ${2:items}:\n    ${3:pass}",
    "with": "with ${1:context} as ${2:var}:\n    ${3:pass}",
    "try": "try:\n    ${1:pass}\nexcept ${2:Exception} as e:\n    ${3:pass}",
    "main": 'if __name__ == "__main__":\n    ${1:main()}',
    "init": "def __init__(self${1:}):\n    ${2:pass}",
    "test": "def test_${1:name}(${2:}):\n    assert ${3:True}",
    "async": "async def ${1:name}(${2:args}):\n    ${3:pass}",
    "dataclass": "@dataclass\nclass ${1:Name}:\n    ${2:field}: ${3:str}",
}

JS_KEYWORDS = [
    "function ", "const ", "let ", "var ", "if ", "else ",
    "for ", "while ", "return ", "import ", "export ",
    "class ", "async ", "await ", "try ", "catch ",
    "switch ", "case ", "default:", "throw ", "new ",
]

GO_KEYWORDS = [
    "func ", "type ", "struct ", "interface ", "var ",
    "const ", "if ", "else ", "for ", "range ",
    "return ", "switch ", "case ", "default:", "defer ",
    "go ", "select ", "chan ", "map[", "make(",
]


class AutoCompleter:
    """
    Fast autocomplete engine using project brain data.
    
    Usage:
        completer = AutoCompleter(brain=project_brain)
        
        # Get completions for a prefix
        results = completer.complete("engine.gen", language="python")
        # Returns: [Completion(text="generate", label="generate(query, config)", ...)]
    """

    def __init__(self, brain=None):
        self.brain = brain
        self._function_cache: Dict[str, List[Tuple[str, str, str]]] = {}  # name -> [(full_sig, file, kind)]
        self._class_cache: Dict[str, List[Tuple[str, str]]] = {}  # name -> [(file, detail)]
        self._last_rebuild = 0
        self._rebuild_interval = 30  # rebuild cache every 30 seconds
        self._stats = {"total_completions": 0, "brain_hits": 0, "fallback_hits": 0}

        if brain:
            self._rebuild_cache()

    def _rebuild_cache(self):
        """Rebuild the completion cache from the brain's indexed data."""
        if not self.brain:
            return

        self._function_cache.clear()
        self._class_cache.clear()

        try:
            # Index all functions
            for func_name, location in self.brain.graph._function_lookup.items():
                if len(func_name) < 2:
                    continue
                # Parse location (e.g. "core/engine_v3.py:generate")
                file_part = location.split(":")[0] if ":" in location else ""
                key = func_name.lower()
                if key not in self._function_cache:
                    self._function_cache[key] = []
                self._function_cache[key].append((func_name, file_part, "function"))

            # Index all classes
            for rel_path, analysis in self.brain._file_analyses.items():
                if hasattr(analysis, 'classes'):
                    for cls in analysis.classes:
                        cls_name = cls if isinstance(cls, str) else getattr(cls, 'name', str(cls))
                        key = cls_name.lower()
                        if key not in self._class_cache:
                            self._class_cache[key] = []
                        self._class_cache[key].append((rel_path, cls_name))

            self._last_rebuild = time.time()
        except Exception:
            pass

    def complete(
        self,
        prefix: str,
        language: str = "python",
        file_path: str = "",
        line: str = "",
        max_results: int = 10,
    ) -> List[Completion]:
        """
        Get completions for a prefix.
        
        Args:
            prefix: the text to complete (e.g. "engine.gen", "def calc")
            language: programming language
            file_path: current file path (for context)
            line: full current line (for context)
            max_results: maximum completions to return
        
        Returns:
            List of Completion objects, sorted by relevance
        """
        self._stats["total_completions"] += 1
        start = time.time()

        # Rebuild cache if stale
        if self.brain and (time.time() - self._last_rebuild) > self._rebuild_interval:
            self._rebuild_cache()

        results: List[Completion] = []
        prefix_lower = prefix.strip().lower()

        if not prefix_lower:
            return results

        # 1. Brain completions — search indexed functions and classes
        if self.brain and self._function_cache:
            brain_results = self._complete_from_brain(prefix_lower)
            results.extend(brain_results)
            if brain_results:
                self._stats["brain_hits"] += 1

        # 2. Language-specific completions
        lang_results = self._complete_from_language(prefix_lower, language)
        results.extend(lang_results)

        # 3. Snippet completions
        if language == "python":
            snippet_results = self._complete_snippets(prefix_lower)
            results.extend(snippet_results)

        if not results and not brain_results if self.brain else True:
            self._stats["fallback_hits"] += 1

        # Deduplicate by text
        seen = set()
        unique = []
        for r in results:
            if r.text not in seen:
                seen.add(r.text)
                unique.append(r)

        # Sort by priority then alphabetically
        unique.sort(key=lambda c: (c.sort_order, c.text))

        return unique[:max_results]

    def _complete_from_brain(self, prefix: str) -> List[Completion]:
        """Search the brain's indexed functions and classes."""
        results = []

        # Handle dot notation: "engine.gen" → search for "gen" in functions
        search_prefix = prefix
        context = ""
        if "." in prefix:
            parts = prefix.rsplit(".", 1)
            context = parts[0]
            search_prefix = parts[1] if len(parts) > 1 else ""

        # Search functions
        for func_key, entries in self._function_cache.items():
            if func_key.startswith(search_prefix) and search_prefix:
                for func_name, file_part, kind in entries:
                    results.append(Completion(
                        text=func_name,
                        label=f"{func_name}()",
                        kind="function",
                        detail=file_part,
                        sort_order=0,  # brain results have highest priority
                        insert_text=f"{func_name}(",
                    ))
            # Also match partial (e.g. "gen" matches "generate")
            elif search_prefix and search_prefix in func_key and len(search_prefix) >= 2:
                for func_name, file_part, kind in entries:
                    results.append(Completion(
                        text=func_name,
                        label=f"{func_name}()",
                        kind="function",
                        detail=file_part,
                        sort_order=1,  # partial matches slightly lower
                        insert_text=f"{func_name}(",
                    ))

        # Search classes
        for cls_key, entries in self._class_cache.items():
            if cls_key.startswith(search_prefix) and search_prefix:
                for file_part, cls_name in entries:
                    results.append(Completion(
                        text=cls_name,
                        label=cls_name,
                        kind="class",
                        detail=file_part,
                        sort_order=0,
                        insert_text=cls_name,
                    ))

        return results

    def _complete_from_language(self, prefix: str, language: str) -> List[Completion]:
        """Complete from language keywords and builtins."""
        results = []

        if language == "python":
            keywords = PYTHON_KEYWORDS + PYTHON_BUILTINS
        elif language in ("javascript", "typescript", "js", "ts"):
            keywords = JS_KEYWORDS
        elif language == "go":
            keywords = GO_KEYWORDS
        else:
            keywords = PYTHON_KEYWORDS  # default

        for kw in keywords:
            kw_lower = kw.strip().lower()
            if kw_lower.startswith(prefix) and prefix != kw_lower:
                results.append(Completion(
                    text=kw.strip(),
                    label=kw.strip(),
                    kind="keyword",
                    detail=language,
                    sort_order=5,  # lower priority than brain results
                    insert_text=kw,
                ))

        return results

    def _complete_snippets(self, prefix: str) -> List[Completion]:
        """Complete from code snippets."""
        results = []
        for trigger, snippet in PYTHON_SNIPPETS.items():
            if trigger.startswith(prefix) and prefix != trigger:
                # Show first line as label
                first_line = snippet.split("\n")[0].replace("${1:", "").replace("${2:", "").replace("${3:", "").replace("}", "")
                results.append(Completion(
                    text=trigger,
                    label=f"⚡ {first_line}",
                    kind="snippet",
                    detail="snippet",
                    sort_order=3,
                    insert_text=snippet,
                ))
        return results

    def update_brain(self, brain):
        """Update the brain reference (e.g. after /brain scan)."""
        self.brain = brain
        self._rebuild_cache()

    def stats(self) -> dict:
        return {
            "functions_indexed": len(self._function_cache),
            "classes_indexed": len(self._class_cache),
            **self._stats,
        }
