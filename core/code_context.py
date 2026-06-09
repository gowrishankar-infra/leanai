"""
core/code_context.py — reusable structural code-context retrieval.

Generalizes the per-finding context builder that made Sentinel's reasoning pass
work, so ANY feature (review, explain, fix, Q&A) can feed the model the *right*
code instead of an isolated snippet. The single biggest lever for making a
small local model feel smart on code is: show it the relevant structure.

Two tools:
  * CodeContextBuilder.for_function(...) — a target function PLUS its callers,
    callees, and the file's imports, pulled from the brain's dependency graph
    (M8). This is "did the model see the data flow?", solved.
  * rerank_chunks(query, chunks) — order retrieved chunks by lexical relevance
    to a query WITHOUT an embedder (deterministic identifier/token overlap), so
    the most relevant code lands first in a limited context window even when the
    semantic embedder is unavailable (offline).

Pure stdlib + the brain. No model, no network. Never raises on bad input.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STOP = {
    "the", "a", "an", "and", "or", "is", "to", "of", "in", "for", "on", "with",
    "this", "that", "it", "def", "class", "self", "return", "import", "from",
    "if", "else", "true", "false", "none", "what", "how", "why", "does", "do",
}


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _IDENT.findall(text or "")
            if t.lower() not in _STOP and len(t) > 1]


def rerank_chunks(query: str, chunks: List[Any], top_k: int = 5,
                  text_key: str = "content") -> List[Any]:
    """Order chunks by lexical overlap with the query. Deterministic, no
    embedder. `chunks` may be strings or dicts (text read from `text_key`).
    Ties keep original order (stable). Returns the top_k most relevant."""
    if not chunks:
        return []
    q = set(_tokens(query))
    if not q:
        return list(chunks)[:top_k]

    def text_of(c: Any) -> str:
        if isinstance(c, str):
            return c
        if isinstance(c, dict):
            return str(c.get(text_key, "") or "")
        return str(c)

    scored = []
    for idx, c in enumerate(chunks):
        toks = _tokens(text_of(c))
        if not toks:
            scored.append((0.0, idx, c))
            continue
        tset = set(toks)
        overlap = len(q & tset)
        # density: reward chunks where the query terms are a real fraction,
        # not just present in a huge blob.
        density = overlap / (1 + (len(tset) ** 0.5))
        score = overlap + density
        scored.append((score, idx, c))
    # sort by score desc, then original index asc (stable for ties)
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored[:top_k]]


class CodeContextBuilder:
    """Build structural context for a function from the brain graph."""

    def __init__(self, brain: Any):
        self.brain = brain
        self._src_cache: Dict[str, str] = {}

    # ── public ────────────────────────────────────────────────────────
    def for_function(self, filepath: str, qualified_name: str, *,
                     max_chars: int = 1500, include_callers: bool = True,
                     include_callees: bool = True,
                     include_imports: bool = True) -> str:
        parts: List[str] = []
        func = self._find_func(filepath, qualified_name)
        src = self._func_source(filepath, func) if func else ""
        parts.append(f"FUNCTION ({filepath}::{qualified_name}):\n{src[:max_chars]}"
                     if src else f"FUNCTION ({filepath}::{qualified_name}): <source unavailable>")

        node_id = self._node_id(filepath, qualified_name)
        g = getattr(self.brain, "graph", None)
        if g is not None and node_id:
            if include_callers:
                callers = self._names(getattr(g, "_reverse_adj", {}).get(node_id, []))
                if callers:
                    parts.append("CALLED BY: " + ", ".join(callers[:8]))
            if include_callees:
                callees = self._names(getattr(g, "_adjacency", {}).get(node_id, []))
                if callees:
                    parts.append("CALLS: " + ", ".join(callees[:8]))

        if include_imports:
            imps = self._imports(filepath)
            if imps:
                parts.append("IMPORTS: " + ", ".join(sorted(imps)[:15]))
        return "\n".join(parts)

    def callers_of(self, filepath: str, qualified_name: str) -> List[str]:
        g = getattr(self.brain, "graph", None)
        nid = self._node_id(filepath, qualified_name)
        if g is None or not nid:
            return []
        return self._names(getattr(g, "_reverse_adj", {}).get(nid, []))

    # ── helpers ───────────────────────────────────────────────────────
    def _find_func(self, filepath: str, qualified_name: str):
        analysis = getattr(self.brain, "_file_analyses", {}).get(filepath)
        if not analysis:
            return None
        for fn in getattr(analysis, "functions", []) or []:
            if getattr(fn, "qualified_name", None) == qualified_name:
                return fn
        return None

    def _node_id(self, filepath: str, qualified_name: str) -> Optional[str]:
        nid = f"{filepath}:{qualified_name}"
        nodes = getattr(getattr(self.brain, "graph", None), "nodes", {}) or {}
        return nid if nid in nodes else None

    def _names(self, node_ids: List[str]) -> List[str]:
        nodes = getattr(getattr(self.brain, "graph", None), "nodes", {}) or {}
        out = []
        for nid in node_ids:
            n = nodes.get(nid)
            if n is not None:
                out.append(getattr(n, "name", str(nid)))
        return out

    def _read(self, filepath: str) -> str:
        if filepath in self._src_cache:
            return self._src_cache[filepath]
        root = getattr(getattr(self.brain, "config", None), "project_path", "") or ""
        full = os.path.join(root, filepath) if root else filepath
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
        except Exception:
            txt = ""
        self._src_cache[filepath] = txt
        return txt

    def _func_source(self, filepath: str, func) -> str:
        src = self._read(filepath)
        if not src:
            return ""
        lines = src.splitlines()
        start = max(0, getattr(func, "line_start", 1) - 1)
        end = getattr(func, "line_end", None) or (start + 60)
        end = min(len(lines), end)
        return "\n".join(lines[start:end])

    def _imports(self, filepath: str) -> set:
        g = getattr(self.brain, "graph", None)
        fi = getattr(g, "_file_imports", None) if g else None
        if not fi:
            return set()
        # keys may be abs or rel; match by suffix
        fp_norm = filepath.replace("\\", "/")
        for k, v in fi.items():
            if k.replace("\\", "/").endswith(fp_norm) or k.replace("\\", "/") == fp_norm:
                return set(v)
        return set()
