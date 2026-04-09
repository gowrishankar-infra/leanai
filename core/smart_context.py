"""
LeanAI — Smart Context Injector
Automatically enriches every query with relevant context from all systems.

Instead of the model seeing just "fix the auth bug", it sees:
  - Project structure (from Brain)
  - Recent changes to auth files (from Git)
  - What you discussed about auth yesterday (from Sessions)
  - Your known preferences and patterns (from Memory)

This is what makes a 32B model outperform GPT-4 on YOUR project —
it has context GPT-4 can never access.
"""

import os
import time
from typing import Optional, List


class SmartContext:
    """
    Builds rich context from all LeanAI systems for every query.
    
    Usage:
        ctx = SmartContext(brain=brain, git=git, sessions=sessions, hdc=hdc)
        
        enriched_prompt = ctx.build("fix the auth bug in login.py")
        # Now includes: file structure, recent git changes, past discussions
    """

    def __init__(
        self,
        brain=None,
        git_intel=None,
        session_store=None,
        hdc=None,
        max_context_chars: int = 2000,
    ):
        self.brain = brain
        self.git = git_intel
        self.session_store = session_store
        self.hdc = hdc
        self.max_context_chars = max_context_chars

    def build(self, query: str, include_brain: bool = True,
              include_git: bool = True, include_sessions: bool = True,
              include_hdc: bool = True) -> str:
        """
        Build enriched context for a query.
        Returns a context string to prepend to the system prompt.
        """
        parts = []
        budget = self.max_context_chars

        # 1. Brain context — project structure relevant to query
        if include_brain and self.brain and budget > 0:
            brain_ctx = self._get_brain_context(query)
            if brain_ctx:
                parts.append(brain_ctx)
                budget -= len(brain_ctx)

        # 2. Git context — recent changes to mentioned files
        if include_git and self.git and self.git.is_available and budget > 0:
            git_ctx = self._get_git_context(query)
            if git_ctx:
                parts.append(git_ctx)
                budget -= len(git_ctx)

        # 3. Session context — relevant past conversations
        if include_sessions and self.session_store and budget > 0:
            session_ctx = self._get_session_context(query)
            if session_ctx:
                parts.append(session_ctx)
                budget -= len(session_ctx)

        # 4. HDC cache — similar past Q&A
        if include_hdc and self.hdc and self.hdc.count > 0 and budget > 0:
            hdc_ctx = self._get_hdc_context(query)
            if hdc_ctx:
                parts.append(hdc_ctx)

        if not parts:
            return ""

        return "\n\n".join(parts)

    def _get_brain_context(self, query: str) -> str:
        """Get relevant project structure context with actual code content."""
        try:
            ctx_parts = []
            query_lower = query.lower()

            # Find mentioned files and inject their actual description
            matched_file = False
            for rel_path in list(self.brain._file_analyses.keys())[:200]:
                fname = os.path.basename(rel_path).lower().replace(".py", "")
                # Match filename without extension in query
                if fname in query_lower and len(fname) > 2:
                    desc = self.brain.describe_file(rel_path)
                    if desc and "not indexed" not in desc.lower():
                        ctx_parts.append(f"[File: {rel_path}]\n{desc[:600]}")
                        matched_file = True
                        break

            # If no specific file matched, try partial matches
            if not matched_file:
                for rel_path in list(self.brain._file_analyses.keys())[:200]:
                    fname = os.path.basename(rel_path).lower()
                    # Check if any word in the query matches a filename
                    for word in query_lower.split():
                        if len(word) > 3 and word in fname:
                            desc = self.brain.describe_file(rel_path)
                            if desc and "not indexed" not in desc.lower():
                                ctx_parts.append(f"[File: {rel_path}]\n{desc[:600]}")
                                matched_file = True
                                break
                    if matched_file:
                        break

            # Find mentioned functions with details
            for func_name in list(self.brain.graph._function_lookup.keys())[:300]:
                if len(func_name) > 3 and func_name.lower() in query_lower:
                    info = self.brain.find_function(func_name)
                    if info and "not found" not in info.lower():
                        ctx_parts.append(f"[Function: {func_name}]\n{info[:400]}")
                        break

            # Always include project summary
            stats = self.brain.graph.stats()
            ctx_parts.append(
                f"[Project: {os.path.basename(self.brain.config.project_path)}] "
                f"{stats['files']} files, {stats['functions']} functions, "
                f"{stats['classes']} classes, {stats['edges']} dependency edges"
            )

            return "\n".join(ctx_parts) if ctx_parts else ""
        except Exception:
            return ""

    def _get_git_context(self, query: str) -> str:
        """Get relevant git history context."""
        try:
            ctx_parts = []
            query_lower = query.lower()

            # If query mentions a specific file, get its git history
            words = query.split()
            for word in words:
                if "." in word and not word.startswith(".") and len(word) > 3:
                    history = self.git.file_history(word, limit=3)
                    if "No history" not in history:
                        ctx_parts.append(f"[Git] {history[:300]}")
                        break

            # If query is about changes/recent/what happened
            change_words = {"change", "recent", "update", "modify", "broke", "fix", "bug", "last"}
            if any(w in query_lower for w in change_words):
                activity = self.git.recent_activity(days=3)
                if "No commits" not in activity:
                    # Just the summary line
                    lines = activity.split("\n")
                    ctx_parts.append(f"[Git] {lines[0]}")

            return "\n".join(ctx_parts) if ctx_parts else ""
        except Exception:
            return ""

    def _get_session_context(self, query: str) -> str:
        """Get relevant past conversation context."""
        try:
            results = self.session_store.search(query, limit=2)
            if not results:
                return ""

            ctx_parts = ["[Previous conversations]"]
            for exchange, session_id in results[:2]:
                ctx_parts.append(f"  Q: {exchange.query[:100]}")
                ctx_parts.append(f"  A: {exchange.response[:150]}")

            return "\n".join(ctx_parts)
        except Exception:
            return ""

    def _get_hdc_context(self, query: str) -> str:
        """Get similar past Q&A from HDC store."""
        try:
            results = self.hdc.search(query, top_k=2)
            if not results:
                return ""

            ctx_parts = []
            for text, similarity, meta in results:
                if similarity > 0.55:  # only include if reasonably similar
                    ctx_parts.append(f"[Similar] {text[:150]}")

            return "\n".join(ctx_parts) if ctx_parts else ""
        except Exception:
            return ""

    def build_system_prompt(self, base_system: str, query: str) -> str:
        """
        Build an enriched system prompt with context injected.
        This is the main entry point — call before every model generation.
        """
        context = self.build(query)
        if not context:
            return base_system

        return (
            f"{base_system}\n\n"
            f"IMPORTANT: You have access to the user's actual project. "
            f"Use the following real context to give specific, accurate answers. "
            f"Do NOT give generic examples — refer to the ACTUAL code below:\n\n"
            f"{context}\n\n"
            f"Answer based on this real project context. Be specific, not generic."
        )
