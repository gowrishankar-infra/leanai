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
        max_context_chars: int = 4000,
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
                budget -= len(hdc_ctx)

        # 5. DFSG — Dynamic Few-Shot Grounding
        # Inject real code examples from YOUR project so the model matches your style
        if include_brain and self.brain and budget > 500:
            fewshot = self._get_fewshot_examples(query, max_chars=min(budget, 2000))
            if fewshot:
                parts.append(fewshot)

        if not parts:
            return ""

        return "\n\n".join(parts)

    def _get_brain_context(self, query: str) -> str:
        """Get relevant project structure context with actual code content."""
        try:
            ctx_parts = []
            query_lower = query.lower()

            # Find mentioned files and inject their actual description + content
            matched_file = False
            matched_path = None
            for rel_path in list(self.brain._file_analyses.keys())[:200]:
                fname = os.path.basename(rel_path).lower().replace(".py", "")
                # Match filename without extension in query
                if fname in query_lower and len(fname) > 2:
                    desc = self.brain.describe_file(rel_path)
                    if desc and "not indexed" not in desc.lower():
                        ctx_parts.append(f"[File: {rel_path}]\n{desc[:600]}")
                        matched_file = True
                        matched_path = rel_path
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
                                matched_path = rel_path
                                break
                    if matched_file:
                        break

            # If a file was matched, also include actual file content snippet
            if matched_path and self.brain.config and self.brain.config.project_path:
                try:
                    full_path = os.path.join(self.brain.config.project_path, matched_path)
                    if os.path.exists(full_path):
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read(3000)  # first 3000 chars
                        ctx_parts.append(f"[Source code of {matched_path}]\n```\n{content}\n```")
                except Exception:
                    pass  # file read failed, continue without content

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

    def _get_fewshot_examples(self, query: str, max_chars: int = 2000) -> str:
        """
        DFSG: Dynamic Few-Shot Grounding.
        Finds similar functions in YOUR project and injects them as examples
        so the model generates code that matches YOUR patterns and style.

        Only triggers for code generation queries (write, create, implement, etc.)
        """
        try:
            query_lower = query.lower()

            # Only trigger for code generation/implementation queries
            gen_words = {
                "write", "create", "implement", "build", "add", "make",
                "generate", "design", "develop", "code", "function",
                "class", "method", "handler", "endpoint", "route",
                "refactor", "fix", "update", "modify", "improve",
                "show me", "how to", "how do", "example",
            }
            if not any(w in query_lower for w in gen_words):
                return ""

            # Extract action keywords from the query
            # e.g., "write a function to validate API tokens" → ["validate", "token"]
            action_words = self._extract_action_words(query_lower)
            if not action_words:
                return ""

            # Search brain's function index for matching functions
            best_match = None
            best_score = 0

            for func_name in list(self.brain.graph._function_lookup.keys())[:500]:
                func_lower = func_name.lower()

                # Skip very short names and dunder methods
                if len(func_name) < 4 or func_name.startswith("__"):
                    continue

                # Score: how many action words appear in the function name?
                score = sum(1 for w in action_words if w in func_lower)

                # Bonus for exact word match in function name
                func_parts = set(func_lower.replace("_", " ").split())
                score += sum(2 for w in action_words if w in func_parts)

                if score > best_score:
                    best_score = score
                    best_match = func_name

            if not best_match or best_score < 2:
                return ""

            # Get the function's actual source code from disk
            node_id = self.brain.graph._function_lookup.get(best_match)
            node = self.brain.graph.nodes.get(node_id) if node_id else None
            if not node or not node.filepath:
                return ""

            source_code = self._read_function_source(
                node.filepath,
                node.metadata.get("line_start"),
                node.metadata.get("line_end"),
                max_chars,
            )

            if not source_code or len(source_code.strip()) < 30:
                return ""

            # Format as few-shot example
            file_path = node.filepath
            return (
                f"[YOUR PROJECT'S CODE STYLE — match this pattern]\n"
                f"Example from your codebase ({file_path}):\n"
                f"```python\n{source_code}\n```\n"
                f"Follow the same naming conventions, error handling patterns, "
                f"docstring style, and return types shown above."
            )

        except Exception:
            return ""

    def _extract_action_words(self, query: str) -> list:
        """Extract meaningful action/domain words from a query."""
        # Remove common stop words and command words
        stop = {
            "a", "an", "the", "is", "are", "was", "were", "do", "does",
            "what", "how", "why", "when", "where", "who", "in", "on",
            "at", "to", "for", "of", "with", "and", "or", "but", "not",
            "can", "could", "would", "should", "will", "shall", "may",
            "i", "me", "my", "you", "your", "it", "its", "this", "that",
            "write", "create", "implement", "build", "add", "make",
            "show", "give", "tell", "help", "need", "want", "please",
            "function", "method", "class", "code", "new", "using",
            "leanai", "lean", "project", "file", "module",
        }

        words = []
        for word in query.split():
            # Clean punctuation
            clean = word.strip(".,!?()[]{}\"'`")
            if len(clean) > 2 and clean not in stop:
                words.append(clean)

        return words[:8]  # max 8 action words

    def _read_function_source(self, filepath: str, line_start: int = None,
                               line_end: int = None, max_chars: int = 1500) -> str:
        """Read a function's actual source code from disk."""
        try:
            if not self.brain.config or not self.brain.config.project_path:
                return ""

            full_path = os.path.join(self.brain.config.project_path, filepath)
            if not os.path.exists(full_path):
                return ""

            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                all_lines = f.readlines()

            if line_start and line_end and line_start > 0:
                # Read specific function lines
                start = max(0, line_start - 1)
                end = min(len(all_lines), line_end)
                source = "".join(all_lines[start:end])
            else:
                # Fallback: read first 50 lines
                source = "".join(all_lines[:50])

            # Truncate if too long
            if len(source) > max_chars:
                source = source[:max_chars] + "\n    # ... (truncated)"

            return source.rstrip()
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
