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
        indexer=None,
        max_context_chars: int = 4000,
    ):
        """
        brain:              ProjectBrain for AST/dependency lookups (lexical)
        git_intel:          GitIntel for recent-change context
        session_store:      SessionStore for past-conversation context
        hdc:                HDCMemory for cross-session pattern memory
        indexer:            (M6) ProjectIndexer for semantic code retrieval.
                            When provided, `_get_brain_context` uses semantic
                            search first and falls back to lexical matching.
                            Safe to leave as None — falls through to existing
                            behavior.
        max_context_chars:  hard cap on context size per query
        """
        self.brain = brain
        self.git = git_intel
        self.session_store = session_store
        self.hdc = hdc
        self.indexer = indexer
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

    # M6 fix — hard context budget for brain-derived context. Was
    # previously allowed to stack semantic (~2000 chars) + lexical
    # (~3000 chars) = 5000+ chars, which on a 4096-token llama.cpp KV
    # cache combined with conversation history + system prompt + query
    # produced 'decode: failed to find a memory slot' errors and
    # multi-minute generation times. Total cap now enforced.
    _BRAIN_CTX_BUDGET = 2500  # characters, hard cap on _get_brain_context
    _SEMANTIC_TOP_K   = 3     # was 4; tighter to stay within budget

    def _get_brain_context(self, query: str) -> str:
        """Get relevant project structure context with actual code content.

        M6 — Hybrid retrieval with a HARD CHARACTER BUDGET.
          1. Semantic hits render first (highest signal). Top-K reduced
             to 3 from 4 so they fit within budget.
          2. Lexical matches render second, skipping files already
             covered by semantic. Each contribution is budget-checked
             before being appended.
          3. Project summary is guaranteed to fit (80-200 chars) and is
             appended last — even if every other block got trimmed.

        The budget prevents the KV-cache overflow seen in testing when
        conceptual queries were matching lots of lexical file+function
        blocks on top of semantic chunks. Small loss on precision for
        very rich queries; big gain on generation stability and speed.
        """
        try:
            ctx_parts: List[str] = []
            budget = self._BRAIN_CTX_BUDGET
            query_lower = query.lower()

            def _append(block: str) -> bool:
                """Append if it fits in the remaining budget. Returns True
                if appended, False if skipped. Chunks are ATOMIC — either
                the whole block goes in or none of it does, to avoid
                mid-chunk truncation that would confuse the model."""
                nonlocal budget
                if len(block) + 1 > budget:  # +1 for the newline join
                    return False
                ctx_parts.append(block)
                budget -= len(block) + 1
                return True

            # ── PATH A: semantic retrieval (if indexer wired & populated) ──
            semantic_hits: List[dict] = []
            if self.indexer is not None:
                try:
                    if self.indexer.count() > 0:
                        semantic_hits = self.indexer.search(
                            query, top_k=self._SEMANTIC_TOP_K) or []
                except Exception:
                    semantic_hits = []

            # Reserve ~200 chars for the project summary at the end so it
            # always fits. Each semantic chunk capped at 900 chars so 3
            # chunks max ~2700 — we'll budget-check each one.
            PROJECT_SUMMARY_RESERVE = 220
            working_budget_cap = self._BRAIN_CTX_BUDGET - PROJECT_SUMMARY_RESERVE

            semantic_files: set = set()
            for hit in semantic_hits:
                chunk = hit.get('content', '') or ''
                rel_path = hit.get('relative_path', hit.get('file_path', ''))
                name = hit.get('name', '') or hit.get('chunk_type', 'chunk')
                relevance = hit.get('relevance', 0.0)
                start = hit.get('start_line', 0)
                end = hit.get('end_line', 0)
                if not chunk or not rel_path:
                    continue
                semantic_files.add(rel_path)
                # Each chunk capped at 900 chars — lets 3 fit under budget
                # with room for lexical contributions + summary.
                snippet = chunk if len(chunk) <= 900 else chunk[:900] + '\n...'
                block = (
                    f"[Relevant code — {rel_path}:{start}-{end} "
                    f"({name}, relevance {relevance:.0%})]\n"
                    f"```\n{snippet}\n```"
                )
                # Leave room for summary
                if budget - PROJECT_SUMMARY_RESERVE - len(block) < 0:
                    # Try shorter form — just header + first 300 chars
                    short_snippet = chunk[:300] + ('\n...' if len(chunk) > 300 else '')
                    short_block = (
                        f"[Relevant code — {rel_path}:{start}-{end} "
                        f"({name}, relevance {relevance:.0%})]\n"
                        f"```\n{short_snippet}\n```"
                    )
                    _append(short_block)
                else:
                    _append(block)
                # Stop if we've used most of the working budget
                if budget <= PROJECT_SUMMARY_RESERVE + 200:
                    break

            # ── PATH B: lexical filename match (existing behavior) ──
            matched_path = None
            matched_file = False
            for rel_path in list(self.brain._file_analyses.keys())[:200]:
                fname = os.path.basename(rel_path).lower().replace(".py", "")
                if fname in query_lower and len(fname) > 2:
                    if rel_path in semantic_files:
                        # Already covered by semantic — don't duplicate
                        matched_file = True
                        matched_path = rel_path
                        break
                    desc = self.brain.describe_file(rel_path)
                    if desc and "not indexed" not in desc.lower():
                        desc_short = desc[:500]  # was 600 — trim 100 for budget
                        _append(f"[File: {rel_path}]\n{desc_short}")
                    matched_file = True
                    matched_path = rel_path
                    break

            if not matched_file:
                for rel_path in list(self.brain._file_analyses.keys())[:200]:
                    fname = os.path.basename(rel_path).lower()
                    for word in query_lower.split():
                        if len(word) > 3 and word in fname:
                            if rel_path not in semantic_files:
                                desc = self.brain.describe_file(rel_path)
                                if desc and "not indexed" not in desc.lower():
                                    _append(f"[File: {rel_path}]\n{desc[:500]}")
                            matched_file = True
                            matched_path = rel_path
                            break
                    if matched_file:
                        break

            # Source-code block — only if file was lexically named AND not
            # already covered by semantic AND budget allows a useful chunk.
            if (matched_path and matched_path not in semantic_files
                    and self.brain.config and self.brain.config.project_path
                    and budget > PROJECT_SUMMARY_RESERVE + 300):
                try:
                    full_path = os.path.join(self.brain.config.project_path, matched_path)
                    if os.path.exists(full_path):
                        # Budget-aware read: never take more than what's
                        # left minus the project-summary reserve. Used to
                        # always read 3000 chars.
                        avail = budget - PROJECT_SUMMARY_RESERVE - 50  # margin
                        read_size = max(500, min(2000, avail))
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read(read_size)
                        _append(f"[Source code of {matched_path}]\n```\n{content}\n```")
                except Exception:
                    pass

            # Function-name lexical match
            for func_name in list(self.brain.graph._function_lookup.keys())[:300]:
                if len(func_name) > 3 and func_name.lower() in query_lower:
                    info = self.brain.find_function(func_name)
                    if info and "not found" not in info.lower():
                        _append(f"[Function: {func_name}]\n{info[:350]}")
                        break

            # Project summary — guaranteed to fit (we reserved space for it)
            stats = self.brain.graph.stats()
            summary = (
                f"[Project: {os.path.basename(self.brain.config.project_path)}] "
                f"{stats['files']} files, {stats['functions']} functions, "
                f"{stats['classes']} classes, {stats['edges']} dependency edges"
            )
            # Force-append even if over budget — this line is critical UX
            ctx_parts.append(summary)

            return "\n".join(ctx_parts) if ctx_parts else ""
        except Exception:
            # Absolute-silent fallback
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
