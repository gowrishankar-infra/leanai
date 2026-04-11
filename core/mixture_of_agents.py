"""
LeanAI — Mixture of Agents (MoA)
Generates answers from multiple "expert" perspectives, then synthesizes them.

Different from Swarm (which picks the best answer).
MoA SYNTHESIZES — combining insights from each perspective into one superior answer.

How it works:
  1. Same question sent to model with 3 different expert personas
  2. Each persona catches different things
  3. A synthesis pass combines the best of all three

Example:
  Query: "Review this authentication code"
  
  Perspective 1 (Security Expert):
    "SQL injection risk on line 12, password not hashed"
  
  Perspective 2 (Performance Expert):
    "Database query on every request, should use caching"
  
  Perspective 3 (Architecture Expert):
    "Auth logic mixed with business logic, violates SRP"
  
  Synthesis:
    Combined answer with ALL three insights — better than any single pass

Research: Together AI showed MoA with weak models can match GPT-4 quality.
"""

import time
from typing import Optional, Callable, List


# Expert personas — each catches different things
EXPERT_PERSONAS = {
    "security": (
        "You are a security expert. Focus on: vulnerabilities, injection risks, "
        "authentication flaws, data exposure, hardcoded secrets, input sanitization, "
        "CORS issues, and encryption. Flag every security concern you find."
    ),
    "performance": (
        "You are a performance expert. Focus on: time complexity, unnecessary loops, "
        "missing caching, database N+1 queries, memory leaks, resource cleanup, "
        "connection pooling, and scalability issues. Flag every performance concern."
    ),
    "architecture": (
        "You are a software architecture expert. Focus on: separation of concerns, "
        "SOLID principles, error handling patterns, testability, code organization, "
        "naming conventions, missing abstractions, and maintainability. "
        "Flag every design concern."
    ),
    "correctness": (
        "You are a code correctness expert. Focus on: logic errors, off-by-one bugs, "
        "null/None handling, edge cases, race conditions, type mismatches, "
        "unhandled exceptions, and incorrect assumptions. Flag every bug you find."
    ),
}


class MixtureOfAgents:
    """
    Multiple expert perspectives synthesized into one answer.
    
    Usage:
        moa = MixtureOfAgents(model_fn=my_model)
        result = moa.analyze("Review this code: def login(...)...")
        # result.final_answer has insights from all perspectives
    """

    def __init__(self, model_fn: Optional[Callable] = None,
                 perspectives: int = 3, enabled: bool = True):
        self.model_fn = model_fn
        self.num_perspectives = min(perspectives, len(EXPERT_PERSONAS))
        self.enabled = enabled
        self._stats = {
            "total_analyses": 0,
            "perspectives_generated": 0,
        }

    def analyze(self, query: str, context: str = "",
                personas: List[str] = None) -> 'MoAResult':
        """
        Analyze a query from multiple expert perspectives and synthesize.
        
        Args:
            query: the user's question or code to review
            context: project context from brain
            personas: specific personas to use (default: auto-select)
        """
        if not self.model_fn or not self.enabled:
            return MoAResult(final_answer="MoA not configured.", perspectives=[])

        self._stats["total_analyses"] += 1
        start = time.time()

        # Select personas based on query or use provided ones
        selected = personas or self._select_personas(query)
        
        # ── Step 1: Generate perspectives ─────────────────────
        perspectives = []
        for persona_key in selected:
            persona_prompt = EXPERT_PERSONAS.get(persona_key, EXPERT_PERSONAS["correctness"])
            
            user_prompt = query
            if context:
                user_prompt = f"[Project context]\n{context[:800]}\n\n{query}"

            try:
                result = self.model_fn(persona_prompt, user_prompt)
                if result and len(result.strip()) > 20:
                    perspectives.append({
                        "persona": persona_key,
                        "analysis": result,
                    })
                    self._stats["perspectives_generated"] += 1
            except Exception:
                pass

        if not perspectives:
            return MoAResult(final_answer="Could not generate perspectives.", perspectives=[])

        # ── Step 2: Synthesize all perspectives ───────────────
        synthesis_system = (
            "You are a senior technical lead. Below are analyses from multiple experts. "
            "Synthesize them into ONE comprehensive answer that includes the best insights "
            "from each expert. Do not repeat redundant points. Structure your answer with "
            "clear sections. If experts disagree, note both viewpoints."
        )

        perspectives_text = ""
        for p in perspectives:
            perspectives_text += f"\n\n--- {p['persona'].upper()} EXPERT ---\n{p['analysis'][:600]}"

        synthesis_prompt = (
            f"Original question: {query}\n\n"
            f"Expert analyses:{perspectives_text}\n\n"
            "Synthesize these into one comprehensive answer. Include the most important "
            "findings from each expert. Be structured and concise."
        )

        try:
            final_answer = self.model_fn(synthesis_system, synthesis_prompt)
        except Exception:
            # If synthesis fails, combine perspectives directly
            final_answer = "\n\n".join(
                f"**{p['persona'].title()} Analysis:**\n{p['analysis']}"
                for p in perspectives
            )

        elapsed_ms = (time.time() - start) * 1000

        return MoAResult(
            final_answer=final_answer,
            perspectives=perspectives,
            elapsed_ms=elapsed_ms,
        )

    def _select_personas(self, query: str) -> List[str]:
        """Auto-select the most relevant personas for a query."""
        lower = query.lower()

        # Always include correctness
        selected = ["correctness"]

        # Add security for security-related queries
        security_words = {"security", "auth", "login", "password", "token", "jwt",
                         "injection", "xss", "cors", "encrypt", "hash", "secret",
                         "vulnerability", "attack", "sanitize"}
        if any(w in lower for w in security_words):
            selected.append("security")

        # Add performance for performance-related queries
        perf_words = {"performance", "speed", "slow", "fast", "optimize", "cache",
                     "memory", "loop", "query", "database", "scale", "load",
                     "concurrent", "async", "batch"}
        if any(w in lower for w in perf_words):
            selected.append("performance")

        # Add architecture for design-related queries
        arch_words = {"design", "architect", "pattern", "refactor", "organize",
                     "structure", "clean", "solid", "principle", "module",
                     "separate", "decouple", "abstract", "interface"}
        if any(w in lower for w in arch_words):
            selected.append("architecture")

        # If only correctness matched, add the two most generally useful
        if len(selected) == 1:
            selected.extend(["security", "architecture"])

        return selected[:self.num_perspectives]

    def should_use_moa(self, query: str) -> bool:
        """Determine if a query would benefit from MoA analysis."""
        if not self.enabled:
            return False

        # MoA is best for code review and complex analysis
        moa_triggers = [
            "review", "audit", "analyze", "check", "evaluate",
            "improve", "refactor", "what's wrong", "find bugs",
            "security scan", "code quality",
        ]
        lower = query.lower()
        return any(trigger in lower for trigger in moa_triggers)

    def stats(self) -> dict:
        return dict(self._stats)


class MoAResult:
    """Result from Mixture of Agents analysis."""

    def __init__(self, final_answer: str, perspectives: list, elapsed_ms: float = 0):
        self.final_answer = final_answer
        self.perspectives = perspectives
        self.elapsed_ms = elapsed_ms
        self.num_perspectives = len(perspectives)

    def summary(self) -> str:
        """Summary of perspectives used."""
        if not self.perspectives:
            return "No perspectives generated"
        names = [p["persona"] for p in self.perspectives]
        return f"Analyzed from {len(names)} perspectives: {', '.join(names)}"
