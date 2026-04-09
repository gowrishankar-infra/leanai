"""
LeanAI — Reasoning Engine
Closes the reasoning gap between 32B local and 200B cloud models.

Techniques:
  1. Chain-of-Thought: Forces step-by-step thinking before answering
  2. Self-Critique: Model reviews its own logic and finds flaws
  3. Self-Refine: Fixes the flaws and produces improved answer
  4. Structured Decomposition: Breaks complex problems into sub-problems

This turns a single model call into a 3-pass reasoning pipeline:
  Pass 1: Think step by step → raw reasoning
  Pass 2: Critique your reasoning → find flaws
  Pass 3: Fix the flaws → final answer
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List


@dataclass
class ReasoningResult:
    """Result of a multi-pass reasoning process."""
    final_answer: str
    chain_of_thought: str = ""
    critique: str = ""
    refinement: str = ""
    num_passes: int = 0
    total_time_ms: float = 0
    technique: str = ""

    def summary(self) -> str:
        return (
            f"Reasoning: {self.technique} | {self.num_passes} passes | "
            f"{self.total_time_ms:.0f}ms"
        )


# ── Prompt Templates ──────────────────────────────────────────────

COT_SYSTEM = """You are an expert problem solver. Think through problems step by step.
For every question:
1. Break down what is being asked
2. Identify the key concepts and constraints
3. Work through the logic step by step
4. Check your reasoning for errors
5. State your final answer clearly

Be thorough and precise. Show your reasoning."""

COT_USER = """Think through this step by step:

{query}

First, break down the problem. Then work through each step. Finally, give your answer."""

CRITIQUE_SYSTEM = """You are a rigorous logic reviewer. Your job is to find flaws, errors, and gaps in reasoning.
Be critical. Check:
- Are the assumptions valid?
- Is the logic sound at each step?
- Are there missing cases or edge cases?
- Is the conclusion supported by the reasoning?
- Are there factual errors?

If you find errors, explain exactly what is wrong and what the correct answer should be."""

CRITIQUE_USER = """Review this reasoning for errors and gaps:

Question: {query}

Reasoning:
{reasoning}

List any errors, gaps, or flaws. If the reasoning is correct, say "No errors found." If there are errors, explain what should be fixed."""

REFINE_SYSTEM = """You are an expert who produces clear, accurate, well-structured answers.
Given a question, initial reasoning, and critique, produce the best possible final answer.
- Fix any errors identified in the critique
- Keep correct reasoning intact
- Be clear, concise, and well-organized
- If the critique found no errors, improve clarity and completeness"""

REFINE_USER = """Produce the best possible answer to this question.

Question: {query}

Initial reasoning:
{reasoning}

Critique:
{critique}

Write a clear, accurate, complete answer. Fix any errors found in the critique:"""

# ── Planning Templates ────────────────────────────────────────────

PLAN_SYSTEM = """You are an expert strategic planner. Create detailed, actionable plans.
For every planning task:
1. Define the goal clearly
2. Identify constraints and resources
3. Break down into phases with milestones
4. Identify risks and mitigations
5. Define success criteria
6. Create a timeline

Be specific and actionable, not vague."""

PLAN_USER = """Create a detailed plan for:

{query}

Structure your plan with:
- Goal
- Phases (with specific steps)
- Timeline
- Risks and mitigations
- Success criteria"""

DECOMPOSE_SYSTEM = """You are an expert at breaking complex problems into manageable sub-problems.
For each sub-problem, explain:
1. What needs to be solved
2. What information is needed
3. How it connects to the other sub-problems
4. The approach to solve it"""

DECOMPOSE_USER = """Break this complex problem into smaller sub-problems:

{query}

List each sub-problem, explain the approach for each, then synthesize into a complete solution."""


class ReasoningEngine:
    """
    Multi-pass reasoning engine that significantly improves answer quality.
    
    Usage:
        engine = ReasoningEngine(model_fn=my_model)
        
        # Full 3-pass reasoning (best quality)
        result = engine.reason("Why do tides happen?")
        
        # Planning mode
        result = engine.plan("How to migrate from monolith to microservices")
        
        # Decomposition mode
        result = engine.decompose("Design a distributed rate limiter")
    """

    def __init__(self, model_fn: Optional[Callable] = None):
        """
        Args:
            model_fn: function(system_prompt, user_prompt) -> str
        """
        self.model_fn = model_fn
        self._stats = {
            "total_queries": 0,
            "cot_queries": 0,
            "plan_queries": 0,
            "decompose_queries": 0,
            "avg_passes": 0,
        }

    def _call(self, system: str, user: str) -> str:
        if self.model_fn is None:
            raise RuntimeError("No model_fn provided to ReasoningEngine")
        return self.model_fn(system, user)

    def reason(self, query: str, max_passes: int = 3, verbose: bool = False) -> ReasoningResult:
        """
        Full chain-of-thought + self-critique + refinement pipeline.
        3 passes for maximum reasoning quality.
        """
        start = time.time()
        self._stats["total_queries"] += 1
        self._stats["cot_queries"] += 1

        # Pass 1: Chain-of-thought reasoning
        if verbose:
            print("  [Pass 1/3] Chain-of-thought reasoning...", flush=True)
        cot = self._call(COT_SYSTEM, COT_USER.format(query=query))
        if verbose:
            elapsed1 = (time.time() - start) * 1000
            print(f"  [Pass 1/3] Done ({elapsed1:.0f}ms)", flush=True)

        if max_passes == 1:
            return ReasoningResult(
                final_answer=cot, chain_of_thought=cot,
                num_passes=1, total_time_ms=(time.time() - start) * 1000,
                technique="chain-of-thought",
            )

        # Pass 2: Self-critique
        if verbose:
            print("  [Pass 2/3] Self-critique (finding flaws)...", flush=True)
        critique = self._call(
            CRITIQUE_SYSTEM,
            CRITIQUE_USER.format(query=query, reasoning=cot[:2000]),
        )
        if verbose:
            elapsed2 = (time.time() - start) * 1000
            print(f"  [Pass 2/3] Done ({elapsed2:.0f}ms)", flush=True)

        if max_passes == 2 or "no errors found" in critique.lower():
            if verbose and "no errors found" in critique.lower():
                print("  [Pass 2/3] No errors found — skipping refinement", flush=True)
            return ReasoningResult(
                final_answer=cot, chain_of_thought=cot, critique=critique,
                num_passes=2, total_time_ms=(time.time() - start) * 1000,
                technique="chain-of-thought + critique",
            )

        # Pass 3: Refinement
        if verbose:
            print("  [Pass 3/3] Refining answer...", flush=True)
        refined = self._call(
            REFINE_SYSTEM,
            REFINE_USER.format(query=query, reasoning=cot[:2000], critique=critique[:1000]),
        )

        elapsed = (time.time() - start) * 1000
        if verbose:
            print(f"  [Pass 3/3] Done ({elapsed:.0f}ms)", flush=True)
        self._update_avg_passes(3)

        return ReasoningResult(
            final_answer=refined, chain_of_thought=cot,
            critique=critique, refinement=refined,
            num_passes=3, total_time_ms=elapsed,
            technique="chain-of-thought + critique + refine",
        )

    def plan(self, query: str, verbose: bool = False) -> ReasoningResult:
        """Generate a structured plan with critique pass."""
        start = time.time()
        self._stats["total_queries"] += 1
        self._stats["plan_queries"] += 1

        if verbose:
            print("  [Pass 1] Generating plan...", flush=True)
        plan = self._call(PLAN_SYSTEM, PLAN_USER.format(query=query))
        if verbose:
            print(f"  [Pass 1] Done ({(time.time()-start)*1000:.0f}ms)", flush=True)

        if verbose:
            print("  [Pass 2] Reviewing plan...", flush=True)
        critique = self._call(
            CRITIQUE_SYSTEM,
            CRITIQUE_USER.format(query=f"Plan for: {query}", reasoning=plan[:2000]),
        )
        if verbose:
            print(f"  [Pass 2] Done ({(time.time()-start)*1000:.0f}ms)", flush=True)

        if "no errors found" in critique.lower():
            final = plan
            passes = 2
        else:
            if verbose:
                print("  [Pass 3] Improving plan...", flush=True)
            final = self._call(
                REFINE_SYSTEM,
                REFINE_USER.format(query=f"Plan for: {query}", reasoning=plan[:2000], critique=critique[:1000]),
            )
            passes = 3
            if verbose:
                print(f"  [Pass 3] Done ({(time.time()-start)*1000:.0f}ms)", flush=True)

        elapsed = (time.time() - start) * 1000
        self._update_avg_passes(passes)

        return ReasoningResult(
            final_answer=final, chain_of_thought=plan,
            critique=critique, refinement=final if passes == 3 else "",
            num_passes=passes, total_time_ms=elapsed,
            technique="structured planning",
        )

    def decompose(self, query: str, verbose: bool = False) -> ReasoningResult:
        """Break a complex problem into sub-problems and solve each."""
        start = time.time()
        self._stats["total_queries"] += 1
        self._stats["decompose_queries"] += 1

        if verbose:
            print("  [Pass 1/3] Decomposing problem...", flush=True)
        decomposition = self._call(
            DECOMPOSE_SYSTEM, DECOMPOSE_USER.format(query=query),
        )
        if verbose:
            print(f"  [Pass 1/3] Done ({(time.time()-start)*1000:.0f}ms)", flush=True)

        if verbose:
            print("  [Pass 2/3] Critiquing decomposition...", flush=True)
        critique = self._call(
            CRITIQUE_SYSTEM,
            CRITIQUE_USER.format(query=query, reasoning=decomposition[:2000]),
        )
        if verbose:
            print(f"  [Pass 2/3] Done ({(time.time()-start)*1000:.0f}ms)", flush=True)

        if verbose:
            print("  [Pass 3/3] Synthesizing solution...", flush=True)
        final = self._call(
            REFINE_SYSTEM,
            REFINE_USER.format(query=query, reasoning=decomposition[:2000], critique=critique[:1000]),
        )
        if verbose:
            print(f"  [Pass 3/3] Done ({(time.time()-start)*1000:.0f}ms)", flush=True)

        elapsed = (time.time() - start) * 1000
        self._update_avg_passes(3)

        return ReasoningResult(
            final_answer=final, chain_of_thought=decomposition,
            critique=critique, refinement=final,
            num_passes=3, total_time_ms=elapsed,
            technique="decomposition + synthesis",
        )

    def quick_reason(self, query: str) -> ReasoningResult:
        """Single-pass chain-of-thought (fast, moderate quality boost)."""
        return self.reason(query, max_passes=1)

    def _update_avg_passes(self, passes: int):
        n = self._stats["total_queries"]
        old = self._stats["avg_passes"]
        self._stats["avg_passes"] = (old * (n - 1) + passes) / n

    def stats(self) -> dict:
        return dict(self._stats)
