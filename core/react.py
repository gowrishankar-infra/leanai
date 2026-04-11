"""
LeanAI — Tool-Augmented Reasoning (ReAct)
The model doesn't just THINK — it can ACT, OBSERVE, then THINK again.

How it works:
  1. Model receives a question
  2. Instead of answering from memory alone, it can call tools:
     - /brain_lookup: find a function or file in the project
     - /git_check: see recent changes to a file
     - /run_code: execute code to verify a claim
     - /search_memory: search past conversations
  3. Tool results are fed back to the model
  4. Model incorporates real data into its answer

Why this matters:
  A 32B model with access to real project data produces better answers
  than a 200B model guessing without data.

Example:
  User: "What's the time complexity of my search function?"
  
  Without ReAct:
    Model guesses: "Probably O(n) since it's a linear search"
  
  With ReAct:
    Model thinks: "Let me look at the actual code"
    → Calls brain_lookup("search") → Gets the function source
    → Sees nested loop
    → Model: "Your search is O(n²) because of the nested loop on line 45"
"""

import re
import time
from typing import Optional, Callable, Dict, List, Tuple


class ReActReasoner:
    """
    Tool-Augmented Reasoning engine.
    
    The model can call tools during its reasoning process,
    observe the results, and incorporate them into its answer.
    
    Usage:
        reasoner = ReActReasoner(model_fn=my_model)
        reasoner.register_tool("brain", brain_lookup_fn)
        reasoner.register_tool("git", git_history_fn)
        
        result = reasoner.reason("What does the generate function do?")
        # Model looked up the function, read the code, then explained it
    """

    def __init__(self, model_fn: Optional[Callable] = None, max_steps: int = 3):
        self.model_fn = model_fn
        self.max_steps = max_steps  # max tool calls per query
        self.tools: Dict[str, Dict] = {}
        self._stats = {
            "total_queries": 0,
            "tool_calls": 0,
            "tools_used": {},
        }

    def register_tool(self, name: str, fn: Callable, description: str = ""):
        """Register a tool the model can call."""
        self.tools[name] = {
            "fn": fn,
            "description": description,
        }

    def reason(self, query: str, context: str = "") -> 'ReActResult':
        """
        Reason about a query using tools.
        
        The process:
        1. Ask model to analyze the query and decide if tools are needed
        2. If tools needed: call them, get results
        3. Feed results back to model for final answer
        """
        if not self.model_fn:
            return ReActResult(answer="ReAct not configured.", tool_calls=[], steps=[])

        self._stats["total_queries"] += 1
        start = time.time()
        tool_calls = []
        steps = []

        # ── Step 1: Analyze query and gather tool data ────────
        tool_data = self._gather_tool_data(query, context)
        
        if tool_data:
            for tool_name, tool_query, tool_result in tool_data:
                tool_calls.append({
                    "tool": tool_name,
                    "query": tool_query,
                    "result": tool_result[:500],
                })
                steps.append(f"Called {tool_name}('{tool_query}') → got {len(tool_result)} chars")
                self._stats["tool_calls"] += 1
                self._stats["tools_used"][tool_name] = self._stats["tools_used"].get(tool_name, 0) + 1

        # ── Step 2: Generate final answer with tool data ──────
        system = (
            "You are a senior software engineer. Answer based on the REAL data provided below. "
            "Do not guess — use the actual code and information given. "
            "If tool data shows specific code, reference it with file paths and line numbers. "
            "Be specific and accurate."
        )

        user_prompt = query

        if context:
            user_prompt = f"[Project context]\n{context[:1000]}\n\n{query}"

        if tool_data:
            tool_section = "\n\n[REAL DATA FROM YOUR PROJECT — use this to answer accurately]\n"
            for tool_name, tool_query, tool_result in tool_data:
                tool_section += f"\n--- {tool_name}('{tool_query}') ---\n{tool_result[:800]}\n"
            user_prompt = tool_section + "\n\n" + user_prompt

        answer = self.model_fn(system, user_prompt)

        elapsed_ms = (time.time() - start) * 1000

        return ReActResult(
            answer=answer,
            tool_calls=tool_calls,
            steps=steps,
            elapsed_ms=elapsed_ms,
        )

    def _gather_tool_data(self, query: str, context: str = "") -> List[Tuple[str, str, str]]:
        """
        Analyze query and proactively gather relevant tool data.
        Instead of asking the model what tools to use (slow),
        we pattern-match the query and call relevant tools directly.
        """
        results = []
        query_lower = query.lower()

        # ── Brain lookups: if query mentions a function/file/class ──
        if "brain" in self.tools:
            brain_fn = self.tools["brain"]["fn"]
            
            # Look for function names in the query
            func_patterns = [
                r"(?:the\s+)?[`\"]?(\w{3,})\(\)[`\"]?",
                r"(?:function|method|def)\s+[`\"]?(\w{3,})[`\"]?",
                r"what does\s+[`\"]?(\w{3,})[`\"]?\s+do",
                r"how does\s+[`\"]?(\w{3,})[`\"]?\s+work",
                r"explain\s+[`\"]?(\w{3,})[`\"]?",
            ]
            skip_words = {"the", "this", "that", "what", "how", "does", "work",
                         "explain", "function", "method", "class", "file", "code",
                         "python", "javascript", "golang", "java", "rust"}
            
            for pattern in func_patterns:
                for match in re.finditer(pattern, query, re.IGNORECASE):
                    name = match.group(1)
                    if name.lower() not in skip_words and len(name) > 2:
                        try:
                            result = brain_fn(name)
                            if result and "not found" not in result.lower():
                                results.append(("brain_lookup", name, result))
                        except Exception:
                            pass

            # Look for file names
            file_patterns = [
                r"[`\"]?(\w+\.py)[`\"]?",
                r"[`\"]?(\w+\.js)[`\"]?",
                r"[`\"]?(\w+\.go)[`\"]?",
                r"[`\"]?(\w+\.rs)[`\"]?",
                r"[`\"]?(\w+\.java)[`\"]?",
            ]
            for pattern in file_patterns:
                for match in re.finditer(pattern, query):
                    filename = match.group(1)
                    try:
                        result = brain_fn(filename)
                        if result and "not found" not in result.lower():
                            results.append(("brain_file", filename, result))
                    except Exception:
                        pass

        # ── Git lookups: if query is about changes/history/bugs ──
        if "git" in self.tools:
            git_fn = self.tools["git"]["fn"]
            change_words = {"change", "changed", "modify", "modified", "broke",
                          "broken", "fix", "bug", "recent", "update", "last",
                          "commit", "history", "when", "who"}
            if any(w in query_lower for w in change_words):
                try:
                    # Get recent activity
                    result = git_fn("activity")
                    if result:
                        results.append(("git_history", "recent activity", result[:500]))
                except Exception:
                    pass

        # ── Memory lookups: if query references past conversations ──
        if "memory" in self.tools:
            memory_fn = self.tools["memory"]["fn"]
            memory_words = {"remember", "earlier", "before", "last time",
                          "previously", "we discussed", "you said", "I told"}
            if any(w in query_lower for w in memory_words):
                try:
                    result = memory_fn(query)
                    if result:
                        results.append(("memory_search", query[:50], result[:500]))
                except Exception:
                    pass

        return results[:self.max_steps]  # Max tools per query

    def stats(self) -> dict:
        return dict(self._stats)


class ReActResult:
    """Result from ReAct reasoning."""

    def __init__(self, answer: str, tool_calls: list, steps: list, elapsed_ms: float = 0):
        self.answer = answer
        self.tool_calls = tool_calls
        self.steps = steps
        self.elapsed_ms = elapsed_ms
        self.used_tools = len(tool_calls) > 0

    def summary(self) -> str:
        """Human-readable summary of what tools were used."""
        if not self.steps:
            return "Answered from model knowledge (no tools used)"
        
        lines = [f"Used {len(self.steps)} tool(s):"]
        for step in self.steps:
            lines.append(f"  → {step}")
        return "\n".join(lines)
