"""
LeanAI — Code Quality Enhancer
Post-processes code responses to catch bugs and missing best practices.

How it works:
  1. Model generates initial response (Pass 1)
  2. Enhancer extracts code blocks from the response
  3. Runs a focused review pass (Pass 2) — asks the model ONLY to find bugs
  4. If bugs/issues found, appends them to the response

This is how Claude Opus achieves higher quality — it self-reviews internally.
We do it explicitly as a second pass.

Adds ~30-60 seconds but catches bugs the first pass misses.
"""

import re
from typing import Optional, Callable, List, Tuple


# ── Language-specific review prompts ──────────────────────────

REVIEW_PROMPTS = {
    "python": (
        "Review this Python code for bugs and missing best practices. "
        "Check for: missing type hints, bare except clauses, mutable default arguments, "
        "missing input validation, unhandled None/empty cases, resource leaks (unclosed files), "
        "security issues (SQL injection, command injection), and performance problems. "
        "List ONLY the issues found. If no issues, say 'No issues found.'"
    ),
    "javascript": (
        "Review this JavaScript code for bugs and missing best practices. "
        "Check for: == instead of ===, missing async/await error handling, "
        "callback hell, prototype pollution, XSS vulnerabilities, "
        "missing null/undefined checks, memory leaks from event listeners, "
        "and missing 'use strict'. "
        "List ONLY the issues found. If no issues, say 'No issues found.'"
    ),
    "typescript": (
        "Review this TypeScript code for bugs and missing best practices. "
        "Check for: any types used unnecessarily, missing null checks, "
        "incorrect type assertions, missing async error handling, "
        "and unused imports. "
        "List ONLY the issues found. If no issues, say 'No issues found.'"
    ),
    "go": (
        "Review this Go code for bugs and missing best practices. "
        "Check for: unchecked error returns (err not checked), goroutine leaks, "
        "missing defer for cleanup, missing context.Context parameter, "
        "race conditions on shared state, http.Server without timeouts "
        "(ReadTimeout, WriteTimeout), and missing graceful shutdown. "
        "List ONLY the issues found. If no issues, say 'No issues found.'"
    ),
    "rust": (
        "Review this Rust code for bugs and missing best practices. "
        "Check for: unwrap() on Option/Result without proper handling, "
        "unnecessary clone(), unsafe blocks without justification, "
        "missing error propagation with ?, and lifetime issues. "
        "List ONLY the issues found. If no issues, say 'No issues found.'"
    ),
    "java": (
        "Review this Java code for bugs and missing best practices. "
        "Check for: null pointer risks, unchecked exceptions, "
        "resource leaks (use try-with-resources), missing equals/hashCode, "
        "raw types instead of generics, and thread safety issues. "
        "List ONLY the issues found. If no issues, say 'No issues found.'"
    ),
    "sql": (
        "Review this SQL for bugs and missing best practices. "
        "Check for: SQL injection vulnerabilities (use parameterized queries), "
        "missing indexes on WHERE/JOIN columns, SELECT * instead of specific columns, "
        "missing LIMIT on large tables, N+1 query patterns, "
        "and missing transaction handling for multi-step operations. "
        "List ONLY the issues found. If no issues, say 'No issues found.'"
    ),
    "cpp": (
        "Review this C++ code for bugs and missing best practices. "
        "Check for: buffer overflows, memory leaks (missing delete/free), "
        "use-after-free, null pointer dereference, missing bounds checks, "
        "raw pointers instead of smart pointers, and missing RAII. "
        "List ONLY the issues found. If no issues, say 'No issues found.'"
    ),
    "c": (
        "Review this C code for bugs and missing best practices. "
        "Check for: buffer overflows, memory leaks, null pointer dereference, "
        "missing bounds checks, integer overflow, format string vulnerabilities, "
        "and missing error handling on system calls. "
        "List ONLY the issues found. If no issues, say 'No issues found.'"
    ),
    "yaml": (
        "Review this YAML configuration for issues and missing best practices. "
        "Check for: missing test steps before deploy, hardcoded secrets, "
        "missing authentication/service connections, Dockerfile assumptions, "
        "tag strategy risks (latest vs build IDs), missing error handling/rollback, "
        "and missing environment variable configuration. "
        "List ONLY the issues found. If no issues, say 'No issues found.'"
    ),
}

DEFAULT_REVIEW = (
    "Review this code for bugs and missing best practices. "
    "Check for: missing input validation, unhandled edge cases (null, empty, negative), "
    "missing error handling, security vulnerabilities, resource leaks, "
    "performance issues, and missing type safety. "
    "List ONLY the issues found. If no issues, say 'No issues found.'"
)


def detect_language(text: str) -> str:
    """Detect the primary programming language from a response."""
    # Check for explicit language tags in code blocks
    lang_pattern = re.compile(r"```(\w+)")
    matches = lang_pattern.findall(text)
    if matches:
        lang = matches[0].lower()
        # Normalize
        lang_map = {
            "py": "python", "python": "python",
            "js": "javascript", "javascript": "javascript",
            "ts": "typescript", "typescript": "typescript",
            "go": "go", "golang": "go",
            "rs": "rust", "rust": "rust",
            "java": "java",
            "sql": "sql",
            "cpp": "cpp", "c++": "cpp",
            "c": "c",
            "yaml": "yaml", "yml": "yaml",
            "bash": "bash", "sh": "bash",
            "dockerfile": "yaml",
        }
        return lang_map.get(lang, lang)

    # Fallback: detect from content
    if "def " in text and "import " in text:
        return "python"
    if "func " in text and "package " in text:
        return "go"
    if "fn " in text and "let " in text and "mut " in text:
        return "rust"
    if "public class" in text or "public static" in text:
        return "java"
    if "const " in text and ("=>" in text or "function" in text):
        return "javascript"

    return "python"  # default


def extract_code_from_response(text: str) -> List[str]:
    """Extract code blocks from a response."""
    blocks = []
    pattern = re.compile(r"```\w*\n(.*?)```", re.DOTALL)
    for m in pattern.finditer(text):
        code = m.group(1).strip()
        if code and len(code) > 20:
            blocks.append(code)
    return blocks


class CodeQualityEnhancer:
    """
    Post-processes code responses to catch bugs and improve quality.
    
    Usage:
        enhancer = CodeQualityEnhancer(model_fn=my_model)
        
        # After generating a response
        original_response = "Here's a Go HTTP server..."
        enhanced = enhancer.enhance(original_response)
        # enhanced now includes additional bug findings if any
    """

    def __init__(self, model_fn: Optional[Callable] = None, enabled: bool = True):
        self.model_fn = model_fn
        self.enabled = enabled
        self._stats = {
            "total_reviews": 0,
            "issues_found": 0,
            "reviews_skipped": 0,
        }

    def enhance(self, response: str, query: str = "") -> str:
        """
        Review a response and append any found issues.
        
        Args:
            response: the original model response
            query: the user's original query (for context)
        
        Returns:
            Enhanced response with additional findings, or original if no issues found
        """
        if not self.enabled or not self.model_fn:
            return response

        # Extract code blocks
        code_blocks = extract_code_from_response(response)
        if not code_blocks:
            self._stats["reviews_skipped"] += 1
            return response

        # Detect language
        language = detect_language(response)

        # Get language-specific review prompt
        review_prompt = REVIEW_PROMPTS.get(language, DEFAULT_REVIEW)

        # Build review request with the code
        code_text = "\n\n".join(code_blocks)
        if len(code_text) > 2000:
            code_text = code_text[:2000]  # limit to avoid context overflow

        user_prompt = f"{review_prompt}\n\nCode to review:\n```{language}\n{code_text}\n```"
        system = (
            "You are a senior code reviewer. Find bugs, missing best practices, "
            "and security issues. Be specific and concise. "
            "List each issue as a bullet point starting with '- '. "
            "If the code is correct and complete, say exactly 'No issues found.'"
        )

        try:
            self._stats["total_reviews"] += 1
            review_result = self.model_fn(system, user_prompt)

            # Check if any issues were found
            review_clean = review_result.strip()
            if not review_clean or "no issues found" in review_clean.lower():
                return response

            # Count issues
            issues = [l for l in review_clean.split("\n") if l.strip().startswith("- ")]
            if not issues:
                # Try to detect issues without bullet format
                if len(review_clean) < 30 or "no issue" in review_clean.lower():
                    return response
                issues = [review_clean]

            self._stats["issues_found"] += len(issues)

            # Append findings to response
            enhanced = response.rstrip()
            enhanced += "\n\n### Additional Code Review Findings\n"
            enhanced += "A second review pass found these additional issues:\n"
            for issue in issues[:5]:  # max 5 issues
                issue_text = issue.strip()
                if issue_text.startswith("- "):
                    enhanced += f"\n{issue_text}"
                else:
                    enhanced += f"\n- {issue_text}"

            return enhanced

        except Exception:
            return response  # silently return original if review fails

    def should_review(self, query: str, response: str) -> bool:
        """Determine if a response should be reviewed."""
        if not self.enabled:
            return False

        # Only review responses that contain code
        if "```" not in response:
            return False

        # Only review if the query asked for code/explanation
        code_triggers = [
            "explain", "write", "implement", "create", "build",
            "review", "fix", "improve", "refactor", "give me",
            "show me", "code", "function", "class", "script",
        ]
        lower = query.lower()
        return any(trigger in lower for trigger in code_triggers)

    def stats(self) -> dict:
        return dict(self._stats)
