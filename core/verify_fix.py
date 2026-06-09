"""
core/verify_fix.py — generalized produce → verify → self-correct loop.

A lot of what makes a strong model *feel* reliable is that it implicitly checks
its own work. This makes that explicit and reusable: any code-producing output
can be run through compile + lint (and optionally execution) and self-repaired
via the model before it's shown to the user. A weaker model that verifies beats
a stronger one that doesn't.

Safe by default:
  * Syntax check (compile) and lint are static — they never execute the code.
  * Execution is opt-in: pass a `runner(code) -> (ok, output)` only if you want
    the code actually run (e.g. in the agents sandbox). Default does NOT run
    arbitrary model output.

Never raises; degrades gracefully (no fix_fn → just verifies; no pyflakes →
syntax-only lint).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class VerifyResult:
    ok: bool
    code: str
    attempts: int = 1
    fixed: bool = False
    errors: str = ""
    log: List[str] = field(default_factory=list)


def check_syntax(code: str) -> Tuple[bool, str]:
    """Static syntax check via compile(). No execution."""
    if not code or not code.strip():
        return False, "empty code"
    try:
        compile(code, "<verify>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"
    except Exception as e:  # ValueError on null bytes etc.
        return False, f"{type(e).__name__}: {e}"


def lint(code: str) -> List[str]:
    """Light static lint. Uses pyflakes if available; else an AST-based check
    for a couple of common issues. Never executes the code."""
    issues: List[str] = []
    ok, err = check_syntax(code)
    if not ok:
        return [err]
    # Prefer pyflakes if installed.
    try:
        from pyflakes.api import check as _pf_check          # type: ignore
        from pyflakes.reporter import Reporter               # type: ignore
        import io
        out, errs = io.StringIO(), io.StringIO()
        _pf_check(code, "<verify>", Reporter(out, errs))
        for line in (out.getvalue() + errs.getvalue()).splitlines():
            line = line.strip()
            if line:
                issues.append(line)
        return issues
    except Exception:
        pass
    # Fallback: AST-based smell checks (cheap, no false-positive-prone rules).
    try:
        tree = ast.parse(code)
    except Exception as e:
        return [str(e)]
    for node in ast.walk(tree):
        # bare except
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append(f"bare 'except:' at line {node.lineno}")
        # mutable default arg
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.args.defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    issues.append(f"mutable default arg in {node.name}() line {node.lineno}")
    return issues


class VerifyFixLoop:
    """Verify produced code and self-correct via the model up to max_attempts.

    fix_fn(code, errors) -> repaired_code   (the model; optional)
    runner(code) -> (ok, output)            (execution check; optional, opt-in)
    require_lint_clean: if True, lint issues also trigger a fix attempt.
    """

    def __init__(self, fix_fn: Optional[Callable[[str, str], str]] = None,
                 runner: Optional[Callable[[str], Tuple[bool, str]]] = None,
                 max_attempts: int = 3, require_lint_clean: bool = False):
        self.fix_fn = fix_fn
        self.runner = runner
        self.max_attempts = max(1, int(max_attempts))
        self.require_lint_clean = require_lint_clean

    def _verify_once(self, code: str) -> Tuple[bool, str]:
        ok, err = check_syntax(code)
        if not ok:
            return False, err
        if self.require_lint_clean:
            issues = lint(code)
            if issues:
                return False, "lint: " + "; ".join(issues[:5])
        if self.runner is not None:
            try:
                rok, out = self.runner(code)
                if not rok:
                    return False, f"runtime: {str(out)[:300]}"
            except Exception as e:
                return False, f"runner error: {e}"
        return True, ""

    def run(self, code: str) -> VerifyResult:
        log: List[str] = []
        cur = code or ""
        last_err = ""
        for attempt in range(1, self.max_attempts + 1):
            ok, err = self._verify_once(cur)
            log.append(f"attempt {attempt}: {'ok' if ok else err}")
            if ok:
                return VerifyResult(ok=True, code=cur, attempts=attempt,
                                    fixed=(attempt > 1), errors="", log=log)
            last_err = err
            if self.fix_fn is None or attempt == self.max_attempts:
                break
            try:
                repaired = self.fix_fn(cur, err)
            except Exception as e:
                log.append(f"fix_fn raised: {e}")
                break
            if not repaired or not repaired.strip() or repaired == cur:
                log.append("fix_fn produced no change")
                break
            cur = repaired
        return VerifyResult(ok=False, code=cur, attempts=len(log),
                            fixed=False, errors=last_err, log=log)
