#!/usr/bin/env python3
"""
selfcheck.py — end-to-end real-machine health & behavior report for LeanAI.

Runs every major subsystem against a COPY of a target project and writes a
structured report (selfcheck_report.json) plus a readable summary. Paste the
JSON back to your assistant to get evidence-based guidance/fixes.

SAFE BY DESIGN:
  * Copies the target project to a temp dir — never touches your original.
  * Uses a throwaway LEANAI_HOME — never touches your real ~/.leanai graph.
  * Read-only: never auto-fixes or edits your repo.
  * Every stage is isolated: one failure is recorded, the run continues.

MODES:
  default        : everything that does NOT need the model (fast, fully safe).
  --with-model   : also drives /sentinel --reason via a best-effort engine load
                   (slower, loads the 27B; failures are captured, not fatal).

USAGE:
  python tools/selfcheck.py [PROJECT_PATH] [--with-model] [--out FILE]
  # default PROJECT_PATH = "." (the LeanAI repo itself)
"""

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RESULTS = []


def stage(name):
    """Decorator-ish helper: time a stage, capture ok/error/metrics."""
    def run(fn):
        rec = {"stage": name, "ok": False, "ms": 0, "metrics": {}, "error": None,
               "sample": None}
        t0 = time.time()
        try:
            out = fn(rec) or {}
            rec["metrics"].update(out)
            rec["ok"] = True
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-1500:]
        rec["ms"] = int((time.time() - t0) * 1000)
        RESULTS.append(rec)
        status = "OK " if rec["ok"] else "ERR"
        extra = rec["error"] or json.dumps(rec["metrics"])[:90]
        print(f"  [{status}] {name:26} {rec['ms']:>6}ms  {extra}")
        return rec
    return run


def dep(modname):
    try:
        __import__(modname)
        return True
    except Exception:
        return False


def _resolve_real_model(real_home):
    """Find the user's actual model the way engine_v3 does: active_model.txt,
    then scan <real_home>/models for a .gguf. Returns a path or None."""
    try:
        cf = Path(real_home) / "active_model.txt"
        if cf.exists():
            p = cf.read_text(encoding="utf-8").strip()
            if p and os.path.exists(p):
                return p
        md = Path(real_home) / "models"
        if md.exists():
            for pat in ["qwen*7b*.gguf", "qwen*.gguf", "*.gguf"]:
                found = sorted(md.glob(pat))
                if found:
                    return str(found[0])
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project", nargs="?", default=".")
    ap.add_argument("--with-model", action="store_true")
    ap.add_argument("--out", default="selfcheck_report.json")
    args = ap.parse_args()

    src = os.path.abspath(args.project)
    tmp = tempfile.mkdtemp(prefix="leanai_selfcheck_")
    proj = os.path.join(tmp, "project")
    home = os.path.join(tmp, "home")
    # Resolve the REAL model path from the real home BEFORE we redirect
    # LEANAI_HOME, so --with-model can find the actual model while every other
    # stage stays isolated in the throwaway home.
    real_home = os.environ.get("LEANAI_HOME") or os.path.join(str(Path.home()), ".leanai")
    real_model = _resolve_real_model(real_home)
    os.environ["LEANAI_HOME"] = home
    os.makedirs(home, exist_ok=True)

    print(f"LeanAI self-check")
    print(f"  source project : {src}")
    print(f"  working copy   : {proj}")
    print(f"  LEANAI_HOME    : {home}  (throwaway)")
    print(f"  mode           : {'WITH MODEL' if args.with_model else 'no-model (safe/fast)'}")
    if args.with_model:
        print(f"  model (real)   : {real_model or 'NOT FOUND under ' + real_home + chr(92) + 'models'}")
    print("-" * 72)

    # copy the project (skip the heavy/irrelevant dirs)
    def ignore(d, names):
        return [n for n in names if n in {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            ".pytest_cache", "models"} or n.endswith(".pyc")]
    shutil.copytree(src, proj, ignore=ignore)

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "deps": {m: dep(m) for m in
                 ["chromadb", "sentence_transformers", "watchdog",
                  "llama_cpp", "numpy", "pyflakes"]},
        "with_model": args.with_model,
    }
    print(f"  env: py{env['python']} {env['platform']}")
    print(f"  deps: " + ", ".join(f"{k}={'Y' if v else 'n'}" for k, v in env['deps'].items()))
    print("-" * 72)

    brain_holder = {}

    @stage("brain_scan")
    def _brain(rec):
        from brain.project_brain import ProjectBrain
        b = ProjectBrain(proj)
        b.scan()
        brain_holder["b"] = b
        g = getattr(b, "graph", None)
        rec["sample"] = sorted(list(b._file_analyses.keys()))[:8]
        return {"files_indexed": len(b._file_analyses),
                "graph_nodes": len(getattr(g, "nodes", {}) or {}),
                "import_graph_files": len(getattr(g, "_file_imports", {}) or {})}

    @stage("sentinel_scan")
    def _sent(rec):
        from core.sentinel import SentinelEngine, Severity
        b = brain_holder.get("b")
        if b is None:
            raise RuntimeError("brain unavailable")
        f, st = SentinelEngine(b).scan(severity_floor=Severity.LOW, verbose=False, skip_tests=True)
        by_sev, by_cls, cwes = {}, {}, set()
        for v in f:
            by_sev[str(v.severity)] = by_sev.get(str(v.severity), 0) + 1
            by_cls[v.vuln_class] = by_cls.get(v.vuln_class, 0) + 1
            if v.cwe:
                cwes.add(v.cwe)
        rec["sample"] = [f"{v.severity} {v.vuln_class} {v.filepath}:{v.line}" for v in f[:8]]
        return {"findings": len(f), "by_severity": by_sev, "by_class": by_cls,
                "cwes": sorted(cwes), "sources": st.sources_found,
                "sinks": st.sinks_found, "taint_paths": st.taint_paths_traced}

    @stage("memory_sync")
    def _mem(rec):
        from core.memory_forge import MemoryForge
        b = brain_holder.get("b")
        mf = MemoryForge(project_path=proj, db_path=os.path.join(home, "graph.db"))
        mf.set_brain(b)
        mf.sync(verbose=False)
        s = mf.stats() if hasattr(mf, "stats") else {}
        return {"symbols": s.get("symbols"), "findings": s.get("findings"),
                "relations": s.get("relations"), "events": s.get("events")}

    @stage("incremental_crossfile")
    def _incr(rec):
        from core.sentinel import SentinelEngine
        from core.sentinel_incremental import IncrementalSentinel
        b = brain_holder.get("b")
        s = SentinelEngine(b)
        incr = IncrementalSentinel(s, type("F", (), {"vuln_dir": None,
                                   "forget_finding": lambda *a, **k: False})(),
                                   cross_file=True)
        sample = sorted(b._file_analyses.keys())[:1]
        if not sample:
            return {"note": "no files"}
        scope, extra = incr._expand_neighbours(sample)
        rec["sample"] = [incr._resolve_rel(p) for p in scope][:8]
        return {"target": sample[0], "scope_size": len(scope), "neighbours": extra}

    @stage("chainbreaker")
    def _chain(rec):
        from core.chainbreaker import ChainBreakerEngine, Severity as CSev
        b = brain_holder.get("b")
        cbe = ChainBreakerEngine(b)
        chains, st = cbe.analyze(severity_floor=CSev.MEDIUM, verbose=False)
        rec["sample"] = [f"{c.chain_id} {c.severity} {c.capability}" for c in chains[:5]]
        return {"chains": len(chains), "findings_loaded": st.findings_loaded,
                "entries_resolved": st.entries_resolved,
                "stale_skipped": st.stale_findings_skipped}

    @stage("findings_report")
    def _rep(rec):
        from core.findings_report import FindingsReport
        fr = FindingsReport(os.path.join(home, "vulns")).load()
        roll = fr.rollup()
        sarif = fr.to_sarif()
        path = fr.write(os.path.join(home, "reports"), "sarif")
        return {"total": roll.total, "by_severity": roll.by_severity,
                "sarif_valid": sarif.get("version") == "2.1.0",
                "wrote": os.path.basename(path)}

    @stage("security_audit")
    def _aud(rec):
        from core.security_audit import SecurityAudit
        a = SecurityAudit(os.path.join(home, "vulns"),
                          os.path.join(home, "chains")).load()
        s = a.summary()
        paths = a.write(os.path.join(home, "reports"), "both")
        return {"vuln_total": s.vuln_total, "chain_total": s.chain_total,
                "wrote": [os.path.basename(p) for p in paths]}

    @stage("config")
    def _cfg(rec):
        from core.leanai_config import LeanAIConfig
        c = LeanAIConfig()
        ok = c.set("snippet_limit", 12345)
        reloaded = LeanAIConfig().get("snippet_limit")
        return {"set_ok": ok, "reload": reloaded, "all": LeanAIConfig().all()}

    @stage("module_code_context")
    def _ctx(rec):
        from core.code_context import CodeContextBuilder, rerank_chunks
        b = brain_holder.get("b")
        cb = CodeContextBuilder(b)
        # pick any function from the first file
        sample_fn, sample_file = None, None
        for fp, an in b._file_analyses.items():
            fns = getattr(an, "functions", []) or []
            if fns:
                sample_file, sample_fn = fp, fns[0].qualified_name
                break
        ctx = cb.for_function(sample_file, sample_fn) if sample_fn else ""
        rr = rerank_chunks("authentication token", ["login token auth", "css color", "auth user token"], top_k=1)
        return {"built_context_for": f"{sample_file}::{sample_fn}",
                "context_chars": len(ctx), "rerank_top": rr[0] if rr else None}

    @stage("module_verify_fix")
    def _vf(rec):
        from core.verify_fix import check_syntax, VerifyFixLoop
        ok_good, _ = check_syntax("x = 1\n")
        ok_bad, err = check_syntax("def f(:\n")
        # fix loop with a stub fixer
        r = VerifyFixLoop(fix_fn=lambda c, e: "x = 1\n").run("x = (\n")
        return {"good_ok": ok_good, "bad_ok": ok_bad, "loop_fixed": r.fixed and r.ok}

    @stage("module_abstention")
    def _abs(rec):
        from core.abstention import self_consistency, CalibratedDecider
        seq = iter(["exploitable", "exploitable", "uncertain"])
        c = self_consistency(lambda: next(seq), n=3)
        d = CalibratedDecider(threshold=0.8).decide("ans", 0.4)
        return {"consensus": c.answer, "agreement": round(c.agreement, 2),
                "low_conf_abstains": d.abstained}

    @stage("watchguard")
    def _wg(rec):
        from core.watchguard import Watchguard
        from core.memory_forge import MemoryForge
        b = brain_holder.get("b")
        mf = MemoryForge(project_path=proj, db_path=os.path.join(home, "graph.db"))
        wg = Watchguard(project_path=proj, brain=b, memory_forge=mf,
                        leanai_home=home)
        st = wg.status()
        # quick start/stop if watchdog present (safe; immediate stop)
        started = False
        if dep("watchdog"):
            try:
                started = wg.start()
                time.sleep(0.3)
                wg.stop()
            except Exception:
                pass
        return {"constructs": True, "incremental_default": wg.incremental_enabled,
                "started_then_stopped": started}

    if args.with_model:
        @stage("sentinel_reason_with_model")
        def _reason(rec):
            from core.engine_v3 import LeanAIEngineV3 as Engine, GenerationConfig
            from core.sentinel import SentinelEngine, Severity
            b = brain_holder.get("b")
            if not real_model:
                raise RuntimeError(
                    f"no .gguf model found under {real_home}\\models "
                    "(set active_model.txt or place the model there)")
            # Pass the REAL model path explicitly so the throwaway LEANAI_HOME
            # doesn't hide it.
            eng = Engine(model_path=real_model)
            empty = {"n": 0, "total": 0}
            def model_fn(prompt):
                empty["total"] += 1
                r = eng.generate(prompt, config=GenerationConfig(max_tokens=256, temperature=0.1))
                txt = getattr(r, "text", str(r)) or ""
                if not txt.strip():
                    empty["n"] += 1
                return txt
            # smoke test BEFORE trusting the reasoning pass
            smoke = model_fn("Reply with the single word: OK")
            if not smoke.strip():
                raise RuntimeError(
                    f"model loaded but generation returned empty (model_path={real_model}). "
                    "Reasoning data would be void — fix model load before trusting --with-model.")
            rec["metrics"]["smoke_reply"] = smoke.strip()[:60]
            f, _ = SentinelEngine(b, model_fn=model_fn).scan(
                severity_floor=Severity.LOW, reason=True, verbose=False,
                skip_tests=True, reason_min_severity=Severity.MEDIUM, reason_max=8)
            verdicts = {}
            for v in f:
                verdicts[v.model_verdict or "(none)"] = verdicts.get(v.model_verdict or "(none)", 0) + 1
            rec["sample"] = [f"{v.vuln_class} verdict={v.model_verdict} :: {v.model_reasoning[:80]}"
                             for v in f[:6]]
            if empty["n"] and empty["n"] == empty["total"]:
                raise RuntimeError("every model call returned empty — void run")
            return {"model_path": real_model,
                    "model_calls": empty["total"], "empty_replies": empty["n"],
                    "findings_after_reason": len(f), "verdicts": verdicts}

    # ---- write report ----
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": env, "source_project": src,
        "summary": {"stages": len(RESULTS),
                    "ok": sum(1 for r in RESULTS if r["ok"]),
                    "errors": sum(1 for r in RESULTS if not r["ok"])},
        "stages": RESULTS,
    }
    out = os.path.abspath(args.out)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("-" * 72)
    print(f"  {report['summary']['ok']}/{report['summary']['stages']} stages OK, "
          f"{report['summary']['errors']} errors")
    print(f"  report written: {out}")
    print(f"  -> paste {args.out} back for evidence-based next steps.")
    # cleanup the working copy (report is already written)
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass


if __name__ == "__main__":
    main()
