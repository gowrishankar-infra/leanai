"""
M11 — Findings triage & reporting.

Turns the raw VULN-*.json finding store into something you can hand off:
a severity/category rollup, plus exports to SARIF 2.1.0 (for CI / code-scanning
dashboards) and a human-readable Markdown report. Reads the JSON store directly
(the source of truth that Sentinel persists), so it needs no DB connection and
is trivially testable.

This serves the offline / air-gapped security-analysis use case: teams who
cannot send code to a cloud scanner can still produce a standard SARIF artifact
locally.
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# severity -> sort rank (higher = worse) and SARIF level
_SEV_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}
_SEV_SARIF = {"CRITICAL": "error", "HIGH": "error", "MEDIUM": "warning",
              "LOW": "note", "INFO": "note"}


def _sev_key(sev: str) -> int:
    return _SEV_RANK.get(str(sev).upper(), 0)


@dataclass
class Rollup:
    total: int = 0
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)
    by_file: Dict[str, int] = field(default_factory=dict)
    duplicates_collapsed: int = 0


class FindingsReport:
    """Load + summarise + export the VULN-*.json finding store."""

    def __init__(self, vuln_dir: str):
        self.vuln_dir = vuln_dir
        self.findings: List[Dict[str, Any]] = []

    # ── load + dedup ─────────────────────────────────────────────────
    def load(self) -> "FindingsReport":
        raw: List[Dict[str, Any]] = []
        if self.vuln_dir and os.path.isdir(self.vuln_dir):
            for fp in glob.glob(os.path.join(self.vuln_dir, "VULN-*.json")):
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        raw.append(json.load(fh))
                except Exception:
                    continue
        # de-dup by fingerprint (fall back to file:line:class)
        seen: Dict[str, Dict[str, Any]] = {}
        dupes = 0
        for f in raw:
            key = f.get("fingerprint") or (
                f"{f.get('filepath')}:{f.get('line')}:{f.get('vuln_class')}")
            if key in seen:
                dupes += 1
                continue
            seen[key] = f
        self.findings = sorted(
            seen.values(),
            key=lambda f: (-_sev_key(f.get("severity", "")),
                           str(f.get("filepath", "")), f.get("line", 0)))
        self._dupes = dupes
        return self

    # ── rollup ───────────────────────────────────────────────────────
    def rollup(self) -> Rollup:
        r = Rollup(total=len(self.findings),
                   duplicates_collapsed=getattr(self, "_dupes", 0))
        for f in self.findings:
            sev = str(f.get("severity", "UNKNOWN")).upper()
            cat = str(f.get("vuln_class", "unknown"))
            fpath = str(f.get("filepath", "unknown"))
            r.by_severity[sev] = r.by_severity.get(sev, 0) + 1
            r.by_category[cat] = r.by_category.get(cat, 0) + 1
            r.by_file[fpath] = r.by_file.get(fpath, 0) + 1
        return r

    # ── SARIF 2.1.0 ──────────────────────────────────────────────────
    def to_sarif(self) -> Dict[str, Any]:
        rules: Dict[str, Dict[str, Any]] = {}
        results = []
        for f in self.findings:
            cls = str(f.get("vuln_class", "finding"))
            rules.setdefault(cls, {
                "id": cls,
                "name": cls.replace("_", " ").title(),
                "shortDescription": {"text": cls.replace("_", " ")},
            })
            sev = str(f.get("severity", "")).upper()
            results.append({
                "ruleId": cls,
                "level": _SEV_SARIF.get(sev, "warning"),
                "message": {"text": f.get("description")
                            or f"{cls} in {f.get('function_name', '?')}"},
                "properties": {
                    "severity": sev,
                    "confidence": f.get("confidence"),
                    "vuln_id": f.get("vuln_id"),
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": f.get("filepath", "")},
                        "region": {"startLine": int(f.get("line", 1) or 1)},
                    }
                }],
            })
        return {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {"driver": {
                    "name": "LeanAI Sentinel",
                    "informationUri": "https://leanai.local",
                    "rules": list(rules.values()),
                }},
                "results": results,
            }],
        }

    # ── Markdown ─────────────────────────────────────────────────────
    def to_markdown(self) -> str:
        r = self.rollup()
        out: List[str] = []
        out.append("# LeanAI Security Findings Report")
        out.append("")
        out.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_")
        out.append("")
        out.append(f"**Total findings:** {r.total}"
                    + (f"  (deduplicated {r.duplicates_collapsed})"
                       if r.duplicates_collapsed else ""))
        out.append("")
        if r.by_severity:
            out.append("## By severity")
            out.append("")
            for sev in sorted(r.by_severity, key=lambda s: -_sev_key(s)):
                out.append(f"- **{sev}**: {r.by_severity[sev]}")
            out.append("")
        if r.by_category:
            out.append("## By category")
            out.append("")
            for cat, n in sorted(r.by_category.items(),
                                 key=lambda kv: -kv[1]):
                out.append(f"- {cat}: {n}")
            out.append("")
        if self.findings:
            out.append("## Findings (highest severity first)")
            out.append("")
            for f in self.findings:
                out.append(
                    f"### {f.get('vuln_id', '?')} — {f.get('severity', '?')} "
                    f"— {f.get('vuln_class', '?')}")
                out.append(
                    f"- **Location:** `{f.get('filepath', '?')}:"
                    f"{f.get('line', '?')}` "
                    f"({f.get('function_name', '?')})")
                if f.get("confidence") is not None:
                    out.append(f"- **Confidence:** {f.get('confidence')}")
                if f.get("description"):
                    out.append(f"- {f.get('description')}")
                out.append("")
        else:
            out.append("_No findings._")
            out.append("")
        return "\n".join(out)

    # ── write to disk ────────────────────────────────────────────────
    def write(self, out_dir: str, fmt: str = "markdown") -> str:
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        if fmt == "sarif":
            path = os.path.join(out_dir, f"sentinel-report-{ts}.sarif")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.to_sarif(), fh, indent=2)
        else:
            path = os.path.join(out_dir, f"sentinel-report-{ts}.md")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.to_markdown())
        return path


def format_rollup_console(r: Rollup) -> str:
    """Compact rollup for printing in the REPL."""
    lines = [f"  Total findings: {r.total}"]
    if r.duplicates_collapsed:
        lines.append(f"  Duplicates collapsed: {r.duplicates_collapsed}")
    if r.by_severity:
        order = sorted(r.by_severity, key=lambda s: -_sev_key(s))
        lines.append("  Severity: " + ", ".join(
            f"{s} {r.by_severity[s]}" for s in order))
    if r.by_category:
        top = sorted(r.by_category.items(), key=lambda kv: -kv[1])[:5]
        lines.append("  Top categories: " + ", ".join(
            f"{c} ({n})" for c, n in top))
    return "\n".join(lines)
