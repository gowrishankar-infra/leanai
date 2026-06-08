"""
M12 — Unified security audit + combined report.

A single capstone over the whole security stack: it reads what Sentinel
(VULN-*.json) and ChainBreaker (CHAIN-*.json) have persisted, deduplicates,
and produces ONE combined posture report in two formats:

  * SARIF 2.1.0 — two runs (LeanAI Sentinel + LeanAI ChainBreaker), so CI /
    code-scanning dashboards ingest both vulnerabilities and exploit chains.
  * Markdown — an executive posture summary + per-finding + per-chain detail.

This module only *reads + reports*; the /audit command in main.py does the
actual scanning (it needs the brain). That separation keeps SecurityAudit pure
and unit-testable with no brain, model, or engine.

Builds directly on M11's FindingsReport for the vulnerability side.
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.findings_report import FindingsReport, _sev_key, _SEV_SARIF


def _norm_sev(s: Any) -> str:
    """'Severity.HIGH' / 'HIGH' / Severity.HIGH -> 'HIGH'."""
    text = str(s)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


@dataclass
class AuditSummary:
    vuln_total: int = 0
    vuln_by_severity: Dict[str, int] = field(default_factory=dict)
    vuln_duplicates_collapsed: int = 0
    chain_total: int = 0
    chain_by_capability: Dict[str, int] = field(default_factory=dict)
    chain_by_severity: Dict[str, int] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)


class SecurityAudit:
    """Combine persisted Sentinel findings + ChainBreaker chains into one
    report. Call ``load()`` then any of summary()/to_sarif()/to_markdown()/write()."""

    def __init__(self, vuln_dir: str, chain_dir: str):
        self.vuln_dir = vuln_dir
        self.chain_dir = chain_dir
        self.findings_report = FindingsReport(vuln_dir)
        self.chains: List[Dict[str, Any]] = []

    # ── load ─────────────────────────────────────────────────────────
    def load(self) -> "SecurityAudit":
        self.findings_report.load()
        raw: Dict[str, Dict[str, Any]] = {}
        if self.chain_dir and os.path.isdir(self.chain_dir):
            for fp in glob.glob(os.path.join(self.chain_dir, "CHAIN-*.json")):
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception:
                    continue
                cid = data.get("chain_id") or os.path.basename(fp)[:-5]
                raw[cid] = data   # dedup by stable chain_id
        self.chains = sorted(
            raw.values(),
            key=lambda c: (-_sev_key(_norm_sev(c.get("severity", ""))),
                           -float(c.get("confidence", 0) or 0),
                           str(c.get("chain_id", ""))))
        return self

    # ── summary ──────────────────────────────────────────────────────
    def summary(self) -> AuditSummary:
        roll = self.findings_report.rollup()
        s = AuditSummary(
            vuln_total=roll.total,
            vuln_by_severity=dict(roll.by_severity),
            vuln_duplicates_collapsed=roll.duplicates_collapsed,
            chain_total=len(self.chains),
        )
        for c in self.chains:
            cap = str(c.get("capability", "unknown"))
            sev = _norm_sev(c.get("severity", "UNKNOWN"))
            s.chain_by_capability[cap] = s.chain_by_capability.get(cap, 0) + 1
            s.chain_by_severity[sev] = s.chain_by_severity.get(sev, 0) + 1
        return s

    # ── SARIF (two runs) ──────────────────────────────────────────────
    def to_sarif(self) -> Dict[str, Any]:
        sentinel_run = self.findings_report.to_sarif()["runs"][0]

        chain_rules: Dict[str, Dict[str, Any]] = {}
        chain_results = []
        for c in self.chains:
            cap = str(c.get("capability", "chain"))
            rule_id = f"exploit_chain/{cap}"
            chain_rules.setdefault(rule_id, {
                "id": rule_id,
                "name": f"Exploit chain ({cap})",
                "shortDescription": {"text": f"Multi-step attack chain to {cap}"},
            })
            steps = c.get("steps", []) or []
            entry = steps[0] if steps else {}
            related = []
            for st in steps[1:]:
                related.append({
                    "physicalLocation": {
                        "artifactLocation": {"uri": st.get("filepath", "")},
                        "region": {"startLine": int(st.get("line", 1) or 1)},
                    },
                    "message": {"text": f"{st.get('stage', '?')}: "
                                        f"{st.get('function_name', '?')}"},
                })
            chain_results.append({
                "ruleId": rule_id,
                "level": _SEV_SARIF.get(_norm_sev(c.get("severity", "")), "warning"),
                "message": {"text": c.get("impact_summary")
                            or c.get("narrative") or rule_id},
                "properties": {
                    "chain_id": c.get("chain_id"),
                    "entry_vuln_id": c.get("entry_vuln_id"),
                    "capability": cap,
                    "confidence": c.get("confidence"),
                    "steps": len(steps),
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": entry.get("filepath", "")},
                        "region": {"startLine": int(entry.get("line", 1) or 1)},
                    }
                }],
                "relatedLocations": related,
            })

        chain_run = {
            "tool": {"driver": {
                "name": "LeanAI ChainBreaker",
                "informationUri": "https://leanai.local",
                "rules": list(chain_rules.values()),
            }},
            "results": chain_results,
        }
        return {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [sentinel_run, chain_run],
        }

    # ── Markdown posture report ───────────────────────────────────────
    def to_markdown(self) -> str:
        s = self.summary()
        out: List[str] = []
        out.append("# LeanAI Security Audit")
        out.append("")
        out.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_")
        out.append("")
        out.append("## Executive summary")
        out.append("")
        out.append(f"- **Vulnerabilities:** {s.vuln_total}"
                   + (f" (deduplicated {s.vuln_duplicates_collapsed})"
                      if s.vuln_duplicates_collapsed else ""))
        if s.vuln_by_severity:
            sev_bits = ", ".join(
                f"{k} {s.vuln_by_severity[k]}"
                for k in sorted(s.vuln_by_severity, key=lambda x: -_sev_key(x)))
            out.append(f"  - by severity: {sev_bits}")
        out.append(f"- **Exploit chains:** {s.chain_total}")
        if s.chain_by_capability:
            cap_bits = ", ".join(
                f"{k} ({v})" for k, v in sorted(
                    s.chain_by_capability.items(), key=lambda kv: -kv[1]))
            out.append(f"  - by capability: {cap_bits}")
        out.append("")

        # Chains first — they are the highest-signal items.
        out.append("## Exploit chains")
        out.append("")
        if self.chains:
            for c in self.chains:
                steps = c.get("steps", []) or []
                out.append(f"### {c.get('chain_id', '?')} — "
                           f"{_norm_sev(c.get('severity', '?'))} — "
                           f"{c.get('capability', '?')}")
                out.append(f"- **Entry:** {c.get('entry_vuln_id', '?')}  "
                           f"**Confidence:** {c.get('confidence', '?')}  "
                           f"**Steps:** {len(steps)}")
                if c.get("impact_summary"):
                    out.append(f"- **Impact:** {c.get('impact_summary')}")
                if c.get("narrative"):
                    out.append(f"- {c.get('narrative')}")
                if steps:
                    out.append("- **Path:**")
                    for i, st in enumerate(steps, 1):
                        out.append(f"  {i}. `{st.get('filepath', '?')}:"
                                   f"{st.get('line', '?')}` "
                                   f"({st.get('stage', '?')}) "
                                   f"{st.get('function_name', '?')}")
                if c.get("fix_recommendation"):
                    out.append(f"- **Fix:** {c.get('fix_recommendation')}")
                out.append("")
        else:
            out.append("_No exploit chains detected._")
            out.append("")

        # Vulnerabilities
        out.append("## Vulnerabilities")
        out.append("")
        findings = self.findings_report.findings
        if findings:
            for f in findings:
                out.append(f"### {f.get('vuln_id', '?')} — "
                           f"{f.get('severity', '?')} — {f.get('vuln_class', '?')}")
                out.append(f"- **Location:** `{f.get('filepath', '?')}:"
                           f"{f.get('line', '?')}` ({f.get('function_name', '?')})")
                if f.get("description"):
                    out.append(f"- {f.get('description')}")
                out.append("")
        else:
            out.append("_No vulnerabilities detected._")
            out.append("")
        return "\n".join(out)

    # ── write ─────────────────────────────────────────────────────────
    def write(self, out_dir: str, fmt: str = "both") -> List[str]:
        """Write the report. fmt: 'markdown', 'sarif', or 'both'. Returns the
        list of paths written."""
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        paths: List[str] = []
        if fmt in ("markdown", "both"):
            p = os.path.join(out_dir, f"audit-{ts}.md")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(self.to_markdown())
            paths.append(p)
        if fmt in ("sarif", "both"):
            p = os.path.join(out_dir, f"audit-{ts}.sarif")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(self.to_sarif(), fh, indent=2)
            paths.append(p)
        return paths


def format_audit_console(s: AuditSummary) -> str:
    """Compact summary for the REPL."""
    lines = [
        f"  Vulnerabilities: {s.vuln_total}"
        + (f"  (deduped {s.vuln_duplicates_collapsed})"
           if s.vuln_duplicates_collapsed else ""),
    ]
    if s.vuln_by_severity:
        order = sorted(s.vuln_by_severity, key=lambda x: -_sev_key(x))
        lines.append("    severity: " + ", ".join(
            f"{k} {s.vuln_by_severity[k]}" for k in order))
    lines.append(f"  Exploit chains:  {s.chain_total}")
    if s.chain_by_capability:
        top = sorted(s.chain_by_capability.items(), key=lambda kv: -kv[1])[:5]
        lines.append("    capability: " + ", ".join(
            f"{k} ({v})" for k, v in top))
    return "\n".join(lines)
