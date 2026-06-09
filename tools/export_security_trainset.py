#!/usr/bin/env python3
"""
export_security_trainset.py — build a security fine-tune dataset from LeanAI's
own findings (#3 scaffold).

This is the REAL, runnable half of the fine-tune lever. It turns the findings
Sentinel has already persisted (~/.leanai/vulns/VULN-*.json) into instruction
-tuning examples: given a vulnerable function + context, the model should output
a verdict + reasoning + fix. Run Sentinel with --reason first so findings carry
model_reasoning / model_fix; otherwise the generic fix_suggestion is used.

It does NOT train a model — training needs your GPU + the 27B weights + hours.
See SECURITY_TRAINING.md for the LoRA command. This produces the JSONL you feed
that command.

Usage:
    python tools/export_security_trainset.py [--vuln-dir DIR] [--out FILE]
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _home():
    return os.environ.get("LEANAI_HOME", os.path.join(str(Path.home()), ".leanai"))


def build_examples(vuln_dir):
    """Yield instruction-tuning dicts from persisted findings."""
    for fp in sorted(glob.glob(os.path.join(vuln_dir, "VULN-*.json"))):
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        cls = d.get("vuln_class", "vulnerability")
        verdict = d.get("model_verdict") or "exploitable"
        reasoning = (d.get("model_reasoning")
                     or d.get("description")
                     or f"Potential {cls}.")
        fix = d.get("model_fix") or d.get("fix_suggestion") or ""
        cwe = d.get("cwe", "")
        instruction = (
            f"You are a security analyst. Decide whether the code is exploitable "
            f"by untrusted input. Reply: VERDICT, REASONING, FIX.")
        inp = (f"Class: {cls}{(' (' + cwe + ')') if cwe else ''}\n"
               f"File: {d.get('filepath','?')}:{d.get('line','?')}\n"
               f"Function: {d.get('function_name','?')}\n"
               f"Code/snippet:\n{d.get('code_snippet','')}")
        output = f"VERDICT: {verdict}\nREASONING: {reasoning}\nFIX: {fix}"
        yield {"instruction": instruction, "input": inp, "output": output}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vuln-dir", default=os.path.join(_home(), "vulns"))
    ap.add_argument("--out", default=os.path.join(_home(), "trainset", "security.jsonl"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for ex in build_examples(args.vuln_dir):
            out.write(json.dumps(ex) + "\n")
            n += 1
    print(f"[export] wrote {n} examples to {args.out}")
    if n == 0:
        print("[export] No findings yet. Run:  /sentinel --reason  first, "
              "then re-export.")
    return n


if __name__ == "__main__":
    main()
