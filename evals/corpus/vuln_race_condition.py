"""EVAL FIXTURE — known-vulnerable: TOCTOU race (CWE-362)."""
import os


def read_config(path):
    # VULNERABLE: exists()-then-open() check/use race
    if os.path.exists(path):
        handle = open(path)
        return handle.read()
    return None
