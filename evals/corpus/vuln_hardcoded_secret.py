"""EVAL FIXTURE — known-vulnerable: hardcoded credential (CWE-798).

The value below is a synthetic eval fixture, not a real credential.
"""

API_KEY = "FAKEEVALFIXTURE00000000001"


def connect():
    return {"auth": API_KEY}
