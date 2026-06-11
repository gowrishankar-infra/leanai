"""EVAL FIXTURE — known-vulnerable: unsafe deserialization (CWE-502)."""
import pickle


def load_session(blob):
    # VULNERABLE: pickle.loads on untrusted bytes = RCE
    return pickle.loads(blob)
