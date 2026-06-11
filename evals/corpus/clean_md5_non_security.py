"""EVAL FIXTURE — clean look-alike: MD5 explicitly marked non-security."""
import hashlib


def make_cache_key(data):
    # SAFE: usedforsecurity=False — not a crypto use
    return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()
