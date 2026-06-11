"""EVAL FIXTURE — known-vulnerable: weak crypto (CWE-327)."""
import hashlib


def hash_password(user_password):
    # VULNERABLE: MD5 for password hashing (security context)
    return hashlib.md5(user_password.encode()).hexdigest()


def sign_token(data):
    # VULNERABLE: SHA1 for signing
    return hashlib.sha1(data).hexdigest()
