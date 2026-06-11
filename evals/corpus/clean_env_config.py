"""EVAL FIXTURE — clean look-alike: credentials from environment."""
import os


def get_api_key():
    # SAFE: no literal secret in source
    return os.environ.get("API_KEY", "")
