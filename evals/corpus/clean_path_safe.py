"""EVAL FIXTURE — clean look-alike: sanitized path handling."""
import os

SAFE_DIR = "/var/app/data"


def read_user_file(filename):
    # SAFE: basename strips traversal; join against fixed dir; no concat
    safe_name = os.path.basename(filename)
    full_path = os.path.join(SAFE_DIR, safe_name)
    handle = open(full_path)
    return handle.read()
