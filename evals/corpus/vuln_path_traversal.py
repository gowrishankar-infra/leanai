"""EVAL FIXTURE — known-vulnerable: path traversal (CWE-22)."""


def read_user_file(base_dir, filename):
    # VULNERABLE: concatenated path, no basename/realpath validation
    handle = open(base_dir + "/" + filename)
    return handle.read()


def load_report(name):
    # VULNERABLE: f-string path from caller-controlled value
    handle = open(f"/var/reports/{name}.txt")
    return handle.read()
