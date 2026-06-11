"""EVAL FIXTURE — known-vulnerable: SQL injection (CWE-89).

Sentinel MUST report sql_injection here. If it stops, the detector
regressed (this exact class was silently dead once — see handoff).
"""


def find_user(cursor, name):
    # VULNERABLE: f-string interpolation into SQL
    cursor.execute(f"SELECT * FROM users WHERE name = '{name}'")
    return cursor.fetchall()


def delete_user(cursor, user_id):
    # VULNERABLE: % formatting into SQL
    cursor.execute("DELETE FROM users WHERE id = %s" % user_id)
