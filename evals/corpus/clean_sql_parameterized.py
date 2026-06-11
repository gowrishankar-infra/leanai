"""EVAL FIXTURE — clean look-alike: parameterized SQL.

Same shape as the vulnerable file, but safe. Sentinel must report
NOTHING here — a finding is a false positive.
"""


def find_user(cursor, name):
    # SAFE: parameterized placeholder
    cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
    return cursor.fetchall()


def delete_user(cursor, user_id):
    # SAFE: value passed as a parameter, not formatted in
    cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
