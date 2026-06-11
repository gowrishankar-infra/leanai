"""EVAL FIXTURE — known-vulnerable: command injection (CWE-78)."""
import os
import subprocess


def ping_host(host):
    # VULNERABLE: untrusted value concatenated into a shell command
    os.system("ping -c 1 " + host)


def run_command(cmd):
    # VULNERABLE: shell=True with a dynamic string
    subprocess.run(cmd, shell=True)
