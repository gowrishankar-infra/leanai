"""EVAL FIXTURE — clean look-alike: subprocess with arg list, no shell."""
import subprocess


def ping_host(host):
    # SAFE: list argv, shell=False
    subprocess.run(["ping", "-c", "1", host], shell=False, check=True)
