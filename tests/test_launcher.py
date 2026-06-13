"""
Tests for install_launcher.py — the `leanai` command generator.

Pure logic (no disk scanning, no real install needed). Run:
    python -m pytest tests/test_launcher.py -q
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import install_launcher as L


def test_windows_launcher_dir_uses_userprofile(monkeypatch):
    monkeypatch.setenv("USERPROFILE", r"C:\Users\someone")
    d = L.launcher_dir("Windows")
    assert d == Path(r"C:\Users\someone") / "bin"
    assert L.launcher_path("Windows").name == "leanai.bat"


def test_posix_launcher_dir_uses_local_bin():
    d = L.launcher_dir("Linux")
    assert d == Path.home() / ".local" / "bin"
    assert L.launcher_path("Linux").name == "leanai"


def test_render_windows_bat_cds_and_runs_main():
    repo = Path(r"D:\Downloads\LeanAi\leanai-phase1\leanai")
    out = L.render_launcher(repo, r"C:\Python313\python.exe", "Windows")
    assert '@echo off' in out
    assert f'cd /d "{repo}"' in out
    assert '"C:\\Python313\\python.exe" main.py %*' in out
    assert "\r\n" in out                     # CRLF for .bat


def test_render_posix_script_cds_and_execs_main():
    repo = Path("/home/u/leanai")
    out = L.render_launcher(repo, "/usr/bin/python3", "Linux")
    assert out.startswith("#!/usr/bin/env bash")
    assert 'cd "/home/u/leanai" || exit 1' in out
    assert 'exec "/usr/bin/python3" main.py "$@"' in out


def test_install_writes_executable_posix(tmp_path, monkeypatch):
    # Point HOME at a temp dir so ~/.local/bin lands there.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    repo, d, path, content = L.install(os_name="Linux", python_exe="/usr/bin/python3")
    assert path.exists()
    assert path.read_text() == content
    assert os.access(path, os.X_OK)          # chmod +x applied


def test_on_path_detects_membership(monkeypatch):
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/home/u/.local/bin"]))
    assert L.on_path(Path("/home/u/.local/bin")) is True
    assert L.on_path(Path("/nope/bin")) is False


def test_repo_root_is_this_repo():
    # The script lives at the repo root, next to main.py.
    assert (L.repo_root() / "install_launcher.py").exists()
