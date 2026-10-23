"""Run provenance helpers that need no optional dependencies (no torch), so research scripts and their tests can
import them in environments without the deep-learning extras (for example the CI backend job)."""

from __future__ import annotations

import subprocess

from app.core.config import BACKEND_DIR


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BACKEND_DIR, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
