"""Check and pack the audio a manifest points at.

Kaggle sessions are ephemeral, so the audio a manifest references has to be persisted
(e.g. as a private Kaggle Dataset) and re-attached later. ``pack`` copies exactly the files a
manifest lists, keeping their repo-relative paths. ``verify`` confirms every file exists and
still matches its recorded SHA-256.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from indispoof.data.manifest import ManifestRow, sha256_file


def verify(rows: Sequence[ManifestRow], root: str | os.PathLike[str] = ".") -> list[str]:
    """Return one problem string per missing or altered file (empty list means all good)."""
    problems = []
    for r in rows:
        p = Path(root) / r.path
        if not p.is_file():
            problems.append(f"{r.utt_id}: missing {r.path}")
        elif sha256_file(p) != r.sha256:
            problems.append(f"{r.utt_id}: sha256 mismatch for {r.path}")
    return problems


def pack(rows: Sequence[ManifestRow], dest: str | os.PathLike[str], root: str = ".") -> int:
    """Copy each row's file to ``dest/<path>``, verifying the hash. Returns files copied."""
    problems = verify(rows, root)
    if problems:
        raise FileNotFoundError(f"{len(problems)} problem(s), first: {problems[0]}")
    copied = 0
    for r in rows:
        target = Path(dest) / r.path
        if target.is_file() and sha256_file(target) == r.sha256:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(root) / r.path, target)
        copied += 1
    return copied
