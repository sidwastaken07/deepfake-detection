"""Download a seeded subset of a Hugging Face dataset's shard files into ``data/raw/``.

The spec requires asking before any download over roughly 20 GB. ``fetch`` therefore lists
remote file sizes first and refuses to start when the selection exceeds ``max_gb``, unless the
caller passes ``allow_large=True`` after a human has agreed.

Shards are chosen evenly across the matching file list (with a seeded offset) instead of taking
the first N. Corpora are often sharded by region or speaker, so the first N shards would cover
fewer speakers.

Common Voice is not on Hugging Face; download it from Mozilla Data Collective (see
docs/kaggle.md).
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GB = 1024**3


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int


def select_files(
    files: Sequence[RemoteFile], pattern: str, n_files: int, seed: int
) -> list[RemoteFile]:
    """Evenly spaced, seeded choice of ``n_files`` among those matching ``pattern``."""
    matching = sorted((f for f in files if fnmatch.fnmatch(f.path, pattern)), key=lambda f: f.path)
    if not matching:
        dirs = sorted({f.path.split("/")[0] for f in files})[:20]
        raise FetchError(f"no remote files match {pattern!r}; top-level entries: {dirs}")
    if n_files >= len(matching):
        return matching
    stride = len(matching) / n_files
    offset = np.random.default_rng(seed).uniform(0, stride)
    return [matching[int(offset + i * stride)] for i in range(n_files)]


def list_remote(repo_id: str) -> list[RemoteFile]:
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    return [
        RemoteFile(e.path, int(e.size)) for e in entries if getattr(e, "size", None) is not None
    ]


def fetch(
    repo_id: str,
    pattern: str,
    n_files: int,
    dest: Path,
    *,
    seed: int,
    max_gb: float = 20.0,
    allow_large: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    chosen = select_files(list_remote(repo_id), pattern, n_files, seed)
    total = sum(f.size for f in chosen)
    print(f"{repo_id}: {len(chosen)} file(s) matching {pattern!r}, {total / GB:.2f} GB")
    for f in chosen:
        print(f"  {f.path}  {f.size / GB:.2f} GB")
    if total > max_gb * GB and not allow_large:
        raise FetchError(
            f"selection is {total / GB:.1f} GB, over the {max_gb:g} GB limit. Ask before "
            "downloading this much; rerun with --allow-large once agreed, or lower n_files."
        )
    if dry_run:
        return []

    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    return [
        Path(hf_hub_download(repo_id, f.path, repo_type="dataset", local_dir=dest)) for f in chosen
    ]
