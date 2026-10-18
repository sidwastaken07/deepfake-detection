"""Find the on-disk and remote layout of each corpus instead of trusting a guessed config.

Used by the Kaggle notebook (notebooks/phase1_kaggle.ipynb) to fill in the parts of
``configs/ingest_v1.yaml`` that could not be verified from the build environment:
Common Voice's release folder name, IndicVoices' shard paths, and the parquet column names.
The notebook writes the result to a new, committed config, so every run stays reproducible
from a file.
"""

from __future__ import annotations

import copy
import re
import tarfile
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

# Preference order when several columns could hold the transcript or speaker.
TEXT_COLUMNS = ("normalized", "text", "sentence", "transcript", "transcription", "verbatim")
SPEAKER_COLUMNS = ("speaker_id", "speaker", "client_id", "spk_id")
DURATION_COLUMNS = ("duration", "duration_s", "audio_duration")


class DiscoveryError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Parquet columns
# ---------------------------------------------------------------------------


def guess_columns(schema: Any, speaker_fallback: str | None = None) -> dict[str, str]:
    """Map the audio/text/speaker/duration roles onto a pyarrow schema's column names."""
    import pyarrow as pa

    names = list(schema.names)
    lower = {n.lower(): n for n in names}

    audio = [
        f.name
        for f in schema
        if (pa.types.is_struct(f.type) and f.type.get_field_index("bytes") >= 0)
        or pa.types.is_binary(f.type)
        or pa.types.is_large_binary(f.type)
    ]
    if len(audio) != 1:
        raise DiscoveryError(f"expected one audio column, found {audio}; columns: {names}")

    def pick(options: Sequence[str]) -> str | None:
        return next((lower[o] for o in options if o in lower), None)

    text = pick(TEXT_COLUMNS)
    speaker = pick(SPEAKER_COLUMNS) or (speaker_fallback if speaker_fallback in names else None)
    if text is None or speaker is None:
        raise DiscoveryError(f"no text or speaker column among {names}")
    cols = {"audio": audio[0], "text": text, "speaker": speaker}
    duration = pick(DURATION_COLUMNS)
    if duration is not None:
        cols["duration"] = duration
    return cols


# ---------------------------------------------------------------------------
# Hugging Face shard layout
# ---------------------------------------------------------------------------


def hf_language_patterns(paths: Iterable[str], lang_name: str) -> tuple[str, str]:
    """(remote fnmatch pattern, local glob) for one language's parquet shards.

    Finds the shallowest folder named ``lang_name`` (case-insensitive) that holds parquet
    files below it. ``*`` in the remote pattern spans folders (fnmatch); the local glob uses
    ``**`` to do the same with pathlib.
    """
    hits = []
    for p in paths:
        parts = PurePosixPath(p).parts
        if not p.endswith(".parquet"):
            continue
        for i, part in enumerate(parts[:-1]):
            if part.lower() == lang_name.lower():
                hits.append("/".join(parts[: i + 1]))
                break
    if not hits:
        tops = sorted({PurePosixPath(p).parts[0] for p in paths})[:30]
        raise DiscoveryError(
            f"no parquet files under a folder named {lang_name!r}; top level: {tops}"
        )
    prefix = min(hits, key=lambda h: (h.count("/"), h))
    return f"{prefix}/*.parquet", f"{prefix}/**/*.parquet"


# ---------------------------------------------------------------------------
# Common Voice archives
# ---------------------------------------------------------------------------


def find_cv_language_dir(member_names: Iterable[str], lang: str) -> str:
    """The archive folder (e.g. ``cv-corpus-25.0-2026-03-09/ta``) holding ``validated.tsv``."""
    for name in member_names:
        parts = PurePosixPath(name).parts
        if len(parts) >= 2 and parts[-1] == "validated.tsv" and parts[-2] == lang:
            return "/".join(parts[:-1])
    raise DiscoveryError(f"no {lang}/validated.tsv in the archive")


def _safe_target(dest: Path, name: str) -> Path:
    target = (dest / name).resolve()
    if not str(target).startswith(str(dest.resolve()) + "/"):
        raise DiscoveryError(f"archive member escapes destination: {name}")
    return target


def extract_cv_language(archive: Path, lang: str, dest: Path) -> Path:
    """Extract only ``<release>/<lang>/`` from a Common Voice tar or zip; return that folder."""
    dest.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            lang_dir = find_cv_language_dir((m.name for m in members), lang)
            wanted = [
                m
                for m in members
                if m.name.startswith(lang_dir + "/") and (m.isfile() or m.isdir())
            ]
            for m in wanted:
                _safe_target(dest, m.name)
            tf.extractall(dest, members=wanted)  # noqa: S202 - members vetted above
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            lang_dir = find_cv_language_dir(names, lang)
            wanted = [n for n in names if n.startswith(lang_dir + "/")]
            for n in wanted:
                _safe_target(dest, n)
            zf.extractall(dest, members=wanted)
    else:
        raise DiscoveryError(f"{archive} is neither a tar nor a zip archive")
    return dest / lang_dir


def size_from_details(details: Any) -> int | None:
    """Best-effort download size in bytes from a Mozilla Data Collective details payload."""
    found: list[int] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if re.search(r"size|bytes", str(k), re.I) and isinstance(v, (int, float)) and v > 0:
                    found.append(int(v))
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(details)
    return max(found) if found else None


# ---------------------------------------------------------------------------
# Resolved config
# ---------------------------------------------------------------------------


def resolve_config(base: dict[str, Any], updates: dict[str, Any], name: str) -> dict[str, Any]:
    """Deep-merge ``updates`` into a copy of ``base`` and rename it."""

    def merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v

    out = copy.deepcopy(base)
    merge(out, updates)
    out["name"] = name
    return out
