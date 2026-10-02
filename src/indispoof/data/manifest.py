"""The manifest: one JSONL row per audio clip, the single source of truth for every split.

The schema is fixed in Phase 0 (spec section 3). Fields may be *added* by extending
``SCHEMA`` below; renaming or repurposing an existing field is not allowed. Validation is
strict: a row with an unknown field, a missing field, a wrong type or an inconsistent
label/generator combination is rejected with the file and line number.

Rows are immutable once written. ``write_manifest`` refuses to overwrite an existing file;
a correction is a new manifest version with a new hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

LABELS = ("bonafide", "spoof")
LANGS = ("ta", "hi")
BONAFIDE_GENERATOR = "A00"
SPOOF_GENERATORS = tuple(f"A{i:02d}" for i in range(1, 9))  # A01..A08
GENERATOR_IDS = (BONAFIDE_GENERATOR, *SPOOF_GENERATORS)
BONAFIDE_FAMILY = "bonafide"
SPOOF_FAMILIES = ("flow_matching", "vits", "fastpitch_gan", "ar_codec_lm", "vc")
GENERATOR_FAMILIES = (BONAFIDE_FAMILY, *SPOOF_FAMILIES)
SPLITS = ("train", "dev", "eval", "held_out")
# Evaluation-time channel conditions (E4). Extend here, never ad hoc in a config.
CONDITIONS = (
    "clean",
    "opus_6k",
    "opus_12k",
    "opus_24k",
    "amr_nb",
    "mp3_32k",
    "mp3_64k",
    "band_8k",
)
PROCESSED_SAMPLE_RATE = 16000
PROCESSED_ROOT = PurePosixPath("data/processed")

# <lang>_<source>_<gen>_<index>. source has no underscores, so the index may carry
# suffixes such as a condition (ta_commonvoice_A00_000123-opus_12k) without ambiguity.
UTT_ID_RE = re.compile(
    r"^(?P<lang>[a-z]{2})_(?P<source>[a-z0-9]+)_(?P<gen>A\d{2})_(?P<index>[0-9A-Za-z][0-9A-Za-z_-]*)$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when a manifest row or file violates the schema."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestRow:
    """One audio clip. Field order here is the canonical serialisation order."""

    utt_id: str
    path: str
    sha256: str
    label: str
    lang: str
    generator_id: str
    generator_family: str
    speaker_id: str
    source_corpus: str
    text: str
    duration_s: float
    sample_rate: int
    condition: str
    split: str | None  # None until Phase 6 assigns splits
    qc_wer: float | None  # None for bonafide, and for spoof before the Phase 3 gate
    license: str
    seed: int | None  # generation seed where applicable

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCHEMA: tuple[str, ...] = tuple(f.name for f in fields(ManifestRow))
NULLABLE = frozenset({"split", "qc_wer", "seed"})


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return (isinstance(v, (int, float))) and not isinstance(v, bool) and math.isfinite(v)


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != "" and v == v.strip()


def validate_row(
    raw: dict[str, Any],
    *,
    processed: bool = False,
    gated: bool = False,
    assigned: bool = False,
) -> ManifestRow:
    """Validate a raw dict against the schema and return a typed row.

    Stage flags tighten the checks for later phases:
      processed: Phase 4 output (path under data/processed, 16 kHz).
      gated:     Phase 3 quality gate applied (every spoof row carries qc_wer).
      assigned:  Phase 6 protocol applied (every row carries a split).
    """
    if not isinstance(raw, dict):
        raise ManifestError(f"row must be a JSON object, got {type(raw).__name__}")

    keys = set(raw)
    missing = [k for k in SCHEMA if k not in keys]
    unknown = sorted(keys - set(SCHEMA))
    if missing:
        raise ManifestError(f"missing field(s): {', '.join(missing)}")
    if unknown:
        raise ManifestError(f"unknown field(s): {', '.join(unknown)}")

    def fail(field: str, why: str) -> ManifestError:
        return ManifestError(f"{field}={raw.get(field)!r}: {why}")

    for k in SCHEMA:
        if raw[k] is None and k not in NULLABLE:
            raise fail(k, "must not be null")

    for k in ("utt_id", "path", "speaker_id", "source_corpus", "text", "license"):
        if not _nonempty_str(raw[k]):
            raise fail(k, "must be a non-empty string without surrounding whitespace")

    if raw["label"] not in LABELS:
        raise fail("label", f"must be one of {LABELS}")
    if raw["lang"] not in LANGS:
        raise fail("lang", f"must be one of {LANGS}")
    if raw["generator_id"] not in GENERATOR_IDS:
        raise fail("generator_id", f"must be one of {GENERATOR_IDS}")
    if raw["generator_family"] not in GENERATOR_FAMILIES:
        raise fail("generator_family", f"must be one of {GENERATOR_FAMILIES}")
    if raw["condition"] not in CONDITIONS:
        raise fail("condition", f"must be one of {CONDITIONS}")
    if raw["split"] is not None and raw["split"] not in SPLITS:
        raise fail("split", f"must be null or one of {SPLITS}")

    if not isinstance(raw["sha256"], str) or not SHA256_RE.match(raw["sha256"]):
        raise fail("sha256", "must be 64 lowercase hex characters")

    m = UTT_ID_RE.match(raw["utt_id"])
    if not m:
        raise fail("utt_id", "must match <lang>_<source>_<gen>_<index>")
    if m["lang"] != raw["lang"]:
        raise fail("utt_id", f"lang part {m['lang']!r} disagrees with lang={raw['lang']!r}")
    if m["gen"] != raw["generator_id"]:
        raise fail("utt_id", f"gen part {m['gen']!r} disagrees with generator_id")

    path = PurePosixPath(raw["path"])
    if path.is_absolute() or "\\" in raw["path"] or ".." in path.parts:
        raise fail("path", "must be a relative POSIX path inside the repo")

    # Label, generator and family must agree.
    if raw["label"] == "bonafide":
        if raw["generator_id"] != BONAFIDE_GENERATOR:
            raise fail("generator_id", "bonafide rows must use A00")
        if raw["generator_family"] != BONAFIDE_FAMILY:
            raise fail("generator_family", "bonafide rows must use family 'bonafide'")
        if raw["qc_wer"] is not None:
            raise fail("qc_wer", "must be null for bonafide")
    else:
        if raw["generator_id"] == BONAFIDE_GENERATOR:
            raise fail("generator_id", "spoof rows must not use A00")
        if raw["generator_family"] == BONAFIDE_FAMILY:
            raise fail("generator_family", "spoof rows must not use family 'bonafide'")

    if not _is_number(raw["duration_s"]) or raw["duration_s"] <= 0:
        raise fail("duration_s", "must be a finite positive number")
    if not _is_int(raw["sample_rate"]) or raw["sample_rate"] <= 0:
        raise fail("sample_rate", "must be a positive integer")
    if raw["qc_wer"] is not None and (not _is_number(raw["qc_wer"]) or raw["qc_wer"] < 0):
        raise fail("qc_wer", "must be null or a finite non-negative number")
    if raw["seed"] is not None and not _is_int(raw["seed"]):
        raise fail("seed", "must be null or an integer")

    if processed:
        if raw["sample_rate"] != PROCESSED_SAMPLE_RATE:
            raise fail("sample_rate", f"processed audio must be {PROCESSED_SAMPLE_RATE} Hz")
        if path.parts[: len(PROCESSED_ROOT.parts)] != PROCESSED_ROOT.parts:
            raise fail("path", f"processed audio must live under {PROCESSED_ROOT}/")
    if gated and raw["label"] == "spoof" and raw["qc_wer"] is None:
        raise fail("qc_wer", "spoof rows must carry qc_wer after the quality gate")
    if assigned and raw["split"] is None:
        raise fail("split", "every row must carry a split after the protocol is applied")

    values = dict(raw)
    values["duration_s"] = float(raw["duration_s"])
    if raw["qc_wer"] is not None:
        values["qc_wer"] = float(raw["qc_wer"])
    return ManifestRow(**values)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes (audio files and manifests alike)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def hash_manifest(path: str | os.PathLike[str]) -> str:
    """The manifest's own hash: SHA-256 of its exact on-disk bytes."""
    return sha256_file(path)


def sidecar_path(path: str | os.PathLike[str]) -> Path:
    return Path(f"{path}.sha256")


def verify_manifest_hash(path: str | os.PathLike[str]) -> str:
    """Check a manifest against its .sha256 sidecar; return the hash."""
    side = sidecar_path(path)
    if not side.exists():
        raise ManifestError(f"{path}: no hash sidecar at {side}")
    recorded = side.read_text(encoding="utf-8").split()[0]
    actual = hash_manifest(path)
    if recorded != actual:
        raise ManifestError(f"{path}: hash mismatch (sidecar {recorded}, actual {actual})")
    return actual


# ---------------------------------------------------------------------------
# Read / write / merge
# ---------------------------------------------------------------------------


def serialize_row(row: ManifestRow) -> str:
    """Canonical one-line JSON: schema field order, native script kept as UTF-8."""
    return json.dumps(row.to_dict(), ensure_ascii=False, allow_nan=False)


def iter_manifest(path: str | os.PathLike[str], **stage: bool) -> Iterator[ManifestRow]:
    """Stream validated rows. Raises ManifestError naming the file and line on any defect."""
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                raise ManifestError(f"{path}:{lineno}: blank line")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ManifestError(f"{path}:{lineno}: invalid JSON ({e.msg})") from e
            try:
                row = validate_row(raw, **stage)
            except ManifestError as e:
                raise ManifestError(f"{path}:{lineno}: {e}") from e
            if row.utt_id in seen:
                raise ManifestError(f"{path}:{lineno}: duplicate utt_id {row.utt_id!r}")
            seen.add(row.utt_id)
            yield row


def read_manifest(path: str | os.PathLike[str], **stage: bool) -> list[ManifestRow]:
    return list(iter_manifest(path, **stage))


def write_manifest(
    path: str | os.PathLike[str], rows: Iterable[ManifestRow | dict[str, Any]], **stage: bool
) -> str:
    """Validate and write rows, plus a ``<path>.sha256`` sidecar. Returns the manifest hash.

    Refuses to overwrite: manifest rows are immutable once written.
    """
    path = Path(path)
    if path.exists():
        raise ManifestError(f"{path} already exists; manifests are immutable, write a new version")

    lines: list[str] = []
    seen: set[str] = set()
    for i, r in enumerate(rows):
        raw = r.to_dict() if isinstance(r, ManifestRow) else r
        try:
            row = validate_row(raw, **stage)
        except ManifestError as e:
            raise ManifestError(f"row {i}: {e}") from e
        if row.utt_id in seen:
            raise ManifestError(f"row {i}: duplicate utt_id {row.utt_id!r}")
        seen.add(row.utt_id)
        lines.append(serialize_row(row))
    if not lines:
        raise ManifestError("refusing to write an empty manifest")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive create: a concurrent writer can never be silently clobbered.
        with open(path, "x", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    except FileExistsError as e:
        raise ManifestError(f"{path} already exists; manifests are immutable") from e
    except BaseException:
        path.unlink(missing_ok=True)
        raise

    digest = hash_manifest(path)
    sidecar_path(path).write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def merge_manifests(*manifests: Iterable[ManifestRow]) -> list[ManifestRow]:
    """Concatenate row sets. Identical duplicates collapse; conflicting duplicates raise."""
    out: dict[str, ManifestRow] = {}
    for rows in manifests:
        for row in rows:
            prev = out.get(row.utt_id)
            if prev is None:
                out[row.utt_id] = row
            elif prev != row:
                raise ManifestError(f"conflicting rows for utt_id {row.utt_id!r}")
    return list(out.values())
