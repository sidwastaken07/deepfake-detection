"""Phase 1: bonafide corpus ingestion.

One reader per corpus format turns a corpus that is already on disk under ``data/raw/`` into
candidates (speaker, text, audio locator). A seeded, speaker-balanced sampler picks clips, and
each accepted clip becomes a manifest row with ``label=bonafide`` and ``generator_id=A00``.
Every row records its corpus license.

Downloading is separate (see ``indispoof.data.fetch``). Ingestion reads only the corpus's own
metadata files, never a directory scan of audio, and it never modifies anything it reads.
Audio held inside parquet files is written out once to ``<root>/_extracted/`` so that every
manifest row points at a real file with a hash.
"""

from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np

from indispoof.data.manifest import (
    BONAFIDE_FAMILY,
    BONAFIDE_GENERATOR,
    LANGS,
    ManifestRow,
    read_manifest,
    sha256_file,
    write_manifest,
)

RAW_ROOT = PurePosixPath("data/raw")
CORPUS_KEY_RE = re.compile(r"^[a-z0-9]+$")

# Unicode blocks of each language's native script.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "ta": ((0x0B80, 0x0BFF),),
    "hi": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
}


class IngestError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """NFC and whitespace collapse only. Content is never rewritten here."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def script_ratio(text: str, lang: str) -> float:
    """Fraction of letters and marks that are in ``lang``'s native script."""
    ranges = SCRIPT_RANGES[lang]
    letters = [ch for ch in text if unicodedata.category(ch)[0] in "LM"]
    if not letters:
        return 0.0
    native = sum(any(lo <= ord(ch) <= hi for lo, hi in ranges) for ch in letters)
    return native / len(letters)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    native_id: str  # unique within one corpus + language
    speaker: str  # raw speaker label as the corpus gives it
    text: str
    duration_hint: float | None
    locator: Any  # reader-specific


class CorpusReader(Protocol):
    def candidates(self) -> Iterator[Candidate]: ...

    def materialize(self, cands: Sequence[Candidate]) -> dict[str, Path]:
        """Return native_id -> audio path (relative to the repo root) for each candidate."""
        ...


class CommonVoiceReader:
    """An extracted Common Voice language directory: validated.tsv, clips/, clip_durations.tsv."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.tsv = root / "validated.tsv"
        if not self.tsv.exists():
            raise IngestError(
                f"{self.tsv} not found; is {root} an extracted Common Voice language?"
            )

    def _durations(self) -> dict[str, float]:
        path = self.root / "clip_durations.tsv"
        if not path.exists():
            return {}
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            return {r["clip"]: float(r["duration[ms]"]) / 1000.0 for r in reader}

    def candidates(self) -> Iterator[Candidate]:
        durations = self._durations()
        with open(self.tsv, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            missing = {"client_id", "path", "sentence"} - set(reader.fieldnames or ())
            if missing:
                raise IngestError(f"{self.tsv}: missing columns {sorted(missing)}")
            for r in reader:
                yield Candidate(
                    native_id=r["path"],
                    speaker=r["client_id"],
                    text=r["sentence"] or "",
                    duration_hint=durations.get(r["path"]),
                    locator=self.root / "clips" / r["path"],
                )

    def materialize(self, cands: Sequence[Candidate]) -> dict[str, Path]:
        return {c.native_id: c.locator for c in cands if c.locator.exists()}


def _sniff_extension(data: bytes) -> str:
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"OggS":
        return ".ogg"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return ".mp3"
    raise IngestError("unrecognised audio encoding in parquet bytes")


def _classlabel_names(schema: Any, column: str) -> list[str] | None:
    """Names of a Hugging Face ClassLabel column, read from the parquet schema metadata."""
    meta = (schema.metadata or {}).get(b"huggingface")
    if not meta:
        return None
    try:
        feature = json.loads(meta)["info"]["features"][column]
    except (KeyError, ValueError, TypeError):
        return None
    if isinstance(feature, dict) and feature.get("_type") == "ClassLabel":
        return list(feature.get("names") or []) or None
    return None


class ParquetReader:
    """Hugging Face-style parquet shards with an audio column (struct of bytes/path, or bytes)."""

    def __init__(self, root: Path, files: str, columns: dict[str, str]) -> None:
        for key in ("audio", "text", "speaker"):
            if key not in columns:
                raise IngestError(f"columns.{key} is required for parquet corpora")
        self.root = root
        self.columns = columns
        self.files = sorted(root.glob(files))
        if not self.files:
            raise IngestError(f"no parquet files match {root}/{files}")
        self.extract_dir = root / "_extracted"

    def candidates(self) -> Iterator[Candidate]:
        import pyarrow.parquet as pq

        meta_cols = [self.columns["text"], self.columns["speaker"]]
        if "duration" in self.columns:
            meta_cols.append(self.columns["duration"])
        for fi, path in enumerate(self.files):
            pf = pq.ParquetFile(path)
            names = set(pf.schema_arrow.names)
            missing = [c for c in [*meta_cols, self.columns["audio"]] if c not in names]
            if missing:
                raise IngestError(f"{path}: missing columns {missing}; available: {sorted(names)}")
            labels = _classlabel_names(pf.schema_arrow, self.columns["speaker"])
            # Shards in different folders may share a file name, so the id keeps the folder.
            stem = path.relative_to(self.root).with_suffix("").as_posix().replace("/", "__")
            table = pf.read(columns=meta_cols).to_pydict()
            for row in range(pf.metadata.num_rows):
                spk = table[self.columns["speaker"]][row]
                if labels is not None and isinstance(spk, int):
                    spk = labels[spk]
                dur = table[self.columns["duration"]][row] if "duration" in self.columns else None
                yield Candidate(
                    native_id=f"{stem}-r{row:06d}",
                    speaker=str(spk),
                    text=table[self.columns["text"]][row] or "",
                    duration_hint=float(dur) if dur is not None else None,
                    locator=(fi, row),
                )

    def materialize(self, cands: Sequence[Candidate]) -> dict[str, Path]:
        import pyarrow.parquet as pq

        out: dict[str, Path] = {}
        by_file: dict[int, list[Candidate]] = defaultdict(list)
        for c in cands:
            existing = sorted(self.extract_dir.glob(f"{c.native_id}.*"))
            if existing:
                out[c.native_id] = existing[0]
            else:
                by_file[c.locator[0]].append(c)
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        for fi, group in by_file.items():
            pf = pq.ParquetFile(self.files[fi])
            starts = np.cumsum(
                [0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
            )
            by_rg: dict[int, list[Candidate]] = defaultdict(list)
            for c in group:
                by_rg[int(np.searchsorted(starts, c.locator[1], side="right") - 1)].append(c)
            for rg, members in by_rg.items():
                col = pf.read_row_group(rg, columns=[self.columns["audio"]]).column(0).to_pylist()
                for c in members:
                    cell = col[c.locator[1] - starts[rg]]
                    data = cell.get("bytes") if isinstance(cell, dict) else cell
                    if not data:
                        continue
                    target = self.extract_dir / f"{c.native_id}{_sniff_extension(data)}"
                    tmp = target.with_suffix(target.suffix + ".part")
                    tmp.write_bytes(data)
                    tmp.rename(target)
                    out[c.native_id] = target
        return out


def make_reader(corpus_cfg: dict[str, Any], lang_cfg: dict[str, Any]) -> CorpusReader:
    root = Path(lang_cfg["root"])
    if root.is_absolute() or PurePosixPath(root.as_posix()).parts[:2] != RAW_ROOT.parts:
        raise IngestError(f"root {root} must be a relative path under {RAW_ROOT}/")
    fmt = corpus_cfg.get("format")
    if fmt == "commonvoice":
        return CommonVoiceReader(root)
    if fmt == "hf_parquet":
        return ParquetReader(root, lang_cfg.get("files", "**/*.parquet"), corpus_cfg["columns"])
    raise IngestError(f"unknown corpus format {fmt!r}")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def priority_order(cands: Sequence[Candidate], rng: np.random.Generator) -> list[Candidate]:
    """Speaker-balanced, seeded order: round-robin over shuffled speakers.

    The first k items hold as many distinct speakers as possible, so selecting a prefix spreads
    clips across speakers instead of letting prolific speakers dominate.
    """
    by_spk: dict[str, list[Candidate]] = defaultdict(list)
    for c in cands:
        by_spk[c.speaker].append(c)
    speakers = sorted(by_spk)
    rng.shuffle(speakers)
    queues = []
    for s in speakers:
        items = sorted(by_spk[s], key=lambda c: c.native_id)
        queues.append([items[i] for i in rng.permutation(len(items))])
    order: list[Candidate] = []
    for r in range(max((len(q) for q in queues), default=0)):
        order.extend(q[r] for q in queues if r < len(q))
    return order


@dataclass
class Accepted:
    cand: Candidate
    path: Path
    duration_s: float
    sample_rate: int


def select(
    reader: CorpusReader,
    lang: str,
    *,
    target: int,
    max_per_speaker: int,
    min_duration: float,
    max_duration: float,
    min_script_ratio: float,
    drop_text_regex: str | None,
    rng: np.random.Generator,
) -> tuple[list[Accepted], Counter[str], int]:
    """Filter, order and accept clips. Returns (accepted, rejection counts, n_candidates)."""
    import soundfile as sf

    rejected: Counter[str] = Counter()
    drop = re.compile(drop_text_regex) if drop_text_regex else None
    pool: list[Candidate] = []
    n_total = 0
    for c in reader.candidates():
        n_total += 1
        text = normalize_text(c.text)
        if not text:
            rejected["empty_text"] += 1
        elif drop is not None and drop.search(text):
            rejected["text_regex"] += 1
        elif script_ratio(text, lang) < min_script_ratio:
            rejected["script_ratio"] += 1
        elif c.duration_hint is not None and not min_duration <= c.duration_hint <= max_duration:
            rejected["duration_hint"] += 1
        else:
            pool.append(Candidate(c.native_id, c.speaker, text, c.duration_hint, c.locator))

    order = priority_order(pool, rng)
    accepted: list[Accepted] = []
    per_spk: Counter[str] = Counter()
    i = 0
    while len(accepted) < target and i < len(order):
        step = max(64, int((target - len(accepted)) * 1.25))
        chunk = [c for c in order[i : i + step] if per_spk[c.speaker] < max_per_speaker]
        i += step
        paths = reader.materialize(chunk)
        for c in chunk:
            if len(accepted) >= target:
                break
            if per_spk[c.speaker] >= max_per_speaker:
                continue
            path = paths.get(c.native_id)
            if path is None:
                rejected["missing_audio"] += 1
                continue
            try:
                info = sf.info(str(path))
            except RuntimeError:
                rejected["unreadable_audio"] += 1
                continue
            dur = info.frames / info.samplerate if info.samplerate else 0.0
            if not min_duration <= dur <= max_duration:
                rejected["duration"] += 1
                continue
            accepted.append(Accepted(c, path, dur, int(info.samplerate)))
            per_spk[c.speaker] += 1
    return accepted, rejected, n_total


def to_rows(
    accepted: Sequence[Accepted],
    *,
    corpus: str,
    lang: str,
    license: str,
    speaker_template: str,
) -> list[ManifestRow]:
    rows = []
    for k, a in enumerate(accepted):
        rows.append(
            ManifestRow(
                utt_id=f"{lang}_{corpus}_{BONAFIDE_GENERATOR}_{k:06d}",
                path=a.path.as_posix(),
                sha256=sha256_file(a.path),
                label="bonafide",
                lang=lang,
                generator_id=BONAFIDE_GENERATOR,
                generator_family=BONAFIDE_FAMILY,
                speaker_id=speaker_template.format(
                    corpus=corpus, lang=lang, speaker=a.cand.speaker
                ),
                source_corpus=corpus,
                text=a.cand.text,
                duration_s=round(a.duration_s, 4),
                sample_rate=a.sample_rate,
                condition="clean",
                split=None,
                qc_wer=None,
                license=license,
                seed=None,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Config-driven entry points
# ---------------------------------------------------------------------------


def corpus_config(config: dict[str, Any], corpus: str, lang: str) -> tuple[dict, dict]:
    corpora = config.get("corpora", {})
    if corpus not in corpora:
        raise IngestError(f"corpus {corpus!r} not in config; known: {sorted(corpora)}")
    if not CORPUS_KEY_RE.match(corpus):
        raise IngestError(f"corpus key {corpus!r} must match {CORPUS_KEY_RE.pattern}")
    ccfg = corpora[corpus]
    if lang not in ccfg.get("langs", {}):
        raise IngestError(f"corpus {corpus!r} has no config for lang {lang!r}")
    return ccfg, ccfg["langs"][lang]


def part_path(config: dict[str, Any], corpus: str, lang: str) -> Path:
    return Path(config["parts_dir"]) / f"{corpus}_{lang}.jsonl"


def ingest_corpus(ctx: Any, corpus: str, lang: str) -> Path:
    """Select bonafide clips from one corpus and language; write a part manifest."""
    config = ctx.config
    ccfg, lcfg = corpus_config(config, corpus, lang)
    out = part_path(config, corpus, lang)
    if out.exists():
        raise IngestError(f"{out} already exists; manifests are immutable (bump parts_dir)")

    reader = make_reader(ccfg, lcfg)
    accepted, rejected, n_total = select(
        reader,
        lang,
        target=int(lcfg["target"]),
        max_per_speaker=int(lcfg["max_per_speaker"]),
        min_duration=float(config["duration_s"]["min"]),
        max_duration=float(config["duration_s"]["max"]),
        min_script_ratio=float(ccfg.get("min_script_ratio", config["min_script_ratio"])),
        drop_text_regex=ccfg.get("drop_text_regex"),
        rng=ctx.rng,
    )
    if not accepted:
        raise IngestError(f"{corpus}/{lang}: no clips accepted; rejections {dict(rejected)}")
    rows = to_rows(
        accepted,
        corpus=corpus,
        lang=lang,
        license=ccfg["license"],
        speaker_template=ccfg.get("speaker_template", "{corpus}_{speaker}"),
    )
    digest = write_manifest(out, rows)
    ctx.log(
        "ingest",
        corpus=corpus,
        lang=lang,
        version=ccfg.get("version"),
        n_candidates=n_total,
        n_accepted=len(rows),
        target=int(lcfg["target"]),
        n_speakers=len({r.speaker_id for r in rows}),
        hours=round(sum(r.duration_s for r in rows) / 3600, 3),
        rejected=dict(sorted(rejected.items())),
        manifest=str(out),
        manifest_sha256=digest,
    )
    return out


def phase1_checks(
    rows: Sequence[ManifestRow], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    """The Phase 1 Definition-of-Done checks, per language."""
    checks: list[dict[str, Any]] = []
    for lang in LANGS:
        sub = [r for r in rows if r.lang == lang]
        per_spk = Counter(r.speaker_id for r in sub)
        share = max(per_spk.values()) / len(sub) if sub else 1.0
        at_least = [
            ("utterances", len(sub), thresholds["min_utts_per_lang"]),
            ("corpora", len({r.source_corpus for r in sub}), thresholds["min_corpora_per_lang"]),
            ("speakers", len(per_spk), thresholds["min_speakers_per_lang"]),
        ]
        for name, value, limit in at_least:
            checks.append(
                {
                    "lang": lang,
                    "check": name,
                    "value": value,
                    "threshold": limit,
                    "passed": value >= limit,
                }
            )
        limit = thresholds["max_speaker_share"]
        checks.append(
            {
                "lang": lang,
                "check": "max_speaker_share",
                "value": round(share, 4),
                "threshold": limit,
                "passed": share <= limit,
            }
        )
    return checks


CORPUS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3, fixed by corpus order


def speaker_histograms(
    rows: Sequence[ManifestRow], corpora: Sequence[str], out_dir: Path, max_share: float
) -> list[Path]:
    """Clips per speaker, ranked, coloured by corpus; one PNG plus a CSV table per language."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {c: CORPUS_COLORS[i % len(CORPUS_COLORS)] for i, c in enumerate(corpora)}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for lang in LANGS:
        sub = [r for r in rows if r.lang == lang]
        if not sub:
            continue
        counts = Counter((r.speaker_id, r.source_corpus) for r in sub)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        x = np.arange(len(ranked))
        ax.bar(
            x,
            [n for _, n in ranked],
            width=0.8,  # gap keeps neighbouring speakers distinct
            color=[colors[c] for (_, c), _ in ranked],
            linewidth=0,
        )
        ax.axhline(max_share * len(sub), color="#5f5e5a", linewidth=1, linestyle="--")
        ax.annotate(
            f"{max_share:.0%} cap",
            (len(ranked) - 1, max_share * len(sub)),
            ha="right",
            va="bottom",
            fontsize=8,
            color="#5f5e5a",
        )
        present = [c for c in corpora if any(rc == c for (_, rc), _ in ranked)]
        handles = [Patch(color=colors[c], label=c) for c in present]
        ax.legend(handles=handles, frameon=False, fontsize=8)
        ax.set_xlabel(f"speaker rank ({len(ranked)} speakers)", fontsize=9, color="#3d3d3a")
        ax.set_ylabel("clips", fontsize=9, color="#3d3d3a")
        ax.set_title(f"Bonafide clips per speaker, {lang} (n={len(sub)})", fontsize=10, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors="#5f5e5a")
        fig.tight_layout()
        png = out_dir / f"phase1_speakers_{lang}.png"
        fig.savefig(png)
        plt.close(fig)
        with open(out_dir / f"phase1_speakers_{lang}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "speaker_id", "source_corpus", "clips", "share"])
            for rank, ((spk, c), n) in enumerate(ranked, start=1):
                w.writerow([rank, spk, c, n, round(n / len(sub), 5)])
        written.append(png)
    return written


def merge_parts(ctx: Any) -> Path:
    """Merge every part manifest named in the config into the Phase 1 bonafide manifest."""
    from indispoof.data.manifest import merge_manifests

    config = ctx.config
    out = Path(config["output"])
    if out.exists():
        raise IngestError(f"{out} already exists; manifests are immutable")
    parts = []
    for corpus, ccfg in config["corpora"].items():
        for lang in ccfg.get("langs", {}):
            p = part_path(config, corpus, lang)
            if p.exists():
                parts.append(p)
            else:
                print(f"note: no part manifest for {corpus}/{lang} ({p})")
    if not parts:
        raise IngestError("no part manifests to merge")
    rows = merge_manifests(*(read_manifest(p) for p in parts))
    digest = write_manifest(out, rows)

    thresholds = config["checks"]
    checks = phase1_checks(rows, thresholds)
    figures = speaker_histograms(
        rows, list(config["corpora"]), Path(config["figures_dir"]), thresholds["max_speaker_share"]
    )
    for c in checks:
        ctx.log("phase1_check", **c)
    ctx.log(
        "merge",
        parts=[str(p) for p in parts],
        n_rows=len(rows),
        manifest=str(out),
        manifest_sha256=digest,
        figures=[str(f) for f in figures],
        all_passed=all(c["passed"] for c in checks),
    )
    return out


def inspect_parquet(path: str | Path, n: int = 3) -> str:
    """Schema and the first rows (audio column summarised) of one parquet file."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    buf = io.StringIO()
    buf.write(
        f"{path}: {pf.metadata.num_rows} rows, {pf.num_row_groups} row groups\n{pf.schema_arrow}\n"
    )
    first = pf.read_row_group(0).slice(0, n).to_pylist()
    for r in first:
        shown = {}
        for k, v in r.items():
            if isinstance(v, dict) and "bytes" in v:
                shown[k] = f"<audio {len(v['bytes'] or b'')} bytes, path={v.get('path')!r}>"
            elif isinstance(v, bytes):
                shown[k] = f"<{len(v)} bytes>"
            else:
                shown[k] = v
        buf.write(json.dumps(shown, ensure_ascii=False, default=str) + "\n")
    return buf.getvalue()
