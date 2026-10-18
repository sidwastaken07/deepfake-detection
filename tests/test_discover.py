import io
import tarfile
import zipfile

import numpy as np
import pyarrow as pa
import pytest

from indispoof.data import ingest
from indispoof.data.discover import (
    DiscoveryError,
    extract_cv_language,
    find_cv_language_dir,
    guess_columns,
    hf_language_patterns,
    resolve_config,
    size_from_details,
)

AUDIO = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def test_guess_columns_indicvoices_like():
    schema = pa.schema(
        [
            ("audio_filepath", AUDIO),
            ("verbatim", pa.string()),
            ("normalized", pa.string()),
            ("speaker_id", pa.string()),
            ("duration", pa.float64()),
            ("gender", pa.string()),
        ]
    )
    assert guess_columns(schema) == {
        "audio": "audio_filepath",
        "text": "normalized",  # preferred over verbatim
        "speaker": "speaker_id",
        "duration": "duration",
    }


def test_guess_columns_indictts_like_uses_fallback():
    schema = pa.schema([("audio", AUDIO), ("text", pa.string()), ("gender", pa.int64())])
    with pytest.raises(DiscoveryError, match="speaker"):
        guess_columns(schema)
    assert guess_columns(schema, speaker_fallback="gender") == {
        "audio": "audio",
        "text": "text",
        "speaker": "gender",
    }


def test_guess_columns_needs_exactly_one_audio_column():
    with pytest.raises(DiscoveryError, match="audio"):
        guess_columns(pa.schema([("text", pa.string()), ("speaker_id", pa.string())]))
    with pytest.raises(DiscoveryError, match="audio"):
        guess_columns(pa.schema([("a", AUDIO), ("b", AUDIO), ("text", pa.string())]))


@pytest.mark.parametrize(
    "paths, expected",
    [
        (["tamil/train-00000.parquet", "hindi/train-00000.parquet"], "tamil"),
        (["tamil/train/0.parquet", "tamil/valid/0.parquet"], "tamil"),
        (["data/Tamil/train/0.parquet", "README.md"], "data/Tamil"),
    ],
)
def test_hf_language_patterns(paths, expected):
    remote, local = hf_language_patterns(paths, "tamil")
    assert remote == f"{expected}/*.parquet"
    assert local == f"{expected}/**/*.parquet"


def test_hf_language_patterns_reports_layout():
    with pytest.raises(DiscoveryError, match="hindi"):
        hf_language_patterns(["hindi/train-0.parquet", "README.md"], "tamil")


def _archive_bytes(kind, files):
    buf = io.BytesIO()
    if kind == "tar":
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    else:
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, data)
    return buf.getvalue()


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_extract_cv_language_only_takes_one_language(tmp_path, kind):
    files = {
        "cv-corpus-25.0-2026-03-09/ta/validated.tsv": b"client_id\tpath\tsentence\n",
        "cv-corpus-25.0-2026-03-09/ta/clips/a.mp3": b"x",
        "cv-corpus-25.0-2026-03-09/hi/validated.tsv": b"y",
    }
    archive = tmp_path / f"cv.{kind}"
    archive.write_bytes(_archive_bytes(kind, files))
    root = extract_cv_language(archive, "ta", tmp_path / "out")
    assert root == tmp_path / "out/cv-corpus-25.0-2026-03-09/ta"
    assert (root / "clips/a.mp3").exists()
    assert not (tmp_path / "out/cv-corpus-25.0-2026-03-09/hi").exists()


def test_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar"
    archive.write_bytes(
        _archive_bytes("tar", {"x/ta/validated.tsv": b"", "x/ta/../../../evil": b""})
    )
    with pytest.raises(DiscoveryError, match="escapes"):
        extract_cv_language(archive, "ta", tmp_path / "out")


def test_find_cv_language_dir_missing():
    with pytest.raises(DiscoveryError):
        find_cv_language_dir(["cv/hi/validated.tsv"], "ta")


def test_size_from_details():
    assert size_from_details({"name": "x", "size": 123, "files": [{"sizeBytes": 456}]}) == 456
    assert size_from_details({"name": "x"}) is None


def test_resolve_config_deep_merges_without_mutating():
    base = {"name": "a", "corpora": {"iv": {"license": "L", "langs": {"ta": {"target": 1}}}}}
    out = resolve_config(base, {"corpora": {"iv": {"langs": {"ta": {"files": "f"}}}}}, "b")
    assert out["name"] == "b"
    assert out["corpora"]["iv"] == {"license": "L", "langs": {"ta": {"target": 1, "files": "f"}}}
    assert "files" not in base["corpora"]["iv"]["langs"]["ta"]


class _StubReader:
    def __init__(self, cands):
        self.cands = cands

    def candidates(self):
        return iter(self.cands)

    def materialize(self, cands):
        return {}


def test_select_rejects_missing_speakers():
    text = "வணக்கம் உலகம்"
    cands = [
        ingest.Candidate(f"c{i}", spk, text, 2.0, None)
        for i, spk in enumerate(["None", " ", "nan", "s1"])
    ]
    accepted, rejected, n = ingest.select(
        _StubReader(cands),
        "ta",
        target=5,
        max_per_speaker=5,
        min_duration=1.0,
        max_duration=20.0,
        min_script_ratio=0.5,
        drop_text_regex=None,
        rng=np.random.default_rng(0),
    )
    assert n == 4 and rejected["no_speaker"] == 3
    assert rejected["missing_audio"] == 1  # s1 reaches materialisation, stub has no audio
