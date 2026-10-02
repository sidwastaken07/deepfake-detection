import json

import pytest

from indispoof.data.manifest import (
    SCHEMA,
    ManifestError,
    ManifestRow,
    hash_manifest,
    merge_manifests,
    read_manifest,
    sha256_file,
    sidecar_path,
    validate_row,
    verify_manifest_hash,
    write_manifest,
)


def bonafide(i=0, **kw):
    row = {
        "utt_id": f"ta_commonvoice_A00_{i:06d}",
        "path": f"data/processed/A00/ta_commonvoice_A00_{i:06d}.wav",
        "sha256": "a" * 64,
        "label": "bonafide",
        "lang": "ta",
        "generator_id": "A00",
        "generator_family": "bonafide",
        "speaker_id": "cv_spk_001",
        "source_corpus": "commonvoice",
        "text": "வணக்கம் உலகம்",
        "duration_s": 3.25,
        "sample_rate": 16000,
        "condition": "clean",
        "split": None,
        "qc_wer": None,
        "license": "CC0-1.0",
        "seed": None,
    }
    row.update(kw)
    return row


def spoof(i=0, **kw):
    row = bonafide(
        i,
        utt_id=f"hi_indictts_A01_{i:06d}",
        path=f"data/processed/A01/hi_indictts_A01_{i:06d}.wav",
        label="spoof",
        lang="hi",
        generator_id="A01",
        generator_family="flow_matching",
        text="नमस्ते दुनिया",
        qc_wer=0.12,
        seed=1337,
    )
    row.update(kw)
    return row


def test_schema_matches_spec_order():
    assert SCHEMA == (
        "utt_id", "path", "sha256", "label", "lang", "generator_id", "generator_family",
        "speaker_id", "source_corpus", "text", "duration_s", "sample_rate", "condition",
        "split", "qc_wer", "license", "seed",
    )  # fmt: skip


def test_valid_rows_are_typed():
    r = validate_row(bonafide(duration_s=3))
    assert isinstance(r, ManifestRow) and isinstance(r.duration_s, float)
    assert validate_row(spoof()).qc_wer == 0.12


def test_roundtrip_preserves_native_script_and_hash(tmp_path):
    p = tmp_path / "m.jsonl"
    digest = write_manifest(p, [bonafide(0), bonafide(1), spoof(0)])
    assert digest == hash_manifest(p) == sha256_file(p) == verify_manifest_hash(p)
    assert "வணக்கம்" in p.read_text(encoding="utf-8")  # not \u-escaped
    rows = read_manifest(p)
    assert [r.utt_id for r in rows] == [
        "ta_commonvoice_A00_000000",
        "ta_commonvoice_A00_000001",
        "hi_indictts_A01_000000",
    ]
    assert rows[2].to_dict() == spoof(0)


def test_write_is_deterministic(tmp_path):
    rows = [bonafide(0), spoof(0)]
    assert write_manifest(tmp_path / "a.jsonl", rows) == write_manifest(tmp_path / "b.jsonl", rows)


def test_write_refuses_overwrite_and_empty(tmp_path):
    p = tmp_path / "m.jsonl"
    write_manifest(p, [bonafide()])
    with pytest.raises(ManifestError, match="immutable"):
        write_manifest(p, [bonafide()])
    with pytest.raises(ManifestError, match="empty"):
        write_manifest(tmp_path / "e.jsonl", [])


def test_write_rejects_duplicates_and_leaves_no_file(tmp_path):
    p = tmp_path / "m.jsonl"
    with pytest.raises(ManifestError, match="duplicate"):
        write_manifest(p, [bonafide(0), bonafide(0)])
    with pytest.raises(ManifestError, match="row 1"):
        write_manifest(p, [bonafide(0), bonafide(1, lang="en")])
    assert not p.exists()


CORRUPTIONS = {
    "missing field": lambda r: r.pop("license"),
    "unknown field": lambda r: r.update(speaker_gender="f"),
    "bad label": lambda r: r.update(label="real"),
    "bad lang": lambda r: r.update(lang="en"),
    "bad generator": lambda r: r.update(generator_id="A09"),
    "bad family": lambda r: r.update(generator_family="diffusion"),
    "bad condition": lambda r: r.update(condition="opus_99k"),
    "bad split": lambda r: r.update(split="test"),
    "short sha": lambda r: r.update(sha256="abc"),
    "upper sha": lambda r: r.update(sha256="A" * 64),
    "utt_id format": lambda r: r.update(utt_id="ta-commonvoice-A00-1"),
    "utt_id lang mismatch": lambda r: r.update(utt_id="hi_commonvoice_A00_000001"),
    "utt_id gen mismatch": lambda r: r.update(utt_id="ta_commonvoice_A01_000001"),
    "absolute path": lambda r: r.update(path="/data/processed/x.wav"),
    "escaping path": lambda r: r.update(path="data/../../etc/x.wav"),
    "bonafide with spoof gen": lambda r: r.update(generator_id="A01", utt_id="ta_cv_A01_1"),
    "bonafide with spoof family": lambda r: r.update(generator_family="vits"),
    "bonafide with qc_wer": lambda r: r.update(qc_wer=0.1),
    "spoof labelled A00": lambda r: r.update(label="spoof"),
    "zero duration": lambda r: r.update(duration_s=0),
    "nan duration": lambda r: r.update(duration_s=float("nan")),
    "string duration": lambda r: r.update(duration_s="3.2"),
    "float sample rate": lambda r: r.update(sample_rate=16000.0),
    "bool sample rate": lambda r: r.update(sample_rate=True),
    "null required": lambda r: r.update(text=None),
    "empty speaker": lambda r: r.update(speaker_id=""),
    "padded text": lambda r: r.update(text=" வணக்கம்"),
    "string seed": lambda r: r.update(seed="1"),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_read_rejects_corrupted_row(tmp_path, name):
    p = tmp_path / "m.jsonl"
    write_manifest(p, [bonafide(0), bonafide(1), bonafide(2)])
    lines = p.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    CORRUPTIONS[name](row)
    lines[1] = json.dumps(row, ensure_ascii=False)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match=r"m\.jsonl:2:"):
        read_manifest(p)


def test_negative_qc_wer_rejected():
    with pytest.raises(ManifestError, match="qc_wer"):
        validate_row(spoof(qc_wer=-0.1))


@pytest.mark.parametrize(
    "line", ["{not json", "", "[1, 2]", json.dumps(bonafide(0))]
)  # last one duplicates line 1
def test_read_rejects_malformed_lines(tmp_path, line):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(bonafide(0)) + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match=r":2:"):
        read_manifest(p)


def test_tampering_breaks_hash(tmp_path):
    p = tmp_path / "m.jsonl"
    write_manifest(p, [bonafide(0)])
    p.write_text(p.read_text(encoding="utf-8").replace("3.25", "3.5"), encoding="utf-8")
    read_manifest(p)  # still schema-valid ...
    with pytest.raises(ManifestError, match="hash mismatch"):
        verify_manifest_hash(p)  # ... but no longer the recorded manifest
    sidecar_path(p).unlink()
    with pytest.raises(ManifestError, match="sidecar"):
        verify_manifest_hash(p)


def test_stage_checks():
    with pytest.raises(ManifestError, match="sample_rate"):
        validate_row(bonafide(sample_rate=22050), processed=True)
    with pytest.raises(ManifestError, match="path"):
        validate_row(bonafide(path="data/interim/A00/x.wav"), processed=True)
    validate_row(bonafide(path="data/interim/A00/x.wav", sample_rate=22050))  # fine pre-Phase 4

    with pytest.raises(ManifestError, match="qc_wer"):
        validate_row(spoof(qc_wer=None), gated=True)
    validate_row(bonafide(), gated=True)  # bonafide never carries qc_wer

    with pytest.raises(ManifestError, match="split"):
        validate_row(bonafide(), assigned=True)
    validate_row(bonafide(split="eval"), assigned=True)


def test_merge():
    a = [validate_row(bonafide(0)), validate_row(spoof(0))]
    b = [validate_row(bonafide(0)), validate_row(bonafide(1))]
    assert [r.utt_id for r in merge_manifests(a, b)] == [
        "ta_commonvoice_A00_000000",
        "hi_indictts_A01_000000",
        "ta_commonvoice_A00_000001",
    ]
    with pytest.raises(ManifestError, match="conflicting"):
        merge_manifests(a, [validate_row(bonafide(0, duration_s=9.0))])
