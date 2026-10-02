import csv
import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import yaml

from indispoof.cli import main
from indispoof.data import ingest
from indispoof.data.manifest import read_manifest, sha256_file, verify_manifest_hash
from indispoof.data.storage import pack, verify
from indispoof.run import read_results

TA = "வணக்கம் இது ஒரு சோதனை வாக்கியம்"
HI = "नमस्ते यह एक परीक्षण वाक्य है"
SR = 16000


def wav_bytes(seconds, sr=SR, seed=0):
    buf = io.BytesIO()
    sig = np.random.default_rng(seed).normal(0, 0.1, int(seconds * sr)).astype(np.float32)
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def make_commonvoice(root, lang="ta", n_speakers=30, clips_per_speaker=(1, 12), text=TA):
    """A fake extracted Common Voice language dir, plus a few deliberately bad rows."""
    clips = root / "clips"
    clips.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, durations = [], []
    k = 0
    for s in range(n_speakers):
        for _ in range(int(rng.integers(*clips_per_speaker))):
            name = f"common_voice_{lang}_{k:05d}.mp3"
            (clips / name).write_bytes(wav_bytes(1.5, seed=k))  # wav bytes; soundfile sniffs
            rows.append({"client_id": f"spk{s:03d}", "path": name, "sentence": text})
            durations.append({"clip": name, "duration[ms]": 1500})
            k += 1
    bad = [
        ("english", "this is english only", 1.5, True),
        ("empty", "", 1.5, True),
        ("short", text, 0.4, True),
        ("missing", text, 1.5, False),
    ]
    for tag, sentence, secs, write in bad:
        name = f"bad_{tag}.mp3"
        if write:
            (clips / name).write_bytes(wav_bytes(secs))
        rows.append({"client_id": f"bad_{tag}", "path": name, "sentence": sentence})
        durations.append({"clip": name, "duration[ms]": int(secs * 1000)})
    for name, fields, data in [
        ("validated.tsv", ["client_id", "path", "sentence"], rows),
        ("clip_durations.tsv", ["clip", "duration[ms]"], durations),
    ]:
        with open(root / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            w.writerows(data)


def make_parquet(path, n=40, text=TA, start_seed=0):
    """HF-style shard: audio struct, text, gender as a ClassLabel int with HF metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "audio": [
                {"bytes": wav_bytes(1.2, seed=start_seed + i), "path": None} for i in range(n)
            ],
            "text": [text] * n,
            "gender": [i % 2 for i in range(n)],
        }
    )
    features = {
        "audio": {"_type": "Audio"},
        "text": {"_type": "Value", "dtype": "string"},
        "gender": {"_type": "ClassLabel", "names": ["female", "male"]},
    }
    meta = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
    pq.write_table(table.replace_schema_metadata(meta), path, row_group_size=16)


def config_for(tmp_path, **corpora):
    cfg = {
        "name": "ingest_test",
        "seed": 7,
        "results_dir": str(tmp_path / "results"),
        "parts_dir": "manifests/parts",
        "output": "manifests/bonafide_raw.jsonl",
        "figures_dir": "paper/figures",
        "duration_s": {"min": 1.0, "max": 20.0},
        "min_script_ratio": 0.5,
        "checks": {
            "min_utts_per_lang": 50,
            "min_corpora_per_lang": 2,
            "min_speakers_per_lang": 20,
            "max_speaker_share": 0.05,
        },
        "corpora": corpora,
    }
    p = tmp_path / "ingest.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


CV = {
    "format": "commonvoice",
    "license": "CC0-1.0",
    "langs": {
        "ta": {"root": "data/raw/cv/ta", "target": 60, "max_per_speaker": 3},
        "hi": {"root": "data/raw/cv/hi", "target": 60, "max_per_speaker": 3},
    },
}
TTS = {
    "format": "hf_parquet",
    "license": "CC-BY-4.0",
    "columns": {"audio": "audio", "text": "text", "speaker": "gender"},
    "speaker_template": "{corpus}_{lang}_{speaker}",
    "langs": {
        "ta": {
            "root": "data/raw/tts/ta",
            "files": "**/*.parquet",
            "target": 6,
            "max_per_speaker": 3,
        },
        "hi": {
            "root": "data/raw/tts/hi",
            "files": "**/*.parquet",
            "target": 6,
            "max_per_speaker": 3,
        },
    },
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_commonvoice(tmp_path / "data/raw/cv/ta", "ta", text=TA)
    make_commonvoice(tmp_path / "data/raw/cv/hi", "hi", text=HI)
    # Same file name in two folders: ids must not collide.
    make_parquet(tmp_path / "data/raw/tts/ta/train/shard.parquet", text=TA)
    make_parquet(tmp_path / "data/raw/tts/ta/valid/shard.parquet", text=TA, start_seed=100)
    make_parquet(tmp_path / "data/raw/tts/hi/train/shard.parquet", text=HI)
    return tmp_path


def run_ingest(cfg, corpus, lang):
    assert main(["ingest", "--config", str(cfg), "--corpus", corpus, "--lang", lang]) == 0


# ---------------------------------------------------------------- text and sampling


def test_script_ratio_and_normalize():
    assert ingest.script_ratio(TA, "ta") == 1.0
    assert ingest.script_ratio(HI, "hi") == 1.0
    assert ingest.script_ratio(TA, "hi") == 0.0
    assert ingest.script_ratio("hello வணக்கம்", "ta") == pytest.approx(7 / 12)
    assert ingest.script_ratio("123 !!", "ta") == 0.0
    assert ingest.normalize_text("  a \t b\n") == "a b"


def test_priority_order_round_robin_and_deterministic():
    cands = [
        ingest.Candidate(f"{s}-{i}", s, "t", None, None)
        for s, n in [("a", 10), ("b", 1), ("c", 3)]
        for i in range(n)
    ]
    order = ingest.priority_order(cands, np.random.default_rng(0))
    assert len(order) == 14
    assert {c.speaker for c in order[:3]} == {"a", "b", "c"}  # every speaker before any repeat
    assert order[-7:] == [c for c in order if c.speaker == "a"][-7:]  # prolific tail last
    again = ingest.priority_order(cands, np.random.default_rng(0))
    assert [c.native_id for c in order] == [c.native_id for c in again]


# ---------------------------------------------------------------- end to end


def test_ingest_commonvoice(repo):
    cfg = config_for(repo, commonvoice=CV)
    run_ingest(cfg, "commonvoice", "ta")
    part = repo / "manifests/parts/commonvoice_ta.jsonl"
    rows = read_manifest(part)
    verify_manifest_hash(part)

    assert len(rows) == 60
    assert all(r.utt_id.startswith("ta_commonvoice_A00_") for r in rows)
    assert all(r.label == "bonafide" and r.license == "CC0-1.0" for r in rows)
    assert all(r.path.startswith("data/raw/cv/ta/clips/") for r in rows)
    assert all(sha256_file(r.path) == r.sha256 for r in rows)
    assert max(np.unique([r.speaker_id for r in rows], return_counts=True)[1]) <= 3
    assert not any("bad_" in r.speaker_id for r in rows)
    assert verify(rows) == []

    (results,) = (repo / "results").glob("*.jsonl")
    _, records = read_results(results)
    (log,) = [r for r in records if r["type"] == "ingest"]
    # empty text and English text are filtered before sampling, the short clip by its
    # duration hint. The missing clip may or may not be reached by the sampler.
    assert log["rejected"]["empty_text"] == 1
    assert log["rejected"]["script_ratio"] == 1
    assert log["rejected"]["duration_hint"] == 1
    assert log["n_accepted"] == 60


def test_ingest_is_deterministic(repo, tmp_path_factory):
    cfg = config_for(repo, commonvoice=CV)
    run_ingest(cfg, "commonvoice", "ta")
    first = (repo / "manifests/parts/commonvoice_ta.jsonl").read_bytes()
    (repo / "manifests/parts/commonvoice_ta.jsonl").unlink()
    run_ingest(cfg, "commonvoice", "ta")
    assert (repo / "manifests/parts/commonvoice_ta.jsonl").read_bytes() == first


def test_ingest_refuses_to_overwrite(repo):
    cfg = config_for(repo, commonvoice=CV)
    run_ingest(cfg, "commonvoice", "ta")
    with pytest.raises(ingest.IngestError, match="immutable"):
        run_ingest(cfg, "commonvoice", "ta")


def test_ingest_parquet(repo):
    cfg = config_for(repo, indictts=TTS)
    run_ingest(cfg, "indictts", "ta")
    rows = read_manifest(repo / "manifests/parts/indictts_ta.jsonl")
    assert len(rows) == 6
    assert {r.speaker_id for r in rows} == {"indictts_ta_female", "indictts_ta_male"}
    assert all(
        r.path.startswith("data/raw/tts/ta/_extracted/") and r.path.endswith(".wav") for r in rows
    )
    assert all(r.sample_rate == SR and r.duration_s == pytest.approx(1.2) for r in rows)
    assert verify(rows) == []
    names = sorted(p.name for p in (repo / "data/raw/tts/ta/_extracted").iterdir())
    assert len(names) == len(set(names))


def test_inspect(repo, capsys):
    cfg = config_for(repo, indictts=TTS)
    assert (
        main(["ingest", "--config", str(cfg), "--corpus", "indictts", "--lang", "ta", "--inspect"])
        == 0
    )
    out = capsys.readouterr().out
    assert "gender" in out and "<audio" in out


def test_root_must_be_under_data_raw(repo):
    bad = {**CV, "langs": {"ta": {"root": "elsewhere/ta", "target": 5, "max_per_speaker": 1}}}
    cfg = config_for(repo, commonvoice=bad)
    with pytest.raises(ingest.IngestError, match="data/raw"):
        run_ingest(cfg, "commonvoice", "ta")


def test_merge_checks_and_figures(repo):
    cfg = config_for(repo, commonvoice=CV, indictts=TTS)
    for corpus in ("commonvoice", "indictts"):
        for lang in ("ta", "hi"):
            run_ingest(cfg, corpus, lang)
    assert main(["ingest", "--config", str(cfg), "--merge"]) == 0

    rows = read_manifest(repo / "manifests/bonafide_raw.jsonl")
    verify_manifest_hash(repo / "manifests/bonafide_raw.jsonl")
    assert {r.lang for r in rows} == {"ta", "hi"} and len(rows) == 132
    for lang in ("ta", "hi"):
        assert (repo / f"paper/figures/phase1_speakers_{lang}.png").stat().st_size > 0
        assert (repo / f"paper/figures/phase1_speakers_{lang}.csv").exists()

    logs = [read_results(p)[1] for p in (repo / "results").glob("*.jsonl")]
    (records,) = [r for r in logs if any(x["type"] == "merge" for x in r)]
    checks = {(c["lang"], c["check"]): c for c in records if c["type"] == "phase1_check"}
    assert checks[("ta", "utterances")] == checks[("ta", "utterances")] | {
        "value": 66,
        "passed": True,
    }
    assert checks[("ta", "corpora")]["passed"]
    assert checks[("ta", "speakers")]["passed"]  # 20 CV speakers + 2 IndicTTS
    # 3 of 66 clips is 4.5%, under the 5% cap
    assert checks[("ta", "max_speaker_share")]["passed"]


def test_phase1_checks_fail_when_short(repo):
    cfg = config_for(repo, indictts=TTS)
    run_ingest(cfg, "indictts", "ta")
    rows = read_manifest(repo / "manifests/parts/indictts_ta.jsonl")
    checks = ingest.phase1_checks(rows, yaml.safe_load(cfg.read_text())["checks"])
    by = {(c["lang"], c["check"]): c["passed"] for c in checks}
    assert not by[("ta", "utterances")] and not by[("ta", "corpora")]
    assert not by[("ta", "max_speaker_share")]  # 2 speakers -> 50% each
    assert not by[("hi", "utterances")]  # no Hindi at all


def test_pack(repo, tmp_path_factory):
    cfg = config_for(repo, indictts=TTS)
    run_ingest(cfg, "indictts", "ta")
    rows = read_manifest(repo / "manifests/parts/indictts_ta.jsonl")
    dest = tmp_path_factory.mktemp("pack")
    assert pack(rows, dest) == 6
    assert pack(rows, dest) == 0  # idempotent
    assert verify(rows, root=dest) == []
    (dest / rows[0].path).write_bytes(b"tampered")
    assert verify(rows, root=dest) == [f"{rows[0].utt_id}: sha256 mismatch for {rows[0].path}"]
