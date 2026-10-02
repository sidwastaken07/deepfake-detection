import json
import random

import numpy as np
import pytest
import yaml

from indispoof.cli import STUBS, main
from indispoof.run import ConfigError, ResultsError, RunContext, load_config, read_results


@pytest.fixture
def config(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump({"name": "unit", "seed": 3, "results_dir": str(tmp_path / "results")}),
        encoding="utf-8",
    )
    return p


def test_load_config_overrides_and_seed(config):
    cfg = load_config(config, ["model.lr=0.001", "model.name=aasist"], seed=11)
    assert cfg["seed"] == 11
    assert cfg["model"] == {"lr": 0.001, "name": "aasist"}


@pytest.mark.parametrize(
    "body", [{"seed": 1}, {"name": "Bad Name", "seed": 1}, {"name": "x"}, {"name": "x", "seed": -1}]
)
def test_load_config_rejects(tmp_path, body):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_run_context_writes_valid_results(config):
    with RunContext(config, ["extra.k=1"]) as ctx:
        ctx.log("metrics", eer=0.25, lang="ta")
    header, records = read_results(ctx.results_path)
    assert header["run_id"] == ctx.run_id
    assert header["config"]["extra"] == {"k": 1}
    assert header["overrides"] == ["extra.k=1"]
    assert set(header["git"]) == {"commit", "dirty"}
    assert len(header["config_sha256"]) == 64
    assert [r["type"] for r in records] == ["metrics", "run_end"]
    assert records[-1]["status"] == "ok"


def test_run_context_is_seeded(config):
    draws = []
    for _ in range(2):
        with RunContext(config) as ctx:
            draws.append((ctx.rng.normal(size=3), np.random.rand(), random.random()))  # noqa: NPY002
    (a0, a1, a2), (b0, b1, b2) = draws
    assert np.array_equal(a0, b0) and a1 == b1 and a2 == b2


def test_run_context_records_failure(config):
    with pytest.raises(RuntimeError), RunContext(config) as ctx:
        raise RuntimeError("boom")
    _, records = read_results(ctx.results_path)
    assert records[-1] == records[-1] | {"type": "run_end", "status": "failed"}
    assert "boom" in records[-1]["error"]


def test_reserved_record_types(config):
    with RunContext(config) as ctx, pytest.raises(ResultsError):
        ctx.log("run_header")


def test_read_results_rejects_corruption(config):
    with RunContext(config) as ctx:
        ctx.log("metrics", eer=0.1)
    lines = ctx.results_path.read_text(encoding="utf-8").splitlines()
    bad = ctx.results_path.with_name("bad.jsonl")

    bad.write_text("\n".join(lines[1:]), encoding="utf-8")
    with pytest.raises(ResultsError, match="run_header"):
        read_results(bad)

    rec = json.loads(lines[1]) | {"run_id": "other"}
    bad.write_text("\n".join([lines[0], json.dumps(rec)]), encoding="utf-8")
    with pytest.raises(ResultsError, match="run_id"):
        read_results(bad)


@pytest.mark.parametrize("command", sorted(STUBS))
def test_stub_commands_fail_loudly(command, capsys):
    assert main([command]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_smoke_end_to_end(tmp_path):
    out = tmp_path / "results"
    rc = main(
        [
            "smoke",
            "--config",
            "configs/phase0_smoke.yaml",
            "--set",
            f"results_dir={out}",
            "--set",
            "smoke.n_per_class_per_lang=100",
        ]
    )
    assert rc == 0
    (path,) = out.glob("phase0_smoke_*.jsonl")
    header, records = read_results(path)
    assert header["seed"] == 0
    metrics = [r for r in records if r["type"] == "metrics"]
    assert [m["lang"] for m in metrics] == ["all", "ta", "hi"]
    overall = metrics[0]
    assert overall["n_bonafide"] == overall["n_spoof"] == 200
    assert 0.35 < overall["eer"] < 0.65  # random scores
    assert records[-1]["status"] == "ok"
