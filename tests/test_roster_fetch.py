from pathlib import Path

import pytest
import yaml

from indispoof.data.fetch import GB, FetchError, RemoteFile, select_files
from indispoof.data.roster import (
    DEFAULT_REGISTER,
    DEFAULT_ROSTER,
    RosterError,
    load_roster,
    render_register,
)

REPO = Path(__file__).resolve().parents[1]


def test_roster_loads_and_register_is_current():
    roster = load_roster(REPO / DEFAULT_ROSTER)
    assert list(roster.generators) == [f"A0{i}" for i in range(1, 9)]
    rendered = render_register(roster)
    on_disk = (REPO / DEFAULT_REGISTER).read_text(encoding="utf-8")
    assert rendered == on_disk, "docs/overlap_register.md is stale: run `make overlap-register`"


def test_roster_meets_phase3_family_coverage():
    # Phase 3 DoD: at least six generators and three families per language.
    for lang, fams in load_roster(REPO / DEFAULT_ROSTER).coverage().items():
        assert sum(len(v) for v in fams.values()) >= 6, lang
        assert len(fams) >= 3, lang


def test_overlaps():
    roster = load_roster(REPO / DEFAULT_ROSTER)
    assert roster.overlapping_generators("commonvoice") == ["A04"]
    assert "A03" not in roster.overlapping_generators("indictts")
    with pytest.raises(RosterError):
        roster.overlapping_generators("librispeech")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["generators"]["A01"].update(family="diffusion"),
        lambda r: r["generators"]["A01"].update(langs=["en"]),
        lambda r: r["generators"]["A01"].update(overlaps=["librispeech"]),
        lambda r: r["generators"]["A01"].update(verified="maybe"),
        lambda r: r["generators"].update(A09=r["generators"]["A01"]),
        lambda r: r["corpora"]["indictts"].update(license=""),
    ],
)
def test_roster_validation(tmp_path, mutate):
    raw = yaml.safe_load((REPO / DEFAULT_ROSTER).read_text(encoding="utf-8"))
    mutate(raw)
    p = tmp_path / "roster.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(RosterError):
        load_roster(p)


FILES = [RemoteFile(f"tamil/train-{i:03d}.parquet", GB) for i in range(10)] + [
    RemoteFile("hindi/train-000.parquet", GB),
    RemoteFile("README.md", 100),
]


def test_select_files_spread_and_seeded():
    a = select_files(FILES, "tamil/*.parquet", 3, seed=1)
    assert len(a) == 3 and all(f.path.startswith("tamil/") for f in a)
    idx = [int(f.path[-11:-8]) for f in a]
    assert idx == sorted(idx) and idx[-1] - idx[0] >= 5  # spread over the list, not the first 3
    assert a == select_files(FILES, "tamil/*.parquet", 3, seed=1)
    assert len(select_files(FILES, "tamil/*.parquet", 50, seed=1)) == 10


def test_select_files_no_match_lists_entries():
    with pytest.raises(FetchError, match="tamil"):
        select_files(FILES, "Tamil/*.parquet", 3, seed=0)
