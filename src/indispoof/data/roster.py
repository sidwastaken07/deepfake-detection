"""Generator roster and training-overlap register (``configs/roster.yaml``).

The YAML is the source of truth. ``docs/overlap_register.md`` is rendered from it, and
Phase 6 uses :func:`Roster.overlapping_generators` to keep each bonafide corpus out of the
evaluation split of every generator that was trained on it.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from indispoof.data.manifest import LANGS, SPOOF_FAMILIES, SPOOF_GENERATORS

DEFAULT_ROSTER = Path("configs/roster.yaml")
DEFAULT_REGISTER = Path("docs/overlap_register.md")


class RosterError(ValueError):
    pass


@dataclass(frozen=True)
class Corpus:
    key: str
    name: str
    license: str
    role: str
    note: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Generator:
    id: str
    system: str
    checkpoint: str
    family: str
    cloning: str
    langs: tuple[str, ...]
    license: str
    training_data: tuple[str, ...]
    overlaps: tuple[str, ...]
    verified: bool
    note: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Roster:
    corpora: dict[str, Corpus]
    generators: dict[str, Generator]

    def overlapping_generators(self, corpus: str) -> list[str]:
        """Generators trained on ``corpus``; it must be excluded from their eval split."""
        if corpus not in self.corpora:
            raise RosterError(f"unknown corpus {corpus!r}")
        return [g.id for g in self.generators.values() if corpus in g.overlaps]

    def coverage(self) -> dict[str, dict[str, list[str]]]:
        """lang -> family -> generator ids."""
        out: dict[str, dict[str, list[str]]] = {lang: defaultdict(list) for lang in LANGS}
        for g in self.generators.values():
            for lang in g.langs:
                out[lang][g.family].append(g.id)
        return {lang: dict(fams) for lang, fams in out.items()}


def _strs(v: Any, where: str) -> tuple[str, ...]:
    if v is None:
        return ()
    if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
        raise RosterError(f"{where}: expected a list of non-empty strings")
    return tuple(v)


def _str(d: dict[str, Any], key: str, where: str, required: bool = True) -> str:
    v = d.get(key, "")
    if not isinstance(v, str) or (required and not v.strip()):
        raise RosterError(f"{where}.{key}: expected a non-empty string")
    return " ".join(v.split())


def load_roster(path: str | os.PathLike[str] = DEFAULT_ROSTER) -> Roster:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or set(raw) != {"corpora", "generators"}:
        raise RosterError(f"{path}: top level must have exactly 'corpora' and 'generators'")

    corpora = {}
    for key, c in raw["corpora"].items():
        w = f"corpora.{key}"
        corpora[key] = Corpus(
            key=key,
            name=_str(c, "name", w),
            license=_str(c, "license", w),
            role=_str(c, "role", w),
            note=_str(c, "note", w, required=False),
            sources=_strs(c.get("sources"), f"{w}.sources"),
        )

    generators = {}
    for gid, g in raw["generators"].items():
        w = f"generators.{gid}"
        if gid not in SPOOF_GENERATORS:
            raise RosterError(f"{w}: id must be one of {SPOOF_GENERATORS}")
        if g.get("family") not in SPOOF_FAMILIES:
            raise RosterError(f"{w}.family: must be one of {SPOOF_FAMILIES}")
        langs = _strs(g.get("langs"), f"{w}.langs")
        if not langs or set(langs) - set(LANGS):
            raise RosterError(f"{w}.langs: must be a non-empty subset of {LANGS}")
        overlaps = _strs(g.get("overlaps"), f"{w}.overlaps")
        unknown = set(overlaps) - set(corpora)
        if unknown:
            raise RosterError(f"{w}.overlaps: unknown corpora {sorted(unknown)}")
        if not isinstance(g.get("verified"), bool):
            raise RosterError(f"{w}.verified: must be true or false")
        generators[gid] = Generator(
            id=gid,
            system=_str(g, "system", w),
            checkpoint=_str(g, "checkpoint", w),
            family=g["family"],
            cloning=_str(g, "cloning", w),
            langs=langs,
            license=_str(g, "license", w),
            training_data=_strs(g.get("training_data"), f"{w}.training_data"),
            overlaps=overlaps,
            verified=g["verified"],
            note=_str(g, "note", w, required=False),
            sources=_strs(g.get("sources"), f"{w}.sources"),
        )
    return Roster(corpora=corpora, generators=dict(sorted(generators.items())))


def render_register(roster: Roster, roster_path: str = str(DEFAULT_ROSTER)) -> str:
    """Markdown overlap register. Deterministic, so it can be checked for drift."""
    corpora = list(roster.corpora)
    lines = [
        "# Training-overlap register",
        "",
        f"<!-- Generated from {roster_path} by `make overlap-register`. Do not edit by hand. -->",
        "",
        "For each generator in the Phase 3 roster: what it was trained on, according to its model",
        "card or paper. **✗** marks a bonafide corpus that overlaps the generator's training data.",
        "That corpus is excluded from the evaluation split for that generator (Phase 6 enforces",
        "this).",
        "",
        "| ID | System | Family | Langs | License | Training data | "
        + " | ".join(corpora)
        + " | Verified |",
        "|---|---|---|---|---|---|" + "---|" * len(corpora) + "---|",
    ]
    for g in roster.generators.values():
        marks = ["✗" if c in g.overlaps else "" for c in corpora]
        lines.append(
            f"| {g.id} | {g.system} | {g.family} | {', '.join(g.langs)} | {g.license} | "
            f"{'; '.join(g.training_data)} | "
            + " | ".join(marks)
            + f" | {'yes' if g.verified else 'no'} |"
        )

    lines += ["", "## Exclusions by bonafide corpus", ""]
    for c in roster.corpora.values():
        gens = roster.overlapping_generators(c.key)
        lines.append(
            f"- **{c.key}** ({c.license}): excluded from the eval split of "
            + (", ".join(gens) if gens else "no generator")
            + "."
        )

    lines += ["", "## Family coverage", ""]
    for lang, fams in roster.coverage().items():
        n_gen = sum(len(v) for v in fams.values())
        detail = "; ".join(f"{fam}: {', '.join(ids)}" for fam, ids in sorted(fams.items()))
        lines.append(f"- **{lang}**: {n_gen} generators, {len(fams)} families ({detail}).")

    notes = [(c.key, c.note) for c in roster.corpora.values() if c.note]
    notes += [(g.id, g.note) for g in roster.generators.values() if g.note]
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- **{k}**: {n}" for k, n in notes]

    lines += ["", "## Sources", ""]
    for k, srcs in [(c.key, c.sources) for c in roster.corpora.values()] + [
        (g.id, g.sources) for g in roster.generators.values()
    ]:
        lines.append(f"- **{k}**: " + ", ".join(f"<{s}>" for s in srcs))

    unverified = [g.id for g in roster.generators.values() if not g.verified]
    if unverified:
        lines += [
            "",
            f"**Unverified:** {', '.join(unverified)}. These entries were compiled from search "
            "summaries, not read from the model cards directly. Re-check each one against its "
            "model card before Phase 3 uses that generator.",
        ]
    return "\n".join(lines) + "\n"
