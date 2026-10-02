"""Phase 0 dummy end-to-end run: synthetic manifest -> random scores -> metrics -> results file.

Exercises the harness (config, seeding, manifest write/read/hash, metrics, results logging)
with no audio and no model. Scores are random, so EER is expected near 0.5. Nothing this
produces is a benchmark result.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from indispoof.data.manifest import (
    BONAFIDE_FAMILY,
    BONAFIDE_GENERATOR,
    LANGS,
    ManifestRow,
    hash_manifest,
    read_manifest,
    write_manifest,
)
from indispoof.eval.metrics import DCFParams, compute_all
from indispoof.run import RunContext

# A01..A03 with their families from the Phase 3 roster.
_SMOKE_SPOOF = (("A01", "flow_matching"), ("A02", "fastpitch_gan"), ("A03", "vits"))


def synthetic_rows(n_per_class_per_lang: int, rng: np.random.Generator) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for lang in LANGS:
        for label in ("bonafide", "spoof"):
            for i in range(n_per_class_per_lang):
                if label == "bonafide":
                    gen, fam, wer, seed = BONAFIDE_GENERATOR, BONAFIDE_FAMILY, None, None
                else:
                    gen, fam = _SMOKE_SPOOF[i % len(_SMOKE_SPOOF)]
                    wer, seed = float(rng.uniform(0, 0.3)), int(i)
                utt_id = f"{lang}_smoke_{gen}_{i:06d}"
                rows.append(
                    ManifestRow(
                        utt_id=utt_id,
                        path=f"data/processed/smoke/{utt_id}.wav",
                        sha256=hashlib.sha256(utt_id.encode()).hexdigest(),
                        label=label,
                        lang=lang,
                        generator_id=gen,
                        generator_family=fam,
                        speaker_id=f"smoke_spk{i % 20:02d}",
                        source_corpus="smoke",
                        text="synthetic smoke-test row",
                        duration_s=round(float(rng.uniform(1.0, 8.0)), 3),
                        sample_rate=16000,
                        condition="clean",
                        split="eval",
                        qc_wer=wer,
                        license="none (synthetic)",
                        seed=seed,
                    )
                )
    return rows


def run_smoke(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.config.get("smoke", {})
    n = int(cfg.get("n_per_class_per_lang", 250))
    params = DCFParams(**cfg.get("dcf", {}))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.jsonl"
        write_manifest(path, synthetic_rows(n, ctx.rng), processed=True, gated=True, assigned=True)
        manifest_hash = hash_manifest(path)
        rows = read_manifest(path, processed=True, gated=True, assigned=True)
    ctx.log("manifest", role="eval", n_rows=len(rows), sha256=manifest_hash, ephemeral=True)

    scores = ctx.rng.normal(size=len(rows))
    labels = np.array([r.label for r in rows])
    langs = np.array([r.lang for r in rows])

    overall: dict[str, Any] = {}
    for scope in ("all", *LANGS):
        mask = np.ones(len(rows), bool) if scope == "all" else langs == scope
        m = compute_all(
            scores[mask & (labels == "bonafide")], scores[mask & (labels == "spoof")], params
        )
        ctx.log("metrics", scorer="random_normal", lang=scope, condition="clean", **m)
        if scope == "all":
            overall = m
    return overall
