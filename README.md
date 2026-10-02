# IndiSpoof

A cross-lingual benchmark for audio deepfake detection in Indic languages (Tamil, Hindi).

**Claim under test:** English-trained audio deepfake detectors transfer poorly to Indic-language
synthetic speech. IndiSpoof measures the size of that gap, splits it into a language effect and an
unseen-generator effect, and estimates how much target-language data closes it.

The full build specification is in [`docs/SPEC.md`](docs/SPEC.md). Work proceeds strictly phase by
phase; accepted results and gate outcomes are logged in [`RESULTS.md`](RESULTS.md).

## Status

| Phase | State |
|---|---|
| 0 Scaffold and reproducibility harness | done (see RESULTS.md) |
| 1 Bonafide ingestion | code ready; run on Kaggle per [`docs/kaggle.md`](docs/kaggle.md) |
| 2 Text plan → 10 Release | not started |

## Setup

Python 3.10 or 3.11.

```bash
pip install -e ".[dev]"          # harness, metrics, tests
pip install -e ".[dev,data]"     # + soundfile, pyarrow, huggingface_hub, matplotlib (Phase 1)
pip install -e ".[dev,audio]"    # + torch/torchaudio (Phase 3 onwards)
make check                       # ruff lint + format check, then pytest
make smoke                       # dummy end-to-end run -> results/<run_id>.jsonl
```

## Layout

```
configs/      one YAML per run; no hard-coded paths in src
data/raw/     downloaded corpora, never modified           (git-ignored)
data/interim/ per-generator raw synthesis output           (git-ignored)
data/processed/ normalised 16 kHz mono; the only audio read (git-ignored)
manifests/    JSONL manifests + .sha256 sidecars; the single source of truth
results/      one append-only JSONL file per run
paper/        generated LaTeX tables and figures
src/indispoof/
  data/manifest.py   manifest schema: strict read / write / validate / merge / hash
  data/ingest.py     Phase 1 corpus readers, speaker-balanced sampling, checks, histograms
  data/fetch.py      seeded, size-capped Hugging Face shard download
  data/roster.py     generator roster + training-overlap register (configs/roster.yaml)
  data/storage.py    verify / pack the audio a manifest references
  eval/metrics.py    EER, AUC, minDCF, actDCF
  run.py             run context: config resolution, seeding, results file
  cli.py             `indispoof` command
docs/SPEC.md  the build specification
docs/overlap_register.md  which generator was trained on which bonafide corpus (generated)
docs/kaggle.md  how to run the data-heavy phases on Kaggle
```

## Conventions

- **Manifests.** Every audio access goes through a manifest. Rows follow the fixed schema in
  `src/indispoof/data/manifest.py` (spec section 3); unknown or missing fields are rejected.
  Manifests are never overwritten. Each one is written with a `<name>.jsonl.sha256` sidecar.
- **Runs.** Every run takes one YAML config. Its results file starts with a `run_header` line holding
  the resolved config, git commit, environment, and any known non-determinism.
- **Scores.** Higher score means more bonafide. DCF uses the ASVspoof 5 countermeasure costs
  (π_spoof = 0.05, C_miss = 1, C_fa = 10). actDCF assumes the scores are log-likelihood ratios.

## Protocol hashes

None yet. Frozen protocol manifest hashes will be published here in Phase 6.
