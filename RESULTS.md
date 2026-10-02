# RESULTS

Running log of accepted results and phase gates. Rules (from `docs/SPEC.md`):

- Only numbers produced by a run in `results/` appear here. If a run has not happened, its cell stays empty.
- A phase closes only when its Definition of Done is recorded below with a commit hash.
- Failed gates are recorded alongside passing ones and never deleted.

## Phase gates

| Phase | Gate | Status | Commit |
|---|---|---|---|
| 0 Scaffold | soft | **passed** | `560d568` |
| 1 Bonafide | soft | in progress (code ready; Kaggle run pending) | |
| 2 Text plan | soft | | |
| 3 Generation | soft | | |
| 4 Normalisation | soft | | |
| 5 Controls | HARD | | |
| 6 Protocol | HARD | | |
| 7 Detectors | HARD | | |
| 8 Experiments | soft | | |
| 9 Analysis | soft | | |
| 10 Release | soft | | |

## Manifest hash registry

Hashes of frozen manifests (SHA-256 of the file bytes, also stored in `<manifest>.sha256`).

| Manifest | Version | SHA-256 | Commit | Notes |
|---|---|---|---|---|

---

## Phase 0: Scaffold and reproducibility harness

Code commit `560d568`. Environment: Python 3.11.15, numpy 2.4.6, scipy 1.17.1, pytest 9.1.1. torch is
not installed in the Phase 0 environment, so torch seeding was not exercised.

**Definition of Done**

- [x] **pytest passes, including metric tests with analytically known answers.** `make check`: ruff
  is clean and 84 tests pass at `560d568`. The metric tests cover:
  - hand-computed EER, AUC, minDCF and actDCF cases;
  - exact agreement with the ASVspoof 2019/2021 reference `compute_eer` on untied scores;
  - Gaussian score sets with closed-form answers: EER = Φ(−μ/2), AUC = Φ(μ/√2), and minDCF found
    by numerical minimisation of the analytic cost;
  - calibrated LLRs giving actDCF ≈ minDCF, and shifted LLRs inflating actDCF but leaving minDCF
    unchanged.
- [x] **A dummy end-to-end run writes a valid results file from random scores.**
  `results/phase0_smoke_20261002T200453Z_s0_7d89.jsonl` (`make smoke`, seed 0, clean tree at
  `560d568`) passes `indispoof.run.read_results`. An earlier run with seed 0 at `1def2d8` gave
  identical metrics, which confirms the seeding is reproducible. It is a harness check only, not a benchmark
  result: as expected for random scores, EER is close to 0.5.
- [x] **Manifest validation rejects a deliberately corrupted row.** `tests/test_manifest.py`
  corrupts line 2 of a valid manifest 28 different ways. Every one is rejected with `file:line`.
  Malformed JSON, blank lines, duplicate `utt_id` and hash tampering are also rejected.

**Decisions made in Phase 0** (change only through a new schema version):

1. `split`, `qc_wer` and `seed` may be null. Splits do not exist until Phase 6, and `qc_wer`
   does not exist until the Phase 3 gate. Stage flags (`processed`, `gated`, `assigned`) make the
   requirements strict once the relevant phase has run.
2. `condition` is a closed vocabulary: `clean`, `opus_6k`, `opus_12k`, `opus_24k`, `amr_nb`, `mp3_32k`,
   `mp3_64k`, `band_8k`. These are the E4 conditions.
3. The score convention is: higher means more bonafide. A trial is accepted iff score ≥ threshold.
   DCF uses the ASVspoof 5 countermeasure parameters (π_spoof = 0.05, C_miss = 1, C_fa = 10).
4. `utt_id` index may carry dash-separated suffixes (e.g. `ta_cv_A00_000123-opus_12k`), so that
   codec-condition rows stay globally unique.
5. A `smoke` CLI subcommand was added beyond the spec's CLI surface, as the Phase 0 harness check.

**Open question for before Phase 7:** the schema fixes `lang ∈ {ta, hi}` and
`generator_id ∈ A00..A08`. ASVspoof 2019 LA is English, and its own attack IDs A01–A19 mean
different systems. Phase 7 and E1 need it, but it cannot be expressed in this schema without
extending the enums. This needs a decision, e.g. a separate manifest family for external
corpora, or adding `en` plus namespaced generator IDs.

---

## Phase 1: Bonafide corpus ingestion (in progress)

The ingestion code, config (`configs/ingest_v1.yaml`) and Kaggle runbook (`docs/kaggle.md`)
are in place. No corpus has been ingested yet, so the data criteria below are still open.

**Definition of Done**

- [ ] `manifests/bonafide_raw.jsonl` validates and contains both languages. *Pending the Kaggle run.*
- [x] **The overlap register exists and is referenced here:**
  [`docs/overlap_register.md`](docs/overlap_register.md), generated from `configs/roster.yaml`.
  A test fails if the two drift apart. Every entry is marked unverified: it was compiled from
  search summaries because the model cards couldn't be opened from the build environment, and
  each must be re-checked against its model card before Phase 3.
- [ ] Speaker histogram saved, and no speaker above 5% of clips. *Pending.* The merge run writes
  `paper/figures/phase1_speakers_{ta,hi}.png`, logs each check to `results/`, and passes or
  fails the 5% cap per language.

**What the overlap register implies** (provisional until verified):

- **IndicTTS** overlaps 5 of 8 generators (A01, A02, A04, A05, A06). It can serve as bonafide
  only in the evaluation splits of A03, A07 and A08. Its main use is as cloning references.
- **IndicVoices** overlaps A01 and A06, through IndicVoices-R, which is a restored subset of it.
  If IV-R utterance IDs can be matched, an utterance-level exclusion would keep most of
  IndicVoices usable for those generators.
- **Common Voice** overlaps only A04 (the XTTS-v2 Hindi fine-tune). It is the cleanest bonafide
  source, which is why it has the largest target.
- **No Tamil XTTS-v2 fine-tune was found**, so A04 is Hindi-only. Tamil still has 6 generators
  across 5 families (requirement: 6 and 3). Hindi has 8 across 5.
