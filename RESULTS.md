# RESULTS

Running log of accepted results and phase gates. Rules (from `docs/SPEC.md`):

- Only numbers produced by a run in `results/` appear here. If a run has not happened, its cell stays empty.
- A phase closes only when its Definition of Done is recorded below with a commit hash.
- Failed gates are recorded alongside passing ones and never deleted.

## Phase gates

| Phase | Gate | Status | Commit |
|---|---|---|---|
| 0 Scaffold | soft | **passed** | `560d568` |
| 1 Bonafide | soft | | |
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
