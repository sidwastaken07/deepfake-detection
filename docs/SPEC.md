# IndiSpoof — Build Specification

A cross-lingual benchmark for audio deepfake detection in Indic languages

Build specification and implementation handoff · Solo research project · Target output: arXiv preprint with public code

## 0. How to use this document

This is an implementation specification, not a tutorial. It is written to be handed to a coding agent and executed phase by phase. Read these rules before starting work.

- Work strictly in phase order. Each phase has a Definition of Done. Do not begin phase N+1 until phase N passes its acceptance criteria and the criteria are recorded in RESULTS.md.
- Phase 5 is a hard gate. If the confound controls fail, the benchmark is invalid and no amount of later modelling fixes it. Stop and repair the data pipeline instead of proceeding.
- Everything is seeded and manifested. No experiment may read audio from a directory scan. All data access goes through a manifest file with a recorded hash.
- Prefer small and finished over large and half-working. Every phase leaves a runnable artifact and a committed result file.
- Ask the human before any download exceeding roughly 20 GB, before any paid API call, and before deleting anything under data/raw/.
- Do not invent numbers. If an experiment has not been run, the table cell stays empty. Placeholder results must never be written to RESULTS.md.

Terminology: bonafide means genuine human speech; spoof means synthetic or converted speech; a generator is one specific model checkpoint used to produce spoof audio; a generator family is a class of architecture shared by several generators.

## 1. Research claim and scope

### 1.1 The single claim

English-trained audio deepfake detectors transfer poorly to Indic-language synthetic speech. This project builds the benchmark that measures the size of that gap, decomposes it into a language effect and an unseen-generator effect, and establishes how much target-language data is required to close it.

Every experiment in this specification exists to support, refute or qualify that claim. If an idea does not serve it, it is out of scope.

### 1.2 Deliverables

- A reproducible generation pipeline producing Indic spoof speech from a diverse roster of open TTS and voice-conversion systems.
- A published evaluation protocol: manifests, splits, seeds, and checksums, released even where audio redistribution is restricted.
- Two trained detector baselines with full per-generator, per-language, per-condition results.
- Five experiment result sets (E1 to E5) with bootstrap confidence intervals.
- A preprint-ready results package: LaTeX tables, figures, and a reproducibility appendix.

### 1.3 Explicit non-goals

- No novel detector architecture. The contribution is evaluation, not modelling.
- No leaderboard competition against funded labs.
- No real-time or on-device deployment.
- No speaker verification integration. Detection only.
- No cloning of identifiable public figures under any circumstances.

## 2. Environment and repository

### 2.1 Assumptions

- Python 3.10 or 3.11. PyTorch with CUDA. Primary compute is a free-tier T4 or equivalent; code must run on a single 16 GB GPU.
- Audio I/O via torchaudio and soundfile. ffmpeg available on PATH for codec simulation.
- Generation and training may run in separate environments. Isolate each generator in its own virtual environment or container if dependency conflicts appear; this is expected and is not a failure.

### 2.2 Repository layout

```
indispoof/
  configs/                 # YAML, one per experiment; no hard-coded paths in src
  data/
    raw/                   # downloaded corpora, never modified
    interim/               # per-generator raw synthesis output
    processed/             # normalised 16 kHz mono; the only audio read
  manifests/               # JSONL, the single source of truth for every split
  results/                 # JSONL metric dumps, one per run, append-only
  paper/                   # generated LaTeX tables and figures
  src/indispoof/
    data/        ingest.py  text_plan.py  manifest.py
    generate/    base.py  registry.py  <one module per generator>
    audio/       normalize.py  vad.py  codecs.py  quality.py
    controls/    silence_probe.py  onset_probe.py  channel_probe.py  report.py
    protocols/   splits.py
    models/      aasist.py  ssl_head.py
    train/       loop.py  losses.py  sampler.py
    eval/        metrics.py  calibration.py  bootstrap.py  report.py
    cli.py
  scripts/                 # thin wrappers only; logic lives in src
  tests/
  RESULTS.md               # human-readable running log of accepted results
  README.md
```

### 2.3 Conventions

- Config: every run is driven by one YAML file. The resolved config is written into the results file for that run.
- Seeds: a global seed per run, recorded. Any experiment reported in the paper is run with three seeds unless stated otherwise.
- Determinism: set torch, numpy and python seeds; log non-determinism where it cannot be removed rather than pretending it is absent.
- Logging: structured JSON lines to results/, plus plain text to stdout. No metric exists unless it is in a results file.
- Hashing: every audio file carries a SHA-256 in its manifest row. Every manifest carries a SHA-256 of itself, recorded in RESULTS.md when the manifest is frozen.

## 3. The manifest schema

One JSONL row per audio clip. This schema is fixed in Phase 0 and every later phase conforms to it. Adding fields is allowed; renaming or repurposing fields is not.

| Field | Type | Meaning |
|---|---|---|
| utt_id | string | Globally unique. Format: <lang>_<source>_<gen>_<index> |
| path | string | Relative to repo root, always under data/processed after Phase 4 |
| sha256 | string | Hash of the audio file bytes |
| label | enum | bonafide \| spoof |
| lang | enum | ta \| hi |
| generator_id | string | A00 for bonafide, else A01..A08 |
| generator_family | string | bonafide \| flow_matching \| vits \| fastpitch_gan \| ar_codec_lm \| vc |
| speaker_id | string | Source speaker identity; for cloned spoofs, the cloned speaker |
| source_corpus | string | commonvoice \| indicvoices \| indictts \| ... |
| text | string | Transcript or synthesis prompt, UTF-8 native script |
| duration_s | float | After normalisation |
| sample_rate | int | Always 16000 after Phase 4 |
| condition | string | clean \| opus_12k \| amr_nb \| mp3_64k \| band_8k ... |
| split | enum | train \| dev \| eval \| held_out |
| qc_wer | float | ASR word error rate from the Phase 3 quality gate; null for bonafide |
| license | string | License string of the source corpus |
| seed | int | Generation seed where applicable |

Rule: a manifest row is immutable once written. Corrections produce a new manifest version with a new hash, and the old hash stays in RESULTS.md.

## Phase 0 — Scaffold and reproducibility harness

Estimated effort: 3 to 5 days. Nothing scientific happens here, and skipping it costs more later than it saves now.

### Tasks

1. Create the repository layout in section 2.2. Package installable with pip install -e.
2. Implement src/indispoof/data/manifest.py: typed read, write, validate, merge, hash, and a strict schema check that refuses unknown or missing fields.
3. Implement the CLI skeleton in cli.py with subcommands stubbed: ingest, plan-text, generate, normalize, controls, protocol, train, eval, report.
4. Implement a run context: resolves a YAML config, sets all seeds, creates results/<run_id>.jsonl, writes the resolved config and git commit hash as the first line.
5. Implement eval/metrics.py now, before any model exists: EER, AUC, minDCF, actDCF. Unit-test each against hand-computed cases and a synthetic score set with known EER.
6. Set up pytest and a CI-style make target running tests plus a lint pass.

### Definition of done

- pytest passes, including metric tests with analytically known answers.
- A dummy end-to-end run writes a valid results file from random scores.
- Manifest validation rejects a deliberately corrupted row.

## Phase 1 — Bonafide corpus ingestion

Estimated effort: 1 week. The goal is genuine Indic speech that is diverse in channel, speaker and recording condition. Channel diversity in the bonafide class is the main defence against the confound described in Phase 5.

### Sources

| Corpus | Role | Notes |
|---|---|---|
| Common Voice (ta, hi) | Primary bonafide | Crowdsourced, highly varied channels. Permissive license. |
| AI4Bharat IndicVoices | Bonafide diversity | Natural and spontaneous speech, many speakers and districts. |
| IndicTTS (IIT Madras) | Bonafide + cloning references | Studio quality. Note overlap risk: several generators were trained on it. |
| IndicVoices-R / Rasa | Optional bonafide | Check overlap with generator training sets before use. |

### Tasks

1. Write one ingest adapter per corpus producing manifest rows with label=bonafide, generator_id=A00.
2. Record the license string per corpus in every row. Do not defer this.
3. Build a training-overlap register: a markdown table listing, for every generator in the Phase 3 roster, which corpora it was trained on according to its model card or paper. Any bonafide corpus that overlaps a generator training set is flagged and excluded from the evaluation split for that generator.
4. Target volume: at least 2,000 bonafide utterances per language from at least two corpora and at least 100 speakers, before normalisation.

### Definition of done

- manifests/bonafide_raw.jsonl validates and contains both languages.
- The overlap register exists and is referenced in RESULTS.md.
- A speaker histogram is plotted and saved; no single speaker exceeds 5 percent of the clips.

## Phase 2 — Synthesis text plan

Estimated effort: 2 days. Purpose: ensure spoof and bonafide audio carry the same lexical content distribution so a detector cannot win by recognising vocabulary.

### Tasks

1. Sample a text pool from the bonafide transcripts, stratified by length in words and by source corpus.
2. For each selected text, record which bonafide utterance it came from. The paired bonafide clip becomes the content-matched counterpart for every spoof generated from that text.
3. Exclude texts with heavy code-mixing in the main pool; set aside a separate code-mixed subset of roughly 200 utterances per language as an optional analysis slice.
4. Choose cloning reference clips: for cloning-capable generators, 3 to 10 seconds per speaker drawn from a reference partition that is excluded from all evaluation splits.
5. Write manifests/text_plan.jsonl: text_id, text, lang, source_utt_id, length_words, reference_utt_id.

### Definition of done

- Length distributions of the text plan and the bonafide pool match within a Kolmogorov-Smirnov p-value threshold recorded in RESULTS.md.
- Reference clips are disjoint from every evaluation utterance, verified by assertion in code, not by inspection.

## Phase 3 — Spoof generation

Estimated effort: 3 weeks, mostly unattended GPU time. This is the phase that makes the benchmark novel, and the phase where generator diversity matters more than clip count.

### 3.1 Generator roster

Target eight generators spanning at least four architecture families. Family diversity is the requirement; specific checkpoints may be substituted if a model is unavailable, provided the family coverage is preserved. Verify availability and license before committing to each one.

| ID | System | Family | Cloning | Langs |
|---|---|---|---|---|
| A01 | IndicF5 (AI4Bharat) | flow_matching | zero-shot | ta, hi |
| A02 | Indic-TTS (AI4Bharat) | fastpitch_gan | no | ta, hi |
| A03 | MMS-TTS (Meta) | vits | no | ta, hi |
| A04 | XTTS-v2 Indic fine-tune | ar_codec_lm | yes | ta, hi |
| A05 | svara-tts | ar_codec_lm | yes | ta, hi |
| A06 | F5-Hindi (SPRING Lab) | flow_matching | varies | hi |
| A07 | kNN-VC | vc | conversion | language-agnostic |
| A08 | FreeVC or RVC | vc | conversion | language-agnostic |

Two generators from the same family are acceptable and useful, because the leave-one-family-out experiment needs at least one family with more than one member to distinguish family effects from checkpoint effects.

Licensing note: several Indic TTS models restrict commercial use and explicitly prohibit unauthorised voice cloning. Research use with consented open-corpus speakers is within scope; cloning any real identifiable individual outside those corpora is not. Record each model license in the roster file.

### 3.2 Generator adapter interface

Every generator is wrapped behind one interface so the pipeline never special-cases a model. Dependency conflicts are resolved by isolating environments, not by weakening the interface.

```
class Generator(Protocol):
    id: str                  # "A01"
    family: str              # "flow_matching"
    languages: list[str]
    needs_reference: bool

    def load(self, device: str) -> None: ...

    def synthesize(
        self,
        text: str,
        lang: str,
        seed: int,
        reference_wav: Path | None = None,
        reference_text: str | None = None,
    ) -> tuple[np.ndarray, int]:   # waveform, sample_rate
        ...

# registry.py maps id -> class; the CLI resolves by id only.
```

### 3.3 Generation run

1. For each generator and language, synthesize the full text plan. Write raw output to data/interim/<generator_id>/ at the model native sample rate. Do not resample here.
2. Record per clip: generator_id, text_id, seed, reference_utt_id, native sample rate, wall-clock time, model version string.
3. Checkpoint and resume. Generation will be interrupted on free-tier compute; a run must restart without regenerating completed clips.
4. Target volume: roughly 1,500 utterances per generator per supported language.

### 3.4 Quality gate (mandatory)

Low-resource TTS fails in ways that are obvious to a listener and invisible to a loss curve: wrong script handling, truncation, silence, looping, or unintelligible output. Unfiltered garbage makes detection artificially easy and the benchmark worthless.

1. Run an Indic ASR model over every synthesized clip and compute word or character error rate against the prompt text.
2. Compute the same ASR error rate over the bonafide pool to establish the reference distribution, since Indic ASR is itself imperfect and an absolute threshold would be meaningless.
3. Reject spoof clips whose error rate exceeds the bonafide median by a margin fixed once, in config, and recorded. Store the value in qc_wer for every surviving clip.
4. Report the rejection rate per generator in the paper. A generator rejecting more than half its clips is reported as a limitation, not quietly dropped.
5. Listen to 20 random surviving clips per generator yourself. Automated gates miss failure modes that are instantly audible.

### Definition of done

- manifests/spoof_raw.jsonl validates, with at least six generators and at least three families represented per language.
- Per-generator rejection rates are recorded in RESULTS.md.
- A manual listening note exists for every generator.

## Phase 4 — Audio normalisation

Estimated effort: 3 days. One pipeline, applied identically and in the same order to bonafide and spoof. Any step applied to one class only invalidates the benchmark.

### Pipeline order (fixed)

1. Decode to float32, downmix to mono.
2. Resample to 16 kHz with a single fixed resampler implementation.
3. Voice activity detection with fixed parameters; trim leading and trailing non-speech. Save the trimmed-off head and tail segments separately: Phase 5 needs them.
4. Loudness normalise to a fixed LUFS target.
5. Peak guard: scale down if clipping would occur; never clip.
6. Write 16-bit PCM wav to data/processed/, compute SHA-256, emit the manifest row.

### Tasks

- Record pre-normalisation and post-normalisation statistics per class: duration, loudness, peak, DC offset, spectral centroid, energy above 7 kHz.
- Plot each statistic as a paired distribution, bonafide against spoof. These plots go in the paper appendix and are the first thing a skeptical reader will want.
- Assert in code that no processed file is outside the expected sample rate, bit depth, or duration bounds.

### Definition of done

- All audio under data/processed is 16 kHz mono 16-bit, verified by assertion over the full manifest.
- Distribution plots exist for every recorded statistic.

## Phase 5 — Confound controls (hard gate)

Estimated effort: 1 week, possibly more if a control fails. This phase decides whether the project produces a real result or an artifact. A detector that separates classes using silence, channel or duration will report a near-zero error rate and mean nothing. This failure mode is documented in the spoofing literature, and it is the first thing a knowledgeable reader will suspect.

### 5.1 The four probes

| Probe | Input | Method | Pass criterion |
|---|---|---|---|
| C1 Silence | Only the non-speech head and tail segments removed in Phase 4 | Small classifier on spectral features | AUC below 0.60 |
| C2 Onset | First 100 ms of each clip only | Same classifier | AUC below 0.65 |
| C3 Channel | 20 hand-engineered global statistics, no temporal detail | Gradient boosting | AUC below 0.70 |
| C4 Duration | Duration distribution per class | Two-sample KS test | p above 0.05, or matched by resampling |

These thresholds are deliberately permissive. Perfect parity is impossible; synthetic speech really does differ in global statistics. The purpose is to establish that a trivial shortcut cannot explain a strong detection result, and to quantify how much of the signal is shortcut rather than artifact.

### 5.2 If a probe fails

1. Diagnose which feature carries the signal, using feature importance for C3 and ablation for C1 and C2.
2. Repair in the data, not in the model. Options: tighten VAD symmetry, add bonafide sources with matching channel character, resample the duration distribution, apply a shared mild augmentation to both classes.
3. Re-run the full probe suite. Record both the failing and the passing values in RESULTS.md; do not delete the failure.
4. If a probe cannot be brought under threshold, the benchmark is still usable but every headline result must be reported alongside the probe AUC, and the limitation must be stated in the abstract. This is acceptable. Concealing it is not.

### Definition of done

- All four probes run, with results committed, whether they pass or fail.
- A written paragraph in RESULTS.md interpreting the probe outcomes. This paragraph becomes a subsection of the paper.
- Explicit human sign-off before Phase 6 begins.

## Phase 6 — Protocol and splits

Estimated effort: 3 days. Splits are frozen here and never touched again. This is what makes the benchmark citable.

### Split rules

- Speaker-disjoint across train, dev and eval. No speaker appears in two splits, in either class.
- Text-disjoint across splits, so lexical memorisation is impossible.
- Reference clips used for cloning never appear in any evaluation split.
- Generator-held-out folds for E3: five folds, each holding out one family entirely from training.
- Language-held-out variant for the cross-lingual analysis.

### Tasks

1. Implement protocols/splits.py producing manifests per split, deterministically from a seed.
2. Write assertion tests: speaker disjointness, text disjointness, reference exclusion, class balance per split, generator coverage per split.
3. Freeze. Compute the SHA-256 of every protocol manifest and record it in RESULTS.md and in the README.

### Definition of done

- All disjointness tests pass as automated tests, not manual checks.
- Protocol hashes are published in the README.

## Phase 7 — Detectors

Estimated effort: 2 weeks. Two baselines, no more. The paper is not about architecture.

### D1 — AASIST (raw waveform, from scratch)

- Raw-waveform graph attention baseline, small enough to train from scratch on a T4.
- Fixed-length crops of roughly 4 seconds at 16 kHz, random crop during training, centre crop plus optional multi-crop averaging at evaluation.
- Serves as the no-pretraining reference point and establishes that the pipeline can learn at all.

### D2 — Self-supervised front-end plus head

- Front-end: a cross-lingually pretrained speech model (XLS-R family) and, as a second variant, an English-centric one (WavLM). The comparison answers whether multilingual pretraining alone buys transfer, which is a genuine open question and costs one extra training run.
- Head: learnable layer-weighted sum over transformer layers, attentive statistics pooling, linear classifier.
- Adaptation: LoRA on attention projections, or freeze the lower N layers. Full fine-tuning of a 300M model will not fit comfortably on 16 GB; do not fight this.
- Training: AdamW, warmup then cosine decay, mixed precision, gradient accumulation to reach an effective batch of 32, early stopping on dev EER, three seeds for any reported number.

### Shared requirements

- Identical data loading, augmentation and evaluation code paths for both detectors. Any difference is a confound.
- Checkpoint selection on dev only. The evaluation split is scored once per final configuration.
- Score files are saved per utterance so metrics can be recomputed without retraining.

### Definition of done

- Both detectors reproduce a sane in-domain result on ASVspoof 2019 LA, within a reasonable margin of published numbers. If they do not, the training code is wrong and nothing downstream is trustworthy.
- Per-utterance score files are written for every evaluation run.

## Phase 8 — Experiments

Estimated effort: 4 weeks. Write the empty results tables before running anything. If a table cell cannot be described in advance, the experiment is not yet defined.

### E1 — The cross-lingual gap

- Train: English spoofing corpora (ASVspoof 2019 LA, and an ASVspoof 5 subset if compute allows).
- Evaluate: English in-domain, then IndiSpoof Tamil, then IndiSpoof Hindi.
- Report: EER per language, per generator, with bootstrap confidence intervals.
- This delta is the headline number of the paper.

### E2 — Data efficiency

- Fine-tune the E1 model with 0, 0.5, 2, 10 and 50 hours of Indic spoof data, three seeds each.
- Plot EER against hours on a log x-axis, one curve per language, with confidence bands.
- Also report the reverse direction: does Indic fine-tuning degrade English performance? Catastrophic forgetting here is a result worth reporting.
- This is the most practically useful figure in the paper.

### E3 — Leave-one-family-out

- Five folds, each excluding one generator family from training entirely.
- Purpose: decompose the E1 gap into a language effect and an unseen-generator effect.
- Be prepared for the finding that most of the gap is generator novelty rather than language. If that is what the data says, report it plainly. It is a more interesting paper than the one you set out to write.

### E4 — Codec and channel robustness

- Conditions: Opus at 6, 12 and 24 kbps; AMR-NB; MP3 at 32 and 64 kbps; 8 kHz band-limited telephone simulation.
- Apply at evaluation time only, to the frozen evaluation split, producing new manifest rows with the condition field set.
- Also run one training variant with codec augmentation to show whether the degradation is fixable by augmentation alone.
- Cheap in compute, probes a weakness the challenge organisers themselves flag, and strengthens the paper considerably.

### E5 — Calibration

- On the best E2 model: reliability diagrams, expected calibration error with 15 bins, actDCF against minDCF.
- Then apply temperature scaling fitted on dev, and report the same metrics after.
- Report calibration separately per language and per condition, because a model calibrated in-domain and wildly overconfident out-of-domain is exactly the practical failure this paper is about.

### Experiment hygiene

- One config file per experiment, committed.
- Every run appends to results/ with the git commit, the resolved config, and the protocol manifest hash.
- No evaluation-split result is used to select anything. Ever.

## Phase 9 — Analysis and reporting

Estimated effort: 1 week, overlapping Phase 8.

### Metrics module requirements

- EER, AUC, minDCF, actDCF, ECE. All computed from saved score files, never from in-training state.
- Bootstrap confidence intervals, 1,000 resamples, for every reported headline number.
- Breakdowns always by generator, by language and by condition. A single averaged number is never the primary report, for the same reason your earlier project refused to average over noise categories.

### Outputs

- scripts/make_tables.py reads results/ and emits LaTeX booktabs tables directly into paper/tables/.
- scripts/make_figures.py emits the E2 data-efficiency curve, the E4 degradation curves, the E5 reliability diagrams, and the Phase 4 distribution plots.
- No number is ever typed into the paper by hand. Everything is generated from results files.

## Phase 10 — Release and preprint

Estimated effort: 2 weeks.

### Code and data release

1. Public repository with pinned dependency versions, the frozen protocol manifests, all configs, and a one-command reproduction script for each experiment.
2. Audio release depends on license. Where redistribution is unclear or restricted, release generation scripts, text plans, seeds and checksums instead, so the exact dataset can be regenerated. State this clearly rather than apologising for it.
3. Model checkpoints released where license permits.

### Ethics statement (required, one paragraph)

- Only speakers from openly licensed corpora who consented to public release.
- No cloning of identifiable public figures.
- Framing is detection research; generated audio is released or withheld according to the licenses of the models used.
- Acknowledge the dual-use tension directly rather than asserting it away.

### Preprint logistics

- arXiv requires an endorsement for first-time submitters in most categories, including eess.AS and cs.SD. An institutional email often clears this automatically; otherwise an endorsing author must vouch for you. Resolve this in Phase 8, not on submission day.
- Fallbacks if endorsement stalls: TechRxiv or Zenodo, both of which issue a DOI.
- Post the code repository link in the abstract. For a benchmark paper, the artifact is the contribution.

## Acceptance criteria summary

A phase is complete when its criterion is met and recorded in RESULTS.md with a commit hash. Do not proceed otherwise.

| Phase | Gate | Criterion |
|---|---|---|
| 0 Scaffold | soft | Metric unit tests pass against analytically known values |
| 1 Bonafide | soft | Two corpora, two languages, 100+ speakers, overlap register written |
| 2 Text plan | soft | Length distributions matched; reference clips provably excluded |
| 3 Generation | soft | 6+ generators, 3+ families per language, quality gate applied and reported |
| 4 Normalisation | soft | Uniform format verified by assertion; distribution plots produced |
| 5 Controls | HARD | Four probes run; results recorded; human sign-off obtained |
| 6 Protocol | HARD | Disjointness tests pass automatically; manifest hashes published |
| 7 Detectors | HARD | In-domain ASVspoof 2019 LA result near published figures |
| 8 Experiments | soft | E1 to E5 complete, three seeds, eval split scored once |
| 9 Analysis | soft | All paper numbers generated from results files, none typed by hand |
| 10 Release | soft | Repo public, ethics statement written, preprint posted |

## Risks and fallbacks

| Risk | Signal | Fallback |
|---|---|---|
| Confound probes fail and cannot be fixed | C3 AUC stays above 0.70 | Report probe values alongside every result; reframe the paper partly as a study of benchmark construction pitfalls, which is itself publishable |
| A generator is unavailable or unlicensed | Model card restricts use | Substitute within the same family; family coverage is the requirement, not specific checkpoints |
| Tamil synthesis quality is poor | High quality-gate rejection rate | Report rejection rates as a finding; shift the second language to Hindi-dominant analysis |
| Compute runs out | Training a 300M front-end does not fit | Use a base-size front-end, or freeze it entirely and train only the head; report the constraint |
| The gap turns out to be small | Cross-lingual EER barely moves | This is a publishable negative result. Pivot the framing to E3 and E4, which remain interesting regardless |
| Scope creep | Urge to invent a new architecture | Re-read section 1.3. Architecture work is explicitly out of scope |

## Paper artifact map

Write the paper skeleton in week one and fill it as results arrive. Each experiment maps to a known place in the document.

| Paper element | Source |
|---|---|
| Table 1: benchmark composition | Phase 3 manifests, per generator and language |
| Table 2: confound probe results | Phase 5, all four probes |
| Table 3: cross-lingual gap | E1, per generator with CIs |
| Figure 1: data efficiency curve | E2, EER against hours, three seeds |
| Table 4: leave-one-family-out | E3, five folds |
| Figure 2: codec degradation | E4, per condition |
| Figure 3: reliability diagrams | E5, before and after temperature scaling |
| Appendix A: normalisation statistics | Phase 4 distribution plots |
| Appendix B: reproducibility | Protocol hashes, seeds, configs, commit hashes |

## Appendix: CLI surface

The complete command surface. If a task cannot be expressed here, it probably belongs in a config file rather than a flag.

```
indispoof ingest      --corpus commonvoice --lang ta
indispoof plan-text   --lang ta --n 1500 --seed 1337
indispoof generate    --generator A01 --lang ta --resume
indispoof qc          --manifest manifests/spoof_raw.jsonl
indispoof normalize   --manifest manifests/spoof_raw.jsonl
indispoof controls    --manifest manifests/all_processed.jsonl --probe all
indispoof protocol    --config configs/protocol_v1.yaml --freeze
indispoof train       --config configs/e1_xlsr_en.yaml --seed 0
indispoof eval        --ckpt runs/e1/best.pt --protocol manifests/eval_ta.jsonl
indispoof codec       --manifest manifests/eval_ta.jsonl --condition opus_12k
indispoof report      --out paper/
```

## Appendix: first week checklist

1. Scaffold the repository and get pytest running.
2. Implement and unit-test the metrics module before anything else.
3. Download one small corpus slice and one generator; produce 50 spoof clips end to end.
4. Run the Phase 5 probes on those 50 clips. They will probably fail at this scale, and seeing the failure early is the point.
5. Write the empty results tables listed in the paper artifact map.
6. Email a potential arXiv endorser.
