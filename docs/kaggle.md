# Running IndiSpoof on Kaggle

The code is developed in this repo, but the data-heavy steps run in a Kaggle notebook (T4 GPU).
This page covers the one-time setup and then Phase 1 (bonafide ingestion) step by step. Each
block below is one notebook cell.

## One-time setup

1. **Kaggle account.** Phone-verify it so notebooks can use the internet. In the notebook, set
   *Settings → Internet* to on and *Accelerator* to GPU T4 (Phase 1 doesn't need the GPU).
2. **Hugging Face.**
   - Create a read token at <https://huggingface.co/settings/tokens>.
   - Open <https://huggingface.co/datasets/ai4bharat/IndicVoices> while logged in and accept its
     access terms. It is gated, and downloads fail until you do.
3. **Mozilla Data Collective** (Common Voice is only distributed there now).
   - Create an account at <https://mozilladatacollective.com>.
   - Open the Common Voice Scripted Speech 25.0 pages for Tamil and Hindi and accept the terms.
   - Under *Download → API / Python Access*, note each dataset ID. Tamil's is
     `cmn2gfvyp01geo107izoftfki`.
   - Create an API key under *Profile → Credentials*.
4. **GitHub token**, so the notebook can push results: a fine-grained personal access token with
   *Contents: read and write* on `sidwastaken07/deepfake-detection` only.
5. **Kaggle secrets.** In the notebook, open *Add-ons → Secrets* and add `HF_TOKEN`,
   `MDC_API_KEY` and `GH_TOKEN`.

## Disk layout on Kaggle

`/kaggle/working` keeps at most about 20 GB of output between sessions. Everything else is lost
when the session ends. So:

- full downloads (Common Voice archives, parquet shards) go under `/tmp/raw`, which is not saved;
- `data/raw/<corpus>` in the repo is a symlink into `/tmp/raw`, so manifest paths stay
  `data/raw/...` everywhere;
- after ingestion, only the selected clips are copied out (`scripts/pack_audio.py`) and saved
  as a private Kaggle Dataset that later phases attach.

## Phase 1 cells

**1. Clone, install, identity, secrets**

```python
from kaggle_secrets import UserSecretsClient
import os
s = UserSecretsClient()
os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN")
os.environ["MDC_API_KEY"] = s.get_secret("MDC_API_KEY")
os.environ["GH_TOKEN"] = s.get_secret("GH_TOKEN")

%cd /kaggle/working
!git clone -b claude/epic-clarke-40s24l "https://x-access-token:$GH_TOKEN@github.com/sidwastaken07/deepfake-detection.git"
%cd deepfake-detection
# Drop the token from .git/config: /kaggle/working is saved with the notebook version.
!git remote set-url origin https://github.com/sidwastaken07/deepfake-detection.git
!pip install -q -e ".[dev,data]" datacollective
!git config user.name "sidwastaken07" && git config user.email "sp.siddharth2006@gmail.com"
!make check
```

**2. Point `data/raw/<corpus>` at scratch disk**

```bash
%%bash
for c in commonvoice indicvoices indictts; do mkdir -p /tmp/raw/$c && ln -sfn /tmp/raw/$c data/raw/$c; done
ls -l data/raw
```

**3. Common Voice (Tamil and Hindi).** Each archive is downloaded once and only the language
folder is extracted. Check the archive size on the MDC page first: if it is over ~20 GB, stop
and decide before downloading.

```python
import os, glob, tarfile
os.environ["MDC_DOWNLOAD_PATH"] = "/tmp/raw/commonvoice/_archives"
from datacollective import download_dataset

CV_IDS = {"ta": "cmn2gfvyp01geo107izoftfki", "hi": "<Hindi dataset id from the MDC page>"}
for lang, ds_id in CV_IDS.items():
    archive = download_dataset(ds_id)   # resumes if interrupted
    print(lang, archive)
```

```bash
%%bash
# Extract only <version>/<lang>/ from each archive. Adjust the name if the version differs from
# configs/ingest_v1.yaml (corpora.commonvoice.version and the two roots).
cd /tmp/raw/commonvoice
for a in _archives/*.tar.gz; do
  tar -tzf "$a" | head -3
  tar -xzf "$a" --wildcards 'cv-corpus-25.0-2026-03-09/ta/*' 'cv-corpus-25.0-2026-03-09/hi/*' 2>/dev/null || true
done
ls cv-corpus-*/*/validated.tsv
```

**4. IndicTTS and IndicVoices shards.** `--dry-run` lists the chosen files and their total size
without downloading anything. Anything over `max_gb` (20 GB) is refused unless `--allow-large`
is passed, and the spec says to ask before doing that.

```bash
%%bash
C=configs/ingest_v1.yaml
for corpus in indictts indicvoices; do
  for lang in ta hi; do
    indispoof ingest --config $C --corpus $corpus --lang $lang --download --dry-run
  done
done
```

If IndicVoices reports *no remote files match*, the error lists the repo's top-level folders.
Fix `hf.pattern` and `files` under `corpora.indicvoices.langs` in `configs/ingest_v1.yaml`, then
re-run. Once the dry runs look right:

```bash
%%bash
C=configs/ingest_v1.yaml
for corpus in indictts indicvoices; do
  for lang in ta hi; do
    indispoof ingest --config $C --corpus $corpus --lang $lang --download
  done
done
```

**5. Check the parquet column names.** The IndicVoices column names in the config
(`audio_filepath`, `normalized`, `speaker_id`, `duration`) could not be verified from the build
environment. If the printed schema differs, edit `corpora.indicvoices.columns` and commit the fix.

```bash
!indispoof ingest --config configs/ingest_v1.yaml --corpus indicvoices --lang ta --inspect
!indispoof ingest --config configs/ingest_v1.yaml --corpus indictts --lang ta --inspect
```

**6. Ingest each corpus × language, then merge**

```bash
%%bash
set -e
C=configs/ingest_v1.yaml
for corpus in commonvoice indicvoices indictts; do
  for lang in ta hi; do
    indispoof ingest --config $C --corpus $corpus --lang $lang
  done
done
indispoof ingest --config $C --merge
```

The merge run prints, and logs to `results/`, every Phase 1 check per language: utterances
≥ 2000, corpora ≥ 2, speakers ≥ 100, largest speaker share ≤ 5%. It also writes
`paper/figures/phase1_speakers_{ta,hi}.png` and `.csv`.

If a check fails, don't edit the manifest. Raise a `target` or `max_per_speaker`, or add shards,
then re-run into a new `parts_dir` and `output`. Manifests are immutable.

**7. Persist the selected audio**

```bash
!python scripts/pack_audio.py manifests/bonafide_raw.jsonl /kaggle/working/indispoof-audio
```

Then, in the notebook's right panel, use *Save Version*. Turn the output folder
`indispoof-audio` into a private Kaggle Dataset (e.g. `indispoof-audio`). In later sessions,
attach that dataset and point the repo's `data/raw` at it:

```bash
for c in commonvoice indicvoices indictts; do ln -sfn /kaggle/input/indispoof-audio/data/raw/$c data/raw/$c; done
python scripts/pack_audio.py manifests/bonafide_raw.jsonl --verify-only   # all hashes must match
```

**8. Push the manifests, results and figures**

```bash
%%bash
git add manifests results paper/figures configs
git commit -m "Phase 1: bonafide ingestion run on Kaggle

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push "https://x-access-token:${GH_TOKEN}@github.com/sidwastaken07/deepfake-detection.git" HEAD:claude/epic-clarke-40s24l
```

Audio is never committed (`data/` is git-ignored). Manifests, their hashes, results and figures
are.

After pushing, tell Claude. It will check the results against the Phase 1 Definition of Done
and record them in `RESULTS.md`.
