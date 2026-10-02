# Training-overlap register

<!-- Generated from configs/roster.yaml by `make overlap-register`. Do not edit by hand. -->

For each generator in the Phase 3 roster: what it was trained on, according to its model
card or paper. **✗** marks a bonafide corpus that overlaps the generator's training data.
That corpus is excluded from the evaluation split for that generator (Phase 6 enforces
this).

| ID | System | Family | Langs | License | Training data | commonvoice | indicvoices | indictts | Verified |
|---|---|---|---|---|---|---|---|---|---|
| A01 | IndicF5 | flow_matching | ta, hi | MIT | Rasa; IndicTTS; LIMMITS; IndicVoices-R |  | ✗ | ✗ | no |
| A02 | AI4Bharat Indic-TTS (FastPitch + HiFi-GAN V1) | fastpitch_gan | ta, hi | MIT | IndicTTS |  |  | ✗ | no |
| A03 | MMS-TTS | vits | ta, hi | CC-BY-NC-4.0 | MMS-lab (New Testament readings) |  |  |  | no |
| A04 | XTTS-v2 Hindi fine-tune | ar_codec_lm | hi | Coqui Public Model License (non-commercial) | XTTS-v2 base corpus; Common Voice 18 (hi); IndicTTS (hi) | ✗ |  | ✗ | no |
| A05 | svara-TTS v1 | ar_codec_lm | ta, hi | Apache-2.0 | SYSPIN; RASA; IndicTTS; SPICOR |  |  | ✗ | no |
| A06 | F5-Hindi | flow_matching | hi | CC-BY-4.0 | IndicTTS (hi); IndicVoices-R (hi) |  | ✗ | ✗ | no |
| A07 | kNN-VC | vc | ta, hi | MIT | WavLM-Large pretraining (English); LibriSpeech train-clean-100 (vocoder) |  |  |  | no |
| A08 | FreeVC | vc | ta, hi | MIT | VCTK; WavLM-Large pretraining (English) |  |  |  | no |

## Exclusions by bonafide corpus

- **commonvoice** (CC0-1.0): excluded from the eval split of A04.
- **indicvoices** (CC-BY-4.0): excluded from the eval split of A01, A06.
- **indictts** (CC-BY-4.0): excluded from the eval split of A01, A02, A04, A05, A06.

## Family coverage

- **ta**: 6 generators, 5 families (ar_codec_lm: A05; fastpitch_gan: A02; flow_matching: A01; vc: A07, A08; vits: A03).
- **hi**: 8 generators, 5 families (ar_codec_lm: A04, A05; fastpitch_gan: A02; flow_matching: A01, A06; vc: A07, A08; vits: A03).

## Notes

- **indicvoices**: IndicVoices-R is a restored subset of IndicVoices (same speakers and utterances), so any generator trained on IndicVoices-R overlaps IndicVoices. A finer, utterance-level exclusion (only the IndicVoices-R subset) is possible later if IV-R utterance IDs can be matched.
- **indictts**: Two speakers per language (one male, one female), 48 kHz studio recordings.
- **A04**: The spec lists ta and hi, but no Tamil XTTS-v2 fine-tune was found. A04 is Hindi-only unless one turns up; Tamil family coverage still holds without it.
- **A08**: Chosen over RVC because RVC needs a model trained per target voice.

## Sources

- **commonvoice**: <https://mozilladatacollective.com/datasets/cmn2gfvyp01geo107izoftfki>
- **indicvoices**: <https://huggingface.co/datasets/ai4bharat/IndicVoices>, <https://arxiv.org/abs/2409.05356>
- **indictts**: <https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil>, <https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi>, <https://www.iitm.ac.in/donlab/indictts/database>
- **A01**: <https://github.com/AI4Bharat/IndicF5>, <https://huggingface.co/ai4bharat/IndicF5>
- **A02**: <https://arxiv.org/abs/2211.09536>, <https://github.com/AI4Bharat/Indic-TTS>
- **A03**: <https://huggingface.co/facebook/mms-tts-tam>, <https://arxiv.org/abs/2305.13516>
- **A04**: <https://huggingface.co/Abhinay45/XTTS-Hindi-finetuned>
- **A05**: <https://huggingface.co/kenpath/svara-tts-v1>
- **A06**: <https://huggingface.co/SPRINGLab/F5-Hindi-24KHz>
- **A07**: <https://github.com/bshall/knn-vc>, <https://arxiv.org/abs/2305.18975>
- **A08**: <https://github.com/OlaWod/FreeVC>, <https://arxiv.org/abs/2210.15418>

**Unverified:** A01, A02, A03, A04, A05, A06, A07, A08. These entries were compiled from search summaries, not read from the model cards directly. Re-check each one against its model card before Phase 3 uses that generator.
