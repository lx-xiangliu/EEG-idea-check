# Dataset Audit

## Tier 1

### SparrKULee

- Official: [KU Leuven data landing page](https://homes.esat.kuleuven.be/~spchdata/corpora/auditory_eeg_data/) and [RDR record](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi%3A10.48804%2FK3VSND)
- License/access: CC BY-NC 4.0; full access requires a request.
- Scale: 85 participants; 64 channels; 90–150 minutes/person; natural speech; over 100 GB for the full archive.
- Synchrony/splits: synchronized speech and EEG; supports held-out subject. Story-disjoint and annotation support must be verified from restricted metadata.
- Transcript/phoneme/word timestamps: not confirmed from the public landing page.
- Matched negatives: feasible from long single-speaker audio, but speaker/story metadata quality must be checked.
- Limitation: access-controlled and large; cannot be silently downloaded.

### ICASSP 2023 Auditory EEG Challenge

- Official: [IEEE challenge page](https://signalprocessingsociety.org/publications-resources/data-challenges/auditory-eeg-decoding-challenge-icassp-2023)
- License/access: hosted through KU Leuven; access terms follow the data page.
- Scale: 85 subjects, average 108 min, 157 hours total; tasks are match–mismatch and speech-envelope reconstruction.
- EEG/audio: derived from the SparrKULee ecosystem; synchronized natural single-speaker stimuli.
- Strength: ready-made strong baselines and subject evaluation.
- Limitation: primary task is highly vulnerable to envelope shortcuts; transcript/phoneme timing is not guaranteed.

### ICASSP 2024 Auditory EEG Challenge

- Official: [dataset page](https://exporl.github.io/auditory-eeg-challenge-2024/dataset/)
- Scale: 64-channel BioSemi; official test set contains 20 newly measured subjects.
- Strength: external-subject test set and modern challenge protocol.
- Limitation: password/registration access; test labels and stimulus annotations may constrain semantic evaluation.

### ChineseEEG-2

- Official: [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=cf79be5a415c488b9ebbee97e77a2f16), [project repository](https://github.com/ncclab-sustech/ChineseEEG-2)
- License/access: dataset record is public; exact reuse terms must be read before download.
- Scale: about 95.47 GB; high-density EEG; aligned Chinese text/audio/EEG for reading aloud and passive listening.
- Annotations: aligned semantic embeddings are provided; exact phoneme and word timestamp completeness requires file-level inspection.
- Strength: best audited candidate for cross-modal semantic and story-level tests.
- Limitation: reading-aloud EEG has speech-production/muscle contamination; passive-listening and overt conditions must not be pooled.

## Tier 2

### KUL auditory attention dataset

- Official: [Zenodo record](https://zenodo.org/records/4004271)
- License/access: public record; verify file-level license.
- Scale: 16 normal-hearing subjects; 64-channel BioSemi; recorded at 8196 Hz; two simultaneous speakers.
- Strength: synchronized attended/unattended audio and EEG.
- Major limitation: the official page warns of eye-gaze bias; it is an AAD dataset, not a clean semantic retrieval benchmark. Story/transcript annotations are limited.

### DTU auditory attention dataset

- Public description: 18 subjects, 64 channels, BioSemi, approximately 50 min/person, Danish competing speech; commonly distributed at 512 Hz/downsampled versions.
- Strength: second-site AAD transfer.
- Limitations: access/license link and transcript timing were not confirmed in this audit; dual-speaker attention and audiovisual variants add task-specific confounds.

### UGR-MINDVOICE

- Official: [University of Granada repository](https://digibug.ugr.es/handle/10481/111524?locale-attribute=en), data on OSF linked there.
- License: CC BY-NC-ND 4.0 on the repository record.
- Scale: 15 native Spanish speakers; overt and covert speech; synchronized audio for overt speech; covers Spanish phonemes and semantic word categories.
- Strength: phonetic and semantic labels.
- Limitation: production/covert speech is not listened natural speech; overt EEG contains articulation/EMG shortcuts; held-out stories are not the natural unit.

## Tier 3 / unsuitable for the primary question

### BCI Competition 2020 Track 3

- Official: [competition page](https://brain.korea.ac.kr/bci2020/competition.php); [EEGDash record](https://eegdash.org/api/dataset/eegdash.dataset.NM000113.html)
- Scale: 15 subjects, 64 channels, 256 Hz, 5.2 h; five imagined words/phrases; CC BY 4.0 on EEGDash.
- Limitation: imagined speech has no synchronized acoustic stimulus. It cannot test `I(EEG;audio|C)` or lagged acoustic residualization.

## Recommendation

1. Request SparrKULee/Challenge access for a low-risk match–mismatch benchmark.
2. Inspect ChineseEEG-2 file-level timing and use passive listening only for the primary semantic experiment.
3. Treat KUL/DTU as AAD transfer controls, not evidence of semantic shortcut removal.
4. Use UGR-MINDVOICE only as a separate production/covert study.

## Current environment

No licensed synchronized EEG–audio dataset is present under `/Users/liuxiang/EEG/trf_partial_attention`, and no real-data manifest was configured. Therefore **no real-data experiment was run**.

