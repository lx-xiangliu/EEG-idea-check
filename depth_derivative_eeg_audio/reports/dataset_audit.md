# Dataset Audit

## Selection principles

The first real-data task should be match–mismatch with both subject-disjoint and stimulus/story-disjoint evaluation. License applies to dataset files, not automatically to speech copyright or model redistribution. No raw data are redistributed by this repository.

| Dataset | Official source | License / access | Subjects | Channels / rate | Synchronized audio | Conditions | Suitability / caveat |
|---|---|---|---:|---|---|---|---|
| SparrKULee / ICASSP 2023 corpus | [KU Leuven RDR](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND), [challenge docs](https://exporl.github.io/auditory-eeg-challenge-2023/dataset/) | CC BY-NC 4.0 is stated for non-commercial users; verify repository terms before download | 85 challenge participants (71 train, 14 held-out) | 64; raw 1024 Hz and preprocessed 64 Hz | Yes; separate stimuli | listened single-speaker stories | **Best first benchmark.** Official held-out stories and subjects. Audio/stories may have separate rights; no redistribution. |
| KUL auditory attention | [Zenodo 4004271](https://zenodo.org/records/4004271) | Open Zenodo record; verify the per-file license field before reuse | 16 | 64; recorded 8192 Hz, shared preprocessed 128 Hz | Yes | attended two-speaker | Strong AAD benchmark, but authors warn of eye-gaze shortcut bias. Not ideal as sole mechanistic evidence. |
| DTU auditory attention | [Zenodo 1199011](https://zenodo.org/records/1199011) | Open Zenodo record; verify per-record license before reuse | 18 | 64; 512 Hz | Yes | attended competing Danish speech; reverberation | Useful second AAD dataset; small N and attention label can be decoded via shortcuts. |
| UGR-MINDVOICE | [University of Granada repository](https://digibug.ugr.es/handle/10481/111524?locale-attribute=en), [OSF](https://osf.io/6sh5d) | Repository page states CC BY-NC-ND 4.0 for the item; confirm dataset file terms | 15 | See official paper/metadata before preprocessing | Overt audio synchronized; covert has same stimuli but no produced audio | overt and covert Spanish speech | Valuable production/covert test; overt EMG/articulation artifact is a major confound. |
| ChineseEEG-2 | [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=cf79be5a415c488b9ebbee97e77a2f16) | CC0 on dataset record | 12 (4 reading aloud, 8 listening) | high-density; exact montage/rate in metadata | Raw participant voice withheld for privacy; embeddings/materials supplied | reading aloud and passive listening | Good semantic probe corpus, but lack of raw audio limits teacher extraction and subjects differ by task. |
| 3M-CPSEED | [Scientific Data / OpenNeuro ds006465](https://openneuro.org/datasets/ds006465/versions/2.0.0) | CC BY 4.0 | Verify version metadata | 16-channel according to dataset title/record; verify rate | Speech-production prompts; check exact synchronized audio content | overt, mouthed, imagined Chinese pinyin | Useful production control; muscle/articulation leakage must be measured. |
| AASD 2026 | [Scientific Data article](https://doi.org/10.1038/s41597-026-07244-w) | Article states CC BY-NC-ND 4.0; verify linked dataset record separately | See official record | Raw CNT + processed MAT; exact montage/rate in record | Spatialized two-speaker WAV | spontaneous auditory attention switches | Valuable temporal-latency stress test; new and not yet a standard benchmark. |
| ESAA | [Zenodo 7078451](https://zenodo.org/records/7078451) | Open record; verify explicit license before use | See record | See record | HRTF-filtered speech included | auditory attention | Secondary dataset after license/metadata verification; not selected for first run. |

## Recommended first benchmark protocol

1. Use the official ICASSP/SparrKULee subject and held-out-story partitions.
2. Fit every scaler/filter statistic on training recordings only.
3. Segment after partitioning; never random-split overlapping windows.
4. Group by subject and story/stimulus ID in manifests.
5. Report held-out-subject and held-out-story separately; do not hide them in the challenge weighted average.
6. Use matched temporal negatives from the same story plus cross-story negatives as separate difficulty strata.
7. Do not redistribute audio, EEG, cached teacher features, or subject metadata.

## Reproducibility fields required in any real run

Dataset DOI/version; checksum manifest; subject IDs per split; story IDs per split; preprocessing code version; raw/preprocessed choice; sample rate; reference; channels; window and overlap; artifact rejection; audio feature model/checkpoint/layer; teacher stride; latency handling; seeds; optimizer; batch size; epochs; hardware; wall time; package lock.

## Current availability decision

No real dataset is present in this workspace. Downloading 5–95+ GB datasets and accepting their terms is not implied by this task. Therefore the repository implements a strict manifest loader and does not silently substitute synthetic data for a real benchmark. Real-data reports must remain `not_run` until a licensed local dataset root is explicitly supplied.

## Stage status

- Completed: official-source and license/access audit for eight candidates; first-benchmark choice.
- Failed: none.
- Missing: per-file license confirmation for Zenodo records and exact metadata for datasets not selected first.
- Key findings: SparrKULee has the strongest official unseen-subject/unseen-story protocol; KUL has documented eye-gaze bias.
- Blocking issues: no real EEG–audio data in workspace; dataset downloads/terms and large compute not authorized.
- Decision: synthetic closure now; real benchmark deferred without fabrication.
