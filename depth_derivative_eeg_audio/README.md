# Depth-Derivative EEG–Audio Validation

This repository tests—not assumes—the hypothesis that aligning adjacent-layer representation changes is a better inductive bias for EEG–audio learning than aligning accumulated hidden states.

## Current decision boundary

The literature audit found a direct objective-level prior: [Feature Dynamics Distillation (ACL 2025)](https://aclanthology.org/2025.acl-long.1125/) already matches adjacent-layer first differences. The defensible research question is therefore application-specific: whether that objective, combined with a soft monotonic EEG–audio layer map, survives matched-supervision and shuffled-order controls on subject/stimulus-disjoint data.

No real-data result is claimed. The workspace contains no licensed synchronized EEG–audio dataset, and scripts never silently replace missing real data with synthetic data.

## Implemented

- Python 3.10+ / PyTorch, typed dataclass configuration.
- 4-layer pre-LN EEG Transformer and frozen 6-layer controlled audio teacher.
- Final-state, fixed/all-hidden, Matryoshka, DDA fixed, DDA learned, DDA monotonic, shuffled residual, random/cross-sample teacher, and no-audio controls.
- Hierarchical, flat, non-monotonic, parallel, and shuffled-teacher synthetic processes.
- Subject- and stimulus-disjoint synthetic splits.
- Checkpoints, finite-value guards, fixed seeds, CPU smoke test, five-seed suite, CKA/probing/mapping/norm figures.
- Strict manifest leakage checks for subject, stimulus, record ID, and overlapping windows.

## Environment

The completed run used the isolated `.venv` in this repository. To recreate it on another machine:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install --no-build-isolation -e .
```

## Reproduce

```bash
./scripts/run_smoke_test.sh
./scripts/run_ablation.sh
```

For a short diagnostic that still uses five seeds:

```bash
./.venv/bin/python scripts/run_synthetic.py --quick
```

Validate a real-data manifest without training:

```bash
./.venv/bin/python scripts/prepare_data.py --manifest /path/to/manifest.json
```

The manifest must include `dataset_name`, `version`, `sampling_rate`, and records with `record_id`, `subject_id`, `stimulus_id`, `start_sample`, `end_sample`, and `split`. Subject and stimulus IDs must be disjoint across train/validation/test.

## Outputs

- `reports/novelty_matrix.csv`: 50-paper overlap matrix.
- `reports/literature_review.md`, `novelty_audit.md`, `theory_review.md`, `dataset_audit.md`.
- `outputs/synthetic/results.csv`: one row per mode/method/seed.
- `outputs/synthetic/details.json`: mapping, CKA, probes, and norms per run.
- `reports/synthetic_validation.md`, `statistical_analysis.md`, `representation_analysis.md`.

## Real-data gate

Real expansion is allowed only after: (1) local licensed data are supplied; (2) official subject/story splits are encoded in the manifest; (3) H1 teacher-residual hierarchy probes pass; and (4) DDA beats matched-capacity hidden alignment in the synthetic hierarchy while losing its advantage under flat/shuffled controls.

## Limitations

The synthetic teacher deliberately plants hierarchy. A synthetic success proves only that the implementation can recover a controlled structure. Network depth is not cortical depth, model residuals are not guaranteed semantic increments, and a flexible projector can turn DDA into ordinary multi-layer distillation.
