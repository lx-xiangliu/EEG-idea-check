# Smoke Test Report

## Result

- Status: **PASSED**
- Device: CPU
- Seed: 0
- Epochs: 2
- Model: 4-layer EEG Transformer + frozen 6-layer audio teacher
- Method: dda_monotonic
- Probe accuracy: 0.8958
- Best validation loss: 0.438783
- Trainable parameters: 20,812
- Training wall time: 0.121 s
- End-to-end script time: 1.212 s
- Budget gate (<30 min): **PASS**

## Stage status

- Completed: forward/backward pass, checkpoint, unseen-subject probe, CKA, mapping diagnostics.
- Failed: none.
- Missing: GPU execution was not required for the CPU smoke gate.
- Key findings: finite loss and deterministic metrics were obtained.
- Blocking issues: none for synthetic work.
- Decision: implementation passes the smoke gate.
