# Novelty Audit

## Evidence chain

The 62-paper matrix found no direct EEG–audio implementation of

\[
\mathrm{softmax}((M_{C_E}Q_E)(M_{C_A}K_A)^\top/\sqrt d)V_A.
\]

However, every constituent is established: lagged TRF design, ridge/FWL residualization, null-space concept erasure, conditional contrastive learning, hard negatives, and deconfounded attention. The novelty therefore resides in the exact operator placement plus EEG–audio shortcut-focused evaluation.

## Risk assessment

| Claim | Assessment | Risk |
|---|---|---|
| “First use of residualization” | False; extensive prior art. | Critical |
| “First causal EEG–audio attention” | Unsupported; residualization is not causal adjustment. | Critical |
| “First lag-expanded acoustic residualization of EEG Q and audio K” | Not found in audit; defensible only as a scoped claim. | Medium |
| “New conditional-MI estimator” | False without an explicit density/MI estimator and proof. | Critical |
| “New shortcut-resistant EEG–audio protocol” | Plausible, especially matched negatives and true/shuffled/random C controls. | Medium-low |

## Novelty grade

**B, with a substantial risk of being judged C.** The exact attention operator was not found, but reviewers can reasonably characterize it as “TRF + ridge residualization + cross-attention”. A paper needs decisive evidence that attention-level placement outperforms input/loss-level alternatives; otherwise the architecture claim collapses and only the protocol remains.

## Stop condition

One direct prior paper applying lagged observed nuisance residualization to Q and K in cross-modal attention would downgrade the work to D unless the EEG protocol itself adds a separate contribution.

