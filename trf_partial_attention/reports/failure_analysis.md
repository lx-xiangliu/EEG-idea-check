# Failure Analysis

## Triggered failures

- **Architecture failure (E):** in a score-level benchmark, residualizing the representations before Q/K projection is algebraically equivalent to residualizing Q/K when those projections are identity/linear and commute with the tested operation. TPA provides no extra effect.
- **Mechanism failure (D):** random subspace projection is competitive or better on key metrics.
- **TRF necessity failure (H7):** lag expansion does not beat no-lag.
- **Information destruction risk (H):** fuller acoustic control reduces both leakage and semantic probe performance.
- **Generalization/task failure:** primary held-out-subject matched retrieval does not improve.

## Not established

- Novelty failure was not triggered: no direct operator duplicate was found.
- Real no-residual-dependence failure cannot be assessed without real data.
- Compute failure was not triggered on CPU smoke sizes, though GPU cost is unmeasured.

## Alternative explanations

1. Linear projection removes shared phonetic/semantic variance because linguistic events are acoustically coupled.
2. The synthetic semantic latent is partly time-constant; intercept/centering decisions can erase it.
3. Generic rank/energy reduction changes similarity calibration independently of acoustic specificity.
4. Leaving V unchanged may reintroduce acoustic content downstream even when score leakage falls.
5. A trained nonlinear projector could adapt around residualization, restoring leakage.

## Direct falsifier

The clearest real-data falsifier is: on held-out subjects and stories with acoustically matched negatives, true-C TPA fails to beat both simple input residualization and shuffled/random-C controls while higher-level probes do not improve. The current synthetic study already exhibits that pattern.

