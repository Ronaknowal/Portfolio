# Titans teaching experiments: declared protocol

Declared 13 September 2026 before executing these experiments. This packet is research/write only. None of these experiments reproduces the published Titans language-model benchmarks.

## Mechanism experiments

Use the half-squared associative loss and column-vector keys. Hand trace a two-dimensional linear memory with an orthogonal second key and a correlated second key; calculate both reads, gradients, momentum and decay. Inspect zero key, zero learning rate, momentum-only change, full decay and an explicit reset. Compare sequentially recomputed gradients with a fixed chunk-anchor gradient using two scalar examples, plus chunk size one as the equality case. Validate a functional nonlinear memory and an outer derivative with centered finite differences. Include a scalar differentiable write whose outer derivative can also be computed by hand.

A deliberately small three-branch gated sequence block uses identity key/query/value projections, two recent tokens, one fixed prefix vector and a two-by-two linear fast memory. It is an executable teaching specialization of the MAG topology, not a trained Titans model. Its chosen product gate and identity projections are declared. Execute input changes, prefix removal, write disable, explicit reset, separate-request isolation, streaming continuation and future-input perturbation. No token output may depend on a later token.

## Real observations: next-day rentals

Reuse unchanged UCI Bike Sharing daily observations already retained in the forecasting packet, verifying its recorded SHA256. The reuse is appropriate because a dated stream lets learners distinguish information available before a forecast from information available for a subsequent write. Recheck the provider's dataset/license page and read its supplied description. The previous forecasting lesson's results are known; this different model/protocol has not been run. Retain the full CSV and provenance locally so this packet runs independently offline.

Question: after fitting a small nonlinear predictor on 2011, what happens when its parameters continue to learn from each newly observed day in 2012? This is supervised online adaptation after target arrival, exposing the same gradient/momentum/decay mechanism. It is distinct from Titans' learned latent self-supervised key/value objective and end-to-end outer training.

- Data: all731 daily records, oldest first. Assume a day's total is available at that day's end. Use only dates and total count; no future observed weather or casual/registered target components.
- Fit period: zero-based rows0–364; prediction examples at target indices7–364. Fit mean and population standard deviation on counts0–364 only.
- Replay: target indices365–730. First183 predictions are a development report, final183 a subsequent assessment report. No reselection, retraining or reset between reports; each later day may use earlier arrived outcomes. These labels identify reporting windows, not independent IID replicates.
- Key: seven most recent standardized counts in oldest-to-newest order, sine and cosine of the target date's weekday. Normalize the resulting nine-vector to unit L2 norm. Target: standardized next-day total. No intercept appended; memory includes biases.
- Model: nine inputs, eight hidden SiLU units, one output. Float64 CPU, PyTorch, one thread. Seeds3,7,19. Ordinary initial supervised training: full-batch Adam, rate0.01, exactly1000 updates, mean half-squared loss, no decay. No early stopping or candidate search.
- Main matched intervention: frozen copy versus adaptive copy of each fitted model. Adaptive update after recording the forecast and observing that day's target: inner rate0.005, momentum0.5, decay0.0001, half-squared scalar error. No clipping. Each seed's adaptive state starts from the same fitted weights as its frozen baseline; zero momentum. Count predictions are not clipped.
- Predeclared simple baselines: previous-day count and count seven days earlier. Evaluate identical target dates.
- Retain every forecast, error, prewrite residual, gradient norm, update norm, fitted parameter, seed and both reporting-window MAE/RMSE. Show per-seed outcomes, not only the most flattering run. No wall-clock benchmark or architecture-ranking claim.
- Author verification: verify timestamps/order, input maxima, split counts and source identity; independently aggregate errors; perturb a future outcome and check all earlier predictions unchanged; test the zero-update/zero-decay null and exact checkpoint continuation. A controlled score-after-write example belongs to constructed arithmetic, not a legitimate extra test result.

If numerical instability occurs, preserve that result and its fixed protocol rather than silently tuning until adaptation wins. Correct implementation errors with a documented correction and rerun affected computations. Implementation/UI checks remain phase two.
