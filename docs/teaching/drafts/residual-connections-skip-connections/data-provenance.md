# Residual experiment provenance

Prepared research/write packet, 2026-09-12. All attached files are retained production inputs; website implementation is pending.

## Real data

`digits-400.csv`: 400 real examples from UCI [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), E. Alpaydin and C. Kaynak (1998), [DOI10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY4.0. Retain attribution. This is not MNIST.

Copied unchanged from the preceding prepared lessons. From scikit-learn's `load_digits()`1797 rows (historical UCI test partition), select first40 examples of each class and concatenate by class. Source IDs are one-based original rows; pixel0–63 are row-major8×8 values0–16; final digit is class0–9. SHA-256: a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672.

The program makes a stratified 280 training / 120 validation split with random_state22 and saves both source-ID arrays. Divide input pixels by the known feature-range maximum16. All models share the split. No fitted preprocessing, external download, held-out test, official UCI benchmark or unseen-writer claim.

## Actually executed

Python3.12.14, PyTorch2.14.0+cpu, NumPy2.3.5, scikit-learn1.9.1, one torch CPU thread; existing shared runtime unchanged.

39 fitted networks: seeds1/2/3; one stem-only baseline per seed; block depths2/6/12 for plain, additive residual, fixed1/sqrt(depth) scaled residual, and trainable zero-gate residual. Stem64→32+tanh and head32→10 have independent deterministic generators seed+100/200 across all depths/modes. Each block has LayerNorm32 (epsilon1e−5, affine), Linear32→32, tanh, Linear32→32. Both matrices Xavier-normal using seed*1000+block_index; all linear biases0. Thus plain versus ordinary residual matches all learned parameter initial values; fixed scale changes forward scaling; zero gate adds one scalar per block.

Adam0.003,250 full-batch updates, no dropout, weight decay or schedule. Metrics are CE and correct/count at steps0/1/25/100/250. Diagnostics use the fixed first 32 training rows and their actual labels: loss-specific stem-weight gradient norm, state mean square, and activation-gradient RMS. These are not complete Jacobian singular-value measurements.

Final parameter-displacement records distinguish trained branches from a model that only trains its head. Every zero-gate branch matrix changed; smallest such displacement across runs was about2.7634 in Frobenius norm. Initial zero-gate training metrics match the corresponding stem-only baseline exactly. Learned scale values can be negative.

Each nonplain fitted model also has separate single-block omissions evaluated on validation without retraining. They are explanatory ablations, not a selected pruning policy or a test report. No combined-omission results were generated.

`mechanisms` contains actual float64/autograd fixtures: one correction and SGD update; one-entry weight edit; scalar depth sensitivities; post-ReLU/post-LN identity failures; zero final linear versus zero ReLU gradients; zero/nonzero scalar gates; nonlinear path-expansion counterexample; projection/changed-projection gradients; scalar Euler values; pure addition saved-tensor hook.

The complete fitting program ran once. Later exact fixture additions were recomputed with the same `mechanisms()` function and merged into the output; fitted results were not rerun or replaced. No browser tests, app build, SVG rendering or formal implementation review occurred.

## Formula and illustration boundaries

Matrix/gradient identities and simple count/storage expressions are derived calculations. The denoising values are explicitly hypothetical arithmetic, not trained outputs. LayerScale, CNN/Transformer families, checkpointing and neural-ODE discussions are sourced explanations; they are not additional implemented or benchmarked methods. In particular the zero-gate digit model retains LayerNorm and is called ReZero-style, not an exact paper reproduction.

Phase two must preserve these distinctions and validate any translated numeric behavior. Any added experiment or data selection must get its own provenance and outputs.
