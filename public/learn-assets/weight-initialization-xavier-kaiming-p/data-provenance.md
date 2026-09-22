# Data and computation provenance

Implementation replay, 2026-09-22: the complete experiment reproduced the conserved 12 September JSON exactly. The website reads compact equivalents of those measurements. The new orthogonal/μP library bridge was executed with PyTorch 2.14.0+cpu and mup 1.0.0; source hashes and explicit checks are in `native-verification.json`. The historical methodology below still describes the unchanged data and fits.

`digits-400.csv` contains 400 actual optical handwritten digit records from UCI's [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), E. Alpaydin and C. Kaynak (1998), DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49), licensed CC BY 4.0. Retain attribution on download. This is not MNIST.

Extraction used scikit-learn's `load_digits()` copy of 1,797 rows from the historical UCI test partition: take the first 40 occurrences of each class 0–9, concatenate by class. `source_id` is the one-based original row, `pixel_0`–`pixel_63` are row-major 8×8 integer values 0–16, and `digit` is the class. File SHA-256: `a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672`. The file is copied unchanged from the previous prepared digit lessons, not regenerated with a different sample selection.

All applied fits use `train_test_split` over the 400 row positions, test_size 0.3, stratified by class, random_state 22: 280 training/120 validation. The JSON saves the original source IDs for each partition. Divide pixels by the known feature-range maximum 16. There is no fitted preprocessing or held-out test in this packet. These are teaching experiments, not the official UCI evaluation or an unseen-writer study.

## Executed calculations

Environment: Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.3.5, scikit-learn 1.9.1, one PyTorch CPU thread. The shared runtime was only used, not modified. `initialization-experiments.py` writes `calculated-inputs.json`.

- Propagation probe: float64, 128 Gaussian vectors × 64 dimensions, 20 linear/ReLU layers, input and output cotangent generator seed 71. Six initializers × seeds 1/2/3. Weight generator resets to seed for each scheme; normal schemes share Gaussian orientations, while orthogonal construction differs. All biases absent. Scalar probe=sum(output*cotangent). Pooled mean/variance(correction 0)/mean square/zero fraction and gradient RMS recorded at every layer. This is sensitivity to one cotangent, not a full Jacobian conditioning result.
- Digit fits: four 32-unit ReLU hidden layers and ten-logit head, all bias zero. Six hidden initializers × seeds 1/2/3; same head within seed from generator 100+seed. Adam 0.003, 300 full-batch updates. Metrics at steps 0/1/10/100/300. 18 actual fits. No selection or retuning after seeing validation.
- Width fits: bias-free 64→n→n→10, ReLU hidden layers; base width 32, widths 32/64/128; standard/μP conventions explicitly implemented; Adam epsilon 1e−8, no weight decay; rates 0.001/0.003/0.01; seeds 1/2/3; 150 full-batch updates. 54 actual fits. Fixed first 32 training rows for mean-absolute-coordinate probes. Records include steps 0/1/2/5/150 and final validation; no final test. Base-width traces match exactly across modes.
- Exact/finite fixtures: ReLU moments, changed symmetric input, average squared gain and singular values, fixed seed-19 Gaussian matrix spectrum and QR Gram identity, local ReLU Jacobian, hidden-unit gradients, zero-head gradients, truncation 100k float64 draws seed 13, float32→BF16/FP16 conversions, geometry constraints and μP arithmetic cases.

After the full run, fixture-only calculation was rerun to add explicit changed/null investigation inputs. The same exact fixture function and JSON retain these values; no training outputs were replaced by imagined curves. Later explicit `model.train()` calls before optimization do not change these stateless architectures (no dropout or running-stat normalization).

## What is derived or unexecuted

Layer recurrence, fan-in/out factors, uniform bounds, fivefold diagonal gains, and μP group arithmetic are formulas, not empirical benchmarks. LSUV and Fixup descriptions are sourced optional explanations; this program does not claim to implement either full procedure. The original writing checkpoint did not execute `mup`. The 22 September implementation executes `initialization_library_bridge.py --mup` against pinned mup 1.0.0, comparing outputs, all parameter gradients and two Adam updates at widths 32 and 96. This remains a restricted bias-free Adam case, not an arbitrary-architecture certification.

The manuscript's tiny standalone `nn.init` example is illustrative complete code; its printed shape follows directly from the declared modules. The full downloadable program was executed. No GPU speed, memory, billion-parameter throughput, optimum-invariance theorem, or claimed production default was inferred from these CPU experiments.

Phase two must preserve outputs/provenance, validate any adapted executable examples, and verify actual UI calculations against these records. If data, architecture, optimizer, or version changes, keep the old results labeled and generate a new evidence record rather than silently editing numbers.
