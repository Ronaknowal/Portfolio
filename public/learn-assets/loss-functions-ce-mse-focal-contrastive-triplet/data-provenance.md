# Real handwriting input and execution record

Source: UCI **Optical Recognition of Handwritten Digits**, E. Alpaydin and C. Kaynak (1998), DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49), [dataset page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). UCI distributes it under CC BY4.0; attribution must accompany redistributed rows. This is not MNIST.

The supplied digits-400.csv is byte-identical to the approved Perceptrons packet's extraction from scikit-learn load_digits. It selects the first40 occurrences of each original digit0–9 and concatenates by digit, giving400 observations. source_id is the one-based row in the1797-row sklearn copy of the UCI historical test partition. pixel_0 through pixel_63 are row-major8×8 integers0–16; digit is the original label. No synthesized pixels or externally generated examples.

CSV SHA256: a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672. The source, loader description and original extraction were inspected in the Perceptrons packet. This packet copied that exact local asset and uses no runtime source download.

loss-experiments.py defines the binary target digit==9 and stratifies the local70/30 split on original digit with seed22. It stores all280 training and120 validation source IDs. The selected original UCI test rows are repartitioned for this small educational development experiment; these results are not an official UCI test score or a writer-independent estimate. There is no untouched final test.

Actual author execution on2026-09-12: Python3.12.14, NumPy2.3.5, scikit-learn1.9.1, torch2.14.0+cpu; one torch thread. Nine400-update CPU fits, seeds1–3, identical nn.Linear(64,1) initialization per seed/objective, Adam.03. Validation probabilities and metrics are actual outputs in calculated-inputs.json. Raw training objectives have different formulas and are not a common ranking metric. No model was selected from these runs; no timing/throughput claim is made.

The JSON's mechanism_fixtures section is synthetic mathematical input, explicitly separate from the real digit rows. It includes focal values/derivatives, triplet distances/mining, candidate probabilities and losses, collapsed squared-triplet derivative, and analytic regression minima. Regression minima are closed-form calculations rather than numerical solver outputs. Standalone excerpt comments labeled derived are mathematical predictions; clean installation, exact snippet replay and final runtime review remain phase two.
