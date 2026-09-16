# Dropout experiment provenance

Research/write packet prepared 2026-09-12. These are retained authoring inputs; no website implementation is complete.

digits-400.csv contains 400 real records derived from scikit-learn load_digits (1,797 examples in the historical UCI test partition). Dataset: E. Alpaydin and C. Kaynak, [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), 1998, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY 4.0. Retain attribution. This is not MNIST.

The file is unchanged from preceding prepared lessons: first 40 occurrences of each digit0–9, concatenated by class. source_id is one-based row in load_digits; pixel_0–63 are row-major8×8 integers0–16; digit is the label. SHA-256 a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672.

Split is stratified280/120 training/validation, random_state22. Both source-ID arrays are saved. Known pixel maximum16 scales all inputs; no preprocessing is fitted on validation. No test report, official UCI protocol, independent-writer assessment or population-wide confidence claim.

## Executed program

dropout-experiments.py ran with Python3.12.14, PyTorch2.14.0+cpu, NumPy2.3.5, scikit-learn1.9.1, one torch CPU thread. Existing shared runtime was unchanged.

27 complete fits: seeds1/2/3 and nine configurations each. MLP64→64→64→10, tanh, dropout after both hidden activations, p0/.2/.5/.8. Residual model uses64→64 tanh stem, four corrections .5*tanh(Linear64),64→10 head, no normalization. Its five conditions are unmasked, row endpoint.2/.5, batch endpoint.2/.5, all zero-first depth ramps. MLP8,970 parameters; residual21,450. All initial parameters are matched within family and seed; families are not parameter matched.

Adam learning rate.003,400 full-batch updates, no weight decay, augmentation, scheduler or early stopping. Initialization seed1/2/3 is separate from training-mask seed1001/1002/1003. Traces at0/1/25/100/200/400 evaluate both train and validation in eval mode/no_grad. Stored CE is mean negative log predicted target probability; Brier is mean sum of squared class-probability errors. These finite experiments yielded no zero target probabilities.

MC inference is predeclared seed1 MLPp.5 after training,100 draws with seed7001. Only nn.Dropout modules are enabled; no BatchNorm exists in these fitted models. Full mean/std/entropy/disagreement and labels for120 validation rows are retained, but individual100×10 draws are retained only for the first three rows (source IDs251,40,149). Do not fabricate draw-level views for other rows. The model is restored to eval.

Exact float64/autograd mechanisms were run: two-value update; full probability-weighted mask enumeration; structured grids; branch/whole-sum placement; both schedule conventions including L1; BatchNorm moments and actual unbiased running variance; separate no_grad/train and selective-MC state; LN contrast; attention row; p0/p1 boundaries and gradient; eager-versus-lazy branch call counter. Added boundary/counter/hypothetical entropy fixtures by rerunning fixtures() only and preserving all trained outputs.

Shared-mask covariance0/2 and the small practice examples are explicitly derived arithmetic, not empirical estimators. Hypothetical binary probabilities and specialized-family cartoons are labeled illustrative. No pretrained model, large dataset, hardware timing, calibrated uncertainty, human-review deployment or active-learning trial was run.

## Retention and deferred checks

Retain program, CSV, calculated-inputs.json and packet documents for phase-two production. No package was installed and no disposable experiment directory remains. Website rendering, visual/lab implementation, translated calculation parity, download UI, accessibility/mobile/performance and formal integration checks are deferred.
