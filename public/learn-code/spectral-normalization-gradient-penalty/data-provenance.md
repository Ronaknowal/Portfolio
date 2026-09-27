# Data and calculation provenance

Prepared13 September2026 for research-and-writing-only delivery. All trained results below were produced during authoring, not copied from the previous JSX or fabricated for figures. No website model was implemented.

## Recorded observations

`digits-400.csv` is the unchanged400-row extract already retained in the cross-attention packet: first40examples of each class0–9 in scikit-learn's optical digits dataset. Columns source_id(one-based source row),64integer pixels0–16 and digit. Source is [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), E. Alpaydin and C. Kaynak,1998, DOI10.24432/C50P49, licensed CC BY4.0. Attribution and license must accompany later public downloads. The [scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) identifies the bundled8×8 data. The selected extract is a teaching subset, not the official entire dataset split.

CSV SHA256: `a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672`.

Each profile is(left-half sum,right-half sum)/512, with32pixels per half and known maximum16. No estimated centering/scaling parameters are used. Equal integer-sum pairs define394groups; retain multiplicities but assign an entire group to one role. NumPy default_rng91 permutes groups in first-occurrence order; first floor(.6×394) groups fit, through floor(.8×394) development, remaining assessment. Actual images241/79/80. `protocol.roles`, `collision_groups`, `source_ids` and `measurements` retain the exact assignment and observations. Class labels do not enter loss or split. Different images can share a profile; no attempt is made to recover an image from it. Writer IDs are unavailable, so this split does not establish writer-level independence. The same teaching dataset has appeared in preceding lessons; it is not a fresh confirmatory benchmark.

## Declared model experiment

`critic-regularization-study.py` ran completely on Python3.12.14, NumPy2.3.5, SciPy1.18.1 and PyTorch2.14.0+cpu, one thread. The output records package versions, constants, roles, raw observations and all results. Generator2→24→24→2 with ReLU/ReLU/sigmoid,722parameters. Critic2→24→24→1 with leaky-ReLU .2,697parameters; no batch-coupled normalization. Three methods(clipping .1,GP lambda10,SN one power iteration per weight access)×seeds11/29/47. Nine independent paired runs,600generator steps,3critic steps each,batch64,Adam .001 betas(0,.9). Raw initialization and fitting/latent random streams are paired. GP interpolation has its own generator to avoid changing those common draws.

All methods use D minimization fake-minus-real plus regularizer and G minimization negativefake. D uses detached generated inputs. G freezes D parameters and SN buffer iteration through evaluation mode while retaining input derivatives. Clipping applies initially and after each D step to all D parameters. SN wraps all dense D weights; approximate effective norms are measured afterwards by SVD. Checkpoint observations1/100/300/600 include development discrepancy and all256generated evaluation profiles. No early stopping, architecture selection, hyperparameter sweep or assessment-based choice occurred.

Evaluation uses a shared256×2 standard-normal latent sample, Torch seed2026. A fixed NumPy seed2026 bootstrap samples256fitting profiles with replacement. For64directions at anglesjπ/64, `scipy.stats.wasserstein_distance` calculates empirical1D W1 between projected samples; mean over directions gives the reported finite directional discrepancy. It is not FID, exact multivariate W1 or likelihood. The bootstrap is .006793913fit/.009954363development/.010039217assessment. All9method/seed metrics appear in the manuscript and JSON, including the poor GP outcomes; no repeat campaign sought a more flattering result.

`calculated-inputs.json` also preserves final complete G weights/biases and effective D weights/biases, all generated profiles, all400critic scores, assessment gradient norms,41×41grid scores/gradients over[0,1]², actual per-layer spectra/product bound and fixed latent edit outputs. Effective D weights support frozen inference; raw SN weights, cached singular vectors and optimizer state are not a claimed resumable training checkpoint. Source program and seeds reproduce the declared training procedure.

## Exact examples and independent frozen inference

`sensitivity-calculations.py` does not fit models. It computes the small matrix/power/convolution examples, GP derivative/update, missed-probe region, batch-centering Jacobian, composition/residual bounds and margin geometry. `sensitivity-results.json` retains their actual values. The2×2 spectral-normalization gradient is checked against central differences(step1e−5), maximum error6.9703132155e−12. Evaluation-mode removal with `leave_parametrized=True` preserves the small dense layer's outputs exactly.

Independent NumPy matrix/activation inference checks every saved generator on256latents and each critic on400profiles. Largest generated-value discrepancy8.560356335e−8; largest critic-score discrepancy1.092668629e−7. Saved latent edit discrepancies are below5.4e−8. Simultaneously swapping latent coordinates and first-layer columns produces exact unchanged double outputs for all nine models; swapping only latent coordinates changes outputs, with actual maximum differences retained. These are bounded author checks of the handoff's mathematical claims, not formal independent phase-two review.

Fresh convolution kernel[1,2] has kernel norm√5; valid length3 operator AAᵀ=[[5,2],[2,5]], so normalized full norm√(7/5). The author calculation also verifies this value, the moved-probe penalty32/3 and fresh margin radius1.25/√2 in `fresh_investigations`; phase two must check their browser implementation. Other editable-state outcomes must be computed from actual inputs, not generated by interpolating preset screenshots.

## Retention and limits

Keep both programs, CSV, both result files and the three manuscripts/specification/design records as the pending content packet. They are reproducible inputs for implementation and examples, not disposable scratch. No screenshots, downloaded videos, generated raster assets, temporary scripts or large model checkpoints were created. Runtime publication, full browser labs, mixed-precision execution, GPU timing, formal independent review and deployment remain unperformed.
