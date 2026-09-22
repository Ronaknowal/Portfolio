> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Gaussian Processes (GP): prepared-content design and evidence

Mode: **research and write only**, revision 1, 2026-09-12. Author: classical_probabilistic_content. Content packet prepared for root checkpoint; implementation not started. This is an author review record, not independent review or shipped browser evidence.

Files: [full manuscript](lesson.md), [visual/lab contracts](visual-specifications.md), [offline data](mauna-loa-monthly.csv), [provenance](data-provenance.md), [source receipt](data-source.json), [author calculations](author-calculations.py), [calculated results](checked-results.json).

## Contract, sequence and ownership

Stable ID gaussian-processes-gp; title retained. It already names the method, including regression and extensions; adding CO₂ or probes to the title would confuse example with scope. Current location is Classical ML, Probabilistic & Graphical Models, after CRF and before Semi-Supervised Learning. No navigation or catalogue changes.

Learner arrives with vectors, Gaussian vocabulary and a little Python. Earlier matrix/linear-regression lessons exist, but local refreshers introduce covariance units, variance versus standard deviation, joint conditioning, matrix shapes, Cholesky solves and held-out MAE. The GP lesson does not require reading later Evaluation Metrics, Calibration, Active Learning or Time-Series Validation first.

Observable outcomes: translate a finite Gaussian vector into function values; reject an invalid covariance; compute a conditional mean/variance; distinguish latent from measurement intervals; predict effects of observations/kernel/noise; execute and interpret a fixed real-data forecast; diagnose misspecified uncertainty; optionally choose a measurement by target-specific information and explain approximate inference/KRR boundaries.

First-pass route is introductory contract plus §§1–7. §§8–9 are explicit deeper branches with readiness stated. Proposed reading time after the actual manuscript is roughly 45–60 minutes, with 60–100 minutes for core calculation, programs and practice; optional branches add 30–60 minutes. These estimates are editorial, not measured user study or an authoring quota.

### Scope decisions and neighboring owners

| Idea | Inspected evidence | Decision and durable home |
| --- | --- | --- |
| KRR equality, sample-path RKHS boundary, fixed random feature map | Incoming topic note links Functional Analysis & RKHS design and the old GP posterior/approximation discussion; Kanagawa Prop3.6/§4 read | Include in §9 with σ²=nλ under average loss, anchored Brownian example, finite-rank exception, feature storage/solve distinction. Own note disposition below |
| Posterior uncertainty and calibration | Old GP passage called a latent band “confidence”; upcoming calibration/conformal topic has separate owner within current range | Teach correct conditional interval semantics locally, expose real coverage failure; reserve distribution-free calibration/conformal mechanisms for that lesson |
| Sensor selection | Old GP mentioned active learning/BO mostly as named uses | Add target-covariance reduction calculation and I3 because it directly reuses GP conditioning. Broad labeling strategies belong to Active Learning |
| Time series | Existing GP article used sine interpolation; scikit/GPML CO₂ example presents kernel composition | Add independently specified NOAA historical holdout. Root owns fuller rolling-origin protocols; bridge before next method without reordering |
| Bayesian optimization | Original GP includes EI/gp_minimize and generic tool names | Preserve distinct objective through derived EI worked/changed examples; no unsupported ecosystem ranking or unexplained black-box optimizer output |
| GP classification and extensions | Original likelihood/GPC API, scalability, ARD material read | Preserve likelihood break, Laplace, predictive integration, EP/variational alternatives, scaling and ARD. Add derivative/multioutput mechanism in optional §9. No promise of complete standalone classifier training in core |

### Canonical section-list coverage map

Canonical reference: Rasmussen/Williams GPML free book. Read contents pp vii–ix (PDF pages6–8), then targeted primary chapter sections below. A contents read is not a claim to have reread the entire 266-page book.

| Canonical family | Treatment here and reason |
| --- | --- |
| Ch2 weight/function views, Gaussian regression, prediction, mean functions, implementation | Core §§1–3 and real program; direct random-line bridge and nonzero-mean equation |
| Ch3 classification, decision theory, Laplace, multiclass, EP | Optional §9 explains posterior break and approximate integration; detailed multiclass/EP algorithms remain reference-directed extensions |
| Ch4 covariance validity, stationary/dot-product/nonstationary, composition, smoothness | Core §4 plus PSD counterexample. Eigenfunction analysis/string/Fisher kernels not required for current outcomes; RKHS links mathematical route |
| Ch5 Bayesian model selection and cross-validation, hyperparameters, examples | Core §§5–6; no replacement of prospective evaluation by training density |
| Ch6 RKHS, regularization, splines/SVM relationships | §9 exact KRR correspondence; spline/operator details belong to RKHS/numerical-model extensions |
| Ch7 equivalent kernels, consistency, learning curves, PAC-Bayes | Out of this foundational GP page; upcoming learning-theory owners treat generalization, canonical book remains route to GP-specific results |
| Ch8 reduced rank, subsets, Nyström, iterative approximations | §9 mechanism-first inducing/variational/SKI/random-feature overview and truthful costs; no exhaustive obsolete method catalogue |
| Ch9 multiple outputs, dependent noise, non-Gaussian likelihood, derivatives, uncertain inputs, mixtures, optimization, integrals, t processes, invariances, latent variables | Dependent noise core; derivatives/multioutput/optimization developed. Remaining extensions explicitly not claimed as taught; they need specialist prerequisites and reference reading |

## Conservation and corrections

Original source src/learn/data/topics/gaussian-processes-gp.jsx, all 938 lines read in bounded chunks, including references and exercises. Original source SHA256 ee89796c165c0acaa4c97853dd8ce57b036dfb5c803a1b31c1ffa3459999272e, baseline commit 8c5da59f18516be77c29d5aeeafca3decca4f738. Preserved foundations, posterior derivation, kernel menu, Cholesky implementation, marginal likelihood, practical GPR, graphical intuition, method choice, scaling, failure interpretation and practice. Reorganized them around a novice route and real experiment.

Replaced unsupported history/tool rankings/claimed sine numerical results with traceable mechanisms and actual computations. Corrected: parametric models can have uncertainty; not every GP is “all functions”; latent versus observation bands; pointwise versus simultaneous coverage; fixed-parameter variance invariants; noisy observations need not interpolate; RBF uncertainty returns to prior at remote targets; Matérn differentiability convention; product kernels are not products of GP samples; hyperparameter optimization is not full Bayesian integration; cross-validation still useful; known heteroscedastic Gaussian noise is exact; jitter changes the model; full dense matrix at50k needs20GB before overhead; prediction mean/variance have different costs; random features do not make a dense solve O(nD); no universal scalability threshold/ranking.

Original GPC and BO snippets were illustrative wrappers, not conserved as duplicated runnable programs: mechanisms and changed numerical decisions now provide the learning value, while the two complete core programs teach exact inference and genuine empirical evaluation. No claim of classifier/BO toolkit execution remains.

## Hurdle and evidence map

| Hurdle | Mechanism / example | Representation | Learner evidence |
| --- | --- | --- | --- |
| Function prior feels infinite/unusable | Finite vector indexing, random-line GP, PSD difference variance | F1 | Explain forbidden bend/negative variance |
| An observation informs a different input | rho=.5 one-reading conditional, rho0 null | F2/I1 | Predict change, practiceA changed noise/value |
| Mean zero confused with no information | Symmetric two-reading mean cancellation and .468895 variance | F3/I1 | Construct mean-null but variance-change case |
| Width called universal confidence | Latent variance plus new noise, precise bands | F2/I1 | PracticeC settings audit |
| Kernel controls draw geometry | Shared seeded RBF paths, menu, sum/product, periodic counterexample | F4 | PracticeE drifting seasonal signal |
| Fitted density mistaken for future success | LML terms and fixed split/family selection | F5/I2 | PracticeD data/evaluation correction |
| Most uncertain point called most informative | Conditional covariance variance reduction | F6/I3 | Candidate choice, target shift, exact zero-covariance null |
| GP extensions become unrelated definitions | Likelihood changes, inducing variance, KRR equality/limits | Equations plus original EI/KRR practice | Explain each changed assumption and solver cost |

## Claim/retrieval ledger and alternate resources

All retrieval dates 2026-09-12 unless a pre-existing note states otherwise. Explanations and worked examples are newly written. Technical claims use primary sources; no quoted passages copied. Actual source modality is specified.

| Source / URL | Locator actually inspected | Claims / scope |
| --- | --- | --- |
| https://gaussianprocess.org/gpml/chapters/RW2.pdf | Weight-space opening; §2.2 eqs2.21–2.24 and implementation context | Finite GP, Gaussian posterior/noise, equivalence; manuscript calculations independently executed |
| https://gaussianprocess.org/gpml/chapters/RW4.pdf | §§4.1–4.2, eqs4.9,4.14–4.20; Matérn smoothness lines467–517 | RBF/Matérn/RQ forms and mean-square differentiability; composition assumptions |
| https://gaussianprocess.org/gpml/chapters/RW5.pdf | Contents; §5.4 eq5.8 and local-optima discussion; CO₂ kernel/noise passages | MLL integrates latent function, model selection/CV, no generic fixed extrapolation guarantee |
| https://gaussianprocess.org/gpml/chapters/RW.pdf | Contents pp vii–ix; §3.4 posterior Laplace expression; §9.4 derivative covariance | Canonical coverage map and extensions, not entire-book verification |
| https://scikit-learn.org/stable/modules/gaussian_process.html | GPR noise/optimizer, GPC Laplace/multiclass and kernel descriptions | sklearn1.9.1 semantics |
| https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html | alpha, kernel bounds/optimization, prediction API | alpha diagonal noise versus WhiteKernel; author execution validates chosen return_std setup |
| https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html | “Design the proper kernel,” long trend/seasonality and locally periodic construction | Annotated practical alternative; its numbers not imported into our experiment |
| https://distill.pub/2019/visual-exploration-gaussian-processes/ | Written conditioning/GP/kernel sections and textual interactive descriptions | Geometric alternate article read; widgets not operated. Not adopted notation slip equating SD to covariance diagonal |
| https://mlg.eng.cam.ac.uk/teaching/4f13/1213/lect0304.pdf | Slides10–15 and25–27, marginalization/joint generation/weight covariance | Alternate lecture notes read. No video watched or timestamps asserted |
| https://arxiv.org/pdf/1807.02582 | Prop3.6, matching σ²=nλ, Example4.1 and sample-path discussion | Incoming RKHS note resolved carefully; anchored-space and finite-rank qualification retained |
| https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf | Eqs6–11, especially9 and trace-term interpretation | Variational inducing mechanism/approximation |
| https://arxiv.org/pdf/1309.6835 | §§2–3.1, eqs1–4, separable likelihood contribution | Stochastic variational mechanism; no unsupported universal deployment threshold |
| https://proceedings.mlr.press/v37/wilson15.html | Author abstract | SKI structure/interpolation concept only; no quantitative runtime ranking |
| NOAA data/disclaimer/source CSV, linked in provenance | Headers, selected rows and flags, public-domain conditions | Historical measured values, attribution and chosen-range validity |

Video search was performed. The Cambridge2021 course page exposes a syllabus but puts lecture videos on Moodle and warns older links differ; an MIT CBMM candidate page failed to open. Do not add unverified playable-video claims. The verified public lecture notes and interactive article provide substantive alternatives; no resource-format quota overrides usefulness or access.

## Author checks and freeze boundary

Executed author-calculations.py with shared read-only Python3.12.14, NumPy2.3.5, SciPy1.18.1, sklearn1.9.1. It generates all tiny posterior vectors, exact variance/null comparisons, interval endpoints, shared-seed prior samples, EI original/changed cases, NOAA development/final predictions and explorer horizon-prefix equality assertions. Kernel full log theta is retained, not reconstructed from rounded display strings.

Executed the **exact two Python blocks extracted from the manuscript**, not only equivalent author helper code. Outputs matched the manuscript to displayed precision: means/variances, LML−2.952256, covariance invariant True; dev MAEs5.432943/.320601; selected trend_periodic; test1.233775/1.312967/13; seasonal baseline3.813333. Execution used a synthetic __file__ path beside the actual retained CSV, without writing duplicate executable artifacts.

Inspected source CSV negative-spread flags: none in120 selected rows. Separate final test uses only selected family/initialization, refit on96 rows. Explorer uses frozen1990–95 hyperparameters and is clearly distinguished. Author bounded checks are evidence for these examples, not independent validation of every GP implementation or universal coverage.

Manual derivations: covariance PSD counterexample, random-line kernel, constant-kernel support failure, one-reading changed practice, conditioning variance monotonicity under fixed model, kernel sums/products distinction, independent-kernel no-information candidate, EI zero-SD limit, KRR normalization, Brownian anchored-space boundary, derivative variance and dense storage arithmetic. These are mathematical checks, not fabricated empirical runs.

Learning-experience reread checks: introduction supplies purpose; first-pass route appears before advanced detail; symbols/units introduced; core examples have intermediate reasoning and usable programs; uncertainty cautions appear where interpreted; figures are situated at conceptual hurdles; editable investigations require unsolved predictions and include nulls; exercises have closed hints and reasoned changed-input solutions; measured failure remains visible; optional branches are labeled; sequence closes to Semi-Supervised Learning. Final source read and exact output checks are recorded at completion below.

Completion: reread the entire final manuscript in three contiguous ranges and the complete visual specifications. Resolved an ambiguous return from rho0 to rho.5, repaired Markdown math delimiters and normal-distribution notation, and made frozen-hyperparameter explorer semantics explicit in the learner text. The two extracted program executions and the final author calculation succeeded; the last calculation additionally retained exact fitted log theta and all six explorer horizon-prefix checks. These are the source-level content completion checks. No independent or browser verification claimed.

Deferred: actual rendered figures, interactive state/model code, downloadable-program packaging, native tests, independent teaching/correctness review, browser/keyboard/narrow validation, loading/build checks, final route integration and ledger update. Root owns ledger checkpoint; do not mark implementation complete. No pending drafts/data are disposable scratch. No runtime source, manifest, generated navigation, shared handoff or shared ledger changed by this author.
