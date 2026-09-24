# Maximum Likelihood and MAP: lesson design

Stable topic `maximum-likelihood-map-estimation`, Math Foundations position16. Status: authored and author-verified; parent integration/user acceptance are separate. Parent authors the preceding Probability lesson and owns shared integration. The title remains appropriate. Topic inventory has no incoming destination note; unrelated bit-manipulation inbox remains out of scope. Final evidence: [verification record](MAXIMUM-LIKELIHOOD-MAP-VERIFICATION.md).

## Preserve and repair the existing lesson

The old published body is preserved in `scratch/maximum-likelihood-native-verification/original-lesson.jsx` before editing. Retain its parameter-estimation purpose; likelihood-versus-parameter-probability distinction; log-likelihood; Gaussian sample `[2,3,4,7]` with mean4; Gaussian-prior/L2 and Laplace-prior/L1 connections; limited-data prior influence; underflow, overfitting, model mismatch and the outlier-versus-median task.

Weaknesses: formulas precede a worked inference; the iid factorization does not state conditioning assumptions; Gaussian estimates are asserted rather than derived; no likelihood/prior geometry, interactive investigation, independent graded practice or curated sources; no boundary/nonexistent/nonunique optimum or MAP coordinate caveat; variance and uncertainty are absent. “Weight decay is a Gaussian prior” must be qualified to objective-level L2 and the optimizer convention. The recent Gradient Variants section6 already teaches coupled L2 versus AdamW; reuse its precise connection rather than routing an already-resolved duplicate.

## Intended finish and progression

Core finish: formulate an observed-data likelihood with its assumptions/support; maximize it analytically or by a validated one-dimensional computation; derive Bernoulli and normal estimates; combine likelihood and a stated prior; distinguish MAP, posterior mean and predictive probability; diagnose when point estimates or the chosen model fail. Deeper route: parameterization and Jacobians, identifiability, regularization scaling, separated logistic/no-finite-MLE and regularity limits. Next syllabus topic is Hypothesis Testing & Confidence Intervals, then Bayesian Inference & Conjugate Priors. Do not skip ahead to a later published owner.

| Hurdle | Explanation, visual and assessment |
| --- | --- |
| Which quantity varies? | Fixed binary observations, candidate parameter ruler and likelihood curve. Ordered sequence probability versus count probability differs by a constant combinatorial factor; same parameter argmax, different observed events. |
| Products, zero support and log scores | Define conditional independence given parameter; stable xlog terms; explain zero-count endpoint convention and all-success/no-data cases. Native direct/log comparison. |
| What does optimization have to do with a mean? | Preserve `[2,3,4,7]`; connect residual lines to squared and absolute loss. Derive normal mean and variance MLE, the n versus n−1 distinction, and known-scale Laplace median; add an outlier. |
| How does a prior change the answer? | Beta prior density plus observed likelihood gives a normalized posterior; compare MLE, mode, mean and next-event predictive probability. Density is not point probability; formula conditions are explicit. |
| What does “regularization” mean probabilistically? | Complete normal-normal shrinkage calculation; Gaussian and Laplace priors yield scaled L2/L1 objectives. Sum versus mean loss scaling and AdamW distinction. |
| Why are mode and uncertainty different? | Exact finite binomial distribution of possible estimates under a specified true p, contrasted with a fixed-data likelihood and parameter posterior. Sampling variation, finite-sample bias and conditional large-sample intuition without promising universal normality. |
| Why can MAP move under a relabeling? | Beta(5,3) in probability coordinates versus log-odds coordinates; show transformed density and Jacobian, preserve interval mass, distinguish likelihood relabeling from parameter-density transformation. |
| When is there no unique finite solution? | All-identical unknown-scale normal; closed-support uniform boundary; identifiable total versus nonidentifiable component offsets; scalar logistic separation and a proper regularizing prior. |

## Planned representation contracts

- BernoulliLikelihoodLab: editable bounded binary observations, candidate p and ordered/count view; exact data strip, likelihood ratios and model-derived curve; endpoints/no-data explicitly handled; input applies/reset clearly.
- LocationFitLab: actual observed values on a number line with candidate-center residual segments; linked squared/absolute costs and mean/median targets; known scales stated, outlier changes the data explicitly.
- BetaMapLab: bounded positive integer prior shapes and successes/failures; prior/posterior density and relative-likelihood axes kept semantically separate; mode/mean and predictive readouts; uniform/boundary states explain nonuniqueness. Singular noninteger-shape cases are a deeper analytic caveat, not silently clipped plots.
- SamplingEstimateLab: exact binomial mass bars over possible k/n, user-specified true p and sample size; no Monte Carlo randomness or fabricated empirical claims.
- CoordinateModeFigure: computed probability/log-odds density views and numerical mode correspondence. Optional identifiability ridge figure if it improves the ordinary reading flow; no visual quota.

Every graph's coordinates come from stated formulas. Curve discretization is a rendering approximation; maxima and reported quantities use analytic formulas or checked root solving. Readable axes/units, keyboard controls, text equivalents and local scrolling when useful. No uniform lab template imposed on all mechanisms.

## Research and verification plan

Primary materials inspected initially: Cornell CS4780 estimation notes (coin/MAP/predictive derivation, while independently correcting its oversimplified general consistency language); Stanford CS109 MLE lecture notes (Bernoulli/normal/support examples); MIT official 20.9 MLE video page (page inspected, not full playback); original AdamW paper abstract and the existing verified Gradient Variants lesson. Research remaining coordinate/regularity claims against primary statistical references before implementation. Annotated source ledger and alternate written/video resources will distinguish what was actually read.

Complete Python3.12 standard-library programs with exact checked outputs; independent exact-fraction/binomial/polynomial integration oracles for bounded beta/likelihood states, derivative/grid/cost checks, normal residual identities and optimizer brackets. Test malformed inputs, impossible observations, empty data, endpoint estimates, all-identical measurements, nonunique medians and mismatched support. Compare every native output before browser review. Final desktop/390/320 keyboard/ordinary-reading and opened screenshots precede freeze. Parent owns production build and shared registration.

## Final implementation decisions

All planned core and deeper learning hurdles are implemented in nine connected sections. Four investigations use different mathematical objects: fixed observations with candidate likelihood; individual signed measurement residuals with squared/absolute costs; likelihood/prior/posterior curves with explicitly different vertical meanings; and exact repeated-study estimator masses. The inline coordinate figure compares densities in p and log-odds with a preserved interval-mass calculation. The optional additional identifiability plot was unnecessary: the three concrete offset pairs, exact equal costs, support/failure comparison table and complete executable program make that narrow ridge example explicit without another graph to decode.

The original normal dataset, mean4, outlier task, log stability, priors and L2/L1 connections remain. The former broad weight-decay claim is repaired and links to the actual verified Gradient Variants owner. The lesson adds normal variance MLE versus unbiased variance, exact Beta boundaries/no-data, mean versus MAP versus prediction, normal-normal precision weighting, loss-scaling coefficients, soft thresholding, model dependence, estimator sampling variation, transformed modes and nonunique/nonexistent optima. No title, stable identity, order, catalogue membership or unrelated lesson changed.

Source research was completed against the Cornell/Stanford teaching materials, Geyer's primary likelihood notes, the Stan optimization reference and the original AdamW abstract. The video recommendation is the actual MIT OCW20.9 page; its title/description were inspected, not full playback. See the verification ledger for the precise reviewed scope and independently checked computations.

One useful discovery belongs to the next lesson: [Hypothesis Testing & Confidence Intervals](topic-notes/hypothesis-testing-confidence-intervals.md) should assess a worked Bernoulli boundary interval and finite-coverage comparison, following this lesson's explicit zero plug-in-SE failure. The existing pilot mentions Wilson but the scoped inspection found no worked boundary case. That note carries sources/limits and preserves the receiving author's judgment; no destination body was rewritten here.
