# f-Divergences & Integral Probability Metrics — scoped design

10 September 2026. Mathematics position 29; stable ID `f-divergences-integral-probability-metrics`. Design prepared before replacement; completed author implementation and actual evidence are recorded in [verification](F-DIVERGENCES-IPMS-VERIFICATION.md). Independent integration and user acceptance remain separate. The exact inventory and complete original lesson were read. There is no destination note; the unrelated bit-manipulation inbox does not apply. Parent owns shared registration and progress.

## Learning contract and preservation

Retain the title: both named families need a complete comparison, with MMD and statistical estimation developed locally. The pre-revision lesson had useful framing, probability-ratio versus observable intuition, one complete categorical program and practical cautions, but no actual visual investigations, derivations of IPM/KL witnesses, MMD calculation, independent practice solutions or verified learning resources. Preserve the original program and its independently rerun output `0.208 0.054 0.3`; archive the original body at `scratch/divergence-ipm-native-verification/original-lesson.jsx`.

Connect from Entropy/KL, Mutual Information and Optimal Transport. Locally refresh weighted averages, convexity, dot products, kernels as feature inner products, and a supremum as the best permitted score. Do not silently require the later Functional Analysis/RKHS lesson. The later advanced branch may link to it for construction/completeness theory. Module reading order remains Rate-Distortion → this topic → Graph Fundamentals; related-topic links do not replace that sequence.

Beginner finish: explain why a mass-ratio comparison and a geometry-sensitive comparison can disagree; calculate finite examples with correct zero-support conventions; select a comparison aligned to a specified failure. Intermediate: derive an optimal event/critic, compute kernel MMD from actual pair similarities, distinguish an empirical discrepancy from a hypothesis test, and perform a small exact permutation test. Deeper: common-measure definition, data processing, Pinsker/local curvature, variational f-divergence bounds and restricted-critic limits, with careful generative-model connections.

## Coverage and ownership decisions

| Idea | Current evidence / best owner | Decision |
| --- | --- | --- |
| General f-divergence and support | Original ratio formula omits singular contribution | Own finite boundary rules, common dominating measure, generator conventions, nonnegativity, direction and shared-channel DPI here. |
| TV and statistical decisions | Original only defines event supremum | Derive maximizing event and equal-prior classification accuracy; connect risk differences and Pinsker. |
| Metrics versus divergences | Original uses names without distinguishing square roots | Give concrete triangle failure and correct Hellinger/JS roots; explain IPM pseudometric separation conditions. |
| W1 critic and topology | Optimal Transport already teaches transport/duals; original comparison vague | Derive finite ordered critic by summation by parts and moving-atom contrast here, link full transport mechanics. |
| MMD, kernels, estimators and tests | Only mentioned in original; no separate MMD title found in narrow catalogue search | Own feature-mean mechanism, kernel expansion/witness, characteristic versus restricted kernels, biased versus unbiased squared estimators, and exact permutation calibration here. |
| Importance-weight variance | Revealing use of Pearson chi-square, not explicit original coverage | Add the finite identity Var_Q(P/Q)=chi-square under P≪Q and show why proposal coverage matters; this is not a full importance-sampling course. |
| GAN/f-GAN/WGAN | Existing GAN Fundamentals owner; original blanket gradient claims | Give mathematically qualified discriminator/JS and Fenchel-critic bridges here; route full practical training/regularization investigation to GAN Fundamentals with a durable note. |
| Abstract RKHS construction / kernel theory | Functional Analysis & RKHS is a later catalogue topic | Introduce only needed inner-product/feature/reproducing identities locally, route deeper kernel separation/embedding conditions there if a concrete useful gap emerges. |
| Distribution-shift / fairness applications | Original lists domains without calculations | Develop a small harmful-change stress test and the conditional decision meaning; no generic metric ranking or fairness certification. |

## Concept and representation plan

| Hurdle and placement | Mechanism/example | Representation and meaningful action | Evidence / boundaries |
| --- | --- | --- | --- |
| 1. Same mass differences, different penalties | Original four-label extension P=(.7,.2,.1,0), Q=(.4,.5,.1,0) | `DivergenceRatioLab`: aligned masses, ratio-to-penalty rows and contribution lengths; choose a generator, apply bounded weight drafts, swap or remove support. | One active normalized state; integer browser weights avoid arbitrary-range arithmetic; positive versus exact-zero support explicit. Independent direct formulas/Decimal. |
| 2. What survives coarsening? | Merge labels with opposite discrepancies versus retain their sign event | `CoarseningFigure`: actual labeled masses flow into merged output bins, before/after TV and KL. | Deterministic same map applied to both laws; arrows denote grouping, widths need not encode mass. Native matrix pushforward and Jensen check. |
| 3. A divergence may fail triangle inequality | Delta-left → balanced midpoint → delta-right | `MetricTriangleFigure`: direct and two-leg costs show JS versus sqrt(JS), and H² versus H. | Exact finite values and explicit units/conventions; native independent triangle examples. |
| 4. The observer class defines what can be seen | P=(.4,.1,.1,.4), Q=(.1,.4,.4,.1) on positions −3,−1,1,3 | `ObservableCriticLab`: per-outcome score heights linked to signed expectation terms; select bounded-event, linear-slope or 1-Lipschitz optimal critic, inspect values and restrictions. | Linear means agree; TV=.6; W1=1.2. Exact event enumeration and independent linear programming. |
| 5. A tiny geometric move need not be a small f-divergence | P=delta0, Q=deltah | `MovingAtomLab`: shared physical axis plus calculated KL/TV/JS/W1/RBF-MMD readouts as h changes, including h=0. | No invented benchmark; analytic point-mass laws. Fixed positive bandwidth. Separate plot axes/units; KL infinity is symbolic, never a finite bar. |
| 6. Feature means detect differences raw means miss | P=±1 equally, Q=0; phi(x)=(x,x²) | `FeatureMeanFigure`: source values mapped to a two-coordinate feature plane, with both mean locations. | A finite explicit feature map, not an RKHS prerequisite leap or a characteristic-kernel claim. |
| 7. Kernel pairs form a distribution witness | Four-versus-four numeric samples, variance-only and shift presets | `KernelWitnessLab`: sample locations, linked signed Gram blocks and unnormalized witness curve; change applied samples/kernel bandwidth, inspect biased/unbiased squared quantities. | Computed finite empirical laws, not a population density estimate. Negative unbiased estimates remain signed. Kernel amplitude/units specified. Independent Gram/feature calculations. |
| 8. A nonzero empirical score is not a rejection rule | Eight pooled observations, all choose(8,4)=70 label allocations | `PermutationMmdLab`: current labeled split on an axis, exact null-statistic distribution and upper-tail rank; step split/reset and vary prespecified bandwidth. | Exchangeability under equal iid laws; ties included, observed allocation included; exact-enumeration p-value, not Monte Carlo plus-one formula. Root hypothesis lesson linked. |
| 9. A fitted critic is a lower bound, not exact divergence | Fixed positive original P,Q; KL conjugate and shifted optimal scores | `VariationalDivergenceLab`: each outcome's linear reward versus convex penalty and total bound/true value; shift/flatten critic, restore optimum. | Uses f(u)=u ln u, f*(t)=exp(t−1); finite exact known-law expectation. Finite-sample fitted objectives lack an automatic population lower-bound guarantee. |

The initial visible states must expose each mechanism without requiring a hidden control state. Reuse controls/tables where they help; do not duplicate static figures already clear in an initial lab. Diagrams use numerical geometry only where meaningful. All charts identify exact calculation versus empirical statistic. Labels remain readable at 320px; larger Gram tables use named keyboard-scroll regions. Native controls, explicit draft application, deterministic reset, no timers or new dependencies.

## Detailed mathematical conventions

- Natural logarithms for KL/JS and Fenchel bounds. JS is the half-sum midpoint convention, bounded by ln2. H² is half the squared L2 distance of square-root masses, bounded by 1; H is its square root. These factors differ from some references and must remain consistent across all tables/code.
- For q=0,p>0, contribution is p lim(t→infinity)f(t)/t; for p=q=0 it is zero; q>0,p=0 uses qf(0). Common measure may be P+Q, so discrete, continuous and singular laws fit one definition.
- Adding c(t−1) to f leaves total divergence unchanged. Centered nonnegative KL generator may be used in the contribution investigation; the Fenchel branch explicitly switches to u ln u and its corresponding conjugate.
- Jensen proves nonnegativity; zero implies equal laws only with appropriate separation/strict-convexity-at-one conditions. Convexity by itself does not turn every generator into a metric.
- Same-channel DPI uses Q-weighted conditional ratios; prove the finite positive-reference case stepwise and explain boundary extension. Invertible recoding preserves f-divergence; lossy preprocessing can conceal a difference. IPM contraction depends on whether the observer class is preserved by the map, not a universal arbitrary-preprocessing claim.
- TV is sup over 0≤g≤1; sup over |g|≤1 is 2TV. Equal source prior and equal error costs yield optimal accuracy (1+TV)/2. For bounded loss [0,1], expectation difference is at most TV.
- W1 dual over 1-Lipschitz functions requires a genuine metric and finite first moments on the stated space; finite 1D derivation is self-contained. Arbitrary category codes do not supply valid physical geometry.
- MMD is the RKHS norm; its square is a different quantity. Kernel mean existence is ensured for bounded Gaussian kernels; characteristic Gaussian kernels on R^d with fixed positive bandwidth separate Borel probability laws. Finite linear/polynomial features can miss different laws. The plotted witness is unnormalized and its RKHS norm is MMD; divide by that norm only if positive to obtain the unit critic.
- U-statistic squared MMD excludes within-sample diagonal pairs, retains all cross pairs, needs at least two independent observations per group and can be negative. The biased empirical-law squared norm is nonnegative mathematically; no absolute-value repair of U-statistic.
- Bandwidth/feature selection must be prespecified, independent of test labels, trained on separate data, or recomputed within every permutation. No small p-value guarantees practical harm; no large p-value proves equality. Repeated/clustered observations require a valid dependence-aware test design.
- Variational f-divergence identity is distinct from a plain IPM because Q sees f*(T), not the same T. GAN JS identity assumes optimal unrestricted population discriminator and equal mixture weights; practical training and non-saturating generator losses do not simply equal minimized exact JS. WGAN critic restrictions and optimization approximation prevent universal gradient/convergence claims.

## Examples, practice and verification plan

Complete standard-library Python programs will preserve the original and cover generator/support conventions, event/classification witnesses, finite W1 critics, MMD pair sums and a negative U-statistic, exact permutation calibration, Fenchel gaps, and selected harmful-change/importance-weight calculations. Each has a visible question, inputs, intermediate outputs, explanation, changed-case exercise and actually executed stdout. NumPy/SciPy are reviewer oracles rather than unexplained learner runtime prerequisites.

Independent practice will change probability tables, null support, log/convention factors, critic constraints, geometry, kernel features, sample allocation and model-selection strategy. Include a proof repair, hand computation, hidden-difference diagnosis and a complete finite test-design task with acceptance criteria. Hints precede explained solutions; no DSA/LeetCode quota applies to this mathematics lesson.

Independent tests: direct known-form f-divergences and high precision near equality; exhaustive finite event maximum and binary classification; random common stochastic maps and Pinsker; SciPy transport/critic LP versus ordered formula; Gram PSD/explicit feature norm and independent sample estimators; exact combinatorial allocations and rank validity; scalar optimization/conjugate inequality versus critic; original and changed native programs. Validate rejected inputs, copied snapshots and finite work bounds.

Actual browser review after source registration: 1440/390/320 with original fonts, complete native program questions, every anchor/practice/hint, changed lab inputs/reset/invalid preservation, keyboard operation, local scroll, SVG and equation fit, requests/errors. Open screenshots of ordinary reading and each substantive mechanism, then freeze exact source hashes. Parent separately owns production build, loading checks and integrated status.

## Initial primary-source ledger

Retrieved 10 September 2026. Claims will be checked again while deriving/implementing, and final record will distinguish actual review from links alone.

| Source and locator | Read / purpose / convention caveat |
| --- | --- |
| [Polyanskiy and Wu, f-divergences notes](https://people.lids.mit.edu/yp/homepage/data/LN_fdiv.pdf), sections7.1–7.4 and7.6 | Definition, singular contribution, linear-generator invariance, common-channel DPI, metric roots and Pinsker reviewed. Their H² and JS omit this lesson's half normalization; convert factors deliberately. |
| [Gretton et al., A Kernel Two-Sample Test](https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf), sections2–3 | Feature mean/witness, biased and unbiased forms, negative squared estimator and test assumptions reviewed; formulas and numerical examples independently derived. |
| [Nowozin et al., f-GAN](https://arxiv.org/pdf/1606.00709), section2 | Fenchel lower bound, critic domains and optimization interpretation reviewed. Derive our table from the stated generator, not copied OCR/table constants. |
| [Goodfellow et al., GAN](https://arxiv.org/pdf/1406.2661), section4.1 | Optimal discriminator and −ln4+2JS derivation reviewed; finite-network convergence is not inherited. |
| [Arjovsky et al., WGAN](https://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf), section2 | Primary starting point for moving singular supports and critic assumptions; section 2 reviewed for definitions, singular-support example and topology; no general gradient guarantee inferred. |
| [Arthur Gretton's teaching page](https://www.gatsby.ucl.ac.uk/~gretton/teaching.html), MLSS2020 second lecture | Creator describes MMD, testing and generative applications and links [the recording](https://www.youtube.com/watch?v=eANiXrWO1dM) plus companion slides. Video fetch failed and slides exceeded tool size; no playback/slide review claimed yet. JMLR supplies the substantive written alternative. |

Not used as a blanket claim: the exact intersection characterization of IPMs and f-divergences, distribution-free rankings of estimator rates, universally beneficial GAN gradients or fairness certification. They are unnecessary to achieve this lesson's outcomes.
