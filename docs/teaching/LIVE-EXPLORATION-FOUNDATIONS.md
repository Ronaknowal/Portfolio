# Foundation live-exploration specifications and review

21 September 2026. This scoped revision applies to 96 already implemented Programming, DSA and Mathematics lessons. It revises how learners inspect mechanisms; it is not a replacement of their verified mathematics, programs, prose or independent practice. The original per-topic design remains the source for domain depth, fixtures, derivations and visual encodings. The interaction requirements below supersede older lab prediction/reveal instructions. K-Means has its separate current visual contract.

## Interaction requirements

No learner-prediction input, optional guess mode, grading, answer prefill or pedagogical reveal gate. Start with useful visible data/results, recompute inexpensive valid edits, and keep diagrams and readouts on the same state. Preserve true process steps and explicit execution of complete input drafts. State the last-valid boundary if an incomplete/invalid draft cannot be calculated. Keep scientific information restrictions and meaningful comparisons. Controls should expose causes and consequences with the topic’s existing representation rather than collapse every investigation into a generic text box.

The five substantive foundation changes are detailed per topic below. Other topics already had live or meaningful step/run mechanisms; their improvements remove passive prediction-first framing or obsolete shared prediction components where used. In-domain mathematical predictions and separate practice remain. This is a bounded interaction review, not a new independent proof of every unchanged numerical model.

## Topic-specific dispositions

### Vectors, Matrices & Tensor Operations

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Coordinate projection, basis-image transformation, linked row-column contraction and tensor source-cell reduction** — Which quantities contribute to this output, and what relationship survives the operation? Change coordinates or a map, trace one product cell, and select a reduction axis to inspect its exact source values.

Current lesson: [vectors-matrices-tensor-operations](../../src/learn/data/topics/vectors-matrices-tensor-operations.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/vectors-matrices-tensor-operations.js). Previous review: [retained record](../../docs/teaching/VECTORS-MATRICES-TENSORS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/VECTORS-MATRICES-TENSORS-DESIGN.md).

### Matrix Decompositions (SVD, QR, Cholesky, LU)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **linked factor operations with domain-specific geometric investigations** — What does each factor change, and which property makes the resulting task easier? Step pivoted equations; remove a projected column component; vary correlation; follow a vector through analytically constructed singular factors and truncate a direction.

Current lesson: [matrix-decompositions-svd-qr-cholesky-lu](../../src/learn/data/topics/matrix-decompositions-svd-qr-cholesky-lu.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/matrix-decompositions-svd-qr-cholesky-lu.js). Previous review: [retained record](../../docs/teaching/MATRIX-DECOMPOSITIONS-VERIFICATION.md).

### Eigenvalues & Eigenvectors

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Equal-scale direction geometry, raw recurrence trajectories and projected point clouds** — Which part of a change stays along a direction, and what does its signed scale mean in this model? Turn a unit input across six maps; compare six repeated-update rules and three initial vectors; project three small datasets while calculating variance and squared error.

Current lesson: [eigenvalues-eigenvectors](../../src/learn/data/topics/eigenvalues-eigenvectors.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/eigenvalues-eigenvectors.js). Previous review: [retained record](../../docs/teaching/EIGENVALUES-EIGENVECTORS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/EIGENVALUES-EIGENVECTORS-DESIGN.md).

### Matrix Calculus & Jacobians

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Linked nonlinear/tangent curves, directed forward/reverse traces, parameter contribution matrices and computed finite-difference errors** — Which local input changes or sensitivity contributions determine this result? Change the base point and direction, propagate derivative stages, select a shared parameter and compare objective reductions or numerical step sizes.

Current lesson: [matrix-calculus-jacobians](../../src/learn/data/topics/matrix-calculus-jacobians.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/matrix-calculus-jacobians.js). Previous review: [retained record](../../docs/teaching/MATRIX-CALCULUS-JACOBIANS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/MATRIX-CALCULUS-JACOBIANS-DESIGN.md).

### Tensor Algebra & Einsum Notation

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Selectable contraction terms, query-to-key contribution rows, two contraction trees and a linked basis/coordinate diagram** — Which input entries determine this output, and what changes when labels, execution order or basis change? Edit a bounded explicit expression, inspect a selected output, change attention masks, compare matrix dimensions, and change the coordinate basis while checking invariant quantities.

Current lesson: [tensor-algebra-einsum-notation](../../src/learn/data/topics/tensor-algebra-einsum-notation.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/tensor-algebra-einsum-notation.js). Previous review: [retained record](../../docs/teaching/TENSOR-ALGEBRA-EINSUM-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/TENSOR-ALGEBRA-EINSUM-DESIGN.md).

### Randomized Linear Algebra

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Weighted-vector geometry, singular-value/error budgets, influential observation selection and actual running trace estimates** — What did the sketch preserve, what did it miss, and does the smaller answer satisfy the original objective? Choose three column weights; vary spectrum/rank/oversampling/iterations/seed on bounded six-dimensional matrices; select observations and compare fitted lines; advance independent-sign trace probes.

Current lesson: [randomized-linear-algebra](../../src/learn/data/topics/randomized-linear-algebra.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/randomized-linear-algebra.js). Previous review: [retained record](../../docs/teaching/RANDOMIZED-LINEAR-ALGEBRA-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/RANDOMIZED-LINEAR-ALGEBRA-DESIGN.md).

### Multivariate Calculus & Gradients

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Contour and tangent-slice explorer, approach-path comparison, constrained-circle motion, curvature slices and discrete descent trajectory** — Which local changes are allowed, what does the derivative predict, and when does that prediction justify a conclusion? Hold a base point fixed while changing direction and distance; compare line and curved approaches; move around a constraint; rotate a curvature slice; advance or rewind a gradient update.

Current lesson: [multivariate-calculus-gradients](../../src/learn/data/topics/multivariate-calculus-gradients.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/multivariate-calculus-gradients.js). Previous review: [retained record](../../docs/teaching/MULTIVARIATE-CALCULUS-GRADIENTS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/MULTIVARIATE-CALCULUS-GRADIENTS-DESIGN.md).

### Convex Optimization

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Feasible mixing, actual function chords, constrained coefficient geometry, curvature contours and soft-threshold objectives** — Is this choice allowed, why can no allowed choice improve it, and how does the computed result differ from an exact certificate? Vary chord endpoints, budgets/candidates, ridge penalties and bounded gradient steps, and a signed nonsmooth threshold problem.

Current lesson: [convex-optimization](../../src/learn/data/topics/convex-optimization.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/convex-optimization.js). Previous review: [retained record](../../docs/teaching/CONVEX-OPTIMIZATION-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/CONVEX-OPTIMIZATION-DESIGN.md).

### Gradient Descent Variants (SGD, Adam, AdaGrad, RMSProp, LAMB, LARS)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Batch-slope comparison, quadratic momentum trajectory, gradient-history arithmetic, decay paths and block-relative update geometry** — Which information changes this update, and does its direction, magnitude and stored state match the stated rule? Select observations; change a momentum rule and advance it; replay a sparse or alternating gradient history; compare coupled and decoupled decay; alter block norms and inspect trust ratios.

Current lesson: [gradient-descent-variants-sgd-adam-adagrad-rmsprop-lamb-lars](../../src/learn/data/topics/gradient-descent-variants-sgd-adam-adagrad-rmsprop-lamb-lars.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/gradient-descent-variants-sgd-adam-adagrad-rmsprop-lamb-lars.js). Previous review: [retained record](../../docs/teaching/GRADIENT-DESCENT-VARIANTS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/GRADIENT-DESCENT-VARIANTS-DESIGN.md).

### Learning Rate Schedules (Cosine, Warmup, OneCycleLR)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Discrete phase curves, exact scalar noise moments, event lanes and validation-state transitions** — Which rate is actually consumed, why did it change, and what consequence follows under the stated model? Inspect bounded schedule points, alter curvature/noise, step through accumulated/skipped updates and validate threshold-triggered changes.

Current lesson: [learning-rate-schedules-cosine-warmup-onecyclelr](../../src/learn/data/topics/learning-rate-schedules-cosine-warmup-onecyclelr.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/learning-rate-schedules-cosine-warmup-onecyclelr.js). Previous review: [retained record](../../docs/teaching/LEARNING-RATE-SCHEDULES-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/LEARNING-RATE-SCHEDULES-DESIGN.md).

### Convex Duality & Lagrangian Methods (KKT Conditions)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Feasible half-space and exact bound geometry, condition diagnosis, sensitivity curves and separable allocation price traces** — What is this number a bound on, which hypothesis makes it valid, and what does changing the constraint actually change? Change a budget, candidate and multiplier; diagnose KKT conditions independently; compare actual value changes with supporting prices; step a price update and inspect local allocation and feasibility.

Current lesson: [convex-duality-lagrangian-methods-kkt-conditions](../../src/learn/data/topics/convex-duality-lagrangian-methods-kkt-conditions.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/convex-duality-lagrangian-methods-kkt-conditions.js). Previous review: [retained record](../../docs/teaching/CONVEX-DUALITY-KKT-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/CONVEX-DUALITY-KKT-DESIGN.md).

### Second-Order Methods (L-BFGS, K-FAC, Shampoo, Natural Gradient)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Calculated contour geometry, directional models, history transformations, probability bars and matrix-factor correspondence** — Which information changes the direction, which approximation was made, and what can the resulting step actually guarantee? Inspect bounded geometric trajectories, step through retained secant pairs, compare coordinate updates and calculate factor-based transforms.

Current lesson: [second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient](../../src/learn/data/topics/second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient.js). Previous review: [retained record](../../docs/teaching/SECOND-ORDER-METHODS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/SECOND-ORDER-METHODS-DESIGN.md).

### Non-Convex Optimization Landscape

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Calculated well curves, signed-value maps and directional slices, saddle trajectories, symmetry perturbations and same-input inline figures** — What conclusion does this visible geometry actually support, and what could it hide? Change a well tilt and initial point, inspect stationary slices, direct sample noise, and rescale an equivalent predictor while comparing exact curvature and outputs.

Current lesson: [non-convex-optimization-landscape](../../src/learn/data/topics/non-convex-optimization-landscape.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/non-convex-optimization-landscape.js). Previous review: [retained record](../../docs/teaching/NONCONVEX-LANDSCAPE-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/NONCONVEX-LANDSCAPE-DESIGN.md).

### Constrained & Multi-Objective Optimization

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Coupled projection geometry, penalty/domain curves, two-copy agreement traces, discrete Pareto scatter and continuous decision-to-objective mapping** — Is the proposed answer allowed, what preference chose it, and which exact measurement or convergence claim supports it? Change a target and budget, step separate projections and consensus updates, vary finite penalty/barrier strength, filter candidate budgets and select a scalarization.

Current lesson: [constrained-multi-objective-optimization](../../src/learn/data/topics/constrained-multi-objective-optimization.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/constrained-multi-objective-optimization.js). Previous review: [retained record](../../docs/teaching/CONSTRAINED-MULTIOBJECTIVE-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/CONSTRAINED-MULTIOBJECTIVE-DESIGN.md).

### Probability Distributions & Bayes' Theorem

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Event maps, proportionate joint populations, evidence branches, urn/PMF correspondence, linked density/CDF and event/wait timelines** — Which outcomes count toward this event, under which conditioning population, and how does the representation encode their probability? Change base rates and evidence dependence, inspect finite sampling with or without replacement, and compare density area/CDF mass under a unit change.

Current lesson: [probability-distributions-bayes-theorem](../../src/learn/data/topics/probability-distributions-bayes-theorem.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/probability-distributions-bayes-theorem.js). Previous review: [retained record](../../docs/teaching/PROBABILITY-DISTRIBUTIONS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/PROBABILITY-DISTRIBUTIONS-BAYES-DESIGN.md).

### Maximum Likelihood & MAP Estimation

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Observed binary strip and parameter-likelihood curve, measurement residual geometry, prior/posterior densities, exact estimator-mass bars and transformed-mode correspondence** — What is being held fixed, what is being varied, and what justifies calling this parameter value a useful estimate? Change bounded observations, parameter candidates, prior strengths and sampling assumptions; inspect linked likelihood, fit, posterior and repeated-estimate views.

Current lesson: [maximum-likelihood-map-estimation](../../src/learn/data/topics/maximum-likelihood-map-estimation.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/maximum-likelihood-map-estimation.js). Previous review: [retained record](../../docs/teaching/MAXIMUM-LIKELIHOOD-MAP-VERIFICATION.md).

### Hypothesis Testing & Confidence Intervals

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Preserved moving-interval coverage mechanism, raw pair-to-difference view, null and alternative sampling distributions, effect/threshold number line, exact finite resampling distribution and sampling-unit comparison** — What varies across samples, what remains fixed, and which decision does the observed evidence actually justify? Change sample size/confidence, inspect a paired dataset, move the null reference and alternative, compare practical thresholds and planned test power; expose exact states rather than a single unexplained score.

Current lesson: [hypothesis-testing-confidence-intervals](../../src/learn/data/topics/hypothesis-testing-confidence-intervals.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/hypothesis-testing-confidence-intervals.js). Previous review: [retained record](../../docs/teaching/HYPOTHESIS-TESTING-CONFIDENCE-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/HYPOTHESIS-TESTING-CONFIDENCE-DESIGN.md).

### Bayesian Inference & Conjugate Priors

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Prior/posterior density, evidence lanes, predictive count mass, exposure strips, precision intervals and replicated sequence patterns** — Which quantity is uncertain, what observation supplies new information, and what consequences survive averaging over that uncertainty? Change meaningful prior/data/measurement assumptions, compare parameter and outcome distributions, and challenge a common-rate model with the same totals but a different pattern.

Current lesson: [bayesian-inference-conjugate-priors](../../src/learn/data/topics/bayesian-inference-conjugate-priors.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/bayesian-inference-conjugate-priors.js). Previous review: [retained record](../../docs/teaching/BAYESIAN-INFERENCE-CONJUGATE-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/BAYESIAN-INFERENCE-CONJUGATE-DESIGN.md).

### Concentration Inequalities (Hoeffding, Bernstein, Chernoff)

Retain the already playable mechanism and its topic-specific controls; no interaction replacement is needed.

- **Exact discrete tails, exponential domination/objective curves, precision/variance radius curves, contrasted sampling laws and family-error budgets** — Which event is bounded, how does the bound arise, and what assumption changes its validity? Change finite thresholds or λ, compare actual tail mass with theorem guarantees, vary sample/variance/error budgets and contrast dependence/selection structures.

Current lesson: [concentration-inequalities-hoeffding-bernstein-chernoff](../../src/learn/data/topics/concentration-inequalities-hoeffding-bernstein-chernoff.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/concentration-inequalities-hoeffding-bernstein-chernoff.js). Previous review: [retained record](../../docs/teaching/CONCENTRATION-INEQUALITIES-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/CONCENTRATION-INEQUALITIES-LESSON-DESIGN.md).

### Monte Carlo Methods & MCMC (Metropolis-Hastings, HMC, NUTS)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Contribution averages, asymmetric state-flow graph, rejection-aware trace, Hamiltonian phase-space path, NUTS doubling/candidate tree and exact dependent-sample variance** — Does this transition preserve the intended target, and how much reliable numerical information does its finite output provide? Change bounded seeds, targets, proposal/integrator scales and correlation; step actual proposals and trajectory expansions; compare state, acceptance, candidate membership and estimator error under one shared model.

Current lesson: [monte-carlo-methods-mcmc-metropolis-hastings-hmc-nuts](../../src/learn/data/topics/monte-carlo-methods-mcmc-metropolis-hastings-hmc-nuts.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/monte-carlo-methods-mcmc-metropolis-hastings-hmc-nuts.js). Previous review: [retained record](../../docs/teaching/MONTE-CARLO-MCMC-VERIFICATION.md).

### Variational Inference

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Linked posterior mass and ELBO balance, covariance ellipses, coordinate-update path, two-mode density projection, noise-to-gradient contribution map and shared-inference diagram** — What does this approximation preserve, what does the objective reward, and which error remains after optimization? Change a bounded approximation, correlation, factor update or seeded gradient budget; inspect linked probability, geometry, objective and decision readouts from one calculated model.

Current lesson: [variational-inference](../../src/learn/data/topics/variational-inference.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/variational-inference.js). Previous review: [retained record](../../docs/teaching/VARIATIONAL-INFERENCE-VERIFICATION.md).

### Exponential Families & Sufficient Statistics

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Dataset fibers, normalized finite masses, moment triangle and paired coordinate densities** — What survives the summary, what changes when the model changes, and why do the same parameters have different density pictures? Edit allocations under two models, tilt finite weights, fit a mean target and compare a prior in probability versus log-odds coordinates.

Current lesson: [exponential-families-sufficient-statistics](../../src/learn/data/topics/exponential-families-sufficient-statistics.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/exponential-families-sufficient-statistics.js). Previous review: [retained record](../../docs/teaching/EXPONENTIAL-FAMILIES-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/EXPONENTIAL-FAMILIES-LESSON-DESIGN.md).

### Measure Theory & Probability Spaces

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Finite event partitions, preimage mapping, simple-function bands, shrinking-support curves, transformed probability cells and conditional prediction residuals** — Which sets can the information distinguish, where does their probability go, and which operation preserves the integral? Select an event and observation partition, refine simple functions, compare convergent functions with their integrals, transform a selected joint cell and refine a conditional prediction.

Current lesson: [measure-theory-probability-spaces](../../src/learn/data/topics/measure-theory-probability-spaces.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/measure-theory-probability-spaces.js). Previous review: [retained record](../../docs/teaching/MEASURE-THEORY-PROBABILITY-SPACES-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/MEASURE-THEORY-PROBABILITY-SPACES-DESIGN.md).

### Optimal Transport (Wasserstein Distance, Sinkhorn)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Mass-flow links with a conservation matrix, dual slack certificate, cumulative-distribution areas, actual alternating matrix corrections and split-mass versus conditional-mean diagrams** — Which mass moves, what makes that plan feasible and cheapest, and what changes when we smooth or summarize it? Change bounded source/target weights, feasible coupling, cumulative comparison, row/column step or entropy scale; inspect actual mass, cost, slack and residual values.

Current lesson: [optimal-transport-wasserstein-distance-sinkhorn](../../src/learn/data/topics/optimal-transport-wasserstein-distance-sinkhorn.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/optimal-transport-wasserstein-distance-sinkhorn.js). Previous review: [retained record](../../docs/teaching/OPTIMAL-TRANSPORT-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/OPTIMAL-TRANSPORT-LESSON-DESIGN.md).

### Causal Inference & Do-Calculus

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Mechanism surgery, active paths, population weights, rule-specific graphs and paired counterfactual worlds** — Which part of the data-generating mechanism changes, and what information supports the resulting causal query? Condition on graph nodes, modify assignment, inspect do-calculus graph cuts, break a frontdoor assumption and retain the same hidden unit across possible outcomes.

Current lesson: [causal-inference-do-calculus](../../src/learn/data/topics/causal-inference-do-calculus.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/causal-inference-do-calculus.js). Previous review: [retained record](../../docs/teaching/CAUSAL-INFERENCE-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/CAUSAL-INFERENCE-LESSON-DESIGN.md).

### Entropy, Cross-Entropy & KL Divergence

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Binary address tree, codeword decoder, weighted surprise strips, signed mismatch contributions, conditional probability mosaic, density scaling and a feasible entropy slice** — Which distribution supplies the outcomes, which supplies their price, and which changes preserve the probability or constraint? Change a source/model, decode a message, inspect conditional prediction, change coordinates and move along a mean-preserving probability slice.

Current lesson: [entropy-cross-entropy-kl-divergence](../../src/learn/data/topics/entropy-cross-entropy-kl-divergence.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/entropy-cross-entropy-kl-divergence.js). Previous review: [retained record](../../docs/teaching/ENTROPY-CROSS-ENTROPY-KL-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/ENTROPY-CROSS-ENTROPY-KL-DESIGN.md).

### Mutual Information & Information Bottleneck

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Joint-versus-product probability cells, conditional label bars, XOR reveal table, information plane, soft-assignment updates, variational bound gaps and sampled count grids** — Which information was present, retained or lost, and which number is an exact property rather than a bound or estimate? Change bounded channel probabilities, representation choices, bottleneck weights, finite update steps or a declared sampled count budget; inspect linked distributions and information quantities from one coherent state.

Current lesson: [mutual-information-information-bottleneck](../../src/learn/data/topics/mutual-information-information-bottleneck.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/mutual-information-information-bottleneck.js). Previous review: [retained record](../../docs/teaching/MUTUAL-INFORMATION-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/MUTUAL-INFORMATION-LESSON-DESIGN.md).

### Rate-Distortion Theory

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Bit-string codebook and reconstruction ledger, calculated frontier, evolving conditional-probability matrix and Gaussian variance reservoirs** — Which information crosses the bitstream, which error is permitted, and how close is this actual method to a theoretical limit? Change source bias, codebook, error budget, optimization step and distortion allocation; inspect actual reconstructed symbols, probability sums, bounds and units.

Current lesson: [rate-distortion-theory](../../src/learn/data/topics/rate-distortion-theory.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/rate-distortion-theory.js). Previous review: [retained record](../../docs/teaching/RATE-DISTORTION-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/RATE-DISTORTION-LESSON-DESIGN.md).

### f-Divergences & Integral Probability Metrics

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Aligned mass-and-penalty rows, coarsening flows, constrained critic scores, moving atoms, feature-mean geometry, kernel Gram blocks and witnesses, and exact permutation ranks** — What mismatch can this comparison see, what does its numerical value mean, and what additional evidence is needed to make a decision? Apply bounded distribution or sample edits, choose an observer or kernel, move a point mass, inspect a finite relabeling and compare a critic bound with its exact known-law target.

Current lesson: [f-divergences-integral-probability-metrics](../../src/learn/data/topics/f-divergences-integral-probability-metrics.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/f-divergences-integral-probability-metrics.js). Previous review: [retained record](../../docs/teaching/F-DIVERGENCES-IPMS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/F-DIVERGENCES-IPMS-LESSON-DESIGN.md).

### Graph Fundamentals (Adjacency, Laplacian, Connectivity)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Linked network/matrix cells, edge-current accounting, normalization comparison and node trajectories** — Which relationship does each matrix entry or operation represent, and which graph assumptions make its conclusion valid? Change an edge or its direction, inspect a walk contribution, alter node values, compare normalized operators, advance an averaging step and remove an interpolation anchor.

Current lesson: [graph-fundamentals-adjacency-laplacian-connectivity](../../src/learn/data/topics/graph-fundamentals-adjacency-laplacian-connectivity.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/graph-fundamentals-adjacency-laplacian-connectivity.js). Previous review: [retained record](../../docs/teaching/GRAPH-FUNDAMENTALS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/GRAPH-FUNDAMENTALS-LESSON-DESIGN.md).

### Spectral Graph Theory

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Preserved bridge-spectrum and local-row investigations, cut-threshold geometry, normalized embedding coordinates, mode-gain signal reconstruction and circuit correspondence** — Which weighted disagreements does this mode penalize, and what changes when we turn continuous coordinates into a partition or a filtered signal? Change a bridge and mode, sweep an actual node threshold, trace row-coordinate clustering and compare spectral filter gains on the same graph.

Current lesson: [spectral-graph-theory](../../src/learn/data/topics/spectral-graph-theory.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/spectral-graph-theory.js). Previous review: [retained record](../../docs/teaching/SPECTRAL-GRAPH-THEORY-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/SPECTRAL-GRAPH-THEORY-DESIGN.md).

### Combinatorial Optimization & Approximation Algorithms

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Feasible assignment edges, exchange witnesses, branch bounds, incidence-grid charges, dual loads and value-rounding frontiers** — What legal decision did the algorithm produce, and what independent argument bounds how much better any decision could be? Change costs or constraints, inspect an exchange or residual reversal, advance a bounded search, track element charges and vertex loads, and alter epsilon while checking true reconstructed value.

Current lesson: [combinatorial-optimization-approximation-algorithms](../../src/learn/data/topics/combinatorial-optimization-approximation-algorithms.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/combinatorial-optimization-approximation-algorithms.js). Previous review: [retained record](../../docs/teaching/COMBINATORIAL-OPTIMIZATION-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/COMBINATORIAL-OPTIMIZATION-LESSON-DESIGN.md).

### Stochastic Processes (Markov Chains, Brownian Motion, Poisson)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Path-law strips, linked transition-mass flows, first-hit survival lanes, arrival/count timelines, holding-time clocks and coupled Brownian paths** — What relationship across time does this model assert, and what can a finite observation or simulation actually establish? Apply bounded transition, intensity or drift/scale changes; inspect consistent exact laws and repeatable trajectories; advance first-hit mass, route arrivals and reveal a finer observation grid without changing existing sampled points.

Current lesson: [stochastic-processes-markov-chains-brownian-motion-poisson](../../src/learn/data/topics/stochastic-processes-markov-chains-brownian-motion-poisson.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/stochastic-processes-markov-chains-brownian-motion-poisson.js). Previous review: [retained record](../../docs/teaching/STOCHASTIC-PROCESSES-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/STOCHASTIC-PROCESSES-LESSON-DESIGN.md).

### Random Matrix Theory

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Counted data-to-Gram correspondence, finite spectral mass with separate zero atom, repeated maxima, spike alignment and symmetric two-level geometry** — Which features of this spectrum come from the specified random model, and what conclusion does this particular finite observation actually support? Change dimensions or spike strength with a fixed sample seed, draw a new matrix deliberately, inspect eigenvalue bins and zero mass, compare a stated finite reference, and alter symmetric coupling to see the exact level gap.

Current lesson: [random-matrix-theory](../../src/learn/data/topics/random-matrix-theory.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/random-matrix-theory.js). Previous review: [retained record](../../docs/teaching/RANDOM-MATRIX-THEORY-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/RANDOM-MATRIX-THEORY-DESIGN.md).

### Queueing Theory (M/M/1, M/G/1, Little's Law)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Linked job/service timelines and occupancy areas, birth/death probability flow, waiting-time survival and residual-service geometry** — Which work is still present at this time, and why can equal mean capacity produce very different waits? Change the observation horizon, trace actual service intervals, compare stable load and tail metrics, and inspect same-mean service mixtures and admitted throughput.

Current lesson: [queueing-theory-m-m-1-m-g-1-little-s-law](../../src/learn/data/topics/queueing-theory-m-m-1-m-g-1-littles-law.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/queueing-theory-m-m-1-m-g-1-little-s-law.js). Previous review: [retained record](../../docs/teaching/QUEUEING-THEORY-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/QUEUEING-THEORY-LESSON-DESIGN.md).

### Dynamical Systems Theory & Chaos

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Phase lines and potentials, linked time/phase views, cobweb iteration, finite bifurcation atlas, sensitivity and invariant comparisons** — Which feature belongs to the rule, the starting state, a numerical method, or the evidence we actually collected? Change one meaningful parameter or initial state, trace the same state across views, inspect multipliers and errors, and compare exact or declared numerical evolution.

Current lesson: [dynamical-systems-theory-chaos](../../src/learn/data/topics/dynamical-systems-theory-chaos.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/dynamical-systems-theory-chaos.js). Previous review: [retained record](../../docs/teaching/DYNAMICAL-SYSTEMS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/DYNAMICAL-SYSTEMS-LESSON-DESIGN.md).

### Itô Calculus & Stochastic Differential Equations

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Information-aware increment traces, curvature diagrams, analytic growth laws, mean-reversion probability views and coupled numerical paths** — Which accumulated effect survives, which information was available, and which error does this comparison actually measure? Inspect signed interval contributions; vary model parameters with fixed randomness; group one fine Brownian grid into coarse increments; compare model-derived moments, exact solutions and explicit numerical updates.

Current lesson: [it-calculus-stochastic-differential-equations](../../src/learn/data/topics/ito-calculus-stochastic-differential-equations.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/it-calculus-stochastic-differential-equations.js). Previous review: [retained record](../../docs/teaching/ITO-CALCULUS-SDE-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/ITO-CALCULUS-SDE-LESSON-DESIGN.md).

### Numerical Methods (Finite Differences, Quadrature, Root Finding)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Aligned rate/area/threshold geometry, nested root intervals, tangent proposals, derivative stencils and error curves, quadrature shapes and adaptive interval partitions** — Which part of this answer is known from the mathematics, which part is approximated, and what evidence supports the requested accuracy? Retain a root interval, accept or reject an actual tangent proposal, change derivative spacing, refine sampled area and allocate an integral budget while watching a threshold decision become resolved or remain uncertain.

Current lesson: [numerical-methods-finite-differences-quadrature-root-finding](../../src/learn/data/topics/numerical-methods-finite-differences-quadrature-root-finding.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/numerical-methods-finite-differences-quadrature-root-finding.js). Previous review: [retained record](../../docs/teaching/NUMERICAL-METHODS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/NUMERICAL-METHODS-LESSON-DESIGN.md).

### Functional Analysis & RKHS

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Function/error regions, derivative-to-evaluation overlap, kernel feature contributions and observed-versus-invisible curve components** — Which distance controls this measurement, and what can change between observations without the training loss noticing? Narrow a spike, integrate editable slope intervals to a query, expose a negative Gram quadratic form, and add an unseen wiggle before fitting and interpreting a kernel model.

Current lesson: [functional-analysis-rkhs](../../src/learn/data/topics/functional-analysis-rkhs.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/functional-analysis-rkhs.js). Previous review: [retained record](../../docs/teaching/FUNCTIONAL-ANALYSIS-RKHS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/FUNCTIONAL-ANALYSIS-RKHS-LESSON-DESIGN.md).

### Topology & Topological Data Analysis (TDA)

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Equivalence maps, selected chains and fillings, linked complexes/barcodes, boundary-column cancellation, diagram matching, pixel filtrations and Mapper memberships** — Which object or class is changing, which operation made it change, and what does the resulting summary actually establish? Add a face, advance a filtration event, cancel a boundary pivot, choose a diagram match, change a pixel threshold and trace an observation through overlapping clusters.

Current lesson: [topology-topological-data-analysis-tda](../../src/learn/data/topics/topology-topological-data-analysis-tda.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/topology-topological-data-analysis-tda.js). Previous review: [retained record](../../docs/teaching/TOPOLOGY-TDA-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/TOPOLOGY-TDA-LESSON-DESIGN.md).

### Category Theory (Emerging Use in ML)

Retain the already playable mechanism and its topic-specific controls; no interaction replacement is needed.

- **Typed element paths, schema triangles, naturality squares, unique-mediator grids, stochastic wire/joint views, primal/tangent lanes and image/preimage maps** — Which paths must agree, and what information must each arrow carry for that agreement to survive composition? Trace an input, break and repair a schema equation, compare naturality routes, expose missing or duplicate mediators, change a coin probability and inspect derivatives at their actual evaluation points.

Current lesson: [category-theory-emerging-use-in-ml](../../src/learn/data/topics/category-theory-emerging-use-in-ml.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/category-theory-emerging-use-in-ml.js). Previous review: [retained record](../../docs/teaching/CATEGORY-THEORY-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/CATEGORY-THEORY-LESSON-DESIGN.md).

### Differential Geometry & Riemannian Manifolds

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Overlapping circle charts, linked basis/metric ellipses, sphere arcs and tangent updates, polar moving frames and actual transported arrows** — Which object changes when I relabel the coordinates, change the metric, or move to a different point? Cross a chart seam, alter a basis and a physical cost separately, compare sphere maps, inspect connection cancellation and transport a tangent arrow around a chosen loop.

Current lesson: [differential-geometry-riemannian-manifolds](../../src/learn/data/topics/differential-geometry-riemannian-manifolds.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/differential-geometry-riemannian-manifolds.js). Previous review: [retained record](../../docs/teaching/DIFFERENTIAL-GEOMETRY-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/DIFFERENTIAL-GEOMETRY-LESSON-DESIGN.md).

### Algebra, Functions, Exponentials & Logarithms

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Paired equation steps, linked graph probes, ordered function machines, quadratic root geometry and additive-versus-multiplicative scales** — Which operation preserves the question, and why does multiplication become addition on a logarithmic ruler? Step through both sides of an equation, change a function input or inverse branch, move a parabola, and inspect growth on linear and logarithmic axes.

Current lesson: [algebra-functions-exponentials-logarithms](../../src/learn/data/topics/algebra-functions-exponentials-logarithms.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/algebra-functions-exponentials-logarithms.js). Previous review: [retained record](../../docs/teaching/ALGEBRA-FUNCTIONS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/ALGEBRA-FUNCTIONS-LESSON-DESIGN.md).

### Sets, Logic, Relations & Proof Techniques

Toggling the proposed subset matrix immediately rebuilds the diagonal subset and row-specific disagreement witness. Neither construction nor witness is hidden behind a reveal button. Reset restores the original membership matrix. Other truth, relation and proof controls retain their distinct representations.

- **Set regions, truth-assignment countermodels, quantified witness boards, proof scopes, relation/class and order views, growing square borders and diagonal subset construction** — Which exact object, witness, counterexample or valid deduction makes this claim true or false? Change finite memberships and relation pairs, compare witness dependencies, expose an admitted countermodel, inspect relation failures, grow an induction step and construct a subset differing from every proposed row.

Current lesson: [sets-logic-relations-proof-techniques](../../src/learn/data/topics/sets-logic-relations-proof-techniques.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/sets-logic-relations-proof-techniques.js). Previous review: [retained record](../../docs/teaching/SETS-LOGIC-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/SETS-LOGIC-LESSON-DESIGN.md).

### Geometry, Trigonometry & Coordinate Reasoning

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Area dissection, shared-scale similar triangles, radius-normalized arcs, linked circle/component traces and physical-point/reference-frame diagrams** — What changes when the angle, scale or frame changes, and which geometric quantity must remain the same? Scale a triangle, change a circle radius and sweep, recover quadrant-aware bearings, and compare moving a point with rotating/translating its axes.

Current lesson: [geometry-trigonometry-coordinate-reasoning](../../src/learn/data/topics/geometry-trigonometry-coordinate-reasoning.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/geometry-trigonometry-coordinate-reasoning.js). Previous review: [retained record](../../docs/teaching/GEOMETRY-TRIGONOMETRY-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/GEOMETRY-TRIGONOMETRY-LESSON-DESIGN.md).

### Counting, Combinatorics & Mathematical Induction

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Choice trees and fiber groups; movable allocation stars and bars; per-object inclusion–exclusion contributions; induction dependency chains; balanced paths and reflection; coefficient construction; cyclic orbit and fixed-pattern views** — Which actual outcomes or descriptions does this count, what makes the cases disjoint or equally repeated, and why does the proof cover every claimed case? Change outcome identity and repetition, move bounded resources, inspect an overlap correction, trace a constructive induction witness, reflect and restore a bad path, build a coefficient and rotate a pattern to inspect its stabilizer.

Current lesson: [counting-combinatorics-mathematical-induction](../../src/learn/data/topics/counting-combinatorics-mathematical-induction.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/counting-combinatorics-mathematical-induction.js). Previous review: [retained record](../../docs/teaching/COUNTING-COMBINATORICS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/COUNTING-COMBINATORICS-LESSON-DESIGN.md).

### Single-Variable Calculus: Limits, Derivatives & Integrals

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Linked motion/secant views, epsilon–delta bands, product increments, composition sensitivities, extrema candidates, signed rectangles, accumulation strips, growth intervals and approximation/tail comparisons** — What does a local rate reveal, what must be accumulated, and which limit or error guarantee makes the conclusion valid? Change a nearby observation, requested accuracy, composition input, interval endpoint, partition, growth period or approximation degree and inspect the corresponding geometry, exact quantities and limitations.

Current lesson: [single-variable-calculus-limits-derivatives-integrals](../../src/learn/data/topics/single-variable-calculus-limits-derivatives-integrals.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/single-variable-calculus-limits-derivatives-integrals.js). Previous review: [retained record](../../docs/teaching/SINGLE-VARIABLE-CALCULUS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/SINGLE-VARIABLE-CALCULUS-LESSON-DESIGN.md).

### Random Variables, Expectation & Covariance

Retain the already playable mechanism and its topic-specific controls; no interaction replacement is needed.

- **Outcome-to-value grouping, PMF/CDF correspondence, balance and squared-loss geometry, joint/marginal grids, signed covariance products, shared-noise lanes, conditional residuals and transformed-interval preimages** — Which uncertainty is individual, which is shared, and what changes when outcomes are paired, transformed, averaged or observed through extra information? Change a probability law, compare exact joint pairings, inspect centering, cancel or retain common noise, reveal a group and compare independent versus repeated copies without substituting simulated evidence for the theoretical law.

Current lesson: [random-variables-expectation-covariance](../../src/learn/data/topics/random-variables-expectation-covariance.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/random-variables-expectation-covariance.js). Previous review: [retained record](../../docs/teaching/RANDOM-VARIABLES-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/RANDOM-VARIABLES-LESSON-DESIGN.md).

### Sampling, Measurement & Experimental Design

The interaction contrast directly changes the fourth synthetic factorial mean, conditional effects and average effect. The B-mixture slider changes the averaging weights, not the four cell means. Retain the separate observed-versus-potential-outcome distinction in assignment examples: it explains scientific information limits, not a prediction quiz.

- **Population/frame maps, exact sample-mean dot distributions, inclusion-contribution ledgers, grouped measurement diagrams and assignment/factorial contrasts** — Which source of error changes when we collect more data, and which requires a different collection design? Compare complete versus incomplete frames, enumerate permitted samples and assignments, vary independent units separately from repeats, and change the interaction contrast and mixture while inspecting conditional and average factor effects live.

Current lesson: [sampling-measurement-experimental-design](../../src/learn/data/topics/sampling-measurement-experimental-design.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/sampling-measurement-experimental-design.js). Previous review: [retained record](../../docs/teaching/SAMPLING-MEASUREMENT-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/SAMPLING-MEASUREMENT-LESSON-DESIGN.md).

### Ordinary Differential Equations & Linear Systems

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Rate-field geometry and balance diagrams; linked position, velocity and phase traces; evolving basis columns; input-contribution timelines; ordered transformation stages; exact-reference numerical and event investigations** — Which state and input determine this trajectory, how is the solution constructed, and which observed behavior belongs to the equation rather than the approximation? Change initial data separately from the law, inspect exact slope and solution values, compare damping and coupled modes, propagate one input interval, swap noncommuting stages, and diagnose a changed step or solver outcome while keeping physical time fixed.

Current lesson: [ordinary-differential-equations-linear-systems](../../src/learn/data/topics/ordinary-differential-equations-linear-systems.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/ordinary-differential-equations-linear-systems.js). Previous review: [retained record](../../docs/teaching/ORDINARY-DIFFERENTIAL-EQUATIONS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/ORDINARY-DIFFERENTIAL-EQUATIONS-LESSON-DESIGN.md).

### Complex Numbers, Fourier & Laplace Transforms

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Complex-plane operations; aligned harmonic products and integrals; jump-neighborhood partial sums; finite root-of-unity contributions; shared alias samples; observation-window and spectrum views; convolution boundaries; response decomposition; Laplace convergence half-planes** — Which representation preserves the information this task needs, and which apparent frequency, phase or system behavior comes from the chosen observation or mathematical convention? Change complex coordinates, relative phase, selected frequency, sample rate, actual record length, padding and initial state separately; inspect linked intermediate values, exact reconstruction and convergence boundaries with deterministic reset and explicit invalid-input handling.

Current lesson: [complex-numbers-fourier-laplace-transforms](../../src/learn/data/topics/complex-numbers-fourier-laplace-transforms.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/complex-numbers-fourier-laplace-transforms.js). Previous review: [retained record](../../docs/teaching/COMPLEX-FOURIER-LAPLACE-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/COMPLEX-FOURIER-LAPLACE-LESSON-DESIGN.md).

### Conditioning, Stability & Numerical Analysis

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Rounding cells, arithmetic computation graphs, perturbed measurement lines, backward-error entry witnesses, summation trees and exact error-propagation traces** — Which part of this numerical result can be improved by changing the algorithm, and which depends on the problem or its data? Change a representable input, perturbation direction, allowed data model, accumulation order or refinement scale and explain the resulting geometry and verified errors.

Current lesson: [conditioning-stability-numerical-analysis](../../src/learn/data/topics/conditioning-stability-numerical-analysis.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/conditioning-stability-numerical-analysis.js). Previous review: [retained record](../../docs/teaching/CONDITIONING-STABILITY-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/CONDITIONING-STABILITY-LESSON-DESIGN.md).

### Decision Theory, Risk & Cost-Sensitive Decisions

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Concrete loss-contribution tables and decision trees; exact risk envelopes; weighted demand deficits; capacity allocation boards; state-risk geometry; utility chords and atom-aware probability tails** — Which feasible action has the best stated consequence given what is known now, and which change in costs, information or constraints would change that conclusion? Change actual loss cells and inspect all conditional contributions, allocate limited quarantine slots, reveal a test before choosing a contingent action, mix finite rules under a stated information order, and fill exactly the chosen tail mass.

Current lesson: [decision-theory-risk-cost-sensitive-decisions](../../src/learn/data/topics/decision-theory-risk-cost-sensitive-decisions.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/decision-theory-risk-cost-sensitive-decisions.js). Previous review: [retained record](../../docs/teaching/DECISION-THEORY-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/DECISION-THEORY-LESSON-DESIGN.md).

### Real Analysis, Sequences & Modes of Convergence

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Discrete tail bands, exact interval brackets, fixed versus moving witnesses, triangle zooms, error paths, paired height/slope plots and positive-weight polynomial construction** — What must stay controlled beyond the plotted samples for this limiting operation to be valid? Change the tolerance, domain, index or error shape; inspect an analytic witness or bound alongside the finite picture, and test the changed claim.

Current lesson: [real-analysis-sequences-modes-of-convergence](../../src/learn/data/topics/real-analysis-sequences-modes-of-convergence.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/real-analysis-sequences-modes-of-convergence.js). Previous review: [retained record](../../docs/teaching/REAL-ANALYSIS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/REAL-ANALYSIS-LESSON-DESIGN.md).

### Abstract Algebra, Groups & Symmetry Actions

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Labeled square composition lanes, action-state diagrams, orbit collections and fixed-cycle tallies, coset product witnesses and linked sensor/matrix routes** — Which changes are equivalent under the chosen transformations, and does this calculation respect that equivalence? Change composition order, color pattern, coset representative or sensor-map weights and inspect exact intermediate maps, distinct results and symmetry defects.

Current lesson: [abstract-algebra-groups-symmetry-actions](../../src/learn/data/topics/abstract-algebra-groups-symmetry-actions.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/abstract-algebra-groups-symmetry-actions.js). Previous review: [retained record](../../docs/teaching/ABSTRACT-ALGEBRA-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/ABSTRACT-ALGEBRA-LESSON-DESIGN.md).

### Partial Differential Equations, Conservation & Boundary Conditions

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Flux control volumes, space-time characteristics, boundary-selected modes, dependence cones, harmonic fields and weak jump balances** — Which data determine this field, and what changes when information can leave, reflect or diffuse through its boundary? Trace a selected observation to its data, compare heat profiles with different physical boundaries, inspect a wave cone, and change source or flux data while checking the same conservation identity

Current lesson: [partial-differential-equations-conservation-boundary-conditions](../../src/learn/data/topics/partial-differential-equations-conservation-boundary-conditions.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/partial-differential-equations-conservation-boundary-conditions.js). Previous review: [retained record](../../docs/teaching/PARTIAL-DIFFERENTIAL-EQUATIONS-VERIFICATION.md).

### Numerical PDEs: Grids, Finite Elements & Stability

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Stencil-to-row mapping, actual error budgets, calculated refinement curves, mode amplification, interface resistance, cell flux cancellation, shape-function assembly and source-between-node geometry** — Which continuous quantity does this finite answer approximate, and what evidence certifies its error? Change mesh, boundary, source, step, solver tolerance or element geometry; inspect the recalculated field, balance and appropriate error certificate.

Current lesson: [numerical-pdes-grids-finite-elements-stability](../../src/learn/data/topics/numerical-pdes-grids-finite-elements-stability.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/numerical-pdes-grids-finite-elements-stability.js). Previous review: [retained record](../../docs/teaching/NUMERICAL-PDES-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/NUMERICAL-PDES-LESSON-DESIGN.md).

### Python Basics: Types, Control Flow, Functions & Modules

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Name-to-object explorer** — Does this operation change a list or move a name? Compare alias/copy behavior and step append versus rebinding with reference arrows and visible list elements.

Current lesson: [python-basics-types-control-flow-functions-modules](../../src/learn/data/topics/python-basics-types-control-flow-functions-modules.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/python-basics-types-control-flow-functions-modules.js). Previous review: [retained record](../../FIRST-FIVE-REIMPLEMENTATION.md). Original detailed design: [design](../../docs/teaching/python-foundations-reimplementation.md).

### Object-Oriented Programming in Python

Selecting any of the six candidate readings immediately updates type/conversion/finite-value checks, the stopped gate, exception or accepted result and the same before/after list. Reset returns to the Boolean counterexample. Execution-frame and alias/composition steps remain actual process controls.

- **Receiver and identity explorer** — Which object does this method change? Follow aliases and bound methods to independent log objects with reference arrows.

Current lesson: [object-oriented-programming-in-python](../../src/learn/data/topics/object-oriented-programming-in-python.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/object-oriented-programming-in-python.js). Previous review: [retained record](../../FIRST-FIVE-REIMPLEMENTATION.md). Original detailed design: [design](../../docs/teaching/oop-foundations-reimplementation.md).

### Iterators, Iterables & Generators

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Source positions and cursor ownership** — Does a second name have an independent next item? Compare two iter calls with assignment, alternate requests, and try empty/exhausted input.

Current lesson: [iterators-iterables-generators](../../src/learn/data/topics/iterators-iterables-generators.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/iterators-iterables-generators.js). Previous review: [retained record](../../PROGRAMMING-MODULE-COMPLETION.md). Original detailed design: [design](../../docs/teaching/iteration-decorators-design.md).

### Decorators & Context Managers

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Nested callable references and outward result flow** — Does swapping two wrappers preserve the result? Choose cap/double order and inputs 3, 8 or 12; step inward calls and outward transformations.

Current lesson: [decorators-context-managers](../../src/learn/data/topics/decorators-context-managers.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/decorators-context-managers.js). Previous review: [retained record](../../PROGRAMMING-MODULE-COMPLETION.md). Original detailed design: [design](../../docs/teaching/iteration-decorators-design.md).

### Testing, Debugging & Dependency Management

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Execution with consumed inputs and accumulator** — Which readings contribute before return exits? Change return placement and input; step accumulation, unseen inputs and output.

Current lesson: [testing-debugging-dependency-management](../../src/learn/data/topics/testing-debugging-dependency-management.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/testing-debugging-dependency-management.js). Previous review: [retained record](../../PROGRAMMING-MODULE-COMPLETION.md). Original detailed design: [design](../../docs/teaching/reliability-authoring-design.md).

### NumPy: Arrays, Broadcasting & Vectorization

Selection, broadcast and reduction outputs are visible from the initial valid state and recompute with the selected operation, shape, cell or axis. Invalid broadcasting shows the actual incompatibility instead of a fabricated array. Retain linked array coordinates and shape labels; Write/Undo in the memory explorer performs an actual modeled mutation. Remove the three reveal toggles.

- **Broadcast coordinate explorer** — Which input and offset produce this output value? Change the offset shape, inspect aligned dimensions and select an output cell; include incompatible shapes.

Current lesson: [numpy-arrays-broadcasting-vectorization](../../src/learn/data/topics/numpy-arrays-broadcasting-vectorization.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/numpy-arrays-broadcasting-vectorization.js). Previous review: [retained record](../../FIRST-FIVE-REIMPLEMENTATION.md). Original detailed design: [design](../../docs/teaching/numpy-foundations-reimplementation.md).

### Scientific File Formats, Schemas & Reliable Data I/O

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Schema gates** — Which stage should reject this file? Select malformed numbers, nonfinite values, mixed units or duplicate IDs and step from raw text to validation.

Current lesson: [scientific-file-formats-schemas-reliable-data-i-o](../../src/learn/data/topics/scientific-file-formats-schemas-reliable-data-i-o.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/scientific-file-formats-schemas-reliable-data-i-o.js). Previous review: [retained record](../../FIRST-FIVE-REIMPLEMENTATION.md). Original detailed design: [design](../../docs/teaching/data-foundations-reimplementation.md).

### SQL, Relational Data & Transactions for ML

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Linked join rows** — Why did three sensors become four output rows? Select each result row to highlight its sources; vary join, cutoff, predicate placement and an unconstrained duplicate key.

Current lesson: [sql-relational-data-transactions-for-ml](../../src/learn/data/topics/sql-relational-data-transactions-for-ml.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/sql-relational-data-transactions-for-ml.js). Previous review: [retained record](../../FIRST-FIVE-REIMPLEMENTATION.md). Original detailed design: [design](../../docs/teaching/data-foundations-reimplementation.md).

### Pandas: Data Wrangling, Joins & Grouping

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Label-to-source mapping** — Which fee reaches order b when rows move or label a is absent? Switch label/position assignment, order and missing labels; select a destination to reveal the source arrow and total.

Current lesson: [pandas-data-wrangling-joins-grouping](../../src/learn/data/topics/pandas-data-wrangling-joins-grouping.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/pandas-data-wrangling-joins-grouping.js). Previous review: [retained record](../../NEXT-THREE-REIMPLEMENTATION.md). Original detailed design: [design](../../docs/teaching/next-three-reimplementation.md).

### Matplotlib & Scientific Plotting

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Data-to-position mapping** — Can the marker move without a measurement changing? Select a measurement and switch limits/scales while its normalized coordinate calculation stays visible.

Current lesson: [matplotlib-scientific-plotting](../../src/learn/data/topics/matplotlib-scientific-plotting.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/matplotlib-scientific-plotting.js). Previous review: [retained record](../../NEXT-THREE-REIMPLEMENTATION.md). Original detailed design: [design](../../docs/teaching/next-three-reimplementation.md).

### Reproducible Notebooks & Experiment Structure

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Document versus kernel versus saved output** — What changes when a cell is edited but not run? Edit source, execute individual cells, restart and run all while viewing separate states.

Current lesson: [reproducible-notebooks-experiment-structure](../../src/learn/data/topics/reproducible-notebooks-experiment-structure.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/reproducible-notebooks-experiment-structure.js). Previous review: [retained record](../../PROGRAMMING-MODULE-COMPLETION.md). Original detailed design: [design](../../docs/teaching/reliability-authoring-design.md).

### Code Documentation, Type Hints & API Design

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **API validation gates** — Can filtering hide an invalid input? Select zero, bool, NaN, equality and invalid low scores; trace the first failed gate and returned mapping.

Current lesson: [code-documentation-type-hints-api-design](../../src/learn/data/topics/code-documentation-type-hints-api-design.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/code-documentation-type-hints-api-design.js). Previous review: [retained record](../../PROGRAMMING-MODULE-COMPLETION.md). Original detailed design: [design](../../docs/teaching/reliability-authoring-design.md).

### Git, GitHub & Collaborative Version Control

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Three stored snapshots with transfer arrows** — Will committing record version 2 or the version 3 currently in the editor? Step editing and staging; choose whether to stage again; compare exact HEAD/index/worktree contents and both status columns.

Current lesson: [git-github-collaborative-version-control](../../src/learn/data/topics/git-github-collaborative-version-control.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/git-github-collaborative-version-control.js). Previous review: [retained record](../../NEXT-THREE-REIMPLEMENTATION.md). Original detailed design: [design](../../docs/teaching/next-three-reimplementation.md).

### Linux Basics, Filesystems & Processes

Path text, start directory and operation immediately resolve to their destination or first failure. The learner can walk backward/forward from that computed result; changing an input resets the trace to the new result. Stream-route presets and exit-status controls update descriptor paths, terminal/file contents and pipeline status together. The path-prediction selector, stream prediction form and reveal gate are removed. Permission and child-process investigations retain meaningful state transitions.

- **Stepped path tree** — Which directory does this relative path reach, and when does the shell move? Edit a path and see the destination or first failure immediately; walk through its components, reset and compare an absolute path.
- **Stream routing lab** — Where do the result and warning go? Change routes/status and see exact stdout, stderr and exit outcomes immediately; inspect descriptor ordering in optional depth.
- **Permission gates lab** — Which access check fails first? Toggle directory listing/search and file-read bits, then compare listing a directory with accessing a known filename.
- **Process lifecycle lab** — Is this child stopped, alive or finished? Start, pause, continue, terminate and collect status with the parent still visible.
- **Static environment and link diagrams** — What is copied or shared? Trace parent/child environment values and the difference between hard-link identity and a symbolic link's stored path.

Current lesson: [linux-basics-filesystems-processes](../../src/learn/data/topics/linux-basics-filesystems-processes.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/linux-basics-filesystems-processes.js). Previous review: [retained record](../../PROGRAMMING-REWRITE-LINUX.md).

### Bash Scripting & Command-Line Automation

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Expansion to argument boundaries** — How does one filename become two arguments? Select spaces, wildcard or empty value and quoted/unquoted expansion; step parse, expand, split/glob and launch.

Current lesson: [bash-scripting-command-line-automation](../../src/learn/data/topics/bash-scripting-command-line-automation.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/bash-scripting-command-line-automation.js). Previous review: [retained record](../../PROGRAMMING-MODULE-COMPLETION.md). Original detailed design: [design](../../docs/teaching/bash-completion-verification.md).

### OS Processes, Virtual Memory & Isolation

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **CPU timeline and saved process state** — Which process can run next, and which counter changes? Change time slice and I/O request; step execution, waits, next instruction and ready queue.

Current lesson: [os-processes-virtual-memory-isolation](../../src/learn/data/topics/os-processes-virtual-memory-isolation.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/os-processes-virtual-memory-isolation.js). Previous review: [retained record](../../SYSTEMS-STRUCTURES-IMPLEMENTATION.md). Original detailed design: [design](../../docs/teaching/systems-and-structures-design.md).

### Threads, Concurrency, Locks & Deadlocks

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Shared counter and worker-local state** — How can two increments produce one? Schedule either worker one step at a time; toggle a whole-operation lock and inspect blocked attempts.

Current lesson: [threads-concurrency-locks-deadlocks](../../src/learn/data/topics/threads-concurrency-locks-deadlocks.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/threads-concurrency-locks-deadlocks.js). Previous review: [retained record](../../PROGRAMMING-MODULE-COMPLETION.md). Original detailed design: [design](../../docs/teaching/bash-and-concurrency-design.md).

### Arrays, Strings & Hash Maps

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Backing-array cells and source/destination arrows** — What must move to preserve order? Choose insertion position or deletion and spare/full capacity; step copying, shifting and insertion with counted writes.

Current lesson: [arrays-strings-hash-maps](../../src/learn/data/topics/arrays-strings-hash-maps.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/arrays-strings-hash-maps.js). Previous review: [retained record](../../docs/teaching/BITWISE-FOUNDATIONS-INDEPENDENT-REVIEW.md). Original detailed design: [design](../../docs/teaching/systems-and-structures-design.md).

### Linked Lists, Stacks & Queues

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Node arrows and named references** — Which reference must be saved before changing next? Step correct/broken reversal for three nodes, one node or empty input; see all identities even when a suffix becomes unreachable.

Current lesson: [linked-lists-stacks-queues](../../src/learn/data/topics/linked-lists-stacks-queues.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/linked-lists-stacks-queues.js). Previous review: [retained record](../../docs/teaching/LINKED-TRAVERSAL-MONOTONIC-INDEPENDENT-REVIEW.md). Original detailed design: [design](../../docs/teaching/LINKED-TRAVERSAL-MONOTONIC-EXTENSION-DESIGN.md).

### Trees & Binary Search Trees

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Tree topology linked to path, bounds, frontier and output** — Which relationships and pending work must survive this operation? Search or insert bounded integer keys, step four traversals, and trace leaf/one-child/two-child deletion while node identity remains visible.

Current lesson: [trees-binary-search-trees](../../src/learn/data/topics/trees-binary-search-trees.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/trees-binary-search-trees.js). Previous review: [retained record](../../DSA-CORE-STRUCTURES-IMPLEMENTATION.md).

### Heaps, Priority Queues & Tries

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Linked heap array/tree, retained-stream boundary and character-labelled trie** — Which relation must remain true while the next candidate or key changes? Step bounded heap push/pop/build, vary stream and k, query/insert/delete prefixes while inspecting terminal flags and pruning.

Current lesson: [heaps-priority-queues-tries](../../src/learn/data/topics/heaps-priority-queues-tries.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/heaps-priority-queues-tries.js). Previous review: [retained record](../../DSA-CORE-STRUCTURES-IMPLEMENTATION.md).

### Graphs: Representations, BFS & DFS

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Graph/list/matrix correspondence, queue/layer and DFS-frame traces, grid wavefront** — What is already known, what work is pending, and what does the requested answer require? Edit bounded edges/direction, step BFS/DFS and inspect parent routes, then change grid walls and sources while tracking distances.

Current lesson: [graphs-representations-bfs-dfs](../../src/learn/data/topics/graphs-representations-bfs-dfs.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/graphs-representations-bfs-dfs.js). Previous review: [retained record](../../DSA-CORE-STRUCTURES-IMPLEMENTATION.md).

### Disjoint Sets & Union-Find

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Equivalent-partition forest figure, parent-pointer traces and active-island grid** — What changes in the implementation while the same membership relation is preserved? Apply bounded unions, compare attachment policies, step exact compression rewrites, and activate cells while inspecting component memberships and successful joins.

Current lesson: [disjoint-sets-union-find](../../src/learn/data/topics/disjoint-sets-union-find.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/disjoint-sets-union-find.js). Previous review: [retained record](../../docs/teaching/UNION-FIND-VERIFICATION.md).

### Complexity Analysis & Recursion

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Iteration lattice, suspended call frames and recurrence level accounting** — Which execution events occur, which work is still waiting, and how do they grow? Change loop size/shape, step a bounded suffix-sum call trace, and compare shrinking recurrences with exact work and depth.

Current lesson: [complexity-analysis-recursion](../../src/learn/data/topics/complexity-analysis-recursion.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/complexity-analysis-recursion.js). Previous review: [retained record](../../docs/teaching/COMPLEXITY-RECURSION-VERIFICATION.md).

### Binary Search, Sorting & Two-Pointer Patterns

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Boundary gaps, record merge lanes, pair-candidate matrix, moving interval and work-budget strips** — What can safely be discarded, retained or summarized after this operation? Step boundaries, choose merge heads, eliminate rows/columns of candidate pairs, grow/shrink a window and test integer processing rates.

Current lesson: [binary-search-sorting-two-pointer-patterns](../../src/learn/data/topics/binary-search-sorting-two-pointer-patterns.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/binary-search-sorting-two-pointer-patterns.js). Previous review: [retained record](../../docs/teaching/ORDERED-PATTERNS-VERIFICATION.md).

### Backtracking & Divide-and-Conquer

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Decision tree with live path/copied outputs, attacked-square chessboard, divide-tree interval summaries and inversion boundary figure** — What information does this child explore or return, and what must survive its completion? Change positive subset inputs and pruning, step real queen placements/rejections/undo, and inspect signed-array subregions with computed boundary summaries and witnesses.

Current lesson: [backtracking-divide-and-conquer](../../src/learn/data/topics/backtracking-divide-and-conquer.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/backtracking-divide-and-conquer.js). Previous review: [retained record](../../docs/teaching/BACKTRACKING-DIVIDE-VERIFICATION.md).

### Greedy Algorithms & Exchange Arguments

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Appointment timelines, aligned schedule exchanges, capacity fractions, merge trees and reachable-prefix figures** — What can be exchanged without losing feasibility or worsening the objective, and which changed assumption makes that exchange invalid? Compare interval rules against a small exact oracle, swap adjacent processing jobs and observe signed lateness, then change capacity and divisibility while retaining the same items.

Current lesson: [greedy-algorithms-exchange-arguments](../../src/learn/data/topics/greedy-algorithms-exchange-arguments.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/greedy-algorithms-exchange-arguments.js). Previous review: [retained record](../../docs/teaching/GREEDY-EXCHANGE-VERIFICATION.md).

### Dynamic Programming: States, Transitions & Optimization

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Subproblem graph, spatial grid, alignment table, capacity-generation strip, membership-bit/endpoint map, interval shape/expression planner, conditional tree and digit-prefix branches** — Which information makes two histories equivalent, and which already-solved state does this transition use? Step requests or tabulation, toggle obstacles, inspect prefix dependencies, change capacity iteration direction and compare endpoint futures for one subset; inspect interval splits and postorder witnesses, vary tree parent permissions and follow legal digit-prefix completions.

Current lesson: [dynamic-programming-states-transitions-optimization](../../src/learn/data/topics/dynamic-programming-states-transitions-optimization.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/dynamic-programming-states-transitions-optimization.js). Previous review: [retained record](../../docs/teaching/DP-STATE-FAMILIES-INDEPENDENT-REVIEW.md).

### Segment Trees, Fenwick Trees & Range Queries

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Indexed interval trees, binary block lanes, pending-map trees, candidate deque timelines and prefix-coordinate plots** — What does each stored summary cover, which state can safely be reused or discarded, and how does the actual update alter it? Edit bounded signed arrays, trace canonical query covers and point writes, follow Fenwick read/update jumps, force lazy push/pull through overlapping add/set actions, and investigate two distinct deque retirement rules.

Current lesson: [segment-trees-fenwick-trees-range-queries](../../src/learn/data/topics/segment-trees-fenwick-trees-range-queries.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/segment-trees-fenwick-trees-range-queries.js). Previous review: [retained record](../../docs/teaching/RANGE-QUERIES-VERIFICATION.md).

### Algorithm Correctness, Loop Invariants & Termination

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Proof-boundary prefix inspector, original/working occurrence rows, four-region partition and divisor-preserving Euclid trace** — What does this boundary certify, which step could destroy it, and what strictly progresses? Apply tiny inputs, compare candidate assertions, inject a skipped-position or skipped-classification fault, step mutation boundaries and inspect decreasing remainders.

Current lesson: [algorithm-correctness-loop-invariants-termination](../../src/learn/data/topics/algorithm-correctness-loop-invariants-termination.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/algorithm-correctness-loop-invariants-termination.js). Previous review: [retained record](../../docs/teaching/ALGORITHM-CORRECTNESS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/ALGORITHM-CORRECTNESS-LESSON-DESIGN.md).

### Hashing, Collision Resolution & Amortized Analysis

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Key-to-home inline figures, EMPTY/DELETED probe trace, exact hash-family outcome lattice, paired operation-cost timelines and dense reverse-index mutation** — What information permits this lookup to stop, and over which choices or operations is this cost bound taken? Edit tiny operation streams, inject a deletion fault, step probes and rebuilds, vary a hash choice while holding keys fixed, compare resize policies and inspect swap-delete relationships.

Current lesson: [hashing-collision-resolution-amortized-analysis](../../src/learn/data/topics/hashing-collision-resolution-amortized-analysis.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/hashing-collision-resolution-amortized-analysis.js). Previous review: [retained record](../../docs/teaching/HASHING-AMORTIZED-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/HASHING-AMORTIZED-LESSON-DESIGN.md).

### Shortest Paths, Spanning Trees & Topological Ordering

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Weighted directed geometry, priority-entry ledger, distance-generation matrix, accepted-edge component forest, ready-job frontier and a computed critical-path timeline** — Which candidate, connection or dependency is safe to trust now, and which invariant certifies it? Edit bounded exact graph edges, trace Dijkstra and Bellman–Ford, switch source and forest growth method, manually choose ready jobs, and expose negative-cycle influence or blocked descendants.

Current lesson: [shortest-paths-spanning-trees-topological-ordering](../../src/learn/data/topics/shortest-paths-spanning-trees-topological-ordering.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/shortest-paths-spanning-trees-topological-ordering.js). Previous review: [retained record](../../docs/teaching/WEIGHTED-GRAPHS-VERIFICATION.md).

### String Matching, Prefix Functions & Rolling Hashes

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Aligned overlap ribbons, paired prefix/suffix builder, KMP alignment trace, chunk delivery strip, rolling arithmetic and collision ledger, Unicode coordinate rows and Z-box reuse** — Which evidence survives this mismatch, boundary or arithmetic compression? Inspect fallbacks, step and reset exact comparisons, feed chunks with a deliberately faulty reset mode, and vary tiny fingerprints while inspecting actual identity.

Current lesson: [string-matching-prefix-functions-rolling-hashes](../../src/learn/data/topics/string-matching-prefix-functions-rolling-hashes.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/string-matching-prefix-functions-rolling-hashes.js). Previous review: [retained record](../../docs/teaching/STRING-MATCHING-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/STRING-MATCHING-LESSON-DESIGN.md).

### Reductions, P, NP & Computational Intractability

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Clause truth circuit, solver-conversion boundary, clause-grouped literal occurrence graph, exact bit-length versus state count, and cover graph with disjoint-matching lower bounds** — Which precise claim does this certificate, transformation or bound establish? Toggle assignment bits, change a bounded formula, choose compatible occurrences, inspect exact encoding counts, edit a small graph, build covers and inspect an edge-budget branch tree.

Current lesson: [reductions-p-np-computational-intractability](../../src/learn/data/topics/reductions-p-np-computational-intractability.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/reductions-p-np-computational-intractability.js). Previous review: [retained record](../../docs/teaching/REDUCTIONS-INTRACTABILITY-VERIFICATION.md).

### Randomized Algorithms, Sampling & Error Guarantees

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Outcome tree, raw-ticket mapping, shrinking shuffle prefix, stream slots and full subset distribution, partition-survival trace, paired matrix-probe paths, finite error bars and accuracy budget comparison** — Which outcomes receive probability mass, and what exactly can vary or fail under the declared experiment? Map or reject raw outcomes; choose individual shuffle/reservoir/pivot branches; expose cancellation with binary probes; compare independent versus reused evidence and change precision budgets.

Current lesson: [randomized-algorithms-sampling-error-guarantees](../../src/learn/data/topics/randomized-algorithms-sampling-error-guarantees.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/randomized-algorithms-sampling-error-guarantees.js). Previous review: [retained record](../../docs/teaching/RANDOMIZED-ALGORITHMS-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/RANDOMIZED-ALGORITHMS-LESSON-DESIGN.md).

### Network Flow, Minimum Cuts & Bipartite Matching

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Original flow graph with incidence ledger, separately owned residual pairs, stepped cancellation route and cut inspector, compatibility matrix with alternating arrows/cover rings, vertex gate, and binary-label energy grid** — Which feasible change improves this allocation, and which independently checkable obstruction proves that no better answer remains? Edit proposed flow and bounded capacities, preview/apply/back through augmentations, inspect alternate cuts, toggle compatibility and recover shortage certificates, and change binary labels or disagreement penalties.

Current lesson: [network-flow-minimum-cuts-bipartite-matching](../../src/learn/data/topics/network-flow-minimum-cuts-bipartite-matching.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/network-flow-minimum-cuts-bipartite-matching.js). Previous review: [retained record](../../docs/teaching/NETWORK-FLOW-VERIFICATION.md).

### Computational Geometry, Robust Predicates & Convex Hulls

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Equal-scale oriented triangles, segment-contact geometry, exact/Number arithmetic paths, hull candidate/pop/push trace, concave-envelope contrast and polygon-ray parity** — Which geometric relationship determines the branch, and does the arithmetic preserve that relationship? Move integer points, reverse a baseline, compare exact and rounded products, edit bounded hull inputs and boundary policy, step actual chain removals, and inspect ray crossings.

Current lesson: [computational-geometry-robust-predicates-convex-hulls](../../src/learn/data/topics/computational-geometry-robust-predicates-convex-hulls.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/computational-geometry-robust-predicates-convex-hulls.js). Previous review: [retained record](../../docs/teaching/COMPUTATIONAL-GEOMETRY-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/COMPUTATIONAL-GEOMETRY-LESSON-DESIGN.md).

### Persistent Data Structures, Structural Sharing & Versioned Queries

Retain the already playable mechanism and its topic-specific controls; no interaction replacement is needed.

- **Shared-tail reference figure, immutable physical-node DAG with version roots and interval cover, root-reachability identity chips, per-index write timeline, and prefix-frequency histogram with rank descent** — Which objects must change, which references can safely be reused, and which version-specific facts justify this historical answer? Create bounded branches from any root, query saved intervals, inspect identities, release conceptual root handles, choose historical timestamps and follow a computed subarray rank.

Current lesson: [persistent-data-structures-structural-sharing-versioned-queries](../../src/learn/data/topics/persistent-data-structures-structural-sharing-versioned-queries.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/persistent-data-structures-structural-sharing-versioned-queries.js). Previous review: [retained record](../../docs/teaching/PERSISTENT-DATA-STRUCTURES-VERIFICATION.md).

### External-Memory Algorithms, B-Trees & I/O Complexity

Retain the existing live calculations and genuine process controls; replace prediction-first lab prompts with exploration guidance. No new answer-entry flow is added.

- **Storage pages and buffer frames, multi-key page trees, linked B+ leaves, materialized run/merge lanes and volatile/durable root publication** — Which data move across the costly boundary, and which invariant permits the next page operation? Step actual page references and dirty evictions, insert/delete tree keys, follow a range across leaves, vary merge buffers and interrupt staged page publication.

Current lesson: [external-memory-algorithms-b-trees-i-o-complexity](../../src/learn/data/topics/external-memory-algorithms-b-trees-i-o-complexity.jsx). Topic contract: [blueprint](../../src/learn/data/curriculum/blueprints/external-memory-algorithms-b-trees-i-o-complexity.js). Previous review: [retained record](../../docs/teaching/EXTERNAL-MEMORY-VERIFICATION.md). Original detailed design: [design](../../docs/teaching/EXTERNAL-MEMORY-LESSON-DESIGN.md).

## Verified interaction scope

The final production browser receipt records five focused behavior groups (K-Means, Linux, NumPy, OOP, and diagonal/factorial construction) plus a 96-page foundation survey at 390 px. The focused groups exercise actual changed inputs, process controls, invalid states and selected keyboard interactions. The page survey verifies loading, headings, no learner-prediction forms and no document-width overflow; it is not a claim that every control was exercised. All groups passed with no uncaught page errors. Exact font fixtures were served unchanged because the sandbox blocks the external font service.

The [source disposition](evidence/live-exploration-foundations-source.json) identifies the affected files and 159 distinct unchanged mathematical/model/example files whose prior reviewed hashes still match. Other preexisting sources were preserved without inventing retrospective hash evidence. The [independent bounded review](LIVE-EXPLORATION-FOUNDATIONS-INDEPENDENT-REVIEW.md) covers the substantive state/reset changes; it does not assert a new independent mathematical audit of all 96 topics. Root inspected the four retained informative desktop/phone captures linked by the browser receipt.

Treat this scoped record as frozen evidence after closure. Later topic revisions should create or update their own current topic contract and preserve this record, so one lesson update does not rewrite the historical review of every foundation topic.
