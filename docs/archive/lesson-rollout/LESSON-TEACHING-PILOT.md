> HISTORICAL RECORD — archived 9 September 2026. All queue, authorization, approval and teaching instructions below describe an earlier increment. They are not current policy. Start at [the current handoff](../../../LESSON-AUTHORING-HANDOFF.md).

# Teaching-quality pilot

Latest implementation, 9 September 2026: the authorized [Linux single-topic improvement](../../../PROGRAMMING-REWRITE-LINUX.md) is complete and ready for user review. It adds four focused labs and guided practice to the existing lesson; model checks, native Linux execution, desktop/mobile browser tests and the build passed. Bash remains the next topic needing a rewrite. Earlier status notes below are historical; use [the teaching standard](../../../LESSON-TEACHING-STANDARD.md) and the latest increment report to continue.

Continuation review, 9 September 2026: read [LESSON-TEACHING-STANDARD.md](../../../LESSON-TEACHING-STANDARD.md) for the reusable authoring instructions and [LESSON-CONTINUATION-REVIEW.md](LESSON-CONTINUATION-REVIEW.md) for current source status and teaching findings. Git and Linux now have detailed implementations; their latest verification and user-review status are not established by the historical notes below. Linux is the next review/completion point after Git; Bash is the next lesson still needing the full rewrite. This continuation pass changed documentation only. The earlier follow-up below is retained as history.

Follow-up, 9 September 2026: the first ten programming lessons have been rewritten; see [batch 01](PROGRAMMING-REWRITE-BATCH-01.md), [batch 02](PROGRAMMING-REWRITE-BATCH-02.md), the [Pandas single-topic increment](PROGRAMMING-REWRITE-PANDAS.md), and [batch 03](PROGRAMMING-REWRITE-BATCH-03.md). Batch 03 contains Matplotlib, reproducible notebooks, and documentation/type hints/API design, and awaits review. Git, Linux and Bash remain for the next approved batch. The source inventory below is the historical 8 September snapshot, not the latest completion status.

Date: 8 September 2026. Status: three lessons implemented; wider rollout awaits approval.

## Scope and confidence

The source inventory covers 187 registered custom lessons across five tracks (42 mathematics, 13 programming, 36 classical ML, 39 deep learning, 57 LLMs). Planned catalogue entries are not treated as written lessons. DSA is untouched.

This is a broad **structural scan**, plus close reading and numerical verification of the three rewritten lessons. It is not a claim that every statement in all 187 lessons has been fact-checked. Source flags also miss content produced by shared components. Each remaining lesson needs a concept-by-concept review before it can be called teaching-complete.

## Review the three pages

| Pilot | Main improvement | Try this |
| --- | --- | --- |
| [Hypothesis Testing & Confidence Intervals](http://127.0.0.1:5173/learn/topic/hypothesis-testing-confidence-intervals) | One paired latency experiment connects the question, standard error, t interval, p-value and practical decision. Corrects the inappropriate small-sample normal critical value. | Increase sample size from 25 to 100 in the coverage lab; predict the width change first. |
| [Bayesian Inference & Conjugate Priors](http://127.0.0.1:5173/learn/topic/bayesian-inference-conjugate-priors) | Derives the Beta update and distinguishes likelihood, posterior, credible interval and future prediction. | Keep 8 successes / 2 failures and strengthen the prior; explain the change before reading the result. |
| [Spectral Graph Theory](http://127.0.0.1:5173/learn/topic/spectral-graph-theory) | Connects an actual six-node graph to a Laplacian row, energy, eigenvectors, clustering and smoothing. States unit-normalisation and eigenspace caveats. | Remove the bridge and explain the two zero eigenvalues; then inspect higher modes. |

These local links require the development server. Start it with `npm.cmd run dev -- --host 127.0.0.1` from the project root if it is not running. Nothing has been deployed.

## What changed in the teaching

- Start with a concrete problem, prerequisites and a visible reading route. Beginners follow the sequence; returning learners jump to worked examples and practice.
- Use one consistent dataset/graph through the explanation, calculation and conclusion. This reduces the effort of learning a new example at every heading.
- Explain symbols and assumptions before asking the reader to manipulate a formula. Preserve mathematical rendering with raw LaTeX strings.
- Give visuals a job: predict, change one variable, observe the result, explain why. These are working browser labs, not decorative diagrams.
- Show complete runnable Python and its verified output. Follow output with its meaning, not just an instruction to run the code.
- Add answer reveals with reasoning, including counterexamples and common misconceptions. Learners can try independently without becoming stuck permanently.
- Separate method conditions and failure modes from the happy path. Examples include dependent requests, prior sensitivity and non-unique eigenvectors.
- Keep deeper references optional. The core walkthrough does not require following an external video.
- Retain the existing dark/gold styling. Tables, focus states, chart labels and controls work at phone and desktop widths.
- Remove the redundant generic lesson guide only for these three opt-in articles. Other article structure is preserved.

## Recurring gaps and next priorities

| Area reviewed | Finding / risk | Next action after approval |
| --- | --- | --- |
| Statistics | The old confidence-interval example used a normal cutoff with five observations and estimated variability. Definitions alone did not establish when to use a method. | Review distributions, MLE/MAP, concentration, Monte Carlo and variational inference against explicit sampling models and assumptions. |
| Mathematical foundations | Visible outputs were added to many lessons, but a printed number is not necessarily interpreted. Several practice prompts have no worked solution. | Carry a small example through derivation, output, interpretation and a solved transfer exercise. Prioritise calculus, optimisation and matrix topics before dependent advanced lessons. |
| KKT/duality sample | The zero-multiplier explanation is too categorical; regularity, necessity and sufficiency need careful separation. | Work an actual multiplier calculation and a degenerate active-constraint counterexample; state assumptions precisely. Not edited in this pilot. |
| Programming / NumPy sample | A practical API map lists many functions, but its shared component renders one worked pattern without an output block. A function name plus a sentence does not teach its shape behaviour or failure cases. | Keep the reference map, but add task-based modules with input, shape/dtype, output, explanation, edge case and an exercise. Do not dump an entire library reference into one article. |
| Classical ML, deep learning and LLM source inventory | Source files often lack explicit output blocks or recognised visual components. Some already have extensive answer callouts, so they must not be blindly replaced. | Preserve useful material; review each example and existing visual before adding anything. Use tiny datasets, tensor-shape traces, ablations and failure examples appropriate to each concept. |
| Reader navigation | A generic guide can point at a heading that a particular article does not have. Reading-time labels do not establish depth. | Validate every internal link and estimate reading separately from practice. Do not treat a build pass, word count or published status as a learning-quality certificate. |
| Delivery | The build reports a large shared content chunk and existing JSX warnings in another lesson. | Separate performance/technical cleanup from content approval; lazy-load topic content in a later scoped change. |

## The proposed standard (not a rigid template)

Every lesson should answer: What problem does this solve? What do I need first? How does it work? Can I calculate one example myself? What does the output mean? When does it fail? Can I solve a changed problem without copying?

A lesson is ready when its core path has a complete example, explicit assumptions, verified outputs where appropriate, useful visual support where relationships are hard to imagine, and practice with explained solutions. Not every concept needs a slider, and not every API call needs a separate page.

Programming modules should use **task → input → code → output → shape/state explanation → edge case → practice**, with a separate searchable API reference. Mathematical modules should use **question → intuition → notation → derivation → worked calculation → interpretation → limitations → transfer practice**. Model/system lessons should trace **data → computation → objective → evaluation → failure case**. These are guides, not mandatory filler headings.

For large topics, split into sequenced modules with a short core route and optional deeper branches. Completeness means a clear coverage map and linked progression, not a single unbounded wall of text. Revision should offer summaries and retrieval questions; first-time learners should not be confronted with the whole API surface at once.

## Rollout, only after approval

1. Agree whether these three examples have the right depth, pacing, visual usefulness and exercise difficulty.
2. Review programming and foundational math first, module by module. For each, list subtopics and prerequisites before editing, and mark which details belong in a deeper module.
3. Finish probability/statistics and advanced math in dependency order, keeping numerical checks alongside examples.
4. Apply the approved approach to already-written classical ML, deep learning and LLM content. Verify time-sensitive model/software claims against primary sources.
5. Report each finished topic with its coverage, tests and remaining limitations. Keep DSA excluded until explicitly requested.

## Research used

The new prose, examples and interactive labs are original. These references support the concepts or provide another teaching route; no video was claimed to have been watched end to end.

- [NIST: confidence limits for a mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm) — t intervals and sampling assumptions.
- [SciPy: paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html) — paired-test implementation and outputs.
- [Seeing Theory: frequentist inference](https://seeing-theory.brown.edu/frequentist-inference/index.html) and [Bayesian inference](https://seeing-theory.brown.edu/bayesian-inference/index.html) — interactive presentations reviewed for visual teaching.
- [3Blue1Brown: Bayes' theorem](https://www.3blue1brown.com/lessons/bayes-theorem/) — illustrated lesson and companion video reference.
- [StatQuest video index](https://statquest.org/video_index.html) — optional p-value/power video route, not the mathematical authority for the rewrite.
- [The Book of Statistical Proofs: binomial posterior](https://statproofbook.github.io/P/bin-post.html) and [SciPy Beta distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html) — conjugacy and probability calculations.
- [von Luxburg: A Tutorial on Spectral Clustering](https://arxiv.org/abs/0711.0189) — Laplacians and distinct clustering algorithms.
- [NumPy eigh](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html) and [scikit-learn SpectralClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html) — implementation conventions and algorithm options.

## Verification

- `scripts/verify-lesson-pilot.mjs`: executes all three displayed Python examples and compares actual output with the article; checks KaTeX; tests coverage behaviour; checks Beta integrals/CDFs/quantiles and boundary cases; compares graph spectra with NumPy and checks eigenpair residuals.
- `scripts/review-lesson-pilot.cjs`: tests all three pages at 1440 px and 390 px, navigation targets, answer reveals, controls, missing-guide duplication, rendering errors and horizontal overflow. Tests disconnected graphs and zero-observation Bayes updates.
- `npm.cmd run build`: passes. Existing unrelated JSX and large-chunk warnings remain.
- Python examples run in the learner's own Python environment. Browser labs run in the page. This pilot does not add an embedded Python runtime or automatic exercise grading.

To rerun numerical tests, set `LESSON_PYTHON` to a Python executable with NumPy. To rerun browser tests, install Playwright or set `PLAYWRIGHT_PACKAGE` to its package directory, have Edge installed, and start the local server. Screenshots go to `scratch/lesson-pilot-review`; the fixed navigation is hidden only during element screenshots to avoid masking the captured lab.

## Registered-lesson review queue

Reproduce this read-only inventory with `node scripts/audit-lesson-structure.mjs` (add `--headings` for heading extraction). The inventory is after the pilot changes.

**O / V / S** are source signals: explicit output block / recognised visual component / recognised answer or solution pattern. A dash means **not detected**, not proven absent. Shared renderers, differently named components, comments and naming conventions can cause false positives or negatives. These flags prioritise manual review, not score content quality. Only the three pilot rows have been rewritten and numerically checked in this pass.

### Mathematical & Statistical Foundations

| Lesson | O / V / S | Status |
| --- | --- | --- |
| [Vectors, Matrices & Tensor Operations](../../../src/learn/data/topics/vectors-matrices-tensor-operations.jsx) | — / — / — | Manual teaching audit pending |
| [Matrix Decompositions (SVD, QR, Cholesky, LU)](../../../src/learn/data/topics/matrix-decompositions-svd-qr-cholesky-lu.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Eigenvalues & Eigenvectors](../../../src/learn/data/topics/eigenvalues-eigenvectors.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Matrix Calculus & Jacobians](../../../src/learn/data/topics/matrix-calculus-jacobians.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Tensor Algebra & Einsum Notation](../../../src/learn/data/topics/tensor-algebra-einsum-notation.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Randomized Linear Algebra](../../../src/learn/data/topics/randomized-linear-algebra.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Multivariate Calculus & Gradients](../../../src/learn/data/topics/multivariate-calculus-gradients.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Convex Optimization](../../../src/learn/data/topics/convex-optimization.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Gradient Descent Variants (SGD, Adam, AdaGrad, RMSProp, LAMB, LARS)](../../../src/learn/data/topics/gradient-descent-variants-sgd-adam-adagrad-rmsprop-lamb-lars.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Learning Rate Schedules (Cosine, Warmup, OneCycleLR)](../../../src/learn/data/topics/learning-rate-schedules-cosine-warmup-onecyclelr.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Convex Duality & Lagrangian Methods (KKT Conditions)](../../../src/learn/data/topics/convex-duality-lagrangian-methods-kkt-conditions.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Second-Order Methods (L-BFGS, K-FAC, Shampoo, Natural Gradient)](../../../src/learn/data/topics/second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Non-Convex Optimization Landscape](../../../src/learn/data/topics/non-convex-optimization-landscape.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Constrained & Multi-Objective Optimization](../../../src/learn/data/topics/constrained-multi-objective-optimization.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Probability Distributions & Bayes' Theorem](../../../src/learn/data/topics/probability-distributions-bayes-theorem.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Maximum Likelihood & MAP Estimation](../../../src/learn/data/topics/maximum-likelihood-map-estimation.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Hypothesis Testing & Confidence Intervals](../../../src/learn/data/topics/hypothesis-testing-confidence-intervals.jsx) | ✓ / ✓ / ✓ | Pilot implemented |
| [Bayesian Inference & Conjugate Priors](../../../src/learn/data/topics/bayesian-inference-conjugate-priors.jsx) | ✓ / ✓ / ✓ | Pilot implemented |
| [Concentration Inequalities (Hoeffding, Bernstein, Chernoff)](../../../src/learn/data/topics/concentration-inequalities-hoeffding-bernstein-chernoff.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Monte Carlo Methods & MCMC (Metropolis-Hastings, HMC, NUTS)](../../../src/learn/data/topics/monte-carlo-methods-mcmc-metropolis-hastings-hmc-nuts.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Variational Inference](../../../src/learn/data/topics/variational-inference.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Exponential Families & Sufficient Statistics](../../../src/learn/data/topics/exponential-families-sufficient-statistics.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Measure Theory & Probability Spaces](../../../src/learn/data/topics/measure-theory-probability-spaces.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Optimal Transport (Wasserstein Distance, Sinkhorn)](../../../src/learn/data/topics/optimal-transport-wasserstein-distance-sinkhorn.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Causal Inference & Do-Calculus](../../../src/learn/data/topics/causal-inference-do-calculus.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Entropy, Cross-Entropy & KL Divergence](../../../src/learn/data/topics/entropy-cross-entropy-kl-divergence.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Mutual Information & Information Bottleneck](../../../src/learn/data/topics/mutual-information-information-bottleneck.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Rate-Distortion Theory](../../../src/learn/data/topics/rate-distortion-theory.jsx) | ✓ / — / — | Manual teaching audit pending |
| [f-Divergences & Integral Probability Metrics](../../../src/learn/data/topics/f-divergences-integral-probability-metrics.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Graph Fundamentals (Adjacency, Laplacian, Connectivity)](../../../src/learn/data/topics/graph-fundamentals-adjacency-laplacian-connectivity.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Spectral Graph Theory](../../../src/learn/data/topics/spectral-graph-theory.jsx) | ✓ / ✓ / ✓ | Pilot implemented |
| [Combinatorial Optimization & Approximation Algorithms](../../../src/learn/data/topics/combinatorial-optimization-approximation-algorithms.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Stochastic Processes (Markov Chains, Brownian Motion, Poisson)](../../../src/learn/data/topics/stochastic-processes-markov-chains-brownian-motion-poisson.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Random Matrix Theory](../../../src/learn/data/topics/random-matrix-theory.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Queueing Theory (M/M/1, M/G/1, Little's Law)](../../../src/learn/data/topics/queueing-theory-m-m-1-m-g-1-littles-law.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Dynamical Systems Theory & Chaos](../../../src/learn/data/topics/dynamical-systems-theory-chaos.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Itô Calculus & Stochastic Differential Equations](../../../src/learn/data/topics/ito-calculus-stochastic-differential-equations.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Numerical Methods (Finite Differences, Quadrature, Root Finding)](../../../src/learn/data/topics/numerical-methods-finite-differences-quadrature-root-finding.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Functional Analysis & RKHS](../../../src/learn/data/topics/functional-analysis-rkhs.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Topology & Topological Data Analysis (TDA)](../../../src/learn/data/topics/topology-topological-data-analysis-tda.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Category Theory (Emerging Use in ML)](../../../src/learn/data/topics/category-theory-emerging-use-in-ml.jsx) | ✓ / — / — | Manual teaching audit pending |
| [Differential Geometry & Riemannian Manifolds](../../../src/learn/data/topics/differential-geometry-riemannian-manifolds.jsx) | ✓ / — / — | Manual teaching audit pending |

### Classical Machine Learning

| Lesson | O / V / S | Status |
| --- | --- | --- |
| [Linear & Logistic Regression](../../../src/learn/data/topics/linear-logistic-regression.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Decision Trees & Random Forests](../../../src/learn/data/topics/decision-trees-random-forests.jsx) | — / — / — | Manual teaching audit pending |
| [K-Nearest Neighbors (KNN)](../../../src/learn/data/topics/k-nearest-neighbors-knn.jsx) | — / — / — | Manual teaching audit pending |
| [Gradient Boosted Trees (XGBoost, LightGBM, CatBoost)](../../../src/learn/data/topics/gradient-boosted-trees-xgboost-lightgbm-catboost.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Support Vector Machines (SVM)](../../../src/learn/data/topics/support-vector-machines-svm.jsx) | — / — / — | Manual teaching audit pending |
| [Naive Bayes & Probabilistic Classifiers](../../../src/learn/data/topics/naive-bayes-probabilistic-classifiers.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Ensemble Methods & Stacking](../../../src/learn/data/topics/ensemble-methods-stacking.jsx) | — / — / — | Manual teaching audit pending |
| [Recommender Systems (Collaborative Filtering, Matrix Factorization)](../../../src/learn/data/topics/recommender-systems-collaborative-filtering-matrix-factorization.jsx) | — / — / — | Manual teaching audit pending |
| [Multi-Label & Multi-Output Learning](../../../src/learn/data/topics/multi-label-multi-output-learning.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Survival Analysis (Cox Regression, Kaplan-Meier, Hazard Models)](../../../src/learn/data/topics/survival-analysis-cox-regression-kaplan-meier-hazard-models.jsx) | — / — / ✓ | Manual teaching audit pending |
| [K-Means & Hierarchical Clustering](../../../src/learn/data/topics/k-means-hierarchical-clustering.jsx) | — / — / ✓ | Manual teaching audit pending |
| [PCA & Dimensionality Reduction](../../../src/learn/data/topics/pca-dimensionality-reduction.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Clustering Evaluation & Validation (Silhouette, ARI, NMI)](../../../src/learn/data/topics/clustering-evaluation-validation-silhouette-ari-nmi.jsx) | — / — / ✓ | Manual teaching audit pending |
| [DBSCAN & Density-Based Clustering](../../../src/learn/data/topics/dbscan-density-based-clustering.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Anomaly & Outlier Detection (Isolation Forest, One-Class SVM, LOF)](../../../src/learn/data/topics/anomaly-outlier-detection-isolation-forest-one-class-svm-lof.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Gaussian Mixture Models (GMM) & EM Algorithm](../../../src/learn/data/topics/gaussian-mixture-models-gmm-em-algorithm.jsx) | — / — / ✓ | Manual teaching audit pending |
| [t-SNE, UMAP & Manifold Learning](../../../src/learn/data/topics/t-sne-umap-manifold-learning.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Independent Component Analysis (ICA)](../../../src/learn/data/topics/independent-component-analysis-ica.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Non-Negative Matrix Factorization (NMF)](../../../src/learn/data/topics/non-negative-matrix-factorization-nmf.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Feature Scaling, Encoding & Imputation](../../../src/learn/data/topics/feature-scaling-encoding-imputation.jsx) | — / — / — | Manual teaching audit pending |
| [Cross-Validation & Hyperparameter Tuning](../../../src/learn/data/topics/cross-validation-hyperparameter-tuning.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Regularization (L1, L2, Elastic Net, Dropout)](../../../src/learn/data/topics/regularization-l1-l2-elastic-net-dropout.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Feature Selection & Importance (SHAP, Permutation, Mutual Info)](../../../src/learn/data/topics/feature-selection-importance-shap-permutation-mutual-info.jsx) | — / — / — | Manual teaching audit pending |
| [Bias-Variance Tradeoff & Learning Curves](../../../src/learn/data/topics/bias-variance-tradeoff-learning-curves.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Imbalanced Learning (SMOTE, Cost-Sensitive Learning)](../../../src/learn/data/topics/imbalanced-learning-smote-cost-sensitive-learning.jsx) | — / — / ✓ | Manual teaching audit pending |
| [AutoML & Neural Architecture Search (NAS)](../../../src/learn/data/topics/automl-neural-architecture-search-nas.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Hidden Markov Models (HMM)](../../../src/learn/data/topics/hidden-markov-models-hmm.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Bayesian Networks & Causal Graphical Models](../../../src/learn/data/topics/bayesian-networks-causal-graphical-models.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Conditional Random Fields (CRF)](../../../src/learn/data/topics/conditional-random-fields-crf.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Gaussian Processes (GP)](../../../src/learn/data/topics/gaussian-processes-gp.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Semi-Supervised Learning (Label Propagation, Self-Training, Co-Training)](../../../src/learn/data/topics/semi-supervised-learning-label-propagation-self-training-co-training.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Active Learning](../../../src/learn/data/topics/active-learning.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Evaluation Metrics (Precision, Recall, F1, AUC-ROC, AP, R², MAE)](../../../src/learn/data/topics/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae.jsx) | — / — / ✓ | Manual teaching audit pending |
| [PAC Learning & VC Dimension](../../../src/learn/data/topics/pac-learning-vc-dimension.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Calibration & Conformal Prediction](../../../src/learn/data/topics/calibration-conformal-prediction.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Rademacher Complexity & Generalization Bounds](../../../src/learn/data/topics/rademacher-complexity-generalization-bounds.jsx) | — / — / ✓ | Manual teaching audit pending |

### Deep Learning Fundamentals & Architectures

| Lesson | O / V / S | Status |
| --- | --- | --- |
| [Perceptrons, Neurons & Activation Functions](../../../src/learn/data/topics/perceptrons-neurons-activation-functions.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Backpropagation & Automatic Differentiation](../../../src/learn/data/topics/backprop.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Loss Functions (CE, MSE, Focal, Contrastive, Triplet)](../../../src/learn/data/topics/loss-functions-ce-mse-focal-contrastive-triplet.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Batch/Layer/Group/RMS Normalization](../../../src/learn/data/topics/batch-layer-group-rms-normalization.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Transfer Learning & Fine-Tuning Strategies](../../../src/learn/data/topics/transfer-learning-fine-tuning-strategies.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Weight Initialization (Xavier, Kaiming, μP)](../../../src/learn/data/topics/weight-initialization-xavier-kaiming-p.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Residual Connections & Skip Connections](../../../src/learn/data/topics/residual-connections-skip-connections.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Dropout, DropPath & Stochastic Depth](../../../src/learn/data/topics/dropout-droppath-stochastic-depth.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Convolution, Pooling & Receptive Fields](../../../src/learn/data/topics/convolution-pooling-receptive-fields.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Landmark Architectures (LeNet → AlexNet → VGG → ResNet → EfficientNet)](../../../src/learn/data/topics/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Depthwise Separable & Dilated Convolutions](../../../src/learn/data/topics/depthwise-separable-dilated-convolutions.jsx) | — / — / — | Manual teaching audit pending |
| [ConvNeXt & Modern CNN Designs](../../../src/learn/data/topics/convnext-modern-cnn-designs.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Capsule Networks](../../../src/learn/data/topics/capsule-networks.jsx) | — / — / — | Manual teaching audit pending |
| [RNNs, LSTMs & GRUs](../../../src/learn/data/topics/rnns-lstms-grus.jsx) | — / — / — | Manual teaching audit pending |
| [Sequence-to-Sequence & Encoder-Decoder](../../../src/learn/data/topics/sequence-to-sequence-encoder-decoder.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Attention Mechanism (Bahdanau, Luong)](../../../src/learn/data/topics/attention.jsx) | — / — / — | Manual teaching audit pending |
| [State Space Models (S4, Mamba, Mamba-2)](../../../src/learn/data/topics/state-space-models-s4-mamba-mamba-2.jsx) | — / — / ✓ | Manual teaching audit pending |
| [RWKV & Linear Attention Models](../../../src/learn/data/topics/rwkv-linear-attention-models.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Self-Attention & Multi-Head Attention](../../../src/learn/data/topics/self-attention-multi-head-attention.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Transformer Block Architecture](../../../src/learn/data/topics/transformer-block-architecture.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Positional Encodings (Sinusoidal, Learned, RoPE, ALiBi)](../../../src/learn/data/topics/positional-encodings-sinusoidal-learned-rope-alibi.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)](../../../src/learn/data/topics/grouped-query-attention-gqa-multi-query-attention-mqa.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Multi-Head Latent Attention (MLA)](../../../src/learn/data/topics/multi-head-latent-attention-mla.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Sparse & Linear Attention Variants](../../../src/learn/data/topics/sparse-linear-attention-variants.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Vision Transformers (ViT, DeiT, Swin, DiNOv2)](../../../src/learn/data/topics/vision-transformers-vit-deit-swin-dinov2.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Mixture-of-Experts Transformers (MoE)](../../../src/learn/data/topics/mixture-of-experts-transformers-moe.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Interleaved / Cross-Attention Architectures](../../../src/learn/data/topics/interleaved-cross-attention-architectures.jsx) | — / — / — | Manual teaching audit pending |
| [Message Passing & Graph Convolutions (GCN, GAT, GraphSAGE)](../../../src/learn/data/topics/message-passing-graph-convolutions-gcn-gat-graphsage.jsx) | — / — / — | Manual teaching audit pending |
| [Graph Transformers & Geometric Deep Learning](../../../src/learn/data/topics/graph-transformers-geometric-deep-learning.jsx) | — / — / — | Manual teaching audit pending |
| [Boltzmann Machines & Restricted Boltzmann Machines (RBM)](../../../src/learn/data/topics/boltzmann-machines-restricted-boltzmann-machines-rbm.jsx) | — / — / — | Manual teaching audit pending |
| [Spectral Normalization & Gradient Penalty](../../../src/learn/data/topics/spectral-normalization-gradient-penalty.jsx) | — / — / — | Manual teaching audit pending |
| [Modern Hopfield Networks](../../../src/learn/data/topics/modern-hopfield-networks.jsx) | — / — / — | Manual teaching audit pending |
| [xLSTM (Extended LSTM)](../../../src/learn/data/topics/xlstm-extended-lstm.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Hyena & Long Convolution Models](../../../src/learn/data/topics/hyena-long-convolution-models.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Ring Attention & Sequence Parallelism](../../../src/learn/data/topics/ring-attention-sequence-parallelism.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Advanced Optimizers (Lion, Sophia, Prodigy, Schedule-Free)](../../../src/learn/data/topics/advanced-optimizers-lion-sophia-prodigy-schedule-free.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Neural ODE & Continuous-Depth Models](../../../src/learn/data/topics/neural-ode-continuous-depth-models.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Hybrid SSM-Transformer Architectures (Jamba)](../../../src/learn/data/topics/hybrid-ssm-transformer-architectures-jamba.jsx) | — / — / — | Manual teaching audit pending |
| [Titans (Multi-Memory Architecture)](../../../src/learn/data/topics/titans-multi-memory-architecture.jsx) | — / — / ✓ | Manual teaching audit pending |

### Large Language Models — Architecture, Training & Inference

| Lesson | O / V / S | Status |
| --- | --- | --- |
| [Byte-Pair Encoding (BPE), WordPiece, SentencePiece, Unigram](../../../src/learn/data/topics/tokenization.jsx) | — / — / — | Manual teaching audit pending |
| [Byte-Level Tokenization & Token-Free Models](../../../src/learn/data/topics/byte-level-tokenization.jsx) | — / — / — | Manual teaching audit pending |
| [Vocabulary Design & Multilingual Tokenization](../../../src/learn/data/topics/vocabulary-multilingual.jsx) | — / — / — | Manual teaching audit pending |
| [Multimodal Tokenization (Visual, Audio, Video)](../../../src/learn/data/topics/multimodal-tokenization.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Dynamic Tokenization (ADAT, BoundlessBPE, LiteToken)](../../../src/learn/data/topics/dynamic-tokenization.jsx) | — / — / — | Manual teaching audit pending |
| [Causal Language Modeling (Next-Token Prediction)](../../../src/learn/data/topics/causal-language-modeling.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Masked Language Modeling (BERT-style)](../../../src/learn/data/topics/masked-language-modeling.jsx) | — / — / — | Manual teaching audit pending |
| [Data Curation & Deduplication (MinHash, Bloom Filters)](../../../src/learn/data/topics/data-curation-deduplication.jsx) | — / — / — | Manual teaching audit pending |
| [Scaling Laws (Kaplan, Chinchilla, Beyond)](../../../src/learn/data/topics/scaling-laws.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Curriculum Learning & Data Mixing Strategies](../../../src/learn/data/topics/curriculum-data-mixing.jsx) | — / — / ✓ | Manual teaching audit pending |
| [FP8 Training & Low-Precision Pre-Training](../../../src/learn/data/topics/fp8-training.jsx) | — / — / — | Manual teaching audit pending |
| [MoE Training & Expert Load Balancing](../../../src/learn/data/topics/moe-training.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Multimodal Pre-Training (Vision Encoders, Cross-Modal Alignment)](../../../src/learn/data/topics/multimodal-pretraining.jsx) | — / — / — | Manual teaching audit pending |
| [Data Curation Pipelines (Curator Models, Quality Filtering)](../../../src/learn/data/topics/data-curation-pipelines.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Synthetic Data Generation for Pre-Training](../../../src/learn/data/topics/synthetic-data-pretraining.jsx) | — / — / — | Manual teaching audit pending |
| [Supervised Fine-Tuning (SFT)](../../../src/learn/data/topics/supervised-fine-tuning.jsx) | — / — / ✓ | Manual teaching audit pending |
| [RLHF (Reinforcement Learning from Human Feedback)](../../../src/learn/data/topics/rlhf.jsx) | — / — / ✓ | Manual teaching audit pending |
| [DPO (Direct Preference Optimization)](../../../src/learn/data/topics/dpo.jsx) | — / — / ✓ | Manual teaching audit pending |
| [SimPO (Simple Preference Optimization)](../../../src/learn/data/topics/simpo.jsx) | — / — / ✓ | Manual teaching audit pending |
| [P-Tuning & Soft Prompt Methods](../../../src/learn/data/topics/p-tuning-soft-prompts.jsx) | — / — / ✓ | Manual teaching audit pending |
| [GRPO, RLOO, KTO & Advanced Preference Methods](../../../src/learn/data/topics/grpo-rloo-kto.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Constitutional AI (CAI)](../../../src/learn/data/topics/constitutional-ai.jsx) | — / — / — | Manual teaching audit pending |
| [Process Reward Models (PRM) vs Outcome Reward Models (ORM)](../../../src/learn/data/topics/prm-vs-orm.jsx) | — / — / ✓ | Manual teaching audit pending |
| [RLAIF, IPO, ORPO & Emerging Alignment Methods](../../../src/learn/data/topics/rlaif-ipo-orpo.jsx) | — / — / ✓ | Manual teaching audit pending |
| [RLVR (Reinforcement Learning with Verifiable Rewards)](../../../src/learn/data/topics/rlvr.jsx) | — / — / ✓ | Manual teaching audit pending |
| [DAPO (Dynamic Adaptive Policy Optimization)](../../../src/learn/data/topics/dapo.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Knowledge Distillation for LLMs (DeepSeek-R1-Distill, CoT Distillation)](../../../src/learn/data/topics/knowledge-distillation-llms.jsx) | — / — / ✓ | Manual teaching audit pending |
| [RL for Reasoning (DeepSeek-R1 Style)](../../../src/learn/data/topics/rl-for-reasoning.jsx) | — / — / — | Manual teaching audit pending |
| [KV-Cache & Memory Management](../../../src/learn/data/topics/kv-cache.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Decoding Strategies (Greedy, Beam, Top-k, Top-p, Temperature)](../../../src/learn/data/topics/decoding-strategies.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Structured Output & Constrained Decoding (Outlines, XGrammar)](../../../src/learn/data/topics/constrained-decoding.jsx) | — / — / — | Manual teaching audit pending |
| [Continuous Batching & PagedAttention](../../../src/learn/data/topics/continuous-batching.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Queueing Theory for LLM Serving](../../../src/learn/data/topics/queueing-theory-llm-serving.jsx) | — / — / — | Manual teaching audit pending |
| [Speculative Decoding](../../../src/learn/data/topics/speculative-decoding.jsx) | — / — / — | Manual teaching audit pending |
| [Prefix Caching & Prompt Caching](../../../src/learn/data/topics/prefix-caching.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Inference Cost Economics & Compute Scaling](../../../src/learn/data/topics/inference-cost-economics.jsx) | — / — / — | Manual teaching audit pending |
| [Test-Time Compute Scaling](../../../src/learn/data/topics/test-time-compute.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Inference Engines & Serving](../../../src/learn/data/topics/inference-engines.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Inference System Architecture (End-to-End)](../../../src/learn/data/topics/inference-system-architecture.jsx) | — / — / — | Manual teaching audit pending |
| [Request Routing & Load Balancing](../../../src/learn/data/topics/request-routing-load-balancing.jsx) | — / — / — | Manual teaching audit pending |
| [Autoscaling & GPU Resource Management](../../../src/learn/data/topics/autoscaling-gpu.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Disaggregated Prefill & Decode](../../../src/learn/data/topics/disaggregated-prefill-decode.jsx) | — / — / — | Manual teaching audit pending |
| [Caching Strategies (Semantic, Exact, KV-Cache Sharing)](../../../src/learn/data/topics/caching-strategies.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Multi-Model Serving & Model Routing](../../../src/learn/data/topics/multi-model-serving.jsx) | — / — / — | Manual teaching audit pending |
| [Rate Limiting, Quota Management & Fairness](../../../src/learn/data/topics/rate-limiting.jsx) | — / ✓ / — | Manual teaching audit pending |
| [Guardrails, Input/Output Filtering & Safety Layers](../../../src/learn/data/topics/guardrails.jsx) | — / — / — | Manual teaching audit pending |
| [Observability & LLM Monitoring](../../../src/learn/data/topics/observability-llm.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Streaming & Server-Sent Events (SSE)](../../../src/learn/data/topics/streaming-sse.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Cost Optimization & TCO Analysis](../../../src/learn/data/topics/cost-optimization-tco.jsx) | — / — / — | Manual teaching audit pending |
| [Edge & On-Premise Deployment Architectures](../../../src/learn/data/topics/edge-on-premise.jsx) | — / — / — | Manual teaching audit pending |
| [Multi-Region & Global Inference Infrastructure](../../../src/learn/data/topics/multi-region-global.jsx) | — / — / — | Manual teaching audit pending |
| [Context Window Extension (RoPE Scaling, YaRN, NTK-Aware)](../../../src/learn/data/topics/context-window-extension.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Retrieval-Augmented Generation (RAG)](../../../src/learn/data/topics/rag.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Embedding Models & Vector Databases](../../../src/learn/data/topics/embedding-models-vector-db.jsx) | — / — / — | Manual teaching audit pending |
| [GraphRAG & Agentic RAG](../../../src/learn/data/topics/graphrag-agentic.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Model Merging (TIES, DARE, Model Soups, SLERP)](../../../src/learn/data/topics/model-merging.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Hybrid Search (Dense + Sparse + Reranking)](../../../src/learn/data/topics/hybrid-search.jsx) | — / — / — | Manual teaching audit pending |

### Programming & Scientific Computing

| Lesson | O / V / S | Status |
| --- | --- | --- |
| [Python Basics: Types, Control Flow, Functions & Modules](../../../src/learn/data/topics/python-basics-types-control-flow-functions-modules.jsx) | — / — / — | Manual teaching audit pending |
| [Object-Oriented Programming in Python](../../../src/learn/data/topics/object-oriented-programming-in-python.jsx) | — / — / — | Manual teaching audit pending |
| [Iterators, Iterables & Generators](../../../src/learn/data/topics/iterators-iterables-generators.jsx) | — / — / — | Manual teaching audit pending |
| [Decorators & Context Managers](../../../src/learn/data/topics/decorators-context-managers.jsx) | — / — / — | Manual teaching audit pending |
| [Testing, Debugging & Dependency Management](../../../src/learn/data/topics/testing-debugging-dependency-management.jsx) | — / — / — | Manual teaching audit pending |
| [NumPy: Arrays, Broadcasting & Vectorization](../../../src/learn/data/topics/numpy-arrays-broadcasting-vectorization.jsx) | — / — / — | Manual teaching audit pending |
| [Pandas: Data Wrangling, Joins & Grouping](../../../src/learn/data/topics/pandas-data-wrangling-joins-grouping.jsx) | — / — / — | Manual teaching audit pending |
| [Matplotlib & Scientific Plotting](../../../src/learn/data/topics/matplotlib-scientific-plotting.jsx) | — / — / — | Manual teaching audit pending |
| [Reproducible Notebooks & Experiment Structure](../../../src/learn/data/topics/reproducible-notebooks-experiment-structure.jsx) | — / — / — | Manual teaching audit pending |
| [Code Documentation, Type Hints & API Design](../../../src/learn/data/topics/code-documentation-type-hints-api-design.jsx) | — / — / — | Manual teaching audit pending |
| [Git, GitHub & Collaborative Version Control](../../../src/learn/data/topics/git-github-collaborative-version-control.jsx) | — / — / ✓ | Manual teaching audit pending |
| [Linux Basics, Filesystems & Processes](../../../src/learn/data/topics/linux-basics-filesystems-processes.jsx) | — / — / — | Manual teaching audit pending |
| [Bash Scripting & Command-Line Automation](../../../src/learn/data/topics/bash-scripting-command-line-automation.jsx) | — / — / — | Manual teaching audit pending |

