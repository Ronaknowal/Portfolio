# Naive Bayes & Probabilistic Classifiers — scoped teaching design

Prepared 11 September 2026. Status: complete author implementation and numerical/browser review; exact freeze is in evidence/naive-bayes-author-review.json. Independent review and production integration are separate. Classical Machine Learning position 6 of the authorized first-ten rewrite. Preserve stable ID naive-bayes-probabilistic-classifiers, title, publication source and module order. Next is Ensemble Methods & Stacking. Parent owns registration and integration.

## Starting evidence and preservation

Read the entire 899-line original body, six Python blocks, figures, six exercises and references. Original source: 59,687 bytes, SHA256 79e60503d2ecbdf3fc2faeb58274f5a21cf6ce09ba3ccce9d9d3b840bba1d9b9. The parent's ten-topic baseline and evidence/naive-bayes-original-review.json preserve exact source bytes. The latter preserves every actual rendered program, source location and fresh standalone stdout/error. Programs 1 and 3 execute; 2 and 4 contain cooked newline syntax errors; 5 and 6 depend on undefined earlier variables. Do not overwrite this evidence or describe all original snippets as executable. No topic-specific original support file exists: imports are shared content/viz primitives.

Ran exact inventory command with --topic naive-bayes-probabilistic-classifiers; saved scratch/naive-bayes-topic-plan.json. No existing individual brief or canonical destination note exists. Surfaced routing inbox contains resolved DSA history, not an unresolved Naive Bayes instruction. Read current handoff, teaching/design/code standards, ML domain playbook and .impeccable.md. Use the established dark/amber editorial presentation and frontend-design skill. Linux informs mechanism-first quality, not a component or lab quota.

## Scope, prerequisites and ownership

Retain the title: the center is the Naive Bayes family and responsible use of its model probabilities. It does not promise every probabilistic classifier. Teach generic Bayes scoring locally, then Multinomial, Bernoulli, Gaussian, Categorical and Complement mechanisms, working implementations, probability assessment and an end-to-end miniature workflow. Preserve conditional-independence, Gaussian/discriminative-boundary, streaming, failure diagnosis and calibration depth while repairing claims below.

Proposed required prerequisite: **Probability Distributions & Bayes' Theorem**, which already teaches conditional probability, discrete mass versus continuous density and mean/variance. Restate Bayes locally. Its copied-evidence example becomes a concrete classifier decision/risk counterexample here. Sums/logarithms get local refreshers. Python/NumPy are optional execution readiness links, not prerequisites for reading or operating the lesson. Earlier Linear & Logistic Regression supplies a comparison, but the local comparison defines its linear log-odds form. Do not require later calibration/cross-validation lessons to explain the core workflow. Optional Dirichlet integration links the reviewed Bayesian Inference & Conjugate Priors while restating the required identity.

Nearby owners inspected: Probability/Bayesian bodies teach dependence and Dirichlet prediction; Classical positions 1–5 own regression, trees/forests, KNN, boosting and SVM. Later Calibration & Conformal Prediction owns broader calibration and coverage; Cross-Validation & Hyperparameter Tuning owns nested model-selection generality. Naive Bayes still teaches a complete leakage-safe probability evaluation/calibration workflow. Ensemble Methods follows immediately and owns learned combination, not multiplication of purportedly independent evidence. Persist useful destination findings as their examples become concrete; do not rewrite unrelated bodies.

## Retain / repair / extend

| Original coverage | Decision and concrete repair |
| --- | --- |
| Generative class model, prior plus log likelihood | Retain; distinguish true model, estimated plug-in model, score, posterior and action. A prior is before this case's features, not necessarily before all training data. |
| Conditional independence and parameter reduction | Retain; joint factorization is stronger than pairwise independence. Explain independence within each class. |
| Multinomial counts and smoothing | Preserve three-document/five-word and eight-document/eight-word fixtures. Count tokens, not documents, in the denominator. Include the conditional-length multinomial coefficient and why it cancels. Count-vector components are not independent. Distinguish unseen declared words from OOV text. |
| From-scratch Gaussian and Multinomial code | Replace broken/misnamed APIs with complete programs and verified outputs. predict_log_proba must normalize; name joint scores honestly. Preserve original Gaussian seed and eight-word data as interpreted examples. |
| Bernoulli present/absent evidence | Deepen same-message comparison, explicit absence factors, its different denominator and missing-versus-absent distinction. Add Categorical NB: a separate category universe per column; integer codes are not counts/distances. |
| Gaussian boundary comparison | Replace invented polylines with actual analytic equal-score geometry and a separately executed fitted logistic comparison. Explain density/units, zero variance, MLE and smoothing. Shared diagonal variances cancel quadratic terms; separately fitted logistic/LDA models need not have the same coefficients. |
| Complement NB and imbalance | Derive complement counts, score sign and optional normalization. Verify sklearn norm=False and multiclass prior exception. Remove universal performance, rare-class rescue and fabricated 2–10-point gains. Larger raw counts do not automatically imply larger normalized likelihoods. |
| Library text/Gaussian examples | Preserve useful corpus and Gaussian fixture; repair train-only vocabulary, sparse handling, baseline, class coverage, setup and outputs. Tiny-toy accuracy is not deployment evidence. |
| Streaming | Retain sufficient statistics and complete partial_fit; fixed vocabulary/classes, all remainder rows, work proportional to data plus parameter refresh, no automatic forgetting. Separate count identity from floating-point and Gaussian chunk/smoothing behavior. |
| Calibration | Replace blanket overconfidence, guaranteed fixes and class-preservation claims. Show known-law failure, finite reliability bins/proper losses and a complete separate calibration/test protocol. Lower Brier/log loss alone is not proof of calibration. |
| Historical sources/applications | Keep verified Sahami cost connection and Rennie mechanism. Correct reversed email error costs; remove unsupported earliest/fastest/unique/popularity/benchmark claims. |
| Six self-check intentions | Retain recall, smoothing derivation, Gaussian comparison, drift diagnosis, rare-class diagnosis and probability workflow with changed inputs, hints, explained answers and numeric checks. Extend event-model/OOV and streaming diagnosis. |

## Learning route

1. **One message, one label, a decision.** Identify row, feature, label, prior-only baseline and deployment question. Introduce train/validation/test and fit-versus-transform before model complexity. The anchor is an invented message-triage teaching task.
2. **Competing weights.** Derive class prior times likelihood, normalization, log scores and odds. Define probability/density. Explain why an argmax can discard a class-independent constant while a score cannot be called a probability.
3. **Learn and spend token evidence.** Aggregate the retained small corpus, add alpha per vocabulary entry, compute the free-token posterior 56/67. Trace signed log-odds contributions. Derive token/count likelihood and smoothed estimator with its assumptions. Explain constrained MLE and posterior mean versus mode.
4. **Change the event being modeled.** Count repetition versus occurrence, absence and empty-message behavior. Explain categorical support and unknown-category policy. Marginalizing a missing feature is not silently converting it to zero.
5. **Measure continuous evidence.** Gaussian height/area, mean/spread, log likelihood and fitted sufficient statistics. Derive quadratic comparison, shared-variance cancellation and actual equal-score geometry. Class-conditional adequacy matters; preprocessing parameters belong to training.
6. **Test independence.** An alarm with prior .2 and positive rates .8/.4 is copied. True positive posterior stays 1/3; three naive copies report 2/3 and change action. Show XOR as information in a relationship and avoid implying linear logistic regression solves every interaction.
7. **Change classifier for a reason.** Derive three-class complement pooling and actual sklearn score conventions. Compare assumptions, costs and storage without timing rankings. Separate changed priors, limited rare-class data, asymmetric costs and misspecification.
8. **Build/update/evaluate the pipeline.** Complete source-owned corpus, train-only sparse vectorizer and internal selection; baseline and untouched test. Complete all-row streaming with class/feature schema. Explain drift and the conditional assumptions behind prior replacement.
9. **Check probabilities before using costs.** Exact truth versus finite empirical reliability; bins/counts/proper losses. Complete calibration and test protocol, derivation of asymmetric-error threshold and a possible class change after calibration. No clinical recommendation or universal guarantee.
10. **Deeper connections and independent report.** MNB linear log odds; TF–IDF as scoring rather than literal integer-count likelihood; plug-in versus integrated Dirichlet document prediction and resulting dependence. Bridge to out-of-fold ensemble combination.

These are route groups, not a section-count ceiling. Split where learning questions change. Essential derivations stay visible; optional branches deepen already explained foundations.

## Representation contracts

| Form / placement | Initial state, action and meaningful encoding | Evidence and boundary |
| --- | --- | --- |
| Inline observation-model fork and fit/inference flow, early core | Class fans out to features; fitted quantities and new observation travel on separate labeled paths. Introduce conditional-factorization meaning before operating. | Arrows encode model assumptions/data ownership, not discovered causation. Recompose vertically at narrow width. |
| Token/count correspondence and evidence ledger | Retained vocabulary free, money, win, meeting, agenda; initial free and alpha1. Apply edited message/alpha, step/back through prior and token contributions, reset. Signed bars share log-odds units. | Fraction/multinomial/sklearn checks; empty/OOV/repetition/zero-support cases. Explicit lowercase teaching tokenizer; calculations from invented corpus, not measured language prevalence. |
| Present/absent comparison | Same complete message as counts and binary slots. Change repetition/presence; binary absence terms visibly remain. Initial free. | Independently enumerate binary patterns and sklearn likelihoods. All-zero measured message differs from missing evidence. Exact companion values and text state. |
| Gaussian observation investigation | Known Normal(0,1) versus Normal(0,4), equal priors; move one reading. Density curves and a common-width interval connect height/area; posterior readout. | SciPy PDF/CDF, crossings ±1.3595559869; positive finite variance/range. Wider model wins both far tails. State exact versus height-times-width approximations. |
| Equal-score geometry | Retain centers (−2,−2)/(2,2); switch .5/.5 versus .5/2 variance, move probe. Equal case x+y=0; unequal circle center (−10/3,−10/3), radius4.0088171203. | Direct log-density equality along every sampled curve plus changed grid; known model parameters, not a falsely claimed fitted logistic line. Native fitted comparison is separate. |
| Copied-alarm view | One observed alarm branches into 1–5 recorded copies. Select positive/negative, compare actual joint model and product assumption, posterior and action. | Exact two-state law. Tie-to-normal: switch at3copies lowers true accuracy .8→.64. Exact counterexample, not universal degradation. |
| Complement pooling inline figure | Three class rows feed each class's other-row total; one smoothed row and score sign worked in place. | Independently sum and compare norm on/off current sklearn. Arrows mean data pooling. |
| Calibration ownership and reliability | Separate fit/calibration/test lanes. Deterministic observed probability/outcome points; change bins/predeclared dataset; inspect membership, mean prediction, positive fraction and count. | Rational bin/group/score oracle; p=0/1 endpoint policy. Empirical curve is not true calibration. Fitted experiment is a complete native program. |

Every lab has a prediction, meaningful changed-state transfer, visible errors/reset and accessible exact values. Introduce labels/units before use. Keep native readable plot labels and responsive composition; ordinary initial reading must work without discovering hidden states. Numbers of figures/labs follow mechanisms.

## Native programs and changed practice

Self-contained CPU Python programs keep useful original data and output meaning, with executable strings/current APIs. Record actual tested versions; sklearn1.9.1 is installed in the isolated runtime. No external dataset download, notebook state or prior-block variables. Setup precedes first program; explicitly render each example.question because shared RunnableExample ignores that field.

Executable jobs: stable score normalization/zero evidence; complete count NB on old eight-word corpus; Bernoulli/Categorical alternatives; Gaussian estimation retaining old seed/output; exact event/count and integrated prediction comparison; copy/XOR enumeration; complement weights/library; train-only sparse pipeline retaining old text corpus; all-row streaming versus batch; independent calibration/test and cost report; a changed protocol where a separate capstone is needed. Combine only when the workflow remains complete.

Independent tasks: changed prior/opposing tokens; unseen declared word versus OOV; absence versus repetition; transformed Gaussian units and equal variance; copied alarm with .1 prior and .9/.3 rates; complement score/sign; 14-row/four-row-batch bug; prior versus conditional shift; changed calibrated report; plug-in versus integrated prediction. Hints precede explained solutions; fixed inputs have exact fractions/accepted outputs. Open reports include an actual acceptable response and reproducible changed protocol, not criteria alone.

## Research / alternate resources (11 September 2026)

| Resource | Actually inspected and intended use |
| --- | --- |
| [sklearn1.9.1 NB guide](https://scikit-learn.org/stable/modules/naive_bayes.html) | Complete method guide; family/API orientation. Reconcile compressed wording with token event model; do not promote broad empirical remarks to guarantees. |
| [Stanford IR chapter](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html), [multinomial relation](https://nlp.stanford.edu/IR-book/html/htmledition/relation-to-multinomial-unigram-language-model-1.html) | Explanatory text, count example and relation page inspected. Alternate beginner/intermediate token/smoothing/cost explanation. Our data/wording/figures remain original. |
| [Rennie et al.2003](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) | Model/estimation, complement and normalization sections, equations1–6 and explicit non-generative interpretation inspected. Motivates mechanism; no unrun benchmark claim. |
| [Sahami et al.1998](https://cdn.aaai.org/Workshops/1998/WS-98-05/WS98-05-009.pdf) | Task/cost/evaluation text inspected. Blocking legitimate mail is higher-cost in their account; original lesson reverses it. Documented use does not transfer their results to our task. |
| [CS229 generative notes](https://cs229.stanford.edu/summer2023/cs229-notes2.pdf) | GDA/logistic, Bernoulli, smoothing and token-event sections inspected. Stronger alternate derivation route; informal performance remarks not adopted. |
| [Stanford Online Lecture5](https://www.youtube.com/watch?v=nt63k3bfXS0), [official2018 syllabus](https://cs229.stanford.edu/syllabus-autumn2018.html) | Direct official recording and relevant lecture position verified; substantive corresponding generative notes above read. No full-video viewing or invented timestamps. Recommend oral derivation after core; probability/linear algebra readiness, older code not API authority. |
| [ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html), [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html), [CategoricalNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html) | Parameter contracts inspected: default normalization/prior exceptions, scale-based epsilon, category support. Actual installed-source/program checks close semantics before freeze. |
| [Calibration guide](https://scikit-learn.org/stable/modules/calibration.html) | Reliability construction, proper-score caveat, example assumptions and held-out procedure inspected. Further actual API/source checks required for the native experiment. |

Current Stanford SLP 4.pdf was opened but points to a changed chapter: do not cite it as a verified NB chapter without checking current title. Unsupported Bayes/Maron priority/history is not essential and will not be perpetuated.

## Verification plan and evaluated fixtures

scripts/check-naive-bayes-design.py produced scratch/naive-bayes-design-fixtures.json using Fraction and sklearn. **Design-fixture checks only, not production verification:** MNB56/67, binary absence difference, Complement softmax difference, copy1/3→2/3, Gaussian crossings/circle and stable normalization of −1001/−1000.

Production checks require independently formulated bounded exhaustive count/binary/copy/score oracles, alpha/support/empty/OOV/invalid arithmetic, exact likelihood/Dirichlet identities, library comparisons and full stdout execution. Verify fit/inference/calibration data ownership, not only shapes. Dense-array/key validation, NaN/Infinity, all-zero likelihood support, extreme accepted arithmetic and rounded displays must be honest; reject unsupported arithmetic rather than show misleading values.

Actual-font Edge1440/390/320 QA must inspect all controls, meaningful changed/invalid states, keyboard/reset, question/answer/code/output rendering, anchors, formulas, sources and document overflow/errors. Open ordinary-reading and actual-model screenshots, examine labels/curve correspondence, and record paths/hashes. This is an author heuristic review, not a beginner study. Freeze semantic hashes only after final changes/checks; independent review and production integration are separate.


## Implemented assessment — 11 September 2026

The complete body retains the scoped coverage and supplies six distinct investigations, eleven executed complete programs, two early checkpoints and twelve independent practice tasks. Counts reflect the actual teaching mechanisms, not targets. Numerical/final-font browser and opened-reading evidence is in [NAIVE-BAYES-VERIFICATION.md](NAIVE-BAYES-VERIFICATION.md) and the exact author packet. Current installed calibration source inspection confirms the sigmoid receives GaussianNB probabilities; the separate temperature branch is not conflated with it. The changed report includes actual outputs and the empty highest-probability bin/action limitation. The original body and six original program responses remain archived; no original snapshot is overwritten. Destination findings were saved to canonical Calibration and Ensemble notes; those destinations remain open until their own scoped assessment.
