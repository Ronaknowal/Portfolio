# Naive Bayes: full concept-by-concept review

26 September 2026. Entire production lesson, all examples' explanations, advanced branches and all changed practice tasks read. Direct JSX remains authoritative; canonical example programs and recorded outputs unchanged.

| Location / new idea | Assessment and revision |
| --- | --- |
| §1 observation/representation, baseline and data ownership | Existing message example and table sufficient; retained. |
| §1 generative reversal → conditional independence | Existing class-to-observation figure retained. Added equal-size two-class mixture showing independent features within each class yet pooled dependence. This addresses the conditional qualifier with actual numbers. |
| §2 likelihood → weight → posterior → odds | Exact one-word fractions already expose normalization. Retained. Added full max-shift/exponential/sum/divide intermediates for the stable −1001/−1000 example. |
| §3 documents versus tokens → multinomial event → coefficient | Existing corpus diagram, ordered AB versus count-event example and totals sufficient. Retained. |
| §3 zero count → smoothing → MLE/posterior mean → OOV | Added prior-strength versus data-volume explanation immediately before Dirichlet interpretation. Existing live token lab and support/OOV distinctions retained. |
| §4 count versus binary occurrence → absence → missing | Existing live presence lab sufficient for repetition. Added two-branch absent/unobserved likelihood diagram to teach marginalization and the required observation-process assumption. |
| §4 categorical support | Explicit per-column taxonomy, denominator and reserve-support example retained. |
| §5 density versus probability → Gaussian estimates → floor | Existing interval/density lab and exact variance calculation sufficient. Retained. |
| §5 score quadratic → shared-variance cancellation → model comparison | Existing calculated circle lab and independent density check supply the visual/formal correspondence. Retained complete derivation and likelihood-versus-conditional-fitting distinction. |
| §6 copied evidence and interaction-only signal | Live copied-alarm counterexample and full XOR enumeration sufficient; retained. |
| §7 complement pooling → sign → normalization → family choice | Existing pooled-source figure, exact class score and two normalization programs sufficient. Retained. |
| §8 sparse pipeline → cost → absent-feature sparse algebra | Existing full workflow and explicit all-absent bias derivation retained. |
| §8 accumulated counts → streaming → drift/prior shift | Complete remainder-safe streaming example, sufficient statistics and prior correction already locally explained. Retained. |
| §9 reliability → proper loss → calibrator data → threshold cost | Existing live reliability lab, exact Brier identity, ownership diagram, measured calibration and cost outcomes retained. |
| §10 count score → linear logit → TF–IDF | Representation/objective distinction explicitly explained; retained. |
| §10 plug-in table → joint integrated prediction | Added parameter-unit/conditional-step diagram for (4,1) posterior shapes: AA is .64 under a fixed table and 2/3 after integrating shared uncertainty. Supplies the bridge to rising products rather than replacing the full law. |
| §11 independent transfer | All tasks/hints/solutions and actual changed report retained. New missing-value and integrated-token figures support existing changed inputs without revealing a gated lab answer. |

## Visual specifications

`ConditionalMixtureFigure`: equal priors, independently generated A/B within class; each is on with probability .2 in class 0 and .8 in class 1. Class fork has separate A/B branches; pooled single marginals .5 but joint .34. No area encoding is implied by the HTML panel dimensions. Calculation explicitly identifies hidden-group mixing as the reason, distinct from the later copied-sensor counterexample.

`AbsentVersusMissingFigure`: fault prior .2, positive likelihoods .8/.4. Observed negative gives weights .04/.48 and posterior 1/13. Unmeasured value is summed out, contributing 1 under each class and preserving .2, conditional on uninformative missingness. Parallel semantic-HTML routes stack on narrow screens; neither branch is an inert pseudo-control.

`IntegratedTokenFigure`: shapes (4,1) shown as labeled pseudo-count units. Fixed table stays 4/5 on both steps; integrated joint factorization has second factor 5/6 conditional on the first hypothetical A. State labels explicitly distinguish this conditional calculation from feeding an observed test label into training. Original exact Dirichlet-multinomial derivation remains below it.

## Research and author evidence

Read [Stanford IR book, Naive Bayes text classification](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html) through its worked smoothing and log-score example and [scikit-learn's family guide](https://scikit-learn.org/stable/modules/naive_bayes.html). Used the book's separation of evidence, fitting and computation as a comparison. New mixture/alarm/token diagrams are original local calculations, not reproduced figures. Existing primary Complement NB, spam-filtering and generative-learning references remain; no fresh full video read is claimed.

Author checks parse source, enumerate conditional/pool probabilities, verify missing versus absent posterior, stable score normalization and integrated-versus-plug-in token arithmetic. Source hashes in `author-checks.json`. No native fits rerun. Root independent/browser review is pending and separate.

Final bounded live-instruction cleanup: two retained NaiveBayesLabs paragraphs now invite adding the opposing word and moving the Gaussian reading to observe their contributions. No prediction-before-use language, state/control/engine change or new experiment. The topic-specific lab is included in the author source set for final browser/independent binding.
