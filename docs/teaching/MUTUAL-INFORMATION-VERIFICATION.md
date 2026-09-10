# Mutual Information & Information Bottleneck — implementation and verification

10 September 2026. Stable ID `mutual-information-information-bottleneck`, Mathematics Foundations position 27. This is an author implementation record; root integration, independent review and user acceptance are separate states. The original topic identity, module order and published route are preserved.

## What changed and why

The original full lesson was read and archived at `scratch/mutual-information-native-verification/original-lesson.jsx` (SHA256 `19ffffc408e548e1ce2ea86d9819d6e05948521ac8fdbba7dbde028810ae8787`). Its useful uncertainty interpretation, nonlinear-dependence warning, joint-table program, Markov/DPI idea, representation objective, VIB mention and estimation cautions are retained. The exact original program still prints `0.278 1.0 0.722`.

The main gap was that these claims were largely isolated statements. The new lesson teaches the mechanism and evidence behind each: conditional probabilities versus entropy, joint pairing versus independent marginals, a nonlinear map and XOR synergy, the actual chain-rule DPI proof, signal versus nuisance in a stochastic representation, finite IB computation and its failure to certify a global optimum, both variational gap identities, continuous-value limits, and sampled-table estimates versus known population truth. Ten independent practice tasks include changed numerical problems, experimental-design reasoning and a complete changed-label/noise bottleneck project.

The title remains accurate; no unrelated scope was added to its name. [The individual design](MUTUAL-INFORMATION-LESSON-DESIGN.md) records prerequisite bridges and ownership. Entropy immediately precedes this topic; Rate-Distortion immediately follows. The closed-form finite IB update has a block-descent guarantee, not the fixed-distortion convex global certificate being taught by the Rate-Distortion author. The [InfoNCE destination note](topic-notes/contrastive-learning-simclr-moco-infonce-loss.md) preserves a better-owned finite contrastive experiment and its sampling caveats. The destination inventory confirmed that topic is planned; its proposed experiment is not claimed as implemented here.

## Source ownership

- `src/learn/data/topics/mutual-information-information-bottleneck.jsx`: complete narrative, derivations, nine executable examples, ten independent problems with separate hints and explained solutions, annotated resources and actual next-topic bridge.
- `src/learn/data/mutual-information-models.js`: finite laws, immutable information states, stochastic encoder comparisons/updates, exact variational gaps, deterministic bounded simulation and Gaussian model values.
- `src/learn/data/mutual-information-examples.js`: nine self-contained standard-library Python programs with executed expected output.
- `src/learn/components/lesson-labs/MutualInformationLabs.jsx` and `mutual-information-labs.css`: topic-specific probability, assignment, information-plane, bound and count representations.
- `src/learn/data/curriculum/blueprints/mutual-information-information-bottleneck.js`: individual teaching plan, registered by the root agent.

No shared manifest, blueprint index, global CSS, route order or progress source was edited by this author. Math is imported directly from its explicit component; no new dependency was installed. Models and checks were formatted with Babel only within ownership; normalized ASTs, comments and literal text were conserved (`scratch/format-mutual-information.cjs`, original snapshots in `scratch/mutual-information-format-baseline`). Small subsequent visual-label refinements are explicitly represented in the final source snapshot.

## Visual contracts and actual investigation behavior

| Representation | Learner action and linked evidence | Accuracy/accessibility contract |
| --- | --- | --- |
| Joint information | Change prevalence and channel error; select an actual cell; compare its joint and independent-reference probabilities, pointwise log ratio, contribution and conditional label bars. | Bits, zero cells contribute zero without evaluating log zero; zero-mass rows have explicitly undetermined conditional laws. Probability bars do not pretend to be entropy bars. Inverted, independent and degenerate laws are distinguished. |
| Nonlinear map and conditional information | Trace equally likely readings −1,0,1 to their squared labels; reveal A, B or both in XOR; change B’s bias and inspect each conditional slice. | Code-generated node/edge geometry and exhaustive finite truth table. Covariance can vanish while MI is positive; conditioning can increase or decrease MI. Visible probabilities and text duplicate color meanings. |
| Processing figure | Follow Y–X–Z beside its explicit factorization and two chain-rule expansions. | Arrows describe a stochastic model, not causal evidence. Equality means zero remaining conditional label information. |
| Information plane | Switch constant/signal/nuisance/both/noisy encoders and beta; assignment strips, actual rate/relevance/objective and deterministic comparison table update together. | Finite X=(S,N), label error .1. The dashed noisy-signal curve is a declared candidate family, not a proof of the full optimal frontier. Coordinates are information bits, not tensor size or measured bytes. |
| Finite IB update | Step forward/back, inspect update 40, switch initialization or beta, reset; assignment and predictive bars derive from the same state. | Predictive KL values refer explicitly to the preceding decoder used by the update. Exponent uses nats; displayed values use bits. Maximum 40 finite updates. Symmetric/nuisance initializations expose inferior stationarity. No global convergence claim. |
| Variational bounds | Hold encoder fixed and change reference/decoder; inspect upper rate, lower relevance, both independently calculated KL gaps and objective bound. | Exact MI remains fixed; a poor decoder can give a negative lower bound. Shared axis rescales honestly; zero tick is omitted when it would collide with an endpoint. Shape and text distinguish bound from truth. |
| Gaussian/noise figure | Compare computed noise levels and information scale next to the analytic formula. | Values come from the declared independent Gaussian model. Sigma zero is a separate singular infinite-MI case, never a fabricated finite chart point. |
| Sample estimation | Change category count, sample budget, population or seed; inspect actual count grid and original estimate against 20 shuffled estimates. | Bounded simulator (2/4/8 categories, 20–2,000 observations, finite integer seed), reproducible prefix samples. Histogram bars count shuffles, numerical vertical ticks identify that count scale. Shuffling conserves empirical marginals; it is not presented as a confidence interval or calibrated hypothesis test. Invalid seed preserves the active sample. |

All controls are native keyboard-operable elements with visible focus and reset. Wide tables retain readable numbers in local horizontal scroll regions, with captions outside the scroll area and a narrow-screen hint; the page itself does not scroll horizontally. No timers, animation-generated values, learned model dependencies or invented benchmark measurements are used.

## Native and independent numerical evidence

Run `node scripts/verify-mutual-information-models.mjs`. It verifies JavaScript contracts, exports actual model states, then invokes `scripts/verify-mutual-information-native.py` with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14). This independent program uses NumPy/SciPy and high-precision Decimal rather than importing the lesson’s numerical helpers.

The final native run passed:

| Check | Actual coverage |
| --- | --- |
| Displayed standard-library programs | All nine complete programs executed unchanged and stdout matched exactly. |
| Independent changed-task project | The complete IB program was actually rerun after changing label error to .2 and beta to 4; informative final `(rate,relevance,J)=(.698380,.218212,−.174466)`, symmetric `(0,0,0)` and encoder rows match the explained solution. |
| Finite MI | 624 nonempty 2×2 integer-weight laws checked against both independent entropy differences and SciPy `rel_entr`, plus transpose/relabel invariance and boundedness. |
| Near-zero support | Positive joint mass `1e−310` has an underflowed independent product but a finite pointwise information value; difference-of-logs implementation and explicit zero-cell branch pass. |
| Conditional information | 18 biased XOR/reveal laws and exact conditional information; 27 changed-prevalence/noise processing laws independently enumerated to check the DPI gap identity. |
| Representation laws | 80 mode/noise/label-noise cases against closed-form binary-channel expressions. |
| IB computation | 63 complete 40-update traces; 2,583 full-state free-energy identities with deliberately mismatched auxiliary marginals/decoders. Row normalization, implied joint laws, predictions and objective descent checked. |
| Independent update minimization | 72 actual encoder rows compared with SciPy constrained scalar minimization of the original block objective; no use of the exponential formula as the oracle. |
| Variational bounds | 100 reference/decoder/beta cases, both exact gap identities and combined objective-gap signs/constants. |
| Simulated samples | 21 independently reconstructed sample/count tables; 420 Fisher–Yates permutations with exact conservation of both empirical marginals and independent histogram entropy values. Prefix reproducibility, positive seed and zero-derived PRNG seed handling checked. |
| Continuous Gaussian values | Eight scales, including `1e−310`, against 80-digit Decimal logarithms; sigma zero explicitly returns infinity. |
| Invalid contracts | 16 invalid shape/probability/step/initialization/reference/seed/size/scale inputs rejected. |

The largest observed absolute discrepancy, about `6.93e−9`, was in a numerical optimizer’s probability location; those comparisons allow `3e−8`. Objective comparisons use tighter tolerances, finite information identities normally `2e−10`, and displayed examples match exact formatted output. This is bounded numerical and analytical verification, not a proof of arbitrary floating-point robustness or every possible dataset.

Raw evidence lives in `scratch/mutual-information-native-verification/model-fixtures.json`, `results.json`, `verified-*.py` and `changed-task-project-output.txt`. The example preparation script is `scripts/prepare-mutual-information-examples.py`; rerunning it intentionally regenerates only this topic’s example file from actual execution.

## Sources actually checked

All checks below were performed 10 September 2026; no claim of full video playback is made.

- Original [Information Bottleneck paper](https://arxiv.org/html/physics/0004057), sections 3.1–3.3/equations 15–30: objective, predictive KL, normalization, coupled updates and non-joint-convex limitation. Apparent HTML transcription/index errors were not copied into the lesson’s independently derived equations.
- [Deep Variational Information Bottleneck](https://arxiv.org/html/1612.00410v7), section 3/equations 5–17: Markov condition, predictive lower bound, rate upper bound and reparameterized neural implementation. Historical experimental results were not promoted into general performance guarantees.
- [MIT 6.441 chapter 2](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/184197ca5d5418da2415d37e929860b9_MIT6_441S16_chapter_2.pdf): definitions/theorems for conditional divergence, MI, processing, finite versus non-discrete self-information, Gaussian and AWGN examples. Chapter 3 was opened during research but is not represented as a fully reviewed resource or necessary citation.
- [Contrastive Predictive Coding](https://arxiv.org/html/1807.03748v2), section 2.3: candidate sampling, positive-index classification, density ratio and stated expected-loss bound. The finite contrastive enumeration was routed as an unimplemented destination proposal.
- [McAllester–Stratos](https://proceedings.mlr.press/v108/mcallester20a/mcallester20a.pdf), introduction/theorem 1.1: distribution-free high-confidence sample-bound limitation. The lesson retains those qualifiers rather than asserting a ceiling for every estimator.
- [Bristol’s course index](https://comsm0075.github.io/2020-21.html), linked from the lecturer’s course home: exact lectures 4, 5 and 9, their YouTube targets, slides and prerequisite lecture sequence verified. Direct video fetches were unavailable; available search metadata and the lecturer’s own index supported the link/level descriptions, not any claim to have watched the recordings.

## Actual browser and reading review

`node scripts/verify-mutual-information-browser.cjs --require-fonts` passed on the real localhost 5173 route in Edge at 1440×1000, 390×1000 and 320×1000. The run has 15 grouped checks per viewport covering all six investigations, changes in inputs and connected values, invalid drafts, reset, actual keyboard slider changes and stepping, terminal-step disabling, symmetric/nuisance starts, exact/mismatched/negative bounds, independent/channel samples and sample size changes. It also checks all SVG text bounds/effective font sizes, all KaTeX displays, ten route headings, nine code examples and the independently opened practice solutions.

The first ordinary sandbox run could not fetch the site's public Google Fonts and honestly used fallbacks. Those screenshots exposed 11 overwide equations at 320px. The formulas were decomposed into mathematical steps with explicitly introduced shorthand; type was not shrunk to force fit. Two remaining narrow equations were split in a second pass. Captions were moved outside local table scroll regions, narrow scroll hints were added, long bound labels were shortened and colliding axis ticks were suppressed. The final original-font run used approved public-font network access, loaded Space Grotesk, had no failed requests/page errors, no page overflow, no equation overflow/KaTeX errors and no SVG labels outside the viewBox or below the checked 13.5px effective minimum.

`node scripts/review-mutual-information-reading.cjs` then passed at all three widths after the final changed-task practice and count-axis additions. It verifies all ten in-lesson hash destinations exist, the ten independent tasks are present, the changed-task solution opens with the keyboard, complete free-energy/bound/continuous sections render correctly, all equations still fit, and original fonts load without failed requests or page errors. At 320px, actual ArrowRight presses move the focused count grid horizontally while keeping page width fixed. The final mathematical/model source is unchanged by the later histogram tick addition; its numeric vertical count scale was personally inspected in the final rendered page.

Screenshots were actually opened and read, not only saved. Inspected examples include the negative joint cell at 390px; both-feature conditional view at 1440px; all-input and noisy-signal information planes at 1440/320px; first-update assignments/decoder and prior distortion at 1440/390px; negative and extreme-reference bounds at 320px; sparse count grids and keyboard scrolling at 320px; nonlinear and Gaussian figures at 320px; and ordinary-reading captures of the opening, representation introduction, estimation introduction, finite recomputation, full free-energy identity, variational gaps, changed-task solution and continuous limit. Ordinary page captures retain the site navigation; isolated element screenshots temporarily hide only the fixed navigation to prevent an unrelated overlay from obscuring the component.

Reproducible raw files:

- `scratch/mutual-information-browser-review/`: initial fallback/repair evidence, not the final font-enabled result.
- `scratch/mutual-information-browser-review-fonts/results.json` and `geometry-{width}.json`: full font-enabled interaction/geometry checks and screenshots.
- `scratch/mutual-information-browser-review-fonts/final-reading-results.json`: final source reading/hash-target/practice/keyboard-scroll evidence.
- [Durable combined evidence](evidence/mutual-information-verification.json): final six source hashes, native results, full browser results and final reading results.

`node scripts/verify-curriculum.mjs` passed with 28 modules, 1,218 stable topics, 327 individual briefs and seven paths at the final author check. The root agent owns the production build, runtime payload/recovery integration and shared ledger. No claim of a completed production build or independent review is made by this author record. User acceptance and observed first-time-learner outcomes remain pending.

## Author source freeze

Final author production-source SHA256 values (also stored in `scratch/mutual-information-native-verification/final-source-hashes.json`):

| Source | SHA256 |
| --- | --- |
| Lesson body | `5e0e545b07dafdae348914e655c2b2ffe403a3d908125f73ac1910aa67663786` |
| Pure models | `f3c40318bf0235e545f6882f40c92c6bafdbfff622f041b493d444181cb3540a` |
| Native examples | `29574e13ad103256ca0ebd3df5970cc72c825c2b832e92b438ad7119922aca35` |
| Lab components | `dba25a9eb1bd8f15e69299c179e03636fc5ae60242ed5d381eee4daf51b03d1a` |
| Scoped CSS | `9c99aba160c47544cf017c572c9e950cbc2c9f261b342864569a20899555f3c9` |
| Individual blueprint | `9ad67b0e4c345a1258cfab29ba72dd1c0cdd0c9f5a6875540aa151158b1ced05` |

Source complete, native/model verified and author browser/reading reviewed. Await independent review and root integration; amend this record and hashes if a subsequent concrete defect requires a correction.

## Independent-review prompt amendment

The independent reviewer found that the nine stored program questions were not displayed by the shared runnable-example component. A topic-owned wrapper now renders each question immediately before its program; question metadata spacing was corrected in both the generator and generated examples. No executable program, expected output, model, lab, CSS or blueprint changed. Regenerating the examples executed all nine programs again, and the focused browser check additionally compared every code/output/title against the previous snapshot.

`node scripts/review-mutual-information-prompts.cjs` passed at 1440/390/320 with original public fonts loaded: all nine exact questions precede the correct program, fit their reading column, and produce no page errors, failed requests, equation overflow or document overflow. The 390px ordinary-reading screenshot was opened and inspected. Evidence is `scratch/mutual-information-browser-review-fonts/prompt-review-results.json`; the durable combined JSON includes this amendment and the updated body/examples fingerprints. Earlier full model and interactive-lab evidence remains applicable because those sources are unchanged.
