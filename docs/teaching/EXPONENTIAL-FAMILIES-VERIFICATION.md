# Exponential families and sufficient statistics — author verification

Author freeze: **10 September 2026, 16:04:22 UTC**. Stable identity `exponential-families-sufficient-statistics`, Mathematics22. Author implementation/native/browser verification is complete. Independent cross-review and root production integration are separate; user acceptance remains pending.

The [design record](EXPONENTIAL-FAMILIES-LESSON-DESIGN.md) describes scope, concept-specific representations, assumptions, inspected sources and disposition. The [durable evidence JSON](evidence/exponential-family-author-review.json) contains all 14 owned source/script/design/note fingerprints, actual native and browser result payloads, formatting checks and 17 actually opened final screenshot fingerprints. Transient screenshots and execution files remain under `scratch/exponential-family-browser` and `scratch/exponential-family-verification`; rerun the saved scripts when those local artifacts are unavailable.

## What changed and what was preserved

The previous complete 6501-byte body and its exact Python program/output are preserved in [the original-content record](evidence/exponential-family-original-content.json). The original Bernoulli six-of-eight program appears unchanged and passes exact source/output comparison. Its finite-interior contract is now explained, including the all-success divide-by-zero boundary. The old constant-rate weekday exercise is retained as a fully specified changed-model task with a hint and explained solution.

The 42,958-byte final body develops conditional sufficiency and factorization before introducing family notation, then derives normalization, means/covariance, moment matching, Gaussian spread/merging, feature-aware statistics and conjugate updates in declared coordinates/units. It retains the useful Bernoulli, Poisson, Gaussian, categorical, GLM and Bayesian connections while correcting unrestricted moment-matching and canonical-link overstatements. Nine complete Python programs have visible learner questions. Eight changed reasoning tasks plus a capstone each have a separate optional hint before the complete solution; the early sufficiency checkpoint has a short explanation.

Representation choices address distinct jobs: binary dataset fibers; an editable group-allocation comparison; finite normalization weights/probability bars; observed and modeled moments in a triangle; Gaussian number-line spread; feature→score→mean mapping; and paired probability/log-odds densities. Four are interactive investigations; three are immediately visible static figures. Counts describe this lesson, not a template. Graphs are calculated finite/analytic models with assumptions, axes and text equivalents, not invented measurements.

## Actual computational evidence

Final command: `scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-exponential-family.py`, passed **16:01:47 UTC**, using Python 3.12.14, NumPy 2.3.5 and SciPy 1.18.1. Saved result: `scratch/exponential-family-verification/native-results.json`, embedded durably in the author evidence.

| Verified contract | Actual independent evidence |
| --- | --- |
| Complete examples and preservation | Nine actual learner programs executed and matched their displayed stdout; preserved original source/output matched exactly. Actual learner functions also ran changed normalizer, count-fit, Gaussian-merge and exposure inputs. |
| Natural weights, moments, covariance and CGF shift | 135 finite-family configurations across base weights and parameters including ±100. Independent SciPy log-sum-exp/softmax and NumPy centered covariance, with diagonal relative checks that retain tiny positive variances. Maximum A discrepancy 4.44e−16. CGF shift checked against direct finite expectation. |
| Existence/boundary/empty count cases | 729 count triples, including missing categories and empty data. Positive-count inverse fits reproduce empirical probabilities and both moments; direct cross-entropy comparison certifies the finite optimum against a changed candidate. |
| Model-relative sufficiency | 512 direct rational-product likelihood comparisons for all 256 binary datasets under common and grouped probabilities; equal-total common-p ratios are 1; group-move ratio is 3/28. Exact conditional fibers and the parity counterexample are checked. |
| Prior coordinate transformation | 64 integer Beta priors. SciPy Beta CDF/density and independent quadrature over transformed intervals verify mass, densities and Jacobian. Maximum density discrepancy 5.55e−15. Finite displayed η-window mass is distinguished from total probability. |
| Gaussian formulas and summary merging | 100 independently evaluated normal log densities verify canonical signs/normalizer, maximum absolute log-density error 1.14e−13. Ninety summary/merge cases include offsets up to 1e10, checked against long-double centered sums; maximum M₂ discrepancy 4.62e−5 at that large offset is explicitly numerical error, not exact bitwise associativity. |
| Invalid inputs and changed practice | 11 model-contract failures tested; the numeric formatter preserves 1e−12 rather than returning zero. Independent rational/numeric checks cover weighted finite moments 86/225, two changed triangle fits, merged M₂ 20, feature-weighted count changes, exposure/hour-minute conversion and posterior Beta parameters. Conceptual constrained-score and capstone arguments were read, not presented as numerical theorems proved by test counts. |

The first covariance comparison encountered a 3.37e−19 floating cross-term residual in the matrix-product oracle where symmetry gives exact zero. The oracle now permits a small absolute off-diagonal tolerance while retaining stringent relative checks on positive diagonal variances; SciPy softmax avoids fake extreme-state variance from probability sums rounded away from1. This was a reference-comparison issue, not a suppressed negative variance in the model.

## Actual browser, reading and accessibility review

`node scripts/review-exponential-family.cjs` passed at **16:01:40 UTC**, using actual Edge through Playwright on the shared 5173 route at 1440×1000,390×1000 and320×1000. HMR sockets were closed in the review pages to isolate their loaded state from concurrent authoring. The result payload is in `scratch/exponential-family-browser/results.json` and the durable record.

- Ten real route anchors arrived at headings around 100px below the viewport top at each width; no hard-coded source-only link check substituted for navigation.
- Common/group model changes, ratio readouts, keyboard observation toggles, probability sliders, unequal totals and reset passed. A narrow select was widened so its chosen model is visible.
- Weight endpoints and reset, positive mean/variance readouts, exact fitted triangle coordinates, missing-category disabled fit, empty observations, keyboard parameter movement and reset passed.
- Uniform and skewed Beta priors, equal interval mass, differing density heights, fixed finite η window and keyboard reset passed.
- All nine questions, complete displayed source and stdout blocks matched source data. All nine substantial tasks kept solutions closed when hints opened and supported keyboard disclosure. Five annotated external resources retain the learner’s place in a new tab.
- All 15 display equations and all figure/lab boxes fit the narrow reading area without hidden mathematical clipping. No page overflow, SVG text outside viewboxes, KaTeX errors or page exceptions remained. Long code has an explicit horizontal scroll container instead of forcing page overflow.
- Ordinary reading was captured before operating the investigations: route, section introductions, every static figure and every initial investigation. The final pass produced 80 captures across three widths; 17 selected final screenshots were actually opened, including all representation types, fitted/boundary states, changed exercise/solution, original output, resources and the three repaired 320px equations. The durable list distinguishes opened images from merely captured ones.

Real defects fixed during review: range step 0.01 sanitized an internally exact fit 0.202732… to a control value 0.2; using `step="any"` preserves the exact fitted coordinate and keyboard updates. Three 320px equations needed line breaks/shorter equivalent notation. The triangle’s x-axis title needed separation from its endpoint tick; the Gaussian caption baseline needed viewbox space. The 390px model selector now occupies a full row. These are final-source repairs with a fresh successful full browser rerun.

Formatting used `node scripts/format-exponential-family.cjs` at 16:01:11 UTC. Normalized JavaScript AST/string conservation and CSS rule/value conservation passed. Topic-owned imports retain on-demand loading; no all-topic registry, global stylesheet, curriculum order or shared route was edited by this author.

## Research, ownership and limitations

Read the official MIT sufficiency PDF’s conditional/factorization treatment, Berkeley’s full exponential-family page, Stanford’s normalizer/MLE/minimality/reference-category sections, and Geyer’s conjugate/properness sections. Reviewed MIT Rigollet Lecture 21’s official resource page and transcript examples about links and family decomposition; did not claim full audiovisual playback. Direct YouTube embed retrieval failed in the tool, so the working official MIT page is the learner link. Specific transcription/convention inconsistencies were resolved using independently derived formulas, rather than copied into the lesson. The design records exact scope and links.

No incoming destination note existed. A scoped [Entropy/KL destination note](topic-notes/entropy-cross-entropy-kl-divergence.md) preserves the maximum-entropy/KL certificate for an author who can first teach entropy, plus a concrete inconsistent existing classifier-count exercise. Those proposals remain open for reassessment; this work did not rewrite that destination. Minimal sufficiency is distinguished locally from minimal representation; a full completeness/Rao–Blackwell/UMVU course and unrestricted unbounded-family boundary theory are not claimed. The next full-module bridge remains Measure Theory & Probability Spaces.

The finite controls do not represent every possible family or parameter value. Stable floating-point summaries are not exact arithmetic. Source/native/browser review is not an observed beginner study, and completion does not guarantee universal inference mastery. Root owns final publication metadata, integrated production loading/route verification and the 74-topic ledger.

## Frozen runtime and brief fingerprints

| File | SHA256 |
| --- | --- |
| `src/learn/data/topics/exponential-families-sufficient-statistics.jsx` | `64a1ae33f30b28d62cf691c12a2ba86bfab637647f3a323e2793ed534320c75d` |
| `src/learn/data/exponential-family-models.js` | `b371f7f03adebd0d60f21ab7b4ac3e1568465bd0db605c9a34a5cc58d8132a7c` |
| `src/learn/data/exponential-family-examples.js` | `fc115d8d97394fb82a40931a7bdf55776015fe7d71a7505295cdb9465dd41521` |
| `src/learn/components/lesson-labs/ExponentialFamilyLabs.jsx` | `bde1928bd2c16a2f8296a05ecca29eca393bb7f1e20f5c330b7b4d492ab15a3d` |
| `src/learn/components/lesson-labs/exponential-family-labs.css` | `f15c93899232877b131554bb297c483c94388dae33924d08a0d117c70e4459ae` |
| `src/learn/data/curriculum/blueprints/exponential-families-sufficient-statistics.js` | `93b8ab71e3aa79c55167f0cb2b709bd009a2e1106b97caf1425614e07bb1dfbe` |

## Independent review and narrow setup amendment

The independent [mathematical/source review](EXPONENTIAL-FAMILIES-INDEPENDENT-REVIEW.md) found no actionable mathematical defect. It requested moving the existing Python setup instructions from the final practice section to immediately before the first program. That prose-only change is complete; the model, nine program strings/outputs, labs, CSS and individual blueprint retain their original frozen hashes.

`node scripts/review-exponential-family-setup.cjs` passed at 16:16:27 UTC on 10 September 2026: actual 1440/390/320 reading confirms one setup paragraph directly precedes every program, both run commands are present, nine programs/nine practice tasks remain, and no page overflow or exceptions occur. All three saved first-program setup screenshots were opened. The normalized-AST/string and CSS formatting check also passed at 16:15:02 UTC. No redundant full native rerun is claimed for relocating prose.

The durable author evidence preserves the original 16:04:22 freeze and old body hash under `initialFrozenAt`/`initialSourceFiles`, and records this amendment plus the current body hash. Earlier full model/native/browser results remain timestamped as originally executed. The current body hash in the table above includes the setup relocation.
