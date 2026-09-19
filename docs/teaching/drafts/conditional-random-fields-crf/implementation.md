# Conditional Random Fields — prepared-content implementation

19 September 2026. Stable ID `conditional-random-fields-crf`. Authorized work: phase two of the existing complete packet, as the third topic of the user's five-topic continuation. Author: CRF implementation agent. **Author implementation and verification complete; independent review and shared production integration are owned by the increment owner.** User acceptance is separate. No commit or deployment.

## Inputs and scope conservation

Read the full prepared `lesson.md`, `visual-specifications.md`, `design.md`, extraction provenance, author calculations and results, plus the historical published CRF body. The `--work finish` preflight passed. The checkpoint-bound prepared files are unchanged. There is no destination note to resolve. The stable identity and title remain Conditional Random Fields (CRF); module order is unchanged.

All ten manuscript sections and five changed practice tasks are implemented, including the exact four-path calculation, independent-classifier special case, scale invariance, forward/backward/Viterbi distinction, marginal versus whole-sequence losses, observed-minus-expected gradient and regularization, convexity assumptions, label-bias counterexample, actual BIO masking, real-data fit, neural shapes/padding, higher-order complexity, trees/loops, semi-Markov, latent/partial labels, unlabeled conditional likelihood, conditional output sampling, structured SVM and Bayesian extensions. All annotated primary/alternative references remain.

The prepared plan's retention/correction matrix continues to explain preservation of useful original depth and removal of unsupported benchmarks, fabricated outputs, false legality guarantees and universal dominance claims. No new broad research or lesson scope was introduced; the packet's inspected canonical/reference claims are reused. Current actual execution establishes the installed numerical-library behavior used here.

## Production ownership

- Lesson: `src/learn/data/topics/conditional-random-fields-crf.jsx`.
- Figures and lab controls: `src/learn/components/lesson-labs/CrfFigures.jsx`, `CrfLabs.jsx`, `crf.css`.
- Exact bounded models: `src/learn/data/crf-models.js`.
- Executed data and displayed-example adapter: `src/learn/data/crf-data.json`, `crf-examples.js`.
- Complete program, original offline extract and notices: `public/learn-assets/crf/`.
- Detailed blueprint: `src/learn/data/curriculum/blueprints/conditional-random-fields-crf.js`; parent owns registration.
- Reusable native, model and browser checks: `scripts/verify-crf-native.py`, `verify-crf-models.mjs`, `review-crf-browser.cjs`.

Only this selected topic imports its models and the compact 40-sentence development record. The full 200-sentence dataset is a download, not an eager browser import. Browser calculations enumerate only four paths; no training, network data retrieval, timers or unbounded work runs in the page. Displayed code contains the exact downloadable program string, with parsed source equality checked against the `.py` file; it avoids a public-directory raw import and manual escaping.

## Representation decisions and author corrections

The seven inline/inspection groups implement the specified factor chain, shared-scale path ledger, prefix/suffix joins, count balance, BIO predecessors, actual EWT errors and branched neural flow. Two editable investigations have distinct jobs: eight dimensionless factors drive a trellis and full argmax-set comparison; branch evidence drives private-versus-shared normalizers. Their predictions are initially unset, explicitly committed to an input tuple, invalidated by edits, and separate from ungraded exploration. Applied output retires immediately when inputs change; revealed tuples remain available only through ungraded exploration. Reset clears both prediction and applied result.

The optional separate generalized-graph picture was not added: the chain, neural branch and explicit prose on loops/segments carry the stated scoped outcomes without another largely schematic panel. The compulsory neural picture does show separate trainable input and pair branches, their convergence, the training loss, decoding and gradient destinations. Narrow screens stack branches instead of shrinking text.

Author review found and fixed:

1. The first factor-chain layout visually aligned pair factors with the input column. It now uses the output label column for both circle and pair connectors, with separate word-to-unary-to-label links. Actual desktop/phone images were reinspected.
2. A tiny editable probability change could round to the same six-decimal label while receiving a directional verdict. The result now prints the signed difference with significant digits and declares the `1e-10` unchanged tolerance; path ties declare their relative `1e-12` tolerance.
3. The manuscript's post-code output-introduction sentences became dangling after using `RunnableExample`, which places output with code. These were integrated or removed. Stale author-only “execution deferred” text was replaced with the actual executed status; external optional programs remain honestly unexecuted.
4. Long element screenshots contained the fixed site navigation partway through the crop. Capture-only CSS hides `.learn-nav` while taking element images; the real page/navigation is unchanged, and keyboard/overflow assertions run without that override. This is a capture correction, not a hidden lesson defect.

## Actual checks

- `scratch/lesson-tools/Scripts/python.exe scripts/verify-crf-native.py`: **passed**. Executes the exact complete downloadable program and compares stdout. Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1; no package installation. Independent direct-product enumeration checks 24 varied 2–4-position models, partitions, best paths, node and edge marginals. All 40 development predictions/marginals reproduce the packet. The actual test confusion matrix totals 370 tokens with 293 diagonal successes. Independent and chain optimization converge in 25/34 iterations.
- `node scripts/verify-crf-models.mjs`: **passed**. Separate log-space dynamic normalizer versus direct path products; all five lab presets, nulls, tie sets, scale invariance, finite-difference logZ and log-likelihood gradients, label-bias reverse/equal/changed-prior/transfer, all 30 BIO predecessor cases, legal and all-impossible start masks, and invalid inputs. Parses and renders all **94 actual JSX mathematical expressions** with KaTeX, rejecting control-character escaping defects.
- `review-crf-browser.cjs`, Edge headless against the author dev server `127.0.0.1:4184`: **18 grouped checks passed** across **1440, 390 and 320 px**. Exercises committed predictions and controls, factor traces, painted SVG widths, stale inputs, nulls, ties, invalid input, reset, changed-prior/transfer, BIO, real errors/marginals, count sign, keyboard editing/activation, anchors, native output and page/SVG bounds. Production build/load checks remain with the parent.
- All selected final diagram/lab images at 1440 and 390 px were opened for visual review; evidence paths and file hashes are in `docs/teaching/evidence/crf-browser.json`. Exact source binding is in `crf-author-review.json`. The reusable generic SVG overlap scanner is not the basis of the visual claim; rendered figures and their actual encodings were inspected.

Numerical evidence: `docs/teaching/evidence/crf-native.json`, `crf-models.json`. Browser tests do not establish independent review or actual novice performance.

## Learning-experience checklist — author's assessment

1. **Route:** a local prerequisite refresher and section navigation precede the core; first-pass and deeper-route instructions are explicit.
2. **Cautions:** information availability, decision loss, convexity, normalization, legality and small-data evaluation each have a clear conceptual home. Code prints computed outputs only.
3. **Real question:** a whole-person-name labeling problem motivates interactions; actual web-text tokens reveal errors rather than a synthetic perfect score.
4. **Investigations:** both record/check predictions and expose editable entities beyond presets. Contrasting, unchanged and equal/tied fixtures were run; independent transfer prompts accept computed results rather than matching a canned answer.
5. **Figures:** common zero-based mass/probability scales preserve magnitudes; no minimum fake bar length, gradients or performance curves. Narrow text reflows and token annotations stay together. The factor connector defect was found visually and repaired.
6. **Connections:** grouped path sums are explicitly equated with forward/backward products; HMM, independent softmax, neural encoders and the next GP topic are connected. Canonical-scope choices remain in the prepared design.
7. **Code:** two contiguous blocks are the actual complete native source; inference and learning mechanisms dominate the code. Defensive verification lives in separate scripts. External neural/library routes are annotated alternatives.
8. **Practice:** changed factors, changed gold path, a max-as-sum diagnosis, a live-feature/legality diagnosis and an independent feature ablation all have hidden hints and explanatory solutions or assessment criteria.
9. **Screenshots:** inspected contrast with saved prediction, real wrong-token marginals, negative gradient, alternate BIO continuation and both neural branches, including phone views; not just opening defaults.

## Handoff and limits

Independent reviewer should inspect the full lesson and complementary risks, especially conditional output sampling, neural/latent optimization claims, generic BIO legality and the small real-data protocol. Author checks are not independent review. The parent owns registry, generated metadata, phase checkpoints, catalogue checks, production build/loading/recovery tests and final completion. No known material author finding remains. Retained scratch images are source-linked evidence; the one-off manuscript conversion script was removed after source review so it cannot overwrite corrected production text. The shared Python environment is untouched.

## Independent closure and author concurrence — 19 September 2026

Read and agreed with the independent reviewer’s bounded patches: predecessor ties now share the full-path relative tolerance; default/revealed tuples remain ungraded exploration; lab outputs retire on edits; tie-transfer wording is conditional. Source patch reviewed in CrfLabs.jsx and crf-models.js. Reviewer ran the affected author browser campaign again (18 groups/16 captures), inspected four independent phone captures, and added 3,905 full BIO-support and 81 sampling-path comparisons. Closure: [CRF-INDEPENDENT-REVIEW.md](../../CRF-INDEPENDENT-REVIEW.md), [source receipt](../../evidence/crf-independent.json).

The example module now contains the exact downloadable program string instead of a Vite public-directory raw import. The model verifier parses that string and asserts equality with the normalized downloadable file; it passes with all 94 formula checks. Native program/data are unchanged. Author source binding refreshed; shared production integration remains with the increment owner.
