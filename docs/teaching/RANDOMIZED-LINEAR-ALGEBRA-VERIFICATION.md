# Randomized Linear Algebra — implementation verification

10 September 2026. The existing stable ID, publication mapping, title and mathematics position6 are preserved. This record describes author and independent review; integrated delivery is recorded separately. It is not user acceptance or an observed beginner study.

## Coverage and learning decisions

Read the full previous body and topic inventory before authoring. The original100×40 Gaussian/rank5/oversampling4 example is retained with actual captured output. It now distinguishes its poor low-rank compressibility from random-sketch error. The lesson repairs missing output, disconnected power iteration, undefined dimensions, and unqualified performance/accuracy claims.

The connected route develops column probes, orthonormal range finding, compressed SVD and lifting, rank/error budgets, expectation versus a per-draw guarantee, stabilized iteration and data passes, row sketches and original least-squares residuals, leverage rescaling, subspace embedding, trace estimation, fresh validation, applications and independent practice. Separate residual names keep the proof visible on narrow screens. The next topic remains Multivariate Calculus & Gradients.

Four distinct investigations expose different mechanisms: tail-to-tip column contributions and projection; singular-value/rank/approximation error budgets; selected observations and regression fits; random signs flowing through a matrix into a running trace estimate. Two inline figures explain factorization shapes and the difference between right and left sketches. No invented timing data or empirically unsupported speed ranking is used. Four independent practice exercises include complete executable noisy-factor reconstruction, a false validation certificate, a rank-deficient row-sketch family and a finite probability counterexample. Earlier checkpoints require shape, rank, norm and uncertainty reasoning.

Primary research and precisely reviewed source/video scope are recorded in [the design](RANDOMIZED-LINEAR-ALGEBRA-DESIGN.md). The linked resource index and video title were checked; full playback is not claimed. The lesson has six rendered, annotated reference/alternate links.

## Native and independent accuracy evidence

`node scripts/verify-randomized-linear-algebra-models.mjs` passed125 weighted probes,1,440 spectrum/rank/width/iteration/seed states, all256 observation subsets,495 trace states and13 invalid-input groups. Fixtures: `scratch/randomized-linear-algebra-review/model-fixtures.json`.

`scratch/lesson-tools/Scripts/python.exe scripts/verify-randomized-linear-algebra-native.py` checks these actual JavaScript results against independent NumPy SVD/least-squares and exhaustive sign oracles. It verifies projectors, orthogonality, squared residual decomposition, best-rank floors, regression fits, leverage, expectation and variance; the largest projector difference is approximately2.52e-14.

`node scripts/verify-randomized-linear-algebra-examples.mjs` executes all **11 complete Python/NumPy programs** and compares literal stdout. Final formatting rerun passed at09:21 UTC. Runtime: Python3.12.14 and NumPy2.3.5. The four standalone randomized-SVD helper definitions are identical; the skinny-sketch SVD used for numerical rank is distinct from the full-matrix oracle used only to evaluate tiny examples.

[Independent mathematical review](RANDOMIZED-LINEAR-ALGEBRA-INDEPENDENT-REVIEW.md) passed additional1,296 square/tall/wide helper cases,891 zero/rank-deficient cases including scales1e-80 through1e80,1,161 full-range optimum checks,13 invalid cases,24 controlled residual-space embeddings and48 exhaustive sign-trace matrices. Its maximum normalized squared-error decomposition difference is5.33e-14. It also checked the range theorem against HMT's original assumptions and statement. An ambiguous practice question was corrected to distinguish a new run before final review; no numerical blocker remained.

## Actual browser and reading review

`node scripts/review-randomized-linear-algebra-lesson.cjs` passed at **09:12:30 UTC**, in local Edge headless at1440×1000 and390×1000. At each width it checked:

- Eight column-probe states, cancellation/zero behavior, actual SVG projection coordinates and keyboard reset.
- Forty spectral states across fast/slow/flat/rank-two/zero fixtures, rank/width/iteration changes, numeric error budgets and rendered bar values, with deterministic reset.
- Seven observation selections, unique versus underdetermined fitting, original residuals, actual selected-point geometry and keyboard checkbox behavior.
- Ninety-nine trace states covering all three matrices and0–32 probes, actual running means, plotted point counts/positions, final-step disabling, previous and reset.
- All11 rendered complete programs and expected outputs, eight existing anchors, six reference links, no page/KaTeX errors and no ordinary-width equation/page overflow.

`node scripts/review-randomized-linear-algebra-reading.cjs` passed at **09:24:54 UTC**, at1440,390 and320px after final equation repairs and conventional source formatting. It opens all deeper content, verifies every displayed equation fits, activates first/last navigation anchors and a checkpoint using the keyboard, captures six ordinary reading sections and confirms references and the adjacent-topic bridge. All three widths pass without errors or page overflow.

The reading review found real narrow-screen problems. First the long Pythagorean residual expression was split into explicitly named terms. The320px pass then exposed three more wide equations: the expected range bound and two embedding-proof displays. Named error/residual quantities and separate justified inequalities now preserve the whole argument without horizontal equation scrolling. No mathematical assumption or proof step was removed. A test-only checkpoint selector was corrected from a nonexistent class to the actual `.lesson-check`; it now asserts the checkpoint exists rather than silently skipping it.

Actual opened images include mobile weighted contributions/projections, desktop spectral budget and trace history, phone endpoint regression, zero-trace state, both phone inline figures, ordinary sections3/5/6/8, references, final desktop introduction, the320px expectation bound and the full320px spectral investigation with matrix details. All are under `scratch/randomized-linear-algebra-review/`. Numeric tables intentionally scroll within labelled keyboard-focusable regions when wider than the screen; prose, equations and controls do not require page scrolling sideways. Captures hide the fixed header only for isolated components, then restore it.

## Ownership and limits

The lesson, model, examples, component, scoped stylesheet and individual blueprint have semantic topic-owned files. `scratch/format-randomized-linear-algebra.cjs` verified identical normalized Babel ASTs and CSS non-whitespace tokens during formatting; subsequent equation changes received the final reading check. The JavaScript fixtures are bounded at6×6 matrices, eight selectable observations and32 probes. They use documented small numerical routines with independently checked fixture domains, not a general production SVD claim. Native code is displayed rather than executed by the browser; no heavy numerical package, autoplay, worker or eager cross-topic data bundle was added.

Finite fixtures do not prove universal numerical stability, a probability theorem or performance across hardware. Browser and NumPy random streams differ; fixed seeds are repeatable within their stated implementations. Numerical-rank thresholds, rounded readouts, zero denominators, exact-reference cost, matrix-access assumptions and manual-row-selection limits are explicit in the content. The complete74-topic goal remains active.
