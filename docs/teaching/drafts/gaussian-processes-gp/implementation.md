# Gaussian Processes — prepared-content implementation

19 September 2026. Author: `gp_implementation`. This record continues revision 1 under the user's five-topic finish request. It does not replace the checkpoint-bound [design](design.md), [manuscript](lesson.md), [visual contracts](visual-specifications.md), calculation inputs or data provenance. Those files remain unchanged.

## Scope and preservation

The historical source was preserved at `scratch/gaussian-process-implementation/original-lesson.jsx`, SHA256 `ee89796c165c0acaa4c97853dd8ce57b036dfb5c803a1b31c1ffa3459999272e`, matching the content-phase baseline. The implemented lesson retains all ten prepared sections, both complete native programs, seven separately closed hint/solution pairs, annotated alternate resources and the next-topic bridge. Its title and stable ID are unchanged. Unsupported historical benchmark/tool claims and uncertainty simplifications were already corrected by the prepared manuscript; they were not restored.

The six specified visual mechanisms are implemented: finite draw → coordinates → function points; bivariate Gaussian density slices and normalized conditional densities; observation covariance plus signed mean contributions; common-seed kernel samples with fixed-scale covariance maps; actual NOAA forecasts, their distinct fits and residuals; and target-to-candidate covariance links and variance-reduction comparisons. The full 120-month observation overview makes the train/development/test boundaries visible.

Three investigation contracts have four mounted interfaces: direct one-reading conditioning in section 2, editable spatial conditioning in section 3, the reported/exploratory historical forecast workspace in section 6, and editable prospective probe placement in section 8. The split keeps the one-reading mechanism visible before matrix notation. Every result requires an unselected prediction and a written reason; input changes invalidate it. Future forecast outputs are absent from that investigation's rendered text before commitment. The reported worked example elsewhere in the article remains a solved explanation.

The source note about kernel ridge/RKHS is carried into section 9: average-loss noise match σ²=nλ, no deterministic-regularizer interval guarantee, Brownian anchored RKHS with its almost-sure sample-path exclusion and finite-rank qualification, and the frozen-feature-map/solve-cost distinction. The integration owner closes the incoming note only after independent review.

## Source ownership

- Reader: `src/learn/data/topics/gaussian-processes-gp.jsx`.
- Pure model: `src/learn/data/gaussian-process-model.js`.
- Small, lesson-only recorded data and exact programs: `src/learn/data/gaussian-process-data.js`, `gaussian-process-examples.js`.
- Views: `src/learn/components/lesson-labs/GaussianProcessFigures.jsx`, `GaussianProcessLabs.jsx`, `gaussian-process.css`.
- Brief: `src/learn/data/curriculum/blueprints/gaussian-processes-gp.js`.
- Offline downloads: `public/learn/examples/gaussian-processes-gp/` (two programs, the unaltered selected CSV and its provenance).
- Durable generation/check scripts: `scripts/prepare-gaussian-process-data.py`, `verify-gaussian-process-model.mjs`, `verify-gaussian-process-content.mjs`, `verify-gaussian-process-browser.cjs`.

Browser model work is explicitly revealed, bounded by 12 spatial readings or 96 forecast conditioning rows and 24 future months. Each mounted conditioning/forecast workspace holds a one-entry factor cache keyed by positions, noise and exact kernel parameters; value-only changes reuse it. No browser hyperparameter optimization, ML package, background timer or network data fetch is introduced. The author-only fixtures/explorer arrays and manuscript stay outside browser imports. Publication, generated navigation and final production measurement belong to the increment owner.

## Computational and content evidence

Commands run in the repository:

```text
scratch/lesson-tools/Scripts/python.exe scripts/prepare-gaussian-process-data.py
node scripts/verify-gaussian-process-model.mjs
node scripts/verify-gaussian-process-content.mjs
LEARNING_BASE_URL=http://127.0.0.1:4184 node scripts/verify-gaussian-process-browser.cjs
```

The last line expresses the environment setting portably; PowerShell uses `$env:LEARNING_BASE_URL='http://127.0.0.1:4184'` before the node command. This is author dev-browser evidence; final production validation is separate.

The native run used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and scikit-learn 1.9.1 from the retained environment without package installation. It executed both exact manuscript programs, captured complete stdout, regenerated the specified finite draws and historical fits, and compared every retained calculated result with the content checkpoint. [Native receipt](../../evidence/gaussian-process-native.json) records versions and packaged file digests. Runtime output is faithful to those executed programs, not hand-written approximate output.

[Model evidence](../../evidence/gaussian-process-model.json) records 1,196 numeric comparisons. An independently written pivoted row-elimination solver verifies the browser Cholesky result over different lengths/noise levels, repeated noisy positions and changed targets. Prospective measurement reduction is separately checked by conditioning the augmented observation system. Six frozen-kernel forecasts match sklearn fixtures; shorter-horizon prefixes match, and modifying all future measured values cannot alter predictions or covariance. Tests include zero observations, noise-free interpolation, singular duplicate/noise-free failure, invalid covariance, all four change categories, exact independent-kernel nulls, prefix-only centering and invalid bounds.

[Content evidence](../../evidence/gaussian-process-content.json) verifies 138 formula strings with strict KaTeX rendering, all ten prepared headings, fourteen separate hint/solution disclosures and the exact two native programs and downloads. It refuses decoded control characters or escaped HTML disclosure markup in prose. The native receipt is reused after layout changes because the programs and data remain unchanged.

The phase-one source research is reused for unchanged technical claims. The primary GPML regression chapter and current scikit-learn GPR API were opened again on 19 September, confirming the documented regression/noise API context. This is not a fresh claim to have reread the entire textbook or operated third-party interactive resources.

## Author learning-experience and visual review

This is an author's heuristic walkthrough, not independent review or a novice user study.

1. The opening pipe problem gives a reason to model uncertainty; sections 1–7 define the first-pass route and sections 8–9 state their deeper purpose.
2. The manuscript retains scoped interpretations at their decision points. Programs show computation without printing warnings. Figure captions identify finite samples, pointwise intervals and measured-versus-modeled quantities.
3. The native real-data result deliberately retains the GP's poor test coverage (13/24), alongside its lower MAE than seasonal-naive.
4. Learners edit actual readings, covariance/noise, forecast requests and prospective probes. Nulls include value-only covariance invariance, zero cross-covariance, unchanged horizon prefixes and tied zero-gain candidates. No predicted choice is preselected.
5. All six figure types were inspected in desktop and phone captures. The density slice, common-seed geometry, signed mean cancellation, real test misses and target-specific information contrast are discernible. HTML annotation rows wrap independently of quantitative SVG positions; matrices and formula containers preserve keyboard scrolling.
6. The same one-observation conditioning explains the larger matrix update and later measurement selection. The optional KRR equality is connected without transferring unsupported uncertainty claims.
7. Both native programs remain compact mechanism-focused examples, and complete captured outputs are shown by the actual `RunnableExample.expected` contract.
8. Practice changes values/assumptions, asks for a counterexample, critiques evaluation and requires a new kernel proposal. Hints do not reveal their separately closed solutions.
9. Informative captures include edited mean-only conditioning, final-test misses and a moved-target probe choice, not just initial panels.

Author review repaired concrete issues before handoff: separate adjacent Markdown disclosure blocks to avoid escaped HTML text; keyboard-scrollable local formula containers on narrow screens; sufficient SVG left padding for signed/decimal tick labels at 320 px; density tick precision (0.25 must not print as 0.3); a clear distinction between candidate number 2 and its location x=4; and scientific notation for the tiny nonzero RBF covariance that a six-decimal display rounded to zero. The matrix explicitly says its two-decimal zeros need not mean independence. Figure-only screenshots hide the fixed site navigation during capture so it cannot obscure tall figures; production behavior is unchanged.

The [browser record](../../evidence/gaussian-process-browser.json) binds the rendered source and eighteen final capture digests at 1366/390 px; checks also run at 320 px. It covers hidden numerical results, prediction/edit/reset/null/invalid-state behavior, native stdout and data access, keyboard navigation, painted band fill, label clipping/overlap, sampled foreground-path/label intersections, page overflow and runtime errors. It is a bounded author check, not proof of all possible data configurations.

The independent reviewer identified inconsistent coordinates in the density-slice tick labels. The implementation now derives both ticks and labels from the same x/y maps as the contours and slice. The affected browser checks pass again (91 groups); the revised phone capture was inspected. Independent coordinate assertions and final disposition belong to that reviewer.

## Status and next action

- Implementation: authored and available through the real reader.
- Computational verification: passed as linked above.
- Browser/visual review: author checks and selected image inspection completed; current counts/digests are in the browser record.
- Independent correctness and learning-experience review: pending the increment owner's reviewer.
- Production build/loading/integration and phase ledger: pending increment owner.
- User review: not claimed.

Next: independently review these source files and the complete prepared packet, resolve material findings with affected checks, then let the increment owner finalize the production build, generated catalogue, source-bound ledger and destination-note closure. Do not treat the author's review as independent.
