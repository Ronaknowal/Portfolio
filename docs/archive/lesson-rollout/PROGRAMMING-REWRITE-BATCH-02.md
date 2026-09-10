> HISTORICAL RECORD — archived 9 September 2026. All queue, authorization, approval and teaching instructions below describe an earlier increment. They are not current policy. Start at [the current handoff](../../../LESSON-AUTHORING-HANDOFF.md).

# Programming rewrite — batch 02

9 September 2026. Implemented. The subsequent request authorised only one further topic; see [Pandas single-topic increment](PROGRAMMING-REWRITE-PANDAS.md).

## Three topics replaced in place

| Topic | Teaching coverage | Executable examples |
| --- | --- | --- |
| Decorators & Context Managers | Closures, forwarding, wraps, configuration, stacking, timing, caching, enter/exit, suppression, generator managers, ExitStack, ownership and solved practice | 8 |
| Testing, Debugging & Dependency Management | A real failing case, runnable unittest suite and discovery, boundary/property checks, mocks and real-file integration, debugger workflow, environments, dependencies and reproducibility | 5 |
| NumPy: Arrays, Broadcasting & Vectorization | Creation/dtypes, indexing/copies, masks, reshape/axes, broadcasting, reductions, missing values, ranking, numerical checks, linear algebra/SVD, RNG/I/O and a train/test preprocessing project | 12 |

All three retain their URLs and the existing dark style. No DSA or later-topic rewrite is included.

## Significant changes

- Replaced short wrappers and the NumPy API overview with sequenced, topic-specific lessons, visible results and explained answers.
- Added two step-through explanations (decorator call order and evidence-led debugging) plus an interactive broadcasting explorer with per-cell arithmetic and an incompatible-shape case.
- The test example actually runs four test methods, including subcases, instead of just defining test functions. Its normal discovery command is included.
- Timing uses a clearly labelled fake clock for deterministic teaching output; real timing remains machine-dependent.
- NumPy shows SVD shapes, singular values, a reconstruction check and the rank-one approximation. Solver basis ambiguity is explained rather than presenting arbitrary vectors as unique.
- Corrected the old NumPy implication that boolean indexing masks follow arithmetic broadcasting rules. Distinguished advanced-index reads from direct indexed assignment.
- Setup commands are labelled terminal recipes, not fixed-output Python examples. The recorded environment is Python 3.12.14 / NumPy 2.3.5; that NumPy version requires Python >=3.11. The pin is a tested snapshot, not a latest-version recommendation.

The NumPy page provides a detailed core numerical workflow and reference routes, not a claim to cover every library routine. Specialist FFT, structured-array and interoperability material is signposted separately. The testing lesson uses the standard library; it does not pretend to teach every external testing or dependency-locking tool.

## Verification

- `scripts/verify-programming-batch-two.mjs`: all 25 exact code/output pairs execute successfully in isolated temporary directories. It also checks decorator-trace output, unittest discovery and the suite's nonzero failure when a temporary implementation is deliberately broken.
- `scripts/review-programming-batch-two.cjs`: all three pages tested at 1440 px and 390 px for rendered outputs, anchors, answer reveals, trace controls/reset, keyboard activation, broadcasting modes/cell arithmetic, browser errors and page overflow.
- `scripts/review-programming-batch-one.cjs`: previous three pages still pass their desktop/mobile checks after the backward-compatible trace extension.
- Production build passes with existing unrelated Bayesian-networks JSX and large shared-bundle warnings. No unrelated warning cleanup was attempted.
- Package installation recipes were checked against official guidance; no fresh network package installation or cross-platform environment recreation was performed. The displayed numerical examples ran on the existing verified runtime.

Use `LESSON_PYTHON` for an interpreter with NumPy; use `PLAYWRIGHT_PACKAGE` for an existing Playwright package if needed. Browser tests require Edge and the local server at port 5173. Screenshots are under `scratch/programming-batch-two`; fixed navigation is hidden only during element capture.

## Originally proposed next batch

1. Pandas: Data Wrangling, Joins & Grouping
2. Matplotlib & Scientific Plotting
3. Reproducible Notebooks & Experiment Structure

Superseded by the user's request to implement just one more topic: Pandas only. Stop for review after that lesson; do not begin Matplotlib or Notebooks without a further request.
