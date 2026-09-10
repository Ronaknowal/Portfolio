> HISTORICAL RECORD — archived 9 September 2026. All queue, authorization, approval and teaching instructions below describe an earlier increment. They are not current policy. Start at [the current handoff](../../../LESSON-AUTHORING-HANDOFF.md).

# Programming rewrite — Pandas single-topic increment

9 September 2026. Implemented. Only Pandas was rewritten in this increment. The subsequent request authorised the [next three lessons](PROGRAMMING-REWRITE-BATCH-03.md); DSA remains excluded.

## Review

[Pandas: Data Wrangling, Joins & Grouping](http://127.0.0.1:5173/learn/topic/pandas-data-wrangling-joins-grouping)

The existing route and dark/gold style are retained. The local development server is required; nothing has been deployed.

## Teaching changes

- Replaced the previous lesson with twelve sequenced modules: table grain, selection, label alignment/copy-on-write, missing data, dates/timezones, duplicates, joins, grouping, reshaping, ordered windows, an audited report and solved practice.
- Added twelve independently runnable examples with visible, verified outputs and explanations. All data is inline, with no external download required.
- Connected the main workflow to a practical question: paid-order revenue by region. The final report accounts for exact duplicates, invalid amounts, excluded statuses, unmatched customers and accepted orders.
- Added an interactive join explorer with left/inner/outer modes, a duplicate lookup key, cardinality validation, input/output tables and match indicators. It is explicitly a bounded teaching model, not a Python interpreter.
- Explained hidden correctness risks: label versus position alignment, Pandas 3 copy-on-write, null-key matching, row multiplication, missing-key groups, size versus count, implicit pivot aggregation, time ordering and CSV schema loss.
- Added prediction questions and worked answers, including two changed-input exercises for the final project.
- Linked the official Pandas guides for deeper reference. The lesson teaches an end-to-end core workflow; it does not claim to document every Pandas API. Specialist operations are signposted rather than presented as fully taught.

## Verification

- `scripts/verify-pandas-lesson.mjs`: all twelve examples match their expected stdout on Python 3.12.14 / Pandas 3.0.1. All twelve interactive model states match real Pandas merge results or validation failures. Both project exercise solutions pass execution checks.
- `scripts/review-pandas-lesson.cjs`: passed at 1440 px and 390 px. Checks rendered outputs, lesson anchors, nonempty inline code, answer reveals, join modes/counts, duplicate-key rejection, validation toggle, keyboard reset, browser errors and document overflow.
- Desktop/mobile join-explorer screenshots visually inspected under `scratch/pandas-lesson`.
- Production build passes. Existing unrelated Bayesian-networks JSX warnings and the large shared-bundle warning remain.

Use `LESSON_PYTHON` to select an interpreter with Pandas 3.0.1 and `PLAYWRIGHT_PACKAGE` for the installed Playwright package. Browser tests use Edge and port 5173. No fresh dependency installation or cross-platform recreation was performed; the recorded versions identify the checked runtime, not a latest-version recommendation.

## Scope limits and next step

The final project's source contract uses small whole-cent amounts and explicit paid statuses. Fractional/out-of-range amounts, additional status rules, production validation, external IO engines and large-data processing require further policies, as stated in the lesson. Example assertions are development checks, not a replacement for production exceptions.

The later “implement next three” request authorised Matplotlib, Notebooks and Documentation/API Design. Their changes and verification are recorded separately in batch 03.
