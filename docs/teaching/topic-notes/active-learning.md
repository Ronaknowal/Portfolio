# Authoring notes: Active Learning

Canonical topic ID: active-learning

## 2026-09-21 — Map the working oracle loop to a specialist query API

- Status: resolved — independently reviewed and integrated on 22 September 2026.
- Origin: [Classical ML implementation-depth inspection](../implementation-depth/CLASSICAL-ML.md), row 32.
- Destination and ownership: Active Learning owns query strategies, annotation budgets and who can see which labels. Supervised Learning owns the base estimator; keep that model reused.
- Existing coverage: `src/learn/data/active-learning-examples.js:2` supplies a complete runnable banknote experiment with random, entropy, diversity and committee queries, LogisticRegression, paired seeds, development selection and an untouched final report. The lesson displays excerpts and links the full program. This is already a practical ordinary sklearn workflow. scikit-activeml is currently an external resource, without a learner program mapping its query interface to the hand-written loop.
- Proposed treatment: a short optional specialist-library route after the custom loop: represent unlabeled targets with the package's documented sentinel, adapt a compatible classifier, query one item, reveal only the oracle-selected label and refit. Map pool IDs, known-label mask, utility, selection, budget and fitted learner to the existing loop's objects. Use the same tiny pool and probabilities for a one-round acquisition check if the API supports controlled fitted state.
- Correctness contracts: verify the installed version's missing-label handling, sample-ID order, tie-breaking and query return shape. Never pass full hidden targets into the model or strategy. A package-specific sentinel need not equal the existing program's `-1`; translate it explicitly. Log the chosen ID, utility and incremented budget. Check an empty eligible pool and exhausted budget without manufacturing a query.
- Efficiency and customization: bound candidate and committee work, reuse cached probabilities only while fitted model/pool state is unchanged, and explain when refitting is required. The improvement is API interoperability, not a new claimed accuracy advantage or a duplicate expensive banknote benchmark.
- Evidence: [scikit-activeml official documentation](https://scikit-activeml.github.io/latest/index.html), inspected 21 September 2026. Exact strategy/classifier versions and proposed adapter execution remain unverified.
- Resolution: narrow specialist bridge gap; retain the existing usable library composition and full scratch query controller. Assess whether an optional package subsection materially helps the next revision before adding dependency cost.
- Implementation/verification links: none for the proposed bridge; existing lesson/native evidence remains unchanged.
- Final implementation resolution, 22 September 2026: the learner-facing mechanism/tool route, contract comparison and changed-constraint practice are implemented; native checks, independent review and affected production checks pass. [Final scope and evidence](../implementation-depth/REMEDIATION.md). Earlier proposal/evidence-limit wording above is historical, not an unresolved task.

