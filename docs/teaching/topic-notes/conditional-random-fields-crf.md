# Authoring notes: Conditional Random Fields

Canonical topic ID: conditional-random-fields-crf

## 2026-09-21 — Connect the custom chain to an ordinary CRF package

- Status: resolved — independently reviewed and integrated on 22 September 2026.
- Origin: [Classical ML implementation-depth inspection](../implementation-depth/CLASSICAL-ML.md), row 29.
- Destination and ownership: this lesson already owns CRF features, normalization, inference and training. It is the right place to map those objects to a specialist sequence-tagging API; generic optimization or HMM lessons would obscure the discriminative objective.
- Existing coverage: `src/learn/data/crf-examples.js:7` exposes the supplied `crf_example.py` inference/training sections. They implement log-domain forward/backward, node/edge marginals, Viterbi and likelihood gradients, then fit the real tagging task using SciPy L-BFGS and DictVectorizer. The lesson references CRFsuite, but supplies no CRFsuite package fit/inference program. SciPy remains a useful optimization primitive; it is not the missing specialist API bridge.
- Proposed treatment: after the custom fit, add a complete small `sklearn_crfsuite.CRF` example with feature dictionaries, sequences, gold tags, fit, `predict` and `predict_marginals`. Map state/transition features and decoding/marginals to the custom implementation. Keep the existing richer custom route and its real-data experiment intact.
- Correctness contracts: explicitly account for sum-versus-mean likelihood, c1/c2 regularization, intercept/boundary features and `all_possible_states`/`all_possible_transitions`. Do not assert equal fitted weights merely because both are called CRFs. Prefer a small fixed scorer/inference comparison where the package exposes a verifiable score, or show two clearly different trained objectives with held-out outcomes. Assert normalized marginals, consistent sequence lengths and retained tag vocabulary; check an unseen feature and changed transition evidence.
- Efficiency and customization: sparse feature dictionaries are the ordinary route; disclose the bounded dense teaching reference. Add one real feature-template modification with unchanged data roles and explain its effect. Avoid duplicating the whole experiment only to demonstrate package syntax.
- Evidence: [sklearn-crfsuite API](https://sklearn-crfsuite.readthedocs.io/en/latest/api.html) and [official tutorial](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html), inspected 21 September 2026. Installed-version behavior and the proposed new program have not been executed.
- Resolution: confirmed bridge gap; substantial authoring and native package verification deferred to the next scoped revision. Current content/implementation checkpoints are not retroactively erased by this new requirement.
- Implementation/verification links: existing native program evidence remains authoritative for existing code; no new specialist bridge yet.
- Final implementation resolution, 22 September 2026: the learner-facing mechanism/tool route, contract comparison and changed-constraint practice are implemented; native checks, independent review and affected production checks pass. [Final scope and evidence](../implementation-depth/REMEDIATION.md). Earlier proposal/evidence-limit wording above is historical, not an unresolved task.

