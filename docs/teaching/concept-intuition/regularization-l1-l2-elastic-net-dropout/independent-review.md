# Independent teaching review

regularization-l1-l2-elastic-net-dropout

Read the entire eleven-section lesson, ten practices, source annotations and scratch/library explanations. Reviewed partial residuals, coordinate/KKT logic, duplicate-feature ambiguity, likelihood/prior normalization, dropout, group and smoothness penalties, nonconvex factorization and information criteria. New group shrinkage figure correctly scopes its orthonormal-block formula, and MAP scaling distinguishes independent evidence from duplicated rows. No correction found; preserved fit outputs were not rerun.

## Actual checks

- Independent complete lesson reading and concept-transition assessment: passed.
- Current author source bindings and JSX parse: passed.
- Partial residual and group versus coordinate shrinkage: passed.
- Enumerated inverted-dropout expectation: passed.
- Factor penalty reduced objective and ridge convention: passed.

Reproducible command: `node scripts/verify-concept-intuition-independent-representation.mjs regularization-l1-l2-elastic-net-dropout`. The arithmetic checks were authored independently of the topic author's verifier. Exact reviewed source hashes are in independent-checks.json. No new blocking finding. Browser paint and interaction checks remain separate with root; this record does not claim screenshots or a new fitted-data campaign.
