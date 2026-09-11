# Classical ML: first ten lessons, final integration

Completed 11 September 2026. All ten authorized lessons have source-bound author verification, independent review and a passing production integration. The [implementation record](../../CLASSICAL-ML-SUPERVISED-IMPLEMENTATION.md) identifies the exact scope; the [ledger](classical-ml-supervised-progress.json) binds each reviewed source. User acceptance remains separate. No deployment was performed.

## Teaching and source ownership

The lessons progress from regression through trees, neighbors, boosted trees, SVMs, probabilistic classifiers, ensembles, recommenders, multiple outputs and survival analysis. They retain useful mathematical and implementation depth while explaining the learning problem, mechanism and assumptions first. Topic-specific diagrams and bounded investigations are placed where their representations help: partitions, neighbor geometry, residuals and weights, margins, evidence, out-of-fold predictions, feedback matrices, output structure and incomplete event histories. Complete executed examples, changed practice with explained solutions, and annotated alternatives supplement the self-contained teaching.

Bodies, models, examples, labs and authoring briefs have semantic topic-owned files. All 228 published lesson bodies remain distinct dynamic entries. The final source-import and actual browser-network checks found no unrelated lesson body, syllabus outline or DSA practice dataset loaded by the ten selected lessons.

## Final checks and evidence

- `node scripts/verify-curriculum.mjs`: passed for 28 modules, 1,218 stable topics, 351 individual briefs and seven guided paths. Existing IDs, membership and module reading order remain intact.
- `node scripts/verify-learning-artifacts.mjs`: passed with zero stale generated files; all 228 publication mappings and 245 separately loaded outlines agree with authoring sources.
- `node node_modules/vite/bin/vite.js build`: production build passed. The machine's global npm shim was unavailable, so the installed local Vite executable was used directly.
- `node scripts/verify-classical-ml-integration.cjs`: fifteen production cases passed in Edge 152 against the local production preview. [Exact source, build, browser and payload evidence](evidence/classical-ml-production-review.json) records the actual run.
- `node scripts/build-curriculum-inventory.mjs`: regenerated after all ten source-bound ledger entries became implementation-reviewed.

The production cases cover all ten fresh routes, exact titles and reading-time units, the 39-topic module order, compact counts, Previous/Next, completion persistence without navigation, and Survival's actual K-Means successor. Controlled delayed imports cannot replace a newer destination. Import and render failures retain the page shell and completion guard, and recover through the available retry/reload controls. In this browser a failed module import needed reload after Retry because the module map retained the failure; the evidence records that behavior explicitly.

Fresh selected-page JavaScript totals range from 1,406,897 to 1,496,324 built bytes; gzip estimates range from 312,293 to 344,722 bytes, including shared React, navigation and mathematics dependencies. These are file totals from actual requests, not measured network compression, load latency or a device-performance benchmark. The existing shared navigation chunk exceeds Vite's 500 kB raw-size warning threshold; the build still passes. Existing JSX warnings in the untouched Bayesian Networks lesson remain outside this increment.

## Integration corrections

Three bounded metadata corrections were closed by the final source/build/header checks:

- Multi-Output's bare reading-time number became an explicit reading/practice estimate: [original amendment](evidence/multioutput-metadata-amendment.json).
- Recommender's blueprint now follows the existing URL-string, depth and review-focus schema while retaining its seven sources and all previous teaching-plan fields: [amendment](evidence/recommender-integration-amendment.json).
- Survival's lesson object now uses the supported const/default-export form, and two literal greater-than text nodes use equivalent JSX entities: [amendment](evidence/survival-integration-amendment.json). Archived original bytes and parsed-AST conservation justify reusing its native and operated browser evidence.

The earlier amendments retain their historical pending-integration wording; this completed record and the current ledger close those pending checks without relabelling an earlier run. The production report's module count field was clarified to `moduleTopicCount`: 39 denotes topics in Classical ML, not the number of site modules. No test result changed.

## Retention and next action

The requested cleanup removed 2,222 obsolete or duplicate images, approximately 188 MB, and 35 temporary files outside scratch in its main pass. Topic owners subsequently retired their own duplicate drafts, unused captures and extracted programs. [Cleanup evidence](../engineering/WORKSPACE-CLEANUP.md) and the [retention policy](../engineering/WORKING-ARTIFACT-RETENTION.md) distinguish necessary source/runtime/review records from disposable working files.

Keep selected final images, compact numerical/browser records, source baselines and reusable environments. They are evidence and inputs, not a queue to repeat. There are no open findings in the authorized ten-topic increment. Await the user's next scope; the actual next module topic is K-Means & Hierarchical Clustering.
