# Review of previously improved lessons

10 September 2026. The user authorized revisiting all lessons improved so far and implementing topic-specific illustrations and interactions wherever they make the material clearer. Scope: 17 Programming & Scientific Computing lessons, two implemented DSA lessons and the three original pilots: **22 topics reviewed, 21 improved, Linux retained**. No topic was removed, renamed, reordered or newly published. The BPE lesson was a presentation reference and retains its separately recorded, unresolved accuracy concern.

The review assessed the ordinary reading flow and actual lab representations, not just lab counts. The main gap was often a relationship left in prose or a generic state panel despite an otherwise useful lesson. Existing effective diagrams, models, examples, deeper branches and practice were preserved. This is an author-reviewed improvement; it is not a beginner user study, user acceptance or a complete factual re-audit of all 22 lessons.

## Topic-by-topic result

| Topic | Implemented improvement or reason to retain |
| --- | --- |
| Python Basics | Visible shared-list versus copied-list reference maps before operating the mutation lab. |
| OOP in Python | Saved bound method shows its function and original receiver after a variable is rebound. |
| Iterators, Iterables & Generators | One source collection with genuinely shared or separate cursor objects and moving next-item arrows. |
| Decorators & Context Managers | Definition-time rebinding and later wrapper call chain; resource routes show entry, body, cleanup and failure paths. |
| Testing, Debugging & Dependency Management | Mocked text versus real-file paths expose what each passing test actually exercised. |
| NumPy | Same-input reshape/transpose comparison preserves original sensor/time labels at every output cell. |
| Scientific File Formats & Reliable I/O | CSV field boundaries versus naive splitting; publication shows actual records, incomplete and absent artifacts. |
| SQL | Sensor/observation references distinguish one-to-many membership, missing rows and NULL values. |
| Pandas | Interactive pivot grid links output coordinates to source records and demonstrates duplicates, missing cells and explicit aggregation. |
| Matplotlib | Nested Figure/Axes/Axis/Artist ownership with the corresponding drawing and saving operations. |
| Reproducible Notebooks | Edited source, retained output and live kernel state remain visibly separate. |
| Code Documentation, Type Hints & API Design | The same call follows separate static-checking and runtime-execution routes. |
| Git & GitHub | Shared repository and local references make fetch versus integration boundaries explicit. |
| Linux Basics | Retained: tree navigation, stream routing, permission gates, lifecycle and inline link/environment diagrams already suit their different questions. |
| Bash | Received argument slots expose quoting/expansion boundaries; stream output and exit-status selection are separate channels. |
| OS Processes & Virtual Memory | Same virtual address shown through two process maps; byte-offset selection expands the existing translation investigation. |
| Threads & Concurrency | Independently controlled worker lanes expose private snapshots, program order and the one shared counter. |
| Arrays, Strings & Hash Maps | Concrete adjacent slots and byte displacement replace the generic address-formula flow. |
| Linked Lists, Stacks & Queues | Circular physical slots, head/next markers and occupancy stay linked to the FIFO sequence; exact slot inspection remains available. |
| Hypothesis Testing & Confidence Intervals | Interval number line compares zero with a practical threshold; changed data shifts the interval without changing precision. |
| Bayesian Inference | Observed successes/failures and separate Beta parameter updates are visible before the density exploration. |
| Spectral Graph Theory | A local neighbor/weight inspector connects a Laplacian row to its individual signed contributions. |

There is no new required lesson template or diagram quota. A familiar table remains useful for exact values; a static figure can introduce a relationship; a specific interactive operation lets a learner explore its changes. The continuous reading path introduces objects and terminology before relying on the representation. Quantitative additions are verified calculations or explicit small models, not invented measurements.

## Detailed design and verification records

- [Python foundations](docs/teaching/PYTHON-VISUAL-REVIEW.md): five topics, reference/cursor/lifetime mechanisms, source/runtime checks and 1440/390/320px review.
- [Scientific computing](docs/teaching/SCIENTIFIC-VISUAL-REVIEW.md): five topics, shape/field/key/pivot/ownership contracts and native CSV/NumPy/Pandas/SQLite/Matplotlib checks.
- [Developer workflows](docs/teaching/WORKFLOW-VISUAL-REVIEW.md): five topics, notebook/API/Git/Bash/thread contracts and independent Python/mypy/Bash/Git checks.
- [Systems, DSA and pilots](docs/teaching/SYSTEMS-PILOTS-VISUAL-REVIEW.md): seven topics, retained Linux rationale, new diagrams and mathematical/browser evidence.

Each record explains the original weakness, exact placement, concrete example, visual encoding, interaction/prediction where applicable, model limits, accessible/mobile behavior, evidence and retained content. Existing design records link their follow-up, and the current handoff links this complete review. Pandas' live blueprint now includes the pivot investigation rather than describing only its previous four labs.

## Integration evidence

The dedicated native/model and browser reviews passed as recorded above. Browser evidence includes screenshots visually inspected by the authors, keyboard activation, changed inputs, feedback, reset/back, narrow-screen composition and errors/overflow checks. The roots of the recorded artifacts are `scratch/python-visual-review/`, `scratch/scientific-visual-review/`, `scratch/workflow-visual-review/` and `scratch/visual-refinements/`.

- `node scripts/verify-curriculum.mjs`: 28 modules, 1,218 stable topics, 289 briefs and seven paths; module sequence and shared identities remain valid.
- `node scripts/verify-programming-module-conservation.mjs`: all IDs/memberships conserved; all 17 programming topics remain published.
- `node scripts/build-curriculum-inventory.mjs`: 193 published lessons, 339 prerequisite reviews; no publication-count change.
- `node scripts/verify-lesson-pilot.mjs`: displayed native outputs, formula rendering, coverage/Beta/spectrum numerical checks passed. The verifier now normalizes Windows newlines on both compared strings.
- `node scripts/verify-visual-refinements.mjs`: new interval/graph calculations passed independent Python/NumPy checks; all 128 address/process read mappings checked.
- `node scripts/review-visual-refinements.cjs`: 18 desktop/mobile captures passed after small-screen refinements.
- `node scripts/review-systems-structures.cjs`: passed all nine existing labs across controls/steps at 1440 and 390px, keyboard/reset/back, expanded solutions, anchors, topic links, resources, next sequence and SVG/page bounds. A stale assertion about the former Linux-to-OS bridge was corrected to the actual Linux-to-Bash sequence; no learner navigation change was made in this pass.
- `node node_modules/vite/bin/vite.js build`: integration build passed. Existing warnings remain for unescaped arrows in the unrelated Bayesian Networks lesson and the large content bundle; this pass did not introduce them or attempt an unrelated site-wide repair.

Final reruns passed: all 15 Python topic/viewport cases after label and screenshot refinements; all nine systems/DSA lab traces at desktop/mobile sizes; and the production build after the final blueprint/label edits (9.38 seconds; log at `scratch/visual-refinements/build.log`). `git diff --check` passed. The current handoff and repository entry point now direct future authors to this record; acceptance by the user remains separate.
