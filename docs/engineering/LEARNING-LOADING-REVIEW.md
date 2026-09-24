# Learning source organization and loading review

10 September 2026. Scope: preserve the completed visual improvements, repair rollout-based source organization, and keep growing lesson content from becoming an all-at-once browser payload. No new topics were written or removed in this pass. Current policy is [LEARNING-CODE-STANDARD.md](LEARNING-CODE-STANDARD.md), linked from AGENTS, the teaching standard and the authoring handoff.

## DSA confirmation

Both already implemented DSA lessons were part of the previous visual pass and remain improved:

- **Arrays, Strings & Hash Maps** uses concrete equally sized storage slots, byte ranges, indices and an address displacement instead of a generic prose-box flow. The existing movement, text-unit and hash-bucket investigations remain.
- **Linked Lists, Stacks & Queues** shows the circular physical slots, wraparound, head/next markers and occupancy alongside the logical FIFO sequence. Pointer reversal and the vertical bracket stack remain distinct representations.

The [systems/DSA/pilot review](../teaching/SYSTEMS-PILOTS-VISUAL-REVIEW.md) records their learning questions, assumptions and model boundaries. This pass reran their relevant model and desktop/mobile checks. It preserved these representations rather than replacing them with a new uniform lab.

## Source ownership and conservation

`next-three-blueprints.js` held the plans for Pandas, Matplotlib and Git, named after the increment in which they were authored. That source arrangement is removed. All **19 authored plans** now have individual stable-topic-ID files. Eight old blueprint bundles were removed; exact object and whole-catalogue comparisons passed. [Blueprint migration evidence](BLUEPRINT-ORGANIZATION.md).

Larger runtime example/model/lab/figure bundles were split by subject and shared controls by function. **66 original data exports** were conserved, **62 replacement runtime modules** parsed and **29 retired runtime files** removed. Twenty-seven lesson import aliases were clarified; eleven plot/notebook assets moved to semantic folders. Complete example code, expected output and model behavior were preserved. [Runtime source map and native verification](RUNTIME-SOURCE-ORGANIZATION.md).

All **1,218 unique topics, 1,222 module memberships, 28 modules, seven guided paths and 193 published lesson mappings** remain. All 17 programming lessons remain published; Arrays and Linked Lists remain implemented in DSA. The complete resolved authoring catalogue is byte-for-byte equal after serialization to the captured pre-migration catalogue, SHA-256 `c47c3290cbb59806d40f8db19ebb4396df2f1233d33b307f2eb4752b02dd10b3`. Every new browser route equals the route resolved from fresh full authoring sources. No title, dependency, progress identity or reading sequence was changed.

## Loading change

Previously `topics/index.js` eagerly imported every published lesson. The hub and even a planned topic therefore downloaded a shared chunk containing all 193 lesson bodies. Renaming source files alone would not have fixed this.

The new publication manifest maps stable IDs to lesson files. A build/dev generator produces compact navigation metadata, explicit lazy imports for registered lessons, and 270 individual planned outlines. Unregistered drafts are excluded from the production build. The browser loads only the selected lesson and its real dependencies, or that selected planned outline; published lessons do not fetch their authoring plans.

The generic content barrel also pulled KaTeX into non-mathematical topics. Math imports are now direct and separate. Import-only migration of **217 consumers**, including drafts, preserved all bindings and every non-import byte. A guard checks all 427 source files plus the generic barrel's seven-module dependency tree. Non-mathematical routes now omit the engine and its font CSS; Eigenvalues renders mathematical notation with the required engine/styles.

Headers, course position, sidebar and sequence navigation stay available while loading. Import/render failures have retry and reload actions; premature completion is disabled. A late response for an old route cannot replace the current lesson. Successful imports are reused on revisit. In the tested Edge version, retrying an aborted module import required the explicit **Reload page** fallback because the browser cached the failure; that fallback recovered successfully. We do not claim the application can universally clear the browser's module cache.

## Measured production payload

Three runs per route, with a fresh browser context and HTTP cache disabled for each run: 18 runs before and 18 after. Edge/Chromium 152.0.4191.66, 1440 × 1000 viewport, local Vite production preview on port 4173, no CPU/network throttling. Vite preview supplied HTTP compression. Numbers below are same-origin JavaScript **encoded body bytes**, excluding HTTP overhead, CSS and external fonts.

| Route | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Learning hub | 3,553,283 | 176,055 | 95.0% |
| Python Basics | 3,554,323 | 222,096 | 93.8% |
| Planned Trees topic | 3,554,323 | 197,489 | 94.4% |
| DSA Arrays topic | 3,554,323 | 216,567 | 93.9% |
| Existing tokenization lesson | 3,554,323 | 309,140 | 91.3% |

The portfolio entry remains essentially unchanged at approximately 83.6 KB compressed JavaScript. Decoded JS for the hub fell from 10,997,385 to 942,323 bytes; Python fell from 10,999,526 to 1,110,463 bytes. CSS payload also declined, including exclusion of math fonts/styles where unused. The shared catalogue/UI chunk is still about 667 KB decoded / 89 KB gzip and retains Vite's size warning; the warning threshold was not raised. Shared catalogue data supports site-wide search and path navigation.

This is a payload and dependency-boundary result, not a universal speed claim. Local median hub readiness changed from 933 to 432 ms; Python was essentially unchanged at 941 versus 944 ms. Code splitting increases the number of requests on some lesson routes (Python: three to seven), and host contention, network latency and device characteristics affect timings. No field Core Web Vitals, long-term memory guarantees or observed learner-effectiveness result is claimed.

Reproducible evidence: [before](../../scratch/learning-performance/before.json), [after](../../scratch/learning-performance/after.json), [comparison](../../scratch/learning-performance/comparison.json) and [loading cases](../../scratch/learning-performance/load-boundaries.json). The reports contain resource URLs, sizes, run timings, browser details and limitations; intermediate results are saved separately.

## Verification completed

- Production build passed after the final source changes. Generated artifacts were current. A pre-existing unescaped-arrow JSX warning in the separate Bayesian Networks lesson remains, as does the catalogue size warning described above; neither is hidden or claimed fixed by this refactor.
- `verify-learning-artifacts.mjs`, `verify-curriculum.mjs`, inventory generation and `verify-programming-module-conservation.mjs` passed. They preserve all publication mappings, IDs, memberships and planned/module-order navigation.
- Blueprint/runtime migration conservation checks and `verify-content-import-boundary.mjs` passed. `git diff --check` found no whitespace errors; Git emitted its normal Windows line-ending notices.
- `verify-visual-refinements.mjs` independently checked the three t intervals, graph rows and all 128 toy address/read mappings; `verify-lesson-pilot.mjs` checked displayed Python outputs, KaTeX and distribution/eigenpair models. Both used the configured Python environment. DSA/system checks covered nine model contracts, sixteen exported displayed programs and 5,461 bracket strings. Further executable-example checks are recorded in the runtime source map.
- Production `review-visual-refinements.cjs` passed 18 captured desktop/mobile states, both queue capacities through complete traces, keyboard controls, reset, offset translation and bounds checks. The array/queue mobile views and interval desktop view were inspected. Production `review-lesson-pilot.cjs` passed all three pilots at both widths.
- Production `review-module-order.cjs` passed all 17 programming steps, seven path entries/counts, module boundaries, shared-topic reload/Previous/Next/progress, planned resume and prerequisite links at 1440 and 390 pixels.
- All eight production loading cases passed: no lesson/outline on hub; selected body/dependencies only; Next adds only the destination body; Previous and browser Back reuse imports; planned outline isolation and no request for an absent outline; math engine only where used; stale request ignored; transport/render failures preserve the shell, disable completion and recover with the documented actions.

The dev and local preview servers remain available. Nothing was deployed. Current authoring policy, reviewed lesson status and the next scoped lesson remain in the handoff; this engineering pass is not authorization to expand the teaching rollout.
