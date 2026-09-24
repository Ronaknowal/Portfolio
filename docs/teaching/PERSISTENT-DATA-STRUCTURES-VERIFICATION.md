# Persistent Data Structures: author verification

10 September 2026. **Implemented and author-reviewed; ready for shared integration. User acceptance remains pending.** Stable title/ID and DSA position 21 are preserved. Publication registration by the root agent enabled review; registration alone did not establish completion.

Design: [PERSISTENT-DATA-STRUCTURES-LESSON-DESIGN.md](PERSISTENT-DATA-STRUCTURES-LESSON-DESIGN.md). Source freeze hashes: `scratch/persistent-structures-lesson-review/final-source-hashes.json` (seven semantic source files).

## Delivered scope and learning flow

The former planned entry had no published body to preserve. The complete lesson moves from the demand-plan example and a concrete alias failure to immutable stack tails, fully persistent point assignment/range sums, preservation proof and allocation/lifetime accounting. It then offers compact per-index timestamp histories when updates do not branch. Optional deeper routes fully implement compressed prefix frequency order statistics and a no-push immutable lazy range-add/sum tree. Applications, limits, independent practice and curated resources complete the route.

- Seven complete standard-library Python programs, with exact independently executed output: alias, stack, point, history, rank, lazy, queue.
- Eight independent tasks, each with a separately closed hint and explained solution; three in-flow checkpoints also start closed. The practice requires old-version and sibling-branch correctness, identity/allocation checks and changed constraints.
- Two verified official public LeetCode problems: 1146 Snapshot Array and 981 Time Based Key-Value Store. Their single-timeline scope is explicit. The independently checked official CSES Range Queries and Copies task is a full-branching transfer, with one-based inclusive versus half-open indexing conversion and no untested maximum-size acceptance claim.
- Original paper and official Python/Clojure references; annotated MIT official lecture-video page plus Okasaki written alternative. The exact sections reviewed are in the design; the video was not watched end to end and dated open-problem claims were not represented as current results.
- The incoming Range Queries destination note is resolved with actual source and checks. A bounded external-memory/root-publication discovery is saved for that existing next topic. No topic was renamed, deleted, reordered or silently broadened into database recovery.

## Models and proof contracts

**Point tree.** Default `[2,1,4,3,5]`, half-open intervals, index 2 assigned 9. Initial nodes are IDs 0–8, root n8 sum15. New n9/n10/n11 represent [2,3), [2,5), [0,5), with sums9/17/20. The compact figure verifies actual child identities: old/new roots both share n2, and old/new right parents share n6. Every old node remains frozen. A no-op returns the same root and allocates no nodes, though a distinct version handle can refer to it. Empty native arrays use `None`, only sum[0,0) is valid, and all assignments are rejected. Browser input is intentionally narrower: 1–8 integer values, −99…99, at most eight version handles.

The proof separates new-value correctness from old-value preservation. Each changed path has at most ceil(log2 n)+1 nodes; the initial full binary tree has 2n−1 nodes. Queries use a disjoint cover. Python integer bit costs are separate from counted node/arithmetic operations. The browser's copied arena arrays and graph layout are inspection overhead, not a benchmark of production path-copy performance.

**Ownership.** Retained handles determine a union of reachable physical identities. Releasing a root cannot release objects still reachable from another. The browser retains its arena for inspection, so the readout explicitly models potential reclaimability rather than actual garbage collection. Node equality in the native inspection helper is object identity (`eq=False`), avoiding accidental coalescing of distinct equal-valued nodes.

**Histories.** A record is eligible if its snapshot time is no later than the queried saved snapshot. Binary search finds the first later time; its predecessor is the answer. Repeated writes in a still-unsaved ID replace its record; writes after saving append. Creation IDs do not encode ancestry across sibling versions. Native tests cover saved/no-saved boundaries, untouched indices and monotone history timestamps.

**Rank.** Prefix roots store value-bucket occurrence counts, not original array indices. For [l,r), P_r−P_l is a nonnegative population because prefixes are nested. The rank invariant uses left population c; retain k on the left, or subtract c when going right. Equal values occupy multiple occurrence ranks. Arbitrary siblings and a later edited source array do not satisfy the same preprocessing contract. The browser shows expanded exact histograms for inspection; the native algorithm reads two tree nodes per descent level.

**Lazy addition.** Internal `total = left.total + right.total + add*width`. Child totals exclude the parent's tag. Queries carry strict-ancestor additions and add the carry only once at a fully covered return. A full-cover update allocates one node and shares its children; a partial update retains the parent's tag and rebuilds changed descendants. No old node is pushed into or modified. This design promises range addition and sum only, not mixed set/add composition.

**Queue limit.** The deliberately naive tuple demonstration counts repeated reversal visits from the same retained old state. It explicitly includes no O(1) queue performance promise and notes tuple slicing's additional copying. Efficient persistent queue designs are linked as further reading, not falsely ruled out by the counterexample.

## Meaningful automated verification

Commands run from the nested application repository:

```text
node scripts/verify-persistent-structures-models.mjs
node scripts/verify-persistent-structures-examples.mjs
node scripts/review-persistent-structures-lesson.cjs
node scratch/persistent-structures-final-reading.cjs
node scripts/verify-curriculum.mjs
```

- **JS model PASS:** 800 independently copied branching histories; 574,000 historical interval answers plus exact cover checks; 10,434 exhaustive rank comparisons against sorted subarrays (all arrays up to length5 over −1/0/1); identity-shared unmodified children, actual path allocation, prior-store immutability, compact-figure ID/edge/total contract, no-op/empty/invalid inputs and history predecessor checks.
- **Native PASS:** all seven complete programs run separately on Python3.12.14 and exactly match printed output. `verify-persistent-structures-native.py` checks 1,485,000 historical point/lazy sums against eager copied arrays across 450 seeded branching scenarios, including empty arrays; 10,434 exhaustive sorted rank answers; 176,224 historical snapshot lookups against independent copied snapshots. It checks allocation from independently filtered containing intervals, old-version answers, tag invariants, reachability by separate iterative object-ID traversal, immutable fields, invalid endpoints/ranks and large Python integers. These are not merely JS/Python agreement tests.
- **Source PASS:** Babel parses lesson/labs; no accidental literal-number/sequence JSX expressions. Babel formatting preserved normalized ASTs for models/labs/checks; PostCSS preserved selector/declaration signatures. Structural JSX line expansion ignores only non-rendering newline-only text. Native examples remain complete readable Python strings.
- **Curriculum PASS:** 28 modules,1,218 stable topics,314 individual briefs,7 paths at this review snapshot; acyclic resolved prerequisites and module reading order. Shared publication/build/payload/failure-recovery integration belongs to the root agent; no concurrent production build was run here.

## Browser and ordinary-reading evidence

Microsoft Edge via Playwright on the shared Vite route, at **1440,390 and320px**. Final screenshots and JSON: `scratch/persistent-structures-lesson-review/`. **43 final review screenshots** comprise mechanism/interaction captures, hints and ordinary-reading views. A separately named `ownership-debug390.png` preserves the diagnostic image of a subsequently fixed defect; it is not final evidence. `results.json` and `final-reading.json` record the passing runs.

Actual checks: all ten intro anchors exist once and navigate correctly; seven executable-example blocks; two curated problem links; initial v0/v1 roots and sums; branch from v0 after v1 exists; v1 survives sibling edits; physical table grows by the claimed allocation; conceptual ownership leaves nine nodes when only v1 remains and zero when all root handles are released; invalid input leaves active state unchanged; empty sums; no-op zero allocations; eight-version bound; one-element arrays; predecessor before/after writes and never-written index; rank steps, duplicates, invalid rank and all eight value buckets. Reset, select/input controls, Enter/Space activation, separate hint/solution disclosures, scroll focus and the root-centering control are reviewed.

Visual review opened the actual desktop/mobile PNGs, including compact path and stack figures at320, initial full DAG at1440 and390, one-root ownership at320, history search at320, rank decision at320, ordinary path-figure context at320 and ordinary lazy-table context at1440. The root graph uses readable native SVG labels and local horizontal scrolling on narrow screens; a compact entire-path figure appears inline before the investigation. Initial graph roots are centered into the mobile viewport, and the learner can recenter after panning. The compact figure's main text is18px in a330-unit viewBox, displayed252.40625px wide at320: about13.77px effective text. The final mobile root check verifies both root labels stay inside the visible scroll region.

Real defects found and repaired during review: mobile checkbox labels had flex-shrunk their inputs to zero width; explicit nonshrinking handles now provide visible44px target rows. The default full DAG was unnecessarily wide on desktop; geometry was compacted without shrinking node text. Mobile initial scrolling hid both root objects; root-centered initial framing plus a recenter control fixes that. Explicit select accessible names avoid concatenating option text into labels. Scoped list padding keeps trace numbers inside the mobile lab. Independent practice hints no longer reveal the explained solution at the same time.

No horizontal page overflow, page exceptions or lesson console errors remained. The environment blocked the existing external Google Fonts stylesheet with `ERR_NETWORK_ACCESS_DENIED` at `https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap`; the exact failed request is recorded separately, and fallback-font screenshots were reviewed. Concurrent-development Vite HMR sockets were isolated during test sessions; expected websocket diagnostics were excluded specifically. This is author testing, not an observed beginner study or user approval.

## Files and handoff

Seven semantic production sources:

- `src/learn/data/topics/persistent-data-structures-structural-sharing-versioned-queries.jsx`
- `src/learn/data/persistent-structures-models.js`
- `src/learn/data/persistent-structures-examples.js`
- `src/learn/components/lesson-labs/PersistentStructuresLabs.jsx`
- `src/learn/components/lesson-labs/persistent-structures-labs.css`
- `src/learn/data/practice/persistent-data-structures-structural-sharing-versioned-queries.js`
- `src/learn/data/curriculum/blueprints/persistent-data-structures-structural-sharing-versioned-queries.js`

Focused checks are the three Node scripts above and `scripts/verify-persistent-structures-native.py`. Native fixtures/manifest live in `scratch/persistent-structures-native-verification/`. This task changed no shared manifest/index/ledger/handoff/global style files. The parent agent registered the source for review and owns final integration.
