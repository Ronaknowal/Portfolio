# Network Flow, Minimum Cuts & Bipartite Matching — author verification

10 September 2026. Stable ID `network-flow-minimum-cuts-bipartite-matching`, DSA position19. Complete author implementation and review; root/integrated acceptance and user approval remain separate. Preserve the existing title, stable identity, prerequisites and module order. [Design](NETWORK-FLOW-LESSON-DESIGN.md) records conceptual hurdles, scope decisions and initial visual contracts. Root registered the complete draft; publication alone was not counted as completion.

## Source and learning coverage

The seven semantic source owners are:

- `src/learn/data/topics/network-flow-minimum-cuts-bipartite-matching.jsx`
- `src/learn/data/network-flow-models.js`
- `src/learn/data/network-flow-examples.js`
- `src/learn/components/lesson-labs/NetworkFlowLabs.jsx`
- `src/learn/components/lesson-labs/network-flow-labs.css`
- `src/learn/data/practice/network-flow-minimum-cuts-bipartite-matching.js`
- `src/learn/data/curriculum/blueprints/network-flow-minimum-cuts-bipartite-matching.js`

There was no prior lesson body to preserve. The final lesson starts from allocation and explicit capacity/conservation checks, then exposes signed residual changes and original-edge identity, proves the directed cut inequality and stopping certificate, implements integer solvers, recovers discrete matching/cover/Hall witnesses, derives additional constraint transformations and a binary-label cut, and distinguishes global random cuts from specified terminal cuts. Required mechanisms/proofs remain visible; Dinic and Karger are optional depth with complete examples. Nine independent local tasks have separate closed hints and explained solutions, including objective changes and a capacitated assignment capstone. Counts describe this lesson, not an authoring quota.

The default six-node graph is S,A,B,C,D,T with original edges S→A,S→B,A→C,A→D,B→C,C→T,D→T, all capacity1. BFS first chooses S,A,C,T; its second route S,B,C,A,D,T cancels original edge2. Final original flows are `[1,1,0,1,1,1,1]`, value2; source-side cut `{S}` has capacity2. This fixture was independently checked and is shared consistently across teaching, native code and browser state.

## Computation and proof checks

`node scripts/verify-network-flow-models.mjs` passes:

- 1,130 model networks against independently enumerated cut minima, including all729 three-vertex directed capacity0/1/2 assignments,400 seeded networks with parallel/antiparallel/self-loop occurrences, and the instructional fixture.
- Every stored state checked for capacity and incidence balance; every augmentation replayed from its original-edge/direction identity and bottleneck. Input immutability and invalid integer, terminal and starting-flow contracts checked.
- All512 three-by-three compatibility matrices checked against independent matching-choice and vertex-cover subset oracles. Returned cover touches every edge; returned matching endpoints are disjoint; Hall neighbor sets and exact deficiency are checked separately.
- All384 six-cell labelings across penalties0…5 checked using an independently enumerated grid objective; cut optimum and recovered-label energy agree.

`node scripts/verify-network-flow-examples.mjs` passes all nine exact stdout programs on Python3.12.14, standard library only. The script materializes the complete displayed programs in `scratch/network-flow-native-verification/` and calls `scripts/verify-network-flow-native.py`:

- Both Edmonds–Karp and Dinic checked on1,080 networks against finite cut enumeration. Native capacities include a100-digit integer; opposite edges, parallel edges, self-loops, empty edge sets and disconnected vertices are covered.
- 46,656 candidate feasible-flow assignments independently enumerated across the complete three-vertex capacity family, relating incidence feasibility to cut minima without reusing augmenting-path logic.
- All512 matching fixtures plus unequal/empty partitions and duplicate permissions checked; cover and Hall identities checked independently.
- 250 lower-bound circulation cases compared to exhaustive interval-flow assignments, including self-loops, mandatory impossible one-way flow and zero mandatory flow.
- 160 vertex-capacity cases and160 supply/demand cases checked against independent original-flow enumeration; recovered witnesses, used/unused supply and unmet demand are checked.
- 220 small binary energy problems compared to all Boolean assignments, including empty input and zero penalties.
- Exact rational contraction-choice distributions enumerated on all64 four-vertex simple graphs and one parallel-edge fixture. Every returned cut is valid, disconnected cases return zero, each fixed positive minimum cut meets its1/6 lower bound for four vertices, and the parallel fixture's minimum-cut probability is exactly5/7. This tests probability structure, not just whether a fortunate seed returns the right answer.

Native samples: feasibility audit distinguishes edge-bound from conservation failures; default Edmonds–Karp and Dinic return value2; matching returns exact cover and shortage; split vertices limit throughput to5; supplies deliver4/all met or5/one unmet; mandatory cycle returns `[2,2,2]` and one-way lower flow is infeasible; binary minima at penalties0,2,5 are2,8,15; Karger seed12 first returns a valid size4 cut and the best of20 returns size3. All expected output is literal verified stdout, not hand-invented results. Inputs reused across multiple passes are materialized where appropriate.

Proof/contract review explicitly covers:

- Net source output, not a blind sum when incoming source edges exist; edge bounds and internal conservation are both required.
- Paired residual occurrence identity, including a genuine original reverse edge; forward and cancel amounts cannot be conflated.
- Directed cut capacity counts only outgoing original capacities. Equality needs saturated outgoing edges and zero incoming flow in the residual source cut. Nonunique optimal witnesses are allowed.
- Arbitrary-path integer Ford–Fulkerson's numerical F bound versus encoded capacity length; Edmonds–Karp's O(VE) augmentation charges and O(V+E+VE²) implementation bound with reached-vertex dictionaries; bit arithmetic and optional trace storage are separately described.
- Dinic's increasing levels, current-arc cursor, per-phase blocking argument and general O(V+E+V²E) operation bound. Python recursion depth is an explicit implementation limitation.
- Existence of integral optima versus fractional feasible flows; arbitrary real mathematical theorem versus exact-arithmetic algorithm and integer-only local contracts.
- Unit-capacity matching recovery, maximal versus maximum, left-saturating versus perfect, bipartite-only cover equality, constructive Hall obstruction and a triangle counterexample.
- Supply ceilings versus met demands; lower-bound repair sign derived from `out_x−in_x=in_lower−out_lower`; circulation feasibility distinguished from maximum lower-bound s–t optimization.
- Source-side F/sink-side B binary labeling; terminal costs and opposite neighbor arcs account for every candidate energy exactly; no double charging, negative-capacity shortcut or empirical image-accuracy claim.
- Global nonempty undirected cut versus fixed terminals; edge occurrences preserved under contraction, discarded self-loops, conditional survival versus independent whole-trial repetition; reproducible seed is not a probability or optimality certificate.

## Visual and browser evidence

`node scripts/review-network-flow-lesson.cjs` exercises the real registered route at1440,390 and320px in headless Edge. Results/screenshots are under `scratch/network-flow-lesson-review/`. The suite checks all10 real reading anchors, nine native-example blocks, two official problem links, working optional-depth/solution disclosures, Enter/Space, zero capacities, invalid capacity draft preserving the active problem, reset/back, exact cancellation route, linked cut-side/crossing-edge updates, matching reassignment, empty and deficient compatibility matrices, cover/Hall readouts, binary cost decomposition and optimum application. Page overflow and lesson page/console errors are absent.

Visual review opened actual screenshots rather than relying only on layout metrics. It inspected desktop and narrow residual cancellation, directed cut certificate, deficient matching with alternating arrows and original vertex identities, binary labeling, paired residual accounting, vertex splitting and conservation. Final source choices:

- Six-node diagrams fit naturally; no minimum-width crop of a small graph. A discovered undefined font-variable shorthand was replaced by explicit20px SVG text, measured at about13.85px after scaling at320px. The numerical ledger remains available.
- Preview route arrows encode adding versus dashed cancellation; the edge labels remain the pre-send flow until the operation is applied.
- Cut toggles now update both green source-side vertices and outgoing original-edge highlights, as well as the capacity ledger. The model-derived residual side can be restored explicitly.
- Matching has a compatibility matrix and bipartite geometry, with actual assignment edges, alternating directions, reached-node rings and the returned cover. The independent textual sets remain visible.
- The six-cell labeling grid shows each selected label and both costs, with explicit unary/boundary decomposition and the global cut value. Colors are supplemented by B/F and numbers.
- A redundant fifth conservation-table column was removed after mobile review; compact In/Out/Net columns fit without hiding the numerical mechanism. Remaining local scroll containers are keyboard-focusable with visible focus; captions identify their data.
- Scoped source formatting expanded control flow and structural JSX. `scratch/format-network-flow-sources.cjs` proves normalized AST and CSS selector/declaration conservation; `scratch/format-network-flow-jsx.cjs` changes only non-rendering newline-only structural JSX whitespace. Learner-facing number spacing was polished independently of code, URLs and exact program output.

`node scratch/network-flow-final-reading.cjs` checks ordinary reader screenshots at1440/320 with the fixed navigation present. It inspects the actual residual and matching introductions and measures the SVG labels; no page overflow or page errors. Component screenshots hide the fixed navbar only for isolated element capture; ordinary reading screenshots retain it. There are27 component/entry screenshots and four ordinary-reading screenshots,31 total. Representative final images were opened at desktop and mobile widths; this is author visual inspection, not an observed beginner study.

Environment caveat recorded explicitly: this browser environment intermittently blocks the pre-existing Google Fonts stylesheet with `ERR_NETWORK_ACCESS_DENIED`; `network-<width>.json` records the exact external URL/error. The suite distinguishes this existing external-font failure from lesson errors and reviews the available fallback font too. Vite HMR websockets are deliberately blocked during these isolated tests so concurrent authoring cannot reset lab state; the expected Vite websocket diagnostic is not a lesson error. No shared font/global CSS/runtime changes were made.

## Research and resource ledger

Primary resources checked on10 September2026; prose and fixtures are original, with independent derivations/tests:

| Claim / resource | What was actually inspected | Learner use / caveat |
| --- | --- | --- |
| [Stanford CS261 lecture2](https://theory.stanford.edu/~tim/w16/l/l2.pdf) | Full relevant residual/cut proof and BFS/blocking-flow analysis text | Optional written proof route; old frontier/performance remarks excluded |
| [Stanford CS261 lecture4](https://theory.stanford.edu/~tim/w16/l/l4.pdf) | Binary cut construction, pairing reduction and Hall argument | Local energy convention derived explicitly; source's looser use of “perfect” is distinguished |
| [Princeton maximum flow](https://algs4.cs.princeton.edu/64maxflow/) and [Hopcroft–Karp API](https://algs4.cs.princeton.edu/code/javadoc/edu/princeton/cs/algs4/HopcroftKarp.html) | Overview/code contracts, vertex splitting, bipartite matching/cover theorem and specialized cost bound | Overview is under construction; no blind reliance on its incomplete exposition |
| [Kingsford lower-bound/demand slides](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/flowext.pdf) | Full transformation equations and feasibility condition | Its demand sign convention is not silently mixed with the local repair balance |
| [Stanford CS265 contraction notes](https://web.stanford.edu/class/archive/cs/cs265/cs265.1254/Lectures/Lecture2/l2.pdf) | Algorithm, multiplicity, fixed-cut conditional survival and repetition theorem | Local implementation's slower cost stated; no historical fastest-algorithm claim |
| [MIT6.0462015 lecture13 video resource](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/resources/lecture-13-incremental-improvement-max-flow-min-cut/) | Official video/transcript/linked-notes resource page | Useful spoken/board alternative; no playback or precise timestamp claim. The linked PDF request failed in this session and is not reported as reviewed |
| [LeetCode886](https://leetcode.com/problems/possible-bipartition/) | Public official statement, ID/title/Medium and graph bounds | Explicitly a prerequisite structural check, not a max-flow exercise |
| [LeetCode1349](https://leetcode.com/problems/maximum-students-taking-exam/) | Public official statement, ID/title/Hard, conflict directions and8×8 bound | Column parity, not row-plus-column parity, supports the matching reduction |
| LeetCode1820 official endpoint | Returned Premium-only access gate | Excluded from the verified public practice set; no statement/editorial/submission claim |

Both received [Network Flow destination notes](topic-notes/network-flow-minimum-cuts-bipartite-matching.md) were assessed and implemented: exact cover/Hall in the core, Karger as optional global-cut depth. A scoped search and inspection of the current Combinatorial Optimization body found no complete weighted-matching/min-cost-flow treatment; a durable [destination note](topic-notes/combinatorial-optimization-approximation-algorithms.md) records objective changes, residual costs, general-graph matching limits and the sources for future assessment without rewriting that lesson. The unrelated bit-manipulation inbox remains unresolved.

`node scripts/verify-curriculum.mjs` passes28 modules,1,218 stable topics,312 individual briefs and seven paths at this registered snapshot. Root owns final integrated build/loading/conservation review, shared ledger/practice coverage and acceptance. No deployment, external submission or user approval is claimed here.

Final source freeze: all seven semantic source SHA-256 fingerprints are saved in `scratch/network-flow-lesson-review/final-source-hashes.json`. The final browser interaction run passed at all three widths after compact-table, linked-cut, numeric-spacing and focus review. The real global teaching focus style supplies a visible solid amber2px outline; verification checks visibility rather than requiring a particular lesson-local pixel width. Final ordinary320px matching prose and the complete compact conservation table were opened and read. Further source changes require rerunning affected checks and updating this fingerprint record before root integration is recorded.
