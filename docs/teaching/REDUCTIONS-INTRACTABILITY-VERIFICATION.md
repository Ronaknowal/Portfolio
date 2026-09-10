# Reductions, P, NP & Computational Intractability — author verification

10 September 2026. Stable ID `reductions-p-np-computational-intractability`, DSA position 17. Complete scoped source is implemented and author-verified. Root owns independent/integrated acceptance, production build, shared progress and registration. Publication alone is not teaching review or user approval.

## Content and scope

The original planned topic is now a self-contained lesson. Preserve the stable title/order. It teaches certificate versus search, CNF notation, explicit encoded input length, P/NP and complement quantifiers, many-one direction/equivalence/output size, a full two-way 3SAT-to-clique proof with occurrence identity, complement/independent-set/cover identities, NP-hardness versus membership, Cook–Levin as a stated foundational premise, decision-oracle witness recovery, and practical pseudopolynomial, parameterized, restricted and approximate contracts.

Nine complete native programs supply checked output and original witness semantics. Eight independent local exercises have closed hints/explained solutions; three verified official LeetCode statements revisit meaningful complexity contracts (416 bounded numerical target, 198 path restriction, 698 small-item exact search). This count describes the selection, not a quota or universal readiness guarantee. The next module topic remains Randomized Algorithms, Sampling & Error Guarantees.

Core/deeper decisions are documented in [the design](REDUCTIONS-INTRACTABILITY-LESSON-DESIGN.md). The main route does not rely on solving a lab. Cook–Levin's full machine-tableau proof, broad hierarchy, weighted approximation design and advanced parameterized/kernel theorems are outside the declared scope. The long-clause and oracle-search branches are completely implemented within their stated boundaries. No claim that P differs from NP, or that finite enumeration proves all algorithms exponential, appears.

## Source ownership

- Lesson: `src/learn/data/topics/reductions-p-np-computational-intractability.jsx`.
- Models/examples: `src/learn/data/intractability-models.js`, `intractability-examples.js`.
- UI: `src/learn/components/lesson-labs/IntractabilityLabs.jsx`, `intractability-labs.css`.
- Individual brief/practice: matching stable-ID files under `src/learn/data/curriculum/blueprints/` and `src/learn/data/practice/`.
- Checks: `scripts/verify-intractability-models.mjs`, `verify-intractability-examples.mjs`, `verify-intractability-native.py`, `review-intractability-lesson.cjs`.

No shared content barrel, global styles, catalogue/order, manifest, generated artifacts, practice aggregate or lesson loader was edited by this author. Root registered the complete body/brief for browser review. No mathematical rendering engine or external runtime dependency is imported.

## Visual contracts and actual checks

| Representation | Question, state and mapping | Evidence and boundaries |
| --- | --- | --- |
| CertificateLab | Variable buttons feed signed literal truth cells, clause OR and formula AND. Failed witness differs from unsatisfiability. | All eight certificates on the contradictory preset reject; default F,F,F rejects and F,F,T accepts. Applied formula/editor draft separation, invalid literal rejection, keyboard switches and eight-row truth table checked. Browser always exposes three variables, including unused ones. |
| ReductionFlowFigure | The enclosing constructed A solver contains conversion f and hypothetical B solver. Conversion arrows and solver-use direction remain distinct. | Source equivalence reviewed; desktop row and narrow vertical flow opened as actual screenshots. Does not imply an implemented fast target solver. |
| ReductionLab | Every literal occurrence gets its own clause-grouped vertex; only inter-clause noncomplementary pairs have edges. Selected choices expose compatibility and recover an assignment. | Independent assignment/subset oracles and every recovered tiny clique checked. Browser repeated literal vertices, contradictory no-clique result, invalid/valid selections and selected pair table inspected. First-to-third edges route below middle group to avoid crossing unrelated boxes; crossings are not vertices. |
| EncodingFigure | Exact BigInt binary digits and DP slot count for T=2^exponent. | Every exponent1…40 checked; browser exponent40 renders41 digits and1,099,511,627,777 slots. No large allocation, benchmark or timing prediction; full input contains more than T. |
| CoverTradeoffLab | Chosen vertices versus edge coverage; disjoint matching edges give lower bound; a feasible chosen set gives upper bound; bounded exact search returns witness/call tree. | Exhaustive graph oracles validate optimum, seven budgets, cover/matching witnesses and factor bound. Browser default exact size3 vs approximation6, star1 vs2, emptygraph budget0, duplicate orientations, selfloop rejection and keyboard toggles checked. Six-vertex fixed bound; deterministic applied edge order. The exact display search is additional teaching work, not approximation runtime. |
| PathRestrictionFigure | A real conflict path, weights5,1,6,8,4; chosen positions0,2,4 sum15. | Native weighted path oracle independently verifies objective/witness; screenshot opened at320. Extra conflict edges explicitly invalidate the recurrence. |

Color has accompanying labels, buttons, legend and table/text alternatives. Essential controls use native keyboard behavior; no timers or stochastic layout. Draft errors preserve applied state. Every snapshot is derived from the same applied formula/graph; no fabricated benchmark values.

## Native and model verification

`node scripts/verify-intractability-examples.mjs` passes all **nine exact Python outputs** using Python3.12.14 standard library. Independent Python oracles additionally pass:

- **304 formula cases**, including empty formula/clause, contradiction, repeated literals and random signed clauses; complete assignment truth compared with clique existence and SAT decision-oracle witness recovery.
- **421 recovered clique witnesses**, each checked as a satisfying assignment; all source/target mappings keep occurrence identity.
- **798 original assignments** checked against existence of fresh-variable extensions from100 generated long-clause cases.
- **1,024 graphs**: every simple graph on five vertices, cover subsets independently enumerated, exact branch checked at seven budgets, matching disjointness/feasibility/lower bound/2-approximation checked, complement clique-cover identity checked on every subset.
- **350 numerical/path pairs**, exact subset-sum indices checked against combinations, weighted path optimum/witness checked against all independent subsets; negative weights and zero-target cases included.
- Invalid literals, Boolean certificate type, overlong source clauses, contradictory clique witnesses, invalid cover budgets/selfloops and negative subset values rejected.

`node scripts/verify-intractability-models.mjs` passes **358 formula cases**, **328 recovered clique witnesses**, **1,284 graph cases at seven budgets** (all1,024 five-vertex graphs plus260 deterministic six-vertex cases), exact binary-count identities, invalid input and independent snapshot/input-preservation checks. Branch events preserve the uncovered-edge/budget invariant and bounded call count. Enumeration uses independently chosen subset/assignment predicates rather than only comparing the implementation with itself.

Finite tests establish the displayed algorithms on their checked inputs. General proofs are separately written in the lesson. Models/natives are different implementations; browser bounds and native input contracts are explicitly distinguished. Integer-operation complexity and bit-size caveats are visible. SAT oracle queries are not treated as free in the executable demonstration. Long-clause repeated copying is accounted for as quadratic-in-clause-length construction despite linear output size; cover branch copied-state and output-sort work are included.

## Browser evidence and fixes

`node scripts/review-intractability-lesson.cjs` passes at **1440,390,320 px** with normal reader navigation. Each width checks nine real route anchors, nine program blocks, three direct LeetCode links, all eight contradictory assignments, formula/graph invalid-input preservation, repeated occurrences, recovered/failed clique witnesses, exact/approximate cover contracts, empty/duplicate edges, large encoding counts, Enter/Space buttons and disclosure hints, and no page overflow or page errors.

Artifacts: `scratch/intractability-lesson-review/results.json` and **34 component/entry/ordinary-reading screenshots**. Actually opened representative final images: `reduction-witness-320.png`, `reduction-witness-390.png`, `cover-bound-320.png`, `reduction-flow-1440.png`; earlier reviewed `certificate-accepted-320.png`, `encoding-large-320.png`, `path-restriction-320.png`, `reading-entry-390.png`, `cover-exact-390.png` and desktop reduction witness. Header is hidden only for isolated component captures; ordinary entry/reading screenshots preserve it.

Actual review found and fixed:

1. Raw JSX braces in an exercise caused a caught render error (`A is not defined`) and a numeric set could render as a comma expression. Replaced both with unambiguous ordinary text; full browser rerun passed.
2. A minimum SVG width clipped rightmost graph nodes on320px screens. Removed that minimum for these small diagrams and increased logical font sizes, fitting all nodes without tiny labels. Final320 images opened.
3. First-to-third-clause straight edges passed behind unrelated middle-group literal boxes. Routed those connections below the group and explained crossings; pair tables remain the explicit adjacency interpretation. Final screenshots opened.
4. Sharpened verifier checkpoint: a failed clause can rule out multiple candidates but does not by itself settle the existence question. Added explicit k>V rejection in clique membership discussion so a huge encoded threshold does not create a huge certificate obligation.

Formatting: Babel parser/generator expanded owned model/lab/verifier code with normalized AST equality (`scratch/format-intractability-sources.cjs`). PostCSS changed only whitespace, with selector/declaration signatures preserved. Native/model checks rerun after formatting. Source hashes and parse results are saved in `scratch/intractability-lesson-review/final-source-hashes.json` at freeze.

## Research and persistent discoveries

Checked10September2026:

- Clay P versus NP page explicitly says Unsolved; Cook's formal description inspected for definitions, reductions, NP-completeness, certificates, self-reduction and coNP. Historical hardware, best-known time and cryptographic claims are not reused as current facts.
- MIT6.006 fall2011 Lecture23 resource/video page and full six-page typed notes inspected. No playback claimed. Mathematical statements are independently phrased and qualified, including decision/optimization distinction.
- MIT6.046 spring2015 fixed-parameter notes' edge-branch argument and approximation notes' matching proof inspected. Apparent kernel edge-count typos/loose scheme terminology are not copied or treated as authority to skip derivation.
- OpenDSA3SAT-to-clique resource and explanatory page checked. Hosted slideshow execution is not claimed; local diagram/proof is independently complete.
- Official LeetCode416/198/698 statement ID/title/Medium difficulty and relevant constraints directly checked. No submission, editorial, account or video-solving claim.

Two destination notes preserve useful deeper ownership: [weighted approximation/relaxation](topic-notes/combinatorial-optimization-approximation-algorithms.md) appended alongside the earlier greedy note, and [bipartite maximum matching to exact minimum cover](topic-notes/network-flow-minimum-cuts-bipartite-matching.md). Exact destination IDs/plans inspected through the CLI. Both remain open for their receiving authors; the origin does not claim those deeper methods taught. The unrelated bit-manipulation inbox remains unresolved.

`node scripts/verify-curriculum.mjs` passes28modules,1,218stabletopics,310individualbriefs,7paths at this registered snapshot. Root must run final integrated build/artifact/loading checks and independent source review. Author tests and screenshots do not constitute observed beginner mastery or user approval.

Final ordinary-reading smoke: scratch/intractability-final-reading.cjs passed1440/320 after all source edits; final-reading.json records no lesson console/page errors. The expected Vite websocket message is recorded separately because HMR is deliberately blocked to isolate concurrent authoring. The final reading-core-1440.png and reading-reduction-320.png were opened after waiting for anchors and positioning the heading below the ordinary fixed navbar. The complete interaction suite already verifies all real anchors; this last pass inspects readable prose in the actual reader. Source remains frozen.

Independent review subsequently corrected one phrase in the Cook–Levin paragraph: “at-most-three-clause form” now reads “form with at most three literals per clause.” The clause-width condition was already correct in the definition, algorithms and proof; this removes a misleading clause-count reading. Root explicitly authorized this narrow exception to freeze. Focused source/JSX verification confirms exactly this phrase changed and the other six hashes remain unchanged; the body's fingerprint in `scratch/intractability-lesson-review/final-source-hashes.json` is refreshed. See [independent review](REDUCTIONS-INTRACTABILITY-INDEPENDENT-REVIEW.md) for23 independently modeled graph optimizations,106 threshold checks,14 reduction compositions and82 MILP SAT-oracle calls. No broad browser rerun was performed for this wording-only correction. Source is frozen again; root owns the updated integrated snapshot.

