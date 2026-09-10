# Computational Geometry — implementation and author verification

Completed 10 September 2026 for `computational-geometry-robust-predicates-convex-hulls`, DSA position 20. This is author verification, not user acceptance or integrated production verification. The [durable evidence JSON](evidence/computational-geometry-author-review.json) records exact times, source fingerprints and the checked results. Root owns registration, curriculum/generated artifacts and the integrated build.

## Learning and scope disposition

At preflight this entry had a specialist expansion plan and no lesson body or individual blueprint. The [individual design](COMPUTATIONAL-GEOMETRY-LESSON-DESIGN.md) now supports a self-contained route from points and displacements through signed turns, closed-segment contact, numerical robustness, exact rational construction, convexity and two hull output contracts, polygon area/containment and geometric applications. The stable ID, title and module order are unchanged. Network Flow precedes this lesson; Persistent Data Structures follows.

The lesson rebuilds required coordinate/determinant and numerical distinctions locally. Eight complete standard-library Python programs have checked stdout. Six original exercises change inputs or contracts, include initially hidden hints and reasoned solutions, and end in a composed geometry report with concrete acceptance checks. Six official LeetCode tasks give complementary transfer rather than replacing the local exercises: 1037/1232/812/836 are foundation, 587 is the all-boundary hull core, and 149 is an explicitly optional canonical-direction/hashing extension. Their currently published difficulty labels are four Easy and two Hard. Public statements, constraints and boundary policies were inspected; no external solution/submission or universal interview-readiness claim is made.

The motion application preserves the distinction between point paths and finite footprints. An exact inventory lookup identified the existing [Configuration Space/Collision Checking owner](topic-notes/configuration-space-collision-checking-feasible-trajectories.md), whose brief already covers inflation and continuous edge checks. A proposed connection is recorded for its author to assess. Deeper geometry families remain explicit [future scope decisions](topic-notes/computational-geometry-robust-predicates-convex-hulls.md); the current page does not pretend to implement Delaunay/Voronoi, sweep-line arrangements, spatial indexes or 3D/spherical geometry.

## Representation contracts and visual review

| Placement / representation | Meaning and limit | Actual evidence |
| --- | --- | --- |
| First worked calculation and directed-triangle investigation | Exact displacement products determine orientation and unsigned area. Equal x/y units, Cartesian y-up; SVG only reverses display y. Changing/reversing C/A/B is algebraically visible. | Ten changed/reversed states per viewport; zero/coincident/extreme points; keyboard reset; final desktop/phone images opened. The first inline calculation was repaired to preserve its four lines and fits 320px. |
| Closed-segment investigation, section 2 | Solid AB, dashed CD, four side tests, contact classification; pale overlap denotes a segment, grouped labels denote coincident endpoints. | Five presets and one moved-endpoint case per viewport, exact model/readout comparison and keyboard reset. Centered labels repair nearby B/D crowding; mobile overlap and desktop point-segment images opened. |
| Arithmetic comparison, section 3 | Actual BigInt-before-conversion versus JavaScript Number products. Representable inputs can still yield a wrong rounded sign; already lost input coordinates cannot be recovered afterward. | Four exponent cases and one input-collapse case per width; exact strings/readouts, status and reset checked. Both failure paths opened on phone/desktop, with targeted 320px long-integer reading. No fictitious visible gap or universal threshold is drawn. |
| Concave envelope figure before hull construction | Original courtyard has a notch that its convex hull fills; Q belongs to the hull but not the original region. | Actual same-coordinate polygon/hull geometry, equal-unit axes and explicit caption; desktop and phone images opened in ordinary reading. |
| Monotone-hull investigation, section 4 | Sorted occurrences, active chain, candidate, removed vertex and completed lower chain; corners-only versus all boundary locations; unique/collinear/empty contracts. | All 35 steps of a nontrivial trace per width, ten preset/policy outputs per width, back/restart/end controls, invalid input preservation and changed 0/8-edge data. Pop, boundary and edge-coordinate images opened; 320px edge labels additionally reviewed. |
| Polygon-ray investigation, section 5 | Simple hole-free presets; boundary overrides parity; half-open vertical eligibility and directed determinant decide right crossings. Actual signed area and selected edge are shown. | Three shapes × six queries, all 90 inspected edges per width; boundary/inside/outside, next/previous/reset. Final opening and notch-floor screenshots opened; the selected edge is explicitly distinguished from the whole-polygon classification. |

The representations were chosen for the mechanism, not a diagram or lab quota. Spatial views use their actual coordinates and labeled stroke styles. The precision investigation intentionally uses an arithmetic path instead of inventing a large visible near-collinear gap. No graph claims measured performance or unobserved physical accuracy.

Ordinary-reading screenshots were opened for all eight route destinations at phone width and the key mechanism/hull/polygon/practice destinations at desktop width. Additional normal viewport captures inspect the hull proof, ray convention, first complete program/output and references. Source/native explanations were also read as a continuous lesson. The 1440/390 lab screenshots include actual changed states, not only empty initial panels. The evidence JSON distinguishes screenshots captured by scripts from those actually opened.

## Independent native/model verification

Commands run from the application repository:

```powershell
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/prepare-computational-geometry-examples.py
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-computational-geometry.py
node scripts/format-computational-geometry.cjs
node scripts/review-computational-geometry.cjs
node scripts/review-computational-geometry-reading.cjs
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/finalize-computational-geometry-evidence.py
```

The eight displayed programs ran in isolated Python 3.12.14 subprocesses and matched captured stdout exactly. Each is also loaded into an isolated native namespace for the independently formulated checks. Native result: `scratch/computational-geometry-verification/results.json`, 13:02:19.938723 UTC. Its model/example fingerprints still match the final source; later edits only repaired presentation, labels and reset selection.

- 829 orientation cases use an independent three-point shoelace expansion, including the full 3×3 triple grid and larger bounded coordinates.
- 6,561 segment cases use exact Fraction line parameters and independent parallel/degenerate projection membership, not a copy of the displayed four-orientation branch logic. JavaScript and Python classifications agree with that oracle.
- 1,224 hull cases include both output policies over all 3×3-grid subsets plus changed random records. An independent exhaustive supporting-line oracle determines boundary/extreme sets; checks include unique normalized output, supporting edges, native ordering and the completed trace stack.
- 648 polygon cases compare the horizontal parity model with an independent vertical-ray signed-winding calculation using rational interpolation. They include four polygons, both boundary directions, every grid query, signed fan area and boundary handling.
- Five precision fixtures compare exact intended integer arithmetic and actual binary64 calculation; nine invalid model/input contracts are rejected. Large integer bounds and exact intended-versus-represented differences are explicit.
- 200 native canonical-direction cases use brute-force all-pairs collinearity with duplicate multiplicities as the oracle.
- All six local exercise groups have independent numerical/acceptance assertions, including 6/5 rational intersection, all-boundary rectangle ordering, the smaller courtyard's area 20 and the integrated report's four contact classes/support scores.

Finite oracles establish the enumerated cases; the lesson separately justifies the general algorithm invariants and arithmetic assumptions. The browser model stores bounded teaching snapshots and can use quadratic trace storage; the native monotone-chain algorithm has linear auxiliary chain storage after sorting. Python integer operation counts do not pretend to be unit-cost for arbitrarily large bit lengths.

Root additionally read the supporting-line/processed-prefix proof, all-collinear exception, half-open crossing and sign identity, area, precision/construction, support application, canonical keys and changed practice. No actionable defect was reported. This is a bounded independent source/proof review, not a claim of extra numerical tests; root preserves its separate final-fingerprint record.

## Browser, source and formatting evidence

Final functional result `scratch/computational-geometry-browser/results.json` at 13:14:15.200 UTC passes 1440×1000 and 390×1000 in headless Microsoft Edge against the registered lesson on port 5173. Each viewport checks all eight keyboard-activated anchors actually arrive below the fixed header, every displayed example/output by title, six official links and their new-tab attributes, initially closed local answers/LeetCode hints, keyboard solution disclosure, optional prerequisite-dependent practice, resets/back/end/invalid cases, all five investigations, no page/lab overflow, no SVG label outside its view box and no JavaScript page errors. The four-line introductory calculation also has a dedicated assertion.

Supplemental ordinary-reading result at 13:16:11.363 UTC passes 1440,390,320. It captures actual program/output, proof, ray and sources reading; native code retains intentional horizontal scrolling on narrow viewports without widening the page. The calculation and arithmetic labels fit. This is not a claim of a full 320px interaction matrix: the full interaction checks were at 1440 and 390.

Normalized JavaScript AST/string and CSS-meaning preservation passed the formatter. The selected topic imports only its own examples, models and practice plus appropriate small shared teaching components. No aggregate DSA content body or math renderer was introduced. The parent owns production-loading/build validation.

## Research and limits

The design records actual inspected scope and links: Shewchuk's author page, CGAL6.2.1 Kernel/Convex Hull2/Polygon contracts, Python Fraction constructors, and official LeetCode statements. The annotated alternate resource is MIT6.046 Lecture2's official video page with its hull notes pages1–3 and relevant opening transcript inspected. Its divide-and-conquer/general-position/clockwise treatment is explicitly distinguished from this lesson's monotone-chain/degeneracy/CCW contracts. No full video playback, CGAL runtime test, LeetCode editorial or accepted submission was claimed.

The local polygon routine assumes valid simple input; it does not implement production polygon validation, holes or spherical geometry. The arithmetic comparison is not a port of an adaptive-predicate library, and exact arithmetic cannot resolve unknown physical measurement error. No timing benchmarks, novice study or screen-reader speech session were performed. These limitations are stated where they affect the learner's interpretation, and author verification remains separate from user acceptance.
