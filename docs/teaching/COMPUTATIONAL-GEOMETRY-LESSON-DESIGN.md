# Computational Geometry: lesson design

10 September 2026. Stable ID `computational-geometry-robust-predicates-convex-hulls`, DSA position20. Complete the current scoped authoring standard; this design is not completed verification.

The exact inventory command and its authoring notes were read. At preflight the topic had an expansion plan but no existing lesson body or individual blueprint; verified by filesystem lookup, not inferred from publication. No destination note existed then. The unresolved general bit-manipulation inbox does not apply. Keep the title: orientation, robust decisions and hulls are its core, with segment/polygon applications making the mechanism useful. Do not change module order: Network Flow precedes it, Persistent Data Structures follows.

## Coverage and prerequisites

Assume pairs of coordinates, comparisons, loops, stacks and sorting. Recap vectors, signed area and integer/float distinction locally because the planned geometry/numerical prerequisites must not become hidden detours. All core algorithms operate on exact integer coordinates in Python; browser grid models use bounded integers. A separate BigInt calculation exposes precision failures without claiming a production adaptive-predicate implementation.

| Hurdle | Treatment and mechanism | Representation | Independent evidence |
| --- | --- | --- | --- |
| A geometric branch needs a sign | Derive determinant from two displacements; orientation and doubled area; axis conventions, repeated points, translation/scaling | Equal-unit coordinate plane with oriented triangle and live displacement/product arithmetic | Integer determinant against shoelace expansion, permutations and translation |
| Crossing is more than two infinite lines | Bounding intervals, four orientations, proper crossing, touching, overlap and point-segments; explicit closed-set contract | Two labeled segments, movable endpoint, classified intersection and four directed-side tests | Independent Fraction parameter solve including parallel/degenerate cases |
| Rounding changes topology | Exactly representable input/product cancellation; lost input distinctions; fixed epsilon scaling counterexample; exact decimal strings versus stored binary; predicates versus constructions | Arithmetic pipeline comparing exact and binary64 products, not an invented magnified geometric gap | Python int/Fraction versus JS Number/BigInt fixtures |
| A hull is a region, not the original shape | Convexity/convex combinations, extreme vertices versus boundary records, lower/upper monotone chains and invariant | Initial hull silhouette plus actual candidate/pop/push trace tied to sorted points and stack | Exhaustive supporting-line boundary oracle, containment and turn properties |
| Boundary policy changes the output | Strict-corner versus all-boundary collinearity; deduplication, all-collinear case, zero/one/two points | Same point set under both policies; degenerate presets | Unique normalized outputs and independent supporting-edge set |
| Inside, on, outside are distinct | Signed polygon area; boundary-first half-open horizontal crossing; simple polygon assumption; concavity versus hull | Query/ray/selected-edge parity trace on a U-shaped polygon, area/readouts | Exact winding-number oracle, rectangle enumeration, reversal invariance |
| Useful transfer without unsafe geometry claims | Broad-phase bounds, linear support direction proof, conservative convex envelope; canonical rational line directions | Compact original concave envelope contrast; worked support table and complete programs | Full-point versus hull maxima; brute-force maximum collinearity |

Core closes with complete native orientation/intersection/hull/robustness/polygon/support programs, followed by independent changed fixtures, diagnosis and an integrated report. Optional exact rational intersection and normalized directions develop construction versus decision and hash-key transfer; they do not silently require slope division. No artificial lab or question count.

Do not claim this page teaches the entire computational-geometry field. Exact topic-plan inspection identified the existing Configuration Space, Collision Checking & Feasible Trajectories owner before Motion Planning; its brief already covers footprint inflation and continuous edges. A [destination note](topic-notes/configuration-space-collision-checking-feasible-trajectories.md) preserves the bridge without duplicating that chapter. Voronoi/Delaunay, sweep-line arrangement algorithms, 3D hulls and spatial indexes are additional specialist families, outside the planar predicate/hull finish line. The [reasoned scope note](topic-notes/computational-geometry-robust-predicates-convex-hulls.md) keeps future assessment visible without implying those families completed.

## Visual contracts

All spatial diagrams use Cartesian y-up and equal x/y coordinate units. SVG reverses only screen y. Labels and captions state this; no quantity is inferred from decorative area or color. Small diagrams sit next to definitions; lab initial states already show the actual geometry. Keyboard ranges and labeled inputs cover interactions; dragging is unnecessary. Each investigation has prediction, deterministic reset/back, exact text result and changed-state task.

Orientation varies one point while the directed baseline stays fixed. Segment presets expose proper/touch/overlap/disjoint/point cases, then the learner changes one endpoint and sees the reason. Hull input edits take effect only on Apply; invalid input preserves the existing trace, and reset restores the original. A maximum20 input records and coordinates0…8 bound work. Polygon input consists of fixed valid simple polygons; only the query and inspected edge change. Precision controls select actual binary64 counterexamples with exact arithmetic shown; the near-collinear gap is explicitly below a normal plotted pixel, so no false-scale illustration is supplied.

## Research actually inspected

Retrieved10 September2026. These sources support conventions and provenance; examples and proofs are original local teaching constructions.

- Shewchuk's [robust predicates author page](https://www.cs.cmu.edu/~quake/robust.html): complete page; orientation/incircle determinant failure, adaptive precision description and machine assumptions. Did not inspect or port the complete adaptive implementation.
- [CGAL6.2.1 kernel manual](https://doc.cgal.org/latest/Kernel_23/index.html): coordinate types, exact predicates versus constructions, orientation and intersection return variants. No CGAL library was executed by opening documentation.
- [CGAL hull manual](https://doc.cgal.org/latest/Convex_hull_2/index.html): definitions, counterclockwise extreme-point output, Graham–Andrew/Jarvis complexity and distinct collinear boundary handling.
- [CGAL polygons manual](https://doc.cgal.org/latest/Polygon/index.html): inspected the simple-polygon definition, signed-area and three-way bounded-side contracts, and the distinction between representing a polygon-with-holes record and validating its geometry. The local model assumes valid simple presets; it does not implement a polygon validator.
- [Python fractions documentation](https://docs.python.org/3/library/fractions.html): constructor behavior for strings and already rounded floats. Current Python 3.14 documentation was consulted; the complete standard-library programs were executed in Python 3.12.14, with their actual Fraction results checked.
- [MIT6.046 Lecture2](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/resources/lecture-2-divide-conquer-convex-hull-median-finding/), official [notes](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/7463c413c944ed72b46a3c3d02b49448_MIT6_046JS15_lec02.pdf) hull pages1–3 and transcript opening hull explanation inspected. Useful alternate video/notes after the core: it uses divide-and-conquer, clockwise representation and general-position simplifying assumptions, unlike our explicit degeneracies. No full-video playback claimed.
- Official LeetCode1037/1232/812/836/587/149 statements opened: exact ID/title/difficulty/constraints/public statement visible. Four Easy, two Hard.587 includes **all boundary trees**;836 requires **positive area**, excluding touches;149 and1232 inputs are unique. No submission/editorial access or external solution execution claimed. Curated stages follow the actual mechanisms, not difficulty or a quota.

## Finish line

Native stdout for every displayed program, independent model/native oracles, closed-form exercise checks, actual1440/390 keyboard/edge/anchor/reset/normal-reading screenshots opened, source/provenance and unchanged identity. Root owns registration, generated metadata, build and integrated ledger. Keep author evidence separate from user approval.

The implemented disposition and final evidence are recorded in [COMPUTATIONAL-GEOMETRY-VERIFICATION.md](COMPUTATIONAL-GEOMETRY-VERIFICATION.md). All planned core mechanisms are present; future specialist families and physical-motion connections retain the reasoned notes above. Author review does not substitute for user acceptance or the root's integrated build.
