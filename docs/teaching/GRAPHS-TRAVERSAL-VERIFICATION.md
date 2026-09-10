# Graphs: Representations, BFS & DFS — independent verification

10 September 2026. Scope: the third topic in the authorized sequential DSA implementation, following Trees and Heaps/Tries. Read with [the lesson design](GRAPHS-TRAVERSAL-LESSON-DESIGN.md), [DSA practice standard](DSA-PRACTICE-STANDARD.md), [teaching standard](../../LESSON-TEACHING-STANDARD.md) and [learning code standard](../engineering/LEARNING-CODE-STANDARD.md). The exact stable-ID topic plan and returned authoring notes were read before verification; no destination note or relevant unresolved routing item existed then. The coordinating author owns publication, the blueprint and curriculum integration.

## Teaching and contract review

The lesson introduces vertices and permitted relationships before graph notation, storage or traversal code. It separates layout from meaning: physical line length is not an edge weight, a crossing without a labelled vertex is not a junction, and an incoming arrow does not permit backward travel. Explicit vertices retain isolates. The representation comparison uses the same relation in edge-list, adjacency-list and matrix forms, then the search lab holds the lesson's fixed graph constant for reproducible BFS/DFS comparisons. The grid supplies a different mechanism: a neighbor rule creates an implicit graph whose distance layers can be inspected spatially.

Reviewed representation details include declared endpoints, duplicate collapse, self-loops, Boolean existence versus multiplicity, sparse/dense storage costs and adjacency-list neighbor iteration. One stored self-loop membership is distinguished from conventional undirected degree counting. Native builder order follows input encounter order; the browser uses ascending neighbors, and its sorting overhead is not hidden inside a general linear traversal claim.

BFS explicitly marks on enqueue, assigns first parents and explains why a first discovered distance is shortest only for equal-cost edges. Discovery, pending work and completion have separate meanings. Unknown-source errors and unknown/unreachable target results have stated contracts. Equal-length parent routes depend on neighbor order. A source-to-itself route has zero edges. DFS retains the next neighbor in each active frame, so its explicit stack matches actual recursive calls rather than a batch of pushed siblings. Entry, finish, parent depth and shortest distance remain distinct. Long-path examples avoid depending on recursion depth.

The review also covered the full outer scan for undirected components; weak versus strong directed connectivity; four-neighbor movement and edge-count distance; multi-source initialization and nearest-source minima; blocked, unreachable and source-equals-target cases; full grid initialization cost; flood-fill copy and same-color behavior; and the separate eight-neighbor, cell-count practice contract. The diagonal practice allows corner crossing, which is not silently imported into the four-neighbor lab. State-space extensions explain why a coordinate alone may be insufficient when keys or other state change permitted moves.

Graph cloning preserves sharing, cycles and parallel neighbor entries by object identity. Its new graph nodes and neighbor lists do not imply recursively copied payloads; the lesson declares its immutable string payload fixture and shallow payload policy. Directed cycle detection distinguishes active ancestors from finished vertices, scans every component, and only derives reverse finish order as a topological ordering after confirming acyclicity. Undirected parent-edge handling is explicitly different. Bipartition considers every component and distinguishes failure (`None`) from a valid empty mapping. Output-exponential path enumeration, weighted paths and stronger graph machinery remain scoped extensions with links rather than unsupported mastery claims.

Independent practice has prediction/trace/repair/transfer tasks, hints and worked reasoning. The ten linked platform exercises are staged after their required mechanisms, with contract changes called out. The page connects forward to Disjoint Sets / Union-Find, preserving the module sequence. Written sources and MIT video alternatives are annotated in the coordinating author's source ledger. This reviewer inspected the resulting source/practice presentation and code contracts, but did not watch entire recordings, submit platform solutions or conduct a beginner study.

## Findings resolved during review

- The native grid validator originally tested membership in the string `".#"`, allowing substring cells such as `""` and `".#"` when supplied as nested lists. The author changed validation to membership in the tuple `(".", "#")`. Empty, multi-character and numeric cells now reject; all ten displayed programs and the complete independent native corpus passed after the edit.
- The vocabulary definition of a cycle needed distinct edges to avoid including an immediate out-and-back traversal of the same undirected edge. The author added distinct-edge language and the explicit counterexample. Later directed/undirected cycle explanations now agree with that definition.
- The first static directed-cycle comparison used the full eight-vertex drawing width even though each example has only four vertices. At narrow widths D required panning, weakening the side-by-side comparison of states. The model/component owner introduced a compact four-node geometry only for these static examples. All four vertices and state labels now fit at 390px and 320px; the eight-vertex investigations keep their readable local scrolling behavior.

No outstanding correctness or visual blocker remains from this review.

## Native examples and independent references — passed

Command: `node scripts/verify-graph-traversal-examples.mjs`.

The wrapper imports the exact displayed strings, runs all ten complete programs in isolated Python processes, compares actual stdout with the displayed outputs, and invokes [verify-graph-traversal-native.py](../../scripts/verify-graph-traversal-native.py). Runtime: repository `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14), overridable through `LESSON_PYTHON`. Exact fixture programs and their JSON manifest remain in `scratch/graph-traversal-native-verification/`. No installation or network is required.

| Verified behavior | Actual passing coverage | Independent reference |
| --- | ---: | --- |
| Displayed programs and output | 10 complete programs | Actual isolated Python execution |
| Representation construction | 1,630 graphs | Exhaustive directed graphs through three vertices and undirected graphs through four, including loops; deliberately repeated edge input compared with declared-vertex adjacency existence |
| BFS distances/routes and DFS order | 5,876 source cases | Floyd–Warshall all-pairs distances plus a separately recursive DFS reference; route endpoints, real directed edges and parent contracts |
| Undirected components and coloring | 1,099 graphs | Components iff finite all-pairs distance; exhaustive two-color assignments establish whether any valid coloring exists |
| Directed cycle detection | 531 graphs | Exhaustive candidate vertex orderings establish whether a valid topological order exists, including loops and empty graphs |
| Builder rejection and encounter order | 4 grouped contracts | Duplicate labels, unknown endpoints, reserved None label and a deliberately nonalphabetical encounter-order fixture |
| Four-neighbor grid source sets | 4,225 | Every wall mask for 1×1, 2×3 and 3×3 grids; independently generated cell-edge graph and all-pairs distances, with empty/single/pair/repeated source collections |
| Eight-neighbor cell-count grids | 578 | Independent diagonal adjacency and all-pairs distance plus one cell; blocked endpoints and corner crossing included |
| Four/eight-neighbor contract distinctions | 4 explicit fixtures | Empty and blocked boards, diagonal-only route and cell-versus-move counting |
| Invalid grid/source inputs | 12 | Empty/ragged grids, invalid cell values, blocked/out-of-bounds sources and invalid diagonal input |
| Flood-fill regions and copies | 1,152 | All 2×3 two-color patterns, all starts and three replacement colors; independently computed same-color connectivity, unchanged original and distinct copied rows |
| Invalid flood-fill inputs | 4 | Empty/ragged images and invalid start |
| Clone identity and topology | 300 object graphs | Deterministically seeded graphs whose nodes all have the same payload value; original-to-copy bijection, preserved sharing/self-links/parallel entries, distinct neighbor lists and mutation independence |
| Deep iterative execution | 2,000 vertices/cells | Directed chain BFS/DFS/cycle, cycle after closing the chain and long-row four/eight-neighbor grids |

At each applicable source, parent routes use actual graph arcs and minimal edge counts for BFS; DFS is compared for exact entry/finish order and parent assignments without claiming shortest routes. Grid parents move one orthogonal cell and decrease distance by one. Clone verification proves that equal values do not collapse distinct objects in these fixtures; every copied reachable node maps to exactly one original, and no original graph node leaks into the copied topology.

The references use all-pairs closure, exhaustive color/order assignments, a recursive DFS and direct identity constraints. They do not merely duplicate the displayed queue/frame implementations. This is strong finite computational evidence, not a proof for all inputs or empirical evidence of asymptotic runtime.

## Integrated browser and visual review — passed

Command: `node scripts/review-graph-traversal-lesson.cjs` against `http://127.0.0.1:5173/learn/path/full-curriculum/graphs-representations-bfs-dfs`.

Fresh headless Microsoft Edge pages through Playwright used 1440×1000 and 390×1000 viewports with reduced-motion preference. Results are retained at `scratch/graph-traversal-lesson-review/results.json` (10 September 2026, 05:35 UTC). At each width the passing suite covered:

- Six representation cases: the default relation in both directions; reciprocal directed arrows plus a self-loop and duplicate; its undirected collapse; no edges with all eight isolates retained; and a second loop/reciprocal fixture. Every selectable adjacency row and every 8×8 matrix entry agrees with independent edge membership. Directed arrows exist only when appropriate, opposite directions draw separately, malformed endpoints/tokens and capacity overflow preserve the active graph, and Reset restores the sample.
- Eight BFS/DFS configurations totaling **133 trace states**: both methods from A under both directions, BFS from F, DFS from isolated H, and BFS from E with/without direction. Every reached vertex has a valid parent arc; every BFS distance agrees with Floyd–Warshall. Drawn distances and the table agree; queue entries remain unique and in nondecreasing distance order. DFS discovery prefixes and final return order agree with a recursive reference. All pending work empties. A→C gives one edge by BFS versus three through the undirected DFS parent tree; source-equals-target, H unreachable, Back and Reset work.
- Seven grid configurations totaling **100 trace states**: default single-source eight moves; a target that is itself one of two sources; nearest-source four moves to (0,4); the same target eight moves from one source; isolated-source unreachability; wall-free Manhattan distance; and keyboard-selected source-equals-target. Every intermediate discovered number and every final cell matches an independently constructed grid's all-pairs distances. Blocked targets and attempts to wall a source/target reject. New-source opening, immediate trace reset, Clear all walls, Enter/Space cell editing and Reset work.
- Ten complete rendered programs and nonempty code/output blocks, ten official practice links, nine unique working lesson anchors, the actual next Union-Find topic, safe new-tab attributes, initially closed hints/optional material, keyboard opening/closing and guided-practice navigation.
- Keyboard focus/Tab behavior in all investigations, interactive native controls at least 43px tall, no page or lab overflow, and no page errors. At 390px an actual ArrowRight keypress changes scrollLeft in both eight-vertex diagrams; exact tables/route readouts remain available as text alternatives.

Representative screenshots were opened and inspected: reciprocal arrows/self-loop, desktop BFS queue and DFS active frames, mobile DFS, multi-source grid, clone identity and foundation practice. Directional arrows, discovery/active/finished labels, source/target markers, distance units and route readouts are coherent and readable. The grid fits without horizontal panning. The clone figure preserves its shared C/C′ destination and its cycle while separating original and copied objects.

After the compact static-cycle fix, `node scratch/graph-traversal-lesson-review/recheck-static.cjs` independently checked 1440px, 390px and 320px. Both diagrams contain all A/B/C/D nodes within the region, without local or whole-page overflow. Name/state text renders at 23px/15px at 390px, and approximately 19.5px/12.7px at 320px. The corrected 390px and 320px screenshots were opened and inspected, including the previously clipped D and all active/finished labels. Evidence: `static-cycle-recheck.json` (05:41 UTC) and `cycle-contrast-corrected-*.png`. The integrated review script now contains the four-node-fit regression checks for future runs.

The first tall static screenshot included the fixed Learn navigation bar across part of the drawing because element capture scrolled the page. The capture helper was changed to hide that fixed bar only while taking the screenshot and restore it afterward, matching the existing integrated screenshot helper. Corrected images show the complete figure. This was a capture artifact, not a graph state change. The full interaction run preceded the static-geometry edit; the targeted static checks ran afterward. No unnecessary full behavior rerun was claimed for a bounded layout change. The model owner's later source formatting preserved normalized AST values and reran its model tests separately.

The separately owned [visual/model record](GRAPHS-TRAVERSAL-VISUAL-MODEL-REVIEW.md) reports **3,648 BFS/DFS runs and 262 grid configurations**, with copied-frame isolation, exact parent nesting/resume indices, representation/geometry agreement and invalid-input contracts. That evidence is separately attributed; it does not replace the independent native corpus or this integrated browser and screenshot review.

## Status

- Lesson implementation: complete and published by the coordinating author.
- Computational verification: ten displayed programs and the independent native corpus passed after the grid validation correction.
- Browser/visual verification: 1440px/390px integrated behavior and screenshots passed; the final compact cycle figure also passed targeted 320px review. No outstanding blocker from this review.
- User review / real beginner walkthrough: not performed.
- Next action: ready for user review. The coordinating author owns final application build, sequence and curriculum-conservation checks. No additional topic implementation is implied by this verification record.
