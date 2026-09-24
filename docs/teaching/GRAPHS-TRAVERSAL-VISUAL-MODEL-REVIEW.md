# Graph traversal: visual and model contract

10 September 2026. Topic `graphs-representations-bfs-dfs`; third of the user's three sequential DSA implementations. Read with the [lesson design](GRAPHS-TRAVERSAL-LESSON-DESIGN.md), current teaching standard and engineering policy. These representations address separate learning difficulties; their count and layout are not a template for later lessons.

## Representations and teaching decisions

| Difficulty | Representation | What the learner can inspect |
| --- | --- | --- |
| An edge list, adjacency list and matrix can encode the same graph. | `GraphRepresentationLab` connects an editable edge set with node/edge geometry, selectable adjacency rows and a matrix. | Selecting a vertex highlights its actual outgoing neighbors, its matrix row and corresponding edges. Direction changes both storage and permitted moves. Isolated vertices remain visible. |
| Discovery, pending work and completion mean different things. | `GraphSearchLab` couples the graph with a FIFO queue or actual recursive-style DFS frames, parent/state table, discovery and finish orders. | Mark-on-discovery avoids duplicate queue entries. Each DFS frame records the next neighbor to resume after its child returns. A selected parent route is highlighted and described separately from traversal order. |
| Some graphs are implicit in a rule rather than stored as edge objects. | `GridWavefrontLab` uses a clickable wall grid, exact distance cells, frontier borders and a recovered target route. | A whole distance layer expands through four-neighbor equal-cost moves. Multiple sources begin together; a wall differs from an open unreachable cell. |
| A revisit alone does not establish a directed cycle. | `DirectedCycleFigure` compares an actual DFS revisit to finished D in a DAG diamond with a back edge D → A to an active ancestor. | Both states are derived from the same frame model as the interactive lab. Arrows, active/finished labels and the explanatory route distinguish the two situations. |

The representation lab teaches storage and direction; the search lab independently uses the lesson's fixed example so path comparisons remain reproducible. The grid uses cells and layers rather than another generic text panel because adjacency and distance have a spatial mechanism here. Geometry in explicit graph drawings remains only layout, never a weight or geographic distance.

## Source ownership and model data

- [graph-traversal-models.js](../../src/learn/data/graph-traversal-models.js): pure representation, BFS/DFS, parent-route, grid-wavefront and layout models.
- [GraphTraversalLabs.jsx](../../src/learn/components/lesson-labs/GraphTraversalLabs.jsx) and [graph-traversal-labs.css](../../src/learn/components/lesson-labs/graph-traversal-labs.css): four self-contained named exports and scoped styling.
- [verify-graph-traversal-models.mjs](../../scripts/verify-graph-traversal-models.mjs): independent all-pairs, recursive DFS and grid checks.

The explicit graph is `{vertices, edges, directed, adjacency, matrix}`. Browser vertices are fixed A–H. Default undirected edges are A–B, A–C, B–D, C–D, D–E and F–G. H is explicitly present without incident edges. `buildGraph` collapses duplicate pairs; A–B/B–A collapse only in undirected mode. A directed self-loop is one loop arrow; an undirected self-loop is one loop line and one adjacency membership. These are existence lists, not a claim that list length gives the conventional undirected degree when loops occur. The Boolean matrix records existence rather than parallel multiplicity or weights.

`graphTraversalTrace(graph, source, method)` returns copied snapshots containing the graph, source, queue/frames, discovered/finished vertices, parent references, BFS distances, tree depths, entry/finish order and DFS times. A frame records `{vertex, nextNeighborIndex}`. `graphPath(state, target)` returns null until the target has been discovered; after completion null means unreachable. A source-to-itself route is the singleton source with zero edges.

`gridWavefrontTrace` accepts bounded dimensions, wall coordinates, source coordinates and a target. Each frame copies distance, parent, source-owner, frontier, expanded cells and the current target path. Coordinates are row,column. Sources are deduplicated and ordered by row/column. Neighbor order is up, left, right, down. Open cells are marked on first discovery; walls remain separate from undiscovered distance values.

Every saved state is isolated from caller state and other frames. There are no timers, layout forces, external lab dependencies, random choices or learner-code execution.

## Explicit-graph teaching contracts

**Storage comparison.** Start with the six default edges and select A. Predict two matrix memberships for an undirected A–B edge; direction allows only the row-A/column-B move. Edge input accepts uppercase A–H endpoints and at most 24 comma-separated entries. Blank input retains eight isolated vertices. Draft edge and direction changes apply through Apply graph; malformed input preserves the active graph. Reset restores the original relation. Adjacency-row buttons update the linked views immediately and support keyboard activation.

The node/edge diagram draws self-loops as loops. Opposite directed edges use separate curves and arrowheads, so one direction is not hidden behind the reverse. Crossings without a labelled circle are not vertices. Node positions are stable layout choices; matrix rows/columns are labelled from/to. Native SVG labels remain readable through local keyboard scrolling. Text adjacency lists and exact matrix cells provide alternatives to the picture.

**Search.** Default BFS starts at A with target E. Ascending neighbor labels give BFS discovery A, B, C, D, E. First-parent BFS routes assign C directly to A. Actual DFS frames enter A, B, D, C, E and assign C to D, yielding A → B → D → C. The two searches agree on reachability while DFS does not promise a shortest route. F reaches F/G only; H reaches itself only. Directed search from E reaches only E because its incoming D → E arrow does not permit travel backward.

Search, source and direction drafts apply through Start selected search; changing the inspected path target updates the current route without rerunning traversal. The lab states that it uses the fixed lesson graph independently of edits in the storage lab. Back replays copied frames; Reset restores BFS/A/undirected/target E. Before completion, an unobserved target is labelled not yet discovered; the UI does not prematurely call it unreachable.

BFS assigns a parent and distance while marking, before enqueueing. The queue contains each pending vertex once. Dequeue, adjacency inspection, discovery and finishing the adjacency list are separate states. Distances count edges under the equal-cost contract.

DFS increments the current frame's next-neighbor index, descends into one undiscovered neighbor and preserves the caller frame. When no neighbors remain, it records a finish time and returns. The display puts the top frame first and names its next action. Entry/finish times match a recursive reference. BFS distance and DFS tree depth have different table headings; a DFS parent route is never labelled shortest merely because it was found first.

Colors accompany shapes/text: green fill indicates discovery; dashed node outlines indicate active frames; exact labels distinguish active, finished and unreached. The current inspected edge is dashed amber and the inspected parent route is green. The parent/state table and route text remain the authoritative nonvisual readout.

## Grid-wavefront teaching contract

The 5×5 default fixture exactly matches the native examples:

```text
...#.
.#.#.
.#...
...#.
.....
```

Walls: `(0,3), (1,1), (1,3), (2,1), (3,3)`. Single source `(0,0)`, second source `(4,4)` in multiple-source mode, initial target `(4,4)`. All moves cost one and use up/left/right/down neighbors. The default single-source target distance is eight moves. With both sources, the initial target is a source and correctly has distance zero; choosing target `(0,4)` demonstrates the nearest-source distance of four instead of eight.

Predict the next distance layer, then step. Each expansion inspects a complete equal-distance frontier, assigns first parents to new open cells and advances the layer. Parent routes become visible as soon as the target is discovered. The route includes both endpoints, so its cell count is one more than its move count. A null target distance becomes unreachable only when the wavefront is complete.

Cell buttons either toggle a wall or select a target, according to the labelled edit control. Those explicit actions immediately apply and restart the trace; source changes do the same. A wall cannot cover an active source or target. Choosing a wall as target reports an error without mutation. Switching to multiple sources opens the new source cell if necessary and says so. Clear all walls and Reset are deterministic. Keyboard Enter/Space perform the same edit as a pointer click.

Cells show coordinates, numeric distance and S/T/front markers. `#` means wall; a dot means open but not reached; thick frontier borders and route outlines supplement color. Cells use native HTML buttons with a minimum height of 79 px; browser review checks narrow-screen widths. No Euclidean or diagonal distance is implied by their physical arrangement. The default's walls are chosen teaching inputs, not observed geographic data.

Transfer checks include isolating the source by blocking `(0,1)` and `(1,0)`, source-equals-target, comparing two sources, clearing walls to recover Manhattan distance and changing the contract to diagonal moves or cell-count output. The model does not silently mix these contracts.

## Directed-cycle contrast

The acyclic graph uses A → B, A → C, B → D, C → D. In ascending DFS, D finishes before C inspects C → D. The highlighted destination is discovered but no longer on the active stack. In the cyclic example A → B, B → D, D → A, the destination A is still an ancestor frame, and the edge closes a directed route back to it. The model captures these exact inspection states; neither picture is a hand-authored approximation of a trace. The prose explicitly distinguishes this directed rule from an undirected parent-edge rule.

Narrow-screen inspection initially found D outside the visible area of these small four-node static diagrams. They now use a compact 330px diamond with responsive fitting down to 280px and larger 23px vertex/15px state labels, showing both complete situations without panning at 390px or 320px viewport widths. The eight-node interactive graph retains its larger native-label scrolling representation. The [targeted browser recheck](../../scratch/graph-traversal-lesson-review/static-cycle-recheck.json) confirms all four nodes fit both diagrams at 1440px, 390px and 320px. Effective labels are 23/15px at 390px and approximately 19.5/12.7px at 320px. This is measured browser evidence in addition to geometry assertions.

Corrected [390px](../../scratch/graph-traversal-lesson-review/cycle-contrast-corrected-390.png) and [320px](../../scratch/graph-traversal-lesson-review/cycle-contrast-corrected-320.png) captures were taken with the fixed page header hidden only for isolated screenshots, then restored. Both were opened by the coordinated reviewer; the model author also opened the final 320px capture and confirmed all four vertex names, direction arrowheads and active/finished labels remain visible. No browser-review blocker remains.

## Accuracy, sources and limits

Primary sources reviewed 10 September 2026: [Princeton Undirected Graphs](https://algs4.cs.princeton.edu/41graph/) for representations, reachability, BFS equal-edge shortest paths and parent routes; [Princeton Directed Graphs](https://algs4.cs.princeton.edu/42digraph/) for direction and the active/finished distinction. The chapter's representation and terminology are not copied blindly: this browser uses explicit A–H labels, Boolean edge existence, collapsed duplicates, sorted neighbors and a declared self-loop membership policy.

The browser is bounded to eight labelled vertices and 24 edge entries. The visible grid is 5×5; the pure grid model rejects dimensions beyond 6×6. It does not implement weighted paths, arbitrary layouts, all-path enumeration, strong-component decomposition, geographic routing or general source editing. The native lesson supplies broader applications separately. Counts and traces describe exact local algorithm states, not empirical runtime measurements; object lookup, sorting, copying and drawing overhead do not prove asymptotic performance.

## Verification evidence

- `node scripts/verify-graph-traversal-models.mjs` passed **3,648 BFS/DFS runs**. It covers all 64 undirected simple edge subsets on four vertices, all 512 directed edge subsets including self-loops on three vertices, every source, both methods, the full A–H default and duplicate/reversed-edge fixtures. Independent Floyd–Warshall all-pairs distances establish reachability and BFS optimal edge counts. A separate recursive DFS built from edge membership establishes exact entry/finish orders, parents, tree depth and timestamps.
- Every parent route is checked against real adjacencies, with unique vertices, correct endpoints and the promised edge count. Every frame is checked for monotonic discovery, unique queue membership, nondecreasing BFS queue distances, actual DFS parent-frame nesting and bounded resume indices. Representation checks compare all list/matrix memberships with raw edge existence, and geometry checks compare drawn edges with real relationships. Static DAG/back-edge examples are verified against actual frame states and closure.
- **262 grid configurations** passed independent all-pairs grid-distance checks. These include every wall subset of a 3×3 grid excluding fixed source/target positions, single/multiple sources, the exact native 5×5 fixture, isolated source, duplicate source normalization, source-equals-target and no-wall cases. Every discovered distance agrees with the oracle, every parent decreases distance by one, source ownership propagates correctly, and every target route uses adjacent open cells. This independently verifies multi-source distances as the minimum over source distances.
- Input bounds, invalid endpoints, duplicate collapse, self-loops, unknown sources/methods, blocked/invalid grid cells, empty edges and isolated vertices are checked. Snapshots are mutation-isolated. The owned JSX parses with the repository's Babel parser.
- The new Heap/Graph models, labs and model verifiers were formatted using the installed Babel generator, conserving normalized ASTs, comments, literal values and JSX text before/after. Both complete model suites passed again. Baselines are retained under `scratch/dsa-format-baseline/`. The subsequent compact-static geometry change preserves every edge and has separate bounds/conservation checks; the complete Graph model suite passed after that change as well.
- [Integrated browser results](../../scratch/graph-traversal-lesson-review/results.json) cover six representation cases, eight search scenarios (133 states) and seven grid scenarios (100 states) at each of 1440px and 390px. They include direction, reciprocal edges and loops, unreachable/isolated sources, BFS/DFS routes, default and multiple-source grids, wall edits, keyboard target selection, source-equals-target and reset behavior. The compact-static recheck above followed the only geometry revision. Native displayed examples, publication and full build are separately coordinated by the parent author; these model and browser results do not substitute for each other or for learner testing.

No user acceptance or observed beginner learning study is implied by author/runtime verification.
