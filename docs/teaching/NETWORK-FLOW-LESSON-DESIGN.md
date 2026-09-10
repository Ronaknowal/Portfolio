# Network Flow, Minimum Cuts & Bipartite Matching — lesson design

Stable ID `network-flow-minimum-cuts-bipartite-matching`; DSA position19. Complete author implementation and review, 10 September 2026; see [actual verification](NETWORK-FLOW-VERIFICATION.md). Publication, integrated acceptance and user approval remain separate. The current CLI plan and both destination notes were read; no previous lesson body existed. Preserve title: its three promises form one mechanism-to-certificate progression. Depth is `core`, with optional implementation/modeling branches.

## Learning contract and scope decisions

Readiness: adjacency lists and BFS from Graphs, invariant/certificate reasoning from Algorithm Correctness, reductions from Reductions/P/NP. Refresh directed edges, vertices, paths, sums and queue parents locally. The learner should be able to check feasibility, reroute a decision using reverse residual arcs, derive and check a cut certificate, implement exact integer max flow, recover a matching/minimum cover/shortage witness, and translate additional modeling constraints without changing their meaning.

| Idea / evidence | Owner and decision | Boundary / durable location |
| --- | --- | --- |
| Planned augmenting flow, cuts and matching | Teach completely here, including proofs and original-edge identities | Core lesson and executable solver |
| Incoming exact bipartite cover note from Reductions | Include alternating reachability, exact cover proof, Hall deficiency and triangle limit | Resolve destination note after actual verification |
| Numeric capacities versus input bits | Include integer Ford–Fulkerson bound, BFS policy and exact arithmetic contracts | Complexity and Reductions already teach numeric versus encoded size |
| New incoming Karger note from Randomized Algorithms, found during final CLI review | Include optional global-cut branch with complete trial/repetition program, conditional survival argument and65 exact contraction distributions | Explicitly distinguish undirected nonterminal global cuts, preserve parallel multiplicity, precheck disconnected zero cuts and avoid a success-certification claim |
| Dinic blocking flow and current-edge cursor | Optional complete implementation and progress argument; no universal speed claim | Core residual/cut reasoning remains visible |
| Multiple supplies, vertex capacities, mandatory lower bounds | Explain transformations with complete worked programs and recovered original witnesses | Circulation feasibility is not lower-bound maximum-flow optimization |
| Binary labeling with neighbor disagreement penalty | Include original six-cell example, derived cut mapping and independent all-label oracle | Exact for the stated nonnegative binary energy; no claim about real-image accuracy |
| Weighted matching, min-cost flow, general-graph matching, multicommodity flow | Explain changed objective/domain and route useful deeper ownership | Do not pass a cardinality solver off as weighted optimization |
| Unresolved bit-manipulation inbox | Unrelated; retain unresolved | No forced aside |

## Core route and consistent fixture

Six vertices S,A,B,C,D,T; original edge IDs in order: S→A, S→B, A→C, A→D, B→C, C→T, D→T. Default capacity1 each. BFS first chooses S,A,C,T, then S,B,C,A,D,T; the latter cancels A→C. Final original flows `[1,1,0,1,1,1,1]`, value2. This fixture exposes a necessary change of mind under deterministic BFS rather than merely asserting reversals might matter.

1. Allocation, capacity and conservation before terminology. Check edge bounds and each internal vertex; flow value is net source outflow.
2. Residual opportunities, bottleneck and signed changes. Preserve paired residual identity even for parallel/antiparallel original edges.
3. Follow BFS augmentations, then prove a maximum using residual reachability and an original-edge cut. Sum conservation to derive the upper bound; equality certifies both sides.
4. Complete integer solver with independent feasibility/cut checks. Explain integer existence, real-capacity limits and capacity-independent Edmonds–Karp operation bounds. Optional Dinic uses levels and current-edge cursor.
5. Unit-capacity bipartite matching; alternating paths, minimum cover and explicit Hall shortage. Perfect versus left-saturating and maximal versus maximum remain distinct.
6. Transform additional constraints: vertex splitting, bounded supplies/demands, lower-bound circulation. State what each recovered witness guarantees.
7. Minimum cut as a binary labeling optimizer, not a physical-pipe metaphor. Derive every unary/pair edge capacity and avoid double charging.
8. Independent changed-case practice and verified external problems; natural next topic remains the module's actual successor.

## Representation contracts

| Representation and placement | State / question / encoding | Controls, feedback and limits | Independent checks |
| --- | --- | --- | --- |
| Conservation investigation beside definition | Six-node original directed graph; flow/capacity labels; node incoming/outgoing table | Edit proposed flows, including overcapacity; valid/imbalanced presets and reset. Never label an infeasible proposal a valid flow value. | Direct incidence-sum oracle |
| Residual pair figure before augmentation | A→C cap5 flow3, separate original C→A cap4 flow1; four explicitly owned residual opportunities | Static local ledger/lanes, not a globally feasible flow claim. Arrow direction and original-edge identity both explicit. | Forward cap−flow, reverse flow |
| Augmentation investigation | Same six-node graph, original-edge flow, selected residual route and cancellation marks | Apply bounded integer capacities0…9 explicitly; preview shortest path/bottleneck then apply; back/reset. Final residual source side and independently selectable cuts compare lower/upper certificates. | Exhaustive tiny cut minima and feasible edge assignments; exact signed replay |
| Bipartite matching/cover investigation | Compatibility matrix, distinct left/right entities, matching edges, alternating reachability, cover and shortage | Edit a3×3 matrix, all/empty/deficient presets; step rerouting; reveal proof certificate. Geometry is adjacency, not cost. | All512 matrices against subset matching/cover oracles; Hall neighbor sets |
| Vertex split figure | Incoming lanes→v-in→v-out→outgoing lanes, middle edge capacity2 | Static explanation of where the throughput constraint acts | Transformed solver + recovered totals |
| Six-cell binary cut investigation | Two rows of three labels, unary costs, disagreement edges, exact objective decomposition | Toggle labels and penalty; compare current energy with computed global minimum; apply optimum/reset. Labels and numbers supplement color. | All64 labelings for each bounded penalty |

All diagrams keep readable labels at320px, keyboard controls and explanatory tables/captions. No timers, force layout, animation-dependent teaching, mathematical engine or invented timing plots. Browser integer and graph size limits are explicit; native code handles general finite integer networks within Python memory/time and recursion limits. Traces retain snapshots and original-edge occurrence identities.

## Native examples, practice and research plan

Programs are standalone Python3.12+ standard library with visible run instructions and exact stdout. A shared string helper may generate complete code at authoring time; the reader receives complete programs. The meaningful targets are feasibility auditing, Edmonds–Karp/cut certificates, Dinic, matching/cover/Hall, vertex splitting, multiple supplies, lower-bound circulation and binary cuts. Do not add programs just to reach a count. Local tasks include invalid conservation, cancellation, cut direction, nonunique optima, failed greedy matching, cover/Hall, lower-bound signs and incorrect binary-energy mappings, with optional hints and fully explained solutions.

Primary sources inspected: Stanford CS2612016 lecture2 augmenting paths/cut proof/BFS and blocking flows; lecture4 applications/matching/Hall; Princeton Algorithms4 max-flow and matching implementation contracts; Carl Kingsford's circulation/lower-bound slides hosted by CMU. Historical performance/frontier claims are not reused as current facts. MIT6.0462015 lecture13 resource and accompanying notes will be linked as an alternate video route; record actual page/notes review, not unobserved playback.

Official LeetCode1349 statement is public, Hard, matrix at most8×8; conflicts change column parity, so ordinary checkerboard parity would be wrong. LC1820 official endpoint is Premium-only in this session: do not pretend its statement was verified or silently curate it as public. Prefer the self-contained allocation exercise plus directly verifiable problems whose actual statement is available. Final claim/resource ledger and complete native/model/browser evidence belong in NETWORK-FLOW-VERIFICATION.md.
