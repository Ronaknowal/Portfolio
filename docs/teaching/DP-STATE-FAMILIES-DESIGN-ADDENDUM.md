# Dynamic programming: interval, tree and digit state extension

Design prepared and assessed by the parent on 11 September 2026. This bounded extension to **Dynamic Programming: States, Transitions & Optimization**, stable ID `dynamic-programming-states-transitions-optimization`, DSA module position 11, is now implemented. The sections below retain the approved design rationale; current author checks and their exact scope are recorded in [the extension verification](DP-STATE-FAMILIES-VERIFICATION.md). Independent review and production integration remain separate parent-owned steps.

## Scope, existing teaching and preservation

The [bounded practice review](DSA-PRACTICE-OWNERSHIP-REVIEW.md) and [DP destination note](topic-notes/dynamic-programming-states-transitions-optimization.md) identify the same actual gap: section 8 names interval, tree and digit states without developing these families. The existing lesson already has substantial suffix, weighted-scheduling, grid, sequence, capacity, counting, mask/endpoint and LIS teaching. This extension adds the three missing mechanisms in that same owner. It is not an optimization-theorem survey or a claim that a finite lesson/practice list covers every interview.

The title remains accurate. Keep the stable ID, module position, next-topic Range Queries bridge, all sixteen original complete programs and outputs, five working investigations, four inline figures, existing proofs and local practice, and all twelve existing official problem entries. Keep the old model/example/lab/CSS files unchanged unless an independently established defect requires a separately recorded correction. New support lives in purpose-named `dp-state-families-models.js`, `dp-state-families-examples.js`, `DpStateFamiliesLabs.jsx` and `dp-state-families-labs.css`, with CSS scoped to these additions. The current body, individual blueprint and practice dataset receive the new teaching; no shared metadata is edited by this author.

The baseline has been archived **before production mutations**: [exact source/program/practice and historical evidence manifest](evidence/dp-state-families-original.json), with 94 byte-checked archived files under `archive/dynamic-programming-before-state-families/`. It includes all seven relevant production files, original design/verification/note, focused scripts, historical native/browser records, programs and images. The preserved topic-plan output contains the exact current brief and incoming notes. The old evidence remains evidence of its original increment, not evidence that the new branches work.

Placement as implemented: retain section 8's short family-name orientation as a preview, then add three substantive, separately anchored branches after the existing optimization discussion and before the existing final independent task. This preserves all original explanations while the previously named methods now receive their full derivations. Descriptive unnumbered section headings retain every old numbered heading/anchor identity; the local contents list distinguishes the deeper branches and recommends pausing between them. Each branch keeps its essential recurrence/proof/example visible. Additional diagnosis and solutions use disclosures. Reading/practice time now includes the extension.

The tree investigation deliberately inspects an already completed exact table in postorder, rather than pretending its cursor recomputes a state. Its caption and controls say so. Changing a parent condition changes the conditional witness and requested value, while the two complete boundary tables stay fixed. Interval candidate inspection likewise changes the displayed candidate expression tree without pretending to change the separately computed whole-chain optimum. Digit prefix inspection reuses a cache tied to the active bound. These distinctions prevent a display cursor from being confused with an algorithmic mutation.

The final phone pass found that five balloon neighbors could wrap onto two lines and a scrollbar gutter unnecessarily clipped small trees. The row now uses compact, readable left/right sentinels; all neighbors stay in one row, matrix shape pipelines retain their arrow order, and default four-leaf diagrams fit 320px without scaling their 12px SVG text. Larger user-entered trees retain local horizontal scrolling. Exact arithmetic, provenance and state behavior are unchanged by these layout repairs.

### Coverage and prerequisite decisions

| Need | Existing evidence | Decision and prerequisite effect |
| --- | --- | --- |
| Interval boundaries and last split | Only the name occurs in section 8; weighted scheduling is a different recurrence | Fully derive matrix-chain parenthesization, then a last-removal balloon transfer with reconstruction. Introduce matrix dimensions and scalar multiplication counts locally; no linear-algebra course prerequisite. |
| Subtree interacting with its parent | Trees already teaches parent/child structure, postorder and subtree summaries; DP only names a boundary | Fully derive weighted independent set with a parent-selected bit. Propose **Trees & Binary Search Trees** as an additional explicit prerequisite for this branch, with a local shape/postorder refresher. Parent manages any catalogue prerequisite change. |
| Digit prefix, bound and property | Existing mask primer supplies membership/addition; no bounded-digit counting implementation | Fully derive positive integers with distinct digits up to a bound. Reintroduce decimal positions, padding and tightness; use the existing ten-bit set explanation, not unexplained integer tricks. |
| Weighted LIS/range aggregates | Actual next-topic Range Queries now implements the owner | Keep the existing link and scope. Do not duplicate its algorithm or describe it as still absent. |
| General bitwise interview preparation | Separate owner work is now assigned by parent | Preserve the complete local mask primer. No unrelated XOR/word-arithmetic expansion. |
| Divide-and-conquer, convex-hull and monotone-split optimizations | Current lesson explicitly states missing hypotheses | Retain the honest limit. These are not necessary to derive the three state families. |

The revised individual brief should add observable outcomes for interval-boundary sufficiency and witness order; tree parent-boundary independence; and digit-prefix constraint/count semantics. Append these three stages before the final design task, retain every old stage/source/outcome, and extend visual/practice/review contracts. Its depth remains `core`; these are deeper branches within a core method lesson, not a new catalogue difficulty/order decision.

## Interval branch: choose the last operation so the boundary stays fixed

### First derive an operation budget

Anchor: the same fixed sequence of matrix transformations can have very different arithmetic cost depending on its grouping. Locally explain that an `r × s` matrix multiplied by an `s × t` matrix gives an `r × t` matrix. The ordinary row-by-column algorithm performs `r·s·t` scalar multiplications. This count is the optimization objective; it is not measured wall time, GPU behavior, addition count, memory peak or an error bound. Exact mathematical matrix multiplication is associative, while factors cannot generally be reordered. Floating-point evaluation can round differently under different groupings, so the cost-optimal grouping is not asserted to be the most accurate numerical evaluation.

Use dimensions `[8,2,12,3]`: `(A0 A1) A2` costs `192+288=480`; `A0 (A1 A2)` costs `72+48=120`. Show the two intermediate shapes beside their operation counts. The effect is explained before introducing a table.

Define a fixed dimension array `d[0..n]`. State `C(i,j)` is the least scalar-multiplication cost to evaluate matrices `Ai ... A(j−1)` in their original order. The interval is half-open **in matrix indices**. Its result shape is always `d[i] × d[j]`, independent of its internal grouping. This invariant is why earlier internal choices need not be in the state.

For one matrix, `C(i,i+1)=0`: the matrix already exists. For a longer interval, let its last multiplication join `[i,k)` and `[k,j)`:

```text
C(i,j) = min over i<k<j:
         C(i,k) + C(k,j) + d[i]·d[k]·d[j].
```

Prove both directions. Any binary parenthesization has one last split, and its two child costs are at least their respective optima. Conversely any split joins two optimal legal child plans into a legal parent plan, with the stated fixed final multiplication cost. Replacement of a child cannot change the parent shape or legality. Shorter intervals are dependencies; evaluate increasing `j−i`, with any order within one length. Store the first minimum split for deterministic ties. Recover the two children before their parent multiplication, using a postorder operation list and a token list joined once; do not repeatedly concatenate growing strings while claiming linear reconstruction.

Four-matrix default `[8,2,12,3,6]` has five complete parenthesizations, with costs `204,264,456,624,984`. The optimum is `A0 ((A1 A2) A3)` at 204. The design fixture enumerates every full ordered tree, rather than trusting an optimized table. A one-matrix input returns cost 0 and that matrix's witness. Empty dimensions and dimensions of zero are invalid for this displayed problem; do not invent an empty matrix product with a fabricated shape. Positive integer dimensions are required.

Cost: `O(n²)` states, `O(n)` candidate splits per state, `O(n³)` arithmetic operations and `O(n²)` cost/split entries. The witness has `O(n)` nodes/operations. Python integers avoid fixed-width overflow but arithmetic cost depends on bit length; the displayed scalar-operation model is separate from bit complexity. Browser integer limits guarantee all arithmetic is exactly representable. A naive full search may enumerate Catalan-many plans; that is an oracle for tiny cases, not the scalable algorithm.

### Then transfer the boundary idea to removal

Use three numbered balloons `[2,4,3]`. Removing one earns its value times its **current** left/right neighbor values, using 1 outside the row. Every balloon must be removed; equal-valued balloons retain distinct original identities. An immediate decision changes neighbors, so a state that remembers only how many were removed cannot identify future rewards.

Pad with permanent value-1 sentinels. State `B(l,r)` is the maximum reward for removing all positions strictly between surviving boundaries `l` and `r`. Empty interior gives 0. Guess the **last** interior balloon `k`:

```text
B(l,r) = max over l<k<r:
         B(l,k) + B(k,r) + a[l]·a[k]·a[r].
```

Until `k` is removed, it separates the two subproblems; the fixed boundaries shield their internal operations. Every complete removal order has a last interior balloon, and choosing both child plans then removing `k` realizes each candidate. The optimal split tree is not itself chronological removal order: output the left child's removals, the right child's removals, **then** `k`. Explain that alternate interleavings of independent child removals can also attain the value. With `[2,4,3]`, original indices `[1,0,2]` earn `24+6+3=33`; removing the chosen last node first is not reconstruction.

Provide a second full native program returning optimum and original-index order. It replays the recovered order against a changing live row as a visible sanity check. Its table has the same cubic arithmetic/quadratic storage bounds; reconstruction is linear, whereas a simple list-based replay is quadratic and is explicitly only a witness checker. Browser static values come from this independently checked semantic model. Keep nonnegative integer values to match the stated task and explain zero ties/all-removed semantics; a changed signed-value exercise must use an impossible/uninitialized candidate marker rather than retaining 0 as an unattainable optimum.

### Representations and investigation

- **Matrix shape comparison figure**, immediately after the first concrete calculation. Two original row/column shape pipelines expose where the 12-wide intermediate is formed. Rectangles are schematic labeled shapes, not an exact area/benchmark chart. Each multiplication displays its three dimensions and cost. Stack the two plans at narrow widths without shrinking text.
- **`IntervalSplitLab`**, default four matrices. Editable dimensions (2–7 values, each integer 1–20) apply atomically and reset the selected interval/split. A labeled upper-triangular table shows interval costs; selecting `[i,j)` displays the actual consecutive matrix tokens and split candidates with left/right/final costs. Choosing a candidate shows its ordered binary expression tree and dependency arrows. A separate recover/step control executes the **chosen optimal** tree postorder and highlights completed intermediate shapes, explaining why a candidate inspection is not a new globally chosen solution. Default visible state must already show the interval/boundary meaning; it must not start as an empty generic table. Base cells have no split. Keep all numbers derived from the active model, accessible tabular candidates, full keyboard controls and local overflow only where necessary. A width-320 layout uses readable native text and stacked decomposition, not a scaled-down desktop tree.
- **Balloon last-removal figure**, beside the transfer derivation. Labeled permanent sentinels, original object IDs and three visible snapshots show `[2,4,3] → [2,3] → [3] → []`, exact earned products, and a small last-choice tree whose root is removed last. This static sequence has a different learning job from the editable matrix planner; no second large interval lab is needed unless actual reading review finds an unresolved mechanism.

## Tree branch: two answers cross one parent edge

Anchor: choose valuable nonadjacent vertices in a tree. “Independent” means no selected edge endpoints, not statistical independence. Nodes are original identities with weights; their spatial position is layout, not weight or depth cost. Empty selection is legal, so signed weights are supported without forcing a negative leaf.

Root a connected, acyclic, undirected graph. Define `F(v,p)` as the best total in v's subtree when its external parent is selected (`p=1`) or not selected (`p=0`). The parent lies outside the subtree and its weight is never included in `F`. At the global root use `p=0` because no parent constrains it.

```text
skip(v) = sum F(child,0)
take(v) = weight[v] + sum F(child,1)
F(v,1) = skip(v)
F(v,0) = max(skip(v), take(v))
```

Base leaves have `skip=0`, `take=weight[v]`; free value is `max(0,weight[v])`. Childless sums are zero. Explain the equivalent include/exclude pair but use one convention consistently in code/UI. The proof uses the unique parent connection and absence of cross-child edges: once v's choice is fixed, all child subproblems are independent in their **feasibility and additive objective**, and every legal set is either a skip or take case. Induct on subtree size. An extra edge between child subtrees destroys this decomposition; do not run the same recurrence over an arbitrary graph and call it correct.

Evaluate child-before-parent using iterative traversal plus reversed order. The native program validates dense integer weights and a simple tree (node bounds, no self-loops/duplicates, connectivity/acyclicity), does not mutate caller data, and supports an explicit root. Iterative reconstruction propagates the **actual chosen** parent bit, with skip on ties. Returned IDs are a legal witness whose summed weights equal the optimum. Any final sorting of witness IDs is presentation overhead and must not be hidden in the linear algorithm bound; preserve a deterministic traversal-order witness instead. Each edge contributes to a constant number of sums, giving `O(n)` arithmetic work/storage and linear witness recovery. A long chain must not depend on Python's recursion limit.

Default seven-node binary shape has weights `[5,9,2,4,1,6,3]`, edges `0–1,0–2,1–3,1–4,2–5,2–6`. Root free value 19 selects `{0,3,4,5,6}`; an externally selected parent forces root off and yields 18. For subtree 1, free value 9 differs from forced-off value 5: one unconstrained best number cannot serve both parent requests.

**`TreeBoundaryLab`** shows the actual input tree with two clearly labeled conditional answers per active node, not a rectangular list of unrelated text. Hover is optional; node selection has native buttons or a labeled selector. Change weights and choose a binary/chain/star topology preset, bounded to at most nine nodes; explicit Apply preserves old state on invalid input. A parent-selected toggle asks a conditional subtree question and changes the child request arrows, selected witness and sums together. It is labeled as an external condition, not falsely described as a second ordinary whole-tree optimum. Step postorder evaluation and inspect reconstruction; defaults show the meaningful two-state difference at node 1. A linked readable table gives the exact two answers and actual postorder. Narrow views stack node cards by depth, retain drawn parent edges and use a local labeled scroll area only for a genuinely wide tree. Empty input is an explicit no-node state. No animation timers or force layout.

## Digit branch: count completions without enumerating every integer

Anchor: count positive integers from 1 through N whose ordinary decimal spelling has no repeated digit. This is a precise property for generating/checking identifier ranges; do not imply it is a universal security/password rule. We count values, not permutations of padded spellings. N itself is included when valid.

Write the bound as L decimal positions. Pad a shorter candidate on the left with zeros for comparison only. State `(position,tight,started,used)` records the next position, whether the prefix still equals the bound prefix, whether a non-padding digit has appeared, and the set of actual digits already used. The bound/digit length remains fixed within one cache. “Tight” means equality so far, not merely that the prefix is legal.

At a position, allowable upper digit is the bound digit if tight, otherwise 9. For each candidate digit d, `nextTight = tight && d == boundDigit`. Once a smaller digit makes the prefix loose it can never become tight again. A zero before started remains padding and consumes no digit bit; after started, zero is a real digit and is checked/added like every other digit. Reject a digit already present. Every legal next digit identifies a disjoint set of completions, so add their counts. At position L return 1 exactly when started; the all-padding string is 0 and is excluded. The recursion advances one position, so it terminates.

State sufficiency proof: two histories with equal position, bound status, started status and used set have exactly the same allowed continuation strings and endpoint rule. Original prefix order does not otherwise affect this property. The `started` flag is pedagogically explicit; for this particular distinct-digit encoding it can be inferred from an empty used set, but that compression needs the invariant that every actual digit consumes a bit. It is not a general permission to discard leading-zero state for other properties.

For N=213, prefixes 12 and 21 have the same used set `{1,2}` at position 2, but 12 is loose and has eight valid final digits while tight 21 permits only 0 and 3. Prefix 00 has used no actual zero; ordinary 102 is valid, while 100 repeats a real zero. Exact counts are `count(0)=0`, `count(99)=90`, `count(100)=90`, `count(102)=91`, `count(213)=172`, `count(999)=738`. For inclusive positive range `[a,b]`, use `count(b)−count(a−1)` with an explicit below-1 empty count. Reject reversed/invalid ranges. Changing the task to include the number zero changes the endpoint contract, not its padded multiplicity.

Full native program uses Python integers, memoization scoped to one bound and at most 19 decimal positions for stated inputs through `10**18`; this covers the official exercise's bound. Its transition upper bound is `O(L·2·2·2^10·10)`, with `O(L·2·2·2^10)` cached entries and `O(L)` call depth. This is an upper bound, not a claim that every state is reachable; no loop through all N integers is in the solution. State-count arithmetic and Python bit costs remain distinct; the small browser uses exact finite integers. The optional endpoint/range helper returns counts, not every number. Do not append all witnesses and retain the count-only complexity claim.

**`DigitPrefixLab`** uses an editable integer bound 0–999999, with explicit Apply/reset and a current prefix strip aligned with the bound. Ten labeled digit controls show permitted branches, repeated-digit rejections, above-bound rejections, and completion counts from each allowed branch. Padding zero and actual digit zero have distinct visible labels. A ten-bit/ten-digit tray represents the used set. Selecting a digit moves along one prefix spine and updates tight/started/used and every branch count; Previous/Reset restore the exact earlier state. A nonzero branch count can produce one explicit completion for inspection, without enumerating all values. The root count and chosen prefix's remaining count are clearly different quantities. A small inline zero comparison (007,102,100) appears before the lab so leading-zero semantics do not require discovering a hidden state. No exponentially large full decision tree; the bounded visible spine and ten branches expose the mechanism directly.

## Programs, independent practice and external transfer

Four new complete runnable programs are planned: matrix-chain optimum plus expression/operation witness, last-removal balloons plus original-ID replay, general-tree weighted independent set plus reconstructed IDs, and distinct-digit count/range. Each is self-contained standard-library Python, with visible task question, full inputs, save/run instruction, exact executed stdout and an interpretation. Preserve the old sixteen program values/outputs byte-for-byte. Inputs distinguish empty, base, tie and invalid cases; do not silently rely on the UI validator to justify a native helper's broader contract.

New local changed tasks have hints followed by explained acceptance results:

1. Change dimensions to `[3,7,2,5]`: left grouping costs 72 versus right 175. Equal dimensions `[2,2,2,2]` tie at 16; explain why a deterministic witness does not imply uniqueness. Ask why minimizing only the immediate multiplication does not prove the global optimum.
2. Recover and replay balloons `[2,0,3]`, then diagnose root-first traversal on `[2,4,3]`. Force removal of all objects even when their value is zero. A signed singleton change demonstrates why zero initialization would be unsound if the contract broadened.
3. Change tree topology/weights and predict both boundary answers; include all-negative weights and an external-parent toggle. Add one cross-child conflict and construct a concrete falsely selected pair before deciding that one parent bit no longer suffices. Independent exhaustive subsets establish the small changed expected answer.
4. Explain the two/eight completion difference for bound 213 without running code; count a changed inclusive range and diagnose a tightness flag that incorrectly becomes true again. The range `[100,130]` has 17 all-distinct values.
5. Change the property to **adjacent** digits unequal. Now 101 is valid, while 100 is not; a previous-actual-digit state replaces the full used set, and padding must not create artificial adjacency. `[100,130]` has 19 valid values. Derive the new recurrence/base and decide how including 0 changes the answer. The solution includes explicit accepted numbers/branch reasoning and small brute-force checks; it is a changed-state task, not a renamed distinct-digit exercise.

Append one meaningful deeper-practice group; retain every old problem object. Official public statements were inspected 11 September 2026:

| Official task | Actual fit and platform contract | Hint and changed transfer |
| --- | --- | --- |
| [312 Burst Balloons](https://leetcode.com/problems/burst-balloons/), Hard | Last surviving split; 1–300 balloons, values 0–100, all removed; returns value | Ask what is still present when the last interior object is removed. Adapt local witness to judge value-only return; test zeros/ties and explain signed-value initialization separately. |
| [337 House Robber III](https://leetcode.com/problems/house-robber-iii/), Medium | Tree boundary; binary-tree node objects, nonnegative values, up to 10,000 nodes | Translate node identities and missing children to the local include/exclude contract. Do not sum the two unconstrained child optima when taking their parent. Explain iterative postorder for a deep chain and signed/empty-choice variation. |
| [2376 Count Special Integers](https://leetcode.com/problems/count-special-integers/), Hard | Positive distinct-digit integers up to n, with `1≤n≤2·10^9` | Keep bound equality and padding separate. Change to a range or adjacent-only property and justify the changed state/base. |

The optional [CSES Counting Numbers](https://cses.fi/problemset/task/2220/) statement is a direct adjacent-property/range transfer, including 0 and bounds through `10^18`. Its public statement was read; no judge/editorial/submission access or wall-clock success is claimed. It can be linked in the local changed exercise rather than forced into the LeetCode schema. Retain the old practice verification date for old entries; state the new three-entry inspection date explicitly without pretending all old links were freshly rechecked.

## Sources and what was actually checked

Research informs original explanations and independent fixtures. No third-party wording or screenshot is copied. Inspection date: 11 September 2026.

- [MIT 6.006 Fall 2011, Lecture 21 video page](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-21-dp-iii-parenthesization-edit-distance-knapsack/) and [official transcript](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/76f04a14b5c609e5157e7423ab083821_ocZMDMZwhCY.pdf), matrix/last-operation discussion around transcript pages 3–6: inspected the official page and relevant transcript. Annotated alternate video route for interval subproblems; full audiovisual playback is not claimed. Older course conventions differ in indexing and cost notation.
- [MIT 6.006 Spring 2020 Lecture 17 notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/665523227a175e9e9ce26ea8d3e5b51c_MIT6_006S20_lec17.pdf), arithmetic-parenthesization section on PDF page 4: inspected the min/max boundary warning, last operation, increasing length and two-child witness. The arithmetic-expression objective is **different** from minimizing matrix multiplication cost; retain this as a caveat against mechanically replacing any merge with max or min. The lesson derives its own fixed-shape recurrence.
- [Jason Eisner, Johns Hopkins Declarative Methods: Dynamic Programming](https://www.cs.jhu.edu/~jason/425/PDFSlides/16dyna.pdf), slides 49–64: inspected tree boundary, included/excluded-root formulations and weighted objective. The source uses Dyna/Prolog and an old illustrative application; the new lesson uses original numerical fixtures and Python, with local signed/empty semantics proved explicitly. Link as a deeper written comparison with an honest language/prerequisite annotation.
- The four official practice statements above were directly inspected for exact objective, difficulty where available, ranges and zero conventions. They support practice selection; they do not substitute for a proof of the authored digit recurrence. The digit proof is a self-contained partition of continuation strings, checked against independent enumeration.
- The Competitive Programmer's Handbook was examined as a resource candidate, but the tool did not establish a relevant digit section. The Caltech tree-PDF opening failed. Neither is cited as inspected support for this increment. Secondary search summaries were not used as proof sources.

No fixed video/article quota is imposed. The existing useful MIT video/written routes remain, and the added MIT interval lecture plus tree written comparison each have a distinct purpose. Do not attach an unreviewed digit tutorial merely to fill a media-format slot.

## Evidence contracts and completion gate

The [design fixture record](evidence/dp-state-families-design-fixtures.json), generated by `scripts/verify-dp-state-families-design.py`, currently records all five full ordered parenthesizations of the default chain, every live-neighbor removal permutation of selected balloon cases, independent tree vertex-subset optima, both changed digit-property lists and **2,501** digit-count recurrence versus direct-enumeration comparisons. These establish the proposed fixtures only; they are not the forthcoming production model/browser verification.

Implementation checks must additionally:

- Compare actual exported JS and actual displayed Python against independent full ordered expression trees/live-row removal permutations for many small signed/zero/tie-supported inputs, not against a retyped DP recurrence alone. Independently replay every returned operation/removal witness; confirm dimensions/IDs/terminal results and the first-split tie policy.
- Compare tree values and witnesses with all vertex subsets across enumerated small tree shapes, multiple roots, signed weights and both parent-boundary states. Check every selected edge, objective sum, empty/singleton/star/deep-chain behavior. Reject cycles, disconnected graphs, duplicate/self edges, bad IDs, sparse arrays, booleans where integers are promised, nonfinite inputs and out-of-range browser states. Keep snapshots immutable.
- Compare digit counts with direct decimal enumeration across every small bound and ranges, and with independent combinatorial length counts at larger all-9 bounds. Check exact branch-completion sets for tight/loose and leading-zero states, 0 endpoints, repeated real zeros, exact bound inclusion, negative/reversed/invalid inputs, cache reset across changed bounds and accepted native maximum. Verify count-only complexity instrumentation is not confused with UI trace-copying overhead.
- Execute all sixteen preserved programs and four additions, compare exact stdout and baseline hashes, and assert the twelve old problem objects and old model/lab/CSS bytes are conserved. Run the old focused models/native suite as regression; any unrelated discovered defect is coordinated and separately versioned.
- Review actual route at **1440, 390 and 320 px** with intended fonts. Operate every new branch/state/reset/apply/error, keyboard controls and disclosures; inspect old labs and all old/new program and practice placements. Assert no genuine console/page/request error or document overflow, readable SVG labels and all reachable data within plotted bounds. Test first-pass reading independently of controls, and actually open representative default/changed/error/reconstruction/narrow screenshots.
- Save a new semantic author verification and immutable exact-source packet, preserving old author records. Record what was actually tested/opened, final owned file hashes, source-review limits and any amendments. Parent then performs independent review and integration. Publication, author verification, independent review and user acceptance remain different stages.

The number of diagrams, programs and practice links above follows their learning jobs. Reading review may remove a redundant form or add a needed explanation while preserving complete coverage and a reasoned record. No universal layout, section count or lab quota is being introduced.
