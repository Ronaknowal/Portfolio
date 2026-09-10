# Trees & Binary Search Trees: independent verification

10 September 2026. Scope: the first newly implemented topic in the authorized sequential DSA increment. This record covers author review, exact displayed Python programs and integrated browser behavior. It does not claim an observed beginner study or user acceptance.

Read with [lesson design](TREES-LESSON-DESIGN.md), [visual/model review](TREES-VISUAL-MODEL-REVIEW.md), the [teaching standard](../../LESSON-TEACHING-STANDARD.md) and [learning code standard](../engineering/LEARNING-CODE-STANDARD.md). The current exact-ID topic-plan command was run and its returned notes read; no destination note or relevant unresolved routing item existed.

## Teaching and correctness review

The lesson distinguishes branching shape from ordered keys before asking a learner to use comparisons. The continuing nine-key example connects diagrams, search, traversal, deletion and height. The recursion bridge makes suspended calls and emitted output explicit because the full Complexity Analysis & Recursion lesson occurs later in the module. The implemented ordered collection has integer keys and set semantics; duplicates are ignored. Global validation assumes an already well-formed finite tree, and does not promise to detect arbitrary aliased/cyclic object graphs.

Reviewed claims include inherited strict bounds, inclusive floor/ceiling/range contracts, successor deletion with a right child, key-copy versus object-transplant identity, height and diameter in edges, empty height −1, traversal/output/stack distinctions, width-dependent BFS queue space, O(h) versus a maintained logarithmic-height guarantee, and the separate responsibilities of rebuilding, rotating and balancing. The general-tree LCA algorithm explicitly requires both supplied identities to occur in the tree; it is not presented as a missing-target validator. Serialization reserves None as a missing-child marker and distinguishes shape from values. Arbitrary expression parsing, full balanced-container engineering, persistence and external-memory structures remain appropriately scoped extensions or future owners.

Useful transfer is concrete: numeric version floor lookup, pruned ranges, expression grouping, ceiling implementation, global-bound diagnosis and a diameter entirely within a child subtree. The next topic remains Heaps, Priority Queues & Tries. The ten linked platform problems supplement local complete exercises; optional reconstruction/combination tasks expose additional prerequisites and do not gate module progression.

One prose error was found in task C: a sentence first said five edges and then corrected itself to four. The author replaced it with a direct four-edge explanation (1 + 3); the six-edge internal diameter is independently verified.

## Native programs and independent oracles — passed

Command: `node scripts/verify-tree-examples.mjs`.

The wrapper runs all ten exact displayed programs with isolated Python processes, checks their displayed stdout and then runs [verify-tree-native.py](../../scripts/verify-tree-native.py) against the same source strings. Environment: repository `scratch/lesson-tools/Scripts/python.exe`, Python 3.12.14. No package installation or network execution is required. Reproducible source fixtures are retained under `scratch/tree-native-verification/`; `LESSON_PYTHON` can select another compatible Python interpreter.

| Coverage | Actual passing cases | Independent reference |
| --- | ---: | --- |
| Exact standalone programs and displayed stdout | 10 | Real Python execution |
| Insertion shapes and all four traversals | 878 | All insertion permutations through six distinct keys, plus empty, duplicate, negative/extreme and deeper-successor cases; finite path maps and lexicographic emission markers |
| Search, floor and ceiling query bundles | 9,670 | Set membership, valid reference path steps and Python bisect on separately sorted keys |
| Inclusive range queries | 13,170 | Filtering the independently sorted key set; reversed intervals separately reject |
| Deletions with object-identity checks | 6,818 | Set difference, all-ancestor ordering, reachability, exact surviving identity set and the successor object's removal |
| Arbitrary binary-tree validation, summaries and serialization | 205 | Independent ancestor comparisons, maximum path depth, all-pairs BFS graph distance, path-key reconstruction and missing-child token placement |
| General-tree LCA identity pairs | 39,295 | Shared path prefixes/ancestor identity, including same-node and ancestor/descendant pairs on duplicate-valued non-BST shapes |
| Balanced rebuild input sizes | 130 | Inorder set, minimal possible height and independent descendant-depth balance checks |
| Right rotations | 722 valid; 156 invalid | Exact retained object identities, old-root/pivot/middle-child references and independent ancestor order |
| Expression evaluations | 75 plus 5 malformed inputs | Direct arithmetic over signed/fractional/zero operands and required ValueError boundaries |
| Malformed serializations | 6 | Empty/truncated/trailing token rejection |
| Deep iterative chain | 1,500 nodes | Exact full inorder and endpoint search/floor/ceiling, beyond the usual recursive comfort range |

The test oracles intentionally differ from the lesson mechanisms: diameter uses graph distances rather than repeating the recursive height/diameter formula; LCA uses identity paths rather than repeating the recursive split algorithm; validation compares every ancestor rather than copying bound propagation. Tests remain finite evidence, not proofs for every possible input or empirical proofs of complexity.

The separate model owner reports 2,190 passing pure-model cases in [the visual/model record](TREES-VISUAL-MODEL-REVIEW.md). That is separately attributed evidence; the native program suite here does not pretend to replace it.

## Integrated browser behavior — passed

Command: `node scripts/review-tree-lesson.cjs` against the Vite server at `http://127.0.0.1:5173/learn/path/full-curriculum/trees-binary-search-trees`.

Playwright used headless Microsoft Edge, fresh pages at 1440×1000 and 390×1000, reduced-motion preference. The successful result is retained at `scratch/tree-lesson-review/results.json` (10 September 2026, 05:04 UTC). The script is configurable through `PLAYWRIGHT_PACKAGE` and `LESSON_URL`.

Both viewport runs passed:

- 53 search/insertion trace states: found/absent/equality, duplicate insertion, empty root, original versus sorted insertion order, the twelve-node browser limit, invalid key lists/targets, target-invalid preset atomicity, and deterministic reset.
- All 29 states of each preorder, inorder, postorder and level-order trace. Every intermediate output is the correct prefix, final outputs agree with the independently specified order, the frontier empties, and Back/Reset work.
- Nine deletion cases: sample root, leaf, one-child node, two-child internal node, missing key, deeper successor with right child, singleton root, one-child root and empty tree. Exact-link table contents preserve the remaining key set and identities; the temporary duplicate is explicitly marked and resolves; key 35 reconnects through 40.left.
- Keyboard opening/closing of the optional practice group and hint, focus and Tab movement in all three labs, keyboard enlargement/scroll-region access, exact-link disclosure, native control heights at least 43px, and no whole-page or lab overflow.
- Ten complete rendered program sections with nonempty code/output blocks, ten official practice links with safe new-tab attributes, nine unique route anchors, the guided-practice hash route, a link to the actual next module topic, and references.
- No page errors. Two initial harness issues (an exact implicit-select label selector and reading visible text from a closed disclosure) were corrected before the passing run; they were not lesson failures.

## Actual visual inspection

Screenshots under `scratch/tree-lesson-review/` cover search shape/enlargement, recursive stack and FIFO queue, deletion before/after reconnection, all four inline figures and the guided practice set at desktop and narrow widths. Representative screenshots were opened and inspected, independently of the DOM overflow checks.

The first narrow anatomy screenshot revealed tiny in-SVG root/leaf annotations intersecting a child edge. The visual owner removed these redundant annotations, kept depth/root/leaf information in the readable caption and enlarged static key/edge labels at narrow widths. Updated anatomy and rotation were inspected at both 390px and 320px: branch structure, left/right labels and the transferred middle subtree remain legible, without page overflow. The 320px captures are `inline-1-320.png` and `inline-3-320.png`.

The diagram, suspended stack, emitted sequence and causal feedback are spatially linked. At narrow widths, the stack and output become successive regions. The search and deletion diagrams also provide explicit enlargement with horizontal keyboard scrolling and a readable exact-links table; these are useful for small stable-identity labels in the default fit-to-width view.

The deeper-successor deletion screenshot also exposed long labels `n1 · target` and `n4 · successor` intersecting outgoing edges. The coordinating author replaced these with short IDs inside the drawing; the existing adjacent identity readout retains the target/successor roles. A targeted browser recheck at 1440px and 390px verified the short labels, unchanged target n1/successor n4 readouts and no overflow. Both corrected screenshots were opened and inspected: the labels no longer intersect the edges. Evidence: `delete-labels-corrected-1440.png`, `delete-labels-corrected-390.png` and `deletion-label-recheck.json` in the same scratch evidence directory. The final edit changes label text only; the complete behavior suite did not need repetition.

## Status and limits

- Implementation: complete in the coordinating author's lesson and the model/practice owners' components.
- Computational verification: exact Python examples and independent native cases passed; separate model evidence linked above.
- Browser behavior: passed at desktop/390px; additional 320px static figure inspection passed.
- Visual review: anatomy and deletion-label corrections verified by targeted desktop/narrow screenshot inspection; no outstanding review blocker.
- User review / real beginner walkthrough: not performed in this author review.
- Sources: the lesson design and practice standard record the author/resource owner's source inspection. This reviewer checked contracts and link placement, but did not watch entire videos, submit platform solutions or independently re-audit all external statements.
- Next action: this scoped lesson is ready for user review; continue the authorized module sequence. Parent retains responsibility for publication, application build and curriculum conservation checks.
