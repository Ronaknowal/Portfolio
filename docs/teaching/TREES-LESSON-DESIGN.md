# Trees & Binary Search Trees: lesson design

10 September 2026. First of the user's next three DSA lessons, implemented before Heaps/Priority Queues/Tries and then Graphs. The existing entry was planned, with no lesson, individual blueprint or destination note. Topic-plan retrieval found an empty unresolved inbox. Preserve the title, stable ID, prerequisites and module position.

## Finish line and scope

A learner can read a rooted binary tree, distinguish it from an ordered BST, trace recursive and iterative traversals, search/insert/delete with an explicit duplicate policy, validate ancestor bounds, reason about height and auxiliary space, and adapt these mechanisms to range queries and subtree summaries. The core uses integer keys with set semantics, explicit None children, empty height -1 and leaf height 0 (edges).

The catalogue currently places formal Complexity Analysis & Recursion later, despite recording it as a prerequisite. Teach the necessary call/return and O(n)/O(h)/O(log n) bridge locally; keep the supporting prerequisite link and module order. This is not a claim that the full later lesson is already taught. Python functions/classes and the preceding stack/queue lesson remain necessary.

| Hurdle | Original example / support | Assessment |
| --- | --- | --- |
| Binary shape versus ordered keys; root/leaf/subtree/depth/height | Labelled nine-key tree, visible left/right structure and empty-child convention | Classify a non-BST binary tree; compute height without counting nodes |
| Discarding an entire subtree safely | Search 7 and missing 5 through [8,3,10,1,6,14,4,7,13]; live bounds and comparison path | Explain each discarded region; insert duplicate/empty/skew cases |
| Suspended recursion versus emitted output | Same tree, four traversals, frontier/call stack and output linked to structure | Predict next emission, recover inorder using an explicit stack |
| Preserving references while deleting | Leaf, one-child and successor replacement states; identity/key distinction | Root deletion and successor with a right child |
| Global validity versus parent-only checks | Counterexample 10 → left 5 → right 12; propagated open intervals | Implement validator; reject duplicate/global violation |
| Efficient shape and ordered queries | Same-key skew/branched contrast, balanced rebuild and rotation | State O(h) before promising O(log n); prove inorder preservation |
| Transfer beyond definitions | Version floor lookup, inclusive numeric range, expression tree evaluation, path/subtree summaries | Independent ceiling and diameter tasks with hints, full solutions and cases |

Three focused labs expose search/insertion, traversal and deletion. Static figures introduce structure and local/global contrast before requiring interaction. Optional advanced material explains balance maintenance/rotations and subtree metadata rather than suggesting that an ordinary BST self-balances. Complete red-black/persistent/B-tree implementations are distinct deeper engineering work; existing Persistent Data Structures and External-Memory Algorithms own their specific mechanisms. Save reasoned discoveries for those authors instead of silently expanding this page into those lessons.

## Sources and interpretation

- Princeton Algorithms, https://algs4.cs.princeton.edu/32bst/: inspected chapter for ordering, height-dependent operations and successor deletion; https://algs4.cs.princeton.edu/code/javadoc/edu/princeton/cs/algs4/BST.html for API/height conventions. Teaching scenarios, Python implementations and figures are original.
- MIT OCW 6.006 lectures 5 and 6: inspected course resource pages and companion notes for search/sort and AVL balance/rotation explanations. Video links are alternate learning routes; no claim of watching entire videos.
- VisuAlgo BST/AVL page: inspected tutorial text and available operations as an alternative interactive explanation, not as evidence of learner mastery or an executed browser oracle.

## Verification contract

Run every displayed Python program and compare stdout; exercise empty/singleton/duplicates, malformed ordering, skew, traversal consistency, missing deletion and successor-right-child deletion. Compare the JS model against independent Python/ordered-set answers, including intermediate topology validation. Inspect diagram identity, edge direction and stack/output agreement. Check desktop/mobile, keyboard, resets/invalid input, no overflow, direct anchors, next module-order entry, lazy body loading and publication conservation. Passing checks remains author review, not user acceptance or an observed beginner study.

The separate DSA practice standard governs verified LeetCode links, pattern coverage, delayed/retrieval practice and transfer. External submissions complement complete local exercises; they do not guarantee solving every possible interview question.
