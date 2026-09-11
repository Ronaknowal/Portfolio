# K-Nearest Neighbors — independent review

Closed 11 September 2026 at 14:32:00 UTC by `/root/scientific_visual_improvements`. No unresolved material finding remains within this bounded review. This does not claim user acceptance, a learner study or a new full author-suite run.

The [independent packet](evidence/knn-independent-review.json) binds all seven final source hashes, the preserved [13:59:23 author packet](evidence/knn-author-review.json), and the [14:30:38 input/numerical amendment](evidence/knn-input-amendment.json). The [archive manifest](archive/tree-knn-input-amendment/manifest.json) preserves the initial sources. Only the pure model and two actual programs in the example bundle changed; body, validation observations, labs, CSS and blueprint are unchanged.

## Source and teaching assessment

Read the complete thirteen-section body and twelve changed practice tasks, full pure model, six labs/CSS, recorded validation fixture, individual brief/design/verification, and the exact displayed `LocalNeighbors` and KD-tree definitions. The author's eleven complete program executions and unchanged full browser records were inspected and reused, rather than repeated. Pre-rewrite conservation is attributed to its preserved baseline and author review.

Retrieval precedes aggregation, and the original eight-point example now has consistent distances, tie rules and weighting. The unit and cosine explanations, exact-match-only weights, local-mean bounds, train-owned preprocessing, candidate recall versus task quality, and branch-plane lower bound are coherent. The KD discussion does not turn a balanced index into a worst-case logarithmic query guarantee. The asymptotic branch distinguishes local disagreement from the Jensen risk bound, fixed 1NN from growing-k consistency, and smooth-regression rate assumptions from universal rules. Geography, multioutput and learned-representation applications state useful limits.

Read the current [sklearn neighbors guide](https://scikit-learn.org/stable/modules/neighbors.html), including supported metrics and the triangle-inequality justification for ball-tree bounds. Read the relevant [Cornell lecture companion](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html) volume discussion; the lesson's declared fixed-cube identity avoids mistaking it for a measured random-neighborhood curve. No video viewing is claimed by this reviewer.

## Findings and focused closure

Three input/arithmetic findings were reported before the author amended production:

1. Hole-skipping array checks accepted sparse queries/candidates, leading to NaN distances or a later incidental property error. Explicit own-index and declared-identity validation now rejects them before computation.
2. The actual Python estimator and KD helper squared very small differences inside `np.linalg.norm`. With stored `0` and `1e-200`, a query exactly at the second value could incorrectly choose the first. Both displayed helpers now use stable hypot reduction, retaining their explicit overflow rejection.
3. The JS cosine helper multiplied two tiny nonzero norms before division and consequently rejected nonzero vectors. Separately normalizing each vector before the dot product fixes the same scale boundary without changing its geometry.

The [complementary record](evidence/classical-tree-knn-independent-numerical.json) passed at 14:31:21 UTC. It includes four rotation/Euclidean/cosine identities, six tiny-scale cosine cases, sparse rejections, and twelve changed exact-match queries using both actual Python helpers at scales `1`, `1e-200`, and `1e150`. An independent oracle rescales coordinates before calculating distances. Contradictory exact duplicates and a changed numerical-target weighted mean are also checked. Shared Trees cases are identified separately.

The author re-executed only the two affected complete programs, retaining identical stdout and the nine unchanged execution records. Its focused six-scale/overflow checks and actual 390px amended code/output check are separately attributed in the amendment. `node scripts/verify-classical-tree-knn-independent.mjs` reproduces this review's complementary checks without rerunning all demonstrations.

## Actual visual review and limits

Actually opened six author captures: desktop neighborhood and recorded validation curve, 390px unit contributions and KD split-plane state, and 320px candidate restriction and volume curve. Exact paths and hashes are in the packet. Equal axis scales, meaningful row IDs, class shapes, contribution proportions, explicit candidate differences and the stated volume model make different mechanisms visible. The validation figure uses actual tested losses; it does not invent a U-shaped result.

Some element screenshots contain the fixed header across part of the captured element. This is retained as an evidence limitation, not described as unobscured ordinary reading. The numerical/source checks and visible surrounding labels support the bounded assessment; no new full route operation is claimed. Narrow spatial figures use visible local-scroll instructions and equivalent values. The author's full three-width keyboard, reset, disclosures, programs, anchors and overflow evidence remains the basis for unchanged behavior. These finite checks are not a proof over arbitrary floating-point arrays or all data distributions.
