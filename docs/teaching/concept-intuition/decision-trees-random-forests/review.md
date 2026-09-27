# Decision Trees & Random Forests: complete conceptual review

26 September 2026. Author revised the production JSX, which has no topic-body generator. Existing canonical programs and their execution records are preserved.

| Location / transition | Assessed support and disposition |
| --- | --- |
| §1 task → rows → tree vocabulary → leaf distribution | Existing training diagram, 3/4 example, split ownership and baseline suffice. Retained. |
| §2 questions → paths → regions → constant outputs | Existing partition lab traces the same (4,3) query through named rows; sufficient. Retained. |
| §3 Gini → mixture versus error | Formula was right but two-draw interpretation remained abstract. Added a 4×4 equiprobable pair grid showing six disagreements and contrasting majority error. |
| §3 entropy → weighted gain | Added surprise-to-average-surprise explanation before entropy. Existing 6/2 split calculation and live split ledger clearly teach child weighting. |
| §3 threshold candidates → prefix implementation | Added three actual scan states for labels [0,0,1,1], with left/right counts and recalculated gains. Explains the saved work before the optimized code. |
| §4 greedy search versus capacity; transformations | XOR lab, depth-two argument and transformed midpoint counterexample already expose both failures. Retained. |
| §5 regression leaf mean → SSE split → extrapolation → alternative losses | Mean identity explicitly reuses regression, with six-row calculation and complete program. Retained. |
| §6 constraints → pruning objective → weakest link | Added buying-specificity explanation before formal complexity cost. Existing live crossover and exact α explain removal decisions; retained. |
| §7 scratch tree/forest → library boundaries | Complete prefix-scan code and prediction contracts retained; no fit/code change. |
| §8 bootstrap → omission → eligible OOB members → workflow leakage | Membership lab and actual omission calculation already trace every dependency; retained. |
| §9 vote versus probability average | Existing 0.49/0.49/0.99 figure is adequate. Retained. |
| §9 averaging → covariance → limit floor → margin bound | Added canceling versus shared-error intuition before covariance. Preserved assumptions and distinction between conditional seed variation and repeated-dataset risk. Original margin bound is explicitly deeper and has its own definitions. |
| §10 held-out experiment → schema/categories/missing/weights | Actual independent results and isolated mechanism examples retained. No novel API claim or unexecuted output added. |
| §11 MDI versus permutation → correlated copies → group perturbation | Added questions-first framing before definitions. Existing duplicate-feature lab shows why model reliance differs from available information. |
| §12 compute/storage → alternatives → forest proximity/kernel | Existing operation counts/conditions suffice. Added leaf-signature diagram and slot-by-slot dot-product interpretation before PSD statement. |
| §13–14 changed practice and connections | All thirteen tasks, hints, solutions, capstone and next-topic order retained. New representations support existing gain/pruning/variance exercises. |

## Representation specifications

Gini grid: four constructed row identities [A0,B0,C0,D1] with replacement give 16 ordered pairs; six disagreements are amber and explicitly marked ≠, agreements neutral with =. Grid is compact at phone widths. The shape encodes sampling outcomes, not class probability by rectangle area.

Prefix scan: three states of one four-row sorted fixture, candidate thresholds 1.5, 2.5, 3.5. Moving the marker transfers one row. Gains are computed from each state's actual class counts: 1/6, 1/2, 1/6. Count text and cut markers remain together on narrow screens. Static sequence is appropriate because the existing split lab already allows live alternatives.

Proximity: three observations retain separate tree-indexed leaf identities. A–B and B–C share one of two slots; A–C share none. Wrapping HTML signatures make this construction legible without shrinking a large tree diagram. The following mathematical derivation preserves the √B normalization needed for an exact inner product.

## Research and checks

Read [scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html), mathematical formulation, classification criteria and minimal cost-complexity pruning sections on 26 September 2026. Used to check criterion-to-leaf interpretation and pruning tradeoff. The local pair and prefix examples are original constructed calculations. Existing Breiman paper/companion and explanatory resources stay in the lesson; this pass does not claim a fresh full video viewing.

Author checks parse both JSX files and independently enumerate pair disagreements, split gains and shared-leaf fractions. Full lesson read included all thirteen practice solutions and later computation/proximity branches. New source hashes are in `author-checks.json`. Root browser and independent review remain separate; unchanged native fitting evidence was not rerun or relabeled.
