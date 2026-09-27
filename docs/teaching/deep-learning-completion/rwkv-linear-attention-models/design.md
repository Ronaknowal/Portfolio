# RWKV implementation map

The complete prepared manuscript is generated into the stable existing production path. Its 11 main sections, equations, original code, all ten practice problems and annotated primary-source references remain. The older published article's unsupported family/quality claims are replaced by the prepared distinctions.

## Concept-to-runtime map

| New hurdle | Exact explanation location | Worked steps and representation |
| --- | --- | --- |
| Match versus value contribution | Lesson §1 | `RwkvFigure(transcript)` calculates each score/weight/product; `parameters` separates training weights from sequence state. |
| Fixed summary equivalence | §2 | `outer` writes 2×1 cells and normalizer; `ledger` exposes all four intermediate states; `RwkvSummaryLab` edits actual query/key/value records and compares direct/chunk calculations. |
| Chosen versus approximated kernel | §2 scaling and optional branch | `kernels` matches weight and output axes; `random` distinguishes expectation, finite draw and normalized ratio. |
| Chunk causality | §3 | `chunk` separates incoming and local triangular contributions; `timeline` distinguishes known training inputs from sequential generation. Summary lab includes uneven remainder and sizes exceeding the sequence. |
| Read-before-write and bonus | §4 | `circuit` has separate numeric current-read and future-write branches; `retention` plots three half-lives. `RwkvWeightedLab` exposes every saved a,b,p and independent enumerated error. |
| Signed stable arithmetic | §4 stable arithmetic | `scale` explains common rescaling; weighted lab handles ±1000 keys without overflow. No clipping or arbitrary epsilon inserted. |
| Complete trainable block | §5 | `block` names all five state slots; `objective` aligns input prefixes and next-token targets; deferred complete PyTorch program is linked at §8. |
| Replacing an association | §6 | `correction` shows before-read/residual/new state; `interference` uses [.6,.8] key; `RwkvDeltaLab` provides an editable angle compass, target, query and rate. |
| Matrix versions and rank-one correction | §6 | `versions`, `goose` and `reflection` distinguish state orientation, decay/erase/write tiles and theorem-only factor2. Goose direction toggle immediately changes all five matrices. |
| Different forgetting shapes | §7 | `forgetting` applies scalar, row and targeted changes to the same signed matrix. |
| Storage versus timing | §7 | `cache` contains exact formula lines, denominators and core-only label; adjacent table retains exact bytes. No timing measurements invented. |
| Real fitting and state continuation | §8 | `pipeline`, `outcomes`, `continuation`; `RwkvTrajectoryLab` loads only 50 validation rows + seed17 weights on opening, edits real coordinates, cut and chronology, then computes all three routes. |

23 distinct inline figures are implemented at their requested local positions. The numerical ledger, original manuscript tables and compact HTML stages replace some originally proposed animations: they preserve intermediate states without motion, scale down cleanly, and remain directly readable with a keyboard. No authoring-only figure instructions reach the reader. All four investigations have visible current results, real editable entities, reset and contrasting/null cases; independent practice stays separate.

## Verification and review boundary

Native Python checks reuse selected frozen weights and never retrain for preferred results. Both complete PyTorch models reproduce all 50 saved validation logits; 16 changed/full/carry/reset native cases supply an independent trust root for the JavaScript port. Maximum JavaScript/native logit difference is recorded in model-checks.json (about 2.64e-6). Kernel denominator rejection, chunk remainder, stable ±1000 keys, retention1, bonus-only-read, constant values, correlated-address interference and zero-rate state are checked independently from the UI.

Read on 26 September: project architecture history still labels RWKV8 branches experimental; official API_DEMO and RWKV7 paper record remained available. Existing annotated research is preserved. No large checkpoint was downloaded; the optional official `rwkv` package program is available but unexecuted. This is separate from the actual ordinary PyTorch model route.

Author learning-experience pass: problem precedes terminology in each branch; core route is preserved; concrete numerical states support the algebra; weak and misclassified real outcomes remain visible; no repeated caveats added; original code is complete and deferred on demand; practice changes constraints rather than replaying traces. Author checks are not independent review. Root owns browser geometry, keyboard checks, integration and the central ledger.

## Independent review correction

Root's first independent pass found that several original implementation substitutions supplied the numerical reasoning but weak visual paths. `RwkvMechanismFigures.jsx` now supplies original connected query-to-weight-to-value wires, an explicit local causal triangle plus incoming state, read/store circuits with separate bonus/decay wires, two residual bypasses with named state slots, a key compass next to signed correction states, and an actual source-row7 movement path beside dimensioned model flow. `RwkvSummaryLab` starts at read4, retains a pinned input baseline and shows signed contribution ribbons. `RwkvWeightedLab` supports1–12 records and displays linked current-read versus stored-history ribbons; current bonus cannot enter the stored ribbon. Old prediction-gate and awkward residual wording was corrected in the current manuscript and regenerated. These changes address a teaching finding, not a numerical-model defect. Browser geometry and independent confirmation remain root-owned.
