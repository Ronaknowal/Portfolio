# Combinatorial Optimization — independent review

Reviewed 10 September 2026 UTC by the scientific-visual agent, independently of the root author. This review does not edit the lesson, examples, models or visual sources. It supplements [author verification](COMBINATORIAL-OPTIMIZATION-VERIFICATION.md); the author owns actual browser operation and root owns production integration.

## Scope actually read

The complete eleven-section body, every changed practice solution, individual design and the full core model source were read. All ten actual displayed Python programs were executed independently. Detailed code inspection focused on signed required-size assignment, exact search bounds, filtered value scaling, greedy forests and fractional-cover rounding. Author native/model/browser evidence was read as attributed supporting evidence, not represented as independently rerun.

The mathematical pass checked:

- Feasibility and objective units; correct minimization/maximization bound directions; zero and signed objectives; per-instance certificates versus universal polynomial approximation guarantees.
- Matroid heredity/augmentation, the forest component-count argument, high-weight prefix maximality and telescoping proof. Unrestricted signed weights differ from mandatory bases.
- Fractional upper bounds and retained-frontier stopping certificates; feasible-item filtering and the prefix-versus-best-singleton half guarantee.
- Refund signs for negative as well as positive assignment costs, fixed-flow residual-cycle optimality, global potential certificates, the reachable-region proof for successive shortest augmentation, and the initial negative-cycle limitation.
- Exact nonnegative weighted bipartite cover by a cut; the distinct odd-set issue for general matching and separation from general vertex cover.
- Weighted set-cover harmonic charges including simultaneous coverage, zero costs and the singleton endpoint.
- Weighted-cover threshold inclusivity, nonnegative costs, redundant upper bounds and the load dual; feasible dual versus optimal dual; the half-integral extreme-point perturbation.
- Normalized nonnegative monotone submodularity under a cardinality budget, complete marginal evaluation and the finite-k residual recurrence.
- Filtered Vmax, downward loss accounting, exact rounded-score DP recovery, polynomial bit costs and PTAS/FPTAS conventions. The native program explicitly calls its reported cell count value-state updates rather than allocated memory.
- Metric completeness/symmetry/triangle assumptions, MST and parity repair, two-odd-vertex case, perfect matching versus bipartite assignment, and the unrestricted-TSP gap reduction.
- Changed practice computations and the distinction between an optimal modeled assignment and its realized operational cost.

No unresolved substantive mathematical or algorithmic defect was found in this bounded review.

## Complementary executed checks

Run node scripts/verify-combinatorial-independent.mjs. It uses scripts/verify-combinatorial-independent.py and stores raw evidence in scratch/combinatorial-independent-review/. The durable [review evidence](evidence/combinatorial-optimization-independent-review.json) retains tested and final fingerprints separately.

Actual run at 18:56:01 UTC:

| Check | Independent construction and result |
| --- | --- |
| Ten learner programs | Execute actual current source strings and compare complete stdout |
| 228 assignments | Changed signed/missing 2×3, 3×2, 3×4, 4×3 and 4×4 matrices at every permitted requested size; native and browser models compared to subset/permutation optima |
| 554 stages | Each reported intermediate matching has the optimum cost for its attained cardinality and valid distinct endpoints |
| 228 residual certificates | Rebuild residual arcs from final pairs independently; verify exact arc correspondence, all-pairs Floyd negative-cycle absence, reduced costs and source-to-sink reachability |
| 54 value-scaling states | Changed item/capacity fixtures at inverse error 2, 7 and 100; 45 nonempty cases checked against exact Fraction floors and the maximum rounded objective over every feasible subset |
| 40 signed forests | Compare the actual greedy helper with every subset; independent forest test uses edge count versus connected components, rather than the helper's union-find |

All passed. These intentionally complement the author's larger randomized/exhaustive checks and do not imply exhaustive proof for arbitrary inputs.

## Minor teaching clarification and closure

Practice 2 originally described earliest-finish maximum interval count as solving a “different weighted objective.” The main matroid section already correctly distinguished the weighted counterexample from cardinality scheduling. The reviewer requested removal of “weighted” so the solution matches that distinction without asking a beginner to infer implicit unit weights. This is a wording-only correction; native functions and model contracts are unchanged. Final source closure and the author's targeted rendering evidence are recorded in the durable review JSON.

The original tested body fingerprint is 08208a16b3d083069099564e0db29845a8d386c8c75825e94881f27b97e38118; model fingerprint is 3f04f45e48774be13206264d3143e8607eddac126264472b8bb485b3ac42dcf1. The six tested fingerprints matched the author's 18:54:24 freeze. Any subsequent final wording fingerprint is preserved separately rather than silently replacing the tested record.

## Visual evidence and limits

The author's final browser report covers actual 1440/390/320 interactions, fonts, all anchors/programs/disclosures, keyboard controls, equations and responsive figures. Independently opened author screenshots: assignment-refund-1440.png, scaling-fine-320.png and vertex-fractional-390.png under scratch/combinatorial-optimization-review/browser/. These visually support the signed refund trace, computed DP frontier and distinction between a feasible dual and the LP optimum. This reviewer did not independently rerun the full browser suite.

This is a finite source/math/implementation review, not a learner study, formal proof assistant result, performance benchmark, screen-reader listening session or user approval. Root owns integration and full-goal status.

Final closure: the author corrected the phrase to "a different, unweighted objective" and refroze at 18:58:36.802 UTC. The reviewer independently inspected that exact wording and confirmed all six current hashes against the final author record. Final body SHA256: 6b1ffdcc3d5f6c350cd57aaf3f3fe8bace7c4a1417929dea6c51f299c2098eda. The other five hashes remain identical to the independently executed tests. The author final-reading evidence reports the amended answer visible at 1440/390/320; this attribution does not claim a second browser run by this reviewer.
