# Lesson visual-layout follow-up

14 September 2026. The user reported overlapping labels in the Gaussian-mixture allocation figure and requested repairs to similar defects elsewhere and prevention in future implementation. This is a presentation repair of published lessons, not a new content rollout. Prepared manuscripts, topic identities, module order and deferred implementation requests remain unchanged.

## The reported figure

The figure put explanation, allocation columns and both components' before/after statistics into one fixed SVG coordinate space. The second component's heading collided with the preceding update. A diagram can stay entirely within its SVG bounds and still be unreadable; the earlier five-topic audit's selected screenshots did not establish that every inline figure was clear.

`AllocationFigure` in `GmmFigures.jsx` now separates the fixed measurements from the weighted update. Ordinary HTML carries stage headings and explanations. Each component has its own before/after strips, with a common numeric domain and equal physical widths; the components stack on narrow screens. Every observation still contributes total mass one. The effective counts, weights, means, variances, exact responsibility table and model calculations are preserved. Standard-deviation bars are explicitly distinguished from confidence intervals.

The same targeted review separated GMM mean guides from observation labels and moved the covariance mean label away from its eigenvector line. A final independent review also caught coincident Left/Right lane names in the EM identical-components preset. Those names now occupy a wrapping color/dash key outside the plot; all four presets are checked before and after a full EM cycle. PCA point labels, DBSCAN row labels/cutoff annotations and the anomaly lab's reach-neighbour labels received separate annotation space without moving the data.

## Broader repairs and shared behavior

- [Mathematical-axis review](MATHEMATICAL-AXIS-LAYOUT-REVIEW.md): annotation gutters, inward endpoint anchors, readable phone type and wrapping axis descriptions.
- [Diagram-label review](DIAGRAM-LABEL-LAYOUT-REVIEW.md): tree/graph and point labels, plotting captions, CSS font specificity and deliberate label backplates.
- [Shared-plot and foreground-label review](SHARED-PLOT-AND-FOREGROUND-LABEL-REVIEW.md): responsive shared charts, exact-data preservation and reference-line/label clearance in selected interactive states.
- Shared `viz/Plot`: captions, axes and legends reflow outside the SVG; container measurement keeps tick glyphs readable. Compact ticks retain distinct endpoints. An optional paginated exact-data disclosure preserves original numeric values and order and mounts rows only while open. The shared plot has 298 JSX uses; this does not mean 298 new lesson implementations.
- Foreground-line review: multivariate point labels move away from outgoing arrows; reference markers are confined to data rows in hypothesis and exponential-family plots; the dual-price annotation uses the heading margin; a radial sphere label and backtracking choice labels have local background halos.

These repairs adapt to the representation. They do not impose one diagram template or replace topic-specific investigations. No numerical model, example program, generated dataset or lesson body was changed by this follow-up.

## Verification method and limits

`scripts/audit-lesson-visual-layout.cjs` is an author-only triage tool. It measures transformed SVG glyph boxes, text intersections, foreground line crossings, SVG bounds and unequal axis scaling. It inspects currently visible/default states and does not test every possible interaction, path, canvas or HTML layout. Grid lines, opaque graph nodes and deliberate label halos need source and rendered interpretation. Its candidate count is not a defect count or an approval score.

The broad pass visits every published lesson at 1366, 390 and 320 pixels with actual fonts loaded. Focused interaction checks and actual screenshot inspection close the affected representations. Evidence from unchanged sources is reused; a later isolated repair receives a targeted rebuild/recheck rather than silently inheriting an earlier result.

## Final verification and source reconciliation

The [source-bound integration record](evidence/lesson-visual-layout-review.json) records **39 topic families with targeted repairs**, plus the shared plot, across 56 presentation source files. Three reviewers' topic-owned reports above retain the actual inspected images, source hashes, representative interactive states and limits. The root's [affected-figure captures](evidence/visual-layout-root-captures.json) cover PCA, DBSCAN, anomaly and backtracking; the [GMM browser record](evidence/gmm-browser.json) includes the reported allocation figure. Wide DBSCAN strips and the decision tree intentionally retain labelled horizontal scrolling; both ends were inspected where affected.

| Check | Actual result |
| --- | --- |
| All published routes, loaded fonts | 228 lessons × 3 widths = 684 route/width snapshots; no load errors. |
| Later bounded closure | Nine changed lesson routes rechecked at all three widths, then GMM alone after the final preset-label repair; unchanged routes retain their previous source-bound evidence. |
| Recent lesson interactions/regressions | PCA 14, DBSCAN 12, anomaly 13, GMM 14 browser cases passed: 53 total. GMM includes all four EM presets before/after a cycle at desktop/320px. |
| Shared plot | 17 cases passed at desktop, intermediate and narrow sizes, including exact-data values, 50-row pagination and unmounting. |
| Foreground-label interactions | 38 default/changed-state cases passed; selected screenshots actually inspected. |
| Mathematical and diagram review | Per-pass screenshot inventories and final closures are in the two linked reviews; captions, endpoint gutters and line/label intersections were inspected. |
| Integration | Final production build, curriculum verification, lazy-loading/navigation artifact verification and all eight delivery-ledger behavior groups passed. |

The initial scan produced 353 **candidate SVG occurrences**, counting repeated widths. The reconciled final passes retain 45 line/label candidates across 11 topic families, all explicitly source/render-dispositioned: opaque graph nodes/backplates, protected axis-origin labels, or a stem covered by its point marker. There are no remaining detected text/text intersections, out-of-SVG labels or unequal-axis-scaling flags in these recorded default states. This is not a claim that every possible state or every future illustration has been exhaustively tested. The detector was also refined to recognize grid/axis classes on parent groups, so the reduction in candidates must not be represented as a count of fixed bugs.

The original GMM bindings and broader baseline are retained separately: the broader snapshot was recorded after the first GMM repair. All lesson bodies in that baseline, and all unaffected model/example/data bindings in the repaired lessons, remain unchanged. Prepared manuscripts and specifications are retained. Only the four recently implemented topics' design records gain this follow-up; 39 central entries receive current presentation evidence, 35 legacy source maps are refreshed for reviewed changes with previous verification references retained, and the other **138 central entries are byte-identical**. Revision numbers, phase statuses and completion timestamps are preserved. The generated inventory remains **1,218 topics, 28 modules, 228 publications, seven paths, 176 current content-complete checkpoints and 111 current implementation-complete checkpoints**. Pending content-first revisions and the separate K-Means proposal remain pending.

The final build predates only documentation/ledger/evidence updates. No renewed native-program campaign was needed because the programs, models and datasets did not change. Existing numerical evidence remains applicable to those unchanged inputs; new layout evidence covers the altered presentation. Superseded captures created during this repair were removed after useful evidence and before/after records were retained.

## Future implementation contract

The [teaching standard](../../LESSON-TEACHING-STANDARD.md#visual-layout-is-a-separate-completion-check), [code standard](../engineering/LEARNING-CODE-STANDARD.md#diagram-layout-and-svg-legibility) and [topic-design brief](TOPIC-DESIGN-BRIEF.md) now explicitly require internal visual-layout review. Reserve room for labels and stages; put prose in normal document flow; inspect loaded-font desktop, narrow/intermediate widths and enlarged text; preserve meaningful geometry and common scales. A new or rewritten lesson needs every distinct inline figure and informative lab state inspected. A targeted repair needs its affected representations rechecked, with unchanged evidence retained.

The GMM browser verifier includes a regression for the allocation figure at 1366, 1024, 768, 390 and 320 pixels and enlarged text. It checks stage geometry, responsive stacking, equal-scale component widths, all four before/after strips and conserved allocation-column mass. Automated checks support actual visual inspection; they cannot guarantee that a future drawing will never have a layout defect.
