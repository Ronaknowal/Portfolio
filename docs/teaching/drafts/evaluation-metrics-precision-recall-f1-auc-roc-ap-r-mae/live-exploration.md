# Current live exploration contract — 21 September 2026

This topic uses playable exploration with **no learner prediction fields, optional guesses, answer commitments or grading gates**. Changing a valid control updates the connected calculation, topic-specific visual and explanation immediately. This contract supersedes the prediction/commit/reveal interaction ordering in the earlier authoring packet; those earlier clauses are historical and must not be reimplemented.

Move decision thresholds, scores, observed classes, error costs and residuals; reorder retrieval items and change relevance/gain/cutoff. Update score blocks, denominator bins, curves, residual geometry and retrieval shelves with matching tables immediately.

- Keep the existing equations, datasets, assumptions, controls, numerical limits, figures, worked examples and independent practice. Do not replace the topic's representation with a generic output box.
- Provide reset, meaningful contrasting/null presets and labelled comparisons. Saving a comparison baseline captures actual inputs; it is never a guess. Describe what changed and why, including exact ties and undefined outcomes.
- A Run/Step/Acquire/Open action may represent an actual algorithm operation, acquisition, expensive batch or frozen assessment boundary. It must never require an expected answer. Model predictions, pseudo-labels and held-out observations keep their scientific meanings.
- Validate incomplete/out-of-range drafts. Either suppress their result or clearly retain the last valid calculation; never label old output as the edited input's result. No silent substitution or normalization, except an explicitly labelled control that redistributes a probability vector.
- Recompute from actual input dependencies, with bounded data/iterations and memoization where useful. Replays should not rerun expensive fits on focus, scroll or narration. Preserve the on-demand topic import boundary.
- Verify real control-to-mechanism-to-output changes, a meaningful null, reset and invalid-input recovery; review phone widths and keyboard operation. Do not rerun retired prediction-gate assertions as current acceptance criteria.

Runtime ownership: `src/learn/components/lesson-labs/EvaluationMetricsLabs.jsx` and its topic-specific helpers. Current migration evidence: [Classical ML live exploration review](../../LIVE-EXPLORATION-CLASSICAL-LATE.md).

## Residual ruler affordance repair — 21 September 2026

The inline residual-square figure is now an editable experiment. Its gold forecast diamonds are genuine pointer/touch and keyboard sliders; the observed circles remain fixed. Start with A/T5 at 4, move it toward observed 10, and watch the signed error segment, squared area, MAE, RMSE, SSE and R² change together. A and B are independent. Reset restores every original forecast; Match all observations demonstrates zero error on nonconstant targets, including R²=1.

Use one stable ruler domain −2 to 12 minutes and one area scale throughout inline edits. A 64 px square side encodes error magnitude 14 minutes; zero error has exactly zero area. All displayed quantities derive from the existing `regressionMetrics` helper; preserve the original example and negative-duration caution. Pointer positions use the SVG's inverse screen transform, preserve the initial grab offset and capture the pointer until release/cancel. Arrow keys move ¼ minute, Page Up/Down move 1 minute, and Home/End select ruler limits. Keyboard focus stays on the diamond; do not make results depend on hover or introduce guesses.

The full residual investigation retains editable observations, baseline, row add/remove and minutes/seconds. Its forecast diamonds are also draggable. The shared domain includes all values, remains stable within a gesture, and expands for larger numerical edits within the existing −100 to 100 base-minute limits. The square scale is shared and frozen during each gesture. The numeric fields provide exact arbitrary-value entry beyond the ¼-minute drag increment. A seconds display changes every geometry/readout consistently; editing still uses base minutes.

Pair each short ruler with its area square so learners can see both without an oversized empty panel. Explain actual control capabilities next to the figure. Do not give read-only rulers drag cursors or slider roles. Button hover/focus/disabled states remain in the topic's dark/amber palette. Verify pointer movement at desktop and narrow widths, keyboard movement, boundaries, independent A/B state, unchanged observations, area ratios, metric agreement, reset, zero error, invalid numeric recovery, seconds conversion, and a stable ruler/area scale while dragging. The new browser receipt belongs to this repair; earlier numerical and migration reviews remain immutable.


The separate probability-penalty comparison keeps its original probability vectors fixed. Its gold positions are now vertical readout ticks, explicitly labelled as non-handles; the item selector chooses the displayed row. The genuine threshold slider still changes alert membership without changing probability losses. This distinction prevents a read-only encoding from advertising nonexistent dragging.
