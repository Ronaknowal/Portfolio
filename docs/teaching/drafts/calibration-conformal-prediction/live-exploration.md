# Current live exploration contract — 21 September 2026

This topic uses playable exploration with **no learner prediction fields, optional guesses, answer commitments or grading gates**. Changing a valid control updates the connected calculation, topic-specific visual and explanation immediately. This contract supersedes the prediction/commit/reveal interaction ordering in the earlier authoring packet; those earlier clauses are historical and must not be reimplemented.

Edit probability cards, isotonic blocks, calibration residuals, alpha and interval scales. Keep threshold ranks discontinuous where the mathematics is discontinuous; display exact nulls and infinity/empty-set boundaries rather than inventing smooth motion.

- Keep the existing equations, datasets, assumptions, controls, numerical limits, figures, worked examples and independent practice. Do not replace the topic's representation with a generic output box.
- Provide reset, meaningful contrasting/null presets and labelled comparisons. Saving a comparison baseline captures actual inputs; it is never a guess. Describe what changed and why, including exact ties and undefined outcomes.
- A Run/Step/Acquire/Open action may represent an actual algorithm operation, acquisition, expensive batch or frozen assessment boundary. It must never require an expected answer. Model predictions, pseudo-labels and held-out observations keep their scientific meanings.
- Validate incomplete/out-of-range drafts. Either suppress their result or clearly retain the last valid calculation; never label old output as the edited input's result. No silent substitution or normalization, except an explicitly labelled control that redistributes a probability vector.
- Recompute from actual input dependencies, with bounded data/iterations and memoization where useful. Replays should not rerun expensive fits on focus, scroll or narration. Preserve the on-demand topic import boundary.
- Verify real control-to-mechanism-to-output changes, a meaningful null, reset and invalid-input recovery; review phone widths and keyboard operation. Do not rerun retired prediction-gate assertions as current acceptance criteria.

Runtime ownership: `src/learn/components/lesson-labs/CalibrationLabs.jsx` and its topic-specific helpers. Current migration evidence: [Classical ML live exploration review](../../LIVE-EXPLORATION-CLASSICAL-LATE.md).

## Inline conditioning explorer — 21 September 2026

`ConditioningForkFigure` owns a separate live two-group experiment. Four sliders edit each group's class-1 forecast and exact constructed class-1 rate, in .01 steps from 0 to 1; shares remain .5 each. Reset and contrasting presets are starting points, not the only interactions. Matching forecasts to current group rates supplies a meaningful calibrated null. Both plots, the table, accessible descriptions and interpretation follow the current state without any learner-answer feature.

Pool observations by the variable each diagram conditions on. Equal class-1 forecasts share a rate-weighted point; equal top confidences share a correctness-weighted point. Match complementary decimal confidences despite binary floating-point roundoff. At .5 the selected class is 1. Check calibration separately at every distinct conditioning value: equality of overall mean confidence and accuracy is insufficient. Keep exact constructed rates distinct from empirical counts and coverage guarantees. Verify the original .2/.8 example, arbitrary edits, equal forecasts, complementary forecasts, .5, both endpoints and reset. Theme all inline preset buttons and focus states, including figures outside the investigation wrapper.
