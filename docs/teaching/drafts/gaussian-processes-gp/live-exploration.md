# Current live exploration contract — 21 September 2026

This topic uses playable exploration with **no learner prediction fields, optional guesses, answer commitments or grading gates**. Changing a valid control updates the connected calculation, topic-specific visual and explanation immediately. This contract supersedes the prediction/commit/reveal interaction ordering in the earlier authoring packet; those earlier clauses are historical and must not be reimplemented.

Drag covariance, noise, position and horizon controls, retaining precise numerical entry. Update target mean/variance, posterior bands, forecast rows and probe variance gains directly. Frozen historical training/development/test roles remain explicit.

- Keep the existing equations, datasets, assumptions, controls, numerical limits, figures, worked examples and independent practice. Do not replace the topic's representation with a generic output box.
- Provide reset, meaningful contrasting/null presets and labelled comparisons. Saving a comparison baseline captures actual inputs; it is never a guess. Describe what changed and why, including exact ties and undefined outcomes.
- A Run/Step/Acquire/Open action may represent an actual algorithm operation, acquisition, expensive batch or frozen assessment boundary. It must never require an expected answer. Model predictions, pseudo-labels and held-out observations keep their scientific meanings.
- Validate incomplete/out-of-range drafts. Either suppress their result or clearly retain the last valid calculation; never label old output as the edited input's result. No silent substitution or normalization, except an explicitly labelled control that redistributes a probability vector.
- Recompute from actual input dependencies, with bounded data/iterations and memoization where useful. Replays should not rerun expensive fits on focus, scroll or narration. Preserve the on-demand topic import boundary.
- Verify real control-to-mechanism-to-output changes, a meaningful null, reset and invalid-input recovery; review phone widths and keyboard operation. Do not rerun retired prediction-gate assertions as current acceptance criteria.

Runtime ownership: `src/learn/components/lesson-labs/GaussianProcessLabs.jsx` and its topic-specific helpers. Current migration evidence: [Classical ML live exploration review](../../LIVE-EXPLORATION-CLASSICAL-LATE.md).
