# Naive Bayes: independent review

11 September 2026. Root reviewed the complete current lesson independently of its author. The [source-versioned review](evidence/naive-bayes-independent-review.json) closes the topic review with no open material finding. Final increment integration remains separate.

The review followed the learner's progression from the prediction unit and representation through Bayes scores, conditional-independence assumptions, word counts, absence and missingness, categorical support, Gaussian densities and geometry, repeated evidence, ComplementNB, sparse pipelines, streaming, calibration and integrated prediction. The core reasoning is visible before the deeper branches. Twelve changed exercises require applying the mechanism rather than copying the worked inputs; complete programs retain their data and runtime context.

Accuracy checks included the distinction between conditional token counts and independent column counts; smoothing as a posterior mean rather than an unconditional MAP statement; density units and interval probability; equal/unequal-variance boundaries; copied evidence's effect on decisions and confidence; ComplementNB's version-specific score contract; and calibration inputs with disjoint data ownership. No source or UI correction was required.

Seventeen complementary invariance cases were executed: common log-score offsets, bag-of-words permutations at three smoothing settings, density-unit Jacobians, equal-variance boundary symmetry, and the invariance of Brier/log loss to display binning. All six production hashes matched the author's reviewed version. The full unchanged native programs and browser suite were reused from the author's evidence, rather than rerun.

Four existing author captures were independently opened: the 320px introductory explanation and 390px unequal-variance geometry, density crossing and reliability plot. Labels, exact versus sampled boundaries, density/area distinctions, bin counts, values and transfer prompts were readable. These are independently assessed existing captures, not a new independent execution of every interaction. No learner study or user acceptance is claimed.
