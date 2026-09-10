# Authoring notes: Calibration & Conformal Prediction

Canonical topic ID: calibration-conformal-prediction

## 2026-09-10 — Preserve conditioning populations and correct plot direction

- Status: open
- Origin: [Probability Distributions design](../PROBABILITY-DISTRIBUTIONS-BAYES-DESIGN.md), scoped review of its original calibration preview and the destination's introduction/section 2.1.
- Ownership: this existing published lesson owns calibration definitions, reliability diagrams and validation. The probability foundation will teach only the local population/conditional-rate bridge, with no rewrite of this destination.
- Learning benefit: distinguish useful ranking from calibrated numerical probability, class-probability calibration from confidence-of-correctness calibration, and a finite bin estimate from an exact conditional identity. Explain that Bayes' algebra does not guarantee the assumed population/rates still describe deployment.
- Existing coverage: the inspected destination section defines confidence on x and empirical accuracy on y, then says overconfidence bows above the diagonal. Under those stated axes, accuracy below confidence is below the diagonal. Its exact plot and broader claims still need scoped review; this note does not endorse the rest of the page.
- Proposed treatment: use small declared bin counts with uncertainty, a changed base-rate example under explicitly unchanged class-conditional behavior, and a comparison that preserves ranking while changing numerical scores. Resolve the axis-direction inconsistency against actual rendered data. Keep exchangeability/population/selection assumptions adjacent to any conformal coverage claim.
- Prerequisites: conditional probability, joint populations, repeated sampling and the difference between confidence and correctness. No empirical data or universal calibration behavior is inferred from the local toy.
- Evidence: current destination JSX section 2.1 and probability body's previous “only useful when calibrated” sentence were inspected on 10 September 2026. The elementary diagonal counterexample is x=.9, y=.6. Primary calibration/conformal sources and the destination's full plots require fresh research when that topic is authorized.
- Resolution: awaiting the destination author; no production code there changed.
- Implementation/verification links: origin design above; no completed destination verification.
