# Capsule Networks implementation review

22 September 2026. Reviewer: root, separately from topic author `initialization_implementation`. Scope: the complete prepared revision3 of Deep Learning position13, `capsule-networks`.

## Content and mechanism review

Read the complete manuscript and visual specification, the design's source/coverage decisions, generated content, model and interaction source, canonical training/check contracts and author's native evidence. The implementation retains all eleven major sections and17 prepared disclosures: grouping/vote semantics, parent-axis allocation, squash/routing/gradient distinctions, full classifier and loss units, six actual paired fits, fixed-weight interventions, decoder information boundaries, geometric counterexamples, diagonal EM, architecture scaling, eight practice tasks and annotated alternatives. Optional full-paper architectures are accurately distinguished from the bounded runnable local model.

The full NumPy mechanisms, independently reconstructed inference and explicit trainable Torch program are available beside their explanations. No generic capsule import conceals the owned routing algorithm. The retained six600-update fits were reproduced by the author; the reviewer did not duplicate that campaign. Class lengths remain scores, internal agreement does not certify recognition, and transformed-input failure is taught rather than hidden. Labels do not enter encoder inference; selected decoder masks are described as separate diagnostics.

## Complementary numerical review

`verify-capsule-independent.py` builds a separately written Torch contraction/softmax reference. Thirty-six new cases vary parent count1/2/4, coordinate width2/5, routing rounds1/3/8 and temperature0.31/1.7. The browser's entire coupling/sum/output trace agrees. Three new fixed-model inputs—constant intensity, stripes and a continuous ramp—also match native encoder/classifier/reconstruction outputs. These are constructed interventions, not new dataset measurements.

`verify-capsule-independent.mjs` reports maximum absolute difference `9.43689570931383e-16`. Native and browser source hashes are retained in `evidence/capsule-independent-native.json` and `evidence/capsule-independent.json`. These complement, rather than replace, the author's original native, derivative, objective, mask and changed-image checks.

## Finding and correction

**P2: negligible activation was displayed as identified EM evidence.** Valid input `[0,0,1e-20]` creates mass below the declared `1e-12` denominator guard. The model intentionally retains that guarded arithmetic, but a drawn mean/ellipse could misrepresent it as an identified estimate. The author now labels each affected parent as not reliably identified, suppresses its mean/ellipse, and distinguishes the guarded numeric calculation. Input is not silently clamped; actual votes remain visible. The exact tiny-mass and equality-at-guard fixtures are retained, with a browser regression covering the typed tiny value and reset.

No remaining source/content correctness or scope gap was found. Final current-source browser checks, responsive screenshot inspection and source/ledger reconciliation are recorded in [the integration record](DEEP-LEARNING-SEQUENCE-IMPLEMENTATION.md). That record closes the parent integration boundary; this review does not claim full MNIST/Matrix-Capsule/VB/STAR experiments, universal browser performance or user acceptance.
