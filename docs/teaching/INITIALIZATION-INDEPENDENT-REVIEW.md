# Weight initialization independent implementation review

22 September 2026. Reviewer: parent agent, independently of author `initialization_implementation`. Current revision 3 prepared manuscript and visual specification were read completely alongside the generated lesson, models, labs, native producer and library bridge. Production interaction/visual closure is recorded separately by the integration owner.

The complete eleven-section teaching route is preserved: symmetry and signal scale before initialization recipes; mean square versus centered variance; Jacobian directions versus average gain; finite-width experimental evidence; explicit width conventions and restricted μP/Adam bridge. References, changed-input practice, solutions and prerequisite/next-topic relationships remain available. The source is explained around the complete downloadable programs; it does not substitute API imports for the mechanism. Native QR is an explicitly reused linear-algebra primitive. Optional LSUV/Fixup discussion does not imply an executed experiment.

Six live investigations have distinct instructional jobs. Saved propagation, digit fits and width sweeps are distinguished from recomputed toy mechanisms and hypothetical configuration controls. Fixed sampling, matched seeds, finite-run limits and development-set use are stated. Logarithmic charts carry exact zero on a separate rail. The spectrum and truncation figures retain their narrower scopes. Library equivalence claims are confined to the copied state, conventions and optimizer actually checked.

## Finding and correction

The default smaller gain was sqrt(0.1), while its HTML range stepped in 0.001 increments. The browser could therefore display 0.316 while the mechanism used a more precise state. The author changed the control to squared gain, with exactly reachable default 0.1 and preset 0.04; the resulting square-root gain is explicitly explained. This changes only the control parameterization, not native measurements. Author model evidence was refreshed. Browser gesture/reset validation remains in the integration record.

Production geometry review also found the long logarithmic label `10^-12.5` extending beyond the plot's left edge with the real site font. The author reserved 80 SVG units for ticks and selected an integer-decade middle tick at its actual logarithmic coordinate. The reviewer checked that both points and ticks retain the same domain/coordinate transform; no empirical values or mathematical meanings changed. Final production tests assert label containment at desktop, 390 and 320 pixels.

## Complementary evidence

`scripts/verify-initialization-independent.py` passes twelve rectangular/singleton/zero-gain matrices via singular values; six changed signed/zero-input autograd cases; and a width-48 multiplier-1.5 bridge with changed seed, batch and learning rate, comparing output, every derivative and three actual Adam updates with mup 1.0.0. This avoids rerunning the author's 72 fits and probes.

`scripts/verify-initialization-independent.mjs` compares the native changed derivatives to the JS mechanism, checks twelve gain/depth combinations using a 128-direction angular average and the Frobenius identity, and checks variance via pairwise squared differences. Final reviewed sources are SHA256-bound in `evidence/initialization-independent-review.json`; native evidence is `evidence/initialization-independent-native.json`. All these substantive checks passed. The author’s full native replay remains separate evidence.

No unresolved content or numerical finding. This review does not claim pretrained performance, arbitrary architecture/optimizer μP equivalence, browser completion or user acceptance. Final browser checks and visual inspection must pass before implementation completion.
