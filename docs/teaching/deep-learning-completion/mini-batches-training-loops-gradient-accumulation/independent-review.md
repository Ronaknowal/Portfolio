# Independent mini-batch review — 27 September 2026

Reviewer: graph/attention author, distinct from the training-mechanics author. Read the complete ten-section manuscript, all five changed exercises and solutions, original/current designs, full visual specification, provenance and original outputs, all three Python programs, the pure JavaScript model, all three components, scoped CSS, generator and author verification programs. The review covers source correctness, concept-level teaching and complementary native calculations. Root owns actual browser and integration results.

## Independent science

Six fresh real-row chains start from the two saved initial/trained snapshots and use different eight-row then twenty-four-row groups. Hypothetical feature/target edits exercise nondefault logits. Twelve actual float64 `torch.optim.SGD` steps compare every loss, logit, parameter gradient, next parameter and retained momentum buffer against JavaScript under four physical limits. Every one of the twelve 67-coordinate gradients is also checked by central finite differences: 804 derivatives, maximum error 1.23e−10. Maximum native discrepancy is 1.78e−15.

Whole-row replication/reversal preserves the intended mean and optimizer update. Hidden-unit relabeling carries parameters, derivatives and existing momentum consistently. One hundred seeded cases additionally verify target-mass rescaling/regrouping/permutation invariance, zero-mass undefined outputs, coefficients reconstructing the intended derivative, normalized-input translation and sign symmetries, downstream-scale finite differences, and scalar forward/backward/optimizer clock invariants. The reviewer records 22,257 complementary comparisons and verifies all 46 author source hashes. The 82,096 existing author comparisons, true BatchNorm checks and complete unchanged twenty-epoch 32/12/7 fits remain reusable; no extra fitting run or result selection was performed.

Read the primary [PyTorch cross-entropy reduction formulas](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html) and [AMP accumulation/unscale ordering](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation). The manuscript distinguishes class-weight mass for integer targets from position count for probability targets. It also correctly keeps a common scale until accumulation completes, unscales/clips at the update boundary and scopes skipped updates separately from data consumption. GPU AMP and multiprocess DDP are explicitly explained contracts, not claimed executed experiments.

## Separate learning-experience assessment

1. The twelve-row memory limit and thirty-two-row desired update give the opening a concrete purpose. The first-pass route reaches runnable code before the specialized execution branches.
2. Example, physical chunk, effective group, update and epoch are counted separately with actual remainder rows. The numerical counts are not disguised timing measurements.
3. Forward values, gradient buffers and optimizer memory have distinct visible stores. Scalar chain-rule arithmetic precedes tensor/API terminology; the figure and live trace preserve the fact that backward does not update the weight.
4. Common-scale coefficient bars explain unequal chunk means before the algebra. Fresh rows, editable boundaries and three policies expose discarded gradients versus gradients evaluated at changed parameters.
5. Weighted inclusion uses separate numerator and mass rails, explicit row identities and actual coefficients. Undefined means remain distinct from zero. The equal-mass preset and identical-derivative example distinguish structural from accidental agreement.
6. Complete offline Iris code explains split-only preprocessing, shapes, fixed row orders and actual denominators. All epoch observations, reversals and initialization are retained. Parameter/momentum gaps measure accumulation equivalence separately from classification quality.
7. The real-data live investigation computes every layer and all 67 derivatives. Its edits are clearly hypothetical single updates, with original source values and restoration controls. Source and model resources are lazy with friendly retry and input ownership above the loading boundary.
8. Normalization moves actual observations relative to full/local means before comparing derivatives. The frozen reference is copied rather than silently refreshed. Equal outputs with different moving means, singleton errors and no-normalization/null cases resolve distinct conceptual traps.
9. Dropout, sampling variability, clipping, AMP and DDP each receive a representation of their own mechanism and a qualified transfer argument. Existing scratch derivative/optimizer owners are actual programs, while this lesson owns the new grouping rule.
10. Five substantial changed problems, the seven-row transfer and closed explanatory solutions require applying the contract. No answer entry or reveal gate interrupts the labs. Root must establish painted readability, keyboard behavior and final browser state preservation separately from SSR/source review.

## Findings and final delta

The only new independent implementation finding was a trace arrow anchored permanently to the first chunk. The author now computes its origin from `event.group`, so a later backward visibly contributes from the corresponding highlighted chunk. Inspected that source delta. The remaining specification sentence requiring explicit execution was also reconciled with immediate bounded recomputation; Step/Back inspect the already-computed chronology.

Root independently found and the author corrected the equal-mass preset's inherited comparison mode, and a long unrounded inline worked expression; those final changes preserve the mechanism and exact table. Native engines and original programs are unchanged. After these deltas, reran the complementary checks and bound the final author/evidence hashes. No unresolved source, scientific or teaching finding remains. Root browser/build evidence remains a separate handoff.

Inspected the final annotation-only correction: the optimizer-store label moved from (372,164) to (470,149), above its rectangle. No numerical model or program changed; the 22,257 comparisons remain applicable. Refreshed the final source and scoped-render bindings.
