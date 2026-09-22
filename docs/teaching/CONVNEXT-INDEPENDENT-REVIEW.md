# ConvNeXt implementation review

22 September 2026. Reviewer: root, independently of author `convnext_finish`. Scope is only Deep Learning position12, `convnext-modern-cnn-designs`, prepared revision3. This review does not authorize another lesson.

## Review coverage

Read the full prepared manuscript, its implementation/visual contract, generated teaching route, model arithmetic, mechanism components, canonical block/hierarchy builder and Torchvision bridge, and author/native checks. The introduction separates architecture changes from training recipe; the block and hierarchy precede GRN and masked reconstruction; actual paired outcomes, leakage counterexample, folding boundary and changed-design practice are retained. The full core and optional references remain. Static route rendering and on-demand canonical source preserve depth without bundling training artifacts into navigation.

The complete local V1/V2 builder owns block and hierarchy composition. The maintained-library bridge copies matching V1 state/layout and compares outputs and gradients; the executed author probes include signed, non-tiny LayerScale so initialization cannot conceal a wrong branch. Optional pretrained photograph use is explicitly unexecuted, and is not confused with the real small masked-learning experiment. Six existing training fits remain conserved evidence; this review did not refit them.

## Complementary verification

`verify-convnext-independent.py` produces reviewer-owned native fixtures; `verify-convnext-independent.mjs` compares the browser calculations to them. Four groups passed:

- Changed three-channel and constant-group LayerNorm values under two different reduction boundaries.
- Signed and zero-response GRN cases, with a changed-state native input Jacobian check.
- Six actual native V1/V2 parameter counts and exact zero-projection residual identities.
- Three full25-position branch-fusion references using independent Torch convolutions and evaluation BatchNorm, including negative and zero gamma.

The largest browser/native absolute difference was `2.842170943040401e-14`. Source-bound reports are `evidence/convnext-independent-native.json` and `evidence/convnext-independent.json`. These supplement the author's24 native reconstruction fixtures and library comparisons; they do not assert measured speed, ImageNet reproduction or a full sparse FCMAE implementation.

## Finding and disposition

Continuous feature/GRN/kernel/pixel controls accepted off-step numeric values while their discrete range inputs rounded the visible thumb. The author changed continuous controls to `step="any"`, preserving integral architecture dimensions, and added browser numeric/range parity checks. No mathematical, coverage or reference defect remained in this source review. Theme overrides keep plot axes/text neutral and controls amber.

Final rendered interaction, responsive screenshots, source currency and delivery reconciliation belong to [the batch integration record](DEEP-LEARNING-SEQUENCE-IMPLEMENTATION.md). Its current evidence supersedes any earlier author note reserving those checks for the parent. This source review alone does not claim browser or user acceptance.
