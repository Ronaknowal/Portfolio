# Residual connections — independent concept review

26 September 2026. Reviewer: classical_representation_intuition, separate from the author. Read the complete manuscript, all seven practice solutions/references, generated learner text and generator additions, and the new ResidualGateIntuition component/CSS. This is a source-level review, not browser evidence or an actual learner trial.

## Full progression inspected

| Source location | Assessment |
| --- | --- |
| “One block…” through “The skip cannot promise…” | The vector forward/loss/update leads to two gradient routes. Jacobian order and cancellation explain why an identity summand does not guarantee a nonzero total derivative. The scalar counterexample is retained. |
| “Operation order…” and “Shape agreement…” | Pre/post-activation, normalization and the final ReLU change the preserved route. Projection and coordinate meaning are distinguished from mere compatible shapes. |
| “Start near identity…” | Zero last map, dead ReLU, trainable gate and fixed small scale have different available gradients; the learner can trace the first update. |
| “A complete comparison…” and “Own the addition…” | Actual digit ablations, the unfavourable case and complete code remain. Generator additions explain train/eval, buffers, parameters, the return expression and no accidental broadcasting. The prior initialization connection now exists in canonical source. |
| “Many routes…” | Nonlinear paths are not independent ensembles. DenseNet, ResNeXt and Res2Net remain separate mechanisms. The new input-dependent Highway example includes the derivative of the gate, unlike a scalar ReZero parameter. |
| “Connections…” and “What really costs…” | Denoising subtraction, Euler steps and stability remain tied to their separate purposes. Checkpoint memory counts explain what must be retained and recomputed, rather than treating an addition as a memory solution. |
| Practice and references | All seven solutions preserve changed correction, cancellation, projection, dead branch, channel count, actual evidence and stability reasoning. |

## Learning-experience checklist

1. **First-pass route:** one complete block comes before depth, operation order and architecture relatives.
2. **Cautions:** failure examples qualify the particular mechanism; the added Highway explanation does not add repetitive caveats.
3. **Question/data:** fixed arithmetic is distinguished from actual digit runs and their limited comparison.
4. **Investigations:** correction, depth, operation order, shape, initialization and evidence investigations retain immediate controls and separate decisions.
5. **Representations:** the gate derivative is broken into carry, transformed route and changing gate; it is not a decorative gate icon. CSS/source inspected, rendering left to root.
6. **Connections:** initialization's scale links to residual routing; normalization, differentiation and convolution owners are reused instead of silently assuming an unexplained operator.
7. **Implementation:** full programs, pullbacks and library mode behavior remain. The canonical-source status wording was corrected by the author; the old generator already replaced it for learners, so this was not evidence of a live status bug.
8. **Practice:** all seven changed tasks retain worked reasoning and empirical interpretation.
9. **Evidence:** current manuscript, generator and learner bytes are bound; no unchanged fitting campaign rerun or browser claim.
10. **Every transition:** review included Highway's input derivative, nonlinear path limitations, Euler stability and memory lifetimes, not only basic addition.

Complementary checks use a nonlinear Highway transform and a different gate slope, a 36-layer checkpoint balance, and a nonidentity projection checked by finite differences. No remaining source-level blocker found. Existing resources were assessed in context; no new web/video inspection claimed. Accepted at source level with browser/integration evidence separate.
