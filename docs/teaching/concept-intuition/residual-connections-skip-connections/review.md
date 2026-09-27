# Residual Connections: concept-level author review

Author pass 2026-09-26. Read all thirteen sections, seven main practice tasks and vector-gate implementation extension. Inspected generator and existing memory/path figure contracts. This lesson already supports most concepts with exact local examples and live investigations; selective additions preserve that work.

## Entire-lesson concept map

| Transition | Assessment / action |
| --- | --- |
| Replacement versus correction → coordinate addition → loss and outer-product gradient | Complete arithmetic and editable correction lab adequate. Retained. |
| Degradation versus representability → two backward paths → cancellation | Existing gradient-lane figure, example and scalar-depth lab adequately separate direct contribution from total gradient. Retained. |
| Scalar product → ordered Jacobians and norm bounds | Existing explanation states order and depth dependence correctly. Retained. |
| Post-activation/post-norm versus pre-activation/pre-norm | Zero-correction and common-shift examples plus live operation-order lab provide sufficient support. Retained future-attention scope boundary. |
| Equal shapes versus aligned coordinates → projection and transpose gradient | Named vector example and projection lab sufficient; broadcasting and image-alignment caveats retained. |
| Zero branch output → first trainable parameter → ReZero/LayerScale/depth scaling | Existing gradient examples and opening lab already explain differences. Retained covariance term and no universal-depth promise. |
| Matched digit experiment → measured optimization/generalization/ablation | Full actual fits, counts, code, diagnostics and smaller-model baseline preserved. |
| Scratch routing → PyTorch reuse → buffer versus Parameter → vector gate | Existing code walkthrough and changed-case practice adequate; retained implementation owners. |
| Linear path expansion → nonlinear counterexample → deletion intervention | Existing 6-versus-4 example and path figure sufficient. Retained shared-weights and ablation interpretation. |
| Addition versus concatenation → DenseNet / ResNeXt / Res2Net | Concrete feature counts and explicit future-convolution prerequisites adequate. Retained. |
| Highway coupled gate → derivative paths | Gap: gate changing derivatives was stated without showing the missing path. Added residual rewrite, product-rule decomposition and numerical contribution diagram. |
| Denoising residual target → Euler stability | Existing arithmetic figures/live Euler mechanism adequate. Retained distinction from an arbitrary trained solver. |
| Addition derivative storage → checkpoint recomputation | Existing memory-lifetime figure retained. Added where L/k and k come from and a 16-layer chunk example. |
| Practice, primary/alternative resources and next-topic bridge | All retained. |

## Research and visual contract

Read [Highway Networks](https://arxiv.org/pdf/1505.00387), §2 coupled carry/transform equations and sigmoid construction. The original lesson already links it. Derived the full input derivative directly instead of relying on limiting-gate intuition. Also opened D2L's ResNet/ResNeXt route for orientation; no new claim of completed chapter/video review.

`HighwayGateFigure` traces H(x)=2x, T(x)=sigmoid(x), x=1. Forward carry .2689 plus transform 1.4621 gives 1.7311. Derivative contributions .2689, 1.4621 and .1966 sum to 1.9277. The input-dependent gate adds the third term. Separate forward/derivative columns use semantic labels that remain valid when stacked on mobile. This is an exact static walkthrough; it is not labeled a playable lab. Existing numerous live labs remain.

## Actual checks / remaining gates

Four groups passed: changed JSX parse; residual rewrite and derivative finite differences at five signed inputs; checkpoint-count minimizer; generator byte equality. Native historical fits and full training programs remain unchanged and were not rerun. Browser and independent review are pending root; receipt binds the manuscript, generator and runtime additions.
