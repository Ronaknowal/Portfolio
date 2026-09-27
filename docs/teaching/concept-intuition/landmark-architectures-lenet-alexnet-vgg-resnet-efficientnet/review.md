# Concept-level author review: Landmark Architectures

Read all thirteen current sections, complete composition/library route, seven core questions with solutions and the VGG changed-head task. Production JSX is the current checkpoint, with matching canonical manuscript/renderer. Browser and independent review remain pending root.

## Entire-lesson concept map

| Transition | Assessment / change |
| --- | --- |
| Opening and §1 architecture versus weights/recipe; shapes, stages, backbone/head; budget types | Existing feature-route figure and detailed parameter/MAC example sufficient. Reworded an early mandatory prediction prompt as a comparison. |
| §2 LeNet local-to-global hierarchy, C5 versus dense, historical fidelity | Existing historical-shape diagram and C5 input-size explanation sufficient. Preserved details on historical subsampling, partial connectivity and class penalties. |
| §3 AlexNet stride/pool grids, training versus architecture; VGG support/nonlinearity/head | Existing kernel figure, head live lab, real parameter arithmetic and evaluation-protocol distinctions strong. Retained without extra general illustration. |
| §4 Inception parallel spatial routes and width reduction; ResNet addition and bottleneck | Existing concatenation/addition diagram and numerical examples support each distinction. Residual-gradient depth reused from earlier lesson. |
| §5 spatial/channel factorization; inverted bottleneck; SE global context; compound scaling | Existing narrow/wide flow, gate example/live lab and scaling lab sufficient, including linear-projection rationale and idealized exponent limits. |
| Complete builders/library route | Five primitive-based full constructors, native matched-state comparisons, complete pretrained route with explicit unexecuted download boundary, and transfer-policy reuse preserved. |
| §6 controlled comparisons, top-k, packages/recipes, Pareto | Existing evidence table and later budget selector sufficient. No unsupported benchmark curve introduced. |
| §7 DenseNet | Added channel-provenance diagram showing original maps retained and three-channel groups appended. Explicit example shows growing input cost despite fixed growth rate; actual bottleneck variants kept distinct. |
| §7 RegNet | Added full proposal → rounded logarithmic exponent → per-block width → stage-count example/figure. Channel/group compatibility remains explicit. |
| §7 NFNet | Added unit-wise AGC numeric norm/cap/direction/SGD example and zero-norm floor; avoids treating a residual scale or deleting BN as the full recipe. |
| §7 frozen features as a loss | Added two-pixel flow/derivative figure showing fixed weights with nonzero input derivative, safe target branch versus detached generated branch, and feature equality with pixel inequality. |
| §8 real small-model experiment | All twelve actual fits, fixed training protocol, partial matching, development-only caveats and live results retained. No extra native training needed. |
| §9 CAM correspondence and limits | Existing exact score-map example, live weights/maps and recorded-map viewer explain signed contributions, bias and spatial resolution well. Retained. |
| §10–11 practice, synthesis, next topic | All seven worked problems preserved; changed class-map question asks calculation/explanation without prediction gate. Next depthwise topic remains explicit. |

## Sources actually read for gaps

- RegNet §3.3 equations 2–4: https://arxiv.org/pdf/2003.13678 — the new six-block example uses the actual exponent-rounding mechanism, not arbitrary rounding of widths.
- NFNet §3 equation 3 and unit-wise definition: https://arxiv.org/pdf/2102.06171 — norm-relative clipping, norm floor and direct SGD interpretation.
- Johnson et al., §3.2 equation 2 and feature reconstruction discussion: https://arxiv.org/pdf/1603.08155 — fixed feature network and matching representations; original toy arithmetic makes the gradient and information boundary visible.

## Checks

JSX parsing, DenseNet group widths/costs, all six RegNet quantizations and stage counts, AGC arithmetic and finite-difference input gradient passed. Canonical generation exactly reproduces the topic and keeps all program/measurement routes. Topic-scoped DOM figures wrap at narrow widths without fixed SVG label positions. Existing numerical fits were preserved rather than rerun. Root still owns rendered and independent closure.
