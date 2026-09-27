# Weight Initialization: concept-level author review

Author pass 2026-09-26. Read the complete eleven-section lesson, six main practice tasks and changed-case library exercise. Inspected the existing live geometry, symmetry, width and measured-training components plus the topic generator. Canonical manuscript and generator remain synchronized; preserved all programs, measured records, practice and existing investigations.

## Entire-lesson map

| Concept transition | Assessment / action |
| --- | --- |
| Amplifier intuition → twenty-layer signal probe → backward cotangent | Existing measured forward/backward plots explicitly state what is measured. Retained. |
| Mean, variance, second moment → ReLU changes | Four values and editable moment lab already expose the distinction. Retained. |
| Weighted sum → second-moment identity | Added the two-input expansion and exact x=[1,2] expectation, explaining why weight independence removes cross terms without requiring zero-mean inputs. |
| Symmetry → ReLU half-mass → Kaiming rule | Existing derivation and finite-network caveats adequate. Retained. |
| Fan-in/out → Xavier compromise → gain and uniform bounds → activation/API orientation | Numerical 100→25 example and practical code sufficient. Retained. |
| Singular gains → average versus each direction → orthogonal/rectangular maps → nonlinear gates | Existing ellipse and Jacobian lab already gives constructive counterexamples. Retained. |
| QR draw/sign convention → tall/wide Gram checks → library agreement | Complete implementation and complexity explanation preserved. |
| Identical units → symmetry gradients → zero head versus dead ReLU | Existing live steps directly expose which path learns; retained. |
| Digit comparison → training versus validation probabilities | All actual data, counts, seeds and measured selectors preserved. No rerun claimed. |
| μP random sums → correlated updates → restricted Adam architecture | Gap: correlation was named without showing its arithmetic. Added four-term gradient-sign flow, exact linear-SGD calculation and random-vs-aligned width table. Explicitly not a derivation of Adam μP. |
| μP layer categories, forward divisor, group rates → package shapes → coordinate diagnostics | Existing recipe controls and measured coordinates adequate; retained base-width identity and finite-width caveats. |
| LSUV variance rescaling | Added why divide by standard deviation, not variance, and why later layers must be remeasured after earlier changes. |
| Fixup branch scaling, truncation versus clipping, precision, defaults and diagnostic checks | Existing examples, schematic and measured precision table sufficient. Retained. |
| Practice / references / residual bridge | Read all, retained. |

## Diagram and research

`CorrelatedUpdateFigure` shows a single linear squared-loss SGD step, x=[1,−1,1,−1], w=0, y=1, η=.01. Although gradients alternate signs, multiplying updates by the same inputs gives four +.01 contributions. Independent random sign updates have RMS η√n; this aligned update gives ηn. The expectation and exact calculation are labeled separately, with no deep-network benchmark implication. Responsive sign columns and a keyboard-scrollable width table use scoped charcoal/amber styles.

Read [Tensor Programs V](https://arxiv.org/pdf/2203.03466), §2 primer and opening parameterization discussion. Used its distinction between initialization and training scales as a prompt for the original finite linear calculation. Existing paper, official code and alternative lecture links remain. No new video playback or library fit claimed.

## Checks and gates

Four groups passed: JSX parse; complete two-input and four-sign enumerations; explicit SGD changes for widths 4/16/64 and variance-rescaling arithmetic; generator byte equality. Existing programs/measurements remain unchanged; native training was not repeated. Browser and independent review pending root, with source hashes in author-checks.json.
