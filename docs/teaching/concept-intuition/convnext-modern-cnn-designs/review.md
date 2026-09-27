# ConvNeXt: concept-level author review

Full production lesson read, including the complete generated sections, experiment interpretation, library bridge and all eight practices. Canonical manuscript and generator updated together; frozen programs, measurements and live investigations retained. This is an author checkpoint, not independent or browser acceptance.

| Location / transition | Disposition and learner support |
| --- | --- |
| Opening / §1 architecture, training recipe, checkpoint | Retained: controlled genealogy separates architecture attribution from a changing training recipe. |
| §2 spatial filtering → channel mixing → nonlinear block | Retained: shape table, native block diagram, expansion/projection reasoning and parameter/MAC decomposition. |
| §2 normalization axes, ε, train/eval behavior | Retained: concrete shifted-location values and live axis investigation expose the distinction from GroupNorm. |
| §2 residual LayerScale vs DropPath | Retained: one mask per specimen, expectation vs final prediction, contribution count vs actual branch execution. |
| §3 hierarchy → head operation order | Fixed gap: original noncommutation statement now has a two-location counterexample and side-by-side native flow figure. Same shape, different information. |
| §3 stage cost, meta tensors, resolution, isotropic variant | Retained: explicit area/width calculation, shape-only vs actual computation distinction, downstream multi-scale use. |
| §4 channel-map norm → denominator → broadcast | Retained: two-map arithmetic and live cell interventions, learned sign and identity cases. |
| §4 identity initialization → trainable parameters | Fixed gap: parameter-gradient chain-rule intermediates explain the existing numerical results, including why zero γ still has a nonzero slope. |
| §5 visibility vs target path, sparse/dense masks | Retained: branch figure, hidden-target intervention and encoder active-set explanation. |
| §5 patch-normalized reconstruction objective | Fixed gap: original two-pixel contrast/brightness counterexample explains what the target normalization discards and why its statistics cannot leak into the encoder. |
| §6 paired experiment, hidden-pixel MSE, frozen probe, diversity diagnostic | Retained: actual data provenance, repeated-seed limitations, raw-pixel baseline, numeric reconstruction and existing labs. No re-fit or changed empirical claim. |
| §7 checkpoint contract, logical vs storage axes, branch fusion | Retained: concrete transforms/metadata, fixed-statistic algebra, linearity boundary and live fusion counterexample. |
| §7 global/local hybrid placement | Fixed gap: exact position-pair counts before/after downsampling connect communication pattern to cost and lost resolution. Explicitly not a runtime estimate. |
| Scratch/library bridge and all practices | Retained: full source, matched weights/axes/gradients, hidden-width modification and eight worked tasks retain depth and advanced choices. |
| References / next topic | Retained annotated primary code, paper, slides, D2L and broader routing to Capsules. |

## Actual research

- Read [official ConvNeXt source](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py), Block, stage construction, forward_features and LayerNorm definitions. Confirmed head normalizes after spatial pooling; the new counterexample is independently constructed arithmetic.
- Read [ConvNeXt V2](https://arxiv.org/pdf/2301.00808), §3 FCMAE and §4 GRN (PDF pages 3–6, pseudocode, initialization and target-loss discussion). Used to check mechanism boundaries, not reproduce wording or experiment rankings. An attempted HTML v3 URL returned 404; the PDF was read instead.

## Visual / verification contract

`ConvNeXtHeadOrderFigure` follows [1,3] and [9,5] through two different orders. Semantic HTML grids stack below 600px; no fixed-height text or raster figure. Unit affine LayerNorm and ε are stated. All original lab diagrams and source disclosures remain in place.

`node scripts/verify-structure-learning-intuition.mjs convnext-modern-cnn-designs` passed four arithmetic groups, JSX parsing and four baseline identities. The receipt binds canonical manuscript, generator, rendered topic and added visual/CSS. Production build, independent full reading and actual rendered desktop/phone inspection remain with the integrating reviewer.
