# ICA: whole-lesson concept-transition review

26 September 2026. Read all eleven sections, complete surrounding implementation explanations, six practices and resources. Production JSX is the content checkpoint. Independent teaching and browser review remain pending.

| Concept / exact source location | Finding and action | Representation / transfer evidence |
| --- | --- | --- |
| §1 sensor mixtures, orientation and recoverability | Sufficient weighted sensor recipe and conditions. Retained. | IcaMixingFigure; practice 1. |
| §2 independence versus covariance | Sufficient dependent-but-uncorrelated distribution and exact source probabilities. Retained. | IcaDependenceFigure, exact binary source construction. |
| §3 centering, eigenscaling and whitening | Sufficient saved-state transformations and rotated support. Retained. | IcaWhiteningFigure; practice 3. |
| §4 equal covariance → higher-order information | Gap: kurtosis calculation lacked an explicit marginal representation. Added binary-source and diagonal-projection probability masses. | New ica-projection-mass figure: same variance, fourth moments 1 and 2, excess kurtosis −2 and −1. |
| §4 Gaussian ambiguity | Sufficient population versus finite-sample distinction. Retained. | Contrast existing rotated square with Gaussian symmetry. |
| §4 rotation lab and negentropy | Sufficient live joint-support/contrast investigation. Removed stale instruction to predict before revealing a rotation; directs immediate manipulation. | IcaRotationLab; practice 2. |
| §5 fixed-point update | Added why a linear nonlinearity would cancel to zero after whitening; higher-order shape must supply a direction. | Existing IcaFixedPointFigure and Newton derivation retained. |
| §5 deflation, normalization and convergence | Sufficient explicit steps and sign-invariant stopping. Retained. | Complete handSeparation program; practice 3. |
| §6 real recording, split and library controls | Sufficient data provenance and protected interval. Removed another prediction-first prompt without changing computations. | IcaSplitFigure, realRecording program, IcaOutcomeFigure; practice 4. |
| §7 reconstruction and component removal | Sufficient additive contribution and removal tradeoff. Retained. | IcaContributionLab; practice 5. |
| §8 entropy / mutual information | Added joint-density versus product-of-marginals intuition: preserve individual distributions while removing their relationship. | Existing objective and assumptions retained. |
| §8 likelihood Jacobian | Added a coordinate-volume example: B = diag(2,1), area 0.5 becomes 1; density changes reciprocally to preserve probability. | Exact HTML arithmetic alongside determinant formula. |
| §8 convolutive/overcomplete/nonlinear alternatives | Sufficient failure conditions and correct distinct-owner links. Retained. | No claim that instantaneous FastICA solves these cases. |
| §8 costs; §9–11 practice/readiness/resources | Sufficient bounded compute, ambiguity alignment, six explained practices and annotated references. Retained. | From-scratch and library programs unchanged. |

## Research actually inspected

- [Hyvärinen and Oja, Independent Component Analysis: Algorithms and Applications](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf): mixture/independence exposition, section 5.2 whitening, section 6.1 update and Newton derivation (PDF pages 11–13). The original author's progression supports separating second-order normalization from remaining shape; no wording or graphics copied.

The binary projection distribution and rectangle-volume example are original calculations. The existing annotated resource already links this tutorial, so no duplicate alternative was added.

## Visual specification and author checks

The two mass plots share coordinate and height scales, explicitly use probabilities rather than continuous densities, and show all possible projected values. HTML supplies all moments and the interpretation; responsive panels and bounded SVGs avoid shrinking one wide comparison. The existing live lab remains the place for changing the rotation.

The scoped verifier checks exact projection moments, cancellation with a linear contrast, probability-preserving volume change and JSX syntax. Existing scratch/library scripts, engines, numerical datasets and published run evidence are preserved. Browser inspection and independent explanation review are pending; this record is not a completion claim for those stages.

Browser follow-up from root: the new probability-mass labels inherited SVG's default black fill. Both numerical label rows now explicitly use the figure's light currentColor, and the new figure uses neutral charcoal/text/border. Mathematical geometry is unchanged. Rendered recheck remains with root.
