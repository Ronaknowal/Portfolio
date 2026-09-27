# Bias–variance: whole-lesson concept-transition review

26 September 2026. Read the full eleven-section lesson, eight practice solutions, implementation explanations and reference annotations. Inspected the existing fixed-design figure's projector mechanism to avoid duplicating it. Revised production JSX is the content checkpoint.

| Concept / exact source location | Finding and action | Representation and transfer evidence |
| --- | --- | --- |
| Intro / §1 fixed input, random fit and fresh target | Sufficient carefully separated random objects; retained. | Existing sensor construction and observation/prediction rulers. |
| §2 squared-error decomposition | Sufficient algebra tied to deviations and vanished cross terms; retained. | Squared-deviation representation and noisy-target versus mean-target distinction. |
| §2 irreducible relative to available information | Sufficient conditional example; retained. | Existing explanation of how adding useful input changes the target/noise split. |
| §3 finite training worlds | Sufficient enumeration rather than invented repeated-fit curves; retained. | Exact finite-world lab and complete scratch program. |
| §3 bootstrap | Sufficient resampling distribution versus inaccessible true population; retained. | Existing limits on claiming true bias. |
| §4 learning curves | Sufficient nested sizes, training/validation ownership and repeated-split variability; retained. | Actual airfoil curves and run evidence remain unchanged. |
| §5 optimization versus model-size diagnosis | Sufficient changed setting and trajectory distinction; retained. | Existing trajectory investigation and controls. |
| §6 affordable diagnostic protocol | Sufficient fit budgets and changed estimands; retained. | Existing practical pipeline and leakage cautions. |
| §7 same-input optimism | Sufficient projector geometry and train/fresh noise derivation; retained. | Existing SameInputsFigure, checked in its component source. |
| §7 effective degrees and general smoother | Gap: trace and squared norm could look interchangeable. Added self-influence explanation and S=diag(0.8,0.2) direction table. | New smoother-direction-budget gives trace 1, squared norm 0.68, mean prediction variance 0.34 and optimism gap 1 at n=2, noise variance 1. |
| §8 classification loss | Sufficient separate zero-one example, no false universal squared-loss decomposition; retained. | Existing two contrasting randomized decisions and loss figure. |
| §8 averaging | Sufficient pair-covariance count and common-bias caveat; retained. | Explicit v=4, correlation=0.5, four-member average. |
| §9 double descent | Added the missing causal chain: unidentified directions, minimum-norm choice, small singular values amplifying noise, and redundancy beyond the interpolation boundary. | Existing labelled asymptotic curve and formula retained; no empirical neural-network claim added. |
| §10–11 transfer and sources | Eight explained practices, readiness and alternatives retained. | New direction budget supports the existing same-design reasoning. |

## Research actually inspected

- [Caltech Learning From Data, lecture 8 slides](https://work.caltech.edu/slides/slides08.pdf): opening bias–variance progression and repeated-dataset interpretation. Slide text was inspected; no claim of watching the lecture video.
- [Nakkiran, More Data Can Hurt for Linear Regression](https://arxiv.org/pdf/1912.07242): sections 2–3, pseudoinverse signal/noise split, near-square conditioning and the stated isotropic setting. This is the source for the linear mechanism; the new paragraph does not extrapolate it to every neural network.
- Also inspected [Nakkiran et al., Deep Double Descent](https://arxiv.org/pdf/1912.02292), the interpolation discussion and its explicit informal-hypothesis qualification. It reinforced keeping the linear construction separate from general deep-learning claims.

## Checks and representation

The added HTML direction table gives amplitudes and variances in separate columns rather than reusing the existing projector illustration. The verifier enumerates independent symmetric unit-variance noise, calculates train/fresh errors, and verifies both trace identities. It also checks a two-direction near-singular solve so the noise-amplification bridge is concrete.

All original fitted outputs, examples, figures and lab models are retained. No untouched native fit campaign was rerun. JSX author parse passed only when recorded in author-checks.json; independent teaching review and browser review remain pending.
