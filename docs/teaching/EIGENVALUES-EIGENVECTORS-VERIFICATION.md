# Eigenvalues & Eigenvectors — completed author review

10 September 2026. Stable ID `eigenvalues-eigenvectors`, mathematics module position 3. [Individual design and research ledger](EIGENVALUES-EIGENVECTORS-DESIGN.md). This record covers this lesson; it does not imply completion of the full 74-topic rollout or user acceptance.

## Teaching and accuracy

Retained the original matrix [[2,1],[1,2]], its two eigendirections, covariance/PCA application, eig/eigh distinction and nonunique sign/basis cautions. Replaced incomplete snippets and non-executed output placeholders with complete programs. Added the nonzero definition, shifted-null-space derivation, characteristic roots and pair checks, eigenbasis operations, spectral theorem/orthogonality reasoning, repeated and defective cases, complex directions, raw dynamics, transient growth, variance derivation/reconstruction, mixing, graph smoothing, coupled vibration and numerical interpretation.

Three investigations have different jobs: compare an input/output line and its perpendicular remainder; follow unnormalized repeated updates with a linked norm plot; turn a measurement direction through observations and inspect projections, variance and reconstruction error. Two inline figures explain basis coordinates and simultaneous compartment transfers. The four independent tasks change the matrix, challenge an invalid stability claim, change a mixing rule and interpret a tied variance budget. Hints and solutions start closed. Optional deeper branches preserve a readable core route.

Primary references and annotated written/video alternatives are recorded by actual inspected scope in the design ledger. All ten reference/alternative links render in the final source. NumPy programs were executed with 2.3.5; newer online API documentation is identified separately. No full-video viewing is claimed. Examples and physical parameters are declared teaching fixtures, with the assumptions and units needed to interpret them.

Independent peer review inspected field/multiplicity, finite-matrix asymptotic versus transient behavior, sample variance versus total error, residual versus dominant-eigenpair claims, and mixing/spring units. No blocking numerical or conceptual issue was found. Adopted its two precision improvements: **n distinct eigenvalues in the chosen field** suffice for an n×n eigenbasis; continuous-time eigenvalues carry inverse-time units, whereas discrete multipliers are dimensionless. Root additionally clarified the conserved left row vector and complex triangular versus real block-triangular Schur forms.

## Executable and independent model evidence

Command: `node scripts/verify-eigenvalue-examples.mjs`; independent oracle: `scripts/verify-eigenvalue-native.py`. Python 3.12.14 / NumPy 2.3.5. Results: `scratch/eigenvalue-verification/native-results.json`.

- **10 standalone programs** matched literal displayed stdout: symmetric pairs, basis powers, missing/complex directions, nonorthogonal coordinates, complete PCA/SVD comparison, transient growth, two-compartment mixing, bounded normalized power iteration, coupled modes and an independent spectrum/rank-one task.
- **438 direction states** checked against native multiplication and a separately solved least-squares line projection, including reflection, null output, scalar, shear and rotation cases.
- **558 raw recurrence states** checked against `matrix_power`, explicit shear/nilpotent formulas and rotation norm preservation.
- **219 variance states** checked against native covariance, score variance, eigvalsh, projections and total squared error. Verified perpendicular residuals and error = (n−1)×discarded sample variance.
- **100 independently constructed power-iteration cases** checked with orthogonal bases and positive/negative dominant values. The line comparison uses the actual residual/separation bound, not an arbitrarily tighter entrywise tolerance. Invalid matrices/vectors/tolerances and bounded nonconvergence of a real quarter-turn were checked.
- Additional independent hand checks cover three graph modes and smoothing multipliers, the changed mixing matrix, 31 changed-shear powers, the defective perturbation split and variance budget.

Pure models reject unknown presets and invalid finite ranges, return independent state arrays and do not mutate presets. Browser investigations use small bounded matrices and event-driven calculations; they are not a general eigensolver, continuous animation or benchmark. No dependency or eager runtime bundle was introduced.

## Browser and actual reading review

`node scripts/review-eigenvalue-lesson.cjs` passed in Edge at **1440×1000 and 390×1000**. At each width: 48 direction states, 234 recurrence states, 18 variance states, all ten rendered programs and exact outputs, eight unique working anchors, reset/previous/next limits and keyboard disclosures. Checks compare numerical readouts, actual vector endpoints, projected points and every norm-plot position with the independently verified model. No page/figure overflow or browser errors.

`node scripts/review-eigenvalue-reading.cjs` is the final supplemental check after intentional prose/reference/formula changes. It verifies ten actual reference links, six technical list entries, six ordinary reading sections, absence of displayed-equation/page overflow and valid longest norm-axis labels at both widths. The initial run caught an incorrect `Sources` prop contract that hid six references and two equations that required sideways reading; both were repaired, then the check passed. The formulas are now stacked rather than shrunk. No model changed in these repairs.

Opened screenshots include desktop diagonal direction, non-eigenvector remainder, transient trajectory/norm, PCA projections, mixing and lesson entry; mobile null-output direction, perpendicular remainder, recurrence, PCA, eigenbasis flow and ordinary eigenbasis/PCA/practice pages; desktop numerical-contract reading. Read scales, ticks, arrow/zero meanings, projected overlaps, values, formula wrapping and prose transitions in the rendered pages. Diagram refinements made matrices compact, gave the covariance caption room and replaced spatial transfer arrows with explicit destinations that remain meaningful on stacked mobile layouts.

Artifacts are in `scratch/eigenvalue-lesson-review/`, including `results.json`, `reading-results.json`, isolated figures and ordinary reading screenshots. Fixed navigation was hidden only for isolated element screenshots and restored afterward. Native controls and disclosures retain keyboard focus and touch targets. The actual next topic is Matrix Calculus & Jacobians; publication does not change that sequence.

## Integration and limits

[Rollout integration evidence](DSA-MATH-FOUNDATIONS-INTEGRATION.md) owns the final production build, conservation, module-order and lazy-loading checks. The per-topic progress entry records the reviewed source and its owned dependency hashes only after those checks. The original source was retained in the captured baseline and local authoring backup; no topic ID or publication mapping was removed.

Tests and author reviews support the mathematical calculations and implemented behavior. They do not establish external learner outcomes, physical validation of the illustrative applications or user approval. Linux remains the explicitly approved quality reference.
