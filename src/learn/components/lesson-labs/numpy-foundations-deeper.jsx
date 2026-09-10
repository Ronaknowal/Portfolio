import { H3, Prose } from "../content";

import PythonExample from './PythonExample';
import { numpyReferenceExamples as ex } from "../../data/numpy-reference-examples.js";

// Useful verified API depth retained from the prior NumPy lesson, now optional.
export default function NumpyFoundationsDeeper() {
  return <div className="numpy-optional-reference">
<details className="numpy-deeper"><summary>Sort records while preserving their labels</summary>
<PythonExample example={ex.numpySort}><Prose>Argsort supplies indices; applying those indices orders the scores. Reversing the last two sorted indices selects the highest two. Unique counts correspond to its sorted unique values by default, here 1 then 2. Nonzero identifies positions 1 and 3. Partition only guarantees the kth value's position relative to smaller/larger values, not a fully sorted array.</Prose></PythonExample>
    <Prose>Sorting values and sorting their indices solve different problems. Use indices when associated labels or other columns must follow the same permutation. For large top-k problems argpartition can avoid a full sort; then sort the selected candidates if their order matters. Define tie handling explicitly, and use a stable sort when preserving original tie order is important.</Prose>
</details>
<details className="numpy-deeper"><summary>Linear algebra after the mathematics prerequisites</summary>
<Prose>This branch assumes vectors, linear systems and matrix decompositions; the dedicated mathematics lessons develop their geometry. Here, focus on reconstructing or checking each answer.</Prose>
<PythonExample example={ex.numpyAlgebra}><Prose>Elementwise X * w scales columns and keeps shape (2, 2). X @ w forms weighted sums, giving a length-two vector. X @ X.T compares rows by dot products. Einsum's indices ij,j→i state the same contraction as X @ w: sum over j, retain i.</Prose></PythonExample>
    <Prose>For AX=b, solve produces x=[3, 2]; checking A @ x against b verifies the result. Prefer solving the system to explicitly computing an inverse and multiplying. A square matrix may be singular or ill-conditioned; successful execution alone does not establish an accurate solution. Inspect conditioning and residuals when numerical reliability matters.</Prose>
    <H3>Read the SVD output</H3>
    <Prose>SVD factors A as U @ diag(s) @ Vh. Here the singular values are 4 and 2, in descending order. Multiplying U * s scales U's columns, so (U * s) @ Vh reconstructs A. Keeping only the largest singular component retains the bottom-right 4 and discards the smaller 2: the printed rank-one matrix shows exactly what was lost.</Prose>
    <Prose>Singular-vector signs and bases in repeated subspaces can vary between valid solvers. That is why the example prints shapes, singular values, the reconstructed-result check and the approximation rather than claiming a unique U. The dedicated matrix-decomposition lesson develops the geometry; this module shows how to call and verify the API.</Prose>
    <Prose>Lstsq fits an overdetermined linear system; the design contains an intercept and x values 0, 1, 2, so the coefficients are intercept 1 and slope 2. It also returns residuals, rank and singular values; the residual array can be empty depending on rank and dimensions. Eigvalsh is for symmetric/Hermitian matrices; eigh also returns eigenvectors. General eig is not interchangeable with a symmetric solver. The default norm of a 2-D array is the Frobenius norm, not its largest singular value.</Prose>
    <Prose>For batched matrix products, matmul/@ broadcasts leading batch dimensions and contracts the final matrix axes. Dot has different higher-dimensional rules. Prefer the operation whose shape contract matches your intention, and use explicit einsum indices when a contraction is otherwise difficult to audit.</Prose>
</details>
<details className="numpy-deeper"><summary>Reproducible generators and array storage</summary>
<PythonExample example={ex.numpyRandomIO}><Prose>Two generators initialised with the same seed and used identically produce matching draws in this environment. We check that relationship instead of pretending a random array has universal values across all future library releases. Each generator advances after every draw; reusing a seed and reusing an already-advanced generator are different operations.</Prose></PythonExample>
    <Prose>Integers uses an exclusive upper bound by default; normal and uniform sample distributions, choice selects items with optional probabilities/replacement, and permutation reorders items without changing membership. Record NumPy/version and the generator setup for reproducibility; a seed alone does not fix changes in algorithms, call order or parallel execution.</Prose>
    <Prose>Save/load preserve an array's shape and dtype using .npy; savez stores named arrays in an archive. The example uses BytesIO for a self-contained round trip; a real filename works too. Do not enable pickle for untrusted object-array files. Loadtxt/savetxt are useful for simple numeric text; complex tabular data and missing-value schemas often fit a dataframe tool better.</Prose>
</details>
<details className="numpy-deeper"><summary>Preview: train-test preprocessing without leakage</summary>
<Prose>Training data is used to fit model-related quantities; test data is held aside to evaluate the resulting system. Fit preprocessing statistics from training observations and reuse them on test observations, so evaluation does not incorporate information it should hold out.</Prose>
<Prose>Now combine reductions, broadcasting, masks and matrix products. The training array has three observations, two varying features and a constant third feature. Estimate the feature means/scales using training data only, then apply those same statistics to test observations.</Prose>
    <PythonExample example={ex.numpyProject}><Prose>Mean has shape (1, 3) and broadcasts over observations. Standard deviations use ddof=0 here. The constant feature has zero scale, so replacing only that scale with 1 avoids division by zero and leaves its centred values at zero. It cannot acquire unit variance from data that never varied.</Prose></PythonExample>
    <Prose>The test point standardises to about 2.449 on the first two features because it is above the training range. Fitting a new mean/scale on that test point would instead erase this information. The positive-first-feature mask keeps one training row, so selected is (1, 3) and its Gram matrix is (1, 1), containing a squared norm of 3.</Prose>
    <Prose>This is a teaching pipeline for finite arrays, not a complete estimator implementation. Real code should validate dimensions, feature ordering, finite values and an empty training set. Near-zero scales may also require a domain-specific policy rather than only exact-zero handling. Keep fitted preprocessing statistics with the trained model and reuse them consistently.</Prose>
</details>
</div>;
}
