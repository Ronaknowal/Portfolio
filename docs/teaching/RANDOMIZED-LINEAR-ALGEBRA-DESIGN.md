# Randomized Linear Algebra — active individual design

10 September 2026. Active authoring within the 74-topic DSA/mathematics goal. Preserve `randomized-linear-algebra`, its title, existing publication mapping and mathematics position 6. Author and independent review are complete; see RANDOMIZED-LINEAR-ALGEBRA-VERIFICATION.md for actual evidence. Integrated review remains separate.

## Existing content and scope

Read the entire previous lesson and exact inventory/returned notes. No destination note exists; the unassigned DSA bit-manipulation item is unrelated. Retain the random-column range finder, small compressed SVD, oversampling/power iteration, fixed-seed reproducibility, the 100×40 Gaussian example with k=5 and p=4, and the proposed structured-matrix comparison. Repair the output block that substitutes prose for an actual error, disconnected optional power code, missing exact shapes/error contracts, unstable orthogonalization guidance, and unsupported general speed/quality implications.

The broad title warrants a connected account of random range finding and low-rank compression, row sketching for least squares, random norm/subspace preservation and matrix-vector trace estimation. These are different goals and require different guarantees. Deeper randomized algorithms, concentration proofs, specialized sketch families, matrix-free iterative solvers and distributed systems have later owners; introduce the relevant mechanism and limits rather than claim an exhaustive software survey. Reassess during research before adding an application or branch.

## Implemented progression and representations

1. Ask a concrete question: how can a few matrix products discover a useful subspace without factoring the full input? Refresh column combinations, orthogonal projection, rank and Frobenius versus spectral error. A small two-dimensional weighted-column figure exposes what one probe can miss or cancel.
2. Derive Y=AΩ, orthonormal Q, B=QᵀA and a lifted rank-k factorization. Keep target rank k distinct from sketch width ℓ=k+p. Identify original-space and small-space residuals and the exact-SVD tail benchmark; deterministic arithmetic is also floating-point.
3. Explain Gaussian probe coverage, oversampling and the σ^(2q+1) effect of subspace iteration. Orthogonalize between Aᵀ and A products, count actual data passes and work, and compare spectra/seeds using calculated errors rather than invented timing curves. A bounded spectral-budget investigation should expose approximation tradeoffs without pretending a small fixture proves a theorem.
4. Turn from right sketches (column-space discovery) to left sketches (compress observations). Derive a sketched least-squares objective, evaluate the solution in the original data, and distinguish unbiased quantities from reliable solutions. A row-selection/sketch investigation should reveal influential rows and rank loss. An embedding of span([A,b]) supports the residual guarantee; preserving a few arbitrary sample distances alone does not.
5. Develop useful alternative products: an independently checked Rademacher trace estimate, its expectation mechanism and nonmonotone finite-run error. Optional leverage-score, sparse/structured sketch, one-pass and preconditioning branches need explicit assumptions, bounds and useful ownership. The number of labs is not fixed; each must answer a distinct question.
6. Complete executable NumPy programs and hand examples, independent changed-spectrum/failed-sketch/estimation tasks, annotated alternate written/video learning and the actual next Multivariate Calculus topic. Practical uses must state the operator/data objective and approximation consequences, not simply list industries.

## Verification contracts

Pure browser calculations must be bounded and deterministic for a chosen seed, with independent NumPy or exhaustive finite oracles, rank-deficient/zero/flat-spectrum cases and actual plot-value checks. Any small eigensolver/QR used by a visual requires independent reconstruction, orthogonality and residual checks; avoid adding that machinery if a simpler accurate representation answers the same question. Native examples must be full separate programs with captured literal stdout, original-error evaluation and meaningful counterexamples. Verify current API options against primary documentation. Browser review must include actual desktop/mobile/keyboard interactions and opened normal-reading/diagram screenshots, sources, formulas and next-topic order.

## Source selection

The initial Cornell CS6220 index supplied reference leads, but its linked notes were not reviewed or used as lesson evidence. HMT and Martinsson–Tropp provide the primary mathematical sources; the exact selected sections actually read are listed below. The full papers and linked videos were not read/watched in their entirety.

## Targeted research and implementation decisions

The complete draft now retains the original Gaussian100×40/k5/p4 example with real captured stdout, and adds the connected range/least-squares/trace learning route. Four distinct investigations and two inline figures serve the recorded hurdles. Eleven standalone NumPy programs and final desktop/mobile/keyboard checks pass. The independent numerical/content review and final320px equation repair are recorded in RANDOMIZED-LINEAR-ALGEBRA-VERIFICATION.md; this design is not a substitute for those records.

Additional primary-source scope actually read on10 September2026:

- HMT: Algorithms4.3–4.4, Remark4.3 and the compressed-SVD stage around PDF pages25–29; Theorem10.5's assumptions/statement and surrounding explanation. Apply normalization between both alternating products. The expected Frobenius range bound is not silently relabeled as a final rank-k guarantee. No claim to have read the whole proof or paper.
- Martinsson–Tropp monograph (arXiv version3,2021): probability and trace sections around pages9–12, restricted singular values/Gaussian embeddings in section8, row leverage and rescaling in section9.6, least-squares sketching/preconditioning around pages40–42, and selected Krylov/Nyström/single-view definitions around pages50–52,61–65. The source's historical claims of what is currently known are not repeated as current2026 findings. General branch orientation is explicitly separate from complete implementations.
- Current scikit-learn randomized_svd API signature, oversampling/iteration/normalization/transposition/seed options and factor shapes read. The available native environment uses NumPy2.3.5; no scikit-learn execution is claimed. Documentation contains wording/version inconsistencies, so the lesson uses explicit local settings rather than repeating an ambiguous default.
- Tropp's official ICML2014 tutorial entry and selected slides16–41: sampling versus low-rank factorization, two-stage shapes and costs. Its theoretical/illustrative figures are not copied as benchmarks. No full talk playback claimed.
- Brunton/Kutz official Chapter1 resource page and section1.8's video index inspected; the direct Randomized SVD YouTube title verified. This provides a useful optional visual route, not a full-video accuracy endorsement. Attempts to fetch two Tropp-linked MP4s returned cache misses; those are not represented as watched resources.

Pure model checks currently pass125 weighted probes,1440 spectra/rank/width/iteration/seed combinations, all256 observation subsets,495 trace states and13 invalid-input groups. Independent NumPy SVD/lstsq and exhaustive sign enumeration verify original residuals, projectors, eigenvector orthogonality/residuals, exact rank floors, leverage and variance. The maximum independent projector difference is about2.52e-14. These checks do not replace rendered or mathematical review.

The implementation uses bounded6×6 spectra,8 selectable observations and32 trace probes. Browser orthogonalization and a small Gram eigensolver are explained as checked fixture machinery, not recommended production SVD code. The complete native helper uses an SVD of the skinny sketch for numerical rank, with real-input validation and explicit zero/rank-deficient behavior. Keep the final source and exact stdout in semantic topic-owned files. Do not regenerate it from an older scratch draft after final corrections.
