# Independent review of the thirteen mathematics depth assessments

22 September 2026. Reviewer: root integration agent, distinct from the author. **The scoped content/code review passes after the corrections below.** Production browser verification and final phase integration are separate receipts.

I read all thirteen added learner sections, the ten complete new programs, the relevant existing derivative, second-order and stochastic-process implementations, and the author's source/evidence inventory. The review checks implementation ownership, explanation continuity, numerical and resource contracts, ordinary-tool mapping and independent practice. It does not claim that a finite set of fixtures proves every numerical method or every production variant correct.

## Findings and closure

1. The first blockwise MMD implementation retained a list of all block sums, contradicting its bounded auxiliary-memory claim. It now streams a generator into `math.fsum`; kernel arrays remain block-sized. The reviewer and author identified this independently during the first source pass. Numerical and block-partition checks pass on the corrected source.
2. Matrix and Multivariate Calculus linked to an unpublished `backpropagation-computational-graphs` ID as the reusable AD owner. Both now use the actual published `backpropagation-automatic-differentiation` lesson and its current title. A nonexistent route cannot satisfy the reuse requirement.
3. Queueing's opening package-free setup statement was too broad after adding a SimPy comparison. It now scopes that claim to the original core programs and identifies the later dependency-bearing route. The causal fitted-model bridge was also moved after the existing IPW/AIPW definitions to preserve mechanism-first reading.

No unresolved issue remains in the reviewed additions. The broad initial “needs deeper review” labels are resolved against each lesson's declared outcomes, not by requiring an arbitrary package for every theorem or workflow.

## Complementary numerical evidence

[`review-math-depth-remediation.py`](../../../scripts/review-math-depth-remediation.py) and its [source-bound receipt](../evidence/math-depth-remediation-independent.json) provide ten additional groups. These investigate different fixtures or mathematical routes rather than repeating the author's 71-group campaign:

- Affine derivatives: finite differences of every input, weight and bias coordinate on two changed rectangular shapes, exposing reduction and transpose mistakes.
- Simplex-constrained quadratic: independent scalar KKT root solve, changed active sets, projection translation.
- Entropic transport: analytic symmetric 2×2 solution at three epsilon values and invariance to nonconstant row/column cost shifts.
- Causal estimation: exact rational standardized contrasts with unequal cell sizes and treatment-label reversal.
- MMD: scalar pairwise kernel oracle for unequal sample sizes, both estimators, several block partitions and large coordinate translations.
- Mutual information: rational cell probabilities, empty margins, transpose invariance and exact perfect three-class information.
- Graph operators: series-resistance harmonic solution, edge-energy identity, loop cancellation and zero-weight isolate.
- Queueing: independently hand-scheduled ties/zero services and interval-sweep integration of censored occupancy.
- SDEs: closed multiplicative Euler and Milstein path products for specified nonrandom increments.
- Finite elements: constant-flux solution with piecewise conductivity, reused factors and an exactly nodal point load.

The initial verifier had a local helper-shadowing error; it recorded failure, was corrected, and the full ten-group run passed. Failed runs are not counted as passing evidence. No plots or hardware benchmarks were inferred from these values.

## Learning and ownership assessment

Matrix Calculus exposes derivative maps and an affine pullback; Multivariate explicitly reuses that actual executable bridge with a scalar function/direction example. Second-Order retains its two-loop application, actual PyTorch LBFGS fit, matrix-free CG/HVP, finite exact/factored Fisher solves and original matrix Shampoo recurrence. Its added boundary makes the numerical primitives and the additional specifications needed for multilayer/distributed variants explicit. It does not falsely call a two-layer library import a scratch optimizer.

Constrained optimization now carries its projection into a reusable solver; optimal transport preserves exact/LP routes while adding the entropic API comparison. Causal identification stays an assumption/theorem task and is distinguished from fitted nuisance estimation. Finite IB updates remain owned locally, with MI feature estimation identified as a different task. Divergence tests separate estimator conventions and statistical assumptions. Graph operators expose the package's different loop-normalization convention rather than hiding it.

Stochastic Processes appropriately reuses standard random primitives while owning how they form process laws; a second process package is not necessary. Queueing and Itô provide actual simulator/solver mappings with shared jobs or increments. Numerical PDEs improve the practical one-dimensional representation to banded assembly/factor reuse, while pointing to the already implemented sparse two-dimensional and triangle mechanisms. Cost and scope statements are bounded, and changed-input exercises include reasoned solutions. Existing diagrams/labs are retained; these additions are implementation routes, not replacement generic labs.

Final body, program and existing-owner bindings are in `math-depth-remediation-independent-sources.json`. Browser checks must still establish reachable disclosures, readable code, actual downloads and responsive containment before phase completion.
