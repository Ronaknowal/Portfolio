# Second-Order Methods — author verification

10 September 2026. Stable ID `second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient`, Mathematics position 12. Complete author implementation and independent review; integration and user acceptance are separate. [Design](SECOND-ORDER-METHODS-DESIGN.md) records retained coverage and representation contracts.

## Teaching and source ownership

The original `100x²+y²` Newton example and useful warnings about local curvature, memory and comparisons are retained. The previous short method descriptions now have self-contained mechanisms: local quadratic minimization, positive-definite descent and line search; secant pairs, inverse BFGS and the two-loop product; a locally introduced two-outcome probability model, Fisher and KL; explicit affine-layer K-FAC factorization; and original Shampoo matrix contractions and spectral powers. The closing HVP/CG example separates matrix-free direction computation from a small dense validation oracle.

Eight conceptual sections contain six distinct investigations, two immediate inline figures, nine complete native programs and seven changed practice groups with initially closed hints and explained solutions. A worked changed-data L-BFGS/GD comparison completes the experimental practice. These are this lesson's choices, not quotas. Probability basics and the affine layer are introduced locally because their later modules cannot be assumed. The next lesson remains Non-Convex Optimization Landscape.

The six semantic owners are the stable-ID body and blueprint, `second-order-methods-models.js`, `second-order-methods-examples.js`, `SecondOrderMethodsLabs.jsx` and its named stylesheet. Browser work is bounded to matrices no larger than 4×4 and short deterministic traces; no Python, PyTorch, continuous training loop or benchmark payload enters the browser. Multiline example formatting preserves every code/output character, as recorded in `scratch/second-order-review/native/format-equivalence.json`.

## Native and independent mathematical evidence

`node scripts/verify-second-order-examples.mjs` materializes and runs every actual displayed program with isolated Python 3.12.14, NumPy 2.3.5 and CPU PyTorch 2.14.0. It compares full stdout, preserving results in `scratch/second-order-review/native/verification.json`. Each program includes its own imports, inputs and checks. Earlier captured runs remain in that directory.

- Exact and rotated quadratic solves retain the original subtraction convention and compare residuals with finite GD progress.
- The nonlinear Newton program distinguishes an indefinite raw direction, positive-definite damping and an accepted multiplier. The near-flat fixture rejects full steps and accepts α=.125; the lab plots actual objective versus the undamped local Taylor model along its explicitly chosen direction.
- Complete L-BFGS calculations compare memory budgets and reject a reversed curvature pair. The native nonlinear fit uses a real repeatable PyTorch closure and reevaluates the final loss rather than mislabelling the optimizer's returned initial loss.
- Bernoulli score expectations distinguish model Fisher from observed-label outer products. Direct probability and logit Euler endpoints differ even though their infinitesimal natural directions agree.
- Full joint-outcome enumeration checks K-FAC's exact Fisher; conditional dependence explains the factor approximation. Column-stacked vectorization, supervised gradient and factored versus full diagonal damping remain explicit.
- Three actual gradient matrices produce Shampoo accumulators and spectral inverse-quarter roots. Rotation equivariance is tested; the replay is not advertised as a comparison of training performance.
- The HVP/CG program uses a positive-definite six-parameter objective and checks its residual and descent. Dense Hessian construction appears only in the independent small verification step.
- Changed-data fitting uses a common objective/gradient evaluation budget and stopping criterion. L-BFGS reaches the gradient tolerance in 15 evaluations including the final check; the chosen GD rate uses 1000 and does not. The actual results and fixed-fixture limits are explained; no general speed ranking is claimed.

The separate [independent mathematical review](SECOND-ORDER-METHODS-INDEPENDENT-REVIEW.md) passed its exact/changed calculations and primary-source reading. Its executable checks use Hessian-form BFGS then inversion (16 cases), finite-difference joint score expectations (6), NumPy spectral powers and orthogonal transforms (9), Bernoulli geometry (6), matrix-power GD (4), Newton systems (4) and six changed closed-form practice groups. A separate follow-up reruns the seventh experiment and checks full-precision losses/gradients with an independently derived NumPy Jacobian. These bounded oracles and reviewed proofs do not claim arbitrary floating-point accuracy or general optimizer convergence.

The independent review found one genuine display defect: a probability near one rounded to exactly `1`. The probability-specific formatter now shows `1 − 2.639e-6`, including accessible labels, while retaining the complementary mass and finite KL. The before/after evidence and actual opened 390px keyboard reproduction are linked in the independent record. No scoped independent finding remains open.

## Rendered reading and interaction evidence

`node scripts/review-second-order-methods.cjs` checks the actual registered route in Edge at 1440, 390 and 320px. Final combined results are in `scratch/second-order-review/browser/results.json`, completed at 12:45:54 UTC. All eight reading links reach the correct headings; all nine full code/output strings render; all six investigations update/reset through meaningful states; all seven hint/solution pairs and early checkpoints work by keyboard. Reviewed states have no page errors, displayed-math overflow, page overflow or out-of-bounds SVG labels.

The six representations expose different mechanisms: equal-axis rotated contours and actual parameter paths; actual/Taylor loss with rejected line-search attempts; chronological secant history and vector transformations; paired probability masses and exact/local KL; shared-scale factor/full/error matrices; and row/column accumulators, inverse roots and transformed gradients. Immediate vector correspondence and row/column dot-product figures support ordinary prose instead of replacing it with six generic control boxes.

Actual images were opened and read: desktop curvature, K-FAC difference, rejected secant history and boundary probabilities; 320px K-FAC cells, rotated Shampoo matrices, safeguarded Newton and contraction figure; and 390px resources. All eight ordinary section openings were inspected across representative desktop/narrow captures. Keyboard and layout checks cover each of the three widths; this is not a claim to have watched a beginner or tested every assistive technology.

Reading review repaired an apostrophe-derived section anchor, a right-edge plot label and two long equations at narrow widths. The formulas now break into mathematical steps instead of shrinking text. Matrix cells have real row roles. Isolated tall-element screenshots may include the fixed site header crossing a capture; ordinary page screenshots retain that header and verify actual reading placement. No shared navigation change was made for this capture artifact.

## Research and remaining boundaries

The finalized design records inspected primary material: Boyd/Vandenberghe's local Newton discussion; UW's two-loop algorithms; Martens/Grosse's Fisher factorization and damping overview; Martens's local KL, finite-coordinate and empirical-Fisher distinctions; original Shampoo matrix/tensor algorithms; and the installed PyTorch version's closure/HVP documentation. Stanford's official Newton lecture page, outline and relevant transcript passages were reviewed as an alternative route; no full-video playback is claimed. Source links are annotated in the rendered references.

Final six source fingerprints are stored in `docs/teaching/evidence/second-order-methods-author-review.json`. The shared ledger must match these and the production build before integration status is granted. No deployment, universal benchmark result or user approval is claimed.
