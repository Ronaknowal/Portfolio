# Convex duality and KKT: bounded independent review

10 September 2026. Root read the complete new body, all ten standalone programs, all pure model functions and the six changed-input practice solutions. Author native/browser evidence remains in [the verification record](CONVEX-DUALITY-KKT-VERIFICATION.md); production integration is separate.

The mathematical review checked weak-duality signs and effective domains; the projection's exact dual and gap decomposition; all four KKT conditions and their sufficiency under convexity; Slater's strict-feasibility assumptions, dual attainment and the separate need for primal attainment; the degenerate min x subject to x²≤0 example; the stated nonconvex dual gap; multiplier sensitivity and nondifferentiable supporting slopes; resource-price projection and its smoothness/rate bound; and the small hard-margin SVM derivation. The written assumptions and six independent exercise explanations were consistent.

One actual display defect was found during review. A reachable resource-price state used subtraction of nearly equal objective/dual values and displayed a negative certificate gap. The author replaced this evaluation with its equivalent nonnegative decomposition: weighted local displacement squares, a boundary-gradient contribution and the multiplier times repair slack. The explanation and complete Python example now state the same calculation and floating-point limitation. This preserves a small positive gap rather than arbitrarily clamping subtraction to zero.

The targeted independent command `node scripts/review-duality-kkt-mathematics.mjs` passed on the frozen source at 12:35:52 UTC. [Its durable result](evidence/duality-kkt-independent-review.json) records:

- Three original quarter-step reproductions whose naive subtraction remains negative while the stable formula returns positive values, including budget2.75/rate1.75/state18 with stable gap approximately8.62e−18.
- Eleven independently derived active-face resource optima, spanning zero budget, the2.5 active-set boundary, interior allocations and inactive resource constraint.
- Three boundary-price gap calculations and a changed projection certificate.
- Exact source hashes for the reviewed body, pure model and examples.

No further actionable mathematical issue was found in this bounded review. It does not claim outward-rounded rigorous interval certificates, a beginner study, clinical conclusions, or duplicate every author browser/native case. Source hashes tie these observations to the actual reviewed version; new substantive changes require assessing the affected evidence.
