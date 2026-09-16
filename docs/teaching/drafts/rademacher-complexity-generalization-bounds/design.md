# Rademacher Complexity & Generalization Bounds — design and author record

Canonical ID: `rademacher-complexity-generalization-bounds`. Title unchanged. Module: `classical-ml`. Delivery: research and writing only; diagrams and labs are specified, not implemented. The immediate predecessor is Calibration & Conformal Prediction; the successor is ML Problem Formulation, Baselines & Data Leakage. The manuscript preserves that route without changing runtime order.

## Scope, baseline and conservation

Read the current teaching handoff, full teaching standard and design brief, ML domain strategy, code ownership, coordination and retention instructions earlier in this assignment. Read this topic's scoped inventory with `--topic rademacher-complexity-generalization-bounds --work content`. It has no preexisting destination note; the unassigned inbox has no relevant unresolved obligation. Read the full existing topic, including all code, plot arrays, framework comparisons, scaling claims, sources and exercises.

Original: `src/learn/data/topics/rademacher-complexity-generalization-bounds.jsx`, SHA256 `0a3d172ba8033d61842e5d17c20b3bca341ecdf6449d2631f3a23dd864ca5972`, baseline commit `8c5da59f18516be77c29d5aeeafca3decca4f738`. No original production code, route, shared ledger or inventory was edited. Root owns reconciliation and the eventual content checkpoint.

| Existing strand | Prepared disposition |
|---|---|
| Why sample-dependent capacity is useful | Preserved; start with a complete three-input noise game instead of a long history before intuition |
| Empirical and expected complexity | Both retained; explicit no-absolute-value, 1/n convention and independent sources of randomness |
| Uniform bounds, symmetrization and McDiarmid | Rebuilt with the correct loss class, constants and full explanation of each inequality; actual bounded finite-population author probe |
| Contraction, finite and norm bounds | Preserved and expanded with exact loss domains, finite-vector radius, Euclidean/L1/kernel derivations and fixed-intercept treatment |
| Margin theory and SVM connection | Full ramp construction, errors versus margins, scale invariance and selection accounting; no false identification of a C sweep with a computed certificate |
| From-scratch linear and threshold estimates | Replaced with complete executable programs, exact enumeration, threshold endpoints and ties, finite Monte Carlo upper correction and inner-optimization distinction |
| Synthetic learning curves and SVC code | Removed unverified stdout and mismatched train/test simulation; replace with real offline constrained-logistic models under explicit data roles. The mathematical norm budget is directly controlled rather than guessed from C or support-vector counts |
| Neural, PAC-Bayes and stability scope | Preserved as explained optional branches with actual primary theorem conditions, correct prior-selection accounting and no unsupported 2026 universal ranking |
| Visual traces and comparisons | Retain the idea of visual proof and measured comparisons, but replace generic score heatmaps with actual best responses, signed geometry, margins and real evidence |
| Sources and practice | Annotated primary notes, papers and official course recordings; eight changed independent exercises with closed hints and solutions |

Material corrections include: predictor complexity substituted for an arbitrary loss class; confusion between empirical and expected theorem constants; incorrectly attributed theorem number; incorrect 2/n bounded-difference argument for [0,1] losses; an extra ghost-sample factor; calling logistic loss 1/4-Lipschitz; squared-loss domain/range omissions; claiming contraction always reduces complexity; silently taking an absolute value in the threshold estimator; missing threshold endpoints; treating Gaussian expected energy as a deterministic sample-maximum bound; labeling a finite noisy test gap “true”; changing the target-generating rule between train and test; claiming observed curves were monotone when their numbers were not; guaranteed overfitting from high class capacity; unbounded kernel/intercept claims; unsupported universal framework ratings; arbitrary data-dependent PAC-Bayes priors; and expected stability statements presented as universal high-probability certificates.

The current title covers the useful additions: L1 geometry, convex hulls, sample-dependent class selection and the distinction between estimation and certification are part of this topic. Full neural architecture training, advanced PAC-Bayes optimization, local-complexity fixed-point proofs and dependent-data theory remain explicit deeper routes. No new curriculum topic or title expansion is needed.

## Learner model and explanatory design

The learner may remember PAC/VC definitions but need local refreshers for dot products, supremum, expectation, norms, loss composition and margins. Start with an operation they can perform without those terms: choose the best matching prediction row for one coin pattern. Introduce the symbols only after computing the complete example.

Observable outcomes are to compute exact small complexity; distinguish max-then-average from average-then-max; preserve conventions; construct the loss class; explain the uniform theorem's data and range assumptions; derive a closed-form norm-ball best response; use finite, kernel and L1 bounds; calculate a bounded margin loss; account for predeclared selection and Monte Carlo failure probabilities; and diagnose a misleading certificate using actual evidence.

| Hurdle | Treatment and observable action |
|---|---|
| A supremum feels abstract | Editable prediction matrix; identify and justify a winning row for each sign pattern |
| Different randomness gets conflated | Separate sample draw, ghost sample and sign-draw lanes; distinguish empirical from expected complexity |
| Predictor quality is mistaken for capacity | Singleton-zero example and rich-class counterargument; retain the training loss term |
| The loss disappears in a theorem | Convert binary predictions to actual mistake vectors; exact factor1/2 calculation |
| The proof looks like symbol manipulation | Pair swapping and the split from one shared winner into two independent winners |
| Norm bounds seem unrelated to data | Signed vector sum and supporting point; same-energy duplicate/perpendicular contrast |
| Margins look like arbitrary scores | Ramp loss and simultaneous scale null; explicit B/rho and data-role accounting |
| A Monte Carlo result is treated as a guarantee | Checked finite estimate trajectories, one-sided correction and the wrong direction of imperfect optimization |
| Theory is forced onto a familiar empirical curve | Real fixed budget sweep where validation improves and all simple bound expressions remain vacuous |

Four focused investigations are specified, with static explanations at first use: best-response construction, signed geometry, editable margin evidence, and a real predictor workbench. The real workbench allows bounded coefficient edits and actual row calculations; it does not require a costly live optimizer or repeat a generic text panel. The manuscript's first eight sections form the core route; optional advanced sections are clearly separated. Every numerical exercise changes the worked inputs and has a closed hint and solution.

## Canonical coverage map

The scope comes from actual primary section lists, not a keyword checklist.

| Reference structure inspected | Coverage decision |
|---|---|
| Mohri lecture: empirical/expected complexity; two generalization bounds; proof; binary loss corollary; growth function; VC dimension; lower bounds | Sections2–4 own the exact conventions and guarantees. Section5 connects finite restrictions to VC. PAC/VC already owns the complete lower-bound teaching; refresh locally rather than duplicate an entire preceding topic |
| Understanding Machine Learning chapter26: complexity, calculus, linear classes, SVM, low-L1 predictors, exercises; chapter27 covering numbers | Core sections2–6 cover all chapter26 mechanism categories, including L1. Section9 introduces covering/chaining and points to the full proof route |
| Bartlett–Mendelson: definitions/Gaussian comparison, risk bounds, structural combinations, trees/networks/kernels | Preserve the broader context, convention difference, convex mixtures and kernel calculation; no unverified formula copied across normalizations |
| Oxford lecture3: contraction; L2/L2 and L1/L-infinity predictors; further regression examples | Use the exact norm and coordinate assumptions for the geometric comparisons, and provide notes plus official recording route |
| Neyshabur2015: group norms, magnitude control and width/depth dependence, path regularization, implications | Optional neural branch names the actual kinds of restrictions and their limits; no replacement with an unsupported product-of-norms slogan |
| Spectral-margin2017: normalized complexity and multiclass margins, empirical case studies, analysis | Explain why norm and margin normalization matter; link exact research theorem rather than inventing a universal computable certificate |
| Dziugaite/Roy2017: PAC-Bayes, prior choice, posterior optimization, final Monte Carlo correction, experiments | Optional branch preserves stochastic-predictor and prior-selection distinctions; no claim of having trained their networks |
| Hardt/Recht/Singer2016: stability definitions, convex and nonconvex SGD, stability-inducing operations | State the convex assumptions and sum-of-step-sizes bound; distinguish nonconvex conditions and expected from high-probability claims |

## Research record and claim locators

Research was performed on 12 September 2026. The following lists what was actually inspected. It does not imply complete playback or reading every appendix.

1. [Mohri's official lecture PDF](https://cs.nyu.edu/~mohri/mls/lecture_3.pdf): title/lecture structure; slides4–11 for definition, empirical/expected versions, sample replacement, pair-swap proof and binary loss corollary; growth/VC connection in the same lecture. Supplies the exact theorem convention in sections2–4. The original JMLR theorem1 is a VC result, so the old attribution was removed.
2. [Understanding Machine Learning, author PDF](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf): full table-of-contents entries for chapters26–27; chapter26 definition/proof and calculus excerpts, lemmas26.6–26.11, coordinatewise contraction proof, Euclidean expansion, L1 derivation and SVM setup around printed pages375–386. Used for core mechanisms. Did not claim a full449-page read. Its centered finite-vector Massart form also supports the simpler valid radius bound derived here.
3. [Bartlett and Mendelson2002](https://jmlr.org/papers/volume3/bartlett02a/bartlett02a.pdf): abstract/introduction, definition2 with 2/n and absolute value, theorem5/6 context, theorem7 margin setup; structural theorem12 and its contraction/translation conventions; kernel theorem21 setup on page13. Used to prevent mixing normalizations and support optional structural branches. Did not claim all appendix proofs independently verified.
4. [Vatsal Sharan lecture3](https://vatsalsharan.github.io/fall23/lec3.pdf): final pages10–12, definition13, geometric picture and symmetrization lemma14; previous PAC discussion was read earlier in this range. Useful proof-sequence cross-check. Did not adopt its informal “will probably overfit” sentence as a theorem about every algorithm.
5. [Oxford official course page](https://www.stats.ox.ac.uk/~rebeschi/teaching/AFoL/22/): syllabus, prerequisites, source list and lecture2/3 resource links. [Lecture3 notes](https://www.stats.ox.ac.uk/~rebeschi/teaching/AFoL/22/material/lecture03.pdf): introduction/sections3.2–3.3, contraction and L2/L1 derivations through page3. Official Panopto targets are `37a26a46-56dc-40c9-b6f4-ac2c0098ea64` and `34d50f99-9151-4a03-b18a-ac2c0098ead0`; both tool fetches failed. **No video or transcript was watched, and public playback access was not established.** The learner-facing course link is a verified route to notes and listed recordings, qualified by accessibility rather than promised playback.
6. [Neyshabur/Tomioka/Srebro2015](https://proceedings.mlr.press/v40/Neyshabur15.pdf): publication page/abstract, theorem1 and corollary2 with width/depth/norm conditions on pages4–6, discussion of when width disappears and the lower-bound qualification; appendix lemma18's no-absolute contraction statement. The whole26-page proof appendix was not read or implemented.
7. [Bartlett/Foster/Telgarsky2017](https://papers.neurips.cc/paper/7204-spectrally-normalized-margin-bounds-for-neural-networks.pdf): abstract/contributions and normalized-margin explanation on pages1–3, research case-study scope. Used for a qualitative, explicitly scoped neural branch, not a reproduction of its numerical figure or full theorem calculation. Official NeurIPS presentation page was located, but no session video was watched.
8. [Dziugaite/Roy2017](https://arxiv.org/pdf/1703.11008): theorem/algorithm setup on page4; sections3.1–3.3 on prior mean/variance, discrete-family union bound, posterior optimization and separate Monte Carlo loss correction; table1 scope. No arbitrary same-sample prior or unverified current “best framework” claim retained; no networks trained.
9. [Hardt/Recht/Singer2016](https://proceedings.mlr.press/v48/hardt16.pdf): official abstract, smoothness/Lipschitz definitions, theorem3.7 with 2L²/n times sum of steps, proof conditions, theorem3.8's different nonconvex assumptions. Used section9; not recast as a universal high-probability2epsilon bound. No stability experiment claimed.
10. [Local Rademacher complexities](https://arxiv.org/abs/math/0508275): abstract and publication metadata inspected for the optional research route. The local fixed-point proof is not presented as verified or implemented. The lesson's short conceptual explanation is a pointer, not a claimed reproduction of a theorem.
11. [SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html): analytic Jacobian, inequality constraints, stopping criteria and result/multiplier contract. [StandardScaler API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): fitted mean/scale and transform behavior; actual installed1.9.1 usage executed. The author program computes its own stationarity residual and projected returned objective rather than treating solver success as a proof.
12. [UCI267](https://archive.ics.uci.edu/dataset/267/banknote+authentication): metadata, attribution/license and actual archive were inspected/retrieved earlier in this assigned range. This packet copies the exact retained subset, preserving source IDs; the data record states that reuse honestly. No fabricated new acquisition or real-world iid certification.

## Author computation and remaining work

`complexity_calculations.py` and `bounded_norm_experiment.py` are complete instructional programs. Both ran in the shared read-only Python runtime with existing packages. Small exact calculations check the convention, loss factor, finite restrictions, threshold orientations, translation/duplicate/convex/scaling nulls, signed-vector and Gram geometry, margin scale invariance, finite Monte Carlo corrections and changed exercises. A bounded two-point population probe checks the direction of the ghost/symmetrization inequalities.

The real program uses 80 disjoint representation-design rows, 240 coefficient-fit rows, 80 validation rows and 80 assessment rows in a declared finite-corpus experiment. It fixes five budgets/two margins before running and selects B by validation before computing assessment results. All measured outcomes, fitted weights and objective/feasibility/stationarity records are retained. The first run revealed only tiny solver-radius violations from floating point; a small inward projection now returns feasible vectors and the final output was regenerated. No candidate errors or rounded conclusions changed.

The final author pass read the full 7,678-word manuscript, both complete programs and all visual contracts. Exact calculations were rerun after finite-input guards; all five stored fits were reconstructed from the frozen CSV transformation and checked for feasible norms, margins, errors, log losses and both bound sums. The 480 source IDs and four disjoint roles, source hash, seven local links, balanced math delimiters and 16 closed practice hint/solution disclosures passed. Real coefficient-permutation and row-reversal nulls were checked, along with the zero-score tie case. The compact results are retained in `author-checks.json`. This is bounded author checking, not a formal independent review. Root performs scoped reconciliation before the content checkpoint.

Phase two must implement the full manuscript, inline figures, investigations and downloads; independently verify mechanisms, numerical outputs, source claims and current APIs; and perform browser/accessibility/responsiveness/performance checks. Preserve the actual class and data contracts, not just the plotted shapes. The real-data bound expressions remain diagnostics, not certificates of an unjustified iid deployment model. Native author runs do not mark implementation complete.

Retain all manuscript/specification/source/data/program/result files for that handoff. Delete only this packet's disposable import cache after author checks. No runtime assets, redundant downloaded archives or broad scratch cleanup were created here.

### Root content reconciliation

Read the full manuscript, visual contracts, provenance and design. The theoretical and recorded numerical distinctions were preserved. A focused exact-feature check prompted by AutoML's grouping policy found four duplicate feature pairs in the 480-row subset, including two fitting/validation overlaps and no assessment overlap. Section 8 and provenance now disclose the exact limitation and the latter records every source-ID pair. This repairs the distinction between disjoint source IDs and distinct feature vectors without changing the declared finite-corpus calculation, selection rule or measurements. No model was refitted for this documentation clarification. Formal independent review remains deferred.
