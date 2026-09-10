# Causal Inference & Do-Calculus — bounded independent review

10 September 2026. Read-only source review by a second author. No actionable mathematical or teaching-contract defect was found in the inspected scope. This is a finite cross-check, not a proof that every implementation input or every causal inference extension is covered.

The author confirmed the final freeze at **2026-09-10T17:25:46.682374Z** in `docs/teaching/evidence/causal-inference-author-review.json`. The independently inspected body, model and example hashes match that freeze: body `cc99f7d4ffdce7e7fa30c47e12454bb56354002a858cfa234c64ea004e61bf9d`, model `25b4f62b7a2ef3f426be665354f8e0536be7bd9c1541c4110cf6c1a890255245`, examples `37a96b8a41c67183059f1ab3a0c5aca5819cffaaa38a0d7bc650bc185de02401`. Full reviewed semantic fingerprints accompany the independent result below. No production or shared source was edited by this reviewer.

## Inspected reasoning

Read the scoped design; the body’s mechanism/truncated-factor assumptions, path/selection reasoning, backdoor and potential-outcome derivation, observational equivalence, all three do-calculus rules, frontdoor derivation, paired-world counterfactuals, nuisance correction, IV interpretation and changed practice solutions. Read the core pure models in full and the corresponding complete Python mechanisms where relevant. The separate author record owns execution of all ten learner programs and browser coverage; these were not claimed or redundantly repeated by this reviewer.

Cross-checked the rule definitions and identification distinction against [Pearl, The Mathematics of Causal Inference](https://ftp.cs.ucla.edu/pub/stat_ser/r416-reprint.pdf), particularly definitions 1–2 and section 3.1. The parsed PDF loses bar/underline styling; the source body explicitly spells out both operations, and the independent numerical oracle below tests the actual resulting probability equalities. No full-paper visual review is claimed.

- Rule 1 cuts existing action inputs only; rule 2 additionally cuts the exchanged variable’s outputs for its diagnostic graph. The source distinguishes that diagnostic operation from a real intervention.
- Rule 3 determines `Z(W)` using ancestors **after** the existing X intervention cuts. It preserves inputs to action nodes that remain ancestors of conditioned W. Its downstream-selection counterexample correctly keeps the selection event and yields unequal conditional probabilities.
- The frontdoor derivation uses the original population treatment weights in its inner mixture. It states all three graphical requirements and observed support. Its direct-path counterexample compares an invalid unchanged formula with the true intervention rather than calling finer estimation a repair.
- All measured X/Y cells can be positive in the nonidentification example while hidden mechanism values remain unconstrained observationally. The two models share the same causal graph and measured law and produce different do risks.
- The paired-world calculation conditions a single shared response type. Experimental marginals do not determine the coupling. Outcome monotonicity and IV receipt monotonicity are correctly distinguished.
- Augmentation is a population nuisance identity under identifying assumptions and support, not immunity to hidden confounding or automatic finite-sample inference. IV exclusion/relevance/randomization/monotonicity justify a complier effect, which is explicitly distinguished from population ATE.

## Complementary numerical evidence

Command: `node scratch/review-causal-independent.mjs`. Actual result [`scratch/causal-independent-review/results.json`](../../scratch/causal-independent-review/results.json), passed **2026-09-10T17:24:45.603Z**.

The reviewer’s oracle enumerates the full binary structural-factor distribution and applies intervention equation replacements directly. It does not reuse the production path-classification algorithm or the author’s ancestor-moralization oracle. Three distinct positive conditional-table parameterizations are used with changed action and selection values.

- **390 graph/rule conditions:** 2,970 equalities asserted for rules whose graph tests pass; 185 failed tests exhibit an explicit numerical counterexample among these parameterizations. A failed graph test need not force inequality in every parameterization.
- **Two extra ancestry cases:** in Y→Z→X→W, an existing do(X) removes Z’s ancestry of W, making Z eligible; without that existing action it stays ineligible. These test the order of graph modifications rather than only the simple downstream fixture.
- **108 frontdoor states:** independent closed-form do risks and proposed formula, including mediator probabilities 0/1, missing required cells, unused zero-weight terms and direct-path magnitudes 0/.05/.1. Unsupported factors remain unavailable; exact absent paths remain distinct from zero sampling noise.
- **96 augmented expectation states:** closed-form conditional residual expectations under varied true/estimated propensities and target shares, including zero-population strata and actual overlap failures.
- **40 counterfactual states:** independently constructed 2×2 response coupling, selected evidence and cross-world success probabilities.
- Verified identical positive measured ambiguity tables with effects .7 and −.1; four explicit invalid-boundary requests reject.

All comparisons pass. This review found no source correction to request. Actual intended-font desktop/mobile screenshots, keyboard behavior, complete-program stdout and production integration remain the separately attributed responsibilities documented in [the author verification](CAUSAL-INFERENCE-VERIFICATION.md) and root integration evidence.
