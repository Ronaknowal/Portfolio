# Measure Theory & Probability Spaces — independent review

Status: bounded independent content/model review passed after two author-applied wording corrections. This is separate from author implementation verification, browser QA, integration, and user acceptance.

Reviewed stable topic: `measure-theory-probability-spaces`, Mathematics Foundations position 23. The reviewer did not modify the lesson, models, examples, labs, or shared registration.

## Scope and findings

Read the complete lesson, twelve native example programs and their expected outputs, pure models, individual design, incoming topic note, and exact topic-inventory result. Reviewed the logical conditions in the limit, product, conditional-expectation, and Radon–Nikodym arguments. In particular, checked the incoming change-of-variables/Jacobian bridge rather than treating its picture as proof.

Two small but real completeness issues were sent to the author and corrected:

1. The dominated convergence statement and derivation now explicitly require a measurable limit `f`. Agreement almost everywhere alone does not guarantee that an arbitrary version is measurable on a non-complete sigma-algebra. This distinction matters because the lesson explicitly introduces Borel versus completed measures. The author now explains retaining a measurable version. Axler's **Measure, Integration & Real Analysis**, theorem 3.31, explicitly includes measurability of the limit; the author's current PDF was read at PDF page 106 (printed page 92). [Primary text](https://measure.axler.net/MIRA.pdf).
2. The uniqueness paragraph for conditional expectation now permits different values on null sets **provided the versions remain G-measurable**. Equality almost surely does not license arbitrary edits that break the information-measurability requirement already given in the definition.

Verified both corrections in the final reviewed body; no model or numerical semantics changed.

No further actionable defect was found in this bounded review. Specifically:

- Finite information cells, preimages and pushforward laws separate observability from probability correctly. Countable additivity is not extended to uncountable unions.
- The Vitali argument specifies countable rational translates, disjointness, coverage and bounded containment. Cantor finite-stage displays are distinguished from the infinite singular/atomless conclusion.
- Nonnegative integrals, signed integrability and the undefined infinity-minus-infinity case are separated. The dyadic simple approximation to `x²` has a valid monotone construction and explicit error bound.
- MCT's increasing-set argument, Fatou's tail infima, and DCT's finite-integral subtraction use the stated conditions. The moving spike is a genuine counterexample to exchanging pointwise limits and integrals without an additional condition.
- Product construction and Tonelli/Fubini use the stated sigma-finiteness, sign and absolute-integrability conditions. A product reference measure is distinguished from a product joint law/independence.
- The signed infinite array has iterated sums 0 and 1 and infinite absolute sum. The finite square and wider-rectangle traces use different shapes; they do not claim different orders change the sum of the same finite entries.
- The joint-density cell and diagonal coordinate transformation preserve mass. The equality between center density times cell area and exact mass is correctly restricted to the illustrated bilinear density/rectangular cell.
- Conditional expectation's event-integral definition, uniqueness, nested tower argument and L² projection identity are consistent. The extension from simple test factors uses Cauchy–Schwarz, and reduced average risk is not described as guaranteed improvement on every outcome.
- The stated finite-positive versus sigma-finite-positive Radon–Nikodym form supports both probability reweighting and the conditional-expectation construction. A pure Lebesgue density is not claimed for the mixed law.
- Changed practice values, weighted conditional risk, mixed-law atom inclusion, spike exponents and changed coordinate scaling agree with their explained solutions. Later analysis prerequisites are bridged locally rather than silently required.

## Independent numerical checks

Commands run from repository root:

```powershell
node scratch/export-measure-independent.mjs
scratch/lesson-tools/Scripts/python.exe scratch/measure-theory-independent-review/check.py
```

The exporter reads the actual JavaScript models and stores results. The Python reviewer check uses separately constructed cell-indicator matrices and weighted least-squares projection `A (AᵀWA)^+ AᵀW`; it does not call or reproduce the model's cell-mean implementation. It checks predictions only on positive-mass outcomes when a null-cell version is arbitrary.

Actual checks and results:

| Investigation | Independent evidence |
| --- | --- |
| Conditional means | 96 states: twelve changed signed outcome vectors, four information partitions, fair and zero-mass-sixth-outcome laws. Prediction agreement, idempotent projection, weighted self-adjointness, cell-residual orthogonality, mean preservation and variance decomposition all passed. |
| Tower property | Ten coarse/fine projection-operator identities passed across the two laws. |
| Missing nesting | For outcomes `[0,2,4,4,8,12]`, parity then pair conditioning produces `[5,5,5,5,5,5]`; direct pair conditioning gives `[1,1,4,4,10,10]`. This confirms the concrete limitation of removing the nesting condition. |
| Null-cell versions | Versions 0 and 99 on the zero-mass singleton preserve all reported risks, means and variances. |
| Jacobian/mass | 64 cells at four independently chosen scale pairs, including non-integer and irrational factors. SciPy adaptive double integration evaluates both original `4xy` and transformed `4uv/(a²b²)` densities: 128 integrals. All masses and displayed area-density products agree. Maximum before/after mass discrepancy: `8.326672684688674e-17`. |
| Signed finite arrays | Independently enumerated 12×12 and 12×13 arrays have totals 1 and 0, absolute totals 23 and 24. |

Maximum numerical discrepancy across all checks: `1.4210854715202004e-14` (tolerance `1e-10`). Full machine-readable evidence is `scratch/measure-theory-independent-review/results.json`; fixtures, source snapshot and check script are in the same directory. These finite checks can falsify examples and models; they do not prove the infinite-space theorems.

## Reviewed source identity and limits

After verifying the two wording corrections, SHA-256 values were:

| Source | SHA-256 |
| --- | --- |
| `src/learn/data/topics/measure-theory-probability-spaces.jsx` | `0f1fc496e0259136c7013a69a19c0ca9dd93cefdee2ad56f5e9c7c7901dc2c1d` |
| `src/learn/data/measure-theory-models.js` | `4ec09eb7b22b456f4e4157c096e2a1931109985ce9aebde4638fb8556eb3daa3` |
| `src/learn/data/measure-theory-examples.js` | `9b45f281c45121d50b5a444ad1b75bbec635fe84e7cf8a3633016d38ee00c75f` |

The initial source snapshot in `reviewed-source-hashes.json` predates the two wording refinements; model and example hashes are unchanged. The reviewer did not rerun the author's twelve programs or duplicate browser testing. The author separately owns native execution, desktop/mobile/keyboard behavior, ordinary-reading screenshots and final source freeze. Any later semantic change needs a targeted review of the affected claim; this document does not certify future edits.
