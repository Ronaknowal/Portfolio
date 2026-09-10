# Entropy, Cross-Entropy & KL Divergence — author verification

10 September 2026. Stable topic `entropy-cross-entropy-kl-divergence`, Mathematics position 26. Complete source is installed and author verification passes. Root owns independent review, curriculum integration and production-build evidence; this record does not assert those stages or user acceptance.

## Reviewed scope and conservation

Read the original full body, current authoring/engineering standards, exact topic plan and both incoming destination notes. The title, route and module position are retained. [Design and reviewed research](ENTROPY-CROSS-ENTROPY-KL-DESIGN.md) records the teaching choices and actual resource-review bounds. Both incoming notes are [resolved](topic-notes/entropy-cross-entropy-kl-divergence.md): the inconsistent accuracy exercise is repaired, and finite moment-constrained maximum entropy is derived and investigated with a nonuniform-base counterexample.

The original source is retained at `scratch/entropy-authoring/original-lesson.jsx`. Both original complete Python programs are preserved byte-for-byte inside the eleven-program collection, independently checked by the verifier. Useful old entropy, cross-entropy, KL, likelihood, confidence, calibration/perplexity and alternative-divergence coverage remains and is expanded with actual code lengths, support conventions, a KL proof, conditional prediction and density/coordinate caveats.

The body has ten concept sections, seven distinct interactive investigations, prefix-tree and per-observation-loss inline figures, two early checkpoints and nine independent changed-input practice tasks. Hints and explained solutions are separate reveals. The next-topic bridge remains Mutual Information; prior Huffman construction is linked to the existing Greedy Algorithms §6, verified by a scoped source read. The figures and programs are analytic finite examples, not empirical performance results.

## Numerical and complete-program checks

Command: `node scripts/verify-entropy-information.mjs`, which invokes `scripts/verify-entropy-information-native.py` through the isolated lesson Python runtime. Actual result: [`scratch/entropy-verification/native-results.json`](../../scratch/entropy-verification/native-results.json), passed **2026-09-10T16:57:03.428993Z**. Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and Torch 2.14.0+cpu were recorded by the executing process.

| Scope | Executed independent checks |
| --- | --- |
| Complete examples | All 11 standalone programs executed; actual stdout matches every expected output. The two original programs pass byte-conservation checks. |
| Finite entropy and mismatch | 1,600 weighted P/Q states against SciPy entropy, `rel_entr` and `xlogy`, including zeros, reversed support, signed terms and both units. |
| Binary uncertainty | 101 selected probability states and their analytic curve values. |
| Prefix mechanism | 1,734 decoding positions against independently derived cumulative word boundaries, plus changed native codec/invalid-code cases. |
| Conditional prediction | 408 states against independently enumerated Fraction joint probabilities and conditional loss calculations. |
| Stable logits | 84 changed score-gap/offset/target states compared with actual Torch CPU log-softmax/cross-entropy. |
| Continuous coordinates | 1,440 states, with interval and density/KL quadratures; the complete suite performs 5,841 independent quadratures. |
| Maximum entropy | 882 feasible/boundary states, with 840 independent moment roots and scalar entropy optimizers. |
| Near equality | 40 comparisons using 70-digit mpmath arithmetic to check the stable nonnegative KL accumulation. |
| Input contracts and native helpers | 36 rejected model inputs; 991 additional changed native-helper cases including alternative valid codes, ambiguous/incomplete streams, varied-dimensional logits, 81 Normal KL integrals, triangle failure and all changed practice calculations. |

The pure model does not take the absolute value of signed KL rows. Its total uses the equivalent nonnegative integrand `p ln(p/q) − p + q`, exploiting cancellation of the linear terms across normalized distributions; a local series avoids near-equality cancellation. Very small positive floating-point residuals remain visible rather than being labeled exact mathematical zero. Support mismatch remains infinity. These checks cover the documented bounded controls and chosen native contracts, not arbitrary-range floating-point safety.

New Python examples were formatted with Black only after normalized AST comparison; originals were not reformatted. Babel formatting of owned model/body/labs and verifier scripts preserved normalized AST including literal and JSX text values. CSS formatting preserved its parsed selectors/declarations and ordering. Records: [`formatting-results.json`](../../scratch/entropy-verification/formatting-results.json), [`verifier-formatting-results.json`](../../scratch/entropy-verification/verifier-formatting-results.json).

## Actual browser and visual review

Command: `node scripts/review-entropy-information-lesson.cjs`, Edge headless via the existing Playwright package and local Vite route. Full result [`scratch/entropy-browser/results.json`](../../scratch/entropy-browser/results.json) passed **2026-09-10T17:14:27.222Z**.

| Viewport | Changed lab states | Rendered programs / checkpoints / practice | Anchors / displayed equations | Keyboard controls / scrolling tables |
| --- | --- | --- | --- | --- |
| 1440 | 200 | 11 / 2 / 9 | 10 / 14 | 34 / 0 |
| 390 | 200 | 11 / 2 / 9 | 10 / 14 | 34 / 2 |
| 320 | Reading and keyboard review; full state sweep at the larger two widths | 11 / 2 / 9 | 10 / 14 | 34 / 3 |

Actual checks include applied/invalid draft preservation, codebook changes and all decoding steps, source/model matching and support zeros, noise/model-confidence combinations, extreme finite logits, unit/bin changes, constrained means/interior positions/boundaries, resets, visible questions, complete code and stdout, keyboard-opened nonempty hints/solutions, focus and 44-pixel controls, keyboard table scrolling, every in-lesson anchor, KaTeX parsing, formula and paragraph fit, SVG label bounds and document overflow. All three widths had zero page errors, console warnings/errors and failed requests. Document widths equal viewport widths.

The local sandbox initially denied the site's existing public font request. The final browser run used the approved network-capable execution of the same local test; no failure was filtered from the final result. A too-early keyboard-scroll assertion was changed to wait for actual scroll movement. The actual product fixes found during review were four wrapped narrow equations, two unbroken arithmetic expressions, a distinct initial maximum-entropy selection and the tiny-total-KL cancellation repair.

Final visual polish changed only the narrow logit pipeline to a downward sequence and spaced the classifier captions. Targeted `node scripts/review-entropy-pipeline-polish.cjs` passed at 1440/390/320; [`pipeline-polish-results.json`](../../scratch/entropy-browser/pipeline-polish-results.json) records actual layout direction, geometry and zero errors. `logits-final-*` and `classifier-final-*` are the final screenshots for those regions. The full numerical sweep was not redundantly repeated for this CSS/caption-only amendment.

Screenshots were actually opened, not judged solely from geometry: every ordinary section opening `reading-1` through `reading-10` at 390 (section 4 reopened after its final arithmetic wrapping), ordinary sections 2/8 at 1440 and sections 4/5 at 320; all seven changed lab mechanisms at 390; prefix-tree at 320, classifier at 390, sources at 390 and final changed-mean practice at 320. The final downward pipeline at 320 and classifier caption at 390 were also opened. Earlier desktop mismatch, prefix decoding and maximum-entropy screenshots were opened. Ordinary reading captures retain the fixed navigation; isolated long lab captures temporarily hide it solely for unobscured evidence, then restore it. Data tables may scroll locally and are keyboard-operable; the whole page does not scroll horizontally. No actual first-time learner study or full screen-reader session is claimed.

## Research and review limits

Primary MIT notes, Stanford maximum-entropy notes and current PyTorch contracts were inspected within the exact scope in the design ledger. The alternate MIT 6.02 recording was verified against its official page/metadata and substantive companion slide content. It was not watched in full, and no full transcript was obtained. The learner-facing annotation states this. Wainwright/Jordan and MacKay access failures are not represented as completed research. No external source was copied as the lesson's structure or wording.

The finite prefix theorem is not a promise about every message or omitted file-format overhead. Conditional population bounds do not guarantee each finite sample. Density entropy uses a stated coordinate/reference; measure KL is distinct from empirical negative log density. The maximum-entropy proof is finite with explicit positivity, feasibility and boundary conditions. These scope limits are taught in the body, not only recorded here.

Final semantic fingerprints and timestamp: [`scratch/entropy-verification/final-source-hashes.json`](../../scratch/entropy-verification/final-source-hashes.json). Root should use that snapshot for subsequent independent review/integration. No shared registry, manifest, progress ledger, generator or global teaching document was edited by this author.

## Post-freeze independent-review amendment

The independent reviewer found two accepted arithmetic-range defects after the initial author freeze. A positive weight `1e-322` alongside 100 could normalize to zero, incorrectly presenting finite mathematical mismatch as infinite support failure. A strictly interior mean `1e-200` could underflow a positive maximum-entropy candidate probability, likewise producing a false infinite gap. The defects were outside the original slider sweep; the weight case was reachable through the editable scientific-notation draft.

The repaired weight normalizer rejects positive inputs that become zero during normalization. The draft parser also rejects nonzero text such as `1e-9999` that becomes zero during number conversion. An intentional exact zero still produces the intended support infinity, and the changed `1e-310` case stays finite. Errors preserve the previously applied distributions. The maximum-entropy numerical helper now explicitly accepts exact endpoint means 0/2 and interior means within [0.000001, 1.999999]; closer interior requests reject before calculation. Every existing selector choice is preserved. The calculator explains its arithmetic range separately from the unrestricted finite mathematical certificate.

The full native suite was rerun, passing **2026-09-10T17:30:28.721006Z**. Its current result has **1,601** information states, **924** maximum-entropy states, **882** independently solved roots/optimizers and **42** invalid inputs; all other counts and complete-program checks above remain the same. The independent constrained optimizer was reparameterized onto [0,1] so its absolute position tolerance would not swallow a tiny feasible interval; the numerical comparison tolerance was not loosened.

Focused actual-browser command `node scripts/review-entropy-numeric-boundaries.cjs` passed **2026-09-10T17:31:42.698Z** at 1440/390/320. [`numeric-boundary-results.json`](../../scratch/entropy-browser/numeric-boundary-results.json) records the two rejected drafts and preserved values, an accepted tiny finite draft, intentional-zero infinity, 18 existing maximum-control combinations per width, and actual-browser module checks for four rejected/four accepted range cases. There were no page/console/request errors or document overflow. All six `numeric-error-*` and `maximum-range-final-*` screenshots were opened and inspected: the error is readable, prior finite probabilities remain visible, and the range explanation fits at each width. This focused regression complements the earlier full state sweep rather than claiming an unnecessary second full browser run.

The final fingerprint file retains the previous author-freeze snapshot and identifies the revised model/lab hashes. The original body, examples, CSS and blueprint are unchanged by this amendment. Independent closure is recorded separately in the reviewer's entropy review record; root integration should use the revised freeze.

Independent closure was confirmed against all six revised fingerprints in [ENTROPY-INDEPENDENT-REVIEW.md](ENTROPY-INDEPENDENT-REVIEW.md), with durable evidence at `docs/teaching/evidence/entropy-independent-review.json`. Its complementary checks include 21 90-digit Decimal KL cases, 32 maximum-entropy states including accepted range edges, 606 nonuniform-base certificates, 780 changed five-symbol messages, three Normal quadratures and the unequal-context practice. Both reported numerical defects are closed; the reviewer reports no unresolved material issue. Those checks are separately attributed and were not counted as the author's own execution above.
