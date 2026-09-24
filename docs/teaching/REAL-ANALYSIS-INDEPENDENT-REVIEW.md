# Real Analysis — independent final review

Reviewed 11 September 2026 by the workflow author, independently of the lesson's root author. This is a complete final-source review, distinct from the preserved [design-only review](REAL-ANALYSIS-DESIGN-INDEPENDENT-REVIEW.md). It covers all six production sources, the full fourteen-section body and proofs, all fourteen changed practice groups, the thirteen actual complete programs, model contracts, labs, CSS, brief and [design](REAL-ANALYSIS-LESSON-DESIGN.md).

**Outcome: no unresolved material mathematical or teaching-contract finding.** Two narrow display changes were authorized by the root author and are verified below. The [independent evidence packet](evidence/real-analysis-independent-review.json) records the original author identity, amended final identity, actual reviewer checks and opened captures. The [original author packet](evidence/real-analysis-author-review.json) remains unchanged; a [durable original source archive](evidence/real-analysis-original-author-sources.json) preserves all six exact pre-amendment byte sequences. Publication, production integration and user acceptance remain separate.

## Source and proof review

The sequence foundation preserves the order of tolerance/index/later-term choices, strict boundaries and the distinction between boundedness and convergence. The supremum, monotone completeness, nested interval, subsequence, real Cauchy, extreme/intermediate value and uniform-continuity arguments retain their necessary assumptions. In particular, completeness is not inferred from rounded digits, and a rational Cauchy sequence is not asserted to have a rational limit.

The function-sequence sections distinguish a fixed point from a moving witness and an attained maximum from an unattained supremum. The triangle's integral, L1, squared L2 and supremum errors are different quantities; the exact shape and deliberately sampled estimate are labelled separately. Uniform integration uses a finite interval and establishes integrability of the limit. Differentiation requires uniform derivative control and a convergent base value. The radius/interior/endpoint distinctions, absolute versus conditional summation, and smooth-versus-analytic construction are sound.

The Bernstein argument uses actual positive weights, finite moments, a Lipschitz bound and a separately derived basis integral. The certificate is sufficient; failing it is not presented as proof of actual error. The optional probability section introduces its extra prerequisites, keeps shared coupling distinct from laws, and correctly establishes the typewriter counterexample. Its UI argument first obtains integrability of the probability limit, then truncates with a common cutoff before sending the index to infinity. The fourteen changed solutions were read as mathematical arguments, not inferred correct from their number.

The reviewer also inspected the supremum equivalence in [Lebl §6.1](https://www.jirka.org/ra/html/sec_puconv.html) and the bounded-interval derivative/base-value theorem and relevant interchange examples in [Lebl §6.2](https://www.jirka.org/ra/html/sec_liminter.html). The lesson's theorem hypotheses match those mechanisms. This was a focused source check, not a review of the entire book or the linked videos. The author's more extensive written/video inspection ledger remains separately attributed.

## Concrete findings and repairs

1. **Exact endpoint claim versus rounded display.** The original `CompletenessBracketLab` formatted endpoints to nine decimal places but called the displayed dyadics exact. At the ordinary slider setting of 16 bisections, the exact left endpoint is `1.4141998291015625`, whereas the old text displayed `1.414199829`. The model's bracket itself was correct. The lab now uses the exact terminating decimal at that dyadic precision, and explicitly labels squared decimal readouts as rounded. The exact width and internal endpoint-square invariant remain unchanged. Real 1440/390/320 checks assert both exact endpoint strings; the 320/390 images were opened and fit without horizontal page scrolling.
2. **Series endpoint label clarity.** The author flagged an ambiguous negative glyph in the old native-select screenshot. Its selected DOM value, negative function sum and conditional derivative readout already agreed; no arithmetic failure was found. Options now name `−1 (left endpoint)` and `+1 (right endpoint)`, with explicit interior labels. The reviewer checked keyboard movement, reset, selected value and numerical outputs, then captured the closed control. An initially open native popup was closed in the screenshot harness rather than treated as a lesson defect. The final 320/1440 images clearly identify the negative endpoint.

Only `RealAnalysisLabs.jsx` changed. Its final SHA-256 is `e171a1dbb2cc4c4d61db9d9528efe72b6bb22119cb34b2027643f9f63ae19e17`. The body, brief, models, examples and CSS remain byte-identical to the 07:20:48.539 UTC author freeze. The full original body and every program/output are preserved.

## Independent executable and browser evidence

Commands:

```text
node scripts/verify-real-analysis-independent.mjs
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-real-analysis-independent.py
node scripts/review-real-analysis-independent.cjs
```

The final complementary native run passed at **07:32:25 UTC**. It executed all thirteen delivered program strings and compared complete stdout. It then checked changed actual helpers against independently formulated references: 18 enclosures from integer square roots after clearing denominators, harmonic blocks from digamma differences, 35 polylog function tails and 30 separately bounded log derivative tails, 60 exact de Casteljau polynomial evaluations, exact third binomial moments/Jensen bounds, triangle mass and centroid, dyadic binary prefixes, and 20 capstone integrals computed by closed arithmetic progressions. Strict rational tails and changed signed geometric sums also passed. The packet contains the precise cases and counts; these finite checks support, rather than prove, the separately read infinite arguments.

The final reviewer Edge/Playwright run passed at **07:34:13 UTC**, with actual Space Grotesk and JetBrains Mono at 1440, 390 and 320 pixels. All nine investigations received meaningful changed-state checks; reset and selected keyboard changes were exercised. The exact Bernstein `3/16` readout was also checked against the actual plotted marker coordinates. All thirteen rendered programs, outputs and preceding visible questions matched the reviewed source. Sixteen equation blocks fit, page overflow and page errors were absent, and the changed capstone's hint/solution was exercised. This is an independent targeted three-width pass, not a relabelling of the author's larger exhaustive field-state run.

Sixteen final reviewer screenshots were actually opened: exact endpoint readouts, closed negative endpoint selectors, an ordinary desktop introduction, strict tail, compact power domain, missed triangle, paired derivative geometry, weighted polynomial, dyadic observer, both inline proof figures, changed capstone, exact output and resources. Their hashes and precise filenames are in the packet. The original author screenshot was inspected only as pre-amendment evidence and is identified separately. No claim is made to have opened every automatically generated capture.

## Limits and handoff

The original lesson had a planned baseline rather than a previous published body. This review does not certify every advanced analysis theorem, all arbitrary numerical inputs, all browsers or a learner study. The elementary first-pass route, optional probability prerequisites, current module order and specialist boundaries were preserved. Root retains production integration and final progress bookkeeping. No shared metadata, publication map, prerequisite graph or unrelated lesson was changed by this review.
