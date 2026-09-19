# CRF independent review — 19 September 2026

Reviewer: the Gaussian Processes implementer, who did not author CRF. Scope: the complete prepared CRF manuscript, visual specifications and design; all ten reader sections, figures, investigations, pure models and downloadable native program. Reviewed the recorded source/data provenance and author evidence. This is a complementary review, not a repeat of the author's full training campaign or a novice user study. The reviewer subsequently made the three bounded repairs below with the original author's agreement; that repair ownership is distinct from independence of the original review.

Status: **no remaining material finding in the reviewed scope**, subject to the increment owner's production integration. Exact final source hashes and four capture digests are in [the independent receipt](evidence/crf-independent.json). Prepared checkpoint files remain unchanged.

## Findings and closure

1. **P2: inconsistent numerical ties.** Full-path winners used the stated relative tolerance of 10⁻¹², but best predecessors used an absolute tolerance. With inputs `[16,16,16,16,16,16,15.999999999999,15.999999999999]`, all four complete paths were reported as tied while each predecessor cell showed only A. Both now use relative tolerance. The complementary verifier reproduces the formerly conflicting example and checks predecessor sets and full winners. This was a reachable disagreement, not a cosmetic change to rounded numbers.
2. **P2: solved inputs could be graded as predictions.** Both investigations initially displayed outputs for the exact default tuple they allowed the learner to predict. The packet calls for an edited input before commitment. Solved prose/figures remain before the investigations. Lab output now appears only after Apply/Explore and retires on edits. Numeric tuple keys reject formatting-only edits, default inputs and any tuple already revealed during that mounted investigation for a fresh graded commitment; ungraded exploration remains available. Return-to-input, reset, stale prediction and keyboard exploration were checked in both investigations. No numerical answer is rendered inside an uncommitted fresh investigation.
3. **P3: tie transfer wording.** “Create a tie” previously shared an unconditional instruction to predict different path sets, although a four-way tie can agree with the independent model. The transfer now asks the learner to decide whether the complete sets agree; the reverse-disagreement task retains its intended prediction.

The original CRF author reviewed and agreed with the final patch; concurrence is recorded in the implementation record. Its separate correction to the example-source import preserved the exact downloadable Python text and removed a dev-only public-asset import warning. The reviewer read the added parsed-string/download equality guard and ran it successfully; the independent source receipt was refreshed for that verifier-only change without repeating unchanged browser or native checks.

## Correctness assessment

- Read the score/partition derivation, forward/backward/Viterbi recurrences, node/edge marginals, marginal-versus-sequence decisions, expected-feature gradient and L2 convention. The displayed code uses previous-label rows/current-label columns consistently. The coefficient `.05 * ||theta||²` differentiates to `.1 * theta`; fixed-feature convexity retains its neural/latent exceptions.
- The author's actual native execution, 40 development predictions and 370-token selected test are reused for unchanged program/data. Source inspection confirms training-only vocabulary, sentence-level official splits, matched independent/chain features and final-test separation. No new fit or optional CRFsuite/PyTorch execution is claimed here.
- Independently enumerated all **3,905** BIO sequences of lengths 1–5 using a complete-path legality rule. Legal counts are compared with the masked forward normalizer; this expands the author's edge-level cases to full supports. Every legal/illegal status agrees.
- Independently reconstructed all **81** probabilities of an asymmetric three-label, four-position model from the manuscript's backward conditional-sampling formula, and compared them with normalized complete-path products. Every value agrees within 10⁻¹¹. This specifically checks the deeper sampling claim.
- Verified the allowed-input near-tie regression and the distinction between finite preferences and excluded support. Label bias is presented as private-denominator cancellation, not a universal performance ranking. The canonical [Sutton–McCallum tutorial](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf) was reopened for the conditional-model context; the packet's scoped primary-source research is retained.
- Changed practice values, gradients and confusion denominators agree with solutions. The feature-ablation task requires a measured result instead of supplying an invented ranking.

Commands actually run after repairs:

```text
node scripts/review-crf-independent.cjs
node scripts/verify-crf-models.mjs
PLAYWRIGHT_PACKAGE=<bundled Playwright> node scripts/review-crf-browser.cjs
```

The independent receipt has six complementary groups, including the enumerations above and two fresh-input interaction campaigns at 390 px. The affected author browser campaign passed its 18 groups and recaptured 16 images at 1440/390 px, with checks also at 320 px. The model/content verifier passes all five presets and 94 parsed formula strings. Passing assertions alone are not visual inspection.

## Learning-experience assessment

1. **Entry and progression:** the whole-name problem gives labels an immediate purpose. BIO and features precede arithmetic; the A/B chain removes incidental NLP complexity. First-pass, real-code and deeper routes are clear.
2. **Scoped cautions:** input availability, score/probability, legality, convexity and the small sample appear where the learner makes those decisions. The lesson does not begin with a defensive checklist.
3. **Concrete meaning:** four paths expose what competes for the common denominator. Real error inspection shows that token accuracy and complete-sentence success answer different questions.
4. **Investigations:** unary/pair edits answer a structural-decision question; local/global route factors answer an information-flow question. Distinct visual mechanisms include reversals, nulls, ties and transfer. The initial grading leak is closed.
5. **Visual communication:** inspected all four new phone captures—factor chain, neural flow, corrected near-tie trellis/ledger and changed-prior normalization. Pair connectors remain under output nodes; labels stay with words; bars share zero-based scales. The selected edge and dashed alternatives work without color alone. No clipped labels, crossing annotations or gold gradients were observed in these captures.
6. **Connections:** grouped path sums become prefix/suffix products; expected counts become training signals. The independent-classifier special case and GP continuation state what remains familiar and what changes.
7. **Code:** complete inference/training programs expose mechanisms. External neural/library routes are annotated and do not masquerade as executed results.
8. **Practice:** calculation changes inputs, update practice changes the gold edge, diagnosis changes information constraints and real data requires an unsolved ablation. Hints/solutions remain closed and distinct.
9. **Interaction:** checked commitment, hidden numerical outcomes, numeric-equivalent input return, reveal/edit/return, reset and keyboard exploration. Live phone overflow/runtime checks pass. Desktop and 320 px author checks remain bounded evidence. This is heuristic learning review, not proof of mastery or a real learner study.

Publication/route loading, generated metadata and phase-ledger binding remain the increment owner's work. User acceptance is not claimed.
