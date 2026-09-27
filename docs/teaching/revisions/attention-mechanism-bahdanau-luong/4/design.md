# Attention: intuition and reading progression, revision 4

26 September 2026. Full implementation revision requested after the user found the three recently implemented lessons difficult to follow. This revision concerns Attention, Long Context and State Space; it does not authorize implementing the next topic.

## Diagnosis and intended result

Revision 3 established numerical and runtime correctness, but its opening demanded knowledge of the very entities it was about to introduce. Definitions preceded a reason to use them, the first lab exposed unexplained training loss, and the linear-cancellation proof interrupted the first complete read. Several diagrams named components without establishing the problem the wiring solves.

The new route follows a familiar spelling decision, saved reading notes, retrieval versus returned information, one exact weighted read, the decoder loop, an executable core, learning feedback, and an actual trained experiment. Advanced scorer diagnosis comes after a complete decoder operation. The full derivative follows its signed interpretation. Local, copying and speech branches retain depth while explaining the problem before the formula. The original data, complete training/inference programs, all seed results, limitations and eight changed exercises remain.

## Research consulted

These sources inform teaching choices, not copied language, layouts or invented correctness guarantees:

- [Olah and Carter, Distill (2016)](https://distill.pub/2016/augmented-rnns/): read the Attentional Interfaces discussion and surrounding mechanism explanation. Its whole-task-to-memory-read progression informed the opening wiring comparison. This is a classic, not a recent implementation guide.
- [D2L 1.0.3, Queries, Keys, and Values](https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html): read retrieval and pooling definitions. Adopt the teaching distinction between an address and returned information. Use an independently constructed weather example, explicitly labeling manually chosen mixture shares.
- [3Blue1Brown, Attention in transformers (7 April 2024)](https://www.3blue1brown.com/lessons/attention/): inspected the article's motivating examples and query/key/weight/value diagram explanations. The useful principle is desired behavior before matrix notation, with stable entities followed through the operation. Its self-attention scope differs from recurrent attention. Do not import its illustrative linguistic labels as empirical features or its occasional informal claims as proofs. The accompanying full video was not watched.
- Existing Bahdanau/Luong/pointer-generator/speech primary references remain the source of the technical architecture definitions. Their formulas and historical reproduction limits remain intact. The independent reviewer also consulted Alammar's original seq2seq explainer; see the independent review for exactly what that reviewer inspected.

Recentness is useful for current APIs and accessible alternate resources, not a reason to discard a clearer classic explanation. The lesson's resources identify the scope of each and what was actually inspected.

## Content and implementation contract

The complete current manuscript is [lesson.md](lesson.md); [visual-specifications.md](visual-specifications.md) describes the new explanatory representations and revised first lab. The original revision-3 packet remains unchanged under `docs/teaching/drafts/attention-mechanism-bahdanau-luong/` and owns the canonical programs, data and measurements. Required unchanged programs remain in the new ledger content hash map; do not copy or retrain them merely because prose changed.

New runtime ownership: `RecurrentAttentionIntuition.jsx` and its topic-scoped CSS. Existing `RecurrentAttentionLabs.jsx` keeps all seven investigations. The initial read lab ends at the returned context; training probabilities, loss and validity controls appear in the later learning lab after their introduction. Heavy fitted weights and full programs remain fetched only on demand.

The short `attention-read.py` is extracted from the current manuscript by the existing topic generator. It makes the score–mask–normalize–read core visible before the larger library program, while retaining the complete scratch and native routes. Its one-query operation avoids a needless source-by-source matrix.

## Verification and review boundary

Reuse revision-3 numerical/native evidence only for byte-identical numerical engines, trained weights, data and native programs. Verify the new core Python program against independently calculated weights/context, changed masks and NumPy versus Torch. Check explanatory figure arithmetic, current manuscript anchors, retained practice, source identity and revised first-lab behavior. Inspect actual rendered figures at desktop, narrow reading-column and phone widths, including edited controls. A passing build is not teaching review.

The independent reviewer first diagnosed the old lesson; that baseline is not approval of this revision. Final review must read the revised body and drawings in order and record any remaining findings. No human beginner trial has been performed; learning-experience assessment is an author/reviewer heuristic.

Current state: both revision-4 phases complete. The independent full reading and bounded CSS follow-up are in [independent-teaching-review.md](independent-teaching-review.md); fresh arithmetic and executable-core results are in [author-checks.json](author-checks.json). The integration owner inspected all six new figures on desktop, checked container stacking at 760px and phone rendering at 320px, and exercised the changed read lab. The phone label-size and PAD wrapping findings were corrected and re-inspected. See the [shared integration record](../../../ATTENTION-MEMORY-INTUITION-REVISION.md) and [root browser receipt](../../../evidence/attention-memory-intuition/browser-review.json). User acceptance remains separate.
