# Attention revision 4: independent teaching audit

26 September 2026. Reviewer: `attention_review`, separate from the revision author.

**Status:** baseline audit, resource comparison and revision-4 independent source,
scientific and teaching review complete. Site visual verification belongs to the
integration owner's browser record; see the explicit review limit below. This is
not an implementation completion record. The user's criticism of intuition and readability is the reason
for this revision. The earlier numerical/coverage review does not establish that
a novice can construct the mechanism from the explanation.

Read the complete current lesson and its associated visual/lab implementation.
Baseline: `src/learn/data/topics/attention.jsx`, SHA-256
`5e0b7b4bcd52da5aae3c8857eefcb863454687f3dbf2121f2e5aca7f6e7c77f1`.
The audit changes no runtime, prepared manuscript, shared standard or ledger.

## What is already valuable

Preserve the exact weighted-read example, score-cancellation counterexample,
historical scorer/schedule distinction, mask and gradient calculations, explicit
scratch and library routes, real saved-model investigations, and honest experiment
results. The simple rules beating every neural fit is useful evidence. So are
alignment ambiguity, local-window normalization, copying repeated words, location
features, coverage and complete cost terms. A clearer route must not remove these
details or replace measured results with a more attractive fictional demonstration.

The principal problem is the distance between **a definition being present** and
**the reader understanding why the next operation follows**. Several technically
correct sections introduce abstractions, implementation policy and caveats before
completing one ordinary decoding decision. Some diagrams restate formulas rather
than adding the missing explanation. The remedy is causal buildup and useful
representations, not a word-count or diagram-count target.

## Educational resources actually inspected

These are three independent educational works, not templates to copy.

| Resource and inspected scope | Useful teaching method for this lesson | Limits of this inspection |
| --- | --- | --- |
| [Olah and Carter, Attention and Augmented Recurrent Neural Networks, Distill](https://distill.pub/2016/augmented-rnns/): introduction, memory/read explanation, attentional interfaces, translation/application captions. | Begin with the action the system needs, keep visible memory objects, then introduce a differentiable weighted read. Show the decoder consulting memory repeatedly rather than treating attention as an isolated formula. | Read text and captions; did not play the animations. Later adaptive-computation/programmer branches were not fully reviewed. |
| [D2L, Queries, Keys, and Values](https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html), with the core regression construction in [Attention Pooling by Similarity](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-pooling.html). | Ground the three roles in retrieval; contrast an exact selection, an equal average and a query-dependent mixture. Connect the normalized weights to the returned quantity before introducing architectural variants. | Read the retrieval/formula discussion, pooling construction and relevant plotting explanation. Did not execute notebooks or verify every framework rendition. |
| [Jay Alammar, Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/): complete substantive article text and figure captions. | Explain the whole task and the numerical objects before unrolling a single output step. Keep one visual vocabulary for encoder states, decoder state, read and output; reveal the scoring internals after their purpose is understood. | Embedded animations/videos were not played. Its illustrated updated-state query is not a reason to erase Bahdanau's different order or copy historical software recommendations. |

Source explanations still need scientific scrutiny. In particular, do not repeat
D2L's blanket assertion that a softmax gradient never vanishes: identical values
give zero score gradient through this read, and saturation can make derivatives
arbitrarily small or numerically zero. Our existing counterexample must survive.
The adaptation recommendations below are the reviewer's analysis of the local
lesson, informed by these resources; they are not claims that those sources tested
this site's teaching effectiveness.

## Concrete repairs, in learning order

| Current hurdle | Proposed repair and useful visual | Coverage to retain |
| --- | --- | --- |
| The opening lab-policy paragraph lists queries, keys/values, scorer parameters, masks and local windows before introducing them. | Open with the spelling task and one output decision. Move interaction guidance beside the relevant lab; let beginners discover controls in the same order as the mechanism. | Immediate playable controls and no learner-prediction submission or unlock gate. |
| The memory shelf introduces `h_j`, dimensions and two paths, but does not first make the bottleneck versus repeated access tangible. | Draw the same source and decoder twice: final summary initializes the decoder in both; only the attentive version also has a path from saved source states to each output step. Explicitly show what stays fixed and what changes. | Encoder information can already be lost; attention is not perfect retrieval or a guarantee of unlimited memory. |
| Query/key/value role definitions jump directly to arbitrary two-coordinate vectors. | Use an exact read and an equal average as contrasts, then a question-dependent mixture. Label the source of the query (decoder state), relevance features (keys) and returned information (values). Carry one example through every arrow. | Keys and values may originate in the same encoder vector; their numerical roles remain different. Avoid assigning invented linguistic meanings to learned coordinates. |
| Softmax appears as a completed formula, without showing why its parts are needed. | Add a tiny scores → positive masses → total → shares trace. State why negative scores are allowed, why division makes a comparable mixture, and why one changed score affects all shares. Show stable max subtraction as the same calculation. | Exact scores `1,0,-1`, normalized weights and signed value contributions; attention is not output confidence. |
| The first read lab already reports class probability and loss, although the output-head simplification is explained in section 5. | Keep the initial lab about reading: scores, weights, values and context. Introduce class logits/loss in the learning version with an explicit toy-head bridge. A value edit and a key edit should visibly demonstrate different effects. | Shared trustworthy calculation model; do not imply every real context coordinate is a class logit. |
| The scorer-cancellation proof interrupts before the learner has completed a decoder step; scorer alternatives precede a concrete full loop. | Complete a single read → decoder update → output decision first, then repeat it with a changed query and fixed source. Put cancellation alongside scorer design, after a working query-sensitive score. | Both historical schedules, BOS handling, previous token/state roles, input feeding and the full counterexample. |
| Additive projection notation and shapes are correct but the transformation is hard to visualize. | Show two unequal-width feature vectors entering separate learned maps, addition in a common space, `tanh`, then reduction to one scalar per source. Explain each arrow's purpose before listing dimensions. | Dot/general/additive distinctions, concat equivalence, biases as a declared choice and exact parameter counts. |
| The tensor table and three kinds of masking arrive as a compact catalogue. | First depict one short and one long source in a rectangular batch. Compare a zero PAD value admitted to the denominator with a PAD score excluded before normalization. Then map the picture to axes and the shape table. | Source packing, target-loss mask, output constraints, all-masked rejection and source-versus-output softmax axes. |
| The gradient section moves quickly from an artificial head to a derivative identity. | Explain that the desired class needs less first-coordinate and more second-coordinate support. Use the actual three values to show signed support relative to the current mixture, then derive score and query gradients. Distinguish an inference edit from parameter learning. | Exact local derivatives, finite step recomputation, the identical-values null case and no universal convergence promise. |
| The first visible runnable code is checkpoint loading; the readable core read is buried in larger programs. | Add a short, complete masked-read implementation beside a line-to-operation explanation, before the full saved-model/training route. Show score/softmax/source axis/value mix and the illegal-row policy. Connect that core to the larger model rather than creating a competing unexplained program. | Efficient numerical operations, declared array shapes, native-library parity, complete training/inference files and genuine reuse of prerequisites. |
| The real experiment begins with provenance/split machinery before explaining what the comparison asks. | Start with “can repeated reads help generate forms for unseen spellings?” Explain one example and one metric row, then the full protocol/results. Keep provenance and confounds available near the data rather than deleting them. | Full counts, seeds, rules, negative results, architecture confounds and the development-not-test distinction. |
| Alignment is displayed after several abstractions, and numerical maxima can become a reading exercise. | Use one real saved trace at two output steps: source memory fixed, query/read/output changed. Add a guide to row/column meaning before the full matrix, then use the ambiguity example to bound interpretation. | Actual weights and output probabilities, contextual states, diffuse EOS and the nonidentifiability counterexample. |
| Deeper branches have good initial motivations but quickly become equations. | Preserve copying's existing out-of-vocabulary problem. Show duplicate source positions merging into one word probability; for windows show missed evidence versus fewer reads; for speech show repeated content plus a previous-location cue. Narrate one concrete change before the general formula. | Unnormalized versus renormalized local-p, locality versus monotonicity/streaming, generator/copy mixture, location convolution, coverage and complete caching costs. |
| Practice starts with strong arithmetic/diagnostic tasks, but there is little confirmation of the complete basic mental model. | Add or adapt a short “follow one output and identify each object's origin” task and a minimal-read coding task before advanced diagnosis. Keep answers separate from live labs and preserve worked hints/solutions. | Current rigorous exercises and research modification practice, without compulsory guesses in investigations. |

## Recommended route, not a mandatory template

Concrete output need → fixed summary versus revisitable memory → one weighted read
→ one complete decoder step → scorer design and schedules → batch/mask mechanics
→ how the read learns → small scratch implementation and ordinary library use
→ real experiment and saved alignments → optional extensions → practice/readiness.

The author can retain useful section anchors and choose a different ordering if
the dependencies are explicit. Repeated concepts should use consistent symbols,
source tokens and visual objects; each deeper section should begin with the
question it answers and reconnect to the main read. Details can be progressively
disclosed, but essential explanation must not disappear into a collapsed code file.

## Follow-up review criteria established by the baseline audit

Review the completed revision as a learner before looking at its verification
record. Can the reader explain why weights are needed; where query, key and value
come from; which quantities are learned versus recomputed; what softmax normalizes;
and how a read affects one output? Can they map the visible small program to that
same mechanism and then use the normal library route?

Independently check new worked arithmetic and diagram arrows against the actual
calculation, inspect new representations in the browser at a useful desktop and
narrow width, and check that controls expose the promised cause/effect without
premature unexplained outputs. Reuse prior numerical evidence for unchanged native
models; do not repeat all six fits merely because explanatory prose changed.
Bind the final review to the actual changed files and record its limits. This
audit alone does not close those requirements or certify learner mastery.

## Revision-4 independent follow-up review

Read the revised complete teaching route, design/specifications, generated article,
`RecurrentAttentionIntuition.jsx`, its CSS, the revised first/learning lab split and
the generator's source mapping. The new opening earns the vocabulary through a
concrete task; the retrieval bridge distinguishes compared and returned information;
softmax has inspectable intermediates; the first lab ends at context; the visible
NumPy core precedes checkpoint-loading machinery; and learning now has a signed
interpretation before its derivative. Original scorer, schedule, tensor, training,
experimental, alignment, extension and practice coverage remains available.

One diagram finding was sent to the author: the new revisitable-notes drawing's
single amber read into `s1`, without a query path, could be read as a fixed pooled
summary. Its prose correctly says that each step chooses different shares. The
diagram should either depict that dependency/repetition or explicitly identify
itself as one illustrated read whose shares depend on the current writer. **Closed:**
the reviewed final source labels the connection “writer chooses this read” and
explicitly explains that only one read is drawn, the current writer determines its
shares, and the next step asks again with its new state. This prevents treating the
new connection as another fixed summary without pretending the schematic is a
complete unrolled computation.

Independent checks executed using the shipped `attention-read.py` and NumPy, with
explicit expected values rather than expected results copied from its implementation:

- Different key and value widths: keys have width 2 and values width 3. Scores
  `(0, log(3))` on the two valid rows, plus a masked distractor with extremely large
  key/value entries, give weights `(.25, .75, 0)` and context `(5, 2, 2)`.
- A common score shift of `+1000` leaves the finite read unchanged; an all-masked
  source raises the documented `ValueError`.
- A nontrivial interpretation check: with the lesson's three values, shares
  `(.2, .7, .1)` give context `(.3, 1.5)`, whereas `(.2, .1, .7)` give `(-.3, .9)`.
  Contexts differ, but both have margin `1.2` and class-2 probability
  `0.7685247834990175` under the illustrated identity-logit head. Thus the new margin
  explanation does not imply that equal class support means identical contexts,
  nor that this invariance holds under an arbitrary downstream projection.
- Independently calculated weather means and additive-scoring intermediates:
  uniform `21`, specified mixture `22.5`, and score `2*tanh(1.5)-tanh(0)` =
  `1.8102965072897328`.

All checks passed with tolerance `1e-12`. Reviewed core SHA-256:
`89962ed58c5f13c59961d7e7b850d31303f218bf7a3735d03b8b463891c4f2c3`.
No fits, broad native experiments or unrelated lesson audits were repeated.

### Conclusion, limits and source binding

No material source-level scientific or teaching finding remains open in this
revision's reviewed scope. The six new representations each address a distinct
learning hurdle, and the first lab now exposes the mechanism taught at its location.
The revised lesson preserves advanced detail while supplying a workable first-pass
route. This is a reasoned educational assessment, not evidence from a human learner
trial and not a promise that any reader will master the subject without practice.

**Browser limit:** after refreshing CUA documentation, creating a fresh tab in the
announced browser `1` returned “Browser is not available” to this reviewer. No
revision-4 rendered screenshot or control observation is claimed by this reviewer.
The author/integration owner must inspect the new figures, changed lab disclosure
and actual layout at the specified widths and retain that separate evidence before
implementation closure. Source geometry inspection is not substituted for that work.

Final reviewed files and SHA-256 bindings (integration records/ledger excluded):

| File | SHA-256 |
| --- | --- |
| `docs/teaching/revisions/attention-mechanism-bahdanau-luong/4/lesson.md` | `ba1816022f4fa4e13f819c45be08923652095a44de9852abebedd42dcefe19a7` |
| `docs/teaching/revisions/attention-mechanism-bahdanau-luong/4/visual-specifications.md` | `93860a15b54eabb4be0c8a04b8f392a1bd1b631789975c5c735d4aea35406ed0` |
| `src/learn/data/topics/attention.jsx` | `502bd1c2be9206d9b0b0931392fe7c980e08456c4cbb0581a4824b891a6e7f3a` |
| `src/learn/components/lesson-labs/RecurrentAttentionIntuition.jsx` | `92d08faa4b3d8a4cdd0d66fa9bbf1d1dece2d116ab8e11de236033b60ea68de4` |
| `src/learn/components/lesson-labs/recurrent-attention-intuition.css` | `2e1bde29045805987913851641a4dd0ba8585ca2f6afcb6a22f677b2fb19412d` |
| `src/learn/components/lesson-labs/RecurrentAttentionLabs.jsx` | `a1a23a8fd44f2ca83653677f84135482ee631bcaedf10d7d92933d18885d656c` |
| `src/learn/data/recurrent-attention-models.js` | `fb5f5507c6c52db20772b7f789e6a48a20ff9265cadf6da8678204a8fce3c7f2` |
| `scripts/generate-recurrent-attention-lesson.mjs` | `348b7aca3c63a07d310a40df824205b5540fd4739de9f94a768e62f753c45cbe` |
| `public/learn-code/attention-mechanism-bahdanau-luong/attention-read.py` | `89962ed58c5f13c59961d7e7b850d31303f218bf7a3735d03b8b463891c4f2c3` |

If those files change, review the delta and refresh only the relevant bindings;
this record does not authorize treating arbitrary later content as reviewed.

Bounded final CSS follow-up: reviewed the author's larger PAD storage minimum width
(`4.7rem`) and container-width-at-most-450px label sizes (route SVG `16px`, learning
margin SVG `21px`). These changes address the wrapping and small-text findings from
the author's browser inspection without changing plotted positions, numerical
values, proportions, controls or the teaching sequence. Existing container stacking
continues to apply. The revised label extents are compatible with the declared SVG
view boxes on source inspection; the author is checking the actual rendered result.
Only the CSS binding above changed. No numerical or whole-lesson rerun was needed,
and this follow-up does not remove the independent browser-access limit.
