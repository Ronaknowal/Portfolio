# Attention: independent implementation review

26 September 2026. Reviewer: `attention_review`, separate from the implementation author.
Scope: `attention-mechanism-bahdanau-luong`, revision 3, finish the prepared lesson.
The bounded independent correctness, coverage and learning-experience review is complete.
All findings below are closed; no material finding remains in this reviewed scope. User
acceptance and the integration owner's complete browser campaign are separate evidence.

## Prepared content and original coverage

Read the full prepared manuscript, design, visual specifications, displayed training program,
constructed calculation program and saved-model correspondence program. Compared with the
original `src/learn/data/topics/attention.jsx`, whose retained baseline is identified in the
topic's design record. Useful scoring, schedules, masking, training, alignment, input feeding,
local attention, application and computational-cost coverage survives. The new real-data
experiment replaces unsubstantiated reversal outputs and historical curves. Broad production
recommendations, universal length thresholds, unsupported speed claims and inaccurate
input-feeding descriptions should not be carried forward merely to conserve words.

Independent primary-source inspection of [Luong et al., arXiv v5, sections 3.1–3.3](https://arxiv.org/html/1508.04025v5)
confirmed the nonlinear concat scorer, updated-state query, Gaussian multiplication and
previous attentional-vector input connection. This was a focused reading, not a reproduction
of the paper's experiments or a review of its entire bibliography.

### Correctness and interaction-copy findings sent to the author

1. The implementation-route shortcut `O(TSA)` omitted the value-width contribution of the
   weighted read. State the projection and read terms, or explicitly refer to the full
   decomposition already given in the deeper branch. With query width `d_s`, memory/value
   width `d_h` and scoring width `A`, the additive attention work excluding the recurrent
   decoder is `O(S d_h A + T(d_s A + S A + S d_h))`.
2. The saved-model paragraph still requested a learner prediction before editing, and the
   local-window visual description still described predicting the entering positions.
   Replace these lab-adjacent instructions with immediate experimentation and comparison,
   in accordance with the user's current interaction policy.

These are bounded corrections, not a reason to regenerate the prepared experiment.

## Independent numerical checks already executed

A separate Node calculation used elementary vector operations and stable softmax rather
than importing the author's Python calculation functions. It matched all ten saved
constructed fixtures, their context/loss and centered finite-difference query derivatives
(gradient tolerance `1e-8`). Complementary four-memory inputs, different from the packet's
three-memory example, verified that jointly permuting positions preserves the context and
adding one common translation to all keys preserves the attention distribution. Both
relationships follow directly from a reordered sum and cancellation of a shared score term.

The six saved models' trainable-array element counts matched their parameter totals, with
the nontrainable output mask excluded. Their character-error rates matched the saved edit
count divided by the reference-character denominator. No model was refitted and no
historical experiment result was overwritten.

## Prepared-content learning-experience assessment

This was the initial independent heuristic assessment before runtime handoff, not an
observed beginner study. The runtime and closure sections below resolve its deferred checks.

1. **Route:** the first-pass path appears immediately after the opening; local, copy and speech
   branches are explicitly optional. The next-topic link follows catalogue order.
2. **Cautions:** the material gives the important qualifications at their mechanism: contextual
   memories, interpretation, inspected development data and streaming dependencies. Displayed
   training code computes results rather than printing disclaimers.
3. **Real question:** requested word inflection persists from the introduction to the saved
   `lactated` result and measured unseen-spelling comparisons, including the stronger rule baseline.
4. **Live investigations:** the specifications require meaningful entity edits, immediate
   outputs and contrasted/null cases. The two stale prediction instructions were reported above.
   Actual runtime behavior remains to be reviewed.
5. **Figures:** prescribed shelf, vector plane, decoder timelines, mask surgery, checkpoint
   curves, alignment matrix and window/copy diagrams fit different mechanisms. Their rendered
   legibility and geometry remain unverified at this stage.
6. **Connections:** the scalar read, batched tensor code and gradient chain are explicitly
   connected. Parameterized recurrent attention is correctly distinguished from a library
   multi-head module. The complete original coverage audit gives legitimate later owners.
7. **Code:** the training program devotes most of its body to the model and its experiment;
   the initial saved-model route avoids requiring six fits before trying the mechanism.
8. **Practice:** eight changed problems include exact arithmetic, debugging, experiment
   design and transfer, with closed hints and solutions; the temperature extension supplies
   a separate implementation task and numerical target.
9. **Screenshots:** no independent runtime screenshots inspected yet. Browser evidence must
   include the informative edited/masked/null states, not only an untouched page.

## Runtime review scope

The implementation review covered lesson/model/visual sources, the generator's complete
manuscript consumption, ordinary edits and non-preset inputs, empty masks, saved-model
correspondence, prefix causality and cache invalidation, and displayed/downloaded code
identity. Targeted browser operation and screenshot inspection closed the reproduced
padding defect. The author separately completed the broader desktop, phone and intermediate
width campaign recorded in [the implementation record](ATTENTION-IMPLEMENTATION.md).
That broader visual campaign is author evidence, not a claim that the independent reviewer
personally inspected every viewport and screenshot.

## First runtime review

Read the complete numerical model and lab components, the topic generator/rendering
contract, the new self-contained NumPy inference program and the generated manuscript
integration. The full prepared prose is rendered; author-only figure/investigation
paragraphs become the corresponding teaching components. The generator corrects both
stale prediction instructions and the incomplete complexity expression. The complete
training source is loaded only when opened, rather than dropped from the lesson.

The browser numerical model matched all sixteen stored additive/general fixture traces,
including queries, contexts, output probabilities and generated strings. Maximum checked
difference from the prepared NumPy trace was `2.904143592274977e-11`. Complementary tests
used `prone + third_person`, a forced `ab` prefix, and added masked storage. Forcing the
emission preserved the first distribution; masking preserved every output row. A new
four-memory case matched the finite-difference query derivative and the joint-position
permutation invariant. Center `2.37`, radius `1.19` with independently edited local scores
correctly selected positions 2 and 3. Empty windows and all-invalid reads were rejected.
These were direct Node imports of the production model, not another training run.

**Runtime finding, subsequently closed:** the padding inspector formatted an absent clean
reference row as `undefined` when a faulty run lasts longer than the correctly masked
run. Reproduced with additive seed 1, source `cry`, request `participle`, and both PAD
slots admitted: faulty `crigging` has nine rows, while clean `crying` has seven. At rows
8–9, identify that the correct branch already ended instead of displaying a numeric
placeholder. This is an ordinary supported source, not an out-of-domain stress input.

The author added that explicit ended-reference message. On the built preview at port
4196, the reviewer independently opened the padding investigation, entered `cry`, selected
`participle`, admitted both PAD slots and selected output step 8 through the actual controls.
The UI showed `crigging`, a clean `crying` reference, PAD mass `0.10015591`, and “the correctly
masked run has already ended.” The desktop screenshot showed a legible selected matrix
row, synchronized number/range fields and clearly separate source/output probability
bars in the dark/amber theme. This closes the reproduced finding. It is a targeted
independent visual check; the author's broader phone and all-figure review is still separate.

The initial independent browser connection on port 4194 timed out creating the route and subsequently
showed a blank page with an empty DOM and no captured console errors. No lesson defect
or browser pass is inferred from that failed connection; the temporary reviewer tab was
closed. Port 4196 subsequently supported the targeted check above. The integration author's
complete working-session browser evidence is linked from the implementation record.

## Final correction review and limitations

The final source replaces the incomplete complexity shortcut, removes the obsolete
lab-adjacent prediction requests, and implements explicit ended-reference handling.
Reviewed the final container-query/checkbox/spacing changes, the dependency paragraph
for the supplementary mechanics report, and the generator's local split-integrity
provenance adaptation. These changes preserve the frozen data, models and content packet.

The self-contained inference route still requires only its own weights. The more extensive
mechanics report now names its actual sibling encoder–decoder report dependency. The
published provenance links to a real copied audit rather than an undeployed draft path.

No additional model fitting, benchmark or learner study was performed by this reviewer.
Network failure injection and assistive-technology testing were not claimed. The independent
reviewer inspected the 4196 desktop contrast directly; the author's full phone and 760px
reviews remain distinguishable. Numerical evidence for unchanged source is reused rather
than rerunning all six fits. All reported findings are closed with the specific evidence
above, and final source identities are recorded below.

## Final reviewed source identities

SHA256 of the reviewed production sources and relevant learner downloads.
A hash identifies the source to which the review applies; it is not the correctness argument.

| Path | SHA256 |
| --- | --- |
| `src/learn/data/topics/attention.jsx` | `5e0b7b4bcd52da5aae3c8857eefcb863454687f3dbf2121f2e5aca7f6e7c77f1` |
| `src/learn/data/recurrent-attention-models.js` | `fb5f5507c6c52db20772b7f789e6a48a20ff9265cadf6da8678204a8fce3c7f2` |
| `src/learn/data/recurrent-attention-measurements.json` | `a6d62dca596c00bfc486d3580358ad149fde37462cacc7f294c280c65373b6ae` |
| `src/learn/components/lesson-labs/RecurrentAttentionLabs.jsx` | `038fdee01417ac20d115c71445111bb7f2462457336ab50d18583d9d87371046` |
| `src/learn/components/lesson-labs/recurrent-attention-labs.css` | `1480fd216bc08a1f641809d52b21f4a4d257e48eb2809abe0ba5fdfe91cc397a` |
| `scripts/generate-recurrent-attention-lesson.mjs` | `8e2e397d2b637a1e9ba52e9437db467f06a3b0fac2b6b302fd78e98ba3e784a3` |
| `scripts/lib/prepared-lesson-renderer.mjs` | `5c5428df3e7fd127c8961d21f51d23d49c17ae4d8c58d6f90af390dd06c8e8f6` |
| `public/learn-code/attention-mechanism-bahdanau-luong/attention-inference.py` | `6b2cbfd4c0b9fc651bcb4f2b3bbe55602e272005a5e115a89968515a20362848` |
| `public/learn-code/attention-mechanism-bahdanau-luong/attentive-inflection.py` | `0b006def11a1bac2a8f30b5ed43da3f31b1f06dbd0cba95cb267fc89e76dc05a` |
| `public/learn-code/attention-mechanism-bahdanau-luong/attention-calculations.py` | `54615f3153dd52e703b728f2932b33c9d472d8f978a593f715597df54766cd1a` |
| `public/learn-code/attention-mechanism-bahdanau-luong/attention-mechanics.py` | `2b65810ead1e48f4eb3fa302ca1b6cc7f53f076326a1f4bd25dc209290750b2e` |
| `public/learn-code/attention-mechanism-bahdanau-luong/saved-inflection.py` | `777be3c954b6f6a9712058f96baab72da0c31123a3337e6b1bd1af10044f1af2` |
| `public/learn-code/attention-mechanism-bahdanau-luong/calculated-inputs.json` | `046557cd27776737c94ca76dd35facd8fd956b0196535b3642ae79628f7602c6` |
| `public/learn-code/attention-mechanism-bahdanau-luong/additive-seed-one.json` | `3b652129ac75e37147c03b96d252cb67ace5953bf114f60b372777fee707b62f` |
| `public/learn-code/attention-mechanism-bahdanau-luong/general-seed-one.json` | `bbd00d43e4e788e993b87ad608ff40761ed450c5867f7ac3648d9c4fc520c3e6` |
| `public/learn-code/attention-mechanism-bahdanau-luong/data-provenance.md` | `b44cf108ea626b2ffe87a8374489a6d2a9b114d8dea3ed1f983205e62c2a653c` |
| `public/learn-code/attention-mechanism-bahdanau-luong/split-integrity-repair.json` | `14e67b88114d03d37a4caaa4e469bbd50149ad2ed3d6430254f44df2a5e21767` |
