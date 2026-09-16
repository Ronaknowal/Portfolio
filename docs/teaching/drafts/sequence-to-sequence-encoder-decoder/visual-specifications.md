# Sequence-to-sequence visual and investigation specifications

Content-first contracts, 13 September2026. No production visuals, browser labs or implementations are included. Use the shapes, exact computations, data and distinctions in the manuscript rather than adapting every hurdle into the same panel.

## Shared interaction contract

Every **gated investigation** begins with the answer unset and no visible solved default. Worked explanatory figures are explicitly ungraded. Prediction state includes current input/token arrays, weights/model revision, selected output coordinate or token, scoring settings, expected relationship/value and a revision identifier. Editing a consequential entity invalidates the prediction and result; cosmetic captions and plot zoom must leave computed values fixed.

Require an actual entity edit or arrangement where specified. Predictions are not preset demonstrations with a “correct” label. On reveal, compute the result from the same state used to draw the figure; compare the learner's selected relationship and optional number separately. Show their prediction beside the actual result and describe the causal path. Token strings, probability changes, gradient directions and stop reasons require different feedback.

Reset restores the declared fresh default, clears prediction/grade/history and removes stale edited states. Source edits rerun the encoder; prefix edits restore the correct decoder state and recompute the affected suffix. Never keep an old context for a changed source. Reject invalid numeric or unsupported-character inputs while preserving the last committed valid state. Bounded inference is deterministic and uses saved weights; no browser training or network call.

Use 1e-9 absolute tolerance for constructed exact-arithmetic fixture comparisons where appropriate; 0.005 for a learner's three-decimal numerical estimate; 3e-6 for independent float64 versus saved float32 native parity. Do not grade a token-probability direction from rounded bars; use raw values with a disclosed equality tolerance.

Lazy-load the seed1 model only for its investigation. The manuscript's full research report is an author artifact, not a page-level eager import. Bound real input to1–12 ASCII lowercase characters, three request tokens, at most14 source tokens including request/EOS,64 hidden coordinates, at most16 generated tokens and at most three live beams. No unrestricted exponential search. Debounce committed edits, provide keyboard fields, and invalidate asynchronous stale results.

All figures need text/numeric equivalents, labeled axes/units, color-independent symbols and keyboard-operable controls. A character's source position and identity are separate. No essential hover-only information or autoplay. Use learner-controlled stepping; reduced-motion mode removes transitions. On mobile, stack source/bridge/decoder views and keep horizontal scrolling within bounded token/matrix regions. Announce a short committed result through an accessible live region, not every slider movement.

## A. Two timelines and target-shift workbench

**Home:** sections1–2. **Hurdles:** source/output lengths, BOS versus EOS versus PAD, teacher forcing without answer leakage.

First show an **ungraded** two-track diagram for actual `lactate+past→lactated`. Source has request+7letters+EOS =9 tokens. Decoder predicts8letters+EOS =9 targets, with BOS then the first8 target tokens as inputs. A context bridge separates the two processing phases; do not imply aligned source/output positions. A separate tiny `ate` table is the worked target-shift example.

**Fresh gated default:** source `care+past`, reference `cared`, plus a second `try+past→tried` row. Learner arranges actual target/input token cards in initially unsolved rows, including BOS and EOS, then predicts which rows feed which network. The answer key is created by the real `batch` function, saved under `mechanics-results.json.loss_mask`, not an unrelated static permutation checker.

After a valid arrangement, edit the reference `cared→carts` while retaining source `care`. Ask for the first decoder-input/prediction position affected. Reference target position4 changes e→t, but input at prediction position4 is still r; the first changed decoder input is position5. The native calculation verifies first4 output-logit vectors unchanged. Source state stays unchanged. Display the reference as a **constructed counterfactual target**, not a newly verified English inflection.

Mask branch: add two actual PAD cells to each target row and ask whether the valid-token mean should change. Checked native valid NLL stays1.0740114450 for the two-example fresh batch. Source packing and target-loss masking have separate labeled boundaries. PAD positions are storage; EOS remains a scored token. Require the learner to identify the actual denominator,12 valid tokens, rather than select a prefilled “ignore padding” answer.

Nulls: reorder display labels alone; add correctly ignored target PAD positions; edit a future target without changing the earlier prefix; restore original token arrangement. Contrast: misplaced same-position target input can leak the answer; do not animate it as a valid training trace.

## B. Context bridge and joint-gradient path

**Home:** sections3–4. **Hurdles:** encoder/decoder weights are distinct, context is a numeric dependency, output loss trains encoder.

The manuscript scalar x=[.2,.8] figure is **ungraded**. Draw two encoder state nodes, the context edge, two decoder nodes and A/EOS probability pairs. Expand the selected affine/tanh/softmax step with actual scalar values. Reverse gradient arrows are signed and different from forward state arrows.

**Fresh gated default:** x=[−.3,.6], encoder input weight.7, recurrent weight.4,bias.1,h0=0; decoder input weight.6,recurrent.5,bias.05; BOS/A embeddings.1/.4; output scores[s,−s], reference A,EOS; learning rate.1. The prediction is unset. Ask for final context, the sign of encoder input-weight update, or which of the two losses is larger; show outputs only after reveal.

Author-only fresh values are in `scalar_fresh`: context.4431763680; mean loss.8125134135; encoder-weight derivative−.01004846337; updated weight.7010048463; updated loss.8125033251. The worked fixture's different values stay available as an explanation tab, not as solved defaults. Fresh actual x2 edit .6→.2 has separately computed outputs in `scalar_fresh_input_edit`. Zero-rate null has checked identical parameter/loss. The reference remains A,EOS in this two-step mechanism.

Controls: x1/x2 [−1,1], encoder input weight[−1.5,1.5], rate[0,.2]. Recompute the analytic chain rule or bounded autodiff-equivalent reference. If an allowed step raises loss, display it honestly. A source/weight edit invalidates prediction and recomputes the full dependency path.

Optional graph branch: detach the context. Forward values are unchanged, while the encoder gradient through the decoder is cut. A real four-example native check gives encoder input-weight gradient norm.1699930280 before detach and no such gradient afterward. The displayed control must name the path being cut, not claim all encoder learning is impossible under every possible auxiliary objective.

## C. Probability tree and bounded beam candidates

**Home:** section5. **Hurdles:** local versus whole-route probability, candidate-owned states, EOS, scoring policy.

The manuscript's A.60/B.40 tree is **ungraded** and exactly enumerated. Render branch probabilities as labeled edges, prefixes as nodes and complete EOS leaves with products/log scores. It is a constructed probability model, not measured language-model output.

**Fresh gated default:** root A.55/B.45; afterA EOS.60/C.40; afterB EOS.85/C.15; afterC EOS1. The learner edits at least one conditional probability (its complement updates, keeping row sums1), chooses greedy or beam width1–2, predicts the complete winner and records its probability before reveal. Bounds [.01,.99] for editable nontrivial probabilities.

Author-only fresh complete probabilities: A-EOS.33,AC-EOS.22,B-EOS.3825,BC-EOS.0675. Greedy returnsA-EOS; beam2 returnsB-EOS. Changing P(EOS|A) to.90 gives A-EOS.495 and AC-EOS.055; both methods now chooseA-EOS. All default/changed histories are in `tree_fresh_*`. Width1 agrees with greedy. Repeated run or renaming node labels leaves numeric probabilities fixed. A finished branch never grows children.

Each beam candidate shows prefix, raw logP, completion flag and owned decoder-state identity. For the small probability tree the prefix defines the distribution; explain that a neural candidate instead carries a numerical state. Do not use one hidden-state reference for different consumed prefixes.

Length-score branch uses a different fresh pair from the worked manuscript: lengths2 and5 with raw logP−1.2/−1.4. Ask for the winner at alpha0 versus1 under the explicit GNMT denominator. Author-calculated scores at1 are−1.0285714286/−.84, so the longer wins at1 and shorter at0. The denominator applies alpha exactly once; L includesEOS, excludesBOS. This two-candidate scoring exercise does not claim its candidates came from one fitted tree. Null alpha0 recovers raw ranking.

## D. Fitted source/prefix interventions

**Home:** section8, after the real experiment. **Hurdles:** source conditioning, reused stale context, generated-prefix effects, probability versus correctness.

The worked real trace is seed1 `lactate+past`, producing `lactated`. It is training data and labeled so. Show source-token embeddings/states, context, decoder previous token, selected hidden coordinates and next-token probabilities with numeric inspection. Hide arbitrary semantic names for coordinates.

**Fresh gated default:** real development entry `emmove+past` (reference `emmoved`), seed1. Prediction unset; no future output or direction shown. Source ID is the pinned CSV row's lemma/request/source_rows tuple. Learner edits actual characterv→d, yielding constructed query `emmode+past`, or changes request to participle. Require prediction of first-token P(e) direction and whether the generated string remains the same. Grade both, and do not grade “correct English” for a constructed query without a reference.

Author-only original P(e)=.6076041629; source edit raises it to.8741401680 and changes generated `emoves` to `ememmis`. Request edit raises it to.8126019281 and produces `emomming`. The original model is wrong against the real reference; preserve that fact. Full states and probability vectors are under `mechanics-results.json.traces`. Native versus independent decoder probabilities agree within5.87e−7 over the checked traces.

Prefix branch: on the fresh original input, force its first generated e→a after prediction. The first probability vector stays identical because it was computed before the intervention. At the next step P(m) changes from.9689718671 to.5478859994; generated result becomes `amves`. Ask about this temporal boundary before revealing the second-step state.

Context branch: replace with zeros or the actual `lactate+past` context. Saved outputs are `pled` and `lactated` respectively. The latter demonstrates that this decoder's source access is through its initial state. It does not imply zero is a meaningful linguistic input. Preserve the original computed context for reference, and make the overridden context explicit.

Nulls checked: repeat identical source; replay first two already generated tokens; force the already selected token; change only source-label text; original context supplied unchanged. Prefix forcing leaves earlier distributions intact. Return source to its original characters and rerun the encoder to restore original results. All actual probabilities normalize, and masked output IDs remain zero.

Cap branch: limits1–16, default16. Predict effect of cap3 on a fresh source and show explicit capped/natural-EOS status. On the default source it yields `emo`, EOS=false, logP−.8332651146. A prefix may outrank its complete extension in raw probability. Display source/input and generated-token lengths independently. Do not force EOS at the cap and then claim the model chose it.

## E. Actual outcome plots, no new generic dashboard lab

**Home:** section6. This is a **read-only exploratory figure**, not another gated investigation. Use retained final-split checkpoints and every declared seed. Plot update versus development teacher-forced NLL and update versus generated exact rate in separate aligned axes. Training values are final-only; do not invent training checkpoint curves from the final point.

Show all three final training/development counts with the copy and predeclared-rule baselines. Include actual denominator447, CER numerator/reference3490 characters, EOS count and seed. A grouped length view shows168 short/279 long development examples; switching grouping changes aggregation, not weights. No causal “memory limit” boundary or interpolated benchmark data.

Beam comparison is seed1 fixed weights: greedy53/447,beam3 56/447,alpha0,max16. Label this a measured change to search. Do not pool its56 with another seed's parameters or compare test/training mixes. A text table is the complete fallback.

## Implementation handoff

All numerical defaults, revised split, baseline/fit outcomes and changed/null fixtures are authored and executed on CPU. The fresh length score is an explicit calculation, not a measured model probability. The complete program must remain available with setup, input data, interpretation and real results.

Phase two implements topic-owned diagrams/investigations, extracts only needed model/data subsets, verifies model-parity and input-bound grading, invalid/reset/async cases, source packing, output support, target mask, termination and search state ownership. It also performs independent content/native checks where justified, accessibility/mobile/browser/integration and lazy-loading checks. No phase-two completion is implied by these author calculations.
