# Dropout: visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Play with mask geometry and mode.** Edit features/weights, probability, survivor scale, mask grouping, branch position, per-block rates and train/eval mode; inspect retained Monte Carlo prefixes.
**See the consequence.** Update weighted outcome means/variances, gradient routes, call counts, state buffers and saved prediction distributions immediately. Keep the sampled mask fixed while comparing a parameter, with resampling a separate action.
**Decision connection.** Choose masking scope and evaluation behavior from their actual effects; distinguish expected active depth from work that was really skipped.


Content/specifications only; no renderer or browser lab implemented. Inputs are produced by dropout-experiments.py and retained in calculated-inputs.json. Static tensor diagrams, probability outcomes, circuit rewiring and recorded-fit comparisons serve different purposes; do not replace them with a single generic lab shell.

## Shared behavior

Each investigation follows the live exploration contract above: current results are visible immediately, valid entity edits update all linked views, and comparisons explain the mechanism. Reset restores the declared inputs and recomputes their result. No prediction or answer-submission state is retained.

Use numeric labels and dropped/kept symbols in addition to color. Each grid has a row/column text table; each circuit has an equation alternative. All edits work by keyboard or direct entry, touch controls have adequate target sizes, result feedback uses a polite live region, and no answer depends on animation. Stack sample/channel panels on mobile without losing axis labels. Respect reduced motion. Bounds below keep work to tiny arrays and scalars; select saved digit results instead of browser training. Lazy-load only the current lesson's data/interactive homes during implementation.

## 1. Values, masks and the actual backward pass

Home: lesson §1. Show two activation lanes h=[1,2], fixed original weights [1,−0.5], mask [1,0], q=0.5, target1. Separate the masking and survivor rescale stages. Join weighted contributions at output and display half-squared loss. The backward trace reuses the same mask, highlighting zero contribution versus a permanently deleted parameter.

Source: mechanisms.one_update. Output2, loss0.5, weight/input gradients [2,0]; SGD0.1 gives weights[0.8,−0.5], output1.6, loss0.18.

Initial live result: which coordinate gets a nonzero weight gradient? Keep both selected coordinates and their computed contributions visible together. Guided reveal follows an actual gradient step; learner may step forward/back without changing the underlying fixture.

Changed contrast: original weights with mask[0,1] gives output−2, loss4.5, gradient[0,−12] by the same formula. Null all-zero mask gives output0 and weight gradient0, while loss remains0.5; do not mark loss zero because gradients vanish. Allow fixed-mask free exploration with h/weights in[−3,3], target[−3,3], p in{0,.25,.5,.75}; valid input changes recompute every dependent result and explanation; retained baselines keep their original inputs.

Do not animate new randomness during backward. Keep the mask revision visible.

## 2. Probability outcome tree and scaling repair

Home: §2. Two binary choices expand into four leaves; leaf areas/probabilities change with p. Show weighted mean and variance accumulation, not a claim that a finite random sample equals its expectation.

Sources: mechanisms.enumerations for p0.25/0.5, h[1,2]. At p0.25 probabilities [.0625,.1875,.1875,.5625]; means[1,2], variances[1/3,4/3]. p0.5 all leaves.25; variances[1,4].

Live observation: change p0.5→0.25; show whether means and variances increase/decrease/stay. Calculate and explain each separately.

Construction: fix p0.25 and h[1,2], initial survivor multiplier0.75, unsolved. Learner enters multiplier c in[0,5] to make both weighted coordinate means equal their inputs, tolerance1e−8. Correct c=4/3; initial means[.5625,1.125] fail, c1 gives[.75,1.5], c0 gives zeros. the live result uses probability-weighted leaves, not a text match.

Nonlinear switch: signed contributions[1,−1], ReLU after sum. At p0.5 expected ReLU0.5 versus deterministic0; p0.25 expected0.25. Null p0 makes them equal. No claim of exact nonlinear ensemble averaging.

Boundary panel: mechanisms.boundaries p0 identity; p1 training zeros with zero local gradient, evaluation identity. Never display infinity from an unguarded divide by zero.

## 3. A mask has geometry

Home: §3. Show two examples, each with two 2×2 channel maps. Values1–16 in row-major order. A synchronized axis diagram explains element[2,2,2,2], channel[2,2,1,1], row[2,1,1,1], batch[1,1,1,1] broadcasting. Probability p0.5 fixes survivor scale2.

Source: mechanisms.granularity. Retain explicit bit arrays; do not infer a mask from whether an input value was already zero.

Live observation: display how many independent random decisions exist in each mode (16/4/2/1), computed from the shape product.

Construction: start all four channel bits1, unsolved. Task: hide only example1/channel2 completely while preserving every other map. Learner must choose channel geometry and edit bits to [[1,0],[1,1]]; allow equivalent element bits only if the stated task explicitly asks for an output rather than a channel-sharing mechanism. Here the learning objective includes sharing, so channel shape is required. Check every cell and shape.

Contrast selected example2/channel1 must remain; row masking of example1 removes too much. Null all-keep preserves the mask pattern but training values are still scaled2: label identity only in evaluation or p0. Changed-input check replaces all values by values+1; the bit-sharing rule remains, while outputs change accordingly. The task cannot pass from a fixed colored picture.

Covariance inset: compare all four independent masks against two shared masks for[1,2] at p0.5. Covariance0 versus2 is derived from enumerated outcomes. Distinguish fixed activations/covariance over masks from empirical data correlations.

## 4. Protect the residual lane

Home: §4. Circuit has direct x lane and correction F lane; mask operator can be moved before branch join or after whole join with a select alternative.

Source: mechanisms.branch, x[2,−1], F[.5,1], q.5. Correct dropped output[2,−1], surviving[3,1], mean[2.5,0]. Wrong whole-sum outputs[0,0]/[5,0].

Live observation: does dropping the mask preserve x? Compute the comparison from the complete current inputs.

Construction: initial mask after sum with m0 is unsolved. Move masking onto correction only; check equality to x on both original input and changed input[1,3] with correction[.5,1]. Shape and correspondence must remain valid. Output equality at x0 is a null that cannot establish identity for arbitrary input.

Derivative overlay uses I+(m/q)JF; label it conditional on the sampled mask. Preserve predecessor's cancellation warning rather than presenting an indestructible gradient path.

## 5. Active depth and execution timeline

Home: expected-count subsection. Place one probability label and one mask switch over each residual branch; add an expectation bar and separate executed-call counter.

Source: mechanisms.schedules. Original-style L4 endpoint.5 rates[.125,.25,.375,.5]→2.75 active; zero-first[0,1/6,1/3,.5]→3. L12 endpoint.2 zero-first→10.8. L1 zero-first[0] explicit.

live comparison distinguishes expected active branches from whether F was called. mechanisms.execution with dropped batch: eager calls1, lazy calls0, both branch outputs zero. This exact counter is not runtime or GPU savings.

Optional schedule construction: four blocks, first rate0, nondecreasing rates ending.5, expected active count3. Start[0,0,0,.5] giving3.5, unsolved. Require both middle rates sum.5 with bounds0≤p≤.5; [0,1/6,1/3,.5] is one solution, [0,.25,.25,.5] another. Calculate and explain current sum and constraints, not resemblance to a line. Null endpoint0 gives all active. Do not require independence to derive the expected count; it follows by linearity.

## 6. Mode and normalization state

Home: §5. A state panel has independent gradient-recording and module-mode controls, plus BatchNorm running mean/variance/counter and dropout mode. Show state transition, not just output text.

Sources: mechanisms.state and normalization. no_grad with train mode still increments BN counter to1. Selective dropout after eval leaves it at1 and running mean unchanged. No need to pretend a particular random mask proves the mode contract.

Variance diagram: clean equally likely1/3 has mean2,var1; dropout outcomes0/2/0/6 each.25 have mean2,var6. Exact four-row BatchNorm with momentum1 stores running variance8 because of unbiased estimation; evaluation outputs ±.35355317. Clearly label divisor and epsilon1e−5.

Live observation: which state changes when only gradient recording is disabled? compare against actual recorded state. Repair task starts with all modules in train mode during scoring; learner sets ordinary eval behavior and no recording, or separately chooses the explicitly labeled MC goal and enables only dropout. These goals have different conditions; do not accept a reset RNG as deterministic evaluation.

LN inset: [1,3] normalized≈[−1,1], masked[2,0]≈[1,−1]. It lacks running-statistic mismatch but is not invariant. No additional full lab needed.

## 7. Real fitting comparison

Home: §6. Architecture strips show the two distinct model families and their parameter counts; the unmasked baseline remains visible. Plot actual points at0/1/25/100/200/400, with training and validation CE clearly distinguished, and show correct/count in a separate table. No fabricated intermediate points or smooth idealized curves.

Data: fits. Three seeds, four element-rate fits and five residual configurations per seed. Show source-ID-labeled actual images from the attached CSV. Restrict random-looking controls to selecting measured configurations; arbitrary new rates require executing new experiments outside the browser and retaining results.

Live observation: selecting a baseline and variant displays both final validation CE values and their difference immediately. Calculate and explain exact stored values; note ties only within an explicit numerical display tolerance, not tied rounded accuracies. Valid input changes recompute every dependent result and explanation; retained baselines keep their original inputs. Reset restores the stated inputs and immediately displays their computed result.2.

Feedback explains that the baseline is slightly better on seed1, slightly worse on seed2; zero gap does not establish superiority. Display hypothesis, measurements and interpretation separately. No new implementation benchmark implied.

## 8. Optional MC prediction distribution and noise family map

Home: §7. For first three validation specimens only, show actual 100 per-pass probability vectors from monte_carlo.sample_probabilities, with mean and standard deviation. Their source IDs are251/40/149; mean records for all120 specimens also exist. The wrong specimen299 has saved mean/entropy/disagreement but no retained individual sample vectors: do not invent its sample trace.

Select T as a prefix of saved draws for first-three specimens and recompute means, entropy and disagreement; initial T10. The live readout shows whether the selected specimen's entropy increases, decreases or stays equal between the selected prefix and all100 draws; Calculate and explain the recomputed values with tolerance1e−7. Compute the comparison from the complete current inputs. Changing either clears it. Explain separately that neither direction guarantees correctness; that conceptual caution is not a scored input-independent quiz.

The two hypothetical binary panels use[.9,.1]/[.1,.9] versus two[.5,.5] draws. Same mean entropy.693147, disagreement.368064 versus0 (mechanisms.hypothetical_disagreement). Label hypothetical. A changed-input comparison shows which has larger between-mask disagreement; null identical draws zero. These are not calibrated risk scores.

Retain actual aggregate deterministic/MC CE, Brier convention and116/120 count. More draws are repeated inference work; no claimed device timing.

Optional taxonomy inset uses a region stencil, missing weight edges, retained recurrent state and branch coefficients. Each is a separate operation with an annotated source, not an implied implemented backend. Attention row[.25,.75]→[.5,0] sum.5 illustrates that a sampled row need not sum1.

## Deferred implementation checks

Verify translated formulas, exact-fixture parity, genuine input-bound output consistency, incorrect/contrast/null/reset behavior, keyboard and screen-reader equivalents, mobile grids, reduced motion, file downloads, data-loading scope and memory use during phase two. No formal browser/build review was performed in content-only work.


## Code-to-mechanism implementation contract — 22 September 2026

Place the manuscript's new implementation route next to its stated concept section. Preserve the named axes, state and algorithm steps when implementing figures; the complete teaching programs are content inputs, not a hidden replacement for learner-visible code. Render long source only on demand with keyboard-scrollable code and wrapping download labels. Keep constructed comparison fixtures separate from recorded training experiments. No browser execution of Python, pretrained-model download or GPU experiment is required to operate a lab.

The ownership map in design.md identifies which operations are implemented here and which actual sources are reused. Both paths must be findable: the transparent mechanism and the ordinary package/tool route, followed by the changed-input practice. There is no guess-entry, prediction submission or answer-unlock state. Current outputs remain visible while the learner edits meaningful inputs; separate written practice can retain hints and solutions.
