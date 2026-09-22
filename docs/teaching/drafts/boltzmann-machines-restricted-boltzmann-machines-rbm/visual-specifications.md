# RBM visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Move energy and follow probability mass.** Edit small-model biases/interactions, data counts, transition/sampling settings and supported retained digit states.
**See the consequence.** Show normalized joint/marginal probabilities, data-model statistics, exact transition mass and sampled chain trajectories simultaneously.
**Decision connection.** Distinguish energy from normalized likelihood, reconstruction from probability and finite mixing behavior from an equilibrium claim.


Prepared content only, 13 September 2026. These are required phase-two implementations, not existing browser features. Retain lesson.md, rbm-study.py, digits-400.csv, calculated-inputs.json and data-provenance.md. Exact fixtures and nine real CPU fits are author evidence; browser translation, interaction, accessibility, rendering and independent reviews remain deferred.

## Shared behavior without a shared generic lab

Use switch/edge ledgers for energy, probability-mass paths for sampling, difference ledgers for learning, and pixel masks/uncertainty tiles for the real model. Reuse typography and accessible controls; do not flatten these mechanisms into repeated text-output boxes. Place each representation next to the concept it explains. Color is supplemental: state labels, signed values, numeric tables and masks must remain sufficient without it.

Each investigation includes an initially closed hint and a separately closed explanation/solution. The hint identifies the next calculation or dependency without displaying the answer; the explanation uses the current input and the checked worked alternatives below. Construction tasks accept any numerically verified solution. Do not expose a solved default as a learner challenge.

Open with the current normalized distribution and computation visible. An entity edit immediately updates the result; a pinned baseline retains the old parameters. Show numerical differences with stated tolerances and explain the energy/probability mechanism. There is no answer field, submission or prediction grade.

Support labeled buttons/number fields and keyboard focus, without requiring drag, hover or animation. Provide static diagrams plus result tables and closed worked solutions as the noninteractive alternative. On mobile stack stages in dependency order and preserve row/column labels. Honor reduced motion; announce a settled result. Validate finite bounded inputs before calculation and retain the last valid state after an error. No fitting, installations or whole-curriculum data loading in the browser. Lazily load only this topic and the currently opened real investigation's data; use semantic filenames/components under current code-structure rules, not batch names.

## 1. Switch graph, energy ledger and hidden-state sum

Place in sections 1–3: first a two-visible general BM edge, then the RBM's two visible switches below one hidden switch. Show explicitly that the direct visible edge is removed in the RBM; the two models use different energy formulas. A connecting line is undirected and not an attention score. Selecting a state highlights contributions to energy, the exponential mass, the sum over the two hidden alternatives and the final visible probability. Numerical state tables are the source of every bar height.

Default RBM: W=[ln3,ln3], a=[0,0], b=0; visible order00,01,10,11 and hidden order0,1. calculated-inputs.json exact.joint_mass=[[1,1],[1,3],[1,3],[1,9]], Z20, visible probabilities[.1,.2,.2,.5], hidden probability[.2,.8]. Use stable softplus/logsumexp; label energy as dimensionless and probability as0…1. The general-BM contrast has direct couplingln3, no biases, four masses[1,1,1,3] and Z6. Do not reuse the RBM masses for that diagram.

Allow actual switch edits and bounded weights/biases−6…6. Present state toggles with state0/1 and conditional probability as separate quantities. A hidden-switch toggle changes the selected joint event and its energy; it does not change the model's marginal visible probability unless a parameter changes. This is an intentional null of marginalization, not a broken control.

Investigation: baseline shows the worked original distribution. Stage a1=ln2, inspect direction of p(11), then evaluate. New masses[2,4,8,20]/34, p(11)=10/17. show whether states had their unnormalized mass doubled and separately which normalized probabilities increased. Stage common energy offset+100: all probabilities unchanged even though displayed absolute energies rise. Carry the offset consistently through every state and normalization. Setting both interactions zero with zero biases gives four visible probabilities.25 and zero visible covariance. Editing only a display label never changes numbers.

Meaningful construction challenge: from the default correlated model, adjust actual interaction coefficients to make p(11)=p(v1=1)p(v2=1) while keeping p(v1=1)=p(v2=1)=.5, tolerance1e−8. It begins unsolved. Zero weights/biases is one solution but do not prefill it. Calculate and explain computed distribution and marginals rather than literal coefficients. With only one interaction zero, independence holds but the marginal constraints may still fail; show the separate residuals. Present this as constructing one simple distribution, not identifying unique parameters.

Deferred checks: enumerate all joint states independently for every tested parameter change; compare analytical conditionals/marginals, p sum1, nonnegative masses, covariance, offset invariance, selected-hidden-state null, actual-entity edits, unsolved challenge/reset and grayscale/keyboard state identification.

## 2. Data-versus-model co-occurrence ledgers

Place in section4 beside the one-update table. Display one column from the chosen observed visible state and a second from the model's entire distribution. Edge rows show positive statistic, negative statistic, difference and proposed update. Visible and hidden biases have their own rows rather than pretending all parameters are pairwise weights. A selectable highlight follows which states contribute to a model expectation.

Default observation11 and same three-switch model. Positive weight statistics[.9,.9]; negatives[.6,.6]; gradients[.3,.3]; visible gradients[.3,.3], hidden.1. Learning rate.1 gives W=ln3+.03, a=.03, b=.01; logp11−.69314718056→−.65690172920. All changes are simultaneous from old-state statistics. Model expectations are recomputed after edits, not retained from baseline.

Stage observation10 and show whether interaction increases/decreases. Exact gradients W[.15,−.6], a[.3,−.7], b−.05. Current results are visible on opening. until submission. Expose a numeric calculation option for the selected edge and a three-way sign option for the full update. Change learning rate0… .5; rate0 is a true unchanged-parameters null. Changing the selected observed state still changes the displayed data expectation even if step size0 makes the proposed parameter update zero. Ask about the quantity actually being computed.

Optional deeper mode uses a learner-edited four-state empirical probability vector, nonnegative entries summing1 after explicit validation/normalization choice. Starting at observation11 is delta mass; setting data mass equal to [.1,.2,.2,.5] makes all exact expected gradients zero. This is an input-distribution control and does not require training. Do not silently normalize invalid negative entries or present an automatically solved configuration.

Deferred checks: full finite-difference derivative at several bounded asymmetric weights/biases, old-state simultaneous update ordering, negative expectation recomputation, source-of-sign explanations, Compute the comparison from the complete current inputs.

## 3. Probability transport, sampled paths and CD/persistence

Place in section5 after the first-transition calculation. Four state nodes labeled00/01/10/11 have probability bars and transition ribbons; keep at most16 edges visible and allow selecting one source to reduce clutter. This is a Markov transition diagram, not a smooth energy landscape. A separate single-particle strip displays actual uniform draws, conditional probabilities and binary outcomes. Switching views must not conflate exact mass with estimated frequency.

For each visible source, compute hidden probabilities, then mix the two product-Bernoulli visible distributions. Default transition rows are calculated-inputs.json exact.transition:

~~~text
00: [.15625, .21875, .21875, .40625]
01: [.109375, .203125, .203125, .484375]
10: [.109375, .203125, .203125, .484375]
11: [.08125, .19375, .19375, .53125]
~~~

Default q0=[0,0,0,1]. One mass step computes q←qT and the expected negative statistics under q, then compares expected CD gradient with the exact gradient under p. The exact.traces record provides steps0…10 including gradients and total variation. Parameters remain fixed during those mass steps. A “one learning update” action, if included, is separate and creates a new parameter revision; its transition matrix and equilibrium bars must change together.

Fresh primary task: keep initial visible11 but stage hidden bias0 to ln2; show whether the next-step probability of11 rises from the worked.53125. Its new value is.5460526315789473 (investigations.hidden_bias_ln2_next_probability11), calculated from hidden probability18/19, not from a new training fit. Additional tasks may change the initial state, interaction or step count; show the next-step probability mass and its incoming contributions immediately. Steps0…100, two-visible/one-hidden only; parameters−6…6. Strong interactions may make progress slow, so do not label the final displayed step “converged” by construction. Set q0 to the exact stationary mass as an explicit comparison case: it remains stationary, although an individual sample still moves. This is a particularly useful null for understanding a distribution versus one trajectory.

Single-particle default demonstration uses state10, hidden uniform.8, visible uniforms.3/.6: hidden0 then visible10. Support a fully specified seeded PRNG with retained draw stream for reset, or expose a short fixed editable uniform list in[0,1). Do not call Math.random and claim the saved Python stream is reproducible. A changed draw crossing its probability threshold changes the state; an edit that remains on the same side can be a legitimate no-change. Sample labels must include the current parameter revision.

Persistence view uses a bounded collection of at most16 actual particles and two explicit minibatches. Default4particles start[00,01,10,11]; minibatch A=[11,11,10,01], minibatch B=[00,01,10,10]. The first-step demonstration may use hidden uniform draws[.1,.9,.4,.8] and visible uniform pairs[[.2,.8],[.6,.3],[.9,.4],[.1,.7]] for both methods, then fresh explicitly displayed draws for the next step. Identical draws control randomness but do not imply identical conditional probabilities or outcomes. CD starts each chain at the current data; PCD carries particles forward. Show before/after particle states and start provenance; never simulate persistence by replaying a fixed textual list. This is a constructed method demonstration, not the64-particle real campaign. Fixed three-switch model probabilities supply the update; expected mass view can remain noise-free for clarity. If parameter learning is enabled, allow at most20 manual updates with rates0… .1 and record old states/statistics. No automatic infinite animation or browser training campaign.

Deeper comparison: default q1's E[h after next conditional]=.809375, while sigmoid at mean-visible [.725,.725] gives.8310360531. Label the latter as a deterministic replacement, not another sample of the same exact chain. Recompute both for current parameters and q; zero interactions makes the hidden response constant so the two agree, a valid null.

Deferred checks: T rows sum1; pT=p; enumerate hidden/visible paths independently; q update versus sampled outcomes; persistence survives minibatch changes; correct final hidden-probability statistic; random draw threshold edge cases; state versus probability labeling; invalidation/reset and slow-mixing display without fake convergence.

## 4. Reconstruction arrows against normalized probability bars

Place at section7's counterexample before the real-image completion. Two tiny model diagrams show input11, deterministic mean reconstruction and all four visible probabilities. Use one visual scale for probability bars and a separate labeled MSE scale. This is a mathematical fixture, not a fitted comparison.

Model A: W[20,20], a[−10,−10], b−20. Model B: W[0,0], a[ln9,ln9], b0. calculated-inputs.json exact.reconstruction_counterexample stores full metrics. A MSE2.06096665264e−9/NLL.6932379763; B MSE.01/NLL.2107210313. These fixed extreme parameters are not part of the editable Gibbs lab's−6…6 controls. Stable arithmetic is required; do not round probabilities to exact0/1 before subsequent computation.

The prose works input 11; use input 00 as a contrasting inspection case. Show reconstruction and NLL for both models side by side immediately, with their different winners where applicable. Selecting another input recomputes both metrics and explains why reconstruction quality does not determine normalized likelihood.

Deferred checks: all4 inputs, exact normalizers, no misleading log-axis tick labels, MSE per pixel versus total error, Valid input changes recompute every dependent result and explanation; retained baselines keep their original inputs.

## 5. Real-digit model, sampling and missing-data evidence

Place after experiment results and in section7. Keep three modes with distinct purposes: inspect weights/hidden responses, independent generation, and conditional completion. Use native8×8 pixel tiles and signed8×8 feature-weight tiles with one shared scale across hidden units/models when comparing. Avoid auto-rescaling each filter to look equally strong. A weight pattern is not automatically a named stroke detector. Hidden probabilities use a0…1 scale and input-dependent values.

Source data: digits-400.csv; threshold>=8; drop228duplicate12 and300duplicate274 BEFORE split. calculated-inputs.json protocol.roles contains238/80/80 source IDs,8 hidden units and all9 final parameter sets. Allow model method/seed selection with explicit labels. Quantitative experiment plot, if useful, uses recorded post-epoch1/10/50/100/300 fit/development exact NLL, actual x coordinates and nats/image. No synthetic smooth interpolated claims, no assessment-per-epoch curve, no timings. All nine final assessment scores remain visible together; no default winner badge.

Model inference is small: 64×8 weights, plus biases. Full output JSON also contains histories/per-image evidence; phase two may extract purpose-specific constants with source hashes and equivalence checks instead of loading all research evidence into the runtime. Retain the full author packet. Loading one model and one selected image should not load other lesson modules or run training. For normal inference compute actual sigmoid/free energy from the selected weights. Exact normalization/conditional calculations enumerate256 hidden states and64 visible terms, bounded and recomputed on explicit evaluation; cache model-dependent logits until weights/model change.

Independent generation: show all16 saved exact samples in original order for each method/seed, with matching hidden-state samples. Mark them “independent draws using exact hidden enumeration”; they are not reconstructions, assessed examples or outputs of a short Gibbs chain. If adding fresh draws, use explicit reproducible PRNG and correct categorical hidden draw then conditional Bernoulli visible draws. Do not generate an arbitrary plausible digit illustration and attribute it to this model.

Show four distinct roles together: editable observed pixels, hatched/labeled missing cells, conditional missing-pixel probabilities and the original withheld truth for comparison. The missing-pixel truth must never enter conditioning or sampling, even though the learner can inspect it. Use distinct labels to separate an observation from a model completion.

The prose works pixel27; the fresh primary task instead stages observed pixel index18 toggle and asks whether any missing probability changes. Compare old/new missing probabilities and highlight the largest absolute change, .0687647026712648 in the checked default model. Fresh original/edited arrays live in investigations.fresh_completion. The prose pixel27 arrays remain under fits[].intervention for explanatory comparison. Allow choosing a missing pixel and immediately display its signed probability change; compute that pixel’s delta rather than infer it from the maximum across pixels. Null task: change only placeholder at missing index28; result must remain unchanged because that value is excluded from evidence. Label both index and row/column so the learner can reason about the mask.

Allow mask editing on any64 cells, plus binary edits on currently observed cells. Valid input changes recompute every dependent result and explanation; retained baselines keep their original inputs. All observed: output equals input and no missing-error denominator exists—show “no hidden pixels to score,” not NaN/100% completion. All missing: compute unconditional pixel marginals, which do not depend on placeholders. No arbitrary continuous pixel input into the binary model. On switching sources, load the actual thresholded values and reset the default mask; preserve no stale show the computed values and their cause.

Optional conditional sampling uses the same exact p(h|observed) weights, samples h, samples missing pixels, and restores observed cells. Show uncertainty across multiple whole samples rather than representing the mean as a unique true reconstruction. Cap16 displayed samples. The held-out true image is one observation, not proof that another plausible conditional sample is mathematically invalid.

Metric view: over all80 assessment inputs under the fixed half mask, MSE averages only2560 missing probabilities; threshold accuracy uses>=.5. Default exact11MSE.117704 and2112correct versus baseline.132506 and2028. Other model rows use their actual values. If the user edits an image/mask, show a local edited-case comparison; never relabel it as the original80-image assessment. Source186's individual correctness does not certify method quality.

Deferred checks: independently reconstruct loaded model scores/normalizer/completions from retained weights, compare all stored assessment outputs, probability sums, observed invariance, ignored-placeholder null, all/none masks, masked-target leakage, selected-pixel direction computed comparisons, categorical draw validity, lazy loading/CPU bounds, fixed shared scales, narrow-screen role order, keyboard masks and reset. These checks have not been performed in a browser.

## Precise fixture and scope details retained for implementation

**Reconstruction and likelihood fixture.** At input 00, A wins both: MSE 2.06096665e−9/NLL .6932379763; B MSE .81/NLL 4.605170186. Source: `investigations.input00_comparison`. Selecting 01/10/11 recomputes the metrics. B's reconstructed mean is [.9,.9] for every input, while error and likelihood vary. A global energy offset changes neither metric. Show probability bars over every visible state so another mode remains visible.


## Scratch/tool bridge presentation — 22 September 2026

Beside the new bridge show W(D,H) -> components_(H,D) with explicit visible/hidden labels. An editable hidden-bias changes transform probabilities and the normalized four-state oracle together; keep sampled next state separate. No pseudo-likelihood line may be relabeled exact log likelihood. Full code/downloads must share one canonical source.

The complete source and teaching explanation are already written in the manuscript and companion programs. Phase two implements the presentation and verifies actual behavior; it does not invent an omitted algorithm. Show code only when requested, load large code assets on demand, preserve exact source equality, and keep immediately visible numerical explanations usable without running Python in the browser. No learner-prediction entry or grading state is permitted.
