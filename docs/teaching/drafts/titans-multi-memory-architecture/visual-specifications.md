# Titans: visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Write into memory and inspect later reads.** Edit key/query/value cards, rate, momentum, decay, request tokens and chunk size; step bounded writes or continue/reset request state.
**See the consequence.** Show weight/update state, residual and gradient terms, current query output, gated topology and anchor/current-gradient comparison.
**Decision connection.** Choose write timing, state isolation and chunk semantics from the outputs they can affect, rather than from the word memory alone.


Phase one, 13 September 2026. Consume the full [manuscript](lesson.md), [design](design.md), [mechanism records](mechanism-results.json), [real observations and results](rental-results.json), [programs](memory_mechanisms.py) and [author checks](author-checks.json). This document specifies the website work; no visual is claimed rendered or browser-tested yet.

## Placement and teaching vocabulary

Keep the calm dark/gold site direction. Geometry carries meaning: observed positions are small dated/token rows; stored fast weights are parameter grids; momentum is a separate same-shaped grid; queries/keys are arrows; returned values are output coordinates. Prefix vectors have a distinct labeled row, rather than a padlock that could imply security or literal permanent fact storage. Gold highlights the currently inspected operation; sign is encoded with +/− labels as well as diverging color. Reserve familiar terms for their exact roles: loss, gradient, momentum and update cannot all be relabeled surprise.

Show the current memory, key/query state and read result immediately. Use live edits and bounded write steps to expose how updates affect later queries. The layout may introduce advanced momentum/decay controls with their explanatory section, but never hides an output behind a learner answer.

## Inline reading figures

| ID / placement | Composition and consequence | Model / caption / boundary |
| --- | --- | --- |
| memory-lifetimes / §1 | Six observed positions, a moving last-two bracket, arrows into a fixed-size weight grid, a separate shared prefix row. Advance illustration to a seventh position as a static before/after pair. | Qualitative structure. Recent positions are individually addressable representations; compressed weights are not literal archived tokens. Prefix parameter values stay fixed, contextual outputs need not. |
| associative-write / §2 | Key arrow in 2D, two weight cells, answer ruler from 0 toward target 2. Highlight zero second-coordinate contribution. Beside the second write, show orthogonal and correlated keys on the same axes. | Exact trace: final weights [1,2] versus [2.5,1.5], old-key read 1 versus 2.5. The vector coordinates are constructed feature values, not measured semantic embeddings. |
| update-decomposition / §3 | Signed retained-momentum arrow + new gradient-step arrow + shrink arrow = actual parameter change. Weight and momentum occupy separate lanes. | Scalar worked state w=1, S=.2, g=−2, rate .1, retention .5, decay .1 yields S′=.3, w′=1.2, Δw=.2. Do not label S′ as Δw when decay is nonzero. |
| nonlinear-shapes / §4 | Key d_k → W1(H×d_k), bias H → SiLU → W2(d_v×H), bias d_v → read. Mark backward paths to both weights and biases. | Structural shape diagram; hidden-unit count does not label storage slots. Show a small affine XOR contradiction separately if helpful. |
| architecture-wiring / §5 | Three separate small diagrams: MAC retrieved rows enter attention, MAG parallel branches join at a gate, MAL memory precedes attention. State crossing time is distinct from data moving through depth. | Schematic based on paper §§4.1–4.3, equations21–31. MAC includes attention-driven write and post-write read/gated output; no claim that a plain sum is MAC. Segment-level notation is not an unambiguous autoregressive mask specification. |
| outer-gradient / §6 | Slow parameters above; old fast state → gradient → new fast state → query → task loss below. Backward arrow crosses the gradient operation; alternate inference legend keeps only local write graph. | Exact scalar branch: rate .25 → w′=.5 → output1.5 → outer loss .125, derivative −3. A smaller picture may show nonlinear finite-difference agreement as numbers, not a fabricated learning curve. |
| forecast-timing / §7 | Seven observed count tiles end before the target date; target weekday enters from calendar. Save prediction, reveal actual, compute loss, write, move to next date. | Real data with explicit assumed day-end availability. Current target may not enter an earlier forecast. Use the actual first seed3 record and date; the date labels in all views agree. |
| state-bands / §9 | Separate 1.5 GiB fast-state band .75 GiB local-KV band; named but unquantified shared-weight/workspace categories beside them. A second exact full-cache row is48 GiB. | Hypothetical stated shapes/dtypes; label payload bytes and GiB conversion. No hardware-fit assertion or visual implying unspecified categories are zero. |

## Investigation A: which read does a write change?

**Question.** How do key overlap, residual direction and retained updates affect another query? Place beside §§2–3. Begin with a one-output, two-input linear memory at zero and two editable key/value cards. Default cards are (1,0)→2 and (0,1)→4. Query is (1,0). Rate .5, momentum0, decay0. Show the first write worked; Show the current computed result and its contributing terms immediately. Weight cells, 2D key/query arrows, current residual, gradient and output ruler share one state.

**Live read difference:** display the old-query read before and after the selected next write, including its signed difference and the key-overlap term. Use absolute tolerance 1e−10 for equality. Direct key/value/query or update-parameter edits recompute the same bounded model immediately; no radio prediction or reveal gate is included.

**Entity edits.** Key/query coordinates and target accept bounded finite decimals in [−2,2] and [−5,5]; rates0–.5, momentum0–.8, decay0–1. Permit selecting either card next; at most12 writes per investigation. Store the starting state and event history so Back reconstructs the exact prior state. Edits are a staged configuration; Apply starts a fresh investigation and clears derived process state/history. No stale label or score may survive an edit. Reset restores the initial values and current comparison.

**Consequence.** Draw the key arrow, then gradient/update on the two weight cells, with actual before/after reads. Overlap itself appears as a labeled value. On narrow screens, stack the coordinate view above weight/momentum lanes; numeric tables remain accessible. Input values and targets stay beside the consequence, rather than disappearing above a giant console.

**Actual fixture evidence.** `associative_trace` executes the orthogonal and correlated cases; author checks verify exact arrays. Separate checked controls include zero key with target10 and zero gradient, momentum-only change, full decay leaving momentum, and zero rate with zero momentum/decay preserving weights. Offered finite controls use the same analytic model; corner/extreme input behavior and reset/Back must be exercised in phase two. Nonfinite output stops the trace and requests a reset; never silently clamp values or mislabel divergence as learning. Zero key is valid here and must not be rejected just because some normalized models exclude it.

**Transfer.** Pick a changed query and second target that reverses the old-read change; Show the current computed result and its contributing terms immediately. Offer practice1 after the lab. For the core explanation, do not turn on all momentum/decay controls at once: reveal them when §3 introduces them.

## Investigation B: request state through a gated topology

**Question.** Which outputs can an input change, and which state must be carried or reset? Place after the worked gated block in §5. `gated_sequence` is the authoritative complete teaching model: identity projections, two-position causal window including current token, optional fixed prefix [1,0], W and S initially zero, inner rate .5, retention .5, no decay; observed x→x write then read; output attention⊙tanh(memory). This is a MAG-topology specialization, not a published trained checkpoint. Do not silently add a convex mixing gate or label it MAC.

**Starting entities.** Four editable token vectors [[1,0],[0,1],[1,1],[1,−1]]. Separate lanes show attention rows, softmax weights, resulting vector, W/S updates, memory read and final coordinate product. Click a token to inspect its exact stage. Show the two cached recent rows and persistent row with different provenance labels.

**Live manipulation:** edit tokens, split boundary and continue/reset choice, then inspect every affected output and carried state. The current result updates from the specified initial state, with an immutable baseline if pinned. The explanation distinguishes new writes, retained state and request isolation.

**Controls.** Token values in [−2,2]; four to eight tokens; prefix checkbox; enable/disable new writes; split after any nonfinal position; Continue with state / Start fresh. Every edited scenario restarts from its specified initial state, so toggling writes cannot leave an accidental old momentum buffer. Reset restores tokens and prefix; Back changes the inspected step without mutating the underlying run. No stochastic redraw. There is no cross-request shared mutable W/S/cache; a named Request A and Request B have distinct state objects.

**Visible result.** First default output [.462117,0]; second [0,.232671]; prefix removal leaves first output equal but changes second to[0,.309508]. With zero writes and fresh zero memory, the product gate gives all-zero outputs even though attention is active: show both branches so the learner sees why. Continuation exactly matches an uninterrupted run; a fresh suffix differs. Future-token perturbation leaves earlier outputs identical. Author checks exercise these contrasts, the null and separate-request isolation; phase two verifies the same arrays in the actual browser model and control behavior.

**Follow-up.** Ask the learner to name why two outputs can agree despite different attention weights, using the duplicated first prefix row. Distinguish that numerical null from whole-model equivalence. Practice8 diagnoses the two wrong read schedules without requiring an unsafe implemented option that leaks data.

## Investigation C: where was this gradient evaluated?

**Question.** Why can two chunking rules give different final weights? Place in §8. Use scalar key1, initial w0, targets1 and2, rate .5, no momentum/decay. Draw two dependency graphs with identical target cards: current-state gradient edges in one; both gradients attached to the fixed starting anchor in the other. Updating the running weight and evaluating a gradient must appear as different operations.

**Live comparison:** display sequential and anchor-gradient final weights, their difference and both residual/evaluation-state ledgers. Step advances the actual write; changing chunk size or rate recomputes both. The equality cases remain visible, without an answer entry or Step/Reveal gate.

**Controls and consequences.** Editable numeric entities, chunk size1 or2, step and reset. The default produces1.25 vs1.5; the changed practice case produces .875 vs1.0. Chunk size1 and zero rate are verified null cases. A table lists residual, evaluation weight and resulting running weight beside the graph. No claim about GPU timing, training throughput or accuracy follows from this two-step model. Show the affine composition formula as a static deeper explanation; a separate animation for every scan stage would add work without a distinct learning need here.

**Transfer.** Find a target pair for which the two methods agree, explain why, then change one target to break that equality. The program's simple fixed two-target reference can be generalized with the same declared equations; verify offered inputs in phase two rather than treating the saved default as a universal fixture.

## Real-study quantitative figures and accessible output

Use `rental-results.json` directly; never hand-author a smoothed trend. Figure1 is two panels of paired MAE dots: three individually labeled seeds, frozen/adaptive joined, previous-day horizontal baseline, previous-week baseline optionally dashed with a label. Show the same relevant MAE range across panels; integer rental ticks, readable two-decimal tooltips/exact table, no log transform. Make the fact that some adaptive runs lie above the simple baseline visible. Include all seeds and both periods rather than presenting only seed19.

Figure2 shows actual counts, frozen seed3 predictions and adaptive seed3 predictions for a bounded date interval. Optional date-range selection chooses from already retained data; it does not retrain or reselect a model. Plot actual units, date labels, line styles and exact selected-day values. To show adaptation history, pair a chosen day with its prewrite residual, gradient and update values. These are different scales: separate aligned panels or inspection labels, not a misleading shared axis. State the one-run measured status and date/split protocol in the caption. No causation is assigned to an unusual date merely from the curve.

All data can be lazily imported with this topic and figure. The 731-row source is a downloadable offline artifact, not an app-wide eager import. Render a bounded point window and pause work when hidden. No browser Python/PyTorch training or arbitrary-code execution is needed; the small analytic labs run only on valid input changes. Store arrays in one derived state and memoize by current valid configuration rather than running an animation loop. Request cleanup stops pending computation on unmount; no whole-curriculum visual bundle.

## Phase-two acceptance

Implement the specified figures beside their explanations; read through without operating a lab to check inline teaching. Then exercise each live comparison, meaningful edit, null case, reset/Back and invalid value. Compare every exact default array and real metric to retained evidence. Inspect desktop and narrow widths with keyboard and reduced motion, including the paired-dot baseline crossing and nondefault lab states. Confirm lazy topic loading and no hidden training or unbounded render loop. Formal independent content/implementation review, browser screenshots and integration checks remain explicitly outstanding.


## Scratch/tool bridge presentation — 22 September 2026

Alongside the existing two-loop figure show an actual dependency arrow through a write-rate tensor. Toggle graph retention as a clearly labeled derivative-contract comparison: live forward values can agree while outer gradients differ. Preserve the real rental forecast before actual-count reveal; that reveal is scientific event timing, not a learner-prediction gate. Display value state and retained differentiation history separately.

The complete source and teaching explanation are already written in the manuscript and companion programs. Phase two implements the presentation and verifies actual behavior; it does not invent an omitted algorithm. Show code only when requested, load large code assets on demand, preserve exact source equality, and keep immediately visible numerical explanations usable without running Python in the browser. No learner-prediction entry or grading state is permitted.
