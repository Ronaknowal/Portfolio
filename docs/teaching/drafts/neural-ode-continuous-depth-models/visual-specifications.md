# Neural ODE visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Follow the field and the numerical solver.** Edit vector-field parameters, initial state, step/tolerance, differentiation route, augmentation and supported real measurements.
**See the consequence.** Show field arrows, accepted/rejected solver stages, current trajectory/error, derivative target and class output immediately or through bounded process steps.
**Decision connection.** Choose a solver/tolerance or representation from error, work and topology, distinguishing numerical approximation from the continuous equation.


Content phase only, 13 September 2026. Stable ID: neural-ode-continuous-depth-models. Read the entire manuscript, design and provenance before implementation. No figure or lab has been built in this request. Source calculation programs and saved actual fits are implementation inputs, not browser evidence.

## Shared semantic and interaction contract

State and derivative have the same dimension; a derivative is change per unit time/depth, not the next state. Label physical time only for a genuinely temporal example. Iris uses learned representation depth. Use distinct visual grammar for a vector field, numerical stage, accepted trajectory, class readout, and observation jump. Never flatten every mechanism into a text-console box.

Show the initial state and current numerical outcome immediately. Edits to field, initial condition, method, steps or tolerance recompute a bounded solution; show pending versus current state if needed. Step/play/pause exposes solver stages and accepted/rejected work, without collecting a prediction or hiding a result.

Explain each current outcome from the field, numerical approximation or chosen observation, and suggest a meaningful parameter change. Show the actual difference from a labeled baseline. Independent practice can assess reasoning separately; no answer submission gates the solver.

Use code-native SVG/HTML/canvas as appropriate. Supply keyboard/numeric alternatives to dragging and a table equivalent for quantities encoded spatially. Titles, legends, color-independent styles, sufficient contrast, focus, concise live status and non-hover access are mandatory. Motion is learner-triggered, with instant state changes under reduced motion. At narrow widths stack coordinated panels; allow a labeled inner table scroll instead of whole-page overflow. Preserve axes/scale rather than shrinking text beyond usability.

No browser training, remote model loading or unbounded automatic solver loop. Load only this lesson and the selected small model/fixture when needed. Programmatic numeric computations run on Run; stepping a completed trace only selects stored points. Use a bounded worker if profiling shows a main-thread issue, with cancel/stale-result protection. Do not ship all twelve fit curves, source data and author metadata eagerly. Retain them as attributable downloads and derive compact runtime assets with hashes.

## Inline figures at the point of explanation

| ID | Location and geometry | Exact content and accessible meaning |
| --- | --- | --- |
| O01 | Opening: input → field/solver → readout | Distinguish one shared learned field from repeated calls. The path denotes feature change; text names available inputs and outputs. |
| O02 | §1: exact scalar decay and tangent step | z0=1, rate−2, h.1. Euler.8, exact exp(−.2).818730753. Draw a same-time endpoint residual. |
| O03 | §2: classical RK4 stage construction | Four formula-defined arrows at tentative locations, weighted final move. Stage points are not four sequential accepted states. Rotation curves/errors from calculated-inputs.json. |
| O04 | §3: accepted/rejected timeline | Worked rates[−1,−100], initial[1,1], T.4; two tolerances give58/4 versus144/5 accepted/rejected. Widths reflect time, not arbitrary equal-width steps. All attempt values retained. |
| O05 | §4: discrete tape versus continuous augmented solve | Match manuscript adjoint signs and column-vector shapes; name objective, terminal state and accumulator limits. Memory labels are components, never constant total RAM. |
| O06 | §4: four gradient references | Worked scalarθ−.7,z0=1.2,target.4,T1.3,n4. Common gradient axis plus clearly labeled absolute residual inset; preserve tiny RK4 differences. |
| O07 | §5: order-preserving tracks and added dimension | Exact1D order theorem with assumptions; augmented field[0,x²] and linear threshold. Coarse Euler order reversal is explicitly an approximation counterexample. |
| O08 | §6: actual twelve-run results | All validation curves and assessmentCE/counts from study-results.json, split90/30/29, seeds13/37/61, actual counts15/671/179/251. No fabricated smoothing/CI/timing/benchmark ranking. |
| O09 | §6: full real input-to-probability path | Worked source64,seed37, full4D/6D states, fixed standardization, head/softmax. Projected state path is labeled projection; coordinate traces expose omitted dimensions. |
| O10 | §7: continuous propagation plus observation jumps | Worked times[.2,.9,1.3],values[1,−.5,.8]. Query1.6, output.205575705; omission.310852828 versus zero.279567954. Query alone never changes hidden state. |
| O11 | §8: matrix-exponential patch transport | A[[.2,2],[.4,−.1]],T2, volumeexp(.2), densityexp(−.2), mass invariant. Rademacher estimates2.5,−2.3,−2.3,2.5 are individual noisy trace probes. |
| O12 | §9: pair paths versus marginal field | Pairs0→2/2→0 meet at t.5,x1 with targets±2 and mean0. Label this finite local regression illustration, not a complete density-flow fit. |

These figures remain inline even when an investigation explores a related mechanism. Add a small diagram for a derived explanation if it improves learning; no fixed figure quota or requirement to turn every optional application into another lab.

## O-I1 — Follow the field

**Fresh question and inputs.** Rotation matrix A=[[0,−1],[1,0]], initial[.6,.8], endpoint1.2, Euler four steps. “Will its radius increase, decrease or stay fixed compared with the exact path? What changes if we use classical RK4?” Worked preset is initial[1,0],T1 from §2. Show the current output.

Expose all four matrix entries in[−3,3], two initial coordinates in[−2,2], endpoint .1…2 and integer steps4,8,16,32,64. A paired method choice compares Euler/classicalRK4 at the same steps or same NFE, with explicit labels. For arbitrary2×2 matrices compute a trustworthy matrix exponential, including repeated/complex eigenvalue cases; do not invert an eigenvector matrix blindly. A bounded matrix-exponential series with scaling/squaring is suitable, verified against SciPy. If an input exceeds a numerical bound, report inability instead of plotting Infinity.

**Representation.** Vector arrows on a labeled equal-aspect state plane; exact curve, numerical path and discrete step points use different line styles. Select one step to reveal tangent or four RK stages. A radius/error strip below the plane shows numerical differences. Autoscale all compared trajectories jointly; a zoomed residual is separate and named. Include state/time/value table.

**Checked behavior.** calculated-inputs.rotation contains exact fresh/worked outcomes for4/8/16/32steps. Worked four-step Euler radius1.12890625, RK4 .9999932713091974; errors.13066074649445 and.000032531826420384. Fresh values are different and must be consumed from their exact records. Nulls: zero initial state under a linear field stays zero; zero matrix leaves any initial state unchanged; restoring inputs restores outputs. Rotational exact norm remains one for fresh initial; a changed general matrix need not conserve norm.

**Feedback and checks.** Explain Euler's multiplicative radius drift, RK4 convergence and differing NFE. Do not generalize radial growth to arbitrary matrices. Bind each computed result to the current matrix, initial state, endpoint, steps and method/comparison. Verify exact references, derivative arrows, stages, equal-aspect axes, endpoints and zero cases at1e−10; port convergence fixtures rather than merely two matching implementations. Full Reset restores the fresh inputs and their immediately visible result.

## O-I2 — Spend an error budget

**Fresh controls.** rates[−2,−50],initial[1,.3],T.4, relative tolerance.01, absolute tolerance relative/100, initial proposed step.1. Tightening both tolerances tenfold immediately displays the resulting step/error comparison. Both rates are editable in[−100,−.1]; initial coordinates[0,2],T.1…1; relative tolerance choices.1,.01,.001. State the coupling of absolute to relative tolerance instead of presenting it as an independently changed control. Initial step may vary .01…min(.2,T).

Use the exact Euler/Heun controller in ode_calculations.py: two calls per attempted step, RMS of componentwise scaled differences, accept≤1, safety.9, exponent−1/2, factor clamp[.1,5], endpoint-clipped step. Ratiozero permitsfactor5. Bounded10,000attempts andminimumstep1e−14 must produce explicit local failure states. Failed attempts do not advance state or disappear from the accounting.

**Representation.** A chronological accepted-interval strip with rejected proposals stacked above their start; two component trajectories and a local normalized-error strip share time. Select an attempt to see before/Euler/Heun/error/accepted. Retain complete computation but virtualize the attempt table and draw dense marks efficiently; do not silently discard rejected steps. Separate final error versus exact diagonal exponential from the local controller ratio.

**Fixtures.** Fresh tolerance.01 gives41accepted,3rejected,88NFE; tolerance.001 gives113accepted,2rejected,230NFE. Worked rates−1/−100 give58/4/124 and144/5/298. Zero both initial coordinates gives exactzero and no rejection; changing only requested sampling/query markers leaves this solver's integration unchanged. A stiff-slow-manifold adjacent optional figure uses actual SciPy RK45/Radau measurements, not this Heun controller. Labels distinguish NFE,njev,nlu and no wall-clock claim.

**Feedback.** Tightening reduces permitted local discrepancy and changes proposed/accepted steps; rejected count need not increase monotonically. Identify whether global error decreases in the actual run, rather than promising it for every input. Same active-input invalidation/reset requirements as above. Phase two verifies all attempt states and final metrics, acceptance threshold, rejected no-advance, exact reference, endpoint closure and clear failure/cancel behavior.

## O-I3 — Which loss are you differentiating?

**Fresh setup.** θ=0.3, z0=0.8, target=1, T=0.7, three steps. Ask for the gradient sign and whether the exact continuous and finite-program gradients must agree. The worked setup is θ=−0.7, z0=1.2, target=0.4, T=1.3, four steps. Controls: θ in [−2,1], z0 and target in [−2,2], T in [0.1,2], steps 3/4/16/64, Euler or classical RK4.

**Computation.** Use z'=θz and half squared terminal loss. Exact z=z0 exp(θT), with gradient (z−target)Tz. Euler's amplification is R(u)=1+u. Classical RK4 uses R(u)=1+u+u²/2+u³/6+u⁴/24, where u=θT/n. Differentiate the finite recurrence analytically, propagating state sensitivity step by step; this avoids division when R=0. Compare central differences with ε=1e−6.

The numerical continuous backsolve starts from the chosen numerical endpoint, a=endpoint−target and g=0. Integrate [z',a',g']=[θz,−θa,−az] backward with classical RK4 and negative step size. The retained scalar example is autonomous; its shifted-clock implementation must not be copied into a general time-dependent backsolve.

**Display.** Show an initial/terminal trajectory lane, the named loss, four gradient lanes and numerical residuals. Include reconstructed z0 in the backsolve lane. At 64 steps use precision-aware labels and a residual inset instead of claiming exact equality because rounded values look identical.

**Checked contrasts.** Fresh three-step Euler prediction is 0.9800344; its gradient is −0.0128008246464. The exact continuous gradient is −0.00902093665863 and the numerical continuous backsolve gives −0.01369687552596. Fresh three-step RK4 gradient is −0.00902095491354. The complete grid is in calculated-inputs.gradients.

If the target equals a method's actual endpoint, that finite loss gradient is zero; the exact continuous gradient need not be zero. At θ=0, Euler/RK4 forward states are constant and state sensitivity is Tz0. With z0=0, the parameter gradient is zero even for a nonzero target: changing θ cannot move the zero state.

**Feedback and verification.** Name the objective each number differentiates. Same sign is not equal gradient, and finite-difference parity does not verify an unrelated continuous objective. Add an optional ungraded inverse-conditioning strip for rate −20 and endpoint perturbation 1e−8, showing actual amplification exp(20) and recovered state. Compute the comparison from the complete current inputs. Check central-difference agreement within 2e−8 on recorded cases, analytic identities, reset and keyboard access.

## O-I4 — Lift the middle out

**Fresh inputs.** Points [−2,0,1], depth 0.4, threshold 0.5, desired labels [1,0,1]. Display selected points immediately, then change only depth to 0.7 and inspect the comparison. Permit three ordered distinct inputs in [−3,3], depth [0,1.5] and threshold [0,10]. Labels stay attached to point identities. If point editing changes their order, sort the display geometry explicitly while preserving identity.

**Mechanism and display.** Initial state is [x,0], field is [0,x²], exact state is [x,tx²]. Readout is one iff y>threshold; equality is class zero. Show a two-dimensional lift with three full tracks and a synchronized scalar y/readout strip. An optional one-dimensional mode uses exp(−2t)x and a single threshold. Explain the specific readout limitation without claiming arbitrary nonlinear one-dimensional readouts are impossible.

**Expected behavior.** Fresh depth 0.4 gives y=[1.6,0,0.4], selected=[1,0,0]. At depth 0.7 these become [2.8,0,0.7] and [1,0,1]. Worked inputs [−1,0,1] at depth one are selected [1,0,1] by threshold 0.5. Changing the threshold changes decisions but no trajectory; zero depth leaves every added coordinate zero.

The coarse-solver inset uses z'=−2z and one Euler step h=1, mapping [−1,1] to [1,−1]. The exact map preserves order by multiplying by exp(−2). Label the inset as an invalid inference from a coarse approximation.

**Checks and feedback.** Test full-space/time interpretation, the equality boundary, fresh/worked/null cases, and recomputation on threshold or depth edits. A projected crossing is not proof about full-state trajectories. Supply keyboard point editing, a numeric track table and the shared reset contract.

## O-I5 — Edit a measurement, keep the model

**Fresh task.** Validation source 70, seed 37, Neural ODE, four classical RK4 steps. Show its four actual raw measurements and withhold the species label until after Run. Ask for the effect of adding 0.6 cm to petal length on versicolor probability. Worked source 64 is solved inline; validation source 109 supplies another legitimate case. Do not turn assessment samples into a hidden hyperparameter-tuning control.

**Input and model controls.** Edit all four raw features within the original CSV's observed column ranges, printed in centimeters. The named ±0.6 cm interventions must either use the recorded valid sources or explicitly extend the local editable interval with a hypothetical-input label; never silently clip. Select ODE/augmented ODE, seed 13/37/61, Euler/RK4 and steps 4/16/64. Each selector loads the corresponding actual weights. No browser training.

**Exact model.** Apply the fixed 90-fit mean/SD from study-results.data. Append two zeros only in the augmented model. At each stage concatenate depth to state, apply affine→16-unit tanh→affine derivative, and integrate by the selected exact finite formula. Apply the three-output affine head and stable softmax. Retain every bias and weight. The four-dimensional model has 179 parameters; the augmented model has 251.

Store the initial and every completed fixed step. Both models were trained at four RK4 steps; other numerical choices are inference interventions, not independently trained models. A projected path is optional and must not replace the full coordinate traces.

**Representations.** Coordinate the raw measurement editor, fixed standardization bridge, four/six state-versus-depth panels, optional projected trajectory, class logits/probabilities and a before/after difference panel. The fresh probability shift is small: a separately named probability-difference axis makes it visible without exaggerating the main class bars. Do not assign botanical meanings to evolved learned coordinates.

**Author fixtures.** calculated-inputs.real_input contains 18 source/model/edit cases, each with four method/resolution variants. For source 70, seed 37 Neural ODE, clean versicolor probability is 0.9837685256539058; adding 0.6 cm gives 0.9846357263014778, a slight rise; subtracting gives 0.9692078296783093. The augmented model gives 0.9920369997259414 clean and 0.976560050806133 after the positive edit, a fall. Both retain the versicolor decision. Preserve the opposite probability directions and unchanged classes. The worked source 64 Neural ODE changes from 0.9337388812 to 0.6744000390. Full states and derivatives are retained.

**Nulls and feedback.** Restoring exact input/model/solver restores outputs. Selecting a different displayed projection changes no inference. An unchanged class does not mean an unchanged computation. More steps refine a numerical approximation but need not improve classification. Petal-length derivatives are per centimeter; do not confuse them with standardized-coordinate derivatives.

The independent NumPy/PyTorch maximum logit/state error is at most 8.9e−16 over author-recorded cases. For the JavaScript port, target 1e−9 for logits/states and 1e−10 for probabilities, investigating near ties. Check every recorded case plus a genuinely new arbitrary input against the native full model. Verify low/high bounds, units, all model selections and stable softmax. Loading/retry failures preserve edited features; stale model-switch results must be discarded.

**Live state and reset:** row identity, raw measurements, model, seed, numerical method and step settings jointly define the current result. Editing one recomputes the bounded model and related geometry. Reset restores the original case; a baseline keeps its original configuration for an honest comparison.

## O-I6 — Separate observation from query

**Fresh controls.** Times [0.1,0.7,1.4], values [1,−1,0.5], query 1.6 and initial state zero. Between observations h'=−0.5h; at an observation h+=0.7h−+0.3x. Ask for the effect of deleting the middle observation versus replacing its value with zero.

Allow two to eight observations, strictly increasing times in [0,3], values in [−2,2] and query in [0,3]. The initial version rejects duplicate observation times with a local explanation, rather than silently choosing an order for simultaneous updates.

**Execution and display.** Use exact decay between each available observation, apply the update, then decay to the query. Observations after the query are unavailable and never assimilated. A query at an observation time is defined after its update. Additional query markers inspect the same causal history and cause no update.

Draw exact decay segments, vertical observation jumps, an observation-presence strip, a query-availability strip and a table of before/after states. All positions use actual times. The synthetic mechanism has no medical or physiological unit labels.

**Fixtures.** Fresh query 1.6 gives 0.0712615841177507; omitting the middle gives 0.234922588781007; replacing it with zero gives 0.20516349595832306. Query 1.0, before the final observation, gives −0.12431048108694495. Keep the §7 worked case separate.

Altering the future observation at 1.4 cannot change the state at query 1.0. Adding query markers cannot change any trajectory. Observed zero can change state, while omitting the update is a different intervention. An explicit no-observations null mode returns zero at every query; the ordinary editor's minimum of two observations only preserves a useful comparison.

**Feedback and checks.** Explain elapsed-time propagation and discrete information updates separately. Moving an observation changes retention intervals and availability; moving a query does not retrospectively alter measurements. Compute the comparison from the complete current inputs. Inspecting another point on the same stored history does not invalidate it.

Reset restores the fresh problem. Verify jump ordering, query equality, future availability, zero/missing nulls and numerical values within 1e−10. Provide keyboard time/value editing and a bounded timeline; no autoplay.

## Phase-two continuation

Build only on an explicit finish request. Derive semantically named topic-owned models/assets, an executable download bundle and lazy imports. Verify calculation ports, full displayed programs, named nulls and live comparison contracts. Perform independent correctness and learning-experience review, keyboard/mobile/browser/render/loading/error checks, and application integration. Record actual source-bound evidence and both phases separately. Preserve this prepared packet until its implementation and retention decisions are complete.


## Scratch/tool bridge presentation — 22 September 2026

At the existing solver/gradient comparison add an API contract strip: requested output times versus accepted internal steps; fixed Euler h versus adaptive tolerances; direct versus adjoint derivative path. Label RK4 tableaus separately. Initial state, rate and endpoint edits update analytic and numerical comparisons together. New package results remain unmeasured until phase two captures them.

The complete source and teaching explanation are already written in the manuscript and companion programs. Phase two implements the presentation and verifies actual behavior; it does not invent an omitted algorithm. Show code only when requested, load large code assets on demand, preserve exact source equality, and keep immediately visible numerical explanations usable without running Python in the browser. No learner-prediction entry or grading state is permitted.
