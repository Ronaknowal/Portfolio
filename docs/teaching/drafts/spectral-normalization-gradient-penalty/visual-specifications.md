# Spectral normalization and gradient penalty: visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Control stretch and inspect where a penalty acts.** Edit matrix entries, normalization method, power-iteration steps, critic/input values, interpolation points and margin geometry.
**See the consequence.** Update singular stretch, effective matrix, derivative paths, sampled gradient penalties and local decision distances live.
**Decision connection.** Choose or diagnose a constraint by what it actually bounds and where it was evaluated; a sampled penalty is not a global guarantee.


Content-only handoff, 13 September 2026. These are implementation contracts, not claims that the visuals or labs have been built. Consume the complete manuscript, two programs and their recorded outputs in phase two.

## Interaction and evidence contract

Use different representations for geometry, derivative coverage, computation graphs and real observations. Match the site's typography and controls, but do not force these mechanisms into the same text-output panel. Static inline diagrams carry essential explanations; investigations add deliberate entity edits and a live computed readout.

Each investigation follows the live exploration contract above: current results are visible immediately, valid entity edits update all linked views, and comparisons explain the mechanism. Reset restores the declared inputs and recomputes their result. No prediction or answer-submission state is retained.

For exact small computations use absolute tolerance1e−6 or relative1e−4 where appropriate, displaying the tolerance. Frozen float32 model comparisons accept1e−5. A “no change” category must use the appropriate numerical tolerance and explain the analytical invariant when available. A near-zero denominator, zero direction, invalid coordinate or unsupported operation receives a localized error with retained last valid inputs, not a fabricated zero score. Do not Calculate and explain rounding noise as a conceptual difference.

Provide numeric keyboard editors for every draggable point/vector/matrix. Label coordinates, matrix dimensions and units; keep visible focus and44px targets. Do not rely on color alone. Every SVG/canvas view has an equivalent labeled data table plus concise causal text. At narrow widths stack aligned input→operation→output panels; preserve equal geometric axis scales and readable values instead of shrinking a wide plot. Respect reduced motion; animation is optional stepwise explanation, never the only way to inspect a state. Announce one concise result after a meaningful settled change.

Load lesson assets on demand. Small 2×2 calculations update continuously during valid dragging and numeric edits. Real models are tiny frozen forwards, with a maximum256 displayed generated points and400 recorded points; do not run GAN fitting, download models or import Python/torch into the browser. The stored41×41 grid is for the selected final critic; display selected vectors or a coarse arrow subset, not1,681 permanent DOM arrow components. A canvas raster may show the scalar field with an accessible sampled table. Benchmark the actual phase-two implementation before deciding whether a worker is needed. Do not retain hidden model instances/listeners after unmount.

Graph data must identify exact calculation, actual recorded observation, actual trained output or user-edited counterfactual. No fabricated FID/time curves or extension of measured points into unsupported regimes. Original lessons' named hardware timings are not accepted evidence.

## 1. Critic and generator derivative paths

Placement§1. Two aligned diagrams share the same real/fake score surface. Critic phase: real x and detached G(z) enter f; only phi changes. Generator phase: z→Gtheta→fphi; theta changes, phi is fixed, input derivatives through f remain active. Show detach as an explicit stopped derivative edge, never as deletion of a value edge.

The scalar worked strip uses real0, theta2, f(x)=−x: scores0/−2, generator loss2, gradient1, descent .1→1.9. A wrong-sign branch is labeled a diagnosis, not a second valid algorithm. No saturation or BCE widgets are necessary to explain the sign.

Static fallback narrates exactly which parameter is optimized and which derivative survives. A compact code-graph exercise may ask the learner to repair a detached edge, but does not replace the numerical investigations below.

## 2. Matrix stretch, spectrum and power-iteration investigation

Placement§3. Static solved view W=diag(3,1), unit circle, transformed ellipse, singular-value bars3/1, Frobenius√10, normalized ellipse1/(1/3). Draw circles/ellipses with equal x/y scale and a marked unit input vector. The spectral bar represents maximum amplification, not a typical input. Translations can appear as a separate bias control; translating both output points leaves their distance unchanged.

Fresh challenge starts W=[[2,1],[0,1]], u=[1,1], target1. Matrix entries editable in[-5,5]; u coordinates[-2,2] with nonzero norm. Exact singular values2.28824561127 and.874032048898 are recorded. Learner edits a coefficient or direction, observe whether one-round normalization has true norm below/equal/above1 and optionally gives an estimate. Only show u→Wᵀu→v→Wv→u and sigma estimate, effective matrix, exact norm and spectrum. Step limit16; a small table retains each round's estimate and true normalized norm.

Implement exact2×2 largest singular value from eigenvalues of WᵀW with a numerically stable discriminant; compare with recorded fixtures. The bound is exact for this finite matrix up to floating-point error. Estimate uses the actual normalized vectors. If Wᵀu or Wv is zero, stop with “this direction gives no usable estimate”; let the learner edit/reinitialize. For W=0, exact normalized result is defined as zero and direction undefined, with no misleading division or unit-norm claim. Near-zero cutoff1e−12 is disclosed.

Contrasts: generic diagonal3/1 first estimate2.86356421265527 leaves norm1.04764544365437; orthogonal u=[0,1] stays at estimate1 and true norm3. Slow-gap diag(1.01,1) generic start after8 rounds estimate1.00575293791739, true norm1.00422276875612. A cached [1,0] direction for changed diag(1,3) can miss the new leading axis. These are selectable explanatory traces, not auto-completed practice.

Meaningful null: positive rescale of every coefficient by2 under exact unit normalization gives the same effective matrix. Scaling only one coefficient generally changes it. Offer exact-normalize versus cap-only for W=.2I, where results differ; make method choice part of fingerprint. A bias edit changes absolute output position but not pairwise stretch. Explain zero/negative-scale cases separately rather than overgeneralizing positive-scale invariance.

## 3. Composition ledger and convolution stencils

Placement§2 composition and§4 convolution. Static composition diagram follows both singular directions for diag(3,1/3) then diag(1/3,3): product bound9 versus actual identity norm1. Residual identity edge contributes to bound1+.5=1.5. These diagrams are a reason for a loose bound, not an error badge.

Convolution view links input cells, sliding windows, output sums and full matrix. Solved kernel[1,1] on input length3 valid stride1 gives A=[[1,1,0],[0,1,1]], kernel norm√2, full norm√3 and normalized norm1.22474487139. Length4 stride2 yields disjoint rows and normalized norm1. Circular length4 stride1 yields normalized norm1.41421356237. Exact source: sensitivity-results.json.convolution.

Fresh editable kernel[1,2], input[1,−1,2], valid stride1. Numeric cells within[-3,3], input lengths3–6, stride1/2, padding valid or circular explicitly defined. Build the actual matrix by applying the same kernel-placement rule to basis vectors; preserve the distinction between correlation orientation and convolution reversal. The chosen convention is cross-correlation y_i=sum_j k_j*x_(i*stride+j), with circular wrap only in that mode. Do not support arbitrary padding semantics until tested.

The live readout shows whether normalization of the stored kernel will make the full operator norm≤1, and which outputs change after editing a selected input cell. Use a small eigen/SVD routine for AᵀA or a vetted exact-small linear algebra implementation; do not label a finite power estimate “exact” here. Fresh [1,2] valid length3 has AAᵀ=[[5,2],[2,5]], full norm√7, kernel√5 and normalized norm√(7/5)≈1.18321595662. With stride2,length4 the normalized norm is1. These are analytical acceptance fixtures even if generated during implementation. The fresh matrix arithmetic can also be checked directly.

Display actual output change and the operator bound as separate readouts. An input that misses the worst direction can have smaller stretch than the bound. Moving that input or changing the matrix updates both quantities while preserving their distinct meanings.

## 4. Where a gradient penalty sees the function

Placement§5. Static solved f=x+4ReLU(x−1), probes−.5,0,.5 shows slope1 and zero target-one penalty while slope5 outside. Shade sampled locations as points, not the whole region. The sampled maximum never receives a “global bound” label.

Move an actual probe or knot and display its derivative, the mean penalty and the relevant bound immediately. Explain why a finite set of probes can miss another region; the live curve and probe positions make that limitation visible.

Default sampled gradients are1 and penalty0; moving last probe1.5→2.5 makes the strength2 mean penalty32/3. Moving knot2→3 while all probes remain below leaves measurements unchanged although f differs elsewhere. That is the meaningful unobserved-region null. Show the exact derivative and squared contribution of each probe before averaging.

Toggle target-one, one-sided or zero-centered with visible formulas; do not call every zero-centered location R1/R2. A separate sampling-location schematic identifies real versus generated samples for R1/R2 and real–fake interpolation for WGAN-GP. A target change recompute current outputs. For the linear worked w=(3,4),strength2,rate.1 show gradient(9.6,12.8),neww(2.04,2.72),norm3.4,penalty11.52. Fresh vector[2,−1],strength1,rate.05 allows a learner-controlled penalty step, with closed derivation help. It is explicitly penalty-only, not a simulated GAN optimizer.

Batch-coupling inset draws the2×2 Jacobian[[.5,−.5],[−.5,.5]] and the cancellation in the sum-of-scores gradient. Contrast diagonal per-example dependence with off-diagonal dependencies. Do not substitute a general BatchNorm numerical formula for this exact simplified centering example.

## 5. Real ink profiles and frozen GANs

Placement§7. Explain each coordinate by showing an actual8×8 digit image with left/right halves and sums divided by512. Dataset image, class label, source_id and role remain available. The model generates two-dimensional profiles only; no generated profile is rendered as an invented reconstructed digit.

Use calculated-inputs.json:400 measurements,394 profile groups,241fit/79development/80assessment; bootstrap256; each method×seed has256generated points, declared history, complete G/effective D weights and grid. Default plot shows fitting profiles and seed11 SN output, with optional reserved-role overlay clearly labeled. Equal axes0–1 preserve geometry. Score color scale changes explicitly when switching critics; do not imply raw critic scores share a calibrated unit across methods. Gradient arrows can have normalized direction length only if their true norms are separately shown; otherwise use one shared scale.

First chart:64-direction averaged empirical W1 at declared steps1/100/300/600, using stored development observations. Connect points as a guide, with no unseen interpolation measurements. Final assessment table reports every seed and the bootstrap baseline. Second chart shows actual assessment gradient norms, sampled-grid maximum and matrix product bound; distinguish samples from an upper bound and approximate normalization from exact normalization. Never combine them on an unlabeled common “quality” scale.

Choose a latent point in [−3,3]² and edit either coordinate. Show both frozen generators' actual outputs and the changes from the original point, with their exact retained parameters. The comparison is computed on the current input, not a guessed direction or a new fit.

For seed11 SN, the recorded edit changes(.40882012248,.26194134355)→(.37287068367,.19582489133). Other seeds need not have the same direction. Exact numerical result checks uses each selected saved function, not this one example. A result near the tolerance boundary is reported as indistinguishable numerically.

Coordinate symmetry activity: swap both latent coordinates and the first-layer columns. Observe whether any generated output changes, and show immediately the exact cancellation. Swapping only the latent coordinates is the contrasting intervention; the saved256point maximum changes range .06138–.51810 across the nine models. All joint swaps are zero in independent double inference. Do not pretend a label-only edit constitutes the main meaningful null.

Browser critic forward uses saved effective layers with leaky slope.2 and final scalar output. For fresh gradient probes, compute the2D Jacobian through the piecewise-linear layers, respecting derivative convention at zero; report kink ambiguity when relevant. Validate against saved grid/input gradients in phase two. Grid pixels use recorded output for each corresponding selected final model; arbitrary weight edits require recomputation, so omit weight sliders from this activity. A latent-only edit does not modify the critic or historical metric curve.

## 6. Decision margin geometry

Optional§8, after the first-pass route. Draw logitsF=(x1,x2), decision boundaryx1=x2, point(2,0), shortest perturbation(−1,1), distance√2. Show the gap2 and pairwise slope√2 separately from the joint map norm1. The perturbation circle touches a tie; points strictly inside preserve the decision.

Fresh numeric editor point(1.5,.25), matrix initially identity, bounded2×2 coefficients[−2,2],bias[−1,1]. The gap1.25 gives radius1.25/√2≈.883883476483, retained in sensitivity-results.json.fresh_investigations. Compute the exact row-difference norm for the two-class linear model and distance gap/norm. If the selected class does not win, state no positive winning certificate; if row difference is zero, distinguish equal logits from constant strictly positive gap. Compare the joint-spectral upper bound with the direct row-difference bound, never downgrade a bound by treating a power estimate as exact.

Display the current radius and its geometric derivation immediately as inputs change. Display raw-input preprocessing only when its concrete transform is supplied. This remains a two-class linear calculation, not a certification tool for arbitrary networks. Text alternative derives the same perpendicular distance and names the open-ball versus boundary distinction.

## Phase-two completion

Implement only the necessary topic-owned models/figures/labs with lazy imports and retained semantic filenames. Reuse the nine frozen fits; do not refit to obtain prettier curves. Check exact small operators, changed input/null cases, independent forward agreement, derivative paths, zero matrices/directions, norm thresholds and all live recomputation/reset paths. Test visible numerical geometry and accessibility at desktop/mobile sizes, unknown/loading errors and route context. Formal independent correctness and learning-experience review, browser verification and production integration remain deferred. No diagram/lab count is a substitute for these learning outcomes.

## Precise fixture and scope details retained for implementation

**Probe coverage.** Default f=x+4ReLU(x−2), probes −.5,.5,1.5, strength 2. Editable knot [−2,3], extra slope [−.9,5], 2–6 probes in [−3,4]. A probe exactly at the kink has no ordinary derivative; show the reason. The analytic global Lipschitz value is max(1,abs(1+extra)), initially 5. Moving probes updates the sampled penalty, not that global statement.

**Frozen latent edit.** Initial pair [−.7,.4]→[−.7,1.1], with model-specific outputs in each fit's `latent_intervention`. Edit z∈[−3,3]². All nine complete generators evaluate z@Wᵀ+b, ReLU twice and final sigmoid; show their actual current outputs. These are counterfactual inputs to saved fits.


## Scratch/tool bridge presentation — 22 September 2026

Beside §6 show three separately labeled stores: original weight, effective normalized weight, and power-vector buffers. A mode/access-count change updates the state trace immediately. Connect the R1 extension to the existing penalty-location view: real inputs versus interpolants, target zero versus one, and λ versus γ/2. Preserve the existing exact linear gradient oracle; label any future package output by its actual captured version.

The complete source and teaching explanation are already written in the manuscript and companion programs. Phase two implements the presentation and verifies actual behavior; it does not invent an omitted algorithm. Show code only when requested, load large code assets on demand, preserve exact source equality, and keep immediately visible numerical explanations usable without running Python in the browser. No learner-prediction entry or grading state is permitted.
