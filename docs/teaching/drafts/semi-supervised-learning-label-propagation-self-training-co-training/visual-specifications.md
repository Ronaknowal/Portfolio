# Semi-supervised learning — phase-two visual and investigation contract

Content prepared 2026-09-12. No production diagram, React component, SVG or lab implementation exists in this packet. Preserve the instructional roles below while adapting composition to the site's design system. Do not replace these mechanisms with one repeated parameter panel or synthetic accuracy graph.

## Shared interaction and evidence requirements

Each investigation starts unsolved. Recorded prediction is initially null, with no preselected correct answer. Use a prediction control plus an explicit commit action; compute and expose outcomes only after commit. Store the prediction and comparison against a canonical signature of the full active input (data values, known labels, graph/view choices, parameters, tie policy, algorithm mode and query). A signature change clears prediction, computed trace, feedback, reveal state and any stale success badge. Keep unchanged informational panel state separate from mathematical input state.

Predict → commit → run/step → inspect → explain → construct a changed case is the learning loop. Feedback names what changed and which input caused it. A wrong prediction is useful evidence; do not replace it with the right answer silently. Reset restores default inputs and unset prediction; replay resets only the execution trace while preserving a committed prediction for unchanged inputs. Presets are convenient checked contrasts, not the whole exercise: learners can edit graph edges/labels, point coordinates, or paired categorical rows.

All controls have visible labels, keyboard operation and focus indication. Numeric entry is an alternative to dragging. Class identity uses text and shape/border as well as color. Announce one user-requested step's result in a polite live region without continuously reading the entire graph. Reduced motion disables interpolation without removing any state. At narrow widths place the visual above its numerical contribution table and wrap controls; do not require horizontal panning to reach a result. Every picture has a persistent caption, legend and equivalent table/text explanation. Hints and solutions begin closed and have independent state.

This is small deterministic computation. Resolve only the requested investigation, cache unchanged traces, cap inputs, cancel obsolete calculations and load code only with the topic/investigation. No heavy ML libraries, server fitting, background corpus downloads or animation-frame retraining. Data and scores are dimensionless unless marked otherwise. Measured banknote counts come exclusively from the retained checked results; illustrative fixtures are explicitly authored.

## Inline figures at the point of need

### F1 — ownership of a label (§1)

Three horizontal compartments with count badges: training pool 320 = 6 observed + 314 sealed; development 80; test 80. The six-label total is only fitting supervision. Model-generated labels remain inside the training pool with outlined provenance badges; solid observed-label badges never change style after a promotion. Arrow from development to candidate choice, then a single choice to test reporting. No training arrow returns from test. On mobile make three stacked compartments. Text equivalence is the same arithmetic and role list.

### F2 — one input distribution, two target rules (§1)

Two aligned panels share exactly the same authored points: x = [-2.2,-2,-1.8,1.8,2,2.2], y-position for display [0,.2,-.2,-.1,.1,0]. Panel one target labels [0,0,0,1,1,1]; panel two [0,1,0,1,0,1]. No computed density estimate or learned accuracy. Caption explains that the same inputs admit different targets; the local coordinates are a constructed counterexample. Use labels/shapes plus a point table. Do not claim the second target is predictable from x in an unobserved independent-inspection population: it is one possible realized labeling.

### F3 — hard-clamped flow (§2)

A–B–C–D line with endpoint labels fixed; equal-width unit edges, B/C score bars and incoming contribution arrows. Static initial/equilibrium pair immediately after the equations, before I1. Endpoint shape distinct from inferred score node. Fractions 1/3 and 2/3 accompany decimals. Electrical inset reuses the same graph with conductance and voltage terminology; no ornamental unrelated circuit.

### F4 — raw evidence is not a probability (§3)

Side-by-side raw score bars and normalized two-class readout for the α=.8 chain from checked-results.json → graph.spreading. Shared row IDs A–D. Raw B = [.254408788998,.149652228823], row mass .404061017821; readout B = [.629629629630,.370370370370]. Raw bars have numeric scales appropriate to their observed maximum, not a forced probability axis. Readout axis 0–1. Mark α=0 unknown rows as zero input evidence and readout unavailable. Display S row sums [.707106781187,1.207106781187,1.207106781187,.707106781187] in a separate small row, not as class probabilities. Caption distinguishes S, P and F.

### F5 — provenance across self-training rounds (§4)

Original seeds, accepted batch and remaining pool in separate lanes. Every accepted item displays origin round; arrows return to a newly fitted model, not the previous model. Use the prototype trace for exact coefficients and I2 interaction. No transition from a pseudo-label badge to an observed-label badge.

### F6 — two views, one row (§5)

Paired categorical row cards aligned between two rule boards. An offer travels from donor prediction through the row's recipient-side feature to the recipient rule. Initial rows 0/1 solid; inferred labels outlined with donor/round. Show row6 conflict with two distinct proposed values and an abstention state. This static first-round explanation precedes I3.

### F7 — actual banknote promotion audit (§6)

Two coordinated count panels rather than a dual-axis performance curve. Bars for newly accepted and wrong: accepted [48,106,60,18,11,6,3,0], wrong [0,12,28,14,10,6,3,0]; correct segment = accepted−wrong [48,94,32,4,1,0,0,0]. Cumulative accepted [48,154,214,232,243,249,252,252]. x = promotion round 1–8; y = specimens, integer ticks including zero. The final totals are 252 accepted / 73 wrong / 62 remaining, in addition to 6 seeds. Caption: hidden benchmark targets opened after fitting for explanation; not available to a real unlabeled promotion rule. Pair with the actual candidate development table and selected model's single test result. No interpolated accuracy values, confidence intervals, timing rankings or implied population generalization.

## I1 — edit the neighborhood and explain the influence

Placement: §2 after shortcut calculation; §3 reuses the same input in optional soft-spreading mode, not a second unrelated lab.

Editable entity: undirected weighted graph with 2–10 named nodes. Edit/add/delete edges and set each node's observed label to 0,1 or unknown. Weights finite in [0,5], zero removes an edge; no self-edge, symmetric assignments. Coordinates affect layout only and are explicitly excluded from the metric; edge weight is the data. Layout editing does not invalidate a mathematical prediction, but changing any edge/label does. At least one unknown node is needed for the main task; no anchors is a valid null state, not a validation error.

Prediction: select a target unknown node and predict the direction of its class-1 score versus the last committed comparison input: increase/decrease/unchanged/unavailable, plus optional numeric estimate. Initially compare the line without B–D to the editable shortcut input. To prevent a solved preset from passing as independent practice, require one learner-authored edge/label change and a new prediction for the transfer question. For a newly created graph without a baseline, ask which side of .5/unavailable the target will reach.

Hard mode: Dii=sum W; Pij=Wij/Dii for positive degree. Initial unknown scores .5; observed scores fixed 0/1. Synchronous update from previous state, restore observed labels. Identify connected components first. An unanchored component is flagged unknown and excluded from the hard solve; do not regularize it secretly into a label. Solve Luu f = Wul y using a stable small-system solver for the displayed equilibrium. Step trace remains explicit; never label a partial iterate equilibrium. Compute residual Luu f−Wul y and display precision to 6 decimals, with checked fixture agreement within 1e−9 in double precision.

Soft mode: one-hot Y on known nodes, zero otherwise; S=D−1/2WD−1/2 with inverse degree set zero for isolated nodes. α finite in [0,.99], default .8. Start F0=Y. Update αSF+(1−α)Y; show raw and normalized views separately. Readout unavailable when row sum≤numerical tolerance for true zero evidence; isolated observed node retains (1−α)Y and normalizes to its label. Zero-support detection should use reachability as well as numerical magnitude so a tiny reachable score is not declared structurally unanchored. Equilibrium solves I−αS. Iteration cap5000 and residual tolerance1e−10; if cap reached show incomplete convergence. Readout ties .5 are displayed as tied, not necessarily a class0 winner.

Checked contrasts in checked-results.json → graph:

- line AB=BC=CD=1; A0,D1: B1/3,C2/3; first five displayed updates recorded.
- shortcut BD2: B5/7,C6/7, so B flips relative to .5.
- separated BC0: B0,C1.
- unanchored E–F1 with both unknown: hard solution not unique; soft zero rows and no classification.
- spreading line α.2 and .8: full raw/readout arrays and traces retained. α0 null is Y with unknown rows unavailable; do not describe as hard propagation.
- S row-sum and contraction .99^100/.99^500/1375-step fixtures are retained, not simulated wall time.

Feedback explains the incoming contributions for the selected node and whether the evidence changed or disappeared. Removing B–C is a contrast; unanchored E/F and α0 are nulls. Phase two must additionally check edited degree-zero nodes, contradictory anchors in a strongly connected component (soft compromise/hard fixed), one class only, disconnected labeled components, ties and cancellation after rapid edits. Confirm keyboard/narrow/text equivalence and no stale prediction across mode switches.

## I2 — move an unlabeled point, move the next boundary

Placement: §4, immediately after the authored prototype walkthrough.

Editable entity: 2–10 observed 1D points containing both classes, and 0–20 unlabeled points; point coordinates in [-10,10], up to2 decimal places. Default observed [(-2,0),(2,1)], unlabeled [-1,0,1,3]. Provide editable numeric rows and drag handles; query point defaults1.25 and is not included in fitting. Add/remove points independently. Class labels editable for observed points, but both classes must remain represented; invalid input produces a specific repair message and no computed outcome.

Model: class prototype = arithmetic mean of all currently assigned points. For each remaining input x compute logits −(x−μc)^2, subtract maximum and softmax. Accept confidence≥τ; default τ=.8, range[.5,.99]. Class tie selects0 for deterministic promotion; show the tie explicitly. Promote a synchronous batch then refit; accepted pseudo-labels are retained, never revised in this simplified variant. Stop no candidates/pool empty/max20rounds. Even after pool empty compute and expose final prototypes. Original and pseudo origin lists remain separate.

Prediction before run: boundary movement left/right/unchanged and query label change yes/no/tie, with optional reasoning. Baseline is the observed-only model for the current authored inputs; show its boundary but hide final. If μ0=μ1, scores tie at every x and there is no unique separating boundary: use unavailable/tied prediction choice. If class ordering reverses, the midpoint remains the boundary but class shading reverses accordingly.

Visual: real line with observed class shapes, outlined unlabeled circles, prototype triangles and a dotted boundary. Active offers are outlined in a separate batch lane until the refit action. A linked score table lists x, distances, probabilities, acceptance and origin; rejected rows remain visible. Unit is authored coordinate, not a physical measurement. Feedback names which accepted coordinates moved which mean. Contrast markers compare original and edited input trajectories only when both computations belong to their recorded signatures.

Checked fixtures in checked-results.json → teaching.centroid_self_training:

- Default: first offers x−1→0,1→1,3→1; μ=(−1.5,2); boundary .25. Next x0→0 at confidence≈.8519528; final μ=(−1,2), boundary .5.
- Move only3→9: final μ=(−1,4), boundary1.5. Query1.25 is class1 with original final model and class0 with changed final model.
- Null U=[0,0], same observed points and τ.8: no offers; μ=(−2,2), boundary0 unchanged.

These are transparent authored mechanism fixtures, not fitted banknote results or calibrated confidence claims. The banknote experiment remains separate; do not silently substitute a prototype learner for logistic regression there. Learner transfer: create at least one distinct input list, predict a prototype movement and explain whether an observed validation label would support that movement. No precomputed lookup table for arbitrary edited points.

Phase two verifies every checked trace, stable softmax, empty U, duplicate x, threshold equality, max-round final-refit behavior, overlapping class means, class-order reversal, fast-edit invalidation and keyboard operation. Calculation fits in a small pure function; animate only user-requested state transitions, never refit at animation-frame rate.

## I3 — paired views with explicit donor chains

Placement: §5 after the first two worked transfers and before the complete program.

Editable entity: 2–20 paired category rows with optional observed class0/1; default seven rows from lesson and cotrain-categories.py. Each view category is a trimmed string1–20characters; preserve case and treat equality literally. Learner can change both views, add/delete rows and change an observed label. A copy-view1-to-view2 action demonstrates redundancy. No anchors is a legitimate null; blank categories are input errors.

Reference algorithm: cotrain-categories.py is the exact author program, not a production dependency. Each learner has an independent label array initialized with the same observed labels. It learns a category rule only from a singleton set of received class values; conflicting categories abstain. Both learners produce predictions before any offers are applied. When both predict different classes for a row, defer both. Otherwise donor may send a known prediction only to a recipient whose label on that row is unset. Recipient uses its own feature representation. No pseudo label is called independently verified. Eight-round cap, stop on no offers; arbitrary larger fixtures display cap status if reached.

Prediction: choose an initially unresolved category/row and predict0,1,conflict or remains unknown; commit before donations. Default target green in view1. Then edit the bridge row and commit another prediction. Freeform reason prompts “Which donor can reach this category?” without showing the solved path first.

Visual: left/right rule boards, paired row table between them, directional offer arrows. Step subphases: fit rules → propose synchronously → resolve conflicts → receive → show new rules. Recipient labels have donor/round/origin links. Reciprocal confirmations are visibly different from first access to an unseen category; they are not counted as independent evidence. Table fallback lists full offer records and current rule dictionaries.

Checked fixtures in checked-results.json → teaching.co_training:

- Baseline seven rows: offers per round2,4,2,0; row6 red/square conflict each round. New rules green0,orange1,triangle0,hexagon1.
- Changed bridge row2 blue/triangle: green becomes1. Exact trace retained.
- Duplicate views: row indices3,4,5 remain unknown in both arrays; there is no new green/orange rule. The formerly red/square conflict disappears because the paired data changed, not because it was adjudicated.
- No anchors: rules empty, zero offers in first round, all unknown.

Feedback follows one exact donor chain, reports unresolved conflicts and counts distinct newly learned category rules separately from duplicate row confirmations. No percentages describe the categorical rule's calibration. Predicting an unanchored unknown is a correct reasoning outcome.

Phase two verifies trace identity to the retained program, order-independent batch proposals, preservation of observed labels, conflicting category abstention, no-anchor null, duplicate-view null, row removal/reset, eight-round bound, no stale donor chain after edits, narrow stacked layout and accessible transfer text.

## Downloads, accessibility and phase-two acceptance

Publish topic-scoped CSV/provenance and all three complete example programs using existing downloadable-code patterns. All manuscript-relative links must resolve after integration; author check files remain preparation evidence and need not load in the browser. Render the full categorical source as optional explained code if that better serves the page, while preserving a working download. The complete graph and banknote blocks must match their checked companion programs.

Phase two owns browser rendering, formula typography, mobile/keyboard checks, contrast and focus, numerical port equivalence, lazy imports, resource-link checks and final independent accuracy/pedagogy review. This packet's equations and traces were author-checked, but no production or independent-review claim is made.
