# Neural ODEs: implementation, representations and review route

The complete prepared manuscript is retained: thirteen sections, six distinct investigations, twelve original figure placements, all worked calculations, the complete study and solver bridge, nine independent practice tasks and their hints/solutions, and primary reading references. The library execution statements now report the actual local torchdiffeq0.2.5 execution; no separate official example suite, JAX/Diffrax campaign or GPU timing is claimed. Historical author-checks.json/check_author_packet.py remain dated packet evidence with their original manuscript hash; they were not misrepresented as current implementation checks.

## Concept map and visual placement

| Learning hurdle | Local explanation and actual representation | Manipulation, null and interpretation |
| --- | --- | --- |
| Derivative versus next state | Opening input→state→shared field→readout geometry; §1 scalar decay curve and tangent endpoint error segment | The worked z′=−2z step0.1 shows0.8 versus exp(−.2), connecting a local rate to approximation error |
| Depth versus physical time; residual sharing | §1 residual formula and separate-weight/shared-field diagram, reused evaluations connected to one field | Changing solver resolution changes computation without inventing more trainable field parameters |
| Euler and classical RK4 | §2 four actual tentative stage arrows plus separate weighted accepted move; exact rotation table retained | `ode-fixed`: edit all four A entries, initial coordinates, endpoint and interval budget; compare same intervals or same field calls |
| Continuous reference versus numerical path | Equal-axis field/trajectory plane and separate enlarged signed residual curves; exact endpoint/radius/work table | Matrix exponential handles complex, repeated and real eigenvalues; zero matrix/state are actual nulls; select any RK4 interval |
| Adaptive local error and rejected work | §3 first ten actual attempt intervals, followed by fresh `ode-adaptive` selectable attempt window, complete Euler/Heun/start/ratio table and exact-versus-accepted coordinate traces | Change rates, initial state, tolerance, initial step and endpoint; rejection consumes two calls but keeps time/state; zero initial state has zero error |
| Stiffness and cost | Exact cos(t) example, fast perturbation equation and executed SciPy RK45/Radau counts in the manuscript | Method counts retain Jacobian/factorization caveat and no unmeasured wall-clock claim |
| Direct differentiation versus continuous adjoint | §4 discrete dependency reverse lane and explicit (z,a,g) backward equation lane; worked gradient main axis and magnified residual plot | `ode-gradients`: fresh .3/.8/1/.7 example, rate/state/target/T/steps/method edits; direct tangent of the actual solver, finite differences, exact continuous and approximate backsolve remain distinct |
| Reverse conditioning and API choices | §4 exponential reconstruction amplification plus full runnable torchdiffeq bridge | Same Euler resolution compares states and two gradient types; classical versus library3/8 RK4 is explicitly distinguished |
| Unique flow order and readout constraints | §5 three actual noncrossing scalar tracks, movable final threshold and desired outer/middle labels | Invertibility alone is not a classification impossibility; the linear scalar threshold assumption is named |
| Augmentation and numerical artifacts | `ode-topology` vertical lifted trajectories of immutable A/B/C identities, explicit threshold and final coordinates; additional exact-versus-coarse-Euler order-flip diagram | Edit three strictly ordered inputs, depth and threshold; equality maps to0, depth0 is a null, depth.7 solves the fresh declared pattern |
| Real learned field and controlled study | §6 all twelve measured validation curves, paired seed-specific assessment counts and parameter counts; worked row64 traces in every hidden coordinate | No altered split, refit, fabricated accuracy or selected favorable example; actual12 saved checkpoints replayed on all declared data roles |
| Actual inference from raw measurements | `ode-iris` raw cm→fit-only standardization→field evaluations→actual readout geometry, all4/6 coordinate traces, optional equal-axis projection, original/current probabilities and enlarged signed sensitivity | Fresh row70 by default; rows64/109 alternatives; all six field checkpoints, three interval counts, Euler/RK4, all four feature edits and ±.6cm petal contrasts; edited inputs receive no new species label |
| Observations versus queries | §7 worked decay arcs with jumps and availability; `ode-observations` real timestamp geometry, exact applied-event table and causal readout | Edit2–8 timestamps/values, omit one or zero its value, add query markers, move query; future observations are unavailable, exact event-time query reads after update; invalid order is explained |
| Density and trace estimation | §8 exact transported parallelogram, area/density reciprocal values and all four Rademacher probes; rotation toggle | Individual probe error and conserved probability mass are distinguished from a stochastic exact-trace claim |
| Flow matching | §9 opposing conditional straight paths meet; current marginal velocity at the crossing and quadratic loss are visible | Edit predicted velocity; zero minimizes the conditional mean-square loss though neither target is zero; sampling still solves a field |
| Physics, events, controls and noise | §10 Hamiltonian level-set directions, equation-residual marks, irregular input control path and stochastic/event-reset schematic | The local prose states which object/objective changes and which specialist lesson owns deeper implementation; none is claimed equivalent to ordinary dropout |
| Diagnosis, implementation and practice | §11 numerical/model/data/optimization diagnosis; §12 nine changed tasks and closed hints/solutions | Exercises change decay, grids, local budgets, gradients, readouts, observations, density, conditional velocity and evaluation protocol |

## Computational and publication design

`neural-ode-models.js` owns importable deterministic scientific calculations. Classical RK4 returns all tentative states/derivatives; adaptive Heun records every attempted step and only appends accepted states. The matrix exponential uses a two-by-two centered decomposition with a near-repeated series branch. Differentiating the fixed program integrates the tangent recurrence, avoiding division by an amplification factor that can be zero. The continuous backsolve uses a distinct RK4 augmented state. Observation updates have explicit availability and no invented imputation.

The body is generated from the entire canonical manuscript through the existing authoring-only renderer. Tilde code fences are normalized during generation; complete study/bridge source remains visible behind ordinary disclosures. The eight public downloads are explicit learner programs/data/provenance; authoring design, manuscript, specs and verification receipts are not published. Full fitted-model/calculation results remain optional downloads. The body imports only relevant metadata, three source flowers, small worked traces and actual history values. Six small individual model files load one at a time near the lab viewport, keyed to architecture/seed with abort, friendly failure and retry. Model switches cannot show stale weights as current.

Each plot has a named focusable local scroll region and minimum nominal340px width. Every custom SVG has its own nominal width and bounded focusable scroll. Equal-axis vector/state views preserve metric geometry. Small residual plots declare a power-of-ten scale; unscaled precise tables remain. Shared number controls use the opt-in stable-error footprint and neutral CSS. No first18 lesson source or shared numerical helper was changed.

## Executed native and scientific checks

`check-neural-ode-native.py` executes the unmodified arithmetic program in an isolated temporary directory, avoiding fixture mutation. It imports the complete actual study model and replays all12 selected checkpoints on fit/validation/assessment, checks validation-only selection and fit-only standardization, executes the installed torchdiffeq bridge and records its state/gradient outcomes. Additional SciPy matrix-exponential cases include Jordan, rotation, near-repeated and zero fields. New cases across all six actual field checkpoints include full trajectories, class probabilities and48 native input derivatives per centimeter. Maximum CPU thread allowance is2; the original study voluntarily uses1.

`check-neural-ode-models.mjs` compares12,972 scalar values: exact/numerical paths, every adaptive attempted decision,19 gradient configurations including zero Euler amplification, topology/readout boundaries, causal observation omissions and zero updates, density/probes,72 prepared and12 additional real-model solver cases. The largest recorded error is6.16016571086675e−10, from a numerical derivative comparison; model traces use2e−12 tolerances. All public programs/data and per-model objects match the intended canonical sources. No native fit was repeated without a new scientific need.

`check-neural-ode-render.mjs` bundles and server-renders the complete initial page, checking all six investigations, finite displayed initial values, import resolution and absence of React warnings. This establishes source execution, not painted geometry, keyboard behavior, network recovery or mobile browser acceptance.

## Author learning-experience review

1. The opening separates the first-pass path from labeled later density/flow-matching/structured branches.
2. Data and numerical caveats have a clear local home; no runnable code prints warning paragraphs.
3. A flower measurement becomes a learned trajectory and returns as an actual editable Iris prediction.
4. Every investigation has immediate current output and direct entity edits; explicit zero/unchanged cases are available. No prediction entry, answer gate or Run-only constraint remains.
5. Actual geometry, work intervals and state traces are used; independent error scales expose small differences without modifying the main path. Browser perceptibility remains root-owned.
6. Connections include residual→Euler, field→solver, direct→continuous gradient, unique flow→readout assumption, observation→jump, divergence→density and conditional paths→mean field.
7. Complete scratch and standard-tool programs are retained, with shape/state/time/gradient semantics explained locally. Verification boilerplate stays outside learner programs.
8. Nine tasks change the problem and supply reasoning, values or experiment criteria; optional solutions remain separate from live lab outputs.
9. No screenshot/browser claim is made in this author source receipt. Root owns those checks and captures.
10. The concept map above records the middle and advanced transitions as well as the introduction, with numerical worked states and topic-specific representations. All twelve prepared figure contracts and six investigation contracts were implemented; additional diagrams clarify order artifacts and structured variants.

Remaining before final delivery: independent review, root browser checks, integration and central ledger closure.

## Independent review closure — 27 September 2026

- Observation lab now has an explicit no-observations mode that passes empty observation lists to the unchanged pure model. Every entered event has a presence/availability/decision row; available, future and absent values have separate markers and text. Empty input remains exactly zero, independently checked in retained numerical evidence.
- Adaptive lab shows every attempt on a time-versus-normalized-error-ratio strip, with a threshold at1, accepted/rejected markers and selection linkage. Height is explicitly log10(1+ratio) with original-ratio ticks; all zero ratios remain representable. The bounded solver exception is caught locally while settings/reset remain usable.
- Gradient lab now connects all current accepted scalar states to terminal residual and squared-error loss before showing derivatives, with stated state-axis bounds and exact values.
- Raw Iris measurements and solver settings are owned above the loading boundary. Model/seed changes and retry preserve them; source change and explicit reset restore measurements deliberately. No altered input is judged by its old species label.
- An API time-grid figure beside the complete torchdiffeq bridge separates requested output times, eight fixed Euler steps and actual adaptive Heun accepted times. It names the illustrative controller and does not label it dopri5; fixed-step resolution, adaptive tolerance, differentiation route and classical versus3/8 RK4 are explicit.

Full SSR rerun passes without warnings. Numerical engine and retained native evidence are unchanged. Browser state-persistence, failure recovery and painted geometry remain the root integration check.
