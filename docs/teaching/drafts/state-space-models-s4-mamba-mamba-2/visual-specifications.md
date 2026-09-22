# State Space Models — visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Change a state-space write and read.** Edit tiny system coefficients, impulse inputs, selective writes, distraction sequence, chunk boundaries and supported real trajectories.
**See the consequence.** Show impulse response, carried state, input-conditioned updates, SSD matrix entries and chunk equivalence live. Stepping exposes current recurrence arithmetic.
**Decision connection.** Decide which information needs selection or state carry and distinguish a mathematically equal scan from a different update rule.


Content-first packet; this document specifies phase-two implementation. No React, SVG, browser lab, production manifest or optimized GPU kernel has been implemented or verified by this authoring increment.

Canonical ID: state-space-models-s4-mamba-mamba-2. Proposed display title: State Space Models: S4 and the Mamba Family. Keep the stable ID. Place each visual beside its first explanatory paragraph; optional derivations remain clearly secondary. The prose is a complete manuscript, not visual placeholder copy to publish.

## Representation and accuracy rules

- Use the manuscript's post-update convention h[t]=Ad h[t−1]+Bd u[t], y[t]=C h[t]+D u[t], initial state h[−1]. Annotate continuous A versus discrete Ad; do not reuse one unlabeled A knob for both.
- Zero time/sequence axes where appropriate; dimensionless toy input and state units remain labeled. Physical continuous plots have declared arbitrary time units, not implied seconds from Libras timestamps.
- Every trace comes from the specified recurrence or retained record. Curves are computed mathematical illustrations, not speed benchmarks. Display negative coefficients with a zero-centered scale; SSD weights are not probabilities.
- Expose values as text and tables. Color supplements labels, directed arrows, solid/dashed differences and selection borders. Equal x/y scale for trajectory/phase-plane geometry. No perspective distortion, rainbow ranking or unexplained normalized importance bars.
- Static diagrams and their explanation remain useful without opening an investigation. Advanced panels can be collapsed; no mandatory animation or automatic simulation.
- Use semantic headings, figure/caption associations and readable SVG text with accessible names. At 360px content width, stack stages; provide a horizontally scrollable semantic table for matrices instead of shrinking labels illegibly. Avoid whole-page horizontal scrolling.
- Pure calculations stay local; import frozen arrays/results only when the related investigation opens. No browser training, tensor library, remote data request, entire-curriculum import, or eager mount of all labs. Cache by canonical inputs, cancel stale results and release per-lab retained arrays on unmount. Do not present an unimplemented spec as a working lab.

## Inline visual map

| ID / manuscript anchor | Form and exact content | Learning purpose / text equivalent |
|---|---|---|
| ssm-state-paths, §1 | Four-path dimensioned flow: old h[N]→Ad[N,N]; u[H]→Bd[N,H]; sum→new h[N]→C[J,N]; independent u→D[J,H] joins at y[J] once. Numeric scalar fixture Ad=.5,Bd1,C2,D.25,hprev1,u3 gives h3.5,y7.75. | Distinguish storage, evolution, write, read and feedthrough. Caption lists both output contributions 7 and .75. |
| ssm-discretization, §2 | Continuous zero-input e^(−t) from h0=1; sample Δ1 at times0…6, ZOH e^(−k), bilinear(1/3)^k. Next panel held inputu1/initial0 rises to1, ZOH1−e^(−k), bilinear1−(1/3)^k. Axes time and state, legends name actual methods. | Clarify Ad/Bd both change and method equality is not assumed. Static table first3 samples. |
| ssm-integrator, §2 | Accumulation rectangles Δ=.5, u[2,−1], initial3; state3→4→3.5. | A=0 is meaningful; no inverse required. Rectangle signed areas1,−.5. |
| ssm-impulse-ledger, §3 | Input[2,0,1,0], two-mode taps and direct path in mechanism-results.json/lti. Separate initial response band if enabled. Contributions aligned by input birth time and output column. | Each column's sum matches recurrent output; label direct path separately. |
| ssm-kernel-and-state, §3 | Kernel stems [.3125,.203125,.11328125,.0595703125] and state coordinates from lti states; output [1.125,.40625,.7890625,.322265625]. | Same map viewed through state or input history. No generic correlation line. |
| ssm-fft-padding, §3 | Four input cells and four kernel cells require at least7 transform slots; show zero padding and a dashed prohibited wraparound into early output. | A causal convolution must not become circular. Label convolution operation, not raw elementwise time-domain multiplication. |
| ssm-timescales, §4 | Computed decay traces a^k for a=.2,.8,.99, k0…100; inset first8steps. Half-life annotation ln(.5)/ln(a), label decay of one mode. | Multiple memory rates; no guaranteed neural recall claim. Avoid truncated y-axis exaggeration. |
| ssm-oscillator, §4 | A=[[-.2,−2],[2,−.2]], Δ.25, initial[1,0], no input; mechanism-results/oscillator states. Equal-axis shrinking spiral plus synchronized two-coordinate traces, time labels. | Complex mode as real rotation+decay. |
| ssm-conjugate-read, §4 | One complex z=x+iy and c=p+iq with partner conjugates; output cz+conj(cz)=2(px−qy). Small computed example c=1+2i,z3−i→cz5+5i,paired read 10. | Why imaginary parts cancel and factor2 remains. |
| ssm-polynomial-memory, §4 optional | On[0,2] f(s)=s; normalized basis1,√3(s−1); coefficients1,√3/3. One-mode approximation 1 and two-mode exact s; independent basis panels and a reconstruction sum. | Meaning of a memory coefficient and the declared ds/2 measure. |
| ssm-dplr, §4 optional | N4 HiPPO A/B and normal correction from mechanism JSON; show A+ppᵀ, unitary basis change to diagonal normal part, low-rank correction retained. Signed heatmap may supplement labeled algebra; no raw heatmap alone. | Normal-part eigenbasis is not direct diagonalization of original triangular A. Do not call finite S4D exact HiPPO. |
| ssm-fixed-delay, §5 | Four input arrows entering three-cell shift register, read cell3; fixture [2,5,−1,7,0]→[0,0,2,5,−1]. | LTI can do fixed delay. Marker-dependent gaps require additional selection. |
| ssm-selection, §5 | Signed input bars, g and1−g complementary bands, retained/write stacked contributions, state trace; checked gates/inputs below. | Distinguish withholding a write from retaining state. |
| ssm-mamba-block, §5 | Full Mamba-1 branches with shapes: width d→expanded D and gate D; feature causal depthwise conv/SiLU; BC[N],Δ[D]; scan state D×N; multiply SiLU gate D; output d/residual. | Full block versus core operator. Mark small experiment omissions without silently equating models. |
| ssm-affine-scan, §5 | Three maps(a,b) grouped in two parenthesizations, label composition order and same resulting map. Use(a1=.5,b1=1),(a2=.2,b2=3),(a3=.8,b3=−2), composite a=.08,b=.56. | Associativity of affine maps, not independence of outputs. Derive 0.8*(0.2*(0.5h+1)+3)−2=.08h+.56. |
| ssm-ssd-influence, §6 | Separate matrices CBᵀ,L,and Hadamard product; future structurally blank, diagonal L=1; chosen row dot with V. Exact source mechanism-results/ssd. | Signed unnormalized operator versus softmax attention. |
| ssm-ssd-chunks, §6 | Four-stage block matrix/state bridge storyboard. q2 second-chunk local[[1,2],[4.4,3.8]] plus incoming[[.75,−.25],[1.4,−.3]]. | Dense local work plus carried memory; changing chunk partition preserves map. |
| ssm-real-paths, §7 | First deduplicated fitting source row in class1,6,10 according to saved roles, from raw data;45points,ordered start/end and ordinal ticks. Labels correspond to raw names. | Real input and temporal order, no generated “representative” path. |
| ssm-training-flow, §7 | 45×2→45×16→2 residual temporal blocks→mean16→15logits→softmax/labeledCE; gradient arrows to exact parameters. | Training turns mechanism into prediction; sizes remain visible. |
| ssm-learning-evidence, §7 | Loss against epoch1…100 for all 4 saved runs, separate fit/validation lines. Mark selected epoch; table errors and denominators. Confusion view fixed class order 1…15 with counts and row-normalized recall toggle. | Actual measurements; no smoothed invented trajectory/ranking. Assessment curves must not appear per epoch because they were never recorded. |
| ssm-cache-counts, §8 | Two labeled rectangles/numeric formulas: 48 KiB state versus 12 MiB full-MHA cache, for the specified arrays; if common-axis bar used state small must remain selectable with text. | Quantitative state count, not runtime benchmark or all-memory claim. |
| ssm-mamba3-endpoint, §9 | Previous write/current write/old state separate contributions .5,.5,3 for λ.5; selector λ=1 changes to .5,0,6. | Three-term discrete update and initialization boundary. |
| ssm-mamba3-rotation, §9 | Bits[1,0,1,1], unit circle points[−1,0],[−1,0],[1,0],[−1,0], read odd parity 1,1,0,1; local π/2 practice separate. | Positive forgetting limitation correctly scoped. |
| ssm-mamba3-rank, §9 | N=P=2,R2 example B=I, X=I writes I (rank 2), versus a single outer product e1e1ᵀ(rank1); retained state 2×2 remains the same. | Arithmetic/state distinction; no unsupported latency bar. |

The block/projection diagrams are authored explanations. Equations are the source of truth; do not copy research figures without their permissions/attribution. Static toy fixtures are original and freely editable within the specified investigations.

## Common investigation contract

Each investigation follows the live exploration contract above: current results are visible immediately, valid entity edits update all linked views, and comparisons explain the mechanism. Reset restores the declared inputs and recomputes their result. No prediction or answer-submission state is retained.

Every substantive edit invalidates the old result and prediction and returns to the unsolved state. Reset restores the stated inputs and immediately displays their computed result. Switching a purely presentational plot/table view or stepping through an already solved trace does not invalidate the experiment. If a prior result is retained for comparison, label it “previous inputs,” do not present it as current feedback.

live comparison is explanatory, not a score assigned to prose. Say what changed, what stayed invariant and which update terms explain it. Use tolerances rather than exact float equality; show values/error magnitude and a reason. An arbitrary edited input can produce a null outcome—compute it rather than forcing a preferred result.

Keyboard alternatives exist for every drag: selected index and signed numeric field, add/remove where allowed, arrow movement, labeled reset and step. Announce completed comparisons using a short polite live region, not each animation frame. A semantic table includes the entire displayed numerical trace. Do not require color or hover to find the current step. Step/playback is learner-initiated, honors reduced-motion and pauses offscreen; static Next/Previous always works.

Compute only on explicit run or bounded debounced updates after a new Show the current computed result and its contributing terms immediately. State must belong to this open lesson instance. No cross-lesson singleton or localStorage persistence that restores a stale answer into a different fixture.

## Investigation 1 — system and impulse laboratory (§3)

**Learning question.** Which output contribution came from old memory, the current write or feedthrough? Do recurrence, direct convolution and FFT evaluate the same system?

**Editable entities and initial state.** Allow 1–16 signed input samples in [−10,10]; two diagonal continuous rates in [−3,0]; B and C coordinates in [−2,2]; D in [−2,2]; Δ in [.05,2]; and two initial-state coordinates in [−5,5]. Start with A=diag(−1,−2), B=[1,1], C=[1,−.5], D=.25, Δ=ln2, initial state zero, and input [1,−2,3,0]. This input differs from the worked example. An integrator preset uses A=0,B=1,C=1,D=0,Δ=.5,initial=3,input=[2,−1]. Presets restore named input entities and immediately display their computed results.

**Live comparison:** for an input edit, display the earliest changed output and every affected contribution; for an initial-state edit, display its carried contribution separately. Compute maximum discrepancy from the full current parameters and selected intervention. Explain why causality preserves earlier outputs and why initial state may affect the whole sequence.

**Exact calculation.** For each scalar mode, Ad=exp(Δa) and Bd=B·expm1(Δa)/a, with the exact a=0 limit ΔB. Update before reading. K[l]=sum(C·Ad^l·Bd). Add the initial response and Du once to the causal convolution. Pad the FFT to at least 2T−1. A direct computation copied into three columns is not an independent FFT comparison: implement a bounded FFT or explicitly label the FFT comparison as downloadable-only. Invalid empty or nonfinite fields block Run with a field-level message.

**Observable representation.** Connect signed input stems, both state coordinates, kernel taps, the input-contribution ledger, initial-memory response and Ch/Du components to an output table. Axes identify time index and arbitrary input/state/output units. The caption identifies the held-input model and initial-state convention.

**Checked fixtures.** The fresh default returns [.5625,−.921875,1.39453125,.4423828125]; recurrence, direct convolution and FFT agree in investigation-checks.json. Restoring the worked input [2,0,1,0] returns [1.125,.40625,.7890625,.322265625]. With initial state [1,−2], add C Ad^(t+1)[1,−2], beginning with .75. The integrator returns states [4,3.5]. C=0,D=1 returns the input regardless of internal state. Changing only the final input leaves earlier outputs unchanged. The zero-input, zero-initial-state null returns zero for any allowed parameters. Use 1e−10 float64 tolerance in tiny comparisons.

**Bounds and accessibility.** At most 16 times and two state coordinates; O(T²+NT) reference work. Stack views on narrow screens and expose a per-time semantic table. Every drag has a numeric field. Phase two must verify the contrasting/null/singular/initial-state/padding fixtures, T=1 and 16, independently computed forms, all input/model identity fields, reset, keyboard/mobile operation and finite-value errors.

## Investigation 2 — selective writes and distractions (§5)

**Learning question.** Can the chosen gate schedule retain a marked value through distractors, and why is closing a write different from stopping decay?

**Editable entities.** Start with [3,−8,5,−2] and marked flags [true,false,false,true]. Allow 2–16 signed values in [−10,10], add/delete, per-item markers and gates in [0,1]. Initial gates are .99 at marked items and .01 elsewhere. The constant comparison gate starts at .5 and is editable in [0,1]. Initial state is 0, editable in [−5,5]. The chosen read index starts at 2. An optional separate-retention mode exposes independent a and g in [0,1], explicitly labeled h=a·h_previous+g·u rather than the coupled exact gate.

**Live comparison:** show the selective-schedule and constant-gate states, their reads and absolute errors from the most recent marked value. Editing an actual event or write/gate parameter updates both from the same input, including ties and cases where neither is useful. Explain the retained versus injected terms.

**Mechanics and observations.** Evaluate h=(1−g)h_previous+g·u in coupled mode, or the independent recurrence above. Show retained and incoming contributions at each step, the state trace, target and absolute error. Use signed bars, marker labels, complementary g/(1−g) bands and an aligned table. A gate value alone is not a probability of remembering. Feedback connects the actual distractor contribution to the resulting error.

**Checked contrast and null.** The fresh selective trace is [2.97,2.8603,2.881697,−1.95118303]; the constant trace is [1.5,−3.25,.875,−.5625]. At read index 2, target 3 gives errors .118303 and 2.125. These differ from the prose example and are retained in investigation-checks.json. If all inputs and the initial state are zero, both traces are zero for any gate schedule. With initial state 4 and coupled g=0, the state remains 4. With independent a=.5,g=0, it becomes [2,1,.5,.25], showing that no write does not imply no forgetting.

**Lifecycle, limits and checks.** The common live-update/reset contract applies to all fields and add/delete operations. No solved target output appears . At most two 16-step float64 recurrences; use a 1e−10 tolerance. A numeric table supplies every drag alternative; narrow views stack timelines. Phase two checks changed gaps and markers, nulls, gate endpoints, independent mode, missing target, input bounds, keyboard/drag agreement and every invalidation path.

## Investigation 3 — SSD matrix and chunk workshop (§6)

**Learning question.** Does changing a computational partition change the answer? Which earlier writes influence a selected output?

**Editable entities.** Use 1–8 positions with N=P=2. Each scalar decay is in [0,1]; each b,c,v coordinate is in [−8,8]. Start from the manuscript's four-row table with v[2] replaced by [−1,3], and chunk size q=3. This creates an unsolved answer and an uneven final chunk. Allow q=1…8, including q>T. Initial state is zero, with an optional editable 2×2 initial matrix in [−5,5]. Presets for a future edit, reset-before-row or zero writes supplement free editing.

**Live comparison:** compute recurrent, matrix and chunked outputs for the current inputs and show their maximum discrepancy. Highlight which rows change under the selected intervention. Vary the computational partition independently from the operator parameters; a correct chunk implementation preserves the function.

**Exact calculation.** S=aS+outer(b,v), y=cᵀS. L[i,j] is the product of decays from j+1 through i, with unit diagonal and absent future entries. M[i,j]=(c_i·b_j)L[i,j]. Compute MV and add the initial-state response c_iᵀ(product(a[0…i]) S_initial). For chunks, compute local outputs, the chunk's own final write, total decay, the passed state and its within-chunk readout. Do not apply softmax or row-normalize the actual operator. Direct products are adequate for at most eight float64 steps and handle a=0; the scalable log-segment discussion does not imply a kernel was built.

**Observable views.** Show CBᵀ, L and their elementwise product separately, with signed cell values and a zero-centered color key. A selected row expands into contributions from V. The four-stage chunk view shows local, incoming and total output side by side. A semantic output table reports all three results and maximum absolute difference. Caption: coefficients are unnormalized signed weights; q is a computational partition.

**Checked fresh fixture.** With v[2]=[−1,3], outputs are [[2,1],[4,−.5],[−.25,2.75],[1,5.9]]. Chunk sizes 1,2,3,4,8 agree in investigation-checks.json. With the worked baseline restored to v[2]=[1,2], output is [[2,1],[4,−.5],[1.75,1.75],[5.8,3.5]]. The following checked contrasts use that restored baseline: a[2]=0 gives [[2,1],[4,−.5],[1,2],[4.4,3.8]]; changing final v to [8,−3] changes only the final output, to [−4.2,7.5]. Zero b with zero initial state gives the zero-output null. Chunking must preserve the answer for every active input. Nonzero initial state requires its additional term.

**Lifecycle and constraints.** Apply the common live-update, reset and input-validity rules, including q and initial-state changes. Matrix size is at most 8×8 and state 2×2. On a phone, use a selected-row table and vertically stacked chunk stages rather than tiny labels. Phase two checks all fixtures, uneven chunks, q>T, nonzero initial state, zero decays, bounds, keyboard operation and independently implemented evaluations at 1e−10 tolerance.

## Investigation 4 — a fitted trajectory under an edit (§7)

**Learning question.** What changes when a coordinate or its order changes, and which differences reflect only the evaluation algorithm?

**Data and models.** Use movement_libras.data, trajectory-state-fits.npz and the saved results. The default is source row 7, the first validation row, with original class 1 (curved swing). Start with diagonal_seed17; offer selective_seed17. Restrict record selection to saved validation IDs, never assessment IDs. All 45 points remain editable in unit coordinates [0,1], transformed by 2x−1 for the model. Expose a point index, x/y numeric fields, equivalent drag controls, reflection of a chosen x about .5, reversal of order and reset. These are genuine changed inputs, not controls mapped to a prewritten output.

**Live fitted result:** show the original and edited class-probability vectors, top class and their difference immediately. Changing the trajectory recomputes the exact retained model. A top class may stay fixed while its probability moves; show both effects. Display the source label separately and keep a clearly identified original baseline.

**Exact frozen inference.** Match the full saved program: input affine projection; two blocks of LayerNorm over 16 coordinates using population variance and eps=1e−5 with learned scale/bias; temporal mixer; GELU xΦ(x); output affine projection; residual; mean over 45 steps; final affine classifier; stable 15-class softmax. The diagonal model has four complex modes per channel, paired readout 2Re, and Ad/Bd/C from saved parameters. The selective model has eight real states, softplus Δ, A=−exp(log_decay), exp(ΔA) decay, ΔB write and Du once. Implement a small local forward computation with real pairs for complex arithmetic. Do not load PyTorch, train weights or contact a model service. Phase two may export arrays to a compact lossless runtime representation with digests and metadata.

**Exact contrasting fixtures.** The author chose the first validation row before inspecting its output. Point 23 has normalized x=.24952000379562378; reflection makes it −.24952000379562378, equivalent to unit x .6247600019→.3752399981, with y unchanged. investigation-checks.json retains all logits:
- Diagonal seed17: base predicted class1, probability of original class .35335370898246765; edited class1, probability .33841976523399353; reversed class2, probability .3007948398590088.
- Selective seed17: base predicted class10, probability of original class .18970493972301483; edited class10, probability .20461773872375488; reversed class14, probability .14461970329284668.

**Null and causality evidence.** Resetting the same input yields a zero maximum logit difference. Across all original rows, diagonal FFT/recurrence agreement is within 4.76837158203125e−6 for seed17 and 4.291534423828125e−6 for seed41. A separate first-mixer probe changes projected features from step30 onward to 2; earlier 30 mixer outputs change by at most 9.5367431640625e−7 for the diagonal model and exactly zero for the selective reference. This is an internal causal-prefix check, not a claim that the final mean-pooled classification is unchanged. Only show an FFT comparison if an independent FFT was actually implemented; a repeated serial forward is not one.

**Representations and interpretation.** Show the equal-axis directed path, ordinal point markers, selected coordinate table, a selected channel's temporal response and 15 named class-probability bars in a fixed class order. Display the original source label separately from predictions. After recording, allow a baseline ghost path and a changed-segment highlight. Internal state coordinates are an inspection view, not an importance or causal-attribution score. Report the class/probability difference and ask why traversal order might matter even when geometric shape looks similar. An edited trajectory is hypothetical; its original label is not a newly certified edited-input label.

**Bounds, accessibility and later checks.** Keep 45×2 points, width16, two blocks and at most the two seed17 parameter sets loaded on opening. Run explicitly, cache by input key, and release owned arrays on close/unmount. Provide numeric keyboard controls, adequate touch targets and a full probability table. At narrow widths stack the plot and timeline; never shrink 15 class labels into unreadable bars. Phase two verifies source/parameter digests, all recorded comparisons, class mapping, LayerNorm/GELU/softplus/complex numerics, whole-forward agreement within 1e−4 logits and 1e−5 probabilities, live recomputation, reset, validation-only selection, no eager imports/network/training, and mobile/accessibility/performance behavior.

## Publication handoff

Convert “Inline figure” descriptions into adjacent visuals and learner-facing captions; do not print implementation directions as content. Keep these four investigations independently useful and optional. Preserve the nine changed practice problems and eighteen initially closed Hint/Solution disclosures, annotated resources, local sequence links, and complete downloads with provenance. Root owns the ledger checkpoint. Browser implementation, formal independent review, native integration and the production status change remain phase two.


## Code-to-mechanism implementation contract — 22 September 2026

Place the manuscript's new implementation route next to its stated concept section. Preserve the named axes, state and algorithm steps when implementing figures; the complete teaching programs are content inputs, not a hidden replacement for learner-visible code. Render long source only on demand with keyboard-scrollable code and wrapping download labels. Keep constructed comparison fixtures separate from recorded training experiments. No browser execution of Python, pretrained-model download or GPU experiment is required to operate a lab.

The ownership map in design.md identifies which operations are implemented here and which actual sources are reused. Both paths must be findable: the transparent mechanism and the ordinary package/tool route, followed by the changed-input practice. There is no guess-entry, prediction submission or answer-unlock state. Current outputs remain visible while the learner edits meaningful inputs; separate written practice can retain hints and solutions.
