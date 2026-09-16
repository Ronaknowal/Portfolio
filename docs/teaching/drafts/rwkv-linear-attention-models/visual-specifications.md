# RWKV & Linear Attention Models — visual and investigation handoff

Content phase only,13 September 2026. These specifications are not implemented visuals. Read the complete [lesson](lesson.md), [design](design.md), [provenance](data-provenance.md), [mechanism results](mechanism-results.json) and [fitted-model fixtures](investigation-checks.json). Preserve the named operators and tensor orientations.

## Shared rendering and interaction contract

Use existing reader typography and control tokens, with topic-owned semantic files during phase two. Inline figures belong beside their mechanisms. Choose forms for the reasoning task: causal connection map, outer-product grid, two-stage circuit, scale ruler, key-direction compass, matrix subtraction, trajectory path and measured outcome plot. A lab does not replace the first explanatory figure.

Each figure has a visible caption stating the intended inference, every axis/shape/unit, and exact calculation versus measured fitted output. Accessible text/tables communicate the same relationship without color or motion. Use outline, labels and patterns alongside color. Signed matrix entries are not probabilities. No decorative speed races or fabricated performance curves.

All investigations start with editable inputs and **no recorded prediction**. Use explicit numeric/choice/free-text answers without preselected radio responses. A prediction record contains the exact current input snapshot, the learner's answer and the model/operator revision. “Run and compare” requires that record. Editing any computation-bearing input invalidates the record and hides the old answer until the learner records another prediction. Preserve an unsubmitted explanation draft if useful. Purely presentational zoom/tabs do not invalidate results.

Every lab has Reset to its fresh starting problem, Clear prediction and appropriate restore controls. Reset restores all inputs, states, selected row/cut/model/seed and controls, then clears predictions/results. No state leaks between topics or source examples. Hints and solutions initially remain closed.

Keyboard users can edit every entity available through a pointer. Pair coordinates/matrix entries with labeled numeric fields, row insert/delete controls with clear names, and sliders with numeric fields. Dragging is optional. Announce completed runs concisely, not every animation frame. Respect reduced motion and preserve focus through rerender.

At320px, stack controls and result panels; confine horizontal scrolling to labeled math/matrix/table regions. Never shrink labels until illegible or cause page-wide overflow. Use matched numerical scales in comparisons. Math needs adjacent verbal interpretation. Do not assign physical units to normalized trajectory positions.

Toy calculations use bounded finite numbers and float64. Validate invalid inputs explicitly rather than silently changing the operator. Round only for display. Numerical equality is within a stated tolerance, not bit identity. During phase two, verify the shipped models independently and review rendered learning experience separately from calculation correctness.

## Inline figures in manuscript order

| Figure and location | Representation and exact contract |
| --- | --- |
| 1. Transcript/query,§1 | Three key/value pairs → query scores → normalized bar → weighted vector sum; future item disconnected. Example scores[0,log(2),0], values [2,8,−1], weights [.25,.5,.25], output 4.25. Accessible products table. |
| 2. Slow weights/fast state,§1 | Training updates projection weights; each new sequence advances state while those weights stay fixed. Draw separate boundaries for parameters, sequence state and external cache. |
| 3. Outer-product write,§2 | Length-m key column × length-p value row → m×p state tile; separate key accumulation into z. Worked two writes: S=[2,8]ᵀ,z=[1,1]ᵀ,q=[2,1], numerator 12,denominator 3,answer 4. |
| 4. Four-update ledger,§2 | Exact Q,K,V from mechanism results; intermediate S,z and output per row. Final S=[11,12],z=[4,3], numerator 45,denominator 15. Worked trace distinct from fresh investigation. |
| 5. Two kernels,§2 | Query2,keys 0/1,values 0/10. Softmax weights [.119202922,.880797078],output 8.80797078; ELU+1 weights [1/3,2/3],output 6.66666667. Matched0–1 weight axes and 0–10 output axis. |
| 6. Random-feature identity,optional§2 | Gaussian direction feeds two positive exponentials, expected product equals exp(x·y). Separate expectation, finite estimator and normalized ratio. No implied guaranteed finite convergence. |
| 7. Chunk's two sources,§3 | Lower-triangular within-chunk contribution plus incoming S,z contribution; combine before division. Chunk3 on four updates creates a short final chunk. Future cells disconnected. |
| 8. Training/decoding timeline,§3/5 | Training inputs available together for projection, then structured memory work; generation's next input depends on preceding output. Do not depict every RWKV kernel as a time-parallel scan. |
| 9. RWKV read/write circuit,§4 | Current exp(u+k) only enters read; stored write uses exp(k) and decay on old state. Worked t2: A2,B1,currentweight4,read6.8,storedA17,B2.5. Highest priority for index clarity. |
| 10. Retention ribbons,§4 | Isolated contribution under λ=.5,.9,.99, half-lives1,6.57881348,68.96756394 storage updates. Axis extends enough to show crossings. No invented semantic channel names. |
| 11. Stable scale ruler,§4 | Common rescaling of signed numerator and positive denominator; weight exponents end at or below0. +1000 key shift preserves output within 1.1e−13 while raw exp overflows. Overflow is text, not a finite plotted point. |
| 12. Block/state slots,§5 | Norm→time mix→residual→norm→channel mix→residual. RWKV4 slots: previous time input,a,b,p,previous channel input. Parameters outside state boundary. Teaching channel expansion 2 labeled explicitly. |
| 13. Shifted-label objective,§5 | Input tokens1…T aligned with target tokens2…T; processing t predicts t+1. Reset removes context; detach preserves value but stops gradient arrows. |
| 14. Add/correct,§6 | Orthogonal A/B writesA2,B7,A5: additivefinal[7,7],deltafinal[5,7]. Last correction reads2,target 5,residual 3. Adjacent unaffected address remains visible. |
| 15. Address interference,§6 | Unit B[.6,.8],A[1,0]. After second delta write[5.48,4.64]; after A5:[5,4.64],B retrieval6.712. Geometry labels refer to dot products, not estimated statistical correlations. |
| 16. Version shapes,§6 | RWKV4 vector summary→RWKV5 matrix/fixed row retention→RWKV6 dynamic rows→RWKV7 coupled removal. A structural map, not accuracy ladder. Label orientation/read order. |
| 17. RWKV7 tiles,§6 | Initial[[2,7],[-1,3]],w[.8,.9],a[.6,.2],replacement[1,0],v[5,2]. Removal[1,0]→transitiondiag(.2,.9),next[[5.4,6.3],[1.8,2.7]]. Removal[.6,.8]→[[.584,−.096],[−.288,.772]],next[[4.152,5.212],[.552,2.412]]. Separate decay/subtraction/addition tiles. |
| 18. Optional reflection,§6 | n=[1,−1]/√2 gives I−2nnᵀ=[[0,1],[1,0]], swapping coordinates. Label theoretical factor2/boundary construction, not released c=1 core. |
| 19. Ways to forget,§7 | Scalar valve, row valves and erase direction over matched state. Checked unnormalized constant-.5 outputs [2,10,5.5,35.75]; third gate[.1,.9]→[2,10,11.5,36.75]. Full RetNet/GLA blocks include other components. |
| 20. Cache formulas,§7 | Fixed12 layers,width 512,8×64 heads. RWKV4 fp32all5vectors122,880 bytes; matrixcore fp32only1,572,864 bytes; MHA fp16KV24,576Tbytes; GQA2KVheads6,144Tbytes. T4096 gives 96 MiB/24 MiB. Default T0–8192; calculated curves, no timings. |
| 21. Movement pipeline,§8 | Actual row 7 unit-space path→2x−1→45×2→width 16→two memory blocks→mean→15 logits. Attribution, start/end/order labels, training-only backward arrows and forward-only evaluation roles. |
| 22. Fit outcomes,§8 | Assessment errors/60: baseline22,RWKV17=19,RWKV41=23,kernel17=34,kernel41=23. One point per seed, axis0–60. Parameters1365/5263/5199. Optional actual fit/validation histories mark epochs 67/61/54/69. No interval inferred from two seeds. |
| 23. Stream continuation,§8 | Worked row 7 split22/23, complete carried-state bridge versus reset; weighted feature sums/counts. RWKV carry error≤2.4e−7; reset class 1p .753939→.047025,pred1→10. Kernel originalpred10/resetpred7. Equivalence indicator separate from correctness. |

## Investigation A — two summaries and a causal table

**Learning question.** Can a query or write change a selected answer while future edits preserve the past and chunk size preserves the operator? Place after§2; unlock chunk explanation after§3 or provide its brief local refresh.

**Fresh editable problem.** Q=[[1,1],[2,1],[1,2],[3,1]],K=[[1,0],[0,1],[1,1],[2,1]],values [3,−2,7,1],chunk3,readposition4. Those values are not solved in the manuscript. Allow1–12 sequence rows, two features and scalar values. Query/key entries0–4, values −12…12, chunk1…12 and chosen read1…T. Insert/delete changes actual entities.

**Prediction.** Ask for the selected output or its direction of change and which earlier outputs can change. Initially unset. Bind Q,K,V,chunk,readposition and operator revision. After record/run show causal weight table, stateS/z, normalized output and direct-versus-chunk difference. Inspection controls can reveal individual contributions without changing the problem.

**Checked contrasts and null.** Fresh outputs [3,4/3,10/3,2.8]. Chunk sizes1,2,3,4,8 agree. Worked values [2,8,−1,5] give [2,4,2.5,3]; changing only last value to 11 leaves earlier outputs unchanged and gives last 5.8. Every value 4 gives every output 4. Zero-query input is an undefined normalized operator, with a local error rather than invented output 0. Check the actual denominator for each row.

**Feedback and reset.** Explain which key score, write or total caused the observed change. Equivalent algorithms agree within 1e−10. Reset fresh inputs, empty S/z and unset prediction; chunk changes invalidate the recorded input snapshot. A purely presentational panel switch does not.

**Accessibility, bounds and phase two.** At most144 scores, no worker required. Accessible row/column headings and named table/state/output panels; keyboard row editing. Independently compare direct and chunk computations for changed inputs, shorter final chunk,T1,zero-denominator recovery,last-edit causality,constant-value null and full reset. Then review the visible learning flow.

## Investigation B — today's answer versus stored history

**Learning question.** Which controls change today's read, future stored information or only the numerical scale? Place after§4.

**Fresh editable problem.** Logkeys[0,log(2),0,log4],values [3,−2,7,1],retention.5,currentbonuslog(2),commonoffset0,readposition3. Allow1–12 positions; keys −8…8, values −12…12, retention.01…1, bonus−4…4 and common key offset−1000…1000. Edits alter actual sequence values and weights.

**Prediction.** Record output increase/decrease/same or numeric answer, whether stored history changes, and a reason. Bind sequence,retention,bonus,offset,selected read and operator revision. All computational edits invalidate the record. Stepping an already computed trace is presentation-only.

**Mechanism view.** Read and write branches remain separate. Old/current contributions are labeled, with stable a,b,p below. p is a dimensionless weight log scale. Optional direct history view evaluates shifted logweights; it must not turn overflowing raw exponentials into valid finite bars.

**Exact contrasts and null.** Fresh outputs [3,−1,23/9,55/41]. Worked values [2,8,−1,5] with bonuslog(2) give [2,6.8,10/3,190/41]; bonus0 gives [2,6,32/7,4.4]. Stored history is identical under this bonus change. Common offset+1000 gives maximum difference1.05694e−13 while raw exp overflows. All values 7 from empty state produce 7 throughout. Retention1 is a meaningful no-decay boundary; phase two must independently verify its unrolling.

**Feedback/reset.** Explain why bonus belongs only to the read and retention changes history. A scale-change misconception should reveal the same normalized weights. Reset all fresh inputs, empty a/b,p=−∞ and unanswered prediction. User values outside bounds get a local error, not silent clipping.

**Accessibility/budget/phase two.** One channel,12 steps; optional second channel only if it clarifies distinct timescales. Numeric logkey/retention/bonus fields, readable signed contribution labels, stacked circuit on narrow screens. Check direct enumeration versus stable recurrence, signed values, first-step initialization, current-bonus/state invariance, large offset, constant-value null, retention1 and recovery/reset. The internal initial −∞ is valid empty-state representation; it is not an editable nonfinite key.

## Investigation C — write, retrieve, correct

**Learning question.** Can a memory update one address without damaging another? When does address geometry prevent it?

**Fresh editable problem.** Unit keys [[0,1],[1,0],[0,1]],values [4,−3,9],rate .5,query [0,1],initialM=[0,0]. Allow1–10 writes, key/query angles−180…180degrees converted to [cos,sin], values −12…12 and rate 0…1. Optional initial-state entries−6…6. This differs from the solved A2/B7/A5 sequence.

**Prediction.** Initially unset retrieved-value guess, preserved-address judgment and explanation. Bind order,angles,values,rate,initial state,query and operator revision. Run additive and delta memory on identical inputs. If a query has no specified target, show its retrieval without inventing a ground-truth error.

**Visual result.** Key compass, numbered writes, query arrow, signed memory row, before-read/target/residual/correction per step. Show changed-address and untouched-address retrieval together. State is value-by-key M, explicitly transposed relative to the earlier S. No claim this ordinary delta simulation is the full RWKV-7 model.

**Contrasts/nulls.** Fresh additivefinal[−3,13],deltafinal[−1.5,5.5]; querysecond retrieves 13 versus5.5. Worked orthogonal writes atβ1 yield[7,7] versus[5,7]. Correlated key [.6,.8] yields intermediate[5.48,4.64],final[5,4.64],final B retrieval6.712. Rate0 leaves empty delta memory zero. Repeating an identical unit-key/value write after β1 gives zero correction. With nonempty initial state, rate 0 preserves that state. An optional nonunit branch must explain the ||k||² factor rather than silently normalize.

**Feedback/reset/bounds.** Feedback identifies the key dot product and both retrieval changes. Reset fresh keys/values/.5rate/empty memory/unset prediction. At most10×2 arithmetic. Angle fields duplicate dragging; keys have numeric names and visible direction labels. Narrow layout stacks compass, write editor and result. Phase two checks fresh/orthogonal/correlated/rate 0/repeated-target/changed-initial fixtures, independent gradient calculation, edited keys, reset and keyboard use.

## Investigation D — interrupt a real stream

**Learning question.** Did a result change because the data changed, state was lost, or the stream was split equivalently? Include misclassified originals so equality is not confused with accuracy.

**Offline data/model.** Only the50 deduplicated validation source IDs are selectable. Default seed 17, RWKV-style versus positive-kernel model; seed 41 is optional with its own frozen weights. The retained NPZ is a handoff source; phase two exports only needed arrays to safe browser data. No pickle, online training or eager import of every lesson. Read the full training program for both previous-input slots, memory state, layer normalization and mean pooling.

**Fresh start.** Source20, second validation row, class 1 curved swing, cut after 15of45 points, RWKVseed17, original path. Its answers are not worked in the manuscript. Choose a source row, edit any point through numeric fields or dragging, cut1…44, reverse order, and compare carry/reset. Display source coordinates in unit-space[0,1]; compute2x−1. Edited paths are labeled modified and do not receive certified new class labels.

**Prediction contract.** Before fitted outputs are revealed, ask whether carry matches uninterrupted computation, whether reset changes the predicted class and/or how a chosen probability changes, with a reason. Bind all 45 coordinate pairs, source row,cut,model,seed,fitted-weight digest and comparison mode. All computational edits invalidate the record. The original known class may remain visible; model bars wait for record/run.

**Result.** Show uninterrupted, carry-all-state and intentional-reset routes. Mark the cut on a chronological path with start/end arrows. Carry the previous normalized time/channel inputs and all memory slots. Temporal mean uses sum/count across chunks. A max-logit-difference indicator is separate from original-label correctness. All15 class probabilities are available in an accessible text table; plot selected original/predicted classes without silently hiding alternatives.

**Fresh checked fixture: source 20/cut15/seed 17.**

| Model | Original prediction / class 1 probability | Carry max logit difference | Reset prediction / class 1 probability | Point16 x reflection | Reversed order |
| --- | --- | --- | --- | --- | --- |
| RWKV-style | 7 / .0125182085 | 9.53675e−7 | 7 / .133891672 | 7 / .00101422914 | 1 / .744798541 |
| Positive kernel | 8 / .0998163223 | 4.76838e−7 | 15 / .233017176 | 8 / .0756031275 | 1 / .309542328 |

Carry probabilities are .0125181954 and .0998163074. These unequal and sometimes non-flipping results are intentional; do not manufacture a universal class change.

**Additional contrast/null.** Worked source 7/cut22: RWKV originalpred1,p.753938913;carry equal within 2.38419e−7;resetpred10,p.0470251255. Kernel originalpred10,p.251895875;resetpred7,p.0546496287. Point23 normalized x reflection+.249520→−.249520 and reversal have exact results in JSON. Restore-original produces the same CPU result. Future point31 perturbation leaves earlier per-position features unchanged, CPU error0; final mean prediction may change. All four fitted models preserve features within 7.62940e−6 over chunk lengths 1/12/16/16 on the first five source rows.

**Feedback, bounds and reset.** Explain the lost state or changed path and distinguish equivalence from correctness. One active45-point example, two width 16 blocks, one selected fit, at most three main forwards per recorded run. Compute on explicit run, not every pointer event; share fixed arrays and lazy-load this lab. Never recompute the whole corpus in the browser. Reset row 20,cut15,model/seed,path,empty state and unanswered prediction. Restore-original only changes coordinates and invalidates the old prediction as expected.

**Accessibility/mobile/phase-two checks.** Point selector and numericx/y, cut range plus numeric entry, named reversal/restore actions and sourceID/class/role labels. Match unit-space axes. Stack editor/state/results without tiny labels. Independently verify the complete frozen forward, state orientation, transformed retention, normalization epsilon, squared-ReLU channel branch, all carried slots, weighted mean, source-role selection and coordinate conversion. Ported logits target 1e−5 tolerance, adjusted only after documented precision investigation. Verify all contrasts/nulls, actual input editing, reset, focus and keyboard path controls; measure payload and long tasks. No assessment dataset or training path is needed for this interactive lab.

## Phase-two acceptance boundary

Read the complete manuscript before implementation. Build topic-owned visuals/investigations, verify the new shipped model against the exact fixtures, and complete independent correctness and learning-experience review plus browser/accessibility/performance checks. Author calculations are reusable evidence for unchanged inputs, not proof that a JavaScript port works. A changed or omitted visual must preserve the teaching need and record why its replacement works better.

