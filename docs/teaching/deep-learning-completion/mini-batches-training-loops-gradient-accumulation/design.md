# Mini-batches: implementation design and teaching review

The complete ten-section manuscript and references are rendered, including all five changed exercises, the seven-row transfer, derivative-engine ownership bridge, loss-convention caveats, and advanced AMP/DDP material. The original current manuscript was substantively different from the stale ledger hash, so it was fully read and preserved; only the opening invitation and obsolete gated visual directions were reconciled. No summary manuscript replaced it.

## Concept and representation map

|Transition|Intuition and exact correspondence|Representation / action|
|---|---|---|
|Examples → physical chunks → updates → epoch|Different clocks count different operations; the last group still exists.|ClocksFigure shows ten stable rows, three chunk brackets, two update brackets and one epoch.|
|Forward → backward → step → clear|A derivative is information about the current weight, not a changed weight.|StoresFigure separates parameter, nullable gradient and momentum stores; StateLab cursor traces all operations and counters.|
|Scalar rule → tensor training|Reducing examples retains every parameter coordinate.|ShapeFigure shows B×D times D×C to B×C and backward C×D/C shapes. The complete scalar PyTorch source remains inline and downloadable.|
|Fresh graph → persistent derivative buffer|Each chunk can release activations while its parameter derivative survives.|GraphLifetimeFigure connects three independent forward graphs to one additive buffer.|
|Equal item objective → unequal chunk means|A chunk mean assigns its items a coefficient before chunks are combined.|CoefficientsFigure has common-scale 1/3 versus 1/4,1/4,1/2 bars. StateLab edits actual rows/boundaries and exposes clear/step mistakes at later changed weights.|
|Count → target mass|The denominator answers what receives equal influence.|TargetSlotsFigure separates eight eligible slots from twelve tensor slots and traces numerator/count sums. MassLab edits inclusion, weight, input, target and noncontiguous groups; every row coefficient and numerator is shown.|
|Unequal counts → equal mass null|Group size alone cannot diagnose a weighting mismatch.|MassLab equal-mass preset agrees with unequal included counts; zero eligible mass stays undefined with no gradient marker or update.|
|Complete training code → measured result|A reproducible comparison holds update membership and state fixed.|Full train_iris.py is opened on demand with shape, scale, split, order, zero/backward/step and validation explanations retained from the manuscript. History uses every recorded epoch including reversals; state gaps are separate exact quantities.|
|Recorded run → current real-data computation|The same rows can enter the same gradient through different physical partitions.|IrisLab lazily loads initial/trained20 parameters and optimizer state, edits actual measured features or a hypothetical target, computes all67 derivatives and one update, and inspects each row/chunk contribution. Edited data remain above the loading boundary.|
|Twelve rows → seven-row physical cap|The actual effective group, not the nominal physical cap, determines the divisor.|SevenFigure draws 7+7+7+7+4 over32 and 7+7+7+3 over24; changed exercise retains19 calls/four updates per epoch.|
|Correct gradient weighting → changed forward computation|Different normalization groups can change each example before the loss is reduced.|Worked normalization and NormalizationLab connect actual raw points to full/local mean lines and signed z coordinates, losses, downstream derivatives and running means. Singleton local groups explain their missing result; shared frozen stats/no-normalization restore partition invariance for their own computations.|
|Matching outputs → different persistent state|Training statistics and moving statistics have different lifetimes.|Equal-statistics preset agrees in output gradients but shows .1 versus .19 running means. Constant-input and one-group nulls are available.|
|Seed → realized dropout mask|Call shape can change random draws; aligned masks define aligned computations.|DropoutFigure shows retained/removed entities, scaling and exact derivatives8,20,6.5. No sampled learning curve is invented.|
|Group size → gradient variability|Averaging changes the possible derivative samples at a fixed model.|VarianceFigure enumerates one-row/two-row/all-row derivatives on the same axis and retains exact finite-population variances.|
|Gradient sum → nonlinear processing|Clipping before addition changes the answer.|ClippingFigure uses common signed coordinates for3,−2.5,.5 and0.|
|Backward clock → accepted-update clock|Scaling, clipping, optimizer state and schedule have separate dependencies.|BoundaryFigure joins scaled gradients, unscales once, normalizes when required, clips once, attempts a step, updates scale, then branches scheduler advancement on acceptance before clearing. CPU evidence does not claim AMP execution.|
|Local means → global distributed objective|Default rank averaging must be canceled by the explicit world-size factor.|DdpFigure connects rank-owned sums/masses to the global denominator and shows −2 versus−5/3. Multiprocess execution is not claimed.|
|Explanation → transfer|New counts, rows and targets force the learner to use the rule.|All five changed problems retain closed hints/solutions; the fresh real-data lab has no answer submission or gated reveal.|

## Learning experience checklist

1. Purpose and relevance: opening invites edits and explains why memory limits need correctly owned updates.
2. Intuition at each concept: map above covers elementary, advanced and code transitions; quantities are explained before rules and API claims.
3. Mechanism-specific geometry: membership brackets, separate memory stores, coefficient bars, tensor cells, target slots, normalization positions, dropout masks and update dependency paths have concrete entity correspondence.
4. Immediate exploration: all four labs render a defined default computation; Step only inspects an already computed trace. No prediction-entry or answer gate exists.
5. Real changes: row/target edits, stable identities and boundary changes recompute the actual model; Iris uses all layers and all parameter derivatives.
6. Meaningful contrasts/nulls: one chunk, zero rate, equal masses, no eligible targets, matched batch stats with changed running state, constant activations, frozen stats and no normalization.
7. Scratch and tools: manual JS algebra is native-checked against PyTorch; complete standalone scalar and real-data programs execute; existing derivative/optimizer engine links retain the intended ownership distinction.
8. Numerical honesty: recorded fitting results remain separate from live hypothetical updates; no timing claim, test-set claim or GPU/DDP execution claim is introduced. All plotted history rows are retained.
9. Accessibility/responsiveness: scoped neutral surfaces, native-size focusable chart scrolling, exact-number controls, explicit select names, captions/exact tables, visible undefined-state explanations and friendly retry preserving edits. Painted verification belongs to root.
10. Practice and readiness: all original changed constraints, closed worked solutions, readiness and references preserved. Primary sources are linked beside claims; linked video is not claimed watched.

## Verification and limits

The successful native verification executes unchanged physical32/12/7 twenty-epoch Iris runs, confirming all21 epoch evaluations, 80 optimizer updates and80/220/380 backward calls. An earlier fixture export failed on an integer tensor after fitting; it was corrected to floating dtype and the same configurations replayed, with no tuning/selection. Fifty-four scalar cases, twelve weighted objectives, twenty-seven true BatchNorm cases and twelve actual networks supply fixtures. The browser implementation passes82,096 scalar comparisons with maximum absolute error1.4210854715202004e−14. This includes all67 parameter derivatives, all per-row derivatives, multiple physical partitions and immutable source state.

PyTorch2.14 primary references were checked for weighted integer-target cross-entropy (sum of target-class weights), probability-target reduction (target-position count), BatchNorm forward population variance/running unbiased variance behavior, and AMP accumulation/unscale/step ordering. Sources: https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html ; https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html ; https://docs.pytorch.org/docs/2.14/notes/amp_examples.html . The lab intentionally models running mean only, not running variance or gradients through an earlier layer.

The public allowlist is trace_update.py, train_iris.py, iris.csv, data-provenance.md and author-results.json, plus two lazily loaded parameter/optimizer snapshots. No authoring manuscript, design, visual specification or review receipt is copied to public. Initial display metadata retains the complete history, split IDs, and first32 measured group rows. Real-data loading errors preserve current edits and offer Retry. Native evidence, model checks and SSR are source-bound; independent review and root browser/integration remain separate.
