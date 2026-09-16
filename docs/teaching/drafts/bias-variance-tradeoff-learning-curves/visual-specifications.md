# Bias–variance visual and investigation contracts

Content specifications, 12 September 2026. Consume [lesson.md](lesson.md), [design.md](design.md), [calculated-inputs.json](calculated-inputs.json) and the retained data/calculation files. These contracts are written; no production component, interactive model, rendered figure or independent phase-two review exists yet.

## Shared contract

Choose forms for the question: prediction/outcome dots, fitted-function bundles, paired learning curves, an information-flow diagram and a projection illustration. The measured Airfoil curves must not be replaced by a standard U-shaped cartoon. Exact finite calculations, recorded model results and asymptotic formulas have different visible evidence labels.

Every scored investigation starts with an unset prediction. Commit records the full old/proposed state and quantity predicted; Apply calculates from that state and compares the result with the committed answer. Changing any relevant setting invalidates the prediction. A separate ungraded Explore action may reveal a state without creating a retrospective correct answer. Reset restores documented defaults and clears grading/history. Keep draft versus applied controls explicit and replace all linked displays atomically.

Keyboard numeric inputs provide the same edits as pointer dragging. Native labeled controls, a visible focused entity, and a concise polite result announcement are required. Use text/shape as well as color, an exact-value table, and plain captions. Avoid automatic animation. At 320 CSS px stack panels while retaining normal-size labels; never scale an entire large plot down until its axes become illegible. A text table may scroll in its own named region; the page must not overflow horizontally.

Round only display values. Reject nonfinite/out-of-range entries next to their input without overwriting the active result. Any numerically singular fit must be detected and explained if an implementation broadens the supported input domain. User code execution, unbounded fitting and unrelated global datasets are unnecessary.

## I1. Move predictions, distinguish two kinds of spread — §1

Question: does changing one fitted prediction alter bias, prediction variance, or both?

Use three equally likely training-world predictions A=[8,10,12], true target mean10 and fresh-target variance1. Display one horizontal prediction ruler with three dots labeled D1,D2,D3, a mean tick and the target mean reference. A second, separate ruler shows outcome possibilities [9,11]. The rulers share numeric scale but have explicit titles: “predictions after different training samples” and “fresh outcomes at the same input.” Do not make outcome noise another fitted-model dot.

Controls: each prediction editable in[0,20] step.25; true mean in[0,20] step.25; fresh-noise standard deviation in[0,4] step.25. The constructed fresh outcomes are mean±sigma. Initial ruler bounds[0,20], extend by padding if outcome markers reach−4/24. A constant B=[9,9,9] comparison is accessible, but arbitrary values remain the principal edit.

Before Apply, choose expected-error direction relative to the previous applied state: decrease/unchanged/increase. Prediction key contains the entire previous and proposed tuple. Calculate mean m, squared bias(m−f)², variance=sum((a_i−m)²)/3 and total=mean((a_i−f)²)+sigma². Use tolerance1e−10 for equality. A waterfall adds the three nonnegative quantities with explicit values, while a distance bracket identifies the signed mean offset. Never draw the signed bias as a positive error-bar length without the square distinction.

Checked fixtures by direct arithmetic:

- Default total11/3; change [8,10,12]→[9,10,11]: total5/3, lower.
- Default→[6,10,14]: total35/3, higher.
- Permute predictions to[12,10,8]: exactly unchanged quantities.
- Same bias, different spread: default versus[10,10,10], error11/3→1.
- Same variance, shifted mean: default versus[9,11,13], error11/3→14/3.
- With every prediction and true mean translated by2, all error components stay unchanged.

Feedback identifies changed operands, not just correctness. For shifting all predictions up1, their variance stays8/3 while squared bias becomes1. Transfer: construct two different sets with the same mean but different total error; then explain a null. Accept calculated conditions, not a hidden exact string.

No need for a third generic metrics box. Put one equation directly beneath the selected dot geometry; exact data table lists all six prediction/outcome squared errors if the learner opens “enumerate.” On mobile separate the rulers vertically and put entity edits beneath the selected dot. Check the arithmetic both by decomposition and six equally weighted prediction/outcome pairs, plus edited mean/noise, permutation null, stale predictions, reset and keyboard operation.

## F1. The error splits into centered deviations — §2

Static three-part number-line equation, using the same target and prediction notation as §1. Label Y−f as fresh target deviation, f−mean as fixed offset, mean−prediction as training-draw deviation. A small expansion grid displays the three squared terms and the three cross terms; cross terms are marked “expectation zero under stated independence/centering,” not silently erased.

The display encodes algebra, not measured spatial lengths unless actual values are substituted. Pair it with the text derivation. Mobile: a vertical list of the three deviations followed by the resulting identity. Include conditional x and actual expectation labels in the accessible equivalent. The separate hidden-setting example uses a two-lane information diagram: X alone leaves Z+noise, observing X,Z leaves noise. Values1.25/.25 are calculated, with no implied prediction-variance change.

## I2. A finite population of fitted curves — §3

Question: when does changing polynomial degree improve expected error at the chosen input?

Actual model: train inputs initially[-1,0,1], true mean1+x+c*x², independent training noise ±sigma at each input with equal probability. Enumerate8 training target vectors; optional fixed five-input design[-1,−.5,0,.5,1] enumerates32. Fit degree0/1/2 by unregularized least squares. Each design matrix has full column rank; no train-input dragging is required. Stored JSON contains independently calculated fixtures, but edited c/sigma/probe states require the same actual model rather than interpolation between canned prose.

Controls: curvature c in[−1,2] step.05; sigma[0,1.5] step.05; probe x[−1.5,1.5] step.05; reference/candidate degree0–2; three/five input design. These are meaningful edits to the data-generating mechanism and queried prediction. Choose comparison “candidate expected error lower/same/higher than reference,” commit with the complete tuple, then apply. Display true mean, each fitted curve, average fitted curve and a vertical probe slice with one dot per training world. The error table at that same probe shows mean, squared bias, prediction variance, target-noise variance and total. Number of training worlds and their equal probabilities are explicit.

Do not describe this as a bootstrap or random-X experiment. Switching to five inputs adds particular measurement locations and changes the design, not only an i.i.d. sample count. Keep that boundary in the caption next to the switch. Each training-world selector shows its actual noisy observations and selected fitted curve; no arbitrary “lucky seed” search. Selection does not change the average error distribution.

Exact checks from retained calculations at x=.5, sigma=.5:

| Design / curvature | Constant error | Line error | Quadratic error |
| --- | ---: | ---: | ---: |
| Three inputs,c=1 |49/144|155/288|55/128|
| Three inputs,c=0 |7/12|35/96|55/128|
| Five inputs,c=1 |.3625|.3875|12/35|

The line→quadratic comparison improves at c1 and worsens at c0. Three-input c1 at x0 makes constant and line predictions identical for every noise vector; their error comparison is a meaningful null. At sigma0,c0, both line and quadratic reproduce truth exactly and tie at0. With c1 and increasing noise .5→1, quadratic x.5 variance23/128→23/32 and total55/128→55/32.

Important observed contrast: adding the two fixed inputs slightly worsens the constant fit at x.5 (.340278→.3625), although variance falls. Its average predicted level shifts and pointwise bias rises. This is correct under this design-specific experiment, not evidence that random additional population data always harms constant models. Feedback must show both terms.

Plot geometry: x range[−1.5,1.5]; y shared across both model panels and fitted realizations, dynamically bounded using actual extrema plus padding. Never clip curves to manufacture stability. At least the true mean and average fitted curve stay solid/labeled; individual curves are subdued with a selectable thick highlight. No invented uncertainty interval around the truth; these are actual finite fitted curves. The probe ruler uses the same current data.

On phones show one fit at a time with accessible reference/candidate switch and a persistent comparison table; changing displayed panel does not alter model state or grading. Probe numeric input is the touch/keyboard alternative. Render a dataset table for8/32worlds behind a details disclosure; load only this lesson's small inputs. Bound draws32, input points5, degree2 and grid61; deterministic algebra, no workers or package-sized plotting dependency required.

Formal phase-two verification: compare all stored fixtures with an independent polynomial-weight implementation; verify interpolation weights[−.125,.75,.375], full-rank least squares and exact eight-world sums; sigma0 and degree equality; reference/candidate exchange; application-state consistency; negative curvature; edge probes; reset/input invalidation. Actual visual spread/perceptibility requires future desktop/phone inspection.

## F2. Fit size is a count within each fold — §§4–5

Use a data lane showing1,503 retained rows→1,200development/303reserved. Split development into five folds; highlight one with960available training/240validation. Beneath it, nested fitting subsets of60/120/240/480/900 reach separate fresh model boxes while the same240validation rows reach each evaluator. A StandardScaler state is learned only inside the Ridge pipeline.

The diagram need not draw1,503 individual tiny cells. Proportional bars and exact count labels communicate sizes; individual retained IDs remain in JSON/table. Explain that the four model families reuse the same protocol. No result arrows leave the303reserved rows. Mobile: one fold and one subset size illustrated at a time; counts remain visible. Check disjoint development/reserved IDs, each fold count, nested subset eligibility and absence of evaluation-to-fit arrows.

## F3. Real learning curves and a crossover — §5

Source: JSON real.learning, actual five-fold MSE values. Small multiples by model with identical x-axis fitted rows[60,120,240,480,900] and common y-axis0–50 squared dB. Training/validation distinguished by line pattern and explicit legend. Include training-mean baseline; it must not disappear just because it is worse. Display fold values as small jitter-free markers or an expandable table, and the mean curve. Do not call fold spread a95%confidence band.

Default useful contrast: Ridge and leaf1 tree. At60 rows the tree is worse (41.8405 vs24.9831); at900 it is better (8.1412 vs23.3720). The tree's training curve lies exactly at zero, so give it a visible marker/offset-free baseline and adequate axis padding below0 for marker visibility without representing negative MSE. Leaf20 comparison exposes the changed restriction. No arbitrary interpolation is called a newly fitted result.

Selecting a model or fold is inspection, not a scored investigation; no claim that dropdown presets are an independent experiment. Show exact ratios/differences only when they help answer the question and match input units. An optional paired-comparison table retains all5fold values rather than misreading means as paired raw observations.

Phone layout: compare two models in stacked panels with shared axes; avoid showingfour shrunken plots side by side. Use readable tick labels60/240/480/900 and a table containing120 as well. Caption states row-level development protocol and links provenance. Future checks: all displayed means from stored folds, current model identity, zero training curve, visible crossover, no unused reserve results and accessible exact-value alternatives.

## F4. Restriction and training time have different axes — §5

Two separate charts, each labeled with its own split:

- Validation curve: JSON real.validation; x minimum leaf size1/2/5/10/20/40, yMSE0–25, five-fold train/validation means. This uses960fitrows, explaining why its leaf1score8.2079 differs from the900-row learning curve.
- Boosting trajectory: JSON real.stage, rounds1…120 and actual train/monitor errors. yMSE0–50. Mark best inspected round120 with a right-edge annotation “lowest observed in this range,” not “optimal stopping round.” No unmeasured turn-up. The monitoring split is960/240 within development, not the same five-fold estimate.

Optional step inspection selects an actual recorded round and reveals both values; this is an explanatory trace, not another mandatory lab. Mobile stack; table of1/10/30/60/120present. Avoid y-limits that hide the direction or the starting baseline. Check all120values when computing argmin, not only printed sample rows.

## F5. Same inputs, new outcomes — §7

A paired schematic fixes the same matrixX on both sides. On left, training noiseε enters y→H y and the residual(I−H)ε. On right, independent noiseε′ enters new outcomes while predictions still depend on the originalε. Link the two residual formulas to their expected error bars for n6,p2,sigma²4:8/3 versus16/3. A third bar shows prediction variance4/3; label the gap8/3.

A small projection picture is qualitative geometry, not a data-trained numerical plot. Label column space and residual-orthogonal component; no claim that each coordinate of ε is represented by the sketch. A separate new-input arrow to x_* states that new-location leverage differs from the same-X average.

No interaction necessary; the point is the changed dependence, so mirrored formula lanes beat a generic slider. Accessible equivalent gives assumptions and each number. Phase two checks trace ranks, same-X qualifier, intercept count, squared-error units and all hand arithmetic.

## F6. Theoretical double descent — §9

Use only JSON doubleDescentApproximation and the piecewise asymptotic formula in the manuscript. x ratio gamma=n/p, yexpectedMSE; noise variance.04, signal norm1, isotropic Gaussian design, min-norm ridgeless estimator. Plot gamma .1…3 with a break at1. Mark1as singular/asymptotic boundary; do not join left/right branches. Values are theoretical asymptotic calculations, not measured dataset performance.

A log y-axis with explicit values.04/.1/.4/1/4 can preserve both the peak and lower branches; label the transform in caption. Include the .04 noise asymptote as a dashed horizontal reference. The plotted finite endpoints at.99/1.01 do not define a finite maximum; outward arrows toward the vertical boundary communicate divergence of the approximation. A linear inset on[.1,.95] may aid the initial decrease/increase if the main transform obscures it. Use actual plotted states in phone/desktop review before selecting final geometry.

Text table includes the manuscript's nine ratios and values. No spline overshoot or parameter-count labeling: this is sample-wise behavior at fixed high-dimensional model size. A companion note gives the bias/variance expressions; no new complex simulator required. Checks: piecewise branches, additive noise, meaningful .75→.9 reversal, no NaN/Infinity serialized as numeric points, reciprocal limits, and honest asymptotic labels.

## Later implementation

Build semantic topic-owned lesson, figures/investigations, numerical models and examples under the code standard when finish is authorized. Export only needed data/fixtures into runtime assets; the whole 692 KB author JSON is a retained research input, not automatically a browser payload. Keep the real .dat download and attribution accessible. Do not import other lesson bodies, scratch tools or the phase ledger at runtime.

Execute the full displayed programs verbatim, verify actual model exports and independent arithmetic, obtain formal independent correctness/learning-experience review, and inspect browser/mobile/keyboard states. Rendering, real interaction, final asset budgets, integration and publication are deferred. Improve a representation for a recorded learning reason if implementation exposes a better choice.
