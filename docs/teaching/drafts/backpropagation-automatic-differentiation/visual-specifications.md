# Backpropagation: visual and investigation contracts

Research/write handoff only. No browser models, figures or interactions have been implemented. Source manuscript markersA–H; exact/calculated inputs in calculated-inputs.json and complete teaching source teaching-autodiff.py. Author calculations are actual NumPy/PyTorch CPU results, not browser evidence.

## Shared assessment state

InvestigationsA,B,F open with mathematical inputs visible and prediction unset. Commit records the actual response and a snapshot of every input relevant to the question. Reveal computes that snapshot's result and grades it, with numeric terms in feedback. Any input/function/step/dtype change invalidates the old grade and hides answers until a new commitment. Editing a prediction after reveal is a new attempt. Reset restores example values and unset prediction; step-back in a read-only derivation preserves its known model and changes only reveal stage.

Use labeled keyboard-operable numeric inputs alongside any geometric drag controls. No answer preselected, no correctness message tied to old input, no hardcoded answer based on a preset name. Nonfinite/out-of-range/incomplete input disables commit with an adjacent explanation. Numeric values and grades come from one model state; no independent graph and table calculations. Controls are finite and bounded; no live arbitrary Python execution or neural training in the browser.

Every diagram needs a visible description, exact table and labels/line styles independent of color. On narrow screens stack dependency stages and preserve variable names. Live messages are concise and do not reread the whole graph. Respect reduced motion; no auto-running timelines. Numerical browser agreement, keyboard/screen-reader operation and actual desktop/mobile graph legibility are deferred checks, not author-verified renders.

## A. Forward/backward trace and update investigation

**Placement/purpose:** §1. Start with scalar x2,w3,b−1,y1; z5,a5,L8 for half-squared error. Forward values above nodes, reverse adjoints below; labels distinguish the coincident8loss/8weight-gradient. Fixed trace steps: products and bias;ReLU;error and half-square;seed1;loss→activation4;ReLU→score4;parameter/input branchesw8,b4,x12. Read-only Prev/Next shows these steps, with full textual table always accessible.

**Actual investigation:** separate clearly named “Two-example mean-squared error” fixture x=(1,2),y=(1,3),w1,b0. This is mean((wx+b−y)^2), not the preceding half-square. Show two observed points, current fitted line and residual segments. New prediction line remains hidden until commitment.

**Entity controls:** edit w,b∈[−3,3],η∈[0,1], targets y1,y2∈[−3,5]; step.1 with exact numeric entry. x1=1,x2=2 fixed and labeled. Formula gradients gw=mean(2r*x),gb=mean(2r), update both from the same pre-update state. Ask initially unset “loss decreases / is unchanged / increases.” Grade from old/new MSE with relative1e−10 and absolute1e−12 equality tolerance.

**Checked fixtures:** baseline MSE.5,gw−2,gb−1. η.1 givesw1.2,b.1,predictions1.3,2.5,MSE.17;η1givesw3,b1,predictions4,7,MSE12.5;η0nullMSE.5. Store matches calculated-inputs.updates. Optional analytic ηplot follows .5−5η+17η² and is explicitly the fixed baseline fixture; editing another input recomputes its curve from actual formulas, not this fixed polynomial.

**Feedback/transfer:** state residuals, both derivative sums, simultaneous update and new loss. Explain why a harmful large update does not show a bad derivative. Independent task: edit a target or starting parameter and find a step with lower loss; prediction remains required. Do not auto-solve an optimal learning rate.

**Mobile/text:** graph followed by before/after table ofx,target,prediction,residual; all arithmetic visible without color. Distinguish original and updated line by stroke pattern and label. Fixed x means the horizontal axis does not jump when weights change.

## B. Repeated paths and cancellation

**Placement:** §2 beside u=x*x,L=u+c*u. Graph has a true shared u node, two consumers and two distinct x input slots. A visited-node count must not erase an operand edge. Keep the operation graph, not a generic slider panel.

**Initial:** x3,c2,u9,L27. Prediction asks numeric dL/dx, initially blank. Editable x∈[−3,3],c∈[−3,3],step.25. Apply creates a new forward state; graph results are hidden until commit/reveal. Grade derivative2x(1+c) using1e−9 absolute/relative tolerance.

**Reveal sequence:** seedL1; contributions1 andc accumulate at u; then each multiply slot contributes(1+c)x to x. Show both arrows and their sum. A side table lists forward values and incoming contributions. The same quantity cannot be labeled “gradient at u” before its second consumer has contributed.

**Checked contrast/null:** x3,c2→grad18,L27; c0→grad6,L9; c−1→grad0,L0. At c−1, further x edits leave loss/gradient zero; null follows the exact algebra, to be explicitly browser-checked at x1.5 andx−2 in phase2. At x0 all these c settings give zero derivative; provide this second null rather than suggesting every edit changes output.

**Independent edit:** choose a negative x and a coefficient below−1, predict gradient sign and value. c and x are direct graph entities, not answers in presets. Feedback explains both chain multiplication and branch addition; no arbitrary semantic grading of explanatory prose.

**Rendering:** use duplicated small x labels only as labeled aliases of one node when a phone cannot show both return edges; announce shared identity explicitly. Color cannot be the only indication of shared parameters. No randomized graph generation necessary.

## C. Broadcasting correspondence

Static §3figure:3×2 output sensitivities[[1,2],[3,4],[5,6]], shared bias shape(2,), reverse sums9,12. One bias cell appears once and has three forward-use arrows; backward arrows converge. Labels name row/example and column/feature, not batch as a neuron. An exact table repeats each column calculation. On phone show one feature-column at a time with both totals retained. The numbers are direct integer sums; verify sums and shape before rendering. An average here is incorrect; distinguish downstream reduction already included in G.

## D. Two-layer shape/pullback map

§3static paired lanes: X[N,D]→W1[D,H],b1[H]→Z1/A1[N,H]→W2[H,C],b2[C]→logits/probabilities[N,C]→scalar mean CE. Reverse follows G2=(P−onehot)/N;W2=A1ᵀG2;b2=row sum;A1=G2W2ᵀ;G1=A1gradient⊙(1−A1²);W1=XᵀG1;b1=row sum. Store input×output weights explicitly; PyTorch nn.Linear stores transposes. Do not reverse arrows without showing the actual contracted indices. Include a table of every shape as the text alternative. Phone uses vertically paired local forward/backward blocks; no tiny full-network screenshot.

## E. Recorded training observations

§3source calculated-inputs.xorTraining and digitTraining. Separate charts with full titles, objectives, denominators and known sample/update counts. XOR2→4→1tanh,seed3,lr.1,2000steps,allfourtruth-table rows; digits64→16→10tanh,seed4,lr.2,500steps,280train/120validation,split22,float64,known pixels/16. Ordinary full-batch gradient descent in teaching-autodiff.py. Do not imply a controlled comparison with Perceptrons' Adam/width32float32experiment.

XOR observed steps0,1,10,100,500,2000; digit0,1,10,100,250,500. Irregular update positions use actual numeric spacing. Keep step0(loss2.903633 and3.597477 respectively) visible; class count does not force random starting loss logC. Optional log y is labeled; XOR near1e−29 is finite arithmetic fit, not “negative loss” or unseen-data accuracy. A table preserves exact outputs and counts. Interpolating lines connect samples only.

Digit validation correct17,17,40,101,105,112 out of120, shown on full0–120count axis. No fabricated gradient norm, heatmap, neuron-death diagnosis or missing intermediate samples. If a paired image panel is added later, it requires retaining actual per-case results; this packet only retains final counts for this engine run. Data-source IDs and images remain in the CSV, but do not invent this model's predictions for individual rows.

Phase2 validates plot transformations, starting point and late behavior on phone/desktop; alternate table view always available.

## F. Finite-difference evidence laboratory

**Placement/question:** §4: does a smaller perturbation necessarily provide a better gradient check? Show a magnified local function panel plus the *actual two evaluated numbers* and a derivative/error table. This topic's object is a numerical experiment, not an abstract “precision” slider.

**Editable state:** function sin(x),C+x,orx²; x∈[−2,2] with numeric entry; offsetC∈{0,1e6,1e12} plus bounded numeric field[0,1e12] for the linear family; h from explicit powers10^0…10^−15. Use binary64 calculations initially. Do not offer a float32 toggle without implementing and testing float32 rounding at each intended primitive. Set the starting pair of candidate h values to1e−3 and1e−5 for sine at1.

**Prediction:** initially unset “smaller h improves / equal / worsens absolute derivative error.” Commit binds function,x,C,bothhvalues andnumericformat. Reveal each f(x±h),differencequotient,analyticderivative and absolute error. Compare errors using a documented tolerance tied to browser arithmetic; preserve exact zero separately. Relative error is available only when the chosen reference denominator is nonzero; show undefined otherwise.

**Checked actual fixtures:** sin at1: h.1error9.00053698e−4,h1e−3error9.00504502e−8,h1e−5error1.11408660e−11,h1e−9error2.96988523e−9,h1e−15error.0148092064. ForC1e12,x1,h1e−5 derivativeestimate0 vsanalytic1;h.1≈.999755859. x²at0 yields derivative0 and centraldifference0 for symmetric representable±h: null agreement with relative error undefined. ReLU0 is a separate fixed explanation fixture with central.5 and chosenAD0; do not classify it as an ordinary analytic derivative error.

**Contrast/precision limits:** exact browser libm results may differ in last bits from Python; compare qualitative order only where author margins are large, and retain the binary64 saved values as source-calculated reference. Use1e−5→1e−9 sine and1e−3→1e−5offset for robust worsening examples. Do not promise a smooth monotonic U curve at every sampledh. Optional log-log error chart uses saved observations, handles zero by a distinct baseline annotation and never applieslog(0).

**Transfer:** learner editsC andx to discover an unchanged mathematical derivative with altered numerical check; can varyh outside named presets through exponent entry. Feedback distinguishes truncation, cancellation, zero-reference reporting and nonsmoothness. Reset also clears magnification and prediction. x±h coordinates can coalesce at tinyh; text must show that honestly instead of drawing visibly separated points as measured distance.

## G. JVP versus VJP correspondence

§6static figure forf=(x1x2,sinx1,x2²),x(.3,.7),J[[.7,.3],[cos.3,0],[0,1.4]]. Input directionv(1,2)maps to(1.3,.955336489,2.8); output weightu(1,−1,2)pulls back to(−.255336489,3.1). Direction/sensitivity have units of output/input change as appropriate; bars show signed coordinate values, not probabilities. Label dimensions2→3 versus3→2. Inner-product check both5.944663510874394 from calculated-inputs.products. This is a correspondence demonstration, not an extra forced lab. Table lists matrix-vector calculations; mobile stacks the two distinct questions.

## H. Checkpoint storage and recomputation

§6static/stepped idealized eight-operation chain. State upfront: equal-size layer outputs and simple segmentation, not a measured network memory profile. Show boundary snapshots after operations0,4,8 and the local intermediates regenerated during the second segment's reverse pass, then first. Recomputedoperations have a distinct hatch/label; saved buffers occupy slots rather than arbitrary pixel area suggesting byte counts. Explain K+L/K conceptual storage and other uncounted memory. No fixed slowdown percentage.

Text timeline lists each operation/recompute/reverse action; mobile vertical timeline. In phase2 verify the schedule actually retains every needed input and does not count output/parameter storage twice in any displayed total. The accompanying deterministic checkpoint program has a derived expectedTrue, not a claimed executed performance comparison.
