// Generated from the complete prepared manuscript; semantic generator retains every section.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { OptimizerAnatomy, AdamHistoryFigure, LionWorkedFigure, CurvatureBowlFigure, CurvatureLanes, ScheduleWeightsFigure, OptimizerSelectionFigure, OptimizerPolarFigure, CurvatureJacobianFigure } from '../../components/lesson-labs/AdvancedOptimizerDiagrams.jsx';
import { LionDirectionLab, SophiaInstrumentsLab, ProdigyScaleLab, ScheduleFreeLab, OptimizerMemoryLab } from '../../components/lesson-labs/AdvancedOptimizerLabs.jsx';
import { OptimizerDigitLab, OptimizerPixelFigure, OptimizerHistoryFigure, OptimizerProgram } from '../../components/lesson-labs/AdvancedOptimizerStudy.jsx';
export default { title: 'Advanced Optimizers: Lion, Sophia, Prodigy and Schedule-Free', readTime: '~90 min read + investigations and practice', content: () => <div className="neural-lesson neural-lesson-neutral advanced-optimizer-lesson">
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Edit gradient/momentum, curvature estimate, step scale, averaging state and supported real digit input for one update. Show the exact update vector and state changes for Lion, Sophia, Prodigy and Schedule-Free, including where gradients and evaluation occur. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose a debugging question from sign, curvature, scale and averaging effects rather than extrapolating a universal optimizer ranking."}</Prose>

<Prose>{"Two people can receive the same directions and make different journeys. One takes a fixed stride, another slows down on a steep slope, and another changes stride after seeing how far they have travelled. A neural-network optimizer faces a related problem: the gradient tells it how the current loss changes locally, but does not specify a safe, useful next step."}</Prose>

<Prose>{"An "}<strong>{"optimizer"}</strong>{" turns gradients and stored history into changes to model parameters. Those changes affect the next prediction. An optimizer does not add attention heads, change the labels or discover information that the input lacks. Its job is to make better use of the learning signal that the model and objective provide."}</Prose>

<Prose>{"The preceding "}<a href={"/learn/path/full-curriculum/ring-attention-sequence-parallelism?module=deep-learning-fundamentals"}>{"Ring Attention lesson"}</a>{" reorganized a computation across devices while preserving its mathematical result. Changing the optimizer deliberately changes the learning trajectory. A faster update kernel and fewer updates to reach useful quality are separate advantages."}</Prose>

<Prose opening="route">{""}<strong>{"First pass:"}</strong>{" read the update anatomy, follow one calculation for each method, then inspect the handwriting experiment and memory accounting. Try the core practice before opening the optional theory and matrix-method branches. You need derivatives, weighted averages and basic probability; each is refreshed where it enters. The "}<a href={"/learn/path/full-curriculum/second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient?module=math-foundations"}>{"second-order methods lesson"}</a>{" supplies a deeper route through curvature."}</Prose>

<H2>{"1. What has to happen between a gradient and the next prediction?"}</H2>

<Prose>{"Suppose a classifier assigns only 0.6 probability to the correct handwritten digit. Its cross-entropy loss is −log(0.6). Backpropagation calculates how changing each weight would change that loss. A positive gradient component says that a small increase in that parameter raises this batch's loss; descent would move in the negative direction."}</Prose>

<Prose>{"That statement is local. If a slope changes rapidly, a long step can go past the useful region. A gradient from a small batch also contains sampling noise. Finally, millions of parameters can have very different scales. These are reasons to transform the gradient, rather than reasons to stop trusting calculus."}</Prose>

<Prose>{"We will use θ for the parameter vector, g for the current gradient, η for a learning rate and λ for a weight-decay coefficient. A subscript t identifies an update, not an example or an epoch. Multiplication, square roots and division between equally shaped parameter arrays are elementwise unless a dot product or matrix product is written explicitly."}</Prose>

<OptimizerAnatomy />

<NeuralTable caption={"1. What has to happen between a gradient and the next prediction?"} headers={[<>{"Method"}</>,<>{"Central question"}</>,<>{"Information it retains"}</>]} rows={[[<>{"AdamW"}</>,<>{"How large is the recent signed gradient relative to its recent squared magnitude?"}</>,<>{"First and second gradient moments"}</>],[<>{"Lion"}</>,<>{"Which direction does a blend of history and this gradient support?"}</>,<>{"One momentum buffer"}</>],[<>{"Sophia"}</>,<>{"How sharply does the loss change along each parameter coordinate?"}</>,<>{"Momentum and an estimated curvature diagonal"}</>],[<>{"Prodigy"}</>,<>{"Can observed progress supply a useful unknown step-scale estimate?"}</>,<>{"Scaled moments, displacement statistics and initialization"}</>],[<>{"Schedule-Free"}</>,<>{"At which parameters should we take gradients, and which parameters should we evaluate?"}</>,<>{"A fast trajectory, an averaged trajectory and any base-method scaling state"}</>]]} />

<Prose>{"These questions overlap. “Adaptive,” “second-order,” “parameter-free” and “schedule-free” name different properties. None means that data selection, validation or implementation details cease to matter."}</Prose>

<H2>{"2. AdamW gives us a concrete reference"}</H2>

<H3>{"Smooth direction and scale separately"}</H3>

<Prose>{"An exponential moving average keeps part of the old value and adds part of the new one:"}</Prose>

<Prose>{"mₜ = β₁mₜ₋₁ + (1−β₁)gₜ,"}</Prose>

<Prose>{"vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²."}</Prose>

<Prose>{"The first buffer keeps signed information. Opposite gradients can cancel. The second stores squared magnitude, so opposite signs cannot cancel. It is a "}<strong>{"raw second moment"}</strong>{", not a variance: variance would subtract the square of the mean. With β₂ different from β₁, even that subtraction needs care about the weighting scheme."}</Prose>

<Prose>{"If both buffers start at zero, their early values include too much of that initial zero. For a constant gradient g, the first recurrence produces mₜ=(1−β₁ᵗ)g. Dividing by 1−β₁ᵗ corrects this initialization effect. Thus"}</Prose>

<Prose>{"m̂ₜ = mₜ/(1−β₁ᵗ), v̂ₜ = vₜ/(1−β₂ᵗ),"}</Prose>

<Prose>{"θₜ₊₁ = (1−ηₜλ)θₜ − ηₜ m̂ₜ/(√v̂ₜ+ε)."}</Prose>

<Prose>{"The small positive ε stabilizes division. Its placement outside the square root is part of this algorithm, not cosmetic notation. The "}<a href={"https://arxiv.org/abs/1711.05101"}>{"AdamW paper"}</a>{" motivates separating shrinkage from adaptive scaling; the "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html"}>{"PyTorch 2.14 rule"}</a>{" makes the implemented order explicit."}</Prose>

<Prose>{"At the first step, take g=[2,−4], β₁=.9 and β₂=.999. Then m=[.2,−.4], v=[.004,.016], m̂=[2,−4], v̂=[4,16]. Ignoring only the tiny ε for mental arithmetic, the normalized direction is [1,−1]. The larger raw component does not produce a twice-as-large first step."}</Prose>

<Prose>{"With θ=[.5,−.7], η=.03 and λ=.1, shrinkage first gives [.4985,−.6979], followed by approximately [.4685,−.6679]. Our complete float64 implementation was compared with native PyTorch AdamW for four unequal gradients, including a zero gradient; all four maximum parameter differences were zero in that run. This verifies that particular calculation, not every optimizer in this lesson."}</Prose>

<AdamHistoryFigure />

<H3>{"Weight decay is its own change"}</H3>

<Prose>{"Adding λθ to the gradient is the derivative of an L2 penalty. In an adaptive optimizer, that addition also enters the moment buffers and is rescaled with the gradient. Decoupled weight decay instead directly multiplies the parameter by 1−ηλ. These operations are generally different."}</Prose>

<Prose>{"With no gradient contribution and constant ηλ=.01, ten shrinkage steps multiply a parameter by .99¹⁰. The product over steps, ∏ₜ(1−ηₜλ), explains why changing the learning-rate schedule changes the cumulative shrinkage even when λ is unchanged. A zero current gradient does not generally stop an optimizer with nonzero momentum."}</Prose>

<H2>{"3. Lion: keep a directional memory, discard the final magnitude"}</H2>

<Prose>{"Imagine receiving several “move left” instructions followed by one “move right.” It matters whether the new instruction is weak or strong relative to the stored history. Lion first blends those quantities, then keeps only the sign of that blend for this update:"}</Prose>

<Prose>{"uₜ = sign(β₁mₜ₋₁ + (1−β₁)gₜ),"}</Prose>

<Prose>{"θₜ₊₁ = (1−ηₜλ)θₜ − ηₜuₜ,"}</Prose>

<Prose>{"mₜ = β₂mₜ₋₁ + (1−β₂)gₜ."}</Prose>

<Prose>{"The direction uses the "}<strong>{"old"}</strong>{" momentum and β₁. The stored momentum then uses β₂. Replacing both with one average defines a different method. Lion's original defaults are β₁=.9 and β₂=.99. There is no second-moment denominator and no Adam-style bias correction in this rule. Its "}<a href={"https://github.com/google/automl/blob/master/lion/lion_pytorch.py"}>{"official Google implementation"}</a>{" is small enough to follow alongside the equations."}</Prose>

<Prose>{"For θ=.5, old m=.2, g=−1, η=.01 and λ=.2:"}</Prose>

<ol start={1}><li>{"Blend: .9(.2)+.1(−1)=.08."}</li><li>{"Direction: sign(.08)=+1, despite the current negative gradient."}</li><li>{"Parameter: .5(.998)−.01=.489."}</li><li>{"New memory: .99(.2)+.01(−1)=.188."}</li></ol>

<Prose>{"The old history wins this step. If the current gradient were negative enough to reverse the blend, the step would reverse. Gradient magnitudes are therefore still important "}<strong>{"before"}</strong>{" the sign and when storing history. The buffer cannot be replaced by one sign bit without changing the algorithm. A zero blend has sign zero; the gradient-driven update can be zero. Weight decay can make the total change larger or smaller than η."}</Prose>

<LionWorkedFigure />

<LionDirectionLab />

<H3>{"Tuning and an interesting origin"}</H3>

<Prose>{"The "}<a href={"https://arxiv.org/html/2302.06675v4"}>{"Lion paper's tuning section"}</a>{" recommends trying a learning rate roughly three to ten times smaller than its AdamW comparator, with a correspondingly larger λ to retain a similar ηλ product. That is an empirical starting range. It is not an equivalence theorem, a requirement to multiply every decay by 100, or a diagnosis of any later training failure."}</Prose>

<Prose>{"The striking origin of Lion is "}<strong>{"program search"}</strong>{". Candidate optimizer programs were mutated, screened on inexpensive tasks, selected on harder tasks and simplified. The two different blend coefficients survived that process. This is an application of automated discovery where the result is an inspectable algorithm. A proxy task can still favor the wrong behavior, so transfer to larger tasks and ablations remain essential. The resulting method is not a neural network secretly deciding updates at runtime."}</Prose>

<Prose>{"Lion can reduce persistent optimizer storage when that storage is the limiting resource. It does not halve activations, model weights or every communication collective. Its sign threshold can also make tiny numerical changes consequential near zero; precision, loss scaling and accumulation should be assessed in the actual training setup rather than reduced to “one floating-point format always works.”"}</Prose>

<H2>{"4. Sophia: distinguish a large gradient from high curvature"}</H2>

<H3>{"The shape of the slope"}</H3>

<Prose>{"For the one-dimensional bowl L(θ)=½h(θ−a)², the gradient is h(θ−a), and the second derivative is h. A large gradient may mean that we are far from a, that the bowl is sharp, or both. Dividing by h returns the displacement θ−a. With an exact quadratic and step multiplier one, Newton's update reaches a immediately."}</Prose>

<Prose>{"For many parameters, the "}<strong>{"Hessian"}</strong>{" H contains second derivatives. Off-diagonal entries describe coupling: changing one coordinate changes the slope in another. Storing a full matrix costs quadratic space in parameter count. Sophia estimates only the diagonal, smooths it, divides a momentum estimate by it and clips the resulting coordinate updates."}</Prose>

<CurvatureBowlFigure />

<Prose>{"Write ρ for the positive curvature scale in our implementation:"}</Prose>

<Prose>{"mₜ = β₁mₜ₋₁ + (1−β₁)gₜ,"}</Prose>

<Prose>{"hₜ = β₂h_previous + (1−β₂)ĥₜ on refresh steps; otherwise keep h unchanged,"}</Prose>

<Prose>{"θₜ₊₁ = (1−ηₜλ)θₜ − ηₜ clip(mₜ/max(ρhₜ,ε),−1,1)."}</Prose>

<Prose>{"Max and clipping are coordinatewise. The estimated diagonal is refreshed less frequently than the gradient. A negative or near-zero estimate receives the ε floor, and clipping prevents it from producing an arbitrarily large gradient-driven step. This bounds each such coordinate change by η; it does not certify that the whole loss decreases."}</Prose>

<Prose>{"For m=[.2,−.3,0], h=[5,.1,0] and ρ=.5, the ratios are [.08,−6,0] and the clipped direction is [.08,−1,0]. At θ=[1,2,3], η=.1 and λ=.2, the new parameters are [.972,2.06,2.94]. Only the second gradient-driven component is clipped. Decay still moves the third component."}</Prose>

<H3>{"Two different curvature estimators"}</H3>

<Prose>{""}<strong>{"Sophia-H uses Hessian-vector products."}</strong>{" Draw a random vector u with E[uuᵀ]=I and calculate u⊙(Hu). Its expected ith entry is Hᵢᵢ, because the cross terms have zero expectation. Gaussian probes are used in the paper; independent ±1 probes also satisfy this identity."}</Prose>

<Prose>{"For H=[[2,3],[3,1]], the probe [1,1] returns [5,4], while [1,−1] returns [−1,−2]. Averaging the four equally likely sign probes gives exactly [2,1]. An unbiased estimator can have negative individual entries. One sample is not a proof of negative true diagonal curvature. Automatic differentiation can form Hu without constructing H explicitly."}</Prose>

<Prose>{""}<strong>{"Sophia-G uses model-sampled labels."}</strong>{" For a classification model, form its current probabilities, sample an independent label for each example, and differentiate cross-entropy using those sampled labels. Square that mean gradient and multiply by batch size B:"}</Prose>

<Prose>{"ĥ = B ĝ², ĝ = ∇θ mean_b CE(logits_b, sampled_label_b)."}</Prose>

<Prose>{"The labels used to train the model remain the real labels. Sampled labels are a separate instrument for estimating curvature. Squaring the ordinary real-label gradient is not this estimator."}</Prose>

<Prose>{"Here is an exact binary example. Let the input be x=2, predicted positive-class probability p=.8, and the real label be 1. The real-label gradient is x(p−1)=−.4; its square is .16. The curvature of the logistic loss in its scalar weight is x²p(1−p)=.64. If we sample label 1 with probability .8 and label 0 with probability .2, the expected squared gradient is"}</Prose>

<Prose>{".8(−.4)² + .2(1.6)² = .64."}</Prose>

<Prose>{"This exposes the key distinction without a giant neural model. For a two-example batch x=[2,1], p=[.8,.3], enumerating all four sampled-label pairs gives E[2ĝ²]=.425, the average of the two exact diagonals. Omitting B gives half that value."}</Prose>

<CurvatureLanes />

<SophiaInstrumentsLab />

<section data-lesson-teaching="" className="lesson-teaching-section">

<h3 className="lesson-teaching-section__title">Deeper: what curvature does Sophia-G actually estimate?</h3>

<Prose>{"Let J have shape classes × parameters, the Jacobian of logits with respect to parameters, and p be the probability vector. The generalized Gauss–Newton matrix for softmax cross-entropy is"}</Prose>

<Prose>{"G = Jᵀ[diag(p)−ppᵀ]J."}</Prose>

<Prose>{"The full Hessian additionally includes second derivatives of the logits weighted by the loss's logit derivatives. For a linear classifier those second derivatives vanish, so G equals the Hessian. For a nonlinear network they generally do not vanish. Sophia-G is unbiased for the G diagonal under the sampled-label construction; it is not generally unbiased for the full Hessian diagonal."}</Prose>

<CurvatureJacobianFigure />

<Prose>{"The batch factor follows because independent sampled-label gradients have mean zero. When we square their sum, expected cross-example terms vanish. With unequal example weights, correlated sampling, masks or distributed averaging, derive the normalization for the actual reduction; do not copy an unrelated "}<code>{"bs"}</code>{" default. A formula involving only probabilities and squared logits misses the parameter Jacobian and cannot be the general parameter-curvature formula."}</Prose>

</section>

<H3>{"Implement the refresh at a coherent parameter state"}</H3>

<Prose>{"The paper uses a batch-scaled estimate and a maximum with ε. The official Sophia-G source instead stores the unscaled squared sampled gradient and multiplies by "}<code>{"bs"}</code>{" inside its denominator, adding a small ε. Those conventions must be matched deliberately. Using both B-scaled storage and "}<code>{"bs=B"}</code>{" would apply the factor twice."}</Prose>

<Prose>{"For a neural training loop, compute a fresh forward/backward for the sampled-label estimate at the intended parameter state, clear those gradients, then compute the real-label gradient for the update. Do not reuse a freed graph or reuse pre-update logits after mutating weights. The complete classroom program avoids this ambiguity by explicitly evaluating both gradients at the same parameter array."}</Prose>

<Prose>{"The "}<a href={"https://arxiv.org/html/2305.14342v4"}>{"Sophia paper"}</a>{" reports improvements on its specified language-model tasks and budgets. Its result is a reason to test a faithful implementation, not a guaranteed twofold speedup on another model. Measure refresh cost, clipped fraction, validation quality and actual elapsed training time. A clipped momentum method is not automatically Lion: their histories and update rules differ."}</Prose>

<H2>{"5. Prodigy: use progress to estimate an unknown scale"}</H2>

<Prose>{"A learning rate that is sensible for one parameterization can be poor for another. In convex optimization, useful step-size bounds often contain the unknown distance D from initialization to a solution. "}<strong>{"D-adaptation"}</strong>{" methods try to estimate a useful distance scale while learning. Prodigy changes that adaptation so the estimate can grow more effectively."}</Prose>

<Prose>{"The basic signal compares the current gradient with displacement from initialization. Suppose we have moved in a direction that earlier gradients supported, and the current gradient still says there is useful progress in that direction. Their agreement supplies information about the scale of the problem. Cancellation and reversal supply different information. This is not an exact oracle for the location of a neural-network optimum."}</Prose>

<H3>{"Follow a fully specified Adam-style version"}</H3>

<Prose>{"We use Algorithm 4 of the "}<a href={"https://arxiv.org/html/2306.06101v3"}>{"Prodigy paper"}</a>{", without weight decay or optional bias correction. Let d start at a positive d₀; m, v, s and scalar r start at zero. Let γ be a user multiplier, and b=√β₂. At a step with current θ, g and d:"}</Prose>

<Prose>{"m_new = β₁m + (1−β₁)d g,"}</Prose>

<Prose>{"v_new = β₂v + (1−β₂)d²g²,"}</Prose>

<Prose>{"r_new = b r + (1−b)γd²〈g, θ_initial−θ〉,"}</Prose>

<Prose>{"s_new = b s + (1−b)γd²g,"}</Prose>

<Prose>{"d_estimate = r_new / ||s_new||₁,"}</Prose>

<Prose>{"d_next = max(d, d_estimate),"}</Prose>

<Prose>{"θ_next = θ − γd m_new/(√v_new+dε)."}</Prose>

<Prose>{"The norm ||s||₁ adds absolute coordinate values. The numerator is a scalar sum across parameters, not a per-coordinate learning rate. If its denominator is zero, keep the previous d rather than divide by zero. The parameter update shown here uses the "}<strong>{"current d"}</strong>{", while d_next is for the next step. The momentum buffers themselves include d and d²; inserting d into an otherwise unchanged Adam update is not Algorithm 4."}</Prose>

<ProdigyScaleLab worked />

<Prose>{"On the declared bowl ½(θ−3)², start θ=0 and d₀=.01, using γ=1, β₁=.9 and β₂=.999. The first parameters are .0316228, .0741064 and .1514404. The distance used for those steps is .01, .01 and .0157316. At step 7 the parameter overshoots to 7.11008; d never decreases in this rule. The trace is useful precisely because automatic scale growth does not mean monotonically improving loss."}</Prose>

<ProdigyScaleLab />

<H3>{"What “parameter-free” leaves for the practitioner"}</H3>

<Prose>{"Prodigy aims to remove the need to supply a well-tuned problem-scale learning rate. It still has initialization, betas, ε, a multiplier, regularization and variant choices. The theoretical guarantees apply to the stated convex algorithms and assumptions; they are not an assertion that the Adam-style version solves every nonconvex training problem without tuning."}</Prose>

<Prose>{"The "}<a href={"https://github.com/konstmish/prodigy"}>{"official package"}</a>{" recommends starting with "}<code>{"lr=1"}</code>{", offers "}<code>{"d_coef"}</code>{", optional bias correction and sliced adaptation statistics, and permits constant or cosine schedules. Its warmup safeguard changes the estimator normalization; it does not merely wait a fixed number of steps before allowing d to grow. Current library arithmetic also differs from the compact paper presentation in numerical rescaling and update details. Record which implementation you use before comparing trajectories."}</Prose>

<Prose>{"A practical application is training many small models with different scales, where a full learning-rate search for every model is expensive. Architecture search and multiple objectives provide such situations. A fair study still includes the tuning compute that was actually saved, failures that required reruns and validation quality. “No sweep needed” cannot be asserted from a single favorable run."}</Prose>

<H2>{"6. Schedule-Free: train and evaluate at different points"}</H2>

<Prose>{"Learning-rate schedules often assume that you know the final update count. If you plan 100,000 steps and later extend the run, the decay may already have changed your trajectory substantially. Schedule-Free offers another strategy: maintain a fast-changing sequence and an average, and evaluate the loss gradient between them."}</Prose>

<Prose>{"Call the fast parameters z, the averaged parameters x, and the training parameters y. For the simplest SGD form, use"}</Prose>

<Prose>{"yₜ = βxₜ + (1−β)zₜ,"}</Prose>

<Prose>{"zₜ₊₁ = zₜ − η∇L(yₜ),"}</Prose>

<Prose>{"xₜ₊₁ = (1−1/t)xₜ + (1/t)zₜ₊₁ for our one-based update index t."}</Prose>

<Prose>{"At t=1 the average becomes the first updated fast point. This indexing convention averages the post-update z values. Gradients are calculated at y; validation and inference use x. Neither “always use z” nor “average predictions” describes this method."}</Prose>

<Prose>{"Take ½θ², start x=z=2, β=.9 and η=.2. Add the declared illustrative gradient perturbations [1,−1,.5,−.5]. These are arithmetic inputs, not measured training noise."}</Prose>

<NeuralTable caption={"6. Schedule-Free: train and evaluate at different points"} headers={[<>{"Update"}</>,<>{"Training point y"}</>,<>{"Perturbed gradient"}</>,<>{"New fast point z"}</>,<>{"New average x"}</>]} rows={[[<>{"1"}</>,<>{"2"}</>,<>{"3"}</>,<>{"1.4"}</>,<>{"1.4"}</>],[<>{"2"}</>,<>{"1.4"}</>,<>{".4"}</>,<>{"1.32"}</>,<>{"1.36"}</>],[<>{"3"}</>,<>{"1.356"}</>,<>{"1.856"}</>,<>{".9488"}</>,<>{"1.222933"}</>],[<>{"4"}</>,<>{"1.19552"}</>,<>{".69552"}</>,<>{".809696"}</>,<>{"1.119624"}</>]]} />

<Prose>{"The third gradient is evaluated at 1.356, neither 1.32 nor 1.36. That small distinction changes every later step. At the end, report the loss at x, not whichever point happens to look best."}</Prose>

<ScheduleFreeLab worked />

<H3>{"Averaging does not mean exponential decay"}</H3>

<Prose>{"With equal weights, x after t updates is the average of those t post-update z values. Each has final coefficient 1/t. The newest value's insertion coefficient is 1/t, but older values are subsequently diluted too. It is incorrect to compare their insertion coefficients as though those were their final weights."}</Prose>

<Prose>{"This is not the same as a momentum EMA, nor the same as applying a 1/t learning rate directly to the current gradient. Earlier gradients influence many later z values. The equations specify that influence; there is no universal “effective schedule” curve that makes every method identical."}</Prose>

<ScheduleFreeLab />

<H3>{"The AdamW form and the warmup weights"}</H3>

<Prose>{"Schedule-Free AdamW replaces the SGD step with a second-moment-scaled gradient. In the base form used here there is no ordinary first-moment EMA: interpolation supplies the momentum-like behavior. Update v with g², bias-correct v, and use g/(√v̂+ε) to move z. Decay can be calculated at y, as in the paper's algorithm."}</Prose>

<Prose>{"Warmup still changes η during the opening updates. With the paper's weighting, let wₜ=ηₜ² and cₜ=wₜ/Σᵢ≤ₜwᵢ, then average with cₜ. If the first learning rates are .1,.2,.3, the three post-update z values have normalized weights [1,4,9]/14 after step 3. A plain one-third average is a different calculation. Our monotone warmup then constant rate matches this weighting; the current library's maximum-rate weighting also agrees for this schedule."}</Prose>

<ScheduleWeightsFigure />

<H3>{"The training/evaluation switch changes parameters"}</H3>

<Prose>{"The efficient "}<a href={"https://github.com/facebookresearch/schedule_free"}>{"Schedule-Free library"}</a>{" stores only the sequences it needs and reconstructs the other. Calling "}<code>{"optimizer.eval()"}</code>{" changes the parameter buffer to x; "}<code>{"optimizer.train()"}</code>{" restores y. These calls are separate from "}<code>{"model.eval()"}</code>{" and "}<code>{"model.train()"}</code>{", which affect such layers as dropout and BatchNorm."}</Prose>

<Prose>{"For BatchNorm models, statistics collected at y may not match x. Recompute appropriate running statistics using training inputs at the evaluation weights, following the method's guidance. Do not use validation labels or validation examples to fit those statistics. Save checkpoints with the optimizer in its documented mode and retain optimizer state if training will resume."}</Prose>

<Prose>{"Validating at y is a different measurement, but it is not guaranteed to be worse on every batch. Save both values in a diagnostic; compare models at the evaluation point defined by the method. A framework's automatic scheduler or missing optimizer-mode hook can silently change the intended recipe."}</Prose>

<section data-lesson-teaching="" className="lesson-teaching-section">

<h3 className="lesson-teaching-section__title">Deeper: a theorem is not a universal anytime guarantee</h3>

<Prose>{"The Schedule-Free paper's introductory SGD bound assumes convex, Lipschitz stochastic losses with independent samples. Its displayed choice η=D/(G√T) contains the horizon T even though the practical update can run with a constant chosen rate. The broader online-to-batch result and the discussion of larger rates explain more of the connection. It would be incorrect to cite the introductory bound as proof that any constant rate converges optimally at every stopping time on a nonconvex network."}</Prose>

<Prose>{"The useful practical distinction is that an evaluation iterate exists at each update without prescribing a decay endpoint. Constant-rate SGD, alternative schedules and restart strategies also exist; Schedule-Free is not the only imaginable way to extend a run. If the data distribution changes, averaging across the full old history may be inappropriate. That is a new learning problem, not a consequence that the original stationary analysis settles."}</Prose>

</section>

<H2>{"7. Train the same real classifier with different update rules"}</H2>

<H3>{"Fix the learning problem before comparing the optimizers"}</H3>

<Prose>{"Our practical task is to recognize ten handwritten digits from 64 pixel features. We use 400 actual images, forty per digit, from the UCI Optical Digits data distributed through scikit-learn. These are 8×8 block-count images, not MNIST. Pixel values range from 0 to 16; divide by 16 and append a constant 1 for a bias feature."}</Prose>

<Prose>{"The model is a linear softmax classifier. Its parameter array Θ has 65 rows and 10 columns, giving 650 parameters. For a batch X, the score matrix is XΘ; a row-wise softmax produces ten probabilities per image. The loss is the mean negative log probability of the correct label. Inference chooses the largest score. This small model makes the optimizer's behavior inspectable without confusing it with architecture changes."}</Prose>

<Prose>{"The gradient is"}</Prose>

<Prose>{"∇ΘL = Xᵀ(P−Y)/B,"}</Prose>

<Prose>{"where P contains predicted probabilities and Y has a one in each example's correct-label column. A pixel's gradient contribution is its intensity multiplied by a class residual. The bias row uses intensity one. For a nonlinear model, backpropagation supplies the corresponding parameter gradients; the optimizer then consumes arrays of those gradients."}</Prose>

<OptimizerPixelFigure />

<Prose>{"Use a fixed stratified split: 240 fitting images, 80 validation images and 80 assessment images. Source identities and full pixel signatures are distinct across these roles. The subset comes from the historical UCI test file, so this new classroom partition is not the official UCI train/test evaluation. Writer identities are unavailable here, and the data have appeared in earlier lessons. We use them to study mechanisms, not claim a new untouched benchmark."}</Prose>

<Prose>{"We declare the experiment before fitting:"}</Prose>

<ul><li>{"Two seeds, 11 and 29; the same initial parameter array and minibatch sequence for every method at a given seed."}</li><li>{"Exactly 400 updates per candidate, batches of 64 fitting images sampled with replacement; no early stopping, augmentation or weight decay."}</li><li>{"Two candidate scales per method: AdamW constant/cosine and Schedule-Free [.01,.03]; Lion [.003,.01]; Sophia-G [.01,.03]; paper-version Prodigy multipliers [.3,1]. Other method settings are those specified above."}</li><li>{"Twenty-step warmup for cosine AdamW and Schedule-Free. Cosine decays to zero at update 400. Constant AdamW and the other methods use their stated constant scales."}</li><li>{"Sophia-G refreshes at updates 1,11,…,391: forty additional sampled-label gradient evaluations per fit. They are computed at the same parameters as the real gradient. No full Hessian is formed."}</li><li>{"Select one scale per method by "}<strong>{"mean final validation cross-entropy across the two seeds"}</strong>{", then assess both selected runs. All 24 candidate curves are retained; no candidate or seed is erased because it is less flattering."}</li></ul>

<Prose>{"The two-value grids are deliberately small and method-specific. Equal numbers of candidates do not prove equally good tuning. Equal updates do not imply equal work: Sophia has extra gradient evaluations, and schedules deliberately change the update scales. There are no CPU or GPU speed measurements in this comparison."}</Prose>

<OptimizerSelectionFigure />

<H3>{"Actual outcomes"}</H3>

<NeuralTable caption={"Actual outcomes"} headers={[<>{"Selected method"}</>,<>{"Scale"}</>,<>{"Assessment cross-entropy, seeds 11 / 29"}</>,<>{"Correct out of 80, seeds 11 / 29"}</>]} rows={[[<>{"AdamW, constant"}</>,<>{".03"}</>,<>{".06075 / .05581"}</>,<>{"80 / 80"}</>],[<>{"AdamW, warmup + cosine"}</>,<>{".03"}</>,<>{".09489 / .09190"}</>,<>{"79 / 80"}</>],[<>{"Lion"}</>,<>{".003"}</>,<>{".09008 / .08580"}</>,<>{"78 / 78"}</>],[<>{"Sophia-G, paper-scaled estimator"}</>,<>{".01"}</>,<>{".02786 / .02523"}</>,<>{"79 / 79"}</>],[<>{"Prodigy, paper Algorithm 4"}</>,<>{"1"}</>,<>{".03591 / .04906"}</>,<>{"79 / 79"}</>],[<>{"Schedule-Free AdamW, evaluated at x"}</>,<>{".03"}</>,<>{".07985 / .08085"}</>,<>{"79 / 80"}</>]]} />

<Prose>{"A constant-class baseline gets 8 of 80 correct; uniform probabilities have loss log(10)≈2.30259. Every selected model learns useful structure. The more interesting comparison is between the two metrics: AdamW gets every assessment label right here, while Sophia assigns probabilities that give a lower average cross-entropy despite one error. Classification accuracy counts decisions; cross-entropy also measures how probability was allocated."}</Prose>

<Prose>{"Do not choose a universal winner from those eighty images. The validation results differ substantially: Prodigy's final losses are .41470 and .49441, while Schedule-Free's are .19111 and .19220. The small partitions expose different cases. An apparently excellent assessment number does not justify retuning on that partition."}</Prose>

<Prose>{"The Schedule-Free diagnostic also illustrates its mode contract. For seed 11, final validation loss is .19111 at x and .19281 at y. For seed 29 it is .19220 at x and .18660 at y. The training iterate happens to score lower in the second case. We still report x because that is the method's defined evaluation model, not because it wins every comparison."}</Prose>

<OptimizerHistoryFigure />

<Prose>{"For Prodigy, a companion trace shows d rather than pretending it is exactly the AdamW learning rate. In these selected runs its final values are .07640 and .09739. For Sophia, show the actual clipped fraction; at the final update it is about .03385 and .06. Those diagnostic quantities help explain an update, but neither alone measures successful learning."}</Prose>

<H3>{"Reproduce and investigate"}</H3>

<Prose>{"Download "}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/digits-400.csv"}>{"the data"}</a>{", "}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/optimizer_rules.py"}>{"the complete update rules"}</a>{" and "}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/optimizer_study.py"}>{"the complete study program"}</a>{" into one directory. The "}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/data-provenance.md"}>{"provenance"}</a>{" documents attribution, source identities, split construction and versions. The program needs Python and NumPy; the optional calculation checker also uses PyTorch. This packet ran with Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu."}</Prose>

<CodeBlock language={"text"}>{"python -m venv .venv"}</CodeBlock>

<Prose>{"Activate that environment using its platform's normal activation command, then run:"}</Prose>

<CodeBlock language={"text"}>{"python -m pip install numpy==2.3.5\npython optimizer_study.py"}</CodeBlock>

<Prose>{"The program prints the twelve selected assessment records and writes all candidate histories to "}<code>{"study-results.json"}</code>{". "}<code>{"fitted-optimizer-states.json"}</code>{" retains the selected models' full weights and optimizer state, including Schedule-Free's different iterates. The exact arithmetic/probe checker is "}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/optimizer_calculations.py"}>{"optimizer_calculations.py"}</a>{"; it also needs the saved study files and PyTorch. The files contain complete programs, with no missing loader, model or loss function."}</Prose>

<OptimizerProgram file="optimizer_rules.py" /><OptimizerProgram file="optimizer_study.py" /><OptimizerProgram file="optimizer_calculations.py" />

<OptimizerDigitLab />

<Prose>{"This is a temporary copy of the fitted model, not an alteration of the reported experiment. For example, changing pixel 28 of source 277 from 0 to 16 and applying one AdamW update toward its original class 0 changes its class-0 probability from .97019 to .97777. That is evidence about this edited example and copied state. It does not establish improved generalization. The Sophia diagnostic can use the exact expected-label diagonal for this linear model; label it distinctly from the sampled estimator used during fitting."}</Prose>

<Prose>{"Cosine AdamW has already reached zero learning rate at update 400. Continuing its frozen schedule gives no parameter change at the next step. That is an informative control, not a reason to silently restart its schedule to produce a visible animation. Restoring the original snapshot must recover the same weights and predictions."}</Prose>

<H3>{"Use all four optimizer APIs, then resume the same process"}</H3>

<Prose>{""}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/optimizer_rules.py"}>{"optimizer_rules.py"}</a>{" owns the complete scratch update for each named method, including its persistent arrays. The real digit experiment above uses those rules. The next program provides a second capability: taking control of the ordinary optimizer objects people put into a PyTorch loop. It keeps the objective and initial weight matrix fixed across methods, but it is an eight-update API study, not a ranking of optimizer quality."}</Prose>

<Prose>{"Use "}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/optimizer_library_bridge.py"}>{"optimizer_library_bridge.py"}</a>{" beside "}<code>{"optimizer_rules.py"}</code>{". Its authoring targets are PyTorch 2.14.0, "}<code>{"prodigyopt==1.1.2"}</code>{" and "}<code>{"schedulefree==1.4.1"}</code>{". Install those packages in the lesson environment. For the other two, the exact original licensed author files are provided as "}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/lion_pytorch.py"}>{"lion_pytorch.py"}</a>{" and "}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/sophia.py"}>{"sophia.py"}</a>{", with their "}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/LICENSE-lion.txt"}>{"Apache license"}</a>{" and "}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/LICENSE-sophia.txt"}>{"MIT license"}</a>{". Place them beside the bridge. Their pinned upstream sources are: "}<a href={"https://raw.githubusercontent.com/google/automl/b21d6ced9bc9b748e1c8ab9fdebf2c44c57e63ae/lion/lion_pytorch.py"}>{"Google's pinned Lion source"}</a>{" as "}<code>{"lion_pytorch.py"}</code>{", and "}<a href={"https://raw.githubusercontent.com/Liuhong99/Sophia/2fc52f24d4bc008658111b0237a70953ada22398/sophia.py"}>{"the pinned Sophia source"}</a>{" as "}<code>{"sophia.py"}</code>{". Preserve their license notices. These are explicit imported dependencies, not unnamed implementations the learner has to invent. Run the program with "}<code>{"--method lion"}</code>{", "}<code>{"--method sophia"}</code>{", "}<code>{"--method prodigy"}</code>{", or "}<code>{"--method schedule_free"}</code>{"."}</Prose>

<Prose>{"The state map explains which equivalence we can legitimately claim:"}</Prose>

<NeuralTable caption={"Use all four optimizer APIs, then resume the same process"} headers={[<>{"Scratch quantity"}</>,<>{"Package quantity/control"}</>,<>{"Comparison boundary"}</>]} rows={[[<>{"Lion m and two β values"}</>,<>{""}<code>{"exp_avg"}</code>{", "}<code>{"betas=(.9,.99)"}</code>{""}</>,<>{"Same gradient, initialization, decay and rate give matched parameters and momentum; the program checks every update"}</>],[<>{"Sophia sampled curvature EMA of B times squared mean gradient"}</>,<>{""}<code>{"hessian"}</code>{" stores the unscaled squared-gradient EMA; "}<code>{"step(bs=B)"}</code>{" supplies B"}</>,<>{"Multiply once. "}<code>{"bs"}</code>{" counts terms in this mean cross-entropy, not the number of loader batches"}</>],[<>{"Prodigy d, initial weights, displacement history and moments"}</>,<>{"Group "}<code>{"d"}</code>{", "}<code>{"d0"}</code>{", "}<code>{"d_numerator"}</code>{", parameter "}<code>{"p0"}</code>{", "}<code>{"s"}</code>{", "}<code>{"exp_avg"}</code>{", "}<code>{"exp_avg_sq"}</code>{""}</>,<>{"The original package has implementation conventions beyond paper Algorithm 4; matching names does not imply identical trajectories"}</>],[<>{"Schedule-Free training y, evaluation x and fast z"}</>,<>{"Parameter buffer changes under "}<code>{"optimizer.train()"}</code>{"/"}<code>{".eval()"}</code>{"; z and second moment persist"}</>,<>{""}<code>{"model.eval()"}</code>{" alone does not select x. Evaluate and save the intended iterate"}</>]]} />

<Prose>{"For Sophia we refresh the estimate at the current parameters every second update, clear those sampled-label gradients, and calculate a fresh real-label gradient before "}<code>{"step"}</code>{". There is no accidental reuse of curvature-probe gradients as the training gradient. The package denominator adds an epsilon; the scratch rule uses a denominator floor. The difference matters near zero and is another reason not to promise bitwise identity. The scaling and method calls come from "}<a href={"https://github.com/Liuhong99/Sophia"}>{"the original Sophia implementation"}</a>{"."}</Prose>

<Prose>{"Prodigy settings deliberately expose bias correction, safeguard behavior, the initial distance and statistic subsampling. This route uses full statistics ("}<code>{"slice_p=1"}</code>{") and turns the first two options off, but still identifies its output as package behavior rather than relabeling it paper-Algorithm-4 parity. "}<a href={"https://github.com/konstmish/prodigy"}>{"Original Prodigy implementation"}</a>{". Schedule-Free 1.4.1 includes the post-1.3 warmup/decay behavior; the reference paper variant is a distinct class. Weight decay is zero in this small comparison. "}<a href={"https://github.com/facebookresearch/schedule_free"}>{"Author usage and release notes"}</a>{"."}</Prose>

<section data-lesson-teaching="" className="lesson-teaching-section"><h3 className="lesson-teaching-section__title">Read the complete runnable library bridge</h3><CodeBlock language={"python"}>{"\"\"\"Four actual optimizer APIs on one small fixed classification problem.\n\nTargets: torch 2.14.0, prodigyopt 1.1.2, schedulefree 1.4.1; the lesson\npins the original Lion/Sophia source files. Choose --method explicitly.\nThis is an API/state demonstration, not an optimizer quality benchmark.\n\"\"\"\nimport argparse\nfrom copy import deepcopy\nfrom importlib.util import module_from_spec, spec_from_file_location\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch.nn import functional as F\nfrom optimizer_rules import Optimizer\n\n\ndef source_class(filename, class_name):\n    path = Path(__file__).with_name(filename)\n    spec = spec_from_file_location(path.stem, path)\n    module = module_from_spec(spec)\n    spec.loader.exec_module(module)\n    return getattr(module, class_name)\n\n\ndef make_optimizer(parameters, method):\n    if method == \"lion\":\n        return source_class(\"lion_pytorch.py\", \"Lion\")(\n            parameters, lr=.01, betas=(.9, .99), weight_decay=0.)\n    if method == \"sophia\":\n        return source_class(\"sophia.py\", \"SophiaG\")(\n            parameters, lr=.01, betas=(.965, .99), rho=.04, weight_decay=0.)\n    if method == \"prodigy\":\n        from prodigyopt import Prodigy\n        return Prodigy(parameters, lr=1., betas=(.9, .999), d0=1e-6,\n                       weight_decay=0., use_bias_correction=False,\n                       safeguard_warmup=False, slice_p=1)\n    if method == \"schedule_free\":\n        from schedulefree import AdamWScheduleFree\n        return AdamWScheduleFree(parameters, lr=.01, betas=(.9, .999),\n                                  weight_decay=0., warmup_steps=2, foreach=False)\n    raise ValueError(method)\n\n\ndef switch(model, optimizer, method, training):\n    model.train(training)\n    if method == \"schedule_free\":\n        optimizer.train() if training else optimizer.eval()\n\n\ndef update(model, optimizer, method, inputs, labels, step):\n    switch(model, optimizer, method, True)\n    optimizer.zero_grad(set_to_none=True)\n    if method == \"sophia\" and step % 2 == 0:\n        # Curvature and true-label gradients are evaluated at the same parameters.\n        logits = model(inputs)\n        fake_labels = torch.distributions.Categorical(logits=logits.detach()).sample()\n        F.cross_entropy(logits, fake_labels, reduction=\"mean\").backward()\n        optimizer.update_hessian()   # EMA of squared mean gradient, not B*g^2\n        optimizer.zero_grad(set_to_none=True)\n    F.cross_entropy(model(inputs), labels, reduction=\"mean\").backward()\n    if method == \"sophia\":\n        optimizer.step(bs=len(inputs))   # B is applied once, in Sophia's denominator\n    else:\n        optimizer.step()\n\n\ndef build(method):\n    model = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)\n    with torch.no_grad():\n        model.weight.copy_(torch.tensor([[.1, -.2], [-.1, .2], [.05, -.05]]))\n    return model, make_optimizer(model.parameters(), method)\n\n\ndef main(method):\n    torch.manual_seed(23)\n    inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]], dtype=torch.float64)\n    labels = torch.tensor([0, 1, 2, 1])\n    model, optimizer = build(method)\n    reference = Optimizer(model.weight.detach().numpy(), \"lion\", .01) if method == \"lion\" else None\n    checkpoint = None\n    for step in range(8):\n        if reference is not None:\n            loss = F.cross_entropy(model(inputs), labels)\n            gradient = torch.autograd.grad(loss, model.weight)[0].detach().numpy()\n            reference.step(gradient)\n        update(model, optimizer, method, inputs, labels, step)\n        if reference is not None:\n            np.testing.assert_allclose(model.weight.detach().numpy(), reference.parameters,\n                                       atol=1e-12, rtol=1e-12)\n            np.testing.assert_allclose(optimizer.state[model.weight][\"exp_avg\"].numpy(),\n                                       reference.state[\"momentum\"], atol=1e-12)\n        if step == 3:\n            switch(model, optimizer, method, False)\n            checkpoint = deepcopy({\"model\": model.state_dict(), \"optimizer\": optimizer.state_dict(),\n                                   \"rng\": torch.get_rng_state(), \"next_step\": step+1})\n    switch(model, optimizer, method, False)\n    with torch.no_grad():\n        expected_logits = model(inputs).clone()\n    resumed, resumed_optimizer = build(method)\n    resumed.load_state_dict(checkpoint[\"model\"])\n    resumed_optimizer.load_state_dict(checkpoint[\"optimizer\"])\n    torch.set_rng_state(checkpoint[\"rng\"])\n    for step in range(checkpoint[\"next_step\"], 8):\n        update(resumed, resumed_optimizer, method, inputs, labels, step)\n    switch(resumed, resumed_optimizer, method, False)\n    with torch.no_grad():\n        torch.testing.assert_close(resumed(inputs), expected_logits, rtol=1e-12, atol=1e-12)\n        print(method, \"evaluation loss:\", F.cross_entropy(expected_logits, labels).item())\n    print(\"State keys:\", sorted(optimizer.state[model.weight]))\n    print(\"Uninterrupted/resumed evaluation maximum error:\",\n          (resumed(inputs).detach()-expected_logits).abs().max().item())\n\n\nif __name__ == \"__main__\":\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\"--method\", choices=[\"lion\", \"sophia\", \"prodigy\", \"schedule_free\"], required=True)\n    main(parser.parse_args().method)"}</CodeBlock></section>

<Prose>{"Each run saves model state, optimizer state, the next update index and CPU random state after update four, in the evaluation representation. It resumes from those states and checks the eighth evaluation against uninterrupted execution. The call to "}<code>{"optimizer.train()"}</code>{" then restores the training representation before computing a new gradient. Sophia's sampled labels make the random state part of the process. The code uses an in-memory checkpoint to keep the mechanism visible; the final diagnostics lesson supplies actual serialization and more complex data-order/scheduler restoration. The four routes were executed on CPU with PyTorch 2.14.0, Prodigy 1.1.2 and Schedule-Free 1.4.1, using the pinned Lion and Sophia sources above. Every uninterrupted/resumed pair had maximum final-logit difference 0. The final evaluation losses were .8761122277 (Lion), .8761122277 (Sophia), .9517085309 (Prodigy) and .9106840650 (Schedule-Free). These are the tiny bridge fixture’s eight-update outputs, separate from the measured digit study; they establish API/state behavior rather than an optimizer ranking."}</Prose>

<Prose>{"Updates for these dense single-model examples take O(N) optimizer arithmetic per update and method-specific O(N) persistent state, as counted next. Sophia adds a forward/backward curvature pass on refresh updates. Prodigy's adaptation adds reductions, so an equivalent distributed implementation needs globally coherent statistics. This tiny program establishes neither GPU throughput nor distributed equivalence."}</Prose>

<Prose>{""}<strong>{"Take control."}</strong>{" Change the constructed objective to a class-weighted mean cross-entropy. Explain why passing the row count as Sophia's "}<code>{"bs"}</code>{" is no longer automatically a valid estimator correction. Separately remove only Schedule-Free's optimizer mode switch, and locate the first semantic error even if accuracy stays the same."}</Prose>

<details><summary>Hint and reasoned solution</summary>

<Prose>{"The B correction assumed B equally weighted independent sampled-label contributions averaged by B. Class weights alter both contributions and the denominator; reproducing the intended curvature requires deriving the estimator for that weighted objective, not substituting an arbitrary count. Keep the unweighted route as the checked baseline until that derivation is implemented. Without "}<code>{"optimizer.eval()"}</code>{", reported logits come from y rather than x; model dropout mode is a separate switch. A checkpoint saved in the wrong representation may also resume with inconsistent optimizer metadata. Compare logits, parameter buffers and optimizer state, not only rounded accuracy. For an independent implementation exercise, add an omission flag that removes only RNG restoration in the Sophia route; the expected failure is a changed subsequent sampled-label curvature history, not an obligatory accuracy decrease."}</Prose>

</details>

<H2>{"8. Count the resources you actually need"}</H2>

<H3>{"Persistent state is not peak training memory"}</H3>

<Prose>{"An array with N entries, each s bytes, occupies Ns payload bytes. When comparing state counts, first decide whether you are counting additional arrays, the current parameter buffer, master weights, gradients or temporary workspaces."}</Prose>

<NeuralTable caption={"Persistent state is not peak training memory"} headers={[<>{"Method / stated implementation"}</>,<>{"Additional parameter-sized arrays beyond current parameters"}</>]} rows={[[<>{"AdamW without AMSGrad"}</>,<>{"m and v: 2"}</>],[<>{"Lion"}</>,<>{"m: 1"}</>],[<>{"Sophia"}</>,<>{"m and h: 2"}</>],[<>{"Prodigy paper reference here"}</>,<>{"m, v, s and initialization: 4"}</>],[<>{"Schedule-Free teaching code here"}</>,<>{"x, z and v: 3"}</>],[<>{"Efficient Schedule-Free base implementation"}</>,<>{"z and v: 2; the parameter buffer switches between x and y"}</>]]} />

<Prose>{"Scalar counters are excluded from this table. Current library options can change the count: extra inner momentum, sliced adaptation statistics, factored state or a redundant reference copy must be counted explicitly. State dtype is implementation-dependent. Our NumPy experiment uses float64; a count table does not magically make its arrays float32."}</Prose>

<Prose>{"For a hypothetical 70-billion-parameter model with two float32 AdamW moments, those two arrays contain 560 billion bytes, or 560 decimal GB. One float32 Lion moment contains 280 GB. Bfloat16 model weights separately contain 140 GB. These figures exclude gradients, master copies, activations and temporary storage. Dividing a global byte count by device count is justified only for the arrays actually sharded under that scheme."}</Prose>

<OptimizerMemoryLab />

<H3>{"Adafactor offers a different compression"}</H3>

<Prose>{"For an n×m weight matrix, Adafactor can store row and column statistics instead of an nm-entry second-moment array. Using row sums R and column sums C, a rank-one reconstruction has entries Ṽᵢⱼ=RᵢCⱼ/ΣᵢRᵢ. It preserves these marginals, not every individual entry."}</Prose>

<Prose>{"For a 4096×4096 matrix, two float32 factors hold 8192 numbers, requiring 32,768 bytes. A dense second moment holds 16,777,216 numbers, requiring 67,108,864 bytes. This is not a fixed “half a parameter array” saving: it depends strongly on shape. Vector/scalar parameters and optional first moments need separate accounting."}</Prose>

<Prose>{"The "}<a href={"https://proceedings.mlr.press/v80/shazeer18a.html"}>{"Adafactor paper"}</a>{" also discusses update clipping, changing second-moment decay and parameter-relative step sizes. Factoring the state is one part of the method. It does not inherently prove a fixed loss of accuracy or a fixed slowdown."}</Prose>

<H3>{"Steps, work, traffic and elapsed time"}</H3>

<Prose>{"Imagine a hypothetical method needs 600 updates at 1.1 time units per update, while another needs 1000 at 1 unit. Total time is 660 versus 1000, a speed ratio about 1.515. Reporting “40% fewer steps” as “40% faster updates” confuses two measurements. Tuning trials and failed runs also belong in a project-cost comparison."}</Prose>

<Prose>{"Reducing optimizer state does not automatically reduce network traffic by the same factor. Data-parallel gradient reduction communicates gradients; parameter all-gathers communicate parameter shards. Persistent moments can remain local to their owner. Prodigy's global adaptation statistic may itself require a reduction. An actual distributed implementation determines which values move, when and at what precision."}</Prose>

<Prose>{"This connects directly to the previous Ring Attention discussion: bandwidth claims require a communication schedule, buffer sizes and an overlap model. A memory-count heatmap cannot establish them. For real timing, use a representative workload, warmup, synchronization appropriate to the device, repeated measurements, matched quality targets and a record of software/hardware versions."}</Prose>

<H2>{"9. Choose a method by a question you can test"}</H2>

<Prose>{"If optimizer state dominates memory, measure the benefit of a smaller or factored state while checking validation quality. If step-scale tuning dominates repeated experiments, compare adaptation against the total tuning budget of the baseline. If a run's endpoint is uncertain, test an evaluation strategy that remains useful at multiple stopping times. If curvature information seems promising, include its estimation cost and verify that the curvature path is actually used."}</Prose>

<Prose>{"This gives a practical decision table without invented suitability scores:"}</Prose>

<NeuralTable caption={"9. Choose a method by a question you can test"} headers={[<>{"Observation"}</>,<>{"Candidate experiment"}</>,<>{"Evidence to collect"}</>]} rows={[[<>{"Persistent moment arrays dominate"}</>,<>{"Lion or factored-state method"}</>,<>{"Measured peak/steady memory, matching quality, tuning budget"}</>],[<>{"Different coordinates behave very differently"}</>,<>{"Sophia or a suitable preconditioner"}</>,<>{"Curvature/clipping diagnostics, quality versus total work"}</>],[<>{"Many model scales make LR searches expensive"}</>,<>{"Prodigy with a declared implementation"}</>,<>{"Actual trials, d trace, failures and validation"}</>],[<>{"Useful stopping time is uncertain"}</>,<>{"Schedule-Free plus explicit evaluation modes"}</>,<>{"Quality at predefined horizons, x/y handling, normalization statistics"}</>],[<>{"Baseline already trains well"}</>,<>{"Keep it as a reference when testing alternatives"}</>,<>{"Same task, useful paired controls and honest tradeoffs"}</>]]} />

<H3>{"Matrix geometry: a useful current extension"}</H3>

<Prose>{"The four methods above do not exhaust optimizer design. "}<strong>{"Muon"}</strong>{" transforms a matrix-valued momentum direction using an approximate orthogonalization procedure. An idealized polar factor of M=UΣVᵀ is UVᵀ: singular directions are retained while nonzero singular magnitudes are flattened. This is different from applying an entrywise sign."}</Prose>

<Prose>{"For M=[[2,1],[1,2]], all entries are positive, so entrywise sign gives an all-ones rank-one matrix. The ideal polar factor is the identity because M is positive definite. One transformation acts on entries; the other acts on singular geometry. Neither calculation requires a Hessian."}</Prose>

<OptimizerPolarFigure />

<Prose>{""}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html"}>{"PyTorch 2.14 documents Muon"}</a>{", including finite iteration coefficients, matrix-shape requirements and learning-rate adjustment. Non-matrix parameters need a suitable companion update, and an embedding's two-dimensional shape alone does not establish that every recipe treats it like a hidden matrix. The exact recipe and parameter grouping matter. Muon is a useful connection to the matrix-preconditioning lesson, not evidence that every advanced optimizer lives outside native PyTorch."}</Prose>

<H3>{"Diagnosing a changed training run"}</H3>

<Prose>{"Start with the actual objective and parameter changes. If loss jumps after switching optimizers, record the gradient norm, update norm, parameter norm, rate, decay and active state. Check whether the batch or reduction changed. Reusing a familiar rate may be a poor choice, but a failure at one particular update does not identify its cause by itself."}</Prose>

<Prose>{"For Sophia, inspect curvature refreshes and scaling before concluding that “second-order information failed.” For Prodigy, inspect the estimator and d trace before treating a slow start as convergence. For Schedule-Free, confirm which iterate produced the reported metric. For Lion, inspect the blended direction and ηλ product. These are mechanisms that can be checked, not diagnoses based only on an optimizer's name."}</Prose>

<Prose>{"The next "}<a href={"/learn/path/full-curriculum/neural-ode-continuous-depth-models?module=deep-learning-fundamentals"}>{"Neural ODE lesson"}</a>{" considers continuous state evolution inside a model. An optimizer also generates a trajectory, but its trajectory lives in parameter space while training. The distinction between evolving model state and updating model parameters remains essential."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"10. Practice and transfer"}</H2>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. A different AdamW first step"}</H3>

<Prose>{"Use θ=[1,−2], g=[−3,6], η=.02, λ=.5, β₁=.9 and β₂=.999. Ignore ε only for this hand calculation. Find the corrected moments and next parameters. Would adding λθ to g first give the same rule?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The first corrected moments recover g and g². Apply shrinkage independently of the normalized direction."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"m̂=[−3,6], v̂=[9,36], so the direction is [−1,1]. Shrink to [.99,−1.98], then obtain [1.01,−2]. Adding λθ to g would contaminate the moment calculation with the regularizer; it is not decoupled AdamW even if a particular first-step sign happens to agree."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. Lion follows history"}</H3>

<Prose>{"Use θ=−.4, old m=−.3, g=2, β₁=.9, β₂=.99, η=.02 and λ=.1. Find the update and new momentum. What current g would make the blend zero in exact arithmetic?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Solve −.27+.1g=0 separately from the β₂ memory update."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The blend is −.07, so θ_next=−.4(.998)+.02=−.3792. New m=−.297+.02=−.277. The zero threshold is g=2.7. This threshold depends on the magnitude of the old memory; knowing only its sign is insufficient. Finite-precision implementations may place a mathematically exact decimal tie just to one side, which is why the worked zero-state null uses exact zero inputs."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Curvature is not the real-label gradient squared"}</H3>

<Prose>{"For a scalar logistic weight, x=3 and p=.25. Calculate the gradient squared for real label 1 and the expected squared gradient under a model-sampled label. Explain why the results differ."}</Prose>

<details><summary>Hint</summary>

<Prose>{"The gradient is x(p−y). Enumerate y=0 and y=1 with their model probabilities."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"For label 1 the gradient is −2.25 and its square is 5.0625. The sampled expectation is .25(5.0625)+.75(.5625)=1.6875, equal to 9(.25)(.75). The first quantity reflects one observed residual; the second averages over the model's possible labels and isolates the logistic curvature."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. A curvature probe can disagree with the diagonal"}</H3>

<Prose>{"For H=[[4,−2],[−2,1]], calculate u⊙Hu for u=[1,1] and u=[1,−1]. What is their average? Does the negative sample prove H has a negative eigenvalue?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Multiply by H before multiplying elementwise by u. The two remaining sign probes duplicate these results."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The estimates are [2,−1] and [6,3]; their mean is [4,1]. H has eigenvalues 5 and 0, so it is positive semidefinite despite that negative sampled entry. An estimator's individual signs do not determine the matrix's eigenvalues."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Find the wrong Prodigy implementation"}</H3>

<Prose>{"An implementation computes ordinary Adam moments, estimates d from only the current displacement dot product, caps d at .003, and multiplies the ordinary Adam update by d. Is this paper Algorithm 4? Identify three repairs needed before presenting a comparison under that name."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Look at the histories, where d enters and which quantities persist across updates."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"No. The moments must accumulate dg and d²g²; numerator and vector denominator statistics have their own weighted histories; the stated paper rule takes a nondecreasing maximum without that arbitrary cap. The update's d/ε convention and old-versus-next state also need matching. A useful alternative algorithm can be studied, but it must be named and evaluated as that alternative."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. Compute the model that will be evaluated"}</H3>

<Prose>{"Run the simple Schedule-Free SGD rule with x=z=−1, target 1, η=.2, β=.9 and perturbations [−.5,.5,1,−1]. Find x, z and the next training point after update 2; then find the gradient at update 3. How do the final averages differ for β=0 and β=1?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The first update gives x=z=−.5. Remember that the gradient is evaluated at an interpolation, not necessarily at z."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"After update 2, z=−.3 and x=−.4. The next y is −.39 and the next gradient is −.39−1+1=−.39. After all four updates, x≈−.19456 for β=.9, −.208 for β=0 and −.193 for β=1. This tiny noise sequence does not make the commonly chosen .9 universally optimal."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. Which resource saving did you measure?"}</H3>

<Prose>{"A matrix has shape 2048×1024. Compare a float32 dense second moment with two float32 row/column factors. Separately, a model has eight billion parameters and two float32 moments sharded evenly over eight ranks. How many decimal GB of moments belong to each rank? Does replacing two moments with one halve a gradient all-reduce?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Use nm entries for the dense matrix and n+m for the factors. State payload and communicated payload are different objects."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The dense moment is 8,388,608 bytes; factors require 12,288 bytes. For the model, global moments occupy 64 GB and each ideal equal shard occupies 8 GB. One float32 moment would give 4 GB per rank under the same sharding. The gradient array is unchanged, so its all-reduce is not automatically halved. Account for the actual algorithm's collectives and other allocations."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. Accuracy and loss disagree"}</H3>

<Prose>{"Classifier A assigns correct-label probabilities [.51,.99,.99,.99]; B assigns [.49,.999,.999,.999] on four binary examples. Which has more correct decisions, and which has lower mean cross-entropy? Why must optimizer selection state its objective?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Use a .5 decision threshold and average the four negative logarithms."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"A gets 4/4 correct, B gets 3/4. A's loss is about .17587; B's is about .17909, so A wins both for these inputs. Now replace A's three .99 probabilities by .90: its decisions remain 4/4 but its loss rises to about .24736, and B has the lower loss. The changed case shows the tradeoff; do not assume the first probability list must demonstrate it. Accuracy counts boundary decisions, while cross-entropy measures allocated probability throughout the set."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"9. Design a comparison that can answer your question"}</H3>

<Prose>{"You can afford twelve small training runs and care about useful quality at both 200 and 400 updates. Compare a baseline with two alternatives. Specify data roles, tuning allocation, seeds, evaluation iterates, additional curvature work, reported metrics and a rule for failures. What would you refuse to infer from the result?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Three methods × two declared scales × two seeds use all twelve runs. Intermediate evaluation does not require a new fit."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"One defensible design fixes three disjoint data roles, two candidate scales and two seeds per method; uses common batches at each seed; reports both horizons; and selects scales by a prespecified validation criterion. It reports the defined evaluation iterate, actual extra gradient work and any failed run rather than replacing it after assessment. Assessment is reserved for selected settings. Twelve small runs cannot establish universal superiority, broad hardware speedups or robust performance on an unrelated large model. Different learning-rate grids or a different quality target may answer a different question."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"References and other ways to learn"}</H2>

<Prose>{"These links support different parts of the lesson. The papers and executable local calculations carry the technical claims; external learning resources offer another presentation."}</Prose>

<ul><li>{""}<a href={"https://arxiv.org/abs/1711.05101"}>{"Loshchilov and Hutter: Decoupled Weight Decay Regularization"}</a>{", plus "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html"}>{"the versioned PyTorch AdamW documentation"}</a>{". Read for the regularization distinction and an exact implementation contract."}</li><li>{""}<a href={"https://arxiv.org/html/2302.06675v4"}>{"Chen et al.: Symbolic Discovery of Optimization Algorithms"}</a>{", especially the algorithm, tuning and limitations, with "}<a href={"https://github.com/google/automl/blob/master/lion/lion_pytorch.py"}>{"Google's Lion source"}</a>{". Useful after the one-coordinate trace; the paper also explains the search process and evaluated tasks."}</li><li>{""}<a href={"https://arxiv.org/html/2305.14342v4"}>{"Liu et al.: Sophia"}</a>{", method and estimator sections, with "}<a href={"https://github.com/Liuhong99/Sophia"}>{"the official implementation"}</a>{". Use the paper to check the sampled-label normalization; do not treat a package's batch default as a universal value."}</li><li>{""}<a href={"https://arxiv.org/html/2306.06101v3"}>{"Mishchenko and Defazio: Prodigy"}</a>{", especially Algorithm 4 and its distinction from the proved convex variants; "}<a href={"https://github.com/konstmish/prodigy"}>{"official package guidance"}</a>{" explains practical options and schedules. The executable classroom rule explicitly identifies its paper version."}</li><li>{""}<a href={"https://arxiv.org/html/2405.15682v2"}>{"Defazio et al.: The Road Less Scheduled"}</a>{", method, large-rate discussion and implementation concerns; "}<a href={"https://github.com/facebookresearch/schedule_free"}>{"official Schedule-Free repository"}</a>{" for current modes and options. The paper's theorem conditions deserve the same attention as its curves."}</li><li>{""}<a href={"https://slideslive.com/39024867/the-road-less-scheduled"}>{"The Road Less Scheduled, NeurIPS 2024 author presentation"}</a>{". An alternate video route after the x/y/z diagram. The recording page, title and conference attribution were checked; the recording was not watched or transcribed for this manuscript."}</li><li>{""}<a href={"https://en.d2l.ai/chapter_optimization/adam.html"}>{"Dive into Deep Learning: Adam"}</a>{". A ground-up article with algebra and code for the prerequisite averages. Its notation differs, and some surrounding “variance” wording is loose; distinguish the raw second moment as taught here. The article was read; its remote notebooks were not executed."}</li><li>{""}<a href={"https://proceedings.mlr.press/v80/shazeer18a.html"}>{"Shazeer and Stern: Adafactor"}</a>{". Follow the factored-state mechanism beyond a memory slogan. Optional momentum and tensor shapes change storage."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html"}>{"PyTorch 2.14 Muon"}</a>{". A current matrix-update branch; inspect parameter groups, finite iterations and scaling before adapting a recipe. This lesson did not benchmark Muon."}</li><li>{""}<a href={"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"}>{"UCI Optical Recognition of Handwritten Digits"}</a>{", E. Alpaydin and C. Kaynak, "}<a href={"https://doi.org/10.24432/C50P49"}>{"dataset DOI"}</a>{", CC BY 4.0. The retained extract and classroom transformations are described in "}<a href={"/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/data-provenance.md"}>{"data provenance"}</a>{"."}</li></ul></section>
</div> };
