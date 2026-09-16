# Advanced Optimizers: Lion, Sophia, Prodigy and Schedule-Free

Two people can receive the same directions and make different journeys. One takes a fixed stride, another slows down on a steep slope, and another changes stride after seeing how far they have travelled. A neural-network optimizer faces a related problem: the gradient tells it how the current loss changes locally, but does not specify a safe, useful next step.

An **optimizer** turns gradients and stored history into changes to model parameters. Those changes affect the next prediction. An optimizer does not add attention heads, change the labels or discover information that the input lacks. Its job is to make better use of the learning signal that the model and objective provide.

The preceding [Ring Attention lesson](/learn/path/full-curriculum/ring-attention-sequence-parallelism?module=deep-learning-fundamentals) reorganized a computation across devices while preserving its mathematical result. Changing the optimizer deliberately changes the learning trajectory. A faster update kernel and fewer updates to reach useful quality are separate advantages.

**First pass:** read the update anatomy, follow one calculation for each method, then inspect the handwriting experiment and memory accounting. Try the core practice before opening the optional theory and matrix-method branches. You need derivatives, weighted averages and basic probability; each is refreshed where it enters. The [second-order methods lesson](/learn/path/full-curriculum/second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient?module=math-foundations) supplies a deeper route through curvature.

## 1. What has to happen between a gradient and the next prediction?

Suppose a classifier assigns only 0.6 probability to the correct handwritten digit. Its cross-entropy loss is −log(0.6). Backpropagation calculates how changing each weight would change that loss. A positive gradient component says that a small increase in that parameter raises this batch's loss; descent would move in the negative direction.

That statement is local. If a slope changes rapidly, a long step can go past the useful region. A gradient from a small batch also contains sampling noise. Finally, millions of parameters can have very different scales. These are reasons to transform the gradient, rather than reasons to stop trusting calculus.

We will use θ for the parameter vector, g for the current gradient, η for a learning rate and λ for a weight-decay coefficient. A subscript t identifies an update, not an example or an epoch. Multiplication, square roots and division between equally shaped parameter arrays are elementwise unless a dot product or matrix product is written explicitly.

**Figure 1 — anatomy of an update.** Trace a real image through scores → probability → loss → gradient, then split the optimizer into history, direction, scale and parameter update. Loop the new parameters back to a new prediction. Keep the image label outside the optimizer's state boxes.

| Method | Central question | Information it retains |
| --- | --- | --- |
| AdamW | How large is the recent signed gradient relative to its recent squared magnitude? | First and second gradient moments |
| Lion | Which direction does a blend of history and this gradient support? | One momentum buffer |
| Sophia | How sharply does the loss change along each parameter coordinate? | Momentum and an estimated curvature diagonal |
| Prodigy | Can observed progress supply a useful unknown step-scale estimate? | Scaled moments, displacement statistics and initialization |
| Schedule-Free | At which parameters should we take gradients, and which parameters should we evaluate? | A fast trajectory, an averaged trajectory and any base-method scaling state |

These questions overlap. “Adaptive,” “second-order,” “parameter-free” and “schedule-free” name different properties. None means that data selection, validation or implementation details cease to matter.

## 2. AdamW gives us a concrete reference

### Smooth direction and scale separately

An exponential moving average keeps part of the old value and adds part of the new one:

mₜ = β₁mₜ₋₁ + (1−β₁)gₜ,

vₜ = β₂vₜ₋₁ + (1−β₂)gₜ².

The first buffer keeps signed information. Opposite gradients can cancel. The second stores squared magnitude, so opposite signs cannot cancel. It is a **raw second moment**, not a variance: variance would subtract the square of the mean. With β₂ different from β₁, even that subtraction needs care about the weighting scheme.

If both buffers start at zero, their early values include too much of that initial zero. For a constant gradient g, the first recurrence produces mₜ=(1−β₁ᵗ)g. Dividing by 1−β₁ᵗ corrects this initialization effect. Thus

m̂ₜ = mₜ/(1−β₁ᵗ), v̂ₜ = vₜ/(1−β₂ᵗ),

θₜ₊₁ = (1−ηₜλ)θₜ − ηₜ m̂ₜ/(√v̂ₜ+ε).

The small positive ε stabilizes division. Its placement outside the square root is part of this algorithm, not cosmetic notation. The [AdamW paper](https://arxiv.org/abs/1711.05101) motivates separating shrinkage from adaptive scaling; the [PyTorch 2.14 rule](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) makes the implemented order explicit.

At the first step, take g=[2,−4], β₁=.9 and β₂=.999. Then m=[.2,−.4], v=[.004,.016], m̂=[2,−4], v̂=[4,16]. Ignoring only the tiny ε for mental arithmetic, the normalized direction is [1,−1]. The larger raw component does not produce a twice-as-large first step.

With θ=[.5,−.7], η=.03 and λ=.1, shrinkage first gives [.4985,−.6979], followed by approximately [.4685,−.6679]. Our complete float64 implementation was compared with native PyTorch AdamW for four unequal gradients, including a zero gradient; all four maximum parameter differences were zero in that run. This verifies that particular calculation, not every optimizer in this lesson.

**Figure 2 — two histories, one step.** Show signed-gradient bars and squared-gradient bars feeding their own averages. Expand the first step into raw and corrected values. A later impulse after 999 zero gradients produces |m̂|/√v̂≈2.51457 with these betas, before ε. Therefore the Adam direction is not universally bounded by one.

### Weight decay is its own change

Adding λθ to the gradient is the derivative of an L2 penalty. In an adaptive optimizer, that addition also enters the moment buffers and is rescaled with the gradient. Decoupled weight decay instead directly multiplies the parameter by 1−ηλ. These operations are generally different.

With no gradient contribution and constant ηλ=.01, ten shrinkage steps multiply a parameter by .99¹⁰. The product over steps, ∏ₜ(1−ηₜλ), explains why changing the learning-rate schedule changes the cumulative shrinkage even when λ is unchanged. A zero current gradient does not generally stop an optimizer with nonzero momentum.

## 3. Lion: keep a directional memory, discard the final magnitude

Imagine receiving several “move left” instructions followed by one “move right.” It matters whether the new instruction is weak or strong relative to the stored history. Lion first blends those quantities, then keeps only the sign of that blend for this update:

uₜ = sign(β₁mₜ₋₁ + (1−β₁)gₜ),

θₜ₊₁ = (1−ηₜλ)θₜ − ηₜuₜ,

mₜ = β₂mₜ₋₁ + (1−β₂)gₜ.

The direction uses the **old** momentum and β₁. The stored momentum then uses β₂. Replacing both with one average defines a different method. Lion's original defaults are β₁=.9 and β₂=.99. There is no second-moment denominator and no Adam-style bias correction in this rule. Its [official Google implementation](https://github.com/google/automl/blob/master/lion/lion_pytorch.py) is small enough to follow alongside the equations.

For θ=.5, old m=.2, g=−1, η=.01 and λ=.2:

1. Blend: .9(.2)+.1(−1)=.08.
2. Direction: sign(.08)=+1, despite the current negative gradient.
3. Parameter: .5(.998)−.01=.489.
4. New memory: .99(.2)+.01(−1)=.188.

The old history wins this step. If the current gradient were negative enough to reverse the blend, the step would reverse. Gradient magnitudes are therefore still important **before** the sign and when storing history. The buffer cannot be replaced by one sign bit without changing the algorithm. A zero blend has sign zero; the gradient-driven update can be zero. Weight decay can make the total change larger or smaller than η.

**Figure 3 — a balance followed by a direction switch.** Put .18 and −.10 on a signed number line; their sum crosses a sign threshold at zero. Below it, show the separate .198 and −.01 memory calculation. Do not draw an arbitrary confidence meter.

**Investigation: when does the new evidence win?** Begin with a different weight and history, edit the actual gradient sequence, and predict the next direction before stepping. Compare two sequences with the same signs but different magnitudes. Then set both gradient and history to zero and observe the no-decay null. The explanation should identify which blend crossed zero.

### Tuning and an interesting origin

The [Lion paper's tuning section](https://arxiv.org/html/2302.06675v4) recommends trying a learning rate roughly three to ten times smaller than its AdamW comparator, with a correspondingly larger λ to retain a similar ηλ product. That is an empirical starting range. It is not an equivalence theorem, a requirement to multiply every decay by 100, or a diagnosis of any later training failure.

The striking origin of Lion is **program search**. Candidate optimizer programs were mutated, screened on inexpensive tasks, selected on harder tasks and simplified. The two different blend coefficients survived that process. This is an application of automated discovery where the result is an inspectable algorithm. A proxy task can still favor the wrong behavior, so transfer to larger tasks and ablations remain essential. The resulting method is not a neural network secretly deciding updates at runtime.

Lion can reduce persistent optimizer storage when that storage is the limiting resource. It does not halve activations, model weights or every communication collective. Its sign threshold can also make tiny numerical changes consequential near zero; precision, loss scaling and accumulation should be assessed in the actual training setup rather than reduced to “one floating-point format always works.”

## 4. Sophia: distinguish a large gradient from high curvature

### The shape of the slope

For the one-dimensional bowl L(θ)=½h(θ−a)², the gradient is h(θ−a), and the second derivative is h. A large gradient may mean that we are far from a, that the bowl is sharp, or both. Dividing by h returns the displacement θ−a. With an exact quadratic and step multiplier one, Newton's update reaches a immediately.

For many parameters, the **Hessian** H contains second derivatives. Off-diagonal entries describe coupling: changing one coordinate changes the slope in another. Storing a full matrix costs quadratic space in parameter count. Sophia estimates only the diagonal, smooths it, divides a momentum estimate by it and clips the resulting coordinate updates.

**Figure 4 — bowl geometry.** Show both an axis-aligned bowl and a rotated bowl with principal curvatures 1 and 20. From the same point, compare a gradient step, a diagonal-curvature step and the full Newton solution. On the rotated bowl, retaining only the diagonal misses coupling. This is a calculated geometry example, not a model benchmark.

Write ρ for the positive curvature scale in our implementation:

mₜ = β₁mₜ₋₁ + (1−β₁)gₜ,

hₜ = β₂h_previous + (1−β₂)ĥₜ on refresh steps; otherwise keep h unchanged,

θₜ₊₁ = (1−ηₜλ)θₜ − ηₜ clip(mₜ/max(ρhₜ,ε),−1,1).

Max and clipping are coordinatewise. The estimated diagonal is refreshed less frequently than the gradient. A negative or near-zero estimate receives the ε floor, and clipping prevents it from producing an arbitrarily large gradient-driven step. This bounds each such coordinate change by η; it does not certify that the whole loss decreases.

For m=[.2,−.3,0], h=[5,.1,0] and ρ=.5, the ratios are [.08,−6,0] and the clipped direction is [.08,−1,0]. At θ=[1,2,3], η=.1 and λ=.2, the new parameters are [.972,2.06,2.94]. Only the second gradient-driven component is clipped. Decay still moves the third component.

### Two different curvature estimators

**Sophia-H uses Hessian-vector products.** Draw a random vector u with E[uuᵀ]=I and calculate u⊙(Hu). Its expected ith entry is Hᵢᵢ, because the cross terms have zero expectation. Gaussian probes are used in the paper; independent ±1 probes also satisfy this identity.

For H=[[2,3],[3,1]], the probe [1,1] returns [5,4], while [1,−1] returns [−1,−2]. Averaging the four equally likely sign probes gives exactly [2,1]. An unbiased estimator can have negative individual entries. One sample is not a proof of negative true diagonal curvature. Automatic differentiation can form Hu without constructing H explicitly.

**Sophia-G uses model-sampled labels.** For a classification model, form its current probabilities, sample an independent label for each example, and differentiate cross-entropy using those sampled labels. Square that mean gradient and multiply by batch size B:

ĥ = B ĝ², ĝ = ∇θ mean_b CE(logits_b, sampled_label_b).

The labels used to train the model remain the real labels. Sampled labels are a separate instrument for estimating curvature. Squaring the ordinary real-label gradient is not this estimator.

Here is an exact binary example. Let the input be x=2, predicted positive-class probability p=.8, and the real label be 1. The real-label gradient is x(p−1)=−.4; its square is .16. The curvature of the logistic loss in its scalar weight is x²p(1−p)=.64. If we sample label 1 with probability .8 and label 0 with probability .2, the expected squared gradient is

.8(−.4)² + .2(1.6)² = .64.

This exposes the key distinction without a giant neural model. For a two-example batch x=[2,1], p=[.8,.3], enumerating all four sampled-label pairs gives E[2ĝ²]=.425, the average of the two exact diagonals. Omitting B gives half that value.

**Figure 5 — two gradient lanes.** Real labels feed the training-gradient lane; independently sampled labels feed the curvature lane. Show the binary branches, their probabilities, squared gradients and weighted sum. A batch panel explains that squaring an average is not averaging squares.

**Investigation: what does the label sampler estimate?** Edit input magnitudes and probabilities in a new two-example problem. Predict whether a real-label squared gradient equals the expected sampled-label result. Enumerate the small outcome space, then compare with the analytic curvature. A separate probe view lets you edit a symmetric two-by-two matrix and see positive and negative Hutchinson estimates. These are different estimators, not interchangeable presets.

<details>
<summary>Deeper: what curvature does Sophia-G actually estimate?</summary>

Let J have shape classes × parameters, the Jacobian of logits with respect to parameters, and p be the probability vector. The generalized Gauss–Newton matrix for softmax cross-entropy is

G = Jᵀ[diag(p)−ppᵀ]J.

The full Hessian additionally includes second derivatives of the logits weighted by the loss's logit derivatives. For a linear classifier those second derivatives vanish, so G equals the Hessian. For a nonlinear network they generally do not vanish. Sophia-G is unbiased for the G diagonal under the sampled-label construction; it is not generally unbiased for the full Hessian diagonal.

The batch factor follows because independent sampled-label gradients have mean zero. When we square their sum, expected cross-example terms vanish. With unequal example weights, correlated sampling, masks or distributed averaging, derive the normalization for the actual reduction; do not copy an unrelated `bs` default. A formula involving only probabilities and squared logits misses the parameter Jacobian and cannot be the general parameter-curvature formula.

</details>

### Implement the refresh at a coherent parameter state

The paper uses a batch-scaled estimate and a maximum with ε. The official Sophia-G source instead stores the unscaled squared sampled gradient and multiplies by `bs` inside its denominator, adding a small ε. Those conventions must be matched deliberately. Using both B-scaled storage and `bs=B` would apply the factor twice.

For a neural training loop, compute a fresh forward/backward for the sampled-label estimate at the intended parameter state, clear those gradients, then compute the real-label gradient for the update. Do not reuse a freed graph or reuse pre-update logits after mutating weights. The complete classroom program avoids this ambiguity by explicitly evaluating both gradients at the same parameter array.

The [Sophia paper](https://arxiv.org/html/2305.14342v4) reports improvements on its specified language-model tasks and budgets. Its result is a reason to test a faithful implementation, not a guaranteed twofold speedup on another model. Measure refresh cost, clipped fraction, validation quality and actual elapsed training time. A clipped momentum method is not automatically Lion: their histories and update rules differ.

## 5. Prodigy: use progress to estimate an unknown scale

A learning rate that is sensible for one parameterization can be poor for another. In convex optimization, useful step-size bounds often contain the unknown distance D from initialization to a solution. **D-adaptation** methods try to estimate a useful distance scale while learning. Prodigy changes that adaptation so the estimate can grow more effectively.

The basic signal compares the current gradient with displacement from initialization. Suppose we have moved in a direction that earlier gradients supported, and the current gradient still says there is useful progress in that direction. Their agreement supplies information about the scale of the problem. Cancellation and reversal supply different information. This is not an exact oracle for the location of a neural-network optimum.

### Follow a fully specified Adam-style version

We use Algorithm 4 of the [Prodigy paper](https://arxiv.org/html/2306.06101v3), without weight decay or optional bias correction. Let d start at a positive d₀; m, v, s and scalar r start at zero. Let γ be a user multiplier, and b=√β₂. At a step with current θ, g and d:

m_new = β₁m + (1−β₁)d g,

v_new = β₂v + (1−β₂)d²g²,

r_new = b r + (1−b)γd²〈g, θ_initial−θ〉,

s_new = b s + (1−b)γd²g,

d_estimate = r_new / ||s_new||₁,

d_next = max(d, d_estimate),

θ_next = θ − γd m_new/(√v_new+dε).

The norm ||s||₁ adds absolute coordinate values. The numerator is a scalar sum across parameters, not a per-coordinate learning rate. If its denominator is zero, keep the previous d rather than divide by zero. The parameter update shown here uses the **current d**, while d_next is for the next step. The momentum buffers themselves include d and d²; inserting d into an otherwise unchanged Adam update is not Algorithm 4.

**Figure 6 — progress accounting.** Draw initialization and current position on the parameter line. Attach the current gradient, displacement dot product, weighted cumulative numerator and denominator. Keep the old d beside this step's update and the new estimate beside the next step's state.

On the declared bowl ½(θ−3)², start θ=0 and d₀=.01, using γ=1, β₁=.9 and β₂=.999. The first parameters are .0316228, .0741064 and .1514404. The distance used for those steps is .01, .01 and .0157316. At step 7 the parameter overshoots to 7.11008; d never decreases in this rule. The trace is useful precisely because automatic scale growth does not mean monotonically improving loss.

**Investigation: adaptation without an oracle.** Change the target, starting point and initial estimate before a fresh run. Predict when d first increases and whether the first crossing of the target settles the trajectory. Show the entire calculated trace, including overshoots. A stationary starting point with zero gradient and zero history is a null: neither the parameter nor d changes.

### What “parameter-free” leaves for the practitioner

Prodigy aims to remove the need to supply a well-tuned problem-scale learning rate. It still has initialization, betas, ε, a multiplier, regularization and variant choices. The theoretical guarantees apply to the stated convex algorithms and assumptions; they are not an assertion that the Adam-style version solves every nonconvex training problem without tuning.

The [official package](https://github.com/konstmish/prodigy) recommends starting with `lr=1`, offers `d_coef`, optional bias correction and sliced adaptation statistics, and permits constant or cosine schedules. Its warmup safeguard changes the estimator normalization; it does not merely wait a fixed number of steps before allowing d to grow. Current library arithmetic also differs from the compact paper presentation in numerical rescaling and update details. Record which implementation you use before comparing trajectories.

A practical application is training many small models with different scales, where a full learning-rate search for every model is expensive. Architecture search and multiple objectives provide such situations. A fair study still includes the tuning compute that was actually saved, failures that required reruns and validation quality. “No sweep needed” cannot be asserted from a single favorable run.

## 6. Schedule-Free: train and evaluate at different points

Learning-rate schedules often assume that you know the final update count. If you plan 100,000 steps and later extend the run, the decay may already have changed your trajectory substantially. Schedule-Free offers another strategy: maintain a fast-changing sequence and an average, and evaluate the loss gradient between them.

Call the fast parameters z, the averaged parameters x, and the training parameters y. For the simplest SGD form, use

yₜ = βxₜ + (1−β)zₜ,

zₜ₊₁ = zₜ − η∇L(yₜ),

xₜ₊₁ = (1−1/t)xₜ + (1/t)zₜ₊₁ for our one-based update index t.

At t=1 the average becomes the first updated fast point. This indexing convention averages the post-update z values. Gradients are calculated at y; validation and inference use x. Neither “always use z” nor “average predictions” describes this method.

Take ½θ², start x=z=2, β=.9 and η=.2. Add the declared illustrative gradient perturbations [1,−1,.5,−.5]. These are arithmetic inputs, not measured training noise.

| Update | Training point y | Perturbed gradient | New fast point z | New average x |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2 | 3 | 1.4 | 1.4 |
| 2 | 1.4 | .4 | 1.32 | 1.36 |
| 3 | 1.356 | 1.856 | .9488 | 1.222933 |
| 4 | 1.19552 | .69552 | .809696 | 1.119624 |

The third gradient is evaluated at 1.356, neither 1.32 nor 1.36. That small distinction changes every later step. At the end, report the loss at x, not whichever point happens to look best.

**Figure 7 — three named points, two kinds of arrow.** On a shared number line, connect x and z by the interpolation that forms y. A gradient arrow starts at y but updates z; the averaging arrow updates x. Show the selected row of the table beside the geometry.

### Averaging does not mean exponential decay

With equal weights, x after t updates is the average of those t post-update z values. Each has final coefficient 1/t. The newest value's insertion coefficient is 1/t, but older values are subsequently diluted too. It is incorrect to compare their insertion coefficients as though those were their final weights.

This is not the same as a momentum EMA, nor the same as applying a 1/t learning rate directly to the current gradient. Earlier gradients influence many later z values. The equations specify that influence; there is no universal “effective schedule” curve that makes every method identical.

**Investigation: which model are you measuring?** Use a new target and editable perturbation sequence. Predict the training point, fast point and evaluated point after an update. Compare β=0, β=.9 and β=1 without assuming the middle choice wins on every small problem. At the exact target with zero noise, all three remain there.

### The AdamW form and the warmup weights

Schedule-Free AdamW replaces the SGD step with a second-moment-scaled gradient. In the base form used here there is no ordinary first-moment EMA: interpolation supplies the momentum-like behavior. Update v with g², bias-correct v, and use g/(√v̂+ε) to move z. Decay can be calculated at y, as in the paper's algorithm.

Warmup still changes η during the opening updates. With the paper's weighting, let wₜ=ηₜ² and cₜ=wₜ/Σᵢ≤ₜwᵢ, then average with cₜ. If the first learning rates are .1,.2,.3, the three post-update z values have normalized weights [1,4,9]/14 after step 3. A plain one-third average is a different calculation. Our monotone warmup then constant rate matches this weighting; the current library's maximum-rate weighting also agrees for this schedule.

**Figure 8 — rates and averaging coefficients are different quantities.** Place the warmup/constant rate above a separate coefficient strip. Label rate and dimensionless mixture weight explicitly. Compare against the actual cosine formula only in a separately labeled schedule panel.

### The training/evaluation switch changes parameters

The efficient [Schedule-Free library](https://github.com/facebookresearch/schedule_free) stores only the sequences it needs and reconstructs the other. Calling `optimizer.eval()` changes the parameter buffer to x; `optimizer.train()` restores y. These calls are separate from `model.eval()` and `model.train()`, which affect such layers as dropout and BatchNorm.

For BatchNorm models, statistics collected at y may not match x. Recompute appropriate running statistics using training inputs at the evaluation weights, following the method's guidance. Do not use validation labels or validation examples to fit those statistics. Save checkpoints with the optimizer in its documented mode and retain optimizer state if training will resume.

Validating at y is a different measurement, but it is not guaranteed to be worse on every batch. Save both values in a diagnostic; compare models at the evaluation point defined by the method. A framework's automatic scheduler or missing optimizer-mode hook can silently change the intended recipe.

<details>
<summary>Deeper: a theorem is not a universal anytime guarantee</summary>

The Schedule-Free paper's introductory SGD bound assumes convex, Lipschitz stochastic losses with independent samples. Its displayed choice η=D/(G√T) contains the horizon T even though the practical update can run with a constant chosen rate. The broader online-to-batch result and the discussion of larger rates explain more of the connection. It would be incorrect to cite the introductory bound as proof that any constant rate converges optimally at every stopping time on a nonconvex network.

The useful practical distinction is that an evaluation iterate exists at each update without prescribing a decay endpoint. Constant-rate SGD, alternative schedules and restart strategies also exist; Schedule-Free is not the only imaginable way to extend a run. If the data distribution changes, averaging across the full old history may be inappropriate. That is a new learning problem, not a consequence that the original stationary analysis settles.

</details>

## 7. Train the same real classifier with different update rules

### Fix the learning problem before comparing the optimizers

Our practical task is to recognize ten handwritten digits from 64 pixel features. We use 400 actual images, forty per digit, from the UCI Optical Digits data distributed through scikit-learn. These are 8×8 block-count images, not MNIST. Pixel values range from 0 to 16; divide by 16 and append a constant 1 for a bias feature.

The model is a linear softmax classifier. Its parameter array Θ has 65 rows and 10 columns, giving 650 parameters. For a batch X, the score matrix is XΘ; a row-wise softmax produces ten probabilities per image. The loss is the mean negative log probability of the correct label. Inference chooses the largest score. This small model makes the optimizer's behavior inspectable without confusing it with architecture changes.

The gradient is

∇ΘL = Xᵀ(P−Y)/B,

where P contains predicted probabilities and Y has a one in each example's correct-label column. A pixel's gradient contribution is its intensity multiplied by a class residual. The bias row uses intensity one. For a nonlinear model, backpropagation supplies the corresponding parameter gradients; the optimizer then consumes arrays of those gradients.

**Figure 9 — one pixel contributes to ten updates.** Select a real image pixel, follow its normalized intensity through ten weighted scores, then show its ten residual-weighted gradient contributions. Retain row/column labels and the bias row. This connects the abstract update rules to an actual prediction.

Use a fixed stratified split: 240 fitting images, 80 validation images and 80 assessment images. Source identities and full pixel signatures are distinct across these roles. The subset comes from the historical UCI test file, so this new classroom partition is not the official UCI train/test evaluation. Writer identities are unavailable here, and the data have appeared in earlier lessons. We use them to study mechanisms, not claim a new untouched benchmark.

We declare the experiment before fitting:

- Two seeds, 11 and 29; the same initial parameter array and minibatch sequence for every method at a given seed.
- Exactly 400 updates per candidate, batches of 64 fitting images sampled with replacement; no early stopping, augmentation or weight decay.
- Two candidate scales per method: AdamW constant/cosine and Schedule-Free [.01,.03]; Lion [.003,.01]; Sophia-G [.01,.03]; paper-version Prodigy multipliers [.3,1]. Other method settings are those specified above.
- Twenty-step warmup for cosine AdamW and Schedule-Free. Cosine decays to zero at update 400. Constant AdamW and the other methods use their stated constant scales.
- Sophia-G refreshes at updates 1,11,…,391: forty additional sampled-label gradient evaluations per fit. They are computed at the same parameters as the real gradient. No full Hessian is formed.
- Select one scale per method by **mean final validation cross-entropy across the two seeds**, then assess both selected runs. All 24 candidate curves are retained; no candidate or seed is erased because it is less flattering.

The two-value grids are deliberately small and method-specific. Equal numbers of candidates do not prove equally good tuning. Equal updates do not imply equal work: Sophia has extra gradient evaluations, and schedules deliberately change the update scales. There are no CPU or GPU speed measurements in this comparison.

**Figure 10 — selection has a boundary.** Fitting arrows reach weights; validation arrows select a declared scale; assessment arrows produce the report. Beside it, show both candidate values and both seeds. Assessment has no arrow back into the selection rule.

### Actual outcomes

| Selected method | Scale | Assessment cross-entropy, seeds 11 / 29 | Correct out of 80, seeds 11 / 29 |
| --- | ---: | ---: | ---: |
| AdamW, constant | .03 | .06075 / .05581 | 80 / 80 |
| AdamW, warmup + cosine | .03 | .09489 / .09190 | 79 / 80 |
| Lion | .003 | .09008 / .08580 | 78 / 78 |
| Sophia-G, paper-scaled estimator | .01 | .02786 / .02523 | 79 / 79 |
| Prodigy, paper Algorithm 4 | 1 | .03591 / .04906 | 79 / 79 |
| Schedule-Free AdamW, evaluated at x | .03 | .07985 / .08085 | 79 / 80 |

A constant-class baseline gets 8 of 80 correct; uniform probabilities have loss log(10)≈2.30259. Every selected model learns useful structure. The more interesting comparison is between the two metrics: AdamW gets every assessment label right here, while Sophia assigns probabilities that give a lower average cross-entropy despite one error. Classification accuracy counts decisions; cross-entropy also measures how probability was allocated.

Do not choose a universal winner from those eighty images. The validation results differ substantially: Prodigy's final losses are .41470 and .49441, while Schedule-Free's are .19111 and .19220. The small partitions expose different cases. An apparently excellent assessment number does not justify retuning on that partition.

The Schedule-Free diagnostic also illustrates its mode contract. For seed 11, final validation loss is .19111 at x and .19281 at y. For seed 29 it is .19220 at x and .18660 at y. The training iterate happens to score lower in the second case. We still report x because that is the method's defined evaluation model, not because it wins every comparison.

**Figure 11 — read the curves as evidence.** Plot actual saved fitting and validation losses against optimizer updates. Provide separate method/seed selections and a shared comparison; show both candidates in the tuning view. Keep probabilities, accuracy and loss on different labeled scales. A log-loss-axis option must say that the axis is logarithmic, and no smoothed curve may hide a measured increase.

For Prodigy, a companion trace shows d rather than pretending it is exactly the AdamW learning rate. In these selected runs its final values are .07640 and .09739. For Sophia, show the actual clipped fraction; at the final update it is about .03385 and .06. Those diagnostic quantities help explain an update, but neither alone measures successful learning.

### Reproduce and investigate

Download [the data](digits-400.csv), [the complete update rules](optimizer_rules.py) and [the complete study program](optimizer_study.py) into one directory. The [provenance](data-provenance.md) documents attribution, source identities, split construction and versions. The program needs Python and NumPy; the optional calculation checker also uses PyTorch. This packet ran with Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu.

```text
python -m venv .venv
```

Activate that environment using its platform's normal activation command, then run:

```text
python -m pip install numpy==2.3.5
python optimizer_study.py
```

The program prints the twelve selected assessment records and writes all candidate histories to `study-results.json`. `fitted-optimizer-states.json` retains the selected models' full weights and optimizer state, including Schedule-Free's different iterates. The exact arithmetic/probe checker is [optimizer_calculations.py](optimizer_calculations.py); it also needs the saved study files and PyTorch. The files contain complete programs, with no missing loader, model or loss function.

**Investigation: one new image, one real update.** Start from a saved model and a fresh validation image. Edit a stroke pixel, choose a target label and predict which class probabilities will increase after one diagnostic update. The actual edited pixels produce the scores and gradients. A change of label changes the gradient, while leaving the pre-update inference probabilities unchanged.

This is a temporary copy of the fitted model, not an alteration of the reported experiment. For example, changing pixel 28 of source 277 from 0 to 16 and applying one AdamW update toward its original class 0 changes its class-0 probability from .97019 to .97777. That is evidence about this edited example and copied state. It does not establish improved generalization. The Sophia diagnostic can use the exact expected-label diagonal for this linear model; label it distinctly from the sampled estimator used during fitting.

Cosine AdamW has already reached zero learning rate at update 400. Continuing its frozen schedule gives no parameter change at the next step. That is an informative control, not a reason to silently restart its schedule to produce a visible animation. Restoring the original snapshot must recover the same weights and predictions.

## 8. Count the resources you actually need

### Persistent state is not peak training memory

An array with N entries, each s bytes, occupies Ns payload bytes. When comparing state counts, first decide whether you are counting additional arrays, the current parameter buffer, master weights, gradients or temporary workspaces.

| Method / stated implementation | Additional parameter-sized arrays beyond current parameters |
| --- | --- |
| AdamW without AMSGrad | m and v: 2 |
| Lion | m: 1 |
| Sophia | m and h: 2 |
| Prodigy paper reference here | m, v, s and initialization: 4 |
| Schedule-Free teaching code here | x, z and v: 3 |
| Efficient Schedule-Free base implementation | z and v: 2; the parameter buffer switches between x and y |

Scalar counters are excluded from this table. Current library options can change the count: extra inner momentum, sliced adaptation statistics, factored state or a redundant reference copy must be counted explicitly. State dtype is implementation-dependent. Our NumPy experiment uses float64; a count table does not magically make its arrays float32.

For a hypothetical 70-billion-parameter model with two float32 AdamW moments, those two arrays contain 560 billion bytes, or 560 decimal GB. One float32 Lion moment contains 280 GB. Bfloat16 model weights separately contain 140 GB. These figures exclude gradients, master copies, activations and temporary storage. Dividing a global byte count by device count is justified only for the arrays actually sharded under that scheme.

**Figure 12 — an allocation ledger.** Stack separately named weight, gradient, moment and workspace allocations, each with a dtype and ownership rule. Change the number of sharding ranks and recompute only the sharded terms. Do not represent hypothetical global storage as a measured per-device peak.

### Adafactor offers a different compression

For an n×m weight matrix, Adafactor can store row and column statistics instead of an nm-entry second-moment array. Using row sums R and column sums C, a rank-one reconstruction has entries Ṽᵢⱼ=RᵢCⱼ/ΣᵢRᵢ. It preserves these marginals, not every individual entry.

For a 4096×4096 matrix, two float32 factors hold 8192 numbers, requiring 32,768 bytes. A dense second moment holds 16,777,216 numbers, requiring 67,108,864 bytes. This is not a fixed “half a parameter array” saving: it depends strongly on shape. Vector/scalar parameters and optional first moments need separate accounting.

The [Adafactor paper](https://proceedings.mlr.press/v80/shazeer18a.html) also discusses update clipping, changing second-moment decay and parameter-relative step sizes. Factoring the state is one part of the method. It does not inherently prove a fixed loss of accuracy or a fixed slowdown.

### Steps, work, traffic and elapsed time

Imagine a hypothetical method needs 600 updates at 1.1 time units per update, while another needs 1000 at 1 unit. Total time is 660 versus 1000, a speed ratio about 1.515. Reporting “40% fewer steps” as “40% faster updates” confuses two measurements. Tuning trials and failed runs also belong in a project-cost comparison.

Reducing optimizer state does not automatically reduce network traffic by the same factor. Data-parallel gradient reduction communicates gradients; parameter all-gathers communicate parameter shards. Persistent moments can remain local to their owner. Prodigy's global adaptation statistic may itself require a reduction. An actual distributed implementation determines which values move, when and at what precision.

This connects directly to the previous Ring Attention discussion: bandwidth claims require a communication schedule, buffer sizes and an overlap model. A memory-count heatmap cannot establish them. For real timing, use a representative workload, warmup, synchronization appropriate to the device, repeated measurements, matched quality targets and a record of software/hardware versions.

## 9. Choose a method by a question you can test

If optimizer state dominates memory, measure the benefit of a smaller or factored state while checking validation quality. If step-scale tuning dominates repeated experiments, compare adaptation against the total tuning budget of the baseline. If a run's endpoint is uncertain, test an evaluation strategy that remains useful at multiple stopping times. If curvature information seems promising, include its estimation cost and verify that the curvature path is actually used.

This gives a practical decision table without invented suitability scores:

| Observation | Candidate experiment | Evidence to collect |
| --- | --- | --- |
| Persistent moment arrays dominate | Lion or factored-state method | Measured peak/steady memory, matching quality, tuning budget |
| Different coordinates behave very differently | Sophia or a suitable preconditioner | Curvature/clipping diagnostics, quality versus total work |
| Many model scales make LR searches expensive | Prodigy with a declared implementation | Actual trials, d trace, failures and validation |
| Useful stopping time is uncertain | Schedule-Free plus explicit evaluation modes | Quality at predefined horizons, x/y handling, normalization statistics |
| Baseline already trains well | Keep it as a reference when testing alternatives | Same task, useful paired controls and honest tradeoffs |

### Matrix geometry: a useful current extension

The four methods above do not exhaust optimizer design. **Muon** transforms a matrix-valued momentum direction using an approximate orthogonalization procedure. An idealized polar factor of M=UΣVᵀ is UVᵀ: singular directions are retained while nonzero singular magnitudes are flattened. This is different from applying an entrywise sign.

For M=[[2,1],[1,2]], all entries are positive, so entrywise sign gives an all-ones rank-one matrix. The ideal polar factor is the identity because M is positive definite. One transformation acts on entries; the other acts on singular geometry. Neither calculation requires a Hessian.

**Figure 13 — entries versus singular directions.** Show the two matrices and their actions on [1,1] and [1,−1]. Label UVᵀ as the idealized polar operation, not the output of a fixed finite Newton–Schulz iteration.

[PyTorch 2.14 documents Muon](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html), including finite iteration coefficients, matrix-shape requirements and learning-rate adjustment. Non-matrix parameters need a suitable companion update, and an embedding's two-dimensional shape alone does not establish that every recipe treats it like a hidden matrix. The exact recipe and parameter grouping matter. Muon is a useful connection to the matrix-preconditioning lesson, not evidence that every advanced optimizer lives outside native PyTorch.

### Diagnosing a changed training run

Start with the actual objective and parameter changes. If loss jumps after switching optimizers, record the gradient norm, update norm, parameter norm, rate, decay and active state. Check whether the batch or reduction changed. Reusing a familiar rate may be a poor choice, but a failure at one particular update does not identify its cause by itself.

For Sophia, inspect curvature refreshes and scaling before concluding that “second-order information failed.” For Prodigy, inspect the estimator and d trace before treating a slow start as convergence. For Schedule-Free, confirm which iterate produced the reported metric. For Lion, inspect the blended direction and ηλ product. These are mechanisms that can be checked, not diagnoses based only on an optimizer's name.

The next [Neural ODE lesson](/learn/path/full-curriculum/neural-ode-continuous-depth-models?module=deep-learning-fundamentals) considers continuous state evolution inside a model. An optimizer also generates a trajectory, but its trajectory lives in parameter space while training. The distinction between evolving model state and updating model parameters remains essential.

## 10. Practice and transfer

### 1. A different AdamW first step

Use θ=[1,−2], g=[−3,6], η=.02, λ=.5, β₁=.9 and β₂=.999. Ignore ε only for this hand calculation. Find the corrected moments and next parameters. Would adding λθ to g first give the same rule?

<details><summary>Hint</summary>

The first corrected moments recover g and g². Apply shrinkage independently of the normalized direction.

</details>
<details><summary>Solution</summary>

m̂=[−3,6], v̂=[9,36], so the direction is [−1,1]. Shrink to [.99,−1.98], then obtain [1.01,−2]. Adding λθ to g would contaminate the moment calculation with the regularizer; it is not decoupled AdamW even if a particular first-step sign happens to agree.

</details>

### 2. Lion follows history

Use θ=−.4, old m=−.3, g=2, β₁=.9, β₂=.99, η=.02 and λ=.1. Find the update and new momentum. What current g would make the blend zero in exact arithmetic?

<details><summary>Hint</summary>

Solve −.27+.1g=0 separately from the β₂ memory update.

</details>
<details><summary>Solution</summary>

The blend is −.07, so θ_next=−.4(.998)+.02=−.3792. New m=−.297+.02=−.277. The zero threshold is g=2.7. This threshold depends on the magnitude of the old memory; knowing only its sign is insufficient. Finite-precision implementations may place a mathematically exact decimal tie just to one side, which is why the worked zero-state null uses exact zero inputs.

</details>

### 3. Curvature is not the real-label gradient squared

For a scalar logistic weight, x=3 and p=.25. Calculate the gradient squared for real label 1 and the expected squared gradient under a model-sampled label. Explain why the results differ.

<details><summary>Hint</summary>

The gradient is x(p−y). Enumerate y=0 and y=1 with their model probabilities.

</details>
<details><summary>Solution</summary>

For label 1 the gradient is −2.25 and its square is 5.0625. The sampled expectation is .25(5.0625)+.75(.5625)=1.6875, equal to 9(.25)(.75). The first quantity reflects one observed residual; the second averages over the model's possible labels and isolates the logistic curvature.

</details>

### 4. A curvature probe can disagree with the diagonal

For H=[[4,−2],[−2,1]], calculate u⊙Hu for u=[1,1] and u=[1,−1]. What is their average? Does the negative sample prove H has a negative eigenvalue?

<details><summary>Hint</summary>

Multiply by H before multiplying elementwise by u. The two remaining sign probes duplicate these results.

</details>
<details><summary>Solution</summary>

The estimates are [2,−1] and [6,3]; their mean is [4,1]. H has eigenvalues 5 and 0, so it is positive semidefinite despite that negative sampled entry. An estimator's individual signs do not determine the matrix's eigenvalues.

</details>

### 5. Find the wrong Prodigy implementation

An implementation computes ordinary Adam moments, estimates d from only the current displacement dot product, caps d at .003, and multiplies the ordinary Adam update by d. Is this paper Algorithm 4? Identify three repairs needed before presenting a comparison under that name.

<details><summary>Hint</summary>

Look at the histories, where d enters and which quantities persist across updates.

</details>
<details><summary>Solution</summary>

No. The moments must accumulate dg and d²g²; numerator and vector denominator statistics have their own weighted histories; the stated paper rule takes a nondecreasing maximum without that arbitrary cap. The update's d/ε convention and old-versus-next state also need matching. A useful alternative algorithm can be studied, but it must be named and evaluated as that alternative.

</details>

### 6. Compute the model that will be evaluated

Run the simple Schedule-Free SGD rule with x=z=−1, target 1, η=.2, β=.9 and perturbations [−.5,.5,1,−1]. Find x, z and the next training point after update 2; then find the gradient at update 3. How do the final averages differ for β=0 and β=1?

<details><summary>Hint</summary>

The first update gives x=z=−.5. Remember that the gradient is evaluated at an interpolation, not necessarily at z.

</details>
<details><summary>Solution</summary>

After update 2, z=−.3 and x=−.4. The next y is −.39 and the next gradient is −.39−1+1=−.39. After all four updates, x≈−.19456 for β=.9, −.208 for β=0 and −.193 for β=1. This tiny noise sequence does not make the commonly chosen .9 universally optimal.

</details>

### 7. Which resource saving did you measure?

A matrix has shape 2048×1024. Compare a float32 dense second moment with two float32 row/column factors. Separately, a model has eight billion parameters and two float32 moments sharded evenly over eight ranks. How many decimal GB of moments belong to each rank? Does replacing two moments with one halve a gradient all-reduce?

<details><summary>Hint</summary>

Use nm entries for the dense matrix and n+m for the factors. State payload and communicated payload are different objects.

</details>
<details><summary>Solution</summary>

The dense moment is 8,388,608 bytes; factors require 12,288 bytes. For the model, global moments occupy 64 GB and each ideal equal shard occupies 8 GB. One float32 moment would give 4 GB per rank under the same sharding. The gradient array is unchanged, so its all-reduce is not automatically halved. Account for the actual algorithm's collectives and other allocations.

</details>

### 8. Accuracy and loss disagree

Classifier A assigns correct-label probabilities [.51,.99,.99,.99]; B assigns [.49,.999,.999,.999] on four binary examples. Which has more correct decisions, and which has lower mean cross-entropy? Why must optimizer selection state its objective?

<details><summary>Hint</summary>

Use a .5 decision threshold and average the four negative logarithms.

</details>
<details><summary>Solution</summary>

A gets 4/4 correct, B gets 3/4. A's loss is about .17587; B's is about .17909, so A wins both for these inputs. Now replace A's three .99 probabilities by .90: its decisions remain 4/4 but its loss rises to about .24736, and B has the lower loss. The changed case shows the tradeoff; do not assume the first probability list must demonstrate it. Accuracy counts boundary decisions, while cross-entropy measures allocated probability throughout the set.

</details>

### 9. Design a comparison that can answer your question

You can afford twelve small training runs and care about useful quality at both 200 and 400 updates. Compare a baseline with two alternatives. Specify data roles, tuning allocation, seeds, evaluation iterates, additional curvature work, reported metrics and a rule for failures. What would you refuse to infer from the result?

<details><summary>Hint</summary>

Three methods × two declared scales × two seeds use all twelve runs. Intermediate evaluation does not require a new fit.

</details>
<details><summary>Solution</summary>

One defensible design fixes three disjoint data roles, two candidate scales and two seeds per method; uses common batches at each seed; reports both horizons; and selects scales by a prespecified validation criterion. It reports the defined evaluation iterate, actual extra gradient work and any failed run rather than replacing it after assessment. Assessment is reserved for selected settings. Twelve small runs cannot establish universal superiority, broad hardware speedups or robust performance on an unrelated large model. Different learning-rate grids or a different quality target may answer a different question.

</details>

## References and other ways to learn

These links support different parts of the lesson. The papers and executable local calculations carry the technical claims; external learning resources offer another presentation.

- [Loshchilov and Hutter: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), plus [the versioned PyTorch AdamW documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html). Read for the regularization distinction and an exact implementation contract.
- [Chen et al.: Symbolic Discovery of Optimization Algorithms](https://arxiv.org/html/2302.06675v4), especially the algorithm, tuning and limitations, with [Google's Lion source](https://github.com/google/automl/blob/master/lion/lion_pytorch.py). Useful after the one-coordinate trace; the paper also explains the search process and evaluated tasks.
- [Liu et al.: Sophia](https://arxiv.org/html/2305.14342v4), method and estimator sections, with [the official implementation](https://github.com/Liuhong99/Sophia). Use the paper to check the sampled-label normalization; do not treat a package's batch default as a universal value.
- [Mishchenko and Defazio: Prodigy](https://arxiv.org/html/2306.06101v3), especially Algorithm 4 and its distinction from the proved convex variants; [official package guidance](https://github.com/konstmish/prodigy) explains practical options and schedules. The executable classroom rule explicitly identifies its paper version.
- [Defazio et al.: The Road Less Scheduled](https://arxiv.org/html/2405.15682v2), method, large-rate discussion and implementation concerns; [official Schedule-Free repository](https://github.com/facebookresearch/schedule_free) for current modes and options. The paper's theorem conditions deserve the same attention as its curves.
- [The Road Less Scheduled, NeurIPS 2024 author presentation](https://slideslive.com/39024867/the-road-less-scheduled). An alternate video route after the x/y/z diagram. The recording page, title and conference attribution were checked; the recording was not watched or transcribed for this manuscript.
- [Dive into Deep Learning: Adam](https://en.d2l.ai/chapter_optimization/adam.html). A ground-up article with algebra and code for the prerequisite averages. Its notation differs, and some surrounding “variance” wording is loose; distinguish the raw second moment as taught here. The article was read; its remote notebooks were not executed.
- [Shazeer and Stern: Adafactor](https://proceedings.mlr.press/v80/shazeer18a.html). Follow the factored-state mechanism beyond a memory slogan. Optional momentum and tensor shapes change storage.
- [PyTorch 2.14 Muon](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html). A current matrix-update branch; inspect parameter groups, finite iterations and scaling before adapting a recipe. This lesson did not benchmark Muon.
- [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), E. Alpaydin and C. Kaynak, [dataset DOI](https://doi.org/10.24432/C50P49), CC BY 4.0. The retained extract and classroom transformations are described in [data provenance](data-provenance.md).
