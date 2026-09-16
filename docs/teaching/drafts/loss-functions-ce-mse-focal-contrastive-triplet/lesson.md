# Loss Functions: Predictions, Probabilities, and Learned Similarity

A learning algorithm needs more than examples of correct answers. It needs a way to say how an imperfect answer should change. Predicting a delivery ten minutes late, assigning a wrong label with 99% confidence, and retrieving the wrong photograph are different failures. A **loss function** assigns a numerical penalty to a prediction and its target. Its derivatives tell backpropagation how that penalty responds to changes in the model.

The preceding lesson explained how to compute those derivatives. Here we choose what to differentiate. A small loss is useful only when the objective represents the behavior we need.

**First pass:** follow the prediction-to-update map, work the regression and probability examples, investigate focal loss, then build the pair→triplet→candidate-selection connection. Run the small digit experiment and attempt the practice before its solutions. The likelihood, angular-margin, and mutual-information branches add depth; their derivations are not prerequisites for the first experiment.

## From a prediction to an update

Take a prediction \(\hat y=3\) for a measured value \(y=5\). Squared error gives \(L=(3-5)^2=4\). Its derivative with respect to the prediction is \(2(\hat y-y)=-4\): increasing the prediction slightly decreases loss. If \(\hat y=wx+b\), backpropagation continues with \(\partial L/\partial w=-4x\) and \(\partial L/\partial b=-4\). The optimizer then decides the step size.

**Visual — three linked views:** a prediction on the target axis; its height on the loss curve; the tangent and resulting parameter update. Label the axes separately. A loss of four does not mean “move four units,” and a negative derivative is not a negative loss.

For \(B\) examples, a common training objective is
\[
J(\theta)=\frac1B\sum_{i=1}^{B}L(f_\theta(x_i),y_i)+\lambda R(\theta).
\]
Here \(f_\theta\) is the model, \(\theta\) its parameters, \(R\) a regularization penalty, and \(\lambda\) its strength. Architecture, data, and sampling also shape what is learned. The loss is one part of that system.

The quantity reported to users can differ from the training objective. A classifier can minimize differentiable cross-entropy and be assessed using recall, a confusion matrix, and a decision cost. We choose a smooth surrogate because a count of correct labels is flat over most small parameter changes. We must still evaluate the intended outcome.

## Regression: what should “a typical answer” mean?

Let the residual be \(r=\hat y-y\). Four useful penalties produce different responses:

| Penalty | Per-example formula | Derivative with respect to \(\hat y\), away from corners | What it emphasizes |
|---|---|---|---|
| Squared error | \(r^2\) | \(2r\) | Large numerical errors |
| Absolute error | \(|r|\) | \(\operatorname{sign}(r)\) | An error's direction, with bounded magnitude |
| Huber | \(r^2/2\) if \(|r|\le\delta\); \(\delta(|r|-\delta/2)\) otherwise | \(r\) inside; \(\delta\operatorname{sign}(r)\) outside | Smooth small-error correction, bounded large-error slope |
| Quantile, level \(q\) | \(q(y-\hat y)\) if \(y\ge\hat y\); \((1-q)(\hat y-y)\) otherwise | \(-q\) below the observation; \(1-q\) above it | A chosen asymmetric underprediction/overprediction balance |

MSE means the **mean** of squared errors. At residuals one and ten, squared-error losses are one and one hundred; prediction-gradient magnitudes are two and twenty. The loss ratio and gradient ratio are different.

Consider a constant predictor for seven measurements: \(0,0,0,0,0,0,10\). MSE is minimized at the arithmetic mean \(10/7\approx1.429\). MAE is minimized at the median, zero. For Huber with \(\delta=1\), the optimum is \(1/6\): six small residuals contribute derivative \(6c\), the large residual contributes \(-1\), and \(6c-1=0\).

**Investigation — move one measurement:** predict what happens to these three fitted constants when the last measurement changes from 10 to 100. Move the measurement, inspect each point's contribution, and compare your prediction. MSE's optimum becomes \(100/7\); the MAE and Huber optima stay at zero and \(1/6\). Replacing all measurements by three is a useful contrast: all three losses agree on three.

This is not permission to delete a troublesome observation. A rare large value can be the event the application must predict. Check the measurement and choose the estimand: the mean for expected total cost, a median for a typical case, or a high quantile for a capacity target. Huber reduces sensitivity to large residuals; it does not decide whether those residuals are mistakes.

### Deeper: why squared error estimates a mean

For a random target \(Y\), conditioning on the available input \(x\),
\[
\mathbb E[(Y-c)^2\mid x]=\operatorname{Var}(Y\mid x)+(\mathbb E[Y\mid x]-c)^2.
\]
The variance term does not depend on \(c\). With finite second moments, the optimal constant is the conditional mean, without any Gaussian assumption. Similarly, an absolute-error optimum is a conditional median; a quantile-loss optimum is a conditional quantile, possibly nonunique for discrete distributions.

A likelihood interpretation adds an explicit probability model. Under independent Gaussian errors with fixed common variance \(\sigma^2\), negative log likelihood is a constant plus \(\sum r_i^2/(2\sigma^2)\). Its optimizer matches squared error. Fixed-scale Laplace errors give absolute error. If the model also learns a different \(\sigma(x)\) for each input, the Gaussian objective includes both \(r^2/(2\sigma(x)^2)\) and \(\log\sigma(x)\); omitting the latter rewards inflating uncertainty indefinitely.

For a practical nonstandard application, a service can predict the 90th percentile of demand to plan reserve capacity. That prediction is deliberately above the median. A quantile of 0.9 corresponds to nine times the local penalty slope for underprediction as for overprediction. The operational cost ratio, not a desire for uniformly high predictions, motivates that choice.

## Classification: confidence is part of the answer

For a binary label \(y\in\{0,1\}\), the model produces a real **logit** \(z\). The sigmoid \(p=1/(1+e^{-z})\) converts it into a number between zero and one. Binary cross-entropy is
\[
L=-y\log p-(1-y)\log(1-p).
\]
Only one term remains for a hard label. If the correct label is one, assigning probabilities 0.9, 0.5, and 0.1 gives losses approximately 0.1054, 0.6931, and 2.3026. Correct but uncertain and confidently wrong predictions receive different penalties. Natural logarithms give units of **nats**.

For mutually exclusive classes, use one logit per class and
\[
p_c=\frac{e^{z_c}}{\sum_j e^{z_j}},\qquad
L=-\log p_y=\operatorname{logsumexp}(z)-z_y.
\]
Differentiate: \(\partial L/\partial z_c=p_c-\mathbf1[c=y]\). The correct class receives a negative derivative unless its probability is already one; other classes receive positive derivatives. This is the output gradient used by the previous lesson's engine.

Cross-entropy is also negative log likelihood for the observed categorical outcome. At a population level, expected cross-entropy decomposes as \(H(q,p)=H(q)+D_{\mathrm{KL}}(q\Vert p)\), where \(q\) is the target distribution. The irreducible entropy \(H(q)\) remains even when probabilities are correct. Extra expected coding cost from using the wrong distribution is the KL term. [Stanford CS231n's softmax discussion](https://cs231n.github.io/linear-classify/#softmax) provides an alternate derivation.

### Stable logits and explicit targets

Avoid calculating a tiny softmax probability and then taking its logarithm. For logits \([1000,-1000]\) and correct index one, the loss is approximately 2000, not infinity. Compute log-sum-exp with a maximum shift. Binary CE has the stable form \(\max(z,0)-yz+\log(1+e^{-|z|})\).

```python
import torch
from torch.nn import functional as F

logits = torch.tensor([[1000., -1000.], [1., 2.]], dtype=torch.float64)
class_indices = torch.tensor([1, 0], dtype=torch.long)
print(F.cross_entropy(logits, class_indices))  # about 1000.656631
binary_logits = torch.tensor([-2., 1.5, .2, -.4])
binary_targets = torch.tensor([0., 1., 1., 0.])
print(F.binary_cross_entropy_with_logits(binary_logits, binary_targets))
```

These are complete API examples; the first result was also checked in the preceding backpropagation packet. The second is a prediction to verify when running this snippet. At the 0.5 threshold, all four binary examples are classified correctly, although their confidences differ.

Multilabel classification is different from multiclass classification: an image can have both “outdoors” and “vehicle.” Use independent binary targets and logits for those labels when that matches the task, not one softmax that forces exactly one category. Neither output format by itself guarantees calibrated probabilities.

With a probability target \(t_c\), CE becomes \(-\sum_ct_c\log p_c\), with gradient \(p_c-t_c\) for a normalized unweighted target. PyTorch label smoothing uses \(t=(1-\epsilon)\,\text{one-hot}+\epsilon/C\); a different convention allocates smoothing only to incorrect classes. State the convention. For \(C=3,\epsilon=.1\), PyTorch's target is \([.93333,.03333,.03333]\) when class zero is correct, not \([.9,.05,.05]\). Smoothing changes the desired probabilities; its effect on measured calibration depends on the model and evaluation. [CrossEntropyLoss's API contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html) specifies targets, shapes, weights, and reductions.

## Imbalance: inspect contributions before changing the objective

An always-negative classifier scores 90% accuracy on a task with 10% positives. That baseline should trigger inspection of recall and probability quality. It does **not** establish that BCE cannot learn rare positives. A constant-only model minimizes BCE by predicting prevalence; a model with informative inputs can do better.

A positive-term weight \(r\) gives
\[
L_r=-r\,y\log p-(1-y)\log(1-p).
\]
For a location with actual positive probability \(\eta\), minimizing its expected weighted loss yields
\[
p^*=\frac{r\eta}{r\eta+1-\eta}.
\]
For \(\eta=.1,r=9\), this is .5. The output now represents a changed cost balance, not automatically the original probability .1. In the ideal unrestricted population solution, subtracting \(\log r\) from the fitted logit recovers the unweighted log-odds. Finite-data models need validation; this is not a guaranteed calibration repair.

The ratio of negative to positive training counts balances aggregate class weights. It is a candidate choice, not a universal optimum. Changing the decision threshold is another intervention that leaves the learned ranking intact. Resampling changes the training distribution and can interact with weighting; combining both without accounting for the changes can unintentionally count the same preference twice.

### Focal loss changes emphasis as confidence changes

Define \(p_t=p\) when \(y=1\), otherwise \(p_t=1-p\). Unweighted focal loss is
\[
L_{\mathrm{focal}}=-(1-p_t)^\gamma\log p_t,\qquad \gamma\ge0.
\]
At \(\gamma=0\), this is CE. At \(\gamma=2,p_t=.9\), the loss is multiplied by .01. But the multiplier also depends on the model. Differentiating the product gives a gradient about .028965 times the CE gradient, **not .01 times**.

Let \(s=2y-1\), so \(p_t=\sigma(sz)\). Its derivative is
\[
\frac{\partial L}{\partial z}
=s(1-p_t)^\gamma\left[\gamma p_t\log p_t-(1-p_t)\right].
\]
This formula connects the moving multiplier to backpropagation. A class factor \(\alpha_t\), if used, multiplies both the loss and gradient. With the convention \(\alpha_t=\alpha\) for positives and \(1-\alpha\) for negatives, setting \(\alpha=1\) discards negatives. To recover ordinary BCE at \(\gamma=0\), omit class weighting rather than setting that \(\alpha\) to one.

**Investigation — who moves a shared bias?** Use 1000 negatives each with predicted positive probability .01, and one positive with probability .1. For an additive bias shared across the examples, sum each logit's gradient. Under CE the negatives contribute +10 and the positive contributes −.9. Under unweighted focal loss with \(\gamma=2\), negative contributions shrink to about +.002990 and the positive contributes about −1.102019. The net direction reverses. Display per-example and summed gradients, not just attractive loss curves. These are an intentionally constructed mechanism example, not observed digit probabilities.

Focal loss was introduced for the many easy background candidates in dense object detection; its paper's preferred hyperparameters are results for that setting. Hard examples may include label errors, so emphasizing them is not a general robustness strategy. [Lin et al., §3](https://arxiv.org/html/1708.02002v2#S3) explains that distinction. Focal confidence scores also need evaluation as probabilities: theoretical classification consistency does not imply strict propriety as a probability score. [Charoenphakdee et al.](https://arxiv.org/abs/2011.09172) studies this explicitly.

## One real experiment: recognize a nine

The downloadable [loss-experiments.py](loss-experiments.py) and [digits-400.csv](digits-400.csv) run entirely on a CPU after dependencies are installed. The data contains 40 real 8×8 handwriting images per digit from UCI Optical Recognition of Handwritten Digits. It is not MNIST. [Data provenance](data-provenance.md) explains the source, license, row selection, and split limitations.

We ask a binary question: “Is this digit nine?” There are 40 positives and 360 negatives. Split the specimens into 280 training and 120 validation rows, stratifying on the original digit with seed 22. Divide pixels by their documented maximum 16. Fit the same 64-input linear logit using BCE, positive-weight-nine BCE, or unweighted focal loss with \(\gamma=2\). Each gets Adam at .03 for 400 full-batch updates. Reset the initial seed for every objective; repeat seeds one, two, and three.

The code does not use validation examples in gradients or choose a winner. This small comparison measures the declared protocol. It has no separate final test and does not establish writer-independent handwriting performance.

```text
python -m venv .venv
.venv\Scripts\python -m pip install numpy==2.3.5 torch==2.14.0 scikit-learn==1.9.1
.venv\Scripts\python loss-experiments.py
```

On macOS/Linux, use `.venv/bin/python`. Put the CSV beside the program. Installation needs network access; training uses only the included data. The program writes `calculated-inputs.json`, containing split IDs, training traces, validation probabilities, metrics, and mathematical fixtures. The author's run used Python 3.12.14 and torch 2.14.0+cpu.

Read the implementation in three parts. `binary_focal` computes **unreduced** per-example losses from logits, retaining the focusing factor in the differentiation graph. `digit_experiment` performs the fixed split and nine fits. Its evaluation uses the same unweighted metrics across objectives, since raw training loss values from different formulas are not directly comparable.

| Seed | Objective | False positives | False negatives | Average precision | Brier score | Unweighted log loss |
|---|---|---:|---:|---:|---:|---:|
| 1 | BCE | 1 | 2 | .979070 | .012382 | .042302 |
| 1 | Positive weight 9 | 1 | 1 | .976389 | .013921 | .047416 |
| 1 | Focal, \(\gamma=2\) | 1 | 2 | .979070 | .017337 | .085298 |
| 2 | BCE | 1 | 2 | .979070 | .011961 | .043178 |
| 2 | Positive weight 9 | 1 | 1 | .986645 | .011386 | .042535 |
| 2 | Focal, \(\gamma=2\) | 1 | 1 | .979070 | .019137 | .097343 |
| 3 | BCE | 1 | 2 | .979070 | .012168 | .040372 |
| 3 | Positive weight 9 | 1 | 0 | .979070 | .010817 | .042596 |
| 3 | Focal, \(\gamma=2\) | 1 | 1 | .979070 | .015298 | .073454 |

These are executed results. There are 108 negatives and 12 positives in validation. At threshold .5, BCE's recall is \(10/12\), with specificity \(107/108\); its balanced accuracy is the average of those two fractions, about .9120. A single positive changes recall by \(1/12\), so small count changes deserve restraint.

Average precision summarizes the precision-recall ranking, using recall increments to weight precision; it is not an unspecified trapezoidal PR area. Brier score is mean squared probability error and log loss penalizes confident mistakes strongly. Lower is better for the last two columns; higher is better for average precision. The focal run's similar ranking and worse probability scores show why one metric cannot answer every question. None of these observations proves that a different learning rate, model, or loss variant would behave the same way.

**Visual — probability-to-decision audit:** choose a stored run, view its actual validation probabilities and labels, then set a threshold. Predict how false positives and false negatives will change before applying it. A threshold changes labels and confusion counts, while stored probabilities, average precision, Brier score, and log loss remain fixed. A threshold sweep is validation work; any future final test must remain unconsumed while choices are made.

## When the output is a location: pair and triplet losses

An **embedding** is a vector representing an item. Two recordings of the same machine state or two photographs of the same object may need nearby vectors even when their raw inputs differ. A shared encoder converts each input into coordinates; a similarity loss teaches relationships between those coordinates.

For a pair, let \(D=\|a-b\|_2\), and use \(y=1\) for a matching pair. One explicit contrastive convention is
\[
L_{\mathrm{pair}}=yD^2+(1-y)\max(0,m-D)^2.
\]
Matching items are pulled together; nonmatching items receive a penalty only while they are closer than margin \(m\). Some papers reverse the label convention or include a factor one-half. Convert labels and prefactors before comparing code. The original pair-learning research is [Hadsell, Chopra, and LeCun](https://yann.lecun.com/exdb/publis/).

At \(m=1,D=.2\), matching and nonmatching losses are .04 and .64. At \(D=1.2\), the nonmatching loss is zero. The zero-distance corner deserves care: the Euclidean norm has no unique derivative direction there. A library's zero-gradient convention can leave coincident negative embeddings stuck even though their loss is positive.

A triplet instead says “this positive should be closer than this negative.” Using **squared** distances,
\[
L_{\mathrm{triplet}}=\max(0,\|a-p\|^2-\|a-n\|^2+\alpha).
\]
The margin has squared-coordinate units. For \(a=(0,0),p=(1,0),\alpha=1\), negatives at \((.5,0),(1.2,0),(2,0)\) give losses 1.75, .56, and zero. They are hard, semi-hard, and easy respectively. Semi-hard means
\[
D_{ap}^2<D_{an}^2<D_{ap}^2+\alpha.
\]
An easy triplet supplies no local gradient; it remains useful as evidence that this particular constraint is satisfied. [FaceNet, §3.1–3.2](https://arxiv.org/pdf/1503.03832) motivates squared distances and explains the role of triplet selection.

**Investigation — choose a useful negative:** move the three candidate points and predict which is eligible under a stated mining rule. Our program selects the nearest strictly semi-hard candidate, breaking ties by row order; it skips the anchor if none exists. This is a transparent teaching policy, not a universal best miner. A no-candidate result should be displayed as “skipped,” never silently substituted with a zero-loss example.

Inside the active region, derivatives are \(2(n-p)\) for the anchor, \(2(p-a)\) for the positive, and \(2(a-n)\) for the negative. Take a small joint step and recompute both distances. Describing the gradients as attraction and repulsion is helpful, but a large finite step is not guaranteed to improve all desired distances.

At \(a=p=n\), squared-triplet loss with positive margin is \(\alpha\), yet every derivative above is zero. Positive margin makes complete collapse costly but does not make it impossible. Unit normalization prevents the zero vector from being a valid unit vector, but every item can still collapse onto the same nonzero unit vector. Inspect embedding spread, norms, pair labels, active constraints, and retrieval outcomes together.

### Match the distance used by the API

PyTorch `TripletMarginLoss(p=2)` uses unsquared Euclidean distances, with an epsilon convention. It does not directly match the squared FaceNet formula. Our `squared_triplet` implements that formula explicitly; `TripletMarginWithDistanceLoss` can also accept a squared-distance function. [The API documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TripletMarginLoss.html) makes this distinction reviewable.

```python
import torch
from torch.nn import functional as F

a = torch.tensor([[0., 0.]])
p = torch.tensor([[1., 0.]])
n = torch.tensor([[1.2, 0.]])
squared = ((a-p).square().sum(-1) - (a-n).square().sum(-1) + 1).clamp_min(0)
unsquared = (torch.linalg.vector_norm(a-p, dim=-1)
             - torch.linalg.vector_norm(a-n, dim=-1) + 1).clamp_min(0)
print(squared, unsquared)  # derived: approximately .56 and .8
```

This changes more than notation. Copying the same numerical margin between distance definitions changes which examples are active.

## InfoNCE: finding the correct candidate is classification

For one query \(q\), suppose there is one designated positive key and \(K\) designated negative keys. Convert their similarities \(s_j\) into logits \(s_j/\tau\), where temperature \(\tau>0\). Then
\[
L=-\log\frac{\exp(s_+/\tau)}{\sum_{j=0}^{K}\exp(s_j/\tau)}.
\]
This is ordinary categorical CE over **candidate items**, rather than over class names. For cosine similarity, use nonzero vectors normalized to unit length. Normalization is a design choice that makes the geometry angular; general InfoNCE does not mathematically require it. A temperature of one is also a valid choice, not an absent parameter that makes the objective invalid.

**Visual — candidate competition:** a query points to three labeled keys. Their similarity bars become temperature-scaled logits, then probability shares. Lower temperature sharpens preferences. Predict both a correctly ranked and incorrectly ranked query: sharpening helps the first and can increase the second's loss dramatically. If all similarities are equal, the probabilities stay uniform at any positive temperature.

For \(N=K+1\) equal candidates, loss is \(\log N\). This is the uniform baseline, not an upper bound. A positive that receives much less probability than \(1/N\) has larger loss.

The program's `paired_info_nce` uses a \(B\times B\) score matrix with matching query/key rows as positives. A row's remaining keys are designated negatives. It is one-way paired learning. SimCLR constructs two views of each input, excludes each view's self-comparison, and averages both positive directions among \(2B\) views. Those masks define different candidate sets. [SimCLR's method and Algorithm 1](https://arxiv.org/pdf/2002.05709) are useful to inspect after this simpler matrix.

A same-class key may be a **false negative** for the intended task. If two candidate keys are identical and only one is designated positive, no scoring function can give that positive more than half their combined probability; even if every other key becomes irrelevant, loss cannot fall below \(\log2\). This is different from an ID or timestamp shortcut that lets training loss approach zero without learning useful semantics.

When multiple items really are positive, one option is supervised contrastive learning: average the negative log probability assigned to each positive within the candidate set. Keep self-comparisons out and define what happens when an anchor has no positive. An average of log probabilities differs from taking the log of the summed positive probability. [Khosla et al., §3.2](https://arxiv.org/pdf/2004.11362) compares these formulations.

### Deeper connections and useful boundaries

For exact nonzero unit vectors, \(\|a-b\|^2=2-2a^\top b\). Therefore squared-triplet constraints can be written with cosine similarities. They do not become identical to CE over candidates: margins, candidate weighting, and gradients still differ. As \(\tau\to0\), **\(\tau L\)** tends to \(\max_j s_j-s_+\). The unscaled loss can diverge when a negative wins, or retain a log-tie penalty. Temperature is not an explicit triplet margin.

The CPC derivation connects expected InfoNCE with a mutual-information lower bound \(I\ge\log N-L_N\), under its joint-positive and marginal-negative sampling assumptions. \(N\) counts all candidates, including the positive. Increasing \(N\) also changes \(L_N\); simply adding \(\log2\) to a claimed bound without reevaluating the loss is unjustified. [CPC §2.3](https://arxiv.org/pdf/1807.03748) gives the assumptions and density-ratio interpretation. Useful representations should still be assessed on retrieval or downstream prediction.

An angular-margin classifier such as ArcFace normalizes features and class weights, scales their cosine logits, and modifies the target logit to \(s\cos(\theta_y+m)\) during training. This directly shapes angular separation; the scalar \(m\) is an angle, unlike a squared-triplet margin. The purpose is an embedding useful beyond the training class head, not a guarantee that any cosine margin wins. The [ArcFace paper](https://arxiv.org/pdf/1801.07698v3) is a deeper application, after the pair and candidate geometry are secure.

## Practical reductions, scaling, and diagnosis

Always specify what receives one vote. Averaging per pixel, per sequence, or per example can produce different objectives. With a mask \(m_i\), an explicit masked mean is \(\sum_i m_i L_i/\sum_i m_i\), with an intentional policy for an empty mask. A long sequence should not accidentally carry more weight merely because its loss was summed while another term was averaged.

Class-index weighted PyTorch CE divides a mean by the sum of included target weights; CE with probability targets uses a per-observation mean. Binary `pos_weight` multiplies positive terms but its mean still divides by the number of elements. Our focal loss returns a vector and explicitly averages its elements. These denominators matter when comparing gradients or combining losses.

Full CE over \(C\) already-computed logits costs order \(BC\); a dense head mapping \(D\) features into those logits costs order \(BDC\). A contrastive \(B\times B\) similarity matrix costs order \(B^2D\) and stores \(B^2\) scores. For \(B=1024\), float32 scores alone use 4 MiB; at 4096 they use 64 MiB, before gradients and encoder activations. These are dimensional calculations, not runtime benchmarks.

Batch-hard mining can share a pairwise distance matrix of order \(B^2D\), then use label masks and row reductions. Enumerating every possible dataset triplet is unnecessary. Sampled-softmax and noise-contrastive approaches change how candidate normalization is estimated; hierarchical softmax changes the factorization. Their bias, sampling corrections, and inference behavior need their own treatment before substitution. Ordinary gradient accumulation does not create similarities between separate microbatches; enlarging the negative set requires retaining or gathering the relevant embeddings.

Use the observed failure to choose the next check:

| Observation | First useful inspection | What it does not prove |
|---|---|---|
| High accuracy, no positive predictions | Class counts, ranking, probabilities, threshold, recall | CE cannot learn an imbalanced task |
| Large loss from a few residuals | Units, measurement validity, desired estimand | Those observations should be removed |
| Focal improves recall, worsens log loss | Threshold and calibration on validation | Focal is uniformly better or worse |
| Triplet loss stays at the margin | Embedding spread, gradients, positive/negative labels | A larger margin alone will fix collapse |
| Candidate loss becomes tiny, retrieval fails | Split leakage, shortcuts, candidate identities, gallery protocol | Duplicates necessarily explain tiny loss |
| Combined losses change with batch size | Reduction denominators and term gradients | Equal displayed loss values give equal influence |

## Practice: change the problem, then explain the consequence

1. **Different measurement units.** Delivery errors change from minutes to seconds. How do squared loss, absolute loss, and a Huber threshold change if the intended behavior should remain the same? Hint: write \(r'=60r\). **Solution:** squared loss multiplies by 3600, absolute by 60. Scale Huber's threshold by 60; its numerical loss then scales by 3600. An optimizer or a combined objective may need corresponding scale adjustments.

2. **Different cost balance.** The actual positive probability is .2 and the positive weight is four. Derive the ideal weighted-BCE output. Hint: use expected loss before differentiating. **Solution:** \(p^*=.8/(.8+.8)=.5\). This does not mean the original event has probability .5. A .5 threshold in this ideal weighted space corresponds to an original probability threshold .2.

3. **Different margin.** With \(a=(0,0),p=(1,0),n=(1.2,0)\), change the squared margin from one to .3. Predict the active status and loss. **Solution:** \(1-1.44+.3=-.14\), so loss and its local gradient are zero. The geometry did not change; the required separation changed.

4. **Different candidates.** Three candidates have equal scores; add a fourth identical candidate. What happens to loss, and can lowering temperature reverse that? **Solution:** it rises from \(\log3\) to \(\log4\); temperature cannot distinguish equal scores. Now move only the designated positive score upward and explain why lowering temperature can help.

5. **Repair the experiment.** A colleague selects the threshold with the lowest error count on the final test, then reports that count as untouched performance. Identify the information consumed and propose a valid continuation. **Solution:** test labels selected a modeling decision. Treat that test as development information, freeze the revised protocol using development data, and acquire or reserve another untouched evaluation set if an unbiased final assessment is required.

6. **Run a changed objective.** Add `focal_gamma_0` to the program with no class weighting, resetting seeds exactly as before. Predict its relationship to BCE, then inspect losses, parameters, and validation probabilities with floating-point tolerances. **Solution:** the formulas are identical; differences should be limited to implementation/numerical effects. Adding the balanced \(\alpha=.25\) convention would change the objective and invalidate this null comparison.

7. **Diagnose a gradient.** Two identical embeddings designated negative have positive pair loss but no useful update under the library's zero-distance convention. Explain why displaying only the loss misses the problem. **Solution:** the loss value says the constraint is violated; the norm's derivative direction at coincidence is not defined. Inspect representation initialization, symmetry, nonzero variations, and the actual gradient instead of treating positive loss as proof of movement.

## Another way to learn, and the next connection

Start with [Stanford CS231n's linear-classification notes](https://cs231n.github.io/linear-classify/) if a second worked softmax explanation helps; its score→loss→probability diagrams complement the local regression view. [Stanford Lecture 3: Loss Functions and Optimization](https://www.youtube.com/watch?v=h7iBpEHGVNc) provides a spoken explanation of classification objectives and how optimization uses them. Watch it after the probability example; it does not replace the later focal and metric-learning sections. The official channel description was checked; the video itself was not watched for this packet.

For paper reading, use focal §3 after the contribution lab, FaceNet §3 after the mining exercise, and SimCLR Algorithm 1 after drawing the candidate mask. Read CPC's information-theory branch only after candidate CE is comfortable. The articles are alternate routes and sources; the local explanation and program stand on their own.

The next topic is **Batch, Layer, Group, and RMS Normalization**. We now know how scores are judged. Next we examine the intermediate numbers entering those scores: which collections of activations are rescaled, how that changes dependence between examples, and why training and inference sometimes use different statistics.
