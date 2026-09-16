# Batch, Layer, Group, and RMS Normalization: Which Values Belong Together?

A neural layer may receive numbers whose typical size changes as earlier layers learn. A tanh unit that used to receive values near zero can start receiving values near ten, where its output barely changes and its derivative is small. Normalization layers deliberately rescale collections of intermediate values. Their most important design decision is **which values share the calculation**.

The preceding loss lesson explained how predictions are judged. Here we inspect the intermediate activations that produce those predictions. We will follow a small collection of numbers through normalization, connect the operation to gradients, and compare real training runs.

**First pass:** calculate one normalization, explore the axis map, compare BatchNorm's training/evaluation modes, and run the small handwriting experiment. The gradient derivation, residual placement, precision, and distributed sections are deeper branches. You do not need to know convolutions or Transformers to follow the core: the tensor axes are introduced locally.

## A reference frame for a collection of numbers

Suppose a collection contains \(1,3,5,7\). Its mean is four. Subtracting four gives deviations \(-3,-1,1,3\). Their mean squared value is five, so the standard deviation is \(\sqrt5\). Dividing each deviation by it produces approximately \(-1.342,-.447,.447,1.342\).

For a group \(S\) of \(M\) activation values, the centered normalization is
\[
\mu=\frac1M\sum_i x_i,\qquad
v=\frac1M\sum_i(x_i-\mu)^2,\qquad
\hat x_i=\frac{x_i-\mu}{\sqrt{v+\epsilon}}.
\]
The small positive \(\epsilon\) prevents a zero denominator. Here variance divides by \(M\), not \(M-1\). It describes the collection being normalized; later we will distinguish the running-variance convention used by PyTorch BatchNorm.

**Visual — shared ruler:** show the raw points, their mean, centered deviations, and the standard-deviation ruler. Each point remains individually labeled through the transformation. The operation changes its coordinate relative to the collection, rather than “squeezing every value between zero and one.”

Before any learned affine transformation, the normalized collection has mean zero and variance \(v/(v+\epsilon)\), which is close to one only when \(v\) is large relative to \(\epsilon\). The result is not necessarily Gaussian. A bimodal collection remains bimodal.

Networks often follow the normalization with learned scale and shift:
\[
y_i=\gamma_i\hat x_i+\beta_i.
\]
These are trainable parameters, not measured means or variances. When the same \(\gamma,\beta\) apply across a normalization group, its output mean is \(\beta\) and variance is \(\gamma^2v/(v+\epsilon)\). With different per-feature scales and shifts within the group, use the actual transformed values to calculate those statistics.

The affine transform lets the network choose useful output scales and offsets. It cannot generally reconstruct the different mean and magnitude removed from every possible input. For example, \([1,3]\) and \([11,13]\) produce the same centered normalized vector. A single fixed affine transform sees identical inputs and cannot infer which original offset was present.

This loss of information is sometimes desirable and sometimes costly. Normalization is a modeling choice, not a universally harmless numerical cleanup.

## Tensor axes: name the collection before applying a formula

A tensor is a multidimensional table. In an image-like representation with shape \((N,C,H,W)\):

- \(N\) indexes examples in the current batch.
- \(C\) indexes channels: different measurements or learned features at a position.
- \(H,W\) index positions in a two-dimensional grid.

A channel could initially contain grayscale brightness; inside a model it can contain a learned response. We do not need the convolution operation yet to understand which cells are grouped.

Consider \((N,C,H,W)=(2,4,1,2)\). The first example contains channel rows \([1,2],[3,4],[5,6],[7,8]\); the second contains \([9,10],[11,12],[13,14],[15,16]\).

| Method and this declared layout | One statistics group contains | Number of values per group |
|---|---|---:|
| Training BatchNorm2d | One channel, all examples and spatial positions | \(NHW=4\) |
| LayerNorm over \((C,H,W)\) | One complete example | \(CHW=8\) |
| GroupNorm with \(G=2\) | One example, two adjacent channels and all their positions | \((C/G)HW=4\) |
| InstanceNorm2d | One example, one channel, all positions | \(HW=2\) |

**Investigation — who shares my ruler?** Select the first value, one. Highlight every value used in its mean and variance. For BatchNorm, the group is \([1,2,9,10]\). For the declared LayerNorm, it is \([1,\ldots,8]\). For GroupNorm with two groups, it is \([1,2,3,4]\). For InstanceNorm, it is \([1,2]\).

Now change only the value nine to nineteen, in the second example. Predict which methods change the normalized first example. Only training BatchNorm does: its shared reference group crossed the batch axis. In the executed fixture, the largest first-example output change is about .150221 for BatchNorm and exactly zero for the other three methods.

This is a dependency question, not a claim that BatchNorm is worse. Sharing across examples provides different statistical information and also introduces batch-composition dependence. The [Group Normalization paper's formulation and Figure 2](https://arxiv.org/pdf/1803.08494) offer another useful view of these grouping choices.

### Group limits do not erase parameter differences

With \(G=1\), GroupNorm uses the same centered statistics as LayerNorm over the complete \((C,H,W)\) example. With \(G=C\), its statistics match InstanceNorm over \((H,W)\). The program checks both equalities with common epsilon and identity affine parameters.

But PyTorch's LayerNorm can learn a distinct scale and shift for every element of its declared normalized shape. GroupNorm learns them per channel and shares them over positions. Thus `GroupNorm(1,C)` and `LayerNorm((C,H,W))` need not represent identical learned functions once their affine parameters differ. Moreover, `LayerNorm(C)` normalizes the **last dimension**; it does not automatically find the channel axis in an NCHW tensor. [LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html) and [GroupNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html) document these contracts.

The number of groups must divide the channel count. Choose a valid count for each layer, then evaluate it. There is no rule that 32 groups is possible or optimal for every channel width.

## LayerNorm and RMSNorm on a feature vector

For a sequence representation \((B,T,D)\), \(B\) indexes examples, \(T\) token positions, and \(D\) features describing each token. A token can be a word piece, an audio step, or another sequence element. LayerNorm with `normalized_shape=D` calculates statistics across features **within each token**, independently of other tokens and examples.

RMSNorm uses the same feature group but a different statistic:
\[
\operatorname{RMSNorm}(x)_i
=\gamma_i\frac{x_i}{\sqrt{\frac1D\sum_jx_j^2+\epsilon}}.
\]
It rescales by the root mean square without subtracting the mean. PyTorch's module has a learned scale and no additive bias. Do not describe every normalization method as centering to zero. [The RMSNorm paper, §4](https://arxiv.org/pdf/1910.07467) motivates retaining rescaling while removing recentering; its empirical results are not a guarantee for every model.

With identity affine parameters and \(\epsilon=10^{-5}\):

| Input | LayerNorm output, approximately | RMSNorm output, approximately |
|---|---|---|
| \([1,3]\) | \([-1,1]\) | \([.447213,1.341639]\) |
| \([11,13]\) | \([-1,1]\) | \([.913500,1.079591]\) |
| \([-1,1]\) | \([-1,1]\) | \([-1,1]\) |
| \([5,5]\) | \([0,0]\) | \([1,1]\) |
| \([0,0]\) | \([0,0]\) | \([0,0]\) |

These values were calculated by the accompanying program. RMSNorm retains information about the vector's common offset relative to its magnitude; LayerNorm removes a common offset. Both reduce sensitivity to positive overall scaling, exactly when epsilon is zero and denominators are nonzero, approximately when epsilon is small compared with the relevant squared scale. Negative scaling reverses signs rather than leaving outputs unchanged.

**Investigation — change brightness or contrast?** Edit an actual two-feature vector, first by adding the same offset to both entries, then by multiplying both by a positive scale. Predict which outputs remain unchanged. Use the zero-mean vector as a null comparison where the two methods agree. For values close to zero, increase epsilon and observe why the approximate scale invariance weakens.

For a causal sequence model, an output at position \(t\) must not depend on future inputs. Per-token normalization over \(D\) respects that boundary. Normalizing over \((T,D)\) mixes token positions and can introduce a future dependency even if attention itself has a causal mask. Padding included in a reduction can likewise change its statistics. Check the axes rather than assuming the layer's name guarantees the desired dependencies.

## BatchNorm remembers information between forward passes

Training BatchNorm computes current-batch statistics. With running statistics enabled, it also updates stored estimates that evaluation will use. That makes it a **stateful** layer: the same parameters can behave differently depending on stored buffers and mode.

In PyTorch's ordinary fixed-momentum convention,
\[
\mu_{\mathrm{run,new}}=(1-m)\mu_{\mathrm{run,old}}+m\mu_{\mathrm{batch}}.
\]
Here \(m\) is the weight on the new statistic. For the running variance, PyTorch uses the batch's variance with denominator \(M-1\), while the training normalization itself uses denominator \(M\). This is a library contract; an exponential average is not a guarantee of the exact whole-dataset variance. [BatchNorm2d's documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html) specifies the behavior and the `track_running_stats=False` exception.

Start with running mean zero and running variance one. A channel contains \([1,3,5,7]\), mean four and within-batch variance five. The corrected variance is \(5\times4/3=20/3\). With momentum .1:

| Item | After one training forward pass |
|---|---:|
| Running mean | \(.9(0)+.1(4)=.4\) |
| Running variance | \(.9(1)+.1(20/3)=1.566667\) |
| Training-normalized first value | \((1-4)/\sqrt{5+10^{-5}}\approx-1.341639\) |
| Same value in evaluation mode | \((1-.4)/\sqrt{1.566667+10^{-5}}\approx .479360\) |

Calling `eval()` does not force these two functions to agree. It switches to the stored estimates, which are still immature after one pass. A mode bug, insufficiently representative running statistics, a distribution change, and incorrect preprocessing are different diagnoses.

**Investigation — follow the state:** predict the next running mean and variance before applying a batch. Change a batch's measurements or momentum, commit the prediction, then inspect the current-batch output and the updated memory. Switch to evaluation and verify that another forward pass leaves that memory unchanged.

### One image is not always one value

Training BatchNorm2d with shape \((1,C,H,W)\) has \(HW\) values per channel. For a single channel with spatial values \([1,3]\), it produces approximately \([-1,1]\); the variance is not zero. With exactly one value per channel, PyTorch's training BatchNorm rejects the input rather than silently treating it as a normal training case.

Spatial values can be correlated. Having a thousand adjacent pixels is not equivalent to having a thousand independent images. This explains why the useful amount of batch information depends on the task and representation, not only an advertised batch-size threshold.

Three switches control three different things:

| Action | What it changes | What it does not do |
|---|---|---|
| `model.eval()` | Mode-sensitive layer behavior; ordinary BatchNorm uses running statistics | Disable gradient recording |
| `torch.no_grad()` | Autograd recording inside the context | Stop training-mode BatchNorm buffer updates |
| `parameter.requires_grad_(False)` | Gradient calculation for that parameter | Freeze all module buffers or switch modes |

A forward pass alone does not perform an optimizer update. If a model adapts BatchNorm statistics to a deployment sample, that still consumes information and changes the predictor even without changing weights. Use an authorized adaptation set and evaluate the resulting protocol honestly. Do not quietly recalibrate on a final test and call it untouched.

Ordinary gradient accumulation over eight microbatches does not give BatchNorm one eight-times-larger batch: each forward pass uses its own statistics and updates buffers. SyncBatchNorm can combine statistics across participating devices for a simultaneous step, at a communication cost. It does not automatically pool sequential accumulation steps. Whether to retain BatchNorm, synchronize it, freeze its state, or change the architecture must be evaluated with the actual pretrained model and training protocol.

## The operation is differentiable, including its statistics

The mean and variance depend on every value in their group. Detaching them from autograd changes the gradient. For one centered group, write \(s=\sqrt{v+\epsilon}\), upstream derivatives \(d_i=\partial L/\partial y_i\), and \(u_i=d_i\gamma_i\). Then
\[
\frac{\partial L}{\partial x_i}
=\frac1s\left[u_i-\operatorname{mean}(u)
-\hat x_i\operatorname{mean}(u\hat x)\right].
\]
The two subtraction terms account for changing the group's center and scale. The formula is valid with the stated variance and epsilon convention; do not replace \(\hat x\) with an independently sampled normalized vector.

For RMSNorm, with \(r=\sqrt{\operatorname{mean}(x^2)+\epsilon}\), there is no centering correction:
\[
\frac{\partial L}{\partial x_i}
=\frac{u_i}{r}-\frac{x_i\,\operatorname{mean}(ux)}{r^3}.
\]
Scale gradients accumulate \(d_i\hat x_i\) across uses of each shared parameter, and shift gradients accumulate \(d_i\). These reductions follow the affine parameter's sharing pattern, which can differ from the statistics group's axes.

Take \(x=[1,3]\), LayerNorm, \(\gamma=[1,2]\), \(\beta=[0,0]\), target \([0,1]\), and mean squared error. The initial output is about \([-1,2]\), loss .999985. The scale gradients are approximately \([.999990,.999985]\) and shift gradients \([-.999995,.999990]\). One gradient step of .1 updating only scale and shift produces output approximately \([-.799997,1.799993]\) and loss .639992. Those are executed values, with \(x\) held fixed for the update.

The tiny input gradient in this two-feature example is not evidence that the engine is broken. Away from ties, centering and dividing by the spread of two values leaves almost only their order when epsilon is small. The affine parameters can still learn. Use more features and changed inputs when investigating broader gradients.

## A complete CPU experiment

Download [normalization-experiments.py](normalization-experiments.py) and [digits-400.csv](digits-400.csv) into the same directory. The program contains readable centered, group, RMS, and stateful BatchNorm formulas, comparison fixtures, and twelve complete training runs. It uses no pretrained model or dataset download during training.

```text
python -m venv .venv
.venv\Scripts\python -m pip install numpy==2.3.5 torch==2.14.0 scikit-learn==1.9.1
.venv\Scripts\python normalization-experiments.py
```

Use `.venv/bin/python` on macOS/Linux. The author used Python 3.12.14, torch 2.14.0+cpu, and one CPU thread. A clean-environment replay remains part of later implementation review.

The real data has 400 UCI handwritten digit specimens, 40 per class, each with 64 pixel values from zero to 16. We divide by 16 and use the same stratified 280/120 development split as the earlier neural lessons. The [provenance record](data-provenance.md) explains why this is not an official UCI or writer-independent benchmark.

The model is `64 → Linear(32) → normalization → tanh → Linear(10)`. The normalized vector contains learned features, not spatial convolution channels. Compare no normalization, BatchNorm1d, LayerNorm, and RMSNorm; all affine defaults and epsilon are explicitly set or documented in code. Train with ordinary SGD at .1 for 50 epochs, ten batches of 28 per epoch. The example order and initial linear weights are matched for each seed. Use seeds one, two, and three.

The program switches to evaluation mode for reported training and validation loss. This matters for BatchNorm: the reported training-set loss uses stored running statistics and is not the sequence of instantaneous batch losses optimized during that epoch. The code records epochs0,1,5,20,50 rather than fabricating smooth curves between arbitrary values.

| Seed | Normalization | Training-set CE in evaluation mode | Validation CE | Correct /120 |
|---|---|---:|---:|---:|
| 1 | None | .101865 | .162959 | 116 |
| 1 | Batch | .021480 | .116648 | 117 |
| 1 | Layer | .023192 | .118822 | 116 |
| 1 | RMS | .023227 | .116965 | 116 |
| 2 | None | .095411 | .163503 | 116 |
| 2 | Batch | .019784 | .128102 | 117 |
| 2 | Layer | .022219 | .121958 | 116 |
| 2 | RMS | .021885 | .117832 | 116 |
| 3 | None | .096612 | .174415 | 116 |
| 3 | Batch | .019923 | .142852 | 117 |
| 3 | Layer | .021124 | .146711 | 115 |
| 3 | RMS | .021135 | .145417 | 115 |

These are actual outputs for a fixed small protocol. All four variants learned; the unnormalized model did not diverge. Normalization reduced the final losses in these runs, while the count differences were small and seed dependent. This experiment does not prove a universal ranking, a speedup, or a requirement for normalization. Learning rates and architectures were not separately tuned to each variant, and no untouched final test was used to endorse a selected model.

The formula fixtures also compare outputs against torch modules using float64. Maximum forward discrepancies were below \(3.2\times10^{-15}\). The stateful BatchNorm function matched running means and variances exactly on its fixture and evaluation output within \(1.8\times10^{-15}\). Layer and group input-gradient comparisons were within \(1.4\times10^{-15}\). These are focused author checks, not certification of arbitrary production inputs.

**Visual — measured learning trajectories:** connect only the recorded epoch points, show individual seeds, and allow the learner to select training-evaluation CE, validation CE, or correct counts. An explanatory caption should identify the common experiment and mode. Do not label a lower-loss line “stable” without inspecting the underlying behavior.

## Deeper: placement, information, and numerical behavior

### A residual connection is an extra path

A residual block adds a learned change \(F(x)\) to its input: \(x+F(x)\). Detailed residual architectures come later. This small definition is enough to compare normalization placement:
\[
\text{post-norm: }y=\operatorname{LN}(x+F(x)),\qquad
\text{pre-norm: }y=x+F(\operatorname{LN}(x)).
\]
If \(F=0\), pre-norm returns \(x\); post-norm returns \(\operatorname{LN}(x)\). The positions are not interchangeable wrappers. In pre-norm, one direct derivative path is an identity. The total derivative also includes the learned branch and can still amplify or cancel directions.

[Xiong et al.](https://proceedings.mlr.press/v119/xiong20b.html) analyze how placement affects initialization-time gradients and warm-up in specified Transformer settings. That supports a useful design explanation, not a guarantee that arbitrary pre-norm networks need no warm-up or that post-norm fails above a fixed depth. Swapping placement in a trained checkpoint changes its function.

**Visual — zero-branch contrast:** trace \(x=[1,3]\) through the two diagrams with \(F=0\). Then enable a specified linear branch and recompute; do not introduce unexplained attention code, rotary positions, or a noncausal “LLM block” here.

### What normalization explanations can claim

The original BatchNorm paper framed its motivation using changing internal input distributions. Later work demonstrated that this explanation alone does not account for its benefits and studied smoother optimization behavior. [Ioffe and Szegedy](https://proceedings.mlr.press/v37/ioffe15.html) and [Santurkar et al.](https://arxiv.org/abs/1805.11604) are useful to read together. A careful explanation separates an operation we can calculate, a mechanism supported under stated analysis, and empirical results that depend on architecture and training.

Normalization can alter parameter scale sensitivity, gradients, and the stochasticity introduced by batch composition. It need not make the full loss landscape globally smooth in every model, eliminate the need for good initialization, or guarantee a larger learning rate is safe.

### Precision and epsilon

Epsilon is added to a variance or mean square, so it shares those squared units. Smaller epsilon can preserve more scale sensitivity at tiny magnitudes and can also produce larger inverse denominators. It is not automatically safer.

Floating-point **range** and **precision** are different. BF16 can represent small numbers such as \(10^{-5}\); its coarse spacing near one does not imply that every smaller number is zero. Squaring a large FP16 activation can overflow before epsilon has any chance to help. For a hand-written low-precision RMS implementation, promote values before squaring and reducing:

```python
import torch

x = torch.tensor([[300., 400.]], dtype=torch.float16)
scale = torch.ones(2, dtype=torch.float32)
working = x.float()
y = (working * torch.rsqrt(working.square().mean(-1, keepdim=True) + 1e-5) * scale)
y = y.to(x.dtype)
print(y)  # derived: approximately [.8486, 1.1318] after float16 rounding
```

This snippet states its output dtype; it is not an exact emulation of every autocast or fused-kernel rule. The packet's mathematical checks use float64 and training uses float32. Consult the actual library/version and keep checkpoint epsilon consistent. [PyTorch RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html) documents the current default when epsilon is unspecified; our experiment supplies \(10^{-5}\) explicitly.

### Costs and related ideas

An activation normalization examines order one value per element, with reductions and affine work. Linear complexity does not mean free: reductions, memory traffic, synchronization, and kernel launches can matter. For \((B,T,D)=(8,8192,8192)\), there are 536,870,912 elements. One float32 read is 2 GiB; one two-byte read is 1 GiB. These are traffic calculations, not a measured end-to-end speedup.

Fused implementations can reduce intermediate allocations and memory traversals. The exact passes and runtime depend on kernel, shape, dtype, hardware and compiler. RMSNorm's simpler statistic can save work, but fewer source-code operations do not establish a universal percentage improvement.

InstanceNorm's per-image, per-channel grouping has a useful connection to stylization: changing global contrast or channel offsets need not change normalized patterns. That property helps some image transformations but can erase intensity information needed elsewhere. [Ulyanov et al.](https://arxiv.org/abs/1607.08022) introduced the method in a fast-stylization setting. Weight normalization is a different family: it writes a weight vector as \(w=g\,v/\|v\|\), separating magnitude and direction rather than computing activation statistics. [Salimans and Kingma](https://arxiv.org/abs/1602.07868) is an alternate deeper route.

## Practice and diagnosis

1. **Different grouping.** A tensor has shape \((2,6,3,4)\). How many values contribute to one BatchNorm, GroupNorm with three groups, and InstanceNorm statistic? Hint: hold the unreduced indices fixed. **Solution:** BatchNorm24; GroupNorm24 per example/group; InstanceNorm12. Equal group sizes do not mean equal memberships.

2. **Fix an axis bug.** With shape \((2,5,8)\), a causal model uses LayerNorm((5,8)). An early output changes when only the last token changes. Explain and repair the dependency. **Solution:** the normalization includes time. LayerNorm(8) normalizes each token's features separately; also inspect other layers for future dependencies. The change affects the model, so a trained checkpoint requires an appropriate retraining/evaluation plan.

3. **Different running state.** Start running mean2 and variance4. A batch has mean6, population variance9, and four values. Use momentum .25. **Solution:** new mean3; corrected batch variance12, so running variance6. Its training denominator still uses9+epsilon. In evaluation, an input6 uses(6−3)/sqrt(6+epsilon).

4. **Predict a null.** Add the same constant to all values in one centered normalization group with identity affine. What changes? **Solution:** the mean shifts by that constant; deviations, variance and outputs remain unchanged. Changing just one member generally changes other members' normalized values. RMSNorm is not invariant to the same offset.

5. **Follow two kinds of state.** Under `torch.no_grad()`, a model in training mode changes its BatchNorm running mean. Is this evidence that the optimizer ran? **Solution:** no; the buffer update is part of the forward operation. Inspect parameters and buffers separately. `eval()` changes ordinary BatchNorm's statistics source; gradient recording is a different switch.

6. **Change the real experiment.** Reduce batch size from28 to14, keeping50epochs. Predict the number of optimizer steps, running-stat updates, and why this is not a comparison that changes only the number of values in a mean. **Solution:** both steps and BatchNorm updates double from500 to1000. A controlled study must declare whether it matches epochs, updates, data exposure, or compute and may need more than one comparison.

7. **Check apparent equivalence.** GroupNorm(1,4) and LayerNorm((4,1,2)) agree with identity affine parameters. Give a LayerNorm affine setting that GroupNorm's per-channel affine cannot match on arbitrary inputs with fixed shared statistics. **Solution:** give the two positions of one channel different shifts or scales. GroupNorm shares that channel's affine parameters across positions.

8. **Choose a diagnosis from evidence.** Validation becomes worse after changing cameras. Someone proposes running all test images through training-mode BatchNorm. Explain what information this uses and what should happen first. **Solution:** it adapts the predictor to the test input distribution and changes buffers; even without labels it is a changed evaluation protocol. First check preprocessing, mode, source/target distributions and representative development data; any adaptation should use a declared allowed dataset and be assessed under the intended deployment protocol.

## Another explanation and the next step

Use [Dive into Deep Learning's Batch Normalization chapter](https://d2l.ai/chapter_convolutional-modern/batch-norm.html) for an alternate derivation and model example. Its from-scratch teaching code uses gradient-recording state as a shortcut for mode; keep the explicit train/eval versus no_grad distinction from this lesson when using real torch modules. The primary GroupNorm figure is especially useful for the axis investigation; the LayerNorm and RMSNorm papers explain why removing batch dependence and removing centering are separate decisions. These resources supplement the local lesson rather than supplying missing prerequisites.

Next is **Transfer Learning and Fine-Tuning Strategies**. Reusing a learned network means deciding which weights and states are allowed to change. BatchNorm's distinction between trainable affine parameters, running buffers, and mode will be particularly useful there.
