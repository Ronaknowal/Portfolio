# Weight Initialization: Xavier, Kaiming, Orthogonal Methods & μP

**Explore as you read.** Inspect saved initialization/seed traces; edit four activation values, singular directions, depth and width/rate scaling. Show forward/backward second moments, means/variance, directional gain and shape/update formulas together. Continuous tiny models are distinct from selectors over measured training records. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose an initialization/parameterization by the signal and update behavior it preserves, without treating average scale as every-direction stability.


A network begins making predictions before it has learned anything. Its initial weights determine whether useful differences between inputs survive the journey through its layers—and whether a change in an early weight can still affect the loss.

Think of passing a sound through twenty amplifiers. A small gain error repeated twenty times can make the signal nearly inaudible or enormously loud. A neural network adds another complication: nonlinear gates can remove parts of the signal. Choosing a starting scale is therefore a problem about a whole sequence of transformations, not simply drawing “small random numbers.”

**First pass:** follow the signal example, the second-moment calculation, the initialization recipes, the geometry counterexample, and the complete digit experiment. Then solve the first three practice problems. The μP section is a second route for learning how to change network width while keeping a meaningful training procedure. LSUV, Fixup, and precision details are reference branches; they are not prerequisites for the next lesson.

You need weighted sums, a nonlinear activation, and the idea that backpropagation multiplies local sensitivities. We will refresh the statistics and matrix geometry where they are used. The [previous lesson on transfer learning](/learn/path/full-curriculum/transfer-learning-fine-tuning-strategies?module=deep-learning-fundamentals) reused learned weights. Here we study fresh initialization. Later, **μTransfer** will mean transferring selected hyperparameters across widths, which is a different operation.

## A signal can disappear before learning starts

Suppose every hidden layer has 64 inputs, zero bias, and a ReLU, which replaces negative values with zero. Consider three normal distributions for its weights:

| Weight standard deviation | Intended question |
|---|---|
| 0.01 | Are small values automatically safe? |
| \(\sqrt{2/64}\approx0.177\) | Does accounting for the number of inputs and the ReLU help? |
| 0.5 | What happens when repeated layers amplify too much? |

The experiment accompanying this lesson sends the same 128 Gaussian input vectors through twenty layers. These are diagnostic inputs, not digit images. It measures

\[
q=\operatorname{mean}(h^2),
\]

the mean squared activation, pooling every row and coordinate of that layer. A useful signal does not have to maintain exactly the same \(q\). But changes by dozens of orders of magnitude deserve investigation.

Actual float64 results for seed 1:

| Initialization | Layer 1 \(q\) | Layer 5 \(q\) | Layer 10 \(q\) | Layer 20 \(q\) |
|---|---:|---:|---:|---:|
| Normal, std 0.01 | 0.00294 | \(3.61\times10^{-13}\) | \(1.36\times10^{-25}\) | \(9.06\times10^{-51}\) |
| Xavier, gain 1 | 0.459 | 0.0337 | 0.00118 | \(6.81\times10^{-7}\) |
| Kaiming, ReLU | 0.918 | 1.077 | 1.207 | 0.714 |
| Orthogonal, gain \(\sqrt2\) | 0.976 | 0.682 | 0.630 | 0.232 |
| Normal, std 0.5 | 7.343 | 35,302 | \(1.30\times10^9\) | \(8.24\times10^{17}\) |

The input \(q\) was 0.970. These are observations from one finite network, not theoretical curves. The downloadable record also contains seeds 2 and 3. In particular, the orthogonal run drifted downward; its name does not guarantee a flat line after nonlinearities.

**Visual investigation: follow the signal.** Compare the forward mean square with a backward sensitivity trace on aligned layer axes; show a new scheme, observe whether its final signal will shrink, grow, or remain within a moderate range. Switching from forward signal to backward sensitivity changes what is measured; it must not silently relabel the same curve.

For the backward probe, the program forms a scalar by multiplying the final activations by a fixed random array and summing. It then differentiates that scalar with respect to every layer's activations. The input gradient RMS is \(9.54\times10^{-26}\) for small initialization, 0.848 for Kaiming, and \(9.10\times10^8\) for large initialization. This measures sensitivity to one chosen output direction. It does not measure all possible directions or prove that a classifier will train.

## What the scale calculation actually preserves

A **mean** describes location. A **variance** measures squared spread around that mean. A **second moment** measures squared distance from zero:

\[
\operatorname{Var}(z)=E[z^2]-(E[z])^2.
\]

They coincide when the mean is zero. After ReLU, that is usually no longer true.

Take four equally weighted values:

| | Values | Mean | Second moment | Variance |
|---|---|---:|---:|---:|
| Before ReLU | −2, −1, 1, 2 | 0 | 2.5 | 2.5 |
| After ReLU | 0, 0, 1, 2 | 0.75 | 1.25 | 0.6875 |

The second moment halved. The variance did **not** halve: the output also moved away from zero. Follow the four values as the negative ones move to zero, then compare their distances from zero with their distances from the new mean.

Now consider one preactivation,

\[
z=\sum_{i=1}^{n}w_i x_i.
\]

Here \(n\) is **fan-in**, the number of contributions to this output. Assume initially independent, zero-mean weights with variance \(s^2\), independent of the input vector. For a fixed input, cross terms vanish when averaging over those random weights:

\[
E_w[z^2\mid x]=s^2\sum_i x_i^2.
\]

If the input coordinates share second moment \(q\), averaging over inputs gives \(E[z^2]=ns^2q\). Input coordinates do not have to have zero mean for this calculation. What removes the cross terms is the assumption about the weights. After training, weights depend on the data and such independence is no longer a reliable description.

For a symmetric preactivation distribution, exactly half its squared mass is on either side of zero:

\[
E[\operatorname{ReLU}(z)^2]=\tfrac12 E[z^2].
\]

Combining these equations gives

\[
q_{\mathrm{next}}=\tfrac12 ns^2q.
\]

Choosing \(s^2=2/n\) makes the approximate layer-to-layer factor one. This is the ReLU **Kaiming**, or **He**, rule. The symmetry condition matters; a substantial bias can change how much of the distribution is removed. The original rectifier analysis also develops the backward calculation, with assumptions about gates and incoming derivatives. [He et al., §2.2](https://arxiv.org/pdf/1502.01852)

There are several different averages here. The derivation averages over random initializations and input assumptions. The program measures one realized tensor. A mean of individual coordinates' variances across examples would be yet another statistic. The chart reports pooled second moment explicitly, so it does not disguise those differences.

## Choose a recipe for the operation it initializes

For a linear layer stored as an \(m\times n\) matrix, fan-in is \(n\) and fan-out is \(m\). Forward signals collect \(n\) terms. Backward signals collect \(m\) terms.

| Method | Listed variance or construction | Starting use |
|---|---|---|
| LeCun normal | \(1/n\) | Preserve a linear forward second moment under the assumptions above; also part of particular self-normalizing recipes |
| Xavier/Glorot normal | \(2/(n+m)\) | Balance forward and backward scale for approximately linear, centered activations |
| Kaiming normal, fan-in | \(2/n\) for ReLU | Preserve the forward second-moment scale through ReLU |
| Kaiming normal, fan-out | \(2/m\) for ReLU | Prioritize the corresponding backward scale |
| Orthogonal | Construct orthogonal rows or columns, then multiply by a gain | Control the linear map's geometry, with attention to dimensions and later nonlinearities |

For a normal draw with variance \(v\), the standard deviation is \(\sqrt v\). For a uniform draw on \([-a,a]\), the variance is \(a^2/3\), so use \(a=\sqrt{3v}\). A gain \(g\) multiplies the weights and therefore multiplies variance by \(g^2\). The listed Kaiming variances already include the ReLU gain; do not apply it twice.

For example, a 100-input, 25-output linear layer has Xavier variance \(2/125=0.016\). Its normal standard deviation is about 0.1265 and its uniform bound is about 0.2191. The idealized forward factor is \(100(0.016)=1.6\), while the backward factor is \(25(0.016)=0.4\). Xavier compromises; it cannot preserve both exactly when dimensions differ. In extreme aspect ratios one of those factors can be arbitrarily small.

For leaky ReLU with negative slope \(a\), the squared multiplier under symmetry is \((1+a^2)/2\). This gives variance \(2/[(1+a^2)n]\). Setting \(a=0\) recovers ReLU. For tanh, a variance calculation near zero is only a local approximation because tanh saturates. A library's suggested tanh gain is a starting convention, not proof of constant moments at all depths. GELU and SiLU similarly deserve measurement in the actual architecture.

Xavier's original study connects activation saturation, initialization, and observed training behavior. Its assumptions and empirical comparisons motivate a diagnostic approach rather than a universal promise. [Glorot and Bengio, §§3–5](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

Here is a complete small API example:

```python
import torch
from torch import nn

torch.manual_seed(7)
hidden = nn.Linear(100, 25)
head = nn.Linear(25, 3)
nn.init.kaiming_normal_(hidden.weight, mode="fan_in", nonlinearity="relu")
nn.init.zeros_(hidden.bias)
nn.init.xavier_normal_(head.weight)
nn.init.zeros_(head.bias)
x = torch.ones(2, 100)
logits = head(torch.relu(hidden(x)))
print(logits.shape)  # torch.Size([2, 3])
```

These functions change the supplied tensor without recording the initialization in the autograd graph. PyTorch assumes the matrix will be used as `x @ weight.T`, with shape `[fan_out, fan_in]`. If your custom matrix is stored for `x @ weight` instead, initialize its transpose so the fan calculation corresponds to the operation. [PyTorch initialization API and orientation note](https://docs.pytorch.org/docs/2.14/nn.init.html)

## Average preservation can hide a collapsed direction

Imagine a circle of possible small changes to a two-dimensional input. Multiplying by a matrix turns it into an ellipse. The ellipse's longest and shortest radii are the matrix's **singular values**: its strongest and weakest directional gains.

Consider

\[
M=\begin{bmatrix}\sqrt{1.9}&0\\0&\sqrt{0.1}\end{bmatrix}.
\]

The mean of its squared singular values is \((1.9+0.1)/2=1\). Nevertheless, it stretches the horizontal direction by about 1.378 and shrinks the vertical one to about 0.316. Repeating this same matrix makes the discrepancy much larger. An average scale is not a guarantee about every direction.

There is also a distinction between averaging over draws and inspecting one draw. For a square \(d\times d\) matrix with independent, zero-mean entries of variance \(1/d\),

\[
E_W\|Wx\|^2=\|x\|^2
\]

for each fixed \(x\). A particular sampled matrix need not preserve that norm. Our saved 64-by-64 Gaussian draw has singular values from approximately 0.00374 to 1.936.

A square orthogonal \(Q\) satisfies \(Q^\top Q=I\), so \(\|Qx\|=\|x\|\) for every \(x\). The program's QR construction checks this identity to maximum absolute error \(8.9\times10^{-16}\) in float64. For a tall matrix, orthonormal columns can preserve input norms. For a wide matrix mapping to fewer dimensions, some input directions must be lost.

The next operation still matters. For \(f(x)=\operatorname{ReLU}(\sqrt2x)\) at \(x=(-1,1)\),

\[
J_f=\begin{bmatrix}0&0\\0&\sqrt2\end{bmatrix}.
\]

One local direction is completely blocked. The total input and output norms happen to agree at that particular point, which makes it an especially useful counterexample to judging the Jacobian from a single norm.

**Geometry investigation:** keep the average squared gain at one while changing the two directional gains. Inspect which input direction will lose sensitivity, then test it with an editable vector. Add or remove the ReLU gate and compare the actual local Jacobian. The identity map is the null comparison.

Keeping the singular values of a network's full input-output Jacobian close to one is the idea of **dynamical isometry**. The linear and nonlinear cases require different conditions. Orthogonal initialization is useful evidence about an individual linear map, not a certificate that an entire ReLU network has this property. [Saxe et al., dynamical-isometry discussion](https://arxiv.org/pdf/1312.6120)

## Construct an orthogonal draw, then match a width-aware optimizer

The scale formula tells us what to sample; an orthogonal initializer instead constrains a whole matrix. The complete [initialization bridge](initialization_library_bridge.py) exposes both cases. For a matrix with more rows than columns, sample a Gaussian matrix, take reduced QR, and multiply each column of Q by the sign of the matching diagonal of R. The sign choice removes the QR routine's arbitrary sign convention. For more columns than rows, construct the tall counterpart and transpose it. Multiply by the requested gain last.

```python
def orthogonal_matrix(rows, columns, gain=1.0, generator=None):
    tall = torch.randn(max(rows, columns), min(rows, columns),
                       dtype=torch.float64, generator=generator)
    basis, triangular = torch.linalg.qr(tall, mode="reduced")
    signs = torch.where(triangular.diagonal() < 0, -1.0, 1.0)
    basis = basis * signs
    return gain * (basis if rows >= columns else basis.T)
```

This uses the QR factorization already taught in [Matrix Decompositions](/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu); it does not hide initialization inside `orthogonal_`. With gain g, test `Q.T @ Q = g²I` for tall matrices and `Q @ Q.T = g²I` for wide ones. Testing the wrong identity would claim preservation in a dimension the map cannot preserve. Reduced QR costs O(max(m,n) min(m,n)²) arithmetic and O(mn) storage. This is a dense matrix construction, not a special accelerator kernel.

Run `python initialization_library_bridge.py` with PyTorch. It compares rectangular Gram contracts against `nn.init.orthogonal_`. Identical seeds need not yield identical wide matrices when routines draw arrays in different shapes; the meaningful comparison here is the distribution/construction contract and Gram identity.

After the μP derivation below, continue with `python initialization_library_bridge.py --mup` in an environment containing Microsoft's `mup` package. That optional branch reuses `WidthMLP` directly from the adjacent experiment file without rerunning its training sweep. Its ordinary model substitutes `MuReadout`, calls `set_base_shapes` with widths 32 and 64 to identify changing axes, copies the scratch model's already-parametrized weights using `rescale_params=False`, and constructs `MuAdam`. The input matrix has one changing axis, the hidden matrix two, and the readout one. Those annotations determine which optimizer group receives the width-divided rate. The readout performs the forward division.

The supplied comparison uses widths 32 and 96, identical inputs and targets, zero initial optimizer state and two Adam updates. It asserts output, every parameter gradient and updated-weight agreement. Turning parameter rescaling back on **after** copying the custom μP weights would change the experiment. This code is prepared for the matched run; no optional-package result is claimed here. The [readout](https://raw.githubusercontent.com/microsoft/mup/main/mup/layer.py), [shape registration](https://raw.githubusercontent.com/microsoft/mup/main/mup/shape.py) and [optimizer source](https://raw.githubusercontent.com/microsoft/mup/main/mup/optim.py) specify the current mapping inspected on 22 September 2026. Pin the tested package version when executing it.

**Implement a changed case.** Use a 3×7 orthogonal draw with gain 0.5 and change the μP target width to 160. Which identity and rates should the comparison check?

<details><summary>Hint</summary>Distinguish output-row orthogonality from preserving every seven-dimensional input direction; the width ratio is measured against 32.</details>

<details><summary>Solution and success criteria</summary>The Gram check is `Q @ Q.T = 0.25 I₃`; `Q.T @ Q` has rank at most three. The width ratio is five. For base Adam rate 0.003, input/readout rates remain 0.003, the hidden matrix rate becomes 0.0006, and the raw readout divides its input by five. Preserve the same copied state and compare two updates. A passing shape check alone does not establish those identities.</details>

## Why equal hidden units can stay equal

Suppose two hidden units have identical incoming weights and biases, the same activation, and identical outgoing weights. Interchanging the units changes nothing. On the same example, they receive the same gradients, so an identical update preserves their equality. Two slots have learned one feature twice.

In the saved two-unit example, input 1 passes through tanh, both incoming weights are 0.2, both outgoing weights are 0.3, the target is 1, and the loss is half squared error. Both incoming gradients are −0.25417. Changing the incoming weights to 0.1 and 0.3 gives gradients −0.26218 and −0.24234. Deliberately different deterministic values already break this symmetry; randomness is convenient, not logically necessary.

“Never initialize anything to zero” is too broad. With distinct tanh features and a zero output head, the first hidden gradients are zero, but the head gradients are −0.09967 and −0.29131. The head can move first and open a later gradient route into the features. This is related to the zero-initialized LoRA output factor from the preceding lesson.

An all-zero hidden ReLU network is a different case. Its features are zero, and PyTorch's ReLU derivative at zero is zero. The digit experiment below leaves such a model predicting equal classes. The useful question is which paths can learn on the first and subsequent updates.

## Run a complete initialization comparison on real inputs

[Download the complete CPU program](./initialization-experiments.py) together with [digits-400.csv](./digits-400.csv). The [provenance and protocol](./data-provenance.md) describe the real handwritten digits and exact split. The file contains 400 samples from UCI's optical digit dataset, not MNIST. Each 8-by-8 image becomes 64 inputs divided by the known feature-range maximum, 16.

In an environment containing PyTorch, NumPy, and scikit-learn:

```text
python initialization-experiments.py
```

The program includes imports, data loading, all model definitions, deterministic splits and initialization generators, optimization, metric calculations, and output recording. It needs no pretrained download. The prepared outputs were executed with Python 3.12, PyTorch 2.14.0+cpu, NumPy 2.3.5, and scikit-learn 1.9.1. Reproduction in another version can differ slightly.

Read `DigitMLP` first. It has four 32-unit ReLU hidden layers and a ten-logit head. The six choices change hidden initialization; within each seed, all choices use the same randomly initialized head. All biases start at zero. The 280 training and 120 validation samples are identical across choices. Adam uses learning rate 0.003 for 300 full-batch updates.

`digit_fits` records cross-entropy and correct counts before training and after 1, 10, 100, and 300 updates. Cross-entropy measures assigned probability, while the correct count uses the largest logit. Neither alone describes every error.

Actual seed-1 final results:

| Hidden initialization | Training CE | Validation CE | Validation correct / 120 |
|---|---:|---:|---:|
| Zero | 2.302586 | 2.302585 | 12 |
| Normal, std 0.01 | 0.002644 | 0.933934 | 106 |
| Xavier | 0.000274 | 0.220745 | 116 |
| Kaiming | 0.000303 | 0.175867 | 117 |
| Orthogonal, gain \(\sqrt2\) | 0.000298 | 0.124550 | 117 |
| Normal, std 0.5 | 0.001471 | 0.778704 | 105 |

Across three seeds, Kaiming gives 116–117 correct and orthogonal gives 116–118. Xavier gives 113–117. These small differences are not evidence for a universal winner. The much poorer small/large validation CE also shows why fitting the training set is not the same as assigning good probabilities on other examples.

The tiny initialization can eventually learn in this four-hidden-layer model. That does not contradict the twenty-layer signal probe: the architectures and questions differ.

**Try a changed experiment:** keep the split and seed fixed, change only the number of hidden layers, and record the initial signal statistics before training. Inspect how the changed depth alters the measured forward and backward statistics. Inspect both early optimization and final validation, and record your change as a new experiment rather than replacing the prepared observations. Repeated validation comparisons consume development information; this packet does not provide a final test estimate.

## μP: changing width changes more than parameter count

Suppose a learning rate worked well in a 32-unit network. Can you tune cheaply there and reuse it in a 128-unit network?

A forward sum of independent zero-mean random terms tends to grow on the order of the square root of their count. But a learning update is correlated with the inputs that produced its gradient. Summing those correlated changes can scale differently. Therefore an initialization that controls the first forward pass is not enough to control the size of the first learned feature change.

**Maximal update parametrization**, written μP, coordinates initialization, forward multipliers, and optimizer scaling as width changes. Its associated μTransfer procedure tunes a smaller model and transfers eligible settings under the matching parametrization. It does not mean that every finite model has exactly the same best learning rate, nor that a rule derived for Adam can simply be copied into SGD. [μTransfer paper](https://arxiv.org/abs/2203.03466), [Microsoft's implementation and coordinate-check guide](https://github.com/microsoft/mup)

Here is the exact restricted case in `WidthMLP`: a bias-free 64→\(n\)→\(n\)→10 MLP with two ReLUs, fixed input/output sizes, base width \(n_0=32\), and width multiplier \(m=n/n_0\).

| Component | Standard comparison | μP convention used here |
|---|---|---|
| Input weight std | \(\sqrt{2/64}\) | same |
| Hidden-to-hidden weight std | \(\sqrt{2/n}\) | same |
| Raw readout weight std | \(1/\sqrt n\) | \(1/\sqrt{n_0}\) |
| Readout input | \(h\) | \(h/m\) |
| Adam input learning rate | \(\eta\) | \(\eta\) |
| Adam hidden learning rate | \(\eta\) | \(\eta/m\) |
| Adam raw-readout learning rate | \(\eta\) | \(\eta\) |

The raw μP readout is deliberately distinguished from its **effective** multiplication by \(1/m\). Leaving out that multiplier changes the model. At the base width \(m=1\), both columns describe exactly the same training procedure.

A readout has one dimension that grows with width; the middle matrix has two. The package calls these its infinite dimensions because it tracks what happens as width grows. This table follows the corresponding readout and hidden-matrix conventions of `MuReadout` and `MuAdam` for this architecture. The program makes the rules explicit to keep the example runnable without another dependency. For tied parameters, multiple changing dimensions, biases, attention, a different optimizer, or an existing architecture, use and inspect the package's base-shape machinery rather than treating this class as a generic replacement. [Readout implementation](https://github.com/microsoft/mup/blob/main/mup/layer.py), [optimizer implementation](https://github.com/microsoft/mup/blob/main/mup/optim.py)

The complete experiment fits widths 32, 64, and 128 with learning rates 0.001, 0.003, and 0.01, for both parametrizations and three seeds. Each fit uses 150 full-batch updates on the same digit split. These are 54 small CPU fits.

Mean final validation cross-entropy across the three seeds:

| Parametrization | Width | LR 0.001 | LR 0.003 | LR 0.01 |
|---|---:|---:|---:|---:|
| Standard | 32 | 0.25880 | **0.10939** | 0.11739 |
| Standard | 64 | 0.12406 | **0.10485** | 0.12290 |
| Standard | 128 | **0.08466** | 0.08567 | 0.12092 |
| μP | 32 | 0.25880 | **0.10939** | 0.11739 |
| μP | 64 | 0.20471 | **0.09638** | 0.11020 |
| μP | 128 | 0.19483 | **0.08208** | 0.09354 |

The narrow-model choice, 0.003, remains best among these three candidates at the larger μP widths. In the standard width-128 run, 0.001 has slightly lower average CE than 0.003; the difference is only about 0.00101. This is an illustrative local experiment, not proof of exact transfer or a discovery of the global optimum. It also does not establish a compute or memory speedup.

A **coordinate check** inspects activation magnitudes as width changes, before spending a long run on tuning. The program records mean absolute values of five corresponding tensors at steps 0, 1, 2, 5, and 150 on the same 32 training examples. Compare the same tensor at the same step.

For seed 1 and learning rate 0.01, the μP mean absolute output at initialization is 0.381, 0.308, and 0.170 for widths 32, 64, and 128. It is not flat. A nonzero random μP readout can have a decaying initial output scale before correlated updates develop. The official guide discusses this transient; demanding exact equality would reject a valid behavior.

**Width investigation:** select a width, then construct the hidden learning rate and readout multiplier from the base rate. Inspect the result alongside the actual recorded curves. For width 128 and base rate 0.003, the hidden rate is 0.00075 and the readout input multiplier is 0.25. Changing only the learning rate while forgetting the forward multiplier is an explicit contrasting configuration, not another name for the same μP model.

## Deeper tools and practical failure checks

**Data-dependent initialization.** LSUV starts with orthogonal weights, sends a calibration batch through successive layers, and rescales each layer toward a chosen output variance. This adapts to an observed distribution rather than only an assumed one. The procedure is bounded by a tolerance and maximum iteration count. A zero or tiny variance needs a diagnostic stop, not division by zero. Calibration on training inputs is legitimate; using a held-out evaluation distribution to fit those scales consumes that information. The chosen measurement point—before or after an activation—must be explicit. [LSUV, Algorithm 1](https://arxiv.org/pdf/1511.06422)

**Residual initialization.** A residual block computes \(x+F(x)\). Making \(F\) initially small can begin near a usable identity map. Fixup uses a coordinated recipe including depth-scaled earlier branch weights, zero final branch/classification layers, and specified scalar multipliers and biases. For a branch containing \(r\) weight layers, its earlier branch weights use a factor \(L^{-1/(2r-2)}\), where \(L\) is the number of residual branches. This is not a universal instruction to multiply any second layer by \(1/\sqrt L\). The next lesson makes the paths and placement concrete. [Fixup, §3](https://arxiv.org/pdf/1901.09321)

**Truncated normal is not clipping.** Values outside the interval are redrawn; they do not pile up at the endpoints. In `trunc_normal_`, bounds are absolute values. With `std=0.02`, the default bounds −2 and 2 are one hundred standard deviations away. To truncate at two standard deviations, specify −0.04 and 0.04. In the program's 100,000-value float64 sample, these choices produce standard deviations about 0.02000 and 0.01758 respectively. The truncated distribution has less variance; the function does not silently restore 0.02 afterward. [PyTorch truncated-normal contract](https://docs.pytorch.org/docs/2.14/nn.init.html#torch.nn.init.trunc_normal_)

**Small numbers and precision.** BF16's reduced precision does not mean every number below 0.001 becomes zero. It has a broad exponent range. In our conversion fixture, \(10^{-5}\), \(10^{-6}\), and \(10^{-8}\) all remain nonzero in BF16. The last becomes zero in float16. Loss of a small update when added to a much larger weight is a different issue from representing the small value by itself. Measure the operation you are concerned about.

**Normalization and defaults.** Normalization can change activation scale, but its axes, epsilon, affine parameters, and Jacobian still matter. It does not certify that all initial gradients or update sizes are useful. Inspect the actual module rather than relying on a catalogue of alleged architecture defaults: for example, PyTorch's LSTM documents a hidden-size-based uniform initialization for all its weights and biases. It does not promise a default orthogonal recurrent matrix. [LSTM initialization note](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html)

When a model fails, inspect initial preactivations, post-activation mean square, zero/saturated fractions, gradient magnitudes, and actual update-to-weight magnitudes. Check inputs, loss, and labels too. Large logits do not automatically make a stable cross-entropy implementation overflow; large finite losses, saturation elsewhere, and excessive parameter updates are separate diagnoses.

## Practice: use the mechanism on a changed case

### 1. Initialize a wider-input layer

A ReLU layer has 200 inputs and 50 outputs. Find the fan-in Kaiming normal standard deviation and uniform bounds. Then calculate Xavier's idealized linear forward and backward factors.

<details><summary>Hint</summary>

square the standard deviation to obtain variance; uniform variance is bound squared divided by three.

</details>

<details><summary>Worked solution</summary>

Kaiming variance is \(2/200=0.01\), so std is 0.1 and bounds are \(\pm\sqrt{0.03}\approx\pm0.1732\). Xavier variance is \(2/250=0.008\), giving forward factor 1.6 and backward factor 0.4. Fan-out Kaiming would be a different priority, with variance \(2/50=0.04\).

</details>

### 2. Repair a misleading statistic

A layer's pooled mean square stays at 1, while its pooled mean changes from 0 to 0.8. Someone reports “the variance stayed at 1.” Correct the report.

<details><summary>Hint</summary>

subtract the squared mean.

</details>

<details><summary>Worked solution</summary>

the final variance is \(1-0.8^2=0.36\). The second moment stayed constant. We also need to know whether the samples and coordinates were pooled consistently before comparing those measurements.

</details>

### 3. Keep average gain, lose a direction

Construct a diagonal 2-by-2 matrix whose average squared singular value is 1 but whose smaller singular value is 0.2. Find the larger singular value and the gain after five repetitions along the smaller direction.

<details><summary>Hint</summary>

the two squared singular values must sum to 2.

</details>

<details><summary>Worked solution</summary>

the larger is \(\sqrt{1.96}=1.4\). The smaller direction has gain \(0.2^5=0.00032\). Preserving the average does not protect this direction.

</details>

### 4. Follow the first update

Two tanh hidden units have different incoming weights, but their outgoing weights are both zero. Is there no learning signal anywhere? Contrast this with an all-zero hidden ReLU network.

<details><summary>Hint</summary>

Trace the chain rule from the output backward, using the actual hidden features at each parameter.

</details>

<details><summary>Worked solution</summary>

the output weights can have nonzero gradients because the hidden features differ and are nonzero; earlier weights receive no gradient through the zero output weights on that first step. After the output moves, that route can open. In the all-zero ReLU case used here, the features and their chosen derivatives at zero block the relevant paths. Zero initialization must be assessed by location.

</details>

### 5. Transfer a base learning rate carefully

For the exact bias-free μP model above, base width is 32, target width is 256, and base Adam learning rate is 0.004. Specify the three group rates, raw readout std, and forward divisor.

<details><summary>Hint</summary>

Find the width ratio, then follow the separate input, hidden-matrix and readout conventions.

</details>

<details><summary>Worked solution</summary>

\(m=8\). Input and raw-readout rates are 0.004; the hidden matrix rate is 0.0005. Raw readout std remains \(1/\sqrt{32}\); divide its input by 8. These answers depend on the stated Adam parametrization, not a general rule for all optimizers.

</details>

### 6. Plan a useful failure investigation

You observe a roughly constant forward mean square but a tiny gradient in an early layer. Propose two checks that distinguish explanations.

<details><summary>Hint</summary>

Ask whether an average scale conceals different directions, and whether the current activations pass the incoming gradient.

</details>

<details><summary>Worked solution</summary>

inspect the local Jacobian or directional sensitivities to test whether some directions are blocked despite average scale preservation. Separately inspect activation gates/saturation and the backward signal arriving from the head. Record which scalar output or loss supplied that gradient. A single final norm cannot distinguish all these cases.

</details>

## Another way to learn, and what comes next

- [Stanford CS231n, Lecture 6: Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M) offers a lecture-based route through activation, initialization, and normalization. The official [2017 syllabus](https://cs231n.stanford.edu/2017/syllabus) identifies its scope. Use it for intuition; this lesson's μP material and versioned API checks go beyond that lecture.
- [Microsoft Research's μTransfer article](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/) explains why random forward sums and correlated training updates require different reasoning. It is a historical 2022 introduction, not a current catalogue of every supported model.
- [The original Xavier paper](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) is useful after the local calculation: read the assumptions and compare its activation diagnostics with the measurements here.
- [The μP repository](https://github.com/microsoft/mup) is the implementation route after the restricted example. Study base shapes, readout layers, optimizer choice, and coordinate checks together.

The next topic in this module is [Residual Connections & Skip Connections](/learn/path/full-curriculum/residual-connections-skip-connections?module=deep-learning-fundamentals). Initialization controls the starting transformations. A residual path changes how those transformations are connected, giving the network a direct route alongside the learned correction.
