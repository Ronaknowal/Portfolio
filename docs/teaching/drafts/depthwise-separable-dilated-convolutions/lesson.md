# Depthwise Separable & Dilated Convolutions

**Explore as you read.** Edit depthwise/pointwise filter entries, channel cells, stencil dilation/offsets, serial rates and parallel branch choices; manipulate retained digit inputs where weights are available. Update output contributions, rank restrictions, visited lattice sites, branch union and exact frozen-model outputs. Keep coverage geometry separate from learned influence. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose separability, dilation or parallel context from expressiveness, blind spots and the measured budget.


A convolution usually answers two questions at once: **what pattern appears nearby, and how should evidence from different channels be combined?** A depthwise separable layer separates those jobs. A dilated layer changes a different choice: **how far apart should the sampled positions be?**

These choices matter when a camera model must fit a device budget, when a segmentation model needs both local boundaries and distant context, or when you want to understand why a seemingly cheaper replacement changes predictions. The previous [Landmark Architectures lesson](/learn/path/full-curriculum/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet?module=deep-learning-fundamentals) compared complete CNN designs. Here we open two of their building blocks and follow the actual numbers.

**First pass:** read sections 1–4 to understand the operators, 6–7 for a practical experiment and diagnosis, then attempt practice 1–5. Section 5 connects the mechanisms to mobile models and segmentation. The rank proof, gradient derivation and interval-coverage argument deepen the explanation; you can return to them without delaying the basic investigations. You need weighted sums and the idea that training changes weights to reduce a loss. Shapes, channel mixing and sampling geometry are refreshed here.

## 1. Separate the spatial and channel questions

An image-like tensor has height, width and channels. RGB channels are measured colors; later channels are learned feature maps. In the PyTorch examples the axes are **batch, channels, height, width**, written $N\times C\times H\times W$. A channel is a whole map, not one pixel or one class.

For a standard $3\times3$ convolution, each output channel has a separate $3\times3$ filter for **every** input channel. At one output location it multiplies each local patch by the corresponding filter, sums within patches, then adds across channels. With $C_{\rm in}$ input and $C_{\rm out}$ output channels there are $9C_{\rm in}C_{\rm out}$ weights, before biases.

Imagine two input maps: one describes vertical contrast and another brightness. Standard convolution can use a different spatial treatment of each map for each output. A depthwise separable layer instead:

1. Applies a spatial filter to each input channel independently. This is the **depthwise** step.
2. Combines the resulting channel values at each location with learned weights. This is a **pointwise**, or $1\times1$, convolution.

The pointwise operation has no spatial reach by itself. It can still perform substantial computation because it mixes all channels at every location. “One by one” describes its spatial kernel, not its channel connectivity.

**Visual: follow two colored channel lanes.** Each lane passes through its own small spatial stencil; the lanes then enter a channel-mixing matrix. Selecting an output highlights the actual scalar terms that contribute to it. Before mixing, a spatial result depends on only its own input channel; afterward, one output can depend on both.

### A complete forward calculation

Use one spatial dimension so every multiplication fits on the page. Take two three-value channels, two-tap filters, stride one and no padding:

| Quantity | Channel A | Channel B |
| --- | --- | --- |
| Input | $[1,2,3]$ | $[2,0,1]$ |
| Depthwise filter | $[1,-1]$ | $[0.5,1]$ |
| First filtered value | $1(1)+2(-1)=-1$ | $2(0.5)+0(1)=1$ |
| Second filtered value | $2(1)+3(-1)=-1$ | $0(0.5)+1(1)=1$ |

Let the output's pointwise weights be $[2,-1]$. Its two values are both $2(-1)-1(1)=-3$. The spatial filters made local summaries; the pointwise weights decided how those summaries interact. A negative pointwise weight subtracts evidence rather than discarding a channel.

Try changing B's first input from 2 to 4. The first filtered B value becomes 2 and the first output becomes −4. The second output stays −3 because that input lies outside its patch. This is a useful locality check: not every edit should change every output.

For two-dimensional filters without an intermediate activation:

$$
z_c(i,j)=\sum_{u,v}D_c(u,v)x_c(i+u,j+v),\qquad
y_o(i,j)=b_o+\sum_c P_{o,c}z_c(i,j).
$$

$D$ holds spatial weights, $P$ holds channel-mixing weights, and $b$ is an output bias. Index offsets above suppress padding and dilation for readability; section 3 restores them. Frameworks normally implement **cross-correlation**, using weights in their stored order. Mathematical convolution reverses the kernel; learned kernels make either convention usable, but hand calculations must match the chosen one.

### How its weights learn

Training differentiates through both steps. For the first output above, target 1 and loss $L=\frac12(y-1)^2$, we have $y=-3$, $L=8$ and $\partial L/\partial y=-4$.

The mixing gradient is $(-4)[-1,1]=[4,-4]$. The gradient reaching filtered channels is $(-4)[2,-1]=[-8,4]$. Multiplying these by the input patches gives spatial gradients A $[-8,-16]$ and B $[8,0]$.

A simultaneous gradient step of size 0.01 changes the mixing weights to $[1.96,-0.96]$ and the spatial filters to $[1.08,-0.84]$, $[0.42,1]$. The new filtered values are −0.6 and 0.84; the output is −1.9824 and the loss is 4.44735488. Both sets of weights contribute to the improvement. This exact example is checked in the supplied author calculations. Larger steps need not reduce loss: a gradient describes a local direction, not a promise for every step size.

With many locations, the gradient for a shared filter sums contributions from all its uses. With nonlinear activations, their derivatives also enter this chain.

## 2. What is saved—and what is restricted?

At one output location, standard convolution uses $k^2C_{\rm in}C_{\rm out}$ multiply-accumulates, or **MACs**. One MAC multiplies two numbers and accumulates their product. Our convention counts it as one MAC; a convention counting separate floating-point operations often counts it as two FLOPs.

With one depthwise filter per input channel, the depthwise and pointwise counts are:

$$
k^2C_{\rm in}+C_{\rm in}C_{\rm out}.
$$

Multiply either expression by $H_{\rm out}W_{\rm out}$ for one image. These counts exclude bias additions, activation, normalization and memory movement. They equal the number of weights before multiplying by spatial area.

For a $14\times14$ output, 64 input channels, 128 output channels and $3\times3$ filters:

| Layer | Spatial weights | Mixing weights | Total weights | MACs/image |
| --- | ---: | ---: | ---: | ---: |
| Standard | 73,728 | Included | 73,728 | 14,450,688 |
| Depthwise + pointwise | 576 | 8,192 | 8,768 | 1,718,528 |

The separable-to-standard ratio is

$$
\frac{1}{C_{\rm out}}+\frac{1}{k^2}.
$$

Here it is about 0.119. The pointwise operation now owns most arithmetic. The ratio approaches $1/9$ for a $3\times3$ filter with very many output channels; it does not become exactly $1/9$ at some magic channel threshold. For one output channel the ratio is $1+1/9$: a separable factorization can cost **more**.

These are exact operation counts under stated shapes, not measured speed. A runtime must read tensors, schedule kernels and exploit a particular processor's parallelism. A fast dense kernel can outperform an inefficiently implemented depthwise pipeline. Measure the actual deployment shape, batch, dtype and device before making a latency claim.

### Why one filter per channel cannot represent every standard layer

Substitute the first equation into the second. The effective standard kernel is

$$W_{o,c,u,v}=P_{o,c}D_{c,u,v}.$$

Fix one input channel $c$. Every output's spatial filter is a multiple of the **same** pattern $D_c$. Standard convolution imposes no such restriction.

A tiny counterexample makes the consequence visible. For one input channel with a two-value patch $[a,b]$, suppose output 1 must equal $a$ and output 2 must equal $b$. The desired filter matrix is

$$
W_c=\begin{bmatrix}1&0\\0&1\end{bmatrix}.
$$

One shared spatial filter cannot be both $[1,0]$ and $[0,1]$ after scalar rescaling. One possible rank-one approximation keeps only the first row: on patch $[2,3]$ it produces $[2,0]$ instead of $[2,3]$. On $[2,0]$ the same approximation happens to agree exactly. Agreement on one input does not prove equivalence of layers.

**Rank** counts independent patterns needed to express a matrix. This matrix has rank two. A **depth multiplier** $m$ gives each input channel $m$ different spatial filters, followed by mixing all $mC_{\rm in}$ intermediate channels:

$$
W_{o,c,u,v}=\sum_{r=1}^{m}P_{o,c,r}D_{c,r,u,v}.
$$

For each input channel, this permits rank at most $m$. Any standard kernel can be represented when $m$ reaches the largest of these per-channel ranks, which is no more than $\min(C_{\rm out},k^2)$. Its weight count becomes $mC_{\rm in}(k^2+C_{\rm out})$; increasing expressiveness spends some or all of the savings.

**Investigation: build two independently controllable outputs.** Start with one spatial filter and two mixing weights. Try to preserve the response to patch $[2,0]$ while recovering the missing response to $[0,3]$. Then add a second spatial filter. The objective is to satisfy both probes, not to match one visible answer by chance.

### Two important distinctions

A $3\times1$ filter followed by a $1\times3$ filter factors the **spatial axes**. Depthwise followed by pointwise factors **spatial treatment and channel mixing**. They are different restrictions and may be combined.

An activation between the two operations changes the function. Even the scalar mapping $x\mapsto\max(0,x)$ cannot be replaced by a fixed linear weight for both positive and negative inputs. The effective-kernel and rank equations above describe the linear pair, not a whole nonlinear MobileNet block. [MobileNet V1](https://arxiv.org/abs/1704.04861) places normalization and ReLU after both operations; [Xception's activation experiment](https://arxiv.org/abs/1610.02357) studies a different placement and finds an intermediate activation harmful in that setting. Architecture-specific evidence does not establish a universal activation rule.

## 3. Dilation changes positions, not the number of learned taps

A three-tap filter normally samples positions $i-1,i,i+1$. At **dilation two**, it samples $i-2,i,i+2$. The filter still has three weights. Dilation increases their spacing.

With signal $[1,2,3,4,5,6,7,8,9]$, center index 4 and filter $[1,1,1]$:

| Dilation | Sampled indices | Sampled values | Sum |
| --- | --- | --- | ---: |
| 1 | 3, 4, 5 | 4, 5, 6 | 15 |
| 2 | 2, 4, 6 | 3, 5, 7 | 15 |
| 8, with zero padding | −4, 4, 12 | 0, 5, 0 | 5 |

The first two sums agree because this particular signal is an arithmetic progression and the symmetric weights balance around the center. Their sampling patterns are still different. Change index 6 from 7 to 20: dilation two's sum becomes 28, while dilation one's sum stays 15. Change index 5 from 6 to 20 instead: dilation two stays 15, while dilation one becomes 29.

**Investigation: edit the signal under a movable stencil.** Move the stencil or edit a signal value and watch the selected output, individual products and sum update together. Highlight sampled values and show padded positions separately. A large outline without sampled-site markers hides the mechanism.

For one spatial axis, input length $H$, kernel size $k$, dilation $d$, symmetric padding $p$ and stride $s$:

$$
H_{\rm out}=\left\lfloor\frac{H+2p-d(k-1)-1}{s}\right\rfloor+1.
$$

The kernel's **bounding span** is $k_{\rm eff}=1+d(k-1)$. A $3\times3$ kernel at dilation two spans a $5\times5$ box but directly reads only nine sites. At stride one, an odd kernel preserves size with $p=d(k-1)/2$. For a $3\times3$ kernel, that is simply $p=d$.

Stride moves the **output centers**; dilation spaces the **taps within each output's stencil**. Zero padding supplies values outside the input. A dilated layer can also be depthwise: one choice concerns channel connectivity, the other concerns spatial sampling.

At fixed output dimensions, changing dilation does not change the number of kernel weights or MAC terms. With no padding, larger dilation shrinks the output, so even the operation count changes. Always state the shape policy before comparing cost.

### Receptive field, jump and actual coverage

Suppose incoming features have receptive-field bounding width $r$ and neighboring centers are $j$ original-input positions apart. Adding a layer gives

$$r_{\rm new}=r+(k-1)dj,\qquad j_{\rm new}=sj.$$

Starting at $r=j=1$, two $3\times3$ stride-one layers with dilations 1 and 2 produce width $1+2+4=7$. A stride-two layer increases the jump for later layers; that is why simply adding all kernel spans fails after downsampling.

The bounding width says how far apart the extreme reachable positions are. It does not prove that every enclosed position is reachable, that learned weights are nonzero, or that a particular input has a nonzero gradient through every path. Those are different questions.

For serial stride-one three-tap layers on an unbounded line, compute exact reachable offsets by starting with $\{0\}$ and repeatedly adding each layer's set $\{-d,0,d\}$. In two dimensions with full $3\times3$ stencils, take the Cartesian product of the one-dimensional set with itself.

| Serial dilation rates | Bounding width | Reachable 1D sites | Reachable 2D sites / box |
| --- | ---: | ---: | ---: |
| 2, 2, 2 | 13 | 7 | 49 / 169 |
| 1, 2, 4 | 15 | 15 | 225 / 225 |
| 1, 4 | 11 | 9 | 81 / 121 |
| 1, 2, 5 | 17 | 17 | 289 / 289 |
| 1, 2, 9 | 25 | 21 | 441 / 625 |

Repeated even rates never connect an output to odd offsets. This is **gridding**. But “the rates have greatest common divisor one” is not enough to remove all holes: rates 1 and 4 miss offsets −2 and 2. Their gcd is one. Check the reachable set instead of replacing geometry with a slogan. The original [hybrid dilated convolution paper](https://arxiv.org/abs/1702.08502) also imposes a gap condition; a warning against shared factors is not its complete construction rule.

<details>
<summary>Deeper: construct a gap-free schedule and see its limit</summary>

Suppose all offsets from $-S$ through $S$ are reachable. A new three-tap layer of rate $d$ makes three shifted intervals centered at $-d,0,d$. They join without missing integers if $d\leq2S+1$. The new covered interval is $[-S-d,S+d]$.

Starting with $S=0$ forces the first rate to be 1 for this construction. Choosing the largest permitted rate each time gives rates $1,3,9,27,\ldots$ and bounding widths $3,9,27,81,\ldots$. This is the same place-value idea as representing offsets with digits −1, 0 and 1 in powers of three.

This is an exact sufficient construction for the stated unbounded, stride-one setting. It is not a recommendation to use enormous rates on tiny images. On a finite map, boundary padding can consume almost all the outer taps. Nor does complete structural coverage imply equal influence or better accuracy.

</details>

## 4. Implement the operator you actually mean

In PyTorch, grouped convolution partitions input and output channels into separate groups. Both channel counts must be divisible by the group count. Depthwise convolution uses one group per input channel. With multiplier $m$, it produces $mC_{\rm in}$ channels; a following ordinary $1\times1$ layer mixes them.

This complete program runs a linear depthwise/pointwise pair, constructs its effective standard kernel, and checks equality:

```python
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(4)
torch.set_num_threads(1)
inputs, outputs, multiplier, dilation = 3, 5, 2, 2
image = torch.randn(1, inputs, 7, 7, dtype=torch.float64)
depthwise = nn.Conv2d(
    inputs, inputs * multiplier, 3, padding=dilation,
    dilation=dilation, groups=inputs, bias=False
).double()
pointwise = nn.Conv2d(inputs * multiplier, outputs, 1).double()

filtered = depthwise(image)
separated = pointwise(filtered)
spatial = depthwise.weight[:, 0].reshape(inputs, multiplier, 3, 3)
mixing = pointwise.weight[:, :, 0, 0].reshape(outputs, inputs, multiplier)
effective = torch.einsum("ocm,cmuv->ocuv", mixing, spatial)
standard = F.conv2d(image, effective, pointwise.bias,
                    padding=dilation, dilation=dilation)
print(filtered.shape, separated.shape)
print(torch.allclose(separated, standard, atol=1e-12, rtol=1e-12))
```

The intermediate shape is $1\times6\times7\times7$, the output is $1\times5\times7\times7$, and the equality check is true within floating-point tolerance. The construction sums the $m$ shared spatial patterns for each input/output channel pair; it does not train or approximate anything.

The complete [direct-loop reference and checks](author-checks.py) also implement grouping, padding, dilation and stride without calling a convolution library inside the reference. Nine float64 comparisons against PyTorch, across groups 1/2/4 and several dilation/stride settings, had maximum absolute error $5.33\times10^{-15}$. That validates these fixtures, rather than claiming every future dtype/device/input is tested. The stable [PyTorch Conv2d API reference](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html) defines the same group and output-shape conventions.

### Biases, normalization and small batches

If the depthwise operation has bias $b_D$ and the pointwise operation bias $b_P$, combining the linear pair produces bias $Pb_D+b_P$. Ignoring the first bias changes the function.

At evaluation, a BatchNorm channel uses fixed stored mean $\mu$ and variance $v$. A preceding convolution with weights $W$ and bias $b$ can absorb this affine transformation:

$$
W'=\frac{\gamma W}{\sqrt{v+\epsilon}},\qquad
b'=\beta+\frac{\gamma(b-\mu)}{\sqrt{v+\epsilon}}.
$$

This is inference folding. Training BatchNorm depends on current batch statistics, so the same fixed-folding argument does not apply. A global-pooling branch with one image and a $1\times1$ map has only one value per channel; training-mode BatchNorm cannot estimate its usual channel variance from that singleton. Use an appropriate trained inference state or deliberately choose a different normalization design; changing only gradient tracking does not change module mode.

## 5. Put the two mechanisms into useful architectures

### Mobile networks: spend channel mixing carefully

MobileNet V1 repeatedly applies depthwise spatial filtering followed by pointwise mixing. Reducing every internal channel width by a factor $\alpha$ reduces the depthwise term linearly and the pointwise term quadratically:

$$
\text{MACs}\approx H_{\rm out}W_{\rm out}
\left(k^2\alpha C_{\rm in}+\alpha^2C_{\rm in}C_{\rm out}\right).
$$

Reducing both spatial dimensions by $\rho$ multiplies this by approximately $\rho^2$, subject to integer rounding and boundary stages. Width and resolution are different choices: narrowing removes feature capacity, whereas downsampling can remove fine spatial evidence. The broad architecture comparison in the previous lesson now has a mechanistic explanation.

[MobileNet V2](https://arxiv.org/abs/1801.04381) expands a narrow representation with a pointwise operation, performs depthwise spatial work in the expanded space, and projects back without a final clipping activation in the branch. When shapes match, a residual path connects the narrow endpoints. Expansion provides multiple nonlinear features before compression; the linear projection avoids obligatorily zeroing every negative projected value.

For equal input/output width $C$, expansion factor $t$, stride one and a $k\times k$ spatial operation, branch weights before biases/normalization are

$$tC^2+k^2tC+tC^2.$$

Expansion is not free. At $C=32,t=6,k=3$ there are 14,016 weights and 2,747,136 MACs on a $14\times14$ map. If the depthwise operation downsamples to $7\times7$, expansion still runs on $14\times14$, while spatial filtering and projection run on $7\times7$; the count becomes 1,589,952. Multiplying every operation by the smaller area would undercount the block.

MobileNet V3 adds design choices including channel gates, architecture search and hard-swish. Its gate approximation $\operatorname{clip}(x+3,0,6)/6$ is piecewise linear; multiplying by $x$ gives **hard-swish**, which is piecewise quadratic in the middle:

$$
\operatorname{hswish}(x)=
\begin{cases}
0 & x\leq-3,\\
x(x+3)/6 & -3<x<3,\\
x & x\geq3.
\end{cases}
$$

For example, hard-swish(−2)=−2/3. It preserves some negative responses, unlike ReLU. Deployment benefit depends on an implementation and processor, not merely on avoiding a sigmoid. The exact design and measured platform belong to the [MobileNet V3 paper](https://arxiv.org/abs/1905.02244), not a universal speed rule.

### Segmentation: ask for context at every location

Classification returns one label per image. **Semantic segmentation** assigns a class to each pixel. A patch may resemble road or roof locally; its surroundings help distinguish them. But repeated downsampling can erase a thin pole before the final prediction.

Dilation lets later layers sample more distant context while keeping their output grid dense. **Output stride** is the spacing between output-feature centers in original-image coordinates. If it is 16, a feature-map dilation of 6 spaces adjacent taps $6\times16=96$ input pixels apart. If output stride changes to 8, dilation 12 keeps that spacing. The complete receptive-field size still depends on the backbone's incoming field; tap spacing alone does not specify it.

DeepLab V3's **atrous spatial pyramid pooling**, or ASPP, processes one map through parallel branches: a pointwise branch, several dilated $3\times3$ branches, and an image-summary branch. Concatenation preserves their separate channels; a learned pointwise projection combines them. The original V3 setting uses rates 6/12/18 at output stride 16, with doubled rates at output stride 8. These are **parallel** branches, so applying the serial gridding formula to the list 6/12/18 is a category error. [DeepLab V3](https://arxiv.org/abs/1706.05587)

**Visual: branch a single map into several stencils.** Follow one common output center through each branch's actual sample sites and the global mean's whole-map dependency. The branch results become different channels at the same location. Which branches can react when a distant cell changes?

The image-summary branch averages each channel over space, projects the vector and broadcasts/upsamples it back. It brings global context, but discards where that context occurred. Two spatial arrangements with the same channel means give the same image-summary vector.

Large dilation can also fail on small feature maps. On an $8\times8$ map, a $3\times3$ kernel at dilation 8 has only its center tap inside the map at every output position. Most nominal taps multiply padding. This is still a learned operation, but it has lost the intended distant-image context. DeepLab V3 discusses this boundary effect; its global branch is one response.

DeepLab V3+ adds a decoder that combines coarse semantic features with higher-resolution features and refines boundaries. Its use of atrous separable convolutions combines our two independent choices. The decoder does not magically reconstruct all information lost during downsampling; its skip features provide additional spatial evidence. [DeepLab V3+](https://arxiv.org/abs/1802.02611)

These ideas also transfer to other domains. In an audio feature map, an axis may represent time and another frequency; dilation rates should reflect the corresponding units. For real-time sequence prediction a centered temporal stencil reads future samples. A causal version uses only current and earlier positions, with left padding; the recurrent and sequence lessons will develop that decision further. A large mathematical field is useful only when the sampled information is available and relevant.

For a complete building-block program, run [context-blocks.py](context-blocks.py) beside the supplied CSV. It defines an inverted residual block and an ASPP-style parallel context module, processes two real digit images through randomly initialized layers, and differentiates a scalar probe back to the stem. The mobile output is $2\times8\times8\times8$ and the context output $2\times4\times8\times8$. This demonstrates shape correspondence and gradient connectivity, not fitted segmentation accuracy. It uses rates 1 and 2 on the small map, four channels per context branch, and a projection input width derived from the actual number of branches. Two images keep the global branch's training-mode BatchNorm from receiving a singleton channel sample. The full segmentation system would additionally need an appropriate backbone, pixel labels, a supervised pixel loss and evaluation.

## 6. A real experiment: compress a trained layer, then inspect the damage

Can a trained standard convolution be replaced by a cheaper linear depthwise/pointwise pair without retraining? This is a concrete model-compression question, different from training a new mobile architecture from scratch.

We use 400 real optical digit images, 40 per label, from the UCI dataset distributed with scikit-learn. Each image is $8\times8$ with intensities 0–16; divide by 16. The [attributed offline CSV](digits-400.csv) and [provenance](data-provenance.md) make the experiment reproducible without a download. All 400 pixel vectors and source IDs were checked for exact duplicates before splitting. Writer IDs are unavailable, so this does not establish recognition of independent writers.

The stratified split uses 280 training and 120 development images, seed 22. It is a teaching split inside a selected subset, not the original benchmark protocol and not an untouched final test. We inspect all candidates on development; a deployment claim would need a subsequent frozen evaluation.

The model is deliberately small:

$$
1\times8\times8
\xrightarrow{\;3\times3,\ 8;\ \mathrm{ReLU}\;}
8\times8\times8
\xrightarrow{\;3\times3,\ 12;\ \mathrm{ReLU}\;}
12\times8\times8
\xrightarrow{\;\mathrm{average\ pool}\ 2\;}
12\times4\times4
\xrightarrow{\;\mathrm{flatten,\ linear}\;}
10\ \text{logits}.
$$

We train it separately with dilation 1 or 2 in the second convolution, always padding by the dilation to preserve shape. Within each of seeds 1/2/3, weights initially match across the two models. Both have 2,886 parameters and 61,824 convolution/linear MACs per image. Adam, learning rate 0.003, runs 400 full-batch updates with no augmentation, dropout, normalization, weight decay or early stopping. All six final models classify all 280 training images correctly.

### Factor only the trained second layer

For each input channel, flatten its 12 output filters into a $12\times9$ matrix. **Singular value decomposition**, or SVD, writes this matrix as a sum of independent rank-one patterns, ordered by strength. Keeping the first $m$ patterns gives the smallest squared error between original and reconstructed weights among rank-at-most-$m$ matrices. This statement concerns the weights; it does not minimize classification loss or guarantee better predictions as $m$ increases.

The saved program puts the retained right singular vectors into depthwise filters and the scaled left vectors into pointwise mixing weights. It copies the old output bias to the pointwise layer, adds no activation between the factors, and retains the original ReLU after them. Everything else stays fixed. Multipliers 1, 2, 4 and 9 are declared before observing results. No compressed model is retrained.

Download [the complete training and factorization program](convolution-factorization.py) beside the CSV. In a Python environment with NumPy, scikit-learn and PyTorch:

```text
python convolution-factorization.py
```

The program contains all imports, model definitions, split, training loop, SVD conversion, evaluation and result writing. It produces [calculated-inputs.json](calculated-inputs.json), including all six runs, development predictions, exact model costs and two saved seed-one dense states. The following is the key factor construction explained in isolation; it is an excerpt from that complete program:

```python
matrix = layer.weight[:, channel].reshape(outputs, -1)
left, values, right = torch.linalg.svd(matrix, full_matrices=False)
depthwise.weight[start:start + multiplier, 0] = (
    right[:multiplier].reshape(multiplier, height, width)
)
pointwise.weight[:, start:start + multiplier, 0, 0] = (
    left[:, :multiplier] * values[:multiplier]
)
```

There is one such assignment per input channel. The intermediate channels are ordered as all retained filters for input channel 0, then channel 1, and so forth. Grouped convolution and the reshape must agree on that ordering.

### Results from the actual CPU run

Each entry below is the number correct out of the **same 120 development images**:

| Seed | Dilation | Original dense | $m=1$ | $m=2$ | $m=4$ | $m=9$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 117 | 112 | 115 | 117 | 117 |
| 1 | 2 | 115 | 95 | 110 | 113 | 115 |
| 2 | 1 | 116 | 90 | 112 | 116 | 116 |
| 2 | 2 | 116 | 56 | 96 | 112 | 116 |
| 3 | 1 | 115 | 107 | 114 | 115 | 115 |
| 3 | 2 | 115 | 98 | 106 | 114 | 115 |

| Whole-model form | Parameters | Convolution/linear MACs/image |
| --- | ---: | ---: |
| Original dense | 2,886 | 61,824 |
| $m=1$ | 2,190 | 17,280 |
| $m=2$ | 2,358 | 28,032 |
| $m=4$ | 2,694 | 49,536 |
| $m=9$ | 3,534 | 103,296 |

Rank one can make a drastic difference: seed 2 with dilation two drops from 116 to 56 correct. That is evidence against treating arbitrary depthwise replacement as function-preserving compression. It is not evidence that trained depthwise networks are inherently poor classifiers; we did not train these compressed networks for their new constraint.

Rank nine recovers every per-channel $12\times9$ matrix, up to numerical error, and therefore recovers predictions. Its maximum absolute logit difference across all six runs is about $1.72\times10^{-5}$ in this float32 execution. It also costs more than the original layer. Exact reconstruction and worthwhile compression are distinct goals.

Even equal accuracy can hide different confidence. Seed 1, dilation one has development cross-entropy 0.128695 before compression and 0.137114 at rank four, although both have 117 correct. Cross-entropy penalizes assigning low probability to the actual label; the count correct only records the largest logit. Inspect both.

**Investigation: inspect a compressed prediction.** Compare the saved seed-one model with its rank-one replacement on a real selected image, then inspect a digit where their predictions differ. Source 299 is a digit 1 selected for the dilation-one comparison; source 32 is a digit 9 selected for dilation two. These disagreement examples are diagnostic selections, not random representatives. A per-image logit view and filter reconstruction show what changed; a success/failure counter alone cannot explain it.

A sensible next experiment would fine-tune the compressed model using training data only, predeclare its update budget, then use development to assess the trade-off. Another would compare training the separable architecture from scratch. Neither experiment was run here, so the page does not supply imagined recovery curves. The six fits show a bounded mechanism and its variability, not a hardware benchmark or a general ranking of dilation rates.

## Reuse the spatial operator and own the factorization

There are two different implementation tasks here. The preceding [convolution packet, `direct_conv2d`](../convolution-pooling-receptive-fields/convolution-experiments.py) owns address calculation for grouped/dilated filtering; its newly prepared [pullback program](../convolution-pooling-receptive-fields/convolution_pullbacks.py) opens the batched dense derivatives. Those improved pages are still prepared, so the source links are the exact reuse contract, not a claim that their new website versions are already published.

This topic owns the new decomposition. In [convolution-factorization.py](convolution-factorization.py), `factor_spatial` takes an existing kernel, performs one truncated SVD per input channel, writes the depthwise and pointwise weights, and preserves the output bias. `reconstructed_weight` independently contracts the factors back into the dense tensor. The `nn.Conv2d(groups=in_channels)` route is the normal tool implementation, with intermediate channel order fixed as `input_channel * multiplier + component`. The script compares reconstructed weights, outputs and the unchanged full model before interpreting prediction changes. `context-blocks.py` separately composes the inverted residual and parallel-context mechanisms already explained above.

For I inputs, O outputs and K spatial taps, factoring all full O×K matrices costs O(I·min(O,K)²·max(O,K)) for a dense SVD; storing factors costs O(Im(K+O)). The multiplication benefit depends on m; raising m to full rank can remove approximation error while costing more than the original layer. The SVD itself reuses the published [Matrix Decompositions owner](/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu), rather than making this lesson secretly responsible for a numerical factorization library.

**Take control:** add a relative discarded-energy report before replacing a layer. For each input matrix, divide the sum of squares of discarded singular values by the sum of squares of all singular values; define the all-zero matrix's relative error as zero. Select the smallest per-input m meeting an energy budget, then decide whether to pad all groups to a common m or implement heterogeneous groups separately. A fixed grouped Conv2d expects equal intermediate multiplicity, so silently assigning variable ranks to its regular tensor layout is wrong.

<details><summary>Hint</summary>The singular values already exist during factorization; use their squares, not their sum, for Frobenius reconstruction energy.</details>

<details><summary>Solution and success criteria</summary>For singular values [4,3,0], rank1 leaves9/25=0.36 relative squared error; rank2 leaves zero. A0.1 budget therefore needs rank2. If two input channels need ranks1 and2, a regular multiplier2 representation keeps the second slot of the first channel zero, or a custom grouped implementation must explicitly carry different slices. Verify reconstructed-weight error against the spectral sum, preserve bias/dilation/padding, and measure task outcomes separately from weight energy.</details>

## 7. Diagnose before adding another architectural feature

If a separable replacement changes predictions, first check whether it was intended to be an exact algebraic rewrite or an approximation. Confirm intermediate activations, bias transfer and channel ordering. Then inspect rank error, calibration/error slices and the opportunity for training adaptation.

If a dilation change seems ineffective, inspect actual sampled positions on the actual map. A linear ramp can hide the difference; oversized rates can sample mostly padding; repeated rates can leave unreachable offsets. The receptive-field outline alone cannot distinguish these cases.

If inference is slow despite fewer MACs, profile the deployed graph. Separate measured elapsed time from operation estimates, and include normalization, activation, memory transfers and kernel overhead. Warm up the actual implementation; synchronize asynchronous devices before timing; report shape, batch, dtype, device and software. This lesson supplies no latency ranking.

If grouping fails, inspect divisibility. A 96-input, 128-output layer with 32 groups is valid: each group has 3 inputs and 4 outputs. Changing 128 outputs to 130 is invalid. Rounding widths to multiples of 8 for a hardware convention does not automatically make them divisible by every chosen group count.

If RGB handling worries you, remember that pointwise mixing can combine color-channel responses. The real restriction is the spatial/channel factorization, not an inability to use colors. Whether that restriction is appropriate in an input stem is an empirical design question, not a universal prohibition.

## 8. Practice on changed problems

### 1. Follow a changed channel through the layer

Use input channels A $[2,1,0]$, B $[1,3,2]$, spatial filters A $[1,2]$, B $[-1,1]$, and pointwise weights $[0.5,2]$. Find both outputs. Then change A's final value from 0 to 4: which output changes, and by how much?

<details>
<summary>Hint</summary>

Compute the two valid two-value patches in each channel before mixing. The final A value participates only in the second patch.

</details>

<details>
<summary>Solution</summary>

A filters to $[4,1]$ and B to $[2,-1]$. Outputs are $[6,-1.5]$. Editing A's last value changes its second filtered value from 1 to 9, so the second output increases by $0.5(8)=4$ to 2.5. The first remains 6.

</details>

### 2. Budget expressiveness

A $3\times3$ layer has 16 inputs and 24 outputs. Compare a dense layer with a separable pair using multiplier two. Ignore biases. Does a smaller parameter count prove equivalence or faster inference?

<details>
<summary>Hint</summary>

The multiplier repeats spatial filters and increases the input width of the pointwise layer. Count both.

</details>

<details>
<summary>Solution</summary>

Dense: $9(16)(24)=3456$ weights. Separable: $16(2)(9+24)=1056$. The effective per-input-channel rank is at most two, so an arbitrary dense layer need not be representable. Hardware execution and all other operations determine elapsed time.

</details>

### 3. Repair a misleading receptive field

Two three-tap layers use rates 1 and 4. List reachable offsets, identify the gaps, and choose a replacement second rate that reaches every offset in a seven-position span.

<details>
<summary>Hint</summary>

Shift the first layer's set $\{-1,0,1\}$ left, nowhere and right by the second rate.

</details>

<details>
<summary>Solution</summary>

Rates 1 and 4 reach $\{-5,-4,-3,-1,0,1,3,4,5\}$ and miss −2,2 inside their eleven-position bound. Second rate 2 produces every offset −3 through 3, a seven-position span. The repaired field is smaller but fully covered.

</details>

### 4. Respect original-image units

Incoming features have receptive-field width 7 and jump 2 input pixels. Add a kernel of size 3, dilation 3 and stride 2. Find the new field width and jump. Explain why $2(2\cdot3+1)$ is not the field width.

<details>
<summary>Hint</summary>

The new extreme taps enlarge an existing field; each incoming feature already summarizes several original pixels.

</details>

<details>
<summary>Solution</summary>

$r'=7+(3-1)(3)(2)=19$, and $j'=2(2)=4$. Multiplying a seven-tap-span outline by the old jump gives 14, which ignores the incoming field width and conflates center spacing with the coverage of one feature.

</details>

### 5. Design the next compression experiment

A rank-two replacement has worse development accuracy than the original, but fewer MACs. Propose an experiment that answers whether training adaptation can recover useful accuracy. Identify which data may drive updates and what evidence is still needed before a latency claim.

<details>
<summary>Hint</summary>

Fix what stays constant, predeclare what training changes, and separate the role of training examples from development examples.

</details>

<details>
<summary>Solution</summary>

Start from the same saved dense model and the declared rank-two conversion. Predeclare an optimizer, update budget and which layers may change. Fine-tune only on training images; retain the original and unadapted compressed models as baselines. Compare development counts, cross-entropy and relevant slices after the fixed run, and report all chosen seeds. Selecting a strategy on development consumes that evidence; later reporting needs a frozen evaluation. MAC reduction still requires a measured deployment timing comparison with documented device, software and input contract.

</details>

### 6. A spatial summary can miss a rearrangement

An ASPP image-summary branch averages a channel whose $2\times2$ values are $[1,3;5,7]$. Swap the top-left and bottom-right values. Can this branch alone identify the swap? Could a local branch respond differently?

<details>
<summary>Hint</summary>

The global branch sees a channel mean, while a local stencil sees values at particular offsets.

</details>

<details>
<summary>Solution</summary>

The mean remains 4, so the pooled vector and its deterministic projection remain identical. A local branch may change because the values at its sampled positions changed. A particular symmetric filter or location could still give the same result; dependency permits a change but does not guarantee one.

</details>

### 7. An advanced construction challenge

Using four serial three-tap, stride-one layers on an unbounded line, construct a gap-free schedule with the largest possible number of distinct reachable positions. State why the result is a structural upper bound, and why it may be unsuitable for an $8\times8$ feature map.

<details>
<summary>Hint</summary>

Each layer offers three tap choices per path. Use the interval construction to make those paths reach distinct offsets.

</details>

<details>
<summary>Solution</summary>

There are $3^4=81$ tap-choice paths, so at most 81 distinct positions can be reached. Rates 1,3,9,27 reach every integer from −40 to 40, attaining that bound. Finite maps invalidate the unbounded-input premise: many taps address padding, and the nominal span does not create new image information.

</details>

## 9. Readiness and the next design question

You are ready to continue when you can trace spatial filtering and channel mixing separately; state the rank restriction of a linear separable pair; calculate weights and MACs with the correct output shapes; distinguish dilation from stride; and test sampled-site coverage instead of trusting a bounding rectangle. In practice, you should also be able to identify whether an architectural change was retrained, whether its evaluation data was already used for selection, and whether a speed claim was actually measured.

Next in the module is **[ConvNeXt & Modern CNN Designs](/learn/path/full-curriculum/convnext-modern-cnn-designs?module=deep-learning-fundamentals)**. It builds on this distinction between spatial mixing and channel mixing, then changes normalization, activation placement, stage design and training. We will ask what evidence supports those changes rather than assuming a newer name explains the improvement.

### References and other ways to learn

- [MobileNets V1, Howard et al.](https://arxiv.org/abs/1704.04861): sections 3.1–3.4 explain separable filtering and width/resolution choices. Read after sections 1–2; its historical model and reported costs have a specific shape/training context.
- [Xception, Chollet](https://arxiv.org/abs/1610.02357): section 4.7 is a useful contrasting activation-placement experiment. Its conclusion is about the tested architecture.
- [MobileNet V2, Sandler et al.](https://arxiv.org/abs/1801.04381): sections 3.2–3.4 and the block table explain linear bottlenecks and inverted residuals. Its geometric motivation needs more linear algebra than the first-pass route.
- [Searching for MobileNet V3, Howard et al.](https://arxiv.org/abs/1905.02244): section 5 distinguishes the hard gate from hard-swish and documents architecture-specific deployment decisions.
- [Multi-Scale Context Aggregation by Dilated Convolutions, Yu and Koltun](https://arxiv.org/abs/1511.07122): sections 2–3 introduce the operator and a context module; later sections separate the front end and evaluation. Useful after drawing exact tap positions.
- [Understanding Convolution for Semantic Segmentation, Wang et al.](https://arxiv.org/abs/1702.08502): section 3.2 explains hybrid dilation and gap conditions. Use it with explicit support enumeration rather than reading its common-factor warning as a sufficient theorem.
- [DeepLab V3](https://arxiv.org/abs/1706.05587) and [V3+](https://arxiv.org/abs/1802.02611): respectively, parallel context branches and a boundary-refining decoder. Keep versions, output stride and training protocol distinct.
- [Dive into Deep Learning: Multiple Input and Multiple Output Channels](https://d2l.ai/chapter_convolutional-neural-networks/channels.html): an alternate visual and code route for the local channel operations and $1\times1$ mixing. Its full standard-convolution examples are especially useful before the factorization proof; the book's own environment setup differs from our self-contained files.
- [Distill: Computing Receptive Fields](https://distill.pub/2019/computing-receptive-fields/): an interactive geometry explanation with derivations for size, location and multi-path networks. Pair the outline diagrams with this lesson's explicit sampled-site sets.
- [PyTorch Conv2d reference](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html) and [Torchvision ASPP source](https://docs.pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html): API and implementation references, not substitutes for the mechanism. The former is a stable 2.9 documentation page; the latter is a changing main-branch view inspected on 13 September 2026. Our saved calculations ran with PyTorch 2.14.0+cpu.
