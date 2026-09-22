# Convolution, Pooling & Receptive Fields

**Explore as you read.** Edit image/kernel cells, stride/dilation/padding, pooling inputs, shared-weight targets/rate and receptive-field threshold. Synchronize the selected patch, products, output map, transpose contributions, gradient accumulation and ancestry paths. Geometry edits visibly change output size, alignment and holes rather than only a summary label. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose window geometry and pooling from reach, alignment, information loss and update behavior.


A handwritten 7 can move a little to the right and still be a 7. Its strokes are local, but their arrangement matters: a short horizontal stroke and a long diagonal belong together. How can a neural network use those facts instead of learning an unrelated detector for every possible pixel position?

Convolution reuses a small calculation across a grid. Pooling summarizes nearby values. A receptive field tells us where one result can obtain information. Together, these ideas let us reason about an image model as a sequence of visible operations.

**First pass:** follow one patch through a filter, learn how that filter changes, plan the sizes, and train the small digit classifier in sections 1–7. You should then be able to build and diagnose a basic CNN. Sections 8–10 are optional deeper routes into influence, reconstruction and efficient execution. The practice separates core readiness from those extensions.

You need multiplication, sums and array indexing. We refresh neural terminology as it appears: a parameter is a learned number, an activation is a computed value, and a gradient says how a small change affects a chosen result. Our task is to predict one of ten digit labels from an 8×8 intensity image. A dense network is the baseline; convolution is a different useful constraint, not a promise to beat it.

## 1. A small rule that travels

Consider this image and a 2×2 filter:

| Image | Filter |
| --- | --- |
| 1, 2, 0<br>0, 1, 3<br>2, 1, 0 | 1, −1<br>0, 1 |

Put the filter over the top-left patch. Multiply corresponding cells and add:

\[
1(1)+2(-1)+0(0)+1(1)=0.
\]

Move it one column right, keeping the **same four weights**:

\[
2(1)+0(-1)+1(0)+3(1)=5.
\]

Repeating for the second row produces \(\begin{bmatrix}0&5\\0&-2\end{bmatrix}\). This output grid is a **feature map**. A positive value means the weighted pattern has a positive response there. It is not yet a probability or necessarily a meaningful named feature.

[Visual placement: patch-and-products investigation. Move the outlined window; each image cell, filter weight and product retains its correspondence. The selected output and its contributing sum update together. Then edit a pixel and identify exactly which outputs change.]

This is the operation deep-learning libraries usually call convolution, although the precise mathematical operation is **cross-correlation**: the filter is used in the displayed orientation. Mathematical convolution reverses its spatial axes. For learned filters either convention can represent the same family of operations, but matching a fixed filter or an implementation requires knowing the convention. [PyTorch Conv2d contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html).

A filter \([1,0,-1]\) on a one-dimensional signal compares an earlier and a later value. A constant signal gives zero away from padding; an abrupt change gives a response. On an image, a similar pattern can respond to an edge. We do not need to hand-design every detector: training adjusts the weights to reduce the task's errors.

Why reuse them? A dense layer connecting 64 input pixels to 512 outputs has 32,768 weights before biases. Eight 3×3 filters on a one-channel image have 72 weights, yet produce eight whole maps. The local pattern detector shares evidence across positions. Dense layers can learn correlations too; they simply do not impose these particular locality and weight-sharing constraints. Positions in overlapping patches are also correlated: they are not hundreds of independent extra training specimens.

## 2. Channels are several measurements at each location

An RGB image has three channels. A later hidden representation might have eight channels corresponding to eight learned responses. A normal convolution combines a patch from **every input channel** to make one output channel.

For one output at location \((u,v)\), with stride one and no padding:

\[
y_{o,u,v}=b_o+\sum_{c=0}^{C_{\rm in}-1}
\sum_{a=0}^{k_h-1}\sum_{b=0}^{k_w-1}
W_{o,c,a,b}\,x_{c,u+a,v+b}.
\]

Here \(o\) selects an output channel, \(c\) selects an input channel, and \(a,b\) select a location inside its patch. The bias \(b_o\) is added once after combining the channels. Its subscript distinguishes the bias from the column index \(b\).

Suppose two single-cell channels contain 2 and 3. An output channel with weights 4 and −1, bias 1, gives \(4(2)-3+1=6\). Another output channel can apply different weights to the same inputs. A 1×1 convolution therefore mixes channels even though it does not combine neighboring spatial positions.

[Visual placement: channel stack with separate partial sums feeding one output cell. Construct two output filters for the sum and difference, then test them on changed inputs and a zero-input bias check. A separate isolation example shows what happens when one input channel has zero weight.]

For a batch, PyTorch uses logical shape \([N,C,H,W]\): specimens, channels, height, width. A weight tensor has shape \([C_{\rm out},C_{\rm in},k_h,k_w]\) when groups=1. It has no separate batch or output-location axis because those uses share its values.

**Grouped convolution** restricts which input channels connect to which outputs. With two groups, each half of the outputs sees only its corresponding half of the inputs; both channel counts must be divisible by two. A depthwise convolution uses one group per input channel, optionally producing several outputs per input channel. The multiplier need not be one. A later pointwise layer can mix those separate channels. We will study this design properly after the landmark architectures.

## 3. How a shared filter learns

Before fitting an image classifier, use a one-dimensional example small enough to calculate completely. Input \(x=[1,3,2]\), filter \(w=[1,-1]\), no bias, produces \(y=[-2,1]\). Suppose the desired output is \([0,0]\), and use half the **sum** of squared errors:

\[
L=\tfrac12((-2)^2+1^2)=2.5.
\]

The output gradients are \([-2,1]\). The first weight contributes to both windows, so its gradient adds both contributions:

\[
\frac{\partial L}{\partial w_0}=(-2)(1)+(1)(3)=1,\qquad
\frac{\partial L}{\partial w_1}=(-2)(3)+(1)(2)=-4.
\]

Gradient descent with step size 0.1 gives \(w'=[0.9,-0.6]\). New outputs are \([-0.9,1.5]\), and new loss is 1.53. One output became worse, but the specified total objective improved. Sharing a filter means negotiating all its uses rather than independently fixing each location.

Gradients also accumulate where windows overlap. The middle input participates with weight −1 in the first window and weight 1 in the second:

\[
\frac{\partial L}{\partial x_1}=(-2)(-1)+(1)(1)=3.
\]

The full input gradient is \([-2,3,-1]\). If we had used mean squared error, its denominator would scale the gradients. Weight sharing itself asks us to **sum** contributions; averaging comes from the chosen loss reduction.

[Visual placement: two windows feeding a shared parameter rail. Show both contributions, the current one-step update, new outputs and loss together. Changing the second target recomputes the actual arithmetic, and Step highlights each contribution without hiding the current result.]

This complete program runs the calculation:

```python
import torch
from torch.nn import functional as F

x = torch.tensor([1., 3., 2.], dtype=torch.float64, requires_grad=True)
w = torch.tensor([1., -1.], dtype=torch.float64, requires_grad=True)
y = F.conv1d(x[None, None], w[None, None]).flatten()
loss = 0.5 * y.square().sum()
loss.backward()
new_w = w.detach() - 0.1 * w.grad
new_y = F.conv1d(x.detach()[None, None], new_w[None, None]).flatten()
print("output:", y.detach().tolist(), "loss:", loss.item())
print("weight gradient:", w.grad.tolist(), "input gradient:", x.grad.tolist())
print("updated:", new_w.tolist(), "new loss:", (0.5 * new_y.square().sum()).item())
```

For classification, hidden convolutions feed activations such as ReLU, \(a=\max(0,z)\), and a final layer produces one logit per class. Cross-entropy compares those logits with the actual digit. Backpropagation performs the same accumulation through the larger graph. There is no special second optimizer for filters.

## 4. Plan the geometry before stacking layers

**Stride** is the movement between output windows. **Padding** supplies values outside the image boundary. **Dilation** spaces the sampled positions inside a filter. A three-weight filter with dilation two touches offsets 0, 2 and 4: three learned values spanning five positions.

For one dimension, let input size be \(n\), kernel size \(k\), dilation \(d\), stride \(s\), and left/right padding \(p_l,p_r\). The sampled span is \(k_{\rm eff}=d(k-1)+1\). Count the window starts that fit:

\[
n_{\rm out}=\left\lfloor\frac{n+p_l+p_r-k_{\rm eff}}{s}\right\rfloor+1.
\]

Apply this independently to height and width. The floor means a leftover strip can be unused. A nonpositive result means the requested operation does not fit; it is not a valid zero-sized feature map.

| Input | Kernel / dilation | Padding left,right | Stride | Output |
| --- | --- | --- | --- | --- |
| 8 | 3 / 1 | 1,1 | 1 | 8 |
| 8 | 3 / 1 | 1,1 | 2 | 4 |
| 8 | 4 / 1 | 1,2 | 1 | 8 |
| 8 | 3 / 2 | 2,2 | 1 | 8 |
| 7 | 3 / 1 | 0,0 | 2 | 3 |

[Visual placement: geometry ruler showing actual sampled dots, padding cells, skipped starts and unused remainder. Solve an output-size target by editing the layer parameters; matching dimensions alone should not hide a changed center alignment.]

“Same” describes an output-size policy, not a universal padding number. For stride one, total required padding is \(d(k-1)\). With an even effective kernel this may need unequal sides. A four-wide kernel can use left 1/right 2 or left 2/right 1; both preserve width but align outputs differently. In PyTorch 2.14, `padding="same"` supports stride one. Explicit `F.pad` handles a chosen asymmetric policy.

Two other names describe useful boundary choices. With stride and dilation one, **valid** uses no padding and produces \(n-k+1\) outputs. **Full** uses \(k-1\) padding on each side and produces \(n+k-1\), including windows with only a partial overlap with the original input. Both follow the same window-count formula.

Zero padding assumes an outside value of zero. Reflection, replication and circular padding impose different boundary conditions; choose them based on what the data means. A periodic signal can justify wrapping. An ordinary photograph does not automatically continue from its right edge to its left.

Two stride-one 3×3 layers have a five-wide possible input region; three have seven, provided dilation is one. Intermediate nonlinearities make this a different function family from a single wider linear filter. Fewer parameters or greater expressivity depends on channel counts and the precise comparison, not just the number of layers.

## 5. Pooling: summarize, then notice what disappeared

A pooling operation normally works independently in each channel. For a 2×2 window \(\begin{bmatrix}1&4\\2&3\end{bmatrix}\), max pooling gives 4; average pooling gives 2.5. With stride two, the next window starts two cells away.

Max pooling asks for the strongest response in the window. Average pooling asks for its mean level. Neither is universally better, and either loses information: many different windows share the same maximum or mean. A decoder may learn a plausible reconstruction using other evidence, but that does not make pooling invertible.

[Visual placement: movable pooling windows with winner routes versus distributed average routes. Move a bright response within a window, then across its boundary. Solve a reconstruction ambiguity by drawing two different patches with the same pooled result.]

For a sum of max-pool outputs on \([1,4,3]\), kernel two and stride one, the outputs are \([4,4]\). The input gradient is \([0,2,0]\): the middle value wins both windows. At a tie the mathematical maximum has several valid subgradients; the implementation chooses an index. Our CPU fixture on \([2,2,1]\) yields gradient \([1,1,0]\). Do not build an argument that depends on a universal tie winner across all backends.

Padding needs care with negative values. Max-pool padding behaves like negative infinity, so an invented zero cannot beat a real negative input. Average pooling can include or exclude padded zeros from the denominator. With \([-2,-3]\), kernel three and one padded cell on each side, the two averages are both \(-5/3\) when padding counts, versus \(-5/2\) when it does not. These policies change numbers despite matching output shapes. [MaxPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html), [AvgPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AvgPool2d.html).

Adaptive average pooling specifies an output size instead of one fixed stride. For input length five and output length three, its bins are indices \([0,2)\), \([1,4)\), \([3,5)\). On \([1,2,3,4,5]\), this gives \([1.5,3,4.5]\). Bins can overlap; “adaptive” does not mean an exact equal disjoint partition for every size. [PyTorch's bin-boundary implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/AdaptivePooling.h).

**Global average pooling** averages an entire spatial map into one value per channel. It allows a classifier head to receive the same number of features at different spatial sizes. It discards location in that final map, which can be useful or harmful depending on the task. It does not guarantee that the earlier map is unaffected by cropping or shifting.

This connects to dropout: dropping responses before max pooling changes which value wins, and a sampled zero can exceed a negative activation. That is a particular noisy objective, not an algebraically forbidden ordering. Our comparison below omits dropout so we can inspect the pooling choice clearly.

## 6. Receptive fields: where can this number get information?

A first-layer 3×3 output sees a 3×3 patch. A later output sees several earlier outputs, each with its own input region. The receptive field is built by tracing those connections backwards.

Track three values per spatial axis:

- \(r\): width of the theoretical bounding region in input coordinates.
- \(j\): spacing between adjacent output centers, measured in input pixels.
- \(a\): center of the first output, with input pixel centers at \(0.5,1.5,\ldots\).

Start with \(r=1,j=1,a=0.5\). A layer with \(k,d,s,p_l\) changes them by:

\[
r'=r+d(k-1)j,\qquad j'=sj,\qquad
a'=a+\left(\frac{d(k-1)}2-p_l\right)j.
\]

The old \(j\) belongs on the right side of all three equations. A pooling window expands the region too.

For a 32-wide input:

| Operation | Output width | \(r\) | \(j\) | First center \(a\) |
| --- | --- | --- | --- | --- |
| 3-wide convolution, pad 1 | 32 | 3 | 1 | 0.5 |
| 2-wide pooling, stride 2 | 16 | 4 | 2 | 1 |
| 3-wide convolution, pad 1 | 16 | 8 | 2 | 1 |
| 2-wide pooling, stride 2 | 8 | 10 | 4 | 2 |
| 3-wide convolution, pad 1 | 8 | 18 | 4 | 2 |
| Average over all 8 positions | 1 | 46 | 4 | 16 |

A width of 46 on a 32-wide input includes padded coordinates; it does not mean 46 observed pixels or wraparound. For output index \(u\), center is \(a+uj\); its bounding endpoints are that center plus/minus \((r-1)/2\). Intersect with actual input coordinates to distinguish observed values from padding.

[Visual placement: stacked input-coordinate rulers. Select an output and trace its exact sampled ancestors; show the bounding box separately from holes and clipped padding. A second branch can be added only after its output shape is compatible, and its centers remain visible.]

The region can have holes. Two three-wide, dilation-two layers reach offsets \(\{-4,-2,0,2,4\}\): bounding width nine, only five positions. Using dilation one followed by two reaches all seven offsets from −3 to 3. A bounding width is not a count of connected pixels.

For residual additions or concatenated branches, take the union of contributing input regions. Equal tensor shapes do not prove equal spatial alignment. Branches with different center offsets may add features referring to different image locations. The [receptive-field coordinate derivation](https://distill.pub/2019/computing-receptive-fields/) gives a useful deeper treatment of alignment.

## 7. Build and inspect a real digit classifier

Our complete [CPU experiment](convolution-experiments.py) includes the direct NumPy operation, tensor fixtures, twelve training runs and saved intermediate maps. Keep [the attributed CSV](digits-400.csv) next to it. Setup from that folder:

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.3.5 scikit-learn==1.9.1 torch==2.14.0
.\.venv\Scripts\python.exe convolution-experiments.py
```

On macOS/Linux, use `python3 -m venv .venv`, then replace each `.\.venv\Scripts\python.exe` above with `.venv/bin/python`. Calling the environment's interpreter directly avoids depending on shell activation.

It needs no GPU, network-loaded model or unspecified image folder. Installation needs network access; after installation the data and experiment run offline. The author executed it on Python 3.12.14, PyTorch 2.14.0+cpu, one CPU thread.

The CSV contains 400 actual UCI optical-digit images: first 40 examples of each digit from sklearn's 1,797-row copy of the historical UCI test partition. Values are integers from 0 to 16, divided by 16 using the documented measurement range. A fixed stratified split, seed 22, uses 280 rows to fit and 120 for development comparisons. This is a small educational split, not the official dataset evaluation or evidence of performance on unseen writers. [Data provenance](data-provenance.md).

The main CNN is:

```text
[N,1,8,8]
→ Conv 1→8, 3×3, pad 1 → ReLU       [N,8,8,8]
→ Pool 2×2, stride 2                [N,8,4,4]
→ Conv 8→16, 3×3, pad 1 → ReLU      [N,16,4,4]
→ Pool 2×2, stride 2                [N,16,2,2]
→ Flatten 64 values → Linear 64→10  [N,10] logits
```

Convolutions provide local weighted sums, ReLU makes the composition nonlinear, pooling reduces spatial size, and the final layer combines the remaining features into class scores. The second convolution's weights have shape \([16,8,3,3]\), not \([16,1,3,3]\).

We compare four declared configurations: dense 64→32 tanh→10; CNN with max pooling; the same CNN with average pooling; max-pool CNN whose final 2×2 map is globally averaged before a smaller head. The max and average CNNs start with identical weights for each seed. The global-average model shares the initial convolution tensors but has a differently sized head; the dense baseline is a different architecture. All use Adam, learning rate 0.003, 400 full-batch updates, no augmentation, dropout or weight decay. Seeds 1, 2 and 3 were chosen in advance.

| Model | Parameters | Seed 1: CE / correct of 120 | Seed 2 | Seed 3 |
| --- | --- | --- | --- | --- |
| Dense baseline | 2,410 | 0.07949 / 117 | 0.08647 / 118 | 0.09342 / 116 |
| CNN, max pooling | 1,898 | 0.08332 / 118 | 0.10153 / 116 | 0.05194 / 119 |
| CNN, average pooling | 1,898 | 0.10358 / 116 | 0.11126 / 118 | 0.13594 / 115 |
| CNN, global-average head | 1,418 | 0.08249 / 117 | 0.13742 / 113 | 0.17901 / 113 |

These are actual final-update measurements. Every run fits all 280 training labels. The CNN uses fewer parameters than this dense baseline; it does not win every seed or metric. Accuracy counts and cross-entropy measure different aspects: a few confident mistakes can increase CE even with similar counts. The program retains intermediate train/development traces rather than drawing an imagined smooth learning curve.

[Visual placement: real specimen → saved post-ReLU feature maps → pooled maps → class probabilities. Inspect the saved seed-1 maps for three source IDs, with numeric values available. Raw preactivations and logits were not retained and must not be invented. A separate evidence comparison uses the actual seed/step records and displays the matching measured outcomes as soon as a record is selected.]

A deliberately difficult stress check shifts each development image one pixel right or down, fills the exposed edge with zero and keeps its original label. This tests a changed input condition; it can also clip meaningful strokes, so it is not a clean proof of translation behavior on an unlimited canvas.

For seed 1, the dense model gets 57/120 right-shifted images correct, and the max-pool CNN gets 72/120. Their unshifted counts were 117 and 118. Down-shift counts are 71 and 75. Even the global-average CNN falls from 117 to 63 right-shifted. All twelve stress results are saved. Weight sharing helps organize the model, but does not remove the need to define and validate deployment variation.

**Before changing the experiment**, write the expected effect, the single change and the metric. An augmentation trial should transform training data only; assess the predeclared development views consistently. Once their outcomes guide choices, these rows remain development data. A final performance claim needs a separate untouched evaluation protocol.

## 8. Optional: reach, influence and shifts

The theoretical field says which paths exist. An input gradient \(\partial z/\partial x_{u,v}\) says how a particular scalar output changes locally around a particular input, with the current parameters. A ReLU can close a route; two contributions can cancel. A zero gradient does not erase an architectural connection.

Our exact linear example repeatedly applies the averaging filter \([1,1,1]/3\), without nonlinearities. At depth two the central output's input weights are \([1,2,3,2,1]/9\). More paths reach central positions. The support width is \(2L+1\), while the distribution's variance is \(2L/3\): it behaves like a sum of \(L\) independent offsets chosen from −1,0,1 for this mathematical construction.

[Visual placement: exact path-weight profile with theoretical support brackets. Change depth and a stated relative-to-peak threshold; separately inspect a trained digit model's signed logit gradient. Preserve the distinction between the exact linear model and the empirical nonlinear map.]

At depth 20, support width is 41; positions with at least 1% of the peak span 21. That threshold is a chosen display rule, not a universal definition of effective receptive field. Learned filters, input-dependent gates and averaging across examples change the profile. The [effective receptive field paper](https://arxiv.org/pdf/1701.04128) develops qualified Gaussian-like behavior under assumptions and empirical settings; it does not make every individual trained gradient a Gaussian. A completely closed ReLU route should display “zero gradient” rather than normalize zero into a misleading heatmap.

Now distinguish **equivariance**, where shifting input shifts the output map correspondingly, from **invariance**, where output stays identical. Stride-one correlation with circular boundaries commutes with circular shifts. Our exact fixture gives difference zero. Switching that fixture to zero-padded boundaries gives maximum difference one.

Downsampling introduces a grid phase. Sampling every second element of \([1,-1,1,-1,1,-1]\) gives \([1,1,1]\); shift the input one position first and it gives \([-1,-1,-1]\). A local two-value average before sampling gives zeros in both cases. This is a simple illustration of aliasing and low-pass filtering, not a claim that blurring makes every classifier invariant.

Max pooling can tolerate some within-window motion, yet moving a response across a window boundary moves its pooled result. Global averaging is invariant to a permutation of an already-computed map. Input shifts need not produce only a permutation of that map. [Zhang's anti-aliasing study](https://proceedings.mlr.press/v97/zhang19a.html) investigates this distinction in trained networks.

## 9. Optional: the reverse operation is a scatter, not an undo

Our one-dimensional filter can be written as a sparse matrix:

\[
C=\begin{bmatrix}1&-1&0\\0&1&-1\end{bmatrix},\qquad y=Cx.
\]

The transpose sends each output-side value back along the same connections, adding where they meet. It obeys \(\langle Cx,g\rangle=\langle x,C^\top g\rangle\). For \(x=[1,3,2]\), \(g=[2,-3]\), both sides are −7, and \(C^\top g=[2,-5,3]\).

But \(C^\top Cx=[-2,3,-1]\), not the original input. **Transposed convolution** is this structured transpose operation, not an inverse. A constant added to every element of \(x\) disappears under this difference filter, so exact recovery from its output alone is impossible.

[Visual placement: scatter-and-overlap canvas. Each input-side value paints a weighted footprint; contributions stack numerically. Compare kernel sizes three and four at stride two and distinguish interior from boundary coverage.]

Three input values equal to one, kernel \([1,1,1]\), stride two, produce coverage \([1,1,2,1,2,1,1]\). In two dimensions, uneven overlap can multiply into a checkerboard. A kernel divisible by stride can equalize interior coverage, yet learned weights can still make artifacts. Resizing followed by an ordinary convolution offers a different constraint; it is not an unconditional artifact cure. [Visual explanation of checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/).

For symmetric padding \(p\), transposed output size is
\[
n_{\rm out}=(n_{\rm in}-1)s-2p+d(k-1)+{\rm output\_padding}+1.
\]
Different input sizes can map to the same strided forward size. `output_padding` selects a compatible output size; it does not mean appending that many zero-valued output cells. [ConvTranspose2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html).

## 10. Optional: implement the same math efficiently

The retained program's `direct_conv2d` follows the nested sums explicitly, including groups, symmetric padding, stride and dilation. Five float64 cases compare it with PyTorch, including rectangular kernels and depthwise multiplier two, with maximum absolute discrepancy below \(10^{-12}\). This checks this bounded CPU arithmetic; it is not a GPU benchmark.

An alternative extracts each patch into a column. For the first image, the column matrix is
\[
P=\begin{bmatrix}
1&2&0&1\\2&0&1&3\\0&1&2&1\\1&3&1&0
\end{bmatrix}.
\]
Multiplying \([1,-1,0,1]P\) yields \([0,5,0,-2]\), then reshape to 2×2. This is often called im2col. It exposes matrix multiplication, but explicitly copying overlapping patches can use substantial memory.

Folding those columns back adds overlaps. The center input appears four times, an edge-middle twice, a corner once. `Fold(Unfold(x))` therefore equals an overlap-count grid times `x`. Dividing by that grid recovers covered positions; an uncovered position with count zero cannot be recovered this way. [Fold contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Fold.html).

Implicit matrix-multiplication algorithms generate patch addresses without materializing the whole expanded matrix. Transform methods such as FFT or Winograd change how the arithmetic is organized. Shape, datatype, hardware and workspace constraints affect the useful choice. These are implementation strategies for the operator, not new learned representations. [NVIDIA convolution algorithm guide, sections 3–4.2](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html).

For groups \(g\), a \(k_h\times k_w\) layer has
\[
C_{\rm out}(C_{\rm in}/g)k_hk_w
\]
weights, plus \(C_{\rm out}\) biases if used. Forward multiply-accumulates per specimen are that weight count times \(H_{\rm out}W_{\rm out}\). If counting one multiplication and addition separately, use two FLOPs per MAC, state that convention, and account for other operators separately.

Our first convolution has 72 weights, 80 parameters including biases, and \(8\cdot8\cdot72=4{,}608\) MACs. The second has 1,152 weights, 1,168 parameters and \(4\cdot4\cdot1{,}152=18{,}432\) MACs. The head adds 650 parameters and 640 MACs: total 1,898 parameters and 23,680 MACs for these affine operations. Pooling and activations have additional work. The dense baseline has 2,368 MACs despite more parameters. A parameter comparison is not a latency comparison.

A 3×3 depthwise-plus-pointwise pair with multiplier one uses \(9C_{\rm in}+C_{\rm in}C_{\rm out}\) weights, versus \(9C_{\rm in}C_{\rm out}\) for a dense 3×3 layer. The dense-to-separable weight ratio is \(9C_{\rm out}/(9+C_{\rm out})\), before biases. It also constrains the computation differently; fewer arithmetic operations do not guarantee proportional wall-clock savings.

Memory format is another independent choice. PyTorch channels-last storage can preserve the logical \([N,C,H,W]\) shape while changing strides in memory. It is not the same operation as permuting the tensor's logical axes. [Channels-last tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html).

An evaluation-mode convolution followed by BatchNorm with fixed running statistics can be combined algebraically. For output channel \(o\), set
\[
\alpha_o=\frac{\gamma_o}{\sqrt{v_o+\epsilon}},\qquad
W'_o=\alpha_o W_o,\qquad b'_o=\beta_o+\alpha_o(b_o-\mu_o).
\]
Our float64 fixture matches within \(3.4\times10^{-16}\). Training BatchNorm depends on the current batch, so that fixed folding argument does not apply. ReLU remains nonlinear even if a backend executes it in the same kernel. Memory layout changes, algebraic folding and actual kernel fusion should be evaluated separately when performance becomes the task.

One useful connection goes beyond recognizing images. A fixed grid stencil
\(\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}\)
computes a discrete Laplacian numerator. With center temperature 30 and four neighbors at 20, it gives −40: local curvature toward cooler surroundings. For grid spacing \(h\), divide by \(h^2\); a diffusion equation also needs diffusivity, a time discretization and boundary conditions. This uses the same local weighted-sum mechanism, but the weights represent a specified numerical operator rather than parameters learned from labels.

## Implement the pullback and batch the arithmetic

The direct `direct_conv2d` routine in [convolution-experiments.py](convolution-experiments.py) is a transparent indexing oracle: every output, input group and tap is visible. Its scalar Python loops are not the final recommendation for processing images. The companion [convolution_pullbacks.py](convolution_pullbacks.py) retains the spatial-tap loops but performs all examples, channels and output locations together using `einsum`. This opens the operation without allocating a full expanded patch matrix.

For valid dense NCHW cross-correlation, one tap contributes `images[:, :, row:row+OH, col:col+OW]` contracted with `weights[:, :, row, col]`. The forward contraction is `bihw,oi->bohw`. Given upstream derivative `bohw`, the weight gradient contracts `bohw,bihw->oi`; the input gradient contracts `bohw,oi->bihw` and **adds** it into the corresponding input slice. Overlapping windows write to the same input, so assignment would lose contributions. Bias gradients sum batch and spatial axes.

The complete program contains all forward/backward functions and a same-input `F.conv2d` comparison for every derivative. It has the deliberately stated boundary of dense, stride-one, unpadded valid convolution; the earlier general routine still owns grouped, dilated and strided address calculation. Extending this pullback means applying exactly those same addresses in reverse. Autograd can perform that composition in the real classifier, so there is no need to reimplement its engine.

For B examples, I input channels, O outputs, K spatial taps and P output positions, the arithmetic is O(BIOKP), with activation/parameter/output storage plus a tap-sized contraction temporary; there is no explicit O(BIKP) im2col buffer. The maintained backend can use different kernels and layouts. This cost statement is not a measured speed ranking.

The same companion implements valid one-dimensional max/average pooling and their pullbacks. `sliding_window_view` exposes windows without copying each one. Maximum routing saves the first maximizing index per window; average routing distributes an upstream value over the window. `np.add.at` accumulates when maxima or average windows share an input. This is the implementation of the overlap diagrams, not an import of an opaque pooling operation. The script checks both with native PyTorch pooling. Padding, adaptive-window geometry and two-dimensional indexing follow the explicit conventions taught earlier; the real model continues to use native pooling.

Run `python convolution_pullbacks.py` with NumPy and PyTorch. The declared result is equality within float64 tolerances, not a new accuracy measurement. These code paths are separate from the saved digit fits.

**Change the implementation:** replace the upstream all-ones pooling vector by `[2,−1,3]` for values `[1,4,3,2,−1]`, size3, stride1. Then extend the convolution pullback to stride2 by using the original forward sampling slice in both directions.

<details><summary>Hint</summary>Max pooling sends each upstream value to one saved winner; mean pooling sends one third to each covered value. A stride changes selected input addresses, not the summation rule.</details>

<details><summary>Solution and success criteria</summary>The max winners are input indices1,1,2, so the derivative is `[0,1,3,0,0]`. Mean pooling gives `[2/3,1/3,4/3,2/3,1]`. For strided convolution select `row:row+stride*OH:stride` and its analogous column slice; accumulate the input derivative into that same strided slice. Match forward, input, weight and bias derivatives to `F.conv2d(..., stride=2)` on a rectangular image. Equality of outputs alone does not catch a mistaken overlapping scatter.</details>

### Adaptive pooling without a hidden implementation

The same program implements `adaptive_average1d` and its explicit pullback. For output bin `i`, use `start = floor(i*n/m)` and `end = ceil((i+1)*n/m)`, then average the half-open input range `[start, end)`. These bins can overlap; they are not necessarily a disjoint partition. A prefix sum makes each bin sum a subtraction, for O(n+m) time and storage. The pullback adds `upstream[i] / (end-start)` to every member of that bin; a range-add difference array accumulates this in O(n+m), including overlapping bins. Prefix subtraction can lose relative precision when subtracting two large, almost equal cumulative sums; this float64 teaching implementation is not a guarantee of identical summation error to a native kernel.

The program compares values and input gradients with `torch.nn.functional.adaptive_avg_pool1d` for 5→3, 3→5 and global 5→1 pooling. The upsampling-shaped case is intentional: adaptive average pooling can create overlapping repeated bins even though no interpolation rule is being applied. **Changed-input task:** pool `[1, 2, 3, 4, 5]` into three bins with output cotangent `[1, 2, 3]`. The bins are `[0,2)`, `[1,4)` and `[3,5)`, giving values `[1.5, 3, 4.5]` and input gradient `[1/2, 7/6, 2/3, 13/6, 3/2]`. Verify the middle inputs collect all their participating bins, then extend the same construction along both axes for adaptive 2-D average pooling.

## 11. Practice: construct, diagnose, transfer

Try the questions before opening the solutions. A correct number without explaining which cells and parameters participated is incomplete.

1. **Changed filter.** On the first 3×3 image, replace the filter by \(\begin{bmatrix}1&0\\0&-1\end{bmatrix}\). Calculate all outputs. Then increase only the central pixel from 1 to 2 and identify every changed output.
2. **Mix channels.** Two input channels have single-cell values 2 and 3. Build two output channels with a 1×1 convolution: first equals their sum, second equals their difference. Supply the complete weight and bias arrays. Which output changes if only channel two increases by one?
3. **A shape is not a coordinate.** Input width 10, kernel four, stride one, dilation one. Choose padding that preserves width. Give both asymmetric choices nearest to symmetry and their first output centers. Explain why equal-sized branches may still be misaligned.
4. **Pooling ambiguity.** Give two different 2×2 windows with max four and average two. Can their pooled outputs reconstruct the original window? Then calculate the gradient of the sum of stride-one, size-two max pools on \([3,1,4]\).
5. **A new receptive field.** Start with a 20-wide input. Apply kernel three/stride two/pad one, then kernel three/dilation two/stride one/pad two. Find output width, \(r,j,a\). Does a bounding width alone establish that all enclosed coordinates connect?
6. **Experiment diagnosis.** A model changes from 118 correct and CE 0.08 to 118 correct and CE 0.16. A colleague says nothing changed because accuracy is identical. Explain what else to inspect. Propose one controlled training change to investigate the observed shift failures, including what data can inform selection.
7. **Optional adjoint problem.** Use \(C\) above and output-side values \([1,1]\). Compute \(C^\top[1,1]\), and name two distinct inputs with the same forward output.
8. **Optional cost problem.** For 16 input and 32 output channels, compare weights in a dense 3×3 convolution with multiplier-one depthwise 3×3 followed by pointwise 1×1. Explain what measurement is still missing before claiming a speedup.

<details><summary>Hints</summary>

1. Name each window by its top-left coordinate; a pixel can occupy different filter positions. 2. Each output has its own row of input-channel weights. 3. Total padding must equal \(k-1\); use the center recurrence. 4. A window's sum must be eight. Overlapping gradients add. 5. Update \(r\) using the jump from before the layer. 6. CE depends on confidence in the true label, not just the largest logit. 7. Think about the difference filter acting on a constant. 8. Count separately before dividing; do not equate arithmetic and elapsed time.
</details>

<details><summary>Worked solutions</summary>

1. Outputs are \(\begin{bmatrix}0&-1\\-1&1\end{bmatrix}\). After editing the center, they become \(\begin{bmatrix}-1&-1\\-1&2\end{bmatrix}\). The other windows contain that pixel but multiply it by a zero coefficient, so they stay unchanged.
2. Weights, in output/input/spatial order, are \([[[[1]],[[1]]],[[[1]],[[-1]]]]\), biases \([0,0]\). Outputs are \([5,-1]\). Changing the second input to four gives \([6,-2]\): both change, in opposite directions.
3. Padding \((1,2)\) gives first center 1; \((2,1)\) gives center 0, using input centers \(0.5,1.5,\ldots\). Both output widths are ten. Their center grids differ by one input pixel, so elementwise addition would combine different locations.
4. Examples are \(\begin{bmatrix}4&2\\1&1\end{bmatrix}\) and \(\begin{bmatrix}4&0\\2&2\end{bmatrix}\). Both summaries match although the windows differ. The one-dimensional max outputs are \([3,4]\); gradient is \([1,0,1]\).
5. First layer produces width ten, \(r=3,j=2,a=0.5\). Second preserves width ten and produces \(r=11,j=2,a=0.5\). Bounding widths alone are insufficient in general. Here the first layer's contiguous offsets plus the second layer's spaced offsets reach \(\{-5,-4,-3,-1,0,1,3,4,5\}\), leaving holes at −2 and 2 before boundary clipping.
6. Inspect per-row true-class probabilities, especially incorrect and nearly tied examples; equal accuracy can hide worse confidence. One controlled trial adds small label-preserving translations to training images while keeping architecture, seed protocol and learning budget fixed. Inspect whether crops remain interpretable, then compare the same declared development conditions. These comparisons cannot be relabeled as untouched final-test evidence.
7. The transpose result is \([1,0,-1]\). Inputs \([1,3,2]\) and \([2,4,3]\) both produce \([-2,1]\), because adding a constant leaves adjacent differences unchanged.
8. Dense: \(9(16)(32)=4,608\) weights. Separable: \(9(16)+16(32)=656\), a ratio of about 7.02, excluding biases. Actual latency needs measurement for the target input/output sizes, batch, dtype, layout, backend, device and memory behavior; predictive quality also needs validation.
</details>

Core readiness means you can trace one output and update, connect channels correctly, plan spatial sizes and input regions, explain information lost by pooling, and interpret the controlled digit comparison. The optional questions extend that understanding; they are not a hidden requirement to begin the next lesson.

The next topic in this module is [Landmark Architectures: LeNet through EfficientNet](/learn/path/full-curriculum/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet?module=deep-learning-fundamentals). It will use these operators to explain architectural choices. Depthwise/dilated designs follow that topic; the route does not skip over an unprepared lesson.

## References and other ways to learn

- [Dumoulin and Visin, A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/pdf/1603.07285), v2: a visual reference for padding, stride and transposed operations. Use sections 2–3 after the geometry ruler, section 4 after the scatter example. Its operator arithmetic is durable; library-specific policies are checked separately here.
- [Stanford CS231n, Lecture 5: Convolutional Neural Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo): an alternate spoken introduction to convolution, pooling and the transition from dense layers. The official 2017 syllabus and video description were checked; the full video was not watched for this draft. Historical architecture examples are context, not current performance recommendations.
- [Araujo, Norris and Sim, Computing Receptive Fields](https://distill.pub/2019/computing-receptive-fields/): interactive coordinate and multi-path diagrams, useful after calculating \(r,j,a\). Historical model comparisons do not prove that increasing receptive field alone causes better accuracy.
- [Luo and colleagues, Understanding the Effective Receptive Field](https://arxiv.org/pdf/1701.04128): the advanced analytical and empirical source for distinguishing possible reach from concentration of input gradients. Read the assumptions before generalizing its profile shapes.
- [Odena, Dumoulin and Olah, Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/): an especially useful visual alternate route through overlap patterns and decoder choices.
- [Zhang, Making Convolutional Networks Shift-Invariant Again](https://proceedings.mlr.press/v97/zhang19a.html): primary research motivating anti-aliasing around downsampling. The abstract was reviewed; the specific experiment results here come from our retained calculation and digit program.
- [PyTorch 2.14 Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html), [MaxPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html), [AvgPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AvgPool2d.html), [ConvTranspose2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html), and [Fold](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Fold.html): exact API policies, especially groups, padding, pooling denominators and overlap accumulation.
- [Channels-last tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) and [NVIDIA convolution guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html): optional engineering references for storage and algorithm choices. The NVIDIA examples include older device/software benchmarks; no timings from them are presented as our measurements.
- [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits): original data source, authors and license. Our subset, source IDs and exact split are documented in the adjacent provenance file.
