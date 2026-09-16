# Landmark Architectures: LeNet, AlexNet, VGG, ResNet & EfficientNet

You can recognize a handwritten **8** even when one loop is wider than the other. A program receives a grid of numbers. How should its computation be arranged so that it can learn useful visual patterns, combine them, and make a decision within a resource budget?

An **architecture** is that arrangement: which operations run, the shapes they accept, and the connections through which information travels. Its weights are the numbers learned inside the arrangement. Changing the architecture changes what the model can express, how gradients reach its parameters, and what computation it requires. Changing the training recipe can also change its performance—even when the architecture stays identical.

Building on [Convolution, Pooling & Receptive Fields](/learn/path/full-curriculum/convolution-pooling-receptive-fields?module=deep-learning-fundamentals), we will read famous networks as answers to concrete design questions. Then we will compare four small, fully specified networks on actual handwritten digits and inspect how their spatial features produce a class score.

**First pass:** follow §§1–6 for the architectural ideas, §8 for the runnable investigation, and §§9–11 for interpretation and practice. The detailed historical fidelity notes and §7's additional families are optional branches. You need not memorize publication years or reproduce ImageNet training to become ready for the next topic.

## 1. Learn to read the diagram before learning the names

For one image, a feature tensor has shape **channels × height × width**, written `C × H × W`. A channel is a learned map of responses, not necessarily a named concept such as “eye.” With several images, the leading batch axis gives `N × C × H × W`.

A typical classifier has three roles:

```text
pixels → stem → repeated feature-processing stages → spatial summary → class scores
          │                  │                            │              │
     first local maps   combine/refine maps         one vector      one number
                                                                   per class
```

The **stem** receives pixels. A **stage** usually processes maps at one spatial resolution; a boundary may reduce that resolution and increase channel count. The **backbone** is the feature-producing portion. A task-specific **head** converts its output into scores, boxes, masks, or another required answer.

A convolution reuses the same local weight pattern across positions. An activation makes the computation nonlinear. Pooling or a strided convolution can reduce the number of positions. A final linear layer gives **logits**, unrestricted class scores. Softmax turns those scores into a distribution; cross-entropy penalizes assigning little probability to the correct class. Backpropagation computes how every participating weight affects that loss. None of these operations knows what an “8” is before learning.

<!-- Figure placement: architecture-feature-route. -->

**Reading the feature pyramid.** Our later example starts with an 8×8 input, produces twelve 8×8 maps, reduces them to twelve 4×4 maps, and eventually produces sixteen 2×2 maps and sixteen summary numbers. A grid represents positions within one channel; a stack represents separate channels. Reducing spatial size does not automatically reduce the channel axis.

Three different budgets are easily confused:

| Quantity | What it counts | What changes it |
| --- | --- | --- |
| Parameters | Stored learned scalar values | Width, kernels, connectivity, classifier dimensions |
| Multiply-accumulates (MACs) | Products accumulated in convolution/linear outputs | Parameters **and** how many positions reuse them |
| Runtime memory and elapsed time | What the actual execution needs | Shapes, batch, precision, activations, optimizer, operator implementations, device |

For a dense `k × k` convolution with `C_in` input and `C_out` output channels:

\[
P=k^2C_{\mathrm{in}}C_{\mathrm{out}}+C_{\mathrm{out}}
\]

when each output channel has a bias. Omitting biases removes the final term. At `H_out × W_out` output positions, one image requires

\[
M=H_{\mathrm{out}}W_{\mathrm{out}}k^2C_{\mathrm{in}}C_{\mathrm{out}}
\]

convolution MACs. These counts omit bias addition, activation, normalization, pooling and other operations. Some reports count a multiply and add as two FLOPs; others report one combined operation. State the convention before comparing numbers.

For `64 → 128`, a 3×3 kernel and 14×14 output, the bias-free weight count is **73,728**, but the convolution uses **14,450,688 MACs**. Each learned weight is reused at 196 positions. That is why a parameter count cannot stand in for runtime.

Before continuing, predict the effect of doubling both channel counts while leaving the spatial grid unchanged. Compare bias-free weights, convolution MACs and output-map elements.

<details><summary>Hint</summary>

Weights and MACs contain both channel dimensions. The output tensor contains only the output-channel dimension.

</details>
<details><summary>Check your prediction</summary>

The weight and convolution-MAC counts multiply by four; the number of output-map elements only doubles. These quantities have different scaling laws.

</details>

## 2. LeNet: learn local features and combine them

Start with the digit task. A small patch might contain a short stroke, a curve, or background. Convolution lets one detector inspect many possible locations. A later layer combines several learned response maps, so it can respond to a configuration of earlier patterns. Subsampling reduces the spatial representation before another round of processing.

The 1998 LeNet-5 paper describes this progression:

```text
1×32×32 → C1:6×28×28 → S2:6×14×14 → C3:16×10×10
         → S4:16×5×5 → C5:120×1×1 → F6:84 → 10 class penalties
```

The first 5×5 convolution has six filters. It needs `6 × (25 + 1) = 156` parameters, including biases. It does not learn a separate set for every patch. Its 28×28 output follows from `32 − 5 + 1` valid placements.

Notice C5: a 5×5 convolution applied to a 5×5 map has one output location. On this input size, it behaves like a fully connected operation over that map. The operation still has spatial meaning: on a larger incoming map, the same convolution would slide over multiple locations. A shape can make two implementations coincide without making them interchangeable for every input.

The historical model used learned subsampling coefficients, partially connected C3 channels, scaled tanh activations, and an output based on distances to class templates. Many modern “LeNet” tutorials replace these with average pooling, fully connected channel mixing and a linear classifier. Those adaptations are useful if labeled as adaptations. The architecture figure and historical details come from [LeCun and colleagues, §II.B](https://gwern.net/doc/ai/nn/cnn/1998-lecun.pdf).

**What to carry forward:** the network learns a hierarchy of spatial features, and the classification loss trains that hierarchy jointly. “Early edges, later objects” can be an intuition for some learned networks; it is not a guarantee that every channel has a clean human label.

**Optional historical connection.** The same paper connects character recognition to a larger document-reading system. A good isolated digit recognizer still needs field extraction, segmentation and contextual decisions to read a complete check. This distinction returns in modern systems: a backbone is a component, while the deployed task is an entire pipeline.

## 3. AlexNet and VGG: make richer features practical

### AlexNet: architecture and training work together

Recognizing varied color photographs needs more representational capacity than recognizing centered digits. AlexNet combined five convolutional and three fully connected learned layers with ReLU, augmentation, dropout and GPU training. ReLU preserves positive inputs and sets negative ones to zero; its positive-side derivative avoids the saturation of a large positive tanh input. Negative ReLU inputs still have zero derivative.

A simplified, explicit geometry example uses a 227×227 RGB input and 96 filters of size 11, stride 4, no padding:

\[
H_{\mathrm{out}}=\left\lfloor\frac{227-11}{4}\right\rfloor+1=55.
\]

A following size 3, stride 2 max pool gives 27 positions. The pool reduces the number of positions; it does not turn 96 channels into 27 channels.

<!-- Figure placement: architecture-stride-versus-pool. -->

**One stride, then one pool.** One first-layer output reads an 11×11 input region. At its adjacent output, that support moves four pixels. The pool then groups 3×3 convolution responses. These are two different operations on two different grids.

That geometry is the familiar teaching variant used in the [Stanford architecture lecture](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf). The original paper's input description and two-GPU connectivity, and modern library variants, need explicit matching before reproducing parameter totals. A model called “AlexNet” is not a sufficient implementation specification.

The historical result also illustrates a measurement trap. In the original paper's ILSVRC 2012 table, one CNN has 18.2% validation top-5 error; five CNNs have 16.4%; the seven-network submission involving extra pretraining has 15.3% test error. These are different evaluation setups. Do not place 16.4 on a graph labeled “the winning single model.” [Krizhevsky, Sutskever and Hinton, §§3–6 and Table2](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

### VGG: compose small filters

Suppose every layer keeps `C` channels, stride 1 and the spatial grid. One 5×5 convolution has 25C² weights. Two 3×3 convolutions have 18C² weights, with an activation between them.

Why do two 3×3 layers reach a 5×5 input region? The first layer reaches one pixel either side of its output center. The second reaches one first-layer position either side; each of those already depends on a 3×3 input region. The radius becomes 2. Three 3×3 layers give radius 3 and a 7×7 support.

The same support does **not** mean the same function. The intermediate representation and activation create a different computation. Nor is the saving automatic when the intermediate width changes: `9 C_in C_mid + 9 C_mid C_out` must be compared with `25 C_in C_out`.

VGG-16 organizes 13 convolutional layers into stages with repetition counts **2,2,3,3,3**, channel widths **64,128,256,512,512**, and pooling between stages. Three dense layers complete the 16 learned layers. This regular layout is easy to reason about, but its original classifier is expensive. [VGG paper, §2 and Table1](https://arxiv.org/pdf/1409.1556).

For the familiar 1000-class VGG-16 with biases:

| Part | Exact parameters |
| --- | ---: |
| Convolutional trunk | 14,714,688 |
| `7×7×512 → 4096` | 102,764,544 |
| `4096 → 4096` | 16,781,312 |
| `4096 → 1000` | 4,097,000 |
| Total | 138,357,544 |

The first dense layer alone has over 102 million parameters. The **whole head**, not that one layer, has 123,642,856. The total agrees with the [Torchvision VGG-16 model specification](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html).

**Budget investigation — where did the memory go?** Start with the 7×7×512 representation. Choose either its original dense head or spatial averaging followed by `512 → 1000`. Before revealing the totals, predict which individual component accounts for most of the difference. Then change the class count or final spatial size. A new classifier is a different function that must be trained; the arithmetic alone cannot predict its accuracy.

The global-average alternative has **513,000** head parameters. It keeps one average per channel and discards within-channel spatial arrangement at this boundary. In return it avoids multiplying the head input dimension by 49. Averaging has no learned parameters; the following classifier still learns combinations of channels.

For scale, storing VGG's first dense layer as float32 weights, gradients and two float32 Adam moment arrays requires 1,644,232,704 bytes for those four arrays. Activations, other layers and execution buffers are additional. This is an exact storage estimate under the stated representation, not a claim that a particular GPU will run out of memory.

## 4. Inception and ResNet: change the routes information can take

### Inception: parallel views, then concatenate

A small local pattern and a wider arrangement might both matter. An Inception-style block processes the same input along several branches and places their outputs next to one another along the channel axis:

```text
                 ┌─ 1×1 convolution ────────────────┐
input ───────────├─ 1×1 → activation → 3×3 ─────────┤
                 ├─ 1×1 → activation → 5×5 ─────────┼→ concatenate channels
                 └─ 3×3 pool → 1×1 convolution ────┘
```

Every branch must return the same batch and spatial dimensions. If their channel counts are 64,128,32,32, concatenation gives 256 output channels. The operation preserves distinct branch outputs; it does not average them.

A 1×1 convolution is learned channel mixing at each location. It can reduce width before an expensive spatial convolution. Reducing 480 channels to 32 before a 5×5 projection to 480 uses

\[
480(32)+25(32)(480)=399{,}360
\]

weights, instead of `25(480)(480)=5,760,000`. The reduction is about 14.42-fold in this branch's weights and convolution MACs at matched output grids. It also restricts the intermediate representation. “Cheaper” is the established arithmetic; retaining enough useful information is the learning question.

GoogLeNet combined these branches with other design and training choices. The paper acknowledges earlier work on 1×1 channel networks; it did not invent that operation in isolation. [Szegedy and colleagues, §§4–5](https://arxiv.org/pdf/1409.4842).

### ResNet: refine a representation through addition

A residual block offers another route:

\[
y=x+F(x).
\]

The branch `F` learns a change to the incoming representation. If the desired transformation is close to identity, a small branch output can supply it. Backpropagation also gets a direct contribution through the identity path:

\[
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}.
\]

This helps explain the design; it does not guarantee that gradients cannot cancel or that any depth trains successfully. The preceding [Residual Connections lesson](../residual-connections-skip-connections/lesson.md) examines those limits.

The original post-activation basic block computes two convolutions with normalization, applies ReLU between them, adds the skip, then applies ReLU again. A projection can match the skip's channels and spatial size when a stage changes shape.

<!-- Figure placement: architecture-branch-lanes, add-versus-concatenate inset. -->

**Add versus concatenate.** For `x=[1,2]` and `F(x)=[3,−1]`, addition yields `[4,1]`; concatenation yields `[1,2,3,−1]`. Two aligned lanes merge at a plus sign, whereas concatenation retains four lanes side by side. The former preserves width; the latter increases it.

A ResNet-50 bottleneck uses 1×1 reduction,3×3 processing, then 1×1 expansion. For 256→64→64→256, the convolutions use

\[
256(64)+9(64)(64)+64(256)=69{,}632
\]

weights. Two dense 3×3 convolutions at 256 channels use 1,179,648. Both return 256 channels, but their internal capacities differ. BatchNorm parameters and any skip projection must be added when counting a complete implemented block.

ResNet's motivating **degradation** observation was increased **training** error in deeper plain networks. Calling it merely overfitting misses the optimization issue. The paper compares matched plain and residual variants; a record score also depends on its training and evaluation recipe. [He and colleagues, introduction and §3](https://arxiv.org/pdf/1512.03385).

## 5. EfficientNet: separate the block from the scaling rule

Two questions are involved: **what should one block do**, and **how should a good base network grow**?

### Spatial filtering, channel mixing and a narrow skip

A depthwise convolution filters each input channel separately. A pointwise 1×1 convolution then mixes channels. With one spatial filter per input channel, the bias-free 3×3 pair uses

\[
9C_{\mathrm{in}}+C_{\mathrm{in}}C_{\mathrm{out}}
\]

weights. For 64→128 this is 8,768, compared with 73,728 for a dense 3×3 convolution. At 14×14 it uses 1,718,528 convolution MACs. That is about 8.41 times fewer, not a measured 8.41 times lower latency. The factorization imposes a restriction on the spatial filters that each output can use. The next topic develops that restriction explicitly. [MobileNet V1, §3](https://arxiv.org/pdf/1704.04861).

An **inverted bottleneck** first expands channels, performs depthwise spatial filtering, and projects back to a narrow representation. The skip connects the narrow representations when their shapes match:

```text
x: C channels ────────────────────────────────────────────────┐
       └→ expand to tC → depthwise spatial → project to C ──── + → y
```

This reverses the wide→narrow→wide pattern of a ResNet bottleneck. The final projection is linear in the sense that it has no subsequent pointwise activation on that branch output; the whole block is still nonlinear. A ReLU applied after a narrow projection would erase its negative coordinates. MobileNet V2 motivates preserving information there and evaluates that choice. [MobileNet V2, §3](https://arxiv.org/pdf/1801.04381).

### Squeeze-and-excitation: use the image's context to gate channels

A normal convolution has learned weights shared across input examples. An **SE gate** computes additional channel multipliers from the current example.

For maps `U[c,h,w]`, first average each channel:

\[
s_c=\frac{1}{HW}\sum_{h,w}U_{c,h,w}.
\]

Pass the summary through a small learned network,

\[
g=\operatorname{sigmoid}\!\left(W_2\operatorname{ReLU}(W_1s+b_1)+b_2\right),
\qquad V_{c,h,w}=g_cU_{c,h,w}.
\]

The gate is constant across positions within a channel but can differ between images. Sigmoid multipliers lie between 0 and 1; they change the relative contribution of channels. A conventional convolution already mixes channels with unequal learned weights. SE adds **input-dependent** modulation, not the first ability to distinguish channels. [SE paper, §3](https://arxiv.org/pdf/1709.01507).

For a hand-sized case, take two channel means `a=2,b=1`. Define one hidden unit `h=max(a−b,0)` and gate logits `[h,−h]`. The gates are approximately `[0.7311,0.2689]`. If the second mean becomes 3, the hidden unit becomes 0 and both gates become 0.5. Changing one channel's global content can change another channel's multiplier.

**Context-gate investigation.** Edit actual cells in either of two small feature maps, predict how the other map's gate changes, and then inspect the average→hidden unit→gate→broadcast multiplication. Rearranging values within one map preserves its mean and therefore both gates in this specific model. This null case distinguishes a global summary from a spatial attention map.

### Compound scaling: spend additional resources deliberately

EfficientNet-B0 combines mobile inverted bottlenecks with SE in a staged backbone. Its base-network search optimizes accuracy and FLOPs; the paper explicitly distinguishes this from targeting a particular device's latency. Its other central idea is to scale depth, width and input resolution together. [EfficientNet, §§3–4](https://proceedings.mlr.press/v97/tan19a/tan19a.pdf).

For intuition, imagine a family dominated by same-width dense convolutions. If depth is multiplied by `d`, both channel dimensions by `w`, and both spatial dimensions by `r`, then

\[
\text{parameters}\ \propto dw^2,\qquad
\text{convolution MACs}\ \propto dw^2r^2.
\]

Increasing resolution gives more input positions to analyze; increasing depth supplies more successive transformations; increasing width supplies more features at each stage. None alone guarantees better development performance.

Compound scaling writes `d=α^φ, w=β^φ, r=γ^φ`. Choosing `αβ²γ²≈2` makes one increment of `φ` approximately double this model's convolution work. The paper reports coefficients 1.2,1.1,1.15 from a search around B0. Their actual product is **1.92027**, not exactly 2. At `φ=2`, the idealized parameter multiplier is 2.108304 and the MAC multiplier is approximately 3.687437.

Do not identify every named B-index with that integer `φ` and then present the approximation as an exact model count. Implementations round channels and repeat counts; depthwise, pointwise, SE, stem and classifier terms do not all scale with the same exponents. Input resolution changes MACs without directly changing stored convolution weights.

**Scaling investigation.** Allocate a hypothetical twofold dense-convolution budget to depth, width and resolution. Predict the effect on parameters before revealing it. Compare doubling depth, multiplying width by √2, and multiplying resolution by √2. All have the same idealized MAC factor, but the last keeps the parameter count unchanged. The plot shows algebraic budget contours, not an invented accuracy surface.

## 6. Read an architecture comparison as evidence

A useful comparison tells you **which model, which weights, which data and split, which preprocessing, which metric, and which execution conditions**.

Top-1 accuracy asks whether the largest score selects the label. Top-5 accuracy asks whether the label appears among the five largest scores. Error is one minus accuracy. A multi-network ensemble is a different deployed computation from a single model. A model trained with extra data or a newer recipe is a different experiment even if its diagram looks familiar.

For example, Torchvision documents **76.130%** ImageNet-1K top-1 for ResNet-50's `IMAGENET1K_V1` weights and **80.858%** for `IMAGENET1K_V2`. The architecture and 25,557,032 parameter count are the same. The difference is evidence that the trained-weight package and recipe matter, not a new residual-connection discovery. The V2 inference transform resizes to 232 before a 224 center crop; it is not simply “every image model takes 224.” [ResNet-50 weights and transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).

Use this separation when selecting candidates:

| Deployment question | Evidence to obtain |
| --- | --- |
| Can this recognize the target classes? | Task-specific development metric, baseline, error examples and important slices |
| Can this preserve small details? | Input resolution and stage geometry; inspect what preprocessing discards |
| Does it fit the device? | Export/operator support, measured latency at the intended batch and precision, peak memory |
| Can it reuse a pretrained model? | Exact weight identifier, preprocessing, class/head contract and appropriate adaptation |
| Will its features feed another task? | Required stage resolutions, channel dimensions and output meanings |

A **Pareto** comparison concerns competing objectives: a candidate is dominated if another is at least as good on all declared objectives and strictly better on one. You cannot declare one architecture family universally Pareto-optimal from unrelated runs. A faster model on one CPU may be slower on another accelerator because operator efficiency and memory traffic change.

Keep fine-tuning decisions with the preceding [Transfer Learning lesson](../transfer-learning-fine-tuning-strategies/lesson.md): freezing parameters and choosing BatchNorm behavior are separate decisions. A small target dataset does not establish a universal transfer gain or require full fine-tuning in every case.

## 7. Optional: additional branches in the architecture family

These ideas complete the useful historical context without making every named family a prerequisite for the next lesson.

**DenseNet retains earlier maps by concatenation.** A layer receives `[x₀,x₁,…,xₗ₋₁]` and produces a small set of new channels. Starting with 8 channels and adding 3 per layer gives widths 8,11,14,17,20. Reusing prior maps can support feature and gradient access, while the widening inputs and retained activations affect computation and memory. Transition layers can compress channels and downsample. This is a different connectivity contract from residual addition. [DenseNet, §3](https://arxiv.org/pdf/1608.06993).

**RegNet asks about a family of designs.** Start with proposed block widths `u_j=w₀+w_a j`, quantize them into repeated widths, and group consecutive equal-width blocks into stages. The result is a small set of design parameters controlling an entire network. Evaluating distributions of sampled designs asks whether a design space reliably produces good candidates, rather than celebrating one searched winner. Its empirical conclusions depend on its search and evaluation protocol. [RegNet, §3](https://arxiv.org/pdf/2003.13678).

**NFNet separates normalization from the requirements it helps satisfy.** Its construction combines scaled weight standardization, controlled residual-branch scales and adaptive gradient clipping. The clipping threshold depends on a gradient norm relative to a parameter norm; it is not the same operation as multiplying a residual branch by a constant. This illustrates a general lesson: removing BatchNorm responsibly requires addressing training behavior, not simply deleting a module and expecting the old recipe to work. [NFNet, §§3–4](https://arxiv.org/pdf/2102.06171).

**Learned features can define another model's loss.** In perceptual-loss work, an image transformation network produces an image `ŷ`. A separately pretrained, frozen feature network `φ` maps `ŷ` and a target image `y` into features. A loss such as

\[
L_{\mathrm{feature}}=\frac{1}{CHW}\|\phi_j(\hat y)-\phi_j(y)\|_2^2
\]

compares one layer's representation. Gradients pass through the frozen feature computation to the generated image, even though the feature network's weights are not being updated. A deeper layer can tolerate pixel changes that a pixelwise loss heavily penalizes, but the chosen features can also overlook changes that matter to a human. This is a useful application of VGG's intermediate maps, not a guarantee of perceptual correctness. [Johnson, Alahi and Fei-Fei, §3.2](https://arxiv.org/pdf/1603.08155).

MobileNet refinements and ConvNeXt follow in their own lessons. Wide residual networks change channel capacity; grouped ResNeXt branches change the transformation grouping. Stochastic-depth training already has a [separate home](../dropout-droppath-stochastic-depth/lesson.md). These are combinations of design choices, not steps on a ladder where every later name makes every earlier one obsolete.

## 8. A complete, small architecture investigation

### The question and the observations

Can different feature-processing blocks learn to recognize actual digits, and what do they cost? Use 400 real 8×8 images from the **UCI Optical Recognition of Handwritten Digits** data: 40 images of each digit 0–9. The original collection supports studying handwritten-digit recognition. Each pixel is an integer intensity 0–16; this dataset is different from MNIST. The supplied subset, source-row identifiers and attribution are in [digits-400.csv](digits-400.csv) and [data provenance](data-provenance.md). The UCI record distributes this dataset under CC BY 4.0. [Dataset and collectors](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

All 400 source IDs and pixel vectors are distinct. We use a stratified 280/120 training/development split with seed 22 and divide pixels by the known bound 16. No fitted preprocessing consumes development rows. The source copy does not provide writer identities for these rows, so this does not establish performance on unseen writers. Development results are available for inspection and model choice; there is no untouched final test in this teaching experiment.

Every candidate has the same external shape route:

```text
N×1×8×8
  → 3×3 convolution,1→12; ReLU; max pool2
N×12×4×4
  → one selected body
N×12×4×4
  → 3×3 convolution,12→16; ReLU; max pool2
N×16×2×2
  → average each map
N×16
  → linear16→10
N×10 logits
```

The four bodies are small teaching constructions:

| Body | Operation on 12×4×4 input |
| --- | --- |
| Plain | 3×3→ReLU→3×3→ReLU, all 12 channels |
| Residual | The same two convolutions, add the input before the final ReLU |
| Parallel | Four branches returning 3 channels each:1×1;1×1→3×3;1×1→5×5;pool→1×1; concatenate and apply ReLU |
| Inverted with gate |12→36 expansion; depthwise 3×3; SE hidden width 9;36→12 linear projection; add input |

These are **not miniature benchmark reproductions** of VGG, GoogLeNet, ResNet or EfficientNet. We omit BatchNorm and use fixed small stages so the bodies are readable. The inverted block uses SiLU, the smooth activation `x × sigmoid(x)`, in expansion/spatial operations; the others use ReLU. The comparisons between different body families change several properties at once.

Within a seed, stem, tail and head start with matching tensors in all four models. Plain and residual also share the same initial branch weights: their only forward-rule difference is the skip addition. They are trained separately, so their weights can subsequently diverge.

### Run it offline and follow one update

Download [architecture-experiments.py](architecture-experiments.py) beside the CSV. In a Python environment with PyTorch, NumPy and scikit-learn installed, run:

```text
python architecture-experiments.py
```

The complete program defines every body, loads the supplied data, checks the split inputs, runs 12 small fits, prints final numerical summaries, and writes `calculated-inputs.json`. It needs no pretrained download or image folder. The recorded run used Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.3.5 and scikit-learn 1.9.1 with one CPU thread. Exact low-order values can vary with numerical libraries.

Read its two-convolution body first. The complete program supplies `torch`, `nn` and `F` imports and calls this inside the classifier:

```python
class PlainOrResidual(nn.Module):
    def __init__(self, channels=12, residual=False):
        super().__init__()
        self.first = nn.Conv2d(channels, channels, 3, padding=1)
        self.second = nn.Conv2d(channels, channels, 3, padding=1)
        self.residual = residual

    def forward(self, x):
        branch = self.second(F.relu(self.first(x)))
        return F.relu(branch + x if self.residual else branch)
```

Padding 1 preserves the 4×4 grid. Both convolutions preserve 12 channels, so addition is shape-compatible. The last ReLU belongs after the merge in this example. Moving it into only the branch changes the function.

The training mechanism is compact:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=.003)
for step in range(400):
    model.train()
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x[train]), y[train])
    loss.backward()
    optimizer.step()
```

This excerpt is the update loop from the complete supplied program, not a standalone script. Each step uses all 280 training images. The logits select neither labels nor probabilities in advance: cross-entropy performs the needed log-softmax internally. Clearing gradients prevents the previous step's gradients from being added accidentally. Backpropagation reaches the head, tail, selected body and stem. Adam then updates their learned parameters.

There are no augmentations, dropout, weight decay, early stopping or development-selected epochs. Every model receives 400 updates at learning rate 0.003 with seeds 1,2,3. We record training/development cross-entropy and correct counts at steps 0,1,25,100,200,400. This fixed recipe is a comparison condition, not a promise that it is optimal for all four architectures.

### Inspect the result, including the baseline

All 12 final models correctly classify all 280 training images. Their development behavior differs:

| Body | Parameters | Conv/linear MACs per image | Development correct, seeds 1/2/3, out of 120 | Development CE, seeds 1/2/3 |
| --- | ---: | ---: | --- | --- |
| Plain |4,650|76,192|116 /115 /112|0.1180 /0.2325 /0.3061|
| Residual |4,650|76,192|112 /115 /114|0.2065 /0.2074 /0.3737|
| Parallel |2,502|41,920|119 /117 /115|0.0367 /0.1056 /0.2091|
| Inverted with gate |3,999|54,376|116 /112 /117|0.1971 /0.3331 /0.0928|

The MAC count excludes addition, pooling, gate multiplication and activations. Thus plain and residual have equal counted MACs, although residual addition still performs work. The saved layer-by-layer counts make this convention inspectable.

What can we conclude? All four small constructions learn the training set. The parallel candidate has fewer counted parameters and MACs here, and its three development counts are promising. The residual path does not consistently beat the plain model in this shallow fixed-recipe setting. A result about helping train very deep networks is not a guarantee that a skip improves every small model.

Notice the inverted candidate's seed 1 score: 116 correct, the same as plain, but higher cross-entropy. Correct counts discard confidence information. Cross-entropy also reacts to probability assigned to wrong labels and to the confidence of correct predictions.

**Result investigation.** Choose a parameter/MAC budget and predict which recorded candidates are eligible. Then inspect matched plain/residual learning traces for a chosen seed. Changing the budget changes eligibility, not measured model outputs. The available observations cover exactly the declared model configurations, seeds and learning rate.

The three seeds show initialization variation on one shared split. They do not provide independent samples from a deployment population. If you pick the parallel candidate after inspecting this table, that choice has consumed development information; a later final assessment needs new held-out evidence.

## 9. How can a class score become a spatial map?

The final representation in our classifier is 16 maps of size 2×2. The head averages each map, then linearly combines those 16 averages. Because averaging and a weighted sum are linear, we can reverse their order.

Let `A_c(h,w)` be the final map in channel `c`, and let `w_kc,b_k` be the weights and bias for class `k`:

\[
z_k=b_k+\sum_c w_{kc}\left(\frac{1}{HW}\sum_{h,w}A_c(h,w)\right).
\]

Define the **class activation map**

\[
M_k(h,w)=\sum_c w_{kc}A_c(h,w).
\]

Then `z_k=b_k+mean(M_k)`. The mean of the map, plus the bias, recovers the class logit exactly in real arithmetic for this head. A map is not itself a probability map, and a negative contribution can reduce the class score. This correspondence is the mechanism behind [class activation mapping](https://arxiv.org/pdf/1512.04150); our definition explicitly uses an average and keeps the bias.

Here is a complete hand-sized computation you can run independently:

```python
import numpy as np

maps = np.array([[[1., 2.], [0., 3.]],
                 [[0., 1.], [2., 1.]]])
weights = np.array([2., -1.])
bias = .5
class_map = np.einsum("c,chw->hw", weights, maps)
via_features = weights @ maps.mean(axis=(1, 2)) + bias
via_map = class_map.mean() + bias
print(class_map)
print(via_features, via_map)
```

Output:

```text
[[ 2.  3.]
 [-2.  5.]]
2.5 2.5
```

The second channel has a negative class weight. Raising its bottom-left cell from 2 to 6 changes that location's class-map value from −2 to −6 and lowers the score from 2.5 to 1.5. More activation can mean less evidence for this particular class.

**Map investigation.** Edit a cell or a signed class weight, predict the change in the score, and reconcile two routes: average-then-linear and linear-then-average. Rearranging the same positions jointly across channels moves the map but preserves the score. Setting all class weights to zero leaves only the bias. These nulls help identify what the head retains and discards.

Then inspect real saved examples from seed 1. Source 251, an actual 4, is correctly classified by all four models. The first misclassified development specimen for the parallel model is source 379: an 8 predicted as 5. Its class 5 and class 8 maps come from the **same** feature tensor with different head weights. Looking only at a vivid predicted-class map would hide the competing explanation.

The saved features, all ten class maps, head weights, biases, logits and probabilities support the correspondence. Float32 reconstructions differ by at most approximately 5.73×10⁻⁶ across the saved cases. That is numerical ordering error in an algebraic identity.

Use the original 2×2 cells alongside any enlarged overlay. Upsampling does not create extra spatial detail. A large positive cell describes a contribution from learned features whose receptive field can extend beyond that cell's displayed location. It does not prove a causal explanation, a precise object boundary, or that editing the corresponding input pixels will have the predicted effect. Editing intermediate features in the hand calculation is explicitly a different intervention from rerunning an image through the whole trained backbone.

## 10. Practice and diagnosis

Attempt each task before opening its hint or solution.

### 1. A new head budget

A backbone returns 128 channels of size 7×7. Compare a direct flattened linear head for 7 classes with global averaging followed by a linear head. Include biases. Which spatial information can only the first head use?

<details><summary>Hint</summary>

One head receives 6,272 scalars; the other receives 128. Both return 7 logits.

</details>
<details><summary>Solution</summary>

Flattened: `(7×7×128+1)×7=43,911` parameters. Averaged: `(128+1)×7=903`. The flattened head can assign different weights to different positions within a channel. The averaged head cannot distinguish permutations of those positions at its input. This does not establish which trained model has better task performance.

</details>

### 2. A bottleneck that is no longer cheap

Compare one 5×5 convolution from 32 to 32 channels with two 3×3 convolutions, first 32→64 then 64→32. Ignore biases, use the same spatial grid, and put an activation between the small convolutions. Is “two small kernels use fewer weights” true here?

<details><summary>Hint</summary>

Write both channel dimensions of both 3×3 layers. The intermediate width is 64, not 32.

</details>
<details><summary>Solution</summary>

The 5×5 layer has `25×32²=25,600` weights. The pair has `9×32×64+9×64×32=36,864`. Their interior receptive-field support is 5×5, but the pair is larger and has an intermediate nonlinearity. The familiar 18C² comparison assumes the intermediate width is C.

</details>

### 3. Repair the merge

An input has shape `N×24×16×16`. A strided branch returns `N×48×8×8`. Give a shape-compatible learned skip for addition. If another design concatenates two `N×24×8×8` branches, what is its output shape? Explain why these are different computations despite one matching final shape.

<details><summary>Hint</summary>

An addition needs both summands to match. A 1×1 convolution can change channels and use a stride.

</details>
<details><summary>Solution</summary>

A 1×1 skip convolution 24→48 with stride 2 returns `N×48×8×8`. The branch and projected skip can then be added. Concatenating two 24-channel branches also returns `N×48×8×8`, but retains their outputs in separate channel ranges. Addition combines corresponding entries. With no bias, the projection has 24×48=1,152 weights; its 8×8 outputs require 73,728 convolution MACs per image.

</details>

### 4. Change the class-map question

Use the two maps in §9 with new class weights `[−1,2]` and bias −0.5. Compute the map and score. Increase the first map's top-left cell from 1 to 5. Predict, then calculate, the new score.

<details><summary>Hint</summary>

The first map now has negative weight. A change at one of four positions changes the spatial average by one quarter of the weighted change.

</details>
<details><summary>Solution</summary>

The original class map is `[[-1,0],[4,-1]]`. Its mean is 0.5, so the score is 0. The edit reduces the top-left contribution by 4; the map's mean falls by 1 and the score becomes −1. The other class from §9 can respond differently to exactly the same features.

</details>

### 5. Can you draw this benchmark curve?

A draft has a point for an old model's single-crop validation accuracy, another for a later seven-network test ensemble with extra pretraining, and invented points filling missing years. Its caption says “architecture progress.” Describe a defensible replacement.

<details><summary>Hint</summary>

Separate what is known, what is comparable, and what was not measured.

</details>
<details><summary>Solution</summary>

Remove invented points. Either select a genuinely matched protocol or show discrete reported results with explicit weights, data, split, inference and source labels. A historical table can preserve the milestones without implying a controlled causal comparison. For this lesson's architecture budgets, use exact shape/parameter calculations separately from the recorded small-data results. Connecting points is an additional claim about what the line means.

</details>

### 6. Diagnose a failed adaptation

A team freezes all parameters whose name lacks the text `"classifier"` or `"fc"`. Its supposedly frozen backbone contains SE layers named `fc1` and `fc2`. It also calls `model.train()` globally. Why might this fail to implement a linear probe? Propose direct checks.

<details><summary>Hint</summary>

Module identity is stronger than a substring. Learned tensors and running-state buffers need separate inspection.

</details>
<details><summary>Solution</summary>

The substring rule can leave backbone SE weights trainable. Global train mode can update BatchNorm running statistics even for parameters with `requires_grad=False`. Freeze the backbone module's actual parameters, explicitly make only the intended head trainable, and set the chosen backbone evaluation policy after any global mode change. List optimizer parameter identities and compare backbone parameters/buffers before and after a step. A chosen fine-tuning policy may deliberately update some of these; label that policy accurately.

</details>

### 7. Design a next experiment

You have a 5,000-parameter limit and a 60,000-convolution/linear-MAC limit for the small task. Which recorded candidates qualify? Choose one development question to investigate next and state what would remain unknown.

<details><summary>Hint</summary>

Eligibility is arithmetic. Choosing a model and estimating its final deployment performance require different evidence.

</details>
<details><summary>Solution</summary>

Parallel and inverted-with-gate qualify; plain/residual exceed the MAC limit. One reasonable next question is whether the parallel model's errors concentrate in a particular pair of digits, inspected on development rows with denominators. Another is measured device latency, since counted MACs omit important work. Either investigation consumes development or engineering evidence. Unseen-writer reliability and final selected-model performance remain unestablished. More than one next experiment can be sensible if its question and decision rule are explicit.

</details>

## 11. What you should now be able to do

Explain a network as a flow of tensors, not a list of names. Predict the shape and budget consequences of a new layer. Distinguish stacking small kernels, concatenating branches, adding a residual, gating channels and scaling a family. Follow data through a complete training/evaluation example and explain why a compelling architectural idea can still lose a particular comparison. Reconstruct a class score from its maps and identify the interpretation's limits.

You are ready to continue when you can repair the mismatched merge in practice 3, derive the changed map in practice 4, and propose a defensible comparison in practice 7 without copying an architecture recommendation.

The next topic is [Depthwise Separable & Dilated Convolutions](/learn/path/full-curriculum/depthwise-separable-dilated-convolutions?module=deep-learning-fundamentals). We have used a factorized convolution as a building block; next we examine exactly which channel/spatial interactions it can express, how dilation changes the positions a filter samples, and when those choices help or fail.

## References & another way to learn it

**Alternate explanations and practice**

- [Stanford CS231n, Lecture9: CNN Architectures](https://www.youtube.com/watch?v=DAOcjicFr1Y), video, with [companion slides](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf). Useful after §3 for shape questions and visual comparisons of AlexNet, VGG, GoogLeNet and ResNet. It assumes basic convolution and predates EfficientNet. The relevant slide content and the creator's video description were reviewed; no claim of watching the complete video or verifying timestamps is made. Historical variants and current APIs still need the distinctions in this lesson.
- [Torchvision's model and weight guide](https://docs.pytorch.org/vision/stable/models.html), official documentation. Use after §6 to connect a model builder with its weight identifier, transformations and output categories. It is an API guide rather than a beginner explanation of the architecture.
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155), Johnson, Alahi and Fei-Fei, paper. Optional after §7:§3.2 and its reconstruction figures make intermediate-feature losses concrete. Basic backpropagation is useful; its historical feature loss is not a universal human-similarity metric.

**Precise technical and historical sources**

- [LeNet and document recognition](https://gwern.net/doc/ai/nn/cnn/1998-lecun.pdf), LeCun and colleagues, 1998, original paper mirrored as a PDF; §II.B for the actual layer contract.
- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), 2012, §§3–6 and Table 2; [VGG](https://arxiv.org/pdf/1409.1556), §2 and Table 1; [GoogLeNet/Inception](https://arxiv.org/pdf/1409.4842), §§4–5; [ResNet](https://arxiv.org/pdf/1512.03385), §3. Read a model's experimental conditions together with its diagram.
- [MobileNet V1](https://arxiv.org/pdf/1704.04861), §3; [MobileNet V2](https://arxiv.org/pdf/1801.04381), §3; [SE](https://arxiv.org/pdf/1709.01507), §3; [EfficientNet](https://proceedings.mlr.press/v97/tan19a/tan19a.pdf), §§3–4. These supply the efficient-block and scaling definitions.
- [DenseNet](https://arxiv.org/pdf/1608.06993), §3; [RegNet](https://arxiv.org/pdf/2003.13678), §3; [NFNet](https://arxiv.org/pdf/2102.06171), §§3–4. Optional family branches rather than required extra reading.
- [Class activation mapping](https://arxiv.org/pdf/1512.04150), Zhou and colleagues, §2. Compare its pooled-feature convention with the explicit mean and bias kept here.
- [UCI optical digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), E. Alpaydin and C. Kaynak, 1998, [DOI10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY 4.0; [complete program](architecture-experiments.py), [retained numerical outputs](calculated-inputs.json), [data provenance and limits](data-provenance.md).
