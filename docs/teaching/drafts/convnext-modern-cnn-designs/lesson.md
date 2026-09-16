# ConvNeXt & Modern CNN Designs

A visual model has two jobs inside each layer: combine nearby evidence and combine different kinds of evidence. A dark curve beside a vertical stroke is a spatial relationship. Combining “curve,” “stroke” and “enclosed region” detectors is a channel relationship. ConvNeXt organizes these jobs into a small repeated block, then asks an equally important question: how much of a model's success comes from its architecture, and how much comes from how it was trained?

The preceding [Depthwise Separable & Dilated Convolutions](/learn/path/full-curriculum/depthwise-separable-dilated-convolutions?module=deep-learning-fundamentals) lesson explained inexpensive spatial filtering and its restrictions. Here we use those operations to read a complete modern network. We will also train a small model to reconstruct hidden parts of real handwritten digits, then test what its representation makes available to a separate classifier.

**First pass:** follow §§1–6, run or inspect the small experiment, and attempt practice 1–5. You should be able to trace one block, identify which numbers a normalizer sees, prevent hidden-input leakage, and distinguish reconstruction from recognition. The deployment, kernel-fusion and hybrid-design branches in §7 and practice 6–8 deepen that understanding; they are not prerequisites for the next lesson.

## 1. Start with the comparison, not the model name

Suppose one model gets more examples right than another. Before explaining the difference using their diagrams, ask whether they used the same images, labels, input resolution, training duration, augmentation, optimizer, regularization and evaluation procedure. Changing several of these changes the question.

An **architecture** defines the function and its trainable parameters. A **training recipe** defines how those parameters are obtained. A **checkpoint** is one resulting set of parameter values. ConvNeXt is a useful case study because its authors first strengthened the training procedure for an existing ResNet, then changed the architecture in stages. The old and enhanced ResNet training runs reported 76.1% and 78.8% ImageNet top-1 accuracy, respectively. That improvement happened before introducing the final ConvNeXt block. [Original study, §2.1](https://arxiv.org/pdf/2201.03545)

The recipe included longer training, AdamW, image transformations, mixed training examples and regularization. You do not need to memorize their hyperparameters to understand the evidence: a newer architecture compared with an older training recipe does not isolate the value of the architecture.

Some individual architecture steps also made the intermediate model worse. Replacing spatial convolutions with depthwise ones reduced computation but initially reduced accuracy; widening the channels recovered capacity. Moving the depthwise filter before expansion initially hurt accuracy; increasing its spatial extent helped in that resulting configuration. These are conditional experiments, not universal bonuses that can be added to any network.

**Visual: an experiment genealogy.** Each node records the architecture, fixed recipe, parameter/MAC budget and measured outcome. Edges identify exactly what changed. A dashed side branch marks a tested change that was not kept. This makes a non-monotonic design process visible without turning a historical leaderboard into a law.

A practical comparison record can be short:

| Question | What must be written down |
| --- | --- |
| What is being predicted? | Label definition, unit of observation and decision metric |
| What may differ? | The specific block or training change being investigated |
| What is held fixed? | Data partition, preprocessing, training budget and paired seeds where possible |
| What did it cost? | Parameters and defined operation counts; separately measured latency if available |
| What supports the conclusion? | Counts/errors on reserved development data, variability and the actual comparison |

Our small experiment will use this structure. It will not use an ImageNet result to predict an accuracy gain on digits.

## 2. Read one block from the input outward

A feature tensor contains a batch of images, spatial positions and channels. Write its shape as \(N\times C\times H\times W\): specimens, channels, rows, columns. At one location, its \(C\) numbers summarize different learned features. A **depthwise** filter looks at neighbors within each channel; a **pointwise** transformation mixes channels at one location.

ConvNeXt V1 uses the following residual branch:

| Operation | Logical output shape | What changes |
| --- | --- | --- |
| Input \(x\) | \(N,C,H,W\) | Starting evidence |
| Depthwise 7×7, padding 3 | \(N,C,H,W\) | Each channel gathers its local spatial neighborhood |
| Channel LayerNorm | \(N,H,W,C\) in the code | The channel vector is standardized separately at each location |
| Linear \(C\rightarrow4C\) | \(N,H,W,4C\) | Features are mixed into a wider set of combinations |
| GELU | \(N,H,W,4C\) | A smooth nonlinear response changes which combinations contribute |
| Linear \(4C\rightarrow C\) | \(N,H,W,C\) | The expanded features are projected back |
| Learned channel scale, then DropPath | \(N,C,H,W\) | Branch contribution is scaled and optionally masked during training |
| Add \(x\) | \(N,C,H,W\) | Existing evidence plus the learned correction |

The two linear layers are applied independently at every spatial location with shared weights. They are equivalent to 1×1 convolutions with the same parameters. The permutation changes where the channel axis appears in the tensor interface; it does not mix information between locations.

Writing the two channel transformations together, the projected vector at one location is

\[
v_j=\sum_{k=1}^{4C} W^{(2)}_{jk}\,\operatorname{GELU}
\left(\sum_{i=1}^{C}W^{(1)}_{ki}\,\widehat{x}_i+b^{(1)}_k\right)+b^{(2)}_j.
\]

Here \(\widehat{x}\) is the normalized output of the spatial filter. The inner sum mixes features; the nonlinearity prevents the two linear transformations from collapsing into a single fixed linear map. GELU is \(z\Phi(z)\), where \(\Phi\) is the standard normal cumulative distribution function. Small negative values can contribute, large positive values mostly pass through, and the response is smooth. This does not make GELU a guarantee of better training than ReLU.

### Which values does LayerNorm actually normalize?

For one specimen and one location, calculate the mean and variance across its \(C\) channels:

\[
\mu=\frac1C\sum_c x_c,\qquad
\sigma^2=\frac1C\sum_c(x_c-\mu)^2,\qquad
\widehat{x}_c=\gamma_c\frac{x_c-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_c.
\]

Each location has its own \(\mu,\sigma^2\). Learned \(\gamma,\beta\) are shared across locations. The implementation uses \(\epsilon=10^{-6}\) to keep constant channel vectors well-defined. These are current-input statistics in both training and evaluation; there is no BatchNorm running average.

Consider two locations with channel vectors \([1,3]\) and \([101,103]\). Channel LayerNorm maps both to approximately \([-1,1]\) before its learned affine transformation. A single GroupNorm group instead uses all four values from the specimen; it preserves the large offset between locations in its normalized output. Replacing one operation with the other changes the function even if the output shapes agree.

**Investigation: move the reduction boundary.** Edit one value in a small feature lattice. Predict which normalized cells will change, then reveal the affected set. Compare channel LayerNorm with a single-group normalization. The goal is to identify the numbers used in a statistic, not to guess which normalizer “wins.”

### Why the spatial filter comes before expansion

A 7×7 depthwise filter at width \(C\) uses \(49C\) weights. Moving that filter after a fourfold expansion uses \(196C\). Both designs can be useful, but the latter spends more spatial-filter work at the expanded width. ConvNeXt spends most of its arithmetic on the dense channel transformations.

With biases, channel LayerNorm and one learned LayerScale vector, a V1 block has

\[
(49C+C)+(2C)+(4C^2+4C)+(4C^2+C)+C
=8C^2+58C
\]

parameters. At \(C=96\), that is 79,296; the two pointwise weight matrices contain 73,728 of them, about 93%. Biases and normalization parameters are small, but counting them correctly matters when checking an implementation.

Ignoring bias additions, normalization, activation and the residual addition, its convolution/linear work is

\[
HW(49C+8C^2)\quad\text{MACs per specimen}.
\]

One MAC here means one product accumulated into a sum. This is not a wall-clock measurement or a count of every floating-point operation.

### Residual scaling and dropping have distinct jobs

LayerScale learns one multiplier per output channel, initially \(10^{-6}\) in the V1 reference implementation. It starts the residual branch contribution very small. The block is therefore close to its input initially; the whole network is not an identity map, because its stem, downsampling and head still change shapes and values. Learned scales need not remain positive or small.

DropPath draws one branch mask per specimen, shared across channels and positions. With drop probability \(p\), it returns the branch divided by \(1-p\) when retained and zero when dropped. Evaluation uses the whole branch. This preserves the expected branch contribution for fixed inputs, not the expected final prediction of an arbitrary nonlinear network.

In the supplied code the branch is calculated before applying the mask. Dropping its contribution does **not** automatically save its computation. For 18 blocks whose drop probabilities range linearly from 0 to 0.1, the expected number of retained contributions is 17.1, while all 18 branch calculations still execute. [Dropout, DropPath & Stochastic Depth](/learn/path/full-curriculum/dropout-droppath-stochastic-depth?module=deep-learning-fundamentals) develops the distinction between masking, expectation and execution.

## 3. From the block to a feature hierarchy

ConvNeXt starts with a 4×4 stride 4 convolution. Each initial output position reads one non-overlapping 4×4 input patch, then channel LayerNorm is applied. For a 224×224 image, this produces a 56×56 feature grid.

Four stages progressively reduce spatial resolution while increasing channel width:

| Tiny stage | Grid | Channels | Repeated blocks | What becomes possible |
| --- | --- | --- | --- | --- |
| 1 |56×56|96|3|Local detail represented at many positions|
| 2 |28×28|192|3|Broader combinations at fewer positions|
| 3 |14×14|384|9|More processing at a wider intermediate representation|
| 4 |7×7|768|3|A compact, semantically useful feature map|

Between stages, channel LayerNorm precedes a 2×2 stride 2 convolution. For classification, average the final map over its two spatial axes, apply LayerNorm to the resulting 768-vector, and use a linear classifier. Averaging before versus after a nonlinear normalization is a real ordering choice; these operations generally do not commute.

The hierarchy is useful beyond classification. A segmentation head can use fine-grid features to locate boundaries and coarse-grid features for context. A detector can attach heads to multiple scales. This explains why exposing stage outputs matters even when the original model's final head produces only one label.

Increasing channels while decreasing area also explains stage cost. Doubling \(C\) and dividing \(HW\) by 4 leaves the leading \(8HWC^2\) term unchanged **per block**. The depthwise term halves. Adding more blocks to the third stage concentrates work there. The Tiny stage depths are 3,3,9,3:18 blocks. Small, Base, Large and XLarge V1 configurations use 3,3,27,3:36 blocks.

The attached [complete architecture program](convnext-blocks.py) implements the block, stage transitions, residual masking, initialization, feature outputs and head. It checks large configurations using **meta tensors**: shapes and parameter counts are represented without allocating full model weights. It also runs small actual forward/backward calculations. Its computed V1 counts, with a 1,000-class head, are:

| V1 configuration | Initial width | Parameters | Conv/linear MACs at 224² |
| --- | --- | --- | --- |
| Tiny |96|28,589,128|4,455,531,264|
| Small |96|50,223,688|8,683,712,256|
| Base |128|88,591,464|15,354,729,472|
| Large |192|197,767,336|34,361,433,600|
| XLarge |256|350,196,968|60,921,030,656|

These are calculations for the stated topology, not training results. Input resolution changes activation sizes and work; it does not change a convolution's learned kernel count. Strided sampling also means shifting an image by one pixel need not simply shift its final features. Shared convolution weights do not make an entire downsampled classifier exactly translation invariant.

The block and the hierarchy are also separable choices. An **isotropic** variant keeps the same grid size and channel width through its repeated blocks, using an initial projection to establish that grid. It gives up the native four-scale output hierarchy. The original study tested such configurations too; their result asks whether the block remains useful without staged downsampling, not whether all tasks should discard multiple resolutions.

**Visual: unfold the hierarchy.** Show the same chosen position through patch extraction, successive grids, spatial neighborhoods and channel widths. Keep the grid sizes geometrically meaningful; show block repetition as a stack, not as extra downsampling.

## 4. Global response normalization: look across the feature map

Imagine two channels that produce almost the same spatial pattern. Both may vary strongly across an image, so neither is a “dead channel.” Yet they may offer redundant evidence. Conversely, a quiet channel might encode a rare useful pattern. Counting channels with nonzero variance does not measure representation quality.

ConvNeXt V2 adds **global response normalization**, or GRN, inside the expanded branch after GELU. It compares the spatial magnitude of each channel with the other channels in the **same specimen**. This differs from channel LayerNorm, which compares channels separately at each location.

For \(X\) with logical shape \(N,H,W,C\), define

\[
G_{n,c}=\sqrt{\sum_{h,w}X_{n,h,w,c}^2},\qquad
R_{n,c}=\frac{G_{n,c}}{\frac1C\sum_jG_{n,j}+\epsilon},
\]
\[
Y_{n,h,w,c}=X_{n,h,w,c}
+\gamma_c X_{n,h,w,c}R_{n,c}+\beta_c.
\]

The first reduction summarizes each whole channel map. The second compares these channel magnitudes. The result broadcasts back to every location. GRN does not subtract a spatial mean or force each channel to unit variance. The definition above matches the dense reference implementation. [Official GRN implementation](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py)

### A two-channel example you can calculate

Take channel A's two locations as \([3,4]\), channel B's as \([0,12]\). Their spatial lengths are 5 and 12. Their average length is 8.5, so relative responses are approximately 0.588235 and 1.411765.

Set \(\gamma_A=.5,\gamma_B=-.5,\beta=0\). Channel A becomes approximately \([3.882353,5.176470]\), and channel B becomes \([0,3.529413]\). Now edit only B's second value from 12 to 0. A's original values have not changed, but its relative response becomes almost 2, so its output becomes almost \([6,8]\).

That is global coupling through a statistic. It is not a new spatial convolution. It also shows why “GRN always boosts strong channels and suppresses weak ones” is misleading: the learned signs matter.

**Investigation: channel maps feeding a shared denominator.** Edit an actual cell in either map, predict the direction of A's output change, and watch its norm, the common denominator and the broadcast response update. Include the zero-scale case: when \(\gamma=\beta=0\), the same edit still changes the statistics but the GRN output is exactly its input.

Initial identity does not mean the layer is absent from learning. For the same two maps and loss \(L=\frac12\sum Y^2\), at \(\gamma=\beta=0\),

\[
\frac{\partial L}{\partial\gamma_A}\approx14.705881,\quad
\frac{\partial L}{\partial\gamma_B}\approx203.294094,\quad
\frac{\partial L}{\partial\beta}=[7,12].
\]

Those parameters can change on the first optimizer update. The initial input derivative equals \(X\) for this loss; afterward the learned GRN response changes the input derivative too. The [author calculations](author-checks.py) check the scale gradients against independent central differences.

V2 removes the V1 LayerScale and adds two GRN vectors at width \(4C\). Thus its block has \(8C^2+65C\) parameters: \(8C\) added and \(C\) removed, a net \(7C\). At \(C=96\), this is 79,968 parameters; 672 more than V1. GRN itself starts as identity; removing LayerScale does not make the **whole V2 residual branch** tiny.

## 5. Learn from missing pixels without giving away the answer

A supervised digit classifier receives an image and its class during training. A masked reconstruction model receives only selected image regions and learns to predict the missing regions. The original image supplies a training target even when no class label is used for that training stage. This is **self-supervised learning**: the training signal is constructed from the data.

The central constraint is informational. If the hidden pixels enter the encoder through a convolution, normalization statistic or another route, a good reconstruction may reflect access to the answer.

### Separate what is visible from what is scored

Let \(M\) be 1 at visible pixels and 0 at hidden pixels. An input-masking operation forms \(M\odot x\), where \(\odot\) is elementwise multiplication. A hidden-pixel loss is

\[
L=\frac{\sum_{i:M_i=0}(\widehat{x}_i-x_i)^2}
{\#\{i:M_i=0\}}.
\]

The encoder sees the visible values. The loss compares predictions with the original hidden targets. Editing a hidden target therefore can change the loss without changing the prediction. This is correct behavior, not a contradiction.

In a multistage convolutional encoder, masking the raw input alone is not the same as maintaining a fixed set of active feature locations. Convolution can write features into inactive locations; biases and channel transformations can make zero inputs nonzero. A masked-dense implementation must keep its intended active set masked at the relevant operations. A sparse implementation explicitly represents and computes on active coordinates. Their runtime costs and normalization behavior must be checked separately.

The published **fully convolutional masked autoencoder**, FCMAE, masks 60% of 32×32 input patches, matching the final encoder-grid granularity, and propagates that mask through the hierarchy. Its lightweight decoder receives encoded visible features and mask tokens at missing positions. Its loss uses patch-normalized hidden targets. Our small experiment below preserves the visibility/target distinction while deliberately using smaller patches and a simpler loss. [FCMAE construction](https://arxiv.org/pdf/2301.00808)

**Visual: two paths from the same image.** One path passes through the visibility mask to the encoder. The other retains the original pixels until the loss. Join them only at the comparison between predicted and target hidden pixels. In the decoder, distinguish a learned mask token from an encoded visible patch.

### Why a second evaluation is needed

An encoder might learn local interpolation that reconstructs textures well but does not separate object categories. To ask whether labels are accessible in its features, freeze the encoder, extract representations, and fit a small supervised classifier using training labels. A **linear probe** tests what a linear readout can use. It differs from fine-tuning, which updates the encoder too.

Neither reconstruction error nor a channel-diversity statistic can substitute for that task evaluation. Even a successful probe on a small development split does not establish deployment performance.

## 6. An actual masked-digit experiment

The [offline CSV](digits-400.csv) contains 400 real 8×8 optical digit images,40 per class, drawn from UCI's Optical Recognition of Handwritten Digits dataset through scikit-learn's local copy. They are not MNIST images. Each integer pixel lies in 0–16; divide by 16 using the known scale. Preserve the [dataset attribution, subset construction and split record](data-provenance.md).

The unit here is an image specimen. Before splitting, the program checks 400 unique source IDs and 400 distinct pixel vectors. Writer identifiers are unavailable, so this is not an independent-writer assessment. It reserves 120 stratified development images and trains on 280, using split seed 22. Development images are excluded even from unlabeled reconstruction training.

### The small model and the controlled difference

The input 8×8 image is divided into sixteen 2×2 patches. Exactly six are visible and ten hidden:62.5% hidden, rather than the paper's 60%. A 2×2 stride 2 stem creates a 4×4 grid with 12 channels. Two ConvNeXt-style encoder blocks use 3×3 depthwise filters, channel LayerNorm,12→48→12 channel mixing and residual addition. A one-block decoder plus a pixel head reconstructs 8×8 pixels.

We compare two variants that differ only in whether the encoder expansion includes GRN. **Neither has LayerScale**, and both use the same small decoder. These are paired GRN experiments, not miniature reproductions of every difference between official V1 and V2.

The training program uses AdamW with learning rate 0.002, weight decay 0.01 and 600 full-batch updates. Weight decay applies to all parameters in this teaching experiment; this is not the paper's optimizer grouping. Each update generates a new six-visible-patch mask. Each paired seed uses identical initial shared tensors and the same mask sequence; GRN's additional scale/shift vectors start at zero.

For evaluation, four fixed masks per specimen allow comparisons on the same missing pixels. The loss is raw normalized-pixel MSE on the 40 hidden pixels per image, not the paper's patch-normalized target loss. No augmentation, DropPath, checkpoint search or model selection is used. Steps 0,1,100,300,600 are recorded; only the declared final step supplies the comparison.

### Run the complete program

Download [masked-reconstruction.py](masked-reconstruction.py) and [digits-400.csv](digits-400.csv) into the same directory. In a Python environment with PyTorch, NumPy and scikit-learn installed, run:

```bash
python masked-reconstruction.py
```

The script sets one PyTorch CPU thread and requires no network access or pretrained checkpoint. It contains the full model, mask generator, loss, train/development split, optimizer loop, fixed-mask evaluation and frozen-feature probe. It writes [calculated-inputs.json](calculated-inputs.json); the supplied copy contains the actual author run on Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.3.5 and scikit-learn 1.9.1. Small numerical differences across library/platform versions are possible.

The core masking/loss excerpt is worth reading before running the complete file:

```python
def masked_mse(predictions, targets, visible):
    hidden_pixels = (1-visible).repeat_interleave(2,2).repeat_interleave(2,3)
    return ((predictions-targets).square()*hidden_pixels).sum()/hidden_pixels.sum()
```

Here `visible` has shape \(N,1,4,4\). Repeating each grid location twice along each spatial axis makes its 2×2 pixel patch share the same visibility. The numerator sums error only where the mask is hidden; the denominator is the number of those pixels. An all-visible mask would have denominator zero and is not an allowed reconstruction-loss input. All-visible images are valid for feature extraction, where this loss is not called.

Inside the encoder, the input is masked before the stem; inactive feature positions are suppressed after spatial filtering, expansion and residual addition. The decoder can fill missing positions, as it must to predict them. [Independent scalar-loop checks](author-checks.py) reproduce all four saved real examples to within \(3.3\times10^{-7}\) of the stored outputs.

### What actually happened

A training-mean-image baseline predicts the same image regardless of visible content. Its development masked MSE is 0.071914. The learned models do better on this reconstruction criterion:

| Paired seed | GRN absent: masked MSE | GRN present: masked MSE | Frozen probe correct, absent / present |
| --- | --- | --- | --- |
|1|.052188|.052313|115/120 /116/120|
|2|.052336|.051906|115/120 /116/120|
|3|.051670|.051488|115/120 /116/120|

The probe standardizes each extracted feature using training statistics, then fits logistic regression with \(C=1\). It sees the clean image through a frozen encoder. A separate standardized logistic regression fitted directly to the 64 raw pixels gets 118/120 correct. All six feature probes fit the 280 training labels perfectly.

Several conclusions now become possible, and several do not:

- The learned reconstructions beat this training-mean-image baseline on the specified missing-pixel task.
- GRN does not improve reconstruction in every seed. Its probe gets one extra image correct in each paired run on this one development split.
- The simpler raw-pixel classifier gets more development labels right than any frozen-feature probe here.
- We did not compare against a random untrained encoder, vary the label budget, fine-tune the encoder or reserve an untouched test. These results do not isolate the benefit of pretraining or establish a universal architecture ranking.

Repeated seeds vary initialization and training masks; they are not independent new datasets. The development data are now consumed by interpretation. If you use these findings to choose a model, obtain a separate appropriate final evaluation before making a deployment claim.

### Look at a specimen, then intervene

The packet retains the two seed 1 models' complete weights and the first two development examples, source 251/label 4 and source 40/label 9. Display their original, visible-only input, reconstruction and hidden-pixel squared error side by side. Show the raw reconstruction values in a numeric view; a clipped display palette must not silently clip the values used for MSE.

For source 251 with GRN, flipping hidden pixel (row 0, column 0) from 0 to 1 leaves every prediction unchanged, while masked MSE changes from approximately 0.054047 to 0.080387. Flipping the visible pixel (0,2) instead changes the reconstruction, with maximum absolute output change about 0.506610. The paired model without GRN has the same hidden-input invariance and a different visible-input response.

**Investigation: which side of the information boundary did you change?** Before revealing a result, record whether you expect the prediction, the target loss, both or neither to change. Edit a pixel or swap a visible and hidden patch while keeping six patches visible. Then compare the new prediction with the original. Changing a specimen's displayed class label alone must not affect reconstruction, because the model never receives that label.

### Inspect features without overinterpreting a diagnostic

The program measures spatial cosine distance between distinct nonzero channel maps at the final encoder expansion. For maps \(a,b\), it uses \((1-\cos(a,b))/2\); identical positive-direction maps have distance 0. Near-zero maps are counted separately instead of assigning an arbitrary cosine.

Every run has zero near-zero-channel fraction under the declared threshold. Mean nonself distances with GRN are slightly higher in seed 1 and slightly lower in seeds 2–3. Thus “more active channels” does not explain the small probe difference, and this diagnostic is not a quality score. Inspect what a statistic measures before attaching an architectural story to it.

## 7. Deeper routes: deploy, reparameterize, or combine mechanisms

### Use a checkpoint as a complete input/output contract

For a practical pretrained model, record the exact library/version, architecture, weight identifier, input transforms, output classes and intended downstream evaluation. A weight file is not useful independently of this contract.

The inspected [Torchvision ConvNeXt Tiny documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html) exposes `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` and its `transforms()`. Its reported 82.52% ImageNet result belongs to Torchvision's modified recipe; it is not the original paper's 82.1% result. For those weights, use the provided resize/crop/normalization and category metadata. Replace the classifier and evaluate your task if the target classes differ; the ImageNet head does not acquire new classes by renaming its outputs.

The attached architecture program offers `return_features=True` for the four stage maps. This is the useful interface for a downstream head: inspect its exact shapes and scale meaning before connecting it. Full transfer-learning training and split design belong to [Transfer Learning & Fine-Tuning Strategies](/learn/path/full-curriculum/transfer-learning-fine-tuning-strategies?module=deep-learning-fundamentals); no pretrained download is required for this lesson's executed experiment.

Logical layout and memory layout are different. `permute(0,2,3,1)` makes a view whose dimension order is NHWC. `to(memory_format=torch.channels_last)` retains the logical NCHW shape while changing storage strides. A permutation itself does not copy values, but later operations may need to materialize a suitable layout. Measure the complete workload before promising a speedup. [PyTorch's channels-last tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)

### Fold several training branches into one inference kernel

A useful deployment idea is **structural reparameterization**: train using several linear branches, then combine them into a simpler equivalent inference operation. This is different from changing the trained function through approximate compression.

Suppose a convolution \(z=Wx+b\) is followed by BatchNorm using fixed evaluation statistics \(\mu,v\) and learned \(\gamma,\beta\). Then

\[
\operatorname{BN}(Wx+b)=
\left(\frac{\gamma}{\sqrt{v+\epsilon}}W\right)x+
\left(\frac{\gamma(b-\mu)}{\sqrt{v+\epsilon}}+\beta\right).
\]

Each output channel gets its own multiplier and bias. Fold each linear branch this way. Pad a smaller odd-sized kernel with zeros so its center aligns with the larger kernel; represent an identity path as a center coefficient 1 for corresponding input/output channels. Add the aligned kernels and biases.

This works only when the branches have compatible input/output shapes, stride, coordinate alignment and groups, and the normalization statistics are fixed. A shared activation **after** the summed branches can remain after the fused convolution. Separate nonlinear activations inside branches generally cannot be folded this way.

The author calculation folds a 3×3 branch, a 1×1 branch and identity on a 5×5 constructed input. Separate and fused outputs agree within \(2.9\times10^{-14}\); the center output is 111.049826. Moving separate ReLUs inside branches creates a different function. The supplied counterexample differs by more than 23 in one output. These are exact-function checks, not latency benchmarks.

Large-kernel models such as [RepLKNet](https://arxiv.org/pdf/2203.06717) use this idea to aid training while retaining a large spatial filter at inference. Its main blocks use a parallel 5×5 branch with the large kernel. [MobileOne](https://arxiv.org/pdf/2206.04040) applies related deployment-oriented reasoning to small blocks. A 31×31 depthwise kernel reads a broader dense stencil, but its coefficients remain shared learned values; they are not automatically input-dependent attention weights.

**Investigation: collapse the branch graph.** Edit an actual kernel coefficient or BatchNorm statistic, predict a selected output and verify the separately evaluated and fused paths. Insert a branch-local nonlinearity to expose the equivalence boundary. Display the resulting fused kernel as a spatial stencil, not just an equation.

### When a hybrid is a useful hypothesis

Attention forms a weighted sum of value vectors, with weights obtained from the current input and query. Convolution uses a learned spatial stencil shared across inputs. Both can mix spatial evidence, but attention layers also include channel projections, and their complete block topology differs from a ConvNeXt block.

A hybrid can use local convolution where the grid is large and more global input-dependent interactions after the grid has shrunk. [CoAtNet](https://arxiv.org/pdf/2106.04803) studies such stage arrangements. [MaxViT](https://arxiv.org/pdf/2204.01697) alternates local block attention with a sparse grid arrangement that connects distant positions. These are concrete choices about which positions communicate, not evidence that adding attention anywhere must help.

For a factory-defect application, local texture may matter alongside long-range alignment between repeated parts. A ConvNeXt feature hierarchy, a larger convolutional receptive field and a hybrid interaction pattern are competing hypotheses. Split by production unit or scene when multiple images share an origin, establish a simple baseline, then examine the errors that distinguish those hypotheses. A smaller-input label classifier and a high-resolution localization system need different evaluation and memory budgets.

The later attention and vision-transformer lessons develop the weighted-sum mechanism in full. Here the useful connection is to ask **which evidence can reach this output, through which operation, at what resolution and cost?**

## 8. Practice: reason about a changed design

### 1. Catch a shape-correct normalization error

A tensor has shape \(N,32,7,32\). Someone applies `nn.LayerNorm(32)` directly and says it normalizes channels. Explain what it actually does and give a correct channel-normalization route.

<details><summary>Hint</summary>

LayerNorm matches its normalized shape to the trailing dimensions; equal dimension sizes can conceal the wrong axis.
</details>

<details><summary>Solution</summary>

It normalizes the final width axis of length 32, independently for each specimen/channel/row. Permute to \(N,7,32,32\) with the original channel axis last, apply LayerNorm(32), and permute back. Name the axes explicitly: both trailing dimensions happen to be 32 after the permutation, so shape inspection alone is insufficient. The unchanged intended output shape does not prove the operation is correct.
</details>

### 2. Change the expansion and count what changed

Use a 5×5 depthwise kernel, input/output width 64 and expansion factor 3 in a V1-style block. Include all biases, channel LayerNorm and LayerScale. How many parameters and convolution/linear MACs does the block use on a 14×14 grid?

<details><summary>Hint</summary>

Write the two pointwise matrices and their different bias lengths before adding the small vectors.
</details>

<details><summary>Solution</summary>

Depthwise has \(25C+C\); LayerNorm \(2C\); the two linear layers \(3C^2+3C\) and \(3C^2+C\); LayerScale \(C\). Total \(6C^2+33C=26,688\). Conv/linear MACs are \(196(25\cdot64+6\cdot64^2)=5,130,496\). Normalization, activations and additions are excluded from that declared operation count.
</details>

### 3. Predict a cross-channel effect

For the GRN example, keep \(\gamma_A=.5\), set \(\gamma_B=0\), and change B's second value 12→24. Does A's first output rise or fall? Does setting both scales to zero make the statistics stop changing?

<details><summary>Hint</summary>

A's norm stays 5; the denominator compares it with B's new norm 24.
</details>

<details><summary>Solution</summary>

A's relative response becomes \(5/(14.5+\epsilon)\), smaller than before, so its first output falls to approximately \(3(1+.5\cdot5/14.5)=3.517241\). Zero scales remove the response-dependent contribution from the output; the norms still change internally. GRN's identity initialization and its statistic computation are different facts.
</details>

### 4. Diagnose suspiciously good reconstruction

The encoder masks pixel values, but first subtracts each full image's mean, calculated using visible and hidden pixels. A hidden-pixel edit changes the model's prediction. Is this necessarily a defect in the convolution code? Propose a repair and a direct check.

<details><summary>Hint</summary>

Ask whether the original hidden value can reach a visible input through preprocessing.
</details>

<details><summary>Solution</summary>

The full-image mean carries hidden information into the centered visible pixels. The convolution can be implemented correctly while the input contract leaks. Use a fixed permitted scale, training-set statistics learned without the evaluated image, or a clearly specified visible-only statistic. Change one hidden target while holding mask and visible values fixed; predictions should stay unchanged under the repaired contract. The hidden-target loss can still change. Target-only patch normalization is a separate path and must not be reused to normalize the encoder input.
</details>

### 5. Choose the conclusion supported by the experiment

A colleague reports, “GRN learns more diverse channels, therefore it improves digit recognition and should replace the raw baseline.” Use the saved outcomes to rewrite the conclusion and name one additional experiment that would answer a genuinely missing question.

<details><summary>Hint</summary>

Compare paired reconstruction results, paired diversity results and the raw-pixel classifier separately.
</details>

<details><summary>Solution</summary>

On this development split the GRN variants' probes get 116/120 versus 115/120, while the raw-pixel probe gets 118/120. GRN does not increase the measured diversity in all seeds and does not improve masked MSE in seed 1. This supports a small paired probe difference in this setting, not the proposed causal explanation or replacement decision. For a pretraining question, predeclare matched random-encoder and pretrained-encoder probes with the same architecture and label budget. For a deployment choice, select using development data and evaluate once on a new appropriately grouped final set.
</details>

### 6. Check a branch-fusion boundary

For scalar input \(x=-2\), compare \(\operatorname{ReLU}(x)+\operatorname{ReLU}(-x)\) with \(\operatorname{ReLU}(x-x)\). Can the separate branch activations be removed while preserving the function?

<details><summary>Hint</summary>

Apply each activation before summing in the first expression.
</details>

<details><summary>Solution</summary>

The first gives \(0+2=2\); the second gives 0. Linear branch kernels can be added only where the intermediate operations permit the algebra. Keeping one shared activation after an equivalent linear sum is valid; replacing separate nonlinear branches with that shared activation is a different model.
</details>

### 7. Budget a finer stem

Replace a 4×4 stride 4 stem with a 2×2 stride 2 stem on 224×224 inputs while keeping all later stage widths and depths unchanged. What happens to the first grid, the block MACs and the head's parameter count?

<details><summary>Hint</summary>

Track spatial area through the later stride 2 transitions. The classification head receives an averaged channel vector.
</details>

<details><summary>Solution</summary>

The first grid becomes 112×112 instead of 56×56. Every corresponding later grid has twice the side length, so block conv/linear MACs and transition MACs become four times larger. Stem MACs happen to remain equal here: four times as many outputs each use one quarter as many spatial weights. Stem parameters fall, but the global-average-pooling classifier's input width and head parameter count stay unchanged. Activation memory also grows; accuracy and latency cannot be deduced from this count alone.
</details>

### 8. Design an informative masked-learning extension

You have 20 labeled specimens per class and many unlabeled images from repeated capture sessions. Propose a comparison to ask whether masked pretraining helps when labels are scarce. Include the split unit, baseline, preprocessing, model comparison and final evaluation.

<details><summary>Hint</summary>

Unlabeled access is still access to data. Keep the question of representation learning separate from the number of labels used by the readout.
</details>

<details><summary>Solution</summary>

Split capture sessions before training so related views do not cross partitions. Restrict both supervised and unlabeled pretraining inputs to training sessions. Fix the same labeled subset and feature-readout recipe for raw pixels, a random frozen encoder and the pretrained frozen encoder; optionally add a separately declared end-to-end supervised model. Fit preprocessing on training data, select any settings on development sessions, and report the chosen protocol once on held-out sessions. Record reconstruction and downstream task outcomes separately and repeat paired seeds. Do not call extra unlabeled access “the same data budget” unless that is explicitly the question.
</details>

## 9. Readiness, connections and other ways to learn

You are ready to move on when you can explain a block using spatial and channel operations, mark the inputs used by each normalization statistic, trace a masked target without leaking it into the encoder, and state what the actual experiment demonstrates. Memorizing every model size or reproducing ImageNet training is not required.

Next in this module is [Capsule Networks](/learn/path/full-curriculum/capsule-networks?module=deep-learning-fundamentals). It asks a different representation question: instead of only scalar feature activations, can groups of values represent a part's properties and help parts agree on a whole? ConvNeXt does not become obsolete at that transition; the proposed inductive bias changes.

Useful routes through the references:

- [Official ConvNeXt source](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py): inspect the block, channel-first LayerNorm, stage transitions and initialization after working through §§2–3. Code reading is particularly useful for distinguishing logical axes from the diagram.
- [ConvNeXt V1 paper](https://arxiv.org/pdf/2201.03545): read §2 as an experiment-design argument, then compare the small-regime roadmap in Appendix C with the final result table. The roadmap's roughly 82.0% average and final 82.1% checkpoint report are different records.
- [ConvNeXt V2 paper](https://arxiv.org/pdf/2301.00808) and [authors' CVPR slides](https://cvpr.thecvf.com/media/cvpr-2023/Slides/22892_lw8881R.pdf): the paper supplies the mask/GRN details; the slides offer a visual second pass through masking, feature maps and co-design. Their full-scale experiments differ from our bounded dense-masked probe experiment.
- [PyTorch channels-last tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html): a hands-on storage-stride explanation. Its hardware results belong to the measured configurations; use the concepts to inspect your own workload.
- [Dive into Deep Learning: Designing Convolution Network Architectures](https://en.d2l.ai/chapter_convolutional-modern/cnn-design.html): study the AnyNet→RegNet design-space argument as an alternative to memorizing model families. The chapter's broader historical rankings are time-specific; its distribution-of-designs perspective is the useful complement here.
- [RepLKNet](https://arxiv.org/pdf/2203.06717), [MobileOne](https://arxiv.org/pdf/2206.04040), [CoAtNet](https://arxiv.org/pdf/2106.04803) and [MaxViT](https://arxiv.org/pdf/2204.01697): optional mechanism-focused extensions for large kernels, inference-time branch folding, stage ordering and local/global communication. Read the ablation conditions before generalizing a result.

All local experimental numbers come from the accompanying programs and retained results. The dataset and its transformations are attributed in [data provenance](data-provenance.md). The figures and investigations described in this manuscript are specified for later website implementation.
