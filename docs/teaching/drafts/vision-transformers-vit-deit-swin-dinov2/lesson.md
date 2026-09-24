# Vision Transformers (ViT, DeiT, Swin, DINOv2)

**Explore as you read.** Edit patch pixels/projection coefficients, patch-position swaps, window/shift geometry, teacher/student logits and feature angles. Update patch contributions, actual frozen-model logits/maps, spatial dependency sets, teacher targets/gradients and relational loss immediately. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to separate image edits from position changes, direct neighbors from multi-hop reach and target sharpening from learned quality.


A handwritten digit is an arrangement of marks. A loop near the top and a loop near the bottom can suggest an eight; the same dark pixels rearranged across the page can suggest something else. An image model needs both the appearance of small regions and their relationships.

A **Vision Transformer**, or ViT, turns small image regions into vectors, lets those vectors exchange information, and uses the resulting representation to make a prediction. The Transformer machinery is familiar. The new questions are visual: how do pixels become tokens, how is location preserved, how can windows communicate, and what should a model learn before we have labels for a particular task?

We will build these ideas on an actual, small image problem: recognizing handwritten digits from 8×8 intensity arrays. We will also work through the mechanisms used in larger systems. The small study is fully reproducible; its accuracy is evidence about that study, not a substitute for evaluating a pretrained vision model on your own images.

**First pass:** read §§1–2, the practical parts of §3, §§4–5, §6 through “What DINOv2 adds,” and §§7–8. Attempt the core exercises in §9. The marked deeper branches develop symmetry, operation counts, Gram geometry and model adaptation; they extend the same ideas without hiding a prerequisite for the first pass.

You need vectors, a linear layer, softmax, cross-entropy and the residual Transformer block. [Self-Attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals) and [Transformer Block Architecture](/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals) teach those mechanisms. We refresh their visual meaning here. The immediately preceding [Sparse and Linear Attention](/learn/path/full-curriculum/sparse-linear-attention-variants?module=deep-learning-fundamentals) explains why restricting who can communicate changes an operator; Swin makes that restriction follow an image grid.

By the end, you should be able to track an image through a patch Transformer, explain what each special token does, calculate the cost of a resolution change, trace a shifted-window connection, distinguish supervised distillation from self-distillation, and design an honest evaluation of image features.

## 1. Turn an image into a sequence without losing the meaning of its axes

An RGB image of height 224 and width 224 has 224×224=50,176 pixel positions, each containing three color values. It therefore contains 150,528 scalar values. If each RGB pixel became one token, the sequence would have 50,176 tokens, not 150,528. One attention head would have about 2.52 billion query–key pairs.

Instead, divide the image into nonoverlapping 16×16 patches. There are 14 rows and 14 columns of patches:196 tokens. Each patch contains 16×16×3=768 scalar values. Flatten those values in a fixed order, then apply the same learned linear projection to every patch. A patch token is a vector describing a region; it is neither a text word nor an object label.

For height $H$, width $W$, channels $C$, square patch side $P$, and embedding width $d$:

$$
N=(H/P)(W/P),\qquad x_j\in\mathbb R^{P^2C},\qquad e_j=x_jE+b,\qquad E\in\mathbb R^{P^2C\times d}.
$$

Here $H$ and $W$ are divisible by $P$. Otherwise choose and document resizing, cropping or padding; silently discarding the last pixel strip changes the input. The width $d$ need not equal $P^2C$. A projection to a smaller width can discard information, while a larger width does not create new measured pixels.

**Inline figure: an image becoming patch vectors.** Keep the image grid visible while four selected patches unfold into rows. Connect every scalar to its source coordinate and connect every patch to the *same* projection. The result is a sequence with a retained grid index, not a bag of detached decorative tiles.

### A projection you can calculate

Take this constructed one-channel 4×4 image. Coordinates and patch order run from the top left, across each row:

```text
 0  1 |  2  3
 4  5 |  6  7
------+------
 8  9 | 10 11
12 13 | 14 15
```

With 2×2 patches, the flattened rows are `[0,1,4,5]`, `[2,3,6,7]`, `[8,9,12,13]`, `[10,11,14,15]`. Use two output features:

$$e_{j,1}=x_{j,1}+x_{j,4}+0.5,\qquad e_{j,2}=x_{j,2}-x_{j,3}+1.$$

The first patch becomes `[5.5,−2]`; the four embeddings are `[5.5,−2]`, `[9.5,−2]`, `[21.5,−2]`, `[25.5,−2]`. The first feature responds to one diagonal sum. The second compares two other locations. Real models learn many such combinations; an individual feature need not retain this simple interpretation.

The projection is exactly a convolution with kernel size $P$ and stride $P$, provided the weights, bias and flattening order match. This is a useful connection to the earlier convolution lesson: “patch projection” and a particular convolution are two implementations of the same operation.

```python
import torch
from torch.nn import functional as F

image = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
weight = torch.tensor([[1., 0., 0., 1.], [0., 1., -1., 0.]], dtype=torch.float64)
bias = torch.tensor([.5, 1.], dtype=torch.float64)
patches = F.unfold(image, kernel_size=2, stride=2).transpose(1, 2)
explicit = patches @ weight.T + bias
convolution = F.conv2d(image, weight.reshape(2, 1, 2, 2), bias, stride=2)
convolution = convolution.flatten(2).transpose(1, 2)
print(explicit)
print(torch.equal(explicit, convolution))
```

Executed output is the four embeddings above and `True`. Creating two unrelated random layers and checking their output *shapes* would not establish this equality. For RGB, PyTorch `unfold` uses channel-major patch coordinates; preserve that order when reshaping a linear layer's weights.

**Investigation — edit the image, inspect the token.** Edit a pixel or projection coefficient and watch its signed contributions and output features update. The fresh patch differs from the worked example. Try changing the patch while keeping both output features unchanged; the null exposes information lost by the projection.

Why not always choose the smallest possible patches? Halving $P$ at fixed image size quadruples the token count. It preserves finer spatial distinctions at a higher computation and memory cost. Overlapping projections, convolutional stems and learned grouping are meaningful alternatives; nonoverlapping 16×16 patches are a design choice, not a proof that all other tokenizations are inferior. [The original ViT method](https://arxiv.org/html/2010.11929v2#S3) provides the starting architecture and a convolutional hybrid alternative.

## 2. Let regions exchange information, then read out the image

The patch projection produces $N$ vectors. A plain classification ViT prepends one learned **class token**, written CLS. This is an initially image-independent parameter vector. It becomes image-dependent as attention lets it read information from the patches.

Add a positional vector to every token, including a separate slot for CLS:

$$Z_0=[c;e_1;\ldots;e_N]+P_{\mathrm{pos}}\in\mathbb R^{(N+1)\times d}.$$

Each pre-normalization block then performs

$$U=Z+\operatorname{MHA}(\operatorname{LN}(Z)),\qquad
Z_{\mathrm{next}}=U+\operatorname{FFN}(\operatorname{LN}(U)).$$

Layer normalization acts across the $d$ features of each token. In each attention head, a query asks what information this token needs, keys determine compatibility, and values supply the information to mix. For one query $i$:

$$a_{ij}=\frac{\exp(q_i^Tk_j/\sqrt{d_h})}{\sum_m\exp(q_i^Tk_m/\sqrt{d_h})},\qquad o_i=\sum_j a_{ij}v_j.$$

The sum includes CLS and all patches in ordinary full-image classification. There is no temporal “future” that requires a causal triangle. Padding masks or deliberately restricted windows are separate reasons to exclude pairs. The FFN then transforms each token individually using shared weights; it does not add another spatial communication step.

After the last block, normalize the CLS vector and project it to class logits, the real-valued scores used by cross-entropy. Softmax converts them to a distribution over the trained label set. Training changes the patch projection, positions, CLS vector, block weights and head so that this entire path helps predict labels.

**Inline figure: two coordinate systems at once.** Above, show the spatial patch grid. Below, show its sequence plus CLS. A selected CLS query connects to a full row of attention weights; a separate panel shows its weighted value sum. Distinguish attention weights from final class probabilities. Both sum to one in their own domains, but one is a distribution over tokens and the other over classes.

### Why locations must enter somewhere

Suppose the image consists of a dark upper patch and a light lower patch. Swapping them changes the arrangement. A Transformer receiving only their appearance vectors has no built-in row label that identifies “upper.” Learned positions supply that missing association.

There are two different swaps:

1. Swap patch content but leave positional vectors in their original slots. This changes which content is associated with which location and can change the prediction.
2. Reorder the already combined content-plus-position rows together. This just changes how the same located patches are listed. In an otherwise permutation-equivariant encoder with CLS kept fixed, the CLS result stays the same.

The distinction makes an excellent debugging test. If the second operation changes the mathematical result, look for an unaccounted order-dependent operation, mask or incorrect index. Small floating-point differences are expected.

In our saved plain ViT, swapping patch 5 with patch 6 on the first test image while holding positions fixed changes some logits by as much as 7.8398. Moving content and position together changes logits by less than 0.000001. These are calculated interventions on the same trained weights, not two separately trained models.

**Deeper: prove the reordering claim.** Let $R$ be a permutation matrix that fixes the CLS row. Shared tokenwise maps give $Q'=RQ$, $K'=RK$, $V'=RV$. The score matrix becomes $R(QK^T)R^T$. Row-softmax respects this simultaneous reordering, so $A'=RAR^T$ and $A'V'=RAV$. Tokenwise normalization, FFNs and residual additions preserve the same property. Thus patch outputs permute and CLS is unchanged. The proof assumes the mask and other positional structure are moved consistently, and excludes independently resampled dropout masks. It does not imply invariance to moving objects in pixel space.

### CLS is a readout choice

A model can instead average patch representations and train a classifier on that average. Then every patch contributes directly to the pooled vector, although earlier attention has already mixed their information. CLS provides a learned accumulation site; mean pooling provides a prescribed accumulation rule. Neither rule is universally better. Changing a trained checkpoint's pooling rule without adapting the head changes the representation the head receives.

ViT also has inductive biases: a patch grid, shared projection, shared tokenwise weights, the chosen position scheme and training augmentations. Compared with a stack of small convolutions, its attention layers impose less local spatial structure. “Less image-specific bias” is useful; “no inductive bias” is not.

The preceding [ConvNeXt lesson](/learn/path/full-curriculum/convnext-modern-cnn-designs?module=deep-learning-fundamentals) shows that CNN training and architecture also evolve. The comparison should be between actual systems and data protocols, not a story in which convolution stopped working once attention appeared.

## 3. Change resolution deliberately

At 224×224 with 16×16 patches, the patch grid is 14×14. At 384×384, it is 24×24. The patch projection still consumes 768 values per patch, so its weights can remain unchanged. The learned table of spatial positions, however, needs 576 patch slots rather than 196.

Treat the old patch-position vectors as a 14×14 grid with $d$ channels. Interpolate that grid to 24×24, flatten it back to a sequence, and keep the CLS position separate. This adapts the representation's coordinates. It does not recover image detail that was already lost in a low-resolution source image, and it does not guarantee that a model trained at one resolution has equal accuracy at another.

For a rectangular image, retain both grid dimensions. A sequence of 192 patches could come from 12×16 or 8×24; the square root of 192 cannot recover either layout. Additional DIST or register tokens are also nonspatial slots; follow the checkpoint's exact token-construction convention rather than reshaping every token into a square.

The complete `resize_positions` function in [vision-mechanisms.py](vision-mechanisms.py) takes explicit old and new grid shapes and a prefix count. It uses bicubic interpolation with `align_corners=False`, preserves prefixes exactly, and returns an exact copy for an unchanged grid. Those settings are part of the method. Bicubic interpolation can overshoot; it is not always a convex average that reduces feature norms. Some checkpoints use different interpolation details or position schemes.

Changing the patch side from 16 to 14 is a different operation: the projection's input width changes from 768 to 588 for RGB. Resizing only the position table cannot fix that mismatch. Methods designed to adapt patch embeddings can address it, but they change more than a position lookup.

### Deeper: count work without inventing a speed measurement

Use **one multiply–accumulate (MAC)** as our matrix-operation counting unit. If a report counts multiplication and addition as separate floating-point operations, its corresponding dense matmul count is roughly twice this number. Neither convention includes all execution costs.

Let $S=N+1$, width $d$, and FFN expansion ratio 4. Per block:

| Operation | MACs | Why |
| --- | ---: | --- |
| Q, K, V and attention output projections | $4Sd^2$ | Four width-to-width linear maps |
| FFN's two linear maps | $8Sd^2$ | $d\to4d\to d$ at each token |
| Scores and weighted values | $2S^2d$ | $QK^T$ and $AV$ across all heads |

Splitting a fixed width into more heads does not multiply these leading totals by the head count: $h d_h=d$. Biases, normalization, softmax, activation, residuals, memory traffic and kernel overhead are excluded here.

For a 12-block ViT-B-width model, $d=768$, RGB patches 16, one CLS and a 1,000-class head, the calculated counts are:

| Input | Tokens $S$ | FFN MACs per block | Pair MACs per block | Pair fraction of block matmul MACs | Whole-model matmul MACs |
| --- | ---: | ---: | ---: | ---: | ---: |
|224×224|197|0.930 billion|0.0596 billion|4.10%|17.56 billion|
|384×384|577|2.723 billion|0.5114 billion|11.13%|55.48 billion|
|512×512|1,025|4.837 billion|1.6138 billion|18.20%|107.03 billion|
|1,024×1,024|4,097|19.332 billion|25.7824 billion|47.06%|659.78 billion|

The whole-model sum includes patch projection and the classifier, with the same exclusions. Doubling image height and width approximately quadruples tokenwise work and multiplies dense pair work by sixteen. The FFN remains linear in token count, not quadratic.

**Inline calculated graph:** plot the projection, FFN and pair counts against image side length with their actual equations. A fixed-dimension table remains available. These are operation counts, so the vertical axis must not say milliseconds or measured throughput.

At 224, optimizing pair attention alone cannot eliminate the much larger projection/FFN work. At larger resolutions, pair work and attention intermediates become more significant. Fused exact attention can reduce intermediate memory without changing the dense mathematical pair count. Local windows change the allowed pairs. That is our next mechanism.

## 4. Swin: change the neighborhoods, then build a hierarchy

Imagine dividing a patch grid into rooms. Inside a room, every token can exchange information with every other token. If every layer uses the same rooms, the attention layers never connect different rooms. Swin alternates the boundaries so that the next layer brings some previously separated tokens together.

These rooms are **windows**, measured in patch tokens, not original pixels. With an $M\times M$ window, each query attends to at most $M^2$ keys. For an $h\times w$ token grid, window pair work is $2hwM^2d$ MACs instead of $2(hw)^2d$. This is linear in the number of tokens when $M$ is fixed. Projections and FFNs still need to be calculated.

### Follow one two-layer connection

Use a constructed 4×4 token grid and 2×2 windows:

```text
A B | C D
E F | G H
----+----
I J | K L
M N | O P
```

In the first layer, F reads A, B, E and F. In the second layer, shift the window boundaries by one row and one column. F now shares a window with G, J and K. Those four *first-layer states* already contain information from all four original windows. Consequently, F's second-layer state can depend on all 16 original tokens.

This is a path through two attention layers. It is not the union of only F's two direct neighbor lists. If each layer simply averages its allowed inputs, every original token contributes $1/16$ to F after these two layers. A corner such as A remains in a clipped shifted window and has only four original contributors at this depth. Receptive fields depend on the query location, boundary, window size and number of layers; there is no universal “13×13 after two blocks” rule.

**Inline spatial trace:** show the original rooms, the shifted rooms and the backward dependency set for a selected token. Use labels and outlines for first-layer versus two-layer reach. The simple averaging case provides exact numbers; learned attention changes those weights, while the allowed graph bounds which information can arrive.

A stack with permanently fixed windows is restricted, but not incapable of every global classification: a final pooling or other cross-window operation can combine separate features. The precise limitation is the absence of cross-window *contextual interaction within those attention layers*.

### Why rolling needs a mask

Directly cutting shifted windows produces partial rooms at image boundaries. For batching, an implementation can cyclically roll the feature grid, partition it into equal-size windows, compute attention and reverse the roll. Rolling moves a top-edge token beside a bottom-edge token in memory. That does not make them spatial neighbors.

An attention mask blocks those artificial wrap-around pairs. On a 6×6 grid with 2×2 windows and shift 1, the rolled corner window can contain original coordinates `(5,5)`, `(5,0)`, `(0,5)` and `(0,0)`. These belong to four different boundary regions. If values are zero except a value 16 at `(5,5)`, an *unmasked* averaging window incorrectly gives `(0,0)` the value 4. With the boundary mask, its value stays 0.

When the feature dimensions are not divisible by the window size, padding is an additional issue. The complete educational implementation in [vision-mechanisms.py](vision-mechanisms.py) marks padded keys unavailable, handles padded query rows without an undefined all-masked softmax, reverses the shift and removes padding. Other implementations can use different documented padding conventions. A general instruction to resize every image to a multiple of 224 is not a substitute for inspecting that contract.

### Relative position belongs in the actual score

Within one window, a head uses

$$a_{ij}=\operatorname{softmax}_j\left(q_i^Tk_j/\sqrt{d_h}+b_{\Delta r,\Delta c}+m_{ij}\right),$$

where $m_{ij}=0$ for allowed pairs and $-\infty$ for blocked pairs. The learned bias is indexed by the row and column displacement from key to query. Each displacement lies between $-(M-1)$ and $M-1$, giving $(2M-1)^2$ learned entries per head. Many pairs share an entry because they share a relative displacement.

For a 2×2 window, query `(0,0)` and key `(1,1)` have displacement `(-1,-1)`. If content scores are all equal and that displacement's bias increases, this pair receives more weight, with the other legal weights decreasing through the shared softmax denominator. A bias table that is allocated but never added to the scores has no effect.

**Investigation — open a path, repair a boundary.** The fresh task uses a 6×6 grid, editable values, window 2 and shift 1. Observe whether a source value can affect a selected destination after two steps. Move the source or destination, change a value, and compare masked versus wrapped computation. You can inspect the relative-bias table separately from the connection graph. Reset restores both the image state and the Show the current computed result and its contributing terms immediately.

### A hierarchy changes resolution and width

Swin starts with smaller patches, commonly 4×4 image pixels. A patch-merging layer groups 2×2 neighboring token vectors, concatenates them into a $4d$-wide vector, normalizes, and projects to width $2d$ in the original Swin V1 convention. Spatial resolution halves along each axis, token count quarters, and channel width doubles. This is a learned reduction, not merely averaging four cells.

For an input 224×224 and initial width 96, stage shapes are:

```text
56×56×96 → 28×28×192 → 14×14×384 → 7×7×768
```

The small, medium and coarse grids can feed different parts of a detection or segmentation system. A tiny object may need finer spatial features; broader context can use coarser features. A head must still learn how to turn those features into boxes, masks or labels. A plain ViT's single-resolution features can also support dense prediction through appropriate adapters; “hierarchical features are convenient” does not mean other backbones cannot work.

The source paper's [method section](https://arxiv.org/html/2103.14030v2#S3) is the reference for windows, shifting, relative bias and merging. Swin V2 changes details such as attention and normalization; do not silently combine its rules with this V1 derivation.

### Carry the window and merge into torchvision

The [complete library bridge](vision_library_bridge.py) reuses `ShiftedWindowAttention` and `PatchMerge` from [the supplied mechanism file](vision-mechanisms.py). It copies QKV/output projections and transposes the relative-bias table into torchvision's offset-by-head layout, then checks output, feature gradients and all parameter gradients. The merge comparison copies LayerNorm and reduction weights and includes an odd 5×7 grid. Install a matched Torch 2.14/torchvision 0.29 environment and run `python vision_library_bridge.py` beside the mechanism file. This optional torchvision route is written but **unexecuted** in this content revision; no measured result is asserted.

The comparison deliberately uses divisible 6×9 dimensions for window attention. Our teaching operator excludes padded donor positions; torchvision's documented implementation pads values and uses a finite -100 boundary penalty rather than our negative-infinity mask. Therefore arbitrary padded shapes and extremely large logits are not claimed equivalent. Window size, shift, boundary policy, projection bias, relative offsets, dropout and V1/V2 normalization ordering all belong to the model. [Torchvision Swin source](https://docs.pytorch.org/vision/0.29/_modules/torchvision/models/swin_transformer.html).

**Take control:** change to a 9×12 grid and compare again, then try a 5×7 grid and inspect the padding-policy difference rather than silently loosening the tolerance. **Hint:** first separate an index/layout bug from an intentionally different set of permitted keys. **Solution:** matched divisible shapes should agree; padded differences require aligning the masks if exact equivalence is the goal. The DINOv2 application in §8 separately supplies normal checkpoint feature extraction, while DeiT/DINO objectives below own their visible custom loss and teacher updates.

## 5. DeiT: use a teacher's decisions as another learning signal

A Transformer architecture does not specify the entire training recipe. Data selection, augmentation, optimization, regularization and supervision all affect the result. DeiT demonstrated that a carefully trained image Transformer could be effective using ImageNet-1k alone, and introduced a particularly visual way to distill a teacher into a student.

First distinguish three objects:

- The **teacher** is an already trained image classifier used to supply targets.
- The student's **CLS head** learns from the dataset's labels.
- The student's **DIST head** learns from the teacher's predictions.

DIST is a second learned token inside the student. It attends alongside CLS and the patch tokens. The teacher's answer is used in the loss; it is not inserted into the student's input sequence. At deployment the student can run without the teacher.

**Inline two-target diagram:** the same training image goes to a frozen teacher and a trainable student. Show one label arrow to CLS and one teacher-target arrow to DIST. Use a stop-gradient boundary on the teacher branch. Inside the student, both special tokens participate in ordinary attention; outside, they have distinct heads.

### Hard and soft targets solve related but different problems

With hard distillation, let $y$ be the true label and $\hat y_t=\arg\max_c z_{t,c}$ the teacher's predicted class. For student logits $z_c$ at CLS and $z_d$ at DIST:

$$\mathcal L=\tfrac12\operatorname{CE}(z_c,y)+\tfrac12\operatorname{CE}(z_d,\hat y_t).$$

The cross-entropy implementation takes logits and applies its own log-softmax. If the true class is 2 and the teacher predicts 7, the heads receive different targets. This disagreement can carry a teacher error or reflect a crop whose visible content no longer matches the original image label. Inspect the image and protocol before deciding which.

Soft distillation instead retains a distribution $p_t=\operatorname{softmax}(z_t/T)$. A common objective is $T^2\operatorname{KL}(p_t\|p_s)$ with $p_s=\operatorname{softmax}(z_d/T)$, mixed with the label loss. The temperature $T$ changes the target's sharpness; the $T^2$ factor is a conventional gradient-scale compensation. State the KL direction explicitly. The term means $\sum_c p_{t,c}\log(p_{t,c}/p_{s,c})$, so the teacher supplies the weighting distribution.

For a teacher distribution `[0.6,0.3,0.1]`, the hard target retains only class 0. A student assigning probability 0.3 to class 1 receives no information from the hard target about whether the teacher considered class 1 a plausible alternative. The soft target retains that relation, but it can also transmit poorly calibrated similarities. Neither target form is a universal winner.

The original DeiT paper's Table 3 reports 83.4% ImageNet top-1 for the distilled base model at 224 resolution after its 300-epoch training setting; the corresponding non-distilled base row is 81.8%. These are specific published comparisons, not expected scores for a new dataset or our small program. [DeiT paper, §4 and Table 3](https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf).

### Be precise about combining the heads

The paper describes fusion of the two softmax outputs. The official released `DistilledVisionTransformer.forward` instead returns the mean of the two **logit** vectors at evaluation. Our teaching program follows that released-code convention:

```python
cls_logits, dist_logits = student(images, return_heads=True)
loss = 0.5 * F.cross_entropy(cls_logits, true_labels)
loss += 0.5 * F.cross_entropy(dist_logits, frozen_teacher_labels)
inference_logits = (cls_logits + dist_logits) / 2
```

The methods are not generally interchangeable. For two three-class heads with logits `[-3,−2,0]` and `[2,3,0]`, mean logits are `[-0.5,0.5,0]`, choosing class 1. Mean probabilities are approximately `[0.1508,0.4098,0.4395]`, choosing class 2. Averaging happens on different sides of a nonlinear transformation. This is a concrete reason to inspect a checkpoint's forward method rather than infer its rule from the word “fusion.” [Official DeiT implementation](https://github.com/facebookresearch/deit/blob/main/models.py).

Our small experiment in §7 deliberately records which rule it uses, how teacher targets were obtained and which weights are shared at initialization. It does not claim that adding DIST reproduces the entire DeiT training recipe.

## 6. DINO and DINOv2: learn features before assigning task labels

DeiT's teacher already knows a supervised label space. Self-distillation asks a different question: can different views of the same image learn to agree without a human-supplied class label?

In DINO, a student and a teacher process image crops. Their output heads produce scores over learned **prototypes**. A prototype is an output coordinate learned for this objective; it is not automatically “dog,” “wheel” or any other named class. We compare prototype distributions from different views of the same image.

The student changes through gradient descent. The teacher is updated as an exponential moving average of the student's parameters:

$$\theta_t\leftarrow m\theta_t+(1-m)\theta_s.$$

With $m=0.9$, an old teacher parameter 2 and an updated student parameter 4 give a new teacher parameter 2.2. This is an average of weights over time, not an average of all stored image features. The teacher target is detached from the current loss gradient.

### A cross-view target, one step at a time

DINO uses two larger crops for the teacher, and both larger and smaller crops for the student. Compare each teacher global view with the student views except the identical view. This encourages a local crop to predict a representation also supported by a broader view. The augmentation must preserve a relation worth learning: a blank crop or a crop containing an unrelated object can make the target difficult or inappropriate.

Write teacher prototype logits as $t$, student logits as $s$, a running teacher-logit center as $c$, and positive temperatures as $\tau_t,\tau_s$:

$$p_t=\operatorname{softmax}((t-c)/\tau_t),\qquad
p_s=\operatorname{softmax}(s/\tau_s),\qquad
\ell=-\sum_k p_{t,k}\log p_{s,k}.$$

Centering subtracts a running average *per prototype*. It counters a prototype that tends to dominate across images. A low teacher temperature sharpens the target, countering a tendency toward overly uniform predictions. These operations have different roles. Neither is a mathematical guarantee that every training run avoids collapse.

For a constructed example, use teacher logits `[0.4,0.1,−0.2]`, center `[0.1,0,−0.1]`, and $\tau_t=0.2$. Subtract and divide to obtain `[1.5,0.5,−0.5]`, then softmax gives `[0.66524,0.24473,0.09003]`. Student logits `[0.1,0.2,−0.1]` at $\tau_s=0.5$ give `[0.34581,0.42238,0.23181]`.

Their cross-entropy is 1.048919 nats. The student-logit gradient is

$$\frac{\partial\ell}{\partial s_k}=(p_{s,k}-p_{t,k})/\tau_s
\approx[-0.63885,0.35530,0.28355].$$

A gradient-descent step therefore increases the first logit and decreases the other two. The teacher is not pulled toward the student by this gradient; its separate EMA update happens afterward. The complete two-view loss and update functions are in [vision-mechanisms.py](vision-mechanisms.py). They exclude same-view pairs and were checked for absent teacher gradients.

**Inline objective diagram:** crop A and crop B remain recognizable pieces of the same image; two score strips pass through different center/temperature operations. Crossed arrows connect teacher A to student B and teacher B to student A. Beside them, a separate timeline shows parameter EMA and the running center. This avoids conflating the teacher, its prediction and its stored center.

**Investigation — change a target, inspect the learning direction.** Edit prototype logits, the center or a temperature. Watch the teacher target, student probability and exact signed gradient together. A common shift of all teacher logits preserves its softmax distribution; changing one prototype can move it substantially. Follow that difference to the direction of a student-logit update.

### Agreement alone is not enough

If every image produces the same distribution, different views can agree while the features contain no useful image information. In the especially clear three-prototype case where all teacher and student logits are zero, both distributions are uniform. The cross-entropy is $\log3$, and the student gradient is exactly zero. Centering an already zero center and sharpening equal logits do not break this symmetry by themselves.

This is why training needs its full recipe, and evaluation needs actual tasks or representation diagnostics. A low or declining pretraining loss by itself does not establish useful recognition. During training the teacher targets also change, so the loss is not a fixed-target accuracy score. [DINO §3 and §5.3](https://arxiv.org/html/2104.14294v2#S3).

### What DINOv2 adds

DINOv2 is a complete representation-learning system, not merely the same DINO loss on a bigger image. Its important ingredients fit together as follows:

| Ingredient | What the model must do | Why it matters |
| --- | --- | --- |
|Image-level DINO objective|Match cross-view CLS prototype distributions|Learn features that relate different views of an image|
|Patch-level iBOT objective|Predict teacher representations at student-masked patch positions|Keep local information relevant as well as an image summary|
|Separate image and patch heads|Map the two feature types to their own prototype scores|Avoid forcing the two objectives through identical head weights|
|Teacher assignment balancing|Use the chosen centering/balancing method, including Sinkhorn–Knopp in the reported recipe|Control prototype use across a batch|
|KoLeo feature-spreading term|Penalize very close nearest neighbors among normalized image features|Discourage the batch from occupying only a tiny part of feature space|
|Curated data and a high-resolution adaptation phase|Train on diverse, deduplicated images and later finer patch grids|Support transferable global and local features|

For the masked-patch objective, the teacher sees the corresponding image region without the student's mask. The student sees a mask token at that location and must use visible context. Matching a teacher feature distribution differs from reconstructing exact RGB values. The masking objective is a learning target, not a claim that the model has recovered an unobserved pixel with certainty.

**Deeper: balancing and spreading address different objects.** Sinkhorn–Knopp alternately rescales a nonnegative prototype-by-sample assignment matrix toward specified row and column marginals. It encourages balanced prototype assignments *across samples*; ordinary softmax independently normalizes one sample's prototype scores. A few iterations produce an approximate balanced assignment. Numerical stability and batch composition matter.

KoLeo works on normalized feature vectors $u_i$, not on those prototype probabilities. Its idealized form is

$$\mathcal L_{\mathrm{KoLeo}}=-\frac1n\sum_i\log\min_{j\ne i}\|u_i-u_j\|_2.$$

If two vectors approach each other, their nearest-neighbor distances approach zero and their negative log distances increase. Practical implementations need a numerical floor. Spreading features does not itself assign semantic labels or prove that rare categories are represented fairly. These terms complement rather than replace task-based evaluation. [DINOv2 §4 and implementation details](https://arxiv.org/html/2304.07193v2#S4).

DINOv2's smaller released models can be distilled from a larger fixed teacher; that stage differs from maintaining an EMA teacher of the same student during initial pretraining. The output backbone can be frozen while a task-specific head is trained. “Frozen features” means no backbone update; it does not mean no labels or fitting are used in downstream evaluation.

### Deeper and current: registers and DINOv3

An output patch token is associated with a spatial position, but attention can also use it for internal computation. The registers study found high-norm outliers in some patch features and introduced additional learned tokens as nonspatial working locations. These **register tokens** participate in attention and are normally excluded from the returned spatial patch grid. They are not extra image patches, class labels or a hardware register file. The study's observed improvements do not establish that every architecture requires exactly four registers. [Registers study, §2–3](https://arxiv.org/html/2309.16588v2).

As of the September 2026 source check, DINOv3 provides a later family and an instructive new objective, **Gram anchoring**. It compares relationships among patch features with relationships from a separate teacher snapshot. If rows of $X$ are normalized patch features, their Gram matrix $XX^T$ contains every patch-to-patch cosine similarity. The objective penalizes a difference between the student's and teacher's Gram matrices.

For two orthogonal features `[1,0]` and `[0,1]`, the Gram matrix is the identity. Rotating both features together changes their coordinates but preserves that identity. Collapsing both to `[1,0]` changes the off-diagonal similarities from 0 to 1, giving squared Frobenius difference 2. Preserving relationships is a different constraint from demanding identical feature coordinates.

The paper develops this idea to address deterioration of dense features during prolonged training. Its vision models also use a two-dimensional RoPE construction rather than DINOv2's learned absolute position table. These differences make “all DINO models are interchangeable patch 14 feature extractors” incorrect. Here we teach the geometric bridge; the later [Self-Distillation topic](/learn/path/full-curriculum/self-distillation-byol-dino-dinov2?module=self-supervised-learning), currently planned, is the home for a full comparison of these training systems. [DINOv3 §3.2 and §4](https://arxiv.org/html/2508.10104v1#S4), [official models and access conditions](https://github.com/facebookresearch/dinov3).

**Inline relationship view:** a small patch grid sits beside its feature arrows and Gram matrix. A common rotation moves the arrows while preserving the matrix; moving one arrow changes its similarity row and column. The values are constructed geometry, not a claim about measured DINOv3 embeddings.

## 7. Run a small image study and interpret every result

The **Optical Recognition of Handwritten Digits** collection was designed to investigate handwritten-digit recognition. The supplied 8×8 images were obtained by counting marked pixels in 4×4 blocks of normalized 32×32 bitmaps. Each cell is an integer from 0 to 16. Our input divides by 16, using the published bound rather than a statistic fitted across the data. This is not MNIST and not a collection of natural RGB photographs. [UCI dataset and CC BY 4.0 attribution](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

The original files contain 3,823 images from 30 writers and 1,797 images from 13 different writers. We retain the original test pool and choose 120 training plus 30 validation images per digit from the original training file, using NumPy generator seed 173. That gives 1,200 training,300 validation and 1,797 test images. The other training-file rows remain unused. All pixel vectors were checked: no exact duplicates within either file or across files. Individual writer IDs are unavailable, so our internal validation subdivision is not a writer-disjoint validation split; the original final test pool is separate by the collectors' protocol.

### What is held fixed, and what changes?

The complete [vision-study.py](vision-study.py) contains data loading, model definitions, losses, checkpoint selection, feature readouts and JSON export. Run it beside the three supplied `optdigits` files:

```text
python vision-study.py
```

The recorded run used Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu and scikit-learn 1.9.1, one CPU thread. It does not need a network connection or a GPU. The [declared protocol](experiment-contract.md) was saved before fitting.

The CNN has two small convolutions and a 32-wide feature layer. The ViT has sixteen 2×2 patch tokens, a 32-wide CLS representation, two pre-normalization blocks, four heads of width 8, and FFNs of width 64. The distilled student adds DIST and its head. Its common tensors are copied from the plain ViT's initial state; the extra token/position start at zero and its head starts as a copy of the CLS head.

Each model receives 600 AdamW updates, learning rate 0.002, weight decay 0.01 on all parameters, and batches of 128 sampled with replacement from the same seed 301 index stream. There is no augmentation, learning-rate schedule, dropout or stochastic depth. Every 50 updates, validation cross-entropy is measured; the smallest value selects the checkpoint, with ties going to the earliest. The fixed test pool is evaluated afterward. This is deliberately a modest training recipe, not a DeiT or DINO reproduction.

The selected teacher is frozen. Its hard labels disagree with 8 of the 1,200 training labels. Even where the teacher agrees, the student differs from plain ViT through an extra token, extra head and two loss paths. The comparison therefore does not isolate only “teacher information” or match parameter counts exactly.

The classifier results were:

| Model | Parameters | Selected update | Validation correct/300 | Test correct/1,797 | Test cross-entropy, nats |
| --- | ---: | ---: | ---: | ---: | ---: |
|Small CNN teacher|15,386|550|293|1,723|0.13587|
|Plain patch ViT|18,218|350|285|1,630|0.38152|
|Student with DIST|18,612|600|283|1,658|0.30758|

Distillation improves this student's test count and cross-entropy, but the CNN remains stronger. The distilled student's selected validation *count* is lower than the plain student's, even though its validation cross-entropy is lower. That is possible because checkpoint selection minimizes a probability-sensitive loss, not simply the number of correct argmax decisions. A few confident mistakes can matter substantially to cross-entropy.

These are one-seed results with different architectures and a fixed small budget. They do not estimate seed uncertainty or establish a universal ordering. Keeping an unfavorable result is useful: the experiment teaches that a successful mechanism and a good final model choice are different questions.

**Inline observed-results view:** show the actual correct counts and cross-entropies on separate scales. Let the reader select a model's recorded validation trace without confusing it with test performance. Label the selection point on validation only. No smoothed invented curve should imply measurements between saved updates.

### A frozen representation and its classifier are separate components

For each selected model, freeze the feature extractor. Fit a training-only `StandardScaler` followed by `LogisticRegression(C=1,max_iter=2000)` on its 32-wide features. Also fit the same readout on the 64 raw pixels. Validation and test rows are only transformed and predicted.

| Input to the newly fitted linear readout | Test correct/1,797 | Test cross-entropy |
| --- | ---: | ---: |
|Raw 64-pixel vector|1,695|0.16831|
|Frozen CNN feature|1,725|0.12642|
|Frozen plain ViT CLS|1,637|0.41823|
|Frozen distilled student's CLS|1,648|0.34947|

The raw-pixel baseline is strong on these already normalized small images. The distilled CLS readout uses only CLS, whereas that network's original classifier fuses CLS and DIST. Its score need not match the two-head score. A linear readout can improve the count but worsen cross-entropy, as happens for plain ViT here. It fits a different decision surface with a different regularizer.

These feature extractors were trained with labels. Their readouts demonstrate the mechanics of frozen-feature evaluation; they are not evidence that self-supervised DINO was trained. A real DINO evaluation must identify its actual pretrained checkpoint, data boundary, feature definition and fitted readout.

### Inspect a real error as well as a success

The saved visual examples are the first three test-file rows, chosen by order rather than by how attractive their maps look. The plain ViT assigns the first image, a 0, probability 0.99787 for its correct class. For the third image, a 2, its probability for class 2 is only 0.01899. Show both the image and all ten scores. A polished attention heatmap should not hide that the classifier is wrong.

On the first image, change intensity `(row2,column4)` from its original value to one minus that value. The model's probability for 0 changes from 0.99787 to 0.99665. This establishes a response to that edit; it does not identify a general causal explanation of how humans recognize zero.

**Investigation — pixels, positions and predictions.** Start from a fresh unsolved specimen. Inspect the consequence of a specific edit, then change a pixel or swap patch contents. Compare that with moving the corresponding positional vectors together. Keep the true label visible as a dataset annotation, not as an input to the network. The model state stays frozen, so changes have an identifiable cause.

All saved models were reloaded and reproduced their recorded test predictions. An independent NumPy forward calculation agreed with the plain ViT logits on the first three images within 0.000004 and with its attention weights within 0.0000003. Those checks substantiate the written numbers and inputs; implementing and checking the browser model remains a later step.

## 8. Use image features for a real task

### A spatial feature map is not a segmentation mask

The final patch outputs can be reshaped into a grid. For the small ViT, they form 4×4 vectors of width 32. One way to visualize them is to fit PCA on training patch vectors, then project each example into the same three axes and map those coordinates to color.

In this study the first three PCA directions explain approximately 22.72%,16.76% and 10.50% of training patch-feature variance. Together that is about 49.98%, not the entire representation. The axes are fitted only on the training patch features and reused for all displayed test images. Refitting PCA independently on each image would make colors incomparable; sign flips of equally valid PCA directions can also reverse a color axis.

**Inline feature atlas:** place the original grayscale image beside its patch grid, common-basis PCA colors and a selected attention row. Legends identify three different quantities: observed intensity, feature coordinates and attention weight. Preserve numeric inspection and a text description; neither hue nor a brighter attention cell establishes a semantic object boundary.

A patch-to-patch matching task uses a different calculation. Normalize feature vectors and compute cosine similarity. For patch 5 of test image 1 compared with all 16 patch vectors of test image 2, the top matches are patch 10, patch 2 and patch 14 with cosines 0.3620,0.2958 and 0.2895. The algorithm always returns a maximum when candidates exist. That maximum is not automatically a meaningful correspondence: these are different digits, the scores are modest, and the features were trained for image classification rather than matching parts.

The mechanism nevertheless explains a useful application: **copy detection and image retrieval**. A global image embedding can find similar images despite some appearance changes; local patch matching can inspect whether corresponding regions agree. Distinguishing a cropped copy from two different but similar products requires labeled evaluation pairs and geometric verification. Similarity by itself cannot establish identity, ownership or authenticity. DINO's paper explicitly studies retrieval and copy detection; the later vision lessons can develop the full evaluation. [DINO §4.2.1](https://arxiv.org/html/2104.14294v2#S4.SS2).

For **dense prediction**, a head can map each patch feature to class scores or a continuous target, then a decoder or interpolation can produce a finer grid. Patch 14 features on a 518-pixel side give 37 positions per axis. Enlarging the output grid does not create additional independent feature measurements or guarantee sharp boundaries. A head trained on masks or depth labels supplies task-specific supervision even if the backbone remains frozen.

This makes a less obvious application easy to interpret: a remote-sensing encoder may supply patch features to a canopy-height estimator. The features encode image patterns; a regressor learns the relationship to measured heights. Pixel size, geographic split, acquisition conditions and target units become part of the evaluation. The DINOv3 report explores this domain, but there is no reason to assume a generic natural-image checkpoint gives calibrated heights without that task-specific work. [DINOv3 geospatial study, §8](https://arxiv.org/html/2508.10104v1#S8).

### Attention is one internal quantity, not the entire explanation

A CLS-to-patch row shows how one head mixes its values at one layer. It omits other heads, the value projection, earlier layers, residual streams and the classifier. With 16 patches plus CLS, uniform attention has weight $1/17$ at every token. If a display excludes CLS and renormalizes the remaining weights, it must label that conditional normalization rather than silently implying the original row sums to one across patches.

Attention rollout composes attention matrices, often with a residual approximation. It can be an informative visualization but still does not replace the network's actual value/FFN computation. Perturbation gives an actual input–output contrast for the chosen edit, with its own limit: masking a region can create an out-of-distribution image. Use maps to pose a testable question, then measure the effect relevant to that question.

### Checkpoint preparation is part of the model

For a pretrained system, record the exact checkpoint and its model configuration. Then preserve its expected color channels, pixel range, resize/crop policy, interpolation, normalization, patch size, special tokens and output layout.

A PIL RGB image usually supplies byte-range input to a processor that rescales and normalizes. An already rescaled float image may require disabling the processor's rescaling; otherwise it can be divided twice. “All ViTs use mean 0.5” is false: normalization belongs to a checkpoint's training pipeline.

In `timm`, resolve the data configuration before creating a transform:

```python
import timm
from PIL import Image

model = timm.create_model("vit_base_patch16_224", pretrained=True).eval()
data_config = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_config, is_training=False)
image = Image.open("example.png").convert("RGB")
image_tensor = transform(image).unsqueeze(0)
```

This is an optional API example, not an executed checkpoint inference in this packet. Preparing `num_classes=8` creates an 8-class head; it does not train that head. A frozen-feature setup must freeze the backbone, create/train the intended head, and verify which parameters and state buffers can change. `eval()`, disabling gradients and freezing parameters serve different purposes. [Current timm quickstart](https://huggingface.co/docs/timm/quickstart).

The complete optional [inspect-pretrained-features.py](inspect-pretrained-features.py) reads a locally prepared official `facebook/dinov2-small` checkpoint, its processor and local image files. It performs inference, extracts CLS and spatial patch features, and saves a cosine matrix. It makes no network requests. Prepare the checkpoint and the PyTorch/Transformers/Pillow dependencies first, then run:

```text
python inspect-pretrained-features.py --checkpoint ./dinov2-small --output features.npz image-a.png image-b.png
```

This optional program was written against the inspected non-register DINOv2 interface and has not been executed here; no sample prediction is claimed. For that model family the expected token count is one CLS plus the processor's patch grid, with hidden width given by the configuration. The program checks those relationships. A register variant or DINOv3 needs its own documented interface. Official DINOv2's native `forward_features` dictionary and Transformers' `last_hidden_state` tensor are different APIs. [Transformers DINOv2 documentation](https://huggingface.co/docs/transformers/model_doc/dinov2), [native feature interface](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py).

### Choose an experiment before choosing a fashionable model

For a new catalog classifier, begin with an explicit label set, grouped image/product split and a held-out evaluation. Fit a modest baseline and a frozen readout from an appropriate available checkpoint. Inspect errors by category, crop size and image source. Measure latency and memory on the actual deployment device with the intended batch, dtype and preprocessing.

If the frozen representation cannot support the distinctions, compare partial or full fine-tuning under a validation protocol. If there is useful unlabeled in-domain data, continued representation learning is another experiment, with data contamination, compute and forgetting to assess. There is no universal “fine-tune only when the gap exceeds 3%” or guaranteed ranking between those choices.

Text-defined zero-shot labels require an image–text alignment mechanism. CLIP and SigLIP are examples, and DINO-based systems can add text alignment too; a bare visual feature vector does not acquire that ability simply by calling softmax. The planned [Vision-Language Models](/learn/path/full-curriculum/vision-language-models-clip-siglip-blip-2?module=nlp-cv-multimodal) lesson owns that objective. Promptable masks require a prompt encoder and mask-prediction system, developed in [Foundation Models for Segmentation](/learn/path/full-curriculum/foundation-models-for-segmentation-sam-sam-2?module=nlp-cv-multimodal). Pixel reconstruction objectives belong in [Masked Autoencoders](/learn/path/full-curriculum/masked-autoencoders-mae-beit-data2vec?module=self-supervised-learning). These are distinct uses of learned visual representations, not interchangeable promises attached to “ViT.”

## 9. Practice: explain, calculate, intervene and choose

The core finish line is Exercises 1–4 and 7–8. Exercises 5–6 and 9 develop the deeper branches. Attempt each before opening its hint or solution; their inputs differ from the worked demonstrations.

### 1. A rectangular camera image

An RGB image is 96×160. Use 8×8 patches and width 64, with one learned CLS and a learned absolute position for every token. How many pixel positions, patches and sequence tokens are there? What is the input width of the patch projection? Count all patch-projection weights and biases, the CLS parameter and the position table. What changes if DIST and its position are added, before counting its classifier?

<details><summary>Hint</summary>

Keep channel values separate from pixel positions. The grid is rectangular. Count the special token parameter independently from its positional vector.

</details>

<details><summary>Solution</summary>

There are 15,360 pixel positions and 46,080 RGB scalars. The patch grid is 12×20, so $N=240$ and $S=241$. Each patch has $8^2\cdot3=192$ scalars. The stem parameters are $192\cdot64+64+64+241\cdot64=27,840$. Adding DIST adds one 64-wide token and one 64-wide position, so the subtotal becomes 27,968 and the sequence has 242 tokens. Its head adds further parameters. Treating every RGB scalar as a token or forgetting one special-token parameter gives a different, incorrect count.

</details>

### 2. Change the image while preserving a token

Use the two projection features from §1 on a patch `[1,4,2,3]`. Calculate the embedding. Find a different patch with the same embedding, then explain why a classifier receiving only this projected patch cannot distinguish those two inputs from that patch alone.

<details><summary>Hint</summary>

One feature depends on the first-plus-fourth sum. Try adding to one of those entries and subtracting from the other.

</details>

<details><summary>Solution</summary>

The embedding is `[4.5,3]`. The patch `[2,4,2,2]` has the same first-plus-fourth sum and unchanged second-minus-third difference, so it gives the same embedding. The difference `[1,0,0,−1]` lies in the projection's nullspace. Later processing cannot reconstruct which input was used if every other input and position is identical. This example shows loss caused by this particular projection; it does not prove every learned patch projection loses the same distinctions.

</details>

### 3. Two steps through shifted rooms

On an 8×8 token grid, apply 2×2 regular windows, followed by windows shifted by 1 along both axes, with correct boundary masks. Can an original signal at `(1,1)` affect destination `(3,3)` after those two layers? What about a signal at `(2,2)`? For uniform averaging, find the nonzero source coefficients at that destination.

<details><summary>Hint</summary>

Trace backward from the destination through the *second* layer, then through each selected first-layer state.

</details>

<details><summary>Solution</summary>

The shifted window for `(3,3)` contains rows 3–4 and columns 3–4. Their original regular windows cover rows 2–3 or 4–5, and columns 2–3 or 4–5. The full dependency set is rows 2–5 by columns 2–5. Thus `(1,1)` cannot contribute at this depth, while `(2,2)` can. Each of the 16 source tokens has coefficient 1/16 in the two-layer averaging model. Learned attention and nonlinear processing change the coefficients and response, but do not create an unavailable path. A third layer can extend reach, so the conclusion is depth-specific.

</details>

### 4. A teacher makes a different decision

The true class is 2. A frozen teacher predicts class 0. The student's CLS probabilities are `[0.2,0.5,0.3]`, and its DIST probabilities are `[0.6,0.1,0.3]`. Compute the half-and-half hard distillation loss. Which head is trained toward which class? Does the teacher need to run when deploying the student?

<details><summary>Hint</summary>

Each cross-entropy selects the probability assigned by its own head to its own target. Class indices are zero-based.

</details>

<details><summary>Solution</summary>

The loss is $-\tfrac12\log0.3-\tfrac12\log0.6\approx0.85740$ nats. CLS is trained toward class 2, DIST toward class 0. The teacher gets no gradient and is unnecessary for the student's deployed forward pass. The two targets conflict for this image; distillation does not make the teacher automatically correct. The final prediction additionally depends on the declared head-fusion rule, which is separate from these two training targets.

</details>

### 5. Resize positions, preserve identity

A checkpoint has a 2×4 patch grid, embedding width 3 and two nonspatial prefix positions. You want a 4×4 grid with the same patch size. What are the old and new position-table shapes? Which entries must be kept outside spatial interpolation? Why does resizing this table fail to solve a change from 8×8 patches to 4×4 patches?

<details><summary>Solution</summary>

The shapes are `(1,10,3)` and `(1,18,3)`: two prefixes plus 8 or 16 patches. Preserve the two prefix entries and interpolate only the 2×4 spatial grid to 4×4. Halving patch size changes the patch projection's input width by a factor of four, as well as changing the number of patches at a fixed image size. Position-table interpolation cannot reshape or train that different projection. Both changes require a specified checkpoint-adaptation method.

</details>

### 6. A self-distillation gradient

A teacher has two equal centered logits, hence target `[0.5,0.5]`. Student temperature is 0.5 and student logits are `[0,0.5 log3]`. Calculate the student probabilities and logit gradient. Does sharpening equal teacher logits make one prototype win? What happens when teacher and student both become uniform for every image?

<details><summary>Hint</summary>

Dividing the second student logit by 0.5 gives `log3`. Apply the gradient formula from §6.

</details>

<details><summary>Solution</summary>

The student probabilities are `[0.25,0.75]`. Its logit gradient is `[(0.25−0.5)/0.5,(0.75−0.5)/0.5]=[−0.5,0.5]`. Gradient descent raises the first logit and lowers the second. Equal teacher logits remain equal under any positive temperature. If both distributions are uniform for every image, the student gradient is zero even though the representation may be useless. The existence of this stationary case is why agreement is not a sufficient feature-quality test.

</details>

### 7. Diagnose a weak validation result

You prepare a pretrained image Transformer for an eight-class task. Training accuracy is high, validation accuracy is close to 12.5%, and the model runs without a shape error. Propose four plausible checks. Each must inspect an actual mechanism; avoid saying that a single symptom proves a particular cause.

<details><summary>Possible solution</summary>

Check that train and validation use the same class-to-index mapping by inspecting filenames and decoded labels from both loaders. Inspect the actual transformed image tensors and the checkpoint processor configuration for double rescaling, color order, cropping or normalization errors. Check which head/backbone parameters require gradients, which are in the optimizer, and whether they actually change after one step; a frozen random head and a supposedly trained head are different situations. Audit product/writer/source groups, duplicates and split distribution so that validation reflects the intended task rather than an accidental domain or label shift.

Also inspect the position-table/token shapes at the actual input size and any explicit adaptation code. A missing interpolation usually produces a shape mismatch in a straightforward implementation; do not assume silent zero-padding occurred just because validation is poor. Different train/validation augmentations are normal by design, so a visual difference alone does not prove a bug. The useful diagnostic is whether the difference matches the intended training and evaluation policies.

</details>

### 8. A representation experiment for product search

You have images of repeated product instances from several shops, including crops and near-duplicate photographs. Design a comparison between a simple image baseline and frozen vision features for finding the same product. Specify the split unit, candidates, metric, one visual inspection and a failure you expect. Would a linear classification probe answer exactly the same question?

<details><summary>Possible solution and criteria</summary>

Group related captures and product identities deliberately. For generalization to unseen products, split by identity; if testing new captures of known identities, keep the intended identity overlap but separate capture/source groups and declare that different goal. Fit any transforms only on training data, deduplicate across boundaries and define a fixed retrieval gallery containing distractors. Report recall at chosen ranks and the distribution of same-instance versus different-instance similarities, with a policy for queries having no correct gallery item.

Inspect a mistaken query, its top matches and patch correspondences. Similar packaging or a large shared background can dominate the embedding while a small identifying mark is missed. A classification probe tests labeled decision boundaries on its class set; retrieval tests the ranking among candidate images and does not require a closed label set in the same way. A good answer makes the task and error costs concrete, rather than assuming a high ImageNet score guarantees product identity matching.

</details>

### 9. Preserve relationships while features move

Three unit patch features are `[1,0]`, `[0,1]`, and `[−1,0]`. Write their Gram matrix. Rotate all three by 90 degrees. Then instead change only the third feature to `[0,−1]`. Which operation preserves the matrix? Calculate the unnormalized squared Frobenius difference for the other.

<details><summary>Solution</summary>

The original Gram matrix is `[[1,0,−1],[0,1,0],[−1,0,1]]`. A common orthogonal rotation preserves every dot product. Changing only the third feature makes its dot products with the first and second become 0 and −1 instead of −1 and 0. Four off-diagonal entries change in magnitude by 1, so the squared Frobenius difference is 4. If an implementation divides by the number of entries or uses a different loss reduction, its scalar will differ by that declared normalization. Gram equality preserves pairwise geometry; it does not force the feature coordinates themselves to be equal.

</details>

### A compact capstone

Use the provided study without changing its frozen test decisions. Read one successful and one failed image through patch extraction, positions, two blocks and the head. Run a new intervention and compare its result with the unchanged input. Explain whether the outcome supports a claim about projection, spatial arrangement, attention reach, or only this model's decision. Then propose a *new, separately evaluated* training experiment justified by the error analysis; do not keep tuning against the already inspected test pool.

A successful explanation names the actual input, the operation being changed, what remains fixed, the output being measured and a limit. You are ready to continue when you can connect those decisions to the shapes and learning objectives, rather than only recall the four architecture names.

The next topic in this module is [Mixture-of-Experts Transformers](/learn/path/full-curriculum/mixture-of-experts-transformers-moe?module=deep-learning-fundamentals). It changes which feedforward experts process a token. Keep that separate from Swin's restriction on which tokens exchange information through attention.

## 10. References and another way to learn it

The lesson is self-contained. These resources offer a different explanation, original evidence or a concrete implementation to inspect. Research was checked on 13 September 2026; older papers describe their own experiments rather than today's universal model ranking.

- [Dive into Deep Learning, §11.8: Transformers for Vision](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html): a free, code-oriented route through patch embedding, the encoder, model assembly and training. The complete chapter and its exercises were read. Its convolutional patch implementation is particularly useful; the discussion's broad scaling language should be read alongside the conditions in the original experiments.
- [Stanford CS231n Lecture 8 slides, 2025](https://cs231n.stanford.edu/slides/2025/lecture_8.pdf) and [Stanford Online's accompanying lecture video](https://www.youtube.com/watch?v=RQowiOF_FvQ): another visual explanation for learners who benefit from a spoken progression. Slides 100–109 were inspected for image→patch→projection→attention→pooling. The video listing was verified; the recording/transcript was not reviewed and no timestamp guidance is asserted. The slides illustrate mean pooling, a useful contrast with CLS.
- [Dosovitskiy et al., An Image Is Worth 16×16 Words](https://arxiv.org/html/2010.11929v2): original ViT architecture, resolution adaptation, data/compute comparisons and attention analyses. Read §3 before interpreting the particular benchmark protocols in §4.
- [Touvron et al., Training Data-Efficient Image Transformers and Distillation Through Attention](https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf), with [official model code](https://github.com/facebookresearch/deit/blob/main/models.py): supervised hard/soft distillation and the second token. Compare the paper's fusion wording with the released forward method, as developed in §5 here.
- [Liu et al., Swin Transformer](https://arxiv.org/html/2103.14030v2): §3 and Figures 2–4 explain the spatial hierarchy and shifted windows. Trace original coordinates through the roll before reading its throughput comparisons.
- [Caron et al., Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/html/2104.14294v2): DINO's cross-view objective, EMA teacher, centering/sharpening and representation evaluations. Its algorithm and collapse discussion complement the worked gradient here.
- [Oquab et al., DINOv2](https://arxiv.org/html/2304.07193v2): §4–5 explain the combined objectives and implementation, with the evaluation protocol distinguishing a frozen backbone from a trained downstream head.
- [Darcet et al., Vision Transformers Need Registers](https://arxiv.org/html/2309.16588v2) and [Siméoni et al., DINOv3](https://arxiv.org/html/2508.10104v1): deeper reading about nonspatial tokens, dense-feature behavior and relational regularization. The inspected DINOv3 sections include §3.2 and §4.2–4.3; its full training and deployment are beyond this lesson's small experiment.
- [timm quickstart](https://huggingface.co/docs/timm/quickstart), [Transformers DINOv2 interface](https://huggingface.co/docs/transformers/model_doc/dinov2), and [official DINOv3 repository](https://github.com/facebookresearch/dinov3): use the interface and preprocessing for the actual checkpoint. DINOv3's access/license terms differ from DINOv2's; inspect the version you intend to use.
- [UCI Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49): E. Alpaydin and C. Kaynak, 1998, CC BY 4.0. The supplied files and the [provenance record](provenance.md) make our numerical study available offline.
