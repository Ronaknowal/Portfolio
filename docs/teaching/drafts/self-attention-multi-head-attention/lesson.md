# Self-Attention & Multi-Head Attention

A point on a recorded hand movement tells you where the hand was at one moment. A word tells you something about a sentence. Neither necessarily tells you enough on its own. We need a way for each part of an input to gather useful information from other parts.

**Self-attention lets each input position build a different mixture of information from the same input.** It learns how to choose that mixture. Multi-head attention runs several such mixing operations side by side, then combines their results.

This is useful for contextual word representations, relationships between image patches, and interactions within a set of measurements. It also has a revealing limitation: if we give it points without their order, it cannot recover order merely by making the network larger. We will compute attention by hand, implement it, and inspect that limitation in an actual classifier of recorded hand trajectories.

**Your first route through the lesson:** follow §§1–6, try the message-mixing and mask investigations, and complete practice 1–4. That gives you the mechanism, a working implementation and a real result to interpret. The gradient, interpretation and efficiency branches in §§7–9 deepen that understanding; practice 5–8 accompanies those branches. You can return to them without losing a prerequisite for the next lesson.

You need to recognize a vector as a list of numbers, a matrix multiplication as weighted sums, and a learned linear layer as `x @ W + b`. A batch is several examples processed together. If gradients are new, the first route only needs this idea: training adjusts parameters to reduce a loss; §7 shows the detailed derivatives. The earlier recurrent and attention lessons explain other ways to move information through a sequence. Here we make pairwise self-attention explicit.

## 1. Give each position a way to ask, match and contribute

Imagine three measured positions, A, B and C. Position A is about to update its representation. It needs two decisions: **whose information is useful to A, and what information should those positions contribute?** Keeping these decisions separate gives us three roles.

| Role | Plain meaning | Who uses it? |
| --- | --- | --- |
| Query, `q` | A description of what a receiving position is looking for | The receiver |
| Key, `k` | A description used to decide whether a donor matches that query | Each potential donor |
| Value, `v` | The information a donor contributes if it receives weight | Each potential donor |

A query is not a text question, and a key is not an identifier looked up in a dictionary. They are learned numerical representations. The query–key comparison produces a score. The values are the vectors that actually get mixed.

<!-- Figure SA1: receiver-and-donors with a separate match path and value path. -->

**Read the arrows this way:** A's query compares with the keys of A, B and C. Those three matches determine three weights. The values of A, B and C then travel to A with those weights. B performs its own comparisons and can receive a different mixture. A position may include itself among its donors.

For a sequence of `L` positions, stack the input vectors as rows of a matrix `X`. If each vector has `d` features, `X` has shape `L × d`. Three linear maps produce

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
\]

For one head, `W_Q` and `W_K` each have shape `d × d_k`; `W_V` has shape `d × d_v`. Consequently, query and key vectors have the same length `d_k`, because we will take their dot product. Values may have a different length `d_v`.

The word **self** says that the query, key and value maps all act on the same source `X`. It does **not** say that `Q`, `K` and `V` are equal. Their learned matrices are usually different. In **cross-attention**, the receiving queries come from one source while the donor keys and values come from another. For example, a translation decoder can query representations of the input sentence. The earlier [Bahdanau/Luong attention lesson](/learn/path/full-curriculum/attention-mechanism-bahdanau-luong?module=deep-learning-fundamentals) develops that arrangement; the matching-and-mixing idea carries over.

These numerical features do not automatically correspond to human labels such as “adjective head” or “motion-direction head.” Training may produce useful structure, but the name of a head is not a built-in fact about its parameters. We will examine what its weights can tell us in §8.

## 2. Compute one complete attention head

The computation has four operations: compare, scale, normalize and mix. We will use small, deliberately constructed vectors so that every number is visible. These are arithmetic fixtures, not embeddings extracted from a trained language model.

| Position | Query | Key | Value |
| --- | --- | --- | --- |
| A | `[1, 0]` | `[1, 0]` | `[2, 0]` |
| B | `[0, 1]` | `[0, 1]` | `[0, 2]` |
| C | `[1, 1]` | `[1, 1]` | `[1, 1]` |

The query and key happen to be identical in this one fixture. They need not be identical in a model.

### Compare: each receiver gets a row of scores

For receiver A, the dot products with keys A, B and C are `1, 0, 1`. A dot product multiplies corresponding coordinates and adds them: `[1,0] · [1,1] = 1*1 + 0*1 = 1`.

Doing this for every receiver gives `QKᵀ`:

\[
QK^T=
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}.
\]

Rows are **receivers / queries**. Columns are **donors / keys**. Cell `(A,C)` answers “how strongly does A's query match C's key?” It does not yet contain a probability. Scores may be negative, and a large vector norm can increase a dot product even without better angular alignment. This is why dot-product attention is not simply cosine similarity.

### Scale: keep the score range manageable

Divide by `sqrt(d_k)`. Here `d_k=2`, so A's scaled scores are approximately `[0.7071, 0, 0.7071]`. We will derive the usual initialization argument in §7. For now, the scaling controls how sharply the next operation distinguishes the scores.

### Normalize: turn one row into a distribution over donors

For scores `s_1,...,s_L`, softmax computes

\[
a_j=\frac{e^{s_j}}{\sum_r e^{s_r}}.
\]

The weights are nonnegative and sum to one. A's exponentials are approximately `[2.0281, 1, 2.0281]`; their sum is `5.0562`. Dividing gives `[0.4011, 0.1978, 0.4011]`.

Apply softmax separately to **each row**:

\[
A=\operatorname{softmax}_{\text{donors}}(QK^T/\sqrt{d_k})
\approx
\begin{bmatrix}
.4011&.1978&.4011\\
.1978&.4011&.4011\\
.2483&.2483&.5035
\end{bmatrix}.
\]

Every receiving position has one unit of weight to distribute. The whole matrix does not sum to one, and its columns need not sum to one. A source can receive high weight from many different queries.

### Mix: use the weights on values

The output at A is

\[
z_A=.4011[2,0]+.1978[0,2]+.4011[1,1]
\approx[1.2033,.7967].
\]

Perform all these weighted sums together with the matrix product `AV`:

\[
Z=AV\approx
\begin{bmatrix}
1.2033&.7967\\
.7967&1.2033\\
1&1
\end{bmatrix}.
\]

That is a full scaled dot-product attention head:

\[
\boxed{\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}(QK^T/\sqrt{d_k})V.}
\]

The output has one row per query and `d_v` columns. It has the same number of positions as the input in self-attention, but its feature width need not be the input width. [Vaswani et al., §3.2](https://arxiv.org/pdf/1706.03762) gives the original Transformer formulation; our numerical fixture is computed independently.

<!-- Figure SA2: QK score grid → row distribution → value-space weighted point. -->

There is a geometric interpretation worth keeping. Before any output projection, residual addition or attention dropout, each row of `AV` lies in the **convex hull** of its allowed value vectors: the line segment, triangle or higher-dimensional region reachable by nonnegative weights summing to one. Here all three values lie on the line `x+y=2`, so every output stays on that segment. The C output is `[1,1]` even though C gives itself more than half the weight: the symmetric contributions from A and B balance exactly. A distinctive heatmap need not imply a distinctive output vector.

**Investigate the two paths.** In the message-mixing investigation, move one key and predict which donor weight will rise for a selected query. Then move that donor's value while keeping its key fixed. Predict whether the weights, output, or both will change. Record your prediction before revealing the recomputed distribution. Finally, add the same constant to every score in the row. The displayed arithmetic will let you explain the result instead of guessing from the colors.

## 3. Decide which information is allowed to travel

A high match score is irrelevant if that donor is illegal for the task. A **mask** records allowed query–key connections before normalization.

### Causal attention: only the available prefix

Suppose input positions are ordered A, B, C and each position will predict what comes next. A may use A; B may use A and B; C may use all three. The allowed-edge pattern is lower triangular, including its diagonal.

| Receiver | Donor A | Donor B | Donor C |
| --- | --- | --- | --- |
| A | Allowed | Blocked | Blocked |
| B | Allowed | Allowed | Blocked |
| C | Allowed | Allowed | Allowed |

Set blocked scores to negative infinity, then normalize the remaining scores. For our fixture, the attention matrix becomes

\[
A_{\rm causal}\approx
\begin{bmatrix}
1&0&0\\
.3302&.6698&0\\
.2483&.2483&.5035
\end{bmatrix}.
\]

The outputs become `[2,0]`, `[0.6605,1.3395]` and `[1,1]`. The second row is **renormalized** over two legal donors. Multiplying the old full-attention weights by zero at blocked positions would leave its weights summing to less than one and would compute a different operation.

<!-- Figure SA3: legal-edge graph synchronized with causal matrix and shifted targets. -->

The diagonal is normally allowed because the current input token is already known. For next-token learning, the input and target arrays are shifted:

| Query position | Input available at this position | Target scored here |
| --- | --- | --- |
| 0 | `we` | `study` |
| 1 | `study` | `attention` |
| 2 | `attention` | `<end>` |

If the input at position 1 were allowed to read position 2, it would receive the very token it is asked to predict. A training loss could look excellent while measuring information leakage. The loss target, the allowed-key relation and the query's absolute position must agree.

All query rows can be computed in parallel during full-sequence training because the correct prefix tokens are already in the batch and the mask blocks the future. Autoregressive generation still has a dependency between **generated** tokens: the next token must be chosen before it can become an input for the following step.

### Padding: exclude dummy donors and dummy outputs

Different-length examples are often padded to share a rectangular batch. A dummy position should not contribute as a key/value. Masking its key column removes it from other positions' mixtures.

That does not automatically remove the dummy position as a **query**. Its output may still be computed, but it must not enter a sequence average or a per-token loss. These are separate decisions:

| Decision | What it prevents |
| --- | --- |
| Key mask inside attention | Real receivers taking information from dummy donors |
| Valid-query mask in pooling or loss | Dummy receiver outputs contributing to the task |

Our classifier's masked mean is `sum(valid * features) / sum(valid)`. A plain mean would change the denominator when extra padding is added, even if padded key columns were correctly masked.

Do not ask softmax to normalize a row with no legal keys: mathematically its denominator is zero. For nonempty examples, a simple implementation can allow padded queries to read the real keys, then discard those query outputs. Packed documents, empty records and combined causal/padding masks need a declared policy ensuring every computed query has a legal donor. A huge negative finite sentinel is not a general remedy: it can overflow in low precision, and an entirely masked row can accidentally turn into a uniform distribution. This lesson's reference function rejects an empty legal set.

### Cached decoding: a rectangular mask needs positions

While generating, a model can keep earlier keys and values in a **KV cache**. Suppose the only new query is at absolute position 2 and cached keys correspond to positions `0,1,2`. All three keys are legal. The one-row mask should be `[True, True, True]`.

A top-left triangular mask of shape `1 × 3` is `[True, False, False]`; that belongs to a query at position 0, not position 2. In our numeric fixture, the correct cached result is `[1,1]`, whereas the top-left mask returns `[2,0]`.

The robust idea is explicit: `allowed[i,j] = key_position[j] <= query_position[i]`, combined with padding or document boundaries as needed. Full-prefix and cached computations must use matching positions, the same parameters and evaluation behavior. The later GQA/MQA and MLA lessons change what is stored; the legality rule remains.

**Investigate a leak.** Construct a four-position input/target alignment and decide which cells must be blocked. Predict whether editing a future value can change the current query output under your mask. Reveal, repair any illegal edge, and repeat with a one-query cached prefix. The all-legal result is not always correct; its correctness depends on the query's actual position.

## 4. Let several heads form different mixtures

One head creates one distribution over donors for each query. That same distribution mixes all value coordinates within the head. Sometimes it is useful to give different feature groups different distributions.

For `H` heads, learn separate maps and compute

\[
Z_h=\operatorname{Attention}(XW_Q^{(h)},XW_K^{(h)},XW_V^{(h)}),
\qquad
Y=\operatorname{Concat}(Z_1,\ldots,Z_H)W_O.
\]

Concatenation places the head outputs beside each other along the feature axis. The output matrix `W_O` learns how to combine those features. It does not choose new donor weights. Equivalently, divide `W_O` into one block per head, project each head's result to the output space, and add those projected results.

### Same parameter count does not mean the same operation

Consider two input rows, `[1,0]` and `[0,1]`, with identity query/key/value/output maps. With one head of width two, the first query's scores are `[1/sqrt(2),0]`; its output is approximately `[0.6698,0.3302]`.

Split the same two coordinates into two heads of width one. Head 1 sees first-coordinate values `[1,0]`; its first query gives scores `[1,0]` and returns `0.7311`. Head 2 sees second-coordinate values `[0,1]`; its first query is zero, so both scores are zero and it returns `0.5`. Concatenate to obtain `[0.7311,0.5]`.

The learned matrices can have the same total dimensions, yet changing where softmax is applied changes the computation. Multiple heads are not produced by slicing an already-computed single-head attention matrix. And their different outputs do not guarantee they will learn useful, distinct roles; that is an empirical question.

<!-- Figure SA4: split–mix–merge lanes, including one-head versus two-head exact fixture. -->

### Follow the axes rather than memorizing a reshape

For a common arrangement, choose `d_k=d_v=d/H` and return output width `d`.

| Tensor | Shape | Interpretation |
| --- | --- | --- |
| `X` | `B × L × d` | Batch, positions, input features |
| Projected `Q`, `K`, `V` | `B × L × d` | All heads' features beside each other |
| After splitting heads | `B × H × L × (d/H)` | Separate position-by-feature table per head |
| Scores and weights | `B × H × L × L` | Receiver–donor table per head |
| Mixed values | `B × H × L × (d/H)` | Head output at each receiver |
| Merged values | `B × L × d` | Heads concatenated in feature order |
| After output map | `B × L × d` | Combined contextual representation |

For `B=2`, `L=3`, `d=8`, `H=2`, the score tensor is `2 × 2 × 3 × 3`. A dot product contracts the final feature axis; it must not contract the batch or head axis. Transposing before flattening back is necessary to put each head's features at the correct position.

With these equal-width choices and no biases, the four learned matrices contain `4d²` parameters. At `d=24`, that is `2304`, whether we use one, two or another divisor of 24 heads. Increasing the number of heads holds the total width fixed and makes each head narrower. Keeping each head's width fixed while adding heads is a different experiment and increases parameters.

More generally, with input width `d`, output width `d_out`, `H` heads, key width `d_k` and value width `d_v`, the bias-free count is

\[
dH(2d_k+d_v)+Hd_vd_{\rm out}.
\]

Biases, if present, add their own output-coordinate counts. There is no rule that every architecture must use `d_k=d_v=d/H`; the familiar count follows from those specific choices.

## 5. Implement the operation you just traced

The following program runs on a CPU with PyTorch. It deliberately materializes the attention matrix so that we can inspect it. The final comparison copies the same parameters into PyTorch's `MultiheadAttention`; it does not compare independently initialized models.

```python
import math
import torch
from torch import nn


def attention(query, key, value, allowed=None):
    # Last two axes: positions, features. True denotes an allowed edge.
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if allowed is not None:
        if not allowed.any(dim=-1).all():
            raise ValueError("Every query needs at least one allowed key")
        scores = scores.masked_fill(~allowed, -torch.inf)
    weights = scores.softmax(dim=-1)
    return weights @ value, weights


class SelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def split_heads(self, tensor):
        batch, length, _ = tensor.shape
        return tensor.reshape(
            batch, length, self.heads, self.head_width
        ).transpose(1, 2)

    def forward(self, inputs, allowed=None):
        query = self.split_heads(self.query(inputs))
        key = self.split_heads(self.key(inputs))
        value = self.split_heads(self.value(inputs))
        values, weights = attention(query, key, value, allowed)
        merged = values.transpose(1, 2).contiguous().reshape(inputs.shape)
        return self.output(merged), weights


torch.manual_seed(19)
manual = SelfAttention(width=8, heads=2).double().eval()
reference = nn.MultiheadAttention(
    embed_dim=8, num_heads=2, bias=False,
    dropout=0.0, batch_first=True
).double().eval()

with torch.no_grad():
    reference.in_proj_weight.copy_(torch.cat([
        manual.query.weight, manual.key.weight, manual.value.weight
    ]))
    reference.out_proj.weight.copy_(manual.output.weight)

inputs = torch.randn(2, 4, 8, dtype=torch.float64)
allowed = torch.ones(4, 4, dtype=torch.bool).tril()
ours, our_weights = manual(inputs, allowed)
theirs, their_weights = reference(
    inputs, inputs, inputs,
    attn_mask=~allowed,  # MHA's True means blocked.
    need_weights=True,
    average_attn_weights=False
)
print(ours.shape, our_weights.shape)
print("maximum output difference:", (ours - theirs).abs().max().item())
print(torch.allclose(ours, theirs, atol=1e-12, rtol=1e-12))
print(torch.allclose(our_weights, their_weights, atol=1e-12, rtol=1e-12))
```

The retained CPU run used Python 3.12.14 and PyTorch 2.14.0. It produced output shape `2 × 4 × 8`, weights shape `2 × 2 × 4 × 4`, two `True` comparisons, and a maximum output difference of about `5.55e-17`. This is numerical agreement for these inputs and double precision, not a promise of bit-identical results for every backend.

`nn.Linear` stores weights as `out_features × in_features` and applies their transpose internally. Our mathematical `W_Q` uses the opposite storage orientation. That difference explains the API layout; it does not change the map. We concatenate the stored query/key/value weights in the order PyTorch expects. We request per-head weights because averaging them would erase the distinction we are trying to inspect.

### Use a fused primitive when you need outputs rather than a full heatmap

PyTorch's `scaled_dot_product_attention` computes the core operation from already-projected, already-split Q/K/V. It does not supply the learned projection matrices or merge heads for you.

```python
import torch
from torch import nn
from torch.nn import functional as F


class AttentionCore(nn.Module):
    def __init__(self, dropout_probability=0.1):
        super().__init__()
        self.dropout_probability = dropout_probability

    def forward(self, query, key, value, allowed):
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=allowed,  # Here True means allowed.
            dropout_p=self.dropout_probability if self.training else 0.0,
            is_causal=False    # The explicit mask defines the relation.
        )


core = AttentionCore().eval()
q = torch.tensor([[[[1.0, 1.0]]]])       # B=1, H=1, one query
k = torch.tensor([[[[1., 0.], [0., 1.], [1., 1.]]]])
v = torch.tensor([[[[2., 0.], [0., 2.], [1., 1.]]]])
query_positions = torch.tensor([2])
key_positions = torch.arange(3)
allowed = key_positions[None, :] <= query_positions[:, None]
with torch.no_grad():
    print(core(q, k, v, allowed))  # approximately [[[[1., 1.]]]]
```

There are three distinct switches here. `.eval()` changes module training behavior. `no_grad()` avoids building the gradient graph. Passing `dropout_p=0.0` disables dropout inside this functional attention call. The functional call applies the probability you pass even when a surrounding module is in evaluation mode. Causality also does not disappear during evaluation: we continue to enforce the task's legal-edge relation.

The current SDPA API treats non-square `is_causal=True` as a top-left causal alignment. This is why the explicit-position example is safer to reason about than guessing what a rectangular triangle means. A boolean mask for this primitive means **allowed**, whereas a boolean attention or key-padding mask for `MultiheadAttention` means **blocked**. Consult the specific API rather than transferring a mask unchanged. [PyTorch SDPA reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [MultiheadAttention reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).

## 6. Inspect attention on real recorded movement

Can a small model recognize the type of a hand movement from its measured points? This gives us something richer to inspect than a heatmap whose entries were chosen to look sensible.

The [UCI Libras Movement dataset](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement) contains 360 trajectories, each represented by 45 two-dimensional points sampled from a movement video. The 15 labels describe movement types such as horizontal swing, anticlockwise arc and clockwise arc. The coordinate pairs are normalized, and the points are ordered samples rather than exact timestamps. These are motion descriptors, not complete signs or sentences. The source metadata describes four performers and two sessions; it does not supply their identifiers with each row, so our split cannot test generalization to unseen performers or sessions.

### Inspect the input before training

One source row contains 90 coordinate values followed by a class. Reshape the first 90 as `45 × 2`: `(x_1,y_1),...,(x_45,y_45)`. Do not reshape them as 45 x-values followed by 45 y-values.

Exact comparison found **330 unique trajectories**, with 30 repeated rows. We kept the first occurrence of each trajectory after verifying that repeated copies had the same label. We then split whole trajectories into 220 training, 50 validation and 60 test examples. The test set has four unique trajectories per class. The retained source file, duplicate groups and row IDs make the split inspectable. This removes exact-copy leakage; it cannot establish independence of unrecorded performer/session groups.

The only coordinate conversion is `2*x - 1`, which maps the source's unit interval to `[-1,1]`. It does not estimate a statistic from the test data. We deliberately omit position information to isolate a limitation of the attention mechanism.

### Build three set models and one order-sensitive reference

Our first model applies the same `2 → 24` linear map and `tanh` to every point, averages the 45 feature vectors, and predicts one of 15 classes. This is a learned **mean-pooling baseline**.

The attention versions use the same structure but insert either one or two attention heads. After attention, they add the mixed features to the original point features. This **residual addition** gives the classifier access to the original representation as well as the contextual change:

\[
H=\tanh(XW_{\rm stem}+b_{\rm stem}),\quad
U=H+\operatorname{MHA}(H),\quad
\text{logits}=\operatorname{mean}_{\text{rows}}(U)W_{\rm cls}+b_{\rm cls}.
\]

Here `X` contains the transformed coordinate pairs, and the biases are added to each applicable row. A **logit** is an unconstrained class score. A final softmax converts the 15 logits to class probabilities, and cross entropy penalizes low probability assigned to the true label.

The last reference is an ordinary linear classifier on all 90 coordinates in their original order. It can attach a different weight to frame 1 and frame 45. It is useful precisely because it retains information our pooled set models discard. The next lesson will assemble a full Transformer block; this small attention classifier has no normalization layers, Transformer feed-forward sublayers or positional encodings.

<!-- Figure SA5: actual source trajectory → point features → mixing → pooling → class output, with the order-sensitive reference beside it. -->

### Reproduce the complete run

Download [the complete Python program](author-calculations.py), [the small input file](movement_libras.data) and [the original variable descriptions](movement_libras.names) into one directory. The program includes the data split, all four models, the training loop, checkpoint selection, every metric and the exact attention fixtures. It uses NumPy, PyTorch and scikit-learn; the recorded run used versions 2.3.5, 2.14.0 CPU and 1.9.1, respectively. Run it from your own Python environment:

```bash
python author-calculations.py
```

The training is deliberately small: full-batch Adam, learning rate `.01`, 200 epochs, and three declared seeds per model. An **epoch** here is one update using all 220 training trajectories. Each fit selects its checkpoint by validation macro F1, then lower validation cross entropy on a tie. Macro F1 computes a separate precision/recall balance for each class and averages those 15 class scores. The test is evaluated after that selection. All 12 runs are reported; the interactive example uses the two-head model with seed 101, selected before seeing its result.

In the program, the core training sequence is `zero_grad → forward → cross_entropy → backward → optimizer.step`. Zeroing prevents the previous update's gradients from accumulating unintentionally. The validation calculation uses evaluation mode and no gradient graph, then saves a copy of the best parameter state. Evaluation does not train on validation or test examples. The program saves the measured results and a small set of learned weights, so the lesson's interactive figure can run inference without training in your browser.

| Model | Parameters | Test correct for seeds 101 / 102 / 103, each out of 60 | Test macro F1 for those seeds |
| --- | ---: | --- | --- |
| Pointwise features + mean | 447 | 22 / 10 / 11 | .361 / .145 / .138 |
| One attention head + residual + mean | 2,751 | 27 / 29 / 30 | .424 / .471 / .486 |
| Two attention heads + residual + mean | 2,751 | 36 / 28 / 35 | .556 / .442 / .577 |
| Ordered linear classifier | 1,365 | 41 / 41 / 41 | .664 / .662 / .664 |

These are actual measurements from the retained program. For the displayed two-head seed-101 model, the selected epoch was 110; it classified 152/220 training examples, 30/50 validation examples and 36/60 test examples correctly. Some other seeds selected different epochs. The JSON record contains the corresponding losses and learning curves.

Attention helped relative to the smaller mean-pooling baseline in these runs, but adding a second head did not improve every seed. The order-sensitive linear model did best on this test. The comparison changes both inductive bias and, for some models, parameter count; it does not isolate a universal causal advantage of any architecture. Three seeds reflect initialization sensitivity on this fixed split, not three independent datasets. A larger search tuned against these displayed test results would need a new appropriate evaluation.

### A wrong prediction that teaches something

The displayed real trajectory is source row 77, labeled class 4, **anticlockwise arc**. The predeclared two-head model assigns about `9.95%` to that class and `81.11%` to class 5, **clockwise arc**. We retain that mistake because it reveals a structural question: can this model even tell the direction in which the points were visited?

<!-- Investigation SA6: editable real trajectory, per-head donor map, and class probabilities. -->

Choose a receiver point and inspect its row in each head's attention map. The horizontal axis is donor point number; the vertical axis is receiver point number. High weight means a large mixing coefficient for that donor; its effect also depends on the value vector and output map. It is not automatically a physical neighbor or a movement label.

Before revealing the next output, predict what will happen if you **reverse all 45 points**. In the retained run, the largest logit change was approximately `2.15e-6`, consistent with floating-point rounding. Reversing both axes of the original heatmap also reproduces the reversed sequence's attention map up to rounding. The model changes which array index names a point, but not the pooled class decision.

Now edit point 23's x-coordinate from about `.64217` to `.35783`, leaving its y-coordinate unchanged. That changes the set of observed coordinates. The clockwise-arc probability drops from about `81.11%` to `49.74%`; the anticlockwise-arc probability rises to about `18.15%`. The predicted class stays clockwise arc, but the distribution changes substantially. This is a hypothetical edited trajectory, so its correct real-world class is unknown. The experiment distinguishes “the model ignores order” from “the model ignores its input.”

Finally add five padding points at `(.75,.75)`. With both the key mask and valid-point pooling, the largest logit difference is about `1.91e-6`. Without either mask it is about `2.2352`. A token that is merely named padding is still a real numerical input unless the computation excludes it.

### Why the reversal result is guaranteed by the architecture

A **permutation** reorders rows. Let `P` be the matrix that performs that reordering. Shared linear maps give `Q'=PQ`, `K'=PK`, `V'=PV`. The score matrix becomes

\[
Q'K'^T=PQK^TP^T.
\]

Row softmax reorders in the same way, so `A'=PAPᵀ`. Therefore

\[
Z'=A'V'=PAP^TPV=PAV=PZ.
\]

The output rows move with the input rows. This property is **permutation equivariance**. Shared pointwise nonlinearities, head merging and residual addition preserve it. Taking the mean removes the row order entirely, giving **permutation invariance** of the final class scores.

This proof assumes there is no position-dependent feature, fixed positional bias or unpermuted mask. A fixed causal mask changes the argument because it encodes an order-dependent relation. In our bidirectional trajectory model, however, the condition holds exactly in real arithmetic.

Permutation invariance is desirable when the input truly is an unordered collection, such as a set of detected objects whose list order is arbitrary. It is a limitation when different visit orders carry different meanings. To represent motion direction, we can supply time/position information or another order-sensitive mechanism. We cannot repair missing information by interpreting a prettier heatmap. Positional Encodings, two lessons ahead, develops several ways to supply that information.

## 7. Deeper branch: why the scores and gradients behave this way

### The square-root scaling is an initialization argument

Suppose each coordinate of `q` and `k` is independent, has mean zero and variance one. A product `q_r k_r` then has mean zero and variance one. If these products are independent across coordinates,

\[
\operatorname{Var}(q\cdot k)
=\operatorname{Var}\left(\sum_{r=1}^{d_k}q_rk_r\right)=d_k.
\]

Its typical magnitude therefore grows with `sqrt(d_k)`. Dividing by that factor gives variance one under these assumptions. Without the division, merely increasing feature width tends to spread the initial scores further apart, making softmax more concentrated.

This is a useful design argument, not a law about learned representations. Trained coordinates may be correlated or have different scales; normalization layers and projection weights also matter. The scale uses the **per-head key width**, not the model width before splitting heads. Other designs can deliberately normalize queries and keys or learn a temperature, but they define a modified scoring function whose behavior should be tested.

For a distribution concentrated almost entirely on one donor, the softmax derivatives become small:

\[
\frac{\partial a_j}{\partial s_r}=a_j(\mathbf1[j=r]-a_r).
\]

As one probability tends to one and the rest tend to zero, all entries of this Jacobian tend to zero. The dominant probability's derivative does not stay large. At the other extreme, uniform attention can dilute a useful donor among many candidates. Neither sharp nor broad attention is automatically healthy; the task determines what needs to be distinguished.

<!-- Investigation SA7: score-gap/temperature curve with probabilities and local derivatives. -->

### Stable softmax changes the arithmetic, not the distribution

Subtract the largest allowed score `m` before exponentiating:

\[
\frac{e^{s_j-m}}{\sum_r e^{s_r-m}}
=\frac{e^{s_j}}{\sum_r e^{s_r}}.
\]

The common factor cancels. Exponentials now have arguments at most zero, avoiding overflow from large finite scores. Adding `1000` to every score in our first fixture row leaves its weights unchanged; the retained double-precision calculation differs by about `1.44e-15` through rounding.

This stabilizes exponentiation. It cannot repair a dot product that already overflowed to infinity, invalid input values, or a row with no legal keys. The masking policy in §3 handles the last case. Standard softmax implementations normally perform stable normalization internally.

### Differentiate the complete mixing operation

Focus on one receiver. Write `z = sum_j a_j v_j` and let `g = ∂L/∂z` be the gradient arriving from later layers. If the weights were fixed, each value would receive the simple gradient

\[
\frac{\partial L}{\partial v_j}=a_jg.
\]

But the weights are learned too. Applying the softmax derivative gives

\[
\frac{\partial L}{\partial s_j}
=a_j\,g^T(v_j-z).
\]

The term `v_j-z` compares a donor's value with the current mixture. Increasing that donor's score moves the mixture toward its value; the upstream gradient says whether that movement increases the loss.

For the fixture's receiver A and an illustrative upstream gradient `g=[1,-0.5]`, the three score gradients are approximately

\[
[.47933,-.35699,-.12234].
\]

They sum to zero, as they must: adding a constant to all scores has no effect. Since `s_j=qᵀk_j/sqrt(d_k)`,

\[
\frac{\partial L}{\partial q}
=\frac{1}{\sqrt{d_k}}\sum_j
\frac{\partial L}{\partial s_j}k_j,
\qquad
\frac{\partial L}{\partial k_j}
=\frac{1}{\sqrt{d_k}}
\frac{\partial L}{\partial s_j}q.
\]

The query gradient in this example is approximately `[0.25243,-0.33894]`; the retained program verifies the calculation against automatic differentiation. For a whole attention matrix, donor keys and values accumulate gradients from **all** receivers that use them. Finally, because the same input feeds the query, key and value maps, the input gradient includes all three paths. Treating attention weights as fixed would omit two of those routes through the scores.

A useful special case connects this to geometry. Hold the key collection fixed. If keys and values are identical and there is no score scaling, `z(q)=sum_j a_j(q)k_j` has Jacobian

\[
\frac{\partial z}{\partial q}
=\sum_j a_j k_jk_j^T-zz^T
=\operatorname{Cov}_{a}(k).
\]

With score division by `sqrt(d_k)`, multiply this covariance by `1/sqrt(d_k)`. The formula follows by substituting `v_j=k_j` in the derivative above. Directions in which all keys have the same coordinate have zero weighted variance and cannot move the output along that direction by changing the query. This is a precise statement for this special case; independently learned value maps do not generally give the same covariance Jacobian. [D2L's Q/K/V chapter](https://www.d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html) develops attention pooling and poses this connection as a deeper exercise.

### Low-rank scores do not imply low-rank attention weights

`QKᵀ` has rank at most `d_k`, which constrains the score matrix. Row softmax is nonlinear, so its output can have a higher rank. For example,

\[
S=\begin{bmatrix}0&0\\0&1\end{bmatrix}
\quad\text{has rank 1, but}\quad
\operatorname{softmax}(S)=
\begin{bmatrix}.5&.5\\.26894&.73106\end{bmatrix}
\]

has nonzero determinant, about `.23106`, and rank 2. This matters when reading claims about low-rank attention approximations: a narrow query/key representation is not by itself a proof that the normalized attention matrix has that same rank.

There is also a kernel-regression connection. A Gaussian distance score expands as

\[
-\tfrac12\|q-k_j\|^2
=q^Tk_j-\tfrac12\|k_j\|^2-\tfrac12\|q\|^2.
\]

The final term is constant across one query's donors and disappears under softmax. The key-norm term disappears only if all the relevant key norms are equal. Thus distance-based weighting and dot-product weighting are related, but they are not generally interchangeable. A normalization layer followed by an arbitrary learned projection does not guarantee equal projected key norms. [D2L's scoring-functions chapter](https://www.d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html) is a useful route into this connection.

## 8. Deeper branch: inspect attention without inventing a story

### Entropy measures concentration, not understanding

For one legal attention row, define entropy in **nats** using natural logarithms:

\[
H(a)=-\sum_{j\in\text{allowed}} a_j\log a_j.
\]

Use `0 log 0 = 0` by continuity. If a query has `m` legal keys, entropy is at most `log(m)`, achieved by uniform weights. It approaches zero when weight concentrates on one key.

In a causal sequence, `m` changes with the row. With zero-based query index `i`, an unpadded full prefix has `m=i+1`. Its uniform-entropy ceiling is `log(i+1)`. For 17 such rows, the mean uniform ceiling is

\[
\frac1{17}\sum_{m=1}^{17}\log m
=\frac{\log(17!)}{17}\approx1.97089,
\]

not `log(17)≈2.83321`. The first row has only one legal key and entropy zero, regardless of how well trained the model is. Comparing a causal head's mean entropy with the last row's ceiling creates a misleading “concentration” effect.

A per-row normalized quantity `H(a)/log(m)` can compare concentration across rows with different `m>1`; the `m=1` case has an undefined ratio and should be reported separately. Even after normalization, `[.9,.1]` and `[.1,.9]` have identical entropy while favoring opposite donors. Head entropy alone does not establish specialization, usefulness or redundancy.

### Same output, different attention

Let two donors both have value `[3,-2]`. Weights `[.9,.1]` produce `[3,-2]`; weights `[.1,.9]` also produce `[3,-2]`. The heatmaps differ strongly, yet the head output is exactly the same.

More generally, a change `δ` in a row's weights preserves its value mixture when `δV=0`. It must also satisfy `sum(δ)=0` and keep the new weights nonnegative. Output projections and downstream layers can create further directions of insensitivity. A large attention weight is a fact about one mixing operation; it is not automatically a measure of the effect of deleting the original input, whose removal can change the keys, values and all other attention rows too.

This does not make attention maps useless. They can reveal a broken mask, repeated donors, broad or concentrated matching, or a hypothesis worth testing. The interpretability debate distinguishes those uses from an unsupported causal explanation. [Jain and Wallace](https://aclanthology.org/N19-1357.pdf) study mismatches between attention and other importance measures; [Wiegreffe and Pinter](https://aclanthology.org/D19-1002.pdf) examine what different explanatory tests and model-consistent interventions actually establish. Their experiments concern particular models and tasks, not a theorem declaring every attention plot meaningless.

For our trajectory, a defensible statement is “this head assigns these weights to these donor frames.” To investigate importance, specify an intervention and measure the resulting change: edit one coordinate, mask a donor while renormalizing, or compare a trained model with a declared baseline. State what was held fixed. Removing a head only at inference is a different question from retraining a model with fewer heads. Neither becomes a general explanation just because the chart is colorful.

**Investigate an unchanged output.** Choose two different attention distributions, then edit the donor values until both distributions yield the same mixture. Predict whether an output map could recover a difference that has already vanished. Reveal the exact vectors and explain your construction. The goal is to reason about information flow, not to find the “right-looking” heatmap.

<!-- Investigation SA9: editable distributions and values, with mixture difference and entropy. -->

### A low loss can answer the wrong question

Consider a synthetic copying task with a fresh independent source of eight symbols, each uniformly drawn from 31 possible symbols:

`s0 s1 s2 s3 s4 s5 s6 s7 SEP s0 s1 s2 s3 s4 s5 s6 s7`

Train a next-token predictor on all 16 shifted targets. Seven targets are new random source symbols; one is the separator; eight are copied symbols. Conditional on the available prefix, each new random symbol still has entropy `log(31)`. Even an ideal predictor of the separator and copies therefore has expected all-position cross entropy at least

\[
\frac{7}{16}\log(31)\approx1.50237\text{ nats per scored target}.
\]

This is a statement about the fresh random data distribution. Memorizing a finite training set could produce a lower training loss without predicting fresh random symbols. If instead you score only the eight copied targets, the irreducible random-source terms are excluded and the copy-only loss can approach zero. These are different metrics.

The query at `SEP`, position 8, predicts the first copied symbol at position 9. It does not wait until the first copied symbol is already present as its own input. Writing this alignment before training helps detect accidental target leakage and off-by-one visualizations. Whenever you report a loss or accuracy, name which positions enter its denominator.

## 9. Deeper branch: computation, memory and choosing an implementation

Dense self-attention connects every query with every legal donor. This gives a short information path, but full pairwise comparison grows with the square of sequence length.

For ordinary equal-width multi-head attention with model width `d`, batch size `B` and length `L`, the major forward matrix products require approximately

\[
\underbrace{4BLd^2}_{Q,K,V,\text{ output projections}}
+\underbrace{2BL^2d}_{QK^T\text{ and }AV}
\]

**multiply-accumulate operations**. Counting a multiplication and an addition as two FLOPs roughly doubles those matrix-product counts. Softmax, masks and elementwise operations add work. A triangular mask reduces the number of legal pairs, but an implementation that computes the whole square before masking may still execute square-sized products.

If a materialized attention matrix uses `b` bytes per element, it occupies `BHL²b` bytes. At `B=1`, `H=16`, `L=2048`, `b=2`, one such matrix uses **128 MiB**. That is one tensor, not a full training-memory estimate: scores, saved activations, parameters, gradients, optimizer states and temporary buffers can add more.

<!-- Figure SA8: derived operation counts and attention/cache memory with explicit units and assumptions. -->

During cached generation, old keys and values avoid recomputing their projections for each newly generated token. For `N_layers` layers with equal query/key/value head counts and width `d/H`, a full KV cache contains

\[
2\,B\,N_{\rm layers}\,L\,d\,b
\]

bytes for keys plus values. For `B=1`, `N_layers=12`, `L=2048`, `d=1024`, `b=2`, that is **96 MiB**. Real caches can also have metadata, padding, allocation granularity or different data types. The attention matrix and KV cache solve different storage problems; their formulas should not be conflated.

For one new query, scoring cached keys and mixing cached values still visits the prefix: approximately `2BLd` multiply-accumulates per layer for those two products. The cache avoids rebuilding old K/V, but it does not make dense attention independent of prefix length. There is usually no need to retain every old query for this ordinary forward-decoding calculation.

### Exact attention does not require storing the whole square

Softmax appears to need all scores at once, but it is possible to combine correctly normalized partial calculations. For one query and a block of keys, keep the largest score `m`, the shifted exponential sum `l=sum(exp(s_j-m))`, and the shifted weighted-value sum `u=sum(exp(s_j-m)*v_j)`.

For two blocks, choose `m=max(m_1,m_2)` and combine

\[
l=e^{m_1-m}l_1+e^{m_2-m}l_2,\qquad
u=e^{m_1-m}u_1+e^{m_2-m}u_2.
\]

Then `z=u/l`. Averaging each block's already-normalized output with equal weights would generally be wrong because the blocks can have different total exponential mass. The rescaling preserves the same global softmax denominator.

FlashAttention uses tiling and such normalization machinery to reduce memory traffic and avoid storing the entire pairwise matrix. It computes the same intended attention operator, subject to floating-point differences. Backward recomputation can even increase arithmetic while reducing costly memory transfers; “faster” is not synonymous with “fewer multiply-adds.” Actual speed depends on shapes, dtype, hardware and kernel support. [Dao et al., §§2–3](https://arxiv.org/pdf/2205.14135) explains the algorithm and its memory analysis. The later Ring Attention lesson extends the coordination question across devices.

In practice, use the supported framework primitive when it fits your mask, dimensions and hardware. Requesting every head's full weight matrix for visualization has an unavoidable output-size cost. Inspect selected rows or small examples when that answers the question; keep dense heatmaps out of a long-sequence production path. The reference implementation in §5 remains useful for understanding and checking the operator on small inputs.

### Identify what an alternative actually changes

| Approach | What changes | What you must still establish |
| --- | --- | --- |
| Exact tiled attention | How the same computation is scheduled and stored | Compatible masks, numerical agreement and actual runtime behavior |
| Sparse attention | Which query–key edges exist | Whether useful information can travel through the chosen pattern |
| Feature-map linear attention | A kernel or approximation enabling a different factorization | The operator being computed, normalization and approximation limits |
| GQA/MQA | Sharing key/value heads across several query heads | Cache reduction and the model's learned quality under sharing |
| MLA | Compressed latent K/V representation and a compatible attention computation | Cache reconstruction/absorption and positional components |
| Recurrent/SSM alternatives | How the prefix is summarized in a state | State update, retained information and train/decode costs |

You cannot move the ordinary row softmax through a matrix product and simply rewrite exact attention as `Q(KᵀV)`. The nonlinear normalization couples each query to all its legal key scores. Earlier RWKV and SSM lessons use particular recurrent operators; they should not be read as algebraic rearrangements of every softmax attention matrix. Sparse & Linear Attention Variants will make those distinctions concrete.

For a short sequence, projections can be a substantial share of the work; for a long sequence, the pairwise term grows more quickly. For generation, bandwidth and cache layout can dominate a small query's arithmetic. Decide which constraint you actually have before replacing the architecture. Operation counts explain scaling; measured timing on a stated workload answers a different question.

## 10. Practice, investigate and explain

Work from the question before opening a hint or solution. The first four tasks test the first-pass route. The later tasks connect the mathematical and systems branches.

### 1. Separate matching from message content

A single query is `[2,0]`. Its two keys are `[1,0]` and `[0,1]`, and their values are `[1,3]` and `[5,1]`. Use `d_k=2`.

Compute the attention weights and output. Then change the second value to `[5,5]` without changing its key. Which intermediate quantities change, and what is the new output's second coordinate?

<details>
<summary>Hint</summary>

The scaled scores are `sqrt(2)` and zero. First compute the second donor's weight; its changed second coordinate contributes an additional four times that weight.

</details>

<details>
<summary>Solution and reasoning</summary>

The first weight is `exp(sqrt(2))/(exp(sqrt(2))+1)≈.8044297`; the second is `.1955703`. The original output is approximately `[1.7822813,2.6088594]`. Changing only the second value leaves scores and attention weights unchanged. It adds `4*.1955703≈.7822813` to the output's second coordinate, making it approximately `3.3911406`. The new mixture is still between its two new value endpoints.

</details>

### 2. Prevent information leaking across packed documents

Five positions contain two short documents. Their document IDs are `[A,A,A,B,B]`, and their local positions are `[0,1,2,0,1]`. Each query predicts the next token in its own document. Write the legal donor indices for every row, and explain why a global lower-triangular mask is insufficient.

<details>
<summary>Hint</summary>

Legality requires both “same document” and “donor position no later than query position.”

</details>

<details>
<summary>Solution and reasoning</summary>

With zero-based indices, the legal sets are `{0}`, `{0,1}`, `{0,1,2}`, `{3}` and `{3,4}`. A global lower triangle would let rows 3 and 4 read document A. The relation is `(doc_i == doc_j) and (position_j <= position_i)`. End-of-document targets also need their own task/loss policy; packing does not authorize predicting the first token of B as the continuation of A. If a sixth slot is padding, exclude it as a donor and from scored targets; skip its query computation or give its unused query a defined nonempty legal set before discarding the output.

</details>

### 3. Account for axes, parameters and one stored tensor

Use `B=3`, `L=5`, `d=12`, `H=3` and equal per-head query/key/value widths. Give the split Q shape, attention-weight shape and merged-output shape. Count bias-free attention parameters. How many bytes would one float32 attention-weight tensor occupy?

<details>
<summary>Hint</summary>

Keep the head axis separate from the position axes. Float32 uses four bytes per element; do not include unrelated model tensors in this question.

</details>

<details>
<summary>Solution and reasoning</summary>

Per-head width is 4. Split Q is `3 × 3 × 5 × 4`; weights are `3 × 3 × 5 × 5`; merged output is `3 × 5 × 12`. The four projections contain `4*12²=576` weights. The attention tensor contains 225 elements and occupies 900 bytes. If all four projections had biases, they would add 48 parameters, but that is outside the stated bias-free count. Neither count is a full training-memory estimate.

</details>

### 4. Duplicate points: a second invariance to test

Take one nonempty trajectory in the no-position, mean-pooled attention classifier. Make a new input by repeating **every** point exactly twice. Predict the output change. Then predict whether the same argument holds if only the first point is repeated once.

You can test this using the retained model: `points.repeat_interleave(2, dim=1)` repeats all points, while `torch.cat([points, points[:, :1]], dim=1)` repeats only the first. These are changes to the represented multiset, not newly collected independent observations.

<details>
<summary>Hint</summary>

When every donor is doubled, the softmax denominator doubles too. Track the total weight of both copies of one value. Then track the final query mean.

</details>

<details>
<summary>Solution and reasoning</summary>

Doubling every donor divides each individual copy's weight by two. Its two identical values together receive the original total contribution. Query outputs are repeated as well, so their mean is unchanged. Shared pointwise maps and residual addition preserve the argument. The saved seed-101 model's maximum logit difference on source row 77 is about `3.81e-6`, consistent with roundoff.

Repeating only one point changes its multiplicity relative to all other donors and also changes the final averaging distribution. There is no general invariance. On that same trace, the maximum logit change was about `.11611`. These results apply to this architecture without positional features or length-dependent readouts; a sum readout, for example, would double when all outputs are doubled.

</details>

### 5. Distinguish two entropies

One unpadded causal sequence has four query rows, each uniform over its legal prefix. Calculate its mean entropy in nats. Separately, compare the entropies of `[.8,.2]` and `[.2,.8]`. What can those entropy values establish about the selected donors?

<details>
<summary>Hint</summary>

The four legal-key counts are 1, 2, 3 and 4. Entropy is unchanged when the entries of a distribution are permuted.

</details>

<details>
<summary>Solution and reasoning</summary>

The mean is `log(1*2*3*4)/4 = log(24)/4≈.7945135`, not `log(4)`. Both two-donor distributions have entropy `-.8 log(.8)-.2 log(.2)≈.5004024`. They have equal concentration but favor different donors. The scalar entropy cannot tell you which donor was selected or whether that selection improved the task.

</details>

### 6. Follow a gradient through a value mixture

There are two scalar values, `v_1=0` and `v_2=4`, with weights `[.25,.75]`. Let `g=∂L/∂z=1`. Compute the output, both value gradients and both score gradients. Explain the sign of the first score gradient.

<details>
<summary>Hint</summary>

Use `∂L/∂v_j=a_j g` and `∂L/∂s_j=a_j g(v_j-z)`. Do not forget that softmax redistributes weight across both donors.

</details>

<details>
<summary>Solution and reasoning</summary>

The output is 3. Value gradients are `.25` and `.75`. Score gradients are `.25*(0-3)=-.75` and `.75*(4-3)=.75`. Increasing the first score moves weight toward the smaller value and decreases the output, which decreases the local loss when `g=1`. The score gradients sum to zero. Treating only the first weight as changing would miss the necessary reduction in the other weight.

</details>

### 7. Combine attention blocks correctly

A query has three already-scaled scores `[0, log(2), log(4)]` and scalar values `[1,3,6]`. Process donors 1–2 as one block and donor 3 as another. Find the full attention output. Why is the equal average of the two block outputs wrong?

<details>
<summary>Hint</summary>

The unshifted exponential masses are 1, 2 and 4. The blocks have total masses 3 and 4. For these small scores, unshifted arithmetic is safe; the running-maximum form gives the same answer.

</details>

<details>
<summary>Solution and reasoning</summary>

The first block's normalized output is `(1*1+2*3)/3=7/3`; the second's is 6. Their masses are unequal, so combine them as `(3*(7/3)+4*6)/(3+4)=31/7≈4.4285714`. Their equal average is `25/6≈4.1666667`, a different value. In the stabilized calculation, the first block has `m_1=log(2)`, `l_1=1.5`, `u_1=3.5`; the second has `m_2=log(4)`, `l_2=1`, `u_2=6`. Rescaling the first block by one half gives `l=1.75`, `u=7.75`, hence the same `31/7`.

</details>

### 8. Audit a copying-task metric before training

A fresh source has four independent uniform symbols from an alphabet of size 10. The full sequence is `s0 s1 s2 s3 SEP s0 s1 s2 s3`, and the model scores all eight shifted next-token targets. What is the irreducible contribution to expected all-position cross entropy? Which query predicts the first copied symbol?

<details>
<summary>Hint</summary>

Count the source symbols that are targets before the separator. The first source symbol is already the initial input, so it is not one of those targets.

</details>

<details>
<summary>Solution and reasoning</summary>

Three targets are fresh independent symbols. Their contribution is `(3/8) log(10)≈.8634694` nats per scored target, even if separator/copy predictions were ideal. The query at `SEP`, zero-based position 4, predicts `s0` at position 5. A copy-only metric scores four different target positions and can approach zero. A very low training loss on a fixed tiny dataset could also reflect memorization; that does not defeat the entropy argument for fresh samples.

</details>

## Continue from the mechanism to a complete model

You can now explain what information a head mixes, how the mask defines legal communication, why heads need separate normalizations, and why an order-blind pooled model cannot infer visit order. The next topic in this module is [Transformer Block Architecture](/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals). It combines this operation with a positionwise feed-forward network, residual paths and normalization, and shows what must remain consistent when blocks are stacked.

After that, **Positional Encodings** explains how to represent order and distance. **GQA/MQA** and **MLA** then focus on how attention stores and reuses the prefix. **Sparse & Linear Attention Variants** changes the communication pattern or operator. **Vision Transformers** applies the same core ideas to image patches, where two-dimensional layout creates new design choices. These are connected questions about what information is available, how it moves and what it costs.

## References and another way to learn

- [Attention in transformers, step-by-step — 3Blue1Brown, Deep Learning Chapter 6](https://www.3blue1brown.com/lessons/attention/). A visual video with creator-hosted explanatory notes. Use the query/key/value and multi-head portions after §2. Its diagrams place query distributions in columns, the transpose of this lesson's row convention; follow the labeled axes. Its `Value_down`/`Value_up` explanation factors the value and output maps. The creator explicitly treats the adjective/noun example as an illustration rather than a measured head role. The written creator notes were inspected for this lesson; no unverified timestamp is required.
- [Dive into Deep Learning: Attention Mechanisms and Transformers](https://www.d2l.ai/chapter_attention-mechanisms-and-transformers/index.html). Work through §§11.1, 11.3 and 11.5 for pooling, scoring and executable multi-head code; §11.6 connects self-attention with order. The surrounding chapter provides the broader section map. Use the mask and gradient qualifications in this lesson when translating simplified examples to another API.
- [Attention Is All You Need — Vaswani et al.](https://arxiv.org/pdf/1706.03762). Read §3.2 for the original scaled dot-product and multi-head formulation, then §4 for the original comparison with recurrent and convolutional computation. The paper's architecture dimensions and historical measurements describe its experiment, not universal defaults.
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) and [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html). Use these as implementation references for shapes, masks, dropout and supported execution paths. The lesson's small CPU reconciliation uses the installed 2.14.0 implementation.
- [UCI Libras Movement dataset](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), by Daniel Dias, Sarajane Peres and Helton Bíscaro, DOI [10.24432/C5GC82](https://doi.org/10.24432/C5GC82). The CC BY 4.0 source and original variable descriptions support the real trajectory investigation. [Data provenance and exact split](data-provenance.md), [complete reproduction program](author-calculations.py), [recorded results](author-results.json).
- [Attention is not Explanation — Jain and Wallace](https://aclanthology.org/N19-1357.pdf), paired with [Attention is not not Explanation — Wiegreffe and Pinter](https://aclanthology.org/D19-1002.pdf). Read the experimental questions and model/intervention assumptions together after §8. They give a more useful framework than assigning semantic labels to arbitrary heatmap patterns.
- [FlashAttention — Dao et al.](https://arxiv.org/pdf/2205.14135). For the systems branch, §§2–3 explain why tiled exact attention can reduce memory traffic; Appendix B supplies the detailed backward algorithm. GPU kernel implementation is beyond this lesson's small CPU experiment.
