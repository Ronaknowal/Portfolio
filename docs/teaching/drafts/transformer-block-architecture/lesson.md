# Transformer Block Architecture

A hand follows a curved path. To recognize the movement, a model needs more than a list of isolated coordinates: it needs to relate different moments, combine the resulting evidence, and revise its description of the movement. The previous [self-attention lesson](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals) supplied the communication mechanism. A **Transformer block** packages that communication with a small feature-processing network and carefully arranged update paths. Several blocks can then refine the same sequence of representations.

This lesson follows one question throughout: **what changes, and what remains available, as information passes through a block?** Answering it lets you read an architecture diagram, implement the actual computation, diagnose a misleading gradient plot, and understand why two models both called Transformers need not have the same behavior.

**First pass:** read §§1–6, try the block and movement investigations, and solve exercises 1, 2 and 4. You should be able to trace the tensor shapes and explain normalization placement before continuing. Sections 7–9 develop gradient geometry, architecture variants and systems costs; they are deeper branches you can return to. The remaining exercises follow those deeper sections. The next topic is positional encoding, not a requirement to master every large-model training method first.

## 1. A repeated workspace for each sequence position

Represent one sequence as a table `X` with `L` rows and `d` columns. A row belongs to a position: a word piece, an image patch, or a sampled hand location. Its columns are learned features. With a batch of `B` sequences the shape is `(B,L,d)`. The raw hand coordinates have only two columns; a learned input projection can turn them into a wider representation before any block runs.

A standard block preserves this outer shape. Its intermediate feed-forward computation may use more columns, and its attention creates relationships between rows, but its output is again `(B,L,d)`. That compatible interface is why blocks can be stacked.

The important operations have different jobs:

| Operation | What it reads to update one position | What it does |
|---|---|---|
| Self-attention | Other permitted positions as well as this one | Builds context-dependent mixtures of projected values |
| Positionwise feed-forward network, or FFN | The current representation at this position | Detects and combines learned feature patterns |
| Residual addition | A saved representation and a same-shape update | Adds the update to the existing representation |
| Feature normalization | The features within this position | Re-centers/rescales, or just rescales, the vector using a declared rule |

“Positionwise” does not mean “a separate network trained for every position.” The **same** FFN parameters are reused at every row. Its input can already contain information from distant positions because attention ran earlier. A local operation on a contextual representation can therefore respond to a global pattern.

**Inline figure — communication and revision.** Three position lanes enter a block. Attention draws legal cross-lane edges; the FFN then operates separately within each lane. Each sublayer has a labeled addition junction with a visible bypass. Mark the sequence axis and feature axis, and keep `(B,L,d)` beside the entering and leaving stream. Follow one position's representation rather than replacing this with a stack of unexplained boxes.

### Refresh the attention part

In one head, project the input into queries `Q`, keys `K` and values `V`. A receiving query compares against permitted keys, normalizes those scores, and mixes their values:

\[
A=\operatorname{softmax}_{\text{keys}}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),
\qquad Y=AV.
\]

Here `d_h` is the query/key width, and `M` represents a mask: zero for a legal pair and negative infinity for a forbidden pair in the mathematical formula. Multiple heads perform separate mixtures; concatenation and an output projection return width `d`. The previous lesson derives these operations and explains numerical masking. In this lesson write the whole same-shape attention transformation as `Attn(X)` so that we can concentrate on its surrounding architecture.

### A block, a stack and a task model are different things

A block transforms representations. A **stack** applies several blocks, usually with separately learned parameters. A **task model** also includes an input representation and an output rule. A language model projects each final row into vocabulary logits. A movement classifier can pool final rows and project the pooled vector into movement classes. An image model first constructs patch representations. The same block does not decide those task boundaries for you.

The original 2017 model combined an encoder stack and a decoder stack. Its encoder block had self-attention followed by an FFN. Its decoder block inserted cross-attention between masked self-attention and the FFN; queries came from the decoder, while keys and values came from the encoder. The original FFN used **ReLU**, and normalization followed each residual addition. These are historical design choices, not a definition requiring every Transformer to retain them. [Vaswani et al., §3](https://arxiv.org/pdf/1706.03762)

## 2. Read the wiring: pre-norm and post-norm

A residual connection saves the incoming representation and adds a learned update. If a sublayer produces `U` from `X`, the addition is `X+U`, not concatenation. Both tensors must describe matching positions and features. The [residual connections lesson](/learn/path/full-curriculum/residual-connections-skip-connections?module=deep-learning-fundamentals) explains why an identity path helps information and derivatives travel through a network.

Normalization can occur **inside the update branch** or **after the addition**. That choice changes the function even if every parameter tensor is identical.

For a **pre-norm block**, omitting dropout for the moment:

\[
\begin{aligned}
H&=\operatorname{Norm}_1(X),\\
U&=\operatorname{Attn}(H),\\
Z&=X+U,\\
G&=\operatorname{Norm}_2(Z),\\
V&=\operatorname{FFN}(G),\\
Y&=Z+V.
\end{aligned}
\]

The attention branch reads a normalized view of the input; the addition still receives the original `X`. The FFN branch reads a normalized view of the already updated `Z`; the second addition retains `Z`. Do not accidentally use `Norm₂(X)` in the FFN branch: it would miss the attention update from this block.

For a **post-norm block**:

\[
\begin{aligned}
U&=\operatorname{Attn}(X),\\
Z&=\operatorname{Norm}_1(X+U),\\
V&=\operatorname{FFN}(Z),\\
Y&=\operatorname{Norm}_2(Z+V).
\end{aligned}
\]

Now every path to `Z`, including the bypass, passes through the first normalization. The second normalization similarly acts on the whole updated state. “Pre” and “post” describe this placement relative to the sublayer/residual computation; they do not mean input-data preprocessing versus postprocessing.

**Inline figure — two circuits, same components.** Put the two equations beside two wiring diagrams. In pre-norm, the normalizer belongs on the fork leading into each sublayer. In post-norm, it lies after the addition where both paths have joined. Selecting a junction reveals its actual input variables. This drawing should make it impossible to confuse `X+Attn(Norm(X))` with `Norm(X)+Attn(Norm(X))`.

### The zero-update test

Temporarily set both learned sublayer outputs to zero. Pre-norm returns `Y=X`: both additions preserve the incoming stream. Post-norm returns `Norm₂(Norm₁(X))`. It can still change the input even though attention and the FFN contribute nothing.

For `X=[1,2,5,8]` with identity LayerNorm gains, zero offsets and epsilon `10⁻⁵`, pre-norm leaves `[1,2,5,8]`. Post-norm produces approximately `[-1.095440,-.730293,.365147,1.460586]` after its two normalizations. The latter is a useful representation, but it is not the identity function.

This is a precise structural difference. It does not establish that one design always learns better. Training behavior also depends on initialization, depth, optimizer, task and the actual loss. We will test the distinction without converting it into a universal architecture ranking.

## 3. What the normalizer actually normalizes

For a vector `x` of width `d`, **LayerNorm** first calculates statistics across its features:

\[
\mu=\frac1d\sum_i x_i,\qquad
v=\frac1d\sum_i(x_i-\mu)^2,\qquad
\operatorname{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{v+\epsilon}}+\beta_i.
\]

The learned gain `γ` and offset `β` have one entry per feature. Epsilon is a small positive number that keeps the denominator usable near a constant vector. The variance here uses divisor `d`, not the sample-estimation divisor `d−1`. This operation is repeated independently at each `(batch,position)` pair. It does not collect statistics across the time axis, and it does not need running averages from earlier batches. [Ba et al., §3](https://arxiv.org/pdf/1607.06450)

For our four-feature vector, `μ=4` and `v=7.5`. Subtracting four yields `[-3,-2,1,4]`; dividing by `√(7.5+10⁻⁵)` gives approximately:

\[
[-1.095444,-.730296,.365148,1.460593].
\]

The centered, rescaled vector has root mean square almost one and Euclidean length almost **two**, because `√d=2`. Calling its length “one” confuses RMS with L2 norm. Its variance is `v/(v+ε)`, slightly below one. Learned unequal gains and offsets can subsequently change its mean, variance and length again.

**RMSNorm** uses the same feature group but skips subtracting the mean:

\[
\operatorname{RMSNorm}(x)_i=
\gamma_i\frac{x_i}{\sqrt{\frac1d\sum_jx_j^2+\epsilon}}.
\]

For `[1,2,5,8]`, the denominator is approximately `√23.5`, giving `[.206284,.412568,1.031421,1.650274]`. Its mean is positive. The common form shown here has learned gain and no additive offset. Removing the centering calculation changes both invariances and computation; it is not a claim that the two functions are numerically interchangeable. [Zhang & Sennrich, §4](https://arxiv.org/pdf/1910.07467)

### An offset and a scale are different interventions

Add five to every feature, producing `[6,7,10,13]`. LayerNorm removes the common offset and gives the same output, up to rounding. RMSNorm changes because it measures distance from zero rather than distance from this vector's mean. For a zero-mean input with matching gain, offset zero and matching epsilon, the two normalizers agree exactly.

Multiplying every feature by a positive constant leaves either normalized direction unchanged when epsilon is zero and the denominator is nonzero. With positive epsilon, that scale invariance is approximate. Negative scaling flips the normalized direction before affine offsets. A constant vector becomes zero under centered LayerNorm before its affine offset; a nonzero constant vector does not become zero under RMSNorm.

**Investigation — choose the ruler.** Edit all four input features, then predict whether a common offset will change LayerNorm and RMSNorm. Record the prediction before computing. Compare the original and shifted vectors as aligned dot plots with their means and zero visible. Next construct a zero-mean vector where both methods agree; changing one feature should break that agreement. The [normalization lesson](/learn/path/full-curriculum/batch-layer-group-rms-normalization?module=deep-learning-fundamentals) develops the wider family and its derivatives.

For causal prediction, these axes matter. `LayerNorm(d)` on `(B,L,d)` only uses one position's features. Normalizing jointly over `(L,d)` can let future inputs change an earlier normalized state even when attention has a perfect causal mask. Similarly, padding must be excluded from any pooling over positions. An architecture name cannot repair an incorrect reduction axis.

## 4. The FFN: detect a feature pattern and write an update

For row-vector notation, a two-layer FFN is

\[
\operatorname{FFN}(x)=\phi(xW_{\rm up}+b_{\rm up})W_{\rm down}+b_{\rm down},
\quad
W_{\rm up}\in\mathbb R^{d\times f},\quad
W_{\rm down}\in\mathbb R^{f\times d}.
\]

The hidden width `f` is the number of intermediate feature responses. A common design expands it beyond `d`, but expansion is a modeling choice. The up projection asks learned questions about the representation; the activation changes each response; the down projection combines those responses into a `d`-feature update. Each row of `W_down` is an update direction multiplied by its corresponding hidden activation.

Without an activation, the composition collapses to one affine map: `x(W_up W_down)+(b_up W_down+b_down)`. The intervening nonlinearity is what makes this particular two-map FFN more expressive than a single affine transformation. **Attention itself is already nonlinear through its input-dependent softmax**, and normalization is nonlinear too. An FFN is not the only source of nonlinearity in a Transformer.

### A feature circuit you can calculate

Consider representation `g=[1,2,-1,-2]` and no biases:

\[
W_{\rm up}=\begin{bmatrix}1&0&1\\0&1&1\\-1&0&0\\0&-1&0\end{bmatrix},
\qquad
W_{\rm down}=\begin{bmatrix}1&0&0&0\\0&1&0&0\\0&0&.5&-.5\end{bmatrix}.
\]

The three up-projection responses are `[g₁−g₃,g₂−g₄,g₁+g₂]=[2,4,3]`. ReLU preserves these positive responses, and the down projection writes `[2,4,1.5,−1.5]`. The third response writes equal-and-opposite changes in two output coordinates. A hidden activation is not itself the final feature update.

For a second row `[-1,0,1,0]`, the up responses are `[-2,0,-1]`; ReLU makes all three zero. The same network can therefore write a substantial update for one position and no update for another. Editing the first row does not change the second row's FFN output **when these are the direct FFN inputs**. If you edit an earlier input before attention, that edit may first change both rows' context.

This pointwise distinction explains a useful connection to a `1×1` convolution over channels: both can apply the same feature transformation at every spatial position without mixing neighboring positions at that step. Stacking such transformations with communication operations gives a different model from either kind alone.

**Inline figure — feature responses and write directions.** Show four input feature bars, three response bars, the activation curve, and four output update bars. Clicking a response shows its explicit dot product and its row of `W_down`; negative output contributions must remain visible. A second token lane reuses the same matrix labels. The learner edits a matrix entry or a feature and predicts which lane can change.

### ReLU, GELU and a multiplicative gate

ReLU is `max(0,z)`: it sets negative responses to zero. GELU is `z Φ(z)`, where `Φ` is the standard normal cumulative distribution function. SiLU, also called Swish with parameter one, is `z σ(z)`, where `σ(z)=1/(1+e⁻ᶻ)`. GELU and SiLU smoothly attenuate negative values; they do not generally turn half of a layer into exact zeros. Their standard scalar forms have no learned parameters. A library may use an approximation to GELU, so specify it when exact comparisons matter.

A **SwiGLU FFN** uses two up projections and an elementwise product:

\[
u=xW_{\rm up},\qquad
g=\operatorname{SiLU}(xW_{\rm gate}),\qquad
\operatorname{FFN}_{\rm SwiGLU}(x)=(u\odot g)W_{\rm down}.
\]

The representation creates both the value-like response and the multiplier. If a hidden coordinate has `u=2` and gate preactivation `1`, its product is `2×.731059≈1.462117`. If the gate preactivation becomes `−1`, the multiplier is `−.268941` and the product is `−.537883`. Unlike a sigmoid gate, the SiLU multiplier is neither restricted to `[0,1]` nor always nonnegative. Calling it a gate is a description of multiplicative control, not a probability interpretation.

The bias-free two-map FFN has `2df` weights. The bias-free SwiGLU FFN has `3df` because it adds an independent up projection. To match the dominant weight count of a conventional FFN with `f=4d`, choose a gated width near `8d/3`: `3d(8d/3)=8d²`. Rounding widths for implementation changes the exact count. For `d=512`, a biased GELU FFN with `f=2048` has 2,099,712 parameters; a bias-free SwiGLU FFN rounded to `f=1408` has 2,162,688. They are close, not identical.

Shazeer's GLU-variant study compared these mechanisms under a declared T5 training setup with reduced gated widths. It is evidence for a useful option, not proof that a ratio or activation is optimal for every task. [GLU Variants Improve Transformer, §§1–3](https://arxiv.org/html/2002.05202v1)

### A useful memory interpretation, with the algebra visible

The sum `Σᵣ φ(x·wᵣ+bᵣ)vᵣ` resembles reading from learned keys and writing their associated values. Here `wᵣ` is a column of the up matrix, and `vᵣ` is a row of the down matrix. These “keys” and “values” are persistent parameters, rather than representations generated from the current neighboring tokens. The coefficients need not add to one; with GELU they can be negative. This is different from the probability-normalized attention mixture.

Researchers have investigated such pattern-reading behavior in trained language-model FFNs. It offers a way to ask which inputs activate a direction and what that direction contributes. It does not require every neuron to mean one clean human concept, or imply that all factual knowledge resides only in FFNs. An interpretable hand-built circuit explains the operation; discovering a trained model's features requires evidence. [Geva et al., §§2–3](https://aclanthology.org/2021.emnlp-main.446.pdf)

## 5. Follow a complete block and implement it

We will first use a small exact fixture so that every stage can be inspected. There are two positions, four features, one attention head, no biases and no dropout. `Q=K=H`, `V=H/4`, and the attention output projection is identity, where `H` is whichever input the wiring supplies to attention. Scaling the values by a quarter keeps the update easy to compare with the bypass. The FFN uses the two matrices above and ReLU. This is a declared mathematical example, not a trained language model.

Start with

\[
X=\begin{bmatrix}1&2&5&8\\3&0&2&1\end{bmatrix}.
\]

For pre-norm, the first row entering attention is approximately `[-1.095444,-.730296,.365148,1.460593]`. The second row is approximately `[1.341635,-1.341635,.447212,-.447212]`. Scores divide their dot products by `√4=2`. After the two-key softmax, mix the rows of `H/4` and add those updates to the **original** `X`. Normalize that new state for the FFN, compute the hidden responses, and add the resulting FFN update to the state.

Follow the second position all the way through. Its attention weights on positions 1 and 2 are approximately `[.076571,.923429]`. The resulting stages are:

| Stage | Second position's vector, approximately |
|---|---|
| Original carried input | `[3,0,2,1]` |
| Attention update | `[.288757,−.323706,.110232,−.075282]` |
| First addition, `Z` | `[3.288757,−.323706,2.110232,.924718]` |
| Normalized FFN input | `[1.330590,−1.356588,.453929,−.427931]` |
| Three ReLU responses | `[.876661,0,0]` |
| FFN update | `[.876661,0,0,0]` |
| Final output | `[4.165418,−.323706,2.110232,.924718]` |

The first hidden response is positive because `1.330590−.453929≈.876661`; its write direction adds only to feature 1. At the first position all three FFN responses happen to be zero, so its final output equals its contextual state. Repeating the computation with the same parameters under post-norm gives second-position output approximately `[.695474,−1.720968,.632590,.392905]`. The changed result comes from the wiring, including which vectors determine attention scores, not from learning new parameters between the two calculations.

**Investigation — build the block.** Select pre-norm or post-norm, edit a real input feature or either FFN matrix, and predict a chosen output feature's direction of change. Commit before revealing the calculation. The trace exposes input, normalized branch view, attention update, first addition, FFN responses, FFN update and final output. Compare the same parameter values under the other wiring. Set branch scale to zero as a separate null experiment: pre-norm must return the input, while post-norm still normalizes it. An independent FFN-only view lets you check that editing a different position cannot affect an unchanged direct FFN input.

The figure is useful because it displays actual vectors at named junctions. A heatmap of vaguely labeled “activation strength” would hide the difference between a normalized branch input, an update and the carried state. We can compute RMS or L2 summaries too, but their labels must say which tensor and which reduction they summarize.

### A complete PyTorch block

The following program uses standard multi-head attention from the previous lesson and implements both wirings directly. Install PyTorch in your own Python environment if needed with `python -m pip install torch`. Save the program as `transformer_block.py` and run `python transformer_block.py`. It uses CPU float64 for a small comparison. All dropout probabilities are zero so that randomness does not obscure the wiring.

```python
import torch
from torch import nn

torch.set_num_threads(1)

class TransformerBlock(nn.Module):
    def __init__(self, width, heads, hidden, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0, batch_first=True)
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, inputs, blocked=None):
        h = self.norm_attention(inputs) if self.pre_norm else inputs
        update, _ = self.attention(
            h, h, h, attn_mask=blocked, need_weights=False)
        context = inputs + update
        if not self.pre_norm:
            context = self.norm_attention(context)
        g = self.norm_feedforward(context) if self.pre_norm else context
        output = context + self.feedforward(g)
        return output if self.pre_norm else self.norm_feedforward(output)

torch.manual_seed(31)
block = TransformerBlock(8, 2, 16).double()
inputs = torch.randn(2, 4, 8, dtype=torch.float64)
blocked = torch.ones(4, 4, dtype=torch.bool).triu(1)

for pre_norm in (True, False):
    block.pre_norm = pre_norm
    reference = nn.TransformerEncoderLayer(
        8, 2, dim_feedforward=16, dropout=0, activation="gelu",
        batch_first=True, norm_first=pre_norm).double()
    reference.self_attn.load_state_dict(block.attention.state_dict())
    reference.norm1.load_state_dict(block.norm_attention.state_dict())
    reference.norm2.load_state_dict(block.norm_feedforward.state_dict())
    reference.linear1.load_state_dict(block.feedforward[0].state_dict())
    reference.linear2.load_state_dict(block.feedforward[2].state_dict())
    actual = block(inputs, blocked)
    expected = reference(inputs, src_mask=blocked)
    print(pre_norm, tuple(actual.shape),
          torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
```

The expected printed lines are `True (2, 4, 8) True` and `False (2, 4, 8) True`. The same-parameter comparisons were executed with PyTorch 2.14.0 CPU; the packet records the actual maximum differences. We are checking the complete computation against another implementation, not expecting separately initialized blocks to agree.

Notice three small decisions. We calculate `Norm(X)` once and reuse it as the input to the three attention projections. The FFN receives the updated contextual state. For this `MultiheadAttention` API, boolean **True means blocked**; do not transfer that convention to an API where True means allowed. Padding masks are additionally needed for variable-length batches, and padded query outputs must be excluded from the task's loss or pooling.

### Defaults and training behavior are part of the model

In the installed reference API, `TransformerEncoderLayer` defaults to post-norm, ReLU, `dim_feedforward=2048`, dropout `.1` and sequence-first tensors. The FFN default does **not** automatically become four times your chosen model width. The explicit arguments above make the intended architecture reviewable. `TransformerEncoder` constructs separate copies of a supplied layer, initially with equal parameter values; those parameters are not shared storage. If independent initial values are intended, initialize the copies appropriately or construct separate blocks with a `ModuleList`. [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html), [TransformerEncoder](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html)

When dropout is part of the intended design, specify its locations. Attention-probability dropout, FFN hidden-activation dropout, and dropout on a branch before residual addition are different operations. In a common pre-norm pattern, for example, `Z=X+Drop(Attn(Norm(X)))`; the bypass itself is retained. The reference encoder layer has additional FFN/residual dropout sites beyond the probability dropout inside its attention module. Matching only the attention constructor's probability does not make two training implementations equivalent. `eval()` disables ordinary module dropout; `no_grad()` alone does not switch it off.

A pre-norm stack commonly applies a final normalizer before the output head because its carried stream can change scale across blocks. That final normalizer is part of the architecture; it is not supplied automatically by a custom block. Keeping shapes identical while silently moving a normalizer, replacing an activation or deleting a bias changes the represented function. Loading a pretrained checkpoint requires matching its exact architecture and conventions before considering optimization.

## 6. A complete model on real ordered movements

Can a stack turn sampled hand positions into useful movement classes? We use the same openly licensed **Libras Movement** source introduced in self-attention: 360 recorded trajectories, 45 normalized two-dimensional points each, and 15 movement types. The task is to classify those movement types. These centroid traces are not full signs, body-pose recordings or language translations. [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement)

### Give the block an explicit time tag

The previous pooled self-attention model had no way to distinguish permutations of the same point collection. A positionwise FFN and feature normalization do not by themselves remove that symmetry: both apply the same rule at every row. Here each sampled point receives a simple third coordinate,

\[
s_t=2t/44-1,\quad t=0,\ldots,44,
\qquad h_t=[2x_t-1,\,2y_t-1,\,s_t]W_{\rm stem}+b_{\rm stem}.
\]

The first and last time tags are `−1` and `1`. They describe position within the sampled trajectory, not seconds measured by a clock. A linear map sends these three numbers into 24 features. The model can now respond differently to the same location early and late in a movement. This is a deliberately simple position representation; the next lesson explores better choices for different sequence tasks.

The complete model is:

\[
\text{tagged points}
\rightarrow \text{Linear}(3,24)
\rightarrow \text{Block}_1
\rightarrow \text{Block}_2
\rightarrow \operatorname{LN}(24)
\rightarrow \text{mean over valid positions}
\rightarrow \text{Linear}(24,15).
\]

Each block has two heads and a GELU FFN of width 48. The last linear layer returns logits; softmax is used to inspect class probabilities, while the training cross-entropy function accepts logits directly. Attention is bidirectional because the whole recorded movement is available for classification. That would be the wrong information boundary for predicting a future point before observing it.

### The comparison is specified before looking at its results

There are 330 unique coordinate sequences: 30 rows repeat other trajectories exactly, with consistent labels. Keep the first occurrence of each unique trajectory, then use the fixed class-stratified seed 73 split of 220 training,50 validation and 60 test records. All 45 points from one trajectory stay together; every test class has four examples. The exact row IDs and duplicate groups are retained in the [results](author-results.json). The source describes four performers and two collection sessions, but does not attach usable performer/session IDs to each row. This study therefore cannot establish generalization to an unseen signer or collection session.

Compare pre-norm and post-norm placement with otherwise identical model definitions. For this controlled comparison, **both** models have the final LayerNorm before pooling, including the post-norm model. This keeps their 10,263 parameters and readout convention matched; it is not a reproduction of the original translation architecture. Each seed gives the same initial parameter tensors to the two wirings. Both use no dropout, no weight decay, Adam at `.003`, and 180 full-training-batch epochs. Three declared seeds are 101,102,103.

For each fit, select the checkpoint with highest validation macro F1, then lower validation cross entropy, then the earliest exact tie. Macro F1 calculates a precision/recall balance per class and averages the 15 class scores. Evaluate the 60 test records after selection. Keeping the same data boundary as the previous lesson supports continuity, but the new task model changes several architectural ingredients; comparing its score with the earlier model does not isolate the benefit of any one added component.

### What actually happened

The complete six-fit CPU experiment produced:

| Placement | Seed | Selected epoch | Training correct /220 | Validation correct /50 | Test correct /60 | Test macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Pre-norm |101|83|216|43|49|.8070|
| Pre-norm |102|146|219|41|49|.8050|
| Pre-norm |103|112|218|43|51|.8435|
| Post-norm |101|125|218|45|52|.8668|
| Post-norm |102|170|220|39|51|.8520|
| Post-norm |103|129|220|42|47|.7814|

Both placements learn this small task. Post-norm has more test-correct examples for seeds 101 and 102; pre-norm has more for 103. The outcome does not support declaring pre-norm a universal winner or post-norm unable to train. These six fits vary initialization within one fixed small data split; they are not independent samples of all possible datasets or a hardware benchmark. They also show a train-to-test gap despite near-perfect training accuracy.

**Inline figure — observed learning, not a preferred winner.** Separate training and validation cross-entropy traces, selected-epoch markers, and a paired dot plot of test macro F1 by seed. Keep all six outcomes. The primary comparison uses the validation-selected checkpoints, not whichever epochs have the most attractive test scores. The retained experiment does not contain per-epoch test curves because the test set was not used that way.

### Inspect a failure before designing a repair

Choose source row 77, the first test example of class 4, before choosing any favorable prediction. Its source class is **anticlockwise arc**. Under the seed 101 checkpoints, the pre-norm model predicts class 7 and assigns class 4 probability about `.000211`; the post-norm model predicts class 3 and assigns class 4 probability about `.200043`. Both are wrong. The trace remains useful precisely because we can inspect an actual error instead of showing only impressive predictions.

**Investigation — what did the model use?** View the recorded trajectory with start/end markers and time tags. Edit an actual point or its time tag, record a prediction about the class 4 probability or a named logit, and run the fixed model. Inspect the first and second block's branch inputs, updates and carried states for the chosen position. A difference between two stages is a computation you can trace; it is not a guarantee that one hidden coordinate has an obvious human meaning.

Three contrasts make the mechanism concrete:

1. **Reverse the coordinates while keeping time slots fixed.** The sequence now visits the observed locations in the opposite order. The largest logit change is about 4.885 for pre-norm and 5.746 for post-norm. Both predict class 5, clockwise arc. This is evidence that these models use the relationship between locations and time tags. It is not a new labeled test case proving accuracy on every reversed gesture.
2. **Reverse coordinates and their time tags together.** This just permutes the complete tagged records. Each block is permutation-equivariant and the mean is invariant, so the pooled classifier should give the same result. The observed maximum logit changes are below `1.5×10⁻⁶`, consistent with floating-point reduction order. Time information lives in the tags; arbitrary storage-row ordering is not an additional secret input.
3. **Append five padding records.** Preserve the original 45 tags, assign zero tag to the five pads, block padded keys in **both** layers, and exclude padded rows from pooling. The output agrees within `10⁻⁶`. If the pads are treated as real observations instead, the largest logit changes are about 1.923 and 2.424. A mask only in the first block would not be enough.

Reflecting only frame 23's x coordinate, `x→1−x`, gives a further genuine editable-input contrast. The seed 101 pre-norm prediction becomes class 2 and the post-norm prediction becomes class 11. The point editor should compute arbitrary new changes from actual saved weights, not look up one of these presets. Keep the original class attached to the original observed record; an edited path has no newly measured ground-truth label.

### Reproduce the study

Download [the complete program](author-calculations.py), [the original data](movement_libras.data) and [source metadata](movement_libras.names) into one directory. Install `numpy torch scikit-learn` in your own environment, then run `python author-calculations.py`. The program contains the full block, time-tagged classifier, deduplication/split, training loop, checkpoint selection, fixtures and interventions. It writes [all measured results](author-results.json) and [the two actual seed 101 models](block-models.json). The [provenance record](data-provenance.md) supplies attribution, exact hashes and the observed runtime versions. No network or GPU is needed after obtaining those files.

This gives you a full system to inspect: real inputs, a defined representation, a differentiable model, a loss, a selection rule and a held-out result. It also leaves a useful research question: would a representation designed around movement direction or a different validation protocol help? That question calls for a new declared experiment, not a retroactive reinterpretation of this one.

## 7. Deeper: what a gradient plot can and cannot establish

A gradient is sensitivity of a **specified output quantity** to a specified input or parameter. It is not an intrinsic score attached to a layer. Before plotting gradients, write down the loss, reduction, tensor being differentiated, initialization and measurement norm.

### A convincing-looking diagnostic can be identically uninformative

Take LayerNorm with all gains one and all offsets zero. Its output coordinates sum to zero:

\[
\sum_i\operatorname{LN}(x)_i
=\frac{\sum_i(x_i-\mu)}{\sqrt{v+\epsilon}}=0.
\]

The gradient of this sum with respect to `x` is therefore zero. If you place `loss=output.sum()` after the final default-affine LayerNorm of a post-norm stack, tiny upstream gradients may simply reveal that you asked the network to change a constant. **One normalizer is already enough.** The learned final gains or offsets can still receive gradients; that does not make the upstream diagnostic informative.

Use a nonconstant contrast, such as the first normalized coordinate minus the second, to see the distinction. This standalone program was executed:

```python
import torch
from torch.nn import functional as F

x = torch.tensor([1., 2., 5., 8.], dtype=torch.float64,
                 requires_grad=True)
y = F.layer_norm(x, (4,), eps=1e-5)
sum_gradient = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
contrast_gradient = torch.autograd.grad(y[0] - y[1], x)[0]
print(sum_gradient)
print(contrast_gradient)
```

The first gradient is `[0,0,0,0]`. The second is approximately `[.328633,−.389491,.012172,.048686]`. The normalizer is differentiable and sensitive to the contrast even though it cannot change the centered sum. If gains are unequal or a nonlinear readout follows the normalizer, the sum need not remain constant; inspect the actual function rather than applying this particular null indiscriminately.

### Derive the normalization Jacobian

This extends the derivative rule in the normalization lesson. Let `c=x−μ1`, `s=√(cᵀc/d+ε)`, and initially take identity affine parameters. Differentiating the centering and scaling steps gives

\[
J_{\rm LN}(x)
=\frac1s\left(I-\frac{11^\top}{d}-\frac{cc^\top}{d s^2}\right).
\]

The first subtraction removes the common-offset direction: `J1=0`. For a vector perpendicular to both `1` and `c`, the Jacobian multiplies it by `1/s`. For the centered radial direction `c`, it multiplies by `ε/s³`. When epsilon is zero and `c≠0`, scaling this vector does not change its normalized direction, so that radial derivative is also zero.

The crucial point is that `1/s` can be greater than one when the input feature spread is small. A LayerNorm Jacobian is **not necessarily a contraction**. With learned gains, left-multiply the expression by `diag(γ)`, introducing further scale and direction dependence. These equations concern one position's feature vector, not a scalar “gradient retention percentage” shared by every block.

For a pre-norm residual sublayer `y=x+F(Norm(x))`, the local derivative is

\[
J_{\rm pre}=I+J_F(Norm(x))J_{\rm Norm}(x).
\]

For a post-norm sublayer `y=Norm(x+F(x))`, it is

\[
J_{\rm post}=J_{\rm Norm}(x+F(x))(I+J_F(x)).
\]

Pre-norm exposes an explicit identity contribution outside normalization. Other terms can reinforce or cancel it. Through a stack, these Jacobians multiply in order. An identity term in every factor is useful architecture, but does not prove that the product's every direction stays well-conditioned at arbitrary depth.

### A declared initialization measurement

The packet computes stacks of 1,4 and 12 blocks at width 8, two heads and FFN width 16. Both placements use the same initial tensors at each depth. A separately seeded random input has shape `(1,3,8)`. After a shared affine-free final LayerNorm, the output is dotted with a fixed random probe of unit L2 length. Autograd measures the L2 norm of the gradient with respect to each **carried state**. This is a vector-Jacobian product for that probe, not the full Jacobian's largest singular value and not an average of incompatible parameter tensors.

| Blocks | Pre-norm input-gradient L2 | Post-norm input-gradient L2 | Pre-norm final carried-state RMS | Post-norm final carried-state RMS |
|---:|---:|---:|---:|---:|
|1|.868298|.828193|1.023500|.999995|
|4|.980193|.851472|1.100291|.999996|
|12|.927944|.893780|1.653763|.999995|

Neither placement loses this probe's sensitivity in the measured depths. Replacing the probe with an all-ones output sum makes both input gradients effectively zero after their shared final normalizer. That contrast explains why specifying the output quantity is essential.

**Investigation — choose a meaningful probe.** Edit the four-feature vector and the output-probe weights. Predict whether the input gradient is exactly zero, then reveal the analytic gradient and a finite-difference comparison. Construct one uninformative probe and one informative probe without changing the normalizer. A separate recorded depth view shows all intermediate-state norms from the actual CPU calculation; it should not extrapolate unmeasured depths or call a larger gradient automatically better.

Xiong et al. analyze initialization with a simplified mean-field setting, including single-head attention, zero-initialized query/key matrices giving uniform weights, Gaussian input assumptions, and a loss on a prediction readout. Their analysis and experiments explain why normalization placement can affect initial gradient scale and the usefulness of learning-rate warmup. In particular, the post-norm concern includes large gradients near the output, not just a slogan about vanishing lower-layer gradients. Their result does not make every pre-norm training setup safe without warmup. [Xiong et al., §§3–4](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf)

### The residual stream's magnitude needs its own explanation

Across a population of examples, for one coordinate of a carried state `x` and update `u`,

\[
\operatorname{Var}(x+u)
=\operatorname{Var}(x)+\operatorname{Var}(u)+2\operatorname{Cov}(x,u).
\]

If independent zero-mean updates of fixed variance accumulate, variance grows linearly with the number of updates. If every update equals the same random vector `z`, then after `k` additions the contribution is `kz`, whose variance is `k² Var(z)`. If an update cancels the state, its variance can shrink instead. Normalizing branch inputs does not erase these correlations or fix the carried stream's variance by itself.

A final norm, scaled residual branches, depth-aware initialization and alternative norm placement are different tools for controlling a model. Their effects should be measured with a real objective and the intended training recipe. Do not confuse RMS across feature coordinates in one activation with variance across a population; the labels in the table above deliberately use the former.

## 8. Deeper: assemble task models and recognize real variants

### Three familiar attention layouts

| Task layout | Self-attention boundary | Extra block component | Typical output rule |
|---|---|---|---|
| Encoder for a complete observation | All valid input positions | None required | Per-position output or pooling/classification |
| Autoregressive decoder-only model | Present and past positions | None required | Next-token logits for each position |
| Encoder–decoder model | Encoder sees valid source; decoder self-attention is causal | Cross-attention from decoder to encoder | Target-token logits conditioned on source |

For next-token training, a causal mask is only half the specification. Shift inputs and targets. With sequence `[BOS, a, b, c]`, input positions `[BOS,a,b]` predict targets `[a,b,c]`. The diagonal of causal attention is legal because the current **input** token is not the next target. If the current input already contains the answer being scored, masking future tokens does not repair the leakage.

At generation time, produce a distribution, choose a next token, append it and repeat. A cache can retain each layer's keys and values from earlier steps; cached values must correspond to the same block inputs, positional convention and causal computation as a full-prefix run. Previous outputs stay valid because their legal information did not include the newly appended future token. Cache arithmetic and sharing get their own lessons after positional encoding.

**Inline figure — one block interface, three information boundaries.** Show source positions, target positions and legal arrows separately. Mark which table supplies each sublayer's queries, keys and values. Keep padding exclusions independent of the causal triangle. A final panel shows the shift between input and target rows rather than placing identical tokens under a misleading mask.

### A copy task teaches why the loss denominator matters

Suppose an autoregressive exercise uses

`[BOS, a₁,…,a₈, SEP, a₁,…,a₈, EOS]`,

where the eight source symbols are drawn independently and uniformly from ten choices for every new example. There are 19 tokens and 18 next-token prediction targets. The first eight source-symbol targets are fresh randomness given the prefix: no causal architecture can predict them better than the uniform distribution in expectation. Their best expected contribution is eight times `ln10`. The repeated symbols and delimiters are predictable in principle. Thus a model that performs the repeat perfectly can still have mean next-token loss approaching

\[
\frac8{18}\ln10\approx1.023371\ \text{nats per target}.
\]

Report accuracy on the eight repeated-symbol targets separately from the total loss. `SEP`, at input index 9 under zero-based indexing, predicts the first repeated symbol at target index 9. Changing the source length, symbol distribution or set of scored targets changes the loss floor. A finite memorized training set can also behave differently from fresh independent examples. This is a useful debugging task, not proof that only Transformers can copy or that any named small model has already achieved perfect copying.

### Modern designs change more than one switch

The precise computation is more informative than grouping every model into one recommended recipe:

| Design | Example update rule for one sublayer or block | What to notice |
|---|---|---|
| Sequential pre-norm | `z=x+A(N₁(x)); y=z+F(N₂(z))` | FFN reads the attention-updated state |
| Parallel attention and FFN | `y=x+A(N(x))+F(N(x))` | Both branches read the same incoming state; the FFN does not consume this block's attention output |
| Residual-post normalization | `y=x+N(F(x))` | Normalize the branch output before adding it to an unnormalized bypass |
| Norms before and after a branch | `y=x+N_out(F(N_in(x)))` | Controls both branch input and branch update |
| Scaled post-norm residual | `y=N(αx+F(x))` with a matching initialization | Changes bypass scale and initialization together |

PaLM uses parallel attention/MLP branches. Swin V2 uses residual-post normalization; its name must not be confused with normalization **after the whole addition**. Gemma 2 normalizes sublayer inputs and outputs with RMSNorm. DeepNorm uses a scaled residual together with architecture-dependent initialization. These are real alternatives, so `x+Norm(F(x))` is not automatically a bug. Their cited studies have their own tasks and ablations; changing a line in a pretrained model does not reproduce them. [PaLM, §2](https://arxiv.org/html/2204.02311v5), [Swin V2, §3.2](https://arxiv.org/html/2111.09883v2), [Gemma 2, §2](https://arxiv.org/html/2408.00118v3), [DeepNet, §§2,4.1](https://arxiv.org/pdf/2203.00555)

A practical architecture description should therefore state the norm's formula and placement, FFN type and actual hidden width, serial versus parallel branches, attention head layout, positional convention, masks, biases, dropout sites and final readout. “Llama-like” or “modern Transformer” alone does not specify a compatible checkpoint. RMSNorm and SwiGLU are useful examples, but neither is universal across contemporary models.

### A compact RMSNorm/SwiGLU branch you can read

This standalone program implements the two components explicitly and checks their shapes. It uses float32; a mixed-precision implementation should consider accumulation precision in its reductions. It is a component example, not a claim to reproduce a named model's complete attention or positional convention.

```python
import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True)
                                 + self.eps)
        return self.gain * x * inverse_rms

class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.up = nn.Linear(width, hidden, bias=False)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(self.up(x) * F.silu(self.gate(x)))

torch.manual_seed(41)
inputs = torch.randn(2, 3, 12)
norm = RMSNorm(12)
ffn = SwiGLU(12, 32)
output = inputs + ffn(norm(inputs))
print(tuple(output.shape))
print(sum(p.numel() for p in ffn.parameters()))
```

It prints `(2,3,12)` and `1152`, since the three bias-free matrices contain `3×12×32` weights. Insert this component only where the intended architecture calls for it. The earlier API-comparison program uses a GELU FFN and LayerNorm, so it would no longer be comparing the same function after this replacement.

## 9. Deeper: count the work, then measure the implementation

### Parameters belong to maps, not sequence positions

For standard attention with total head width equal to `d`, its query/key/value/output weights contain `4d²` scalars. With biases they add `4d`. A two-map FFN adds `2df+f+d`. Two ordinary affine LayerNorms add `4d`. One block therefore has

\[
P_{\rm block}=4d^2+2df+f+9d.
\]

At `f=4d`, this becomes `12d²+13d`. For `d=256`, it is 789,760 parameters. The exact source program checks its chosen model's count rather than assuming every block has this formula: grouped/latent heads, gated FFNs, different biases or extra normalizers change it. Block count multiplies the count when blocks have independent parameters. Token/position embeddings and the output head must be added separately; tying input/output embeddings changes the total.

Increasing sequence length creates more activations and work, but does not create new parameters in these shared maps. This is why a parameter count alone does not tell you whether a long-context run will fit in memory.

### Distinguish MACs, FLOPs and elapsed time

A multiply–accumulate, or MAC, multiplies two numbers and accumulates the result. Under the convention of two FLOPs per MAC, the dominant dense work for one forward block on `(B,L,d)` is:

| Part | MACs |
|---|---:|
| Q/K/V and output projections |`4BLd²`|
| Two FFN projections |`2BLdf`|
| Score matrix and value mixture |`2BL²d`|

At `f=4d`, the approximate forward FLOP count is `24BLd²+4BL²d`. Divide by `BL` for a per-token count: `24d²+4Ld`. These expressions omit softmax, normalization, activations, bias additions and masking overhead. A conventional backward pass through dense maps often adds roughly twice their forward work; that is an approximation for planning, not an exact universal three-times total.

The quadratic term becomes important as `L` grows. For `d=512,f=2048,B=1`, projections plus FFN take about 1.611 billion MACs at `L=512`; the attention matrix products add about .268 billion. At `L=4096`, those terms become 12.885 billion and 17.180 billion. The source of the growth matters more than giving either number the unqualified label “Transformer cost.”

**Inline figure — where the count grows.** Plot these exact derived MAC components against `L` with units and a legend separating shared linear maps from pairwise interactions. Include a parameter-count panel that stays flat as `L` changes. It must be labeled derived arithmetic, not measured latency, and must not draw invented lines for competing software libraries.

### Activation memory and inference cache are different objects

A materialized attention matrix has `BH L²` entries for `H` heads. A training implementation may retain intermediate activations for backward computation, including FFN hidden tensors of shape `(B,L,f)`, subject to its particular schedule and recomputation choices. The autoregressive key/value cache instead stores prior keys and values at every layer. With ordinary equal-width K/V heads, its per-layer element count is `2BLd`. Neither is the same as parameter storage or optimizer-state storage.

Exact tiled attention can avoid materializing the whole score/probability matrix while computing the same mathematical softmax-attention operation. Different reduction orders need not give bit-identical floating-point outputs. It changes memory traffic and the execution schedule, not the fact that every permitted query/key interaction is part of dense attention. [FlashAttention, §§2–3](https://arxiv.org/pdf/2205.14135)

**Activation checkpointing** stores selected boundary activations and recomputes omitted intermediates during backward. It trades additional computation for less saved activation memory. Recomputed stochastic operations must preserve the intended randomness and side effects; otherwise the backward path may not correspond to the forward computation. It does not automatically halve every model's memory or double its elapsed time. Prefer measurements for the exact checkpoint policy and workload.

Fusing a sequence of kernels can reduce launches and intermediate memory transfers. Lower-precision kernels change numerical and hardware conditions. A shorter formula is not a guaranteed speedup: a gated FFN's extra elementwise product, matrix shapes, alignment, fusion and memory traffic all matter. Measure training and inference separately, report batch/length/dtype/hardware, and keep mathematical counts separate from wall-clock results.

### How a block can be distributed

There are two useful levels of partitioning. **Pipeline parallelism** places groups of blocks on different devices and passes activations between them. **Tensor parallelism** splits a large map within a block. In our row-vector convention, split `W_up` by its output columns, so each device computes a subset of hidden features; split the corresponding rows of `W_down`, so each device forms a partial output. Summing those partial outputs recovers the full down projection. That sum requires communication. A gated FFN must keep matching gate and up coordinates together.

Partitioning positions is another choice, with attention communication across position partitions. The later Ring Attention/Sequence Parallelism lesson develops that mechanism. None of these mathematical partitions specifies a universal device count or speed; workload, communication, scheduling and available memory determine the useful implementation.

## 10. Practice: change the problem before checking the answer

### 1. Follow a new four-feature vector

Use `x=[0,2,4,6]`, identity affine parameters and epsilon `10⁻⁵`. Calculate LayerNorm and RMSNorm approximately. Then add three to all entries. Which output remains unchanged? Explain why an output's RMS near one does not make its L2 norm one.

<details><summary>Hint</summary>

The centered vector is `[-3,-1,1,3]`. Compute the mean squared deviation and the mean squared raw value separately.

</details>
<details><summary>Solution</summary>

The mean is 3 and variance is 5. LayerNorm is approximately `[-1.341639,-.447213,.447213,1.341639]`. The raw mean square is 14, so RMSNorm is approximately `[0,.534522,1.069045,1.603567]`. After the offset, LayerNorm is unchanged. RMSNorm uses `[3,5,7,9]` and mean square 41, giving approximately `[.468521,.780869,1.093216,1.405564]`. For four coordinates, `L2=√4×RMS=2×RMS`; an RMS of one corresponds to L2 two. Affine parameters and epsilon affect the exact values as described in §3.

</details>

### 2. Build a new FFN response

Use the matrices from §4, ReLU and no biases, but input `[2,-1,0,1]`. Find the three preactivations, hidden responses and four-feature update. Change only the third row of `W_down` from `[0,0,.5,-.5]` to `[0,0,1,0]`. Which output coordinates change? Would editing a different token's direct FFN input change this update?

<details><summary>Hint</summary>

The three questions remain `x₁−x₃`, `x₂−x₄` and `x₁+x₂`. The third hidden response multiplies the third write direction.

</details>
<details><summary>Solution</summary>

The preactivations are `[2,-2,1]`; ReLU gives `[2,0,1]`. The original update is `[2,0,.5,-.5]`. Changing the third write direction gives `[2,0,1,0]`, so coordinates 3 and 4 each increase by .5. The first two coordinates do not change. A different direct FFN input row cannot affect this row because the function is shared but positionwise. An earlier attention edit is a different intervention because it can change this row's contextual input.

</details>

### 3. Diagnose a result that cannot answer the question

A colleague compares two 20-block networks. The post-norm network ends in `LayerNorm(16)` with all gains one and all offsets zero, and they differentiate `output.sum()` with respect to the input. They obtain nearly zero and conclude the network cannot train. Identify the exact problem, propose a meaningful replacement diagnostic, and name one limitation of that replacement.

<details><summary>Hint</summary>

What is the sum across the features of each final normalized row? Does the number of preceding blocks matter?

</details>
<details><summary>Solution</summary>

The final centered features sum to zero, so the chosen scalar is constant with respect to upstream inputs. The same result occurs with a single default-affine LayerNorm. A fixed nonconstant projection of the normalized output, or a classification cross-entropy through a declared readout and targets, can measure a useful sensitivity. Match the inputs, initial tensors, readout convention and reduction when comparing wirings. One projection measures one vector-Jacobian product; it is neither the whole singular-value spectrum nor proof of optimization/generalization after training. Final affine parameter gradients also need to be distinguished from upstream input gradients.

</details>

### 4. Preserve a tagged movement while changing its storage

A classifier uses point/time records `(x,y,s)`, shared pointwise stems, bidirectional attention, per-position normalization/FFNs and mean pooling. You sort all records by x coordinate, carrying each record's original time tag with it. Should the prediction change in exact arithmetic? What if you replace the carried time tags with newly increasing tags after sorting? What must a padding fix do across two blocks and pooling?

<details><summary>Hint</summary>

Separate a permutation of complete records from changing the relationship between location and time. Follow padding after the first block as well as before it.

</details>
<details><summary>Solution</summary>

Sorting complete tagged records only permutes the inputs. The shared stem, norms and FFNs are equivariant, and attention without an additional order-specific mask or bias is equivariant. Mean pooling removes the permutation, so logits are unchanged in exact arithmetic. Reassigning new time tags changes the data; there is no such guarantee. Padded keys must be blocked at both attention layers, and padded query rows must be excluded from mean pooling. Preserve the original valid records' time tags instead of recomputing their spacing over the padded length. A tiny floating-point change under a legal permutation is different from a large model change caused by incorrect tagging or masking.

</details>

### 5. Count a different block

A biased standard-attention block has `d=96`, FFN width `f=240`, and two affine LayerNorms. Count its parameters. Then replace only the FFN with a bias-free SwiGLU FFN having the same dominant FFN weight count; find the gated width and the resulting complete block count.

<details><summary>Hint</summary>

Keep the attention weights/biases and both normalizers. Match `2df` to `3dg` before deciding what happens to the removed FFN biases.

</details>
<details><summary>Solution</summary>

Attention contains `4×96²+4×96=37,248` parameters. The original FFN has `2×96×240+240+96=46,416`. Two LayerNorms have `4×96=384`. Total 84,048. The gated width is `g=2×240/3=160`; its bias-free FFN has 46,080 weights. The new total is 83,712, which is 336 lower because the original FFN's240+96 biases are gone. Matching leading matrix counts is not the same as matching every scalar parameter.

</details>

### 6. A different copying sequence

An exercise uses `[BOS, six independent uniform symbols from eight choices, SEP, the same six symbols, EOS]`. It trains on fresh samples and scores all next-token targets. Calculate the best possible expected average loss contribution from the unpredictable part. How does that contribution change if only repeated-symbol targets are scored?

<details><summary>Hint</summary>

There are 15 tokens and 14 targets. Count the six first-occurrence predictions, then use the entropy of a uniform eight-way choice.

</details>
<details><summary>Solution</summary>

The unavoidable contribution is `(6/14)ln8≈.891189` nats per scored target. A model can in principle make the predictable targets' loss arbitrarily small, approaching that floor overall. If the loss scores only the six repeated-symbol targets, they are predictable from the observed source and this particular entropy floor becomes zero. That does not guarantee a finite trained model reaches it; it changes what the objective is measuring.

</details>

### 7. Can a normalized branch still accumulate a large stream?

Across examples, let `z` have mean zero and variance one. Start from zero and add exactly the same update `z` eight times. Compare the final variance with adding eight independent copies of `z`. Explain which assumption is required for saying accumulated variance grows linearly.

<details><summary>Hint</summary>

One state is `8z`. The other is a sum whose cross-covariances vanish.

</details>
<details><summary>Solution</summary>

The repeated update gives variance 64. The independent zero-mean sum gives variance 8. Independence is sufficient; zero cross-covariances with controlled per-update variance are enough for the variance addition itself. Normalizing each branch input does not establish those covariance assumptions for learned updates. The result concerns variance across examples, not the across-feature RMS of one vector.

</details>

### 8. Explain a parallel branch to someone reading code

Compare `z=x+A(N(x)); y=z+F(N(z))` with `y=x+A(N(x))+F(N(x))`. Set `A(v)=a`, a nonzero constant update, and let `F` be nonlinear. Is the second an algebraic rearrangement of the first? Explain what information reaches the FFN and why a pretrained checkpoint cannot generally be switched without changing outputs.

<details><summary>Hint</summary>

Look at the FFN's argument, not only the three terms visible near the final addition.

</details>
<details><summary>Solution</summary>

The sequential version returns `x+a+F(N(x+a))`; the parallel version returns `x+a+F(N(x))`. These are generally different because normalization and `F` can respond to the changed input. The sequential FFN consumes this block's attention update; the parallel FFN consumes the original normalized state. They agree under particular nulls, such as zero attention update, but not as a general identity. A switch changes the function even if parameter dimensions and names are compatible.

</details>

## Where this leads, and another way to learn

You can now trace how a block communicates across positions, processes features locally, preserves or normalizes its carried state, and becomes a task model. The next lesson in the module is [Positional Encodings: Sinusoidal, Learned, RoPE and ALiBi](/learn/path/full-curriculum/positional-encodings-sinusoidal-learned-rope-alibi?module=deep-learning-fundamentals). It asks how order and distance should enter the representation or attention calculation; our simple movement time tag is a starting point, not the end of that design problem.

For a different presentation or deeper source reading:

- [D2L 1.0.3, Transformer Architecture](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) connects a post-norm block to an encoder–decoder translation model. Its FFN, add/norm, encoder/decoder and training sections are useful next reading if you want the larger sequence-to-sequence assembly. The examples use the book's helper framework and a different training task; their scores are not comparisons with our movement study.
- [3Blue1Brown, How might LLMs store facts?](https://www.3blue1brown.com/lessons/mlp/) offers an accompanying video and illustrated creator notes on FFN feature detection and update directions. Read the assumptions around its deliberately simplified feature circuit. The current notes also correct a near-orthogonality demonstration from the video; use the corrected notes, and do not treat the numerical capacity illustration as a measured fact about a particular language model. The matrix interpretation is the useful bridge here.
- [Attention Is All You Need, §3](https://arxiv.org/pdf/1706.03762) defines the original post-norm encoder–decoder model and its ReLU FFN. Read it to distinguish historical choices from the family of later variants.
- [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf) is the deeper source for placement, initialization gradients and warmup. Its stated theoretical assumptions are essential to interpreting the conclusions.
- [Layer Normalization](https://arxiv.org/pdf/1607.06450) and [RMSNorm](https://arxiv.org/pdf/1910.07467) explain the distinct normalization mechanisms. The latter's efficiency measurements belong to its evaluated implementations, not a universal current speed ratio.
- [GLU Variants Improve Transformer](https://arxiv.org/html/2002.05202v1) provides the gated FFN equations and parameter-matched experimental setup. [Transformer Feed-Forward Layers Are Key-Value Memories](https://aclanthology.org/2021.emnlp-main.446.pdf) develops the persistent-memory interpretation and investigates trained examples.
- [PyTorch's reference encoder layer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html) is the API matched in §5. The complete topic program and retained inputs are linked in §6 for an offline hands-on route.
