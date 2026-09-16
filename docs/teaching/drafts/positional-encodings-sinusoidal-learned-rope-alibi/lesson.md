# Positional Encodings: Sinusoidal, Learned, RoPE and ALiBi

Imagine recording a hand moving around an arc. The same collection of points can describe clockwise or anticlockwise movement. To tell them apart, a model needs to know how the points are ordered—not just where the hand visited.

The [previous Transformer Block lesson](/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals) supplied that information by attaching a numerical time-slot tag to every point. This lesson studies more structured ways to supply position. Some add a position vector to the input. Some turn query and key vectors before comparing them. Some change the attention score according to distance. These choices affect what the model can represent, how cached generation works and what happens when a sequence becomes longer.

**First pass:** follow §§1–6, run the small program in §7 and try exercises 1–4. You will be able to explain and implement the four main methods and diagnose a position-offset bug. The deeper route in §§8–9 develops context extension, geometric variants and engineering choices; exercises 5–8 use that material. Derivations sit beside the mechanism they explain, so you can return to them after trying the visual investigation.

## 1. What information is missing?

### Rows store order; the computation must use it

An array preserves its row order. That does not mean every function applied to the array uses that order. Adding all the rows gives the same sum after any rearrangement.

Recall one head from [Self-Attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals). A query describes what a row seeks, a key describes how a row can be matched, and a value is the information mixed into the result:

\[
A=\operatorname{softmax}_{\text{keys}}(QK^\top/\sqrt{d_k}),\qquad Y=AV.
\]

Here rows of Q correspond to queries, columns of the score matrix to keys, and each softmax row sums to 1. $d_k$ is the width of **one query/key head**, not the whole model. Dividing by its square root controls score scale.

Suppose we reorder every input row using the same permutation matrix $P$. Shared rowwise projections produce $PQ,PK,PV$. The score matrix becomes $P(QK^\top)P^\top$: its rows and columns are merely rearranged. Rowwise softmax respects that rearrangement, so

\[
\operatorname{Attention}(PX)=P\operatorname{Attention}(X).
\]

This is **permutation equivariance**. Output rows move with their inputs. It is different from invariance: an invariant output stays unchanged. Mean pooling the equivariant output gives an invariant whole-sequence representation. Shared feedforward networks, per-row LayerNorm and residual additions preserve the same symmetry when no other positional signal is present.

For a set of measurements whose order is irrelevant, this is useful. For a movement whose direction matters, it is a limitation. An attention-based set classifier cannot distinguish a sequence from its reversal if the two differ only in row order.

**Figure — the same points, two journeys.** A trajectory panel shows the same marked coordinates with opposite arrow directions, alongside an unordered cloud that is identical in both cases. A second strip attaches slot labels 0, 1, 2,… to the points. Moving a labeled record on the screen preserves its meaning; assigning an old point to a different slot changes the sequence. Keep these two edits separate throughout the lesson.

### A causal mask already supplies some structure

The proof assumed the permitted query–key pairs were also unchanged or consistently permuted. A fixed causal mask permits a query to read only itself and preceding slots. Arbitrarily shuffling content while leaving this triangle fixed changes who can read whom. The earlier equivariance proof no longer applies.

For example, with equal scores and scalar values `[2,6,10]`, causal attention produces prefix means `[2,4,6]`. Reversing the inputs against the same mask gives `[10,8,6]`, not a reversal of the old outputs. With a special beginning-of-sequence marker, uniform attention can even expose a quantity such as $1/(t+1)$, because the marker is one item among a growing prefix. Additional layers can use such structure. This is why decoder-only Transformers without explicit position embeddings, often called **NoPE**, are a meaningful research design. It does not imply that every such model automatically learns robust counting or long-context reasoning. The [NoPE/length-generalization study](https://arxiv.org/pdf/2305.19466) distinguishes the causal setting and evaluates actual tasks rather than equating computability with generalization.

Position can therefore arrive through several routes: explicit coordinates, position vectors, score biases, causal/local connectivity, recurrent state or a combination. Our task is to identify which route the model actually has.

## 2. Add a position vector: learned and sinusoidal encodings

### A label the model can learn

Let a token or measurement have a content vector $x_t\in\mathbb R^d$. The simplest construction is

\[
h_t=x_t+p_t.
\]

Both vectors have the same width. Addition keeps the shape $L\times d$, so the existing Transformer block can consume it.

A **learned absolute embedding** stores a table $P\in\mathbb R^{L_{\max}\times d}$, and chooses $p_t=P[t]$. Its rows begin as ordinary trainable parameters. During training, a prediction error updates the rows used by that example, together with the rest of the network. If several examples use slot 7, their gradients contribute to the same position row 7.

Take a deliberately small table:

| Slot | Position vector |
|---|---|
|0|[0.2, 0.0]|
|1|[0.0, 0.3]|
|2|[−0.1, 0.1]|

If the content vector `[1,2]` occurs in slot 0, its combined vector is `[1.2,2]`. The same content in slot 1 becomes `[1,2.3]`. These numbers are a hand fixture, not learned weights from a language model. They make the operation visible: the table tells the model which slot a row occupies; training decides what to do with that signal.

The table has $L_{\max}d$ parameters. A 512 × 768 table has 393,216. A table with 512 rows supports indices 0–511. Index 512 has no row. Enlarging the table solves the storage problem but leaves a learning problem: new rows need a considered initialization, interpolation or further training strategy. Full retraining from scratch is not mathematically required, and a larger table alone does not establish useful longer-context behavior.

Do not silently replace an out-of-range index with `index % max_length`. That makes different absolute positions share an embedding without having trained the model for this periodic rule.

### Smooth clocks instead of a table

We can compute the position vector from a fixed formula. The original Transformer used sine/cosine pairs at different frequencies:

\[
p_{t,2r}=\sin(t\omega_r),\quad
p_{t,2r+1}=\cos(t\omega_r),\quad
\omega_r=b^{-2r/d},\quad r=0,\ldots,d/2-1.
\]

This definition assumes even $d$, uses radians and commonly starts from $b=10000$. Frequency means radians of phase change per position. The wavelength is $2\pi/\omega_r$ positions for a complete turn.

Think of several clock hands turning at different rates. One fast hand is ambiguous after it comes around again, but the other hands are at different phases. Together they supply a rich position signature. This analogy concerns multiple scales; it does not make a floating-point vector an infinitely precise position identifier.

For $d=8,b=10000$, the four frequencies are `[1,0.1,0.01,0.001]`. Evaluating the formula gives:

| Position | sin pair 0 | cos pair 0 | sin pair 1 | cos pair 1 | sin pair 2 | cos pair 2 | sin pair 3 | cos pair 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|0|1|0|1|0|1|0|1|
|1|0.841|0.540|0.100|0.995|0.010|1.000|0.001|1.000|
|2|0.909|−0.416|0.199|0.980|0.020|1.000|0.002|1.000|
|3|0.141|−0.990|0.296|0.955|0.030|1.000|0.003|1.000|

Slow channels appear constant at three decimal places even when their exact values differ. Conversely, the fastest hand turns nearly half a revolution from position 0 to 3. Adjacent positions need not look similar in every channel.

**Figure — clock hands, traces and a coordinate table.** Select a pair and connect its unit-circle phase to its sine/cosine curves. The encoding heatmap uses a signed scale from −1 to 1, with exact row/column labels. Changing the width recomputes every frequency: the first eight coordinates of a width 32 encoding are not the same as a width 8 encoding. A static 16 × 32 example accompanies the controllable small view.

### Why pairs make relative shifts expressible

The addition identities for sine and cosine give

\[
\begin{bmatrix}\sin((t+\delta)\omega)\\\cos((t+\delta)\omega)\end{bmatrix}
=
\begin{bmatrix}\cos(\delta\omega)&\sin(\delta\omega)\\-\sin(\delta\omega)&\cos(\delta\omega)\end{bmatrix}
\begin{bmatrix}\sin(t\omega)\\\cos(t\omega)\end{bmatrix}.
\]

The matrix depends on the shift $\delta$, not on the starting slot $t$. It is a rotation with a sign/order convention appropriate to the `[sin,cos]` column. This means a shared linear transformation can relate the code of a position to the code of a fixed offset. It does not mean that a learned attention head automatically selects “exactly three positions earlier.”

Another useful identity is

\[
p_m^\top p_n=\sum_r\cos((m-n)\omega_r).
\]

For the unprojected position vectors alone, the inner product depends on relative offset. But attention compares projected **content-plus-position** vectors. Let $M=W_Q^\top W_K$ under a column-vector convention. Then

\[
(x_m+p_m)^\top M(x_n+p_n)
=x_m^\top Mx_n+x_m^\top Mp_n+p_m^\top Mx_n+p_m^\top Mp_n.
\]

There are content–content, content–position, position–content and position–position terms. An arbitrary learned $M$ need not preserve the simple cosine identity. Those interactions are part of the representation, not inherently wasted capacity. They allow position to affect values and residual features as well as attention scores.

The [original Transformer §3.5](https://arxiv.org/pdf/1706.03762) compares learned and fixed input encodings and motivates the shift identity. The [D2L explanation and runnable notebook](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html#positional-encoding) supplies another route through the multiscale picture. Neither the existence of a sine formula at position 100,000 nor a successful table lookup establishes that a trained model will use that position well.

## 3. RoPE: let relative position change the comparison

### Rotate the features after the projections

**Rotary position embedding**, RoPE, applies a deterministic rotation to each query and key. Standard RoPE leaves the values unrotated. Position enters the query–key comparison in each attention layer.

Start with two-dimensional vectors. A counterclockwise rotation through angle $\phi$ is

\[
R(\phi)=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}.
\]

The point `[1,0]` becomes `[0,1]` at $\phi=\pi/2$. Its length stays 1. A general vector turns through the same angle without changing length.

For a head of even width $d_k$, divide the coordinates into pairs. Pair $r$ gets frequency $\theta_r=b^{-2r/d_k}$. At query position $m$, rotate each pair through $m\theta_r$; at key position $n$, rotate its corresponding pair through $n\theta_r$. Write the block-diagonal collection of rotations as $R_m$:

\[
q'_m=R_mq_m,\qquad k'_n=R_nk_n.
\]

The coordinate pairs are fixed by the implementation's basis. The projections that produce their content are learned. The rotation planes themselves are not separately learned in this standard construction.

**Figure — two clocks with arrows inside.** Show raw q and k arrows, then rotate them on separate planes with shared axes and equal scale. A second pair turns more slowly. Below them, add the two pairwise dot-product contributions to obtain the full score. The picture must show signed projections; arrow closeness alone cannot explain a negative dot product.

### The relative-offset identity, step by step

Transpose reverses a rotation, and successive rotations add their angles. Therefore

\[
(R_mq_m)^\top(R_nk_n)
=q_m^\top R_m^\top R_nk_n
=q_m^\top R_{n-m}k_n.
\]

Turn both vectors by 100 extra position units, using the same frequencies, and their dot product stays unchanged. Turn only the key, and the relative angle changes. That is the useful built-in structure.

Notice what the identity holds fixed: the **content vectors** $q_m,k_n$. Different words or hidden states still produce different vectors. RoPE has not replaced content similarity with a distance-only score. The vectors in a later layer may already reflect boundaries, masks and earlier context, so a statement about this local operation must not be mistaken for universal translation invariance of an entire language model.

One pair contributes

\[
(q_0k_0+q_1k_1)\cos(\Delta\theta)
 +(q_1k_0-q_0k_1)\sin(\Delta\theta),\quad \Delta=n-m.
\]

The sine term can distinguish the direction of the offset. With suitable content vectors, “three before” and “three after” can receive different scores. The denominator $\sqrt{d_k}$ and rowwise softmax follow this calculation as usual.

### A complete four-coordinate example

Take $q=[0.8,-0.5,0.3,1.2]$ at position 3 and $k=[1,0.25,-0.5,0.75]$ at position 7. For $d_k=4,b=10000$, frequencies are 1 and 0.01.

1. Query pair 0 turns through 3 radians; query pair 1 through 0.03.
2. Key pair 0 turns through 7 radians; key pair 1 through 0.07.
3. The rotated vectors are
   $q'\approx[-0.721434,0.607892,0.263870,1.208459]$ and
   $k'\approx[0.589656,0.845462,-0.551233,0.713192]$.
4. Their dot product is 0.804961. Dividing by $\sqrt4=2$ gives the attention logit 0.402481.
5. Computing $q^\top R_4k$ gives the same 0.804961. Positions 103 and 107 also give that value.

The query's norm remains 1.555635. Moving the key to position 8 changes the dot product to 1.570549; bringing it to the same position as the query gives the raw dot product 1.425. A farther key can receive a **larger** score. Position modulates a content-dependent comparison; it is not a mandatory recency penalty.

### A diagonal pattern requires a controlled example

A matrix is **Toeplitz** when each diagonal is constant. If the same content query appears at every position and the same content key appears at every position, RoPE scores depend only on their offsets, producing such a matrix. With q above repeated four times for both Q and K, the diagonal entries are all 2.42 and first off-diagonal entries are 2.010793.

Now double the content vector at position 1. Its self-score becomes 9.68; neighboring scores involving it double. Other scores remain unchanged. The matrix is no longer Toeplitz, although every comparison still satisfies the RoPE identity. This is a useful controlled contrast for understanding an attention heatmap.

RoPE also does not make every score decrease monotonically with distance. Concentrate q and k on the first pair as `[1,0]`: the score is simply $\cos\Delta$, which falls toward −1 near distance 3 and rises to 0.96017 at distance 6. Frequency mixtures can create useful distance structure, but they do not remove this counterexample. The [RoFormer construction](https://arxiv.org/html/2104.09864v5#S3) and [Position Interpolation analysis](https://arxiv.org/html/2306.15595v2#S2) help distinguish the exact rotation identity from assumptions about longer-distance behavior.

### Coordinate layout and partial rotation

Our implementation pairs adjacent entries `(0,1),(2,3),…`. Another valid convention pairs the first half with the second half: `(0,d_k/2),(1,d_k/2+1),…`. For four coordinates, the permutation `[0,2,1,3]` converts the adjacent representation into the half-split representation. Permute the projection outputs, frequencies and inverse mapping consistently, and these are two coordinate descriptions of the same operation. Changing the rotation routine while loading unchanged checkpoint weights generally is not equivalent.

Some architectures rotate only an even number $d_r\leq d_k$ of coordinates. Split q and k into rotary and unrotated portions:

\[
(q_m^R)^\top R_{n-m}k_n^R+(q_m^C)^\top k_n^C.
\]

The content-only term and the relative positional term coexist. State whether frequencies use $d_r$ or a checkpoint-specific rule; do not silently substitute the model width. We will use this split again in [Multi-Head Latent Attention](/learn/path/full-curriculum/multi-head-latent-attention-mla?module=deep-learning-fundamentals).

Values are unrotated in the standard mechanism taught here, because the weights already determine how their content is mixed. Rotating values would define a different architecture and requires explaining how its output coordinates transform. It is not an impossible mathematical operation or proof that every such variant trains poorly.

## 4. ALiBi: express a preference in logit space

### A score penalty has a precise probabilistic effect

In causal attention, query position $i$ may read key positions $j\leq i$. **Attention with Linear Biases**, ALiBi, uses

\[
s_{ij}=q_i^\top k_j/\sqrt{d_k}-a_h(i-j),\qquad a_h>0.
\]

The positive slope $a_h$ is fixed per head in the original scheme. Future positions remain masked. This finite penalty does not replace a causal or padding mask: it discourages a distant legal key; a mask prohibits a key.

For two legal keys j and r, softmax gives the exact odds ratio

\[
\frac{A_{ij}}{A_{ir}}
=\exp(c_{ij}-c_{ir})\exp[-a_h((i-j)-(i-r))],
\]

where $c$ denotes scaled content scores. If content scores tie, an extra distance $D$ multiplies the odds by $e^{-a_hD}$. With $a_h=0.5,D=10$, the factor is 0.006738, not one-half. The distance that halves these equal-content odds is $\ln2/a_h\approx1.3863$. These are odds between keys; an individual normalized probability also depends on all other keys.

The logit penalty grows linearly. The induced multiplicative factor in unnormalized attention grows or decays exponentially. That connection makes the name and effect easier to remember.

### Content can overcome the preference

Consider a query in slot 3 and keys in slots 0–3. Suppose their scaled content scores are `[2,0,0,0]`. With slope 0.5:

| Key position | Distance | Content score | Bias | Final score | Final attention weight |
|---|---:|---:|---:|---:|---:|
|0|3|2|−1.5|0.5|0.455054|
|1|2|0|−1.0|−1.0|0.101536|
|2|1|0|−0.5|−0.5|0.167405|
|3|0|0|0|0|0.276004|

Without the bias, key 0 receives 0.711235. The distance penalty weakens its advantage, but it still gets the largest weight. The model can learn content scores that counteract a fixed penalty; the penalty itself does not learn. There is no hard finite attention window in this formula.

**Investigation — a competition between evidence and distance.** Edit the four content scores and move the key positions. Predict whether the farther key will outrank the recent one, then reveal the logit bars and normalized weights. Holding content equal isolates the exponential distance preference; setting slope 0 is the exact no-bias control. Showing only straight bias lines would miss this interaction.

### Different slopes, different preferences

For a power-of-two head count H, the original schedule gives

\[
a_h=2^{-8h/H},\quad h=1,\ldots,H.
\]

With eight heads the slopes are 1/2, 1/4,…, 1/256. With two heads they are 1/16 and 1/256, not 1/2 and 1/256. With sixteen heads the first is $1/\sqrt2$. Steeper slopes prefer shorter distances more strongly when content scores are comparable. Head behavior still depends on its learned Q/K features.

The [author's implementation](https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L693) extends the schedule to a non-power-of-two count by retaining a lower power-of-two schedule and inserting selected slopes from the doubled schedule. Its three-head result is `[1/16,1/256,1/4]`. Preserve that convention when reproducing its checkpoints; a different geometric schedule is a design variation, not a reproduction.

There is also a useful implementation identity. For one causal row,

\[
-a_h(i-j)=a_hj-a_hi.
\]

The last term is constant across that row. Softmax ignores a common additive constant, so adding $a_hj$ alone gives the same probabilities if the legal key set is the same. This permits compact bias construction. It does not mean you can omit the causal mask or forget which entries belong to a packed sequence.

The [ALiBi paper](https://arxiv.org/html/2108.12409v2#S3) reports length-extrapolation results in specified language-model experiments. It does not prove that this bias solves every long-range task, that all distant facts remain retrievable or that runtime overhead is identical across kernels.

### Bidirectional distance needs a direction decision

For an encoder that can read both directions, one possible adaptation is $-a_h|i-j|$. It favors proximity on either side. But absolute distance cannot tell left from right. Reverse a sequence of length $L$: slots i, j become $L-1-i,L-1-j$, and their absolute distance is unchanged.

If all other operations are shared per row and the final readout is mean pooling, this symmetric adaptation produces the same prediction for a trajectory and its reversal. The real investigation below demonstrates it. Signed relative buckets, explicitly directed heads, absolute slots or another directional signal can break that symmetry. The original causal ALiBi model already has a directed mask, so this particular reversal argument does not apply to it.

## 5. Other relative encodings explain the design space

### A learned relation vector: Shaw-style attention

An offset can be represented by a trainable vector instead of a fixed rotation or scalar penalty. For query i, key j, set $r=\operatorname{clip}(j-i,-k,k)$. A Shaw-style head uses

\[
s_{ij}=\frac{q_i^\top(k_j+a_r^K)}{\sqrt{d_k}},\qquad
y_i=\sum_j A_{ij}(v_j+a_r^V).
\]

The score contribution $q_i^\top a_r^K$ depends on what the query seeks. With $q=[2,0]$, relation vector `[0.5,1]` adds 1 to the unscaled dot product. Another query `[0,2]` gets 2 from that same relation. A learned scalar bias cannot express this particular query-dependent distinction on its own.

Clipping at $k=2$ assigns offsets −9 and −2 to the same relation category. This saves parameters but deliberately loses their exact distance at this component. There are $2k+1$ relation vectors per table; the value relation can convey which relation supplied information, beyond just changing the weight. The [original relation-aware attention paper §§3.1–3.3](https://aclanthology.org/N18-2074.pdf) develops this construction and its efficient decomposition.

### A learned scalar by distance bucket: T5-style bias

A simpler mechanism learns one scalar per head and relative-distance bucket, then adds it to the content score. Nearby offsets receive finer categories; large distances share coarser categories. The model learns the preference for each category. Unlike ALiBi, it need not be monotone in distance.

For the common bidirectional 32-bucket, maximum-distance 128 configuration, divide the buckets by direction. Within one direction, distances 0–7 have exact bins and larger distances enter logarithmically widening bins. Under the [T5 implementation's key-minus-query convention](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L198):

| Offset $j-i$ |−129|−128|−16|−8|−7|−1|0|1|7|8|16|128|129|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bucket |15|15|10|8|7|1|0|17|23|24|26|31|31|

The bucket number is an index, not a magnitude. Bucket 31's learned value might be positive or negative. Positions 128 and 129 sharing a bin means this bias component cannot distinguish those two distances; content and other layers may still distinguish their tokens.

For distance $D$ in one direction, with $B$ available buckets and exact range $E=B/2$, the large-distance index is

\[
\min\left(B-1,\;E+\left\lfloor
\frac{\ln(D/E)}{\ln(D_{\max}/E)}(B-E)
\right\rfloor\right).
\]

Use the exact index $D$ for $D<E$, and add the direction offset afterward. A causal variant allocates buckets differently because future keys are prohibited. It still needs the causal mask; mapping an illegal future offset to a bucket does not authorize attention to it.

T5's original attention parameterization also omits the usual explicit $1/\sqrt{d_k}$ score scale. A lesson comparing **bias mechanisms** may use a common scaled-content convention, but a checkpoint reproduction must match its complete attention rule. “Relative bias” names the position component, not every surrounding implementation detail.

These methods are alternatives with different expressive choices. There is no historical rule that every later method strictly replaces every earlier one.

## 6. A real investigation: can the model see movement direction?

### The data and the question

Return to the real **Libras Movement** trajectories used in the preceding lessons. Each record contains 45 ordered x/y hand-centroid samples and one of 15 movement-type labels. Class 4 is an anticlockwise arc; class 5 is a clockwise arc. These are simplified movement measurements, not complete signed-language sentences or a deployed recognition system.

The [UCI source and original metadata](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement) are available under CC BY 4.0, credited to Daniel Baptista Dias, Sarajane Marques Peres and Helton Hideraldo Bíscaro. The [offline input](movement_libras.data), [original metadata](movement_libras.names) and [provenance record](data-provenance.md) accompany this lesson. Coordinates lie in 0–1; our model uses the fixed transformation $2x-1$. Position indices 0–44 describe sample order, not exact elapsed seconds.

The practical question is: **does changing direction become visible to the classifier, and how does that depend on the encoding?** Accuracy on the whole dataset and sensitivity to this particular intervention are different measurements.

The raw 360 rows contain 30 additional exact duplicates. We retain the first occurrence of each unique 90-coordinate trajectory after checking that duplicate labels agree. A fixed classwise split assigns 220 unique records to training, 50 to validation and 60 to test, with four test records per class. Whole trajectories stay together. The collection describes four performers and two sessions but does not provide reliable per-row identities, so this is a row-level study; it cannot establish performance on a new performer or session. The exact split and duplicate groups are saved with the program's results.

### Five controlled models

Each model has a 2-to-24 coordinate projection, one pre-normalized Transformer block with two heads of width 12 and a 48-wide GELU feedforward layer, final LayerNorm, mean pooling and a 24-to-15 classifier. Q/K/V and output maps have biases. LayerNorm uses epsilon $10^{-5}$. There is no dropout.

| Model | Where position enters |
|---|---|
|No-position control|Nowhere; every row uses the same operations|
|Sinusoidal|Add the width 24 vector to the coordinate embedding|
|Learned|Add a learned 45 × 24 table row|
|RoPE|Rotate the 12-dimensional Q/K vectors before comparing them|
|Symmetric ALiBi|Add a two-head $-a_h|i-j|$ proximity bias; no causal mask|

All shared tensors have exactly the same seed 101 initialization. The learned table uses its own generator 303, initialized with standard deviation 0.02, so creating it does not change the shared weights. The table adds 1,080 parameters: 6,447 versus 5,367 in each other model. Fixed sinusoidal and RoPE frequencies use base 10000. The ALiBi slopes are 1/16 and 1/256.

Every model gets 180 full-batch Adam updates with learning rate 0.003 and no weight decay. Select its checkpoint by validation macro-F1, then lower validation cross entropy, then the earliest exact tie. Macro-F 1 averages the 15 classwise F1 scores equally. The test records are evaluated after this selection. This is one predeclared seed with one common training protocol, not a large tuning study or a language-model context-extension experiment.

The actual CPU results were:

| Encoding | Selected epoch | Train correct /220 | Validation correct /50 | Test correct /60 | Test macro-F1 | Test CE, nats/record |
|---|---:|---:|---:|---:|---:|---:|
|None|107|193|35|33|0.528|1.287|
|Sinusoidal|154|217|38|42|0.689|1.039|
|Learned|108|220|40|40|0.656|1.017|
|RoPE|106|203|35|40|0.632|1.120|
|Symmetric ALiBi|180|218|29|39|0.624|1.289|

Sinusoidal happens to have the highest test-correct count in this run. That does not establish a universal ranking. The models have different inductive biases, only one seed was run, the learned table changes parameter count, and the common hyperparameters need not suit all methods equally. Keep the table as an observation of this protocol. The sharper result comes from the following symmetry test.

### Predict before reversing the path

The displayed example is source row 77, chosen beforehand as the first held-out class 4 record in the fixed split. All five selected models classify its original version incorrectly. The example is kept because a real failure can still reveal the mechanism.

There are two different edits:

1. **Move records together.** Reverse the storage/display order of the point records and move each original position ID with its point. The sequence meaning has not changed. Every model's pooled logits stay the same within float32 rounding, with maximum changes below $1.6\times10^{-6}$.
2. **Reverse the movement.** Reverse the points but retain slot IDs 0–44. A point formerly attached to the beginning is now attached to the end. This changes the ordered trajectory.

For the second edit, the observed largest change among the 15 logits was:

| Model | Maximum absolute logit change | Original predicted class → reversed prediction |
|---|---:|---|
|None|0.000001|5→5|
|Sinusoidal|3.652706|9→9|
|Learned|6.880572|7→5|
|RoPE|6.110313|3→5|
|Symmetric ALiBi|0.000002|9→9|

The unrounded ALiBi change is $1.55\times10^{-6}$, numerical noise around an exact real-arithmetic invariance. Its distance matrix is unchanged by reversal after the corresponding row/column rearrangement, so the same proof as in §1 propagates through the block and mean pool. No amount of ordinary parameter training can break this architectural symmetry while its assumptions remain in place.

The sinusoidal classifier's predicted class stays 9, but its logits change substantially. Looking only at the largest-probability class would conceal its directional sensitivity. The RoPE model's class 5 probability changes from 0.084402 to 0.969648. That shows the intervention affected its decision; it does not supply a verified label for an artificially edited recording. We do not automatically grade any reversed or hand-edited sample as a newly labeled real example.

### A visual workspace that exposes the mechanism

**Investigation — reattach a trajectory to its slots.** The main view shows the path with direction arrows, frame numbers and a linked time strip. The learner can reverse the path, jointly reorder records or drag a single point. Choosing a point reveals its actual Q/K features, rotated pair where applicable, selected query's attention weights and final 15-class probabilities. An unset prediction asks whether the **logits**, not merely the winning class, should remain unchanged. Reveal after committing the prediction, then compare the predicted change with the actual result.

The coordinate edit is also real computation: reflect frame 23's x coordinate from $x$ to $1-x$. On this example it changes the maximum logit by 2.890378 in the no-position model and 3.264872 in symmetric ALiBi. Reversal invariance does not mean these models ignore the coordinates. In the ALiBi run this edit even changes the winning class 9→4; an artificially edited sample still has no newly established ground-truth label.

As another control, append five padded points at coordinates $[0.75,0.75]$, assign harmless position IDs 0, and exclude them from attention **and pooling**. Original valid positions remain 0–44. The original logits are recovered within float32 rounding for all five models. Leaving the pads unmasked changes the computation; for the sinusoidal model the largest logit change is 8.816077. Masking keys alone is not sufficient if the final mean still includes padded query rows.

No retraining occurs when the learner edits an input. The workspace evaluates the saved selected model. It must display that scope clearly: predictions are from a small movement classifier, not from a pretrained language model, and attention weights are one internal calculation rather than a causal explanation of the whole decision.

### Reproduce and investigate further

Download [the complete CPU program](author-calculations.py), the two original data files and [its recorded results](author-results.json) into one directory. With Python, NumPy, PyTorch and scikit-learn installed, run:

```bash
python author-calculations.py
```

The program contains the full data-boundary checks, all five model definitions, training/checkpoint selection, metrics, actual interventions and weight export. It has no hidden notebook state, pretrained download or GPU dependency. The recorded run used Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu and scikit-learn 1.9.1, with one CPU thread and deterministic algorithms. Last-digit results can vary with a different numerical environment.

Read `PositionClassifier.forward` in this order: construct the content rows; add an input position signal when selected; normalize; project Q/K/V; apply RoPE or ALiBi at its own location; mask; mix values; perform the residual/feedforward block; normalize and valid-pool; classify. This is the same Transformer mechanism from the previous lesson with a deliberately isolated positional choice.

The retained [model/trace evidence](position-models.json) supports the interactive view, while the full training histories are optional reproduction material. A production page should load only the selected model's compact weights on demand. The source study remains offline; training all variants in the browser would add cost without improving this investigation.

## 7. Implement positions and verify cached attention

### The cache needs a coordinate system

A **KV cache** stores past keys and values so an autoregressive decoder does not project the same past tokens again at every new step. Standard RoPE implementations often cache the already-rotated keys $R_nk_n$, with unrotated values. A new query at position $t$ must use $R_tq_t$.

Imagine a chunk whose real positions are 7, 8, 9. When processing its last token separately, the local query tensor has length 1. Calling `arange(1)` gives position 0, not 9. Its shape is valid, but its relative angles to the cached keys are wrong.

Track three notions separately:

| Quantity | Meaning | Example |
|---|---|---|
|Logical position ID|The coordinate used by the encoding|9|
|Cache slot|Where this key/value is stored|Slot 2 in this three-entry toy cache|
|Valid/causal relation|Which stored entries this query may read|Positions 7, 8, 9 are legal|

In a contiguous cache these may have simple relationships. With left padding, packed documents, sliding windows, evictions or reused prefixes, they can differ. RoPE does not remove the need to maintain those relationships. Learned absolute embeddings also do not intrinsically require a separate position field in every cached vector; implementations track enough metadata for their chosen layout and next position.

### A runnable, transparent reference

This complete program uses NumPy and one head so the geometry and cache contract are visible. It implements sinusoidal features, strict learned lookup, adjacent-pair RoPE, the original ALiBi slope schedule, explicit logical-position causal masking, and a full-versus-cached comparison. It is a correctness reference, not a fused production kernel. The small projected Q/K/V arrays are declared hand fixtures, not claimed learned model activations.

```python
import math
import numpy as np


def sinusoidal(positions, width, base=10000.0):
    if width < 2 or width % 2:
        raise ValueError("Use an even encoding width of at least two.")
    positions = np.asarray(positions, dtype=np.float64)
    frequencies = base ** (-np.arange(0, width, 2) / width)
    angles = positions[..., None] * frequencies
    return np.stack((np.sin(angles), np.cos(angles)), -1).reshape(
        *positions.shape, width)


def learned_positions(table, positions):
    positions = np.asarray(positions)
    if not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("Table positions must be integers.")
    if np.any(positions < 0) or np.any(positions >= len(table)):
        raise ValueError("Position is outside the learned table.")
    return table[positions]


def rotate(values, positions, base=10000.0):
    values = np.asarray(values, dtype=np.float64)
    width = values.shape[-1]
    if width < 2 or width % 2:
        raise ValueError("Use an even rotary width of at least two.")
    angles = np.asarray(positions)[..., None] * base ** (
        -np.arange(0, width, 2) / width)
    even, odd = values[..., 0::2], values[..., 1::2]
    return np.stack((even * np.cos(angles) - odd * np.sin(angles),
                     even * np.sin(angles) + odd * np.cos(angles)),
                    -1).reshape(values.shape)


def slopes(head_count):
    if head_count < 1:
        raise ValueError("A head count must be positive.")
    def powers(count):
        start = 2 ** (-2 ** -(math.log2(count) - 3))
        return [start ** (index + 1) for index in range(count)]
    lower = 2 ** int(math.floor(math.log2(head_count)))
    if lower == head_count:
        return np.array(powers(lower))
    return np.array(powers(lower) + powers(2 * lower)[::2][:head_count-lower])


def mix(query, keys, values, query_ids, key_ids, mode, slope=.5):
    query_ids, key_ids = np.asarray(query_ids), np.asarray(key_ids)
    scores = query @ keys.T / math.sqrt(query.shape[-1])
    if mode == "alibi":
        scores -= slope * (query_ids[:, None] - key_ids[None, :])
    legal = key_ids[None, :] <= query_ids[:, None]
    if np.any(~legal.any(axis=-1)):
        raise ValueError("Every query needs at least one legal key.")
    scores = np.where(legal, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ values


q = np.array([[1, 0, .5, -1], [.5, 1, -1, .25], [2, -.5, .25, 1]])
k = np.array([[.5, 1, 1, 0], [1, -.5, .5, 1], [-.5, .75, 1, -1]])
v = np.array([[2, 0], [0, 3], [1, -1]])
positions = np.array([7, 8, 9])

print("position 1:", np.round(sinusoidal([1], 8)[0], 3))
print("three-head slopes:", slopes(3))
table = np.array([[.2, 0], [0, .3], [-.1, .1]])
print("learned slots 2,0:", learned_positions(table, [2, 0]))

for mode in ("rope", "alibi"):
    q_used = rotate(q, positions) if mode == "rope" else q
    k_used = rotate(k, positions) if mode == "rope" else k
    full = mix(q_used, k_used, v, positions, positions, mode)

    # Prefill positions 7 and 8, then append a correctly positioned new key.
    cached_keys, cached_values = k_used[:2].copy(), v[:2].copy()
    cached_keys = np.concatenate((cached_keys, k_used[2:]), axis=0)
    cached_values = np.concatenate((cached_values, v[2:]), axis=0)
    last = mix(q_used[2:], cached_keys, cached_values,
               positions[2:], positions, mode)
    print(mode, "last:", np.round(last[0], 6),
          "matches full:", np.allclose(last, full[2:], atol=1e-12))

# Keep the legal cache entries fixed, but give the query the wrong RoPE angle.
wrong = mix(rotate(q[2:], [0]), rotate(k, positions), v,
            [9], positions, "rope")
print("wrong query angle:", np.round(wrong[0], 6))
```

Recorded outputs include:

```text
position 1: [0.841 0.54  0.1   0.995 0.01  1.    0.001 1.   ]
three-head slopes: [0.0625     0.00390625 0.25      ]
learned slots 2,0: [[-0.1  0.1]
                  [ 0.2  0. ]]
rope last: [1.035332 1.29716 ] matches full: True
alibi last: [0.340434 2.281648] matches full: True
wrong query angle: [0.65852  1.291697]
```

For the RoPE last query, the actual weights over positions 7, 8, 9 are approximately `[0.487697,0.452366,0.059937]`. With the wrong query angle they become `[0.185156,0.526635,0.288209]`. There is no shape error to warn us. Comparing the actual output with a full causal reference catches the bug.

**Investigation — repair the cache timeline.** The learner receives a new query, occupied cache slots and editable logical position IDs. They must decide whether full and cached output should agree before revealing it. One branch changes an ID; another changes only the storage arrangement while moving keys, values and metadata together. A separate control shifts *every* logical ID by the same constant. The latter preserves RoPE/ALiBi local scores under fixed frequencies and the same legal relation. These controls distinguish a representation permutation, a genuine offset change and a masking bug.

### Padding, packed records and numerical precision

For a left-padded batch, a common logical-position construction is `valid.cumsum(-1)-1`, assigning the first valid token position 0. Replace padding indices with a safe value before table lookup, then exclude those entries with the mask. Whether this convention matches a specific checkpoint depends on its training and generation implementation. A uniform position shift cancels locally in standard RoPE when all relevant content and masks are held fixed; arbitrary per-token renumbering does not. Additive learned/sinusoidal vectors generally change under even a common shift.

When independent documents are packed into one tensor, resetting their position IDs is not enough. A document/block mask must prevent an example from attending to another example. Otherwise equal position IDs do not stop information leakage. Likewise, a key-padding mask often suppresses padded keys but does not automatically erase padded query outputs; exclude those outputs from losses or pooling as appropriate.

Compute phases at adequate precision. Adjacent large integers can become the same number if converted too early to a low-precision format. For example, BF16 cannot represent every integer beyond 256. Constructing all positions directly in BF16 can therefore give neighboring tokens identical phases before sine/cosine is evaluated. A common implementation calculates phases in float32 and casts the resulting sine/cosine values as needed; our small reference uses float64. Neither floating-point choice provides arbitrary precision at unlimited positions. Validate the actual target range, checkpoint convention and kernel.

Standard PyTorch Transformer layers do not automatically choose a positional scheme. Use the model's actual embedding/attention implementation and versioned documentation rather than assuming `nn.TransformerEncoder` inserts RoPE or sinusoidal features for you. The real program above makes the injection site explicit; production implementations may fuse the same operation without materializing the intermediate arrays.

## 8. Deeper route: extending the context without confusing the claims

### Three different length questions

“This model supports a longer context” can mean several things:

1. **The operation is defined.** The position lookup/rotation, mask and cache accept that length.
2. **The model remains useful.** Loss and downstream quality stay acceptable on the new distribution.
3. **The model uses the additional information.** It can retrieve or combine evidence far away, rather than ignoring most of the added context.

An ALiBi penalty is defined at arbitrarily large finite distances in real arithmetic, but grows without a finite bound as distance increases. A sine/rotation formula is also defined beyond training indices. Neither is a theorem about the model's behavior on a longer task. The softmax denominator sees more competitors, content distributions change, and long-range computations may require skills the training examples never demanded.

Even good average language-model loss can hide a failure on an instruction that needs one distant fact. A long-context evaluation should vary both total length and evidence location, include distractors and multiple pieces of evidence, inspect short-context retention, and measure the task that matters. A single successful “needle in a haystack” example does not establish long-document reasoning. The [NoPE study](https://arxiv.org/pdf/2305.19466) is useful here because it evaluates several algorithmic tasks and separates them from perplexity claims.

### Frequency, wavelength and the base

For $d_k=64$, base 10000 gives first frequency 1 and last frequency $10000^{-31/32}\approx0.00013335$. The first wavelength is $2\pi\approx6.2832$ positions; the last is 47,117.243.

Changing the base to 500000 leaves the first frequency exactly 1. For pair $r$ it multiplies the wavelength by

\[
\left(\frac{500000}{10000}\right)^{2r/64}=50^{r/32}.
\]

The last wavelength becomes 2,084,764.773, about 44.25 times as long. It is not 50 times for every pair, and the first pair does not change at all. At position 128,000, a 47,117-position wavelength has completed about 2.72 turns, not 20. One channel wrapping is not the same as the entire multiscale code becoming identical, nor must every channel remain monotone over the whole context.

**Figure — frequency ruler.** Plot wavelengths against pair index on a labeled logarithmic axis, with the target context as a horizontal reference. A companion circle shows the selected pair's phase. Values come directly from the frequency formula; there are no performance axes. Increasing the base creates slower channels, but it does not by itself demonstrate better retrieval or higher accuracy.

The base and rotary dimension are checkpoint/architecture choices. There is no universal formula such as “base equals 60 times the target length.” A suitable choice depends on learned Q/K features, training lengths, positional scaling and task. The [Llama 3 report](https://arxiv.org/html/2407.21783v3#S3.S2) records a 500000 base; its [long-context training section](https://arxiv.org/html/2407.21783v3#S3.S4.SS2) also describes staged long-sequence training. The resulting capability was not created by editing a configuration field after an otherwise unchanged short-context training run.

### Position Interpolation

Let a model be trained with nominal context $L$, and choose a target $L'=sL$. **Position Interpolation**, PI, feeds $t/s$ into the same rotation formula:

\[
R_t\quad\longrightarrow\quad R_{t/s}.
\]

Equivalently, divide all angular frequencies by $s$. Distances are compressed too: two new positions 8 slots apart have the old phase difference of 1 slot when $s=8$. This moves a large range of offsets into a smaller phase range, but also changes local distinctions.

For training length 64 and target 256, $s=4$. Position 255 maps to 63.75, not 63. The interval `[0,256)` maps into `[0,64)`, but the model was trained at integer positions 0–63; fractional positions and altered token spacing are a new input condition. Endpoint-preserving scaling $t(63/255)$ is another possible rule, with slightly different spacing. Name which rule you use.

The [PI paper §2.3](https://arxiv.org/html/2306.15595v2#S2.SS3) gives the rescaling and a bounded interpolation analysis for a fixed trigonometric score function. Its model experiments include further training. A bound on a fixed score function is not an end-to-end guarantee about hidden states, all softmax competitors or task correctness. Uniform rescaling can work well in a particular protocol, but it is not free of a short-distance tradeoff.

### Base scaling, often called NTK-aware scaling

One proposed alternative for rotary width $d$ greater than 2 changes the base to

\[
b'=b\,s^{d/(d-2)}.
\]

Pairr then has

\[
\theta'_r=\theta_r\,s^{-2r/(d-2)}.
\]

At $r=0$ there is no change. At the last pair $r=d/2-1$, the frequency is divided by $s$. Intermediate pairs receive intermediate scaling. This preserves the fastest phase changes while stretching the slowest wavelengths. The $d=2$ formula is undefined and must not be applied blindly.

The name refers to the reasoning that motivated the heuristic; it does not constitute a proof that any pretrained network behaves like its infinite-width neural tangent kernel or that this base change is optimal. The [YaRN paper's methodology and appendices](https://arxiv.org/html/2309.00071v3#S3) explain the relationship among PI, base changes and later interpolation schemes.

### YaRN: select frequencies and adjust score sharpness

Uniform interpolation changes every frequency; a base change changes most frequencies by different amounts. A **by-parts** construction instead asks how many turns each pair made within the original context:

\[
r_i=\frac{L\theta_i}{2\pi}=\frac{L}{\lambda_i}.
\]

Under the paper's linear ramp in this rotation count, let

\[
\gamma(r)=\operatorname{clip}\left(\frac{r-\alpha}{\beta-\alpha},0,1\right),
\qquad
\widetilde\theta_i=(1-\gamma(r_i))\frac{\theta_i}{s}+\gamma(r_i)\theta_i.
\]

Pairs with few turns are fully interpolated; pairs with many turns remain unchanged; the middle blends the two. The paper uses $\alpha=1,\beta=32$ in its Llama experiments. These are thresholds in **rotations during the original context**, not raw feature indices or universal constants.

YaRN combines this frequency treatment with an empirical attention-temperature adjustment. If softmax originally sees $q'^\top k'/\sqrt d$, dividing this logit by temperature $T$ makes it sharper when $T<1$. Scaling **both** q and k by $c=\sqrt{1/T}$ has the same effect because the dot product scales by $c^2$. The paper's suggested fit is

\[
c=1+0.1\ln s,\qquad\text{logit multiplier}=c^2.
\]

At $s=8$, $c\approx1.207944$, so logits are multiplied by about 1.459129. Multiplying them by $\sqrt{1+0.1\ln s}$ is a different rule. The temperature fit is an empirical recipe, not a conservation law that exactly compensates every longer softmax.

The full mechanism therefore changes both relative phases and attention sharpness. In partial-rotation implementations, scaling only the rotary subset does not multiply the entire dot product uniformly. Check whether the attention scale is applied separately and how unrotated channels are treated.

The displayed frequency comparison uses the **paper's ramp in rotation count**. Practical checkpoint libraries can discretize the boundary indices and use a ramp over pair indices, giving a different intermediate curve. A label such as “YaRN” is not enough to reproduce every checkpoint: use its versioned implementation, parameter names, rotary width and scaling factors. [Current Transformers RoPE documentation](https://huggingface.co/docs/transformers/main/en/internal/rope_utils) distinguishes several schemes and per-layer configurations; its `main` documentation is mutable, so pin the version when reproducing a model.

**Investigation — stretch a position system.** Choose original length, target factor and a pair. Overlay unchanged, PI, base-scaled and paper-ramp phases/wavelengths. Predict whether the fastest pair or an adjacent-token phase difference changes, then reveal the computed values. The no-extension $s=1$ case must return the original frequencies and unit score scale. These are geometry and score experiments, not a fabricated perplexity benchmark.

### Dynamic scaling and cached representations

A dynamic rule can change frequencies as the current length grows. That raises a consistency problem: old keys might have been rotated with yesterday's frequency vector while the new query uses today's.

For fixed content keys in one attention layer, you can store unrotated keys or rephase existing keys from the old rotation into the new one. In the hand fixture from §7, changing the base from 10000 to 100 and rotating only the new query gives last output `[0.973912,1.389528]`; recomputing all Q/K rotations consistently at the new base gives `[0.998788,1.344242]`.

There is a further model-level distinction. In a multilayer decoder, cached hidden states and values may themselves have been computed under the old frequencies. Rephasing keys alone does not generally reproduce a fresh full-prefix pass through **every layer** at the new fixed frequency rule. Define the intended dynamic algorithm, maintain internally consistent cache metadata and compare against the appropriate reference. If exact equivalence to a fresh pass under a new global configuration is required, earlier states may need recomputation. A “cache fix” must state what equivalence it promises.

### XPos: a relative amplitude factor as well as a rotation

XPos extends the geometry using reciprocal query/key scaling. In a real-coordinate form, for each pair $i$ choose $0<\zeta_i<1$ and a positive scale $S$:

\[
q'_m=\zeta_i^{m/S}R_mq_m,\qquad
k'_n=\zeta_i^{-n/S}R_nk_n.
\]

The pair's score becomes

\[
\zeta_i^{(m-n)/S}\,q_m^\top R_{n-m}k_n.
\]

For a causal key $n\le m$, the amplitude factor attenuates older contributions. The query/key norms individually are no longer preserved. Scaling both vectors in the same direction would produce an unwanted absolute-position factor instead of this relative factor.

The [TorchScale XPos implementation](https://github.com/microsoft/torchscale/blob/main/torchscale/component/xpos_relative_position.py) uses a pair scale equivalent to $\zeta_i=(2i/d+0.4)/1.4$, default $S=512$, a centered exponent and reciprocal scaling for keys. For the first pair, $\zeta_0=2/7$, so a 512-position causal separation multiplies that pair's score amplitude by 2/7. Higher-frequency pairs have smaller $\zeta$ under this rule. Omitting the division by $S$ would instead apply $(2/7)^{512}$, an entirely different and numerically extreme factor.

The centering choice can improve numerical range while preserving a common relative factor when handled consistently. It also belongs in the cache convention. The [XPos/LEX paper](https://arxiv.org/pdf/2212.10554) separately studies its encoding and blockwise masking; partial rotary dimensions alone are not “XPos.” A fixed amplitude decay also does not establish monotonicity of every content-dependent signed score.

## 9. Choose the position system for the actual task

### What changes, and what stays expensive?

| Mechanism | Position parameters | Main injection site | A question to ask before using it |
|---|---|---|---|
|Learned absolute| $L_{\max}d$ | Input rows | Are supported indices and training coverage adequate? |
|Sinusoidal absolute|None|Input rows|Does the model learn to use this multiscale signal on the relevant lengths?|
|Shaw-style relative vectors|Depends on clipped offset range, width and sharing|Query-dependent scores and optionally values|Which offsets can share a category without losing needed distinctions?|
|T5-style scalar buckets|Buckets×heads per shared table|Scores|How much distance/direction precision should the bins preserve?|
|Standard RoPE|None for fixed frequencies|Q/K coordinates|Which basis, rotary width, frequencies and cache offsets does the model expect?|
|ALiBi|None for fixed slopes|Scores|Is the chosen directional/proximity prior appropriate for the task and mask?|

For a new small model, a simple learned or sinusoidal baseline is useful if its position semantics match the problem. For a pretrained model, preserve its specified scheme first. A frequency change, table extension or bias replacement changes the function the checkpoint computes; it is not a cosmetic implementation substitution.

RoPE adds work linear in the number of rotated coordinates. Full attention still computes pairwise scores, with $O(BHL^2d_k)$ attention arithmetic and an $L^2$ score/probability array if implemented eagerly. A tiled attention kernel can avoid storing the full matrix while calculating the same attention function, within floating-point differences. Positional encoding itself does not make full attention linear in sequence length.

ALiBi can be generated from position vectors and head slopes. A naive implementation materializes an H × L × L bias; a compatible kernel can compute needed entries or use the causal row-constant identity. Thus “ALiBi requires an extra dense matrix” and “ALiBi is free everywhere” are both implementation-dependent claims. Report actual shapes and measured performance if making a speed comparison.

With full rotary width and matching query/key widths, standard RoPE does not reduce KV-cache dimensions. For B batches, N layers, Hkv cached heads, length $L$ and head widths $d_k,d_v$, unquantized cache storage is

\[
BNH_{kv}L(d_k+d_v)\times\text{bytes per stored element}.
\]

An example B=1, N=32, Hkv=8, L=4096, dk=dv=128 with two-byte elements needs 536,870,912 bytes, or 512 MiB, for those K/V tensors. Rotating keys changes their values, not this count. Quantization scales/metadata, padding allocation and other runtime state add their own storage. The **next** lesson, [Grouped-Query and Multi-Query Attention](/learn/path/full-curriculum/grouped-query-attention-gqa-multi-query-attention-mqa?module=deep-learning-fundamentals), changes $H_{kv}$ and explains the actual sharing computation. RoPE can apply once to each stored key head and separately to each query head, provided corresponding frequency/basis conventions match. Reducing stored heads is a different mechanism from supplying position.

### Useful applications beyond words in a sentence

**Movement, irregular time and event streams.** An ordinal sample index measures order. An actual timestamp measures elapsed time. If one sensor records events at 0, 1 and 20 seconds, assigning positions 0, 1, 2 hides the nineteen-second gap. A continuous position formula can accept `[0,1,20]`, but its frequency units are now radians per second and must suit that scale. A measurement system may need both event order and elapsed time; an additive learned lookup can instead encode a finite time-bin category. In our Libras example, only sample order was supplied reliably, so the lesson does not relabel slot differences as exact seconds.

**Images and video.** Rasterizing a patch grid into one long list creates accidental one-dimensional neighbors: the last patch of one row and first patch of the next have consecutive flattened indices. A two-dimensional encoding can represent row and column separately. One construction allocates coordinate pairs to x and y, applying phases $x\theta_i$ in one subset and $y\theta_j$ in another. Their score contributions depend on $\Delta x$ and $\Delta y$, not just a flattened offset. A video can add temporal coordinates. A position design must decide how special tokens, different resolutions and multiple frames share these coordinates. The later [Vision Transformers lesson](/learn/path/full-curriculum/vision-transformers-vit-deit-swin-dinov2?module=deep-learning-fundamentals) develops patch grids, learned-grid interpolation and relative window geometry.

**Coordinates versus identities.** In a set of physical objects, rearranging the storage order should not change a prediction if each object's actual coordinates move with it. In a sentence, swapping words while retaining their slots should change the represented sentence. This is the same distinction as our movement workspace. Choosing position IDs deliberately can preserve the symmetry you want and break the symmetry the task must distinguish.

These examples explain why there is no universal “best position vector.” Start with the relation the learner or application needs—absolute slot, signed distance, elapsed time, two-dimensional displacement—and trace where that relation enters the function.

## 10. Practice: change the situation, then explain the result

Try the questions before opening hints or solutions. Exercises 1–4 check the core route; 5–8 extend it. No pretrained download or long training run is needed.

### 1. Two rearrangements

An encoder receives points A, B, C with position IDs 0, 1, 2 and uses learned absolute input embeddings, no dropout and mean pooling. Compare (a) storing the records in order C, A, B with their IDs 2, 0, 1, and (b) assigning points C, A, B to IDs 0, 1, 2. Which output is guaranteed to equal the original? Explain at the input to the first block.

<details><summary>Hint</summary>

Write the three content-plus-position vectors before and after each edit. Check whether you merely permuted existing rows.

</details>

<details><summary>Solution</summary>

In (a), the combined rows are `[xC+p2,xA+p0,xB+p1]`, a permutation of `[xA+p0,xB+p1,xC+p2]`. The equivariant encoder followed by mean pooling produces the same output. In (b), they are `[xC+p0,xA+p1,xB+p2]`; generally these are different vectors, so equality is not guaranteed. The model can still happen to predict the same class in (b), but the architectural guarantee concerns the full output function and applies to (a).

</details>

### 2. A different rotary pair

Use one pair with frequency 1, query q=`[1,0]` at position 2 and key k=`[0,1]` at position 5. Compute the unscaled dot product after rotation. Then shift both positions by 20. Finally move only the key one further position. Does the score necessarily decrease?

<details><summary>Hint</summary>

Use $q^\top R_{n-m}k$, and work out $R_\phi[0,1]$.

</details>

<details><summary>Solution</summary>

$R_\phi[0,1]=[-\sin\phi,\cos\phi]$. The original score is $-\sin3\approx-0.141120$. Shifting both positions to 22 and 25 leaves offset 3 and the score unchanged. Moving only the key to 6 gives offset 4 and score $-\sin4\approx0.756802$, which is larger. RoPE encodes a relative phase; it does not require scores to fall with distance. If these are two-dimensional attention logits, divide the dot products by $\sqrt2$ before softmax.

</details>

### 3. How much evidence overcomes the bias?

A causal head uses ALiBi slope 0.25. Key A is 8 positions farther from the query than key B. How much larger must A's scaled content score be to tie B's final score? With equal content scores, what are their attention odds? Does that answer depend on how many other legal keys exist?

<details><summary>Hint</summary>

Subtract the final scores. Their difference controls the ratio of softmax probabilities.

</details>

<details><summary>Solution</summary>

A needs an extra $0.25\times8=2$ content-logit units to tie. With equal content scores, $A_A/A_B=e^{-2}\approx0.135335$. Other keys change both normalized probabilities but cancel from their ratio. Neither probability itself is 0.135335 unless the remaining normalization happens to make that so. A finite penalty does not prohibit A.

</details>

### 4. Construct and repair a cache bug

In the reference program, make the last query and all three keys have positions 0, 1, 2 instead of 7, 8, 9, retaining the same content vectors and legal order. Predict the RoPE last output, then check it. Now reset **only** the last query's rotation to 0 while retaining key rotations for 7, 8, 9. Explain why the two edits have different results. How would you separately expose a wrongly offset causal mask?

<details><summary>Hint</summary>

A common shift preserves all pairwise offsets. The query's rotary angle and its legal-key relation are separate inputs in this transparent program.

</details>

<details><summary>Solution</summary>

The common shift preserves offsets, so the last output remains approximately `[1.035332,1.297160]`. Resetting only the query's angle changes those offsets and gives `[0.658520,1.291697]`. The latter experiment keeps the legal entries fixed to isolate the rotation error. To expose a mask error, keep rotations correct but use a local query index 0 to construct legality against key IDs 7–9: this falsely masks every key, and the reference must reject that empty legal set. In a full decoder, use both correct logical IDs and the correct valid/document relation.

</details>

### 5. Design a directional encoder

A model uses $-a|i-j|$, shared rowwise blocks and mean pooling. You want it to distinguish a list from its reversal. Propose one change that can remove the symmetry and one change that cannot. Explain why increasing the number of identical-structure layers does not solve the issue by itself.

<details><summary>Hint</summary>

Check whether the entire pipeline commutes with the reversal permutation. Changing parameter values cannot break an equality that holds for all parameter values.

</details>

<details><summary>Solution</summary>

Adding distinct absolute position vectors, using signed relative categories with independently learned values, or introducing an appropriate directional mask can remove reversal symmetry. Increasing width or adding more shared blocks with the same symmetric-distance rule cannot: each block remains reversal-equivariant, and mean pooling remains invariant. Removing the symmetry creates the capacity to distinguish directions; it does not guarantee successful training or correct predictions on all movements. If order should be irrelevant in the application, breaking it may be undesirable.

</details>

### 6. A stretching rule is not a performance curve

A width 8 RoPE head uses base 10000 and extends nominal length 64 to 256 by PI. Find the four original frequencies, the four new frequencies, the mapped value of position 255 and the original wavelength of the slowest pair. What happens under $s=1$? What evidence would still be missing before claiming good 256-token task performance?

<details><summary>Hint</summary>

Use $\theta_r=10000^{-2r/8}$, divide by 4 for PI, and calculate a full turn as $2\pi/\theta_r$.

</details>

<details><summary>Solution</summary>

Original frequencies are `[1,0.1,0.01,0.001]`; PI gives `[0.25,0.025,0.0025,0.00025]`. Position 255 maps to 63.75. The slowest original wavelength is $2000\pi\approx6283.1853$ positions. With $s=1$ the frequencies and positions are unchanged; a YaRN-style score multiplier also becomes 1. These calculations establish the geometric transformation. They provide no actual loss, retrieval or reasoning result from a trained model at length 256. That requires a declared evaluation protocol, representative held-out examples and controls for retained short-context behavior.

</details>

### 7. A partial-rotation temperature trap

An attention score has rotary contribution 2 and unrotated contribution 3, before the common head-width scaling. A developer multiplies only the rotary Q/K coordinates by $c=2$ and claims to multiply the entire logit by 4. What actually happens? Give a way to achieve the claimed uniform factor.

<details><summary>Hint</summary>

Each rotary dot product receives two factors of $c$, but the unrotated dot product receives neither.

</details>

<details><summary>Solution</summary>

The original unscaled score is 2+3=5. Scaling only rotary coordinates gives $4\times2+3=11$, not 20. To get a uniform factor 4, multiply the completed dot-product score by 4, or scale **all** query and key coordinates by 2. The usual head-width division can remain separate. This is why matching a context-extension checkpoint requires its attention-scale placement as well as its frequency vector.

</details>

### 8. A time unit changes the meaning

A sensor produces three observations at 0, 1 and 20 seconds. A sinusoidal time encoding uses frequency 0.1 radians/second. Another programmer passes timestamps in milliseconds without changing the frequency. Calculate the phase of the last observation under each program. Give the correct frequency in radians/millisecond and explain why index positions 0, 1, 2 solve a different problem.

<details><summary>Hint</summary>

An angle is timestamp multiplied by frequency. Convert both quantities consistently.

</details>

<details><summary>Solution</summary>

The intended phase is $20\times0.1=2$ radians. Passing 20,000 milliseconds with unchanged 0.1 gives 2000 radians. The corresponding frequency is 0.0001 radians/millisecond, restoring phase 2. Index positions 0, 1, 2 retain event order but discard the unequal temporal gaps. Neither is universally wrong: choose the coordinate whose relation matters, then state its units.

</details>

## What comes next?

You can now locate position at the input, in Q/K geometry or in score biases; distinguish invariance from sensitivity; and verify that a cached computation uses the intended offsets. Continue in the planned sequence to [Grouped-Query Attention and Multi-Query Attention](/learn/path/full-curriculum/grouped-query-attention-gqa-multi-query-attention-mqa?module=deep-learning-fundamentals). That lesson asks how several query heads can share keys and values, and what this changes in cache storage and computation. Its position operations build directly on the conventions here.

## References and another way to learn

Use these selectively according to the part you want to understand better. Formula definitions, our independently calculated fixtures and the real local experiment serve different purposes.

- **A visual refresher before the positional details:** [3Blue1Brown, Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention/) includes its video and illustrated written adaptation. The inspected Q/K, dot-product, softmax and masking explanations help reconnect geometry to value mixing. Its diagrams use query columns rather than this lesson's query rows, which the notes explain. The adjective/noun behavior is explicitly hypothetical; this is a background attention resource, not a RoPE implementation tutorial. The written adaptation was inspected; the video was not independently watched for this packet.
- **A compact comparison with diagrams:** [Sebastian Raschka, RoPE versus absolute positional embeddings](https://sebastianraschka.com/faq/docs/rope-vs-absolute-positional-embeddings.html). The full inspected article traces the input-table versus Q/K-rotation distinction, partial RoPE and cache offsets. Use it after §3 for another concise explanation; follow this lesson's examples for the detailed calculations.
- **A book/notebook route:** [Dive into Deep Learning §11.6](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html), especially 11.6.3–11.6.5. The inspected code, frequency visuals, absolute/relative subsections and exercises provide a second path through sinusoidal encodings. Its CNN/RNN comparison connects to earlier architecture lessons. A binary-counter analogy is about scales, not a claim that floating-point encodings always use fewer physical bits.
- **Learned position tables inside a complete GPT:** [Andrej Karpathy's Let's build GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY), linked from the [creator's Zero to Hero course](https://karpathy.ai/zero-to-hero.html), is a longer code-first learning option. The inspected [nanoGPT implementation](https://github.com/karpathy/nanoGPT/blob/master/model.py) shows `wpe`, strict context checks and addition to token embeddings inside a real decoder. It is companion implementation evidence from the same author, not a claim that the video transcript or its unavailable older repository was inspected. This option teaches the learned-input mechanism and surrounding model, not the RoPE/ALiBi extensions.
- **Original fixed encoding:** [Vaswani et al., Attention Is All You Need, §3.5](https://arxiv.org/pdf/1706.03762). Read for the position formula, original comparison and the motivation behind its relative-shift identity.
- **Learned relations and buckets:** [Shaw et al., §§3.1–3.3](https://aclanthology.org/N18-2074.pdf) defines relation vectors and their score/value roles; [T5's maintained attention source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) makes signed bucket conventions and cache offsets concrete. Code on a moving branch must be version-pinned for reproduction.
- **Rotary geometry:** [Su et al., RoFormer §§2–3](https://arxiv.org/html/2104.09864v5) derives the rotation construction. Read its claims about distance behavior together with the explicit counterexample here and the [PI analysis §2](https://arxiv.org/html/2306.15595v2). A geometric identity and a broad empirical tendency are different statements.
- **Linear bias:** [Press et al., ALiBi §§2–3](https://arxiv.org/html/2108.12409v2) supplies the causal mechanism and comparison context; the [author's implementation](https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py) supplies the non-power-of-two schedule and row-constant optimization.
- **Extending a trained model:** [Position Interpolation](https://arxiv.org/html/2306.15595v2), [YaRN §§3–4 and appendices](https://arxiv.org/html/2309.00071v3), and [the Llama 3 training report §3.4.2](https://arxiv.org/html/2407.21783v3#S3.S4.SS2). Use the inspected methods/training/evaluation sections to separate frequency changes from the complete adaptation recipe. This packet did not repeat their large training runs.
- **Further geometric and evaluation depth:** [XPos/LEX](https://arxiv.org/pdf/2212.10554) and its [TorchScale code](https://github.com/microsoft/torchscale/blob/main/torchscale/component/xpos_relative_position.py) explain reciprocal amplitude factors; [Kazemnejad et al.](https://arxiv.org/pdf/2305.19466) tests length generalization in decoder-only models and motivates the causal NoPE distinction. The full appendix proof campaign is optional research depth, not required to start the next lesson.
- **Implementation reference:** [Transformers RoPE utilities](https://huggingface.co/docs/transformers/main/en/internal/rope_utils) documents named variants and layer-specific configuration. Consult the version matching your model; changing a field is not proof of extended-context quality.
- **Real data and reproduction:** [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), [our provenance](data-provenance.md), [complete training program](author-calculations.py), [independent mechanism calculations](mechanism-calculations.py). These support the actual observations and editable examples in this lesson.
