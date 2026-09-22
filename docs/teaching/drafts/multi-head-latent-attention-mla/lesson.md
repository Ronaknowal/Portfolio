# Multi-Head Latent Attention: store a smaller description, read it exactly

**Explore as you read.** Edit latent vectors/projections, rotation, retained rank, payload dimensions and supported frozen-model prefixes. Show expanded and absorbed paths, commutation residuals, singular-direction effects, bytes/arithmetic and resulting outputs together. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose compression by the function and input directions it preserves; distinguish a low parameter error from low task error.


Suppose several people need different summaries of the same record. We could store every summary. Or we could store a compact description from which each person's summary can be computed. The second option is useful only if the description retains what those people actually need.

Multi-head latent attention, or **MLA**, applies this idea to a Transformer's memory. Each past position keeps a learned compact vector, plus the positional information required by its attention design. Different query heads read that shared representation through different learned maps. An algebraic rearrangement lets them do so without rebuilding every past head's keys and values for each new query.

The [previous GQA/MQA lesson](/learn/path/full-curriculum/grouped-query-attention-gqa-multi-query-attention-mqa?module=deep-learning-fundamentals) reduced the number of distinct stored heads. Here we change the coordinates of the stored information itself. We will carefully distinguish **an exact rearrangement of one MLA model** from **a lossy change to what that model can represent**.

**First pass:** follow §§1–6 and exercises 1–5. You will understand the latent representation, compute both equivalent attention paths, preserve the positional and scaling contracts, and interpret a real cache-compression experiment. Section 7 develops rank, conversion, differentiation and deployment connections; exercises 6–9 extend that reasoning. The core drawings and investigations appear beside the mechanisms they explain.

## 1. What should a decoder remember?

### Refresh the dependency before changing the representation

In [Self-Attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals), a query compares with keys. Softmax turns the legal scores into weights, and a weighted sum of values produces a head's output. A causal query at position t can read positions up to t.

For a fixed causal model, a new future input does not change earlier hidden states. We can therefore retain their keys and values instead of projecting them again at every generation step. This is the KV cache. It is separate at each layer. New queries are transient readers; past keys and values are reusable information.

Ordinary multi-head attention stores separate K/V representations for each head. GQA shares one K/V representation within each group; MQA shares one across all query heads. Their outputs still differ because the queries differ. MLA asks another question: **can all those head-specific representations be generated from a smaller common vector?**

### A latent is a learned coordinate vector

Let an attention input have D coordinates. A learned down-projection maps it to a smaller vector c with $d_c$ coordinates. “Latent” means this is an internal representation. Its coordinates need not have names such as direction, subject or verb, and their values are not probabilities.

Each head has its own key and value up-projection. These maps can turn the same c into different keys and values. Sharing c therefore does not mean that the reconstructed heads are identical. The distinction from GQA is concrete: GQA shares a head representation directly; MLA shares coordinates used to produce head representations.

**Figure — a common description, different readings.** Show one input row becoming c. From c, separate learned maps produce head 0's key/value and head 1's key/value. Place a storage outline around c and the positional branch, not around every expanded output. A separate comparison places a GQA shared key beside these distinct reconstructed keys. This resolves the difference before any large matrix equation.

We are describing the dense, decoupled-rotary design introduced in DeepSeek-V2 and retained in the V3 architecture. Later variants can change the positional construction, sparsity or cache layout. The stable principle is to state exactly what is stored and how the model reads it; a model-family label alone does not determine a cache format.

### Three claims that must stay separate

First, restricting information to a latent representation is an architectural choice. It may affect learnability and quality. Second, once an MLA model is defined, its reconstructed and absorbed computations can be mathematically identical. Third, whether either computation is faster depends on the workload and implementation.

A smaller cache does not prove equal quality or the same factor of speedup. Conversely, a lossy rank reduction does not invalidate the exact algebra of the original model. Much of MLA becomes easier once these three questions are separated.

## 2. Build the representations and keep their shapes visible

### Name the dimensions

We use column vectors in the equations and explicit row/batch axes in code.

| Symbol | Meaning |
|---|---|
|D|Attention input and output row width|
|H|Number of query/output heads|
|$d_c$|Width of the cached content latent|
|$d_q$|Width of the optional query latent|
|$d_k$|One head's content-query/content-key width|
|$d_r$|Rotary-query/shared-rotary-key width, an even number|
|$d_v$|One head's value/output width|
|L, T|Number of legal/stored memory positions and number of new query positions|

The important comparison is often $d_c$ versus **all heads' stored coordinates**, not versus one head's width. In a published V2-style configuration, $d_c=512$ while $d_k=128$. The latent is four times wider than one content head, yet much narrower than 128 heads together. Calling every operation on the latent “smaller” would already be misleading.

### Content keys and values come from the same latent

For attention input $h_s\in\mathbb R^D$ at memory position s, begin with

\[
c_s=W^{DKV}h_s,\qquad
k^C_{s,i}=U_{K,i}c_s,\qquad
v_{s,i}=U_{V,i}c_s.
\]

Here $W^{DKV}$ has shape $[d_c,D]$, $U_{K,i}$ has shape $[d_k,d_c]$ and $U_{V,i}$ has shape $[d_v,d_c]$. The superscript C identifies the content-key branch.

The actual implementation can normalize the down-projected vector before these up-projections. For example, a learned RMSNorm has the form

\[
c=\gamma\odot\frac{z}{\sqrt{\frac1{d_c}\sum_a z_a^2+\epsilon}},
\qquad z=W^{DKV}h.
\]

It rescales by a root-mean-square statistic without subtracting the mean. In that case, **c in our subsequent equations is the normalized vector**, and that is the vector to cache. The original report's practical settings and the [official V3 inference implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) use RMSNorm on the compressed latents. Replacing it with LayerNorm changes the model; omitting it is not automatically guaranteed to cause NaNs, but it changes a checkpoint's defined computation.

This distinction also matters for algebra. We can move linear up-projections across sums of c. We cannot treat RMSNorm as a fixed matrix and move it across an arbitrary down-projection or weighted sum.

### Queries have their own path

A query can be produced directly from the current input, or through a separate latent:

\[
c^Q_t=\operatorname{RMSNorm}(W^{DQ}h_t),\qquad
q^C_{t,i}=U_{Q,i}c^Q_t.
\]

The query latent is not the KV cache. It is computed for current queries; its factorization changes parameters, intermediate activations and computation. Whether it reduces peak training memory depends on which expanded tensors are saved or recomputed. A factorized projection does not magically remove every larger activation from autograd.

The [published V2 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json) uses $d_q=1536$ and $d_c=512$: the former is three times the latter. They are separate choices, not a universal ratio. Some implementations allow a direct query path when no query compression is desired.

## 3. Preserve position without rebuilding every key

### Why a usual rotary key obstructs a fixed absorption

Recall [Positional Encodings](/learn/path/full-curriculum/positional-encodings-sinusoidal-learned-rope-alibi?module=deep-learning-fundamentals). RoPE rotates coordinate pairs according to their logical position. Write that rotation as $R_s$. A normally rotated reconstructed key would be $R_sU_Kc_s$, giving score contribution

\[
(R_tq)^T(R_sU_Kc_s)
=q^T R_t^T R_s U_Kc_s.
\]

To compare a single effective query with every c, we would need to remove the key-position dependence from the factor beside c. In general, $R_t^TR_s$ depends on s. There is no one key-position-independent query transform that absorbs all these rotations through an arbitrary $U_K$.

A small example makes the obstruction visible. Let

\[
U_K=\begin{bmatrix}2&0\\0&1\end{bmatrix},\qquad
R=\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
\]

Stretching the x coordinate and then rotating does not equal rotating and then stretching it. Their difference is

\[
RU_K-U_KR=\begin{bmatrix}0&1\\1&0\end{bmatrix}.
\]

With current query $[1,0]^T$, a key with no rotation requires effective latent query $[2,0]^T$; a key with this quarter-turn rotation requires $[0,-1]^T$. The transformation would depend on which past key we are reading.

This is a statement about the general fixed-matrix rearrangement. Special structured maps or different positional architectures can have other identities. Applying RoPE to reconstructed keys is still a valid attention computation; it simply loses this particular easy absorption unless additional structure is supplied.

**Investigation — move a rotation through a projection.** Edit a two-dimensional projection and rotation angle. Compare “project then rotate” with “rotate then project” using the same vector. An identity/isotropic projection supplies a commuting control; an unequal stretch produces a visible difference. Label the order of operations and the actual transformed vector, rather than displaying a generic warning that rotations never commute.

### A separate rotary branch

The V2/V3-style solution keeps content keys unrotated and adds a small rotary branch:

\[
k^R_s=R_s W^{KR}h_s,\qquad
q^R_{t,i}=R_t U_{QR,i}c^Q_t.
\]

There is **one shared rotary key per memory position**, while query heads have distinct rotary queries. Cache $c_s$ and $k^R_s$. The score for head i is

\[
\ell_{t,s,i}=
\frac{(q^C_{t,i})^T U_{K,i}c_s+(q^R_{t,i})^Tk^R_s}
{\sqrt{d_k+d_r}}.
\]

Apply the causal/padding mask, then softmax over the legal memory positions. The resulting weights mix $U_{V,i}c_s$.

The rotary branch is not a stored integer position or a content-free tag: it is a projected input vector rotated using a position. Likewise, “NoPE/content” means that this particular branch receives no direct rotary transform. Its input hidden state may already contain positional information from earlier layers. Do not interpret the name as proof that every latent coordinate contains only semantics and no information about order.

Decoupling preserves a specified positional mechanism. It does not by itself guarantee good behavior at arbitrary unseen lengths. Context extension still depends on frequency/scaling choices, training and evaluation, as the preceding positional lesson explained. A causal mask itself also carries an ordering constraint; removing an explicit positional branch does not justify the blanket statement that a causal network has no way to distinguish order.

**Figure — two score contributions meet.** One rail carries c through a head's content comparison; another carries the rotated shared key. Show two signed dot products added **before one normalization and one softmax**. They are not two independently normalized attention distributions. The storage bracket surrounds both retained rails.

## 4. Read the latent directly: the exact rearrangement

### Absorb the key up-projection into the query

For any fixed head and query,

\[
(q^C)^T U_K c=(U_K^Tq^C)^Tc.
\]

Define $\widetilde q=U_K^Tq^C$, a vector with $d_c$ coordinates. The content score can now be computed directly against the cached c. No historical content key needs to be expanded for that comparison. Keep the separate rotary dot product unchanged.

This is ordinary associativity of matrix multiplication. It does not discard singular directions or approximate a probability. In a row-vector implementation, the same operation is `effective_query = content_query @ key_up`, with the up-projection stored as output-by-input.

### Mix latents before expanding values

Once the weights $a_s$ are known,

\[
o_i=\sum_s a_{s,i}U_{V,i}c_s
=U_{V,i}\left(\sum_s a_{s,i}c_s\right).
\]

Compute a weighted latent $z_i=\sum_s a_{s,i}c_s$, then expand it once for this query/head. Each head still has its own weights and therefore its own z. The shared cache is not one shared output.

If $W_{O,i}$ is the output-map slice for head i, the residual contribution is

\[
y=\sum_i W_{O,i}U_{V,i}z_i.
\]

For fixed weights, the product $W_{O,i}U_{V,i}$ can be precomputed. It can also be left factorized. A larger precomputed matrix may cost more storage and arithmetic than two smaller multiplications, so “can absorb” is not the same as “every implementation must premerge at export.” The official reference computes useful contractions at runtime.

### Keep the original score scale

The effective query has width $d_c$, but its score is the original content dot product expressed differently. The divisor remains $\sqrt{d_k+d_r}$, or the checkpoint's explicitly modified scale.

A generic attention call given concatenated latent and rotary features may default to $1/\sqrt{d_c+d_r}$. When $d_c\ne d_k$, that changes the logits and their concentration. It can produce valid shapes and plausible outputs while implementing the wrong model. For our real example, the intended divisor is $\sqrt6$; the accidental latent-width divisor is $\sqrt{10}$.

**Investigation — two paths, one answer.** An expanded view forms per-head K/V; an absorbed view moves the two linear maps around the dot product and weighted sum. Edit actual vectors or map entries, predict agreement, and show immediately both computations. Include an intentional wrong-scale switch and a nonlinear-value-map counterexample. Equivalence should follow the algebra, not a hard-coded “all views agree” label.

### A complete hand example

Use two heads, $d_c=d_k=d_v=d_r=2$, one query at position 2 and memory positions 0,1,2. These are chosen arithmetic inputs, not trained activations. For easy hand rotations use a quarter turn per position; the trained study later uses ordinary RoPE instead.

| Memory position | Cached c | Raw rotary key | Rotated rotary key |
|---:|---|---|---|
|0|[1,0]|[1,0]|[1,0]|
|1|[0,1]|[1,0]|[0,1]|
|2|[1,1]|[1,0]|[−1,0]|

Content queries are $q^C_0=[1,0]^T$ and $q^C_1=[1,1]^T$. Raw rotary queries `[1,0]` and `[0,1]` become `[−1,0]` and `[0,−1]` at position 2. Use

\[
U_{K,0}=\begin{bmatrix}1&0\\0&2\end{bmatrix},\quad
U_{K,1}=\begin{bmatrix}1&1\\1&-1\end{bmatrix},\quad
U_{V,0}=I,\quad
U_{V,1}=\begin{bmatrix}2&0\\1&-1\end{bmatrix}.
\]

Head 0's effective query is `[1,0]`. Its content scores are `[1,0,1]`; rotary scores are `[−1,0,1]`. Add and divide by 2, obtaining logits `[0,0,1]`. The weights are

\[
\frac{[1,1,e]}{2+e}\approx[0.211942,0.211942,0.576117].
\]

The weighted latent is `[0.788058,0.788058]`. Because $U_{V,0}=I$, that is also head 0's value output.

Head 1's effective content query is `[2,0]`, giving content scores `[2,0,2]`. The rotary scores are `[0,−1,0]`, so scaled logits are `[1,−0.5,1]`. Its weights are approximately `[0.449816,0.100368,0.449816]`. Its weighted latent is `[0.899632,0.550184]`, which its value map transforms into `[1.799265,0.349449]`.

The two heads read the same latent records but have different weights and different value outputs. To complete the example, combine them with

\[
W_O=\begin{bmatrix}1&0&0.5&0\\0&1&0&0.5\end{bmatrix}.
\]

The attention contribution is approximately `[1.687691,0.962783]`. This is an attention output vector; a full Transformer block still applies its residual and feedforward computation before a task prediction.

### Changing coordinates is different from losing coordinates

Let S be any invertible change of latent basis. Store $c'=Sc$, and replace each up-projection U by $U'=US^{-1}$. Then $U'c'=Uc$, so every reconstructed key/value and attention result is unchanged. Our hand example checks $S=\operatorname{diag}(2,0.5)$ exactly.

This is why a latent coordinate does not automatically have a unique semantic identity. Its scale and basis can change with compensating maps. The argument applies to the defined cached vector after any normalization; it does not say that an arbitrary basis change commutes through RMSNorm.

Discarding coordinates is different: a noninvertible projection has no inverse that recovers all possible c. The actual effect depends on what was discarded and what the model needed. Section 6 measures that distinction on real observations.

## 5. Implement it, then count the right things

### A transparent executable reference

This complete NumPy program implements both paths for the hand example. K/V are reconstructed only in the expanded branch. The rotary vectors are already rotated, making the matrix reassociation visible without hiding it inside a larger model.

```python
import numpy as np


def mla_read(query, latent, key_up, value_up, query_rope, key_rope,
             scale, expanded=False):
    # query [H,dk], latent [L,dc], up-maps [H,output_width,dc].
    if expanded:
        keys = np.einsum("lc,hpc->hlp", latent, key_up)
        values = np.einsum("lc,hvc->hlv", latent, value_up)
        content = np.einsum("hp,hlp->hl", query, keys)
    else:
        effective_query = np.einsum("hp,hpc->hc", query, key_up)
        content = effective_query @ latent.T
    logits = (content + query_rope @ key_rope.T) * scale
    # All three keys are legal for this final-position hand example.
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    if expanded:
        output = np.einsum("hl,hlv->hv", weights, values)
    else:
        mixed_latent = weights @ latent
        output = np.einsum("hc,hvc->hv", mixed_latent, value_up)
    return output, weights


q = np.array([[1., 0.], [1., 1.]])
c = np.array([[1., 0.], [0., 1.], [1., 1.]])
uk = np.array([[[1., 0.], [0., 2.]], [[1., 1.], [1., -1.]]])
uv = np.array([np.eye(2), [[2., 0.], [1., -1.]]])
qr = np.array([[-1., 0.], [0., -1.]])
kr = np.array([[1., 0.], [0., 1.], [-1., 0.]])
wo = np.array([[1., 0., .5, 0.], [0., 1., 0., .5]])
expanded, first_weights = mla_read(q, c, uk, uv, qr, kr, .5, True)
absorbed, second_weights = mla_read(q, c, uk, uv, qr, kr, .5, False)
print("head outputs:", np.round(absorbed, 6))
print("final output:", np.round(wo @ absorbed.reshape(-1), 6))
print("same output:", np.allclose(expanded, absorbed, atol=1e-12))
print("same weights:", np.allclose(first_weights, second_weights, atol=1e-12))
```

It produces the two head outputs and final output calculated above, and both checks print `True`. The [complete independent fixture program](mechanism-calculations.py) additionally supplies editable latent/map inputs, rotations, a consistent basis change, broken-position and nonlinear-map contrasts, and exact byte/work counts. The full neural program in §6 adds normalized learned projections, actual causal masks, incremental cache state and task predictions.

### What exactly is in the compact cache?

For B requests, N uniform layers, L occupied positions and s bytes per stored number, the unquantized compact payload is

\[
\boxed{\operatorname{bytes}=BNL(d_c+d_r)s.}
\]

There is one c and one shared rotated key per position/layer. Do not multiply c by H or count it twice because it participates in both the key and value calculations. Queries, score tiles, weighted latents and output vectors are transient work, not additional copies of the persistent history.

At fixed $d_c,d_r$, increasing H leaves this formula unchanged. It still increases the number of comparisons and head outputs, as well as relevant parameters and transient tensors. This is a storage property, not free additional heads.

Different baselines require different denominators. The following are **calculated representations**, all using H=128, content width 128, value width 128, rotary width 64 and latent width 512 where applicable:

| Representation | Stored numbers per token/layer | Meaning |
|---|---:|---|
|Plain MHA, 128-wide keys and values|32,768|RoPE can rotate within its existing key width; this is a different parameterization|
|GQA with 8 such KV heads|2,048|Shared heads with those stated widths|
|MQA with one such KV head|256|Smallest payload in this particular table|
|Same MLA function, literal expanded K/V cache|40,960|128 heads each store a 192-wide key and 128-wide value|
|Same MLA function, expanded content plus one shared rotary key|32,832|Avoids duplicating the 64-wide rotary key across heads|
|Compact MLA|576|One 512-wide latent plus one 64-wide rotary key|

The compact payload is about 56.89 times smaller than the first plain-MHA count and 71.11 times smaller than the literal expanded representation of the **same MLA function**. These answer different comparisons. It is 2.25 times the MQA payload in this table, not smaller than every alternative. A memory table alone contains no quality ranking.

For 60 layers, 32,768 positions, one request and two-byte numbers, compact MLA needs 2,264,924,160 bytes, or 2.109375 GiB. The plain-MHA row needs 120 GiB; GQA-8 needs 7.5 GiB. A GiB is $2^{30}$ bytes. These are payload calculations, not measured allocator memory or statements that a whole model fits on a device.

The V2 report's headline 93.3% cache reduction compares **different complete models**, DeepSeek-V2 and the earlier DeepSeek 67B. It is not the rounded value of our 56.89-fold, same-head-width calculation. Keep the stated comparison attached to any percentage. [Original report](https://arxiv.org/html/2405.04434v5).

Actual allocation may include page rounding, reserved slots, position metadata, quantization scales, distributed replicas and cache layouts that materialize expanded heads. The [official V3 reference](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) visibly allocates different buffers for its `naive` and absorbed branches. An architecture supports a compact representation; an implementation must actually store it to realize that payload.

**Investigation — construct a cache record.** Build a record from its latent and rotary fields, then multiply by occupied positions, layers and requests. Edit the dimensions and storage precision; compare the explicitly labelled representations above. Keep a separate transient-work panel. A plotted byte count must never become an invented latency or model-quality curve.

### Why less data movement can involve more arithmetic

At fixed H, T, L and widths, the expanded attention core uses approximately

\[
2BHTL(d_k+d_r+d_v)
\]

operations for scores and weighted values, counting a multiply-add as two. The absorbed core uses approximately

\[
2BHTL(2d_c+d_r).
\]

It compares with a $d_c$-wide latent and also sums a $d_c$-wide latent. When $d_c$ is larger than $d_k$ and $d_v$, these core arithmetic terms increase. For the widths in the table, the ratio is $1088/320=3.4$.

For one query and 32,768 memory positions, our formula gives about 2.684 billion expanded-core operations versus 9.127 billion absorbed-core operations. The absorbed representation can nevertheless reduce persistent storage and repeated memory traffic by sharing the latent across heads. Effective hardware reuse, tiling, bandwidth and occupancy determine how these facts translate into time.

There is another comparison: rebuilding **all** historical expanded K/V from a compact cache for every new query costs roughly

\[
2BLH d_c(d_k+d_v)
\]

additional operations. Absorption avoids that repeated reconstruction. Instead, it transforms each new query and the resulting weighted latent once per head. In our numerical configuration, full-prefix reconstruction is about $1.10\times10^{12}$ operations; each of those two per-query transformations is about 16.78 million. Avoiding reconstruction is useful, but it is not evidence that absorbed dense attention has 57 times fewer operations than an already-expanded-cache attention core.

Prefill has many known query rows; decode often has one new row per request. A compute-friendly expanded path can be attractive during prefill, with bounded temporary reconstruction, while a compact path can suit decode. [vLLM's MLA implementation notes](https://docs.vllm.ai/en/v0.20.0/api/vllm/model_executor/layers/attention/mla_attention/) explain both forms and chunked prefill. Use the actual dimensions: $T/L$ is near 1 for an uncached full prefill and small for one-query decode. Do not reproduce a reversed small/large ratio from a documentation sentence.

### Parameter counts are also dimension-dependent

Ignoring biases and normalization parameters, the factorized design described here has

\[
D(d_c+d_q+d_r)
+Hd_c(d_k+d_v)
+Hd_q(d_k+d_r)
+DHd_v
\]

projection parameters. The four terms account for input/down maps, KV up maps, query up maps and the output map. A direct-query design changes the query terms. Learned latent normalization adds its scales.

There is no universal “only a few percent more than MHA” answer. In particular, the convenient MHA expression $4D^2$ assumes the usual total head widths equal D. A configuration with $Hd_k\ne D$ does not satisfy that assumption. Count the actual maps before comparing architectures or optimizer-state memory.

### Use the absorbed representation with an ordinary attention primitive

The [complete SDPA bridge](mla_sdpa_bridge.py) supplies a practical tensor API route. Concatenate the absorbed query `q_content @ U_K` and positioned rotary query; concatenate each cached latent and positioned shared rotary key. Use the cached latent as the value. SDPA then returns a latent mixture, which each head's value-up map turns into its output.

Pass `scale=1/sqrt(P+R)` explicitly. SDPA's default would use the concatenated width C+R, changing the function whenever C differs from P. `enable_gqa=True` makes all query heads read the one-head shared memory. The boolean mask means allowed and uses the logical positions; dropout is zero. The complete program compares direct/API output, all six input/factor gradients and one equal update. Run `python mla_sdpa_bridge.py` with PyTorch. A bounded author probe on Torch 2.14.0 CPU gave output discrepancy 4.44e-16 and largest gradient discrepancy 3.33e-16.

This uses the [PyTorch SDPA contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html); it is not a specialized DeepSeek kernel. Device/backend support and physical allocation for grouped queries with unequal value width need separate measurement. The original tensor path remains a useful fallback. The whole model also has projections, latent normalization, residuals, positions and cache identity; those remain explicit in `LatentForecaster`, rather than being inferred from operator agreement.

**Change the constraint:** use latent width 7 with P=3 and R=2. Change the latent and both up-projection shapes together. **Hint:** absorption changes representation width, not the intended temperature. **Solution:** preserve `1/sqrt(5)` in both routes; the equality checks should still pass. Deliberately using `1/sqrt(9)` in only one route creates a shape-valid semantic mismatch.

## 6. A real model: exact execution and a lossy intervention

### Predict the next point of an observed movement

The task is causal coordinate forecasting using [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), an openly licensed set of normalized hand-centroid trajectories. Each row contains 45 ordered x/y pairs. At positions 0–43, predict the next coordinate pair at positions 1–44, without reading future points.

The source has 360 rows, including 30 extra copies of exact trajectories. After checking duplicate labels agree, retain the first occurrence of each distinct trajectory. Reuse the declared seed-73 classwise row split: 220 training, 50 validation and 60 test trajectories. Keep entire trajectories together. Labels determine stratification only; they are not forecast inputs or targets.

Use fixed $2x-1$ scaling and convert RMSE back to the original coordinate unit. The source does not provide reliable per-row performer/session IDs, so this is a row-level diagnostic, not a test on a new signer. Its coordinates have already been processed by the dataset creators; the study does not validate an online camera pipeline. [Full source/protocol provenance](data-provenance.md) accompanies the original offline files.

Persistence predicts the latest point unchanged. A six-parameter affine map predicts the next two coordinates from the current two, fitted only on training transitions. Smooth trajectories make these meaningful baselines.

### The complete small architecture and declared training

Our model has a 2-to-24 stem and one pre-normalized Transformer block. There are four heads with content width 4, rotary width 2 and value width 4. The KV latent has eight coordinates; the query latent has twelve. Both use RMSNorm with epsilon $10^{-6}$ and learned scales. Residual/final LayerNorm uses epsilon $10^{-5}$. The FFN has width 48 with GELU; the final prediction has two unconstrained coordinates.

Attention maps have no biases. Stem, FFN and forecast maps have biases. Ordinary adjacent-pair RoPE uses base 10000, with the correct logical positions; there is no YaRN or dropout. This is a small dense model designed to expose MLA's mechanism, not a reproduction of DeepSeek's expert network.

Train once with seed 131 for 200 full-batch Adam updates at learning rate 0.003, no weight decay. Choose the lowest validation MSE, taking the earliest exact tie. The selected checkpoint is update 200, the budget boundary. That is the recorded selection, not a claim that optimization has converged. The model has 4,118 trainable parameters.

### Check the same function before changing its information

Use the **same weights and inputs** for reconstructed and absorbed paths. In a float64 check on a small real prefix, their complete model outputs differ by at most $3.89\times10^{-16}$. Backpropagating the same squared-output objective through both paths gives corresponding gradients for **every parameter**, with maximum difference $7.11\times10^{-15}$.

This directly contradicts the idea that absorbed attention must have unstable gradients because its normalization statistics somehow change. Both paths normalize the same vector and define the same differentiable function. Floating-point order and a kernel's implementation can affect numerical behavior; they do not change the algebraic identity. An exported, detached product must of course be refreshed if its underlying trainable weights change.

For the 32-point visible prefix, ordinary float32 expanded/absorbed forecasts agree within $1.79\times10^{-7}$ in transformed coordinates. Processing the same prefix one point at a time with the compact cache agrees with the full causal pass within $3.58\times10^{-7}$. This checks projection, rotary positions, masks, residual computation and forecasts—not just concatenating already computed arrays.

### Deliberately remove four latent directions

Now ask a different question. Stack the trained content-key and value up-projections into a joint output-by-latent matrix M. Its singular value decomposition identifies orthogonal latent directions and their contribution to those linear maps. Let P contain the four right singular vectors with the largest singular values.

For each new input, first compute the original normalized eight-coordinate c, then cache only

\[
c'=P^Tc\in\mathbb R^4.
\]

Replace each up-projection U by UP. The reconstructed representation is now $UPP^Tc$, which discards the other four directions. The rotary key and query paths stay unchanged. There is **no additional training** and no test-driven choice of rank.

This intervention reduces persistent content coordinates. It still computes the original eight-coordinate RMSNorm before projection; it is not a separately trained four-coordinate latent architecture. The distinction matters both for arithmetic and for interpreting the result.

The discarded squared singular values sum to about 2.862722. That matches the squared Frobenius error of the rank-4 joint up-projection. This is the optimum for that specified **parameter-space reconstruction objective**, not a guarantee of minimal attention error or forecast loss.

Use all eight singular directions as a control. The complete orthogonal basis changes coordinates but loses none, so the outputs should agree. Across the validation set, the actual maximum difference is $3.58\times10^{-7}$ in transformed coordinates. We have separately checked a basis change and a truncation, instead of calling both “compression.”

### Actual outcomes

RMSE is per original normalized coordinate over all 44 output slots of each trajectory. The held-out set contains 60 whole trajectories.

| Method | Validation RMSE | Test RMSE | Float32 compact payload for one 32-point prefix |
|---|---:|---:|---:|
|Selected MLA, 8 content coordinates + 2 rotary|0.025576|0.024946|1,280 bytes|
|Same model, rank-4 cache intervention + 2 rotary|0.095481|0.091934|768 bytes|
|Persistence baseline|—|0.026458|Not an MLA cache|
|Training-fitted affine baseline|—|0.026222|Not an MLA cache|

The selected small MLA is only modestly better than these simple baselines. The unadapted rank reduction substantially damages its forecasts. Both findings belong in the lesson. A lower parameter reconstruction error than another factorization would not prove good task performance; an important direction can have a modest singular value.

The table does not compare this model directly with preceding lessons' differently shaped attention models. It also does not establish that a model trained from scratch at rank four could not learn well, or that further adaptation would never help. Those are different experiments requiring their own declared protocols.

### Inspect an actual cached prediction

The visible input was chosen before training: source row 77, with its first 32 points observed. The actual next point is approximately `[0.593810,0.250000]`.

The full model predicts `[0.599737,0.263627]`; the rank-4 intervention predicts `[0.613758,0.260927]`. The compact tensors have shapes `[1,32,8]` and `[1,32,2]` in the first case, versus `[1,32,4]` and `[1,32,2]` in the second. The two cache fields have different meanings; the rotary field is not a second copy of the latent.

Reflect observed point 23's x coordinate with $x\mapsto1-x$ and recompute the prefix. The full model's final prediction becomes `[0.600519,0.263407]`. All outputs before the edited point remain **exactly unchanged** in the checked computation. A changed past observation can affect subsequent forecasts, so the cache for the old prefix cannot silently be reused.

Shift all logical positions by 100 under the same ordinary RoPE rule. Full-model outputs agree within $1.79\times10^{-7}$. Moving only queries while retaining old rotated keys is a different operation and generally changes scores. A consistent global position transformation can be a valid equivalence; “a cache can never be reused at another absolute position” is too strong. Reuse must preserve or correctly transform all relevant state and positional conventions.

The wrong scale $1/\sqrt{10}$ changes this full model's final prediction only slightly, to `[0.599745,0.263674]`. The small effect is still a changed function. Do not magnify it into a dramatic failure. For the rank-4 intervention, its latent width happens to equal the content-head width, so the accidental default equals the intended scale and gives a null result. A test that covers only equal widths can therefore miss the bug.

**Investigation — inspect the compressed history.** Follow the observed trajectory through linked latent coordinates, rotary keys and selected-head score contributions. Edit a coordinate or latent/map entry and compare expanded, absorbed, full-basis and rank-reduced outputs live. Display the observed next point separately as a reference; it never enters the forecast input. A head's weights explain its local read, not the complete causal origin of the final forecast.

### Reproduce the entire study

Place [the full CPU program](author-calculations.py), [original data](movement_libras.data) and [original metadata](movement_libras.names) together. With Python, NumPy and PyTorch installed, run:

```bash
python author-calculations.py
```

The recorded environment is Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu, with one CPU thread and deterministic algorithms. The complete program implements preprocessing, duplicate-aware splitting, both baselines, training/selection, both MLA paths, normalized cache truncation, full-parameter derivative checks and actual forecast/cache interventions. It requires no network or GPU. [Recorded results](author-results.json) and [saved weights/traces](forecast-model.json) make the observations inspectable; different numerical environments can change last digits.

Read `LatentForecaster.forward` in the same order as the diagram: normalize the input state, compute content/query latents and rotary branches, append compact fields and logical IDs, calculate the two score terms with the original scale, mask and normalize, mix latents, expand the current head outputs, then complete the residual/FFN/task path.

The website investigation needs only one small model and the selected input. Full author evidence is an optional download. It should not train models, evaluate the entire corpus or eagerly load every experiment on page opening.

## 7. Deeper connections and practical boundaries

### What is low rank, and what is not?

Before normalization, stacking the content-key and value maps gives

\[
\begin{bmatrix}k^C\\v\end{bmatrix}
=\underbrace{\begin{bmatrix}U_K\\U_V\end{bmatrix}}_M
W^{DKV}h.
\]

The effective linear map has rank at most $d_c$. With a nonlinear normalization before the up-map, the entire h-to-output map is no longer one fixed linear matrix. Nevertheless, the stacked outputs still lie in the column space of M, whose dimension is at most $d_c$. These are related but different rank statements.

This restriction explains why a smaller latent can lose useful distinctions. If two cached content vectors are identical, all their reconstructed content keys and values are identical. Their rotary keys can still differ, and the full model has other paths, so do not turn this into a claim that the complete inputs or predictions must be indistinguishable in every setting.

For one head, the unnormalized content-score matrix factors through the latent coordinates. But row-softmax is nonlinear and does not preserve matrix rank. For example, the three-by-three logit matrix with entries $\ell_{ij}=ij$ for i,j in `{0,1,2}` has rank one. Its row-softmax matrix has rank three; the independently computed determinant is about 0.024431.

MLA therefore does not make the full softmax attention matrix low rank merely by using low-rank projection parameters. Dense full-sequence attention still compares query/key positions; its usual score work grows quadratically with sequence length. The [next Sparse and Linear Attention lesson](/learn/path/full-curriculum/sparse-linear-attention-variants?module=deep-learning-fundamentals) changes which comparisons happen or which operator is computed. Latent-coordinate compression and sequence-level approximation are separate ideas.

### Why a good matrix approximation can be a poor predictor

For a matrix M with singular values $\sigma_1\ge\cdots\ge\sigma_r$, a rank-k truncated SVD minimizes the squared Frobenius reconstruction error, with error $\sum_{j>k}\sigma_j^2$. One way to understand this is to use orthogonal singular coordinates: each retained direction preserves one independent squared-energy contribution, so retaining the largest contributions minimizes the discarded sum. The full theorem covers all rank-k matrices, not only a particular coordinate deletion.

This objective weights matrix entries uniformly. Actual inputs need not visit every direction uniformly, and task outputs need not value all errors equally. Take $M=\operatorname{diag}(10,1)$. Its best rank-one Frobenius approximation is $\operatorname{diag}(10,0)$, with squared error 1. For input `[0,10]`, however, the original output is `[0,10]` and the approximation gives `[0,0]`. The discarded direction contains the entire useful signal for this input.

In attention, the consequences can be even less direct: key errors alter normalized weights, value errors alter what those weights mix, and later layers transform the result. A data-aware reconstruction objective might weight directions using input covariance; a task-aware adaptation can optimize prediction loss. Neither is automatically equivalent to minimizing projection-weight distance. Our rank-4 result is a concrete example of the distinction.

**Figure — discarded energy versus discarded information.** Place the two singular directions beside a real input vector. A parameter-error bar reports one objective; an output-error vector reports the effect on that input. Let the learner edit the input direction in the current live view. Inputs along the retained axis supply an exact null; inputs along the discarded axis expose a potentially large effect.

### Converting an existing checkpoint is possible, but not a configuration edit

An arbitrary trained MHA/GQA checkpoint does not already have the required joint latent factorization and shared rotary design. Changing a head-count or rank field cannot create compatible weights. It is also incorrect to claim that useful conversion is impossible.

[MHA2MLA, published at ACL 2025](https://aclanthology.org/2025.acl-long.1597.pdf), investigates partial-RoPE adaptation and joint low-rank factorization, followed by continued training. Its methods explicitly assess retained rotary subspaces and factorize the relevant key/value maps. This is a studied conversion route with measured tradeoffs, not proof that every checkpoint converts losslessly. A converted variant's retained rotary components and cache accounting must be read from that variant, rather than assumed identical to the original shared-key V2 design.

A bounded conceptual initialization is straightforward when the relevant K/V maps are linear and no incompatible positional operation intervenes. Stack their output-by-input weights into M, compute a truncated SVD $M\approx U_k\Sigma_kV_k^T$, choose down-map $\Sigma_k^{1/2}V_k^T$ and joint up-map $U_k\Sigma_k^{1/2}$, then split the up-map into key and value parts. These factors approximate the original parameter matrix. They do not automatically preserve task outputs, compensate for changed rotary dimensions or supply the proper normalization/adaptation recipe.

Our small study starts from a model already trained with MLA and reduces its post-normalization cache. It is not a reproduction of MHA2MLA's full-to-partial-RoPE checkpoint conversion. The original normalization remains part of the declared model, so the two experiments should not be conflated.

### What can move through the value sum?

The value rearrangement requires a map shared across memory positions that is linear in the summed latent. An affine map $Uc+b$ can also be handled: if attention weights sum to one, its weighted output is $U\sum_s a_sc_s+b$. With attention dropout or other unnormalized weights, the bias contributes $b\sum_s a_s$, which must be accounted for explicitly.

A nonlinear map generally cannot move outside the sum. With latents −1 and 1 and weights one-half each, mixing their ReLU values gives 0.5, while applying ReLU to their mixed latent gives 0. Moving the nonlinearity has changed the function.

Head-dependent linear maps are fine because each head has its own weighted latent. Position-dependent maps generally cannot be pulled outside a sum over positions as one fixed map. Data-dependent routing or quantization also needs its own exact contract. This reasoning is more useful than memorizing that every operation called an “up-projection” is absorbable.

### Backward computation and numerical precision

For an output $o=Uz$, the gradient with respect to U is the outer product of the upstream output derivative with z; the gradient with respect to z is $U^T$ times that derivative. For $z=\sum_s a_sc_s$, c receives both a direct value-mixture contribution and an indirect contribution through attention weights when c also influences keys.

The same dependencies exist in the expanded graph. The chain rule combines them in a different order; it does not create a different intended derivative. Our full-parameter check compares those derivatives rather than relying on a loss curve as evidence that attention is correct.

Floating-point multiplication and summation are not perfectly associative. Different intermediate precision, softmax accumulation, kernel reductions or quantization can produce different numerical errors. Compare outputs and gradients with a clear tolerance and inspect which operation changes. Exact algebra does not promise bitwise equality across GPU kernels.

Nor does “FP8 model” mean every tensor, operation and cache field is FP8. The [V3 mixed-precision report](https://arxiv.org/html/2412.19437v2#S3.SS3) retains higher precision for selected operations, including attention and normalization. Cache storage is a separate serving choice. A currently documented V3.2 sparse FlashMLA format has 512 one-byte latent values, 16 scale bytes and 128 bytes for 64 BF16 rotary coordinates: **656 bytes per token**, not simply 576 or 512. This is that specific format, not a universal MLA layout. [Official FlashMLA cache-format documentation](https://github.com/deepseek-ai/FlashMLA#mla-decoding).

### Read empirical claims with their actual comparison

The original attention ablations are in **Appendix D**, not the overall model comparison in Table 2. Table 9's small-MoE comparison reports MMLU 48.7 versus 50.0 for MHA/MLA, but C-Eval 51.6 versus 50.9. Its large-MoE comparison uses a different training scale. These are useful scoped results; even this one table does not justify “MLA never loses quality.” [DeepSeek-V2 attention ablations](https://arxiv.org/html/2405.04434v5#A4).

A quality-versus-cache plot needs comparable models, tasks, training budgets and measurements. Combining an MMLU number from one model family with a task-average score from another paper does not create a measured frontier. Our figures instead show exact formula-derived storage, actual operator differences and actual local forecasting outcomes, with their evidence types labelled.

The [V2](https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json) and [V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json) configurations share several MLA widths, but differ in model width, layer count and other settings. A supported buffer length in a configuration also need not equal a demonstrated effective context length on every task. Use the full positional configuration and training/evaluation evidence before extending an inference length.

### Applications and interactions worth recognizing

**Cached cross-attention.** If encoder outputs are fixed while a decoder generates, their latent content and positional representations can be cached and reused. Source positions and decoder positions need an appropriate cross-attention convention; blindly reusing the self-attention rotary relation may be inappropriate. If the source changes, its cached representation changes. The later cross-attention topic owns the full source/target alignment design.

**Movement, sensor and event streams.** Our forecast demonstrates an actual non-language use. Compact per-observation state can help when many positions are retained. A dense cache still grows with stream length; a window, reset or another architecture is needed for bounded indefinite memory. Sensor calibration or corrected historical observations also require invalidation policies.

**MoE and sparse attention.** Expert routing usually changes the feedforward part of a Transformer; MLA changes attention representation. They can coexist without expert count multiplying every attention-cache field. Likewise, selecting fewer legal/retrieved positions can coexist with a latent representation for those positions. The current [FlashMLA repository](https://github.com/deepseek-ai/FlashMLA) distinguishes dense and sparse kernels and version-specific formats. The next variants lesson and [MoE lesson](/learn/path/full-curriculum/mixture-of-experts-transformers-moe?module=deep-learning-fundamentals) develop these separate mechanisms.

**Encoders and short inputs.** The operator can be used without an autoregressive cache. Its main persistent-history saving then may not apply, but representation/parameter choices can still be studied. There is no mathematical ban on encoder use or universal one-billion-parameter cutoff. Compare the actual task, temporary activation requirements and implementation support.

### Choose the implementation for the workload

Start with an explicit reference for masks, shapes, scale, normalization, rotary layout and cache lifetime. Then select an implementation that supports those dimensions and dtypes. Check whether the kernel expects expanded heads or an MQA-like latent layout; that API label can describe how the same MLA computation is executed, rather than a different trained architecture.

For performance, measure prefill and decode separately at stated lengths, batches, dtypes and hardware, including cache allocation and synchronization. Consider total weights, temporary workspace, communication and scheduling. A cache ratio alone cannot explain a commercial API price or establish how many GPUs serve a complete model under a latency target.

The published [V3 deployment discussion](https://arxiv.org/html/2412.19437v2#S3.SS4) describes substantial distributed, workload-specific arrangements. It is not evidence for a universal single-server fit rule. Current kernels may support formats and variants newer than this lesson's V2/V3 core; pin the actual source/version when reproducing them. Our CPU programs teach correctness and the effects of a declared intervention, with no claim of a production latency crossover.

## 8. Practice: preserve a computation or change it deliberately

Use a fresh calculation or prediction before opening the optional help. Exercises 1–5 cover the first-pass route; 6–9 extend the deeper branches.

### 1. Read one head through a different latent basis

A head has content query `[1,2]`, latent records `[1,0]` and `[0,1]`, key up-map `[[2,0],[0,1]]`, value up-map equal to the identity, and no positional contribution. The defined scale is $1/\sqrt2$. Compute its effective query, weights and output. Then use basis change $S=\operatorname{diag}(2,0.5)$. What must happen to the latent records and up-maps to preserve the output?

<details><summary>Hint</summary>

Compute $U_K^Tq$ first. A consistent basis change uses $c'=Sc$ and $U'=US^{-1}$.

</details>

<details><summary>Solution</summary>

The effective query is `[2,2]`; both scaled scores are $\sqrt2$, so weights are `[0.5,0.5]` and the output is `[0.5,0.5]`. The new latent records are `[2,0]` and `[0,0.5]`. Right-multiply both up-maps by $S^{-1}=\operatorname{diag}(0.5,2)$. Their reconstructed keys and values then equal the originals, so scores, weights and output are preserved. Changing only the stored coordinates would generally change the function.

</details>

### 2. Preserve the temperature after absorption

A model has content-head width 4, rotary width 4 and latent width 12. Its concatenated latent/rotary query has width 16. A library uses its default inverse-square-root feature-width scale. What scale should the model use, what scale did the library choose, and how were the logits changed? Why can a test with latent width 4 miss this mistake?

<details><summary>Hint</summary>

The algebra preserves the old dot product, so compare the original content-plus-rotary width with the new representation width.

</details>

<details><summary>Solution</summary>

The intended scale is $1/\sqrt8$; the default is $1/\sqrt{16}=1/4$. Every finite unmasked logit is multiplied by $\sqrt{8/16}=1/\sqrt2$ relative to the intended value, making unequal logits less separated before softmax. This generally changes weights, although equal-score or other special cases can be unchanged. If latent width also equals 4, both total widths equal 8 and the scales coincide, concealing the bug.

</details>

### 3. Build the compact payload

Two requests each retain 2,048 positions at 12 layers. The content latent has 24 coordinates and the shared rotary key has 8, stored at two bytes each. Compute the compact payload in bytes and MiB. If the model doubles its query-head count while preserving these cache widths, what changes in this count and what can still become more expensive?

<details><summary>Hint</summary>

Multiply requests, layers, occupied positions, coordinates per record and bytes per coordinate. A MiB is $2^{20}$ bytes.

</details>

<details><summary>Solution</summary>

The payload is $2\times12\times2048\times32\times2=3,145,728$ bytes, or 3 MiB. It stays unchanged when only query-head count doubles. Query/head projections, score/value-mixture work and transient tensors can grow, as can communication or kernel overhead. The payload excludes reserved capacity, metadata, replicated copies and other model state.

</details>

### 4. Distinguish three “compression” operations

A programmer proposes: A, rearrange the same MLA weights from reconstructed to absorbed execution; B, change to an invertible latent basis and compensate every up-map; C, retain only the first half of some latent coordinates. Which are algebraically function-preserving under the stated conditions? Does an observed agreement on one input prove C is lossless?

<details><summary>Hint</summary>

Ask whether information or an operation was discarded, rather than whether a tensor has a new name.

</details>

<details><summary>Solution</summary>

A is exact when masking, scale, positions and all other operations remain equivalent. B is exact for the defined cached vector with the compensating inverse maps. C is generally lossy; it is exact only if the discarded directions have no relevant effect for the domain being claimed. Agreement on one input may mean that input has no discarded component or that effects cancel. It does not prove equality for every possible input. A basis change before an uncompensated nonlinear normalization is a different operation from B.

</details>

### 5. Diagnose a cache that forgot its positional field

A developer caches only the normalized content latent, then reconstructs every shared rotary key as if its position were zero. Shapes still match, and next-token outputs look plausible. What information is wrong? Why is a successful one-position test insufficient? Give a useful controlled comparison.

<details><summary>Hint</summary>

Trace the two additive score terms. Think about both different relative distances and a consistent common shift.

</details>

<details><summary>Solution</summary>

The content term can be correct while the rotary term compares queries with keys rotated at the wrong logical positions. A one-position test can have only one legal key, making softmax equal to one regardless of the score; it therefore cannot establish positional correctness. Compare an explicit multi-position reference with the cache path, using unequal vectors and distances. Also shift all positions consistently under fixed ordinary RoPE as a preservation control, then shift only queries or only cached-key positions as a contrasting operation. Retain or correctly reconstruct each required rotary key and its positional contract.

</details>

### 6. Find the nonlinear obstruction

Two cached scalar latents are −2 and 1 with weights 1/3 and 2/3. Values are obtained with ReLU. Compare mixing the values with applying ReLU after mixing latents. Would the rearrangement become valid for a fixed linear value map instead?

<details><summary>Hint</summary>

Evaluate ReLU on each value first in the original expression. In the second expression, add before applying it.

</details>

<details><summary>Solution</summary>

Mixing ReLU values gives $(1/3)0+(2/3)1=2/3$. Mixing latents gives $(1/3)(-2)+(2/3)1=0$, whose ReLU is zero. The two computations differ. A fixed linear map distributes over the weighted sum, so its rearrangement is exact. A position-dependent map, normalization or nonlinear activation needs separate analysis.

</details>

### 7. Explain why the smaller cache did more arithmetic

A single-query attention call has H=8, L=1,024, content width 8, rotary width 4, value width 8 and latent width 32. Ignore projection work and count a multiply-add as two operations. Compute expanded-core and absorbed-core operations. Does the larger arithmetic number decide which is faster?

<details><summary>Hint</summary>

Use $2HL(d_k+d_r+d_v)$ and $2HL(2d_c+d_r)$ for B=T=1.

</details>

<details><summary>Solution</summary>

Expanded core uses $2\times8\times1024\times20=327,680$ operations. Absorbed core uses $2\times8\times1024\times68=1,114,112$, or 3.4 times as many. The absorbed cache can still require much less stored data and support reuse across heads. Hardware bandwidth, arithmetic throughput, kernel layout and other overheads determine latency; neither the FLOP ratio nor the cache ratio alone is a timing result.

</details>

### 8. Challenge the rank interpretation

Someone says, “The content logits have rank at most two, so the softmax weights also have rank at most two and the entire attention is linear in sequence length.” Identify the two unsupported steps. What would a genuine linear-attention method need to specify?

<details><summary>Hint</summary>

Separate a rank bound on a matrix product from the effect of a nonlinear elementwise transformation and normalization.

</details>

<details><summary>Solution</summary>

Softmax need not preserve matrix rank; the lesson gives rank-one logits whose three-by-three probability matrix has full rank. Also, a latent feature factorization does not by itself remove the pairwise softmax computation over query and memory positions. A linear-attention method must specify its actual kernel/operator, normalization and associative accumulation or approximation, including causal-state behavior and its relationship to ordinary softmax. Those are the next topic's mechanisms.

</details>

### 9. Design the next experiment honestly

Our trained rank-eight model worsened after a rank-four cache intervention. A colleague concludes that rank-four MLA can never work. Another concludes that ten more epochs would certainly recover it. What do the recorded results actually establish, and what protocol would investigate either possibility without choosing a result after looking at the test set?

<details><summary>Hint</summary>

Identify which model was trained, which operation happened afterward, and which data chose the checkpoint.

</details>

<details><summary>Solution</summary>

The result establishes the effect of one predeclared, unadapted post-normalization truncation on one selected small model and this row-level task. It does not test a rank-four model trained from scratch or a declared adaptation schedule. Define that new architecture/intervention, training budget, seeds, simple baselines and validation criterion in advance; preserve group/duplicate boundaries and keep test data outside selection. Report all declared outcomes, including failure to recover. A selected checkpoint at the budget endpoint is not evidence of eventual convergence or certain recovery.

</details>

## What comes next?

You can now identify the actual cache fields, calculate both equivalent MLA paths, preserve scale and position, and distinguish representation loss from computational rearrangement. Continue to [Sparse and Linear Attention Variants](/learn/path/full-curriculum/sparse-linear-attention-variants?module=deep-learning-fundamentals), which changes the sequence comparisons or attention operator itself. Carry the same questions forward: what function is defined, what information is retained, what approximation is introduced, and what was actually measured?

## References and another way to learn

- **Original architecture:** [DeepSeek-V2 report](https://arxiv.org/html/2405.04434v5). The full section list, §2.1, practical model settings, context-extension treatment, Appendix C formulas and Appendix D ablations were inspected. Read the operator and cache definitions before the empirical comparisons. Its full model includes other changes, so headline system savings cannot all be attributed to one attention identity.
- **An executable primary implementation:** [DeepSeek-V3 official inference model](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py). The actual rotary function, MLA constructor, RMSNorm/cache branches and complete MLA forward were inspected. Follow the two branches with the same shapes. This is a mutable source branch; pin a revision for reproduction. Our complete small CPU program is independently implemented and executed.
- **A compact alternate explanation:** [Sebastian Raschka's MLA chapter guide](https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/). The complete short body was read. It connects the preceding GQA idea to a compressed cache and offers a useful recap; use the equations and source-bound examples here for scale, positional and performance details.
- **Visual attention prerequisite:** [3Blue1Brown's illustrated attention article and accompanying video](https://www.3blue1brown.com/lessons/attention/). Its substantive written Q/K/softmax/value explanation was inspected in the preceding packets and is reused as background. It helps visualize the weighted sum that MLA rearranges; it is not an MLA kernel tutorial, and no new video viewing is claimed here.
- **Existing-checkpoint conversion:** [MHA2MLA at ACL 2025](https://aclanthology.org/2025.acl-long.1597/) and [the paper](https://aclanthology.org/2025.acl-long.1597.pdf). The background, partial-RoPE selection, split/joint SVD methods and model/task setup with the first results table were inspected. They establish a concrete conversion research route, with task- and rank-dependent costs. Do not interpret the paper's title as a universal lossless-conversion guarantee.
- **Prefill and decode implementations:** [vLLM 0.20.0 MLA notes](https://docs.vllm.ai/en/v0.20.0/api/vllm/model_executor/layers/attention/mla_attention/) were inspected through dimensions, both compute paths and chunked prefill. Match the actual shapes and scale rather than copying informal pseudocode or a reversed ratio description. No vLLM installation or hardware performance test was performed here.
- **Current kernel and precision contracts:** [Official FlashMLA](https://github.com/deepseek-ai/FlashMLA). The dense/sparse capability table, API and version-specific cache formats were inspected on 13 September 2026. These distinguish stored bytes, scaling metadata and rotary precision. Kernel/model variants continue to evolve; their reported throughput is not a measurement made by this lesson.
- **Practical model details:** [V2 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json), [V3 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json), and [V3 report](https://arxiv.org/html/2412.19437v2). Relevant architecture, normalization, mixed-precision, deployment and context-extension portions were inspected. They help distinguish actual fields from shorthand architecture names.
- **Reproducible local learning:** [Data/protocol provenance](data-provenance.md), [complete CPU forecasting program](author-calculations.py), [actual outcomes](author-results.json), [saved model/trace](forecast-model.json), [independent NumPy fixtures](mechanism-calculations.py) and [their exact values](mechanism-fixtures.json). These support the manuscript's observed and constructed examples without requiring a large pretrained-model download.
