# Grouped-Query Attention and Multi-Query Attention

**Explore as you read.** Edit Q/K/V, query-to-KV grouping, cache dimensions, offset masks and supported causal input prefixes. Show each reader, shared K/V record, weighted sum, exact byte/MAC budgets and compact cache outputs immediately. Compare equal-head versus unequal-head regrouping. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose grouping by memory and functional tradeoffs, keeping payload arithmetic separate from measured latency and model quality.


A model predicting the next word repeatedly reads what it has already seen. Several attention heads can ask different questions about that history. Must each head keep its own separate description of every earlier token?

Grouped-query attention lets several query heads read the same keys and values. Multi-query attention shares one key/value head across all query heads. The important distinction is between **how many different reads we perform** and **how many different representations we store**. Sharing the stored representation can reduce the growing inference cache while retaining several distinct attention distributions.

The [previous lesson on positional encodings](/learn/path/full-curriculum/positional-encodings-sinusoidal-learned-rope-alibi?module=deep-learning-fundamentals) explained where positions enter those representations. Here we trace the actual grouped computation, build a compact causal cache and convert a small trained model. Our real example forecasts the next point of a recorded hand movement; it makes the known-prefix versus generated-future distinction visible without downloading a language model.

**First pass:** follow §§1–6 and exercises 1–5. You will be able to calculate a grouped head's output, implement correct caching, explain the storage savings and assess a conversion experiment. The deeper route in §7 develops gradients, representation constraints and serving tradeoffs; exercises 6–9 extend those ideas. The diagrams and short arithmetic examples belong alongside their explanations, not in a separate optional gallery.

## 1. The growing memory behind one prediction

### Read the history without recomputing it

Recall [Self-Attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals): a query is compared with keys, the resulting scores are normalized into weights, and those weights mix values. In a causal decoder, the current position may read itself and preceding positions. Later inputs are unavailable.

Suppose a decoder has already processed positions 0–31. It has computed their keys and values at every layer. When position 32 arrives, earlier causal hidden states do not need to change under a fixed model and position rule: none depended on the new future point. Store those past K/V tensors, project the new input once, append its K/V, and use the new query to read the enlarged history. This stored state is the **KV cache**.

Why do we generally not cache old queries? The next output uses a new query. The keys and values are the old information it reads. Old queries are no longer needed for this ordinary incremental attention step. Other algorithms can store additional state, but that does not alter this basic dependency.

**Figure — write once, read again.** Three time columns show a new input becoming one new query and one new stored K/V record. The next column reads every currently legal cache record. Separate arrows distinguish new projection work from re-reading existing representations. A layer label prevents the picture from implying that one cache is shared unchanged across all Transformer layers.

### Prefill and decode are different workloads

The prompt or observed prefix is already known. Its positions can be processed together with a causal mask; this is **prefill**. During generation, a newly predicted word or point becomes the input for a later prediction, so there is a sequential dependency. This is **decode**. A real decoder can batch independent requests and can verify proposed chunks, but it must preserve the relevant information dependencies.

With many known queries, the same K/V block can contribute to many calculations while resident near the arithmetic units. With one new query per request, repeatedly moving the growing history from device memory can be costly compared with the work done on it. This is the bandwidth motivation behind [Shazeer's multi-query attention paper](https://arxiv.org/pdf/1911.02150). It is not a claim that KV reads dominate every model, batch size, sequence length or device. Weight reads, feedforward arithmetic, communication and scheduling also matter.

Sharing K/V heads addresses the **head axis** of this history. It does not by itself remove old token positions, compress the sequence into a fixed recurrent state or change the causal mask.

## 2. Several questions, fewer stored descriptions

### Use names that cannot swap meanings

We will use:

| Symbol | Meaning |
|---|---|
|$H_q$|Number of query heads|
|$H_{kv}$|Number of distinct key heads and value heads|
|$R=H_q/H_{kv}$|Number of query heads sharing each K/V head|
|$d_k$|Coordinate width of one query/key head|
|$d_v$|Coordinate width of one value head|
|$D$|Width of the model's input/output rows|

The common equal-group construction requires positive head counts with $H_q$ divisible by $H_{kv}$. It also uses the same K-head and V-head count. Query and key widths must match for their dot product; value width can differ. A frequent architecture chooses $d_k=d_v=D/H_q$, but the operator itself does not require this equality.

For eight query heads:

| Mechanism | $H_q$ | $H_{kv}$ | $R$ | Query-head to KV-head mapping |
|---|---:|---:|---:|---|
|Multi-head attention, MHA|8|8|1|0,1,2,3,4,5,6,7|
|Grouped-query attention, GQA|8|4|2|0,0,1,1,2,2,3,3|
|More sharing|8|2|4|0,0,0,0,1,1,1,1|
|Multi-query attention, MQA|8|1|8|0,0,0,0,0,0,0,0|

In [Ainslie et al.'s notation](https://arxiv.org/html/2305.13245v3#S2.SS2), “GQA-8” means **eight KV groups**, not eight queries per group. A model with 32 queries and 8 KV heads has group size 4. A model with 64 queries and 8 KV heads has group size 8. Both have eight KV groups. Keeping these numbers separate prevents several apparent contradictions in model descriptions.

### A shared key does not mean a shared question

Imagine two readers consulting the same indexed records. One asks about the beginning of a movement; another asks about its most recent direction. They can assign different importance to the same records. The analogy describes shared information, not a guarantee that a particular head learns a human-named role.

In equations, the keys and values are shared, but $q_0$ and $q_1$ can differ. Their dot products, softmax weights and mixed outputs can therefore differ. MQA with eight query heads still produces eight head outputs. It is not single-head attention with a new name.

**Figure — wiring, then computation.** Draw query heads on one side and stored KV heads on the other. Each query has exactly one connection to its assigned memory head. Clicking a connection opens that query's full distribution over token positions. The wiring matrix is a fixed head assignment; the attention matrix is an input-dependent distribution over positions. Label these as different objects.

Contiguous groups are a layout convention. An interleaved assignment can define a valid architecture if training, weights and inference consistently use it. Swapping from contiguous to interleaved routing while retaining a checkpoint's unchanged parameters generally changes its function. That is the bug—not a theorem that every noncontiguous grouping is intrinsically inferior.

## 3. Calculate the grouped operation

### From projections to head outputs

Let the query inputs have $T$ rows, memory inputs have $S$ rows and batch size be $B$. In full self-attention these can be the same input; in an incremental step $T$ may be 1 while $S$ includes the prefix. After learned projections and head reshaping:

\[
Q\in\mathbb R^{B\times H_q\times T\times d_k},\quad
K\in\mathbb R^{B\times H_{kv}\times S\times d_k},\quad
V\in\mathbb R^{B\times H_{kv}\times S\times d_v}.
\]

Under the contiguous convention, query head $h$ reads memory head $g(h)=\lfloor h/R\rfloor$. Its score and output are

\[
s_{h,t,j}=\frac{q_{h,t}^{\top}k_{g(h),j}}{\sqrt{d_k}}+b_{h,t,j},\quad
a_{h,t,:}=\operatorname{softmax}_{\text{legal keys}}(s_{h,t,:}),\quad
o_{h,t}=\sum_j a_{h,t,j}v_{g(h),j}.
\]

Here $b$ may contain an appropriate position bias. An illegal key is excluded, conventionally by a score of negative infinity before softmax. Setting an illegal score to zero does not exclude it: its exponential would be 1.

Concatenate all $H_q$ output heads, producing width $H_qd_v$, then apply an output map back to width $D$. This is followed by the residual and feedforward processing from [Transformer Block Architecture](/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals). Sharing K/V changes the K/V projection shapes inside attention; it leaves the external row width compatible with the rest of the block.

### A four-query, two-memory example

Use one query position, three legal memory positions and $d_k=d_v=2$. These are deliberately chosen arithmetic inputs, not learned language-model features.

The four queries are

\[
q_0=\sqrt2[1,0],\quad q_1=\sqrt2[0,1],\quad
q_2=\sqrt2[-1,0],\quad q_3=\sqrt2[0,-1].
\]

Each group holds three keys and values:

| Position | Group 0 key | Group 0 value | Group 1 key | Group 1 value |
|---|---|---|---|---|
|0|[1,0]|[2,0]|[1,1]|[1,3]|
|1|[0,1]|[0,4]|[−1,0]|[−1,2]|
|2|[1,1]|[2,2]|[0,−1]|[3,0]|

Queries 0 and 1 read group 0; queries 2 and 3 read group 1. For query 0, dividing by $\sqrt2$ cancels the chosen query factor, giving scores `[1,0,1]`. Their exponentials are `[e,1,e]`, so the weights are approximately `[0.422319,0.155362,0.422319]`. The output is

\[
0.422319[2,0]+0.155362[0,4]+0.422319[2,2]
\approx[1.689275,1.466087].
\]

Performing the same calculation for the other queries gives:

| Query head | KV group | Scaled scores | Weights over positions 0,1,2 | Mixed output |
|---|---:|---|---|---|
|0|0|[1,0,1]|[0.422319,0.155362,0.422319]|[1.689275,1.466087]|
|1|0|[0,1,1]|[0.155362,0.422319,0.422319]|[1.155362,2.533913]|
|2|1|[−1,1,0]|[0.090031,0.665241,0.244728]|[0.158975,1.600574]|
|3|1|[−1,0,1]|[0.090031,0.244728,0.665241]|[1.841025,0.759549]|

The two readers of group 0 plainly have different answers. Their shared memory did not force their attention weights to coincide.

### Edit a head and inspect which outputs change

Change the first value of group 0 from `[2,0]` to `[3,−1]`. No score changes because scores use Q and K. The two group-0 outputs change by their respective weight on that position times `[1,−1]`. Heads 2 and 3 are unchanged because they read another group.

Now instead change group 0's first key from `[1,0]` to `[2,0]`. Query 0's first score increases. Query 1's score does **not** change in this special fixture: its query has zero x component. A shared key edit can influence every reader in its group, but it need not do so for every particular query. This controlled null is more informative than an animation that always lights up all arrows as “affected.”

**Investigation — edit a shared memory.** Change any query, key or value coordinate; show identify which heads' weights and outputs can change, and explain why. Linked numeric distributions and vector-sum diagrams show the actual result. A second action relabels the two KV groups and updates their connections together; this preserves every output. Changing only the connections generally does not.

### Repeat is one implementation, not the definition

An easy reference implementation repeats each KV head $R$ times and calls ordinary multi-head attention. This gives the correct function, but explicit `repeat_interleave` allocates repeated tensors. Its gradients are valid because backward sums the contributions from copies; valid differentiation does not make the forward a zero-cost view.

We can instead reshape queries as `[B,Hkv,R,T,dk]` and keep K/V compact. Compute scores with

```python
scores = torch.einsum("bgrtd,bgud->bgrtu", grouped_query, keys) / math.sqrt(dk)
weights = scores.softmax(dim=-1)
outputs = torch.einsum("bgrtu,bgud->bgrtd", weights, values)
```

The letters name axes: `g` is a KV group, `r` a reader within it, `t` a query position and `u` a memory position. Summing over `d` makes scores; summing over `u` mixes values. A mask belongs before softmax. The full executable program below includes it.

This eager grouped form avoids an explicit repeated K/V array. It still forms one score distribution per **query head**, so its score tensor has $BH_qTS$ entries. A fused kernel can tile this work and reuse K/V within its execution strategy. Do not infer a particular physical memory-traffic count merely from a high-level `einsum`.

## 4. Build the cache and count the savings honestly

### Store compact K/V, retain the position contract

At a new position, project $H_q$ queries and only $H_{kv}$ keys and values. If using standard RoPE, rotate each query and each unique key at its proper logical position, then append the compact rotated keys and unrotated values to the cache. Read them using the group mapping.

If all copies receive the same position and frequency rule, “rotate then repeat” and “repeat then rotate” are mathematically equivalent. Rotating the unique keys avoids redundant work. Cache correctness depends on retaining a consistent rotated/unrotated convention and logical IDs; the mere ordering of two equivalent operations does not force an expanded cache or a position bug.

A one-position query tensor can correspond to logical position 32. The legal keys are positions 0–32, not only position 0. More generally, a new chunk at positions 3 and 4 against keys 0–4 needs

```text
query 3: 1 1 1 1 0
query 4: 1 1 1 1 1
```

This is a logical-position relation. A generic upper-left triangle on a 2-by-5 tensor would instead permit only the first one and first two keys. Both arrays have valid shapes; only one describes this cached computation.

**Figure — cache growth with two readers per group.** Keep head/group axes and token-position axes visually separate. Each new token adds one record per KV group. Query heads point to compact stored groups; transient head outputs are drawn outside the cache. A small offset-mask inset shows the new chunk reading its prefix.

### Bytes come from actual tensor dimensions

For a uniform unquantized cache with batch size $B$, $N$ layers, prefix length $L$, equal K/V head count $H_{kv}$ and bytes per stored number $s$, the K/V tensor payload is

\[
C=B N L H_{kv}(d_k+d_v)s.
\]

When $d_k=d_v=d_h$, this becomes $2BNLH_{kv}d_hs$. One factor 2 means “key plus value”; another factor may come from using two-byte FP16/BF16 numbers. Do not merge them and accidentally halve the answer.

For an illustrative architecture with 80 layers, 64 query heads, width 128 per K/V head and two-byte cache entries:

| Distinct KV heads | Cache bytes per token, one sequence | Payload at 32,768 tokens | Same payload in decimal GB |
|---:|---:|---:|---:|
|64, MHA|2,621,440|80 GiB|85.899 GB|
|8, GQA|327,680|10 GiB|10.737 GB|
|1, MQA|40,960|1.25 GiB|1.342 GB|

A GiB is $2^{30}$ bytes; a decimal GB is $10^9$ bytes. These are calculated tensor sizes for the stated configuration. They do not claim that a named checkpoint was trained for this context length or that the complete model fits a particular device.

The reduction from 64 KV heads to 8 is eightfold; from 8 to 1 is another eightfold; from 64 to 1 is sixty-fourfold. Group size, head count and the chosen comparison determine the ratio.

For variable request lengths and different attention layer types, sum the actual per-request, per-layer terms. An allocated cache may reserve maximum lengths or rounded pages instead of exactly the occupied tokens. Quantized caches also have scales, packing and other metadata; four-bit values do not imply that every stored quantity occupies exactly half a byte. Prefix sharing and distributed replication further distinguish logical tensor payload from physical allocation.

### Parameter arithmetic changes too

Without biases, projections have parameter counts

\[
P_Q=DH_qd_k,\quad P_K=DH_{kv}d_k,\quad
P_V=DH_{kv}d_v,\quad P_O=DH_qd_v.
\]

Their sum is $D(H_q+H_{kv})(d_k+d_v)$. For the common $d_k=d_v=D/H_q$ case:

\[
P_{\rm attention}=2D^2\left(1+\frac{H_{kv}}{H_q}\right).
\]

With $D=512,H_q=8$, MHA has 1,048,576 attention-projection parameters; two-KV-head GQA has 655,360; MQA has 589,824. Biases, norms, feedforward layers and embeddings add their own parameters. The percentage saved in the **whole model** depends on that full architecture. It is not always 5%, and optimizer-state memory can also decrease when trainable parameter count decreases.

### Which arithmetic remains?

Ignoring small overheads and counting a multiply-add as two operations, one attention application needs approximately

\[
2BH_qTSd_k\quad\text{for QK scores},\qquad
2BH_qTSd_v\quad\text{for mixing values}.
\]

These terms retain $H_q$, even when K/V are shared. Different queries still form different weighted sums. The K/V **projection** arithmetic does decrease with $H_{kv}$. Thus “all FLOPs stay identical” is too broad; “the dense score and value-mixing arithmetic is unchanged at fixed query heads, widths and lengths” is the useful precise statement.

For one decode query with equal widths, these attention operations total about $4BH_qLd_h$. An idealized read-once compact K/V payload is $2BH_{kv}Ld_hs$, suggesting arithmetic per byte of $2R/s$. With two-byte elements this is $R$ operations per byte. This model assumes effective reuse and omits weights, writes, cache hierarchy and other work; it explains why sharing can help a bandwidth-limited kernel without predicting its measured speed.

If 60% of a step's time were reducible by a factor of 8 and everything else stayed fixed, the total speedup would be

\[
\frac{1}{0.4+0.6/8}\approx2.105,
\]

not 8. This is an illustrative Amdahl calculation, not an observed decoder timing. Real group counts may also change kernel occupancy, parallelism, batching and communication. Measure the workload before turning a storage ratio into a latency claim.

**Investigation — account for every axis.** Edit layers, query heads, KV heads, prefix lengths, widths and dtype size. Build the bytes from labeled tensor blocks, then compare MHA/GQA/MQA under the same assumptions. A separate panel shows the explicitly assumed Amdahl calculation. No timing curve is drawn from byte counts.

## 5. Implement the mechanism and verify a real API contract

### A complete transparent NumPy reference

This program loops over query heads so the assignment is easy to inspect. It does not repeat the stored K/V array. Python and NumPy are sufficient; no trained weights or GPU are needed. The displayed arrays are the hand example from §3.

```python
import math
import numpy as np


def grouped_attention(query, keys, values, query_positions, key_positions,
                      mapping=None, causal=True):
    # Q: [B,Hq,T,dk], K: [B,Hkv,S,dk], V: [B,Hkv,S,dv].
    batch, query_heads, query_length, key_width = query.shape
    kv_heads = keys.shape[1]
    if query_heads % kv_heads or keys.shape[:3] != values.shape[:3]:
        raise ValueError("Use equal-size groups and matching K/V head/slot counts.")
    if keys.shape[0] != batch or keys.shape[-1] != key_width:
        raise ValueError("Batch and Q/K coordinate widths must agree.")
    if mapping is None:
        mapping = np.arange(query_heads) // (query_heads // kv_heads)
    mapping = np.asarray(mapping)
    if (mapping.shape != (query_heads,) or np.any(mapping < 0)
            or np.any(mapping >= kv_heads)):
        raise ValueError("Each query head must name a valid KV head.")
    legal = np.ones((query_length, keys.shape[2]), dtype=bool)
    if causal:
        legal = np.asarray(key_positions)[None, :] <= np.asarray(query_positions)[:, None]
    if not legal.any(-1).all():
        raise ValueError("A query has no legal key.")
    outputs, weights = [], []
    for head, memory_head in enumerate(mapping):
        scores = query[:, head] @ keys[:, memory_head].swapaxes(-1, -2)
        scores /= math.sqrt(key_width)
        scores = np.where(legal, scores, -np.inf)
        probabilities = np.exp(scores - scores.max(-1, keepdims=True))
        probabilities /= probabilities.sum(-1, keepdims=True)
        outputs.append(probabilities @ values[:, memory_head])
        weights.append(probabilities)
    return np.stack(outputs, 1), np.stack(weights, 1)


q = np.array([[1,0], [0,1], [-1,0], [0,-1]], dtype=float)[None,:,None,:] * math.sqrt(2)
k = np.array([[[1,0], [0,1], [1,1]], [[1,1], [-1,0], [0,-1]]], dtype=float)[None]
v = np.array([[[2,0], [0,4], [2,2]], [[1,3], [-1,2], [3,0]]], dtype=float)[None]
output, weights = grouped_attention(q, k, v, [2], [0,1,2])
print("weights:", np.round(weights[0,:,0], 6))
print("head outputs:", np.round(output[0,:,0], 6))

# The same function represented as MHA with tied/repeated K/V heads.
tied, _ = grouped_attention(q, np.repeat(k, 2, axis=1),
                           np.repeat(v, 2, axis=1), [2], [0,1,2])
print("tied MHA agrees:", np.allclose(output, tied, atol=1e-12))

# For one query at the last slot, a compact prefix plus the new entry agrees.
cached_keys = np.concatenate((k[:,:,:2], k[:,:,2:]), axis=2)
cached_values = np.concatenate((v[:,:,:2], v[:,:,2:]), axis=2)
cached, _ = grouped_attention(q, cached_keys, cached_values, [2], [0,1,2])
print("compact cache agrees:", np.allclose(output, cached, atol=1e-12))
```

The head outputs are `[1.689275,1.466087]`, `[1.155362,2.533913]`, `[0.158975,1.600574]` and `[1.841025,0.759549]`, and both checks print `True`. The compact-cache check here verifies the **projected-array assembly**. The full neural program in §6 additionally verifies that projecting and processing an observed sequence incrementally reproduces all full-pass causal predictions.

This reference rejects an all-masked query instead of normalizing an empty set. A production kernel may specify another policy, such as a zero output, but the application must still distinguish “there is no legal evidence” from an ordinary attention distribution. Validate head counts, widths, masks and finite data at the actual input boundary.

### Calling PyTorch SDPA is an explicit choice

The [PyTorch 2.14 SDPA contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) exposes `enable_gqa=True`. It is false by default. Do not assume that fewer K/V heads automatically select the intended grouped operation in every API or backend. MQA can also happen to broadcast in some lower-level operations; that does not replace an explicit grouping contract.

For existing tensors Q/K/V with the shapes above, the relevant call is:

```python
result = torch.nn.functional.scaled_dot_product_attention(
    query, keys, values,
    attn_mask=legal_mask,       # True means this key participates.
    is_causal=False,           # The explicit mask already includes causal legality.
    dropout_p=0.0,
    enable_gqa=True,
)
```

This is a call-site fragment; [the complete operator-check program](mechanism-calculations.py) supplies the tensors and executes it. Its CPU float64 output agreed with the direct grouped calculation to $2.23\times10^{-16}$ in the recorded PyTorch 2.14.0 environment. That observation is not evidence about CUDA dispatch, throughput or every supported shape.

Pay attention to two particularly easy API mismatches. SDPA's boolean mask uses `True` for **allowed**, whereas `MultiheadAttention`'s key-padding mask uses `True` for **excluded padding**. Also, SDPA's documented non-square `is_causal=True` aligns a triangle at the **upper left**. A cached query at the last position needs the offset relation from §4. In our one-query/three-key fixture, the upper-left mask allows only the first key, returning its value in each group instead of the correct three-key mixture.

The [FlashAttention interface documentation](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention) describes a bottom-right-aligned causal convention for its current unequal-length operation. These are different API contracts, despite similar parameter names. Compare each call with an explicit logical-position reference rather than transplanting a mask assumption between libraries. Backend support and fused-kernel constraints are version-specific; the recorded native checks do not require installing FlashAttention.

SDPA also applies dropout according to the `dropout_p` argument. Setting a surrounding module to evaluation mode does not automatically override a nonzero argument passed to the functional call. Use zero for deterministic cached inference unless the application deliberately defines another behavior.

The [maintained Llama source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) provides a useful production reading exercise: inspect smaller K/V projections, native-head RoPE, compact cache update, then attention-backend dispatch. Its eager `repeat_kv` helper is a readable reference; it is not proof that an expand-plus-reshape path always avoids allocation. Inspect the actual storage and backend when that distinction matters.

### Make sharing visible to the optimizer

The complete `mechanism-calculations.py` also checks the derivative that sharing creates: reshape per-reader value gradients into `[B,Hkv,readers,S,dv]` and sum the readers axis. That result must equal the compact shared V gradient. This is the bridge from a storage choice to actual fitting, not a claim that averaging separately updated MHA weights implements the same update.

For an independent variation, use six query heads and two KV heads, with value width different from key width. Change the group reshape and output merge consistently; keep the true Q/K scale. **Hint:** each KV parameter now receives contributions from three readers. **Solution:** SDPA and the direct grouped operator should agree, and three per-reader gradients sum into each shared gradient. The real forecast program already owns projection, optimizer, conversion/uptraining and full-versus-cached state. Reuse it for this extension; do not rebuild the attention derivation or compare unrelated random fits.

## 6. Convert a trained model and observe what survives

### Mean pooling is an initialization, not function preservation

An MHA checkpoint has separate K/V projection parameters for each head. To initialize a grouped model, average the original K-head matrices inside each new group and do the same for V. Copy the Q maps, output map and other compatible parameters.

For group $g$ containing head indices $I_g$:

\[
\overline W_{K,g}=\frac1R\sum_{h\in I_g}W_{K,h},\qquad
\overline W_{V,g}=\frac1R\sum_{h\in I_g}W_{V,h}.
\]

Average K/V biases too when those projections have biases. With PyTorch's output-by-input weight layout, reshape a key weight of shape `[Hq*dk,D]` into `[Hkv,R,dk,D]`, average the reader axis and reshape to `[Hkv*dk,D]`. Mixing the coordinate axis with the head axis produces a different projection, even if the final tensor has an acceptable shape.

The mean has a precise limited justification. It minimizes the sum of squared Euclidean/Frobenius distances to the matrices being averaged:

\[
\sum_h\|W_h-M\|_F^2
=\sum_h\|W_h-\overline W\|_F^2+R\|M-\overline W\|_F^2.
\]

The first term is independent of $M$; the second is smallest at the mean. This preserves a particular notion of proximity **in parameter coordinates**. It does not minimize the final model's prediction loss or average attention outputs. Heads can use different learned coordinate bases, and softmax is nonlinear.

For a fixed query q, comparing it with the mean key gives the mean of its comparisons with all keys in that group. This is not the average of the original heads' own comparisons, which use different queries. Applying softmax then mixing averaged values introduces further nonlinear differences.

### A two-head counterexample

Take scalar queries 1 and 2. Head 0 has keys `[2,0]`, values `[1,3]`; head 1 has keys `[0,2]`, values `[5,−1]`. Their original outputs are approximately 1.238406 and −0.892083. Mean-pooling keys gives `[1,1]`, and mean-pooling values gives `[3,1]`. Both retained queries now see equal scores over the two positions, so each produces output 2.

The averaged parameters have not preserved either original output, nor their average. No random numerical accident is needed to demonstrate the issue. Conversely, if the original K and V parameters inside each group are already identical and the positional/masking conventions agree, conversion preserves the function exactly. This tied-head case is a useful null control.

**Investigation — merge two learned descriptions.** Edit small original projection/key/value entries and inspect their mean, the live output change, and the linked old and converted attention distributions. A tied-head control produces exact agreement. Keep a distance-to-parameters display separate from a prediction-error display; one cannot stand in for the other.

Further training lets the model adapt to the new structure. The original GQA study calls this **uptraining**. Its T5 experiment continued the pretraining recipe after conversion, then evaluated downstream tasks. It did not establish that every checkpoint can be converted losslessly with a fixed number of updates. Use the original optimization/data recipe when reproducing a reported result; our local study deliberately uses a much smaller and different task.

### A real causal forecasting task

We use the same licensed [Libras Movement recordings](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement) as the preceding positional lesson, but ask a new question: **given the observed points so far, where will the next point be?** Each record has 45 normalized x/y coordinates. Inputs 0–43 predict targets 1–44. The causal mask ensures a prediction cannot read its target or later points. Movement-class labels are used only to preserve the declared split, not as model inputs or forecast targets.

The source has 360 rows but 330 distinct trajectories. Remove the 30 additional exact duplicates after checking that duplicate labels agree, retain the first occurrence and use the existing seed-73 classwise split: 220 training, 50 validation and 60 test trajectories. Keep whole trajectories together. Fixed scaling $2x-1$ supplies model inputs and targets; output RMSE is converted back to the original coordinate unit. The source lacks reliable per-row performer/session identifiers, so this is a row-level diagnostic, not evidence of generalization to a new signer or a deployable movement system. [Original data, metadata and attribution](data-provenance.md) accompany the program.

A persistence baseline predicts the latest observed point unchanged. An affine baseline fits $\widehat x_{t+1}=A x_t+b$ to training transitions only. These simple models make it harder to mistake smooth motion for impressive architectural learning.

### A controlled conversion protocol

The neural model has a 2-to-24 coordinate projection, one pre-normalized Transformer block, four query heads of width 6, FFN width 48 with GELU, a final LayerNorm and a two-coordinate prediction at every row. It uses full Q/K RoPE with base 10000 and logical positions 0–43, biased maps, epsilon $10^{-5}$ and no dropout. It predicts coordinates rather than classes or token probabilities.

First train the four-KV-head MHA parent using seed 101, 180 full-batch Adam updates at learning rate 0.003. Select the lowest validation MSE, taking the earliest exact tie. The selected parent is update 171.

From that one checkpoint, create three branches: continue MHA unchanged, convert to two-KV-head GQA by group means, and convert to one-KV-head MQA. Each branch then gets exactly 45 additional full-batch Adam updates at learning rate 0.001 with a fresh optimizer. Select by validation MSE, including the zero-update state as a candidate. The continued MHA branch controls for the benefit of simply doing more optimization. All shared weights are copied from the parent; no branch is retuned after seeing test performance.

This is a small conversion study, not a comparison of fully optimized architectures trained from scratch. Different branches have different parameter counts and may require different adaptation schedules. The fraction 45/180 is a count of these local updates, not a claim to reproduce the original paper's pretraining-compute fraction.

### The actual outcomes

RMSE below is per coordinate, in the original normalized coordinate units. Each neural test result averages errors over all 44 prediction slots of the same 60 held-out trajectories.

| Model/branch | Parameters | Selected extra update | Test RMSE before extra training | Test RMSE after selection |
|---|---:|---:|---:|---:|
|MHA continuation, 4 KV heads|5,042|41|0.015789|0.015617|
|Converted GQA, 2 KV heads|4,442|45|0.077182|0.018268|
|Converted MQA, 1 KV head|4,142|42|0.115174|0.025224|
|Persistence baseline|0|—|—|0.026458|
|Training-fitted affine baseline|6|—|—|0.026222|

Averaging sharply worsens both converted models initially. Further training recovers much of that loss. GQA remains worse than the MHA control in this run; MQA is only slightly better than the simple baselines. These are informative outcomes, not failures to obtain the intended lesson. Cache size reductions are exact consequences of shape; quality recovery is an empirical question.

Validation RMSE after selection is 0.015260, 0.018438 and 0.025090 for MHA/GQA/MQA respectively. The GQA selection lands on the last allowed update, so this protocol does not establish its eventual converged quality. More adaptation might change the result; do not extend the schedule solely to improve a displayed test score. A new training question would need a newly declared protocol and appropriate evaluation boundary.

### Inspect one real forecast, including the cache

The visible example was chosen before fitting: source row 77. Observe its first 32 points, then predict point 32 under zero-based numbering. The actual next point is approximately `[0.593810,0.250000]`.

| Selected branch | Next predicted point | Compact float32 K/V payload for this 32-point prefix |
|---|---|---:|
|MHA|[0.600976,0.258456]|6,144 bytes|
|GQA|[0.597412,0.259018]|3,072 bytes|
|MQA|[0.607105,0.253203]|1,536 bytes|

These are actual saved model outputs. The bytes are also checked against the actual K/V tensors: `[1,Hkv,32,6]` for each of K and V. They exclude position metadata and all other model state.

Feeding those same 32 observed points one at a time with the compact cache reproduced **every** full-pass causal forecast within $2.7\times10^{-7}$ in transformed coordinates. Shifting all logical IDs by 100 preserved the outputs within $2.4\times10^{-7}$ under this fixed RoPE model. Neither check means a new unseen input can be ignored; it verifies equivalence of two computations of the same specified function.

Reflect observed point 23's x coordinate using $x\mapsto1-x$. The final GQA next-point forecast changes from `[0.597412,0.259018]` to `[0.597468,0.258925]`; its earlier per-row predictions can change much more. Causality requires outputs **before** the edited point to stay unchanged. Attention to a changed old point can affect later queries even though we use a compact cache. Because the old observation changed, recompute the affected cached states rather than silently reusing a cache for a different prefix.

**Investigation — a shared-memory forecast workspace.** An observed trajectory occupies the left side of a time boundary. Predictions appear on the right only after the learner commits whether a proposed input/cache change should preserve the result. Select any query head to see which KV group it reads, its actual weights over observed positions and its contribution to the forecast. Change an observed coordinate, move the boundary, or compare full and incremental processing. The model selector loads the actual selected branch, not a relabeled copy of one attention map.

### One-step evaluation is not a generated rollout

In the table, each prediction receives the real preceding observations. This is teacher-forced evaluation. To forecast several future points, feed the model's own output back as its next input. Errors can change subsequent inputs and accumulate.

Starting from the same 32-point prefix, the GQA model's five generated points are approximately

```text
[0.597412,0.259018]
[0.590012,0.261214]
[0.583480,0.265978]
[0.577641,0.272512]
[0.571721,0.279467]
```

Only the first prediction used the last true observed point as its newest input; each later prediction used the prior prediction. The workspace may reveal the actual recorded future afterward for inspection, but those values must not enter the rollout. Edited trajectories have no newly established ground truth. Also, the linear output is not constrained to 0–1; do not silently clip out-of-range predictions and describe the clipped path as the model's actual output.

This distinction connects the earlier sequence-model lessons to serving: caching removes repeated computation of an unchanged history; it does not remove the uncertainty or feedback loop of generation.

### Reproduce the complete study

Download [the complete CPU program](author-calculations.py), [the data](movement_libras.data), [original metadata](movement_libras.names) and [the recorded results](author-results.json) into one directory. With Python, NumPy and PyTorch installed:

```bash
python author-calculations.py
```

The recorded environment was Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu, one CPU thread and deterministic algorithms. The program includes the complete model, causal/RoPE/grouped operator, mean-weight/bias conversion, both baselines, all fitting and checkpoint selection, split/duplicate checks, actual weight export and full/cached interventions. It does not rely on a hidden notebook, pretrained download or a GPU. Numerical environments can change last digits.

Read `CausalForecaster.forward` alongside the diagram: normalize the carried state; project native Q/K/V counts; rotate; append compact K/V and logical IDs; reshape the query-head axis into groups/readers; score and mask; mix values; combine heads; complete the residual/FFN path; forecast the next coordinate. The `convert` function shows exactly which parameters are averaged and which are copied.

The [model/trace file](forecast-models.json) supports later interactive implementation. Its full author evidence is an optional download, not an eager browser payload. A page should load only the selected model's compact weights and selected example, and should evaluate changed inputs without training.

## 7. Deeper connections: capacity, optimization and deployment

### Sharing ties parameters and accumulates gradients

For a fixed group, several query heads depend on the same K/V parameters. During training, the shared parameters receive contributions from every use. This is the same chain-rule principle as reusing a function or sharing a convolution kernel across image positions.

If the loss is $\mathcal L$ and group g's value at position j is $v_{g,j}$, its direct value-path derivative is

\[
\frac{\partial\mathcal L}{\partial v_{g,j}}
=\sum_{h:g(h)=g}\sum_t
a_{h,t,j}\frac{\partial\mathcal L}{\partial o_{h,t}}.
\]

Each reader contributes according to how much it used that value and how its output affected the loss. There is no extra automatic average over group size. If the loss itself is averaged across examples or positions, that normalization is already inside the upstream derivatives.

Keys have a more coupled effect because they alter softmax weights. For one reader/query, the derivative with respect to its scaled score is

\[
\frac{\partial\mathcal L}{\partial s_{h,t,j}}
=a_{h,t,j}\left(\frac{\partial\mathcal L}{\partial o_{h,t}}\right)^\top
(v_{g,j}-o_{h,t}).
\]

Multiplying by $q_{h,t}/\sqrt{d_k}$ and summing over that group's readers/positions gives the shared key's gradient when the key enters scores through this dot product. The subtraction of the current mixture appears because increasing one key's weight decreases other normalized weights. Position transforms add their own chain-rule factors.

The [operator-check program](mechanism-calculations.py) differentiates a squared-output loss through direct grouped attention and separately through independent repeated value copies. Summing the per-copy gradients exactly recovers the shared-value gradient within float64 tolerance. This is a substantive check of the sharing mechanism, not merely “a gradient exists.”

**Figure — several gradient contributions meet at one parameter.** Draw each reader's weighted contribution as a vector, then add them at the shared value record. Contributions may reinforce or cancel. Show the scalar loss reduction convention explicitly. The figure should not suggest that shared heads are automatically trained to agree on every use.

### What capacity is being restricted?

At fixed head counts and widths, a grouped model can be represented as an MHA model whose K/V parameters are tied inside each group. MHA allows that tied choice and also permits separate K/V maps. GQA restricts this part of the parameterization while retaining query and output-head diversity.

This explains a capacity tradeoff, not a theorem that a finite trained MHA model must generalize better. Optimization, data volume, inductive bias and training budgets matter. A restriction can help some tasks, hurt others or make little practical difference. The local forecasting table and published studies are evidence under their own protocols.

Shared values also do not force all head outputs to be equal: different distributions form different combinations of the same value rows. Before output projection, each head's mixture is in the convex hull of its legal value vectors when attention is ordinary nonnegative row-softmax without dropout. Distinct heads may choose different points in that hull, and their output projections combine them differently. This geometric interpretation does not impose the same convex-hull restriction on the final residual state or forecast.

If an MHA checkpoint's heads use different feature bases, naive averaging can be a poor functional merge. Even a function-preserving head permutation changes which heads fall into contiguous groups unless grouping is transformed too. Group selection or learned conversion procedures are meaningful research choices. The mean-initialization recipe is a practical baseline, not a proof that the original order yields optimal groups.

### What the original studies actually establish

The [GQA paper's Table 1](https://arxiv.org/html/2305.13245v3#S3) reports this T5-XXL comparison after its specified 5% uptraining and task fine-tuning:

| Attention | Average development-task score | Reported inference time, seconds per sample per TPUv4 chip |
|---|---:|---:|
|MHA|47.2|1.51|
|MQA|46.6|0.24|
|GQA with 8 KV groups|47.1|0.28|

The average combines the paper's summarization, translation and question-answering metrics; it is not accuracy on one dataset. The authors optimized parallelization separately and used their stated batching/timing protocol. These rows illustrate their quality/time tradeoff, not a universal GPU speed multiplier. Their Figure 5 varies **uptraining proportion**; Figure 6 varies groups and reports **time**. Neither is a measured seven-point quality curve proving that eight groups are always optimal. The limitations also identify encoder-decoder-only evaluation and the absence of a from-scratch XXL GQA comparison.

The [MQA paper's model-quality section](https://arxiv.org/pdf/1911.02150) likewise reports task- and metric-specific differences, including a beam-search translation score where MQA slightly exceeds its MHA baseline. Its comparison widens feedforward layers to match total parameter counts. There is no universal “MQA loses exactly 1%” law. The practical questions are which quality measures matter, what adaptations were performed, and what serving workload was measured.

### Three separate ways to reduce attention cost

| Technique | What it changes | What it does not automatically establish |
|---|---|---|
|GQA/MQA|Number of distinct K/V heads|Fewer legal token positions or fixed-size sequence state|
|Sliding/local attention|Which positions each query can read|Identical full-attention function or unlimited direct access to old tokens|
|Tiled exact attention|How the same score/softmax/value calculation is scheduled and stored|A different mathematical attention operator or a smaller logical KV cache by itself|
|Paged cache allocation|Where physical cache blocks are stored and shared|A change to learned projection widths or a universal throughput multiplier|
|KV quantization|Representation precision and payload packing|Exact original floating-point outputs or zero metadata overhead|

These can be combined when their implementation contracts agree. [Mistral 7B's architecture table](https://arxiv.org/html/2310.06825v1#S2) gives a concrete published example with 32 query heads and 8 KV heads, combined with sliding-window attention. That is group size 4, not four KV heads. The window and shared heads act on different axes. Its original context/window choices describe that checkpoint, not a recommendation that every new model use those settings.

Paged allocation rounds token storage into blocks and can support sharing identical prefix blocks with appropriate reference management. It does not guarantee that a logical eightfold GQA reduction multiplies another claimed fourfold saving into thirty-twofold throughput. Allocation efficiency, bandwidth, scheduling and computation interact. Treat the [original PagedAttention paper](https://arxiv.org/abs/2309.06180) as further systems reading; the local plots here contain no unmeasured serving numbers.

Speculative decoding has another role: a draft proposes several tokens and the target verifies them. GQA can supply the target's attention implementation, but its cache layout must support accepted/rejected prefix updates. The benefit depends on acceptance, batch/length and hardware. Do not assign a fixed extra multiplier just because both techniques are present.

### Logical sharing can be replicated across devices

Suppose tensor parallelism partitions query heads across eight devices. A simple implementation may replicate a single MQA KV head on all eight devices so each can perform its local reads. The model logically has one KV head, but aggregate physical storage contains eight copies. With eight GQA KV heads, a suitable one-group-per-device partition can avoid that particular replication.

For this simplified equal partition, aggregate stored-head copies can be $\max(H_{kv},P)$ when head counts and the $P$ partitions are compatible and smaller counts are replicated. This is an example strategy, not a universal distributed-cache formula. A different algorithm might communicate or partition the state differently. Count actual per-device storage and communications; do not divide every cache estimate by the device count automatically. The [GQA method discussion](https://arxiv.org/html/2305.13245v3#S2.SS2) explicitly motivates grouped heads partly through this sharding issue.

### When is this useful beyond chat?

**Streaming trajectories and sensor prediction.** Our point predictor is a small causal example. A long stream would keep adding cached records unless the architecture or application defines a window, reset or another memory mechanism. Shared heads reduce each stored record's width; they do not make indefinite storage finite. A reliable streaming application also needs a policy for missing observations, changed calibration and corrected historical inputs.

**Encoder-decoder systems.** A decoder can use GQA for self-attention over generated tokens and for cross-attention over fixed encoder outputs. The cross-attention K/V can be projected once from those encoder outputs and reused while decoding. Its source length can differ from the generated-prefix length. If encoder outputs change, their cached projections must be refreshed. GQA is not restricted to decoder-only language models, although its incremental motivation is strongest in particular workloads.

**Multimodal prefixes.** Image patches or audio features can contribute many positions to an autoregressive model's context. Shared K/V heads can reduce the representation stored for each such position, but positional coordinates, modality boundaries and legal cross-stream attention remain separate decisions. The later vision and cross-attention lessons own those representations.

**Mixture-of-experts models.** A Transformer may route feedforward computation to experts while keeping a shared attention layer. In that arrangement, expert count does not multiply the attention KV-head count. Read the actual block definition; the phrase “eight experts” does not mean eight independent copies of every attention cache. [Mixture-of-Experts Transformers](/learn/path/full-curriculum/mixture-of-experts-transformers-moe?module=deep-learning-fundamentals) develops that distinction.

### Choosing or reproducing a configuration

For a pretrained checkpoint, reproduce its query/KV counts, grouping, widths, positional convention and weights. Changing a configuration field alone is not a conversion. A changed projection shape needs transformed or newly learned parameters and an evaluated adaptation plan.

For a new model, compare reasonable head counts using the actual task, training budget and serving constraints. A short parallel encoder workload has a different performance profile from long autoregressive decode. Small models can benefit from sharing under suitable contexts, and large models do not all require the same eight groups. Quality, cache capacity, kernel availability and distributed layout jointly determine the choice.

Measure prefill and decode separately, with stated batch size, sequence lengths, dtype, device, kernels and synchronization. Distinguish latency per request from aggregate tokens per second and account for total model state, not only K/V payload. The transparent NumPy/PyTorch programs here teach correctness. Their Python loops and cache concatenation are not production scheduling advice; repeated concatenation copies existing storage, whereas a bounded or paged implementation can manage append positions directly.

Finally, the [next topic, Multi-Head Latent Attention](/learn/path/full-curriculum/multi-head-latent-attention-mla?module=deep-learning-fundamentals), changes a different representation choice: it compresses cache content into a latent representation and carefully handles positional components. Its actual computation cannot be inferred solely from a smaller-looking parameter count. The head/group/cache distinctions here are the prerequisite for understanding that design.

## 8. Practice: reason about new inputs and constraints

Try each question before opening its hint or solution. Exercises 1–5 check the first-pass route; 6–9 use deeper reasoning. The downloadable programs can verify your calculations, but calculate or explain the mechanism.

### 1. Count readers and stored representations

A model has 12 query heads and 3 KV heads, using contiguous equal groups. Which KV head does query head 9 read? How many distinct attention distributions does one query position produce? If the key at one memory position in group 2 changes, which query heads can be affected? Must all of them change numerically?

<details><summary>Hint</summary>

Compute the number of readers per group, then apply integer division to the zero-based query-head index.

</details>

<details><summary>Solution</summary>

Group size is $R=12/3=4$. Head 9 reads group $\lfloor9/4\rfloor=2$. There are 12 attention distributions, one per query head. The changed group-2 key can affect heads 8,9,10,11. A particular score can remain unchanged if its query is orthogonal to the key edit; later output effects can also cancel. Heads outside that group are unaffected by this isolated projected-key change at this operation. A changed original input may influence other projections too, so that is a different intervention.

</details>

### 2. Sharing does not mean agreement

Two queries q0=`[1,0]` and q1=`[0,1]` share keys `[[2,0],[0,2]]` and values `[[1,0],[0,3]]`, with head width 2. Compute each distribution and output. Then change the second value to `[2,3]`. Which weights change, and by how much do the output x coordinates change?

<details><summary>Hint</summary>

The nonzero scaled logit is $2/\sqrt2=\sqrt2$. Let $p=e^{\sqrt2}/(1+e^{\sqrt2})$.

</details>

<details><summary>Solution</summary>

$p\approx0.804430$. The weights are `[p,1−p]` and `[1−p,p]`; outputs are approximately `[0.804430,0.586711]` and `[0.195570,2.413289]`. Changing a value leaves weights unchanged. The x-coordinate increases are $2(1-p)\approx0.391141$ and $2p\approx1.608859$. Distinct queries read the same value edit in different proportions.

</details>

### 3. Count bytes without confusing units

There are two independent requests, 12 layers, 3 KV heads, prefix length 1024, key width 64, value width 32 and two bytes per stored number. Calculate the K/V payload in bytes and MiB. The model has 12 query heads: what would the payload be under MHA with those widths? Name two reasons the measured physical allocation might exceed the compact payload.

<details><summary>Hint</summary>

Use $BNLH_{kv}(d_k+d_v)s$. A MiB is $2^{20}$ bytes. The value width does not have to equal the key width.

</details>

<details><summary>Solution</summary>

The payload is $2\times12\times1024\times3\times96\times2=14,155,776$ bytes, or 13.5 MiB. MHA has four times as many K/V heads, so it needs 54 MiB for these tensors. Reserved/padded capacity, page rounding, quantization metadata, duplicated prefixes or replication across devices can add physical storage; position/cache bookkeeping and other model state also need memory. Do not label this payload as the entire application's memory usage.

</details>

### 4. Repair a non-square causal mask

A decode chunk has query positions `[5,6]` and stored key positions `[0,1,2,3,4,5,6]`. Write its legal mask. A developer passes only `is_causal=True` to an API documented to use an upper-left triangle for non-square inputs. What relation does that represent instead? Would GQA head sharing fix the error?

<details><summary>Hint</summary>

Compare actual logical positions, then separately consider local row indices 0 and 1.

</details>

<details><summary>Solution</summary>

The correct rows are `[1,1,1,1,1,1,0]` and `[1,1,1,1,1,1,1]`. An upper-left triangle gives `[1,0,0,0,0,0,0]` and `[1,1,0,0,0,0,0]`, as if queries were at local positions 0 and 1. Supply the intended explicit mask or an API-specific offset-aware causal bias. The head axis and causal-position axis are independent; fewer KV heads cannot repair the wrong legal set.

</details>

### 5. Design a fair conversion check

After mean-pooling an MHA checkpoint, a researcher trains only the converted model for 100 more updates, then compares it with the original checkpoint. What extra control would help separate conversion recovery from additional training? Which data can select the checkpoint? Under what special condition is mean conversion exactly function-preserving before any further update?

<details><summary>Hint</summary>

Match the continuation budget, keep test data outside selection, and think about already-tied projection heads.

</details>

<details><summary>Solution</summary>

Continue an unchanged MHA branch for the same declared additional-update protocol, and report both models' before/after states. Use validation data for checkpoint selection and held-out test data for final assessment, with group/duplicate boundaries appropriate to the task. Mean conversion preserves the function if original K/V maps, including biases, are identical inside each target group and all relevant positional, mask and output conventions remain consistent. General group averaging is not lossless. Equal update counts also do not necessarily mean equal FLOPs, so state the comparison budget honestly.

</details>

### 6. Accumulate a shared value gradient

Two readers attend to one shared scalar value with weights 0.2 and 0.7 at the selected memory position. Their upstream output derivatives are 3 and −1. There is one query position per reader, and the loss already contains any intended averaging. What is the shared value gradient from these two uses? Would dividing by group size be correct?

<details><summary>Hint</summary>

Each use contributes attention weight times its upstream derivative. Add the contributions.

</details>

<details><summary>Solution</summary>

The gradient is $0.2\times3+0.7\times(-1)=-0.1$. The contributions partly cancel. An extra division by two would change the defined objective's gradient; sharing sums uses. Any averaging desired by the loss must be specified in that loss and propagated normally.

</details>

### 7. A cache ratio is not a timing result

In an explicitly assumed timing model, 40% of a step is work that would become four times faster, while the remaining 60% is unchanged. Calculate the total speedup. If compact KV storage also falls fourfold, may you report the calculated time as measured GPU performance?

<details><summary>Hint</summary>

Normalize the original time to 1 and add the new times of the two parts.

</details>

<details><summary>Solution</summary>

New time is $0.6+0.4/4=0.7$, so speedup is $1/0.7\approx1.429$. It is a consequence of the assumed decomposition, not a measurement. Real kernels may change reuse, occupancy and communication, and the fourfold storage reduction does not establish that the affected time portion accelerates fourfold. Label modeled and measured quantities separately.

</details>

### 8. Account for distributed replicas

A model has 32 query heads, 2 KV heads and 8 tensor-parallel devices. Under a simple layout that puts four queries on each device and replicates each KV head on the four devices reading it, how many physical KV-head copies exist in aggregate? How does that compare with its logical head count? Give another layout choice whose cost would need separate analysis.

<details><summary>Hint</summary>

Each of the two logical KV heads has four physical copies. Avoid assuming that model parallelism always divides every tensor evenly.

</details>

<details><summary>Solution</summary>

There are $2\times4=8$ physical KV-head copies, four times the logical count. The local query computations remain distinct, but the shared memory has been replicated. A communicating or differently sharded cache could reduce replication while adding communication or another scheduling constraint. The correct accounting depends on that algorithm, not on head count alone.

</details>

### 9. Interpret a forecast without leaking its future

An observed prefix ends at point 19. You want five future predictions. A program computes the first forecast, then feeds the true point 20 before computing its second forecast. Has it performed a five-step generated rollout? If a learner edits observed point 7, can the old cache safely be reused unchanged? Explain both dependency errors.

<details><summary>Hint</summary>

Identify what information is available at the observation boundary, then what the cache's old states were functions of.

</details>

<details><summary>Solution</summary>

Feeding true point 20 makes the later calculation teacher-forced, not an autonomous rollout from the original boundary. A generated rollout feeds the model's own forecast as the next input. Editing point 7 changes projections and, in a multilayer causal network, can affect later hidden states and caches. Recompute the affected prefix states or use a correctly designed invalidation strategy; an old cache is valid for its original prefix, model and position convention. Caching accelerates an unchanged computation, not a changed history.

</details>

## What comes next?

You can now distinguish query diversity from stored-head diversity, trace exact grouping and masks, count compact cache payload and evaluate what conversion actually preserves. Continue to [Multi-Head Latent Attention](/learn/path/full-curriculum/multi-head-latent-attention-mla?module=deep-learning-fundamentals). It compresses the cached representation through learned latent coordinates, so its score/value reconstruction and rotary components need a new derivation rather than a relabeled head-count diagram.

## References and another way to learn

- **Original mechanism and bandwidth reasoning:** [Shazeer, Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150). The full nine-page source was inspected, including batched/incremental `einsum` definitions, MQA construction, performance assumptions and translation/language-model protocols. Read §§2–3 for the computational argument, then §4 to see why a particular measured speedup and a particular quality metric must stay attached to their experiment.
- **Grouping and checkpoint conversion:** [Ainslie et al., GQA](https://arxiv.org/html/2305.13245v3). The complete methods, experiments, ablations, limitations and stability appendix were inspected. The section structure is a useful deeper reading route: mean conversion, continued training, grouped heads, then actual comparisons. The [official EMNLP presentation](https://aclanthology.org/2023.emnlp-main.298.mp4) is linked from the [ACL paper record](https://aclanthology.org/2023.emnlp-main.298/). Its provenance and associated full paper were checked; the video was not independently watched or transcribed for this packet. Use it as an optional author presentation, not as separate verified experimental evidence.
- **A short alternative explanation:** [Sebastian Raschka's GQA guide](https://sebastianraschka.com/llms-from-scratch/ch04/04_gqa/). The full inspected guide connects shared heads, smaller cache state and combinations with other architecture choices. It is a compact conceptual recap rather than this lesson's detailed conversion/masking proof. No popularity statement in that guide substitutes for a checkpoint's actual architecture disclosure.
- **An illustrated attention prerequisite:** [3Blue1Brown's Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention/) has a video and written adaptation. The Q/K/softmax/value-mixing portions were inspected during the preceding attention/position packets and are reused as background reading. It explains why different queries can read shared information differently; it does not teach GQA cache implementation. Its visual orientation may use query columns rather than this lesson's query rows.
- **Versioned operator reference:** [PyTorch 2.14 scaled dot-product attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html). The actual shape, grouping, dropout, boolean-mask, non-square-causal and backend sections were inspected. Match the documentation to the version you run. [Our complete native checks](mechanism-calculations.py) record what was executed locally.
- **Read a production attention layer:** [Transformers' Llama attention source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py). The projection, native-head rotation, cache update, repeat helper and backend dispatch were inspected. The branch is mutable; pin a version for reproduction. The lesson's complete model is intentionally much smaller and independently implemented.
- **Kernel layout and mask conventions:** [FlashAttention usage and cache documentation](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention). The GQA head mapping, unequal-length causal alignment, cache update and rotary-layout contracts were inspected. These details are valuable after the transparent implementation; no FlashAttention installation or GPU timing was performed here.
- **A concrete combined architecture:** [Mistral 7B §2](https://arxiv.org/html/2310.06825v1#S2). The actual architecture table, rolling-cache and prefill/chunking explanation show that GQA and a local window address different dimensions. Read the checkpoint's own scope rather than generalizing its reported outcomes to every context or model.
- **Offline real-data work:** [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), [data/protocol provenance](data-provenance.md), [complete forecasting/conversion program](author-calculations.py), [recorded outcomes](author-results.json) and [independent fixtures](mechanism-fixtures.json). The small study makes conversion and actual cached outputs inspectable without a large-model download.
