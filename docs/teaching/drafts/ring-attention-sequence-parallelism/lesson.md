# Ring Attention & Sequence Parallelism: one sequence, several devices

**Explore as you read.** Edit tiny Q/K/V and block ownership, merger order, causal positions, communication budgets and supported trajectory points. Show stable summary accumulation, legal work grid, circulating ownership, payload timeline and current output/error against the dense reference. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to separate mathematical equivalence from communication cost and identify invalid identity, masking or state-carry changes.


Suppose four devices must read one long document. Giving each device a different quarter is easy. Letting a word near the end use information from the beginning is harder: that information now lives elsewhere.

**Ring Attention keeps each device's questions in place and circulates the information those questions need.** Each device gradually builds the same attention result it would obtain if the entire sequence were available locally. The main change is where data lives and when it moves.

In [Hyena](/learn/path/full-curriculum/hyena-long-convolution-models?module=deep-learning-fundamentals), the mixing operation changes. Here we preserve dense softmax attention and distribute its execution. That distinction matters: a systems optimization should first demonstrate equivalence, then establish its resource benefit.

You need the idea of an attention-weighted average and basic array shapes. We refresh both below. On a first pass, follow the ownership picture, the four-value calculation, the causal-work grids and the real movement example. The backward derivation and communication model provide a deeper route for implementing or diagnosing a distributed system.

## 1. What is being divided?

For one attention head, each sequence position supplies three vectors:

- A **query** asks what information this position needs.
- A **key** describes information against which a query can be compared.
- A **value** contains the information to combine.

With query matrix Q, key matrix K and value matrix V, attention computes

\[
S=QK^T/\sqrt d,\qquad A=\operatorname{softmax}(S+M),\qquad O=AV.
\]

Here d is the query/key channel count. Softmax normalizes each query row. The mask M is zero for allowed query–key pairs and negative infinity for forbidden pairs; those pairs receive zero weight. We will reserve A for attention probabilities and P for device count.

If all positions may interact, an L-position sequence has L² pairs per head. Increasing L from 8 to 16 creates four times as many pairs. Avoiding storage of those scores does not remove their computation.

**FlashAttention** tiles this operation within a device, computes stable partial summaries and reconstructs needed intermediates during backward rather than storing the entire attention matrix. **Ring Attention** adds a partition across devices. They address different levels of the memory hierarchy and can be combined. [FlashAttention, especially §3 and Appendix B](https://arxiv.org/pdf/2205.14135)

For a batch of B sequences, Hq query heads, Hkv key/value heads and head width d, Q has shape B×Hq×L×d, while K and V have B×Hkv×L×d. Standard multi-head attention has Hkv=Hq. In [GQA/MQA](/learn/path/full-curriculum/grouped-query-attention-gqa-multi-query-attention-mqa?module=deep-learning-fundamentals), several query heads share a key/value head. The saved KV payload can therefore be smaller than Q without removing query heads.

An attention matrix stored in float32 would occupy 4BHqL² bytes. The Q, K, V and O arrays instead grow linearly in L. Linear growth can still be large. At B=1, L=1,000,000, Hq·d=8192 and two bytes per element, the four equal-width MHA arrays alone total 65,536,000,000 bytes, about 61.0 GiB. This calculation does **not** include weights, optimizer state, other layer activations or temporary buffers. It also does not establish a universal maximum context length for an “80GB GPU”: dimensions, GQA, precision and the rest of the workload change the answer.

**Visual — memory inventory and token ownership.** Put sequence positions on a horizontal strip. Under it, separate Q/K/V/O storage, a hypothetical score matrix, and model-state storage. Splitting the strip moves position-owned arrays into four devices; the model-state bar does not automatically shrink. The score matrix should be shown as a computation grid, with only the current tile filled as stored data.

## 2. Keep the queries; move the key/value blocks

Use eight positions and four devices. Device 0 owns positions 0–1, device 1 owns 2–3, device 2 owns 4–5 and device 3 owns 6–7. Each has local Q, K and V for its positions.

At round 0, every device combines its own queries with its own key/value block. It also prepares to send that block to its neighbor. At the next round, it combines the **same queries** with the incoming block. After four computations, it has visited every key/value block once.

| Round | Device 0 uses KV from | Device 1 uses KV from | Device 2 uses KV from | Device 3 uses KV from |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1 | 2 | 3 |
| 1 | 3 | 0 | 1 | 2 |
| 2 | 2 | 3 | 0 | 1 |
| 3 | 1 | 2 | 3 | 0 |

This table uses sends from rank i to rank (i+1) mod P. Rank means a process's index in the participating group. There are P block computations and P−1 necessary transfers to make every block available after its initial local use. Our forward schedule omits an unused final circulation. A library may circulate once more for convenient buffer ownership; account for the schedule actually implemented.

Why circulate K and V together? A key identifies a value. If the two lose alignment, the weighting can be numerically valid while selecting the wrong information. Position IDs and document membership must travel with the records, too.

Each query owner finally stores only its own output rows. Concatenating outputs in **logical sequence order**, or keeping them sharded for a positionwise operation, preserves the next layer's meaning. Concatenating rank order is correct only when rank ownership is contiguous and ordered that way.

The research design combines this circulation with blockwise computation and overlapping communication. Its empirical context-length and utilization results apply to its reported setups; “near-infinite” describes a scaling ambition, not unbounded resources or unlimited learned understanding. [Liu, Zaharia and Abbeel, Ring Attention](https://arxiv.org/html/2310.01889v4)

**Pause and predict:** if we reverse circulation but preserve every record's identity and visit every block once, should the mathematical answer change?

<details><summary>Reveal the reasoning</summary>

No. We change the order of adding the same contributions. In exact arithmetic, the answer is identical. Floating-point addition and rescaling can produce small rounding differences. Reversing the ring is not inherently a positional error; relabeling a received block as though it were local is.

</details>

## 3. A block's answer is not enough

It is tempting to calculate attention separately within each KV block and average the resulting outputs. That loses how much probability mass each block should receive.

Consider one query with scores [0, ln2, ln4, 0] and scalar values [2, 8, 1, −2]. Its unnormalized weights are [1,2,4,1]. The correct weighted average is

\[
o=\frac{1(2)+2(8)+4(1)+1(-2)}{1+2+4+1}=\frac{20}{8}=2.5.
\]

The first two records alone give 18/3=6. The last two alone give 2/5=0.4. Averaging those block outputs gives 3.2, which is wrong. Their denominators are 3 and 5, not equal. Combining them with those weights gives (3·6+5·0.4)/8=2.5.

Large scores introduce another problem: computing exp(1000) overflows ordinary floating point. Subtracting the same maximum from every score preserves softmax, because the common exponential factor cancels between numerator and denominator.

We can apply that correction incrementally. For each query, retain just three quantities:

\[
m=\max_{j\text{ visited}}s_j,\quad
\ell=\sum_{j\text{ visited}} e^{s_j-m},\quad
u=\sum_{j\text{ visited}}e^{s_j-m}v_j.
\]

m is a scalar score, ℓ a scalar normalizer, and u a vector with the value width. The output is u/ℓ. These are summaries of all visited keys, not trainable model parameters.

When a new block has maximum b, set m′=max(m,b). Convert the old summary to the new exponential reference with α=exp(m−m′), then add the new contributions:

\[
\ell'=\alpha\ell+\sum_{j\in\text{new}}e^{s_j-m'},\qquad
u'=\alpha u+\sum_{j\in\text{new}}e^{s_j-m'}v_j.
\]

The common scale matters for **both** accumulated quantities. It is not a correction for accidentally counting a token twice. The online-normalizer construction also admits associative merging of independently computed summaries. [Milakov and Gimelshein, §3 and §3.1](https://arxiv.org/pdf/1805.02867)

Here is our calculation in the stable representation:

| State | m | ℓ | u | u/ℓ |
| --- | --- | --- | --- | --- |
| After records 0–1 | ln2 | 1.5 | 9 | 6 |
| Rescale old summary to ln4 | ln4 | 0.75 | 4.5 | 6 |
| Add records 2–3 | ln4 | 2 | 5 | 2.5 |

The rescaling line changes the representation without changing its average. The second block contributes 1.25 to ℓ and 0.5 to u. If we neglect rescaling, we obtain 9.5/2.75≈3.4545 instead.

Initialize m=−∞, ℓ=0, u=0. The first valid block gets α=0. A completely masked block contributes nothing. If a query has no valid key in any block, softmax is undefined as a probability distribution: explicitly return the chosen zero-output convention and mark the row invalid, rather than turn 0/0 into a meaningful prediction. The reference program uses zero output and log-normalizer −∞ for that case.

**Investigation — summary merger.** Edit actual scores and values, inspect the final weighted average, and show immediately block denominators and numerator contributions. Rearrange block arrival order or add 1000 to every score as null controls. A second unsolved case uses scores [ln3,0,ln2,0] and values [4,−1,7,2]. The interface should compute from the edited records, not replay a canned output.

## 4. Global meaning must survive local storage

A causal language model lets query position q attend only to keys k≤q. In distributed storage, a record's local array index is not its position in the document.

Device 2's first query may be global position 4. When keys 0–1 arrive, both are allowed. Applying a fresh lower-triangular mask to the local 2×2 block would incorrectly forbid key 1 for that query. Instead evaluate the global condition on each pair.

This distinction becomes even more important when records are deliberately interleaved to balance work. A useful record has at least a payload and its logical identity:

`(document_id, global_position_within_document, key_vector, value_vector)`

With packed documents, the allowed condition becomes `same_document AND key_position <= query_position`. Padding adds another valid-record condition. Without document membership, the beginning of one document can attend into an unrelated previous document even when every numeric position comparison succeeds.

Position encodings must use the same identities. With a two-dimensional rotary pair, unrotated vectors (1,0), angular frequency 1, query position 5 and key position 1, the unscaled rotated dot product is cos4≈−0.6536. Resetting those positions to local indices 0 and 1 changes it to cos1≈0.5403. Moving already correctly rotated Q/K records is safe; computing rotations from wrong indices changes the model. Review [Positional Encodings](/learn/path/full-curriculum/positional-encodings-sinusoidal-learned-rope-alibi?module=deep-learning-fundamentals) for why the relative phase appears.

Training targets have an analogous boundary. Suppose input tokens are [10,11,12,13,14,15] and the task predicts the next token. The targets are [11,12,13,14,15,ignore]. Splitting first and shifting each half independently produces [11,12,ignore,14,15,ignore], silently losing the target across the partition boundary. Construct logical next-token targets before sharding, with document boundaries respected, or explicitly exchange boundary information.

Finally, averaging local mean losses is wrong when ranks contribute different valid-token counts. If one rank has three valid tokens with mean loss 2 and another has one with mean loss 6, the global mean is (3·2+1·6)/4=3, not 4. Gradient normalization and the framework's sum/average collectives must implement that same global objective. The official DeepSpeed integration explains target shifting and valid-token-weighted loss aggregation. [DeepSpeed Ulysses integration, “Nuances” and “Loss averaging”](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/)

**Visual — identity travels with the packet.** Animate only the storage transfer; keep document/position labels attached. A two-document grid should visibly distinguish forbidden cross-document cells from forbidden future cells. A separate boundary arrow shows the target that local shifting would discard.

## 5. Equal token counts can hide unequal work

Consider causal attention over positions 0–15, split four ways. Query 0 has one allowed key; query 15 has sixteen. Four consecutive queries near the end therefore cost more than four near the beginning.

With c=L/P positions per device, zero-based contiguous rank i has

\[
W_i=ic^2+\frac{c(c+1)}2
\]

allowed query–key pairs. The first term counts all previous chunks; the second counts the local triangle. For c=4, totals are [10,26,42,58]. The busiest device has 5.8 times the first device's useful pair work.

The individual block counts make the imbalance visible:

| Query owner \ KV owner | 0 | 1 | 2 | 3 | Total |
| --- | --- | --- | --- | --- | --- |
| 0 | 10 | 0 | 0 | 0 | 10 |
| 1 | 16 | 10 | 0 | 0 | 26 |
| 2 | 16 | 16 | 10 | 0 | 42 |
| 3 | 16 | 16 | 16 | 10 | 58 |

A block whose every pair is masked can be skipped. A mixed block may still execute entire matrix tiles, so useful pairs are not automatically executed FLOPs.

**Striping** assigns rank i positions i, i+P, i+2P, and so on. For our example, rank 0 receives [0,4,8,12], rank 1 [1,5,9,13], and so forth. Each owner now has early and late queries. Its useful work is

\[
W_i=c(i+1)+\frac{Pc(c-1)}2.
\]

The totals become [28,32,36,40]. Each pair of owner blocks contains either c(c+1)/2 or c(c−1)/2 valid cells, depending on their rank relationship. The input sequence has not been semantically reordered; its records have different storage owners. The Striped Attention paper connects this arrangement to per-round work balance and explicitly discusses tile-granularity limits. [Brandon et al., especially §2.2–§4](https://arxiv.org/html/2311.09431v1)

A **zigzag** arrangement divides the sequence into 2P consecutive pieces and pairs an early piece with its mirrored late piece. Here the four owners receive [0,1,14,15], [2,3,12,13], [4,5,10,11], and [6,7,8,9]. All four have 34 useful pairs. The paired-chunk approach appears in PyTorch's context-parallel implementation discussion. [PyTorch authors, “Tensor Sharding”](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082)

Does exact pair balance guarantee the best kernel? No. In our deliberately simple cost calculation, sum the maximum work of any rank in each synchronized round. The resulting critical work is:

| Layout | One-cell tiles | 2×2 tiles | 4×4 tiles |
| --- | --- | --- | --- |
| Contiguous | 58 | 60 | 64 |
| Striped | 40 | 48 | 64 |
| Zigzag | 34 | 36 | 64 |

We count a whole tile whenever any cell is valid, with no special triangular kernel optimization. At 4×4 granularity, all layouts have the same critical executed-cell count. These numbers are exact for this defined toy scheduler. They are not GPU timings, and a production kernel's treatment of diagonal tiles may differ.

**Investigation — causal workbench.** Move labeled positions among equal-size owners. Keep a global causal grid alongside the storage-order grid, round timeline and per-owner counts. Observe whether the change lowers the busiest round's work. Change tile size to see which apparent savings the kernel can actually exploit. Restoring the same ownership under renamed ranks is a null; discarding position labels is a semantic bug, not an optimization.

## 6. Communication can overlap computation, but it takes time

A device can multiply its queries by the current KV block while the next block is in transit. It must not read the receive buffer before the transfer completes or overwrite the send buffer while transport still needs it.

This requires buffer ownership and dependencies, not merely an asynchronous function name. A useful timeline has a compute lane and a communication lane. The next compute block depends on both the current compute finishing and the incoming data becoming ready. An overly early wait serializes work; a missing wait can produce races.

For an explanatory model, let each rank own c positions, batch size be one, and count only the QKᵀ and AV matrix products. A dense block requires approximately

\[
F_{\text{block}}=4c^2H_qd
\]

floating-point operations when one multiply and one add count separately. Softmax, projections, normalization, mask construction and other layer work are excluded. A key/value transfer sends

\[
S_{\text{KV}}=2cH_{kv}d\,s
\]

bytes, where s is bytes per stored element. K and V account for the leading 2. Sending and receiving the same-size payload are distinct network directions; do not add both and then divide by an already aggregate bandwidth without defining that bandwidth.

Suppose effective compute throughput is F, effective one-direction link bandwidth is R and message latency is a. Then C=Fblock/F and D=a+SKV/R approximate one round's compute and transfer times. Under ideal independent overlap,

\[
T_{\text{serial}}=PC+(P-1)D,\qquad
T_{\text{overlap}}=C+(P-1)\max(C,D).
\]

The final compute still has to finish. With C≥D this model hides transfer time; with C<D it exposes network waits. Contention, launch overhead, synchronization, nonuniform causal work and resource interference can all make the measured time worse.

Here are **calculated hypothetical values**, not specifications or measurements for any GPU: P=4, Hq=8, Hkv=2, d=64, s=2 bytes, F=100×10¹² FLOP/s, R=50×10⁹ bytes/s and a=2 microseconds.

| Positions per rank c | Compute C, µs | Transfer D, µs | Serial total, µs | Ideal overlap total, µs |
| --- | --- | --- | --- | --- |
| 128 | 0.336 | 3.311 | 11.274 | 10.268 |
| 1024 | 21.475 | 12.486 | 123.357 | 85.899 |
| 4096 | 343.597 | 43.943 | 1506.219 | 1374.390 |

The rows have different **global sequence lengths**. They show increasing arithmetic intensity, not a claim that a larger input runs faster. Compute grows quadratically in c while transfer payload grows linearly; small shards can leave too little computation to cover communication.

Memory accounting needs equally explicit boundaries. For c=1024 in this example, Q is 1,048,576 bytes, one KV block is 524,288 bytes and a next-block receive buffer adds another 524,288. A float32 accumulated numerator is 2,097,152 bytes; two float32 row statistics add 65,536. If output is distinct it adds 1,048,576. An illustrative 128×128 float32 score tile for each of eight heads adds 524,288. These listed allocations total 5,832,704 bytes.

That is a **forward allocation model**. Aliasing or kernel fusion can change it; backward may retain owned K/V separately and requires gradients and saved/recomputed activations. The CPU reference stores entire arrays and whole local score blocks, so it does not achieve this modeled device footprint. Neither calculation is measured peak GPU memory.

**Investigation — schedule and capacity calculator.** Edit actual dimensional and hypothetical hardware parameters, inspect which lane limits a round, then inspect the dependency timeline and byte inventory. A “keep global length fixed” mode must reduce c as P rises. A “keep local length fixed” mode must visibly increase global length. Never label a calculated line “benchmark” or use a real GPU name with guessed measurements.

## 7. What scales when we add devices?

There are several different questions hidden inside “does it scale?”

With **fixed global L**, adding ranks gives each rank fewer queries. Dense attention matrix-product work per rank is about 4BL²Hq d/P. This is strong scaling. Eventually messages, synchronization and too-small kernels limit speedup. In our hypothetical model with L=4096, moving from P=4 to P=16 does not give a fourfold speedup: ideal overlap time moves from about 85.90 to 70.66 microseconds because the shards become communication-bound.

With **fixed local c** and L=Pc, activation capacity can grow with P. However, each query must now visit P KV blocks. Per-rank attention work grows with P; aggregate work grows with P². This weak-scaling setup does not keep step duration constant.

For a **fixed dataset token budget**, doubling context length means processing half as many sequences. Attention work per sequence grows fourfold, so the attention portion of total dataset work doubles. Positionwise projection/MLP work per token is approximately unchanged. State which quantity is held fixed before interpreting a scaling curve.

The distinction also prevents a learning mistake: fitting a million-token sequence in memory does not demonstrate that a model learned useful million-token dependencies. Position-encoding behavior, training lengths, data quality, optimization and task evaluation remain necessary. A retrieval test measures a particular ability, not end-to-end comprehension of every long document.

## 8. Ulysses and the overloaded term “sequence parallelism”

Ring circulation is one way to distribute attention. **Ulysses** instead changes which tensor dimension is sharded around attention.

Start with L/P positions and all H heads per rank. An all-to-all operation routes head slices so each rank obtains all L positions for H/P heads. It computes attention on those heads, then another all-to-all restores the sequence partition. With L=8, H=4 and P=2:

| Stage | Rank 0 | Rank 1 | Elements per rank, per channel |
| --- | --- | --- | --- |
| Before resharding | Tokens 0–3, heads 0–3 | Tokens 4–7, heads 0–3 | 16 |
| During attention | Tokens 0–7, heads 0–1 | Tokens 0–7, heads 2–3 | 16 |
| After resharding | Tokens 0–3, heads 0–3 | Tokens 4–7, heads 0–3 | 16 |

Holding the whole sequence for **fewer heads** does not restore the original unsharded all-head payload. This is why Ulysses is not limited to the unchanged single-device context capacity. It can also use efficient local attention kernels. [DeepSpeed-Ulysses, §3 and §4.1](https://arxiv.org/html/2309.14509v2)

For equal MHA heads and ideal even partitioning, the before/after per-rank payload for one tensor is BLHd/P elements. Each rank sends a fraction (P−1)/P of that to others in one reshard. Q, K, V and O together therefore send 4BLHd·s·(P−1)/P² bytes per rank for these two forward reshard stages. This arithmetic counts application-level nonlocal bytes, not physical network hops, switch contention or a collective's actual schedule.

For our Ring schedule, forward sends per rank total 2(P−1)B(L/P)Hkv d·s bytes. Comparing these expressions helps identify a tradeoff, but it is not a universal speed ranking. Ring may fit a topology or head arrangement better; all-to-all may move less data but demand a different network pattern. Hybrid approaches can use multiple mesh dimensions.

Head partitioning has practical constraints. H must be divisible by P for the simple equal Ulysses transformation above. GQA complicates matters: with Hq=8, Hkv=2 and P=4, four disjoint owners cannot each receive a nonempty separate KV-head subset. A capable implementation may replicate or specially route KV heads, use a hybrid partition, or limit that mesh dimension. “Impossible for every implementation” is too strong; “unchanged MHA resharding always works” is also wrong.

Terminology depends on the framework. In Megatron, earlier **sequence parallelism** divides certain activations, such as normalization/dropout regions, in conjunction with tensor parallelism. **Context parallelism** extends partitioning across the sequence through the network, with additional attention communication. Inspect the implementation rather than assuming every `sequence_parallel` option means a KV ring. [Megatron Core context parallelism overview](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

Other mesh dimensions answer different ownership questions:

| Strategy | What is divided? | What must still be coordinated? |
| --- | --- | --- |
| Ordinary data parallelism | Different examples among model replicas | Parameter gradients |
| Tensor parallelism | Parts of a layer's operations/weights | Partial layer results |
| Pipeline parallelism | Different depth stages | Activations and gradients between stages |
| Context parallelism | Positions of the same sequence | Cross-position operations and shared-weight gradients |
| Expert parallelism | Different experts | Routed tokens and expert results |
| FSDP/ZeRO | Model-state storage across a chosen group | Parameter availability and gradient/optimizer ownership |

A simple orthogonal DP2×TP2×PP2×CP2 mesh has 16 ranks. It does not mean sixteen independent examples. A CP group participates in the same sequence, and any replicated shared parameters require their contributions to be combined consistently. Real frameworks may overlap or combine groups for model-state sharding; that must be reflected in the actual communication plan.

**Visual — tensor reassembly puzzle.** Give every token/head cell a persistent label. Move cells from sequence-owned boards to head-owned boards and back. Learners should verify both element count and identity. A wrong inverse permutation can retain all shapes and still corrupt results.

## 9. Training: gradients must get home

Correct forward values do not guarantee correct training. A key on rank 0 may affect queries owned by all four ranks. Its gradient must include every such contribution.

For a loss with incoming output gradient G=dLoss/dO, the attention derivatives are

\[
dV=A^TG,\qquad dA=GV^T,
\]
\[
dS_{ij}=A_{ij}\left(dA_{ij}-\sum_k A_{ik}dA_{ik}\right),
\qquad dQ=dSK/\sqrt d,\quad dK=dS^TQ/\sqrt d.
\]

The subtraction expresses competition within each query's softmax row: increasing one score shifts probability away from others. For the four-value example and scalar upstream gradient 1, dV is [1,2,4,1]/8 and dS is [−0.0625,1.375,−0.75,−0.5625]. The score derivatives sum to zero because adding a common score offset changes nothing.

We can reconstruct a tile of A from its scores and the saved row log-normalizer z=m+lnℓ:

\[
A_{ij}=e^{S_{ij}-z_i}
\]

for allowed entries. Also, the row subtraction term simplifies to the dot product G_i·O_i. Thus backward can recompute small probability tiles without retaining the full L×L probability matrix.

At a query owner, contributions accumulate into local dQ. For each visiting KV block, contributions to dK and dV must be reduced across all query owners and returned to the correct original owner. A central sum in our CPU program demonstrates the needed arithmetic; a distributed implementation must supply the transport and synchronization. Its backward communication count is not simply the forward P−1 transfers relabeled “backward.”

Dropout adds state. If attention weights are randomly masked, recomputation must reproduce the same mask for the same global batch/head/query/key identities. Changing shard count must not silently change the intended comparison's randomness. A rank-local random stream without a mapping contract can break this. Our reference disables dropout; adding it requires both forward and derivative changes.

The author calculations compared blockwise gradients with direct dense derivatives, Torch automatic derivatives on valid-row cases, and selected finite differences. Maximum dense/autograd discrepancies were below 7×10⁻¹⁶ in float64; selected finite-difference errors were below 1.9×10⁻¹¹. An intentionally incomplete KV-gradient accumulation differed by about 0.865. These checks establish the small reference's arithmetic, not the correctness of an untested multi-device backend.

**Visual — gradient return map.** Reverse the usual focus: select a key owner and light up every query shard contributing to its dK/dV. Reveal partial sums and the owner that must receive the total. This shows why a detached receive or missing reduction may leave forward outputs looking correct while training diverges.

## 10. A real attention layer under a different execution plan

We reuse a frozen two-head classifier from [Self-Attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals). Its input is a real Libras movement trajectory: 45 recorded two-dimensional hand-centroid positions. The task predicts one of 15 movement classes. It is a small educational classification task, not full sign-language translation. [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement)

The model maps each coordinate pair through a learned 2→24 projection and tanh, computes two attention heads of width 12, applies an output projection and residual connection, averages positions, then classifies with a 24→15 layer. All 2,751 parameters are retained. This particular model has bidirectional attention and no explicit position encoding; do not pretend it was trained as a causal language model.

Its earlier training used duplicate-aware sequence-level splitting: 330 distinct trajectories, 220 fitting, 50 validation and 60 assessment. Each of four attention/baseline variants used three declared seeds, and checkpoints were selected using validation macro-F1, then cross-entropy and earlier epoch. We reuse the predeclared seed-101 two-head model; we did not retrain or choose a model to flatter Ring Attention. The complete provenance records the existing fit rather than manufacturing a new training comparison.

Here the experimental question is: **if the learned Q/K/V arrays are divided into 12,11,11,11 positions, does a four-owner calculation preserve output?** The weights and input are held fixed. Both directions of circulation agree with dense attention to floating-point tolerance.

| Source trajectory | Actual class | Predicted class, both executions | Maximum attention-output difference |
| --- | --- | --- | --- |
| 77 | 4 | 5 | 2.22×10⁻¹⁵ |
| 20 | 1 | 2 | 1.78×10⁻¹⁵ |

Both classifications are wrong. That is useful evidence: correct systems execution preserves a model's mistakes too. The first trajectory's class-5 probability is about 0.8111; it is not evidence of calibrated confidence or correct recognition.

Next change trajectory 77's frame-23 x coordinate by +0.10 within [0,1]. The maximum class-probability change is about 0.00960; the attention values change as well. Recompute **both** dense and partitioned paths from the edited points and compare again. For a different task, edit trajectory 20's frame-10 x coordinate by −0.15; its maximum probability change is about 0.00394. Neither activity should automatically declare the original label valid after an arbitrary coordinate edit.

**Investigation — follow a real query around the ring.** The trajectory and Q/K/V owner cards share editable points. Select a head/query and follow incoming keys, weighted contributions and accumulated output. Show two separate live comparisons: changed model output versus the original, and dense/ring disagreement for the same input. Changing direction or ownership preserves the function; moving a real point usually does not.

This CPU experiment makes no GPU throughput or memory claim. It tests two selected real trajectories and controlled edits, not every input a future implementation might receive.

## 11. From a CPU reference to a distributed implementation

The complete program below needs NumPy. It simulates labeled ownership on one CPU, including uneven shards, causal masks and backward accumulation. Run it as `python ring-attention-reference.py`; the author used Python 3.12.14 and NumPy 2.3.5. Its printed forward difference is approximately 2.22×10⁻¹⁶. The separate [partition study](attention-partition-study.py) also needs Torch and the local data/model files; it creates the real-input and derivative evidence in [partition-results.json](partition-results.json). The [systems calculation](systems-calculations.py) computes the ownership and cost tables without GPU dependencies.

Read the loops as an ownership proof. `query_ids` stay fixed for an owner; `key_ids` change at every step. `allowed[np.ix_(query_ids,key_ids)]` preserves global mask meaning. The numerical guard handles empty rows, and the final assignment places output back at its logical positions.

The program intentionally allocates a dense reference and all shards in one process. To turn this into an efficient GPU operation, replace global-array access with actual owned buffers, use tiled/fused attention kernels, transport labeled KV shards, implement backward reduction, and check the result under the chosen dtype and backend.

<!-- The complete executed reference is inserted below; the executable file remains its source. -->

```python
"""Executable CPU teaching reference, not distributed/FlashAttention code.

Inputs are [heads, positions, channels]; Q and K have equal channels.
One sequence, MHA, no dropout or extra positional bias. A boolean mask
is in GLOBAL position order. Empty rows produce zero, with logsumexp -inf.
The study driver separately verifies gradients and real frozen attention.
"""
import numpy as np


def dense_attention(query, key, value, allowed):
    scores = query @ key.swapaxes(-1, -2) / np.sqrt(query.shape[-1])
    scores = np.where(allowed[None], scores, -np.inf)
    maxima = scores.max(axis=-1, keepdims=True)
    safe_maxima = np.where(np.isfinite(maxima), maxima, 0)
    weights = np.exp(scores - safe_maxima)
    totals = weights.sum(axis=-1, keepdims=True)
    weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
    return weights @ value, weights


def ring_attention(query, key, value, ownership, allowed, direction=1):
    """Simulate P rounds; keep queries local, visit each labeled KV shard once.

    ownership is an exact partition of global positions, possibly uneven.
    direction changes visitation, not sequence positions. CPU arrays already
    exist in one process: this proves arithmetic, not distributed peak memory.
    """
    length = query.shape[1]
    if not ownership:
        raise ValueError("At least one owner is required")
    positions = np.concatenate(ownership)
    if (direction not in (-1, 1) or
            any(len(indices) == 0 for indices in ownership) or
            not np.array_equal(np.sort(positions), np.arange(length)) or
            allowed.shape != (length, length)):
        raise ValueError("Need nonempty exact ownership and a global square mask")
    heads = query.shape[0]
    output = np.zeros((heads, length, value.shape[-1]), dtype=query.dtype)
    logsumexp = np.full((heads, length), -np.inf, dtype=query.dtype)
    trace = []
    ranks = len(ownership)
    for rank, query_ids in enumerate(ownership):
        maxima = np.full((heads, len(query_ids)), -np.inf, dtype=query.dtype)
        totals = np.zeros_like(maxima)
        numerator = np.zeros((heads, len(query_ids), value.shape[-1]), dtype=query.dtype)
        for step in range(ranks):
            owner = (rank - direction * step) % ranks
            key_ids = ownership[owner]
            scores = query[:, query_ids] @ key[:, key_ids].swapaxes(-1, -2)
            scores /= np.sqrt(query.shape[-1])
            scores = np.where(allowed[np.ix_(query_ids, key_ids)][None], scores, -np.inf)
            block_maxima = scores.max(axis=-1)
            new_maxima = np.maximum(maxima, block_maxima)
            safe_maxima = np.where(np.isfinite(new_maxima), new_maxima, 0)
            correction = np.exp(maxima - safe_maxima)
            exponentials = np.exp(scores - safe_maxima[..., None])
            numerator = correction[..., None] * numerator + exponentials @ value[:, key_ids]
            totals = correction * totals + exponentials.sum(axis=-1)
            maxima = new_maxima
            trace.append({"query_owner": rank, "step": step, "kv_owner": owner,
                          "query_ids": query_ids.tolist(), "key_ids": key_ids.tolist()})
        output[:, query_ids] = np.divide(numerator, totals[..., None],
                                        out=np.zeros_like(numerator), where=totals[..., None] > 0)
        safe_totals = np.where(totals > 0, totals, 1)
        logsumexp[:, query_ids] = maxima + np.log(safe_totals)
    return output, logsumexp, trace


def blockwise_backward(query, key, value, ownership, allowed, output, logsumexp, upstream):
    """Recompute probabilities from global LSE; accumulate every KV owner's gradients.

    Central CPU accumulation models the required reduction, not its transport.
    """
    query_gradient, key_gradient, value_gradient = [np.zeros_like(x) for x in (query, key, value)]
    scale = np.sqrt(query.shape[-1])
    for query_ids in ownership:
        row_dot = (upstream[:, query_ids] * output[:, query_ids]).sum(axis=-1, keepdims=True)
        for key_ids in ownership:
            scores = query[:, query_ids] @ key[:, key_ids].swapaxes(-1, -2) / scale
            valid = allowed[np.ix_(query_ids, key_ids)][None]
            lse = logsumexp[:, query_ids, None]
            scores = np.where(valid, scores, -np.inf)
            probabilities = np.exp(scores - np.where(np.isfinite(lse), lse, 0))
            probability_gradient = upstream[:, query_ids] @ value[:, key_ids].swapaxes(-1, -2)
            score_gradient = probabilities * (probability_gradient - row_dot)
            query_gradient[:, query_ids] += score_gradient @ key[:, key_ids] / scale
            key_gradient[:, key_ids] += score_gradient.swapaxes(-1, -2) @ query[:, query_ids] / scale
            value_gradient[:, key_ids] += probabilities.swapaxes(-1, -2) @ upstream[:, query_ids]
    return query_gradient, key_gradient, value_gradient


if __name__ == "__main__":
    generator = np.random.default_rng(91)
    query, key, value = [generator.normal(size=(2, 7, 3)) for _ in range(3)]
    ownership = list(np.array_split(np.arange(7), 3))
    allowed = np.arange(7)[None, :] <= np.arange(7)[:, None]
    reference, _ = dense_attention(query, key, value, allowed)
    distributed, _, _ = ring_attention(query, key, value, ownership, allowed)
    reverse, _, _ = ring_attention(query, key, value, ownership, allowed, -1)
    np.testing.assert_allclose(distributed, reference, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(reverse, reference, atol=1e-12, rtol=1e-12)
    print("Dense/ring maximum absolute difference:", np.abs(distributed - reference).max())
    print("Both directions preserve global causal attention.")
```

For a real backend, begin with its maintained end-to-end example. The PyTorch context-parallel tutorial uses an **experimental** context that shards supplied buffers and replaces supported scaled-dot-product attention calls. Its example also restores logical output ordering for comparison. Declaring a `DTensor` shard alone does not automatically implement a correct KV ring. Position-dependent buffers must be included consistently. The author reviewed the 2.14 documentation; the GPU example was not executed here. [PyTorch context-parallel tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)

If writing lower-level communication, PyTorch 2.14's `batch_isend_irecv` takes a list of `P2POp` records and returns request objects; it does not take an `async_op=True` parameter. Requests, stream dependencies and buffer lifetimes must be respected. Blocking “send to next, then receive from previous” on every rank can deadlock in a cycle. Separate the correctness of a communication schedule from its hoped-for overlap. The installed API documentation was inspected for this lesson. [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)

The independent ring-flash-attention project supplies several attention layouts and packed-sequence APIs, but its README also records numerical/buffer limitations and unsupported dropout/window settings. Read those restrictions and the actual version's tests before adoption; a method name does not prove that every mask, dtype or head configuration works. [Project README and tests](https://github.com/zhuzilin/ring-flash-attention)

### Move the buffers between real processes

The arithmetic reference above is deliberately single-process. [distributed_ring.py](distributed_ring.py) is the complete next implementation step: separate PyTorch processes keep local Q/K/V, send K/V to the next rank, receive from the previous rank, and return accumulated key/value gradients to their owners. It uses ordinary `torch.distributed` primitives; it does not import an opaque Ring Attention function. This small CPU/Gloo protocol is also the lowest useful abstraction for learning buffer ownership before a fused GPU backend.

Use a PyTorch 2.14.0 environment with Gloo support and run `torchrun --standalone --nproc-per-node=3 distributed_ring.py` on one machine. The example uses two heads, equal Q/K/V width three, a single causal sequence and no dropout; the sequence length is `2*world_size+1`, making ownership uneven. Each rank knows shard lengths from `all_gather`; global starts determine the causal mask. Padded packets have a common shape for transport, but only the owner's valid rows enter the attention calculation. The complete authored program below has not yet undergone its multi-process implementation run; it does not supply invented output or timing results.

Forward processing needs P block visits and P−1 transfers. Backward processing makes P visits **and P transfers**: the packet contains K, V, dK and dV, and after a complete circuit its partial sums are back at the original owner. dQ stays with its query owner. This is the exact missing operation in a backward implementation that only passes forward parity. The externally supplied `upstream` is ∂L/∂O; a model layer would pass these Q/K/V derivatives through its projection weights using the already taught chain rule.

```python
"""Explicit CPU/Gloo ring attention and owner-returning manual backward.

Launch: torchrun --standalone --nproc-per-node=3 distributed_ring.py
PyTorch 2.14.0 target; float64, one sequence, equal Q/K/V head width,
contiguous possibly uneven nonempty shards, global causal mask, no dropout.
This readable synchronous protocol makes no overlap or throughput claim.
"""
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn import functional as F


def rotate(packet):
    """Send owned bytes onward; receive into distinct storage before reuse."""
    world, rank = dist.get_world_size(), dist.get_rank()
    if world == 1:
        return packet
    received = torch.empty_like(packet)
    requests = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, packet, (rank+1) % world),
        dist.P2POp(dist.irecv, received, (rank-1) % world),
    ])
    for request in requests:
        request.wait()
    return received


def layout(local_length):
    sizes = [torch.empty(1, dtype=torch.int64) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, torch.tensor([local_length], dtype=torch.int64))
    counts = [int(size.item()) for size in sizes]
    if min(counts) < 1:
        raise ValueError("Every rank needs a nonempty shard")
    starts = [sum(counts[:rank]) for rank in range(len(counts))]
    return counts, starts


def block_scores(query, keys, query_start, key_start):
    scores = query @ keys.transpose(-1, -2) / query.shape[-1]**.5
    q_positions = query_start + torch.arange(query.shape[1])
    k_positions = key_start + torch.arange(keys.shape[1])
    return scores.masked_fill(k_positions[None, :] > q_positions[:, None], -torch.inf)


def pack(key, value, maximum, with_gradients=False):
    packet = key.new_zeros((4 if with_gradients else 2, key.shape[0], maximum, key.shape[-1]))
    packet[0, :, :key.shape[1]] = key
    packet[1, :, :value.shape[1]] = value
    return packet


@torch.no_grad()
def ring_forward(query, key, value, counts, starts):
    rank, world = dist.get_rank(), dist.get_world_size()
    maximum = query.new_full(query.shape[:-1], -torch.inf)
    mass = torch.zeros_like(maximum)
    numerator = torch.zeros_like(query)
    packet = pack(key, value, max(counts))
    for step in range(world):
        owner = (rank-step) % world
        keys, values = packet[:2, :, :counts[owner]]
        scores = block_scores(query, keys, starts[rank], starts[owner])
        updated = torch.maximum(maximum, scores.amax(-1))
        safe = torch.where(torch.isfinite(updated), updated, 0.)
        correction = torch.exp(maximum-safe)
        probabilities = torch.exp(scores-safe[..., None])
        numerator = correction[..., None]*numerator + probabilities @ values
        mass = correction*mass + probabilities.sum(-1)
        maximum = updated
        if step+1 < world:
            packet = rotate(packet)
    # The global causal contract gives each query at least its own key.
    output = numerator/mass[..., None]
    return output, maximum+mass.log()


@torch.no_grad()
def ring_backward(query, key, value, output, logsumexp, upstream, counts, starts):
    rank, world = dist.get_rank(), dist.get_world_size()
    query_gradient = torch.zeros_like(query)
    packet = pack(key, value, max(counts), with_gradients=True)
    correction = (upstream*output).sum(-1, keepdim=True)
    for step in range(world):
        owner = (rank-step) % world
        length = counts[owner]
        keys, values = packet[:2, :, :length]
        scores = block_scores(query, keys, starts[rank], starts[owner])
        probabilities = torch.exp(scores-logsumexp[..., None])
        score_gradient = probabilities*(upstream @ values.transpose(-1, -2)-correction)
        query_gradient += score_gradient @ keys / query.shape[-1]**.5
        packet[2, :, :length] += score_gradient.transpose(-1, -2) @ query / query.shape[-1]**.5
        packet[3, :, :length] += probabilities.transpose(-1, -2) @ upstream
        # P transfers, not P-1: complete sums must return to their original owner.
        packet = rotate(packet)
    return query_gradient, packet[2, :, :key.shape[1]], packet[3, :, :value.shape[1]]


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.set_num_threads(1)
        # A small full oracle is created solely for validation, not in either ring routine.
        generator = torch.Generator().manual_seed(71)
        length, heads, width = 2*world+1, 2, 3
        full = [torch.randn(heads, length, width, generator=generator, dtype=torch.float64)
                for _ in range(4)]
        partitions = torch.tensor_split(torch.arange(length), world)
        indices = partitions[rank]
        query, key, value, upstream = [tensor[:, indices].contiguous() for tensor in full]
        counts, starts = layout(len(indices))
        actual, lse = ring_forward(query, key, value, counts, starts)
        gradients = ring_backward(query, key, value, actual, lse, upstream, counts, starts)
        oracle_inputs = [tensor.clone().requires_grad_() for tensor in full[:3]]
        oracle = F.scaled_dot_product_attention(*oracle_inputs, is_causal=True, dropout_p=0.)
        expected_gradients = torch.autograd.grad((oracle*full[3]).sum(), oracle_inputs)
        torch.testing.assert_close(actual, oracle[:, indices], rtol=1e-11, atol=1e-11)
        for actual_gradient, expected in zip(gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected[:, indices], rtol=1e-11, atol=1e-11)
        print("rank", rank, "positions", indices.tolist(), "forward maximum error",
              (actual-oracle[:, indices]).abs().max().item(), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

The local score tile uses O(H c c_max) memory, with O(H c d) local queries/output and O(H c_max d) circulating packet storage. Per-rank full-sequence arithmetic remains O(H c L d); distributing storage does not make dense attention subquadratic. Backward recomputes score tiles instead of retaining all probabilities. The tiny validator separately creates full arrays and uses `scaled_dot_product_attention` plus autograd as an independent oracle; those deliberately small validation allocations are not part of the ring routines' storage bound. No package parity result is claimed until the program actually runs.

`rotate` submits paired send/receive requests together, waits for completion, and only then returns new storage. The immediate wait makes the schedule synchronous; calling an asynchronous API is not evidence of communication overlap. Gloo/CPU proves a different engineering claim from NCCL/CUDA streams, fused kernels or multi-node performance. Those remain specialized production extensions. The concrete transport API and request-lifetime contract are in [PyTorch's distributed reference](https://docs.pytorch.org/docs/2.14/distributed.html).

**Take control.** Run one, two and three ranks. Then change the validation sequence length to eight at three ranks and retain the uneven-shard checks. Finally remove only the last backward rotation to see which owner receives whose partial sums. Repair the error before introducing any faster kernel.

<details><summary>Hint and reasoned solution</summary>

One rank requires no actual transfer but still computes a complete local forward/backward. The shard counts and offsets must change with eight positions; no rank may attend to padded positions. With P−1 backward transfers the packet at a rank is not its original packet after a full circuit, and it is missing the final return even when every query contribution was added. Matching a global gradient sum is insufficient: compare each owner's dK/dV against its exact logical indices, using nonuniform upstream gradients as supplied. A valid extension to packed documents carries document identities and combines a same-document condition with the global causal comparison; using a local triangle alone fails. This change requires explicit metadata transport, not just a new drawing.

</details>

## 12. Long-input applications and one-token decoding

The same ownership problem appears outside a text document. A video clip may contribute a long sequence of patch tokens. A scientific instrument may produce a long sampled trajectory. A learning agent may consume multiple episodes represented by observations, actions and rewards. In each case, the design question is whether distant interactions are valuable enough to justify dense attention and its communication. These are application possibilities, not claims that every such workload should use Ring Attention.

Our hand-trajectory example is a small version of this input/output contract. Expanding to a long video adds data encoding, training and task-evaluation problems that distributing the attention layer cannot solve. A long-context retrieval result should not be silently renamed a general reasoning result.

Inference also contains two different phases. **Prefill** processes many query positions against the prompt, resembling the attention calculation we studied. **Autoregressive decoding** often adds one query per sequence at a time. With only one query, there may be little computation available to hide movement of a large KV cache.

One alternative is to keep KV shards stationary, distribute the new query, compute each shard's local summary (m,ℓ,u), and merge summaries. Choose global maximum m*, rescale each shard's denominator and numerator by exp(m−m*), sum them, and divide. This is the same stable-merge algebra with a different communication plan. The query and result are small relative to a long cache, though collective latency and batching still matter.

Paged cache allocation solves another problem—how cache blocks are stored and reused. It can coexist with distributed ownership. Neither paging nor Ring Attention by itself specifies scheduling, cache eviction, beam reordering or multi-request batching. Those belong to serving-system design.

The useful question is therefore “which information should move for this workload?” Full-prompt attention, a single new token and a training backward pass have different dependency graphs. Do not infer a proprietary model's architecture from its advertised context window or response latency.

## 13. Diagnose a discrepancy before celebrating a speedup

| Symptom | A targeted investigation | What the result would establish |
| --- | --- | --- |
| Outputs disagree only after a shard boundary | Compare global positions, document mask and target alignment | Whether sharding changed the intended sequence |
| Forward agrees, training diverges immediately | Compare dQ/dK/dV and shared-weight gradient normalization | Whether all loss contributions reach their owners |
| NaNs at early causal queries | Inspect a fully masked incoming block and empty-row convention | Whether stable summary updates handle no contribution |
| More ranks make execution slower | Separate C, message latency, transfer time and waits | Whether smaller shards expose communication |
| Memory greatly exceeds an O(L/P) estimate | Inventory saved activations, logits, buffers and aliasing | Which actual allocations the asymptotic slogan omitted |
| Different layout appears fast but changes quality | Compare identical weights/input/masks and logical ordering first | Whether it is still the same attention operation |
| Correctness differs with dropout or resume | Inspect global random identities and saved recomputation state | Whether the same stochastic computation is being compared |

For a meaningful real benchmark, fix model weights, input lengths/batch, attention semantics, dtype, backward setting and hardware topology. Include warmup and proper device synchronization, measure peak memory with defined allocator semantics, and report whether tokens/second means one long sequence or several shorter ones. Count end-to-end time as well as the attention kernel. Preserve failures and out-of-memory cases rather than plot guessed replacements. These are the next implementation steps, not measurements supplied by this content packet.

## 14. Practice: repair the computation, not just the labels

### 1. Merge a new pair of blocks

Scores are [ln3,0,ln2,0] and values [4,−1,7,2]. The first two and last two entries arrive separately. Calculate the final output and explain why averaging the two local outputs is wrong.

<details><summary>Hint</summary>

Use the unnormalized weights [3,1,2,1]. Keep numerator and denominator separately for each block.

</details>
<details><summary>Solution</summary>

The block numerators are 11 and 16; denominators are 4 and 3. Output is 27/7≈3.85714. Local outputs are 11/4 and 16/3; their equal average is not weighted by their probability masses. In the stable maximum-ln3 representation, ℓ=7/3 and u=9, giving the same answer. Reversing arrival or adding a common score constant changes neither exact result.

</details>

### 2. A valid shape, an invalid mask

A query's global position is 5. An incoming block contains keys at positions [0,3,6] from the same document. Which entries are allowed under causal attention? What if key 3 belongs to a different packed document?

<details><summary>Hint</summary>

The local query index is irrelevant. There are two logical conditions to check.

</details>
<details><summary>Solution</summary>

Positions 0 and 3 are allowed; 6 is future. With different document membership, key 3 is forbidden too. Apply same-document and k≤q to the labeled records. A local triangle over array indices can miss both distinctions. An all-masked block contributes zero; it should not reset summaries from earlier valid blocks.

</details>

### 3. Rebalance a smaller sequence

Take L=12 and P=3. Compare per-rank causal pair counts for contiguous chunks of four, striping, and paired early/late two-position pieces. What is invariant across the layouts?

<details><summary>Hint</summary>

Every global query q contributes q+1 allowed pairs. Sum these over each owner's actual positions.

</details>
<details><summary>Solution</summary>

Contiguous totals are [10,26,42]. Striped owners [0,3,6,9], [1,4,7,10], [2,5,8,11] yield [22,26,30]. Zigzag owners [0,1,10,11], [2,3,8,9], [4,5,6,7] each yield 26. All total 78=12·13/2, preserving the same allowed pairs. Only their ownership changes. Tile granularity and communication still determine executed time.

</details>

### 4. Diagnose an overlap claim

Four ranks each require C=4µs per block; each transfer takes D=7µs. Compute serial and ideal-overlap totals. A slide says “communication is free because the calls are asynchronous.” Repair it.

<details><summary>Hint</summary>

There are four compute rounds and three required transfers. The next round must wait for the slower prerequisite.

</details>
<details><summary>Solution</summary>

Serial time is 16+21=37µs. Ideal overlap is 4+3·7=25µs, compared with compute-only 16µs. Overlap helps but exposes 9µs beyond compute-only. Asynchronous submission permits overlap; it neither removes dependencies nor guarantees sufficient bandwidth or compute duration.

</details>

### 5. Check the bytes before choosing a mesh

Use B=2, c=512, Hq=16, Hkv=4, d=64 and two-byte K/V elements. How many bytes does one KV transfer send per rank? For P=8, what is the forward total in our schedule? Would changing only Hq to 32 double these bytes?

<details><summary>Hint</summary>

Count both K and V, batch size, stored head count, width and bytes. Forward sends happen P−1 times.

</details>
<details><summary>Solution</summary>

One transfer is 2·2·512·4·64·2=1,048,576 bytes. Seven sends total 7,340,032 bytes. Changing only query heads does not alter stored KV payload; it changes query/output storage and attention computation. A backend that materializes repeated KV copies would have a different implementation cost and should be identified explicitly.

</details>

### 6. Forward passes; gradients fail

A ring implementation produces correct outputs but accumulates dK only from queries on the key's original owner. Explain the missing dependency and propose a numerical test.

<details><summary>Hint</summary>

Write dK as a sum over query rows. A key can influence more than its owner's rows.

</details>
<details><summary>Solution</summary>

dK=dSᵀQ/√d includes all allowed queries. Remote query owners must contribute to each key's gradient, and their contributions must be reduced back to its owner. Use small asymmetric Q/K/V, a nonuniform upstream gradient and a mask with cross-owner valid pairs. Compare dense and distributed gradients, then perturb one remote key coordinate and estimate the loss derivative by a central finite difference. A test where every mask is strictly owner-local would miss the bug.

</details>

### 7. Preserve the objective at a boundary

One rank has two valid prediction targets with mean loss 1; another has six with mean loss 3. What should the global mean be? Why does discarding one boundary target remain a bug even if both ranks' local code executes successfully?

<details><summary>Hint</summary>

Reconstruct the global loss sum and denominator. The objective is defined over logical tokens.

</details>
<details><summary>Solution</summary>

The correct mean is (2·1+6·3)/8=2.5. An equal average of local means gives 2. Losing a valid next-token target changes both the numerator and denominator of the training objective. Shift targets using logical sequence/document boundaries before partitioning, or exchange the needed boundary target. The gradient reduction must preserve the same weighted global mean.

</details>

### 8. Make a fresh real-input change

Use the retained trajectory 20 with the frozen model. Choose a different point or coordinate edit from the worked case, then predict separately: will model output change, and should dense versus ring disagreement grow materially? Repeat using only a different owner assignment.

<details><summary>Hint</summary>

Separate the function's input from its execution plan. Recompute Q/K/V after a point edit.

</details>
<details><summary>Solution and acceptance criteria</summary>

There is no fixed class answer for an arbitrary edit. Report the changed coordinates, model identity, before/after probabilities and maximum dense/ring difference. The input edit may change scores, values and classification; a small or null effect is valid evidence. Correct implementations should still agree within a justified floating-point tolerance. Ownership-only changes preserve the mathematical function, provided mask/position/record identities and output order remain correct. Explain any discrepancy instead of accepting a favorable class prediction as proof of systems correctness.

</details>

## 15. Continue learning

You are ready to move on when you can explain why block softmax outputs need their normalizers, trace a global mask through a shard transfer, account for both compute and bytes, and describe how a remote query contributes to a key gradient. The next module-order topic is [Advanced Optimizers](/learn/path/full-curriculum/advanced-optimizers-lion-sophia-prodigy-schedule-free?module=deep-learning-fundamentals). It asks how gradients update parameters once the distributed computation has produced the intended gradient.

For deeper GPU work, implement the maintained small distributed example first, then study fused kernels, network topology and profiling. For serving, revisit the one-query case and cache ownership. For model design, compare this exact execution strategy with the different approximations and state representations in sparse attention, Hyena and state-space models.

Useful alternate routes and references:

- [Ring Attention paper](https://arxiv.org/html/2310.01889v4): the primary algorithm, blockwise setup and experimental context. Read §3 with the ownership/merge example here, then distinguish §5's measured setups from extrapolation. Appendix A supplies its JAX forward/backward structure.
- [Online normalizer calculation](https://arxiv.org/pdf/1805.02867): a short mathematical route to stable normalization and parallel summary merging. Useful before implementing custom attention arithmetic.
- [FlashAttention](https://arxiv.org/pdf/2205.14135): §3 explains the memory hierarchy; Appendix B derives forward/backward tiling. This is an advanced kernel-oriented reference, not a prerequisite for the first pass.
- [Striped Attention](https://arxiv.org/html/2311.09431v1): inspect its causal grids and §4 limitations. The paper's performance measurements are configuration-specific; our toy work grid is independently calculated.
- [DeepSpeed-Ulysses](https://arxiv.org/html/2309.14509v2): a different tensor-ownership route. Follow Figure 2, then §3's communication analysis with the token/head puzzle.
- [PyTorch context-parallel tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html): a maintained, executable GPU starting point after the CPU reference. Experimental APIs and supported backends require version checks; its multi-GPU program was read, not run here.
- [Stanford CS336 Spring 2025 course materials](https://cs336.stanford.edu/spring2025/) and [Stanford Online Lecture 7: Parallelism 1](https://www.youtube.com/watch?v=l1RJcDjzK8M): broader distributed-training background to connect the mesh dimensions. Course schedule and official indexed video identity were verified; the complete video was not watched and no timestamp is claimed. This is supporting parallelism background, not a Ring-specific implementation walkthrough.
- [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement): source, attribution and schema for the real example. The [data/provenance record](data-provenance.md) and [complete frozen model](movement-attention-model.json) make the exact input-to-output path reproducible offline.

The visual investigations described here are specifications for later implementation. Their saved calculations support content correctness; they do not imply that browser labs, a distributed backend or a performance benchmark have already been implemented.
