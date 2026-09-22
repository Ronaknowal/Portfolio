# RWKV & Linear Attention Models

**Explore as you read.** Edit keys/queries/values, decay, current-token bonus, memory write/correction inputs and real stream interruptions. Update summary matrices/denominators, present output versus stored history and chronological state together. Compare a continued stream with an explicit reset. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose the correct current-read and future-memory semantics and see what compact state cannot retain.


A conversation can be remembered in two quite different ways. You can keep a transcript and consult individual lines. Or you can maintain a working summary: who is speaking, what they want, which facts changed, and what remains unresolved. A transcript grows. A fixed-size summary must decide what deserves its limited space.

Sequence models face the same choice. An attention model can retain past keys and values and compare a new query with them. A recurrent model updates a state and carries that state forward. **Linear attention and RWKV investigate how much useful memory we can build without repeatedly searching an ever-growing list of past vectors.** Their versions differ in what they store, how they forget, and whether they can replace a particular remembered association.

Suppose a device receives updates: “sensor A reads 2,” “sensor B reads 7,” and later “sensor A now reads 5.” An average, a running sum and an editable key–value memory produce different answers. That small example will help us understand language-model memory, streamed movement classification and the compromises behind efficient sequence processing.

The [preceding state-space lesson](/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals) explained how input-dependent coefficients update a state. Here we examine memory as weighted aggregation and as a small associative map. You need vectors, matrix multiplication and the idea of a learned projection; we refresh the attention operation locally.

**A useful first route:** read sections 1–5, do the summary and weighted-memory investigations, and then run the real-data example in section 8. Sections6–7 explain the newer architectures and hardware choices; take them after you can trace one update. The optional derivations and later exercises provide a deeper pass. By the end you should be able to derive a causal summary, distinguish an exact reformulation from a different attention operator, preserve state across chunks, and explain what an efficiency claim does—and does not—measure.

## 1. What must the memory answer?

Think of a **key** as an address or description, a **value** as information stored with that address, and a **query** as a request to retrieve information. These are learned numeric vectors, not necessarily readable words. The same token can produce all three through different learned matrices. A projection takes an input vector x and computes, for example, q=Wq x. Its weights are learned from a downstream objective.

A dot product q·k is large when the two vectors align in the directions the model has learned to use. It is not automatically a semantic similarity score: training gives those directions their job. A value can contain negative numbers and several channels even when the weight used to combine values is positive.

For one attention head, let q and k have d coordinates and let v have p coordinates. At sequence position t, causal softmax attention computes

y_t = Σ_{i≤t} a_{ti} v_i,
a_{ti} = exp(q_t·k_i / √d) / Σ_{j≤t} exp(q_t·k_j / √d).

The weights are nonnegative and sum to one. The word **causal** means that the answer at t uses only positions through t. The √d factor controls the scale of the dot products; it does not change which keys are stored. For numerical evaluation, subtract the largest permitted score before exponentiating.

**Inline visual — a query visits a transcript.** Show three address–value pairs in a horizontal history. A query arrow reaches each permitted key; three score labels become a normalized weight bar, then three weighted value arrows add at the output. Future entries are outlined and disconnected. Keep the values as visible vectors so the diagram teaches weighted addition rather than “attention looks here.”

For scores [0, log(2), 0] and values [2,8,−1], the normalized weights are [1/4,1/2,1/4] and the output is 4.25. An attention output is usually a mixture, not a command to select one original token. A later query can assign a different mixture to the same stored history.

Now imagine replacing that history with two running totals. If the new query can use those totals to obtain the answer, we have a bounded recurrent memory. The question is not merely whether its output looks similar once. It is **which weighting rule those totals exactly represent**.

A useful distinction throughout this lesson:

| Object | Changes when a new sequence arrives? | Learned by ordinary training? |
| --- | --- | --- |
| Projection and block weights | Normally fixed during evaluation | Yes, using many training examples |
| Sequence state | Yes, at each position | Its update rule is learned; its current contents depend on this sequence |
| Stored transcript keys/values | Appended as tokens arrive | Their generating projections are learned |
| A delta-rule memory matrix | Updated inside the forward computation | Its update parameters are learned; the matrix itself is the current fast state |

Updating a state at inference does not, by itself, mean changing the pretrained model weights.

## 2. A running summary that really equals its attention rule

First change the weighting rule. Choose a feature map φ that maps a query or key to m features. Define κ(q,k)=φ(q)·φ(k), and use this kernel in place of the exponential score:

y_t = [Σ_{i≤t} κ(q_t,k_i)v_i] / [Σ_{i≤t} κ(q_t,k_i)].

We will assume nonnegative weights and a positive denominator. Nonnegative features alone permit a zero vector or disjoint support, so that last condition must actually hold. A zero total weight makes this normalized expression undefined; returning an arbitrary answer would silently invent a different rule.

For readability, write q̄=φ(q), k̄=φ(k). Let columns represent coordinates in the equations below. A state S_t of shape m×p and a vector z_t of length m are sufficient:

S_t = S_{t−1} + k̄_t v_tᵀ,
z_t = z_{t−1} + k̄_t,
y_t = S_tᵀ q̄_t / (q̄_tᵀ z_t),

starting from S_0=0 and z_0=0. The outer product k̄_t v_tᵀ writes one weighted copy of the value into each feature row. The query then combines those rows. The denominator combines the corresponding total weights.

Why is this exact? Expand the numerator:

S_tᵀq̄_t = (Σ_i k̄_i v_iᵀ)ᵀq̄_t
          = Σ_i v_i(k̄_iᵀq̄_t).

The same distributive law changes the denominator into Σ_i k̄_iᵀq̄_t. We have regrouped the arithmetic of the **chosen kernel**. We have not proved that an arbitrary kernel equals softmax.

**Inline visual — accumulate rows, then ask.** Place an m×p memory grid next to a length-m weight column. Animate or step one outer-product write into both. Draw a separate query across rows, ending at a numerator vector and a denominator scalar. Label these shapes explicitly. A separate small expansion shows how one row is the sum of several writes.

### A complete four-update example

Use the following already-mapped features and scalar values. Zeros are allowed here because every actual query still has positive overlap with the accumulated keys.

| t | q̄_t | k̄_t | v_t | output |
| --- | --- | --- | --- | --- |
| 1 | [1,1] | [1,0] | 2 | 2 |
| 2 | [2,1] | [0,1] | 8 | 4 |
| 3 | [1,2] | [1,1] | −1 | 2.5 |
| 4 | [3,1] | [2,1] | 5 | 3 |

After the first two writes, S=[2,8]ᵀ and z=[1,1]ᵀ. The second query produces (2×2+1×8)/(2×1+1×1)=4. At the third write, S becomes [1,7]ᵀ and z=[2,2]ᵀ; the query returns 15/6=2.5. At the end S=[11,12]ᵀ and z=[4,3]ᵀ, so the final answer is 45/15=3.

Two summaries contain all the information this particular operator needs. They do not contain an individually recoverable copy of every original pair. If two histories have the same S and z, every future query with no intervening write receives the same answer.

**Investigation — can two summaries answer the same questions as a table?** Edit positive-feature queries, keys and values while watching the causal weight table, running matrix and weight vector. Pin the starting case, then change a write or the chunk size. Try equal values, a changed last value and a zero-total-weight query. The live output and unchanged earlier rows expose causality, equivalence and undefined reads separately.

This example also reveals a limitation: an ungated summary retains all writes. Repeating an address can accumulate conflicting values. Normalization may average them, but it does not know that a later measurement was meant to replace an earlier one.

### “Linear” describes scaling, not the whole network

For fixed feature dimension m and value width p, these updates take O(Tmp) work over T positions and O(mp+m) persistent state per head. Computing the learned projections and feature map adds its own cost. The full model has nonlinearities, normalization and input-dependent features; it is not a linear function of the original sequence.

One common map is φ(x)=ELU(x)+1, applied coordinatewise. For x≥0 it is x+1, and for x<0 it is exp(x), so it is strictly positive in exact arithmetic. This chooses a useful kernel. It is **not an exact softmax factorization**.

Take one-dimensional keys [0,1], values [0,10], and query 2. Softmax gives 10exp(2)/(1+exp(2)), about 8.807971. ELU+1 gives feature-query 3 and feature-keys [1,2], hence weights proportional to [3,6] and output 6.666667. Both are correct evaluations of different operators. A resemblance on one random tensor cannot establish equivalence or downstream quality.

**Inline visual — same input, two weighting rules.** Align the two normalized weight bars on the same 0–1 axis, with the same value chips below. Put “exact softmax” above one and “ELU+1 kernel” above the other. The visible gap in their output dots is the learning point.

### Optional: approximating the exponential kernel

Performer approaches the problem differently. For x=q/d^(1/4), y=k/d^(1/4), the desired score kernel is exp(x·y). If ω is a standard Gaussian vector,

E[exp(ω·x−||x||²/2) exp(ω·y−||y||²/2)] = exp(x·y).

The identity follows from E exp(ω·s)=exp(||s||²/2), with s=x+y. Average m samples, dividing each feature by √m, to estimate that kernel with positive features. FAVOR+ additionally uses a carefully constructed orthogonal sampling scheme to reduce variance. It is not the ordinary sine/cosine random-feature approximation to a Gaussian distance kernel.

The kernel estimate can be unbiased while the **ratio** forming normalized attention is biased: in general E[A/B]≠E[A]/E[B]. Feature count, norm scales and numerical stabilization matter. The later [Sparse & Linear Attention Variants](/learn/path/full-curriculum/sparse-linear-attention-variants?module=deep-learning-fundamentals) lesson owns the detailed approximation comparison. Here the essential distinction is exact regrouping for one kernel versus approximate recovery of another. See the [Linear Transformer paper](https://proceedings.mlr.press/v119/katharopoulos20a.html) and [Performer, §2](https://arxiv.org/abs/2009.14794).

## 3. Parallel training and streamed inference are two computations of one operator

During autoregressive generation, the next token is not known until the current step produces it. The model must advance sequentially. During training, the input sequence is already available, so projections for all positions can be computed together, and structured sums or chunk operations can expose parallel work.

For an ungated positive kernel, divide the sequence into a chunk of c rows. Let Q,K,V denote the already-mapped queries, keys and values in that chunk; let S_in,z_in summarize earlier chunks. Define L as a lower-triangular matrix of ones. The chunk numerator is

N = Q S_in + [(QKᵀ) ⊙ L] V.

The denominator for each row is the corresponding element of

d = Q z_in + [(QKᵀ) ⊙ L] 1.

Divide each numerator row by its own denominator. Then advance the carried state once:

S_out = S_in + KᵀV,
z_out = z_in + Σ_rows K.

There are two sources of context: earlier chunks through the carried state, and earlier/current positions inside this chunk through the triangular mask. Omitting the second mask leaks future information. Omitting the first term forgets earlier chunks. Including the entire updated S_out when calculating every within-chunk output also leaks future writes.

**Inline visual — two routes into one chunk.** Draw a small causal score triangle inside a block and a separate state bridge arriving from its left. Two colored contribution arrays add before normalization. Use distinct labels for “earlier chunks” and “this chunk through the current row,” with no future-to-past arrows.

Our four-update example produces exactly[2,4,2.5,3] for chunk sizes1,2,3,4 and 8 in the small NumPy calculation. Size3 deliberately leaves a shorter final chunk; size 8 contains the whole example. That agreement is an algebra check, not evidence that all chunk sizes run equally quickly.

Here is a complete short program for the causal operator. Q and K contain features, so this program does not secretly apply another map:

```python
import numpy as np

def stream_by_chunks(Q, K, V, size):
    state = np.zeros((K.shape[1], V.shape[1]))
    weights = np.zeros(K.shape[1])
    outputs = []
    for start in range(0, len(Q), size):
        q, k, v = [a[start:start+size] for a in (Q, K, V)]
        local = np.tril(q @ k.T)
        numerator = q @ state + local @ v
        denominator = q @ weights + local.sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError("A query has zero total kernel weight.")
        outputs.append(numerator / denominator[:, None])
        state += k.T @ v
        weights += k.sum(axis=0)
    return np.concatenate(outputs)

Q = np.array([[1., 1.], [2., 1.], [1., 2.], [3., 1.]])
K = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 1.]])
V = np.array([[2.], [8.], [-1.], [5.]])
print(stream_by_chunks(Q, K, V, 3).ravel())
# [2.  4.  2.5 3. ]
```

For large chunks, forming c×c local scores costs more temporary space and work. For small chunks, there are more state transitions and smaller matrix multiplications. A useful schematic work count per sequence is O(Tmp + Tc(m+p)), excluding projections, when each chunk uses dense local scores. This is linear in T if c,m,p are held fixed. The best c depends on implementation and hardware.

Training memory is a separate question from persistent inference memory. Naive autograd can retain a state matrix for every time step; fused kernels, recomputation and custom backward passes change that tradeoff. “Constant-size recurrent state” does not mean an entire differentiable training run has constant memory.

When processing a real model in pieces, the carried state may also include token-shift inputs, convolution history, normalization-related state if the architecture has any, or a readout accumulator. Saving only the central matrix is insufficient when other operations cross the chunk boundary. We will test a complete block's continuation in section 8.

## 4. RWKV-4: weighted memories with several forgetting timescales

RWKV stands for **Receptance Weighted Key Value**. Its early published architecture, commonly called RWKV-4, is especially useful for understanding a stable recurrent memory without starting from a full matrix state. Each channel maintains a decaying weighted numerator and denominator. Different channels can learn different retention rates.

Use λ∈(0,1] for retention, u for a learned current-token log bonus, k_t for a key-derived log weight, and v_t for a value. All of these operations are per channel. Before reading token t, let A_{t−1},B_{t−1} contain earlier writes. The current read is

wkv_t = (A_{t−1}+exp(u+k_t)v_t)/(B_{t−1}+exp(u+k_t)).

After that read, store the token for future positions:

A_t = λ A_{t−1}+exp(k_t)v_t,
B_t = λ B_{t−1}+exp(k_t).

The two operations have different jobs. **The current-token bonus changes this read; it does not become a permanent bonus on the stored write.** Also notice the indexing: at t, the most recent earlier write has not yet been decayed by the update for t. Expanding the history gives weight λ^(t−1−i)exp(k_i) for i<t.

A retention of .5 halves an old contribution on each later state update; a retention near1 decays it slowly. Its half-life is log(.5)/logλ updates when0<λ<1. Half-life describes the decay factor, not a guaranteed lifespan of semantic information after nonlinear layers and later writes.

**Inline visual — two-stage read/write circuit.** Split the diagram into “answer now” and “prepare the next state.” Route the current bonus only into the first branch. Put a decay valve only on the old-state input to the second. A dashed time boundary separates the stored state from the new token. This is more informative than a single unlabeled memory box.

### One channel, fully worked

Set λ=.5, exp(u)=2, key weights exp(k)=[1,2,1,4] and values [2,8,−1,5].

| t | A before read | B before read | Current weight exp(u+k_t) | wkv_t | A after write | B after write |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 2 | 2 | 2 | 1 |
| 2 | 2 | 1 | 4 | 34/5=6.8 | 17 | 2.5 |
| 3 | 17 | 2.5 | 2 | 15/4.5=3.333333 | 7.5 | 2.25 |
| 4 | 7.5 | 2.25 | 8 | 47.5/10.25=4.634146 | 23.75 | 5.125 |

Without the current bonus, the outputs become [2,6,4.571429,4.4], but the stored A,B sequence stays exactly the same. This is a useful diagnostic: if changing u alters stored history for these fixed keys/values, the implementation has confused the two branches.

The output of the time-mixing sublayer is W_o[σ(r_t)⊙wkv_t]. The receptance gate σ(r_t) selects how much of each channel's mixture to use. In this RWKV-4 core it does not recompute a different q_t·k_i score for every old token. Across multiple context-dependent channels and layers the full network is still expressive, but the core differs from ordinary query-dependent attention.

### Stable arithmetic without changing the answer

Computing exp(k) directly fails for very large keys even when the final normalized answer is modest. If we add1000 to all keys in this example, every numerator and denominator is multiplied by exp(1000), so the exact outputs should remain unchanged. Raw float64 exponentials overflow; a rescaled calculation changes the answer by only about 1.1×10^−13 in our probe.

Represent A=exp(p)a and B=exp(p)b. Store a,b and the log scale p, rather than enormous A,B. Initially a=b=0 and p=−∞. For the read choose q=max(p,u+k), so

wkv = [exp(p−q)a + exp(u+k−q)v] /
      [exp(p−q)b + exp(u+k−q)].

For the stored update, choose p_new=max(p+logλ,k) and compute

a_new = exp(p+logλ−p_new)a + exp(k−p_new)v,
b_new = exp(p+logλ−p_new)b + exp(k−p_new).

Both exponential arguments are nonpositive. One competing exponential equals1; with valid finite keys, the normalization remains meaningful. Crucially, a may be negative because values may be negative. p tracks the scale of **weights**, not log(a). This method works for signed values.

**Inline visual — the answer survives a change of measuring scale.** Show the same weighted sum using ordinary units and units multiplied by exp(−q). Keep numerator and denominator together. Follow with two exponent rulers where subtracting the maximum places every argument at or below zero. Do not suggest clipping k, which would change relative weights.

Adding the same small ε to the raw and rescaled denominators does not preserve equivalence: ε has a different effective scale in the two formulas. Diagnose invalid inputs or insufficient precision explicitly rather than treating arbitrary ε as an exact algebraic identity.

The complete [linear_memory_mechanisms.py](linear_memory_mechanisms.py) implements direct log-weight enumeration, the stable recurrence, chunked positive-kernel attention and the later memory updates. Run `python linear_memory_mechanisms.py` with NumPy installed. It writes [mechanism-results.json](mechanism-results.json); the direct and recurrent outputs above agree, and the large key-shift test remains finite.

**Investigation — which change affects today's read, and which changes tomorrow's memory?** Edit values, key strengths, retention and current bonus. inspect a chosen output and whether stored history changes. Compare read and write contributions on separate timelines; then add a common large key offset. The lab begins with an visible sequence, includes a constant-value case, and lets you inspect the stable state without displaying overflowing raw values as valid numbers. Its goal is to separate memory semantics from numerical representation.


## 5. From one memory channel to a trainable sequence model

A useful memory operator is only part of a neural network. The model must learn how to encode input, what to write, how to read, and how to turn that read into a task prediction.

RWKV-4 alternates a time-mixing sublayer with a channel-mixing sublayer. “Time mixing” combines information from different positions; “channel mixing” transforms the feature coordinates at one position. Pre-normalization and residual additions provide stable paths through a stack of blocks.

First form shifted inputs. For the key branch, for example,

x̂_t^k = μ_k⊙x_t + (1−μ_k)⊙x_{t−1},
k_t = W_k x̂_t^k.

Separate mixing coefficients and projections produce receptance and value. The previous-position input makes a local change—such as movement direction or a transition between characters—available before the long-memory update. These learned coefficients need not be constrained to a convex interpolation in every original implementation. Our teaching model explicitly uses sigmoid-parameterized coefficients in [0,1].

The channel sublayer forms a projected feature, squares its positive part and projects it back:

c_t = σ(W_r' x̂_t^r) ⊙ W_v' [ReLU(W_k' x̂_t^k)]².

Squaring is elementwise. The hidden projection expands the width, allowing a richer transformation, and W_v' returns to the residual width. This sublayer has its own previous normalized input for token shifting. Reusing the time-mixing previous input would be a different model.

**Inline visual — one block with the state slots exposed.** Draw normalization → time mix → residual addition → normalization → channel mix → residual addition. Beneath the time branch show previous input plus a,b,p; beneath the channel branch show its previous input. Use a single highlighted token moving through the block while the state slots update. Keep projection weights outside the sequence-state boundary.

For language modeling, token IDs become embedding vectors; the final hidden vector becomes one logit per vocabulary token. Softmax converts logits into probabilities. With training text x_1,...,x_T, the loss predicts the following token from each prefix:

L = −Σ_{t=1}^{T−1} log p_θ(x_{t+1}|x_{≤t}).

A causal state update processes x_t before predicting x_{t+1}. Shifting labels incorrectly can ask the model to reproduce an already visible token and create an artificially easy training task. Backpropagation adjusts projections, mixing coefficients, decay and readout weights to reduce this loss. The a,b,p values from one training sequence are not a permanent collection of learned model parameters.

When training across chunks, passing a state carries context forward. **Detaching** that state from the gradient graph preserves its numerical value while truncating credit assignment across the boundary. Resetting it discards context. These two choices can have very different learning consequences. Unrelated documents or movement examples normally start from their designated initial state; state reuse across shuffled examples causes information contamination.

The original RWKV-4 CUDA kernel assigns parallel work across batch/channel coordinates and runs a serial loop through time inside each such computation. Its projections are parallel over known training tokens. Other implementations can use scan or chunk methods, but “parallelizable training” does not identify one universal prefix-scan algorithm. The [paper's architecture and Appendix D](https://aclanthology.org/2023.findings-emnlp.936/) and its [forward kernel](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/cuda/wkv_cuda.cu) make that distinction concrete.

## 6. When a summary needs an editable address

Return to the sensor updates. If the current task asks for the newest reading of A, adding2 and 5 gives 7, while averaging them gives 3.5. Neither operation performs replacement.

A matrix can act as a small associative memory. For this section use the **transposed orientation** M∈R^(p×d): a key is a column of length d, a value has length p, and retrieval is M k. This is the transpose of the key-by-value state S used earlier. Stating the orientation prevents a surprisingly common implementation error when comparing papers.

### Adding and overwriting are different learning rules

An additive memory writes M_new=M_old+v kᵀ. If k_A=[1,0]ᵀ and k_B=[0,1]ᵀ, writes A→2, B→7, A→5 produce

M_add=[7,7].

A **delta rule** first asks what the memory currently predicts for this key, computes a residual and writes a correction:

prediction=M_old k,
error=v−prediction,
M_new=M_old+β error kᵀ.

This is one gradient-descent step on the local loss ½||M k−v||². Its gradient is (M k−v)kᵀ. With unit-norm keys and β=1, retrieval at that key becomes exactly the new value. With 0<β<1, it moves part of the way. If keys are not unit norm, the effective step on the retrieved value is multiplied by ||k||²; normalization is substantive.

For the same three writes and β=1, the states are [2,0], then[2,7], then[5,7]. A's outdated reading is replaced while B survives. At the final write the residual is 5−2=3, so we add[3,0], not[5,0].

**Inline visual — the address that gets corrected.** Show a two-coordinate key compass above a one-row memory matrix. The additive branch deposits a full value; the delta branch first reads, displays its signed error, and deposits only that correction. A neighboring independent key stays highlighted to show whether its answer changes.

The gradient interpretation is real arithmetic, not a metaphor. Starting from M=[2,7], key [1,0] and target 5, loss=4.5, gradient=[−3,0], and a step β=.5 produces M=[3.5,7]. Our NumPy and autograd calculations agree. It is **fast state adaptation inside the forward pass**, while the outer training process learns the projections and update parameters.

Keys interfere when they are not orthogonal. Replace B's key by[.6,.8]. After the first two delta writes the memory is [5.48,4.64]. It retrieves B as7, but it no longer retrieves A as2. Updating A to 5 produces [5,4.64], after which B retrieves 6.712. A finite vector space cannot provide arbitrarily many mutually orthogonal address directions.

**Investigation — can a memory update one address without damaging another?** Create or edit a sequence of key–value writes, choose a retrieval query and Show the current computed result and its contributing terms immediately. Compare additive writes, partial correction and full correction. Rotate a key toward another key, alter a repeated value, and inspect the residual and final retrieval error. The fresh starting task uses different keys, values and a partial update; a zero-update-rate case shows what “no learning” really means.

### RWKV-5 and 6: a matrix state with structured forgetting

RWKV-5, named Eagle, replaces the earlier per-channel weighted average with multi-head matrix states. In key-by-value orientation, a head reads

R_t = S_{t−1}+diag(u) k_t v_tᵀ,

and stores

S_t = diag(w_t)S_{t−1}+k_t v_tᵀ.

The receptance vector reads rows of R_t; per-head normalization, a SiLU gate and an output projection follow. This core is not the normalized positive-kernel operator of section 2, and its keys/values need not be positive. The current-token contribution still has special treatment; removing it would lose a useful original mechanism.

For RWKV-5, the retention vector is learned but fixed over sequence positions, with w=exp(−expω). RWKV-6, Finch, makes both retention and token-shift mixing depend on current/previous input through small low-rank networks. Its retention is w_t=exp(−exp d_t), so it lies in(0,1) in exact arithmetic. This is a constrained parameterization; its unconstrained precursor is not itself the retention factor.

An important practical consequence is the state shape: a head of width d now carries d×d entries, rather than one weighted numerator/denominator per channel. Sequence-length independence is preserved, but the constant can grow substantially. More address interactions cost more memory and arithmetic.

**Inline visual — evolving the stored object.** Put RWKV-4's channelwise numerator/denominator beside RWKV-5's matrix, then animate RWKV-6's row retention changing with an input. Match shared labels while showing changed shapes. Do not draw this as an accuracy leaderboard. The [Eagle/Finch paper, §§3–4](https://arxiv.org/abs/2404.05892) defines the full block and version differences.

### RWKV-7: decay, targeted removal, then a fresh write

RWKV-7, Goose, uses a generalized delta-like update. Its paper switches to a value-by-key state M, which is the orientation used for our correction example. Per head,

M_t = M_{t−1}[diag(w_t)−κ̂_t(a_t⊙κ̂_t)ᵀ] + v_t k̃_tᵀ.

Here κ̂ is a normalized removal key, a is a vector controlling removal, w is the retention vector, and k̃ is the replacement key. The dense-looking update can be evaluated as

M_t = M_{t−1}diag(w_t)
      − (M_{t−1}κ̂_t)(a_t⊙κ̂_t)ᵀ
      + v_t k̃_tᵀ.

The second line needs a matrix–vector read and an outer product, not a generic multiplication of two dense d×d matrices. This is why a **diagonal plus rank-one transition** can provide richer state evolution without paying generic cubic work per token.

The ordinary delta rule is a special case: choose w=1, a=β1, κ̂=k with unit norm, and k̃=βk. RWKV-7 decouples removal, write strength and decay. Therefore its full learned update should not be presented as exactly one ordinary SGD step on that same simple loss for arbitrary parameters.

Consider

M_old=[[2,7],[-1,3]],
w=[.8,.9], a=[.6,.2], κ̂=[1,0],
k̃=[1,0], v=[5,2].

The transition is diag([.2,.9]), and the new memory is [[5.4,6.3],[1.8,2.7]]. First columns undergo targeted removal and replacement; second columns only decay. If κ̂ becomes [.6,.8], the transition becomes

[[.584,−.096],[-.288,.772]],

and M_new=[[4.152,5.212],[.552,2.412]]. Off-diagonal entries now couple key directions. A set of independent scalar forget gates cannot express that same cross-direction transition.

**Inline visual — erase direction, write direction.** Show the initial 2×2 matrix, a diagonal decay, a rank-one subtraction tile and a rank-one addition tile. A compass indicates removal and replacement directions separately. Include both checked fixtures and reveal the resulting matrix after the learner has interpreted the signs. A negative tile means subtraction, not a negative probability.

The full model learns those vectors from token-shifted input using projections and small low-rank branches. It normalizes κ per head; uses a sigmoid to keep a in(0,1); constrains w via exp[−exp(−.5)σ(d)]; and mixes a value precursor from the first layer with the current layer's precursor. The readout applies receptance to the updated matrix, normalizes within the head, adds a separately weighted current-token value term, then gates and projects the combined heads. Its feed-forward branch retains squared ReLU but removes the earlier receptance gate. Thus “change the recurrence” alone does not recreate a released RWKV-7 block.

These architectural details and their ablations are in [RWKV-7, §§3–4](https://arxiv.org/abs/2503.14456). The exact two-dimensional calculation above isolates its state update; it is not a measurement of a pretrained language model.

### Optional: what the expressivity and stability claims actually establish

A diagonal transition rescales coordinate directions independently. A rank-one correction can mix them, enabling richer state tracking. For example, I−2nnᵀ with n=[1,−1]/√2 swaps two coordinates. A product of such transitions can track ordered transformations that an elementwise decay cannot express in the same way.

That reflection is an explanatory extension, not the precise parameter choice of the released RWKV-7 core. The paper's stronger constructions introduce an additional factor c=2 and boundary parameter values; the implemented core uses c=1. Its complexity-class claims also make formal assumptions about precision, depth and standard complexity conjectures. They do not mean a small trained checkpoint solves every finite-state task or is universally better at reasoning.

Similarly, stable eigenvalues of individual transitions do not automatically bound every product of changing transitions. Appendix C's product bound assumes a time-independent a vector; time-varying a is evaluated empirically. Normalization and constrained updates provide useful structure, while actual long-stream state norms, gradients and task behavior still need examination. This is an opportunity to distinguish a theorem's conditions from a model family's headline description.

## 7. Nearby models, shared algebra and different choices

RWKV is one architecture family in a larger design space. Related methods often share state equations without sharing every gate, positional mechanism, normalization or objective.

**RetNet** uses a decayed unnormalized associative state. Ignoring its positional rotations for a moment,

S_t=γS_{t−1}+k_t v_tᵀ,
y_t=S_tᵀq_t.

It assigns different fixed retention rates to heads. The full retention layer includes relative positional rotations, per-head/group normalization and a nonlinear gate. Its parallel, recurrent and chunkwise forms compute the same chosen retention mechanism; they are not three different models. A fixed scalar γ applies one timescale per head.

**Gated Linear Attention, GLA**, makes forgetting input-dependent. In the paper's adopted parameterization,

S_t=diag(α_t)S_{t−1}+k_t v_tᵀ,
y_t=S_tᵀq_t.

The key rows decay separately, using a low-rank input projection and a sigmoid-derived gate. The more general expression G_t⊙S_{t−1} allows a full matrix of elementwise gates, but the paper deliberately chooses a structured row gate for efficient computation. Naming that restriction matters when comparing implementations.

Using the section 2 queries, keys and values with every row gate equal to .5 gives unnormalized outputs [2,10,5.5,35.75]. Change only the third retention vector to [.1,.9] and the outputs become [2,10,11.5,36.75]. At that step, one address direction forgets faster while another preserves more. These values are not weighted averages constrained to the range of the input values.

**Inline visual — one state, several ways to forget.** Use the same tiny matrix under a scalar valve, separate row valves, and a rank-one erase direction. Place RetNet's simple decay, GLA's row gating and the delta/RWKV-7 correction side by side, with equations and the exact dimensions. Distinguish signed outputs from probabilities.

The [RetNet paper, §2](https://arxiv.org/abs/2307.08621) and [GLA paper, §§2–4](https://arxiv.org/abs/2312.06635) derive these relationships and their chunked algorithms. A global cumulative product of tiny gates can underflow, while dividing by that product can overflow. GLA's hardware work therefore uses chunking and log-space treatment where needed; an algebraically convenient quotient is not automatically a stable GPU implementation.

The preceding Mamba-2/SSD construction has a scalar transition per head with matrix-valued state. That brings it close to a gated linear-attention equation. The parameter generators, block structure and normalization still differ. Conversely, **Linformer** compresses the sequence axis of keys/values into a smaller set before attention; it is not simply a feature-map prefix sum. A global sequence projection can mix future positions, so autoregressive use needs a genuinely causal construction. See [Linformer, §4](https://arxiv.org/abs/2006.04768).

This family map continues to evolve. The RWKV project's history page, inspected 13 September 2026, discusses experimental RWKV-8 directions including token-indexed DeepEmbed modulation and a suffix-automaton retrieval mechanism called ROSA. An external table or growing automaton changes the memory accounting: offloading parameters or history to RAM/SSD does not make their storage and access free. The established4–7 mechanisms taught here remain the core; treat new branches through their explicit operator, available code and evidence, not a version number alone.

### Count the state before comparing speed

For one sequence and 12 layers of width 512:

| Stored object | Explicit assumptions | Bytes |
| --- | --- | --- |
| RWKV-4 recurrent slots | Five width 512 vectors/layer, all float32 | 122,880 =120 KiB |
| A matrix-memory core | Eight 64×64 heads/layer, float32; excludes extra shift/readout slots | 1,572,864 =1.5 MiB |
| Full MHA key/value cache | 4,096 tokens, eight KV heads, width 64, float16 | 100,663,296 =96 MiB |
| GQA key/value cache | Same, but two KV heads | 25,165,824 =24 MiB |

These are calculated storage inventories, excluding parameters, activations, temporary buffers, allocator overhead and serving replicas. Mixed state dtypes alter the totals. The [GQA/MQA lesson](/learn/path/full-curriculum/grouped-query-attention-gqa-multi-query-attention-mqa?module=deep-learning-fundamentals) explains why sharing KV heads reduces cache storage without proportionally eliminating all query-head attention arithmetic.

**Inline visual — what grows with the history.** Plot these formulas against token count on clearly labeled byte/MiB axes, with markers at 4,096 and an accompanying numeric table. Label the matrix line as “core only.” These are exact formula curves, not measured RAM or latency. Use linear and log options only if their labels make the difference obvious.

For an attention head, cached decoding at position t reads t keys and values: the per-token attention work grows roughly linearly with t. Generating an entire length-T continuation from scratch can sum to quadratic attention work. Full-sequence training also has quadratic dense attention arithmetic. FlashAttention avoids storing a full T×T score matrix in high-bandwidth memory while computing exact attention; “quadratic score arithmetic” and “must allocate a quadratic score matrix” are different claims.

For recurrent memory, persistent state and per-step memory work are bounded with respect to T, at fixed widths. Projections, feed-forward networks and output vocabulary operations can dominate at small T. Chunk size, precision, batch size, head dimensions, hardware and kernel fusion decide actual timing. There is no architecture-independent crossover where T happens to equal the model width.

A fair deployment measurement separates **prefill** from **decode**, records the exact checkpoint/tokenizer/kernel revision, reports latency distribution and peak memory, warms up and synchronizes the accelerator, and compares equivalent task quality. Equal parameter count alone does not imply equal training data or ability. The [FLA implementation project](https://github.com/fla-org/flash-linear-attention) provides current kernels, model layers and benchmark examples; its API and backend requirements should be pinned when running them.

Fixed state is useful for long-running sensor streams, locally processed interactions and predictable per-request memory. It also creates interference and forgetting pressure. A model may accept another million tokens without retaining an arbitrary exact fact from the beginning. An external retrieval system or occasional attention layer can address different requirements; the later [hybrid architecture lesson](/learn/path/full-curriculum/hybrid-ssm-transformer-architectures-jamba?module=deep-learning-fundamentals) explores that combination.

## 8. Train a real movement classifier, then interrupt its stream

The [Libras Movement dataset](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement) gives us a small real sequence task: classify a two-dimensional hand trajectory into one of 15 recorded movement classes. A row contains 45 ordered x/y coordinate pairs and a class label. Unlike the address examples, the useful keys, values and readout must now be learned.

This continues the real-data thread from the preceding lessons while changing the mechanism. Keeping the raw task and partition visible helps us ask whether the model has learned anything useful, rather than judging an architecture by a decorative synthetic trace.

### Data and a declared evaluation protocol

Download [movement_libras.data](movement_libras.data), [movement_libras.names](movement_libras.names), [trajectory_memory_models.py](trajectory_memory_models.py) and [data-provenance.md](data-provenance.md) into one directory. Credit Daniel Dias, Sarajane Peres and Helton Bíscaro, University of São Paulo; UCI provides the dataset under CC BY 4.0. The provenance file preserves attribution, source hashes and transformations.

The source has 360 rows and 30 exact duplicate copies, with consistent labels. Retain the first occurrence of each exact coordinate row, giving330 unique trajectories. Within each class, use NumPy's default_rng(73) to permute those retained source IDs; put the first floor(2n/3) in fitting, the last 4 in assessment, and the middle rows in validation. Totals are 220/50/60. The saved [trajectory-results.json](trajectory-results.json) records every one-based source ID and duplicate group.

Coordinates are mapped by the fixed rule 2x−1 and processed in their original order. There is no fitted preprocessing that can learn from validation or assessment rows. The dataset records four performers in two sessions but omits row-level performer/session IDs. This partition measures row-level behavior on this small corpus; it cannot establish new-performer transfer. Its 45 sampled points are not a physical velocity trace or a complete sign-language utterance.

Our baseline is multinomial logistic regression on all 90 ordered coordinates, with C=1. It has access to the whole trajectory and to positional information through the feature order. It is a meaningful simple competitor, not a deliberately weak mean-coordinate straw model.

### What exactly are we fitting?

The teaching networks share this route:

45×2 coordinates → learned 2-to 16 projection → two residual memory blocks
→ mean over 45 feature vectors → linear 16-to 15 logits.

Each block has pre-layer normalization, a memory mixer, a residual connection and a gated squared-ReLU channel branch. The RWKV-4-style mixer uses the stable weighted recurrence from section 4. The positive-kernel mixer uses ELU+1 queries/keys and the normalized matrix summary from section 2. Both use a one-position shift before projections, so local transitions are available.

The code deliberately specifies its own small configuration: width 16, two blocks, channel expansion 2, sigmoid-parameterized input mixing, and a classification readout. It is a faithful small investigation of the stated operators inside trainable blocks, not a reproduction of the initialization, scale or training recipe of a published language checkpoint.

For a trajectory with class c, the loss is

−log[exp(ℓ_c)/Σ_{j=1}^{15} exp(ℓ_j)],

where ℓ are its 15 logits. The program uses the library's stable cross-entropy operation, backpropagates through the time steps, and updates the model weights with Adam. Model state starts fresh for every trajectory; the batch dimension holds independent examples.

**Inline visual — the task-shaped learning loop.** Show an actual hand path with a start arrow and 45 numbered positions, then its 2-column sequence, the two memory blocks, the temporal mean and 15 class bars. A separate backward arrow from the known training label reaches the learnable weights. Validation and assessment data have forward-only arrows. This diagram connects the mechanism to supervised learning.

The following loop is the central training operation, already included in the complete program:

```python
optimizer.zero_grad()
logits = model(x[fit])
loss = torch.nn.functional.cross_entropy(logits, labels[fit])
loss.backward()
optimizer.step()
```

Validation chooses which epoch to retain, not which assessment labels to fit. We predeclared 100 full-batch epochs, learning rate .003, seeds 17 and 41 and the two model configurations. Each run retains the epoch with lowest validation cross-entropy. The assessment set is evaluated after that choice. We retain both seeds rather than selecting the one with the prettiest assessment result.

To reproduce in a separate Python environment:

```text
python -m pip install numpy torch scikit-learn
python linear_memory_mechanisms.py
python trajectory_memory_models.py
python author_calculations.py
```

The last command uses [author_calculations.py](author_calculations.py) for the fixed continuation and perturbation checks. The author executed these programs with Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu and scikit-learn 1.9.1, two CPU threads and deterministic algorithms. Full source is included; no GPU, remote checkpoint or external training service is required. A different library/platform can produce small numerical differences.

### The observed result

| Model and seed | Parameters | Selected epoch | Fit errors/220 | Validation errors/50 | Assessment errors/60 |
| --- | --- | --- | --- | --- | --- |
| Ordered logistic baseline | 1,365 | Single fit | 35 | 17 | 22 |
| RWKV-4-style,17 | 5,263 | 67 | 31 | 18 | 19 |
| RWKV-4-style,41 | 5,263 | 61 | 46 | 23 | 23 |
| Positive kernel,17 | 5,199 | 54 | 70 | 31 | 34 |
| Positive kernel,41 | 5,199 | 69 | 34 | 22 | 23 |

One RWKV-style run has fewer assessment errors than the baseline; the other does not. The positive-kernel runs vary considerably. This is evidence about these small fits and this protocol. It does not rank full RWKV, Performer, Transformer or Mamba architectures, nor isolate a single causal reason for the performance differences.

Validation error count and cross-entropy can prefer different epochs: cross-entropy also measures how much probability is assigned to the correct class. For example, RWKV-style seed 41 has 23 validation errors but lower selected validation cross-entropy than seed 17's18 errors. The selection rule was cross-entropy for both.

**Inline visual — honest small-run outcomes.** Plot error counts with explicit denominators and one dot per seed, alongside the baseline. Keep the axis from 0 to 60 for assessment counts and label the table with parameter counts. A separate optional training chart uses the recorded fit/validation loss history and marks the selected epoch. Do not interpolate a trend across architecture names or imply that two seeds estimate a population uncertainty interval.

### Carry the whole state, including the small forgotten pieces

Take the predetermined first validation example, source row 7, labeled curved swing. For RWKV-style seed 17, its class 1 probability is about .753939. Process 22 points, carry every block's memory and previous-input slots, then process the remaining 23 points. Combine the feature sums and counts for the final temporal mean. The carried computation agrees with the uninterrupted logits within 2.4×10^−7 for this example.

Now deliberately reset the block states at that boundary but keep the same 45 points and final pooling rule. Class1 probability falls to .047025 and the predicted class becomes 10, vertical zigzag. Merely splitting an input should not change a computation; discarding its history does.

The positive-kernel seed 17 model already predicts class 10 on the original row, with class 1 probability.251896. Carrying its state preserves the computation; resetting midway changes its prediction to class 7. This contrast separates **implementation equivalence** from **prediction correctness**. A model can consistently compute the wrong classification.

For all four fits, a separate continuation through chunk lengths 1,12,16,16 agrees with the full-sequence features on the first five source rows within 7.7×10^−6. These are floating-point agreements, not bit-identical guarantees.

**Investigation — pause a hand movement without erasing its past.** Choose a validation trajectory, move a point using numeric coordinates or the path, choose a cut point and observe whether carrying or resetting state changes the output. The plot shows the exact path, chronological direction and cut location. Compare uninterrupted, carried and intentionally reset predictions using frozen fitted weights. Include reverse-order and restore-original actions. New edited paths receive model predictions, not newly certified class labels.

For the fixed row 7 contrast, reflecting the23rd normalized x coordinate from +.249520 to−.249520 barely changes RWKV-style class 1 probability, from .753939 to .753752, while reversing the entire order changes it to .580592. The same operations affect the kernel model differently. A small visual edit need not produce a large class change; a null or weak response is informative.

Changing point31 leaves earlier per-position features unchanged in the bounded probe. The final mean-pooled classifier uses the complete trajectory, so its final prediction can change. “Causal internal features” does not mean “a whole-sequence classification is available without seeing the whole sequence.”

### Where these mechanisms are useful

A streamed classifier can process a long observation in pieces without retaining every intermediate input in its model state. On a local sensor device, predictable state size can simplify resource planning. The receiving application still needs to decide when a sequence starts and ends, how missing observations are represented, and whether its training data matches deployment.

A more unusual use of the delta view is **online calibration of associations**: a compact learned state can revise a mapping when a recurring address receives a new value. The sensor A/B example makes the requirement explicit. With nonorthogonal learned keys, revisions interfere, so a practical system must assess both the new answer and answers it should have preserved. This is a mechanism-based design possibility, not a claimed deployment result from the movement dataset.

For language generation, select a compatible checkpoint, tokenizer, numerical backend and state format together. A “RWKV checkpoint” does not identify a version-independent tensor layout. Prefill a prompt, pass the returned state into subsequent steps, and reset or branch state deliberately between conversations. A shared mutable state across unrelated users is both a correctness and information-isolation problem. The project's maintained inference examples are better starting points than assuming an old Transformers class supports every new RWKV variant.

### Carry the same contract into a released model

The small `TrajectoryMemoryClassifier` above is the ordinary trainable PyTorch route: its time state, shift state, channel state and parameters are visible. For an existing language model, use the model's own inference implementation and tokenizer instead of assuming our classroom block has its checkpoint layout. The optional [complete checkpoint-continuation program](rwkv_checkpoint_state.py) uses the official `rwkv` API on a local checkpoint. Its parameters choose generation 4 or 7, matching tokenizer and CPU float32; it requests no model download.

Install the official `rwkv` package in a separate compatible environment and record its version. A World checkpoint uses the matching packaged `rwkv_vocab_v20230424`; a Pile checkpoint needs its matching local tokenizer JSON. For example: `python rwkv_checkpoint_state.py --checkpoint ./model.pth --generation 7 --tokenizer rwkv_vocab_v20230424`. Provide a checkpoint that fits available memory; this is not a promise that a large language model is economical on a CPU.

The program computes final-prefix logits three ways: one full call, two chunks, and one token at a time after the split. `None` starts a fresh state; later calls carry the returned state. It deep-copies the prefix state before branching because the package may update it in place. This is the same continuation invariant used by our scratch model, but not a parameter-equivalence claim between different models. The [official API example](https://github.com/BlinkDL/ChatRWKV/blob/main/API_DEMO.py) was inspected for this interface on 22 September 2026. This optional code is **written, not executed** in the content revision; printed logits and speed results are deliberately absent.

**Take control:** run a different split point, then replace the carried state with `None` for the suffix. **Hint:** chunk boundaries are an execution choice; forgetting the prefix changes available information. **Solution:** the correctly carried final logits should agree within the declared tolerance; resetting need not. Reuse a cached state only for the same token prefix, checkpoint, tokenizer and numerical configuration. A successful continuation check does not test language-model quality.

## 9. Troubleshooting by locating the broken assumption

Use one diagnostic chain rather than adding numerical patches everywhere:

| Symptom | First useful question | Specific check |
| --- | --- | --- |
| A past output changes after a future edit | Did a mask, sequence projection or within-chunk summary include future inputs? | Perturb one later token and compare the full earlier feature prefix |
| Splitting the same input changes outputs | Which state or boundary input was lost? | Compare all recurrent slots and readout sums/counts at the cut |
| A normalized kernel returns NaN | Is its total query weight positive and finite? | Inspect feature values and denominator before changing the operator |
| RWKV-4 changes under a common large key shift | Were numerator and denominator rescaled together? | Compare direct log-weight enumeration and the stable update |
| Current-bonus edits change future stored writes | Was the bonus incorrectly stored? | Separate read and write equations |
| New associations damage older answers | Are keys correlated, or is forgetting applied too broadly? | Measure retrieval before and after each write, including untouched keys |
| Tiny kernels look slower than dense attention | What dominates this workload? | Separate projections, Python overhead, fused kernels, prefill and decode |
| A long stream runs but fails recall | Is the needed information recoverable from bounded state? | Change delay, number of competing associations and distractor structure |

A key outside a vocabulary is normally an indexing/tokenization error, not an explanation for every numerical failure. Likewise, a constrained retention factor cannot simply become positive growth because its raw parameter moved: inspect the actual transformed value. Keep the failing example and find its mechanism before silently replacing invalid values or dropping difficult training batches.

## 10. Practice: change the task, not just the numbers on a trace

Try each problem before opening its hint or solution. The early tasks test the core route; later ones require the optional matrix and systems sections.

### 1. Build the summary

Use key features [1,0] and [1,2], values 3 and 9, then query [2,1]. What are S,z and the normalized answer after both writes?

<details><summary>Hint</summary>

Write each outer product separately. The query combines stored rows and also combines their total weights.

</details>
<details><summary>Solution</summary>

S=[3,0]ᵀ+[9,18]ᵀ=[12,18]ᵀ, z=[2,2]ᵀ. The numerator is 2×12+18=42, denominator 2×2+2=6, and answer 7. Direct key scores are 2 and 4, giving(2×3+4×9)/6=7.

</details>

### 2. A summary cannot distinguish these histories

History A writes key 1,value 2 and key 1,value 8. History B writes key 1,value 5 twice. With the normalized positive kernel and no forgetting, can any later nonzero scalar query distinguish them before another write?

<details><summary>Hint</summary>

Compare both the value total and weight total.

</details>
<details><summary>Solution</summary>

Both produce S=10 and z=2, so every valid positive scalar query returns 5. The histories differ, but this state deliberately loses that distinction. Increasing computation at read time cannot recover information absent from the state.

</details>

### 3. Place the current bonus correctly

One-channel RWKV has λ=.25, exp(u)=3. The first two writes have key weights 2 and 1, values 4 and −2. Compute both reads and the stored state after the second write.

<details><summary>Hint</summary>

Read the second token from the previous state before applying its storage update.

</details>
<details><summary>Solution</summary>

The first read is 4 and stored A=8,B=2. The second read is(8+3×(−2))/(2+3)=.4. The stored state becomes A=.25×8−2=0 and B=.25×2+1=1.5. Including the bonus in this stored write would incorrectly give A=−4 and B=3.5.

</details>

### 4. Does forgetting change a constant value?

All RWKV values equal−3. Keys vary and λ=.9. Does changing the current bonus alter the output? What if the initial numerator and denominator represent an earlier different value?

<details><summary>Hint</summary>

From empty state, A is always−3B. Decide whether that invariant holds for the changed initial condition.

</details>
<details><summary>Solution</summary>

Starting empty, every weighted average is −3, regardless of positive key strengths, retention and current bonus. With a nonempty initial history having another average, the output can differ and the bonus changes the balance with the new value. The constant-value result depends on the state as well as the visible inputs.

</details>

### 5. Repair one address

Start with M=[6,−2], a unit key [0,1], target 4 and β=.25. Find the loss, gradient and updated memory. Does retrieval of key [1,0] change?

<details><summary>Hint</summary>

The current prediction is the second coordinate, and the correction goes along the selected key.

</details>
<details><summary>Solution</summary>

Prediction−2, residual 6 and loss 18. The gradient is [0,−6]. The update is M=[6,−.5]. The first key still retrieves 6. Only one quarter of the error is corrected, leaving a second-key residual 4.5.

</details>

### 6. A nonunit key changes the effective step

M=[0,0], key [2,0], target 3 and β=1. Does one delta step make retrieval equal3?

<details><summary>Hint</summary>

The write is an outer product with the key, and retrieval multiplies by that key again.

</details>
<details><summary>Solution</summary>

M_new=[6,0], so retrieval is 12. The effective correction is multiplied by ||k||²=4. Normalizing the key or choosing β=.25 gives retrieval3 in this example. A step-size claim stated only for unit keys cannot be reused unchanged.

</details>

### 7. Separate memory from its readout

A sequence is split after 10 points. Every block state is carried correctly. The application averages the mean of the first 10 feature vectors with the mean of the remaining 35 vectors. Is that the original classifier?

<details><summary>Hint</summary>

The two chunks have unequal lengths.

</details>
<details><summary>Solution</summary>

No. The original mean is(10m_1+35m_2)/45, not(m_1+m_2)/2. Carry a sum and a count, or weight chunk means by their lengths. Correct central recurrence state does not excuse an incorrect aggregation outside it.

</details>

### 8. Compare storage with the assumptions visible

A matrix-memory core has 6 layers, four 32×32 heads per layer, float32 state. A cached attention model has the same 6 layers, two KV heads of width 32, float16 keys/values and 2,048 cached tokens. Count bytes, excluding everything else.

<details><summary>Hint</summary>

For the cache count both keys and values. A matrix head stores32² numbers.

</details>
<details><summary>Solution</summary>

Matrix core:6×4×32×32×4=98,304 bytes=96 KiB. Cache:6×2,048×2×2×32×2=3,145,728 bytes=3 MiB. Extra state, model parameters and runtime buffers remain uncounted, and the numbers alone do not establish equivalent quality or latency.

</details>

### 9. Design a test that can contradict your favorite model

You want to claim that a delta memory preserves old facts when updating one address. Specify inputs and measurements that could refute it.

<details><summary>Hint</summary>

Testing only the newly updated address misses interference.

</details>
<details><summary>Solution</summary>

Write at least two different keys and values, measure both retrievals, then update only one address. Use an orthogonal-key control and a correlated-key contrast, and measure errors for both addresses after the update. Include a zero-rate null, a repeated identical write and changed update strength. A method that fixes the new address while damaging the other has not established the claim. For a learned model, add unseen key/value combinations and preserve a held-out protocol.

</details>

### 10. Read a speed headline critically

A graph says “linear attention is 8×faster” but gives no sequence length, kernel, hardware, precision, batch size, model quality or prefill/decode distinction. What conclusion can you draw, and what measurement would you request?

<details><summary>Hint</summary>

Asymptotic scaling specifies how a cost grows under fixed dimensions. It does not supply an absolute runtime.

</details>
<details><summary>Solution</summary>

The graph is insufficient to select a deployment method. Request exact model/operator and kernel revisions, input and output lengths, batch/concurrency, head sizes, dtype, hardware, warmup/synchronization, latency statistics, memory and relevant task quality. Separate prompt processing from per-token generation. Reproduce a representative workload before extrapolating its result.

</details>

**Ready to move on:** you can trace a weighted read and stored write separately, derive S and z, explain why a different kernel is not exact softmax, and identify which state crosses a chunk boundary. The advanced readiness test is explaining additive versus corrective writes and the assumptions behind a claimed memory or stability advantage.

The next lesson in the actual module sequence is [Self-Attention & Multi-Head Attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals). It develops the query-based operator and multiple heads systematically. Having seen the alternative memory choices first, you can now ask precisely what retaining individual keys/values buys.

## References and other ways to learn

- [Katharopoulos et al., Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html): start with§3 for the feature-map rearrangement, causal state and backward-memory discussion. Match its tensor orientation to the equations here.
- [RWKV-4 paper and supplemental material](https://aclanthology.org/2023.findings-emnlp.936/):§3 and Appendix D explain the block and stable recurrence; the [official forward kernel](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/cuda/wkv_cuda.cu) makes read-versus-write order explicit. Best after section 4.
- [Eagle and Finch](https://arxiv.org/abs/2404.05892):§§3–4 distinguish matrix state, input-dependent retention and token shifting. Useful for seeing why version changes affect more than model size.
- [RWKV-7 Goose](https://arxiv.org/abs/2503.14456):§§3–4 define the generalized correction; Appendices C–D state the conditions behind stability and expressivity claims. Advanced follow-up rather than a substitute for the worked matrix example.
- [Performer](https://arxiv.org/abs/2009.14794):§§2.3–2.4 explain positive and orthogonal random features. Read alongside the local expectation identity, then continue to the later attention-variants lesson.
- [RetNet](https://arxiv.org/abs/2307.08621) and [GLA](https://arxiv.org/abs/2312.06635): mechanism papers for fixed decay, dynamic row gating and chunked computation. Their benchmark figures describe particular setups, not universal rankings.
- [Linformer](https://arxiv.org/abs/2006.04768):§4 gives a different form of compression along the sequence axis. Useful for separating several ideas that are sometimes all called “linear attention.”
- [RWKV Architecture History](https://wiki.rwkv.com/basic/architecture.html): a project-maintained visual history, version notes and links to inference/training guides. Use it to locate actual implementations; treat experimental features and broad performance language as claims to inspect.
- [Oxen Arxiv Dive: How RWKV-7 Goose Works](https://www.oxen.ai/blog/how-rwkv-7-goose-works-notes-from-the-author): an illustrated article with an embedded YouTube discussion involving RWKV contributor Eugene Cheah. The article's memory and correction walkthrough offers another entry point after section 6. Its broad attention-complexity shorthand needs the prefill/decode and storage distinctions in section 7. The article and embedded-video identity were inspected; the recording was not watched.
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention): maintained code, supported model layers and current benchmark/integration examples. Best for moving from the small programs to an actual accelerated implementation; pin the backend and revision.
- [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement): the original task description and data license. Use the local provenance file when reproducing the duplicate-safe partition.

The [complete mechanism program](linear_memory_mechanisms.py), [complete training program](trajectory_memory_models.py), [checked results](trajectory-results.json) and [perturbation/continuation evidence](investigation-checks.json) let you reproduce the examples without relying on screenshots of someone else's benchmark.
