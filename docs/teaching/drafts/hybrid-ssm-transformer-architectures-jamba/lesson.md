# Hybrid SSM–Transformer Architectures: Jamba and Complementary Memory

**Explore as you read.** Edit record keys/values, decay, score gap, cache budget, expert probabilities/capacity and supported stroke inputs. Show retained state versus explicit memory read, probability mass, exact request memory and continued frozen-model outputs. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose a hybrid arrangement by memory retention and routing costs; named architecture examples do not imply identical mechanisms.


A document assistant may need to follow a developing argument and then retrieve the exact amount beside an invoice number. A handwriting recognizer may need to follow a pen's movement and then compare the closing stroke with an earlier turn. These are related jobs, but the most convenient memory for one is not automatically the most convenient memory for the other.

A **hybrid sequence model** uses more than one kind of sequence-processing operation inside a single network. In the family studied here, some layers update a compact recurrent state; others compare a query with stored keys and read the corresponding values. Jamba combines Mamba state-space layers, attention layers and, at selected feed-forward positions, sparsely chosen experts.

The previous lesson, [Neural ODEs](/learn/path/full-curriculum/neural-ode-continuous-depth-models?module=deep-learning-fundamentals), asked what happens when computation across depth becomes a continuous trajectory. Here depth is a discrete sequence of different operations. A second axis, the positions in the input, must remain separate: a model can have 32 layers while reading 262,144 tokens.

You will learn to trace those two axes, calculate the memory a request actually carries, explain the difference between sparse expert selection and sparse attention, and investigate a small trained hybrid without mistaking it for a miniature benchmark of a giant language model.

**Core route:** follow §§1–8, then solve the first six practice questions. The optional branches on expert arithmetic, cross-layer sharing and deployment experiments add depth without being prerequisites for understanding the central mechanism.

## 1. Two useful questions need two different reads

Imagine three labelled measurements arriving in order:

| Position | Label | Measurement |
|---|---|---:|
| 1 | A | 2 |
| 2 | B | 7 |
| 3 | C | 4 |

One question is “What level has the stream recently been around?” Another is “What measurement belongs to A?” The first invites a running summary. The second invites a label-sensitive lookup.

[Figure J01: the same three records feed a running-summary register and an addressable record strip. Arrows name the two questions. The records are illustrative measurements in arbitrary units.]

For a simple recency-weighted average, keep a numerator $s_t$ and a normalizer $n_t$:

$$
s_t=\lambda s_{t-1}+v_t,\qquad
n_t=\lambda n_{t-1}+1,\qquad
r_t=s_t/n_t.
$$

Start with $s_0=n_0=0$. With $\lambda=1/2$, the state evolves as follows:

| After record | $s_t$ | $n_t$ | Recency-weighted average |
|---|---:|---:|---:|
| A:2 | 2 | 1 | 2 |
| B:7 | 8 | 1.5 | 5.333333 |
| C:4 | 8 | 1.75 | 4.571429 |

The final answer gives the last observation four times the weight of the first. Only two numbers remain in memory, regardless of how many records arrived. This particular summary does not even use the labels.

For the lookup, assign a compatibility score $\log 9$ to a matching label and zero to each nonmatch. Softmax converts these scores into weights proportional to $[9,1,1]$. Asking for A gives

$$
\alpha=[9/11,\;1/11,\;1/11],\qquad
o=\sum_j\alpha_jv_j=\frac{9(2)+7+4}{11}=\frac{29}{11}\approx2.636364.
$$

This is already an important correction to the phrase “exact retrieval.” We computed softmax attention exactly, but its answer is a mixture, not the original value 2. Increasing the score gap makes it concentrate more strongly on A. Equal or misleading keys, finite precision and learned projections can still prevent the intended retrieval.

[Figure J02: weighted numerator bars aligned with each retained record; a labelled denominator 11 and a separate output 29/11. Show the state average alongside it, without suggesting that the two answers should agree.]

A hybrid can make both kinds of information available to later computation. It does not have to average these two answers together. For example, an output head could report a recent level and the value associated with a requested label as separate fields.

### A specific collision explains the limitation

Replace the values $[2,7,4]$ by $[6,5,4]$. Our recurrence again ends at $(s,n)=(8,1.75)$. Once these two histories have collapsed to the same state, any later computation receiving only that state must give the same result for both histories. The label-A attention read instead changes from $29/11$ to $63/11$.

This is a counterexample for **this two-number summary**, not a proof that every recurrent network fails at copying. A larger state, a selective write rule, a different representation or explicit training may preserve the required information. Conversely, retaining separate keys and values does not make a trained attention model infallible.

[Figure J03: two different histories converge to an identical state node, while their retained value strips remain different. Caption identifies the recurrence and avoids a universal capacity claim.]

**Investigation JA — decide what information to retain.** Use the fresh record list A:3, B:8, A:1, C:5. Before running it, observe whether changing the second label from B to A will affect the running summary, the label-A read, both or neither. Then edit a value, the recency factor or the score gap and explain which weights moved. Try the constant-value case and a query that matches no label. The latter still returns a weighted mixture; a production system needs a separate way to represent “no useful match.”

## 2. Read a hybrid stack on two axes

Attention and Mamba are **sequence mixers**: an output at one position can depend on earlier positions. A feed-forward network applies a nonlinear transformation to each position's current vector. It can use context already incorporated into that vector, but it does not independently scan the other positions.

A common layer has two residual updates:

$$
u^\ell=x^\ell+\operatorname{Mixer}_\ell(\operatorname{RMSNorm}(x^\ell)),
\qquad
x^{\ell+1}=u^\ell+\operatorname{FFN}_\ell(\operatorname{RMSNorm}(u^\ell)).
$$

Both additions matter. The second adds its result to $u^\ell$, which already contains the mixer's contribution. RMSNorm rescales each vector using its root-mean-square magnitude and learned channel weights; it does not subtract a mean or erase sequence order.

[Figure J04: two residual paths with addition junctions. The arrow feeding the second junction comes from $u^\ell$. A small four-position grid shows the mixer spanning positions and the FFN operating independently on each contextualized vector.]

In the original Jamba configuration, zero-based layer indices 4, 12, 20 and 28 use attention. The other 28 use Mamba. Every odd-indexed layer uses 16 feed-forward experts, selecting two per token; the even-indexed layers use a single feed-forward network. Consequently, this released pattern's attention layers have dense FFNs. The architecture permits other combinations, but “could combine” and “does combine in this checkpoint” are different statements. These values come from the [released configuration](https://huggingface.co/ai21labs/Jamba-v0.1/blob/main/config.json).

| Layer indices in the first cycle | Mixer | Feed-forward operation |
|---|---|---|
| 0, 2, 6 | Mamba | Dense |
| 1, 3, 5, 7 | Mamba | Top-2 of 16 experts |
| 4 | Attention | Dense |

The cycle repeats across depth, not every eight input tokens. Every token passes through every layer. At an attention layer, its query can read earlier positions of **that layer's input representation**; those representations have already passed through lower layers.

[Figure J05: an eight-layer vertical ladder repeated four times, with separate mixer and FFN columns, zero-based indices, and a horizontal token axis. Show attention at 4 rather than 7. Colour is supplementary to the M/A and dense/expert labels.]

Do earlier attention layers have “nothing to attend to”? No. Even the first layer receives representations of all available input tokens. Deeper layers may provide more contextualized representations, but useful placement is an empirical design choice. Nor is the final attention layer wasted: its outputs can directly influence the prediction head.

The original paper's ratio comparison used 1.3B-parameter models trained on 250B tokens. Its 1:3 and 1:7 hybrids had similar results in that experiment, motivating the cheaper tested ratio. That is a scoped observation, not a theorem prescribing one attention layer per eight layers for every dataset, scale and implementation. [Jamba, §6.1](https://arxiv.org/html/2403.19887v1#S6.SS1)

## 3. What the Mamba and attention layers actually carry

### Mamba: a selective update, followed by a read

Recall the [state-space lesson](/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals): an input-dependent recurrence can choose how strongly to retain old state, write new information and read it back. In a Mamba-1-style channel, a simplified indexing of the implemented update is

$$
H_{t,i,n}
=e^{-\Delta_{t,i}e^{a_{i,n}}}H_{t-1,i,n}
+\Delta_{t,i}B_{t,n}u_{t,i},
\qquad
y_{t,i}=\sum_nC_{t,n}H_{t,i,n}+D_i u_{t,i}.
$$

Here $i$ indexes the expanded channels and $n$ indexes state coordinates within a channel. $\Delta$ is positive; $e^{a_{i,n}}$ is a learned positive decay rate. $B_t$ and $C_t$ depend on the current transformed input. This update is affine in the old state once the current input and coefficients are fixed, while the entire input-to-output model remains nonlinear.

The input $u_t$ has already passed through a learned projection, a short causal depthwise convolution and an activation. The read is gated and projected back into the residual stream. A width-4 convolution needs recent projected inputs too: retaining the recurrent matrix while discarding the convolution history does not preserve the layer's computation.

[Figure J06: expanded channels by state-coordinate matrix; one row update is unpacked into retain, write and read. A small four-slot convolution buffer appears beside the matrix, as a distinct piece of request state.]

Jamba adds RMSNorm to the low-rank timestep features and to $B$ and $C$ before the selective scan. The timestep features are subsequently projected and passed through softplus. Normalizing a final residual output is not an equivalent substitution. The [official implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/jamba/modeling_jamba.py) also distinguishes full-sequence computation from state updates during decoding.

This recurrence uses the common Mamba input discretization $\Delta B u$. It should not be silently substituted for the exact zero-order-hold expression of an arbitrary continuous-time system. The earlier state-space lesson derives that distinction.

With fixed channel and state dimensions, each recurrent update has a fixed amount of work, so processing $T$ inputs this way requires work proportional to $T$. Projection and FFN work must still be included. The teaching program uses an explicit loop; optimized implementations can exploit the associative composition of the state-update maps to parallelize a scan. A Python loop's timing therefore does not predict an optimized Mamba kernel's timing.

### Attention: retain separate projected keys and values

For a causal head, position $t$ forms a query $q_t$. Each permitted position $j\le t$ supplies a key $k_j$ and a value $v_j$:

$$
\alpha_{t,j}
=\frac{\exp(q_t^\top k_j/\sqrt{d_h})}
{\sum_{r\le t}\exp(q_t^\top k_r/\sqrt{d_h})},
\qquad
o_t=\sum_{j\le t}\alpha_{t,j}v_j.
$$

The causal mask excludes future positions. It does not exclude the current position in the convention used here. During generation, caching prior keys and values avoids computing their projections again. The new query still has to interact with the retained keys and values.

Grouped-query attention lets several query heads share each key/value head. In the original Jamba configuration, 32 query heads share eight key/value heads, with width 128 per head. Cache storage depends on **eight**, while the attention score calculations still serve **32** query heads. See [multi-head attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals) for the full head construction.

[Figure J07: four labelled query-head lanes fan into one shared K/V bank; four such banks are not drawn as four independent copies of each query's cache. A triangular mask inset identifies visible positions.]

### Where does order enter?

Jamba does not use RoPE or another explicit positional embedding in its original release. Its recurrent and causal-convolution operations are order-sensitive, so the later attention layers receive representations that can depend on preceding order. The original paper tested RoPE and no-RoPE variants; this does not establish that positional mechanisms are unnecessary in every hybrid.

For example, processing A then B through a recurrence generally differs from processing B then A. An attention layer can operate on those different contextualized vectors even without a separately added position vector. This is different from declaring that an unordered collection of raw token embeddings contains order information.

The model below deliberately includes simple position tags in **all** its neural candidates, including the all-attention baseline. That makes its local comparison easier to interpret; it is one reason it is a teaching model rather than a Jamba checkpoint replica.

## 4. Calculate the cache before making a hardware claim

Separate model weights, persistent request state and temporary working memory. A small KV cache says nothing by itself about whether hundreds of billions of weights fit on a device.

For batch size $B$, context length $T$, $L_A$ full-attention layers, $H_{KV}$ key/value heads, head width $d_h$ and $b_{KV}$ bytes per stored scalar,

$$
M_{KV}=B L_A T\,2H_{KV}d_h b_{KV}.
$$

The factor two represents keys and values. For $L_M$ Mamba layers with expanded width $I$, state width $N$ and convolution-buffer width $C$,

$$
M_{\mathrm{rec}}=B L_M I N b_{\mathrm{rec}},
\qquad
M_{\mathrm{conv}}=B L_M I C b_{\mathrm{conv}}.
$$

The exact buffer layout is an implementation choice. Our accounting reserves $C$ projected inputs per channel, a common update-buffer layout; a minimal mathematical representation could retain only the previous $C-1$. State and convolution dtypes must be stated separately.

Consider the original 32-layer shape: $L_A=4$, $L_M=28$, $I=8192$, $N=16$, $C=4$, eight KV heads and head width 128. Assume batch one, 2-byte K/V and convolution storage, and 4-byte recurrent state. Each attention layer stores 4,096 bytes per token. Across Mamba layers, recurrent state takes 14 MiB and convolution buffers take 1.75 MiB.

| Context tokens | All-attention 32-layer cache | Hybrid K/V | Hybrid recurrent + convolution | Hybrid total |
|---:|---:|---:|---:|---:|
| 1,024 | 0.125 GiB | 0.015625 GiB | 0.015381 GiB | 0.031006 GiB |
| 4,096 | 0.5 GiB | 0.0625 GiB | 0.015381 GiB | 0.077881 GiB |
| 16,384 | 2 GiB | 0.25 GiB | 0.015381 GiB | 0.265381 GiB |
| 65,536 | 8 GiB | 1 GiB | 0.015381 GiB | 1.015381 GiB |
| 262,144 | 32 GiB | 4 GiB | 0.015381 GiB | 4.015381 GiB |

These are calculated tensor sizes, not measured GPU allocations. GiB means $2^{30}$ bytes; GB means $10^9$ bytes. The comparison keeps head configuration and depth fixed. It makes no assertion that the two networks have equal parameter count or predictive quality.

[Figure J08: three additive cache bands versus context length, with the recurrent and convolution bands separately inspectable. Include exact bytes in a table and distinguish the all-attention hypothetical comparator from an actual model.]

The K/V component is exactly eight times smaller in this comparison. The total cache is not exactly eight times smaller because the hybrid also carries recurrent state. At zero processed tokens, our preallocated hybrid buffers already have a nonzero size; the all-attention K/V count is zero. Allocators may reserve memory differently.

Full attention remains present, so this hybrid's cache still grows with context length. Replacing all its attention layers with a fixed-size sliding window would eventually bound the K/V count too, but would remove direct reads of keys outside that window. That is a changed computation, not a free cache optimization.

**Investigation JC — build a memory budget.** Begin with the fresh 12-layer configuration: three attention layers, batch three, width 512, expanded width 1,024, two KV heads of width 64, state width eight and convolution width four. At 2,048 tokens, inspect which components change when context doubles. Then change the attention count, batch size or dtypes. A second view can show a hypothetical windowed operator, with its lost direct-read range drawn explicitly.

### Optional: prefill arithmetic is different from one-token decoding

Prefill processes the supplied prompt. Decode produces a new token using the accumulated state. Ignoring bias additions, normalization, softmax and implementation padding, and counting a multiply and add as two operations, attention projections cost

$$
4BTd(d+H_{KV}d_h)
$$

when query and output widths both equal $d$. The two attention matrix products over an ideally evaluated causal triangle cost

$$
4Bd\frac{T(T+1)}{2}=2BdT(T+1).
$$

A single new token after $T$ retained positions instead requires approximately $4Bd(T+1)$ pair-product operations, plus its projections. The history-dependent term is linear per new token and quadratic when accumulating a whole full-attention sequence. Actual kernels may evaluate additional tiles.

At $T=32,768$, $d=4,096$, $H_{KV}d_h=1,024$ and $B=1$, one attention layer's projections contribute $2.748779\times10^{12}$ operations and the ideal causal pair products contribute $8.796361\times10^{12}$. These are **per prompt**, not per token. The recurrence and FFN costs also matter; the table is not a complete model FLOP estimate or a latency forecast.

[Figure J09: prompt-triangle area versus one additional query row. Place the projection term outside the triangle so it cannot be confused with attention-pair work.]

## 5. Experts are a separate architectural choice

At an MoE feed-forward layer, a router scores experts using the current token vector. Only selected experts transform that vector. This changes which **parameters** participate; it does not select which earlier **tokens** attention reads.

For a SwiGLU expert,

$$
F_e(x)=W_{\mathrm{down},e}
\big[\operatorname{SiLU}(W_{\mathrm{gate},e}x)
\odot W_{\mathrm{up},e}x\big].
$$

Each of its three matrices contributes $df$ parameters when the model width is $d$ and intermediate width is $f$. Ignoring biases, that is $3df$ parameters per expert. The gating branch is why a two-matrix estimate is wrong here.

Let $p=\operatorname{softmax}(r(x))$ and let $S$ contain the top two indices. In the inspected Jamba implementation, the contribution is

$$
z(x)=\sum_{e\in S}p_e F_e(x).
$$

The selected weights retain their probabilities from the full softmax; they are not divided by their selected sum. Other MoE implementations use that additional renormalization. The distinction changes the function and must match the checkpoint.

[Figure J10: a router bar chart, selected experts and a residual bypass. Show unselected probability mass as uncomputed expert contributions, not “missing tokens.”]

For scores $[\log4,\log2,0,0]$, probabilities are $[1/2,1/4,1/8,1/8]$. If the first two expert outputs are $[2,-1]$ and $[0,3]$, Jamba-style mixing gives

$$
\tfrac12[2,-1]+\tfrac14[0,3]=[1,\tfrac14].
$$

Renormalizing the selected weights would give $[4/3,1/3]$. Neither convention can be inferred merely from the phrase “top-2.”

**Investigation JD — selected experts, retained mass.** Use four fresh expert outputs and edit router scores. First observe whether raising the score of an expert that remains unselected can change the output. Run the comparison and trace any change through the probability calculation. Contrast that with changing an unselected expert's output while leaving its score unchanged. Then compare retained-mass and renormalized mixing, and inspect a tie at the selection boundary.

The broader [Mixture-of-Experts lesson](/learn/path/full-curriculum/mixture-of-experts-transformers-moe?module=deep-learning-fundamentals) develops expert specialization, balancing losses, capacity, dispatch and training gradients. Locally, remember that storing 16 experts and executing two does not make 14 experts disappear from model memory. Across a batch, different tokens may activate many different experts, requiring weight movement and possibly communication among devices.

### Optional: count a real-shaped FFN pool

Using $d=4,096$, $f=14,336$, 16 dense FFNs and 16 MoE FFNs with 16 experts each:

| Quantity | Calculation | Parameters |
|---|---|---:|
| One SwiGLU expert | $3df$ | 176,160,768 |
| All FFN weights | $(16+16\cdot16)3df$ | 47,915,728,896 |
| FFN weights active for one token | $(16+16\cdot2)3df$ | 8,455,716,864 |
| Hypothetical 32 dense FFNs | $32\cdot3df$ | 5,637,144,576 |

The FFN pool is 8.5 times the all-dense pool, while active FFN parameters are 1.5 times as many. These counts exclude mixers, embeddings, routers and normalization. They explain how the original model can have roughly 52B total and 12B active parameters, without inventing a claim that sparse capacity is free.

[Figure J11: two aligned parameter shelves, “stored” and “active for this token,” with the untouched model-weight shelf still visible during inference.]

## 6. Distinguish the family from a particular release

| Release | Total / active parameters, reported approximately | Depth and width | Architecture note |
|---|---|---|---|
| Jamba-v0.1 | 52B / 12B | 32 layers, width 4,096 | Original base model; four attention layers |
| Jamba-1.5 Mini | 52B / 12B | Same model-size family | Updated, instruction-tuned successor |
| Jamba-1.5 Large | 398B / 94B | 72 layers, width 8,192 | Nine eight-layer cycles, nine attention layers |

The larger model is not the 32-layer configuration with a larger label attached. Its 64 query heads and eight KV heads also differ from the original model's 32 query heads. Jamba-1.5 retained Mamba-1 after its authors compared Mamba-1 and Mamba-2 hybrids; a newer mixer was not automatically better in their tested combination. [Jamba-1.5, §2](https://arxiv.org/html/2408.12570v1#S2)

At 52B parameters, two bytes per weight alone is approximately **104 billion bytes**, before caches and workspaces. An original single-80GB deployment discussion depended on quantization. The 1.5 paper's ExpertsInt8 technique stores selected feed-forward weights in int8 and dequantizes within the compute kernel. It does not turn the entire model into a uniformly int8 network or mean that only active expert weights require storage.

There is also a numerical-range issue, separate from byte counts. The largest finite float16 value is 65,504; bfloat16 has a much wider exponent range. A computation that remains finite in bfloat16 can therefore overflow after an inference system switches its activations to float16. The Jamba-1.5 authors reported large internal activations and added a small penalty proportional to their mean squared magnitude. This is an example of adapting training to a serving constraint, not a universal requirement to add the same coefficient to every hybrid. [Jamba-1.5, §3.2](https://arxiv.org/html/2408.12570v1#S3.SS2)

For later releases, inspect the exact model card and configuration again. The [Jamba Mini 1.7 card](https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.7), checked on 13 September 2026, describes a 256K context and instruction/grounding updates, and explicitly distinguishes its multi-GPU BF16 requirements. Its example model identifiers and repository title are not consistently ordered, so verify the resolvable identifier before copying a deployment command.

[Figure J12: a release comparison with separate rows for weights, K/V shape and recurrent state. A dated source badge accompanies the configuration facts; there is no synthetic quality leaderboard.]

Model capabilities require training too. Jamba-1.5 describes pretraining, a stage emphasizing long documents, and post-training that mixes skill data with long-context examples. A maximum input-length field, or the ability to allocate a cache that large, does not prove reliable reasoning throughout that context.

## 7. Train and inspect a small hybrid on real pen trajectories

The [UCI PenDigits dataset](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits) contains digit traces collected from 44 writers with a tablet. Each released example contains eight $(x,y)$ points and a digit label. These points were **resampled at approximately regular distance along the stroke**, not at regular time intervals. Connecting them produces a path; treating them as an image grid or attaching a time-in-seconds axis would misrepresent the data.

The original dataset provides 7,494 development rows from 30 writers and 3,498 assessment rows from 14 different writers. Individual writer IDs are not supplied in these flat files. We preserve the official boundary, then use a deterministic classwise shuffle within the development file to select 100 fitting and 30 validation rows per digit. The remaining development rows stay unused. All 10,992 coordinate sequences are distinct in an exact-feature check, with no conflicting labels or cross-file duplicates.

[Figure J13: an eight-point pen trajectory with point numbers, direction arrows and the original 0–100 normalized-coordinate axes. Beside it, show its eight-row sequence representation. Do not invent pen speed, stroke breaks or pressure.]

We divide coordinates by 50 and subtract one, giving values in $[-1,1]$. This fixed transformation learns nothing from validation or assessment rows. All neural candidates receive the same four features at position $t=0,\ldots,7$:

$$
[x_t/50-1,\;y_t/50-1,\;t/7,\;(t/7)^2].
$$

The position tags are fixed to the original eight-point grid. They must not be rescaled when inspecting only a prefix or restarted at a chunk boundary.

### The model and protocol

The neural models embed each four-feature vector into width 16 and pass it through three causal residual layers. M denotes a selective recurrent mixer with state width four and a width-3 causal convolution. A denotes one causal attention head. Every layer has a dense SwiGLU FFN with intermediate width 32. A final RMSNorm and linear classifier read the last position and produce ten logits.

The recurrent mixer includes input-dependent writes, reads and positive timesteps, plus a learned decay, a skip term and an output gate. It omits Jamba's expansion and low-rank timestep factorization. The attention has one head rather than GQA. These simplifications keep every state update inspectable. No MoE is trained in this experiment; expert routing was isolated in §5.

[Figure J14: four three-layer patterns MMM, AAA, MAM and AMM. Each ends in the same last-position classifier. Parameter counts appear beside the actual patterns rather than under an “equal capacity” label.]

The fitting objective is cross-entropy:

$$
\mathcal L=-\frac1m\sum_{i=1}^{m}
\log\operatorname{softmax}(z_i)_{y_i}.
$$

All runs use 80 epochs, batches of 100, Adam at learning rate 0.003, gradient-norm clipping at one and the same sequence of batch permutations. Each run selects its lowest validation cross-entropy checkpoint, retaining the earliest checkpoint if losses tie. We predeclare four neural patterns at seed 37, repeat MAM at seed 73, and include a 170-parameter linear classifier on the flattened 16 coordinates. A matching depth does not imply matching parameter counts.

The complete [training and model program](stroke_models.py), [input data and provenance](data-provenance.md), saved fitted arrays and result files accompany this lesson. In a separate Python environment with NumPy and PyTorch installed, keep the companion files in one directory and run:

~~~sh
python stroke_models.py
python inspect_stroke.py --row 2452 --break-at 3 --mode carry
python inspect_stroke.py --row 2452 --break-at 3 --mode convolution-reset
~~~

The first command prints saved results without training. To reproduce all six fits, run `python stroke_models.py --train`. This author run used CPU PyTorch 2.14.0 and NumPy 2.3.5. No foundation-model weights or GPU kernels are required for these companions.

Here is a complete small inspection program using the supplied modules:

~~~python
import numpy as np
from stroke_models import HERE
from inspect_stroke import run_trace

rows = np.loadtxt(HERE / "pendigits.tra", delimiter=",")
coordinates = rows[2452 - 1, :16].reshape(8, 2)

for mode in ("carry", "recurrent-reset", "kv-reset", "convolution-reset"):
    result = run_trace(coordinates, key="MAM-37", break_at=3, mode=mode)
    print(
        mode,
        result["branch_prediction"],
        round(result["branch_probabilities"][0], 6),
        round(result["maximum_logit_difference"], 6),
    )
~~~

It reports the predicted digit, probability assigned to zero and largest difference from uninterrupted logits. “Recurrent reset” clears both the recurrent state and that mixer's convolution history; “convolution reset” isolates the latter.

The key residual update in the full program is short enough to inspect directly:

~~~python
def forward(self, x):
    x = x + self.mixer(self.norm1(x))
    return x + self.channel(x)
~~~

The channel method normalizes its input, projects separate gate/up branches, multiplies the SiLU gate by the up branch and projects the result back. The selective mixer's step method returns both its output and its updated pair of caches. The attention step appends K/V tensors and reads the complete permitted bank. Thus training, full-sequence inference and streaming implement the same intended function.

### Results, including the uncomfortable comparisons

| Candidate | Parameters | Selected epoch | Fit errors / 1,000 | Validation errors / 300 | Assessment errors / 3,498 |
|---|---:|---:|---:|---:|---:|
| Flattened linear, seed 37 | 170 | 80 | 114 | 30 | 565 |
| MMM, seed 37 | 9,314 | 79 | 2 | 7 | 201 |
| AAA, seed 37 | 8,474 | 64 | 3 | 12 | 167 |
| MAM, seed 37 | 9,034 | 56 | 0 | 8 | 193 |
| AMM, seed 37 | 9,034 | 68 | 6 | 12 | 261 |
| MAM, seed 73 | 9,034 | 78 | 2 | 9 | 183 |

These are actual runs. MMM has the fewest validation classification errors, while AAA has the fewest assessment errors in this table. MAM is not the overall winner. Its second seed also differs from its first. We do not select an architecture retrospectively using the assessment column.

The comparison supports several useful conclusions. Learned sequence models can recognize these real trajectories. The arrangement of mixers can change fitted behavior even at equal parameter count: MAM and AMM differ. A hybrid need not beat every simpler model, and performance on eight-point handwriting does not establish language-model retrieval or 256K-context throughput.

Selection used validation **cross-entropy**, not the displayed error count. That distinction matters when a model becomes more confident on already-correct examples or makes a few highly confident mistakes. Keep both measures when interpreting learning curves.

[Figure J15: validation cross-entropy over epochs from the retained histories, with each selected checkpoint marked. A separate error-count table preserves all six assessment outcomes; no smoothed invented curves or cross-architecture latency axis.]

### Follow a prediction through a real cache

At a three-point boundary, the MAM model carries:

| Layer | Persistent object for one request | Elements |
|---|---|---:|
| M0 | State $16\times4$; prior projected inputs $16\times2$ | 64 + 32 |
| A1 | Three keys and three values, each width 16 | 96 |
| M2 | State $16\times4$; prior projected inputs $16\times2$ | 64 + 32 |

Our tiny convolution implementation retains the previous two inputs, then appends the new one; its cache layout therefore differs from the four-slot accounting convention in §4. Count the representation that the program actually stores.

For validation source row 2,452, labelled zero, the uninterrupted model assigns probability 0.997078 to zero. Carrying all caches reproduces it to floating-point tolerance. Clearing convolution history at the boundary changes intermediate logits by about 9.03, yet the final predicted digit remains zero. Checking only the final label would miss that serious computation change.

Clearing K/V also changes this example's logits while preserving its final class. A changed class is strong evidence of a behavioral difference; an unchanged class is not evidence of equivalence.

[Figure J16: a request timeline split after point three. Draw two compact recurrent matrices, two short buffers and the growing attention bank, each connected to its own layer. Branches show correct carry and individually cleared state types, with a logit-difference strip.]

**Investigation JB — continue the same request.** Start with the different validation trace at source row 2,970. Predict what will happen when only K/V is cleared after point three. Compare the entire probability vector and each prefix logit, then inspect recurrent reset, convolution reset and position-offset reset. Edit the actual coordinate points and make a live comparison before rerunning. A digit that still looks similar to a person may change its model representation.

The saved model was trained to classify after all eight points. Earlier logits reveal its computation, but are not calibrated promises that it can reliably classify every partial stroke.

## 8. Preserve request identity when serving

A recurrent state is specific to the prefix, model weights, layer and request that produced it. It is not a global scratch buffer that can be handed to the next user.

Suppose requests R and S arrive interleaved. To process the next token of R, the server needs R's state and convolution history at every recurrent layer, R's K/V banks at every attention layer, and the correct processed-token count. Reusing S's state can produce a plausible-looking answer that depends on the wrong input.

[Figure J17: two request lanes with separate cache bundles, a scheduler selecting R or S, and layer-specific state returning to the same lane. No visual implies that the model weights are duplicated per request.]

Prefill may be divided into chunks. If a prefix has already been processed, the next chunk must continue from all required state objects. The small model verifies this by comparing full-sequence and token/chunk paths, including gradients. Its largest full-versus-stream output discrepancy across the five neural models' 300 validation sequences was about $1.26\times10^{-5}$ in float32. A separate two-sequence float64 MAM check gave output discrepancy $1.42\times10^{-14}$ and parameter-gradient discrepancy below $3.1\times10^{-15}$. These are author calculations for this small implementation.

Prefix reuse needs more than matching text. The cached state must correspond to the same model revision and any adapters, the same relevant tokenization, the same valid prefix boundary and all necessary layer types. To branch generation from a prefix, preserve an immutable snapshot or use copy-on-write; do not let the first continuation overwrite the second continuation's starting state.

Rewinding is different for the two storage forms. Removing the final K/V entries can undo their append operation. A recurrent update is not generally invertible: decay, selective writes and finite precision may have lost information. To rewind, restore an earlier snapshot or recompute the prefix. Speculative decoding, beam branching and editing previous input all need this distinction.

The [vLLM hybrid-cache design document](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) explains why allocation and prefix-hit rules differ by layer type. Its described algorithm is tied to a named implementation revision and labels parts of Mamba prefix caching as work in progress. Treat it as an explanation of the engineering problem; check the installed release's actual support before depending on a particular optimization.

### A deployment experiment that answers a real question

Begin with a specific workload: perhaps searching a collection of incident reports for all references to a failing component and returning cited evidence. Decide what counts as a correct answer before comparing models.

Use prompts that vary independently in length, evidence position, number of relevant records and distractor similarity. Include missing-evidence questions, conflicting statements and queries requiring several records to be joined. A single inserted sentence that repeats the answer verbatim tests a useful ability, but not all of document reasoning.

Measure time to first token, time per output token, total request latency, peak memory, and supported concurrent requests at fixed output length and sampling settings. Record the GPU count/model, dtypes, quantization, software revision, kernel path and batching policy. Separate a cold model load from warm serving.

Compare models at a constraint you actually care about: acceptable task quality under a fixed memory budget, or throughput while meeting a latency limit. Equal active parameters, equal total parameters, equal training tokens and equal serving cost are different comparisons. No single invented “relative compute” curve can substitute for them.

The optional [deployment program](deployment_example.py) shows a complete single-prompt generation path with an explicit model ID and revision, correct prompt handling and input/output token accounting. It is provided for a suitably provisioned environment; it was not executed for this lesson. The bounded author experiment uses only the small stroke models.

For fine-tuning, inspect the real module names before selecting adapter targets. Attention projections, recurrent input/timestep/output projections and FFNs provide different adaptation choices. Attention-only adaptation is a valid restricted experiment, not automatically a bug; broader targets may help at additional cost. Compare on held-out tasks, including the original context-length requirements. There is no universal rule that recurrent layers require exactly twice the learning rate or a different clipping threshold.

### Reuse primitives, own the composition and the cache

The core implementation is [stroke_models.py](stroke_models.py). `SelectiveMixer.step` explicitly computes the positive step size, stable negative decay rates, selective writes/reads and causal convolution history. `AttentionMixer.forward` constructs masked scores; `AttentionMixer.step` extends the exact K/V cache. `Layer` composes the chosen mixer with the residual and gated channel path, and `StrokeModel` exposes both full and streamed execution. Thus the learner can build the hybrid's defining behavior without importing a ready-made Jamba block.

These classes use ordinary `nn.Linear`, `nn.Conv1d`, tensor operations, parameter registration and Adam. The full study supplies fitting and `state_dict` restoration. For a released model rather than our small instructional network, [deployment_example.py](deployment_example.py) supplies the distinct Transformers tokenizer/model/generation route, with an explicit checkpoint revision and resource assumptions. Its learned projections, dimensions, routing and cache classes belong to that selected release; our tiny output is not a numerical oracle for an unrelated pretrained model. The deployment program remains unexecuted and requires appropriate CUDA, kernels and model storage.

Underlying attention, SSM and expert derivations are taught in their named earlier **prepared** lessons; their new website implementations may still be pending. This packet stays self-contained for its own simplified mixers and routing calculation. It does not claim to have recreated a fused Mamba kernel or trained a full Jamba MoE checkpoint. `hybrid_mechanisms.py::route` explains the selected-expert probability contract; sparse trainable expert dispatch is owned by [the prepared MoE program](../mixture-of-experts-transformers-moe/moe_study.py), not the dense channel network used in this stroke experiment.

**Control request identity.** Run two distinct trajectory prefixes, save each cache and position offset, then continue each with its own suffix. Compare with independent unsplit passes. Swap only the caches, then restore the correct pair.

<details><summary>Hint and reasoned solution</summary>

A request state comprises every layer's recurrent/convolution or K/V state plus its next logical position. Keep an independent cache container for each request; `stream` updates the supplied list, so a shallow alias shared by two requests is unsafe. Full and correctly carried execution should agree within the declared float32 tolerance. Swapping caches changes the past information, and restarting the position offset changes the embedding even if the past tensors were right. Winning labels can remain unchanged, so compare prefix logits and individual state arrays. Recurrent carried storage stays fixed at fixed dimensions; attention K/V storage grows with retained sequence length. The current repeated `torch.cat` in the small reference may copy the old cache; a production serving cache would preallocate or page it while preserving this identity contract.

</details>

## 9. Optional branches: other ways to combine memory

### Share weights across depth: Zamba and Zamba2

Zamba repeatedly invokes a shared attention/MLP module along a Mamba backbone. Its shared module receives a concatenation of the current residual representation and the original input embedding. “Shared” refers to parameters reused at several depths, not to applying attention only to every sixth token. [Zamba, §II](https://arxiv.org/html/2405.16712v1#S2)

This can save parameter storage while still doing attention computation at each invocation. If two invocations receive different hidden vectors, the same projection weights can produce different keys and values. Weight sharing **alone** does not prove that their KV caches can be identified. An architecture must explicitly define any cache sharing.

The [Zamba2 model card](https://huggingface.co/Zyphra/Zamba2-7B) describes Mamba-2, two alternating shared blocks, depth-specific low-rank adapters and RoPE in the shared attention. These are concrete design changes, not interchangeable names for the original Jamba pattern.

### Use local attention: Griffin and Samba

[Griffin](/learn/path/full-curriculum/long-context-sequence-models-transformer-xl-griffin-perceiver?module=deep-learning-fundamentals) combines a gated linear recurrence with local attention. Samba combines Mamba, sliding-window attention and separate FFNs. Its attention directly reads the window; information from further back must arrive through recurrent state or representations propagated through the stack. A fixed window can bound attention storage during streaming.

“The program can continue processing” and “the model can recover any detail from arbitrarily far back” are different claims. Samba's reported length-generalization and passkey experiments have particular data and training conditions. They do not establish unlimited accurate memory for arbitrary histories. [Samba, §§2–3](https://arxiv.org/html/2406.07522v1)

### Fuse heads inside a layer: Hymba

Hymba sends the same layer input to attention and SSM branches in parallel, normalizes and rescales their outputs, then combines them. It also uses combinations of local/global attention, explicit cross-layer KV sharing and learned prefix meta tokens. The latter can serve as learned cache initialization; they are not new external facts retrieved at inference time. [Hymba, §2](https://arxiv.org/html/2411.13676v1#S2)

A useful distinction is whether the second operation receives the first operation's transformed output, as in a serial composition, or both receive the same representation before fusion. Neither composition universally dominates the other; the information flow and parameter budget differ.

[Figure J18: four small architecture diagrams: serial Jamba, repeated shared-parameter Zamba, windowed Samba, parallel Hymba. Distinguish weight-sharing loops from cache-sharing arrows. Each caption names the precise difference instead of assigning a quality rank.]

As a more recent deployment example, [IBM's Granite 4.0 documentation](https://www.ibm.com/granite/docs/models/granite4-0) distinguishes hybrid Mamba-2 variants, dense versus MoE variants, and traditional alternatives. The model suffix and configuration matter: an organization or family name does not mean every model uses an SSM.

## 10. Connect the mechanism to a useful application

The handwriting example suggests a less obvious application of hybrid thinking: a stream can contain both an evolving shape and specific landmarks. A turn near the beginning may distinguish two otherwise similar strokes. A recurrent summary and an address-sensitive operation provide different routes for learning that distinction. Our measured results also show why the architecture still needs to earn its place against simpler baselines.

For incident reports, a running representation could combine a chain of symptoms, while attention could make an earlier part number available when answering a question. To test that hypothesis, change only the part number while holding the story fixed, then change the causal ordering while holding the identifiers fixed. Measure whether the model responds to the information each question requires.

For scientific notebooks, the needed link may be between a final conclusion and a parameter setting far earlier in the record. A useful evaluation asks for the setting **and its evidence location**, includes repeated parameter names in unrelated experiments, and distinguishes the final setting from a superseded value. Increasing context capacity helps only if the model and application preserve those distinctions.

These are application designs to investigate, not observed capabilities of our small classifier. For exact identifiers or auditable numerical records, an external database or structured retrieval step can be a better source of truth than asking any neural architecture to remember and regenerate the value unaided.

The architecture decision can now be made as a sequence of practical questions:

1. What must be preserved: a state summary, addressable details, local structure, or a combination?
2. Which operation provides a route for that information, and where in depth does it enter?
3. What grows with input length, batch size and model width?
4. Can the implementation carry every state correctly through the actual serving workflow?
5. Does the measured task improvement justify the memory, training and software cost?

## 11. Practice

Attempt each question before opening its hint. Questions 1–6 cover the core route; 7–10 explore deeper architecture and engineering decisions.

### 1. An identical summary, a different answer

With $\lambda=1/2$, do $[1,4,2]$ and $[5,2,2]$ end at the same summary state? The labels are A, B, C. For query A, use attention weights proportional to $[4,1,1]$. Compute both attention reads and explain what this proves.

<details><summary>Hint</summary>
Unroll the numerator to $\frac14v_1+\frac12v_2+v_3$. The normalizer is the same for equal sequence lengths.
</details>
<details><summary>Solution</summary>
Both numerators are 4.25 and both normalizers are 1.75, giving 17/7. Attention gives $(4+4+2)/6=5/3$ for the first sequence and $(20+2+2)/6=4$ for the second. This summary cannot distinguish these histories; separate keyed values can. Neither attention result is guaranteed to equal its label-A value, and the example does not prove that all recurrent states have this collision.
</details>

### 2. A different cache budget

A model has 24 layers, six using full attention. Batch size is two, with four KV heads of width 64 and two-byte K/V storage. At 8,192 tokens, calculate K/V bytes. Then double only the sequence length. Which other model information would be needed to calculate total request state?

<details><summary>Hint</summary>
Multiply batch, attention layers, tokens, two banks, KV heads, head width and bytes per scalar.
</details>
<details><summary>Solution</summary>
The count is $2\cdot6\cdot8192\cdot2\cdot4\cdot64\cdot2=100,663,296$ bytes, or 96 MiB. Doubling context gives 192 MiB. Recurrent-layer count, expanded/state dimensions, convolution history layout and their dtypes are needed for the remaining persistent state. Allocator overhead, weights and temporary memory are additional concerns.
</details>

### 3. Repair the residual

An implementation uses $x_{\mathrm{next}}=x+\mathrm{FFN}(\mathrm{Norm}(x+\mathrm{Mix}(\mathrm{Norm}(x))))$. Explain the missing path. In a scalar example with $x=2$, mixer output 3 and FFN output 4, compare it with the intended two-residual layer.

<details><summary>Hint</summary>
Name the intermediate value after the mixer addition before writing the second addition.
</details>
<details><summary>Solution</summary>
The first intermediate value is $u=2+3=5$. The intended result is $u+4=9$. The displayed implementation returns $2+4=6$, so the mixer contribution lacks its direct residual path into the final output. The FFN still depends on it, but that is a different function.
</details>

### 4. Where did the missing probability mass go?

Four router probabilities are $[0.4,0.3,0.2,0.1]$. Top-2 outputs are $[5,0]$ and $[0,5]$. Compute the retained-probability mixture and the renormalized mixture. If only the third expert's output changes, which mixture changes?

<details><summary>Hint</summary>
The selected mass is 0.7. Selection and score computation are held fixed.
</details>
<details><summary>Solution</summary>
Retained mixing gives $[2,1.5]$. Renormalized mixing gives $[20/7,15/7]$. Neither changes when only an unselected expert's output changes, since that output is not evaluated in the mixture. Changing its router score is different and can affect the retained weights even before selection changes.
</details>

### 5. A correct final label hides a cache bug

The model returns digit zero both before and after its convolution history is cleared mid-request. A teammate concludes that convolution history can be discarded. Give a concrete verification strategy and a null case where clearing state really has no effect.

<details><summary>Hint</summary>
Compare numeric outputs at every position under the same weights and inputs, not just an argmax.
</details>
<details><summary>Solution</summary>
Compare full-sequence logits with correctly carried streaming logits, then compare the damaged branch. Check more than one input and inspect the point immediately after the boundary. Intermediate differences or probability changes refute equivalence even when the final class stays zero. Clearing an already-empty cache before the first token is a true null; clearing after the last token cannot alter already-produced outputs. These nulls do not justify clearing a nonempty midstream cache.
</details>

### 6. Choose without looking at the assessment answers

Two candidate models have validation cross-entropies 0.12 and 0.15; their assessment error counts are 50 and 30. Your declared selection rule was lowest validation cross-entropy. Which candidate is selected, and what do you do with the second candidate's better assessment count?

<details><summary>Hint</summary>
Separate an honest report from a new selection decision.
</details>
<details><summary>Solution</summary>
Select the first candidate under the declared rule. Report both assessment results as part of the planned comparison, including the reversal. Do not relabel the second as the validation-selected winner. If the reversal motivates a new selection rule or training experiment, develop it using development evidence and evaluate the resulting decision with an independent assessment protocol.
</details>

### 7. Shared weights do not guarantee shared keys

A shared key projection is the identity. It is called at two depths with vectors $[1,0]$ and $[0,1]$ for the same token. Can the two invocations use the same stored key without changing either computation? What extra design would make cache sharing legitimate?

<details><summary>Hint</summary>
Apply the projection before arguing from parameter identity.
</details>
<details><summary>Solution</summary>
The keys are $[1,0]$ and $[0,1]$, so replacing one with the other changes its attention scores. Explicit architectural sharing could define both layers to consume a key bank generated by a designated layer, with that rule used during training and inference. That is a different contract from merely tying projection weights.
</details>

### 8. Active weights and stored weights

A toy network has eight FFN positions. Four are dense; four have four experts and select two. Each SwiGLU expert has model width 16 and intermediate width 32. Calculate total FFN parameters, active FFN parameters for one token, and the corresponding all-dense total.

<details><summary>Hint</summary>
One expert has three matrices, each with $16\cdot32$ entries.
</details>
<details><summary>Solution</summary>
One expert has 1,536 parameters. Stored FFN parameters are $(4+4\cdot4)1536=30,720$. Active FFN parameters are $(4+4\cdot2)1536=18,432$. Eight dense FFNs have 12,288. Sparse selection increases the parameter pool while limiting, rather than eliminating, extra work. Router and mixer parameters are excluded.
</details>

### 9. Window the attention

Replace every full-attention layer by a window of 512 positions. A query asks for an identifier introduced 10,000 positions earlier. Explain what information routes remain and why the cache can be bounded without proving the answer will be correct.

<details><summary>Hint</summary>
Direct access to a key is different from information carried forward in hidden representations.
</details>
<details><summary>Solution</summary>
The old identifier's original key lies outside every individual attention window. Its information could persist through recurrent state or be copied into later representations and relayed through layers. Those routes may or may not preserve enough detail. With fixed width and state size, each layer's window cache and recurrent buffers have bounded size, but that storage bound is not a guarantee of arbitrary exact recall.
</details>

### 10. Design a useful hybrid experiment

You want to answer questions about a laboratory notebook. Design two interventions that distinguish identifier retrieval from combining evidence across a sequence, and name the measurements required to decide whether a hybrid is useful.

<details><summary>Hint</summary>
Change one explanatory factor at a time. Include questions whose answers are absent.
</details>
<details><summary>Solution</summary>
For retrieval, change one experiment's identifier or final parameter value while preserving the surrounding narrative and add near-matching distractor identifiers. For sequence integration, reorder a superseded setting and its correction, or move the relevant evidence across several separated entries while holding identifier vocabulary fixed. Score correct answers, evidence citations and appropriate no-evidence responses. Measure TTFT, decode time, total latency, peak memory and concurrency at a recorded model/software/hardware configuration. Compare with a simpler model or structured retrieval baseline under the same task and resource constraints.
</details>

## 12. References and other ways to learn

- **Architecture and evidence:** [Lieber et al., Jamba](https://arxiv.org/html/2403.19887v1). Start with §2 and the actual release shape in §3; then read the ratio, format-following and normalization investigations in §6. The authors distinguish measured behavior from their induction-head hypothesis.
- **Scale and serving:** [Jamba-1.5 technical report](https://arxiv.org/html/2408.12570v1). §§2–3 explain the larger configuration, ExpertsInt8 and activation-range issue; §§5–6 explain training stages and evaluation. Read the hardware and output-length conditions beside throughput figures.
- **Implementation reference:** [Hugging Face Jamba documentation](https://huggingface.co/docs/transformers/model_doc/jamba) and [model source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/jamba/modeling_jamba.py). Follow the two residual additions, normalized timestep/B/C features, selected router weights and distinct cache operations. Names and kernel APIs can change between revisions.
- **Data and independent reproduction:** [UCI PenDigits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits), by E. Alpaydin and F. Alimoglu, [DOI](https://doi.org/10.24432/C5MG6K). The original names file explains spatial resampling and the writer-disjoint official assessment set. The accompanying files retain attribution and the CC BY 4.0 source license.
- **A guided video course:** [Build Long-Context AI Apps with Jamba](https://www.deeplearning.ai/courses/build-long-context-ai-apps-with-jamba/), DeepLearning.AI with AI21, taught by Chen Wang and Chen Almagor. The verified course listing describes nine video lessons, architecture, document prompting, tool calling and context/RAG applications. Its listing was reviewed; the videos and notebooks were not watched or executed for this lesson. Use it for an application-oriented second explanation; access and hosted APIs may require an account.
- **Alternative design explanations:** [Zyphra's training cookbook](https://www.zyphra.com/our-work/the-zyphra-training-cookbook) explains the rationale for shared blocks; pair it with [Zamba](https://arxiv.org/html/2405.16712v1) and the [Zamba2 card](https://huggingface.co/Zyphra/Zamba2-7B). [Samba](https://arxiv.org/html/2406.07522v1) and [Hymba](https://arxiv.org/html/2411.13676v1) expose different choices of locality and within-layer fusion.
- **Serving-state design:** [vLLM hybrid KV cache manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/). Useful after the cache investigation; it shows why allocator groups, padding and prefix-reuse rules cannot all be copied from a homogeneous attention stack.

The next topic in this module is [Titans: Multi-Memory Architecture](/learn/path/full-curriculum/titans-multi-memory-architecture?module=deep-learning-fundamentals). Carry forward the questions learned here: what each memory stores, what updates it, how a read is performed, and what must be preserved when processing the next part of a sequence.
