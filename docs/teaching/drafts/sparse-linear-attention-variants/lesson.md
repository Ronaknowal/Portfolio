# Sparse & Linear Attention Variants

**Explore as you read.** Edit sparse edges, feature-memory writes/evictions, random-feature settings, compression coefficients, block layout and supported real trajectories. Show removed mass, reachability, normalized summaries, approximation error, future influence and tile occupancy live under a fixed random draw. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose sparsity or approximation by accessible information, numerical error and actual block work, not a single sparsity percentage.


A long conversation contains many earlier tokens, but the next token may need only a few of them. A stream of hand movements has the opposite possibility: many earlier observations may matter collectively, without needing to retrieve any single observation exactly. These suggest two different ways to reduce attention's work: **read fewer individual records**, or **maintain a smaller summary that can answer a particular kind of query**.

The distinction matters. Removing connections, approximating a similarity function, compressing the sequence, and executing the same calculation more carefully can all reduce a resource cost. They preserve different things. This lesson gives you a way to inspect those choices rather than memorize a ranking of model names.

**First pass:** follow §§1–3 and the opening of §4, the worked sequence-compression example in §5, and the real trajectory experiment in §7. Try the graph and memory investigations, then practice 1–5. You should be able to explain what information an operator can use, calculate a small output, and distinguish work from retained state. **Deeper pass:** study the random-feature derivation, approximation families, gradients and current sparse systems in §§4–6 and §8, then the remaining practice. Those branches develop implementation and research judgment; they are not hidden requirements for understanding the core story.

We build on [Self-Attention & Multi-Head Attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals), [Transformer Block Architecture](/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals), and the immediately preceding [Multi-Head Latent Attention](/learn/path/full-curriculum/multi-head-latent-attention-mla?module=deep-learning-fundamentals). The refreshers below supply the particular mathematics we need.

## 1. Locate the cost before choosing a shortcut

For one head, a **query** asks what to read; each **key** describes an available record; its **value** is the information returned. At position $t$, let $J_t$ be the legal key positions. For a causal decoder, $J_t=\{0,\ldots,t\}$. The head computes

\[
s_{tj}=q_t^Tk_j/\sqrt{d_k},\qquad
p_{tj}=\frac{e^{s_{tj}}}{\sum_{u\in J_t}e^{s_{tu}}},\qquad
o_t=\sum_{j\in J_t}p_{tj}v_j.
\]

The row of weights sums to one. Its denominator depends on every legal score, so the ordinary softmax cannot simply be moved through a matrix multiplication. In general,

\[
\operatorname{softmax}(QK^T)V\ne
\operatorname{softmax}(Q)(K^TV).
\]

There are several bills to pay, and paying less of one does not cancel the others.

| Resource | What it pays for | What changes it |
| --- | --- | --- |
| Attention arithmetic | Key comparisons and value accumulation | Fewer edges, fewer summary slots, or a different factorable kernel |
| Temporary attention storage | Scores, probabilities and backward intermediates | Tiling, recomputation, checkpointing, fused kernels |
| Persistent inference state | Information retained for future tokens | KV sharing, latent compression, eviction, recurrent summaries |
| Whole-model work | Projections, FFNs, routing, normalization and communication | Architecture and implementation beyond the attention core |

For length $L$, dense causal attention has $L(L+1)/2$ legal pairs per head. Computing their dot products and weighted values costs order $L^2(d_k+d_v)$. A simple implementation allocates a full square score tensor, including cells later masked. At batch one, 32 heads and two bytes per score, that tensor alone is 4 GiB at $L=8192$, and 64 GiB at $L=32768$: $32L^2\times2$ bytes, with $1\text{ GiB}=2^{30}$ bytes. These are allocation calculations, not peak memory measurements or claims about which GPU can train a model.

**Exact attention need not allocate that square tensor.** FlashAttention tiles the same softmax computation and accumulates it using stable running statistics; it reduces transfers and intermediates while retaining the dense operator and its quadratic pair arithmetic. The relevant comparison for a proposed approximation is a strong exact implementation, not only a deliberately materialized reference. [FlashAttention paper](https://arxiv.org/abs/2205.14135)

The previous GQA and MLA lessons reduced what is stored per past token. Sparse attention instead asks which past tokens to read. A recurrent feature method changes how the past is represented. These choices can sometimes be combined, but their equations and costs must still be checked together.

**Visual: four ways through the same records.** Follow one query through (a) all individual keys, (b) a highlighted subset, (c) a small bank of learned sequence summaries, and (d) an accumulated feature-by-value matrix. A separate “execution” overlay tiles case (a) without deleting any mathematical connections. The point is to see what changes, not to award a winner.

## 2. Sparse attention: choose which connections exist

Choose a nonempty subset $S_t\subseteq J_t$, then perform ordinary softmax over that subset:

\[
\tilde o_t=\sum_{j\in S_t}
\frac{e^{s_{tj}}}{\sum_{u\in S_t}e^{s_{tu}}}v_j.
\]

This is exact attention **for the specified sparse graph**. It generally differs from the original dense graph. The retained values are the same records, but their probabilities change because the normalization changes.

### A removed value changes the other weights too

Suppose a dense row has weights $[0.5,0.25,0.25]$ and scalar values $[2,-1,4]$. Its output is

\[
0.5(2)+0.25(-1)+0.25(4)=1.75.
\]

Remove the final key. The two retained weights become $2/3$ and $1/3$, so the result becomes $1$, not $0.75$. The removed probability mass was $\delta=0.25$, but the output changed by $0.75$. Probability mass and output error have different units.

There is a useful bound. Let $\mu_{\rm keep}$ and $\mu_{\rm drop}$ be the normalized weighted averages within the retained and removed sets, with $0<\delta<1$. Then

\[
o=(1-\delta)\mu_{\rm keep}+\delta\mu_{\rm drop},\qquad
o-\tilde o=\delta(\mu_{\rm drop}-\mu_{\rm keep}).
\]

If all values satisfy $\|v_j\|\le R$, the triangle inequality gives $\|o-\tilde o\|\le2R\delta$. Here the bound is $2$, which safely exceeds $0.75$. It can be loose: if every value equals the same vector, dropping keys changes the probabilities but leaves the output unchanged. A small removed mass is helpful evidence, not a complete end-to-end model guarantee; later layers may amplify or damp the change. Also, calculating the exact removed mass ordinarily requires the full reference row, so it is an evaluation diagnostic rather than a free sparse-selection algorithm.

### Windows have a precise reach

In this lesson, a causal window of width **$W$ means at most $W$ total keys, including the current position**:

\[
S_t=\{j:0\le j\le t,\ t-j<W\}.
\]

This convention prevents a common off-by-one error. With $W=3$, position 7 can directly read 5, 6 and 7. For $L=12$, the first rows have 1 and 2 keys; the remaining ten have 3, giving $33$ edges. In general, with $m=\min(L,W)$, the count is

\[
m(m+1)/2+(L-m)m.
\]

Read the attention mask as a graph: row $t$, column $j$ means information can travel **from input $j$ to output $t$** in one layer. A second layer reads the first layer's representations, allowing information to move again. With this window and only tokenwise operations between attention layers, $D$ layers can reach at most $D(W-1)$ positions into the past. Residual connections preserve already reachable information. Parallel heads in one layer do not turn a two-edge route into a two-layer computation.

This is a statement about possible dependence. A reachable record may receive negligible weight or lose information while passing through intermediate vectors. A missing path, however, gives an exact inability to depend on that record under the stated architecture.

### Global, strided and random connections

A **global token** can gather and redistribute information. In a bidirectional encoder, it can read the whole input and other positions can read it in a later layer. Longformer's local-plus-global pattern makes this useful for document classification or question tokens. It also uses separate projections for its global attention. Dilated windows sample nearby offsets more sparsely. [Longformer, §3](https://arxiv.org/html/2004.05150v2)

Causality changes that picture. A global token at position 0 cannot collect information from position 2 in a causal decoder. In our 12-position example, adding a causal hub at 6 creates the path $2\to6\to11$; adding a hub at 0 does not. Every edge must respect time, including edges inside a pooling or routing component. Drawing an undirected star hides this distinction.

Alternating local and strided patterns offers another route. A layer can read a nearby range, then another layer can read positions separated by stride $s$. With $s$ around $\sqrt L$, a construction using local spans of order $s$ and strided spans of order $L/s$ costs order $L\sqrt L$. The exact routes depend on offsets and layer order. Sparse Transformer developed local/strided and fixed-block factorizations for sequence generation, including images and audio; the factorization changes the available computation rather than reproducing every dense head exactly. [Sparse Transformer, §4–5](https://arxiv.org/html/1904.10509v1)

BigBird combines local, global and random connections. With fixed numbers of each, the edge count is linear in sequence length. Its universality result is an existence theorem for sufficiently expressive networks and continuous functions on a fixed-length compact domain, using an appropriate graph containing a global star. It is not a promise that a fixed small model will equal dense attention, nor that a particular number of random edges guarantees task accuracy. The paper also studies genomics, where relevant sequence context extends beyond nearby symbols. [BigBird, §2–3 and §5](https://arxiv.org/html/2007.14062v2)

**Investigation: draw a route, then test it.** Edit legal edges and choose a source and destination. Observe whether the source can affect the destination after one, two or three layers. The graph and the matrix highlight the same path. A separate value view lets you remove a key and inspect the re-normalized output. Start the practice state with fresh positions and values; the worked 12-position example remains an ungraded walkthrough.

### Content-based selection: find candidates without comparing every full pair

*Deeper family comparison; continue to §3 on a first pass.*

A fixed window cannot know that an old variable definition is relevant to today's query. Content routing first obtains a manageable candidate set, then applies attention within it.

**Reformer** uses locality-sensitive hashing. Its actual hash chooses the largest component of concatenated positive and negative random projections, $h(x)=\arg\max[xR;-xR]$. Similar directions tend to share a bucket; this is not the same hash as independently taking every projection's sign. Shared query/key representations, normalized keys, sorting by bucket, bounded chunks and multiple hash rounds make candidate comparisons manageable. Causal masks use original positions after sorting. Repeated candidates across rounds must be handled without accidentally multiplying their contribution. A bounded chunk can miss members of a large bucket; an uncapped all-pairs bucket instead risks quadratic work. [Reformer, §2](https://arxiv.org/html/2001.04451v2)

**Routing Transformer** replaces fixed random partitions with online clustering of query/key representations. It uses normalized representations and balanced candidate budgets. With $c$ clusters, assignment costs roughly $Lcd$; balanced within-cluster comparisons cost roughly $L^2d/c$. Balancing those terms suggests $c$ of order $\sqrt L$, hence order $L^{3/2}d$, not automatically linear. Original position masks still matter. [Routing Transformer, §4.1](https://aclanthology.org/2021.tacl-1.4.pdf)

These are approximate search mechanisms: a relevant key can be missed. Their cost includes sorting, assignment, selection and gathers. Computing a full dense score matrix and then zeroing small probabilities produces sparse *weights*, but it has already paid for the dense score computation.

## 3. Linear attention: change the question the memory can answer

Sparse attention retains individual records but skips some reads. A feature-kernel method can include every earlier record by first combining them into a small state.

A **feature map** transforms a vector into another vector, $\phi:\mathbb R^{d_k}\to\mathbb R^m$. A **kernel** here is a similarity of the form

\[
\kappa(q,k)=\phi(q)^T\phi(k).
\]

For a normalized weighted average, we choose features giving nonnegative similarities and require a positive denominator. Instead of softmax, define

\[
o_t=\frac{\sum_{j\le t}\phi(q_t)^T\phi(k_j)v_j}
{\sum_{j\le t}\phi(q_t)^T\phi(k_j)}.
\]

Distribute the multiplication inside the sum:

\[
S_t=\sum_{j\le t}\phi(k_j)v_j^T\in\mathbb R^{m\times d_v},\qquad
z_t=\sum_{j\le t}\phi(k_j)\in\mathbb R^m,
\]

\[
o_t^T=\frac{\phi(q_t)^TS_t}{\phi(q_t)^Tz_t}.
\]

The same result is computed by a different order of operations. A new record adds an **outer product**: each key-feature component scales the entire value vector, creating one row contribution to $S$. The query reads a weighted combination of those rows. The normalizer $z$ keeps the matching amount of key-feature evidence, so output magnitude does not simply grow with the number of records. This causal recurrence is a central construction in [Transformers are RNNs, §3](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf).

### Work one memory update by hand

Use two key features and scalar values:

| Position | Key features | Value | Contribution to $S$ | Contribution to $z$ |
| --- | --- | --- | --- | --- |
| 0 | $[1,0]$ | $2$ | $[2,0]^T$ | $[1,0]^T$ |
| 1 | $[0,1]$ | $-1$ | $[0,-1]^T$ | $[0,1]^T$ |
| 2 | $[1,1]$ | $3$ | $[3,3]^T$ | $[1,1]^T$ |

After all three records, $S=[5,2]^T$ and $z=[2,2]^T$. A query with features $[2,1]$ produces numerator $12$, denominator $6$, and output $2$.

Check it by explicitly comparing all keys: their similarities are $[2,1,3]$, giving weights $[1/3,1/6,1/2]$. The weighted values again sum to $2$. The memory route has not approximated this feature-kernel operator.

**Visual: an outer product being written.** Key-feature bars label the rows of a matrix; value components label its columns. Each arriving record paints its numerical contribution. The query then traces a read across those rows, alongside the separate denominator. Negative values use a diverging scale; positive weights do not imply positive stored values.

**Investigation: edit a memory, not a text explanation.** Change a key-feature vector, a value vector or the query; inspect which cells and output coordinates change. Compare the recurrence with explicit normalized pair weights. Remove the oldest record by subtracting its saved outer product and key features, then verify the remaining-prefix result. A constant-value control makes every legal normalized output identical even when the query changes.

### What “linear” does and does not mean

For fixed $m,d_k,d_v$, each write and read costs order $md_v$, plus the feature-map cost. Across $L$ positions, the core work is order $Lmd_v$. Streaming state per head contains $m(d_v+1)$ scalars. The state size is independent of how many records have arrived, but not independent of the feature width or value width.

The output is generally **nonlinear in the input** because feature maps, normalization and the surrounding network are nonlinear. “Linear attention” refers to sequence-length scaling or the recurrent algebra, not a linear predictive model.

A frequently used map is applied componentwise:

\[
\phi(x)=\operatorname{ELU}(x)+1=
\begin{cases}x+1&x\ge0\\e^x&x<0.\end{cases}
\]

It defines its own similarity. It does not approximate $e^{q^Tk}$ merely because it is positive. In this lesson's model, the map receives $q/d_k^{1/4}$ and $k/d_k^{1/4}$; that scale is part of the declared model, not an identity turning ELU+1 into softmax.

Strictly positive finite features give a positive denominator for a nonempty prefix in exact arithmetic. Merely nonnegative features can give zero overlap. Floating-point exponentials can also underflow. An implementation must detect or handle the problem; adding a denominator floor changes the mathematical operator where the floor is active. Our measured example stays far above its floor.

### A compressed state can forget distinctions

Let every key have one feature equal to 1. Histories with values $[1,3]$ and $[2,2]$ both create $S=4,z=2$. Every positive query returns their mean, $2$. No read of this state can answer “what was the first value?” differently for those histories.

This is a concrete collision, not a claim that all useful information must be lost. More expressive key features can separate other histories. Positional features and gates can make writes depend on order. But a fixed-size state should be evaluated on the retrieval distinctions the task actually needs. A low mean prediction error on a smooth signal does not prove the ability to retrieve an arbitrary earlier identifier.

For a pure additive state, permuting the **already formed key-feature/value pairs** leaves the final sum unchanged. That does not mean a whole causal network ignores order: prefix outputs differ, and adding position to inputs changes the pairs themselves. This distinction connects the algebra to the earlier positional-encoding lesson.

## 4. Approximate softmax with random features

*Read the opening distinction on a first pass. The derivation and sampling details are a deeper branch.*

We can choose a different kernel deliberately, or approximate the existing exponential kernel. These are different experiments.

Set $x=q/d_k^{1/4}$, $y=k/d_k^{1/4}$, so $x^Ty=q^Tk/\sqrt{d_k}$. Draw a fixed random vector $\omega\sim\mathcal N(0,I)$, and define

\[
f_\omega(x)=\exp(\omega^Tx-\|x\|^2/2).
\]

The Gaussian identity $\mathbb E[e^{\omega^Ta}]=e^{\|a\|^2/2}$ gives

\[
\mathbb E[f_\omega(x)f_\omega(y)]
=e^{-\|x\|^2/2-\|y\|^2/2}e^{\|x+y\|^2/2}
=e^{x^Ty}.
\]

With $m$ sampled vectors, place $f_{\omega_r}(x)/\sqrt m$ in feature coordinate $r$. The feature dot product estimates the unnormalized exponential similarity. It can then use the same $S,z$ recurrence. Positive random features and orthogonal constructions are central to [Performer and FAVOR+](https://arxiv.org/html/2009.14794v4).

The result is exact for the **sampled feature operator**, approximate for the original softmax operator. Keep the projections fixed during a sequence. A state constructed under one set of random features cannot be read as though it had been constructed under another.

### Unbiased similarities do not give an unbiased normalized output

The expectation of a ratio is not generally the ratio of expectations. For a small counterexample, suppose two estimated similarities are equally likely to be $(1,1)$ or $(3,1)$. Their means are $(2,1)$. The expected normalized first weight is

\[
\tfrac12(1/2+3/4)=5/8,
\]

while normalizing the mean similarities gives $2/3$. An unbiased kernel estimator therefore does not automatically make a finite-feature attention row or output unbiased.

For the independent Gaussian construction, a single pair has variance

\[
\operatorname{Var}(\widehat\kappa_m(x,y))
=\frac{e^{2x^Ty}}{m}\left(e^{\|x+y\|^2}-1\right).
\]

You can derive this by applying the same Gaussian identity to the square of $f_\omega(x)f_\omega(y)$, then subtracting the squared mean. It explains both the $1/m$ variance reduction and sensitivity to vector norms. It is a variance of the unnormalized pair estimator, not an identical formula for task loss or the normalized ratio. Increasing $m$ improves this expectation-level quantity; one nested random draw can still become less accurate.

### Orthogonal directions still need the right radii

Using mutually orthogonal directions within a block can reduce redundant sampling. To preserve a standard Gaussian marginal for each row, combine a uniformly random orthogonal direction with an independent radius distributed as the length of a $d_k$-dimensional standard Gaussian vector. The supplied program obtains directions using a sign-corrected QR decomposition, then samples those independent radii. Independent blocks allow $m>d_k$.

Multiplying every unit direction by the fixed number $\sqrt{d_k}$ is a different distribution. In one dimension, take $x=y=1$. A fixed-radius vector is $+1$ or $-1$, so the expected feature product is $e^{-1}\cosh2\approx1.384$; the Gaussian construction gives $e\approx2.718$. Both are valid things to compute, but only the latter follows the Gaussian identity above. The Performer paper also studies a fixed-radius regularized kernel; it should be named as such, not described as an unchanged Gaussian estimator.

### Keep the numerical stabilization consistent

Exponentials may overflow. Multiplying all feature coordinates for one query by a common scalar cancels between that query's numerator and denominator. Multiplying every key-feature vector in the whole memory by the **same** scalar also cancels. Multiplying each key by its own unrelated scalar usually changes their relative importance.

For streaming exponential features, if a newly arrived key requires changing the common key scale, rescale the existing $S$ and $z$ by the same factor before adding the new write. Otherwise old and new records use different units. Padding, state reset, query/key feature scaling, and causal prefix boundaries must agree between training and inference.

**Investigation: an approximation has a distribution.** Begin with a fresh editable four-record Q/K/V problem. Inspect the effect of adding features or changing one key, and show immediately the sampled approximation beside exact softmax. A second view shows all saved random-feature trials on an actual trained head. The learner can see seed-specific reversals, rather than a fabricated line that decreases every time.

## 5. Compress the sequence instead of its feature sums

Another strategy replaces many keys and values with fewer summary slots before attention. It deserves its own picture because it is not the same operation as skipping keys or accumulating $\phi(k)v^T$.

### Linformer: learned combinations along the length axis

Let $K\in\mathbb R^{L\times d_k}$, $V\in\mathbb R^{L\times d_v}$. Define learned sequence projections $E,F\in\mathbb R^{r\times L}$:

\[
\bar K=EK,\quad\bar V=FV,\quad
O=\operatorname{softmax}(Q\bar K^T/\sqrt{d_k})\bar V.
\]

Each row of $E$ combines positions into a key summary; each row of $F$ combines positions into a value summary. Learned coefficients need not be nonnegative or sum to one. The model attends over $r$ summaries. With fixed $r\ll L$, projection and attention cost order $Lr(d_k+d_v)$. The original work motivates this using empirical low-rank behavior of attention maps and approximation arguments; it does not establish that every query/key matrix or task has a universally small useful rank. [Linformer, §3–4](https://arxiv.org/html/2006.04768v3)

Now inspect causality. With one value-summary row $F=[0.5,0,0.5]$ and values $[1,2,9]$, the summary is $5$. There is only one summary slot, so its softmax weight is one. A query at position 0 would receive $5$, containing future value 9. Changing that future value to 1 changes the supposedly earlier output to 1. Applying a triangular mask to the single summary slot cannot remove the particular future contribution already mixed into it.

**Visual: follow the future contribution.** Draw three original value rows into one summary, preserving coefficient labels. Highlight the path from the last value through the summary to the first output. The error is visible before any probability calculation.

A causal variant can be constructed, but must define a different prefix-dependent operator. For fixed projection columns $e_j,f_j\in\mathbb R^r$, maintain

\[
\bar K_t=\bar K_{t-1}+e_tk_t^T,\qquad
\bar V_t=\bar V_{t-1}+f_tv_t^T.
\]

The current query attends only to these summaries of positions through $t$. In the one-slot example, position 0 sees $0.5$, not the full-sequence 5. The coefficient has not automatically become a prefix-normalized average. Define how unused summary rows participate, how columns are generated beyond the trained length, and whether coefficients depend on future inputs. This construction shows why “the usual full-sequence projection leaks” is precise, while “sequence projection can never be causal” is too strong.

**Investigation: repair the leak at its source.** Use a fresh four-value example and edit one future value. Compare a full-sequence summary with a prefix-only summary at a selected earlier query. Predict both outputs. Moving the same edit into the legal prefix should remove the earlier invariance; making its projection coefficient zero supplies a different null.

### Nyströmformer: use landmark queries and keys

Nyströmformer selects a smaller set of representative query/key vectors, called **landmarks**. One option takes means of contiguous segments. Here $d=d_k$ is query/key width, and the symbols $F,A,B$ name new factors local to this construction. Define row-softmax matrices

\[
F=\operatorname{softmax}(Q\tilde K^T/\sqrt d),\quad
A=\operatorname{softmax}(\tilde Q\tilde K^T/\sqrt d),\quad
B=\operatorname{softmax}(\tilde QK^T/\sqrt d).
\]

Approximate the attention output by $FA^+BV$, where $A^+$ is the Moore–Penrose pseudoinverse. Compute from the right to avoid forming an $L\times L$ matrix. The paper uses an iterative inverse approximation; the teaching fixture uses a direct numerical pseudoinverse. Pseudoinverse coefficients can be negative, so the approximate full matrix is not automatically a nonnegative probability matrix. Full-sequence landmarks also need a separate causal design before use in autoregressive prediction. [Nyströmformer, §3](https://arxiv.org/html/2102.03902v3)

The mechanism diagram has three small comparisons: query-to-landmark, landmark-to-landmark correction, and landmark-to-original-key. This differs from Linformer's learned length projection, even though both introduce a small intermediate dimension.

For $r$ landmarks, a straightforward implementation pays order $Lrd_k+Lrd_v+r^2d_v+r^3$, including an SVD-based pseudoinverse. “Linear in $L$” assumes $r$ is held fixed; increasing landmarks to preserve quality changes that tradeoff. Singular values near zero make the pseudoinverse sensitive, so approximation and numerical error must be examined together.

## 6. Make the graph efficient on the actual machine

*Deeper practical branch; the real-data core continues in §7.*

### Eight edges can occupy very different amounts of work

A hardware kernel often processes tiles instead of individual matrix cells. In an $8\times8$ mask with $2\times2$ tiles, eight diagonal edges occupy four tiles: 16 candidate cell positions inside those tiles. Place one edge per row at columns $2i\bmod8$, and the same eight edges occupy eight tiles: 32 candidate positions. Both masks have 12.5% token-edge density. They have different tile occupancy and memory access patterns. This example is a bidirectional layout exercise, not a causal mask.

An occupied tile may still contain masked cells; a fully empty tile can be skipped. Neither count alone predicts seconds. Gather overhead, head dimensions, precision, hardware, compilation and other layers matter. A dense implementation of a sparse mask still computes the dense scores if it forms `Q @ K.T` first.

**Investigation: pack the same edges into different tiles.** Toggle mask cells and watch occupied-tile count change. Move edges while preserving their number and compare true edges with candidate cells. The fresh clustered pattern exposes why equal token sparsity need not imply equal tile work.

PyTorch's FlexAttention provides a way to express custom score modifications and block masks and compile suitable attention kernels. Its `mask_mod` receives batch, head, query index and key index and returns whether that pair is allowed. A block mask can skip fully masked blocks. This does not mean every arbitrary mask is equally fast, or that an unsupported device will execute the same compiled path. [FlexAttention introduction and examples](https://pytorch.org/blog/flexattention/), [current API](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html)

### Modern sparse systems also pay to choose the reads

The field has continued beyond the early fixed patterns. These examples are architecture snapshots checked on 13 September 2026, not a leaderboard.

**Native Sparse Attention (NSA)** combines three separately normalized branches: compressed blocks, selected fine-grained blocks, and a local window. Compressed-attention scores help select important blocks, with selection shared across grouped heads. Learned sigmoid gates combine branch outputs; their sum is not required to be one. The three-branch design preserves local access while learning coarser and selected long-range reads. With a fixed compression stride, the compressed branch still grows with the number of compressed positions, so a fixed selection budget alone does not establish linear total work. [NSA, §3](https://arxiv.org/html/2502.11089v1)

**DeepSeek-V3.2-Exp's DSA** uses a small lightning indexer to select past positions for its main MLA read. The indexer aggregates query-head scores of the form $w_{tj}\operatorname{ReLU}((q^I_{tj})^Tk^I_s)$. It is first trained against a dense attention-derived target, then the main sparse model is adapted. Top-$k$ indices are discrete; ordinary backpropagation through their integer selection is not the indexer's training method. The report explicitly distinguishes order $Lk$ main attention from the indexer's still-quadratic sequence comparison, albeit with a smaller cost. [V3.2-Exp technical report, §1–3](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/DeepSeek_V3_2.pdf)

**DeepSeek-V4.1's CSA2** distinguishes layers that build and index new compressed memory, layers that reuse memory but issue fresh index queries, and layers that reuse both memory and selected indices. Its hierarchical decoder indexer obtains a candidate pool from an initial full scan; later reindexing searches that pool. Local sliding-window memory remains a separate source. The architecture's causal encoder–decoder division changes which layer supplies the global memory, so memory reuse and selection reuse must not be conflated. These are separate mechanisms from NSA's three gated outputs. [V4.1 report, §2.2–2.3](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)

The accompanying visual marks **candidate discovery**, **main selected reads**, **local reads**, and **cross-layer reuse** as separate arrows with separate costs. It does not invent a speedup from an edge count. Current [FlashMLA source documentation](https://github.com/deepseek-ai/FlashMLA) provides concrete examples of dense and sparse kernels with architecture-specific formats; a cache format or benchmark cannot be transplanted unchanged between model versions.

## 7. Compare three operators on observed hand trajectories

An abstract graph tells us whether an effect is possible. A trained example tells us what a particular learned system actually does. We will forecast the next two-dimensional point of an observed hand movement using the openly licensed [UCI Libras Movement dataset](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), credited to Dias, Peres and Bíscaro, under CC BY 4.0.

The source has 360 trajectories of 45 points and 15 movement labels. Exact coordinate duplicates reduce it to 330 unique trajectories; duplicate labels agree. We retain the first row in each exact-duplicate group **before splitting**, so identical trajectories do not appear on both sides of the evaluation. A fixed classwise seed-73 split gives 220 training, 50 validation and 60 test trajectories. Movement labels are used only for stratification, not as model inputs. Positions 0–43 predict 1–44, using fixed input scaling $2x-1$.

The source describes four performers and two recording sessions but does not supply row-level performer/session identifiers. This split tests held-out trajectories, not new-person or new-session generalization. The processed coordinate records also do not validate a live image-to-coordinate pipeline. This familiar dataset is intentionally reused from earlier lessons; the new test comparison is not an independent new benchmark across those lessons.

### Hold the experiment fixed

Each model has 4,946 learned parameters: a 2→24 input map, fixed sinusoidal position addition, one pre-norm block with three 8-wide heads, a 24→48→24 GELU FFN, final normalization and a two-coordinate output. There is no dropout or weight decay. The only operator change is dense causal softmax, a five-key causal softmax window, or normalized ELU+1 feature attention. Identical seed-137 initialization and 160 full-batch Adam updates at learning rate 0.003 give an equal-budget local comparison.

Each model selects its lowest validation MSE, taking the earliest exact tie. All selected update 160, the budget endpoint. We did not extend the run to make a preferred outcome emerge, and this result does not establish convergence. A persistence baseline predicts the current point as the next; a six-parameter affine baseline is fitted using training pairs only. RMSE pools both coordinates and all next-point errors in each split, in the source's original coordinate units.

| Predictor | Training RMSE | Validation RMSE | Test RMSE |
| --- | ---: | ---: | ---: |
| Persistence | 0.027529 | 0.026304 | 0.026458 |
| Training-fitted affine | 0.027334 | 0.026154 | 0.026222 |
| Dense causal softmax | 0.024152 | 0.023358 | 0.023219 |
| Five-key causal window | 0.025147 | 0.023907 | 0.024684 |
| ELU+1 feature kernel | 0.026106 | 0.025327 | 0.024680 |

All three learned operators improve on these simple baselines in this run. Dense attention has the lowest test RMSE here. The window and kernel test results differ by only about $0.0000041$, far too little to promote into an architectural verdict from a single split and seed. There is no measured GPU timing comparison in this table.

### Look at one actual prefix and a controlled edit

The worked example uses source row 77, first 32 observed points. The true next point is approximately $(0.593810,0.250000)$. Keep the weights fixed and reflect the x coordinate at zero-based frame 23 around 0.5: $x\mapsto1-x$. This is an artificial sensitivity intervention on a real recorded trajectory, not another observed movement.

| Operator | Original next-point forecast | Forecast after editing frame 23 |
| --- | --- | --- |
| Dense | $(0.598713,0.250658)$ | $(0.601801,0.252266)$ |
| Window | $(0.609244,0.247768)$ | $(0.609244,0.247768)$ |
| Feature kernel | $(0.601817,0.248507)$ | $(0.603256,0.248870)$ |

The window result is an exact null in this model: at final input position 31, its single attention layer reads positions 27–31, and all other operations are tokenwise. Frame 23 has no route to that output. The other two models change modestly; a global path does not require a dramatic response. In all three models, outputs before the edited position remain exactly unchanged, as causality requires.

**Visual: the same trace through three memories.** A coordinate plot preserves the observed trajectory and marks the actual next point separately from each forecast. Under it, the dense model has 32 K/V rows, the window has five, and the feature model has an $8\times8$ matrix plus an 8-vector per head. Selecting a point reveals whether it is individually retained, evicted from the local window, or has contributed to the aggregate state.

At this 32-point prefix, float32 numeric attention payloads are 6,144 bytes for dense K/V, 960 for window K/V, and 864 for kernel $S,z$. These exclude weights, outputs, position counters, index metadata and allocator overhead. They are not peak memory measurements. Dense and window incremental forecasts agree with their full-prefix computation to below $2.4\times10^{-7}$ maximum absolute error in transformed output coordinates. The kernel's recurrent and explicit pairwise implementations agree below $2.7\times10^{-7}$. Separate float64 checks give whole-network gradient agreement below $3.6\times10^{-15}$ for this small checked input.

**Investigation: predict what an old edit can change.** The fresh gated case uses 27 points, changes the y coordinate at frame 19, and hides forecasts until you Show the current computed result and its contributing terms immediately. Drag or numerically edit a point, inspect which models can respond, then inspect actual recomputed outputs. Move the edit inside the local window to test the boundary. Reset returns the genuinely visible fresh problem, not the solved 32-point walkthrough.

### Approximate one trained dense head without retraining it

Take the selected dense model's actual head-0 queries, keys and values from the worked prefix. Compare exact softmax output against positive independent Gaussian features, using seeds 0–7 and nested feature counts 16, 64 and 256. For each run, measure

\[
\text{relative output error}=\frac{\|\widehat O-O\|_F}{\|O\|_F}.
\]

The denominator is nonzero for this saved head. The metric compares the whole prefix's head outputs, not model RMSE or probabilities of a downstream label.

| Features | Mean error over eight seeds | Smallest–largest observed error |
| --- | ---: | ---: |
| 16 | 0.043293 | 0.021708–0.077060 |
| 64 | 0.031098 | 0.016960–0.042411 |
| 256 | 0.017026 | 0.011095–0.022679 |

The average falls, but individual nested draws do not all improve at every step. Seed 5 worsens from about 0.02171 to 0.02813 when going from 16 to 64 features. Seed 6 worsens slightly from 64 to 256. Keep those outcomes in the plot. These are operator approximations of a fixed dense head, not results from training a Performer, and not a worst-case guarantee.

### Run the complete teaching programs

The packet includes the original small data files, [data attribution and provenance](provenance.md), [the complete CPU training and evaluation program](author-calculations.py), [mechanism calculations](mechanism-calculations.py), and the actual saved models/results. Use Python with NumPy and PyTorch in your own environment; the recorded execution used Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu, one CPU thread.

```bash
python -m venv .venv
# Activate .venv using your shell's normal activation command.
python -m pip install numpy==2.3.5 torch==2.14.0
python mechanism-calculations.py
python author-calculations.py
```

Run from the downloaded packet directory. The programs use local data and perform no network access. The mechanism program checks sparse gathered/masked equality, graph reach, feature-state reads and the fresh numerical cases. The training program supplies data loading, duplicate handling, split construction, all model layers, training, validation selection, baselines, incremental evaluation, gradient checks, saved forecasts and the 24 random-feature trials. Every named helper is included in the downloadable source.

Expected recorded model lines are dense/window/kernel selecting update 160 with the RMSE values above; each program ends with a named PASS line. Different supported software or hardware can change last digits. If you alter the data, seed or protocol, label the new result as a new experiment. Training here computes small dense reference masks and, for kernel attention, stores prefix states for automatic differentiation. The code teaches the operator and verifies its recurrence; it is not a high-performance sparse GPU kernel or a constant-memory training implementation.

For a minimal streaming implementation you can inspect the following complete NumPy example. Inputs are already feature vectors, so it cleanly separates the recurrence from the choice of feature map.

```python
import numpy as np

def causal_feature_attention(query_features, key_features, values):
    q = np.asarray(query_features, dtype=float)
    k = np.asarray(key_features, dtype=float)
    v = np.asarray(values, dtype=float)
    if q.ndim != 2 or k.shape != q.shape or v.ndim != 2 or len(v) != len(q):
        raise ValueError("Expected matching (length, features) Q/K and (length, values) V.")
    if not all(np.isfinite(x).all() for x in (q, k, v)) or (q < 0).any() or (k < 0).any():
        raise ValueError("Use finite nonnegative features and finite values.")
    memory = np.zeros((k.shape[1], v.shape[1]))
    normalizer = np.zeros(k.shape[1])
    outputs = []
    for query, key, value in zip(q, k, v):
        memory += np.outer(key, value)
        normalizer += key
        denominator = query @ normalizer
        if denominator <= 0:
            raise ValueError("No positive overlap with this prefix.")
        outputs.append(query @ memory / denominator)
    return np.asarray(outputs)

q = [[2, 1], [2, 1], [2, 1]]
k = [[1, 0], [0, 1], [1, 1]]
v = [[2], [-1], [3]]
print(causal_feature_attention(q, k, v).ravel())
# [2. 1. 2.]
```

### Implement the other compression choices, not just name them

The earlier program owns the trained causal dense/window/kernel comparison. The additional [sequence-compression program](attention_compression_bridges.py) opens the bidirectional Linformer and Nyström operations from §5 and supplies an actual gathered-window route. It requires only PyTorch; run `python attention_compression_bridges.py`.

`linformer` owns two learned length-axis matrices E and F. It computes EK and FV first, then runs attention over those compressed slots. Its normal tool route calls SDPA on the same compressed arrays. E/F receive gradients alongside Q/K/V; a learned summary is not a fixed downsampling label. The program compares values and all five gradients under identical initial arrays. It intentionally has **no causal mask**: making a full-sequence summary and applying a later triangle cannot remove future information already mixed into the summary.

`nystrom` forms segment-mean query/key landmarks. Seven positions split into three segments retain the trailing positions. It constructs the three softmax factors, uses a tolerance-controlled pseudoinverse for the small middle matrix, and multiplies from the value side: `front @ (pinv(middle) @ (back @ V))`. It never materializes the L×L approximate weight matrix. The cost includes O(L r d) pair/factor work and O(r³) pseudoinversion, with O(Lr+r²) factor storage, in addition to the feature/value widths. A pseudoinverse is a well-defined tool here; implementing SVD again would repeat [Matrix Decompositions](/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu?module=math-foundations). Near a rank threshold, derivatives can be sensitive: changing `rtol` changes which directions are retained and must be treated as a model/numerical decision.

`gathered_window` only scores the keys actually in a causal window: O(L W (d_k+d_v)) arithmetic and O(W) temporary scores per query, beyond inputs and outputs. Its SDPA comparison deliberately uses a dense mask as an independent semantic reference, **not** as evidence of sparse execution. The maintained tool takes responsibility for backend selection; a genuinely sparse accelerator path needs a kernel supporting the chosen block pattern.

The author ran these small CPU float64 probes: Linformer and gathered-window maximum API differences were each 1.11e-16; using every position as a Nyström landmark reproduced the dense result to 1.45e-15. Three landmarks gave maximum output error 0.25091 for this declared random fixture. That last number is a single approximation example, not a general error guarantee or a trained accuracy result. The random-feature Gaussian-marginal sampling and stabilizations remain owned by `mechanism-calculations.py`; they are different approximations from these learned/landmark summaries.

**Take control.** Change sequence length to 11, use four landmarks and a window of width 1. Inspect both output and gradient checks. Then reduce `rtol` for nearly duplicate landmarks.

<details><summary>Hint</summary>Width one must return each position's own V. Unequal segment lengths are allowed; the inverse problem remains small.</details>
<details><summary>Solution and success criteria</summary>The window output equals V and has zero Q/K derivative. Linformer's manual/API equality should remain, while approximation quality is a separate measured quantity. Nyström with fewer landmarks need not improve monotonically for each input as count increases. Near duplicate landmarks, record singular values and chosen tolerance before interpreting a large gradient; smaller tolerance is not automatically a better model.</details>

## 8. Deeper connections and practical judgment

### Backpropagation through a recurrent summary

Forward equivalence is not enough if the two training paths differentiate different calculations. For a local read, write $a=\phi(q)$, numerator $n=S^Ta$, denominator $b=a^Tz>0$, and $o=n/b$. If the arriving output gradient is $g=\partial\mathcal L/\partial o$, ordinary quotient differentiation gives

\[
\frac{\partial\mathcal L}{\partial S}=\frac{ag^T}{b},\qquad
\frac{\partial\mathcal L}{\partial z}=-\frac{a(g^To)}{b},\qquad
\frac{\partial\mathcal L}{\partial a}=\frac{Sg-z(g^To)}{b}.
\]

In a causal sequence, a write at position $j$ influences every later state. Its gradient therefore collects contributions from reads $t\ge j$, which can be accumulated by a reverse scan. For its direct outer-product contribution, if the accumulated matrix gradient is $G_j$, then the value gradient includes $G_j^T\phi(k_j)$; the key-feature gradient includes $G_jv_j$ plus the normalizer contribution. The chain rule then differentiates the feature map and input projections.

You can parallelize prefix operations or process chunks, but the memory layout and backward strategy still matter. A naive `cumsum` over all outer products stores an $L\times m\times d_v$ tensor. A custom scan/recomputation method can trade storage for work. Our saved all-parameter gradient comparison checks the defined small model, including surrounding normalization and FFN, rather than checking only one isolated final sum.

### Gates and delta updates change the memory, not just its speed

The additive update $S_t=S_{t-1}+k_tv_t^T$ retains every write with equal temporal persistence. A decay changes it to $S_t=\gamma_tS_{t-1}+k_tv_t^T$, so earlier contributions are multiplied by later decay factors. Featurewise gates allow different parts of memory to forget differently.

A delta-style write instead uses the current prediction error for the key, for example

\[
S_t=S_{t-1}+\beta_t k_t(v_t-S_{t-1}^Tk_t)^T.
\]

For a unit key and $\beta_t=1$, the new read at that key is exactly $v_t$: multiplying by $k_t^T$ cancels the old prediction error. It behaves like correcting a stored association, not merely adding another copy. With a nonunit key, the same conclusion does not follow without adjusting the update. This explains why a family can have linear sequence scaling but different overwrite, normalization and retrieval behavior.

The earlier [RWKV & Linear Attention Models](/learn/path/full-curriculum/rwkv-linear-attention-models?module=deep-learning-fundamentals) and [State Space Models](/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals) own the versioned recurrent mechanisms and selective dynamics. A signed recurrence or an mLSTM normalization is not automatically a positive normalized feature kernel, and none becomes exact row-softmax merely because matrix products can be reassociated. Use the update, read and normalization equations to classify a new model.

### Position and masking are algebraic constraints

A feature recurrence works when each write can be computed from the current/past information and each read uses the appropriate state. An arbitrary pairwise relative-position bias need not factor into a fixed-size query/key state. Some distance factors do: a scalar exponential decay corresponds to the recurrent weighting above. Other position schemes require extra features or a changed kernel. Rotating vectors before a nonlinear feature map does not establish the same relative-position identity as rotating the dot-product vectors; check the resulting similarity explicitly.

Packed documents need state resets or segmented scans at document boundaries. A single global sum over an entire batch of packed text leaks information across samples. During decoding, state updates must match whether the current token is included, the prompt prefix already processed, and the attention layer's positional convention. These are part of the operator, not cosmetic bookkeeping.

### Choose an experiment that can disprove your idea

| Task need | Candidate to investigate | A revealing failure test |
| --- | --- | --- |
| Predict a locally smooth signal | Window or recurrent summary, alongside simple baselines | Insert a relevant remote change; test new entities, not duplicate rows |
| Retrieve an earlier exact identifier | Individual-key access, possibly selected or hybrid | Move the target, add distractors, vary delay and compare missed candidates |
| Classify a complete structured document | Local/global graph or summaries | Remove structure labels; vary document length and cross-section dependencies |
| Compress an existing softmax model | Approximation plus adaptation, with exact reference | Compare operator error, final task loss and required retraining separately |
| Serve a long causal prompt | Measure prefill, decode, cache and selection costs | Include routing/indexing and batch/concurrency effects, not only kernel arithmetic |

An interesting scientific use follows from the graph picture. In a DNA sequence, a local motif and a distant regulatory context may both matter. A sparse graph can preserve cheap local comparisons while adding routes between distant regions. But graph connectivity alone does not establish biological relevance; evaluation must preserve meaningful held-out sequence/entity boundaries. Similarly, a protein's amino-acid sequence is one-dimensional while its interactions can be distant along that sequence. Efficient attention offers a way to examine longer contexts; a token-level prediction score is not itself a validated three-dimensional structure or function prediction. The research examples in the references are motivations for careful task design, not permission to infer such downstream capabilities from our hand-motion experiment.

Retrieval before the model can reduce how much context enters it; efficient attention changes how the supplied context is processed. Neither universally replaces the other. A useful system may use both, and its evaluation should include evidence that retrieval did not discard the needed information.

## 9. Practice and transfer

Work these problems before opening the optional hints or solutions. They use different values and positions from both the worked explanations and the initial investigations.

### 1. Count the real connections

A causal sequence has 20 positions and a window of four total keys including self. How many legal query/key pairs exist in one head? What is the farthest possible input distance after three such layers, assuming all other operations are tokenwise? Does reaching that distance guarantee a useful learned dependence?

<details><summary>Hint</summary>

Count the growing first rows separately. A single layer moves information at most three positions.

</details>
<details><summary>Solution</summary>

The first four rows contribute $1+2+3+4=10$; the next 16 contribute 64, totaling **74**. Three layers can span at most **9** positions. This is possible information flow, not a guarantee that the learned weights preserve or use the information.

</details>

### 2. Compute a summary read

Two key-feature vectors are $[1,2]$ and $[3,1]$, with scalar values $-2$ and $4$. The query features are $[2,1]$. Compute $S,z$, the two normalized weights and the output. Then change the second value to $-2$ without changing any key or query.

<details><summary>Hint</summary>

The unnormalized key similarities are 4 and 7. A value edit changes the numerator, not the denominator.

</details>
<details><summary>Solution</summary>

$S=[10,0]^T,z=[4,3]^T$. The denominator is 11, weights are $4/11,7/11$, and output is **$20/11\approx1.81818$**. With both values $-2$, every normalized weighted average equals **$-2$**. This is a useful null even if you later change the query.

</details>

### 3. Diagnose a misleading speed claim

An implementation forms a $4096\times4096$ score matrix, then masks all but 64 keys per row and calls softmax. Its author says “only 64 keys are visible, so the matrix multiplication is linear in sequence length.” Identify the error and propose a correctness check for a genuinely gathered implementation.

<details><summary>Solution</summary>

The full matrix multiplication has already computed all pairs; a later mask does not undo that work. Gather the legal key/value rows before comparison or use a kernel that skips masked blocks. For small fixed Q/K/V and a nonempty mask per row, compare the gathered outputs to a dense reference with illegal scores set to negative infinity. Test causal boundaries and a mask with uneven row lengths. Matching output establishes the specified sparse operator, not a speedup; timing requires the actual target implementation.

</details>

### 4. A future value inside a summary

One sequence-summary row has coefficients $[0.2,0.3,0.5]$, values $[5,0,6]$, and only one attention slot. At query position 1, compare the full-sequence summary with the prefix-only summary. Change the last value to $-2$. Which earlier output should remain invariant in a causal implementation?

<details><summary>Solution</summary>

The full summary is **4**, then becomes **0** after the future edit. The prefix-only summary is **1** before and after. These coefficients are not renormalized over the prefix. The full summary leaks; masking its sole slot cannot selectively remove the future component.

</details>

### 5. Compare the right memory quantities

For batch one, four heads, $d_k=d_v=16$, $L=2048$ and float32 state, calculate dense K/V payload, a 32-key window payload, and feature state with $m=24$. Exclude metadata and weights. Which calculation tells you peak training memory?

<details><summary>Solution</summary>

Dense: $4\times2048\times(16+16)\times4=\mathbf{1,048,576}$ bytes. Window: $4\times32\times32\times4=\mathbf{16,384}$ bytes. Kernel state: $4\times24\times(16+1)\times4=\mathbf{6,528}$ bytes. **None** gives peak training memory; gradients, saved activations, optimizer state and temporary computations are additional quantities.

</details>

### 6. Test a random-feature claim

Someone reports that their kernel similarities are unbiased, so “the average normalized attention output must be exactly the original output.” Construct a two-outcome counterexample different from the one in §4. What else should be recorded when showing an error-versus-feature-count plot?

<details><summary>One solution</summary>

Let similarity estimates be equally likely $(2,1)$ or $(6,1)$, with values 1 and 0. The expected output is $(2/3+6/7)/2=\mathbf{16/21}$. Normalizing mean similarities $(4,1)$ yields **$4/5$**, a different result. Record the fixed Q/K/V, scale, feature distribution, seeds, counts, whether draws are nested, normalization, error metric and all declared outcomes. Do not discard seeds that worsen as features are added.

</details>

### 7. Repair a streaming feature bug

A program computes exponential key features, subtracting each key's own largest log-feature value before writing it into memory. Queries are similarly centered per row. It claims the result is unchanged because “softmax ignores additive constants.” Which centering is safe, and what state repair is needed when the common key scale changes?

<details><summary>Solution</summary>

A scalar multiplier shared by all features of one query cancels in that query's numerator and denominator. A scalar shared by **all keys** also cancels. Different per-key scalars change relative key contributions. Maintain a common key scale and, when it changes, multiply the existing $S,z$ by the corresponding conversion factor before adding the new write. The usual row-softmax invariance does not justify unrelated rescaling of separate keys.

</details>

### 8. Explain an exact local null

A model has two causal attention layers, each with a three-key window, and tokenwise FFNs. Its last input is at position 14. Can changing raw input position 8 affect output 14? What about position 10? How would you distinguish a missing path from a learned near-zero response?

<details><summary>Solution</summary>

The maximum distance is $2(3-1)=4$. Position 8 is six positions away, so it cannot affect output 14 under these assumptions. Position 10 is reachable, so a dependence is possible but may be weak or absent for particular weights and values. Check graph reach independently of numerical sensitivity; a small observed change cannot prove a missing path. Cross-position normalization, convolution or another global operation would change the assumptions.

</details>

### 9. Design a fair comparison

You want to replace dense attention in a code-assistance model. A window model wins on next-token loss averaged over short files. Specify a test that could reveal a meaningful weakness, and separate an operator-level comparison from a trained-model comparison.

<details><summary>One solution</summary>

Create held-out files or repositories requiring earlier definitions, renamed identifiers and distractors at varied distances; prevent near-duplicate leakage. Compare exact correctness on those dependencies and behavior on ordinary code. For an operator test, hold Q/K/V and weights fixed and measure output differences caused by the replacement. For a model comparison, permit declared adaptation/training and report its budget, validation selection and untouched test results. Measure prefill, decoding, cache and total latency on the target device separately. A short-file average alone does not establish long-range retrieval ability.

</details>

### 10. Interpret a modern indexer's complexity

A sparse main attention layer reads a fixed 512 selected keys per query, but its indexer compares each query with every earlier key. Is the complete sequence computation linear in $L$? If a later layer reuses a candidate pool, does that erase the first layer's cost?

<details><summary>Solution</summary>

The main read is order $512L$ times its per-pair cost, but the all-prefix indexer contributes order $L^2$ comparisons. A smaller indexer dimension or precision can make that term practically cheaper without changing its order. Reusing a bounded candidate pool can reduce later selection work; the initial scan still belongs in the total. The selected memory representations, selected indices and fresh queries must each be accounted for.

</details>

## 10. Another way to learn, and what comes next

For a visual explanation of graph connectivity, read Google Research's [Constructing Transformers for Longer Sequences with Sparse Attention Methods](https://research.google/blog/constructing-transformers-for-longer-sequences-with-sparse-attention-methods/). Its graph, sentence/paragraph and blockification explanations are useful companions to §2 and §6. The 2021 hardware limits and broad performance wording describe that historical setting; use this lesson's explicit causal and cost distinctions when interpreting them.

For a different view of feature factorization, read the authors' [Rethinking Attention with Performers](https://research.google/blog/rethinking-attention-with-performers/). Its matrix-association and prefix-sum visuals accompany §3–4, and its protein example provides another application. Read its “unbiased attention” shorthand with the kernel-versus-normalized-ratio distinction developed here. Both articles offer a useful visual companion to the self-contained derivations here.

For implementation practice, the [FlexAttention tutorial](https://pytorch.org/blog/flexattention/) shows how score modifications and block masks connect to compiled kernels. The current [API reference](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html) is the version-sensitive companion. GPU execution and latency comparisons are separate from the CPU programs supplied here.

Primary reading, by the question it answers:

- [Sparse Transformer](https://arxiv.org/html/1904.10509v1): how alternating spatial patterns create routes through layers.
- [Longformer](https://arxiv.org/html/2004.05150v2) and [BigBird](https://arxiv.org/html/2007.14062v2): local/global graph design and the limits of expressivity claims.
- [Reformer](https://arxiv.org/html/2001.04451v2) and [Routing Transformer](https://aclanthology.org/2021.tacl-1.4.pdf): candidate discovery by hashing or clustering.
- [Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf): feature-state recurrence and causal differentiation.
- [Performer](https://arxiv.org/html/2009.14794v4): positive random features, Gaussian versus fixed-radius constructions, and approximation analysis.
- [Linformer](https://arxiv.org/html/2006.04768v3) and [Nyströmformer](https://arxiv.org/html/2102.03902v3): two distinct routes through a small intermediate dimension.
- [NSA](https://arxiv.org/html/2502.11089v1), [V3.2-Exp report](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/DeepSeek_V3_2.pdf), and [V4.1 report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf): current examples where selection, compression and reuse have separate mechanisms and costs.
- [Efficient Transformers survey](https://arxiv.org/html/2009.06732v3): a historical map of families and evaluation issues, not a current exhaustive ranking.

You are ready for the next lesson when you can identify the legal input path, explain the stored state, calculate one sparse and one feature-kernel read, and propose a control that would expose a misleading efficiency claim. You do not need to memorize every architecture's acronym.

The next topic in the module is [Vision Transformers: ViT, DeiT, Swin and DINOv2](/learn/path/full-curriculum/vision-transformers-vit-deit-swin-dinov2?module=deep-learning-fundamentals). Image patches give the sparse graph a two-dimensional geometry; shifted windows and learned image representations will make the connection concrete. Later [Mixture-of-Experts Transformers](/learn/path/full-curriculum/mixture-of-experts-transformers-moe?module=deep-learning-fundamentals) sparsify which expert computations run, a different axis from selecting attention edges.
