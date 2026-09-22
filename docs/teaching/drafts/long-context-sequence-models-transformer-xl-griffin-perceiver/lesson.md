# Long-Context Sequence Models: Transformer-XL, Griffin and Perceiver

**Explore as you read.** Edit sequence records, segment and memory lengths, recurrence/input gates, latent queries and supported trajectory coordinates. Show legal donors, cache contents, retained/injected state, latent mixtures and exact frozen-model readouts as controls change. Compare reordering whole records with changing their positions. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose a memory/compression scheme by what it preserves and loses, and separate context reach from usable information.


A note at the beginning of a document says, “The backup entrance is on the east side.” Much later, someone asks which entrance to use. A model must carry that earlier information forward or be able to consult it again. Merely accepting the whole document as input does not tell us whether it can answer.

The same problem appears outside text. A sound is a sequence of measurements; its identifying features may be spread across different moments. A long video contains many more pixels than we want to compare with every other pixel at every layer. We need a way to keep useful information available while controlling the work required to process it.

This lesson studies three answers: retain a sequence of earlier representations, update a compact recurrent state, or read a large input into a smaller set of working vectors. Transformer-XL, Griffin and Perceiver make these choices concrete. Their differences will help you ask a better engineering question than “How long is its context window?”: **What can this computation still access, through which path, and at what cost?**

**First pass:** read sections 1–6, doing the three small memory/recurrence/latent investigations where they appear. Run the two supplied programs in section 7; the first takes you through exact mechanisms, and the second trains a small classifier on real hand trajectories. Then attempt exercises 1–5 and 7 in section 9. Section 8 is a deeper route through gradients, scaling and evaluation; return to exercises 6 and 8 with it. Allow roughly 50–65 minutes for the core reading, with practice and experiments in a separate sitting.

The preceding [Bahdanau and Luong attention lesson](/learn/path/full-curriculum/attention-mechanism-bahdanau-luong?module=deep-learning-fundamentals) introduced attention as a way for a decoder to consult different encoder states. We will refresh that operation locally and then change what is available to consult. The later [self-attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals), [transformer-block](/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals) and [positional-encoding](/learn/path/full-curriculum/positional-encodings-sinusoidal-learned-rope-alibi?module=deep-learning-fundamentals) lessons develop those building blocks more fully. No knowledge of their implementations is required here.

## 1. Three kinds of memory are different objects

Think of three possible workspaces:

* A **retained record** has an individual vector for each of several earlier positions. A query can compare itself with those positions separately.
* A **recurrent state** is updated after each input. Earlier inputs influence a fixed number of state coordinates, but their original individual vectors need not remain available.
* A **latent array** contains a chosen number of working vectors. They read a larger input array, process what they have gathered and may read the input again.

“Vector” here means a list of numbers used together as one representation. Its entries are learned features, not necessarily readable facts. A retained vector is also a representation rather than a verbatim copy of a sentence; the distinction is whether earlier positions remain separately addressable.

**Inline figure: three memory workspaces.** Show the same five input positions feeding (a) five retained position vectors, (b) one state that is overwritten through time, and (c) two latents with separate read connections to all five inputs. Keep earlier records visible only in the forms that retain them. An arrow indicates possible information flow, not successful recall.

Four different lengths deserve separate names:

| Quantity | Question it answers |
| --- | --- |
| Input length | How many positions are supplied to a computation? |
| Direct attention span | Which earlier positions can this query consult individually at this layer? |
| State or cache capacity | What representations are stored between steps or segments? |
| Useful dependency length | How far back does the model actually use information successfully on this task? |

A recurrent model can process an arbitrarily long stream using bounded state while losing a particular early fact. A finite-window attention layer may receive a state that already summarizes older material. A latent model can read all pixels of an image without preserving every detail in its bottleneck. These are mechanisms to evaluate, rather than contradictory claims about a single “context length.”

Our document example needs selective recall. A task asking for the running mean of a sensor may need only a sum and a count. The second task is naturally compressible; the first becomes harder as the number of independently retrievable facts grows. The task tells us what memory must preserve.

## 2. A small attention operation you can calculate

An attention query is a vector expressing what information is being requested. Each available item supplies a **key**, used for comparison, and a **value**, the information to combine. Learned linear projections normally produce these vectors from the current representations.

For one query q and available pairs (kⱼ,vⱼ), compute a score, normalize the scores, then average the values:

\[
s_j=\frac{q^\top k_j}{\sqrt{d_k}}+b_j,
\qquad
\alpha_j=\frac{\exp(s_j)}{\sum_{r\in\mathcal A}\exp(s_r)},
\qquad
o=\sum_{j\in\mathcal A}\alpha_jv_j.
\]

Here dₖ is the number of coordinates in a query or key; bⱼ is an optional positional bias; and 𝒜 is the set of positions the query is allowed to use. A dot product multiplies matching coordinates and adds them. The square-root factor helps control the size of dot-product scores as the number of coordinates changes. The softmax weights are nonnegative and sum to one across the allowed keys.

Let q=1, use one-dimensional keys ln 2, 0 and ln 3, and values 2, 4 and 8. The unnormalized weights are 2, 1 and 3. If all three keys are legal, the result is

\[
o=\frac{2(2)+1(4)+3(8)}{2+1+3}=\frac{16}{3}.
\]

Suppose the third item is in the future. It must contribute neither a value nor a term to the normalization. The legal weights become 2/3, 1/3 and 0, so the result is 8/3. Setting its value to zero while leaving its weight in the denominator would give 4/3, a different and incorrect masked computation.

**Inline figure: a query, three key/value pairs and a blocked future edge.** Show the score-to-weight calculation at the edges and the two actual weighted contributions entering the output. Beside it, show the denominator changing from 6 to 3. This is more informative than a grid of unlabeled attention colors.

For many queries, arrange them as the rows of Q. Keys and values are rows of K and V. If Q has shape L×dₖ, K has shape N×dₖ and V has shape N×dᵥ, then QKᵀ is L×N and the output is L×dᵥ. **The number of output positions comes from the queries.** We will use that fact again when a small latent array queries a large input.

For next-token prediction, the representation at position i may use inputs through i and predict token i+1. A **causal mask** permits j≤i. For a complete-trajectory classification task, all measured points are already available before the classification is requested, so a noncausal read is appropriate. The mask follows the task's information boundary.

## 3. Transformer-XL: continue across segment boundaries

### 3.1 Why separate chunks forget

Suppose a long stream is processed four positions at a time. An ordinary independently processed chunk starts fresh at positions 4, 8, 12 and so on. A detail near the end of one chunk is invisible to a query near the beginning of the next. This is **context fragmentation**: a computational boundary cuts across an otherwise coherent sequence.

Transformer-XL carries earlier hidden representations into the next segment. Current positions supply the queries. A concatenation of retained memory and current-segment representations supplies keys and values. The mask still prevents current queries from reading future positions. The memory therefore extends the attention input; it does not remove the causal rule.

**Inline figure: a two-dimensional segment/layer diagram.** Lay segments horizontally and layers vertically. Draw an old lower-layer state feeding the next segment's upper-layer attention, alongside the current lower-layer states. Draw the forward edge solid and the backward-gradient edge stopped at the memory boundary. This exposes the layer shift that a simple circular “memory” arrow would hide.

For layer ℓ in segment s, let Hₛ⁽ℓ⁻¹⁾ have shape L×d and retained memory Mₛ⁽ℓ⁻¹⁾ have shape M×d. The attention inputs are

\[
\widetilde H_s^{(\ell-1)}
=[\operatorname{stopgrad}(M_s^{(\ell-1)});H_s^{(\ell-1)}],
\quad
Q=H_s^{(\ell-1)}W_Q,
\quad
K=\widetilde H_s^{(\ell-1)}W_K,
\quad
V=\widetilde H_s^{(\ell-1)}W_V.
\]

The semicolon concatenates rows, giving M+L available representations. Attention produces L current outputs, not M+L newly recomputed outputs. After the segment, retain the most recent M appropriate hidden states for the following segment.

The [Transformer-XL paper, sections 3.2–3.3](https://arxiv.org/pdf/1901.02860) defines the segment recurrence and relative-position construction. The representation cache is detached during training. Its values affect the current prediction, but the current loss does not backpropagate through the operations that created those old values. Parameters used to project the detached memory into current keys and values still receive gradients.

### 3.2 Follow a complete cache example

Use five scalar keys [ln 9,0,0,0,0] and values [6,1,8,2,0]. Every query is 1. Process segments [0,1], [2,3], [4], using no positional bias in this first calculation.

At the final query, compare three memory lengths:

| Retained positions before the final segment | Legal values | Output |
| --- | --- | --- |
| None, M=0 | Current value 0 | 0 |
| Positions 2 and 3, M=2 | 8, 2, 0 with equal weights | 10/3 |
| Positions 0–3, M=4 | 6, 1, 8, 2, 0 with weights proportional to 9,1,1,1,1 | 65/13=5 |

The output is a soft combination, not a lookup returning exactly 6. The larger cache makes the strong matching key available again. With M=4, every query in this five-position example sees the same legal keys as the full causal calculation, so their outputs agree. This equality holds here because the projections and positions are fixed and every required key remains available.

**Investigation 1 — build the retained context.** Edit the five keys and values, place segment boundaries, and choose a memory length; show a selected query, record which positions you think will be visible and inspect its output. The display exposes the actual cache, legal edges, denominator and weighted contributions. Change a future value and test an earlier query: its result should stay fixed. Move an informative key outside the retained cache and explain the change.

A real multilayer Transformer-XL can pass older information through representations that were themselves contextualized. Its indirect dependency paths differ from a single layer's list of visible positions. With the paper's one-segment memory construction, crossing a segment boundary also moves through a layer, giving a depth-dependent receptive field. A finite network and cache do not imply unlimited exact recall.

### 3.3 A reused position needs a coherent address

Imagine numbering positions within every segment as 0,1,2,3. A cached token from local position 1 and a new token at local position 1 now carry the same absolute-position label. Reusing those labels creates ambiguity about temporal relationships.

A relative position asks, “How far is this key from the query?” A query at global position 5 sees keys at positions 1 and 5 at distances 4 and 0. Shifting all three global positions by 100 preserves those distances. Resetting segment-local counters does not.

**Inline figure: repeated local labels versus relative-distance rulers.** Show the two “position 1” tokens in different segments, then draw distances from the same query using global order. The invariant is the distance, not the printed segment counter.

Transformer-XL uses separate content and relative-position projections. One head's score can be written

\[
s_{ij}=q_i^\top k_j+q_i^\top r_{i-j}
+u^\top k_j+v^\top r_{i-j},
\]

with scaling applied according to the head implementation. The first term matches content; the second makes preferred distance depend on the current query; the third is a learned global content preference; the fourth is a learned global distance preference. Here r is the projected relative-position vector, and u and v are learned vectors. These v and r symbols are score parameters; vⱼ in section 2 denoted an attention value.

Our downloadable scalar calculation can instead add −β(i−j) to the score. It is a transparent recency-bias exercise, **not Transformer-XL's full positional formula**. It lets you observe how recency can compete with a content match. The later positional-encoding lesson explains sinusoidal features, RoPE and other constructions separately.

## 4. Griffin: keep a recurrent summary and consult nearby detail

Transformer-XL retains a sequence of old representations. A recurrent layer updates a fixed-width state instead. To see the tradeoff, begin with

\[
h_t=a h_{t-1}+b x_t.
\]

If a=.8 and b=.6, an input of 1 followed by zeros produces states .6, .48, .384, .3072, .24576 and .196608. The initial input is still influential, but its contribution has decayed. With a fixed state dimension, each new update changes the same limited workspace.

Griffin's **real-gated linear recurrent unit**, RG-LRU, lets the current input influence how strongly to retain old state and admit new input. For vector inputs, all multiplications below are coordinatewise:

\[
r_t=\sigma(W_r x_t+b_r),\qquad
i_t=\sigma(W_i x_t+b_i),\qquad
a_t=a^{c r_t},
\]
\[
h_t=a_t\odot h_{t-1}
+\sqrt{1-a_t^2}\odot(i_t\odot x_t).
\]

The sigmoid σ maps a number into (0,1); a is a learned coordinatewise decay base in (0,1), and c controls the gate's exponent scale. The paper uses c=8. The input gate iₜ controls the incoming signal. The recurrence gate rₜ changes both retention and the normalization of that incoming signal. [Griffin, section 2.4](https://arxiv.org/pdf/2402.19427).

With a=.8 and rₜ=1/8, the effective decay is .8. With rₜ near zero, aₜ approaches 1 and the injection multiplier approaches zero. The unit can preserve a state across an uninformative interval instead of repeatedly replacing it. In the exact limiting case rₜ=0, it holds the state unchanged. Finite sigmoid logits approach that endpoint but do not produce it exactly.

**Inline figure: retention and injection as separate contributions.** For each timestep show a retained-state bar, an incoming-signal bar and their sum. Align these with the actual input strip. A gate shown merely as an open or closed door would conceal the square-root normalization and the two different gates.

The normalization has a useful explanation. If hₜ₋₁ and the gated input each have variance 1 and are uncorrelated, then aₜhₜ₋₁+√(1−aₜ²)xₜ has variance aₜ²+(1−aₜ²)=1 when the coefficient is treated as fixed. Learned, input-dependent signals need not satisfy those assumptions; the calculation explains the design rather than asserting every hidden coordinate has variance exactly one.

For the six-step impulse above, use r₁=1/8 and r₂…r₆=.001. The final state is about .594668, close to the first state's .6. If we only set the later input gates to zero while leaving rₜ=1/8, the final state remains .196608. The inputs were already zero; stopping input admission did not stop the recurrence from decaying.

**Investigation 2 — preserve or replace a memory.** Place nonzero events in a short sequence and edit the two gate sequences separately. Show the current computed result and its contributing terms immediately. Test a later distractor or reverse two input events. The all-zero input from a zero initial state is a checked no-change case. Learn to say which gate caused a change, not just that “gating helps memory.”

### 4.1 The recurrent unit is one component of Griffin

The full recurrent block projects the input into two branches. One branch includes a short causal depthwise temporal convolution and RG-LRU; the other supplies a nonlinear gate. They are multiplied and projected back to the model width. Residual connections, normalization and a gated feed-forward block surround temporal mixing.

Griffin then alternates recurrent mixing with **local sliding-window attention**. Its published construction uses two recurrent blocks followed by one local-attention block; the usual window in that study is 1,024 tokens. Recurrent state can carry a compressed influence from farther back, while local attention can compare recent positions individually. The [official RecurrentGemma architecture walkthrough](https://developers.googleblog.com/en/gemma-explained-recurrentgemma-architecture/) provides a visual implementation-oriented tour of this hybrid.

**Inline figure: recurrent, recurrent, local-attention blocks on a shared sequence.** Show a fixed-width state path traversing time and a bounded window of separately retained key/value pairs. These are different memory allocations. Do not draw distant input tokens as though the local-attention block could directly read them.

A local-attention cache is bounded once its window is full. A recurrent block also needs its state and the short convolution history. Dense projection and feed-forward weights remain part of the model. “Memory independent of stream length” refers to the bounded recurrent/cache state under a fixed architecture, not to the entire training computation or arbitrary batch size.

The next lesson studies state-space models in depth. RG-LRU is not presented in its source as a discretization of a continuous-time system, and this chapter's gated recurrence should not be silently substituted for a Mamba update. Similar diagrams can conceal different equations.

## 5. Perceiver: put deep computation in a smaller workspace

Sometimes the whole input is already available: an image to classify, a recorded sound, or an array of measurements. The challenge is repeatedly processing every pair of input positions.

Perceiver introduces N learned latent vectors, often with N much smaller than the input length T. Their queries attend to keys and values derived from the T input elements. The cross-attention output has N rows because the queries have N rows. Further self-attention operates between those N latent rows. Updated latents can later query the original input again.

The architecture therefore has two distinct kinds of depth: computation among the current latents, and repeated reads from the original input. A second read can collect details that became relevant after the first. Weight sharing between repeated blocks can reduce parameter duplication while still performing additional computation. [Perceiver, section 3.1](https://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf).

**Inline figure: input array → asymmetric read → latent processing → another read.** Keep the large input array in place across both reads. Label a cross-attention score array N×T and a latent self-attention array N×N. This is a change in the number of query positions, not a claim to approximate the original T×T softmax with the same outputs.

### 5.1 Two latents can ask different questions

Take three input positions with scalar position keys [−1,0,1] and values [2,6,10]. Give two latents scalar queries −ln 2 and +ln 2. The first query produces weights [4/7,2/7,1/7]; the second produces [1/7,2/7,4/7]. Their outputs are

\[
z_1=(4\cdot2+2\cdot6+1\cdot10)/7=30/7,
\quad
z_2=(1\cdot2+2\cdot6+4\cdot10)/7=54/7.
\]

The first workspace vector summarizes earlier values more strongly; the second summarizes later values more strongly. Their joint representation contains information that a single uniform mean would lose. A learned model can use richer keys and queries; this hand example isolates the read operation.

A uniform query gives the mean 6 for both [2,6,10] and [0,6,12]. If a later computation receives only that single mean, it cannot distinguish those two inputs. Once the inputs have collapsed to an identical representation, any deterministic downstream function must give them the same output. This is a concrete bottleneck failure, not a claim that every one-latent nonlinear model is exactly a mean.

**Investigation 3 — choose what the workspace can distinguish.** Edit the values and their attached positions, and move the latent queries; show inspect which input will contribute most to each latent and whether two different input arrays will remain distinguishable. Reorder whole position/value records and observe the unchanged read; move values to different positions and observe the changed problem. Add a second query when it preserves a distinction your single summary discarded.

### 5.2 Position is data when the encoder reads a set

If keys and values are permuted together, cross-attention with fixed latent queries produces the same result. The score columns and their corresponding values move together, so the weighted sum is unchanged. This is desirable for a storage reordering, but it means that order must be represented if it matters to the task.

Perceiver supplies position features alongside input features, commonly Fourier features for spatial or temporal coordinates. A frame with its correct timestamp may be moved to another storage row without changing its meaning. Assigning that frame another timestamp changes the input. In the trajectory program, each point carries a normalized position from −1 to 1.

A Fourier feature expands a coordinate x into sine/cosine measurements at selected frequencies: for example, [x, sin(πx), cos(πx)]. At x=0 this is [0,0,1]; at x=1/2 it is [1/2,1,0]. Adding more frequencies gives the learned projections several spatial scales to combine. The original coordinate helps distinguish positions that a periodic component alone would identify. For d coordinate dimensions and K frequency pairs per coordinate, this concatenation has d(2K+1) entries. It supplies location information; it does not change the paired-permutation argument. Our small classifier uses the raw ordinal tag alone so its input representation stays transparent.

**Inline figure: paired permutation versus reassigned positions.** Connect each value to its position tag. One panel moves complete tagged records; the other keeps the tags fixed and moves only values. This makes an invariance that often sounds abstract visible as a concrete difference in what was edited.

### 5.3 The latent workspace can serve many outputs

Original Perceiver aggregated its processed latents for tasks such as classification. **Perceiver IO** adds output queries. A query might specify a pixel coordinate whose motion is requested, a language position to predict, or a desired modality. It reads keys and values from the processed latent array.

If there are O output queries, the output has O rows. For example, four output-coordinate queries can request four motion vectors from the same two latent vectors. The output size is independent of the number of input rows and of the chosen number of latents. What can be accurately reconstructed still depends on the information those latents retained. [Perceiver IO, sections 3.1–3.2](https://arxiv.org/pdf/2107.14795).

The interesting connection is that attention becomes an **interface between array sizes**, not only a way to relate words. Input queries move information into a working representation; output queries specify what information to read from it. The later cross-attention-architectures lesson uses this perspective to connect modalities and pretrained systems.

## 6. Which Perceiver is allowed to generate left to right?

Original Perceiver uses noncausal attention. It can read the whole observed input for a classification task. If an early language-model output reads a latent that already encoded future target tokens, masking only the final output attention cannot undo that leak.

**Perceiver AR** aligns a smaller number of latents with selected final input positions. A latent corresponding to input position i may cross-attend only to positions j≤i; it predicts token i+1. Latent self-attention also uses the causal order. Both stages must preserve the information boundary.

For input positions 0,1,2,3,4, take latents aligned with positions 3 and 4. The first may read inputs 0–3 and predict token 4. The second may read inputs 0–4 and predict token 5. During latent self-attention, the first latent cannot read the second: that second latent has already seen input 4, which is the first latent's target.

**Inline figure: the two masks and a leaking detour.** Draw the legal input-to-latent edges and legal latent-to-latent edges. Then highlight the forbidden detour input 4 → latent 4 → latent 3. The first mask alone leaves that detour open if latent self-attention is unrestricted.

The [authors' ICML presentation slides, pages 3–10](https://icml.cc/media/icml-2022/Slides/17886.pdf) build this input/query/target alignment incrementally. The [Perceiver AR paper](https://proceedings.mlr.press/v162/hawthorne22a/hawthorne22a.pdf) describes the architecture and training/inference choices. A latent bottleneck does not make every Perceiver variant a streaming recurrent model: retaining or rereading a long input and updating a fixed recurrent state are different execution contracts.

Return to the document question. Transformer-XL may retain the entrance's position vector if it remains in memory, or pass its influence indirectly through later states. Griffin may preserve it in recurrent state while consulting recent text explicitly. A Perceiver read can consult it directly while the input remains available, provided the learned queries and latent capacity preserve what the task needs. Those are testable paths, not guarantees that every trained model will answer correctly.

## 7. Run the mechanisms, then try real movement data

### 7.1 The small calculations

Download [sequence_mechanisms.py](sequence_mechanisms.py). It is a complete NumPy program containing attention, a segmented memory trace, the scalar gated recurrence and the latent calculations. It also checks selected invariances with explicit alternative inputs.

Use Python 3.12 or later in your own environment:

```text
python -m venv .venv
python -m pip install numpy
python sequence_mechanisms.py
```

Activate that environment before the `python -m pip` command, or use its Python executable explicitly. On Windows the executable is `.venv\Scripts\python.exe`; on macOS/Linux it is `.venv/bin/python`. The author run used Python 3.12.14 and NumPy 2.3.5. The script writes `checked-results.json` beside itself.

The core operation is short enough to inspect:

```python
scores = query @ keys.T / math.sqrt(keys.shape[1])
scores = np.where(allowed, scores, -np.inf)
weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
output = weights @ values
```

Every query must have at least one legal key. Subtracting the row maximum preserves softmax while avoiding unnecessarily large exponentials. The program's reusable function also accepts a positional bias. The memory example deliberately fixes projections and omits a full neural stack so you can see exactly which input changed an output.

Selected executed results are:

```text
Final output with memory 0: 0.000000
Final output with memory 2: 3.333333
Final output with memory 4: 5.000000
Impulse state after six steps, ordinary decay: 0.196608
Impulse state after six steps, near-hold gates: 0.594668
Two latent outputs: 4.285714, 7.714286
```

These results recur from the hand calculations: 4.285714 and 7.714286 are 30/7 and 54/7 from section 5. The agreement links the weighted-edge picture to the matrix operation. Change the input values before changing the implementation, and calculate one expected output yourself.

### 7.2 A complete trainable latent classifier

Now ask a real question: **Can a compact representation of a hand's path identify which of fifteen recorded movement classes it belongs to?** The UCI Libras Movement dataset provides 360 hand trajectories extracted from videos: forty-five ordered two-dimensional positions per trajectory, followed by a class label. Its classes include circles, arcs, straight lines, waves and zigzags. These coordinates describe one hand's movement; they are not a complete representation of sign-language meaning. [UCI collection description and CC BY 4.0 license](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement).

A point is a measured hand centroid at a selected video frame. The collectors normalized the videos into forty-five sampled positions. We therefore know the point order, but do not infer exact time intervals, hand speed in meters per second or physical distances from these unit-space coordinates. The original metadata describes four performers and two sessions; individual rows do not identify performer or session.

Before splitting, we found thirty duplicate copies of trajectories, all with matching labels. Keep the first occurrence of each exact coordinate sequence: 330 distinct trajectories remain. Within each class, a fixed NumPy seed of 73 determines a permutation. Reserve the last four trajectories for test; use the first two-thirds, rounded down, for fitting and the rest for validation. This gives **220 fit, 50 validation and 60 test trajectories**. No exact coordinate sequence crosses roles. The missing performer/session identifiers mean this row-level experiment cannot establish performance on a new performer or a new recording session.

These forty-five-point sequences are a practical CPU exercise in latent sequence processing. They let us inspect learned reads and compare ways of preserving shape. They do not measure long-context language-model performance.

Download [movement_libras.data](movement_libras.data) and [latent_trajectory_classifier.py](latent_trajectory_classifier.py) into one directory. The [data record](data-provenance.md) supplies attribution, source-row roles, duplicate handling and exact transformations. The program runs offline once dependencies are installed:

```text
python -m pip install numpy torch scikit-learn
python latent_trajectory_classifier.py
```

The tested snapshot is PyTorch 2.14.0+cpu, NumPy 2.3.5 and scikit-learn 1.9.1. The program uses two CPU threads and writes `trajectory-results.json` and `small-fits.npz`. The latter stores the small learned arrays and selected read weights.

**Two baselines ask different questions.** Transform each coordinate with the fixed rule 2x−1, then average the forty-five x and y coordinates. A logistic classifier receives just two numbers describing the path's center. It cannot distinguish paths with the same center. A second logistic classifier receives all ninety coordinates in their original order. It retains the path's shape and ordering, although its decision function is linear in those coordinates. Both use C=1, fit only the fit rows and use no learned preprocessing across roles.

**Latent model:** append a position tag running from −1 to 1, producing 45×3 inputs. Project each row to width 24. Start with N learned latent vectors, read the input positions, process the latent interactions and a feed-forward block, then repeat that read/process step once with shared weights. Average the final latents and map them to fifteen class logits. Compare N=1 and N=4 under the same declared fit/validation protocol.

**Inline figure: the actual trajectory classifier.** A numbered two-dimensional path sits beside its forty-five (x,y,position) rows. These project to width 24, enter two latent read/process rounds and produce fifteen logits. Distinguish coordinate values, ordinal position tags and an optional validity mask. Masked padding is a storage device, not another measured point.

The central model loop is:

```python
inputs = self.input_projection(frames)
latent = self.latent.unsqueeze(0).expand(len(frames), -1, -1)
for _ in range(2):
    update, weights = self.cross(self.cross_norm(latent), inputs, valid)
    latent = latent + update
    normalized = self.self_norm(latent)
    update, _ = self.self_attention(normalized, normalized)
    latent = latent + update
    latent = latent + self.feedforward(self.final_norm(latent))
logits = self.classifier(latent.mean(dim=1))
```

The complete file defines the projections, attention, normalization, feed-forward block, duplicate handling, data roles, training, metrics and saving. The shared second read uses updated latents to ask the input a new question. These are the operations of a small Perceiver-style classifier, not the parameter counts or training settings of the research models.

For a trajectory with true class c, cross-entropy is −ln p(c), with probabilities obtained by softmax of the fifteen logits. Its derivative with respect to logit k is pₖ−1[k=c]. At uniform predictions, the correct logit's derivative is −14/15 and each other derivative is +1/15. A gradient step raises the correct logit relative to the alternatives; backpropagation also changes the latent queries and projections that produced it. Pass logits, not precomputed probabilities, into PyTorch's cross-entropy operation. [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

Train for eighty full-batch epochs with Adam at learning rate .005. Each run retains the epoch with lowest validation cross-entropy. Latent counts 1 and 4, seeds 11 and 29 and the training settings are declared before interpreting test results. Evaluate each selected checkpoint on test only after its validation-based selection.

| Model | Seed | Stored parameters | Selected epoch | Fit errors /220 | Validation errors /50 | Test errors /60 |
| --- | --- | --- | --- | --- | --- | --- |
| Mean coordinates + logistic | — | 45 | — | 183 | 45 | 54 |
| Ordered coordinates + logistic | — | 1,365 | — | 35 | 17 | 22 |
| One latent | 11 | 7,815 | 80 | 95 | 36 | 32 |
| One latent | 29 | 7,815 | 68 | 67 | 24 | 25 |
| Four latents | 11 | 7,887 | 75 | 68 | 29 | 28 |
| Four latents | 29 | 7,887 | 72 | 90 | 30 | 34 |

The mean baseline makes 54/60 errors: knowing a path's center is a poor substitute for its shape. The ordered linear model makes 22/60 errors, about 36.7%, and outperforms all four small latent runs here. The latent models improve on the mean baseline, but their 25–34 test errors show that having learnable selective reads is not enough to make fitting reliable on this small sample. Four latents do not consistently beat one.

This comparison is useful precisely because it separates representational possibility from achieved performance. The models are not parameter-matched; the simple ordered baseline has a helpful representation supplied directly. Two seeds show sensitivity in these runs, not a confidence interval or a universal architecture ranking. No setting was changed after seeing these outcomes.

**Inline figure: individual measured error marks.** Use separate validation and test panels, both on a zero-to-one-hundred-percent error axis. Show both baselines and the four individual runs, labeled with counts and seeds. Avoid a fitted line or a single average that hides disagreement.

**Investigation 4 — inspect a learned read.** Select a validation trajectory, look at its numbered path and each latent's second-read weights, then Show the current computed result and its contributing terms immediately. Recompute the frozen model's logits and weights. Compare its response with the mean-coordinate baseline on the same retained points. A changed hypothetical path has no automatically known new class label: predicting the model's behavior is different from proving it classified the edited movement correctly.

Look for two different outcomes: an edit that changes the winning class, and a smaller edit that changes probabilities without changing that class. The distinction matters because a class label hides all changes that fall short of crossing a decision boundary. Explain both outcomes using the observed inputs and outputs ting the retained baseline.

Two null checks passed for all four fitted models. Permuting complete coordinate-plus-position records preserves logits within floating-point tolerance. Appending five points filled with 1,000 but masked out also preserves logits. If masked points alter the output, a mask or an earlier operation has admitted information that was supposed to be absent.

## Three complete implementation routes, with their boundaries exposed

The local [sequence_mechanisms.py](sequence_mechanisms.py) owns the transparent cache, gated recurrence and latent-read arithmetic. The [latent_trajectory_classifier.py](latent_trajectory_classifier.py) owns a complete differentiable model and its learning/evaluation loop. The new [memory_library_bridge.py](memory_library_bridge.py) connects these mechanisms to ordinary maintained operators without calling a small fixture a full reproduction of three named systems.

**Segment memory:** `check_segmented_attention` builds a real K/V cache, appends the current segment, constructs a causal mask using absolute retained positions, then keeps the last four entries. Its explicit stable score→softmax→weighted-sum path is compared to `F.scaled_dot_product_attention` on exactly the same projected values, including a final short segment. In SDPA's Boolean mask, `True` means allowed. Calling `is_causal=True` on a nonsquare query/cache matrix without considering alignment can select the wrong history. Detaching cache tensors cuts gradient history; it does not reset their values or make a stream's cache suitable for another stream.

This is the segment-memory mechanism described in §3. It is not a complete Transformer-XL checkpoint: the paper also specifies layerwise hidden-state recurrence, relative-position scoring, normalization and a language-model head. Our illustrative distance bias and already-projected one-layer K/V cache must keep those labels. The later prepared [Self-Attention mechanism source](../self-attention-multi-head-attention/lesson.md) owns the full projected multihead attention construction; that improved page is not yet published and is a deeper reference rather than an unstated prerequisite for this local small read.

**Recurrent memory:** `rglru_reference` goes beyond the earlier fixed scalar gates. It calculates learned block-diagonal input and retention gates, stable `-expm1(2*log_decay)` injection scale, per-example reset positions and each updated vector state. No recurrent-cell API hides that computation. The optional `check_griffin_component` copies actual DeepMind `RGLRU` parameters, compares outputs and the last cache, continues in two chunks and checks gradients. The native implementation clips an extreme square-root derivative for training stability; the comparison explicitly uses moderate float32 decay outside that clipped regime. This is the RG-LRU component, not an entire Griffin language model or its short temporal convolution/local-attention schedule. [Official recurrent component source](https://raw.githubusercontent.com/google-deepmind/recurrentgemma/main/recurrentgemma/torch/layers.py), inspected22September2026, owns the exact package convention.

**Latent workspace:** `check_latent_read` reuses our existing learned `Attention` class, obtains its projected query/key/value tensors and passes those exact tensors to SDPA. It then applies the same output projection and compares outputs plus gradients for source, latent query and every parameter. The complete classifier alternates this read with latent self-attention and its feed-forward block. A generic full-input model would erase the architectural question; replacing only the matching read preserves it.

Run `python memory_library_bridge.py` with PyTorch, NumPy and scikit-learn. Adding `--griffin` requires a compatible `recurrentgemma` installation with its Torch dependencies. The ordinary Torch comparisons and optional specialist program are supplied as executable instructional content; optional package parity is not claimed executed. Record the resolved package version when running it.

The explicit attention reference materializes query×visible-key scores; the normal SDPA backend can avoid retaining that matrix under supported conditions. A bounded segment of length Q with M retained positions uses O(Q(Q+M)) score cells per head in the reference. RG-LRU streaming retains one width-sized state per example, while training retains its needed history. Latent reads use L×S score cells for L latents and S source positions. These are separate computational contracts, so there is no one “linear-memory” claim covering all three.

**Change the stream.** Set segment length4, keep only2 previous positions, and process11 inputs. Then place a recurrent reset at position6 while preserving every input. Compare the supplied reference and package paths again.

<details><summary>Hint</summary>The last segment has3 queries. Cache truncation discards earlier keys; a recurrent reset clears prior state for exactly the selected example.</details>

<details><summary>Solution and success criteria</summary>Build legal key positions from the retained absolute indices plus the current segment, mask future indices, and trim the cache only after obtaining that segment's outputs. The first query of the last segment sees indices6,7,8; its later queries can also see9 and10 as they arrive. In RG-LRU, set `segment_pos[:,6]=0` and restart its within-document counter; the reset input uses the special fresh-document injection and cannot depend on the earlier cache. Verify unaffected earlier outputs and a changed post-reset state. This tests state ownership rather than memorizing model names.</details>

## 8. Deeper: gradients, budgets and a useful evaluation plan

This section is a deeper route. It makes the architectures' costs and training boundaries precise after their forward mechanisms are familiar.

### 8.1 Forward memory and gradient memory

A detached cache illustrates two different graphs: the graph of values used in a prediction, and the graph followed by differentiation. Transformer-XL keeps an old representation in the value graph while cutting its backward connection to its earlier computation. Replacing the cache by zero changes the forward computation; detaching it does not change its forward numbers.

For a scalar example, let old state m=2w and current prediction y=w·stopgrad(m). At w=3, m=6 and y=18. The detached derivative with respect to the current w is 6. If we differentiate through the original m=2w as well, y=2w² has derivative 12. This difference explains truncated credit assignment. It does not imply the old state was ignored.

The same distinction matters for recurrent training. Constant inference state does not mean backpropagation stores a constant number of intermediate activations. Training may retain a sequence, recompute activations, use checkpointing or use a specialized scan. State-space and sequence-parallel lessons develop those choices.

### 8.2 Count the particular resource you mean

Let T be input length, L segment length, M retained memory, W local window, N latent count and D the number of latent layers. Hold channel widths and head counts fixed for the following attention-interaction counts:

| Computation | Leading interaction count | What remains separately important |
| --- | --- | --- |
| Dense attention on T positions | O(T²) per layer | Projection/MLP work, causal mask, training activations |
| Segments of length L with memory M | O(T(L+M)) per layer | How memory is filled and detached; hidden-state storage |
| Local attention with window W | O(TW) per layer | Recurrent blocks and their state in a hybrid |
| One Perceiver input read plus D latent layers | O(TN+DN²) | Additional input reads, channel projections, output decoding |
| Perceiver IO with O output queries | Add O(ON) | Output features and output-head work |

If there are R separate input reads, count R·TN, not just TN. If N grows proportionally to T, the bottleneck no longer gives linear input scaling under fixed depth. The recurrent scalar/vector update is linear in sequence length for fixed state width; obtaining its input projections is additional work.

For T=4,096, a dense T×T score array contains 16,777,216 cells. With L=256 and M=128, the rectangular per-segment upper count is 16·256·384=1,572,864 cells across the sequence. Window W=128 gives at most 524,288 score slots. One read to N=32 latents plus eight latent layers gives 4,096·32+8·32²=139,264 cells. These are algebraic score counts under the stated model, not GPU timings or exact complete-model FLOPs.

**Inline figure: attention interaction budgets.** Use a small exact matrix diagram to explain each product, followed by a logarithmic count axis for these large numbers. Label when a count includes eight latent layers versus one input-length layer. Keep the corresponding arithmetic visible and avoid a “fastest model” podium.

Attention computation and persistent key/value cache size have different growth. A full multi-head KV cache for one layer, T positions, total key/value width d and b bytes per stored element uses 2Tdb bytes. With T=4,096, d=64 and b=2, that is 1,048,576 bytes, or 1 MiB. Restricting the retained window to 128 positions gives 32,768 bytes, or 32 KiB. The cache grows linearly with T; the straightforward attention score matrix has quadratic cells. Fused attention can avoid materializing that entire matrix without changing which pairs the operator compares.

Transformer-XL may cache hidden activations rather than the exact projected KV representation assumed in that formula. Multi-query/grouped-query models change stored KV width. Use each implementation's actual cache contract rather than applying the MHA formula to every architecture with “attention” in its name.

### 8.3 Linear recurrences can have parallel training algorithms

For input-dependent coefficients already computed from the input, a recurrent step is an affine map h↦a⊙h+b. Two steps compose to

\[
(a_2,b_2)\circ(a_1,b_1)
=(a_2\odot a_1,\ a_2\odot b_1+b_2).
\]

Composition is associative. Therefore a parallel prefix algorithm can combine these maps in a tree rather than evaluating every dependency with a serial host-language loop. This preserves the mathematical recurrence, with ordinary floating-point-order differences. It is different from making the gates depend on the previous hidden state, which would prevent precomputing those affine maps in the same way. The next state-space lesson will use this distinction repeatedly.

### 8.4 Evaluate the information claim, not just input acceptance

For the document task, construct examples with the queried fact at several distances, distractors with similar wording and multiple independent facts. Hold the question and answer format fixed. Compare short recent context, the intended long-context method, and an explicit retrieval baseline that supplies the relevant passage. Track accuracy by distance and number of competing facts, together with actual retained memory and measured latency if performance is being studied.

A repeated fact at the end is a useful control: it removes the need to remember its early occurrence. An unrelated prefix is another control: extra tokens should not help a task whose answer is entirely local. For summarization, use a different assessment of factual coverage and attribution; success on one isolated “needle” is not a complete test of document understanding.

For the movement exercise, the held-out unit is a complete trajectory, not a shuffled point. Splitting points would mix parts of the same path across fit and test. Exact duplicate trajectories also belong to one role or must be removed before splitting. A new-performer or new-session claim needs the corresponding identifiers and a grouped protocol, which these public rows do not provide. These decisions determine what “works” means before fitting.

The named papers report historical experiments under their own data and hardware settings. We use them to investigate mechanisms and research evidence, without transplanting their numerical speedups into the small CPU examples here. The important practical habit is to preserve one interpretable change, a credible baseline, the task's information boundary and a result you can inspect.

## 9. Practice: implement a change and explain its effect

### 1. Repair a masked weighted average

A one-dimensional query gives unnormalized weights [3,1,5] and values [7,3,100]. The last position is in the future. Calculate the legal output and explain why dividing by 9 is wrong.

<details><summary>Hint</summary>

The future item disappears from both the weighted numerator and the normalizing denominator.

</details>
<details><summary>Solution</summary>

The result is (3·7+1·3)/(3+1)=6. Dividing by 9 would retain the future score in the denominator, shrinking the legal information. A mask is a restriction on participating key/value positions, not only a zeroing of future values.

</details>

### 2. Build a cache counterexample

Process [A,B,C,D,E,F] in segments of length 3 with memory length 2. At the query for E, which earlier/current positions are directly legal? Put the only informative value at a position that cannot be consulted. What memory length would recover it?

<details><summary>Hint</summary>

The second segment starts with the retained tail of the first. Current-segment future positions remain masked.

</details>
<details><summary>Solution</summary>

The second segment retains B and C, then contains D,E,F. The query at E can use B,C,D,E; F is future, and A was evicted. An informative A is a counterexample to direct retrieval with M=2. M=3 would retain A,B,C. In a deeper contextual model an indirect influence could survive elsewhere, but this question concerns direct access in the stated layer.

</details>

### 3. Separate the two gates

An existing scalar state is 2. For the next three steps there is no incoming signal, and effective decay is .75. Calculate the final state. Would reducing only the input gate preserve it? What effective decay would preserve it exactly in this mathematical fixture?

<details><summary>Hint</summary>

There is no incoming term to suppress. Follow the multiplier applied to the old state.

</details>
<details><summary>Solution</summary>

The final state is 2·.75³=.84375. Changing the input gate does nothing when the gated input is already zero. Effective decay 1 preserves state exactly. In RG-LRU that is the limiting recurrence-gate setting r=0 for a fixed base in (0,1); finite sigmoid logits approach the limit.

</details>

### 4. Diagnose a latent collision

A model replaces three scalar values by their uniform mean and then applies an arbitrary deterministic classifier. Construct two different inputs with the same mean but different last values. Can retraining only the downstream classifier let it identify the last value perfectly on both?

<details><summary>Hint</summary>

Make the sum agree while changing the last element. The classifier receives only one number.

</details>
<details><summary>Solution</summary>

[1,4,7] and [0,4,8] both produce mean 4, but their last values differ. Any deterministic classifier of that mean receives identical input in both cases and returns the same result. It needs a richer representation, another input read or a changed task. This proof is about the explicitly uniform mean; a learned one-latent attention mechanism need not compute that mean.

</details>

### 5. Find the indirect causal leak

Latent z₂ predicts token 3 and may read inputs through position 2. Latent z₃ predicts token 4 and may read inputs through position 3. Both cross-attention masks are correct, but z₂ can attend to z₃ in the next layer. Explain the leak and repair it.

<details><summary>Hint</summary>

Trace the target of z₂ through the other latent, rather than inspecting only its direct input edges.

</details>
<details><summary>Solution</summary>

Input token 3 can enter z₃ and then reach z₂. That reveals z₂'s target through a two-step path. The latent self-attention mask must also respect aligned position: z₂ cannot read z₃. End-to-end causality is a property of all paths through the computation.

</details>

### 6. Compare two memory budgets

A full multi-head KV cache has 12 layers, 2,048 positions, 8 KV heads, head width 32 and float16 storage. Calculate its size in MiB for one sequence. Then use a retained window of 256 with everything else unchanged. Name one allocation excluded from this count.

<details><summary>Hint</summary>

Count keys and values separately; a MiB is 2²⁰ bytes.

</details>
<details><summary>Solution</summary>

2·12·2,048·8·32·2=25,165,824 bytes=24 MiB. Replacing 2,048 by 256 gives 3 MiB. Parameters, activation storage, allocator overhead and other state are examples of excluded allocations. This formula assumes the specified 8 stored KV heads, not a multi-query cache with shared keys and values.

</details>

### 7. Design a meaningful trajectory ablation

Keep the supplied data roles. Compare the complete-coordinate/position permutation with a transformation that reverses the coordinate sequence but reassigns increasing positions. Predict which transformation should preserve the frozen latent model's output, then perform the comparison on one validation trajectory. Explain what a difference would establish.

<details><summary>Hint</summary>

The first transformation reorders records. The second changes which measured frame occupies each temporal position.

</details>
<details><summary>Solution</summary>

The paired permutation preserves the weighted reads and therefore the logits up to floating-point roundoff. Reassigning positions changes model inputs, so outputs may differ; they need not differ for every trajectory or fitted model. Retain the original and transformed logits, valid-point count and original label. A difference establishes sensitivity to that temporal reassignment in this example, not that temporal order universally improves movement classification. Never select a new model using the held-out test trajectories to make the result look stronger. The edited path's original label is a reference, not certified ground truth for the hypothetical edit.

</details>

### 8. A detached value still matters

Let m=4w and y=2w·stopgrad(m). At w=2 calculate y, the detached derivative with respect to w, and the derivative if m is not detached. Explain the distinction in words.

<details><summary>Hint</summary>

Hold the numerical memory value fixed in the first derivative; substitute m=4w before the second.

</details>
<details><summary>Solution</summary>

m=8 and y=32. With the cached value fixed, dy/dw=2m=16. Differentiating through m gives y=8w² and dy/dw=16w=32. Both computations use the same forward memory value, but one omits the earlier operation's contribution to the gradient.

</details>

You are ready to continue when you can draw the legal information paths, calculate one attention read and recurrent update, explain a bottleneck collision, and distinguish a stored-state count from a measured performance claim. The next topic is [State Space Models (S4, Mamba, Mamba-2)](/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals). It develops how a hidden state evolves, when a recurrence becomes a convolution, and why input-dependent selection changes the computation.

## 10. References & another way to learn it

* [Dai and colleagues, Transformer-XL](https://arxiv.org/pdf/1901.02860), research paper. Read section 3 with the segment/layer drawing beside you; appendix B is an advanced route to efficient relative-position score construction. The reported model comparisons are from the paper's 2019 setting.
* [De and colleagues, Griffin](https://arxiv.org/pdf/2402.19427), research paper. Section 2 gives the full blocks and RG-LRU equations; appendix A discusses the gate's behavior and stable parameterization. Useful after the scalar recurrence exercise.
* [Google Developers, RecurrentGemma architecture](https://developers.googleblog.com/en/gemma-explained-recurrentgemma-architecture/), illustrated article. A practical second tour through projections, recurrent layers and local attention. Use the paper for exact mathematical conventions and treat model-specific dimensions as the article's versioned examples.
* [Jaegle and colleagues, Perceiver](https://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf), canonical research paper. Section 3 explains the asymmetric read, latent processing, iterative reads and position features. Start here for the bottleneck architecture rather than an unverified implementation summary.
* [Jaegle and colleagues, Perceiver IO](https://arxiv.org/pdf/2107.14795), research paper. Sections 3.1–3.2 explain output queries; the optical-flow example supplies a concrete setting where the output is an array rather than one class.
* [DeepMind, Building architectures that can handle the world's data](https://deepmind.google/blog/building-architectures-that-can-handle-the-worlds-data/), creator article with architecture illustrations and linked audiovisual demonstrations. A less algebraic route to input/latent/output roles; the demonstrations are examples from the authors' historical systems.
* [Hawthorne and colleagues, Perceiver AR](https://proceedings.mlr.press/v162/hawthorne22a/hawthorne22a.pdf), research paper, and [ICML 2022 spotlight with slides and a listed video](https://icml.cc/virtual/2022/spotlight/17886). The inspected [slide deck](https://icml.cc/media/icml-2022/Slides/17886.pdf) builds the two causal masks step by step. The conference page lists a video; playback access was not verified for this packet. Read the slides independently of it.
* [PyTorch attention documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html), API reference for extending the explicit teaching implementation. Inspect mask semantics and dropout behavior when moving between APIs; a Boolean mask does not have the same meaning in every attention interface.
* [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), original data documentation. Read the sampling, coordinate layout and recording context before modifying the real example. Attribution and the exact retained data transformation are in the supplied data record.
