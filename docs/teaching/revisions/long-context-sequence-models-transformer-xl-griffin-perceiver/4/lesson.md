# Long-Context Sequence Models: Transformer-XL, Griffin and Perceiver

A note at the beginning of a document says, “The backup entrance is on the east side.” Much later, someone asks which entrance to use. A model must carry that earlier information forward or be able to consult it again. Merely accepting the whole document as input does not tell us whether it can answer.

Imagine reading the document through a small opening that shows only the last two messages. When the question arrives, the entrance note has disappeared. Reading the visible question more carefully cannot recover the missing direction. A useful system needs a deliberate way to carry information across that boundary.

**Inline figure: the entrance note leaves the visible window.** Show five ordered message positions, the entrance fact at position 0 and the later question at position 4. A two-message window contains positions 3–4. Contrast the missing direct evidence with carrying the entrance representation forward. The semantic answer is east, but the figure is an information-availability thought experiment, not a measured language-model prediction.

Look at what disappeared in the figure: the evidence, not the question. We will compare three ways of preserving useful evidence. **Transformer-XL** keeps earlier records available for later consultation. **Griffin** combines an updated summary with detailed access to the recent past. **Perceiver** reads a large available input into a smaller workspace where most subsequent processing happens. These are architectural choices about information flow; learning determines what each system actually preserves.

The same choice matters when a recorded hand movement must be identified as a line, curve or zigzag. Keeping only the path's average location is cheap, but it can erase its shape. Later we will train and inspect a small model on real hand trajectories, so the discussion ends with a concrete decision rather than a list of model names.

**How to read this lesson.** Sections 1–2 establish the memory objects and a single weighted read. Sections 3–6 explain one architecture at a time: the problem it solves, a small example, the mechanism and then the formal detail. Try each investigation after its worked example; changes appear immediately. Section 7 connects those operations to complete scratch and library programs and real data. Section 8 is a second-pass route through gradients, scaling and evaluation. Finish with the changed problems in section 9. You can take the architecture sections in separate sittings without losing the common question: **what remains available when the next computation needs it?**

The preceding [Bahdanau and Luong attention lesson](/learn/path/full-curriculum/attention-mechanism-bahdanau-luong?module=deep-learning-fundamentals) introduced attention as a way for a decoder to consult different encoder states. We will refresh that operation locally and then change what is available to consult. The later [self-attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals), [transformer-block](/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals) and [positional-encoding](/learn/path/full-curriculum/positional-encodings-sinusoidal-learned-rope-alibi?module=deep-learning-fundamentals) lessons develop those building blocks more fully. No knowledge of their implementations is required here.

## 1. Three kinds of memory are different objects

### Keep the record, update a summary, or make working notes

Start with an ordinary task: record five temperatures and answer questions about them. If the question is “What was the second reading?”, retaining the five readings is a direct solution. If the question is “What is the running average?”, a sum and a count are enough; the individual readings can be discarded. Those two tasks need different kinds of memory.

Neural models usually store a **representation** of an input: a list of feature values used by later computations. Such a list is called a **vector**. Its entries need not be readable facts like “east”; they are numbers learned for the task. The crucial question is whether each earlier position still has its own accessible vector or has been combined into something else.

There are three useful arrangements:

* **Keep separate records.** The next question can consult different earlier positions separately. A bounded collection of such stored representations is often called a **cache**.
* **Update a summary.** Each new input modifies the same fixed-size list of numbers. This repeatedly updated list is the **recurrent state**. A sum and a count are a simple hand-designed example; a neural state learns what to track.
* **Make a small set of working notes.** Keep the large input available, but collect selected information into a few working vectors. These are **latents**: internal representations inferred for the task. A collection of them is a latent array. Further computation happens among those vectors, and they may consult the original input again.

The figure below follows the same five inputs. In the middle arrangement, watch the state being updated: there is no separate slot from which to retrieve input 0. On the right, notice that the input bank remains available outside the smaller workspace. A latent bottleneck therefore differs from a stream that has already discarded its old inputs.

**Inline figure: three memory workspaces.** Show the same five input positions feeding (a) five retained position vectors, (b) one state that is overwritten through time, and (c) two latents with separate read connections to all five inputs. Keep earlier records visible only in the forms that retain them. An arrow indicates possible information flow, not successful recall.

### “Long context” can mean four different things

Returning to the entrance note, accepting every message is only the first requirement. The answering computation also needs an information path from that note, enough capacity to retain what matters, and learned behavior that uses it. Separate these questions when reading a model description:

| Quantity | Question it answers |
| --- | --- |
| Input length | How many positions are supplied to a computation? |
| Direct attention span | Which earlier positions can this query consult individually at this layer? |
| State or cache capacity | What representations are stored between steps or segments? |
| Useful dependency length | How far back does the model actually use information successfully on this task? |

A recurrent model can process an arbitrarily long stream using bounded state while losing a particular early fact. A finite-window attention layer may receive a state that already summarizes older material. A latent model can read all pixels of an image without preserving every detail in its bottleneck. These are mechanisms to evaluate, rather than contradictory claims about a single “context length.”

Our document example needs selective recall. A temperature average is naturally compressible into two numbers; many independently requested facts are harder to compress without losing distinctions. This is why there is no single best memory object for every task. Next we need the operation that allows a question to consult the records that remain.

## 2. A small attention operation you can calculate

### First combine the information; then explain how the weights arise

Attention produces a task-dependent weighted mixture. For the entrance question, the current computation should give more weight to an entrance-related record than to a lunch announcement. The comparison part of a record is its **key**; the information it contributes is its **value**. The request is the **query**. Separate those roles: a record can be easy to match while carrying a small value, or hard to match while carrying a large one.

For a calculation you can follow exactly, replace word features with three scalar readings: 2, 4 and 8. Suppose the query assigns their keys relative supports of 2, 1 and 3. There are six units of support in total. The first reading receives 2/6 of the mixture, the second 1/6 and the third 3/6. Multiply each value by its share and add: (2×2 + 1×4 + 3×8)/6 = 32/6, about 5.333. Larger support means more influence on this read, not a claim that the source is more factually reliable.

**Inline figure: six shares form one weighted read.** Represent supports 2,1,3 as six equal-area units grouped by donor A,B,C, paired with values 2,4,8 and exact contributions. The output is 32/6. The units encode relative support; they are not six extra data points.

Read the figure from the shares to the contributions. The third reading owns half the mixture, so it contributes 4 to the final answer. The first two together contribute 4/3. This is the whole weighted-read mechanism. Neural attention adds a learned way to choose those shares.

### Scores turn a question into a mixture

An attention query is a vector expressing what information is requested. A learned **linear projection** multiplies a representation by a weight matrix to produce its query, key or value vector. The query and each key receive a comparison score. A **dot product** computes that score by multiplying matching coordinates and adding them; a larger result favors that key. Softmax then turns all the allowed scores into nonnegative weights summing to one.

Here is the same recipe in notation, one step at a time: sⱼ is a comparison score; αⱼ is that record's share of the read; o is the resulting mixture. For one query q and available pairs (kⱼ,vⱼ):

\[
s_j=\frac{q^\top k_j}{\sqrt{d_k}}+b_j,
\qquad
\alpha_j=\frac{\exp(s_j)}{\sum_{r\in\mathcal A}\exp(s_r)},
\qquad
o=\sum_{j\in\mathcal A}\alpha_jv_j.
\]

Here dₖ counts query/key coordinates; bⱼ optionally changes a score using position; and 𝒜 contains the positions the query may use. The square-root factor controls score scale as the number of coordinates changes. For our scalar example dₖ=1, so the factor is 1. The sum sign means “add this term for every allowed record.”

We can now generate exactly the supports in the picture. Let q=1 and use keys ln 2, 0 and ln 3 with no positional bias. The natural logarithm is the inverse of exponentiation, so exp(ln 2)=2, exp(0)=1 and exp(ln 3)=3. Softmax normalizes these supports, giving the weighted read we already calculated:

\[
o=\frac{2(2)+1(4)+3(8)}{2+1+3}=\frac{16}{3}.
\]

### A forbidden record gets no share at all

Suppose the third reading has not happened yet. A prediction made now may use 2 and 4, but cannot consult the future 8. We remove the third record before assigning shares. The remaining supports total 3; the weights become 2/3 and 1/3, and the answer is 8/3. Setting the third *value* to zero while keeping its support in the denominator would instead give 4/3. That incorrect answer shows why a **mask** must exclude a record from the entire read, not merely erase its visible value.

**Inline figure: a query, three key/value pairs and a blocked future edge.** Show the score-to-weight calculation at the edges and the two actual weighted contributions entering the output. Beside it, show the denominator changing from 6 to 3. This is more informative than a grid of unlabeled attention colors.

For many queries, do one such read for each request. Two questions about five records require ten comparison scores and produce two answers. In a matrix, rows simply keep those questions separate. Arrange queries as Q with shape L×dₖ, keys as K with shape N×dₖ and values as V with shape N×dᵥ. QKᵀ contains L×N scores; weighting V produces L×dᵥ outputs. **The number of output positions comes from the queries.** The shape diagram below is bookkeeping for the individual reads you just followed, not a new operation.

For next-token prediction, the representation at position i may use inputs through i and predict token i+1. A **causal mask** permits j≤i. For a complete-trajectory classification task, all measured points are already available before the classification is requested, so a noncausal read is appropriate. The mask follows the task's information boundary.

## 3. Transformer-XL: continue across segment boundaries

### 3.1 Why separate chunks forget

Processing a long document in chunks is attractive: each chunk fits a manageable amount of work into memory. But imagine a chunk ending “Use the backup entrance on the east…” and the next beginning “…side.” Starting the second chunk with no earlier information has cut a connected thought in two. This is **context fragmentation**. It can harm understanding even when the needed words are close together.

The repair is to carry some already-computed records into the next chunk. Transformer-XL calls a chunk a **segment**. When reading a new segment, its current positions ask questions; earlier stored representations join the current representations as possible donors. A query near the start can now consult the previous segment. The [authors' illustrated explanation](https://research.google/blog/transformer-xl-unleashing-the-potential-of-attention-models/) makes this segment boundary the central problem; the exact layer construction comes after our small trace below.

That earlier memory is an additional set of allowed sources. It does not permit a query to look ahead within the new segment. We still apply the causal rule from section 2.

### 3.2 Follow a complete cache example

Let the five records have values [6,1,8,2,0]. Record 0 is a strong match for our question: give it relative support 9, compared with support 1 for every other record. With scalar query 1, keys [ln 9,0,0,0,0] produce exactly those supports. This constructed example isolates access to a useful record; it is not a trained text model.

Process segments [0,1], [2,3], then [4]. Follow a cache of length 2 through the boundary. After the first segment it holds records 0 and 1. After the second, keeping only the newest two replaces them with records 2 and 3. When query 4 arrives, record 0 is gone even though it was read earlier. Cache length controls what can still be consulted directly.

At the final query, compare three memory lengths:

| Retained positions before the final segment | Legal values | Output |
| --- | --- | --- |
| None, M=0 | Current value 0 | 0 |
| Positions 2 and 3, M=2 | 8, 2, 0 with equal weights | 10/3 |
| Positions 0–3, M=4 | 6, 1, 8, 2, 0 with weights proportional to 9,1,1,1,1 | 65/13=5 |

The output is a soft combination, not a lookup returning exactly 6. The larger cache makes the strong matching key available again. With M=4, every query in this five-position example sees the same legal keys as the full causal calculation, so their outputs agree. This equality holds here because the projections and positions are fixed and every required key remains available.

The live record strip starts at the middle row of that comparison: memory length 2 and query position 4. Increase retained memory to 4. R0 becomes available, and the answer moves from 10/3 to 5. Notice that its stored value stayed 6; the change came from access and normalization. Next inspect query position 1, edit a later record's value, and observe the unchanged answer. These two edits separate missing history from correctly excluded future information.

**Investigation 1 — build the retained context.** Current cache records, weights, denominator and output are visible; edit the records, segment size, retained memory, query and distance bias. Show exclusions as absence from the read. Keep the completed revision-3 live implementation and exact-value controls.

### 3.3 Put the cache into a layered model

A neural **layer** turns input representations into new representations. A stack of layers repeats that transformation. The records entering one layer may therefore already contain information gathered by earlier layers. In Transformer-XL, an upper-layer read uses the lower layer's current representations together with retained lower-layer representations from the preceding segment.

In the diagram, first follow the solid forward path across the segment boundary. It answers “which values can the current computation use?” Then look at the stopped training path. Training can use those old values while treating their earlier creation as outside the current gradient calculation. This operation is called **detaching** or `stopgrad`; it changes credit assignment, not the value stored in memory. Section 8 calculates the difference explicitly.

**Inline figure: a two-dimensional segment/layer diagram.** Lay segments horizontally and layers vertically. Draw an old lower-layer state feeding the next segment's upper-layer attention, alongside the current lower-layer states. Draw the forward edge solid and the backward-gradient edge stopped at the memory boundary. This exposes the layer shift that a simple circular “memory” arrow would hide.

For layer ℓ in segment s, let Hₛ⁽ℓ⁻¹⁾ contain L current rows of width d and let Mₛ⁽ℓ⁻¹⁾ contain M retained rows of that same width. The following expression is just “put old and current donor rows together; ask questions only for the current rows”:

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

The semicolon joins rows, producing M+L donors. There are still L queries, so attention produces L current outputs. After the segment, retain the latest M appropriate hidden states for the next segment. The W matrices are the learned projections that create the query, key and value roles introduced in section 2.

The [Transformer-XL paper, sections 3.2–3.3](https://arxiv.org/pdf/1901.02860) defines this layerwise recurrence and its relative-position construction. Detached memory still affects the current prediction, and parameters projecting that memory into current keys and values still receive gradients; only the history that created the stored tensor is detached.

A real multilayer Transformer-XL can pass older information through representations that were themselves contextualized. Its indirect dependency paths differ from a single layer's list of visible positions. With the paper's one-segment memory construction, crossing a segment boundary also moves through a layer, giving a depth-dependent receptive field. A finite network and cache do not imply unlimited exact recall.

### 3.4 A reused position needs a coherent address

Imagine numbering positions within every segment as 0,1,2,3. A cached token from local position 1 and a new token at local position 1 now carry the same absolute-position label. Reusing those labels creates ambiguity about temporal relationships.

A relative position asks, “How far is this key from the query?” A query at global position 5 sees keys at positions 1 and 5 at distances 4 and 0. Shifting all three global positions by 100 preserves those distances. Resetting segment-local counters does not.

**Inline figure: repeated local labels versus relative-distance rulers.** Show the two “position 1” tokens in different segments, then draw distances from the same query using global order. The invariant is the distance, not the printed segment counter.

Keeping records creates a second engineering problem: how to tell the old “position 1” from the new “position 1.” The distance ruler above fixes the meaning of “four positions ago” across segment boundaries. For the first pass, this is the important idea. The exact Transformer-XL score below separates content matching from distance preferences.

An attention **head** is one learned set of query/key/value projections; multiple heads can ask different questions. Transformer-XL uses separate content and relative-position projections. One head's score can be written

\[
s_{ij}=q_i^\top k_j+q_i^\top r_{i-j}
+u^\top k_j+v^\top r_{i-j},
\]

with scaling applied according to the head implementation. The first term matches content; the second makes preferred distance depend on the current query; the third is a learned global content preference; the fourth is a learned global distance preference. Here r is the projected relative-position vector, and u and v are learned vectors. These v and r symbols are score parameters; vⱼ in section 2 denoted an attention value.

Our downloadable scalar calculation can instead add −β(i−j) to the score. It is a transparent recency-bias exercise, **not Transformer-XL's full positional formula**. It lets you observe how recency can compete with a content match. The later positional-encoding lesson explains sinusoidal features, RoPE and other constructions separately.

## 4. Griffin: keep a recurrent summary and consult nearby detail

### 4.1 Why ignoring a new input can still erase an old one

A cache grows by keeping more records. A recurrent model has another option: keep updating the same small workspace. Imagine tracking whether an important event occurred while many uninformative events pass by. A state value of .6 might represent the event's current influence. If each update retains only 80% of the previous state, the next value is .48—even when the new input is zero. Silence does not automatically mean “leave memory alone.”

Separate the two contributions to an update: **keep some of the old state, then add some of the new signal**. Write the old state as hₜ₋₁, the incoming number as xₜ and the new state as hₜ. A simple update is

\[
h_t=a h_{t-1}+b x_t.
\]

Here a controls retention and b controls injection. Start from zero, set a=.8 and b=.6, and send one input of 1 followed by five zeros. The states are .6, .48, .384, .3072, .24576 and .196608. The first event is still influential, but repeated multiplication by .8 has weakened it. Unlike the cache, there is no separate old record to reopen.

Griffin uses learned controls called **gates** to change this behavior. An input gate controls the new signal; a recurrence gate controls how much old state survives. To distinguish them, keep the current input at zero. Closing its input gate does nothing: the new signal was already zero. Holding the old state requires changing retention instead.

**Inline figure: three ways to pass a silent step.** Begin every row with old state .6 and input zero. Show ordinary retention .8 giving .48; closed input gate with retention .8 also giving .48; the exact hold limit with retention 1 giving .6. All bars share the same zero-to-.6 scale. Label the exact hold as a mathematical endpoint, not a finite-sigmoid result.

In the figure, compare the first two rows before the third. The first two have different input-gate settings but identical output. Their shared .12 loss happened on the old-state path. That is the practical reason for having a separate recurrence gate.

### 4.2 How the two gates set the update

Griffin's **real-gated linear recurrent unit**, or **RG-LRU**, makes those gates depend on the current input. A sigmoid turns a learned score into a number between 0 and 1. A learned base a between 0 and 1 sets a coordinate's decay behavior; the recurrence gate rₜ changes its effective retention to aₜ=a^(c·rₜ). The scale c is 8 in the paper. With a=.8 and rₜ=1/8, retention is .8. As rₜ approaches zero, retention approaches 1.

The incoming multiplier is √(1−aₜ²). Thus keeping more old state also reduces the scale of the injected signal. With retention .8, that multiplier is √(.36)=.6. With retention approaching 1, it approaches zero. These are coupled choices, not two arbitrary coefficients that can both be turned up without consequence.

We can now write the full vector update. A vector contains several state coordinates; the symbol ⊙ means multiply matching coordinates separately. σ is the sigmoid, W and b are learned gate weights and offsets, and iₜ is the input gate:

\[
r_t=\sigma(W_r x_t+b_r),\qquad
i_t=\sigma(W_i x_t+b_i),\qquad
a_t=a^{c r_t},
\]
\[
h_t=a_t\odot h_{t-1}
+\sqrt{1-a_t^2}\odot(i_t\odot x_t).
\]

Read the last line as two paths meeting: aₜ⊙hₜ₋₁ retains old state; √(1−aₜ²)⊙(iₜ⊙xₜ) admits new signal. The input gate iₜ controls only the signal inside that second path. The recurrence gate rₜ changes both retention and its normalization. This is the update in [Griffin, section 2.4](https://arxiv.org/pdf/2402.19427).

With a=.8 and rₜ=1/8, the effective decay is .8. With rₜ near zero, aₜ approaches 1 and the injection multiplier approaches zero. The unit can preserve a state across an uninformative interval instead of repeatedly replacing it. In the exact limiting case rₜ=0, it holds the state unchanged. Finite sigmoid logits approach that endpoint but do not produce it exactly.

**Inline figure: retention and injection as separate contributions.** For each timestep show a retained-state bar, an incoming-signal bar and their sum. Align these with the actual input strip. A gate shown merely as an open or closed door would conceal the square-root normalization and the two different gates.

Follow the two curves above through the zero-input interval. Both start at .6. Their later difference comes entirely from how much old state survives; no later event adds a signal. The plotted values come from the same recurrence as the live investigation below.

**Why the square root?** It balances the scale of old and new contributions under a useful simplifying assumption. If hₜ₋₁ and the gated input each have variance 1 and are uncorrelated, the combined variance is aₜ²+(1−aₜ²)=1 when the coefficient is fixed. Learned, input-dependent signals need not satisfy those assumptions. This calculation explains the design rather than asserting every hidden coordinate has variance exactly one.

For the six-step impulse above, use r₁=1/8 and r₂…r₆=.001. The final state is about .594668, close to the first state's .6. If we only set the later input gates to zero while leaving rₜ=1/8, the final state remains .196608. The inputs were already zero; stopping input admission did not stop the recurrence from decaying.

Try the live sequence in this order: compare the initial ordinary decay with “Near-hold after first event.” Then reset the recurrence before selecting “Close later input gates only.” The first edit changes the final state; the second does not. Next add a later nonzero event and adjust its input gate; now that gate has an incoming signal to control. The retained and injected terms show exactly where each change enters.

**Investigation 2 — preserve or replace a memory.** Retain the live editable event sequence, gate controls, signed contributions and immediate trace. Preserve zero-state/zero-input and input-gate-only null checks.

### 4.3 Give the summary a nearby detailed view

The recurrence solves one problem—carrying an influence forward in bounded state—but its summary cannot expose every earlier record separately. Griffin combines it with **local sliding-window attention**: a read that keeps nearby positions individually accessible. Recent exact detail and older compressed influence can therefore take different routes.

The published Griffin construction repeats two recurrent blocks followed by one local-attention block; the usual attention window in that study is 1,024 tokens. In the next figure, follow the state through time separately from the row of recent key/value records. An old event can affect the state without still occupying a directly readable attention slot.

**Inline figure: recurrent, recurrent, local-attention blocks on a shared sequence.** Show a fixed-width state path traversing time and a bounded window of separately retained key/value pairs. These are different memory allocations. Do not draw distant input tokens as though the local-attention block could directly read them.

To connect this memory picture to the implementation, open one recurrent block. It projects each input into two branches. One branch first mixes a few recent timesteps with a **causal depthwise temporal convolution**: each channel has its own short filter, which uses the present and past rather than the future. RG-LRU then carries that filtered signal through time. The other branch supplies a nonlinear gate. Multiplying the branches controls which processed features pass on; a final projection restores the model width.

Residual connections add a block's update to its incoming representation, normalization controls feature scale, and a gated feed-forward block transforms features at each position. These supporting pieces do not change which earlier records are directly stored. The [official RecurrentGemma walkthrough](https://developers.googleblog.com/gemma-explained-recurrentgemma-architecture/) illustrates how the branches and dimensions fit together. Its released model settings and Griffin's paper settings should be read separately; “local attention” does not imply one universal window size.

A local-attention cache is bounded once its window is full. A recurrent block also needs its state and the short convolution history. Dense projection and feed-forward weights remain part of the model. “Memory independent of stream length” refers to the bounded recurrent/cache state under a fixed architecture, not to the entire training computation or arbitrary batch size.

The next lesson studies state-space models in depth. RG-LRU is not presented in its source as a discretization of a continuous-time system, and this chapter's gated recurrence should not be silently substituted for a Mamba update. Similar diagrams can conceal different equations.

## 5. Perceiver: put deep computation in a smaller workspace

### 5.1 Why one average can lose the answer

Sometimes the whole input is already available: an image, a recorded sound or a hand's measured path. There is no need to predict before its later elements arrive. The problem is the amount of computation needed to reason about the full collection repeatedly.

A tempting shortcut is to replace a path with its average coordinate. But consider five evenly spaced horizontal positions from 0 to 1. In path A, all heights are .5. In path B, the heights are [.5,.75,0,.75,.5]. Both average to (.5,.5), even though one is straight and the other changes direction. If a classifier receives only that average, the lost shape is unavailable to it.

**Inline figure: two paths collide at one average.** Draw the five points of each constructed path with shared equal x/y scales, a numbered traversal and distinct start/end marks. Show their identical mean (.5,.5) and the summation that produces it. These are constructed illustrations, not measured dataset rows.

Look at the path before the center mark. Their centers agree; their geometries do not. This is the reason to learn several selective summaries rather than automatically average everything. It does not prove a particular learned model will succeed—we will test that question on real trajectories in section 7.

### 5.2 Learn a few questions instead of processing every input pair

Perceiver makes a smaller working array of learned vectors called **latents**. Think of each starting latent as a trainable request for information. Its initial numbers are model parameters shared across examples. After it reads an input, its updated numbers depend on that particular example. The model learns useful requests through the training task, not through hand-written names such as “find a curve.”

Each latent asks a query of the original input records, using the attention operation from section 2. If there are N latent queries and T input records, there are N×T comparison scores and N resulting vectors. This is **cross-attention** because queries and donor records come from different arrays. The resulting latents can then attend to one another—**self-attention** within that N-row workspace—using N×N comparisons.

The next figure shows two different activities. Follow a read from the input bank into the workspace; then follow processing among the workspace vectors. The large input bank stays available. After working with the first read, updated latents can ask it another, better-informed question.

**Inline figure: input array → asymmetric read → latent processing → another read.** Keep the large input array in place across both reads. Label a cross-attention score array N×T and a latent self-attention array N×N. This is a change in the number of query positions, not a claim to approximate the original T×T softmax with the same outputs.

Thus Perceiver has two kinds of depth: processing the current latents and rereading the original input. It often chooses N much smaller than T, so most deep processing happens in the smaller array. Sharing weights across repeated blocks reduces parameter duplication while still doing another computation with updated data. These choices are described in [Perceiver, section 3.1](https://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf). They change the computation performed; they do not promise to reproduce the output of full input-to-input attention.

### 5.3 Calculate two different summaries

Use three records tagged earlier, middle and later, with numerical position keys [−1,0,1] and values [2,6,10]. Give the first read a preference for earlier positions: relative supports 4:2:1. Give the second the opposite preference, 1:2:4. Each has seven shares in total. Their weighted summaries are

\[
z_1=(4\cdot2+2\cdot6+1\cdot10)/7=30/7,
\quad
z_2=(1\cdot2+2\cdot6+4\cdot10)/7=54/7.
\]

The first workspace vector summarizes earlier values more strongly; the second summarizes later values more strongly. Their joint representation contains information that a single uniform mean would lose. A learned model can use richer keys and queries; this hand example isolates the read operation.

Those supports come from ordinary attention, not a new mixing rule. A scalar query −ln 2 multiplied by keys [−1,0,1] gives scores [ln 2,0,−ln 2]. Exponentiation gives [2,1,1/2], whose normalized weights are [4/7,2/7,1/7]. Query +ln 2 reverses the preference. Query zero makes all scores equal, giving the uniform mean. The live controls below let you move continuously between these behaviors.

A uniform query gives the mean 6 for both [2,6,10] and [0,6,12]. If a later computation receives only that single mean, it cannot distinguish those two inputs. Once the inputs have collapsed to an identical representation, any deterministic downstream function must give them the same output. This is a concrete bottleneck failure, not a claim that every one-latent nonlinear model is exactly a mean.

Start the live workspace at its default opposite queries and inspect which end of the input each read emphasizes. Select “Use one uniform query” and compare the two displayed input fixtures. Their identical summaries expose the lost distinction. “Reset latents” restores the opposite queries; the arrays now give different representations. Adding workspace capacity is useful when it preserves information the task needs; the count alone is not a guarantee.

**Investigation 3 — choose what the workspace can distinguish.** Preserve editable queries, values and attached position tags, immediate weights, fixtures, paired permutation and reassignment controls.

### 5.4 Order must travel with the record

Suppose the path is stored in a file and you rearrange its rows. If each point keeps its original timestamp, you have changed storage order, not the path. If you give those points new timestamps, you have changed which movement happened when. A model should distinguish those edits.

If keys and values are permuted together, cross-attention with fixed latent queries produces the same result. The score columns and their corresponding values move together, so the weighted sum is unchanged. This is desirable for a storage reordering, but it means that order must be represented if it matters to the task.

Perceiver supplies position features alongside input features, commonly Fourier features for spatial or temporal coordinates. A frame with its correct timestamp may be moved to another storage row without changing its meaning. Assigning that frame another timestamp changes the input. In the trajectory program, each point carries a normalized position from −1 to 1.

A Fourier feature expands a coordinate x into sine/cosine measurements at selected frequencies: for example, [x, sin(πx), cos(πx)]. At x=0 this is [0,0,1]; at x=1/2 it is [1/2,1,0]. Adding more frequencies gives the learned projections several spatial scales to combine. The original coordinate helps distinguish positions that a periodic component alone would identify. For d coordinate dimensions and K frequency pairs per coordinate, this concatenation has d(2K+1) entries. It supplies location information; it does not change the paired-permutation argument. Our small classifier uses the raw ordinal tag alone so its input representation stays transparent.

**Inline figure: paired permutation versus reassigned positions.** Connect each value to its position tag. One panel moves complete tagged records; the other keeps the tags fixed and moves only values. This makes an invariance that often sounds abstract visible as a concrete difference in what was edited.

### 5.5 Ask for outputs at the places you need them

Our path task needs one class label. Other tasks need an answer at every location—for example, the motion of each pixel between two images. The number of answers need not equal the size of the internal workspace. **Perceiver IO** adds an attention read on the output side: each requested output supplies a query, and the processed latent array supplies the keys and values. Original Perceiver instead aggregated its latents for tasks such as classification.

An output query might specify a pixel coordinate, a language position or a desired modality. A coordinate alone does not determine the answer; the query asks the learned decoder to extract the relevant information from the latents. In the next diagram, count the requests, then count the returned rows. Four requests produce four answer vectors even though there are only two latent donors.

If there are O output queries, the output has O rows. For example, four output-coordinate queries can request four motion vectors from the same two latent vectors. The output size is independent of the number of input rows and of the chosen number of latents. What can be accurately reconstructed still depends on the information those latents retained. [Perceiver IO, sections 3.1–3.2](https://arxiv.org/pdf/2107.14795).

The interesting connection is that attention becomes an **interface between array sizes**, not only a way to relate words. Input queries move information into a working representation; output queries specify what information to read from it. The later cross-attention-architectures lesson uses this perspective to connect modalities and pretrained systems.

## 6. Which Perceiver is allowed to generate left to right?

The movement classifier sees a finished recording before answering. A model generating a sentence has a different job: when predicting the next word, that word must still be unknown to it. This is why we cannot simply reuse an unrestricted full-input latent array for every prediction.

Original Perceiver uses noncausal attention and can read the whole observed input for classification. If an early language-model output consults a latent that already encoded its future target, the answer has leaked into the computation. Blocking a direct connection to the target does not help if an indirect route remains open. The mask must protect every path to the answer.

**Perceiver AR** aligns a smaller number of latents with selected final input positions. A latent corresponding to input position i may cross-attend only to positions j≤i; it predicts token i+1. Latent self-attention also uses the causal order. Both stages must preserve the information boundary.

For input positions 0,1,2,3,4, take latents aligned with positions 3 and 4. The first may read inputs 0–3 and predict token 4. The second may read inputs 0–4 and predict token 5. During latent self-attention, the first latent cannot read the second: that second latent has already seen input 4, which is the first latent's target.

In the first table below, check each latent against the inputs it is allowed to know. Then inspect the second table: the earlier latent must also be prevented from consulting the later one. Trace the displayed detour from input 4 to see why two individually plausible attention stages need coordinated masks.

**Inline figure: the two masks and a leaking detour.** Draw the legal input-to-latent edges and legal latent-to-latent edges. Then highlight the forbidden detour input 4 → latent 4 → latent 3. The first mask alone leaves that detour open if latent self-attention is unrestricted.

The [authors' ICML presentation slides, pages 3–10](https://icml.cc/media/icml-2022/Slides/17886.pdf) build this input/query/target alignment incrementally. The [Perceiver AR paper](https://proceedings.mlr.press/v162/hawthorne22a/hawthorne22a.pdf) describes the architecture and training/inference choices. A latent bottleneck does not make every Perceiver variant a streaming recurrent model: retaining or rereading a long input and updating a fixed recurrent state are different execution contracts.

Return to the document question. Transformer-XL may retain the entrance's position vector if it remains in memory, or pass its influence indirectly through later states. Griffin may preserve it in recurrent state while consulting recent text explicitly. A Perceiver read can consult it directly while the input remains available, provided the learned queries and latent capacity preserve what the task needs. Those are testable paths, not guarantees that every trained model will answer correctly.

## 7. Run the mechanisms, then try real movement data

### 7.1 The small calculations

You have now calculated a weighted read, watched records leave a cache, separated retention from injection, and built different latent summaries. The first program turns those same small examples into reusable functions. It is the shortest route from the diagrams to an implementation you can inspect.

Download [sequence_mechanisms.py](sequence_mechanisms.py). This complete NumPy program implements attention, a segmented memory trace, the scalar gated recurrence and the latent calculations. It also checks selected **invariances**: changes to representation, such as jointly reordering keys and values, that should leave the result unchanged.

Use Python 3.12 or later in your own environment:

```text
python -m venv .venv
python -m pip install numpy
python sequence_mechanisms.py
```

Activate that environment before the `python -m pip` command, or use its Python executable explicitly. On Windows the executable is `.venv\Scripts\python.exe`; on macOS/Linux it is `.venv/bin/python`. The author run used Python 3.12.14 and NumPy 2.3.5. The script writes `checked-results.json` beside itself.

Read the following code as five actions: compare every query with the keys; exclude forbidden records; exponentiate stable scores; divide by the total support; mix the values. The `@` operator performs matrix multiplication, and `.T` swaps a matrix's rows and columns. Each row stays one question throughout:

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

These are the hand calculations made executable: 4.285714 and 7.714286 are 30/7 and 54/7 from section 5. Start by changing one value in the latent example. The weights remain fixed because its keys and queries did not change, so the output change equals that value's change multiplied by its attention weight. This gives you a small independent check before modifying the attention function itself.

### 7.2 A complete trainable latent classifier

Now ask a real question: **Can a compact representation of a hand's path identify which of fifteen recorded movement classes it belongs to?** The UCI Libras Movement dataset provides 360 hand trajectories extracted from videos: forty-five ordered two-dimensional positions per trajectory, followed by a class label. Its classes include circles, arcs, straight lines, waves and zigzags. These coordinates describe one hand's movement; they are not a complete representation of sign-language meaning. [UCI collection description and CC BY 4.0 license](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement).

A point is a measured hand centroid at a selected video frame. The collectors normalized the videos into forty-five sampled positions. We therefore know the point order, but do not infer exact time intervals, hand speed in meters per second or physical distances from these unit-space coordinates. The original metadata describes four performers and two sessions; individual rows do not identify performer or session.

To make the experiment meaningful, give each trajectory one job. **Fit** rows change the model's parameters. **Validation** rows help choose when to stop fitting. **Test** rows assess the resulting choice after those decisions. A copy of the same path must not appear on both sides of that assessment; otherwise apparent recognition could partly be repetition.

Before splitting, we found thirty duplicate copies of trajectories, all with matching labels. Keeping the first occurrence of each exact coordinate sequence leaves 330 distinct trajectories. Within each class, a fixed NumPy seed of 73 determines a permutation. Reserve the last four trajectories for test; use the first two-thirds, rounded down, for fitting and the rest for validation. This gives **220 fit, 50 validation and 60 test trajectories**, with no exact coordinate sequence crossing roles. Because rows lack performer/session identifiers, this experiment cannot establish performance on a new performer or recording session.

These forty-five-point sequences are a practical CPU exercise in latent sequence processing. They let us inspect learned reads and compare ways of preserving shape. They do not measure long-context language-model performance.

Download [movement_libras.data](movement_libras.data) and [latent_trajectory_classifier.py](latent_trajectory_classifier.py) into one directory. The [data record](data-provenance.md) supplies attribution, source-row roles, duplicate handling and exact transformations. The program runs offline once dependencies are installed:

```text
python -m pip install numpy torch scikit-learn
python latent_trajectory_classifier.py
```

The tested snapshot is PyTorch 2.14.0+cpu, NumPy 2.3.5 and scikit-learn 1.9.1. The program uses two CPU threads and writes `trajectory-results.json` and `small-fits.npz`. The latter stores the small learned arrays and selected read weights.

**First give simpler methods a fair attempt.** A baseline is a comparison that tells us what the new machinery adds. Transform each coordinate using the fixed rule 2x−1, then average the forty-five x coordinates and forty-five y coordinates. A logistic classifier receives just those two center coordinates. This is the same information loss shown by the straight and zigzag paths in section 5.

A second logistic classifier receives all ninety coordinates in their original order. It can distinguish shapes that share a center, although its decision function is linear in the supplied coordinates. Both baselines use C=1, fit only the fit rows and use no learned preprocessing across roles. The [linear and logistic regression lesson](/learn/path/full-curriculum/linear-logistic-regression?module=classical-ml) develops these classifiers; here the comparison asks whether preserving shape matters more than adding a complicated model.

**Then learn which information to gather.** The latent model appends an ordinal tag running from −1 to 1 to each coordinate pair. Each of the forty-five rows now has three numbers: x, y and order. A learned projection turns each row into 24 features. N learned latent vectors read those features, interact with each other and pass through a feed-forward block. The model repeats that read/process step once with shared weights, averages the final latents and produces fifteen **logits**—unrestricted numerical scores, one for each class. Softmax converts them into class probabilities. We compare N=1 and N=4 under the same protocol.

Read the diagram from measured geometry to model features. The ordinal tag tells the model where a point belongs in the movement; it is neither another spatial coordinate nor a measurement of elapsed seconds. Count rows at each transition: forty-five input rows can feed one or four latents while the answer still has fifteen class scores.

**Inline figure: the actual trajectory classifier.** A numbered two-dimensional path sits beside its forty-five (x,y,position) rows. These project to width 24, enter two latent read/process rounds and produce fifteen logits. Distinguish coordinate values, ordinal position tags and an optional validity mask. Masked padding is a storage device, not another measured point.

The central loop follows that diagram. `cross` reads input records; `self_attention` lets the working vectors exchange information; `feedforward` transforms each working vector's features. Each `latent + update` is a residual update, preserving a path for the current representation while adding new information:

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

The complete file defines the projections, attention, normalization, feed-forward block, duplicate handling, data roles, training, metrics and saving. First trace `inputs`: it stays available across both rounds. Then trace `latent`: it changes after every update. The second round reuses the same learned projection parameters, but its updated queries generally produce different attention weights over the input records. This is a small Perceiver-style classifier with inspectable dimensions, rather than a reproduction of the research model's scale.

**How do the starting queries learn what to ask?** Training penalizes assigning low probability to the recorded class. For a trajectory with true class c, cross-entropy is −ln p(c). Its derivative with respect to logit k is pₖ−1[k=c], where the indicator is 1 for the correct class and 0 otherwise. At uniform predictions, the correct logit's derivative is −14/15 and each other derivative is +1/15. Subtracting a small multiple of this gradient raises the correct score relative to the others.

Backpropagation carries that signal through the classifier, latent processing and attention reads to the starting queries and projections. No separate labels say where each latent should look. The classification objective supplies the training signal. Pass logits, not precomputed probabilities, into PyTorch's [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html), which performs the required stable log-softmax internally.

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

The error plot makes two comparisons visible: changing the representation from center to ordered coordinates, and changing the latent model's seed or capacity. Read lower marks as fewer mistakes, using the printed denominators to distinguish validation from test. The disagreement among runs is part of the result.

The live workbench below uses a **frozen** fitted model: changing a point recomputes its output immediately, but does not retrain it. Begin with a full validation path and inspect its shape. Select a point, move it, and compare the path, class probabilities and selected latent's read weights. A large read weight means the point contributes strongly to that particular read; it is not proof that the point caused the final class decision on its own.

**Investigation 4 — inspect a learned read.** Retain the complete frozen-model workbench, four fitted models, validation selection, point dragging and numeric controls, validity masks, paired permutations, reassignment, live class probabilities, read weights, mean baseline and reset.

Look for two outcomes: an edit that changes the winning class, and a smaller edit that changes probabilities while the winner stays the same. The latter is still a real response; the winning label hides all changes short of crossing a decision boundary. Compare the mean-coordinate baseline on the same retained points to see whether moving the center alone helps explain the response. An edited hypothetical path has no automatically known new class label, so a changed prediction is evidence of model sensitivity, not proof of correct recognition.

Two null checks passed for all four fitted models. Permuting complete coordinate-plus-position records preserves logits within floating-point tolerance. Appending five points filled with 1,000 but masked out also preserves logits. If masked points alter the output, a mask or an earlier operation has admitted information that was supposed to be absent.

### 7.3 Move from explicit operations to maintained library operators

Once the small calculations are clear, a maintained operator can provide the same operation more efficiently. The useful question is whether it receives the same tensors, masks and state—not whether its API name resembles the architecture. The bridge program checks matching inputs, outputs and gradients before you substitute operators in a larger system.

Keep the three files together. [sequence_mechanisms.py](sequence_mechanisms.py) implements transparent arithmetic. [latent_trajectory_classifier.py](latent_trajectory_classifier.py) adds trainable projections, a learning loop and evaluation. [memory_library_bridge.py](memory_library_bridge.py) reuses the classifier's `Attention` class and compares its operations with library counterparts. This deliberate import avoids maintaining a second subtly different attention implementation.

**Segment memory:** `check_segmented_attention` builds a real K/V cache, appends the current segment, constructs a causal mask using absolute retained positions, then keeps the last four entries. Its explicit stable score→softmax→weighted-sum path is compared to `F.scaled_dot_product_attention` on exactly the same projected values, including a final short segment. In SDPA's Boolean mask, `True` means allowed. Calling `is_causal=True` on a nonsquare query/cache matrix without considering alignment can select the wrong history. Detaching cache tensors cuts gradient history; it does not reset their values or make a stream's cache suitable for another stream.

This is the segment-memory mechanism described in §3. It is not a complete Transformer-XL checkpoint: the paper also specifies layerwise hidden-state recurrence, relative-position scoring, normalization and a language-model head. Our illustrative distance bias and already-projected one-layer K/V cache must keep those labels. The later [Self-Attention lesson](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals) owns the full projected multihead construction. It is a deeper follow-on reference; the local small read supplies the operations needed here regardless of that lesson's implementation status.

**Recurrent memory:** `rglru_reference` goes beyond the earlier fixed scalar gates. It calculates learned block-diagonal input and retention gates, stable `-expm1(2*log_decay)` injection scale, per-example reset positions and each updated vector state. No recurrent-cell API hides that computation. The optional `check_griffin_component` copies actual DeepMind `RGLRU` parameters, compares outputs and the last cache, continues in two chunks and checks gradients. The native implementation clips an extreme square-root derivative for training stability; the comparison explicitly uses moderate float32 decay outside that clipped regime. This is the RG-LRU component, not an entire Griffin language model or its short temporal convolution/local-attention schedule. The [official recurrent component source](https://raw.githubusercontent.com/google-deepmind/recurrentgemma/main/recurrentgemma/torch/layers.py), inspected 22 September 2026, owns the exact package convention.

**Latent workspace:** `check_latent_read` reuses our existing learned `Attention` class, obtains its projected query/key/value tensors and passes those exact tensors to SDPA. It then applies the same output projection and compares outputs plus gradients for source, latent query and every parameter. The complete classifier alternates this read with latent self-attention and its feed-forward block. A generic full-input model would erase the architectural question; replacing only the matching read preserves it.

Run `python memory_library_bridge.py` with PyTorch, NumPy and scikit-learn. Adding `--griffin` requires a compatible `recurrentgemma` installation with its Torch dependencies. The ordinary Torch comparisons are executed in the implementation checks. The optional specialist program is supplied as executable instructional content; optional package parity is not claimed executed. Record the resolved package version when running it.

The explicit attention reference materializes query×visible-key scores; the normal SDPA backend can avoid retaining that matrix under supported conditions. A bounded segment of length Q with M retained positions uses O(Q(Q+M)) score cells per head in the reference. RG-LRU streaming retains one width-sized state per example, while training retains its needed history. Latent reads use L×S score cells for L latents and S source positions. These are separate computational contracts, so there is no one “linear-memory” claim covering all three.

**Change the stream.** Set segment length 4, keep only 2 previous positions, and process 11 inputs. Then place a recurrent reset at position 6 while preserving every input. Compare the supplied reference and package paths again.

<details><summary>Hint</summary>The last segment has 3 queries. Cache truncation discards earlier keys; a recurrent reset clears prior state for exactly the selected example.</details>

<details><summary>Solution and success criteria</summary>Build legal key positions from the retained absolute indices plus the current segment, mask future indices, and trim the cache only after obtaining that segment's outputs. The first query of the last segment sees indices 6, 7, 8; its later queries can also see 9 and 10 as they arrive. In RG-LRU, set `segment_pos[:,6]=0` and restart its within-document counter; the reset input uses the special fresh-document injection and cannot depend on the earlier cache. Verify unaffected earlier outputs and a changed post-reset state. This tests state ownership rather than memorizing model names.</details>

## 8. Deeper: gradients, budgets and a useful evaluation plan

This section is a deeper route. It makes the architectures' costs and training boundaries precise after their forward mechanisms are familiar.

### 8.1 Forward memory and gradient memory

A detached cache illustrates two different graphs: the graph of values used in a prediction, and the graph followed by differentiation. Transformer-XL keeps an old representation in the value graph while cutting its backward connection to its earlier computation. Replacing the cache by zero changes the forward computation; detaching it does not change its forward numbers.

For a scalar example, let old state m=2w and current prediction y=w·stopgrad(m). At w=3, m=6 and y=18. The detached derivative with respect to the current w is 6. If we differentiate through the original m=2w as well, y=2w² has derivative 12. This difference explains truncated credit assignment. It does not imply the old state was ignored.

The same distinction matters for recurrent training. Constant inference state does not mean backpropagation stores a constant number of intermediate activations. Training may retain a sequence, recompute activations, use checkpointing or use a specialized scan. State-space and sequence-parallel lessons develop those choices.

### 8.2 Count the particular resource you mean

The smaller workspace is motivated by a concrete count: how many question–record pairs must be compared? Two queries reading five records make ten score cells. Applying attention to every pair in a 4,096-position input makes over sixteen million. This count explains why changing the array sizes can matter before considering hardware or implementation speed.

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

An update can be understood as an instruction such as “multiply the current state by .5, then add 2.” Follow it with “multiply by .8, then add 1.” Together they mean “multiply by .4, then add 2.6.” Combining the instructions first gives the same result for every starting state. That ability to combine updates is the entry point to parallel recurrence algorithms.

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

For a second explanation, choose the source that matches the mechanism you are trying to picture. The creator articles below offer a gentler visual route; the papers give the exact architecture. Their reported benchmarks belong to their publication settings, not a current universal ranking.

* [Google Research, Transformer-XL: Unleashing the Potential of Attention Models](https://research.google/blog/transformer-xl-unleashing-the-potential-of-attention-models/), creator article. Start with its context-fragmentation example and segment-recurrence illustrations when it is unclear why a chunk boundary loses information. Then return to our five-record cache trace.
* [Dai and colleagues, Transformer-XL](https://arxiv.org/pdf/1901.02860), research paper. Read section 3 with the segment/layer drawing beside you; appendix B is an advanced route to efficient relative-position score construction. The reported model comparisons are from the paper's 2019 setting.
* [De and colleagues, Griffin](https://arxiv.org/pdf/2402.19427), research paper. Section 2 gives the full blocks and RG-LRU equations; appendix A discusses the gate's behavior and stable parameterization. Useful after the scalar recurrence exercise.
* [Google Developers, RecurrentGemma architecture](https://developers.googleblog.com/en/gemma-explained-recurrentgemma-architecture/), illustrated article. A practical second tour through projections, recurrent layers and local attention. Use the paper for exact mathematical conventions and treat model-specific dimensions as the article's versioned examples.
* [Jaegle and colleagues, Perceiver](https://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf), canonical research paper. Section 3 explains the asymmetric read, latent processing, iterative reads and position features. Start here for the bottleneck architecture rather than an unverified implementation summary.
* [Hugging Face, Perceiver IO: a scalable, fully-attentional model that works on any modality](https://huggingface.co/blog/perceiver), implementation-author tutorial. Its architecture illustrations and shape walkthrough help you track which array supplies queries and which supplies keys and values. The linked examples are another implementation route; they are not the program executed for this lesson's results.
* [Jaegle and colleagues, Perceiver IO](https://arxiv.org/pdf/2107.14795), research paper. Sections 3.1–3.2 explain output queries; the optical-flow example supplies a concrete setting where the output is an array rather than one class.
* [DeepMind, Building architectures that can handle the world's data](https://deepmind.google/blog/building-architectures-that-can-handle-the-worlds-data/), creator article with architecture illustrations and linked audiovisual demonstrations. A less algebraic route to input/latent/output roles; the demonstrations are examples from the authors' historical systems.
* [Hawthorne and colleagues, Perceiver AR](https://proceedings.mlr.press/v162/hawthorne22a/hawthorne22a.pdf), research paper, and [ICML 2022 spotlight with slides and a listed video](https://icml.cc/virtual/2022/spotlight/17886). The inspected [slide deck](https://icml.cc/media/icml-2022/Slides/17886.pdf) builds the two causal masks step by step. The conference page lists a video; playback access was not verified for this packet. Read the slides independently of it.
* [DeepMind, Perceiver AR: general-purpose, long-context autoregressive generation](https://deepmind.google/blog/perceiver-ar-general-purpose-long-context-autoregressive-generation/), creator explanation. Its illustrated progression from input positions to aligned latents offers another route through why a generative Perceiver needs causal information paths.
* [PyTorch attention documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html), API reference for extending the explicit teaching implementation. Inspect mask semantics and dropout behavior when moving between APIs; a Boolean mask does not have the same meaning in every attention interface.
* [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), original data documentation. Read the sampling, coordinate layout and recording context before modifying the real example. Attribution and the exact retained data transformation are in the supplied data record.
