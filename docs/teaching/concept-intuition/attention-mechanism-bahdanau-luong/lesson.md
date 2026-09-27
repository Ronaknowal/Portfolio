# Attention: Let Each Output Read the Input It Needs

Imagine changing `walk` into its past tense, `walked`, one character at a time. While writing `wal`, you need the spelling. After `walk`, you need to decide what ending belongs to the requested form. The useful part of the input changes as the answer grows.

The [previous encoder–decoder lesson](/learn/path/full-curriculum/sequence-to-sequence-encoder-decoder?module=deep-learning-fundamentals) gave the writer a summary of the input. That can work. But it asks one fixed-size representation to carry everything that any later output might need. What if the writer could return to its reading notes instead?

**Attention makes that return possible.** At each output step, the model compares its current need with saved input representations, assigns each a share of the read, and combines the information using those shares. It can read differently when its next task changes. We will build that operation before naming its architectural variants.

**First pass:** follow sections 1–2 to understand one read, section 3 to place it inside a decoder, and section 4 to implement it. In section 5, read the direction-of-learning explanation before the optional derivative. Then inspect the real spelling experiment and its alignment in sections 6–7; run the saved model before a full training run. Exercises 1–5 test that route. The scorer diagnosis and section 8 are deeper branches to return to for local windows, copying, speech and efficiency.

## 1. The problem: one summary has to serve every future question

A reader and a writer have different jobs. The **encoder** reads the source. Its state after each token is a vector: a list of learned numerical features summarizing what it has seen. The **decoder** writes the answer. Its own changing state reflects the output prefix and the information it has received so far.

With a fixed-context encoder–decoder, information from the source reaches the writer through the final encoder state. A later output cannot directly request an earlier encoder state that was discarded. Attention changes this wiring: save the intermediate states, and let each writing step read a mixture of them.

**Visual: summary versus revisitable notes.** Contrast one compressed source route with saved per-position notes. Trace the same input to several writing steps; show the recurrent writer in both designs. This is an information-path illustration, not measured alignment or a claim that the fixed summary must fail on `walk`.

Consider two moments in our example. With output prefix `wal`, source information about the next letter can help produce `k`. With prefix `walk`, information about the requested tense can help produce `e`. These are useful behaviors we want the network to learn, not handwritten rules telling it which position to select. Training supplies correct output strings, not a manually assigned attention location for each character.

### What exactly do we keep?

For the measured example later, the source is `<past> lactate <eos>`. The word is less familiar, but the operation is the same: turn a requested spelling into `lactated`. EOS means “end of sequence.” Number the source positions from 0. Save the encoder state at each position and call it \(h_j\), where \(j\) is that position.

| Position |0|1|2|3|4|5|6|7|8|
|---|---|---|---|---|---|---|---|---|---|
|Token|`<past>`|`l`|`a`|`c`|`t`|`a`|`t`|`e`|`<eos>`|
|Memory vector|\(h_0\)|\(h_1\)|\(h_2\)|\(h_3\)|\(h_4\)|\(h_5\)|\(h_6\)|\(h_7\)|\(h_8\)|

The two `a` positions have different notes. A forward recurrent encoder has read a different prefix by the time it reaches the second one. A saved vector is therefore more than a character ID; it may carry information about earlier characters as well.

**Visual: the source-memory shelf.** Keep source positions and output steps on separate timelines. Show the final encoder state initializing the writer once, and all saved states supplying a fresh read at each output step. Introduce the 64-coordinate example only here, after the identity of each stored object is clear.

We keep both paths: final-state initialization and repeated reads. Attention does not recover information the encoder never represented, and mixing memories can itself lose distinctions. Its benefit is access to several saved representations through shorter, selectable paths.

A **bidirectional** encoder also reads backward, so each saved position can contain information from both sides. Bahdanau's original translation system did this; our small experiment uses a forward encoder to keep the preceding lesson's source representation dimensions. Bidirectionality and the rule for reading memory are separate design choices. [Original model, sections 3 and appendix A](https://arxiv.org/pdf/1409.0473)

## 2. Build one read before building the whole network

First separate finding information from returning it. A weather log might store a time beside each temperature. To ask for the temperature at 14:00, compare your requested time with the saved times, then return the associated temperature. The thing you compare is not the thing you return.

**Visual: lookup becomes a weighted read.** Use three time–temperature records to distinguish requested time, matching label and returned measurement. Compare an exact lookup, an unselective average and a specified fractional mixture. Show the products that produce each result. The fractional shares are constructed to explain mixing; they are not a weather model or learned attention weights.

This gives us three names with distinct jobs:

- The **query** is what we are asking for: the requested time in that lookup, or the decoder's current need in our network.
- A **key** is what we compare with the query: a stored time, or a learned description of a source position.
- A **value** is what we read back: a temperature, or that position's feature vector.

An exact lookup gives one entry weight 1 and all others weight 0. An ordinary average gives every entry the same share. Attention allows the query to determine the shares. Softmax attention with finite scores gives every unmasked entry a positive share mathematically; floating-point underflow can round tiny shares to zero. “Pay attention” does not necessarily mean select just one entry.

### Why read a mixture?

If a winner-takes-all rule chooses A, a tiny score change that leaves A the winner changes nothing about the returned value. That discrete choice is hard to train through using ordinary derivatives. A smooth mixture can shift a little toward B as B's score rises. Output loss can then tell the scoring system whether that shift helped. Section 5 follows that learning signal. Hard selection and sparse reads are possible alternatives, but they need their own mechanisms.

Our network uses numerical vectors rather than literal times. The decoder state supplies a query \(q\). Encoder features supply keys \(k_j\) for comparison and values \(v_j\) for the returned information. These are **roles**; basic recurrent attention often uses the same encoder vector as a value and as the input to a key projection. There need not be separate key and value storage.

### Score: how well does each key match this query?

Use small constructed vectors so every operation is visible. The two coordinates here have no claimed linguistic meaning. Memory A, B and C are labels for three rows, not words learned by the fitted model.

| Memory |Key \(k_j\)|Value \(v_j\)|Score \(q^\top k_j\), for \(q=(1,0)\)|
|---|---|---|---|
|A|(1,0)|(2,0)|1|
|B|(0,1)|(0,2)|0|
|C|(−1,0)|(−1,1)|−1|

The dot product multiplies corresponding coordinates and adds them. Query \((1,0)\) reads the first coordinate of each key: A points with it, B contributes zero, and C points against it. For B, \(1\cdot0+0\cdot1=0\). A larger score will get a larger share. Scores can be negative; they are not probabilities. Dot products depend on vector length as well as direction.

### Normalize: turn three scores into shares of one read

We want nonnegative shares that add to 1. Exponentiating turns each score into a positive number; dividing each by their total gives those shares. For numerical stability, first subtract the largest score. The scores \((1,0,-1)\) become \((0,-1,-2)\), their exponentials are approximately \((1,0.367879,0.135335)\), and the total is 1.503215.

**Visual: scores become shares.** Keep memory identities A/B/C aligned while showing score, shifted score, positive mass and normalized share. Link all masses to their shared denominator. Use separate scales for scores and shares, and calculate every displayed value.

A's share is \(1/1.503215\), about 0.665241. The complete operation is **softmax**:

\[
\alpha_j=\frac{\exp(e_j-\max_i e_i)}{\sum_i\exp(e_i-\max_i e_i)},\qquad
\alpha=(0.665241,\ 0.244728,\ 0.090031).
\]

Here \(e_j\) is a score, \(\alpha_j\) its share, and \(i\) runs over every allowed memory. Subtracting the same maximum cancels between numerator and denominator, so it leaves the mathematical answer unchanged. Raising B's score changes A and C's shares too: they compete for the same total.

### Read: multiply each value by its share, then add

For the first coordinate, A contributes \(0.665241\times2\), B contributes zero, and C contributes \(0.090031\times(-1)\). For the second coordinate, B and C contribute. The same three shares apply to both coordinates:

\[
c=\sum_j\alpha_jv_j
=0.665241(2,0)+0.244728(0,2)+0.090031(-1,1)
=(1.240451,\ 0.579488).
\]

The returned vector \(c\) is the **context**. On the value plane below, the diamond is pulled toward the values with greater shares; it lies inside their triangle because the shares are nonnegative and sum to 1. The adjacent contribution table is the same calculation coordinate by coordinate.

**Visual: three weighted contributions.** Put the values on a two-dimensional plane and draw the context inside their triangle. Show the signed contribution of each memory alongside it. Negative value coordinates remain valid even though the attention shares are nonnegative.

We have completed a read: **query → scores against keys → shared normalization → mixture of values**. It returns information, not yet a character. An output layer will combine that information with the decoder state to choose a character in section 3.

**Investigation: edit the memory, then inspect the read.** The lab begins with a different query so you can investigate rather than replay the worked arithmetic. Change B's returned value while keeping the query and keys fixed: the mixture moves but its shares do not. Then change a key: the shares redistribute. Finally make all values identical. Attention can now move without changing the read. The visible result updates immediately; there is no guess to enter.

## 3. Bahdanau and Luong: distinguish the score from the schedule

A read alone cannot write `walked`. It must sit inside a loop: the writer forms a question, reads source information, combines it with what has already been written, and produces a distribution over the next character. The chosen character becomes an input to the following step. The source notes stay fixed while the writer's state changes.

There are two decisions inside that loop. **How do we compare a question with a memory?** That is the scorer. **Do we read before or after updating the writer?** That is the schedule. Bahdanau and Luong are historical names associated with particular choices; keeping these two questions separate makes the implementations easier to follow.

### Additive scoring

The query and the stored memory may use different feature systems. A decoder might have 96 coordinates while the encoder has 192. We cannot directly compare unequal lists with a dot product. A learned projection turns each into a shared comparison space, rather like giving two measurements a compatible set of features.

**Visual: build a learned comparison.** Trace two different-width inputs into equal-width projections; add coordinate by coordinate, apply tanh, then reduce to a single score. Use a tiny two-coordinate intermediate calculation to show why these are different operations. The original source value follows a separate bypass to the weighted read.

An additive score first projects the query and memory into a shared feature width \(d_a\), combines them, applies a nonlinearity, and reduces the result to one number:

\[
e_{tj}=v_a^\top\tanh(W_q q_t+W_h h_j).
\]

Here \(W_q\) has shape \(d_a\times d_s\), \(W_h\) has shape \(d_a\times d_h\), and \(v_a\) has \(d_a\) entries. Decoder width \(d_s\) and encoder width \(d_h\) can differ. We omit biases in this scoring layer; adding a bias inside the nonlinearity is another valid declared parameterization.

Read the formula from the inside out. \(W_q q_t\) describes the current need; \(W_h h_j\) describes one saved position. Their sum is a comparison feature vector, not an attention weight. Applying \(\tanh\) bends this combination so the effect of a memory can depend on the current query. Finally \(v_a\) combines the comparison features into one scalar score. Repeat the same scorer for every source position, then use the softmax and weighted read from section 2.

The word “additive” refers to adding the projected query and memory before the nonlinearity. It does not mean adding the final attention probabilities.

Writing \(W[q;h]\) inside the same \(\tanh\) is equivalent: split the columns of \(W\) into \(W_q\) and \(W_h\). Concatenation itself does not make this version more expressive. The nonlinear concat score is explicitly shown in [Luong et al.'s revised arXiv version, section 3.1](https://arxiv.org/pdf/1508.04025v5).

### Dot and general scoring

With equal query and memory widths, use

\[
e_{tj}=q_t^\top h_j
\quad\text{(dot)}.
\]

To compare different widths, or learn a transformation before comparison, use

\[
e_{tj}=q_t^\top W_h h_j
\quad\text{(general)}.
\]

For general scoring, \(W_h\) has shape \(d_s\times d_h\). A layer mapping a 192-coordinate memory to a 96-coordinate query space is **general attention**, even if its last operation is a dot product.

The parameter comparison depends on dimensions. Without biases, \(d_s=96,d_h=192,d_a=64\) gives 18,496 parameters for additive scoring and 18,432 for general scoring. That is a difference of 64, not a factor of three. Dot scoring adds none when dimensions already match. This count excludes the encoder, decoder and output layers.

### Two valid decoder timelines

Suppose the already written prefix is `wal`. A recurrent state already exists for the preceding step. We also know the most recent output character, `l`. To write `k`, we can either read using that existing state and then update it with `l` and the read, or first update with `l` and then use the new state to read. Both orders are executable. Asking for the new state before calculating the read, while also requiring that read to calculate the new state, would create a circular dependency.

Let \(s_{t-1}\) be the previous decoder state and \(E(y_{t-1})\) the embedding of the previous output. At the first step, that output is a special start token, BOS.

**Bahdanau-style order:**

1. Query the source using \(s_{t-1}\).
2. Compute weights and context \(c_t\).
3. Update the recurrent state using the previous token and that context:
   \(s_t=\operatorname{GRU}([E(y_{t-1});c_t],s_{t-1})\).
4. Use the new state and context to predict \(y_t\).

**Luong-style order:**

1. Update the recurrent state from the previous token:
   \(s_t=\operatorname{GRU}(E(y_{t-1}),s_{t-1})\).
2. Query the source using this new state.
3. Combine the state and returned context into an attentional vector:
   \(\tilde s_t=\tanh(W_c[s_t;c_t]+b_c)\).
4. Apply an output layer and softmax to predict \(y_t\).

**Visual: synchronized decoder timelines.** Use arrows that reveal which state exists before a read. The previous token goes into both timelines; source memory stays fixed. Put the scoring-function choice on the read operation, separate from its position in the timeline. Never draw \(y_t\) as an input to the computation that predicts it.

Our teaching model uses the same form of attentional output projection in both orders. It uses native GRU cells, not the original papers' complete networks: the original Bahdanau decoder had a different output network, and Luong's experiments used stacked LSTMs. These simplifications are explicit so the experiment demonstrates mechanisms without impersonating a historical reproduction.

### Input feeding remembers earlier reads

A Luong decoder can feed \(\tilde s_{t-1}\) alongside the previous token at the next step:

\[
s_t=\operatorname{GRU}([E(y_{t-1});\tilde s_{t-1}],s_{t-1}).
\]

Initialize the fed vector to zero. This is an extra connection from the previous **attentional vector**, not a replacement for the token embedding or a second pass through the source. It gives the state direct access to information from earlier attention decisions. It does not guarantee that each source position is covered once.

The full program supports this connection as a separate extension. The reported fits keep it off for the general-scoring model; switching it on changes the decoder input width and requires a new fit.

### Deeper scorer diagnosis: when the question cancels out

This branch explains why the nonlinearity matters. You can continue to section 4 after understanding one complete decoder step, then return to this counterexample.

Suppose someone tries to learn the score with a linear layer on the concatenated query and key:

\[
e_j=a^\top q+b^\top k_j.
\]

The first term is the same for every memory. Softmax cancels it:

\[
\frac{\exp(a^\top q+b^\top k_j)}
{\sum_i\exp(a^\top q+b^\top k_i)}
=\frac{\exp(b^\top k_j)}{\sum_i\exp(b^\top k_i)}.
\]

The weights no longer depend on the question. The code can run, gradients can exist elsewhere, and the model can appear to “have attention,” yet this scoring rule cannot change its read according to the query. A nonlinear interaction or a query–key product repairs this particular limitation. This is why the details inside a small scoring formula matter.

## 4. Put the read into tensors and code

We have followed one example and one output step. Batching just lays several examples side by side in rectangular arrays; it does not change the meaning of query, memory or read. The bookkeeping matters because an array may contain storage cells that are not real source positions.

Suppose a batch has \(B\) examples, padded source width \(S\), padded target width \(T\), memory width \(d_h\), and decoder width \(d_s\).

| Object |Shape|Meaning|
|---|---|---|
|Source IDs|\(B\times S\)|Embedding addresses, including request and source EOS|
|Source lengths|\(B\)|Number of actual tokens in each source|
|Encoder memory|\(B\times S\times d_h\)|One feature vector per stored source position|
|One query|\(B\times d_s\)|One current question per example|
|One attention row|\(B\times S\)|Distribution over valid source positions|
|One context|\(B\times d_h\)|Weighted memory read|
|All attention rows|\(B\times T\times S\)|One source distribution per decoder step|
|Output logits|\(B\times T\times V\)|Scores over output vocabulary, not source positions|

Attention and output softmax normalize over different things. A model can put 99% attention on one source position while being uncertain among several output characters.

### Source padding, target padding and output constraints do different jobs

A short source is padded to share a rectangular batch with longer sources. PAD is storage, not another observation. Set invalid source scores to \(-\infty\) **before** softmax. The remaining valid positions receive the full probability mass.

Imagine two real memories with equal scores and scalar values 2 and 4. Their read is \((2+4)/2=3\). Add a padding slot with score 0 and value 0, but forget the mask: all three scores are still equal, so the read becomes \((2+4+0)/3=2\). Nothing about the real observations changed. Their shares shrank because storage entered the competition.

**Visual: padding changes the denominator.** Align the same real values in correct and buggy reads. Show two halves becoming three thirds when zero padding is admitted; remove the invalid share before normalization in the corrected case. Connect the displayed shares to the changed scalar read.

Setting padded values to zero while leaving their scores valid is insufficient. A zero-valued memory can still steal probability mass and shrink the context. In a bidirectional encoder, processing padding can also change actual backward states; an attention mask cannot undo that earlier contamination. Our program packs actual source lengths before running the encoder, then also masks the attention read.

Each source here includes a request token and EOS, so there is always a valid memory position. A general-purpose reader must reject or explicitly handle an all-masked row: ordinary softmax on all \(-\infty\) values is undefined.

Target padding has a separate role. For the reference `cared<EOS>`, teacher-forced decoder inputs are `<BOS>cared`. Cross-entropy scores each next target, ignores padded target cells, and averages over the remaining target tokens. Source EOS is a memory; target EOS is a predicted stopping decision. BOS and request tokens are not valid generated outputs in this task.

**Investigation: repair a padded read.** Extend a short source with extra storage cells. At the selected output step, inspect how much attention those cells should receive and whether the context changes. Compare correct masking with allowing the added zero memories into softmax. Inspect both the stolen attention mass and downstream probabilities. Then edit an actual source character: that is a real input change and should rebuild encoder memory and projected keys.

A source edit invalidates its cache. A decoder-prefix edit can reuse source memory but must recompute the affected decoder suffix. Renaming a display label changes neither.

### Implement exactly that read

This small NumPy program implements one dot-product read. `keys` has one row per source position; `values` may have a different feature width. The validity mask says which rows contain actual source data. Finite arrays with compatible dimensions are the function's input contract. The explicit all-masked check protects the one undefined case in the normalization.

```python
import numpy as np

def attention_read(query, keys, values, valid):
    if not np.any(valid):
        raise ValueError("At least one source position must be valid")
    scores = keys @ query
    scores = np.where(valid, scores, -np.inf)
    masses = np.exp(scores - np.max(scores))
    weights = masses / masses.sum()
    return weights, weights @ values

keys = np.array([[1., 0.], [0., 1.], [-1., 0.]])
values = np.array([[2., 0.], [0., 2.], [-1., 1.]])
weights, context = attention_read(
    np.array([1., 0.]), keys, values, np.array([True, True, True])
)
print(np.round(weights, 6), np.round(context, 6))
```

The output is `[0.665241 0.244728 0.090031] [1.240451 0.579488]`, matching section 2. `keys @ query` computes all scores; `weights @ values` computes all context coordinates. Everything between them normalizes only allowed memories. The implementation uses vectorized matrix operations and stores one score row; it does not construct a source-by-source matrix for this one-query operation.

The library route uses the same operations on tensors: a matrix product for scores, `masked_fill` before `.softmax`, and a weighted matrix product for the read. PyTorch then differentiates those operations as part of a custom `nn.Module`. Section 6 supplies the complete trainable model; section 7 links the full NumPy reconstruction. A library multi-head layer changes the architecture, so simply importing it would not reproduce this recurrent decoder.

## 5. Learn which read helps the output

Training does not ordinarily come with labels saying which input position to look at. It comes with desired outputs. If a different read would lower the output loss, gradients adjust the encoder, scoring parameters and decoder.

Think of each source value as proposing information to add to the current mixture. If giving B a little more share would make the desired output more likely, training should raise B's relative score. If the same change would hurt, it should lower it. The feedback depends on what B contains, not just on whether B already has a large weight.

Return to our three-memory arithmetic. Pretend the two context coordinates are logits for two classes, and the correct class is the second one. This tiny output head makes the whole path visible:

\[
p=\operatorname{softmax}(c)=(0.659477,\ 0.340523),\qquad
L=-\log p_2=1.077272.
\]

The first coordinate currently wins, although we want the second class. A's value \((2,0)\) favors the first class, so putting more weight there would hurt. B's \((0,2)\) favors the second; C's \((-1,1)\) also favors it. Moving share away from A toward B or C therefore moves the output in a useful direction. That is the intuition the derivative below will make precise.

**Visual: a memory's effect depends on its content.** Compare each value's class-2-minus-class-1 margin with the current mixture's margin. Draw their locations on one signed number line, with the desired direction labeled. Use actual worked values and keep this figure distinct from the later backward-dependency diagram.

### Deeper calculation: follow one gradient all the way back

The sign argument above is enough to follow the training story on a first pass. For implementing or checking the backward pass, this derivation supplies the exact quantities used by the next lab.

For softmax cross-entropy, the derivative with respect to these logits is

\[
g=\frac{\partial L}{\partial c}=p-(0,1)
=(0.659477,-0.659477).
\]

We want less first-coordinate support and more second-coordinate support. How should a source score change?

Differentiate the softmax-weighted sum:

\[
\frac{\partial c}{\partial e_j}=\alpha_j(v_j-c),
\qquad
\frac{\partial L}{\partial e_j}
=\alpha_j\,g^\top(v_j-c).
\]

The derivative compares each value with the **current mixture**, not just its attention weight. Here the three score gradients are

\[
(0.587450,\ -0.429460,\ -0.157990).
\]

Gradient descent therefore reduces A's score and raises B's and C's. With dot scoring \(e_j=q^\top k_j\),

\[
\frac{\partial L}{\partial q}
=\sum_j\frac{\partial L}{\partial e_j}k_j
=(0.745440,-0.429460).
\]

One learning-rate 0.1 step changes \(q=(1,0)\) to approximately \((0.925456,0.042946)\). Recomputing the entire read gives loss 1.003220, down from 1.077272. These are calculated values, independently matched to automatic differentiation and finite differences. A sufficiently large step need not lower the loss.

**Visual: signed credit through the read.** Connect the output loss to the context, source contributions, scores and query. Positive/negative labels show the gradient sign; arrows show data dependency rather than a claim that every parameter should move the same way. In a full network, the query is produced by a decoder and the keys/values by an encoder, so backpropagation continues into both.

**Investigation: one update and a null case.** Move the query and learning-rate controls and compare the old and updated loss as they change. Set the learning rate to zero to verify no parameter or loss changes. Then make every value identical: the weights can change while the read stays the same. In that case \(\alpha_j(v_j-c)=0\), so this read supplies no score gradient. A dramatic-looking heatmap alone is not evidence of a useful learning signal.

Attention gives a direct weighted path from an output's loss to stored source values. Recurrent dependencies still exist; this additional path does not make vanishing gradients, optimization difficulties or generalization errors impossible.

## 6. Does revisiting memory help on real spellings?

Return to the opening problem: can a writer generate an unfamiliar spelling more reliably if it can consult the source again? We can now test that with trained models, rather than infer success from an attractive attention picture. The unit of success is a complete requested word, including a correct stopping decision.

Use the same small English inflection extract as the preceding lesson: 1,800 records from 600 selected spellings, with three requested verb forms each. The source is the [pinned UniMorph English repository](https://github.com/unimorph/eng/tree/66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b). Its README names Wikipedia and licenses the data under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The supplied extract retains attribution, source rows, filtering decisions and license in [data provenance](data-provenance.md).

All requests for the same lemma stay together. Selected lemmas sharing a target form are also grouped together, preventing the demonstrated `work`/`worke` overlap. The final partition has 451 training lemmas/1,353 rows and 149 development lemmas/447 rows, with no shared lemma spelling or target form across those partitions. This is a conservative identity check, not a proof that the lexicon contains no other aliases.

The development data has already been inspected in the earlier lesson and is inspected here. It remains development data. These experiments are useful for comparing mechanisms and diagnosing errors; they do not supply an untouched final test.

### Keep the task fixed and declare the architecture changes

Inputs are a request token, lowercase letters and source EOS. Targets are the recorded form and target EOS. The 32-token vocabulary is declared from the alphabet and special tokens. Each model uses 24-coordinate embeddings and a 64-coordinate forward GRU encoder. No pretrained downloads or GPU are needed.

The preceding fixed-context model initializes a 64-coordinate decoder from the final encoder state. The new models preserve that initialization and add attention:

|Model|Query/order|Other changes|Parameters|
|---|---|---|---:|
|Fixed context|No repeated source read|Previous lesson's decoder and linear output head|37,408|
|Additive attention|Previous state; read before update|32-coordinate scoring layer; context enters GRU; combined output head|62,080|
|General attention|New state; read after update|Learned 64→64 key map; combined output head; no input feeding|49,760|

The comparison changes complete declared architectures, including parameter counts and output heads. It is not an isolated proof that a score function alone causes every difference. Matched dimensions, data, sampling schedule and training budget make it useful; they do not eliminate every confound.

All neural runs use seeds 1, 2 and 3, Adam learning rate 0.003, 1,200 updates, 64 examples sampled with replacement per update and global gradient clipping at norm 1. The sampling generator is seeded with 100 plus the run seed. The same seed does not imply equal initial parameters across differently shaped networks. Evaluate fixed checkpoints without choosing an early stopping point after seeing them. Decode greedily, allowing at most 16 generated tokens including EOS.

Retain the simple suffix rules from the previous lesson. They handle common `e` and consonant-plus-`y` endings but have no irregular-word lookup or general consonant-doubling rule. A useful model must compete with that task knowledge, not only another neural network.

### Actual results

“Exact” requires the recorded form and natural EOS. Character error rate is total insertions, deletions and substitutions divided by total reference characters, excluding EOS. The 447 development references contain 3,490 characters.

Read one comparison first. After the same number of updates, fixed-context seed 1 produces 53 of 447 development forms exactly; additive-attention seed 1 produces 395. Simple suffix rules produce 407. In this experiment, a repeated source read helps the neural model substantially, but does not beat the task-specific rules. The table retains all seeds so one selected run does not stand in for the whole result.

|Model / seed|Training exact /1,353|Development exact /447|Development character edits|Character error rate|
|---|---:|---:|---:|---:|
|Predeclared suffix rules|1,206|407|57|0.01633|
|Fixed context /1|1,328|53|1,517|0.43467|
|Fixed context /2|1,323|41|1,533|0.43926|
|Fixed context /3|1,287|52|1,475|0.42264|
|Additive /1|1,338|395|86|0.02464|
|Additive /2|1,327|380|108|0.03095|
|Additive /3|1,352|387|97|0.02779|
|General /1|1,292|345|165|0.04728|
|General /2|1,353|392|90|0.02579|
|General /3|1,348|385|101|0.02894|

The attentive models generalize to many more unseen spellings than the fixed-context runs under this protocol. The rule baseline still has the highest exact-match count and the fewest character edits. Treat that as useful evidence about this task and data budget. It is not an inconvenience to tune away.

The outcomes also separate fitting from transfer. General seed 2 gets every training form right but misses 55 development forms. Additive seed 3 fits 1,352 training records but does not produce the best development result. Reading the source again helps, while memorizing training outputs remains possible.

**Measured figure: all run checkpoints.** Plot development exact-match fraction at updates 0, 100, 400, 800 and 1,200 for each seed, with a horizontal rule-baseline line at 407/447. Keep a small table of counts beside the plot. Several curves dip near the end; show those actual dips. Do not smooth them into monotonic progress or attach invented wall-clock values.

Length is another diagnostic, not a universal capacity threshold. Additive seed 1 gets 145/168 shorter lemmas and 250/279 longer lemmas correct. These groups differ in spelling patterns and examples, not just length. An increase in longer-word accuracy does not mean length is intrinsically easier, just as a decrease would not by itself prove a fixed memory limit.

### Run the saved model first

Download [attentive-inflection.py](attentive-inflection.py), [calculated-inputs.json](calculated-inputs.json) and [english-inflections.csv](english-inflections.csv) into one directory. The JSON contains actual trained parameters and recorded results; it is an offline input, not a request to train during page rendering.

Use a Python environment with NumPy and CPU PyTorch. The author run used Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu. Compatible versions can run the program, but numerical kernels and training trajectories can differ. From that directory, save and run this complete small example:

```python
from pathlib import Path
import importlib.util
import json
import sys
sys.dont_write_bytecode = True
import torch

root = Path.cwd()
spec = importlib.util.spec_from_file_location("inflection", root/"attentive-inflection.py")
inflection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inflection)
torch.set_num_threads(1)
report = json.loads((root/"calculated-inputs.json").read_text(encoding="utf-8"))
run = next(item for item in report["runs"] if item["kind"] == "additive" and item["seed"] == 1)
model = inflection.AttentiveInflector("additive")
shapes = model.state_dict()
model.load_state_dict({
    name: torch.tensor(value, dtype=shapes[name].dtype)
    for name, value in run["weights"].items()
})
result = inflection.greedy(model, [{"lemma": "lactate", "feature": "past"}])[0]
print(result["prediction"], result["ended_with_eos"])
```

The executed result is `lactated True`. Generation receives the lemma and request; it never receives `lactated` as a reference input. Change the request or spelling and inspect how the generated answer changes. A constructed spelling has no automatic correctness label just because the model returns something.

### Read and optionally run the complete training program

The full program below creates all inputs, masks, model components, optimizer steps, evaluations and saved parameters. To reproduce training, save it as `attentive-inflection.py` beside the CSV and run `python attentive-inflection.py`. It runs six fits and writes `calculated-inputs.json`; preserve the supplied file under another name if you want to compare your run with it.

Read `encode` first: it packs real source lengths and returns memory, precomputed keys, a validity mask and an initial state. Then read `step`: the two branches put the attention read on different sides of the recurrent update. `forward` feeds reference prefixes for likelihood training; `greedy` feeds generated prefixes. `assess` reports both, so low teacher-forced loss is not silently equated with correct free generation.

<details>
<summary>Complete CPU training, generation and measurement program</summary>

```python

"""CPU attentive inflection: exact shared data, two explicit decoder orders."""
from pathlib import Path
import csv
import json
import platform
import string
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROOT = Path(__file__).resolve().parent
TOKENS = ["<pad>", "<bos>", "<eos>", "<past>", "<participle>", "<third_person>"] + list(string.ascii_lowercase)
INDEX = {token: index for index, token in enumerate(TOKENS)}
PAD, BOS, EOS = 0, 1, 2
ALLOWED = [EOS] + list(range(6, len(TOKENS)))
MAX_OUTPUT = 16

def load_records():
    with (ROOT/"english-inflections.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    train = [row for row in rows if row["partition"] == "train"]
    development = [row for row in rows if row["partition"] == "development"]
    assert (len(train), len(development)) == (1353, 447)
    for field in ("lemma", "form"):
        assert not {row[field] for row in train} & {row[field] for row in development}
    return train, development

def source_batch(rows):
    sources = []
    for row in rows:
        lemma, feature = row["lemma"], row["feature"]
        if not 1 <= len(lemma) <= 32 or any(ch not in string.ascii_lowercase for ch in lemma):
            raise ValueError("Use 1 to 32 lowercase a-z characters.")
        if feature not in ("past", "participle", "third_person"):
            raise ValueError("Unknown inflection request.")
        sources.append([INDEX[f"<{feature}>"]] + [INDEX[ch] for ch in lemma] + [EOS])
    source = torch.full((len(rows), max(map(len, sources))), PAD, dtype=torch.long)
    for index, ids in enumerate(sources):
        source[index, :len(ids)] = torch.tensor(ids)
    return source, torch.tensor(list(map(len, sources)), dtype=torch.long)

def batch(rows):
    source, lengths = source_batch(rows)
    targets = [[INDEX[ch] for ch in row["form"]] + [EOS] for row in rows]
    target = torch.full((len(rows), max(map(len, targets))), PAD, dtype=torch.long)
    for index, ids in enumerate(targets):
        target[index, :len(ids)] = torch.tensor(ids)
    previous = torch.cat([torch.full((len(rows), 1), BOS), target[:, :-1]], dim=1)
    return source, lengths, previous, target

class AttentiveInflector(nn.Module):
    def __init__(self, kind="additive", input_feeding=False):
        super().__init__()
        if kind not in ("additive", "general"):
            raise ValueError("Choose additive or general.")
        if kind == "additive" and input_feeding:
            raise ValueError("Input feeding here names only the Luong extension.")
        self.kind, self.input_feeding = kind, input_feeding
        self.embedding = nn.Embedding(len(TOKENS), 24, padding_idx=PAD)
        self.encoder = nn.GRU(24, 64, batch_first=True)
        extra = 64 if kind == "additive" or input_feeding else 0
        self.decoder = nn.GRUCell(24+extra, 64)
        self.key_projection = nn.Linear(64, 32 if kind == "additive" else 64, bias=False)
        if kind == "additive":
            self.query_projection = nn.Linear(64, 32, bias=False)
            self.score_projection = nn.Linear(32, 1, bias=False)
        self.combine = nn.Linear(128, 64)
        self.readout = nn.Linear(64, len(TOKENS))
        invalid = torch.ones(len(TOKENS), dtype=torch.bool)
        invalid[ALLOWED] = False
        self.register_buffer("invalid_output", invalid)

    def encode(self, source, lengths):
        packed = pack_padded_sequence(self.embedding(source), lengths, batch_first=True, enforce_sorted=False)
        output, final = self.encoder(packed)
        memory, _ = pad_packed_sequence(output, batch_first=True, total_length=source.shape[1])
        valid = torch.arange(source.shape[1])[None, :] < lengths[:, None]
        # Memory, projected keys and mask are source-only; cache once for decoding.
        cache = (memory, self.key_projection(memory), valid)
        return cache, final[0], torch.zeros_like(final[0])

    def attend(self, query, cache):
        memory, keys, valid = cache
        if not bool(valid.any(-1).all()):
            raise ValueError("Every source needs at least one valid position.")
        if self.kind == "additive":
            scores = self.score_projection(torch.tanh(keys+self.query_projection(query)[:, None, :])).squeeze(-1)
        else:
            scores = torch.bmm(keys, query[:, :, None]).squeeze(-1)
        scores = scores.masked_fill(~valid, -torch.inf)
        weights = scores.softmax(-1)
        context = torch.bmm(weights[:, None, :], memory).squeeze(1)
        return context, weights, scores

    def step(self, previous, state, fed, cache):
        embedding = self.embedding(previous)
        if self.kind == "additive":
            context, weights, scores = self.attend(state, cache)
            state = self.decoder(torch.cat([embedding, context], dim=-1), state)
        else:
            decoder_input = torch.cat([embedding, fed], dim=-1) if self.input_feeding else embedding
            state = self.decoder(decoder_input, state)
            context, weights, scores = self.attend(state, cache)
        attentional = torch.tanh(self.combine(torch.cat([state, context], dim=-1)))
        logits = self.readout(attentional).masked_fill(self.invalid_output, -torch.inf)
        return logits, state, attentional, weights, context, scores

    def forward(self, source, lengths, previous):
        cache, state, fed = self.encode(source, lengths)
        logits, weights = [], []
        for token in previous.unbind(1):
            output, state, fed, attention, _, _ = self.step(token, state, fed, cache)
            logits.append(output)
            weights.append(attention)
        return torch.stack(logits, 1), torch.stack(weights, 1)

def edit_distance(left, right):
    previous = list(range(len(right)+1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(current[-1]+1, previous[column]+1, previous[column-1]+(a != b)))
        previous = current
    return previous[-1]

@torch.no_grad()
def greedy(model, rows, max_output=MAX_OUTPUT):
    model.eval()
    source, lengths = source_batch(rows)
    cache, state, fed = model.encode(source, lengths)
    previous = torch.full((len(rows),), BOS, dtype=torch.long)
    finished = torch.zeros(len(rows), dtype=torch.bool)
    outputs, logs = [[] for _ in rows], [[] for _ in rows]
    for _ in range(max_output):
        logits, state, fed, _, _, _ = model.step(previous, state, fed, cache)
        log_probabilities = logits.log_softmax(-1)
        chosen = log_probabilities.argmax(-1)
        for index in range(len(rows)):
            if not finished[index]:
                token = int(chosen[index])
                outputs[index].append(token)
                logs[index].append(float(log_probabilities[index, token]))
        finished |= chosen == EOS
        previous = torch.where(finished, EOS, chosen)
        if bool(finished.all()):
            break
    return [{"prediction": "".join(TOKENS[token] for token in output if token != EOS),
             "tokens": output, "ended_with_eos": bool(output and output[-1] == EOS),
             "log_probability": sum(values)} for output, values in zip(outputs, logs)]

def summarize(rows, predictions):
    errors = [edit_distance(row["form"], item["prediction"]) for row, item in zip(rows, predictions)]
    return {"count": len(rows), "exact": sum(error == 0 and item["ended_with_eos"] for error, item in zip(errors, predictions)),
            "character_edits": sum(errors), "reference_characters": sum(len(row["form"]) for row in rows),
            "character_error_rate": sum(errors)/sum(len(row["form"]) for row in rows),
            "no_eos": sum(not item["ended_with_eos"] for item in predictions)}

@torch.no_grad()
def assess(model, rows):
    model.eval()
    source, lengths, previous, target = batch(rows)
    logits, _ = model(source, lengths, previous)
    loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD)
    predictions = greedy(model, rows)
    return {"teacher_forced_nll": float(loss), **summarize(rows, predictions)}, predictions

def main():
    torch.set_num_threads(1)
    train, development = load_records()
    report = {"versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "protocol": {"kinds": ["additive", "general"], "seeds": [1, 2, 3], "input_feeding": False,
              "updates": 1200, "batch_size": 64, "adam_lr": .003, "gradient_clip": 1., "max_output": MAX_OUTPUT},
              "tokens": TOKENS, "runs": []}
    for kind in report["protocol"]["kinds"]:
        for seed in report["protocol"]["seeds"]:
            torch.manual_seed(seed)
            model = AttentiveInflector(kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            generator = torch.Generator().manual_seed(100+seed)
            checkpoints, clipped = [], 0
            for update in range(1201):
                if update in (0, 100, 400, 800, 1200):
                    metrics, _ = assess(model, development)
                    checkpoints.append({"update": update, **metrics})
                    print(json.dumps({"kind": kind, "seed": seed, **checkpoints[-1]}), flush=True)
                if update == 1200:
                    break
                model.train()
                chosen = torch.randint(len(train), (64,), generator=generator).tolist()
                source, lengths, previous, target = batch([train[index] for index in chosen])
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(source, lengths, previous)
                loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
                clipped += float(norm) > 1
                optimizer.step()
            training, _ = assess(model, train)
            metrics, predictions = assess(model, development)
            report["runs"].append({"kind": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                "clipped_updates": clipped, "checkpoints": checkpoints, "train": training, "development": metrics,
                "development_predictions": [{"lemma": row["lemma"], "feature": row["feature"], "reference": row["form"], **item}
                                            for row, item in zip(development, predictions)],
                "weights": {key: value.tolist() for key, value in model.state_dict().items()}})
            (ROOT/"calculated-inputs.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
            print(json.dumps({"kind": kind, "seed": seed, "train": training, "development": metrics}), flush=True)

if __name__ == "__main__":
    main()


```

</details>

## 7. Connect a real read to a generated character

The opening diagram showed a behavior we wanted: a changing need should lead to a changing read. Now we can inspect actual trained weights. In an alignment picture, each row answers “where did this output step read?” and each column identifies one saved source position. Read across one row before comparing rows; a bright cell is a large source share, not the probability of a correct output.

The seed 1 additive model produces `lactated<EOS>` for the worked input. Its actual attention rows put the greatest weight on these source positions:

|Output being predicted|Highest-weight source position|That weight|Probability of emitted output|
|---|---|---:|---:|
|`l`|1: `l`|0.9755|0.9998|
|`a`|2: first `a`|0.8894|0.9994|
|`c`|3: `c`|0.9056|0.9999|
|`t`|4: first `t`|0.9047|0.9999|
|`a`|5: second `a`|0.6200|0.9804|
|`t`|6: second `t`|0.8527|0.9999|
|`e`|7: `e`|0.4830|1.0000, rounded|
|`d`|0: `<past>`|0.7016|0.9998|
|`<eos>`|6: second `t`|0.3358|1.0000, rounded|

**Visual: the complete alignment matrix and one linked read.** Columns are source positions, rows are generated steps, and brightness is attention weight on a fixed 0–1 scale. The table summarizes maxima; the figure includes every weight. Selecting a row exposes its scores, probability distribution, weighted context and next-character probabilities. Include the request and both EOS roles with distinct labels.

The picture is consistent with copying much of the spelling and consulting the request while appending `d`. The EOS row is more diffuse and does not peak at source EOS in this model. That does not make its prediction invalid: contextual features and decoder state can supply stopping information in other ways.

Do not turn that plausible reading into a claim of uniquely discovered linguistic rules. A memory for one position can encode information from other positions. The output also depends on decoder state and previous outputs, and the value vectors matter in addition to their weights.

Here is an exact counterexample to “the attention distribution uniquely explains the answer”:

\[
v_1=(1,0),\quad v_2=(0,1),\quad v_3=(0.5,0.5).
\]

Both \(\alpha=(0.4,0.4,0.2)\) and \(\alpha'=(0.2,0.2,0.6)\) produce \(c=(0.5,0.5)\). Their heatmaps look different, but a downstream calculation receiving only this context and the same other inputs cannot distinguish them. These are valid softmax outcomes: scores equal to their log probabilities would produce them.

**Investigation: test an alignment hypothesis.** Change the spelling or requested form and follow one generated row through scores, source shares, context and next-character probabilities. Recompute using the saved parameters and compare the actual alignment, context and next-output probabilities. Teacher-forced and generated-prefix rows must be labeled separately. A changed reference label used only for scoring must not change generation.

The masking investigation also has an instructive result: admitting extra zero memories can alter attention and probabilities while leaving the greedy word unchanged. A correct-looking word is not enough to prove the tensor computation is correct. Conversely, two different weights need not imply two different words. Inspect the quantity that your hypothesis actually concerns.

## Implement the read, then compose it into ordinary training

The complete [attentive-inflection.py](attentive-inflection.py) exposes additive and general scoring, source masks, normalized weighted reads and the decoder schedule. [attention-calculations.py](attention-calculations.py) opens the small numerical derivatives and local-window/copy calculations, while [attention-mechanics.py](attention-mechanics.py) reconstructs saved weights through NumPy and compares its trace with the native model. The scores, weights and context all remain learner-visible. The normal library route is this custom `nn.Module` composed with `nn.GRU`, embeddings, losses and optimizers; `nn.MultiheadAttention` is a different architecture and should not be presented as a drop-in Bahdanau decoder.

The encoder's projected keys can be cached because the source and scorer parameters stay fixed during one decoding call. Cache them **after** the source mask has been paired with that source, and rebuild them after a parameter update or source edit. The decoder query changes every step. For source length S, target length T, memory width H, query width D and scoring width A, project the source once in O(SHA). Each output step costs O(DA + SA + SH) for query projection, scores and value mixing. Cached keys occupy O(SA), beside O(SH) source memory. These costs exclude the recurrent update and vocabulary projection. That is a useful avoided recomputation, not a claim to remove attention's source scan.

For a research change, add a positive scoring temperature before masked softmax, not after the weighted sum. The same padding mask must still exclude invalid positions exactly; temperature does not legalize padding. With general scores `[0, log(3)]` and values `[2,10]`, temperature1 gives weights[0.25,0.75] and context8; temperature2 gives weights proportional to[1,√3] and context approximately7.0718. Smaller temperature sharpens competition but can make gradients and alignments less forgiving; it is not an accuracy guarantee.

<details><summary>Implementation hint</summary>Divide scores by temperature, then apply the illegal-key mask before softmax. Reject a row with no legal source token.</details>

<details><summary>Solution and success criteria</summary>Implement `weights = (scores / temperature).masked_fill(~valid, -inf).softmax(-1)` for strictly positive temperature. At temperature2, context is `(2 + 10*sqrt(3))/(1+sqrt(3)) ≈7.0718`. Recompute one output and its score gradient using the same saved source/query parameters in both NumPy and Torch. Append a large-valued masked source token and confirm it changes neither context nor output. Do not compare two different random fits and call that temperature-implementation parity.</details>

The extended topics in the next section remain distinct: exact hard-monotonic alignment algorithms, a complete pointer-generator training system and location-sensitive speech models need their own objectives and datasets. The local copy-aggregation and normalized-window calculations are complete bounded mechanisms; they do not secretly claim those whole systems have been trained here.

## 8. Deeper branches: other reads and other tasks

This section is optional on the first pass. Each branch changes one part of the source-read idea; it does not add a new requirement before beginning the next core topic.

### Global, local and monotonic are different constraints

If a recording contains thousands of frames but the next character depends mostly on a nearby region, rereading every frame may waste work. A window limits the search. The tradeoff is visible: an excluded memory cannot help, even when it contains the best match. This is a choice about which memories are available, separate from how the allowed ones are scored.

Global attention considers every valid source position at every target step. For a source of length \(S\) and output of length \(T\), that creates \(S T\) score comparisons.

Local attention restricts available positions to a window. A window centered at 3 with radius 2 on positions 1–5 contains all five; radius 1 contains only 2, 3 and 4. The decoder cannot consult a distant position outside that window even if it would have received the largest global score.

Luong's local-p predicts a real center from the current state and multiplies its windowed alignment by a Gaussian factor. In the paper's one-based position convention,

\[
p_t=S\,\sigma(v_p^\top\tanh(W_p s_t)),\qquad
w_{tj}=\alpha_{tj}\exp\left[-\frac{(j-p_t)^2}{2(D/2)^2}\right].
\]

The window clips at source boundaries. Membership changes when the center crosses a boundary; the construction is differentiable almost everywhere, not everywhere. The original equation multiplies by the Gaussian **without an additional normalization in that displayed formula**. Those resulting weights need not sum to one. [Revised section 3.2](https://arxiv.org/pdf/1508.04025v5)

For a constructed example, positions 1–5 have scores \((0,0.5,1,-0.5,2)\) and scalar values equal to their position numbers. Center 3, radius 2 gives post-Gaussian weights approximately

\[
(0.010128,0.074836,0.203425,0.027531,0.074836),
\]

with sum 0.390755 and context 1.254375. Explicitly renormalizing those weights changes the context to 3.210133. Renormalization is a valid alternative design, but it changes the read's magnitude. Label the formula in use.

**Visual: window on a source ruler.** Show excluded positions, window-normalized scores, Gaussian multipliers and final weights as distinct stages. Moving center and radius updates which positions enter, their shares and the weight sum immediately. Keep the original formula and renormalized alternative separate.

A local window can move backward; locality alone does not enforce monotonicity. A monotonic mechanism constrains movement through source order. That can suit speech, while unrestricted reordering is useful in translation. Availability is another condition: a bidirectional encoder or a read of the entire future source cannot become streaming merely because its display uses a narrow window.

### Copy a name that the output vocabulary does not contain

An ordinary output softmax can emit only vocabulary entries. Looking closely at an unfamiliar source name does not automatically create a new output token.

A pointer-generator adds a copying route. Let \(p_{\mathrm{gen}}\) be the learned probability of using the vocabulary route. Sum attention over **all occurrences** of a word to obtain its copy mass, then mix:

\[
P(w)=p_{\mathrm{gen}}P_{\mathrm{vocab}}(w)
+(1-p_{\mathrm{gen}})\sum_{j:x_j=w}\alpha_j.
\]

For source `Ada met Ada`, attention \((0.2,0.3,0.5)\), vocabulary probabilities `Ada:0.1, met:0.6, left:0.3`, and \(p_{\mathrm{gen}}=0.4\), the final probabilities are `Ada:0.46, met:0.42, left:0.12`. Ada's two positions contribute 0.7 copy mass. If Ada were outside the vocabulary, its vocabulary contribution would be zero while its copy route could remain available.

**Visual: two routes into one vocabulary.** Connect both Ada positions to one output entry, keeping position probability and word probability separate. A changed example with a repeated out-of-vocabulary item appears in practice. This construction is useful for names, identifiers and source-specific strings; copying source text does not by itself verify factual correctness. [Pointer-generator network, section 2.2](https://arxiv.org/pdf/1704.04368)

### Remember where a speech reader has been

Imagine transcribing a repeated syllable. Two regions of the recording can have very similar acoustic content. Asking only “does this sound like what I need?” may not distinguish the earlier occurrence from the next occurrence. Remembering where the last read was provides another clue.

Speech contains repeated and similar acoustic fragments. A content score can be ambiguous when two memories look similar. A location-aware scorer adds features from the previous attention row:

\[
f_t=F*\alpha_{t-1},\qquad
e_{tj}=v^\top\tanh(W_s s_{t-1}+W_hh_j+W_f f_{tj}+b).
\]

The convolution \(F*\alpha_{t-1}\) measures local patterns around each position in the previous read. It lets a score depend on both current content and recent location. This differs from permanently labeling one position “already used.” [Chorowski et al., section 2.2](https://arxiv.org/pdf/1506.07503)

For example, two acoustic regions can resemble the same vowel. A previous attention peak near the first supplies positional evidence about which occurrence the decoder may be approaching. Learned weights decide how much to use that evidence; the formula alone does not prohibit jumps or guarantee alignment.

Listen, Attend and Spell combines acoustic encoding with character generation. Its pyramidal bidirectional encoder reduces source resolution before reading it, so acoustic frame count need not match output character count. Its original full-input architecture is not inherently streaming. [LAS, sections 3–3.1](https://arxiv.org/pdf/1508.01211)

A separate **coverage** vector \(u_{tj}=\sum_{\tau<t}\alpha_{\tau j}\) records accumulated reads. It can enter a score, or an overlap penalty \(\sum_j\min(\alpha_{tj},u_{tj})\) can discourage repetition. Requiring exactly one unit per source is inappropriate for tasks such as summarization, where some material should be omitted and a phrase may support several output words. [Coverage mechanism, section 2.3](https://arxiv.org/pdf/1704.04368)

Think of coverage as a running tally beside the memory shelf. Every output distributes one unit of attention across the source, so two completed outputs have distributed two units in total. Unlike an individual attention row, the accumulated tally does **not** sum to one. Here is a constructed three-position example; the entries are attention mass, not counts of distinct facts already expressed.

| Read or tally | Source A | Source B | Source C | Total |
|---|---:|---:|---:|---:|
| First completed read | 0.6 | 0.3 | 0.1 | 1 |
| Second completed read | 0.2 | 0.5 | 0.3 | 1 |
| Coverage before the third read | 0.8 | 0.8 | 0.4 | 2 |
| Third read | 0.1 | 0.2 | 0.7 | 1 |
| Overlap: smaller of current share and old tally | 0.1 | 0.2 | 0.4 | 0.7 |
| Coverage after the third read | 0.9 | 1.0 | 1.1 | 3 |

The third read sends most of its mass to C, which had received less attention. Its first 0.4 at C overlaps the old tally; the remaining 0.3 does not. The penalty is therefore 0.7. At the first read, the old tally is all zero and this overlap penalty is zero, whatever the first attention distribution. A penalty discourages repeated allocation; it does not ban it, prove factual coverage, or identify which meaning has already been verbalized. Location-aware features inspect the **previous row's shape**; coverage remembers the **sum of all earlier rows**. They answer different questions and can coexist.

### Cache what is fixed; measure what is expensive

The encoder memory stays fixed while generating one output sequence. Cache projected keys once. Additive scoring's cache costs \(O(Sd_hd_a)\). Each read then projects a query in \(O(d_sd_a)\), scores positions in \(O(Sd_a)\), and combines values in \(O(Sd_h)\). General scoring can cache its \(d_h\to d_s\) projection and use \(O(Sd_s)\) dot products per step.

Across the output, source–target interactions have an \(S T\) factor. It becomes square only when source and target lengths are identified. Retaining every attention row uses \(O(S T)\) extra storage; sequential inference can keep one row at a time, plus source memory, keys and recurrent state. A teaching heatmap deliberately retains extra history.

The recurrent state and input feeding still impose sequential decoder dependencies. Caching keys does not make those dependencies parallel. Beam search additionally needs a state, fed vector and prefix history per live candidate; it can share unchanged source memory. A source change requires a new cache, while corrected prefixes require replaying affected decoder states.

Scaled dot-product attention divides the dot product by \(\sqrt d\). Under independent, zero-mean, unit-variance query/key coordinates, the unscaled dot product has variance \(d\); division keeps that variance near one under those assumptions. Learned recurrent states need not satisfy them. Inserting scaling into a fitted general-attention model changes its computation.

Later multi-head layers add query/key/value projections, several reads and an output projection. A library multi-head module is not a drop-in reproduction of this decoder. Optimized dot-product kernels also do not automatically implement arbitrary additive scorers. Compare math, shapes, masks and outputs before choosing a fast kernel. Benchmark latency at the intended batch sizes, lengths, precision and device rather than inferring it from parameter counts. The [later self-attention lesson](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals) develops those components.

## 9. Practice: calculate, diagnose and transfer

Attempt each question before opening its hint or solution. Problems 1–5 test the core route; 6–8 use the optional branches.

### 1. A changed question

Let \(q=(0,1)\) with the three keys and values from section 2. Calculate scores, weights and context. Which memory gets the most weight?

<details>
<summary>Hint</summary>
The query selects each key's second coordinate. Use the same weights for both value coordinates.
</details>

<details>
<summary>Solution</summary>
Scores are \((0,1,0)\), giving weights approximately \((0.211942,0.576117,0.211942)\). B gets the most weight. The context is \((0.211942,1.364175)\): first coordinate \(2\alpha_A-\alpha_C\), second \(2\alpha_B+\alpha_C\). Equal A/C scores do not imply equal values.
</details>

### 2. A scorer that ignores its question

A developer proposes \(e_j=3q-2k_j+0.7\) for scalar queries and keys. Will changing \(q\) change the attention weights? Suggest a score that can depend on both.

<details>
<summary>Hint</summary>
Factor quantities shared across source positions out of the softmax numerator and denominator.
</details>

<details>
<summary>Solution</summary>
No: \(\exp(3q+0.7)\) cancels. A product \(qk_j\), or a nonlinear score \(\tanh(3q-2k_j+0.7)\), can change relative scores with the query. “Can” is deliberate: symmetry or saturation can still produce little change for a particular input.
</details>

### 3. Where does the first read occur?

A decoder starts in \(s_0\) and receives BOS. Someone computes \(c_1=\operatorname{Attention}(s_1,H)\), then defines \(s_1=\operatorname{GRU}([\operatorname{BOS};c_1],s_0)\). Identify the problem and give two valid repairs.

<details>
<summary>Hint</summary>
Trace which value must exist first. A circular dependency is not yet an executable update rule.
</details>

<details>
<summary>Solution</summary>
These equations require \(s_1\) to obtain \(c_1\) and vice versa, without defining a solver or intermediate state. A Bahdanau-style repair reads with \(s_0\), then updates using the context. A Luong-style repair updates from BOS first without the current context, then reads with \(s_1\) and combines the result in the output head. Input feeding can use the previous attentional vector, initially zero, without creating a current-step cycle.
</details>

### 4. A padding bug hidden by the final word

Two valid positions have scores \((0,0)\) and scalar values \((2,4)\). An unmasked zero-valued PAD position also has score 0. Calculate the correct and buggy contexts. Must their final greedy words differ?

<details>
<summary>Hint</summary>
Compare denominators containing two and three exponentials. Then distinguish continuous logits from their discrete argmax.
</details>

<details>
<summary>Solution</summary>
The correct weights \((1/2,1/2)\) give context 3. The buggy weights \((1/3,1/3,1/3)\) give context 2. Logits can change while their largest entry stays the same, so identical greedy words do not rule out the bug. Mask invalid scores before normalizing.
</details>

### 5. Plan a more specific comparison

You want to learn whether additive versus general **scoring alone** affects this task. Is the two-architecture table a sufficient isolation? Describe a better experiment and what data it consumes.

<details>
<summary>Hint</summary>
List changes besides the scorer: update order, decoder input width, output head and parameter count. Decide which must be held fixed for your narrower question.
</details>

<details>
<summary>Solution</summary>
Keep the encoder, decoder order, context injection, output head, split, training schedule and decoding rule fixed; swap only the scorer. Report the remaining scorer parameter/computation differences. Predeclare seeds/metric, retain all runs and keep the rule baseline. This estimates behavior under one protocol, not a universal ordering. The already inspected partition remains development: repeated comparisons cannot become a fresh final test merely by changing the architecture name.
</details>

### 6. Copying repeated unknown words

Source tokens are `red blue red`. Attention is \((0.15,0.25,0.60)\), \(p_{\mathrm{gen}}=0.2\), and the vocabulary is `blue:0.5, green:0.5`. Compute the final probabilities, including `red`.

<details>
<summary>Hint</summary>
Combine the two red positions. Its vocabulary probability is zero; its copy probability is not.
</details>

<details>
<summary>Solution</summary>
Copy mass is `red:0.75, blue:0.25`. Final probabilities are `red:0.60, blue:0.30, green:0.10`, summing to one. The extended output set includes source words. Taking only the largest red-position weight would discard its other occurrence.
</details>

### 7. Does a Gaussian preserve a probability distribution?

A window-normalized row is \((0.2,0.5,0.3)\), with Gaussian multipliers \((0.5,1,0.5)\). Compute the product's sum and its renormalized alternative. Why is a comment calling both “the same weighted average” wrong?

<details>
<summary>Hint</summary>
Multiply first. A second normalization changes every nonzero weight.
</details>

<details>
<summary>Solution</summary>
The product is \((0.1,0.5,0.15)\), sum 0.75. Renormalizing gives \((2/15,2/3,1/5)\). With fixed values, the original context is 0.75 times the normalized context. Proportions match, but magnitudes and downstream logits need not. Declare the convention.
</details>

### 8. A readable heatmap, an impossible streaming claim

A speech system uses a bidirectional encoder over the complete recording and a local attention window. Its documentation says the window permits instant live transcription with no future audio. What information-path issue should be checked?

<details>
<summary>Hint</summary>
Attention is applied after encoding. Ask what a backward encoder state has already used.
</details>

<details>
<summary>Solution</summary>
A memory can already depend on future audio. Restricting a subsequent read to nearby positions cannot remove that dependency. A streaming design must declare encoder lookahead, chunk/buffering policy, permitted memory and output latency as well as the attention movement rule. It may need a causal or limited-lookahead encoder and an online alignment mechanism. A narrow visible window proves none of those properties by itself.
</details>

## 10. Continue the route and choose another explanation

You are ready to continue when you can calculate a masked read, trace both decoder schedules, explain how output loss trains a score, and distinguish a changed attention picture from a changed output. Copying and speech can remain optional return points.

Next is [Long-Context Sequence Models: Transformer-XL, Griffin and Perceiver](/learn/path/full-curriculum/long-context-sequence-models-transformer-xl-griffin-perceiver?module=deep-learning-fundamentals). It asks how access and computation change when keeping or reading all memory becomes expensive. It introduces additional mechanisms locally; the later [Self-Attention & Multi-Head Attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals) provides their dedicated treatment.

Useful alternate routes:

- **A visual overview:** [Olah and Carter, Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/), especially “Attentional Interfaces,” shows how a changing read connects a reader and writer. The article's attention section was reviewed; it is a classic mechanism explanation, not current library documentation.
- **Lookup before algebra:** [D2L, Queries, Keys, and Values](https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html) starts from retrieval and connects it to weighted pooling. Use it alongside section 2; our weather and vector examples are independently constructed.
- **Visual next step into transformers:** [3Blue1Brown, Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention/) (2024) explains the desired contextual behavior before the matrix operations, with an accompanying [video](https://www.youtube.com/watch?v=eMlx5fFNoYc). The article and its diagrams were reviewed; the complete video was not watched. It teaches transformer self-attention, so return to our decoder timelines when comparing it with Bahdanau or Luong.

- **Textbook with tensor code:** [D2L 1.0.3, Attention Scoring Functions](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html) covers masks, batch matrix products, dot and additive scores. Use it after section 2 to connect vector calculations with batched code. Its helpers/conventions differ from this program; compare definitions before mixing snippets.
- **Recurrent attention chapter:** [D2L 1.0.3, Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) connects stored outputs, valid lengths and decoder queries. Its translation experiment is separate from our inflection measurements.
- **Video and notes:** [Stanford Online CS224N Lecture 8: Neural Machine Translation, Seq2seq and Attention](https://www.youtube.com/watch?v=XXtpJxZBa2c), with [official 2019 notes](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf). A spoken alternative for following the timelines. Lecture identity and companion notes were checked; the full video was not watched during authoring. Use the corrected primary equations here for nonlinear concat and exact input-feeding conventions.
- **Original motivation:** [Bahdanau, Cho and Bengio](https://arxiv.org/pdf/1409.0473), especially sections 3/5 and appendix A. Its historical translation results do not establish a universal input-length boundary.
- **Scorers, windows and input feeding:** [Luong, Pham and Manning, arXiv v5](https://arxiv.org/pdf/1508.04025v5), sections 3/4. Use its explicit nonlinear concat formula. Its local-p multiplication differs from our explicitly renormalized alternative.
- **Applications:** [Pointer-generator networks](https://arxiv.org/pdf/1704.04368), [location-aware speech attention](https://arxiv.org/pdf/1506.07503), and [Listen, Attend and Spell](https://arxiv.org/pdf/1508.01211). Start at the mechanism sections cited above. Their full experimental reproduction is a separate project.

The [provenance](data-provenance.md), [executed program](attentive-inflection.py), [actual trained results](calculated-inputs.json), [constructed calculations](attention-calculations.py) and [saved-model traces](mechanics-results.json) make the page's numbers inspectable.
