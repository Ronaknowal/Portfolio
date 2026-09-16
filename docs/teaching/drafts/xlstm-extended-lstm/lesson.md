# xLSTM: learn what to keep, and how to read it back

A handwritten digit can be read one horizontal strip at a time. After the first strip, the model has a few strokes. After the fourth, it has more evidence. After the eighth, it must classify the complete image. If the original strips are no longer available, what should the model carry forward?

In [Modern Hopfield Networks](/learn/path/full-curriculum/modern-hopfield-networks?module=deep-learning-fundamentals), a query could inspect an explicit bank of patterns. xLSTM explores a different storage choice: continually combine incoming information into a fixed-size state, and learn the rules for writing, retaining and reading that state. One variant maintains normalized scalar memories. Another maintains matrices of associations between keys and values.

The name means **extended long short-term memory**. It identifies a family of recurrent cells and the residual network blocks built around them. It does not mean that every cell is an LSTM with a larger hidden vector, or that every published xLSTM model uses both variants.

**Your route through the lesson.** First follow the scalar ledger, then the matrix address grid, then the row-by-row digit reader. By the end of that core route you should be able to calculate a write and a read, explain why numerical rescaling must preserve the whole formula, train a small model, and distinguish memory efficiency from successful remembering. The marked deeper branches cover parallel/chunk computation, current large-model variants and hardware tradeoffs. You can return to their derivations after completing the core experiment.

[Figure X01: eight image strips enter a fixed state; a separate explicit-bank view makes the storage difference visible. The final classifier reads the state-derived representation, not unseen image strips.]

## 1. Start with an ordinary memory cell

A recurrent model receives an input vector `x_t` and a previous state. It produces a new state and an output. The index `t` can mean a text token, a sensor sample, or, in our experiment, an image row. It need not mean one second.

A classical LSTM has a cell vector `c_t`, which carries information, and a hidden vector `h_t`, which is exposed to the surrounding network and used to compute later gates. For one coordinate, suppressing its index:

\[
c_t=f_t c_{t-1}+i_t z_t,\qquad h_t=o_t\tanh(c_t).
\]

The candidate `z_t` is commonly a tanh of a learned affine transformation of the input and previous hidden vector. The input, forget and output gates are commonly sigmoids of similar transformations. A sigmoid maps a real number into `(0,1)`. The forget gate controls how much old cell content survives; the input gate controls the new contribution; the output gate controls what is exposed.

For example, with old content `0.8`, retention `0.5`, new candidate `−0.4` and write gate `0.25`, the next cell is `0.5×0.8 + 0.25×(−0.4) = 0.3`. This update contains both multiplication and addition. A forget gate is not a switch that removes an entire old token from a list: it scales a mixed numerical state.

[Figure X02: two contribution bars, +0.4 from the old state and −0.1 from the new candidate, join at +0.3. A separate output valve leads through tanh.]

These gates are learned from the task loss. We hand-set them first so that the mechanism is visible. Later, a neural network will produce them from pixels and state.

A sigmoid can make a sharp decision: a write gate near one and a forget gate near zero can replace old information strongly. xLSTM is therefore not motivated by an inability of sigmoids to overwrite anything. The useful question is more specific: **which parameterization of write weights, state normalization and associative storage makes a desired memory operation easier to learn and scale?**

The original xLSTM family investigates two answers:

| Cell | Stored information | What determines the next gates? | Main mechanism to understand |
| --- | --- | --- | --- |
| sLSTM | Scalar content and normalization mass in each channel | Input and previous hidden state, with mixing within a head | Normalize a learned history of writes |
| mLSTM | A key-by-value association matrix, plus normalization state in the exponential variant | Current layer input, without the same hidden-to-gate recurrence | Write outer products and read with a query |

A **channel** is one numerical coordinate. A **head** is a group of coordinates with its own memory operation. A matrix head has a key width and a value width, which need not be equal.

## 2. sLSTM as a weighted ledger

Imagine that each observation contributes a signed estimate and a nonnegative amount of evidence. Keep two totals: a weighted content total `c`, and the total weight `n`. Their ratio is the current estimate. Forgetting scales both totals; a new write adds to both.

For one scalar memory, begin with `c_0=n_0=0` and define

\[
i_t=e^{a_t},\qquad f_t=\sigma(b_t),\qquad
c_t=f_t c_{t-1}+i_t z_t,\qquad
n_t=f_t n_{t-1}+i_t,\qquad
h_t=o_t\frac{c_t}{n_t}.
\]

Here `a_t` is the write **log-weight**, `b_t` is a forget preactivation, and `o_t` is an output gate. The original family also considers exponential forget gates. We use sigmoid forgetting in the worked core and executable model. It makes the raw retention factor lie between zero and one.

The write weight can be larger than one. What matters to the normalized estimate is its size relative to the surviving old mass. With a tanh candidate and a sigmoid output gate, this scalar output remains bounded by the largest magnitude among the candidates written so far. Its internal totals need not be small.

### Work through three observations

Use candidates `[0.2, −0.6, 0.8]`, write weights `[1, 3, 9]`, retention `0.5` on every step, and output gate `0.75`.

| Step | Surviving old content | New content | `c_t` | `n_t` | `h_t` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0.2 | 0.2 | 1 | 0.150000 |
| 2 | 0.1 | −1.8 | −1.7 | 3.5 | −0.364286 |
| 3 | −0.85 | 7.2 | 6.35 | 10.75 | 0.443023 |

At step three, the surviving weights on the original candidates are `[0.25, 1.5, 9]`. Their sum is `10.75`. The latest candidate supplies about `83.72%` of that mass. The estimate before the output gate is `6.35/10.75 ≈ 0.590698`, between the smallest and largest candidates. The output gate then scales it to `0.443023`.

[Figure X03: a chronological contribution ledger. Each old weight shrinks at the next retention gate, while its candidate value stays attached. At the final step show both signed weighted content and positive mass; do not represent negative content as negative weight.]

This picture explains revision. A large new write can dominate surviving evidence without requiring the raw candidate itself to be large. It also explains interference: one scalar remembers a weighted aggregate, not a recoverable list of all individual observations.

Expanding the recurrence makes the history explicit:

\[
w_{t,j}=i_j\prod_{\ell=j+1}^{t}f_\ell,\qquad
\frac{c_t}{n_t}=\frac{\sum_{j=1}^{t}w_{t,j}z_j}{\sum_{j=1}^{t}w_{t,j}}.
\]

The empty product for `j=t` equals one. A new write is not immediately multiplied by its own step's forget gate; that gate affects what was already stored. The nonnegative normalized weights sum to one. This convex-average interpretation applies to this scalar recurrence with empty initialization; it will not transfer unchanged to the signed matrix read.

### Where learning enters

For a vector of channels, learned matrices produce candidates and gate preactivations from `x_t` and `h_{t−1}`. Within a scalar-memory head, these transformations can mix channels: one hidden coordinate can affect another coordinate's next write or retention. Different heads restrict this recurrent mixing to their own groups. Pointwise multiplication in the state update does not imply that the whole cell treats every feature independently.

[Figure X04: three channels enter a small learned recurrent matrix before separate gates. Contrast the mixing arrows with the coordinatewise state update. Label all shown weights as a schematic, not learned digit parameters.]

The next gates depend on a hidden state that depends on previous gates. That dependency matters when we discuss parallel training. It also enables input-conditioned state transitions beyond a fixed decay of past input projections.

## 3. Stabilization changes the representation, not the answer

Exponential write weights are convenient mathematically but dangerous to form directly when their log-weights are very large. We can preserve the answer while storing rescaled totals.

Let `c'_t=e^(−m_t)c_t` and `n'_t=e^(−m_t)n_t`. The shared scale cancels in `c'_t/n'_t`. Choose a new log-scale

\[
m_t=\max\{\log f_t+m_{t-1},\ a_t\},
\]

then compute

\[
f'_t=e^{\log f_t+m_{t-1}-m_t},\qquad
i'_t=e^{a_t-m_t},
\]

\[
c'_t=f'_t c'_{t-1}+i'_t z_t,\qquad
n'_t=f'_t n'_{t-1}+i'_t,\qquad
h_t=o_t c'_t/n'_t.
\]

Each exponent used for a stabilized gate is nonpositive. At least one of the two stabilized gate factors is one, except at limiting or invalid inputs. This avoids constructing a huge common multiplier. A stabilized write equal to one does not mean that the underlying raw write was one or that the model secretly replaced exponential gating with sigmoid gating.

For the three-step example, the stabilized states are:

| Step | `m_t` | `i'_t` | `f'_t` | `c'_t` | `n'_t` | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 1 | 0.5 | 0.2 | 1 | 0.150000 |
| 2 | `ln 3` | 1 | `1/6` | −0.566667 | 1.166667 | −0.364286 |
| 3 | `ln 9` | 1 | `1/6` | 0.705556 | 1.194444 | 0.443023 |

At step two, divide both raw totals by three. At step three, divide both raw totals by nine. Comparing `i'_2` and `i'_3` directly would compare numbers expressed under different scales. Their equality does not erase the original write-weight ratio.

[Figure X05: raw and stabilized ledgers side by side, connected by a shared scale label at each step. The output traces coincide. Permit toggling the intermediate numbers without animating them automatically.]

The max state need not increase monotonically: sufficiently strong forgetting and a smaller new write can lower the relevant scale. It must travel with the memory across a sequence boundary. Resetting only `m`, or only `n`, produces a different state.

Starting with `n_0=1` would insert a zero-valued unit of prior mass if `c_0=0`. That can be an intentional model, but it is not the empty ledger. With finite positive first write, the empty ledger acquires a positive denominator immediately. An implementation still needs a defined response to nonfinite inputs; adding arbitrary epsilon to every formula is not a substitute for choosing the intended operator.

### Try it: a write-weight ledger

As a second worked comparison, use candidates `[-0.4, 0.7, −0.2]`, weights `[2,1,5]`, retention `[0.8,0.6,0.4]` and output gate `0.8`. Before reading the result, consider whether weakening the last write from five to `0.5` makes the final output more or less negative.

The original final output is about `−0.124082`; after that edit it is about `−0.006957`. Earlier positive content has more influence. Now set every candidate to `0.6`. Changing positive write weights and retention does not change the normalized estimate; with output gate `0.8`, every output is `0.48`. This is a useful null experiment: the weights changed, but all the available evidence agreed.

The investigation starts with a separate four-observation problem. Edit its actual candidates or gate inputs and record a prediction before revealing the resulting trace; the worked examples remain available under a separate reset control.

[Investigation XA: editable candidate rows, write log-weights and retention; raw/stabilized state trace, contribution mass and recorded prediction. A separate advanced control adds a common 1,000 to all write log-weights from an empty state. The stable output remains unchanged to rounding while direct raw exponentiation would overflow.]

The scale shift null is a property of this normalized scalar read from an empty ledger. It is not a blanket invariance of every downstream architecture or of the matrix read's fixed floor.

### A learned write, one gradient step at a time

Suppose the two candidates are `−0.2` and `0.8`, the first write is one, retention at step two is `0.5`, and the second write is `e^θ`. Expose the normalized output directly and ask it to approach target `0.7`:

\[
y(\theta)=\frac{-0.1+0.8e^\theta}{0.5+e^\theta},\qquad
L=\tfrac12(y-0.7)^2.
\]

At `θ=0`, `y=0.466667`. Differentiating the quotient gives

\[
\frac{dy}{d\theta}=\frac{0.5e^\theta}{(0.5+e^\theta)^2},\qquad
\frac{dL}{d\theta}=(y-0.7)\frac{dy}{d\theta}\approx-0.051852.
\]

A gradient-descent step with learning rate `0.5` raises `θ` to `0.025926`, raises the prediction to `0.472403`, and lowers the loss from `0.027222` to `0.025900`. The target has encouraged a stronger write of the second candidate. In a trained network the same chain rule reaches the matrices that produced the candidates and gates. Stabilization should preserve that calculation; it does not replace optimization or gradient clipping.

[Figure X06: a local loss curve in θ with the checked before/after points and arrows through write weight → normalized prediction → loss.]

## 4. mLSTM gives memory an address space

A scalar ledger combines evidence along one coordinate. A matrix memory can associate a **key** with a **value**. A key is a learned address vector; a value is the information to retrieve. A **query** asks the memory for values whose keys align with that query.

Use one matrix head. Keys and queries have `d_k` coordinates; values have `d_v`. Throughout this lesson, `C` has **key rows and value columns**, shape `d_k × d_v`. Writing a key and value forms their outer product:

\[
C_t=f_t C_{t-1}+i_t k_t v_t^\top,\qquad
n_t=f_t n_{t-1}+i_t k_t.
\]

The outer product contains one product for every key-coordinate/value-coordinate pair. For key `[1,0]` and value `[2,−1]`, it is `[[2,−1],[0,0]]`. The write changes the first key row. For a general key, it spreads information across rows.

[Figure X07: the key vector labels rows, the value vector labels columns, and each cell is the corresponding product. Move a focus outline from a vector pair to its matrix cell.]

Before output gating and the surrounding normalization layer, read

\[
r_t=\frac{C_t^\top q_t}{\max\{|n_t^\top q_t|,1\}}.
\]

The numerator has `d_v` coordinates. The dot product in the denominator is a scalar. In a learned model we scale a projected query by `1/√d_k` before this formula, exactly once. The hand examples use already-scaled queries so that the arithmetic is transparent.

Why does a query retrieve anything? Substituting the accumulated writes gives

\[
C_t^\top q_t=\sum_{j\le t}w_{t,j}v_j(k_j^\top q_t).
\]

The alignment `k_j^T q_t` multiplies the contribution of the associated value. An orthogonal key contributes zero to that query's numerator. Similar keys can interfere because their values contribute together. The matrix keeps an aggregate of outer products, not a slot for every original item.

### Three writes, then read an earlier address

Start empty; let every retention factor be `0.5`.

| Step | Key | Value | Write weight | Query |
| --- | --- | --- | ---: | --- |
| 1 | `[1,0]` | `[2,−1]` | 1 | `[1,0]` |
| 2 | `[0,1]` | `[0,3]` | 1 | `[0,1]` |
| 3 | `[1,0]` | `[4,1]` | 2 | `[1,0]` |

The first read returns `[2,−1]`. The second matrix is `[[1,−0.5],[0,3]]`, with `n=[0.5,1]`. Query `[0,1]` selects its second key row and returns `[0,3]`.

At step three,

\[
C_3=\begin{bmatrix}8.5&1.75\\0&1.5\end{bmatrix},\qquad
n_3=\begin{bmatrix}2.25\\0.5\end{bmatrix}.
\]

The query reads the first row: numerator `[8.5,1.75]`, denominator `2.25`, output approximately `[3.777778,0.777778]`. The newer `[4,1]` dominates, but the older value has not been deleted. Its surviving weight is `0.25`; the latest weight is two.

[Figure X08: three matrix heatmaps with explicit signed values, matching normalizer vectors, and a query beam selecting rows. Under step three, expand the first-row result into its two surviving writes.]

This is an additive association update. It is not a dictionary assignment, and it is not a delta-rule update that explicitly subtracts the current prediction at an address before writing a correction. One scalar forget gate per head also scales every old association in that head together. It cannot retain one old key and erase another at that same step merely by changing this scalar.

### Signed retrieval is not softmax attention

A dot product can be negative. Consider two keys `[1,0]` and `[-1,0]`, scalar values `2` and `−1`, unit writes and no forgetting. Query `[1,0]` gives coefficients `+1` and `−1`. The numerator is `2−(−1)=3`; the signed normalizer sum is zero. The denominator floor makes the read equal three.

That result is outside the interval `[−1,2]`. It could not be a convex mixture of these two values. Calling these signed coefficients attention probabilities would hide exactly the behavior the learner needs to see.

[Figure X09: positive and negative address alignment on an axis, signed value contributions, normalizer cancellation and the floor at one. Show the scalar-ledger convex interval beside the matrix result outside it.]

The denominator prevents division by a signed sum near zero, but it does not make the read universally bounded by the stored value magnitudes. An outer-product memory is sometimes called a covariance-style memory; it is not automatically an empirical centered covariance matrix. Learned projections and biases matter.

### The stabilization detail that changes the answer

For the exponential variant, use the same `m`, `i'` and `f'` idea as before, storing `C'=e^(−m)C` and `n'=e^(−m)n`. Now the correct read is

\[
r=\frac{C'^\top q}{\max\{|n'^\top q|,e^{-m}\}}.
\]

Both the variable normalizer and the fixed raw floor must be represented in the new scale. Keeping a floor of one after rescaling changes the operator.

A one-dimensional counterexample makes the error visible. Take `q=0.5`, `k=0.25`, `v=4`, and write log-weight two. The raw numerator is `e²/2 ≈ 3.694528`; the raw signed mass is `e²/8 ≈ 0.923632`, so the denominator is one. The answer is `3.694528`.

In the stabilized representation, `m=2`, `C'=1`, `n'=0.25`, numerator `0.5`, and the correct denominator is `max(0.125,e^(−2))=e^(−2)`. The answer is unchanged. A floor of one would instead produce `0.5`.

[Figure X10: raw and scaled denominator rails. The fixed floor moves with the scale; a clearly marked incorrect branch shows the changed answer.]

This is why two implementations agreeing with each other is not sufficient if both copied the same formula error. Compare each to the intended mathematical operator on a case where the floor actually controls the answer.

### Try it: change the address, keep the value

The matrix investigation begins with fresh keys, values and queries. Predict what happens if the last key reverses direction while its value stays unchanged. Inspect both the numerator and the normalizer before interpreting the final read. You can edit a key, a value, a query or a write weight independently.

Then zero every value. Reads must be zero although keys, normalizer and gate history can remain nonzero. Restore the values and try a zero query: its numerator is zero and its denominator is the floor. These null cases distinguish stored address geometry from the information it points to.

[Investigation XB: editable key/value pairs, query coordinates and write/retention factors; linked address plane, outer-product grid, signed contribution bars and raw/scaled read. The changed key fixture produces a larger signed output, not a probability distribution.]

## 5. One matrix operation, several execution schedules

**Deeper branch.** You can proceed to the digit reader after understanding that a memory must be carried across chunks. This derivation explains why matrix-memory training can use large parallel operations while incremental inference updates a state.

For a head whose keys, values, queries and gates have already been computed from the layer input, define the causal write influence

\[
g_{t,j}=\begin{cases}i_j\prod_{\ell=j+1}^{t}f_\ell,&j\le t,\\0,&j>t.\end{cases}
\]

Collect queries, keys and values into row matrices `Q`, `K`, `V`. If `A=(QK^T)⊙G`, row `t` of `AV` is the raw read numerator. The raw denominator for that row is `max(|Σ_j A_tj|,1)`. A triangular causal mask prevents future writes from contributing. This is a dense sequence formulation of the same recurrence, not ordinary row-softmax attention.

[Figure X11: a lower-triangular time-by-time coefficient matrix. A selected entry expands into query/key alignment, the write at its source time, and only the later retention gates.]

For stability, form logarithmic influences with prefix sums of log retention:

\[
\log g_{t,j}=a_j+F_t-F_j,\qquad F_t=\sum_{\ell=1}^{t}\log f_\ell.
\]

The diagonal has no retention factor because `F_t−F_t=0`. Mask future entries to negative infinity, subtract the largest valid log influence in each row, and exponentiate. The denominator's unit floor becomes `exp(−row_scale)`, just as in the recurrent formulation.

This dense form materializes a `T×T` matrix. It is a useful mathematical reference, but defeats the memory goal for very long sequences. A third schedule uses chunks.

### A chunk has old memory and new local writes

Suppose a chunk begins after time `s`. For a time `t` inside it, define incoming retention `p_t=∏_(ℓ=s+1)^t f_ℓ`. The raw numerator is

\[
C_s^\top q_t\,p_t+\sum_{j=s+1}^{t}g_{t,j}v_j(k_j^\top q_t).
\]

The first term reads memory from before the chunk. The second performs a small causal comparison within the chunk. The signed normalizer combines the corresponding old and new contributions **before** applying the absolute value and floor. Normalizing each part separately and then adding would be a different operation.

At the chunk's end, update its boundary state in one aggregate write:

\[
C_{s+b}=\left(\prod_{\ell=s+1}^{s+b}f_\ell\right)C_s
+\sum_{j=s+1}^{s+b}g_{s+b,j}k_jv_j^\top,
\]

with the analogous formula for `n`. A final shorter chunk uses its actual length. Stabilized implementations align the scales of incoming and local quantities before combining them.

[Figure X12: two rails enter each output: an incoming boundary state and a small triangular local tile. They join at one numerator and one denominator. A boundary capsule contains every required state, not only the matrix.]

The supplied `memory_mechanisms.py` includes a recurrent operator, a stabilized dense operator and an explicit chunk operator for moderate unscaled inputs. A seven-token checked example with key width three and value width two agrees across chunk sizes `1,2,3,4,7,9` to less than `3×10^−15`. A nonzero incoming-state case also agrees. Editing the final three values leaves the first four outputs unchanged.

The chunk program is a teaching reference for the decomposition. Its unscaled moderate-input arithmetic is not a safe replacement for a production stabilized kernel on arbitrary large log-weights.

### Try it: move the boundary without changing the story

Choose a seven-write sequence, record whether splitting after the third write should change any output, and compare whole-sequence, recurrent and chunk views. Move the chunk size to two, four or larger than the sequence. Outputs should agree to rounding when state carry is correct.

Now deliberately reset the state at a boundary. The later outputs can change; this is a changed input history, not a faster execution of the same history. Finally edit a future value and check an earlier output. That earlier output must remain unchanged.

[Investigation XC: an editable causal write timeline with movable chunk boundaries, visible boundary state, incoming/local contribution panels, output-difference plot and an explicit reset experiment.]

Why does sLSTM not get the same simple schedule? In sLSTM, this layer's gates depend on `h_(t−1)`, which itself depends on the previous gates. Those gate values cannot all be precomputed from the layer input alone. This prevents the same direct input-precomputed affine scan. It does not prevent parallelism over batch, channels or heads, or efficient fused sequential kernels. In stacked mLSTM networks, layer inputs already contain learned context from earlier layers; precomputability within one layer does not mean the network is context-free.

## 6. Put the cell inside a trainable network

A cell describes sequence mixing. A useful neural block also needs transformations around it. Our small experiment uses this complete path:

`8 pixels in one row → learned 8-to-16 projection → RMS normalization → sequence cell → residual addition → RMS normalization → gated feed-forward network → residual addition → ten class logits`.

A **residual addition** gives information a path around a transformation. **RMS normalization** divides a vector by the square root of its mean squared coordinate plus epsilon, then applies learned coordinate scales. Our feed-forward branch multiplies a SiLU-transformed projection by another projection before contracting back to width 16; this is a SwiGLU-style gated transformation. These operations are applied at each row. Only the final row's logits contribute to the training loss.

[Figure X13: the complete row-reader block with dimensions, both residual routes and the single final-row loss. The cell can be switched between LSTM, sLSTM and mLSTM while the surrounding path remains visible.]

A logit is an unconstrained class score. To turn the final vector `ℓ` into class probabilities, use `p_c=exp(ℓ_c)/Σ_j exp(ℓ_j)` with a stable softmax. For correct class `y`, cross-entropy is `−log p_y`. A high probability on the wrong digit incurs a large loss. Backpropagation differentiates this objective through the classifier, feed-forward block, every recurrent step and the learned projections. Adam updates the parameters.

The scalar cell produces its four groups of preactivations from the current normalized input plus a learned transformation of the previous hidden vector. Its stored state is `(h,c,n,m)`, each width 16. The matrix cell uses one key width of eight and a value width of 16. Its stored recurrent state is `(C,n,m)`, with shapes `8×16`, `8` and scalar. Its query is scaled once, and its read is normalized and output-gated. We cap its gate preactivations smoothly with `15 tanh(raw/15)` to bound the range in this instructional model.

These are deliberately small one-head blocks. They expose the actual mechanisms without reproducing every projection, head arrangement, convolution or training recipe of a published large model.

### Run the complete programs

Download the lesson's program bundle containing `row_sequence_models.py`, `memory_mechanisms.py`, `author_calculations.py` and the attributed `optdigits.tra`, `optdigits.tes`, `optdigits.names` files. Keep those files in one directory. Use a Python environment with NumPy and a CPU-capable PyTorch installation. The author run used Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu; the bundle records those versions for reproduction.

From that directory:

```bash
python memory_mechanisms.py
python row_sequence_models.py
python author_calculations.py
```

The first command produces the exact cell traces and comparison checks. The second trains the six predeclared small models, writes per-epoch loss curves and confusion matrices to `row-sequence-results.json`, and saves selected parameters and inspection arrays to `row-sequence-fits.npz`. The third loads those saved fits to calculate edited-image investigation fixtures; it does not train again. Everything uses the local data; no model download or network call occurs in these programs.

The complete source is included with this lesson. Start by reading `ScalarMemory.forward`, `MatrixMemory.forward`, and `DigitReader.forward`; then follow `main` through fitting, checkpoint selection and held-out assessment. The source uses descriptive state names and explicit loops so that each formula above can be located. It is not presented as a high-performance GPU kernel.

Here is a small standalone scalar trace to reproduce the first calculation before running training:

```python
import math

values = [0.2, -0.6, 0.8]
write_logs = [math.log(weight) for weight in [1.0, 3.0, 9.0]]
cell = normalizer = log_scale = 0.0
for value, write_log in zip(values, write_logs):
    forget_log = math.log(0.5)
    new_scale = max(forget_log + log_scale, write_log)
    retain = math.exp(forget_log + log_scale - new_scale)
    write = math.exp(write_log - new_scale)
    cell = retain * cell + write * value
    normalizer = retain * normalizer + write
    hidden = 0.75 * cell / normalizer
    log_scale = new_scale
    print(f"{hidden:.6f}")
```

Expected output:

```text
0.150000
-0.364286
0.443023
```

This little program computes a mechanism with hand-chosen gates. The training program learns its gates from labeled examples. Keeping those two activities separate helps you identify whether a failure comes from the mathematics, an implementation, or a learned model's behavior.

## 7. Read real handwritten digits one row at a time

The UCI Optical Recognition of Handwritten Digits data contain 8×8 grids of integer values from zero to 16. Each cell counts on-pixels in a 4×4 block of an original normalized bitmap. The 64 input values are followed by a class label from zero to nine. Our model divides inputs by 16 and treats the eight horizontal rows as eight sequence steps. This scan order is a modeling decision; these are not measured time-series samples. [Dataset, acquisition and license](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

[Figure X14: one real 8×8 image, the numeric first row, its normalized input vector and the eight-step scan. Credit E. Alpaydin, C. Kaynak and UCI; identify the displayed training-file source ID.]

The source has 3,823 training images from 30 writers and 1,797 test images from 13 different writers. Per-writer identifiers are not included in the retained rows, so our internal fitting/validation split is a row split within the original training data. All 5,620 feature vectors are distinct.

For each class, a fixed random permutation selects 100 fitting images and the next 30 validation images: 1,000 fit, 300 validation. The remaining 2,523 training-file rows are unused. The 1,797 original test rows form the final assessment. Full source IDs and file hashes are saved. No labels are passed into inference.

[Figure X15: source split → classwise fit/validation/unused roles, with the separate-writer test file on a distinct branch. Arrows show parameters learned from fit, checkpoint selected by validation, and final measurement on test.]

All three models use Adam at learning rate `0.003`, 150 full-batch updates, and gradient-norm clipping at one. Each run saves the epoch with lowest clean validation cross-entropy. Seeds 19 and 43 were specified before evaluating their results. Every model is also assessed after reversing the order of its eight input rows, while retaining the same final digit label. This is a fixed stress test of order dependence; no model is trained on reversed images here.

| Model | Seed | Parameters | Selected epoch | Validation errors / 300 | Clean test errors / 1,797 | Reversed-row test errors / 1,797 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 19 | 4,138 | 85 | 36 | 267 | 1,173 |
| LSTM | 43 | 4,138 | 111 | 29 | 156 | 1,048 |
| sLSTM | 19 | 4,074 | 138 | 22 | 123 | 1,136 |
| sLSTM | 43 | 4,074 | 142 | 22 | 137 | 1,136 |
| mLSTM | 19 | 2,828 | 104 | 67 | 419 | 921 |
| mLSTM | 43 | 2,828 | 93 | 65 | 402 | 865 |

These are executed CPU results for the supplied program, not values copied from a paper. In this setup the scalar variant performs well, while the narrow single-head matrix variant makes substantially more clean errors. The LSTM's two seeds also differ noticeably. The matrix model has fewer parameters, different state geometry and a particular initialization; this experiment does not isolate one universally superior cell design. Increasing its dimensions or changing training would be a new experiment with its own validation protocol.

Reversing the rows hurts all three. In a causal reader, later evidence is processed after earlier evidence through learned transitions. Fixed-size state does not make the function invariant to order. The matrix model's smaller additional reverse penalty does not establish better overall robustness: its clean performance is already weaker.

[Figure X16: paired error-count dots for each seed and condition, with the exact denominator printed. Add a selected run's validation cross-entropy curve with its chosen epoch; do not compare the pre-update fit curve and post-update validation curve as if sampled at identical parameters.]

### Watch evidence arrive, and change it yourself

For the worked image at training-file source ID 3451, the true digit is one. The selected seed-19 scalar model's per-row predicted classes are `[1,2,2,1,1,1,1,1]`. After three rows it favors two; by the fourth it favors one. If rows six through eight are replaced by zeros, the first five predictions remain identical, then the trace becomes `[7,9,9]`. The final class is wrong.

These intermediate predictions come from applying the classifier at each prefix. The model was trained only on final-row loss. A changing prefix score is a view of its computation, not a separately validated early-exit classifier or a calibrated measure of understanding.

[Figure X17: the worked digit, current scan line, class-score trajectory and state summary. The clean and edited paths coincide until the first changed row. Reveal the true label after the prediction step in the investigation.]

The fresh investigation starts from a different image, source ID 187. Before running it, choose which rows to alter and predict whether the final class or only intermediate confidence will change. Edit the actual 0–16 pixel values, not just a decorative corruption slider. Compare a scalar-state trace with the matrix state's changing key-by-value grid. These learned channels do not have inherent names such as “loop detector”; any such interpretation would require separate evidence.

[Investigation XD: editable 8×8 pixel canvas with numeric keyboard editing, eight-step scan, selected model/seed, class scores, state view and carry/reset comparison.]

A useful computational null is to process rows one through three, retain the complete recurrent state, and continue with rows four through eight. The logits should match processing all eight at once to floating-point tolerance. An intentional reset before row four can change them. A blank image need not produce uniform scores because the model has learned biases and its transitions still run; in seed 19, the scalar model predicts nine on the blank fixture. That is a model output without digit evidence, not a genuine recognition success.

For a practical extension, define an occlusion or scan-order augmentation using fitting data, select any new settings with validation, and assess once on a reserved test protocol. Do not tune an augmentation on the test errors in the table and then call the same table an untouched final test.

## 8. Distinguish a cell family from a published model

**Deeper branch.** The term xLSTM spans more than one architecture revision. Use the cell equations, block arrangement, gate choices and checkpoint configuration when identifying a model.

The original 2024 work combines scalar and matrix blocks in different proportions. Its scalar block includes recurrent memory mixing and a feed-forward expansion after the sequence operation. Its original matrix block expands before the memory, with local convolution/projection details and gating around the matrix read. A notation such as `xLSTM[a:b]` describes the ratio of matrix to scalar blocks; it does not require both cell types inside every block. [Original paper and cell/block diagrams](https://arxiv.org/html/2405.04517v2).

The March 2025 xLSTM-7B release uses 32 matrix-memory blocks, not a mixed stack with a few scalar blocks. Its post-expansion block design uses RMS normalization and a SwiGLU feed-forward branch. The released configuration has embedding width 4,096, eight matrix heads, key width 256 and value width 512 per head. It uses a sigmoid forget gate and an exponential write gate, with other gate/normalization choices specified in the model. The paper reports pretraining on 2.3 trillion tokens at an 8,192-token context. These are characteristics of that release, not requirements for every xLSTM. [xLSTM-7B paper](https://arxiv.org/html/2503.13427v1), [released configuration](https://huggingface.co/NX-AI/xLSTM-7b/blob/main/config.json).

[Figure X18: a version-aware block comparison: original scalar block, original matrix block, and 7B matrix block. Match dimension expansion location and identify omitted details rather than drawing three identical boxes.]

There is also a later sigmoid-input matrix variant. It updates `C=σ(b)C_old+σ(a)kv^T`, reads `C^T(q/√d_k)`, and applies a normalization layer and output gate. It drops the exponential variant's `n` and `m` states. For very negative inputs, sigmoid and exponential write activations are close, but the two full operators are not universally identical. Normalization can reduce sensitivity to overall scale, while its epsilon matters for very small inputs. This is a purposeful architectural variant, not permission to silently substitute sigmoid into a checkpoint trained with different equations. [Tiled Flash Linear Attention, §§4.1–4.2](https://arxiv.org/html/2503.14376v2).

### Count state before claiming it is small

For `L` layers, `H` heads and float32 state, the matrix storage is

\[
4LHd_kd_v\ \text{bytes per sequence}.
\]

For the stated 7B configuration, that is `4×32×8×256×512 = 134,217,728` bytes, or **128 MiB**. Adding one key-width normalizer and one scale value per head gives about **128.251 MiB**. Model weights, surrounding activations, training gradients, batching and implementation workspaces are additional allocations.

This state does not grow with the number of already-processed tokens. It can nevertheless be substantial, especially across many concurrent sequences. An attention key/value cache grows with cached length and its own head configuration. A comparison must specify both configurations and dtype; the word “constant” alone gives no crossover point.

[Figure X19: an exact memory accounting diagram and a length-axis schematic contrasting fixed state with a growing cache. Use a schematic curve only unless an explicit attention configuration is entered; do not fabricate measured device memory.]

### Why chunks and kernels still matter

For fixed head dimensions, recurrent matrix updates and reads cost on the order of `T d_k d_v` over a sequence. Dense causal comparison costs on the order of `T²(d_k+d_v)` and stores quadratic intermediates if materialized. A chunked schedule with chunk length `b` combines approximately `T b(d_k+d_v)` local comparison work with matrix-state read/write work on the order of `T d_k d_v`. It does not divide every matrix-memory cost by the chunk size.

Wall-clock speed also depends on where data live and which operations the hardware performs efficiently. Processing one token at a time provides little parallel sequence work. Chunking can use matrix multiplications, while storing too many boundary states consumes memory bandwidth. Tiled Flash Linear Attention separates larger logical chunks from smaller hardware tiles and handles rescaling as contributions are accumulated. Its reported speedups belong to its particular kernels, hardware and shapes; our CPU teaching loop measures none of them. [TFLA algorithm and implementation](https://arxiv.org/html/2503.14376v2), [official kernels](https://github.com/NX-AI/mlstm_kernels).

A cached attention decode step attends to `T` existing keys and is linear in that cached length for that step; full-sequence dense attention is quadratic in sequence length. These are different workloads. Fixed recurrent state gives an attractive decode memory pattern, but projections, feed-forward work, batching and kernel launch overhead still matter.

A subsequent scaling-law study compares dense Llama-2-style models and xLSTM across a stated range of model sizes and token budgets. It fits both equal-compute profiles and a parametric loss surface, accounting for context-dependent sequence-mixing cost. Its reported favorable frontier for xLSTM is evidence for that controlled recipe and range. It does not prove superiority over every transformer variant, task or serving deployment, or guarantee that a fitted curve remains valid arbitrarily far beyond the experiments. [Study methodology and released runs](https://arxiv.org/html/2510.02228v2).

## 9. Useful applications beyond a text decoder

The shared design question is how inputs become a sequence and what the state must preserve for the output task. Three adaptations make that question concrete.

**Images: scan patches in more than one direction.** VisionLSTM turns image patches into vectors and uses matrix-memory blocks with alternating scan directions. Because the entire image is available for classification, later blocks may read it in the reverse spatial order without violating a future-prediction requirement. This changes which patches influence each representation. Our eight-row digit reader illustrates order sensitivity but is not a reproduction of that patch architecture. A useful transfer question is whether the task needs a global image label or a representation at every location; the readout must match. [VisionLSTM paper](https://arxiv.org/abs/2406.04303), [author project](https://nx-ai.github.io/vision-lstm/).

**Forecasts: tell the model what is missing.** TiRex uses scalar-memory blocks on patches of normalized time-series values. Its input includes a presence mask; future patches are represented as missing inputs while the state continues forward. Training includes contiguous masked patches, and the output predicts quantiles rather than only one number. The interesting mechanism is the alignment between a training-time missing interval and an inference-time future interval. It is not equivalent to inserting a plausible-looking point forecast as if it had been observed. Quantiles still need empirical coverage and forecasting validation. [TiRex architecture and masking method](https://arxiv.org/html/2505.23719v2).

A different adaptation, xLSTM-Mixer, begins with shared linear forecasts and uses a scalar-memory mixer on encoded variate information. It demonstrates that the recurrent axis itself is a design choice: recurrence can mix related series rather than simply scan raw time samples. The forecast objective, available covariates and normalization boundary remain essential. [xLSTM-Mixer method](https://arxiv.org/html/2410.16928v3).

**Offline control: keep the episode boundary meaningful.** Large Recurrent Action Models encode observations and conditioning information, process the history recurrently, and predict actions from offline trajectories. In the cited design, observations, desired returns and previous rewards are available before the current action; the current action's future reward is not an input. The model can carry a bounded state during an episode and reset it for a new episode. A recurrent architecture does not make an offline policy safe under unfamiliar states or turn a benchmark action score into a real-robot deployment result. [LRAM method](https://arxiv.org/html/2410.22391v2).

[Figure X20: three task-specific input/state/output flows: image patches with alternating arrows, forecast patches with explicit missing masks, and an episode timeline with the reward arriving after its action. Label which inputs are available at decision time.]

These applications do not establish that one memory variant is best everywhere. They show how changing the input representation, recurrence axis, loss and readout can make the same underlying memory idea useful in a different setting.

## 10. Diagnose the failure at the right level

When a result looks wrong, locate the layer of the problem before changing the model size.

| Symptom | First useful check | What the check distinguishes |
| --- | --- | --- |
| Raw and stabilized scalar outputs differ | Rescale content and normalization together; check empty-state initialization | Representation error versus a different prior |
| Matrix outputs differ only for small query alignment | Compare the raw unit floor with the scaled `exp(−m)` floor | An operator change hidden by ordinary examples |
| A matrix is being displayed transposed | Label key and value dimensions and trace one outer product/read | Shape convention versus incorrect multiplication |
| A future edit changes an earlier output | Check causal masks, input projections and sequence indexing | Leakage versus legitimate later-state change |
| Splitting a sequence changes its answer | Carry every state component and preserve step order | A boundary reset versus the same computation |
| Matrix training is worse than a scalar baseline | Inspect dimensions, normalization, gradients, fit/validation curves and objective | A poor configuration versus an architecture theorem |
| A blank input gives a confident class | Inspect biases and the decision rule; assess calibration separately | A model preference versus actual input evidence |
| A long input runs but retrieval fails | Measure the task at that length with controlled distractors | Executability versus usable memory |

Finite-precision arithmetic, denominator branches and bounded gates deserve targeted tests. They do not justify inventing universally safe numerical thresholds. The supplied programs use controlled float32 training and float64 mechanism checks; a mixed-precision GPU implementation needs its own affected numerical checks.

## 11. Practice: transfer the mechanism

Try each problem before opening its hint or solution. The first four check calculation and representation. The next three check model and sequence reasoning. The final three ask you to design or diagnose a realistic experiment.

### 1. A different scalar history

Start with an empty ledger, candidates `[-0.5,0.25,1]`, writes `[2,1,4]`, retention `[0.5,0.5,0.25]`, and output gate one. Calculate the final content, mass and output. Which candidate has the greatest final influence?

<details><summary>Hint</summary>

At each step, multiply both old totals by that step's retention before adding the new content and weight. Alternatively compute each write's surviving weight at the final step.

</details>

<details><summary>Solution</summary>

After the first write, `c=−1,n=2`. After the second, `c=−0.25,n=2`. After the third, `c=3.9375,n=4.5`, so the output is `0.875`. Final write weights are `[0.25,0.25,4]`. The latest candidate supplies `4/4.5` of the mass and has the greatest influence. Its value is not multiplied by the step-three forget gate because it was not in the old state.

</details>

### 2. A normalization prior

For one candidate `0.8` with unit write, unit retention and output gate one, compare initialization `(c_0,n_0)=(0,0)` with `(0,1)`. Are the different answers numerical errors?

<details><summary>Hint</summary>

Interpret the second initialization as an extra piece of zero-valued evidence.

</details>

<details><summary>Solution</summary>

The empty ledger returns `0.8/1=0.8`. The second returns `0.8/2=0.4`. Both follow their stated recurrences, but the second contains a zero-valued prior with unit mass. They implement different memory semantics. A programmer must not insert `n_0=1` while claiming exact equivalence to an empty ledger.

</details>

### 3. Stabilize a matrix floor

A one-step memory has key `0.2`, value three, query `0.5`, and write log-weight `ln 5`. Compute its raw read and its correctly stabilized read. What does a stabilized floor of one incorrectly produce?

<details><summary>Hint</summary>

The raw matrix equals three and its normalizer equals one. After rescaling by five, the fixed raw floor must be divided by five too.

</details>

<details><summary>Solution</summary>

Raw numerator is `3×0.5=1.5`; signed mass is `1×0.5=0.5`; denominator is one, giving `1.5`. Scaled `C'=0.6,n'=0.2,m=ln 5` gives numerator `0.3` and denominator `max(0.1,0.2)=0.2`, again `1.5`. Keeping a floor of one would give `0.3`. This case deliberately activates the floor, which a large-alignment test might miss.

</details>

### 4. Can a head forget only one address?

A matrix head contains two orthogonal keys with useful values. At the next step you want to halve the old contribution for key A while leaving the old contribution for key B unchanged. Can the head's single scalar forget gate do this by itself? Propose a meaningful architectural or input change.

<details><summary>Hint</summary>

Write the old-state term as one scalar multiplying the entire matrix.

</details>

<details><summary>Solution</summary>

No: `f C_old` scales both old contributions by the same factor. Separate heads could place them under different scalar gates if the learned representation supports that separation. A structured/vector forget operator or a targeted corrective write would be another mechanism, but changes the operation or relies on knowing the needed correction. The matrix's addressability at read time is not selective deletion at write time.

</details>

### 5. A chunk-normalization trap

At one output, the old-state numerator is two with signed mass two, and the local numerator is three with signed mass negative one. Compare normalizing the combined result with normalizing the two parts separately. Use a raw denominator floor of one.

<details><summary>Hint</summary>

Absolute value and the maximum are nonlinear. They cannot be distributed across a sum.

</details>

<details><summary>Solution</summary>

Correct combination gives numerator five, signed mass one and output five. Separate normalization gives `2/max(2,1)+3/max(1,1)=1+3=4`. A chunk boundary must not change where normalization occurs. Align old/local scales, combine their numerator and signed mass, and then apply the shared denominator.

</details>

### 6. What can an early prediction see?

A digit reader has processed rows one through five. You edit row eight, rerun from the beginning, and see its row-three logits change. Name two possible implementation errors. Would reversing every input row be a valid null test for those logits?

<details><summary>Hint</summary>

Check both the recurrent computation and preprocessing that might combine rows before the recurrence.

</details>

<details><summary>Solution</summary>

A noncausal mixing operation or an incorrect future mask could leak row eight. Preprocessing that recomputes a per-image statistic using all eight rows could also change earlier inputs. Our fixed division by 16 avoids that particular dependency. Reversing all rows is not a null: it changes which evidence arrives in the first three steps. Editing only a future row under fixed causal preprocessing is the appropriate prefix-invariance check.

</details>

### 7. Count two different states

A model has 12 matrix-memory layers, four heads per layer, key width 32, value width 64, and float32 state. Calculate matrix bytes per sequence and then add `n` and `m` for the exponential variant. Does processing twice as many tokens double this persistent state?

<details><summary>Hint</summary>

There are `12×4` heads. Each stores `32×64` matrix values, 32 normalizer values and one scale value.

</details>

<details><summary>Solution</summary>

Matrix storage is `12×4×32×64×4=393,216` bytes, or `384 KiB`. Adding normalization and scale gives `12×4×(2048+32+1)×4=399,552` bytes, or `390.1875 KiB`. Persistent recurrent state does not double with processed length at fixed dimensions. Training activations and other execution buffers are separate and may depend on sequence length.

</details>

### 8. Improve the weak matrix digit model honestly

You want to try a larger matrix head and row-order augmentation after reading the result table. Specify which data roles make decisions, which outcome you will optimize, and how you will avoid presenting this development process as an untouched new test.

<details><summary>Hint</summary>

Distinguish fitting parameters, selecting settings and estimating final performance. Previously inspected test outcomes cannot become unseen again.

</details>

<details><summary>Solution</summary>

Fit candidate models on fitting images, including augmentation generated only from those images. Choose the head sizes, augmentation policy and stopping epoch with a declared validation objective, such as clean validation cross-entropy plus a separately declared stress requirement. Keep all candidate and selection decisions recorded. Because the published test table has already informed this development, describe the result as follow-up analysis on that benchmark; use a genuinely new reserved assessment set or a predeclared external protocol for a fresh final generalization claim. Report parameter counts and clean/stress performance together.

</details>

### 9. Design a memory test instead of a speed claim

Two models can process 100,000 tokens. One has fixed recurrent state and the other a growing cache. Design a small controlled retrieval task that tests useful memory without confusing it with execution success.

<details><summary>Hint</summary>

Vary one demand on memory at a time: delay, distractors, number of associations or updates to an address.

</details>

<details><summary>Solution</summary>

Generate sequences that introduce random key/value pairs and later query a specified key. Keep key/value distributions and output scoring fixed. Vary delay separately from the number of intervening unrelated pairs, then add a condition where a key receives a new value and the correct answer is its latest value. Prevent accidental answer cues in position or token frequency. Report retrieval accuracy by condition and length, alongside any measured memory/latency under explicit hardware and batch settings. A model finishing the input is not evidence that it retrieved the association correctly; a small random-key task is still not every long-context application.

</details>

### 10. A forecast mask and an episode boundary

A forecasting system inserts zeros for unknown future values but has no presence mask. A control system carries its recurrent state from one independent episode into the next. Explain the information problem in each, and propose a corrected contract.

<details><summary>Hint</summary>

Ask whether a zero was observed and whether previous state belongs to the same causal history.

</details>

<details><summary>Solution</summary>

Without a mask, a forecast model cannot directly distinguish an observed zero from an unknown value represented by zero. Provide presence information and train the model under the missing-input pattern used during forecasting, while fitting normalization on information available at forecast time. For independent control episodes, reset every required state component and any other cache at the episode boundary. Carrying state is appropriate only when the task explicitly defines a continuous history; otherwise it introduces irrelevant prior-episode information and can invalidate evaluation.

</details>

## 12. References and another way to learn

Use these resources for their different teaching roles. The calculations and small experiments above are self-contained; no video is required to understand an equation.

- [Original xLSTM paper, Beck and colleagues](https://arxiv.org/html/2405.04517v2). Advanced primary reference for both cell families. After the scalar and matrix examples, inspect the main cell equations, then Appendix A's vector forms and block diagrams. Its benchmark settings and architecture variants should not be silently transferred to later releases.
- [xLSTM-7B paper](https://arxiv.org/html/2503.13427v1). Read the architecture changes alongside the [released configuration](https://huggingface.co/NX-AI/xLSTM-7b/blob/main/config.json); this is the best route for understanding why a family diagram and a specific checkpoint may differ.
- [Official xLSTM repository](https://github.com/NX-AI/xlstm) and [7B model card](https://huggingface.co/NX-AI/xLSTM-7b). Practical integration references with current backend and loading instructions. The model card provides a Transformers loading route as checked in September 2026. Check the exact package revision and checkpoint license before using it; the released weights have the NXAI Community License. No pretrained-model execution is needed for this lesson's offline program.
- [Tiled Flash Linear Attention](https://arxiv.org/html/2503.14376v2) and [kernel repository](https://github.com/NX-AI/mlstm_kernels). Read the recurrent/chunk equations first, then the two-level tiling and sigmoid-variant sections. GPU kernel details are an optional branch after the mathematical operator is clear.
- [Maximilian Beck's author presentation](https://www.youtube.com/watch?v=KjvCtslDJv0), with [selected 2026 defense slides](https://maxbeck.ai/resources/talks/2026-03-PhD_Defense_Beck_share_selected.pdf) and [author talks index](https://maxbeck.ai/talks/). An alternate visual route through recurrent memories, kernels and the later work. The author-linked recording was identified and selected slide text was reviewed for this lesson; the recording was not watched and no unverified timestamps are supplied.
- [xLSTM scaling-law study](https://arxiv.org/html/2510.02228v2) and [released analysis materials](https://github.com/NX-AI/xlstm_scaling_laws). Useful for practicing critical reading of equal-compute comparisons. Separate fitted curves, measured training runs and extrapolations; the notebooks are optional and were not run for this lesson.
- [VisionLSTM author project](https://nx-ai.github.io/vision-lstm/), [TiRex method](https://arxiv.org/html/2505.23719v2), [xLSTM-Mixer method](https://arxiv.org/html/2410.16928v3), and [LRAM method](https://arxiv.org/html/2410.22391v2). These are application-specific readings. Focus on the sequence axis, what information is available at a step, the task loss and the readout before looking at benchmark tables.
- [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). Original source and attribution for the executable experiment. The lesson's row-scanning protocol, split assignments, trained models and stress results are its own derived work.

You are ready to move on when you can trace one scalar and one matrix update, explain the changed-scale denominator, distinguish a computational schedule from an architectural change, and interpret the real experiment's success and failure without overclaiming.

The next topic in this module is [Hyena: Long-Convolution Models](/learn/path/full-curriculum/hyena-long-convolution-models?module=deep-learning-fundamentals). It asks how long filters and input-dependent gates can mix a sequence without either an explicit all-pairs attention matrix or this particular recurrent memory cell. For comparison, revisit the earlier [RWKV and linear-attention models](/learn/path/full-curriculum/rwkv-linear-attention-models?module=deep-learning-fundamentals) and [state-space models](/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals); their state updates share some computational ideas while retaining different operators.
