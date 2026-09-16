# Modern Hopfield Networks

A smudged handwritten digit still contains clues: the bend of a stroke, an opening in a loop, the position of a vertical line. Suppose we keep examples of handwriting and ask a model to reconstruct something useful from those clues. The interesting question is not only which example receives the highest score. It is whether repeatedly using the retrieved information improves the cue, whether several examples should contribute, and how to recognize an incorrect reconstruction.

A **Hopfield network** is an associative memory: you give it content that resembles a memory, and its dynamics attempt to complete or refine that content. A conventional database uses an address such as record 42. Associative memory uses a cue such as “the shape with a loop and this downward stroke.”

This lesson moves from four binary features to continuous memories, then to a small trained handwriting classifier. You will calculate an update, explain its energy change, distinguish a retrieved vector from its associated label, and investigate why an apparently cleaner image can represent the wrong digit.

The preceding [Spectral Normalization & Gradient Penalty](/learn/path/full-curriculum/spectral-normalization-gradient-penalty?module=deep-learning-fundamentals) lesson asked how strongly a function can change its output when its input changes. Here that becomes a concrete question about retrieval: does a small cue change shrink after another update, or push the state toward a different memory?

**First pass:** follow §§1–6 and the investigations, then try exercises 1–7. The explicitly optional branches in §7 develop capacity, higher-order memories and energy-derived architectures. The local calculations use vectors, dot products, weighted averages and derivatives, all refreshed where they enter.

## 1. A memory is more than a label

Imagine storing the four-feature pattern

**[on, on, off, off] = [1, 1, −1, −1].**

An arriving cue is [1, −1, −1, −1]: one feature is wrong. We want the two left features to support one another and the two right features to oppose them. Those relationships can correct a feature without being told which feature was corrupted.

There are three different success criteria:

| Goal | What counts as success? | What must be available? |
| --- | --- | --- |
| Exact stored-pattern recall | Output equals a particular stored vector | That vector was stored |
| Reconstruction | Output approaches the clean source under a stated distance | Clean reference for evaluation |
| Classification | Associated class is correct | Labels attached to reference examples |

A new person's handwritten “1” was never in our memory bank. Returning the pixels of an older “1” may classify it correctly while changing its handwriting. A convex combination of several examples may improve average reconstruction error without matching any particular stored image.

[Figure 1: address lookup and cue lookup, followed by the three separate success checks.]

A useful geometric refresher: a dot product adds coordinate-wise agreements. For q = [1, 0], memories [1, 0] and [3, 1] score 1 and 3. The second wins by dot product even though the first has Euclidean distance zero. If all memories have equal norm, minimizing squared distance to a fixed cue is equivalent to maximizing dot product, because

||q − xᵢ||² = ||q||² + ||xᵢ||² − 2qᵀxᵢ.

When norms differ, the middle term matters. Normalizing nonzero vectors to unit length makes the dot product a cosine similarity. That is a modeling choice: it removes information carried only by magnitude.

[Figure 2: the two memory arrows, their lengths, dot scores and distances. Keep the larger wrong-direction advantage visible.]

## 2. Classical Hopfield memory: correct one feature at a time

### Store relationships

Let P binary memories be rows of X, each containing d values in {−1, +1}. The Hebbian storage rule forms

W = XᵀX / d, then sets Wᵢᵢ = 0.

“Hebbian” here means that features with matching signs contribute a positive connection and opposite signs contribute a negative connection. Summing outer products adds each memory's suggested relationships. The connections are symmetric.

For our one memory x = [1, 1, −1, −1], the matrix is

| W | feature 1 | feature 2 | feature 3 | feature 4 |
| --- | ---: | ---: | ---: | ---: |
| feature 1 | 0 | 0.25 | −0.25 | −0.25 |
| feature 2 | 0.25 | 0 | −0.25 | −0.25 |
| feature 3 | −0.25 | −0.25 | 0 | 0.25 |
| feature 4 | −0.25 | −0.25 | 0.25 | 0 |

A neuron does not vote for itself: removing the diagonal makes the local field describe the other features' evidence. A nonzero diagonal can change update behavior; it does not simply force every state to become all ones.

[Figure 3: four feature nodes and signed connections; pairwise sign products assemble one row of W.]

### Read relationships

The **local field** at coordinate i is hᵢ = Σⱼ Wᵢⱼsⱼ. If it is positive, set sᵢ to +1; if negative, set it to −1. At exactly zero, retain the current value.

Update one coordinate, then use the changed state when updating the next coordinate. This is an **asynchronous update**. A **sweep** visits every coordinate once.

Starting from [1, −1, −1, −1], feature 1 sees field +0.25 and stays +1. Feature 2 then sees +0.75 and changes to +1. Features 3 and 4 already agree with their negative fields. After that sweep the state is the stored pattern.

[Figure 4: before/field/after trace. The second update highlights the three votes that repair feature 2.]

### Why this particular update settles

Define an energy, a scalar score assigned to the whole configuration:

E(s) = −½sᵀWs.

This is a mathematical objective, not an amount of electrical energy measured in joules. Positive connections prefer equal signs; negative connections prefer opposite signs. Both preferences lower E.

For symmetric W with zero diagonal, changing only coordinate i gives

E(s after) − E(s before) = −(sᵢ after − sᵢ before)hᵢ.

Our corrected coordinate changes from −1 to +1 with field +0.75, so ΔE = −2 × 0.75 = −1.5. The complete cue has energy 0; the recovered state has energy −1.5.

Every actual flip with a nonzero field decreases energy. Unchanged coordinates leave it constant. There are finitely many binary configurations, so with fair repeated coordinate visits and this tie rule the process reaches a state with no energy-lowering single-coordinate flip.

This is a local guarantee. It does not say the state is the desired memory, the closest memory, or the global minimum.

[Figure 5: energy staircase labeled by individual coordinate updates, including the flat steps.]

**Why update order matters.** From [−1, −1, −1, −1], the forward order 1, 2, 3, 4 reaches [1, 1, −1, −1]; the reverse order reaches [−1, −1, 1, 1]. Both have energy −1.5. With no thresholds, E(s) = E(−s), so the inverse pattern is equally plausible to this energy.

If all coordinates change simultaneously, the proof above no longer applies: each changed field was computed against the old state. For W = [[0, 1], [1, 0]], synchronous updates alternate [1, −1] → [−1, 1] → [1, −1]. A stopping limit is not proof of convergence.

[Investigation A: edit a binary cue and its stored pattern, record a predicted repair, then compare individual updates and alternate visit orders.]

### When memories interfere

With many stored patterns, W combines competing suggestions. Some errors settle into a **spurious state**, an attractor that was never deliberately stored. A mixture of correlated memories can be stable. Exact duplicates also alter the strength of their contributions.

The often quoted 0.138d capacity concerns a particular random-pattern, Hebbian, large-system retrieval regime allowing small errors. It is not a universal hard limit for every learning rule, every finite memory bank, or every definition of successful recall. Exact recovery of most versus every random stored pattern gives different asymptotic conditions. We return to those distinctions in §7.

A small experiment is easier to interpret than an unexplained theoretical line. With d = 64, eight independently generated banks at each size, and exactly six randomly flipped cue bits, our recorded run gives:

| Patterns per bank | Stored patterns tested across 8 banks | Exact fixed states | Exact recalls from damaged cues |
| ---: | ---: | ---: | ---: |
| 2 | 16 | 16 | 16 |
| 6 | 48 | 48 | 47 |
| 10 | 80 | 61 | 56 |
| 16 | 128 | 49 | 29 |
| 24 | 192 | 12 | 2 |

“Exact fixed state” means the original pattern would retain every coordinate under the tie-preserving local rule. Recall starts with damage and uses sequential updates. The two columns ask different questions, and neither eight-bank experiment estimates a universal capacity constant.

[Figure 6: paired proportions with count labels, separately identified fixed-state and damaged-cue criteria; no theoretical curve fitted to these samples.]

## 3. Continuous memories: score, distribute weight, retrieve

The modern continuous construction keeps the memory vectors explicitly. X now has P rows and d real-valued columns. A cue q has d coordinates.

1. **Score:** s = Xq gives one dot product per memory.
2. **Sharpen:** pᵢ = exp(βsᵢ) / Σⱼ exp(βsⱼ).
3. **Read:** q next = Xᵀp = Σᵢ pᵢxᵢ. We call this complete read operation F(q).

The positive number β is the **inverse temperature**. Increasing β increases the relative advantage of higher-scoring memories. The softmax weights are nonnegative and sum to one, so retrieval is a weighted average inside the memories' convex hull.

These normalized weights are an allocation of attention. They are not automatically calibrated probabilities that a memory is correct.

### Work through two memories

Keep x₁ = [1, 0], x₂ = [−1, 0], and start at q = [0.2, 0.4]. The scores are [0.2, −0.2]. With β = 2, the logits are [0.4, −0.4], giving weights approximately [0.689974, 0.310026]. Thus

q next = [0.689974 − 0.310026, 0] = [0.379949, 0].

The vertical component disappears because neither memory contains one. The horizontal component becomes more positive, but it does not jump to +1.

[Figure 7: dot-product scores feed proportional weight bars; the weighted point appears on the segment connecting the memories.]

Apply the same update again:

| Update count | Horizontal coordinate | Vertical coordinate |
| ---: | ---: | ---: |
| 0 | 0.200000 | 0.400000 |
| 1 | 0.379949 | 0 |
| 2 | 0.641017 | 0 |
| 3 | 0.857026 | 0 |

For these two memories the whole horizontal recurrence is

qₓ next = tanh(βqₓ).

Here tanh(z) = [exp(z) − exp(−z)] / [exp(z) + exp(−z)]. The two opposing softmax contributions reduce to this expression.

At β = 2 the positive stable fixed point is near 0.9575, not exactly the stored coordinate 1. A **fixed point** is a state the update leaves unchanged. A **stored pattern** is a row of X. They need not be identical at finite temperature.

Now reduce β to 0.5. The same cue's horizontal coordinates become 0.099668, 0.049793 and 0.024891. Repetition approaches the middle, averaging the memories instead of selecting one.

[Figure 8: aligned β = 0.5 and β = 2 cobweb plots against the identity line, showing the different fixed points.]

A cue [0, 0.6] gives equal weights and reaches [0, 0]. It stays there even at β = 2. The exact symmetric state remains fixed although a small horizontal disturbance grows. A balanced cue does not acquire evidence about which memory was intended just because we turn up β.

### The energy behind the update

For fixed X and β > 0, write

E(q) = ½||q||² − β⁻¹ log Σᵢ exp(βxᵢᵀq).

We omit constants independent of q; adding them changes neither the update nor energy differences. The log-sum-exp is a smooth version of the largest score. Its negative encourages agreement with memories. The quadratic eventually dominates this term as ||q|| grows, keeping the energy bounded below. More directly, after one update q lies in the finite memory bank's convex hull.

Differentiating gives

∇E(q) = q − Xᵀsoftmax(βXq) = q − F(q).

A stationary point therefore satisfies q = F(q). Writing this equality identifies a fixed-point equation; it does not solve it in one step.

Why does the iteration decrease E? Let g(q) = β⁻¹log Σ exp(βxᵢᵀq). Because g is convex,

g(z) ≥ g(q) + ∇g(q)ᵀ(z − q).

Negate this inequality and add ½||z||². We have built an upper bound on E(z) that touches E at q. Minimizing that quadratic upper bound gives z = ∇g(q) = F(q). Consequently,

E(F(q)) ≤ E(q) − ½||F(q) − q||².

This is a short version of the **concave-convex procedure**: replace the concave part by a tangent upper bound, minimize, repeat. Its direction matters: log-sum-exp is convex; negative log-sum-exp is concave.

For the β = 2 example, energies at updates 0–3 are −0.285550, −0.406684, −0.472651 and −0.505746. The second update changes the state substantially. Calling the first read “exact one-step convergence” would contradict the numbers.

[Figure 9: this specific energy surface, downhill iteration arrows and a matching energy-versus-step strip. Draw the saddle at the symmetric point.]

The [Ramsauer paper](https://arxiv.org/abs/2008.02217) proves convergence properties and much stronger local retrieval results under separation assumptions. “One update” in those retrieval results means reaching a prescribed small error near an associated fixed point, not equality after one update for arbitrary memories and cues.

[Investigation B: move continuous memories and the cue, predict which fixed-point region or mixture will appear, then compare β values. Inspect both positions and actual energy.]

### A useful sensitivity connection

The derivative of retrieval is

J_F(q) = βXᵀ[diag(p) − ppᵀ]X
       = β Covₚ(x).

The covariance measures how much the currently weighted memories disagree. If almost all weight lies on one memory, this local derivative can be small: nearby cues yield nearly the same retrieval. If conflicting memories share weight, a cue change can have a larger effect.

In our two-memory example, F′(0) = β. At β = 0.5, small horizontal errors shrink near zero; at β = 2, they grow. This connects the previous lesson's derivative bounds to an observable attraction or repulsion. A small local derivative near one memory does not establish a global contraction over every cue.


## 4. Attention reads associations as well as memories

A library catalogue separates the description used to search from the information returned. A key might describe a book's subject; the value might be its location. Associative neural memory can use the same separation.

Let K contain P keys of width dₖ, V contain P associated values of width dᵥ, and Q contain B query rows. The read is

A = softmax(βQKᵀ), Z = AV.

| Quantity | Shape | Meaning |
| --- | --- | --- |
| Q | B × dₖ | B requests |
| K | P × dₖ | descriptions used to score memories |
| QKᵀ | B × P | one score per query-memory pair |
| A | B × P | row-normalized allocation of weight |
| V | P × dᵥ | payload attached to each memory |
| Z | B × dᵥ | retrieved payloads |

For one query, K = V = X and β = 1/√d, this is exactly the modern Hopfield update written with row vectors. The transpose convention is the only difference. Our complete CPU comparison against PyTorch scaled dot-product attention returns [0.1688805062, 0.4355881251], with maximum difference 2.78 × 10⁻¹⁷ in float64.

A learned value projection can then transform the key-space read into another space. Ordinary attention permits keys and values to have independent projections. Its association formula still makes sense, but the output is no longer automatically a state update of the same scalar energy we just derived.

For example, keys [1, 0] and [0, 1], cue [0.6, −0.2], and β = 1 give weights [0.689974, 0.310026]. Attach scalar payloads 10 and −2. The returned payload is approximately 6.279694, a scalar. It cannot be fed directly into the two-dimensional key-space energy.

[Figure 10: a two-lane memory table: score against keys, then carry the same weights across to values. Show dimensions at the lane crossing.]

### Classify by combining memory labels

Attach one-hot class vectors as values. For classes A and B, these are [1, 0] and [0, 1]. The read sums weight from all memories labeled A into one number, and from all memories labeled B into the other.

Two B memories can jointly outweigh the largest individual A memory. A classifier that chooses the label of the highest-scoring single memory can therefore disagree with one that sums the class weights. Both decisions should be evaluated against the same task.

If every value equals [4, −2], the output is [4, −2] for every query. The keys can change the weights without changing the retrieved payload. This is why an attention heatmap alone cannot explain all output behavior.

[Investigation C: edit query coordinates, key locations and associated values separately. Predict what will change in the score distribution, key-space read and returned payload.]

### Three useful module designs

**Associate two sets.** Queries come from one input and keys/values from another. Image patches can query a set of candidate object descriptions; a decoder can query encoded input. This connects to [Cross-Attention Architectures](/learn/path/full-curriculum/interleaved-cross-attention-architectures?module=deep-learning-fundamentals).

**Pool a variable-sized set.** Learn one or several query vectors. Each query searches the input's keys and returns a weighted summary. If the input rows are permuted and their keys/values stay paired, the summary stays the same. This makes set pooling appropriate when order is irrelevant. An empty set still needs an explicit policy: softmax over no memories is undefined.

**Learn a fixed prototype bank.** Store a trainable parameter matrix rather than every example. Query projections and prototype coordinates can move during training. Such a bank contains learned representations, not necessarily verbatim training records.

These correspond to the author library's Hopfield, HopfieldPooling and HopfieldLayer abstractions. The [official repository](https://github.com/ml-jku/hopfield-layers) is useful for configuration and examples; its README describes an older Python/PyTorch development environment. The runnable programs here use ordinary NumPy and PyTorch, so understanding the lesson does not depend on installing that research package.

[Figure 11: three small data-flow diagrams with the same key/query/value notation. Mark which tensors depend on each input and which are persistent parameters.]

### Learn where to look

For a single target memory t, the loss L = −log pₜ teaches the query to favor its key. With fixed keys and inverse temperature β,

∂L/∂q = βKᵀ(p − eₜ).

Here eₜ is one at the target's position and zero elsewhere. The gradient subtracts the target key from the current weighted key average.

Take K = I₂, q = [0.2, −0.1], β = 1, and target memory 2. We obtain p = [0.574443, 0.425557], loss 0.854355, and gradient [0.574443, −0.574443]. An update q ← q − 0.1∇L gives [0.142556, −0.042556], reducing loss to 0.789980. The second key becomes relatively easier to retrieve.

This is **parameter or representation learning across examples**. It differs from **state refinement for one cue**, which reduces the fixed-bank energy. In a network q is produced by learned parameters, and backpropagation carries this gradient into those parameters.

For a class represented by several memories, let π_c = Σᵢ:yᵢ=c pᵢ. Training uses −log π_c. Its derivative with respect to scaled logit ℓᵢ = βqᵀkᵢ is pᵢ − rᵢ, where rᵢ = pᵢ/π_c for target-class memories and zero otherwise. The target class receives extra weight without requiring every query to match one arbitrarily selected prototype.

[Figure 12: a query gradient arrow, before/after weights and separate arrows for state refinement versus parameter training.]

## 5. A real memory bank for handwritten digits

We use the [UCI Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). Each image has 8 × 8 cells. A cell records how many pixels were on in a 4 × 4 block of a normalized 32 × 32 handwriting bitmap, so its value is an integer from 0 to 16.

The original split contains 3,823 training images from 30 writers and 1,797 test images from 13 different writers. We preserve that test boundary. The released 65-column files do not include individual writer IDs within each split.

Our deliberately small experiment chooses, separately within each digit class and using seed 113:

- 20 training images as memories: 200 in the bank.
- 80 different training images as fitting queries: 800.
- 30 further training images as validation queries: 300.
- All 1,797 original test images for assessment.

The other 2,523 training rows are unused. We checked that all 5,620 images have distinct exact feature vectors, including across the original train/test boundary. The saved source IDs reproduce every role. Validation comes from the original training writers; test evaluates the different-writer split.

Only memory labels are available to retrieval. Fitting-query labels train the projection; validation labels select settings; test labels score the final comparisons. No validation or test image becomes a stored reference.

[Figure 13: two original writer-group containers, with the training container split into memory, fitting-query, validation and unused roles. Keep test outside all learning arrows.]

### Start with a direct memory baseline

Divide each intensity by 16, then normalize the 64-dimensional image vector to unit length. Compare the query with the 200 normalized memory vectors.

The first baseline returns the nearest memory's label by cosine similarity. The second uses softmax weights and sums them by class. Its β is selected from {4, 16, 64, 256} using clean validation cross-entropy. This selects β = 64. The β grid is predefined; we do not select settings on the corrupted test images.

### Learn a more useful association space

The learned model maps an image through a shared linear projection Wₑ of shape 16 × 64:

kᵢ = normalize(Wₑxᵢ), q = normalize(Wₑx).

It has 1,024 trainable parameters, no bias, and a fixed β = 16. The same projection is applied to both memories and queries, so both sides use the same learned geometry. Softmax operates over 200 memories. Summing weights by memory label gives ten class probabilities.

We fit Wₑ by mean negative log probability of the correct class, using all 800 fitting queries in each update. Adam uses learning rate 0.005 for 100 epochs. After each update we evaluate clean validation cross-entropy and retain the best epoch. We run seeds 17 and 41 to expose initialization variation. Seed 17 is the predefined demonstration run, not a winner chosen from test performance.

These are learned associations in a small supervised model. They are not a reproduction of the Ramsauer paper's benchmarks or proof that a Hopfield layer outperforms other architectures.

[Figure 14: 64 intensity values → shared 16-dimensional projection → unit-length query/key vectors → 200 association weights → ten label sums. A second value branch carries the same weights to the original memory pixels.]

### Run the complete programs

Download [digit_memory.py](./digit_memory.py), [optdigits.tra](./optdigits.tra), [optdigits.tes](./optdigits.tes) and [optdigits.names](./optdigits.names) into one directory. [Data provenance and role details](./data-provenance.md) describe attribution and the saved split. In a Python environment with NumPy and PyTorch:

~~~text
python -m pip install numpy torch
python digit_memory.py
~~~

The program reads local files, builds the roles, evaluates the baselines, trains both projections, and saves results and selected weights. It uses the CPU and downloads no model. The recorded run used Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu. Small floating-point differences can occur in another environment.

The core read, written with explicit shapes, is:

~~~python
import torch
import torch.nn.functional as F

def label_read(query_images, memory_images, memory_labels, projection, beta=16.0):
    # query_images: B x 64; memory_images: P x 64
    queries = F.normalize(projection(query_images), dim=-1)
    keys = F.normalize(projection(memory_images), dim=-1)
    log_weights = F.log_softmax(beta * queries @ keys.T, dim=-1)
    # Sum memory mass by class in log space, retaining small probabilities.
    log_classes = torch.stack([
        torch.logsumexp(log_weights[:, memory_labels == label], dim=1)
        for label in range(10)
    ], dim=1)
    return log_classes, log_weights.exp()
~~~

This function assumes that all ten classes are represented in the nonempty memory bank, as they are in the experiment. The complete file supplies inputs, the projection, objective, optimizer, role construction and result reporting. The log-space summation avoids turning very small class probabilities into an artificial zero before taking the loss.

For the smaller exact mechanisms, save [associative_memory.py](./associative_memory.py) and run it with NumPy and PyTorch available. It reproduces the four-feature recovery, continuous energy trajectories, attention equivalence, gradient example and finite binary-memory scan. Its returned traces distinguish unchanged coordinates from actual flips.

### Read the outcomes

To test sensitivity to missing visual evidence, set columns 4 and 5 of each 8 × 8 query image to zero: 16 cells out of 64. The models were fitted and selected on clean inputs. This is a fixed occlusion stress test, not an alternative training distribution selected after inspecting its errors.

| Model | Clean validation errors / 300 | Clean test errors / 1,797 | Occluded test errors / 1,797 |
| --- | ---: | ---: | ---: |
| Nearest of 200 memories, cosine | 25 | 156 | 663 |
| Weighted labels, fixed pixel geometry, β = 64 | 18 | 136 | 667 |
| Learned projection, seed 17, epoch 100 | 13 | 100 | 564 |
| Learned projection, seed 41, epoch 25 | 16 | 113 | 740 |

The learned geometries improve clean classification in these runs. Occlusion reveals a different story: seed 41 makes more errors than the nearest-memory baseline, even though its clean test result is better. Clean validation quality does not automatically identify the most robust geometry for a new kind of missing input.

The seed-17 fitting queries have zero clean classification errors, yet the test set has 100. The memory bank contains only the 200 reference images; the remaining trainable capacity lies in the projection and its learned similarity function. Perfect fitting-query classification is not a guarantee of generalization.

[Figure 15: paired clean/occluded error bars on the same count scale, with denominators and validation-selection labels. Show both learned seeds and both baselines.]

### A cleaner image can tell the wrong story

The association weights can also retrieve a weighted image: x read = Σᵢ pᵢxᵢ, using the original memory pixels as values. This output is inspectable, but the classifier was trained for labels rather than pixel reconstruction.

Consider validation image **training-source row 2946**, labeled “1.” The clean image receives class-1 mass 0.989362 and predicts 1. After zeroing the two central columns, it predicts 0, with class-1 mass only 0.000141. The strongest clean memory is row 1487, with weight 0.939320. Under occlusion, the top three memories become rows 1786, 699 and 104, with weights 0.285921, 0.183328 and 0.120278.

The retrieved occluded image has mean squared pixel error 0.110814 relative to the original. The damaged input's error is 0.203674. The weighted reconstruction is closer in average pixel error while the class prediction is wrong. This is possible because reconstructing common background and stroke regions can reduce many squared errors while the identifying stroke remains incorrect.

For another image, row 3052, labeled “0,” the same occlusion retains class 0, with mass 0.838620. Its retrieved image error is 0.022476 versus 0.074036 for the damaged cue. A memory method can be helpful on one shape and misleading on another.

[Figure 16: two image stories, each showing clean reference, damaged cue, the three most weighted memories and weighted reconstruction; place class mass and reconstruction error beside the relevant images.]

Across all test images, seed 17's mean squared reconstruction error after occlusion is 0.051198, versus 0.124713 for the damaged input. On clean images, reconstruction error is 0.029621, whereas the original clean input has zero error. Retrieval pulls handwriting toward the memory bank; it is not an identity operation and not an unconditional denoiser.

[Investigation D: edit real cue pixels, record a prediction about the class and retrieved stroke, then inspect the resulting memory distribution and reconstructed image. Compare clean, occluded and blank cues.]

## 6. Build a reliable association system

The most useful diagnostics follow the actual read.

**First inspect the cue and score geometry.** Are vectors normalized consistently? Do norms dominate? Is a zero-filled region being treated as “unknown” even though the model interprets it as background? Our occlusion sets pixels to zero; it supplies no missingness mask. A model trained with missingness indicators or augmentation would be a different experiment.

**Then inspect the competing memories.** Are similar keys attached to different labels? Does one class have many more stored examples, gaining aggregate mass from multiplicity? Duplicating one of two equally scored memories changes its side's total weight from one-half to two-thirds. Duplicate entries are not neutral unless the intended weighting accounts for them.

**Then inspect the payload.** The highest-weight memory can be correct while a label-summed read differs. Identical values can conceal large changes in weights. A visually plausible weighted image can conceal a classification failure. Inspect the task output alongside the attention distribution.

**Finally inspect the learning and evaluation boundary.** A trainable bank, a fixed example bank and a current-input activation bank have different storage and leakage properties. Keeping held-out labels in values makes classification trivially easier in a way unavailable on new input. Changing the bank after fitting changes the predictor and should trigger a new evaluation.

High β approaches an argmax over scores when there is a unique maximum; it preserves ties and cannot repair a wrong score ordering. It can also make gradients through losing memories extremely small. Low β gives broad averaging and may reduce query sensitivity too far. The familiar 1/√d attention scale controls dot-product magnitude under particular component-scale assumptions; it is not a universally optimal temperature for unit-normalized memories.

For implementation, stable softmax subtracts the largest logit. All-masked or empty banks need an explicit result policy. A common additive shift of finite logits leaves the distribution unchanged; masking every logit to negative infinity does not produce a valid distribution.

These checks are useful beyond this named architecture. The earlier [Self-Attention & Multi-Head Attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals) lesson develops attention's full projection and masking mechanics. The present energy interpretation adds a tool for reasoning about particular memory dynamics; it does not substitute for that entire model specification.

## 7. Optional depth: what capacity and energy really promise

### A margin explains when a single read can work

Let xₜ be the desired memory, and suppose every memory has norm at most M. For the current cue q, define the score gap

δ(q) = qᵀxₜ − maxⱼ≠ₜ qᵀxⱼ.

If δ(q) > 0, the desired memory outranks every competitor. Divide the softmax denominator by exp(βqᵀxₜ):

pₜ = 1 / [1 + Σⱼ≠ₜ exp(β(qᵀxⱼ − qᵀxₜ))].

Each competing exponential is at most exp(−βδ), so

pₜ ≥ 1 / [1 + (P − 1)exp(−βδ)].

Set a = (P − 1)exp(−βδ). The total competing weight is at most a/(1+a), giving

||F(q) − xₜ|| ≤ 2M a/(1+a) ≤ 2M(P − 1)exp(−βδ).

The last step uses ||xⱼ − xₜ|| ≤ 2M. This derivation explains the three levers: better separation, sharper temperature, and fewer competitors. A high-dimensional memory bank is not enough if the actual query gives the desired memory a poor score.

For P = 100, δ = 3 and β = 2, the target-weight lower bound is 0.802957. With M = 1, the error upper bound from the tighter expression is 0.394086. These are bounds computed from the assumed gap, not measurements from the handwriting experiment.

[Figure 17: a target score and 99 bounded competing scores, with the combined denominator mass; show why many individually weak competitors can matter.]

A separation statement about the clean memory can be connected to noisy cues. Let Δₜ = ||xₜ||² − maxⱼ≠ₜ xₜᵀxⱼ. If ||q − xₜ|| ≤ r, then by Cauchy–Schwarz,

δ(q) ≥ Δₜ − 2Mr.

Noise can spend the available margin. This does not prove that every cue in every such ball reaches a unique attractor; that stronger conclusion needs fixed-point conditions too. It does show exactly where cue error enters a one-read error bound.

### Three different meanings of “capacity”

1. **How many vectors can the hardware hold?** P rows of width d require Pd stored numbers.
2. **How many memories are stable under a specified update?** This is a mathematical property of the patterns and dynamics.
3. **How many relevant examples improve the real task?** This depends on labels, representations, distribution and evaluation.

For classical Hebbian random binary patterns, exact recall of most memories and exact recall of all memories have different asymptotic scales, n/(2 log n) and n/(4 log n), in the corresponding [McEliece et al. analysis](https://authors.library.caltech.edu/records/q92rz-95p89). Allowing small errors changes the criterion behind the familiar roughly 0.138n regime. None of these constants turns our finite binary scan into a calibrated capacity chart.

For continuous modern Hopfield memory, the canonical paper establishes exponentially growing storage under a specified random-sphere construction and associated attraction regions. One form of its result uses random patterns on a sphere of radius M = K√(d−1), where K > 0 is a scalar radius multiplier (separate from the earlier key matrix), d > 1, inverse temperature β > 0 and failure probability 0 < p ≤ 1. Define

a = 2[1 + ln(2βK²p(d−1))]/(d−1), b = 2K²β/5,
c = b/W₀(exp(a + ln b)).

W₀ is the principal Lambert W function, defined by W(z)exp(W(z)) = z. Under the theorem's condition c ≥ (2/√p)^(4/(d−1)), the storage lower bound has form √p × c^((d−1)/4), with probability at least 1−p. The point of displaying the parameters is to expose what must be specified: d alone does not produce a universal exp(d/2) guarantee for arbitrary learned keys.

Our margin calculation is usually the better first diagnostic. The theorem explains why favorable separated configurations can support many memories; it does not certify every geometry learned from real handwriting.

At P = 1,000,000 and d = 64, an explicit float32 pattern bank needs 256,000,000 bytes, about 244.14 MiB, before other model state. Reading one query against all patterns uses O(Pd) score arithmetic and O(Pd) value aggregation when key/value widths both equal d. B queries cost O(BPd). If the memory and query sets both grow with sequence length T, this becomes quadratic in T.

Fused exact attention can avoid materializing all scores in device memory while still doing the dense pairwise arithmetic. Approximate retrieval, sparse subsets and alternative kernels change other parts of the contract. They must be assessed on their own retrieval errors and costs; an SSM such as Mamba is not simply a softmax Hopfield approximation.

[Figure 18: separate axes for stored bytes, query count and pairwise arithmetic. Use exact formulas, not invented implementation timings.]

### Higher-order binary memory and a useful logical example

Modern associative memory is a family broader than the continuous softmax construction. [Krotov and Hopfield](https://arxiv.org/html/1606.01164v2) study energies of the form

E(s) = −Σμ F(xμᵀs),

with polynomial or rectified-polynomial F. A binary coordinate update can compare the energy with that coordinate set to +1 and to −1, then choose the lower-energy state. Higher powers change how sharply strong matches dominate weak ones.

For a concrete parity task, store the four triples

[−1, −1, −1], [−1, +1, +1], [+1, −1, +1], [+1, +1, −1].

Clamp the first two coordinates as inputs and infer the third, which should equal −ab. With F(z) = z², both candidate outputs have energy −12 for every input pair. This storage construction cannot distinguish the answers. With F(z) = z³, E(a,b,z) = 24abz on the binary cube. Minimization chooses z = −ab, giving energy −24 rather than +24.

[Figure 19: four input pairs with a two-column candidate-output energy comparison. Mark inputs as clamped and the output as the only editable state.]

The example shows why changing the interaction function changes representable relationships. It does not mean a quadratic network with additional hidden units can never represent parity, nor that higher degree always trains better.

The higher-order paper also connects learned memories to feature-like versus prototype-like representations and to a feedforward hidden layer with related nonlinearities. This gives a useful design question: should a memory describe a reusable feature or a whole prototype? The answer depends on the task; our 200-image reference bank deliberately uses actual examples.

### Fixed-bank energy, changing memories and stochastic models

The [Boltzmann Machines & RBM](/learn/path/full-curriculum/boltzmann-machines-restricted-boltzmann-machines-rbm?module=deep-learning-fundamentals) lesson uses energy to define probabilities over states and learns through model/data statistics. Classical Hopfield recall here deterministically lowers an energy. A stochastic equilibrium distribution, a deterministic local minimum and a softmax distribution over memory scores are three different objects.

For a growing sequence memory, causal reading uses only the available prefix. An age bias can modify a score to βqᵀxᵢ − γ(t−i): equally matching older memories receive less unnormalized mass. The factor exp(−γ age) acts as a forgetting preference. This does not make an explicit bank occupy constant storage, and changing time or the bank changes the energy being considered.

There is also a genuine architecture consequence to requiring a shared energy when both queries and keys evolve. [Energy Transformer](https://arxiv.org/html/2302.07253v1) derives token dynamics from an engineered energy. Differentiating a token's contribution through both its query role and its key role adds terms absent from ordinary one-way attention. Its memory and attention contributions operate together. This is a specific construction, rather than a new name for an arbitrary transformer stack.

The continuous-time energy argument and a numerical discretization are separate: a large finite step can require its own stability analysis. The later [Neural ODE & Continuous-Depth Models](/learn/path/full-curriculum/neural-ode-continuous-depth-models?module=deep-learning-fundamentals) lesson develops that distinction.

## 8. Applications that make the memory choice matter

### Find a rare signal in a large set

Suppose a bag contains 10,000 short sequence embeddings, and only a small subset carries useful evidence for the bag's label. Plain mean pooling dilutes each instance equally; max pooling forces each feature to use its largest entry. A learned query instead scores instances and builds a weighted summary.

This is the structure used in [DeepRC](https://arxiv.org/abs/2007.13505): receptor sequences become embeddings, attention pools the set, and an output network predicts a repertoire-level label. The training labels apply to the bag, so learning must assign useful credit without being given a correct label for every receptor. Reordering instances should not change the summary. Adding duplicates or sampling a subset can change it.

[Figure 20: variable-sized bags → shared sequence encoder → learned-query pooling → one bag-level label. Highlight that the supervision arrow ends at the bag.]

The broader lesson is useful for document collections and image patches as well: define what one instance is, what the set label means, and whether order or multiplicity matters. Attention weights alone do not establish that a high-weight receptor is a biological cause.

### Revisit examples and features during tabular prediction

[Hopular](https://arxiv.org/abs/2206.00664) uses two memory roles in each block. One reads across stored training examples; another reads across the embedded features of the current input. The representation is refined through successive blocks, and masked attributes are part of training.

A row with an unknown target can therefore ask both “which earlier examples resemble my current representation?” and “which of my own attributes inform one another?” That differs from a one-off nearest-neighbor lookup and from attending only across columns. The actual training examples and the current query's unavailable target must remain distinguishable.

[Figure 21: one sample-to-sample memory read and one feature-to-feature read in an alternating block, with the target masked on the query side.]

The paper's benchmark conclusions belong to its datasets and protocol. For a new table, compare with strong tabular baselines under the same split and preprocessing. Calling a component a memory does not establish that it will help.

### Support sets and stored prototypes

A few-shot classifier can treat labeled support examples as keys and their labels as values. Our handwriting model is already a small instance of that design. Replacing the support set changes which classes and visual styles can receive mass. A learned embedding should be evaluated on the intended episode/class split; a capacity theorem cannot replace that experiment.

For large text retrieval, an external search system may first shortlist documents before a neural model reads them. The shortlist operation, the attention read and the final generated answer are distinct. A correct attention identity proves neither that the search found the right evidence nor that an answer faithfully uses it.

## 9. Practice: predict, calculate and diagnose

Attempt each prompt before opening its hint. Changed inputs are intentional: the aim is to use the mechanism, not recall a number from the worked example.

### 1. Repair a different damaged feature

Store [1, −1, 1, −1] with the Hebbian rule. Start at [1, −1, −1, −1] and visit coordinates 1 through 4. Which update changes the cue, and by how much does energy change at that update?

<details><summary>Hint</summary>

Build only the row needed for the damaged third coordinate. Its three neighbors currently agree with the stored pattern.

</details>
<details><summary>Solution</summary>

The third row is [0.25, −0.25, 0, −0.25]. Its field is 0.75, so the third coordinate changes from −1 to +1. ΔE = −(1−(−1))0.75 = −1.5. Coordinates 1 and 2 remain unchanged before that update, and coordinate 4 remains −1 afterward. The stored pattern is recovered.

</details>

### 2. A changed continuous cue

Keep memories [1, 0] and [−1, 0], but use cue [−0.3, 0.7] and β = 1. What is the first read? Does a second read necessarily equal it?

<details><summary>Hint</summary>

The vertical coordinate disappears; the horizontal update is tanh(βqₓ).

</details>
<details><summary>Solution</summary>

The first read is approximately [−0.291313, 0]. The second is [tanh(−0.291313), 0], approximately [−0.283342, 0]. They differ. A one-step read can be a useful feedforward operation without being an exact fixed point.

</details>

### 3. One high-scoring memory versus a class

Three memories have unnormalized weights 4, 3 and 3. Their labels are A, B and B. What do nearest-memory classification and label-summed retrieval predict?

<details><summary>Hint</summary>

Normalize by the sum, then aggregate by label.

</details>
<details><summary>Solution</summary>

The weights are 0.4, 0.3 and 0.3. The highest single memory belongs to A; class masses are A = 0.4 and B = 0.6, so label summation predicts B. Neither rule is intrinsically the correct classifier for every problem; evaluate the rule you intend to use.

</details>

### 4. Can a shared value hide a changed attention map?

Keep the three weights from exercise 3, but attach value [2, 5] to every memory. Then change the weights to [0.9, 0.05, 0.05]. What changes?

<details><summary>Hint</summary>

Factor the common value out of the weighted sum.

</details>
<details><summary>Solution</summary>

The output remains [2, 5], since the weights sum to one in both cases. The attention distribution changed, but the payload did not. A debugging tool should display both.

</details>

### 5. An apparently excellent handwritten-digit result

A colleague puts all 300 validation images and their labels into the memory bank, then reports near-perfect validation accuracy. Why does that not answer our original evaluation question?

<details><summary>Hint</summary>

Identify which information would be unavailable for a new query.

</details>
<details><summary>Solution</summary>

The bank now contains the evaluation images' correct labels and exact self-matching keys. It evaluates a predictor with information unavailable for a new unlabeled image. Restore the fit-only memory bank and redo the validation protocol. A genuinely transductive task would require a separately stated information boundary; it cannot expose the unknown query labels as values.

</details>

### 6. A noisy cue with the wrong winner

The intended memory's score is 0.8 and an incorrect memory's score is 0.9. What happens to their relative weight as β increases?

<details><summary>Hint</summary>

Write the ratio of their exponentials.

</details>
<details><summary>Solution</summary>

The incorrect-to-intended ratio is exp(0.1β), which grows with β. Sharpening strengthens the wrong winner. Improving the representation, acquiring more cue information or changing the decision rule may help; temperature cannot reverse this score ordering.

</details>

### 7. Choose the metric before judging the image

A reconstruction cuts mean squared error from 0.12 to 0.06 but changes a correctly recognized 7 to a 1. Is this improvement?

<details><summary>Hint</summary>

State the intended task and distinguish two valid measurements.

</details>
<details><summary>Solution</summary>

It improves average squared pixel reconstruction on this example and worsens classification. For a recognition system, report the class failure. For a reconstruction task, inspect whether the metric misses a perceptually or semantically important stroke. A useful report gives both measurements and explains the mismatch rather than choosing whichever makes the method look better.

</details>

### 8. Bound the competing mass

A bank has 11 unit-norm memories. The desired memory beats every other score by at least 2, and β = 1.5. Give a lower bound for its weight and an upper bound for the read's distance to it.

<details><summary>Hint</summary>

Use a = (P−1)exp(−βδ).

</details>
<details><summary>Solution</summary>

a = 10exp(−3) ≈ 0.497871. Target weight is at least 1/(1+a) ≈ 0.667614. The tighter distance bound is 2a/(1+a) ≈ 0.664771. This is a sufficient bound based on the assumed score gap, not the exact error of an unspecified bank.

</details>

### 9. Test energy reasoning against an update

A system evolves both queries and keys through unrelated learned projections and residual updates. Its authors plot the fixed-X energy from §3 at each layer and claim it must decrease. What must they establish?

<details><summary>Hint</summary>

Ask whether the same scalar function is being minimized and whether every update is derived from it.

</details>
<details><summary>Solution</summary>

They must define a shared state and energy, include dependencies through changing keys as well as queries, and show that the actual update lowers that energy under its assumptions. Recomputing a different fixed-bank energy at every layer does not prove descent of one objective. Residuals, projections, normalization and finite step sizes need their own treatment.

</details>

### 10. Plan a meaningful extension of the digit experiment

You can improve robustness by training with masks. Specify a fair experiment and a decision rule without using the existing test results to select a favorable mask.

<details><summary>Hint</summary>

Separate the augmentation design, validation criteria and final assessment.

</details>
<details><summary>Solution</summary>

One acceptable plan predefines a distribution of missing-cell masks using fitting data only; trains both clean and mask-augmented projections with the same memory bank, parameter count and optimization budget; selects epochs using a predefined combination of clean and masked validation losses; and evaluates the frozen models on a separately specified test corruption protocol plus clean test images. Report both seeds, error counts and reconstruction metrics. Preserve the original images and masks so another learner can reproduce the comparison. More robust results on one mask family do not establish robustness to every handwriting distortion.

</details>

You are ready to move on when you can distinguish state refinement from parameter training, trace keys through weights to values, explain a failed retrieval without appealing to a vague capacity claim, and preserve the learning boundary in a memory-based experiment.

The next topic in this module is [xLSTM (Extended LSTM)](/learn/path/full-curriculum/xlstm-extended-lstm?module=deep-learning-fundamentals). It returns to recurrent sequence memory and asks how changing gates and scalar or matrix state changes what a model can retain and read. That is a different storage/update contract from retaining every row of an explicit reference bank.

## References & another way to learn it

- [Ramsauer et al., Hopfield Networks is All You Need](https://arxiv.org/html/2008.02217v3) — the main continuous-memory reference. Read §2 after the worked energy calculation, §3 for layer choices, and Appendix A.1.5–A.1.6 for precise stability and capacity assumptions. The introduction's one-update language is made precise in the theorems.
- [Johannes Brandstetter and the JKU authors, Hopfield layers illustrated article](https://ml-jku.github.io/hopfield-layers/) — an alternate visual explanation of memory retrieval, temperature, keys/queries/values and pooling. Its familiar-image examples help establish the geometry; pair its informal convergence wording with the paper's exact statements.
- [Krotov & Hopfield, Dense Associative Memory for Pattern Recognition](https://arxiv.org/html/1606.01164v2) — study §§2–3 for energy-difference updates and the parity example, and §§4–5 for learned features/prototypes and the feedforward interpretation.
- [Hopfield, Neural networks and physical systems with emergent collective computational abilities](https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/) — the 1982 historical starting point. The archive provides the original scanned article and describes content-addressable memory and asynchronous dynamics.
- [McEliece et al., The capacity of the Hopfield associative memory](https://authors.library.caltech.edu/records/q92rz-95p89) — a more mathematical resource separating exact recovery of most memories from exact recovery of all memories. Useful when evaluating a capacity claim.
- [Official Hopfield layers code and examples](https://github.com/ml-jku/hopfield-layers) — compare the three module interfaces and study the bit-pattern and latch-sequence notebook descriptions. Treat its documented older dependency environment as a research-package detail; the notebook experiments were not executed for this lesson.
- [Yannic Kilcher, Hopfield Networks is All You Need — Paper Explained](https://www.youtube.com/watch?v=nv6oFDp6rNQ) — optional advanced paper walkthrough, also linked by the authors' article. The title and author link were verified; the video itself was not reviewed here, so use the paper and checked examples for the technical guarantees.
- [Widrich et al., Modern Hopfield Networks and Attention for Immune Repertoire Classification](https://arxiv.org/abs/2007.13505) — read the Deep Repertoire Classification section to see how the unit of supervision changes from one sequence to a large set of sequences.
- [Schäfl et al., Hopular](https://arxiv.org/abs/2206.00664) — §3 explains the two memory roles in tabular refinement. Follow the distinction between sample-to-sample and feature-to-feature retrieval.
- [Hoover et al., Energy Transformer](https://arxiv.org/html/2302.07253v1) — an advanced extension. §2 explicitly derives dynamics with changing token representations and explains why its energy attention differs from ordinary attention.
- [UCI digit dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) and [local provenance](./data-provenance.md) — original acquisition, count features, train/test writer split, license, exact downloads and our separate experimental roles.
