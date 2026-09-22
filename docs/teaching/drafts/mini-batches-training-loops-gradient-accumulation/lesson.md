# Mini-Batches, Training Loops & Gradient Accumulation

**Explore as you read.** Edit rows, microbatch boundaries, learning rate, clearing/step policy, target weights and normalization groups. Populate row model outputs/errors/gradients immediately; step backward and optimizer events with distinct clocks and buffers. Compare final policies using the same rows and initial state. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to decide when accumulation is equivalent, what receives influence and why partition-sensitive operations change the computation.


Your model can process twelve examples at a time, but you want one update to use thirty-two. Can you run three forward/backward passes and obtain the same update as a batch of thirty-two? Yes, for an appropriate computation—but the last eight examples must receive the same per-example influence as the first twenty-four. Calling every microbatch's loss `.mean()` and averaging those three numbers gives a different objective.

This lesson makes a training loop inspectable. You will follow examples, predictions, losses, gradients, parameters and optimizer memory as separate objects, then build an offline flower classifier whose full-batch and accumulated updates agree. The central question is: **which examples contribute to this update, with what weight, evaluated at which parameters?**

**First pass:** read §§1–6, run the scalar trace in §4 and the Iris program in §6, and attempt practices 1–4. Use the state and denominator investigations as you encounter them. You should finish able to write a correct accumulation loop and explain its final partial group. §§7–8 are deeper branches on batch-dependent computation, numerical precision and distributed training; return to these before applying accumulation to such models. Practice 5 tests that transfer. Allow roughly 45–60 minutes for the core reading and 45–75 minutes for programs and independent practice.

You need to read a two-dimensional array, recognize a scalar derivative, and know that a gradient tells an optimizer how a local change affects loss. Review [NumPy: Arrays, Broadcasting & Vectorization](/learn/topic/numpy-arrays-broadcasting-vectorization) for shapes and reductions, and [Backpropagation & Automatic Differentiation](/learn/topic/backpropagation-automatic-differentiation) for the chain rule. We refresh the particular derivative needed below. No GPU is required for the programs.

## 1. A batch groups data; an update changes a model

An **example** is one input and its target. A **mini-batch** groups examples for a computation. An **epoch** is a pass through a specified training sampling procedure, commonly one shuffled traversal of the training rows. An **optimizer step** applies an update to parameters and, when present, optimizer state. A **microbatch** is a smaller chunk whose gradient contributes to an update shared with other chunks. The examples combined for that update form its **effective batch**.

For ten examples, microbatch size four, and two microbatches per update, the sequence is:

```text
examples:       [a b c d] [e f g h] | [i j]
microbatch:         1         2     |   3
effective batch: [------- 8 ------]|[-- 2 --]
optimizer step:                   1          2
epoch:          [------------ one traversal ------------]
```

**Figure A — three clocks.** Read across the example lane before reading the update lane. There are three forward/backward computations, two optimizer steps, and one epoch. The last update uses two examples. This is an exact counting example, not a timing measurement.

With a finite ordinary map-style dataset of size $N$, microbatch limit $b$, and $K$ chunks per update, keeping all rows and flushing at each epoch end gives $M=\lceil N/b\rceil$ microbatches and $U=\lceil M/K\rceil$ updates. Write down the sampling and remainder policy before using this formula. Sampling with replacement can revisit a row and omit another within an epoch; a streaming dataset may instead define an epoch by a fixed number of draws.

The loader decides which examples arrive and how they are collated into tensors. The loop decides when to update. For example, PyTorch's `DataLoader(..., batch_size=4, shuffle=True, drop_last=False)` provides batches, but does not create accumulation boundaries for you. `drop_last=True` discards a loader's short final batch; it does not repair an incorrectly scaled accumulation group. Keep it a deliberate data policy. [PyTorch data loading reference](https://docs.pytorch.org/docs/2.14/data.html#loading-batched-and-non-batched-data).

Why group examples at all? Matrix operations can reuse data and amortize framework overhead across rows. Meanwhile, averaging gradients makes the update depend less on any one sampled example. Batch size therefore affects memory, computation and the statistical path taken through training. Gradient accumulation primarily changes how an effective batch fits into working memory. It still performs the microbatches' work; it is not a promise of faster training. [Dive into Deep Learning, §12.5](https://d2l.ai/chapter_optimization/minibatch-sgd.html).

**Pause:** If you change microbatch size but preserve each effective group and its one optimizer step, which clock can change? The number of forward/backward calls can change while the number of examples and updates stays fixed. If instead you step after every newly sized batch, you change the update clock too.

## 2. Follow one complete update

Use a single adjustable weight $w$ and prediction $\hat y_i=w x_i$. These three constructed rows keep the arithmetic visible:

| Row | Input $x_i$ | Target $y_i$ | Prediction at $w=0$ | Half-squared loss $\ell_i=\frac12(wx_i-y_i)^2$ | Derivative $(wx_i-y_i)x_i$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| a | 1 | 2 | 0 | 2 | −2 |
| b | 2 | 0 | 0 | 0 | 0 |
| c | 3 | 1 | 0 | 0.5 | −3 |

The derivative follows the chain rule: changing $w$ changes the prediction by $x_i$ times as much, while changing the prediction changes half-squared loss at rate $wx_i-y_i$. Multiplying those local effects gives the last column. The half factor cancels the derivative of the square. PyTorch's ordinary MSE loss has no half factor; our scalar examples explicitly include it.

The objective is the **mean per example**:

$$
L(w)=\frac{1}{3}\sum_{i=1}^{3}\ell_i(w),\qquad
L(0)=\frac{2+0+0.5}{3}=\frac56,
\qquad g=\frac{-2+0-3}{3}=-\frac53.
$$

A gradient is not a parameter change. Plain stochastic gradient descent (SGD) makes the separate decision $w_{\text{new}}=w-\eta g$. With learning rate $\eta=0.1$, the new weight is $1/6$. Its predictions are now $[1/6,1/3,1/2]$. On the same three rows the mean loss is $67/108\approx0.620370$, down from $5/6\approx0.833333$. The update reduced the combined loss even though row b became less accurate. A mean objective negotiates among examples.

**Figure B — one weight, three distinct stores.** Show the current parameter $w=0$, an initially empty gradient slot, and an optimizer-state slot. Forward computes predictions and loss. Backward writes $g=-5/3$ into the gradient slot while $w$ remains zero. Only the optimizer arrow changes $w$ to $1/6$.

Real networks have many parameters, and their gradients have matching shapes. For a batch with $B$ rows and $D$ input features, a linear classifier in PyTorch computes

$$
X_{B\times D}W^\top_{D\times C}+b_{C}\longrightarrow
\text{logits}_{B\times C}.
$$

Here $C$ is the number of classes; logits are raw scores. One integer target per row has shape $(B,)$. Cross-entropy combines each row's logits and target into a scalar loss after reduction. `backward()` computes a gradient of shape $(C,D)$ for `nn.Linear.weight` and $(C,)$ for its bias. Averaging the loss does not average the parameter dimensions away. [CrossEntropyLoss shapes and target conventions](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

### Parameters, gradients and optimizer memory have different lifetimes

The parameter persists across training. A gradient slot holds contributions for the next update. Optimizer state can retain information from earlier updates. For example, a simple momentum convention is

$$
v_t=\mu v_{t-1}+g_t,\qquad w_{t+1}=w_t-\eta v_t.
$$

The coefficient $\mu$ controls memory. Starting with $v_0=0$, the first update agrees with plain SGD. Suppose the next effective group has gradient $+1$, with $\mu=0.9$. Then $v=0.9(-5/3)+1=-0.5$ and the weight moves from $1/6$ to $13/60\approx0.216667$. It still moves upward because stored momentum outweighs the current gradient. PyTorch SGD agrees with this convention for our zero dampening, no Nesterov, no weight decay setup. Its broader parameter choices have additional semantics. [SGD reference](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html).

`optimizer.zero_grad(set_to_none=True)` resets gradient slots, not weights or momentum. `None` means no gradient has been supplied; it is distinct from a tensor of zeros. PyTorch optimizers skip parameters with `grad=None`, whereas a zero gradient can still allow an update through existing optimizer state or weight decay. This distinction is useful when a branch uses only some parameters. [zero_grad reference](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html).

The normal lifecycle is: **clear → forward → loss → backward → optional gradient processing → step**. Clearing immediately after the preceding step also works if the first group's gradients start clear. `model.train()` selects training behavior; it does not start this lifecycle or change weights by itself. [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

## 3. Accumulation adds derivatives at the same parameters

Return to rows a, b, c and split them into microbatches `[a,b]` and `[c]`. Hold $w=0$ throughout both forward/backward passes. The first chunk contributes the derivative of $(\ell_a+\ell_b)/3$, which is $-2/3$. The second contributes the derivative of $\ell_c/3$, which is $-1$. Adding them gives $-5/3$, exactly the derivative of the full objective from §2.

The mathematical reason is linearity of differentiation. If $S_j(\theta)$ is the sum of losses in chunk $j$ and $D$ is the total number of contributing examples, then

$$
\nabla_\theta \left(\frac{\sum_j S_j(\theta)}{D}\right)
=\sum_j\nabla_\theta\left(\frac{S_j(\theta)}{D}\right).
$$

$\theta$ denotes all model parameters together. The denominator is fixed by the data in this group, not learned. Each call to `.backward()` adds another derivative to the existing gradient slots. Fresh forwards produce fresh computation graphs, so ordinary accumulation does not require `retain_graph=True`. After each backward pass, that chunk's saved activations can be released; the gradient tensors survive.

The equality concerns a particular objective evaluated at the same parameters with the same per-example computation. Use it when examples do not interact across chunks, randomness is absent or aligned, and optimizer state advances once after the complete sum. Batch-dependent layers and cross-example losses are examined in §7. Floating-point summation order can cause small differences even when the mathematical update is identical.

### Why averaging microbatch means can change the question

Our first chunk's mean loss has gradient $(-2+0)/2=-1$. The second chunk's mean has gradient $-3$. Averaging those means gives $(-1-3)/2=-2$, leading to weight $0.2$ instead of $1/6$.

The error is visible before calculus. Under an equal average of chunk means, rows a and b each get coefficient $1/4$, while c gets $1/2$. Under the intended example mean, all three coefficients are $1/3$.

**Figure C — contribution balance.** Align one coefficient under each of the same three rows: `1/3, 1/3, 1/3` versus `1/4, 1/4, 1/2`. Microbatch brackets show where the extra factor arose. Coefficient bars share a zero baseline and linear scale. The smaller chunk's example receives more influence in the second row.

If chunk $j$ contains $n_j$ examples and its mean loss is $\bar L_j$, the correct combination is

$$
L=\sum_j\frac{n_j}{\sum_k n_k}\bar L_j.
$$

Dividing each mean by the number of chunks is the special case where all their denominators are equal. Count what the loss averages; do not assume every tensor called a batch contains the same amount of supervision.

**Investigation 1 — who changed the weight?** Edit a row or target and place microbatch boundaries while watching the current computed results. Step the instructions through separate weight, gradient and momentum lanes. Compare the correct update boundary with clearing or stepping between chunks. Finally use one chunk and inspect which mistaken policies become indistinguishable. Each input edit recomputes the trace; Restart begins the current case again.

## 4. Write the loop around the update boundary

This complete program implements the scalar calculation. Save it as `trace_update.py`, or use the supplied file. Run `python -B trace_update.py` with PyTorch installed. The recorded execution used Python 3.12.14 and PyTorch 2.14.0 on CPU in float64.

```python
import torch

torch.set_default_dtype(torch.float64)
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 0.0, 1.0])
w = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)
optimizer.zero_grad(set_to_none=True)
print("start", f"w={w.item():.6f}", "grad=None")
for rows in ([0, 1], [2]):
    loss_sum = 0.5 * ((w * x[rows] - y[rows]) ** 2).sum()
    (loss_sum / len(x)).backward()
    print("backward", rows, f"grad={w.grad.item():.6f}", f"w={w.item():.6f}")
optimizer.step()
print("step", f"w={w.item():.6f}", f"momentum={optimizer.state[w]['momentum_buffer'].item():.6f}")
optimizer.zero_grad(set_to_none=True)
print("clear", "grad=None", f"w={w.item():.6f}")
```

Executed output, rounded to six decimals:

```text
start w=0.000000 grad=None
backward [0, 1] grad=-0.666667 w=0.000000
backward [2] grad=-1.666667 w=0.000000
step w=0.166667 momentum=-1.666667
clear grad=None w=0.166667
```

The two lines labeled `backward` recover the separate contributions in §3. The final clear preserves both the learned weight and momentum. Logging uses scalar values; accumulating graph-connected losses into a list and calling backward only at the end would retain the chunks' graphs and undermine the activation-memory purpose.

### The final partial group is an actual update

For the ten-row example in §1, the effective groups have sizes eight and two. Divide the first group's summed losses by eight and the last group's summed losses by two. If you keep dividing each microbatch mean by the nominal $K=2$, the last group contains only one mean and receives half the intended gradient. If you step only when a microbatch index is divisible by two, the final two examples never produce an update.

A clear implementation first defines an effective group, counts its actual denominator, then iterates through that group's microbatches. You can buffer its indices or input tensors without keeping their forward graphs. The Iris program uses precisely this layout. Another approach accumulates derivatives of unnormalized loss sums and divides each populated gradient by the final denominator before clipping and stepping. That approach is useful when the denominator becomes known while streaming; it needs extra care with mixed precision as described in §8.

Keeping a small final group is an explicit optimization choice: each completed group produces one update of its own mean loss, so individual examples in a smaller final group receive a larger coefficient in that update. Matching a large-batch reference means matching this same sequence of groups. You can instead drop or carry the tail into the next epoch, but then you have chosen a different data/update schedule and should count it accordingly.

## 5. The denominator defines the objective

For a scalar prediction per example, example count was the denominator. Other losses can average over pixels, tokens, output coordinates or target weights. Write the group objective as

$$
L=\frac{\sum_i a_i m_i\ell_i}{\sum_i a_i m_i},
$$

where $a_i\ge0$ is a fixed importance weight and $m_i\in\{0,1\}$ indicates whether item $i$ contributes. Its numerator and denominator add across chunks. This weighted-mean convention is a declared objective; some library losses implement other normalizations.

For a concrete unequal case, suppose chunk A has two contributing targets with loss sum 2 and chunk B has six with loss sum 18. Their means are 1 and 3. The target-level mean is $(2+18)/(2+6)=2.5$, whereas the equal mean of chunk means is 2. The same arithmetic applies to derivatives because both routes differentiate these differently weighted objectives.

In language modeling, a batch can contain two sequences of different lengths. If one has two eligible next-token targets and the other six, a token mean gives the longer sequence three times the total mass. A sequence mean first averages each sequence's tokens, then weights the two sequences equally. Both can be intentional objectives; they answer different questions. Padding that carries no target must not increase either one's target count. Target shifting and causal attention are developed in [Language-Model Batches, Attention Masks & Loss Alignment](/learn/topic/language-model-batches-attention-masks-loss-alignment), a later curriculum entry.

**Figure D — tokens into two sums.** Show two rows of slots, two versus six eligible targets, plus crossed-out padding slots. Eligible slots send losses into the numerator and counts into the denominator. The denominator is eight even if the rectangular tensor contains twelve slots. Place the alternative sequence-mean calculation below, so its different weighting is visible.

Common PyTorch conventions need separate attention:

| Loss setup | What to sum | Denominator for the stated mean |
| --- | --- | --- |
| Unweighted class-index cross-entropy | Loss for each eligible target | Number of nonignored targets |
| Class-index cross-entropy with class weights, no label smoothing here | Class-weighted target losses | Sum of the eligible targets' class weights |
| Cross-entropy with probability targets and class weights | Class-weighted cross-entropy per target distribution | Number of target positions; it does not use the preceding class-weight denominator |
| MSE over a tensor with multiple output coordinates | All squared coordinate errors | Number of error elements, unless you explicitly construct another reduction |
| Our half-squared scalar regression | Half-squared error per row | Number of rows, or declared weight sum |

The cross-entropy distinction and `ignore_index` semantics come from the current [loss reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html); MSE's element reduction is documented in [MSELoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html). Use `reduction="sum"` and an explicitly computed group denominator when implementing these objectives across uneven chunks. Do not divide once in the loss and again in the gradient unless the second factor is the intended chunk weight.

If a whole group has no eligible target mass, its mean is undefined. Skip that group's update and its update-based scheduler tick, or reject it as an input-construction error. Avoid evaluating an empty mean and then multiplying NaN by zero. A zero-loss sum from an all-ignored microbatch can contribute zero while other chunks make the group's denominator positive; batch-dependent layers may still change state during that forward.

**Investigation 2 — choose the loss mass.** Edit the weighted target table: an input, target, inclusion mark or importance weight. Inspect the global mean gradient, then compare summing numerators with equally averaging microbatch means. The display traces each row's numerator contribution and mass. Find an unequal-size case that nevertheless agrees because the two total masses match; then remove all eligible mass and explain why an update is disabled. The mathematical simulator uses the scalar squared-error model, so the weighting mechanism is visible without requiring an NLP model.

The same accounting governs evaluation logs. Add detached loss numerators and their denominators over the evaluation dataset, then divide once. Averaging batch means during validation reproduces the small-batch weighting error even when no gradients are computed. Accuracies similarly use total correct divided by total eligible targets. A training loss accumulated while the model changes is an online summary across different parameter states; an evaluation pass at epoch end measures one fixed state. Label those differently.

## 6. A complete offline experiment on measured flowers

Fisher's Iris data records sepal and petal length and width for three species. The classification question is whether these measurements help distinguish the species. We use a small neural classifier to ask a narrower training-mechanics question: **can different microbatch partitions produce the same sequence of learned parameters and momentum buffers?**

The packet supplies all 150 observations in [iris.csv](iris.csv). Features are centimeters and class IDs 0, 1, 2 denote setosa, versicolor and virginica. The data is attributed to Fisher through [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris), CC BY 4.0. This CSV is exported from scikit-learn 1.9.1's bundled, corrected variant; scikit-learn documents two corrections relative to the older UCI copy. Row IDs and a header were added, with original row order preserved. See [data-provenance.md](data-provenance.md) for the exact version and checksum.

Before fitting, the program chooses forty rows per species for training and ten per species for validation, using split seed 17. Only training rows determine feature centering and scaling. The validation rows are inspected at epoch ends and are never differentiated. We keep twenty epochs and the stated hyperparameters fixed, with no search or early stopping. A uniform-probability baseline has mean cross-entropy $\log 3\approx1.098612$; predicting one constant species is correct on ten of thirty validation rows. There is no separate final test set in this compact teaching experiment.

The network has shapes `B×4 → B×8 → B×3`: a linear layer, tanh, then a linear output layer. It has no dropout or batch normalization. Both runs clone the same initial model, use the same precomputed shuffled row orders, and use SGD with learning rate 0.05 and momentum 0.9. Their effective groups are `32,32,32,24` each epoch. One run processes each entire group; the other partitions thirty-two as `12+12+8`, and twenty-four as `12+12`. Both make four updates per epoch. Float64 makes the equivalence comparison easy to inspect on CPU.

The full runnable program is [train_iris.py](train_iris.py). Keep it beside `iris.csv`; it needs NumPy and PyTorch and performs no downloads. Run:

```text
python -B train_iris.py
```

Read the following central loop with the complete file open. The full file supplies imports, split, normalization, model, fixed orders, evaluation, both training runs and printed results; this excerpt focuses on the update boundary.

```python
for start in range(0, len(order), 32):
    group = order[start:start + 32]
    optimizer.zero_grad(set_to_none=True)
    for offset in range(0, len(group), microbatch_size):
        rows = group[offset:offset + microbatch_size]
        logits = model(features[rows])
        loss_sum = F.cross_entropy(logits, targets[rows], reduction="sum")
        (loss_sum / len(group)).backward()
    optimizer.step()
```

`len(group)` is 24 in the final update, not the nominal 32. No optimizer state changes while a group's chunks are being processed. Consequently, the same aggregated gradient reaches the same optimizer state, which produces the same next state. Repeating that argument explains why equivalence extends beyond the first update.

Recorded CPU results from the supplied program, rounded as printed:

```text
epoch updates train_loss train_correct validation_loss validation_correct
0 0 1.037300 56/120 1.050925 15/30
1 4 0.806121 79/120 0.823158 20/30
5 20 0.340418 101/120 0.324301 27/30
20 80 0.097719 116/120 0.069435 30/30
full_forward_backward_calls 80
accumulated_forward_backward_calls 220
max_parameter_gap 2.914e-16
max_momentum_gap 8.327e-17
uniform_probability_loss 1.098612
constant_class_validation_correct 10/30
```

The accumulated model learns useful structure on this split, and its final parameters and momentum agree with the full-group run within roundoff. The evidence for accumulation is the paired state comparison, not the classification score. The final validation score describes thirty particular rows in one split; it is not a species-recognition guarantee. No runtime or memory benchmark was collected.

**Figure E — recorded training and paired-state evidence.** Plot loss from epoch zero through twenty using the supplied measured history, with training and validation lines and the uniform-probability reference. Keep a separate exact readout for the maximum parameter and momentum gaps. Two nearly identical training curves would hide the comparison's key quantity; report the difference directly. The complete history includes temporary decreases in validation accuracy, so keep those observations if accuracy is displayed.

### Evaluation selects behavior and suppresses gradient recording separately

Before a training epoch, call `model.train()`. For validation, call `model.eval()` and evaluate within `torch.no_grad()`. The first selects behavior for mode-sensitive layers; the second disables ordinary reverse-mode graph recording for the forward computation. Neither call updates weights. `eval()` alone still permits derivatives; `no_grad()` alone leaves dropout and batch-normalization training behavior active. The supplied model has neither layer, but spelling out both operations makes the loop's intent explicit. [no_grad reference](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html), [BatchNorm1d reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html).

**Try a direct comparison:** change only `train(12)` to `train(7)`, run it and inspect the number of forward/backward calls together with the final parameter and momentum agreement. Keep effective groups, initialization, optimizer and orders fixed. Practice 4 gives the closed reasoning and a second, stronger variation.

### Own the accumulation rule; reuse the derivative and optimizer engines

The mechanism here is the boundary between losses, accumulated derivatives and one update. The complete [train_iris.py](train_iris.py) owns that boundary in `train`: define actual effective-group rows, clear once, divide each chunk's summed loss by the same real group count, call backward per chunk, and step once. `trace_update.py` makes the gradient slot, parameter and momentum lifetimes visible. These are ordinary PyTorch loops, with the accumulation algorithm expressed explicitly rather than delegated to an opaque trainer.

If you want to reopen the two reused engines, the **implemented** [Backpropagation lesson](/learn/path/full-curriculum/backpropagation-automatic-differentiation?module=deep-learning-fundamentals) supplies its [scratch differentiation engine](/learn-assets/backpropagation/teaching-autodiff.py) and [matched engine/library bridge](/learn-assets/backpropagation/engine-library-bridge.py). The **implemented** [Gradient Descent Variants lesson](/learn/path/full-curriculum/gradient-descent-variants-sgd-adam-adagrad-rmsprop-lamb-lars?module=math-foundations) supplies [manual_step and optimizer-state comparisons](/learn-assets/gradient-variants/optimizer_library_bridge.py). Those programs actually implement the mechanisms; a link to a bare library reference would not replace them. This topic need not recopy either engine to teach a new grouping rule.

The correspondence matters: the scratch sum of per-example derivatives becomes repeated `.backward()` additions into `.grad`; the scratch momentum array becomes SGD's `momentum_buffer`; the scratch update clock becomes exactly one `optimizer.step()` per effective group. Actual state comparisons in the Iris experiment check parameters **and** momentum, so similar accuracy cannot conceal a wrong update schedule. Microbatching reduces simultaneously retained activation graphs, while parameter, gradient and optimizer storage remain. Retaining graph-connected losses until the end would lose that intended memory benefit.

**Changed-constraint exercise.** Keep groups of 32 and change the physical microbatch size to seven. Inspect the final group of 24 as well, preserving all input rows. State the denominator at every backward call, including the short last physical chunk.

<details><summary>Hint and reasoned solution</summary>

A 32-row group has physical sizes 7,7,7,7,4, and each summed loss divides by 32. A 24-row group has sizes 7,7,7,3, and each divides by 24. Each group still advances momentum once. With the existing 120 fitting rows, one epoch has groups 32,32,32,24: nineteen physical forward/backward calls and four optimizer updates. The full-group and accumulated parameter/momentum paths should match to floating-point tolerance because the model has no cross-example operation or stochastic forward layer. An equal average of the five or four physical means changes row weights; a step after each physical chunk changes both parameters and momentum between derivatives.

</details>

## 7. Deeper branch: an effective batch need not be a physical batch

The arithmetic proof in §3 assumes that partitioning leaves each loss term's computation unchanged. It can fail before gradients are added.

### Batch normalization sees the physical forward batch

Suppose a layer receives scalar activations $[0,2,10,12]$. Normalization using their full mean 6 and population variance 26 yields approximately $[-1.176697,-0.784464,0.784464,1.176697]$, using $\epsilon=10^{-5}$. If instead `[0,2]` and `[10,12]` are normalized independently, each pair becomes approximately `[-0.999995,+0.999995]`. The third example changes sign: its value 10 is above the full-group mean but below its own pair's mean.

For a downstream trainable scale $\theta$, predict $\theta z_i$ and use half-squared loss against targets $[0,0,1,1]$. At $\theta=1$, the full-group mean gradient is approximately 0.509709, while correctly weighted local-normalization chunks give 0.999990. The denominator is correct in both; the normalized inputs differ. Gradient accumulation has no way to retroactively replace those forward statistics.

Running statistics also update on forwards. With initial running mean zero and update coefficient 0.1, one full forward stores 0.6. Two local forwards store $0.9(0.1)+0.1(11)=1.19$. Even when microbatches have identical means and variances, their training outputs can agree while their repeated running-statistic updates differ. PyTorch uses a population variance for the current normalization and an unbiased estimate for its running variance; the example above traces only its running mean. [BatchNorm1d semantics](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html).

**Figure F and investigation 3 — move the normalization boundary.** Show the actual four activations on a number line, with full-group and local means. Then use a fresh activation table in the investigation: edit values and targets, assign groups, observe whether the downstream gradient agrees, and compare full, local and frozen-statistic computations. Freeze one shared reference set of statistics to see partition invariance return. Find a case where training outputs agree but running means differ. Frozen evaluation statistics define their own fixed computation; they are not a general replacement for training batch normalization.

Layer normalization over features within each independent example does not couple the example axis in this way. A contrastive loss whose negatives come from other batch rows, a batchwise ranking loss, or any operation that explicitly compares examples can have the same partition problem as batch normalization. Choose an implementation that preserves the needed cross-example information; simply adding gradients is insufficient.

### Dropout changes the realized computation

Training dropout samples a mask and scales surviving activations. At drop probability 1/2, the survivors are multiplied by two. If the full and chunked computations use the same realized mask for each example, an otherwise separable loss still accumulates correctly. Separate calls can consume random numbers differently; setting the same seed does not by itself guarantee those masks align across different call shapes.

For a constructed check, take $x=[1,2,3,4]$, targets $[1,0,1,0]$, prediction $\theta z$ at $\theta=1$, and the half-squared mean loss. A keep-mask `[1,0,1,0]` produces $z=[2,0,6,0]$ and gradient 8. The different mask `[0,1,0,1]` produces gradient 20. Partitioning the first fixed masked array preserves gradient 8. Disabling dropout gives another computation with gradient 6.5. These are exact mask calculations, not sampled learning curves. [Dropout reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html).

Stochastic equivalence in distribution and equality of one realized update are different claims. The Iris experiment intentionally uses a deterministic, example-separable network so its state comparison isolates the accumulation rule.

### A larger effective batch changes gradient variability

At fixed parameters, suppose independently drawn examples yield one gradient component with variance $\sigma^2$. Averaging $B$ such draws gives variance $\sigma^2/B$: the variance of their sum is $B\sigma^2$, and division by $B$ scales variance by $1/B^2$. Its standard deviation therefore shrinks as $1/\sqrt B$. Sampling without replacement from a finite dataset adds a finite-population correction; correlations between examples also change the calculation.

You can see the finite case exactly using our three row gradients [−2,0,−3]. Uniform single-row sampling has mean −5/3 and variance 14/9. The three equally likely two-row subsets have means [−1,−2.5,−1.5], with the same expectation but variance 7/18. Taking all three rows has no sampling variability at this fixed weight. These are enumerated possibilities, not a fitted learning curve.

Repartitioning the same effective group into microbatches leaves that group's intended gradient unchanged. Increasing the effective group itself changes how gradients are averaged and how many updates fit into an epoch. A learning-rate scaling rule is therefore a separate optimization choice; it does not follow from the accumulation identity. This is the statistical side of the batching tradeoff introduced in §1. [Dive into Deep Learning, §12.5.2](https://d2l.ai/chapter_optimization/minibatch-sgd.html#minibatches).

## 8. Deeper branch: operations that belong at the boundary

### Clipping and schedules

Gradient clipping limits the norm of a vector. Because it is nonlinear, clipping each chunk and adding differs from clipping the complete gradient. For scalar contributions 3 and −2.5 and a bound of 1, clipping their sum gives 0.5; adding their separately clipped values gives $1+(-1)=0$. To match a clipped full-batch update, first complete the correctly normalized gradient and then clip it once before the optimizer step.

A learning-rate schedule needs a declared clock. An update-based schedule advances after a successful optimizer update, not after each microbatch. An epoch-based schedule advances after the epoch; a metric-driven schedule receives the prescribed validation metric. PyTorch's ordinary scheduler order places the optimizer step first. If four chunks form one update, four scheduler ticks would accelerate an update-based schedule by four. [Optimizer and scheduler reference](https://docs.pytorch.org/docs/2.14/optim.html#how-to-adjust-learning-rate).

Regularization has a boundary too. If the objective includes $\lambda R(\theta)$ once per effective update, add its derivative once, or distribute coefficients across chunks that sum to one. Adding the entire regularizer to every chunk's already normalized data loss multiplies its influence. Decoupled optimizer weight decay happens when that optimizer steps; stepping more often changes its application frequency. Detailed optimizer choices belong to the optimization lessons.

### Automatic mixed precision

Mixed precision can compute selected operations at lower precision; loss scaling multiplies the loss and resulting gradients by a scale factor to help represent small gradients. The accumulation contract still begins with the correctly normalized objective.

For an AMP update group, use this order:

1. Clear gradients and know the group's target denominator.
2. For each chunk, compute its loss numerator under the appropriate autocast context, divide by the group denominator, and call backward on the scaled result.
3. Keep that scale fixed across the whole group. After the last backward, unscale the optimizer's gradients once.
4. Clip the complete unscaled gradient if requested, then let the scaler attempt the optimizer step and update its scale.
5. Advance an update-based schedule only if the optimizer update actually occurred; clear for the next group.

The scaler can skip an update when gradients contain infinities or NaNs. A skipped attempt still consumed data, so the example clock can advance while the successful-update clock does not. If you accumulate unnormalized sums because the denominator arrives late, normalize the complete unscaled gradient before clipping. The ordering is the same reasoning with normalization moved to the boundary. [AMP accumulation and clipping examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation).

**Figure G — boundary dependency strip.** Scaled chunk gradients join before a single unscale; normalized gradients join before a single clip; the accepted-update branch advances parameters and an update-based schedule. A skipped branch leaves those two states unchanged. This is an API-order explanation; this packet's executions are CPU float64, without AMP hardware validation.

### The same denominator problem appears across devices

Under default distributed data parallelism, replicas synchronize gradients and average across ranks. If each of $W$ ranks processes $b$ examples in each of $K$ chunks, the ordinary effective example count is $WbK$. The equality depends on all those contributions receiving the intended weight.

For unequal target masses, let rank $r$ supply numerator $S_r$ and mass $D_r$, with global $D=\sum_r D_r$. The desired gradient is $\nabla\sum_r S_r/D$. Default rank averaging of local means instead gives $\frac1W\sum_r\nabla S_r/D_r$. To recover the global weighted mean under this default averaging convention, each rank differentiates $W S_r/D$; averaging cancels $W$.

For two ranks holding our scalar rows `[a,b]` and `[c]`, the local means have gradients −1 and −3. Their rank average is −2. Scaling local summed gradients as $2(-2)/3$ and $2(-3)/3$ yields a rank average of $-5/3$, recovering the same objective as §3.

DDP's `no_sync()` can defer synchronization during early chunks, but it must surround their forwards as well as backwards. Every rank must follow compatible synchronization boundaries and establish the shared denominator; a rank with zero local mass still participates in the distributed protocol when other ranks have data. Custom communication hooks and uneven-rank termination need their own contract. Continue with [Data Parallelism (DDP)](/learn/topic/data-parallelism-ddp) for that implementation. [PyTorch DDP reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).

## 9. Practice: change the data and defend the boundary

Attempt each before opening its hint or solution. Exact arithmetic is welcome; compare executed float64 results with a small tolerance rather than requiring byte identity across libraries.

### 1. Count work without hiding the remainder

You have 23 training examples, microbatch size five, and three microbatches per update. Keep every example and flush at epoch end. Give the microbatch sizes, effective group sizes, backward-call count and update count for one epoch. What changes if you turn on loader `drop_last`?

<details><summary>Hint</summary>
List the actual chunks before grouping them. Dropping a short loader chunk is different from dropping an incomplete accumulation group.
</details>

<details><summary>Solution</summary>
Chunks are 5,5,5,5,3. The effective groups are 15 and 8, giving five backward calls and two updates. Their denominators are 15 and 8. With loader drop-last, the last three examples disappear: chunks become 5,5,5,5 and groups 15 and 5, with four backward calls and two updates if the remaining group is flushed. A loop that steps only on multiples of three would incorrectly leave the final five processed examples unapplied.
</details>

### 2. Reconstruct a changed scalar update

Use inputs `[1,2,4]`, targets `[0,1,2]`, initial $w=0$, half-squared mean loss, and SGD learning rate 0.1. Partition as a one-row chunk and a two-row chunk. Compute the correct gradient and weight, then the equal average of chunk-mean gradients. Explain which row the naive rule overweights. Next put all rows in one chunk and predict whether the disagreement remains.

<details><summary>Hint</summary>
The per-row derivative is $(wx-y)x$. First write one derivative per row, then write the coefficient assigned to each.
</details>

<details><summary>Solution</summary>
Derivatives are 0,−2,−8. The correct gradient is −10/3, so the weight becomes 1/3. Chunk means have gradients 0 and −5; their equal average is −2.5, giving weight 0.25. The one-row chunk receives half the objective's total mass, so its first row gets weight 1/2 instead of 1/3. It happens to have zero gradient, which reduces the other rows' combined influence. In one chunk, its mean is already the full mean, so both rules agree. Agreement on that null case does not validate the uneven-chunk rule.
</details>

### 3. Pick the objective for variable-length sequences

Sequence A has three eligible targets with loss sum 6. Sequence B has one eligible target with loss sum 5. Each is padded to length five. Compute the token mean, sequence mean and incorrect padding-count mean. If A's loss numerator has derivative 9 and B's has derivative −1, give the corresponding token-mean and sequence-mean gradients. What should happen if all positions are ignored?

<details><summary>Hint</summary>
The ten tensor slots are not ten targets. For the sequence mean, average within each sequence before averaging across sequences.
</details>

<details><summary>Solution</summary>
The token mean is 11/4=2.75. The sequence mean is (6/3+5/1)/2=3.5. Dividing by ten gives 1.1, incorrectly counting padding. The token-mean gradient is (9−1)/4=2. The sequence-mean gradient is (9/3−1)/2=1. Neither legitimate objective can be selected purely by the number of rectangular tensor slots. With no eligible target mass, the mean is undefined and there should be no optimizer or update-scheduler step for that group.
</details>

### 4. Transfer the Iris loop

First change only the accumulated microbatch size from 12 to 7, keeping the thirty-two-example effective groups and final twenty-four-example group. Predict the twenty-epoch call count and the relationship between the two models' states. Then change the effective-group size in both runs from 32 to 25 and use microbatch limits 25 and 6. State the new group sizes, denominators, update count, and the state comparison you would use. Do not predict a new classification score from the old one.

<details><summary>Hint</summary>
A thirty-two-row group needs five chunks of at most seven; a twenty-four-row group needs four. For the second task, partition 120 into groups of at most 25 before splitting those groups into chunks.
</details>

<details><summary>Solution and success criteria</summary>
The first change gives 3×5+4=19 calls per epoch, or 380 in twenty epochs. There are still 80 updates, with the same effective data and optimizer schedule; float64 parameter and momentum differences should remain close to rounding error. The author additionally executed this change and observed maximum parameter gap 4.441e−16 and momentum gap 5.551e−17.

With effective limit 25, groups are 25,25,25,25,20, so each epoch has five updates and twenty epochs have 100. The microbatch limit six produces 6+6+6+6+1 for each group of 25 and 6+6+6+2 for the final 20. The denominators are 25 or 20, never the nominal number of chunks. Compare the two new runs' parameters and momentum after the same 100 updates, with identical initialization and row orders. These new runs need not match the old 32-example runs because gradients are evaluated at different intermediate parameter states. This second changed-group experiment is an independent exercise; its classification outcome is not supplied or preselected. Success is a correctly explained and checked paired-state agreement, not beating the earlier score.
</details>

### 5. An exact match breaks after adding a layer

A deterministic classifier's full-group and accumulated updates agreed. You add training-mode batch normalization, preserve the example-weighted denominator, and the gradients now differ. A colleague suggests dividing by one more factor of $K$. Explain why that is the wrong repair. Propose a discriminating check and explain a case where outputs agree yet some model state still differs. Then place unscale, clipping, optimizer step and an update-based scheduler around two AMP microbatches.

<details><summary>Hint</summary>
Compare the activations used to compute each loss before changing a coefficient. Track forward-updated buffers separately from trainable weights.
</details>

<details><summary>Solution</summary>
Batch normalization changes each example's normalized activation according to the examples in the physical forward. An extra factor changes gradient size without restoring those activations. First compare full versus local means, variances and normalized outputs on the same fixed inputs; then repeat with one frozen shared set of statistics. The frozen computation should be partition-invariant in the otherwise separable model. If both chunks have the same mean and variance, current normalized outputs can agree with the full batch, while two running-statistic updates differ from one. For example, mean 1, coefficient 0.1 and initial running mean 0 gives 0.1 after one update and 0.19 after two.

For AMP, keep one scale across both normalized chunk losses and their backwards. Unscale once after both; clip the complete unscaled gradient; attempt the optimizer step; update the scaler; advance the update-based scheduler only for an accepted optimizer update. Clearing between the backwards discards the first chunk. Changing the scale between them mixes incompatible gradient units.
</details>

## 10. Check readiness and continue

Without looking back, identify the examples and denominator belonging to one update, explain why backward leaves weights unchanged, distinguish gradient clearing from optimizer memory, and handle a partial final group. Then explain why a correct accumulation sum can still differ from a physical large batch with training-mode batch normalization. If these are clear, you have a useful contract against which to judge a real training run.

The next topic in this module is [Neural Training Diagnostics & Reproducible Experiments](/learn/topic/neural-training-diagnostics-reproducible-experiments). It uses this correct-loop contract to isolate faults, design controlled comparisons and decide what training evidence supports. Later [Mixed Precision Training (FP16, BF16, TF32)](/learn/topic/mixed-precision-training-fp16-bf16-tf32) and [Data Parallelism (DDP)](/learn/topic/data-parallelism-ddp) develop the specialized execution branches introduced here.

## References & another way to learn it

**Alternate explanations and practice**

- [PyTorch, Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — beginner article and runnable tutorial connecting loss, backward and optimizer steps. Read after §2. Its simple batch-mean reporting should be adapted using §5 when batch sizes differ; its FashionMNIST setup downloads data, while this lesson's Iris route is offline.
- [Zhang, Lipton, Li and Smola, Dive into Deep Learning §12.5: Minibatch Stochastic Gradient Descent](https://d2l.ai/chapter_optimization/minibatch-sgd.html) — free textbook chapter linking vectorized computation, gradient variability and mini-batch optimization. Read after §4 for the systems/statistics connection. Its section agenda and selected PyTorch explanations/code were reviewed; hardware timing examples are the book's environment, not measurements from this packet.
- [PyTorch, Training with PyTorch — video and companion notebook](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html) — a guided alternate walkthrough from datasets to a training/validation loop, suitable after the first scalar trace. The companion article's introduction, data abstractions and training/validation code were reviewed; the embedded video was not watched, and no timestamps are claimed. It assumes the preceding PyTorch video-series material and uses FashionMNIST and TensorBoard. Keep this lesson's explicit accumulation boundary when adapting its one-batch-per-step loop.

**Precise API and data references**

- [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html), [MSELoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html), and [zero_grad](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html) — inspect the reduction, target-type and missing-gradient semantics when adapting the programs. These annotations refer to the reviewed PyTorch 2.14 snapshot.
- [BatchNorm1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html) and [Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html) — reference contracts behind §7's physical-batch and realized-mask examples.
- [AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation) and [DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync) — intermediate execution references after §8. Their CUDA/distributed examples were read, not run in this CPU packet.
- [Fisher, Iris, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris) and [scikit-learn load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) — attribution, feature meanings and the corrected bundled variant. The packet's provenance record distinguishes its offline export from the older UCI file.
