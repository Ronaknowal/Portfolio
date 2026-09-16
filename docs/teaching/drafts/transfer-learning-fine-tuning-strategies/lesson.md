# Transfer Learning & Fine-Tuning: Reuse, Adapt, and Verify

A model has learned to recognize handwritten digits 0–4. You now need a model for digits 5–9, with only eight labeled examples of each new digit. Can the first model help?

Possibly. Its hidden layers may already respond to useful stroke patterns. They may also discard distinctions that the new task needs. **Transfer learning means reusing something learned on one problem to help with another. Whether it helps is an experimental question.**

Here you will replace a prediction head, decide which parts may change, make a low-rank update by hand, and run a complete offline comparison. The experiment includes a transferred model that performs worse than training from scratch. That result is part of the lesson.

**First pass:** follow sections 1–5, run or inspect the experiment in section 7, and solve practice 1–3. You should be able to explain what was transferred, what was frozen, and which evidence justified choosing a model. Sections 6 and 9 provide optional depth on low-rank derivatives, schedules, and larger-model methods; practice 4–6 checks that depth. You do not need to know convolution or Transformer attention to complete the first pass.

## 1. What exactly moves from one problem to another?

Write a classifier as two cooperating functions:

\[
x \longrightarrow h=g_\psi(x) \longrightarrow a=h_\theta(h).
\]

The **backbone** \(g_\psi\) turns an input into a feature vector. The **head** \(h_\theta\) turns that vector into output scores, called logits. The weights \(\psi,\theta\) determine both functions. Cross-entropy compares those scores with the label; backpropagation computes how trainable weights should change.

In our digit model, 64 pixel values become 32 hidden values, then 16 features, then five scores. The source scores mean digits 0, 1, 2, 3, 4. The target scores must mean 5, 6, 7, 8, 9. Both heads have five outputs, but their label meanings differ. Keeping the old head just because its shape fits is a semantic error.

**Pretraining** is the earlier source learning. **Fine-tuning** is further training a pretrained model for the target setting. A **linear probe** trains a new linear head while keeping the feature function fixed. Transfer also includes reusing learned features without further backbone training.

A domain describes inputs and their distribution: for example, scanned handwriting from particular writers and equipment. A task describes the desired output: digit identity, writer identity, or whether a scan is readable. The task can change within a similar domain, or the input distribution can change while the task remains the same. In our experiment the label set changes; the scans come from the same collection. This is not a test of transfer across hospitals, languages, or sensors.

Picture the source backbone as a learned measuring instrument. Reusing it can save learning useful measurements again. However, a measuring instrument that records only weight cannot recover color, however powerful the new decision rule is. A weak probe may indicate missing target information, but it can also reflect poor optimization, unsuitable regularization, or a nonlinear separation that a linear head cannot express. It does not prove the representation contains no useful information.

For some image networks, earlier features transfer more broadly than later features. The classic study also identified disruption of features that had learned to work together—**coadaptation**—as a reason transfer can fail. Layer position alone does not determine usefulness. [Yosinski et al., *How transferable are features in deep neural networks?*](https://arxiv.org/abs/1411.1792)

## 2. Choose what the target task may change

The choices below impose different restrictions on the function you can learn.

| Strategy | Trainable parts | What it asks |
| --- | --- | --- |
| Train from scratch | Backbone and new head, starting from random weights | Can target data support learning these features directly? |
| Linear probe | New linear head | Are fixed source features already useful for this decision? |
| Partial fine-tuning | New head and selected backbone layers | Can a limited part of the representation adapt enough? |
| Full fine-tuning | New head and all backbone layers | Does adapting the entire representation help? |
| Parameter-efficient fine-tuning, or PEFT | Small added parameters, often plus a head | Can a restricted update adapt the model using fewer trainable values? |

“Full” describes which parameters may change, not whether their learning rates must match. A low learning rate for the pretrained backbone and a higher one for a new head can be a reasonable candidate. The head needs to learn its new label meanings; the backbone already implements a learned function.

Start with an appropriate simple baseline and a probe when available. If the probe fits training examples poorly, allowing representation changes is a useful next investigation. If it fits training examples well but validation performance is poor, adding trainable capacity might worsen overfitting. Inspect the split, preprocessing, labels and error patterns before interpreting every failure as “not enough fine-tuning.”

**Negative transfer** means reuse harms performance relative to an appropriate target-only comparison under a stated protocol. It is not merely low accuracy. A target-only model might perform even worse. Compare the same target rows and evaluation rule, and describe differences in optimization budgets. One experiment does not establish that a source domain is always harmful.

### A freeze has three separate meanings to check

1. **Parameter gradients:** `requires_grad_(False)` prevents accumulation of gradients for those parameters.
2. **Optimizer membership:** an optimizer changes the parameters it owns using their available gradients and update rules. Construct it from the intended trainable parameters and clear stale gradients when changing a freeze policy.
3. **Model state:** training mode can update BatchNorm running statistics or enable dropout even when all weights are frozen.

The preceding [normalization lesson](/learn/path/full-curriculum/batch-layer-group-rms-normalization?module=deep-learning-fundamentals) explained why `eval()` and `no_grad()` are independent. Evaluation mode changes module behavior. Disabling gradient recording changes autograd. Neither substitutes for a complete freeze policy.

Our backbone uses linear layers and tanh, so it has no running statistics or dropout. For a frozen pretrained backbone that does have such state, a common policy is to call `model.train()` and then `model.backbone.eval()` at the start of each training epoch. This deliberately keeps the head in training mode and the backbone in evaluation mode. Adapting BatchNorm statistics is another policy; specify and validate it separately.

Frozen weights can still transmit gradients to their inputs. If an upstream adapter produces \(u\), and a frozen matrix computes \(Wu\), the adapter needs the derivative through \(W\). Wrapping that whole path in `no_grad()` would cut it. When a frozen feature extractor receives ordinary data and only a downstream head is trained, features can instead be computed without a graph, or cached if preprocessing is deterministic.

## 3. A complete small transfer pipeline

We use 400 real 8×8 handwritten scans from the UCI Optical Recognition of Handwritten Digits collection: 40 per digit. The values are integers from 0 through 16. Divide by the known feature-range maximum 16; do not estimate a new scale using the holdout. This collection is distinct from MNIST.

The downloadable [data](digits-400.csv) include `source_id`, 64 row-major pixels, and `digit`. The [provenance](data-provenance.md) identifies the extraction and license.

| Partition | Digits | Rows per digit | Purpose |
| --- | --- | --- | --- |
| Source training | 0–4 | First 30 | Learn the initial backbone and source head |
| Source holdout | 0–4 | Last 10 | Describe retention with the original head |
| Target training | 5–9 | First 8 | Fit each adaptation candidate |
| Target validation | 5–9 | Next 12 | Compare candidates |
| Target test | 5–9 | Last 20 | Report the one selected model |

These are fixed, disjoint row blocks within the extract, not random writer groups. Writer identifiers are unavailable in this file, so the experiment cannot estimate generalization to unseen writers. Row order can also make partitions differ in difficulty. The small experiment teaches the protocol and mechanisms; it is not an official benchmark score. Its source holdout is used for descriptive retention, never for selecting the target model.

The complete [CPU program](transfer-experiments.py) defines every model, optimizer, split, training loop and calculation. Put it beside the CSV. One setup for a separate learner environment is:

```bash
python -m venv .venv
# Activate .venv using your operating system's activation command.
python -m pip install torch==2.14.0 numpy==2.3.5
python transfer-experiments.py
```

The recorded run used Python 3.12.14 and PyTorch 2.14.0+cpu. It needs no pretrained download, account or GPU. The program limits PyTorch to one CPU thread, writes `calculated-inputs.json` and prints validation results followed by the selected test result. Exact floating-point last digits may vary across environments.

Follow its data flow before changing settings:

1. Initialize a 64→32→16 tanh backbone and a five-output source head. Train on the 150 source training rows for 400 full-batch Adam updates, learning rate 0.01.
2. Save both the original random backbone and the learned backbone. Initialize one new five-output target head. Every method in that seed receives an identical copy of this head.
3. Construct six candidates: scratch, probe, full fine-tuning, discriminative rates, rank-2 LoRA, and a bottleneck-4 adapter. Train each for 300 full-batch updates on the same 40 target rows.
4. Compare their final validation cross-entropies. Seed 1 is the predeclared selection seed. Choose its lowest validation loss; exact ties use the listed method order.
5. Seeds 2 and 3 provide sensitivity comparisons on validation only. They do not change the selection rule. Freeze the exact selected seed-1 weights, with no refit, and evaluate that model on the 100 target test rows.

The head rate is 0.01 throughout. Scratch and full fine-tuning use 0.001 for both backbone layers. Discriminative fine-tuning uses 0.0001 for the lower layer and 0.001 for the upper layer. LoRA factors and adapter weights use 0.01. These are six declared training procedures, not a sweep proving each method has received its optimal settings.

## 4. Make an update without replacing the whole weight matrix

Suppose a pretrained linear layer uses \(W\in\mathbb R^{d\times k}\), with \(k\) inputs and \(d\) outputs. Full fine-tuning can independently change its \(dk\) weights. **Low-rank adaptation**, or LoRA, adds a product of two smaller matrices:

\[
y=Wx+sB(Ax),\quad
A\in\mathbb R^{r\times k},\quad
B\in\mathbb R^{d\times r},\quad
s=\alpha/r.
\]

First \(A\) measures \(r\) combinations of the input. Then \(B\) distributes these measurements across output coordinates. This update has rank at most \(r\): its output changes lie in the span of \(B\)'s columns. The restriction applies to the update \(BA\), not to the pretrained matrix \(W\) or to the entire nonlinear network.

For a 4×6 matrix, rank 2 requires \(2(6+4)=20\) factor weights instead of 24 unrestricted weights. Rank 3 requires 30 factor weights, so “low rank” does not automatically mean fewer parameters at every small shape. In general the saving requires \(r(d+k)<dk\). Also count biases, trainable heads and other modules.

The original LoRA method freezes \(W\), initializes one factor randomly and the other to zero, and permits merging a trained update into the base matrix. [Hu et al., *LoRA*](https://arxiv.org/abs/2106.09685) Our program uses Gaussian standard deviation 0.1 for \(A\), zero \(B\), \(r=2\), \(\alpha=2\), and adapters on both backbone matrices. These are explicit teaching choices.

The scale \(s\) controls the multiplier, but \(\alpha/r\) does not make changing rank optimization-neutral. Rank changes the number and initialization of factors, the possible update directions, and their gradients. Keep the data fixed when comparing ranks; do not generate a new task for each rank.

### Watch the first update

Use \(W=I_2\), \(A=[1,-1]\), \(B=[0,0]^T\), \(s=1\), \(x=[2,1]^T\), and target \([0,0]^T\). The initial output is \([2,1]^T\), because \(B=0\). With mean squared error over the two outputs, loss is \(2.5\).

The scalar bottleneck measurement is \(Ax=1\). The gradient for \(B\) is \([2,1]^T\), while the gradient for \(A\) is zero. After one SGD step of size 0.1, \(B=[-0.2,-0.1]^T\). The output becomes \([1.8,0.9]^T\), and the loss is \(2.025\).

This is why zero initial update does not have to mean zero learning. One factor starts ready to transmit a useful signal. Setting **both** factors to zero gives zero gradients for both in this example. The interactive factor editor asks you to predict which factor can change before revealing these calculations.

Once trained, form \(W_{\text{merged}}=W+sBA\). For a plain linear layer this is algebraically equivalent to the separate paths. Floating-point multiplication orders differ: our float64 hand example differs by about \(1.1\times10^{-16}\); the seed-1 float32 digit model differs by about \(1.9\times10^{-6}\) in validation logits. Compare with a suitable tolerance, not a promise of byte-identical outputs.

Merging removes the extra low-rank path for that fixed adapter. Keeping factors separate makes switching tasks convenient. Quantized weights, active adapter dropout, or incompatible module types require their own merging contract. Preserve the original checkpoint and configuration either way.

## 5. Change features through a bottleneck adapter

A bottleneck adapter changes a feature vector rather than directly parameterizing a weight update:

\[
h'=h+U\,\tanh(Dh+b_D)+b_U,
\quad D\in\mathbb R^{b\times d},\quad U\in\mathbb R^{d\times b}.
\]

The narrow intermediate vector has \(b\) values. The added path learns a correction, and the direct \(h\) path keeps the original features available. This introduces the idea of a skip connection; the [later residual lesson](/learn/path/full-curriculum/residual-connections-skip-connections?module=deep-learning-fundamentals) develops its gradient and architecture consequences.

Our \(d=16,b=4\) adapter has \(64+4+64+16=148\) parameters. With the 85-parameter target head, 233 parameters train. We initialize the up-projection weight and bias to zero, making this particular adapter exactly the identity initially. The down-projection starts random. After the up-projection moves, gradients can train the down-projection as well.

This is a small fully specified implementation of the bottleneck idea, not a reproduction of every detail of the original Transformer adapter architecture. [Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*](https://arxiv.org/abs/1902.00751) The program's `BottleneckAdapter` defines the complete forward pass and participates in the same target experiment as the other methods.

An adapter may contain nonlinear computation and therefore is not generally mergeable into one fixed linear weight. It also changes the active feature function even though the base parameters remain untouched.

## 6. Deeper: derivatives tell you what “frozen” actually preserves

For a single input let \(\delta=\partial L/\partial y\). LoRA's derivatives are

\[
\frac{\partial L}{\partial B}=s\,\delta(Ax)^T,\qquad
\frac{\partial L}{\partial A}=s\,B^T\delta x^T,\qquad
\frac{\partial L}{\partial x}=W^T\delta+sA^TB^T\delta.
\]

The frozen base weight has no optimizer update, but \(W^T\delta\) still contributes to the input gradient. In our saved fixture, a frozen identity matrix with the same squared loss transmits input gradient \([2,1]^T\), while its weight has no stored gradient.

With \(B=0\), the \(A\) gradient vanishes on the first step; the \(B\) gradient need not vanish. This is a local explanation, not a claim that identical factor learning rates are optimal. [LoRA+](https://arxiv.org/abs/2402.12354) investigates different rates for the two factors using width-scaling arguments and experiments. Its findings do not provide a universal rate ratio for our small network.

**Freezing parameters preserves their stored values. It does not guarantee preservation of the adapted model's old behavior.** Feed the changed representation into the original source head to measure one form of forgetting. Disabling an unmerged adapter can restore the base function when the same base weights, buffers, preprocessing and original head are restored. Retaining a full pre-fine-tuning checkpoint also lets full fine-tuning be reversed.

## 7. Read the evidence before choosing the method

These are actual results after 300 target updates. Cross-entropy is the mean loss in natural-log units; lower is better. Accuracy is shown as a count so the size of the evidence remains visible.

| Seed-1 method | Trainable values | Target train correct / 40 | Validation CE | Validation correct / 60 | Source holdout correct / 50 after adaptation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scratch | 2,693 | 40 | 0.123143 | 58 | Not applicable |
| Probe | 85 | 37 | 0.738785 | 44 | 49 |
| Full fine-tuning | 2,693 | 40 | 0.346375 | 53 | 49 |
| Discriminative rates | 2,693 | 40 | 0.568471 | 48 | 49 |
| LoRA rank 2 | 373 | 40 | 0.369318 | 53 | 44 |
| Adapter width 4 | 233 | 40 | 3.353089 | 39 | 49 |

Before target adaptation, the source model got 49/50 source holdout rows correct. The probe preserves that result exactly because its backbone and original head remain unchanged. Its new head still cannot fit all 40 target training examples under the declared training procedure.

Several methods fit all target training examples, yet their validation results differ greatly. The adapter's high validation loss alongside 39/60 correct indicates that some errors receive especially costly probabilities. A training score alone would hide this.

Scratch has the lowest seed-1 validation CE, so the predeclared rule selects it. The other seeds also favor scratch under these settings:

| Method | Seed 2 validation CE; correct / 60 | Seed 3 validation CE; correct / 60 |
| --- | --- | --- |
| Scratch | 0.158462; 58 | 0.082299; 59 |
| Probe | 0.781923; 42 | 0.785892; 46 |
| Full | 0.341880; 54 | 0.362536; 51 |
| Discriminative | 0.423951; 50 | 0.677043; 49 |
| LoRA rank 2 | 0.417762; 52 | 0.333299; 56 |
| Adapter width 4 | 2.350006; 40 | 3.027634; 35 |

The selected seed-1 scratch model then gets **77/100 target test rows correct**, with CE **0.757101**. This is substantially worse than its validation result. Small fixed row blocks need not represent equally difficult populations, and choosing by validation can favor a candidate on that validation set. These data alone cannot isolate the causes of the gap.

The correct response is to report the gap and the partition limitations. Trying alternatives on these same test labels would turn the test into more development data. A future study could predeclare a stronger split with writer information and more examples, or compare learning rates and source objectives using development data. It would need new final evidence for a fresh performance claim.

Forgetting also needs careful measurement. Seed-1 LoRA lowers original-head source accuracy from 49/50 to 44/50, although its base weights are frozen. Full fine-tuning retains 49/50 here. This does not prove full fine-tuning always forgets less; it refutes the claim that a frozen base guarantees no forgetting of the adapted function. Accuracy can also remain unchanged while probabilities move: inspect the recorded source cross-entropies.

All plotted training trajectories come from the saved steps 0, 1, 10, 100 and 300. A connecting line shows those samples; it is not a record of every intervening update or a benchmark of elapsed time.

## 8. A usable checkpoint includes meaning, not only tensors

To reproduce a prediction, preserve the architecture, weight values, buffers, preprocessing, output-label order and adaptation configuration. For our target task that includes the 64-pixel ordering, division by 16, hidden widths 32 and 16, tanh, and output labels [5, 6, 7, 8, 9].

A LoRA checkpoint additionally needs the base identity, targeted layers, factor rank, \(\alpha\), scaling convention, biases, and any separately trained head. A file containing only \(A,B\) is insufficient to identify the function. The same issue appears with image transforms and language tokenizers: a compatible tensor shape does not establish the same input or label meaning.

Our program checks a state-dictionary round trip through ordinary lists with the model configuration held fixed; the selected probabilities match exactly in that run. It is a local replay check, not a complete deployment package. When saving a reusable artifact, save a machine-readable configuration alongside it and a small input/output fixture.

For a cached probe, an additional practical advantage is possible: compute fixed features once and train multiple heads on those features. This is exact only for the preprocessing and backbone state used to produce the cache. Random image augmentation creates different inputs, and adapting BatchNorm statistics changes the feature function; either makes a stale cache inappropriate. Name cache entries by source row, preprocessing version and backbone checkpoint, not merely “features.”

For a service hosting many related tasks, one base plus several small adapters can reduce duplicated stored weights. Each task still needs evaluation, correct routing and compatible preprocessing. Merging one adapter favors a fixed serving path; retaining adapters favors switching. This is a concrete architectural tradeoff rather than a guarantee of lower end-to-end latency.

## 9. Further choices once the basic comparison is sound

### Unfreeze gradually; make the schedule explicit

Partial fine-tuning can begin with the head, then release the upper backbone layer, then lower layers. When releasing a layer, include its parameters in the optimizer and decide whether to preserve existing optimizer state. Rebuilding the optimizer resets its moments unless you deliberately transfer them.

ULMFiT combined language-model pretraining, adaptation to target-domain text, and classifier fine-tuning. It used layer-dependent learning rates, a rise-and-decay schedule, and gradual unfreezing. Its empirically chosen layer-rate divisor 2.6 belongs to that study, not a law of neural networks. [Howard and Ruder, *ULMFiT*, §3](https://aclanthology.org/P18-1031/)

For an independently specified teaching schedule, let total duration \(T=100\), peak time \(c=10\), floor fraction \(\rho=1/32\), and peak rate \(\eta_{\max}=0.01\):

\[
q(t)=
\begin{cases}t/c&0\le t\le c,\\(T-t)/(T-c)&c<t\le T,\end{cases}
\qquad \eta(t)=\eta_{\max}\,[\rho+(1-\rho)q(t)].
\]

It starts and ends at 0.0003125 and peaks at 0.01. This triangle is a transparent teaching variant, not a claim to reproduce ULMFiT's printed schedule formula exactly. A scheduler must define whether the rate is sampled before or after each update. Warmup limits early step sizes; it does not guarantee preservation of features or a particular kind of minimum.

### Know which part each efficient method changes

These are optional entry points. Language-model-specific mechanisms are developed after attention and token representations in the later curriculum.

| Method | Distinct mechanism | What to investigate before adopting it |
| --- | --- | --- |
| LoRA | Factorized additive weight updates | Rank, target layers, initialization, scaling and task head |
| DoRA | Separates weight magnitude and direction, with low-rank directional adaptation | Additional state and the evaluated architecture; not an automatic upgrade |
| QLoRA | Trains adapters through a frozen quantized base | Quantization format, dequantization computation, memory overhead and supported hardware |
| Bottleneck adapters | Add small feature transformations | Placement, nonlinearity and serving overhead |
| Soft prompt tuning | Learns continuous input vectors while the model is frozen | Requires differentiable access to the input representation |
| Prefix tuning | Learns continuous conditioning that later tokens can attend to | Layer placement, extra sequence/state cost and task fit |

DoRA's magnitude/direction split and QLoRA's quantized-base training solve different problems. QLoRA includes NF4 quantization, quantization of scaling information and paged optimizers; its reported 65-billion-parameter experiment on a 48 GB GPU is a particular setup, not a general memory-fit promise. [DoRA paper](https://arxiv.org/abs/2402.09353), [QLoRA paper](https://arxiv.org/abs/2305.14314)

Soft prompts are learned numerical vectors, not automatically discovered human-readable instructions. A hosted text-generation endpoint does not necessarily expose the gradients or embeddings required to train them. Prefix methods condition internal generation differently from merely adding the same input-vector count. [Prompt tuning](https://arxiv.org/abs/2104.08691), [Prefix tuning](https://arxiv.org/abs/2101.00190)

AdaLoRA allocates an update budget across weight matrices using importance estimates and a singular-value-style parameterization. This differs from pruning whichever raw LoRA factor entries happen to be small. IA³ learns multiplicative activation scales; it restricts changes to selected feature rescalings rather than adding an arbitrary matrix update. Their value depends on which restriction matches the task; a catalogue of method names is not a selection procedure. [AdaLoRA](https://arxiv.org/abs/2303.10512), [IA³](https://arxiv.org/abs/2205.05638)

### Account for memory in units

For \(P\) parameters, 16-bit weights alone require \(2P\) bytes. Seven billion such weights require 14 GB in decimal units. Raw 4-bit storage would require \(P/2\) bytes, or 3.5 GB, before scale metadata and other overhead.

An illustrative trainable-parameter budget with FP32 gradients and two FP32 Adam moments adds \(4+8=12\) bytes per trainable value, excluding weights and any master copy. “Optimizer moments” alone are 8 bytes, not 12. Actual implementations may use different precision, allocation or sharding.

PEFT can greatly reduce gradients and optimizer-state storage while retaining the base model's weight storage and significant activation memory. Sequence length, batch size, where trainable modules sit, and checkpointing affect the latter. Parameter count is therefore a useful exact calculation, not a measurement of browser speed, GPU throughput or total training memory.

## 10. Practice and transfer

### 1. A head that fits but means the wrong thing

The old head outputs three scores for [cat, dog, horse]. Your new task is [healthy, scratched, broken], also three classes. A colleague loads the old head unchanged because the dimensions match. What should change, and what must be saved for inference?

<details><summary>Hint</summary>

distinguish a vector's length from its meaning.

</details>

<details><summary>Worked solution</summary>

normally initialize and train a new three-output head for the new task. Assess whether the backbone is useful; shape compatibility is insufficient. Preserve the new class order, input preprocessing, architecture and corresponding weights. Keeping the old head is a candidate initialization only if deliberately tested, not a completed transfer.

</details>

### 2. Decide from a new validation table

All candidates use the same 80 validation rows. The rule was declared as lowest validation CE, with a maximum of 500 trainable parameters. Scratch uses 2,000 parameters and CE 0.30. Probe uses 120 and CE 0.55. LoRA uses 480 and CE 0.42. Adapter uses 360 and CE 0.47. Which is eligible and selected? Can you evaluate all four on the final test to reconsider?

<details><summary>Hint</summary>

apply the constraint before minimizing.

</details>

<details><summary>Worked solution</summary>

probe, LoRA and adapter are eligible; LoRA wins among them. Scratch's lower loss does not satisfy the declared resource constraint. Evaluate the selected artifact on the final test to report its performance. Using test outcomes to change the choice consumes that holdout for development; it no longer supports the original untouched-test claim.

</details>

### 3. A freeze that still changes predictions

A probe's backbone parameters stay bit-for-bit unchanged, but its feature vector for the same original image changes between epochs. Name two state or input mechanisms worth checking. When could feature caching be invalid?

<details><summary>Hint</summary>

Look beyond trainable parameters: what else is read or updated during a forward pass?

</details>

<details><summary>Worked solution</summary>

check whether BatchNorm running statistics update in training mode, and whether dropout or random augmentation changes the forward computation. A cache is invalid when its preprocessing or backbone function differs from the active one. `no_grad()` alone prevents none of those training-mode behaviors.

</details>

### 4. Change the LoRA example

Keep \(W=I_2,A=[1,-1],B=0,s=1\), zero target and mean squared loss, but use \(x=[1,3]^T\). Find the first \(B\) gradient and output after one step of size 0.1. Then choose a nonzero input for which this first update vanishes.

<details><summary>Hint</summary>

calculate \(Ax\) before any matrix gradient.

</details>

<details><summary>Worked solution</summary>

\(Ax=-2\), output gradient is \([1,3]^T\), so \(\nabla_B=[-2,-6]^T\). The updated \(B=[0.2,0.6]^T\) adds \([-0.4,-1.2]^T\), yielding \([0.6,1.8]^T\). The new loss is 1.8. For \(x=[1,1]^T\), \(Ax=0\) and \(B=0\), so both factor gradients vanish despite positive loss. Changing the input can remove the learning signal without changing the optimizer.

</details>

### 5. Count, then question the count

A layer has 1,024 inputs and 256 outputs. Find the LoRA factor count at rank 8, excluding bias, and compare with full weight tuning. With 4-byte gradients and two 4-byte moments, how many bytes do those trainable states need? Does that predict total memory?

<details><summary>Hint</summary>

Write the shapes of both factors, then count gradients and the two optimizer moments separately.

</details>

<details><summary>Worked solution</summary>

full tuning has \(1024\times256=262{,}144\) weights. LoRA has \(8(1024+256)=10{,}240\), or 3.90625% as many. The stated trainable states require 3,145,728 bytes versus 122,880 bytes. Base weights, any master weights, activations and temporary buffers remain outside that calculation.

</details>

### 6. Design a rank investigation that can answer its question

A script creates a fresh random target dataset for ranks 1, 2, 4 and 8, then plots accuracy against rank. Redesign it. What can a flat result establish?

<details><summary>Hint</summary>

Identify which quantities besides rank change across runs, and which behavior a flat correct-count metric might conceal.

</details>

<details><summary>Worked solution</summary>

keep target rows, splits, base checkpoint, label mapping, training budget and evaluation rule fixed. State initialization and scaling policies, use controlled seed repetitions, and count head parameters as well. Compare development results; select before using final evidence. Flat accuracy means those settings did not change that discrete metric detectably. Check loss and uncertainty. It does not reveal the true rank of the ideal update or prove higher rank can never help.

</details>

## Where to go next

On the first-pass route, you are ready when you can explain a backbone/head split, construct a deliberate freeze policy, compare transfer with a target-only baseline, and keep selection separate from final reporting. The deeper route adds factor gradients, memory accounting and controlled adaptation experiments.

The next module topic is [Weight Initialization: Xavier, Kaiming & μP](/learn/path/full-curriculum/weight-initialization-xavier-kaiming-p?module=deep-learning-fundamentals). Transfer starts from learned weights, but a new head, adapter or scratch baseline still needs an initial state. We will investigate how that state changes signal and gradient behavior before training has learned anything.

## References and other ways to learn

- [PyTorch: Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — an alternate practical route with pretrained ResNet18 and ants/bees images. Read the data transforms, head replacement, optimizer construction and custom-image inference. It requires external weights/data and knowledge of convolution. Its shared training loop puts the entire model in training mode, so its “fixed feature extractor” freezes weights while BatchNorm buffers can still update. Apply the explicit state policy taught here. Tutorial updated January 2025; documentation served as 2.14 in the author review.
- [Stanford CS231n 2017, Lecture 7: Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA) — official course video covering optimization and transfer, useful after the first-pass experiment for another explanation of adapting image models. It predates LoRA and current library APIs. The author verified the official title, description and syllabus association, not the full recording.
- [Yosinski et al.](https://arxiv.org/abs/1411.1792) — study of generality, specificity and coadaptation; read the experimental setup before generalizing its layer conclusions.
- [ULMFiT, §3](https://aclanthology.org/P18-1031/) — source for the three-stage language-model adaptation strategy and its schedule/unfreezing choices.
- [LoRA, §4](https://arxiv.org/abs/2106.09685) and [PEFT 0.20.0 LoRA documentation](https://huggingface.co/docs/peft/v0.20.0/en/package_reference/lora) — separate the mathematical mechanism from a library's initialization, target-module and merge behavior. The documentation has many model-specific snippets; the self-contained CPU program here does not require that package.
- [Adapter paper](https://arxiv.org/abs/1902.00751), [DoRA](https://arxiv.org/abs/2402.09353), [QLoRA](https://arxiv.org/abs/2305.14314), and [LoRA+](https://arxiv.org/abs/2402.12354) — optional method families with different restrictions and resource goals. Their reported benchmark improvements are evidence for their settings, not predictions for this lesson's data.
- [Prompt tuning](https://arxiv.org/abs/2104.08691) and [Prefix tuning](https://arxiv.org/abs/2101.00190) — optional bridges after learning token embeddings and attention.
