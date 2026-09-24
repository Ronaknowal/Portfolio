# Neural Training Diagnostics & Reproducible Experiments

**Explore as you read.** Edit tiny training rows/step settings, train/eval and graph modes, experimental evidence choices and restored checkpoint fields. Show gradient versus parameter movement, statistic buffers, comparable measured outcomes and the exact next operation under restored/missing state. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose a check that distinguishes a real failure from a null example and preserve the state required for a meaningful replay.


A network has finished training. Its loss fell, its training accuracy reached 100%, and its predictions on new examples are disappointing. You can make it wider, change the learning rate or train longer. But which change would answer the question that matters: **what is preventing useful learning?**

This lesson develops a way to answer that question with evidence. We will train a small classifier on real chemical measurements of wine, deliberately alter one part of its training, and inspect what changes. We will also interrupt training and reconstruct its next update. The goal is to make the next experiment informative before making it expensive.

On a first pass, follow sections 1–4, read the main comparisons in sections 5–7, and finish sections 8–11, including the independent case file. The numerical gradient check and full checkpoint program are marked deeper branches: their examples are useful now, while their implementation detail can wait for a second reading. The complete offline programs and data are supplied with this lesson. Allow roughly 45 minutes for the main reading and additional time for the investigations and practice.

You should be able to follow a forward pass, a mean loss, a gradient and an optimizer update. The preceding [Mini-Batches, Training Loops & Gradient Accumulation](../mini-batches-training-loops-gradient-accumulation/lesson.md) develops that computation. [Bias-Variance Tradeoff & Learning Curves](../bias-variance-tradeoff-learning-curves/lesson.md) explains training versus held-out error. We refresh the particular pieces used here. Both links point to prepared lesson manuscripts awaiting website implementation.

## 1. Start with the question and a known computation

A diagnostic observation is something you measured: “the first-layer gradient is absent,” “the parameters are byte-identical after the attempted update,” or “validation cross-entropy increases while training cross-entropy decreases.” A diagnosis explains the observation. Several explanations may fit the same curve, so write down what would distinguish them.

Use this recurring sequence:

1. **Observation:** identify the data, quantity, units and point in the loop.
2. **Hypotheses:** name plausible explanations that make different predictions.
3. **Minimal experiment:** change or inspect the smallest relevant part while preserving a comparable baseline.
4. **Measurement:** record the result, including a null or failed result.
5. **Inference:** retain, weaken or reject the explanations; choose the next check.

The word *minimal* refers to the question being isolated. A two-line intervention can be a large conceptual change, such as permuting every training label. Conversely, recording an extra tensor may answer a question without changing training at all.

Here is the correct computation we will use as a reference. For a scalar prediction \(\hat y_i=wx_i\), choose the mean half-squared error

\[
L(w)=\frac1n\sum_{i=1}^{n}\frac{(wx_i-y_i)^2}{2},\qquad
g=\frac{\partial L}{\partial w}=\frac1n\sum_i(wx_i-y_i)x_i.
\]

For \(x=[1,2,3]\), \(y=[2,0,1]\), \(w=0\), the per-item gradient contributions are \([-2,0,-3]\). Their mean is \(-5/3\). Plain SGD with learning rate \(\eta=0.1\) therefore produces

\[
w_{\text{after}}=0-0.1(-5/3)=1/6.
\]

The gradient calculation leaves \(w\) at zero. The update changes it. The preceding lesson explains why accumulating microbatches of sizes two and one must preserve the per-item mean; averaging those two batch means equally changes the intended calculation.

**Inline figure — the evidence chain.** Show the three gradient contributions entering their mean, then two separately labeled events: `backward: gradient = −5/3, weight = 0` and `step: weight = 1/6`. A skipped arrow at `step` leaves the same gradient but weight zero. The learner can locate the failed operation without interpreting the shape of a long training curve.

This establishes the lesson's central boundary: a symptom proposes checks; it does not identify a unique cause. Numeric scales depend on the objective, reduction, parameterization, dtype and data units. We will compare values against a defined computation and controlled alternatives, rather than use universal “good loss” or “bad gradient” thresholds.

## 2. Inspect what each row and each split means

The real-data question is whether 13 chemical measurements distinguish wines derived from three cultivars. The [UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine) contains 178 observations from one Italian region; Aeberhard and Forina provide the dataset under CC BY 4.0. This is a cultivar classification exercise, not a prediction of quality or a general claim about future wine production. Our [offline CSV](wine.csv) is the scikit-learn 1.9.1 copy, with an explicit original row ID and target labels changed from 1–3 to 0–2. [Provenance](data-provenance.md) records the transformation and attribution.

One CSV row begins `0,0,14.23,1.71,...`: row ID 0, target class 0, then chemical features. Row ID is an audit key, not a model input. The target is also excluded from the 13 input columns. Before fitting, inspect actual rows together with their targets, shapes, feature ranges and class counts. A shape such as `[72, 13]` tells you how many entries exist; it cannot tell you whether the label beside a row belongs to that row.

Our fixed [split file](split.json) has 72 training rows, 36 validation rows and 70 reserved test rows. Within each class, a seeded permutation selects 24 training rows and 12 validation rows; the rest are reserved. The training and validation sets therefore each have equal class counts. This intentionally small training set makes failure inspection manageable. The split ID lists, rather than just a seed, identify membership unambiguously.

The sets have different jobs:

| Set | Information it supplies | What we do with it here |
| --- | --- | --- |
| Training | Examples the optimizer is allowed to learn from | Fit feature scaling and weights; run small-data diagnostics |
| Validation | Feedback for development decisions | Compare fixed interventions and inspect curves |
| Test | A final evaluation after the development procedure is fixed | Reserve all 70 rows; report no test score in this lesson |

For feature \(j\), compute the training mean \(\mu_j\) and population standard deviation \(s_j\), then transform every split by \((x_j-\mu_j)/s_j\). Validation does not fit a separate scaler. If deployed inputs will arrive from a new producer, time period or instrument, that deployment boundary should determine the split; the supplied random split lacks those grouping and time identifiers.

**Inline figure — information permissions.** Draw row IDs entering three disjoint sets. Only the training set sends an arrow to the fitted mean/scale and optimizer. All sets may receive the fixed transform; validation returns a development-decision arrow, while the test result remains covered. Put “row ID and target excluded” next to the feature extraction, where an accidental leak would occur.

Check the data interface before diagnosing optimization. Do transformations preserve row/target alignment? Do missing values, units and target encodings match the loss? Are augmentations label-preserving for this task? Do the same person, image, recording or source object appear on both sides of a split? Can a training-only feature accidentally reveal the answer? These checks inspect the meaning of the inputs, not merely whether the library accepts them.

For our three-class cross-entropy, a deliberately uniform prediction has loss \(-\log(1/3)=\log 3\approx1.098612\) nats per example. The natural logarithm fixes the unit. This is a reference for uniform logits with hard labels and no extra loss terms; our randomly initialized network need not predict exactly uniformly. A constant prediction of any one class gets 12/36 validation examples correct because this validation set is balanced. These explicit references are more useful than calling an unexplained initial loss “too high.”

## 3. Make a small case learn before interpreting a large run

The small-data check asks a precise question: can the defined model and loop fit a few fixed, inspected training examples under conditions that make fitting feasible?

Select examples with stable row IDs, disable stochastic augmentation and dropout, set explicit loss reduction, and temporarily remove regularization that intentionally resists memorization. Start from a recorded initialization and repeatedly use the same inputs. Include the classes relevant to the check. Inspect predictions as well as the scalar objective.

Our tiny subset contains rows `17, 5, 18, 43, 124, 75, 103, 125, 143, 158, 173, 171`: four examples per class. It uses the training-set transform above, a `13 → 32 → 3` network with tanh between the linear layers, and mean cross-entropy. We fixed SGD learning rate 0.05, momentum 0.9 and 120 updates before running it.

| Measured seed-7 run | Initial training loss | Loss after 120 attempts | Final training accuracy | First gradient norm | First parameter-change norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct update | 1.062082 | 0.000794 | 12/12 | 1.310830 | 0.065541 |
| Deliberately omit `optimizer.step()` | 1.062082 | 1.062082 | 6/12 | 1.310830 | 0 |

Losses are evaluation-mode means on the same 12 examples; norms combine all parameter entries by the Euclidean norm. The correct run has already reached 12/12 at update 20, with loss 0.015172. Accuracy can stop changing while cross-entropy keeps rewarding more probability on the correct class.

The omitted-step run is a complete controlled example of a broken learning process. It has a valid forward computation and nonzero derivatives. The measurements focus the next check on the transition from gradients to parameters: confirm the intended optimizer is called, its groups contain these parameter objects, and its learning rate permits movement. Increasing model size would not repair the omitted call.

A failed small-data check calls for closer inspection, not a declaration that every network must reach zero loss. Identical inputs with contradictory hard targets, a model too restricted for the selected examples, deliberate regularization or a finite update budget can prevent exact fitting. A passed check establishes that this small fitting problem worked. Section 6 will show why its data meaning still needs inspection.

### Investigation: find a check that distinguishes two runs

Use the scalar evidence chain from section 1 and edit its rows or initial weight. Show correct and omitted updates side by side, including both the derivative and parameter change. Find a nonzero-gradient case, then construct a zero-gradient null. The immediate comparison explains why observing a gradient alone cannot establish that an optimizer step ran.

<details><summary>Hint</summary>

A missing update and a correct update at a stationary point can both leave the weight unchanged. Change the inputs so the reference gradient is known to be nonzero.

</details>

<details><summary>Explanation after trying</summary>

With \(x=[1,2]\), \(y=[1,0]\), \(w=0.5\), the mean gradient is \((-0.5+2)/2=0.75\). SGD at 0.1 changes the weight to 0.425. An omitted step leaves it at 0.5, although both runs can calculate gradient 0.75. With \(x=[1,2]\), \(y=[1,2]\), \(w=1\), the gradient is zero, so weight movement fails to distinguish the two. Choose a discriminating fixture before interpreting a null observation.

</details>

## 4. Follow the signals through the network

The loss is the end of a chain. Instrument the earliest point where a comparison begins to disagree: decoded input, transformed features, pre-activations, activations, logits, per-item losses, gradients, then actual parameter changes. A *pre-activation* is the weighted sum before the nonlinear function; an *activation* is the output after it. Logging a small selected batch and layer is usually easier to reason about than collecting every tensor throughout a long run.

At an update boundary, ask different questions of different quantities:

| Evidence | A useful next check |
| --- | --- |
| `parameter.grad is None` after backward | Is this parameter meant to participate? Inspect `requires_grad`, a detached intermediate, the executed branch and whether gradients were just cleared. |
| A gradient tensor exists but is all zero | Inspect the input and derivative along that path; compare a fresh known-nonzero case. A zero tensor is a computed value, unlike an absent buffer. |
| Gradients are finite and nonzero but parameters do not move | Compare parameter identities, optimizer groups, actual step execution, learning rate and update precision. |
| Loss or a gradient first becomes nonfinite | Find the first invalid intermediate and its operands, such as a zero denominator or an overflow; reproduce it on that batch. |
| One layer has a very different scale from its reference | Inspect the layer's role, feature units, initialization, nonlinear derivative and normalization state before changing all layers. |

For a layer with weight vector \(\theta\), record both \(\|g\|_2\) and \(\|\Delta\theta\|_2\). Plain SGD gives \(\Delta\theta=-\eta g\); momentum and adaptive optimizers use additional state. The ratio \(\|\Delta\theta\|_2/\|\theta\|_2\), when the denominator is nonzero, describes change relative to that layer's current scale. Report the absolute change too; a zero or tiny denominator makes the ratio misleading. Compare layers and successive measurements in their context.

Consider a scale mismatch that multiplies already standardized inputs by 100. With the same initial seed-7 weights, the median absolute tanh activation changes from 0.403426 to 1.000000. Since

\[
\frac{d\tanh z}{dz}=1-\tanh^2 z,
\]

values near either endpoint transmit little derivative through the nonlinearity. The measured mean derivative across this first layer falls from 0.759643 to 0.012283. That supports a specific scale hypothesis: this input intervention drives many of these units into saturation. Input scaling, initialization and normalization are places to inspect because they determine the values entering that function.

**Inline figure — values and derivatives together.** Align the tanh curve with two quantile strips of the measured absolute activations: the 10th percentile, median and 90th percentile on a common 0–1 scale. Mark their concentration near the endpoint and show the two mean derivative values beside the same layers. The strips' horizontal axis is the absolute activation value, not update number. They describe the recorded initial forward pass; no learning curve is inferred from it.

For ReLU, inspect which units are zero across the relevant examples and whether changing the examples activates them. A useful sparse representation can also contain many zeros. For normalization, inspect the statistics used in this execution: training-batch statistics or stored statistics, their axes, and whether evaluation data is updating them. An observation about one layer becomes useful when tied to its actual operation.

### Deeper branch: check a derivative independently

Automatic differentiation computes the derivative of the program you executed. If the program implements the wrong objective, an internally correct derivative can still optimize the wrong thing. For a custom loss or suspicious derivative, compare a tiny deterministic smooth calculation with a centered finite difference:

\[
g_{\mathrm{FD}}(w;h)=\frac{L(w+h)-L(w-h)}{2h}.
\]

Take \(x=[1,2]\), \(y=[1,0]\), \(w=0.5\). The residuals are \([-0.5,1]\), the mean half-squared loss is \((0.25+1)/4=0.3125\), and its derivative is 0.75. This complete program checks the calculation:

```python
import torch

torch.set_default_dtype(torch.float64)
x = torch.tensor([1.0, 2.0])
y = torch.tensor([1.0, 0.0])
w = torch.tensor(0.5, requires_grad=True)

def objective(weight):
    return ((weight * x - y).square() / 2).mean()

loss = objective(w)
loss.backward()
print(f"loss={loss.item():.6f}, gradient={w.grad.item():.6f}")
for h in [0.01, 0.0001, 0.000001]:
    estimate = (objective(w.detach() + h) - objective(w.detach() - h)) / (2 * h)
    print(f"h={h:g}, finite_difference={estimate.item():.9f}")
```

```text
loss=0.312500, gradient=0.750000
h=0.01, finite_difference=0.750000000
h=0.0001, finite_difference=0.750000000
h=1e-06, finite_difference=0.750000000
```

The [calculation record](calculation-results.json) retains unrounded estimates. The loss is quadratic, so centered differences are exact in real arithmetic here; the tiny discrepancies come from floating-point evaluation. For a general smooth function, truncation and rounding compete as \(h\) changes. Compare several meaningful step sizes in float64, keep the random draws and model state fixed, and inspect absolute as well as relative differences. At a ReLU kink there need not be a unique derivative for the symmetric difference to match. A small deterministic check is therefore a local diagnostic, with its points and conditions recorded. [CS231n's gradient-check notes](https://cs231n.github.io/neural-networks-3/#gradcheck) give a more detailed treatment of these pitfalls.

## 5. Evaluation behavior has two independent switches

Suppose validation uses `with torch.no_grad():` but the model is still in training mode. Gradients are disabled. Dropout can still draw masks, and BatchNorm can still use the incoming batch and update its running statistics. This makes the evaluation procedure different from the intended one.

`model.eval()` selects module evaluation behavior. `torch.no_grad()` disables graph recording in its scope. Neither replaces the other. Standard validation usually uses both, followed by `model.train()` when training resumes. Evaluation with gradients enabled is useful for some analyses, so these controls are deliberately separate. [PyTorch's autograd notes](https://docs.pytorch.org/docs/2.9/notes/autograd.html#evaluation-mode-nn-module-eval) specify this distinction.

Here is an exact small BatchNorm example. Start a fresh one-feature layer with running mean 0, running variance 1, affine scale 1, affine offset 0 and momentum 0.1. Feed the two values 1 and 3. Their batch mean is 2, their population variance is 1, and their unbiased sample variance is 2. In training mode, forward normalization uses the population variance; the running variance update uses the sample variance:

\[
\mu_{\mathrm{run,new}}=0.9(0)+0.1(2)=0.2,\qquad
s^2_{\mathrm{run,new}}=0.9(1)+0.1(2)=1.1.
\]

These buffer updates occur even inside no-grad. Starting a fresh layer in evaluation mode instead uses the stored mean 0 and variance 1, and preserves them. The distinction between the two variance conventions is part of [PyTorch's BatchNorm contract](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BatchNorm1d.html).

| Fresh layer, same inputs | Output approximately | Stored mean after forward | Autograd graph with default trainable affine parameters |
| --- | --- | ---: | --- |
| `train()`, gradients enabled | −0.999995, 0.999995 | 0.2 | Recorded |
| `train()`, no-grad | −0.999995, 0.999995 | 0.2 | Not recorded |
| `eval()`, gradients enabled | 0.999995, 2.999985 | 0 | Recorded |
| `eval()`, no-grad | 0.999995, 2.999985 | 0 | Not recorded |

The epsilon is \(10^{-5}\). Each row starts from a fresh layer, so the table isolates the switches. A plain linear layer has no module-mode-specific forward rule: switching `train()` to `eval()` gives the same values, providing a useful contrasting case.

**Investigation — two switches, one forward pass.** Edit two input values and the initial stored mean/variance, record whether the forward pass will change the buffer and whether a backward graph will exist, then operate the two switches independently. Inspect the normalization operands alongside the outputs. Use the linear-layer comparison to explain why a mode bug may be invisible in a simpler model.

## 6. Compare losses measured on comparable terms

A training-loop log often reports the loss of each incoming batch *before* its update, possibly with augmentation and dropout. A validation curve often uses the whole held-out set, *after* an update, without augmentation and in evaluation mode. Putting those values on one plot does not make the measurement protocols equal.

For diagnosis, it helps to add a fixed training evaluation set and evaluate it with the same objective, reduction, transforms and module mode as validation. Keep the optimization loss separately when it includes regularization, class weighting, token masks or other terms. Label the horizontal axis: optimizer updates, examples or tokens processed, epochs, or measured time answer different comparison questions. Our curves use completed full-batch optimizer updates. Every measured point evaluates a fixed set in evaluation mode with no-grad and mean cross-entropy.

Now return to the Wine classifier. Before executing, we fixed 400 updates for every full-data run and initialization seeds 3, 7 and 19. One treatment trains on the original 72 targets. The other permutes those targets once with a separately fixed seed, preserving their counts but disrupting correspondence with the chemical measurements. All other settings, features, split membership and validation targets remain the same.

For seed 7, these recorded checkpoints tell the story:

| Completed updates | Clean training loss | Clean validation loss | Shuffled-target training loss | Shuffled-target validation loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.996250 | 0.982391 | 1.151504 | 0.982391 |
| 20 | 0.034856 | 0.054630 | 1.010659 | 1.108418 |
| 120 | 0.002550 | 0.016154 | 0.436862 | 1.285335 |
| 400 | 0.000875 | 0.011259 | 0.029348 | 2.526364 |

At update 400 both models classify all 72 of their *supplied training labels* correctly. The clean model gets 36/36 validation rows correct. The shuffled-target model gets 15/36 correct. On the original labels of its training rows, that shuffled-target model gets only 22/72 correct. The meaning of “training accuracy” therefore depends on which target column you compare against.

This is why looking at the examples and targets is part of training diagnosis. The shuffled run learns the supplied association very effectively. Inspecting row-to-target provenance and restoring the intended labels is the relevant intervention in this controlled case. Adding capacity could help it memorize the wrong associations even more readily.

Inspect one actual row rather than stopping at the aggregate. Training row 5 has original class 0, but the shuffled array supplies class 1. At update 400 the seed-7 shuffled model predicts class 1 with output probabilities `[0.094173, 0.861313, 0.044514]`. The model has learned the label it was given. On validation row 120, whose class remains 1, it instead predicts class 0 with probability 0.999170; the probability assigned to class 1 is only 0.000155. A highly confident wrong answer directs attention to this example and its provenance. The [retained row predictions](wine-results.json) include every final training/validation row so you can inspect other errors without a cherry-picked example defining the conclusion.

**Inline figure — measured loss curves.** Place clean and shuffled-label cases in neighboring panels with common labeled update positions and the initial point present. Plot training and validation separately with a logarithmic loss axis to keep the clean curve's late changes visible; provide linear-scale inspection too. A companion accuracy view shows the memorization/generalization separation. The source is [every measured update](wine-results.json), not a schematic or a fitted trend.

For an unfamiliar run, consider the following as competing questions rather than a lookup table:

| Observation under a specified evaluation protocol | Questions worth distinguishing |
| --- | --- |
| Both training and validation remain poor | Does the small case fit? Is the target meaningful? Are gradients and updates connected? Is the chosen representation sufficient? |
| Training improves while validation worsens | Is the model fitting idiosyncrasies, corrupt labels, a different distribution or a mismatched evaluation procedure? Which inspection/intervention separates these? |
| Validation looks better than training | Are training examples augmented, the training objective penalized, or dropout active only there? Are the sets equally difficult and metrics equally weighted? |
| Both scores appear exceptionally strong | Are the split identities disjoint and the inputs free of target information? Do inspected errors and predictions support the reported score? |

Within this budget, the clean Wine runs show no late validation-loss rise. The shuffled-label treatment supplies the observed contrasting failure. The comparison is useful precisely because the diagnosis follows the recorded evidence: there is no reason to repair an overfitting pattern that the clean run did not exhibit.

### The complete offline program

Download [wine_diagnostics.py](wine_diagnostics.py), [wine.csv](wine.csv) and [split.json](split.json) into the same directory, together with [experiment-protocol.md](experiment-protocol.md), which the program hashes as part of its record. With Python, NumPy and PyTorch installed, run:

```text
python -B wine_diagnostics.py
```

The supplied program has all imports and inputs. `load_inputs` reads the CSV, keeps row identities separate and fits scaling only on training rows. `make_model` initializes the same architecture from the requested seed. `scores` selects evaluation mode and no-grad for each fixed-set measurement. `fit` chooses the training IDs and supplied labels, then records the current scores before performing another update. It stores the gradient norm after backward and the actual parameter-change norm around `optimizer.step()`. `main` executes all eight predeclared runs and writes `wine-results.json` beside the program.

The central update is deliberately ordinary:

```python
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(training_features)
loss = nn.functional.cross_entropy(logits, supplied_targets)
loss.backward()
if treatment != "omitted_update":
    optimizer.step()
```

This excerpt is the fitted computation from the complete file; the surrounding record collection lets us test it. Full-batch updates keep data-order randomness out of this particular comparison. The next experiment introduces order deliberately.

The programs were actually run with Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu, float64, one CPU thread and deterministic algorithms enabled. These are the recorded environment versions, not an instruction to seek a particular newest release. All eight result rows, all intermediate points and the environment are retained. Changing the program produces a new experiment; preserve the original record when comparing it.

## 7. Turn a promising observation into a controlled comparison

Before trying a change, write a short experiment record. For the label experiment it reads:

> **Observation:** a falling training loss can coexist with poor held-out predictions. **Hypothesis:** broken feature–target correspondence allows memorization but removes the intended predictive relationship. **Intervention:** permute only training targets once; preserve class counts, split, features, model, budget and paired initialization seeds. **Measure:** training loss/accuracy against supplied and original targets, and validation loss/accuracy against original targets at fixed updates. **Decision:** if the shuffled run fits its supplied labels but loses validation performance, inspect label provenance before expanding the model.

The prediction is made before looking at the outcome. The result is now available for all planned seeds:

| Initialization seed | Clean validation loss / correct rows | Shuffled validation loss / correct rows |
| ---: | --- | --- |
| 3 | 0.019079 / 36 of 36 | 2.476567 / 11 of 36 |
| 7 | 0.011259 / 36 of 36 | 2.526364 / 15 of 36 |
| 19 | 0.031687 / 36 of 36 | 2.575072 / 14 of 36 |

All runs fit their supplied training labels exactly in accuracy. The shuffled validation accuracy has mean 0.370370 and sample standard deviation 0.057824 across the three initializations. Clean validation accuracy has mean 1 and sample standard deviation 0, while its validation loss still varies. Report the individual observations; a zero spread in coarse accuracy over three runs has a very narrow meaning.

These repeats vary initialization on one fixed split and one fixed target permutation. They do not measure variation from choosing another dataset, split or corruption. Pairing the same initializations is useful for isolating this intervention. In a broader evaluation, vary the sources of variation relevant to the claim, give competing methods comparable selection budgets and keep the final test protocol separate from development. Bouthillier and colleagues' [benchmark-variance study](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf), especially sections 2 and 5, explains why initialization repeats alone miss parts of benchmark uncertainty.

For a new numerical comparison, calculate paired differences \(d_s=m_{B,s}-m_{A,s}\), state whether larger or smaller is better, and inspect their distribution. A claimed improvement should be meaningful for the task as well as larger than plausible variation under the intended evaluation. More repetitions of the same fixed source do not resolve an untested source of uncertainty.

**Investigation — choose what your experiment can answer.** Choose a checkpoint and specific paired seeds from the recorded Wine runs, inspect the sign of the mean validation-loss change and whether training accuracy will distinguish the selected models, then compute those quantities from the selected records. A second input area accepts a small learner-entered table of paired scores with a declared metric direction. The result must identify exactly which runs were included; it must not silently remove an unfavorable row. Test the null by pairing a run with itself.

<details><summary>Changed-score practice and hint</summary>

You have paired validation accuracies for a new task: method A `[0.80, 0.83, 0.79]`, method B `[0.81, 0.80, 0.84]`. Compute each paired difference and its mean before arguing which method to choose. Also name one source of variation the three numbers do not identify.

Hint: the pair, rather than the best B run, is the unit of the comparison.

</details>

<details><summary>Solution</summary>

The differences B minus A are `[0.01, −0.03, 0.05]`, with mean 0.01, or one percentage point. Their sample standard deviation is 0.04. B wins in two pairs and loses in one. An acceptable decision could be to collect more appropriately designed evidence before changing a system whose costs matter; the small observed advantage is inconsistent across these pairs. If the runs use one fixed data split, they leave data-sampling variation unmeasured. The standard deviation describes these paired differences, not uncertainty over every future deployment.

</details>

## 8. A checkpoint preserves a process, not just its weights

Imagine pausing a bicycle on a hill. Its location matters, but restarting with a different velocity changes what happens next. Momentum has a similar hidden-state consequence, which we can compute without an analogy.

Use \(L(w)=(w-3)^2/2\), gradient \(g=w-3\), momentum coefficient \(\mu=0.5\), learning rate \(\eta=0.1\), and the recurrence

\[
v_{t+1}=\mu v_t+g_t,\qquad w_{t+1}=w_t-\eta v_{t+1}.
\]

Start at \(w_0=1,v_0=0\). The first gradient is −2, so \(v_1=-2\) and \(w_1=1.2\). Save there. The next gradient is −1.8. Restoring the velocity gives \(v_2=0.5(-2)-1.8=-2.8\), hence \(w_2=1.48\). Restoring only the weight and resetting velocity gives \(v_2=-1.8\), hence \(w_2=1.38\). Both branches start the resumed step at the same weight and the same loss; their updates diverge.

**Inline figure — first divergence after restart.** Draw two lanes meeting at the saved `w=1.2` box, with the saved velocity `−2` entering one lane and reset velocity `0` entering the other. Align the identical next gradient `−1.8`, different next velocities and resulting weights. The representation should make it clear why comparing the first resumed forward loss can miss this defect.

For a real run, list every state that determines the next computation:

| State category | Why it matters |
| --- | --- |
| Model parameters **and buffers** | Weights and normalization statistics determine the forward computation. |
| Optimizer state and parameter groups | Momentum or adaptive moments, learning rates and parameter membership determine the update. |
| Scheduler state and completed-update count | The same current learning rate can conceal a different future decay time. |
| RNG state of every used generator | The current positions in initialization, dropout, augmentation and sampling streams determine the next draws. |
| Active sample order and cursor | A generator state alone does not identify which entries remain in an already generated permutation. |
| Training/evaluation configuration and preprocessing | Module modes, fitted transforms, target mapping and masks define the computation being resumed. |
| Any unfinished update state | Accumulated gradients, normalization denominator and mixed-precision scaler matter if saving mid-update. |

Save at a documented completed-update boundary when possible: optimizer and schedule have advanced, gradients are cleared, and the next data position is known. That removes unfinished accumulation from the example. Multiworker prefetch, distributed samplers and external streaming sources need their own resume contract; an epoch number alone cannot reconstruct their in-flight state.

`state_dict()` is useful for model and optimizer state. When retaining an in-memory “best model” while training continues, deep-copy the state or serialize it at that moment; a live reference to state is not an immutable checkpoint. [PyTorch's saving/loading tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training) distinguishes a general training checkpoint from saved model weights.

### Deeper branch: an actual uninterrupted-versus-resumed comparison

The complete [checkpoint_replay.py](checkpoint_replay.py) uses the same training data and imports the supplied `load_inputs` helper from `wine_diagnostics.py`. It changes the architecture to `13 → 8 → tanh → dropout(0.25) → 3`, uses batches of 12, and saves after completed update 5. The active permutation has 72 positions and its cursor is 60, so one batch remains before a new permutation is drawn. SGD momentum is 0.9; the schedule halves the learning rate after every three updates.

Run it from the directory containing the supplied files:

```text
python -B checkpoint_replay.py
```

`fresh_training_state` creates the model, optimizer, scheduler and separate sample-order generator. `train_until` consumes batch positions, records the learning rate actually used, and performs `step → scheduler.step → zero_grad`. `snapshot` deep-copies all process state. The program serializes that tensor/plain-container checkpoint through an in-memory buffer, loads it with `weights_only=True`, and compares continuations through update 12. It restores the global Torch RNG after constructing the new model because construction itself consumes random draws. The scheduler is created before loading optimizer state, consistent with the [optimizer loading API](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.load_state_dict.html).

The reference continuation uses learning rate 0.015 at update 6 and 0.0075 at update 7. Here is the actual result of changing only the named state category:

| Restore treatment | Exact continuation trace and final parameters? | Largest absolute final parameter difference |
| --- | --- | ---: |
| Restore all recorded state | Yes | 0 |
| Clear optimizer momentum buffers; retain its parameter groups and current learning rate | No | 0.030061553 |
| Leave global Torch RNG at its reconstructed initial position | No | 0.008614809 |
| Regenerate order and reset cursor instead of restoring the active order/cursor | No | 0.008778305 |
| Leave scheduler at its new initial state | No | 0.023764836 |

The last column is a diagnostic distance under this run's parameterization, not a ranking of practical harm. The [full record](checkpoint-results.json) includes batch positions, pre-update loss and learning rate at every continued step.

Where did the disagreement start? Clearing momentum leaves the update-6 batch, forward loss and learning rate equal; the update changes, so the update-7 forward loss differs. Omitting Torch RNG changes update-6 dropout and therefore its forward loss. Losing the active order/cursor changes the next batch immediately. Losing the scheduler leaves update 6 intact but uses 0.015 at update 7 where the reference uses 0.0075. Inspecting the first different quantity identifies a much narrower search than “the final models differ.”

**Investigation — rebuild the continuation.** On the scalar recurrence, edit the target, initial weight, momentum and save point, then choose which state to restore. Record the predicted next weight and earliest divergence stage before continuing. Compare full restore with reset momentum and the zero-momentum null. The recorded Wine trace is an additional inspection view for RNG/order/scheduler effects; editing the scalar recurrence does not generate new Wine measurements.

### Which machinery the diagnosis opens—and which it reuses

The diagnostic implementation is the measurement-and-intervention procedure, not a replacement neural-network library. [wine_diagnostics.py](wine_diagnostics.py) owns train-only scaling, declared treatments and data roles, gradient norm, before/after parameter differences and actual per-row outputs. `fit` records the model before and after the intended update; it cannot mistake “a gradient exists” for “the parameters moved.” [calculations.py](calculations.py) supplies exact scalar arithmetic and an independent finite-difference route, providing a known reference when a larger run fails.

The reusable model/derivative mechanisms already have concrete owners: the implemented [Backpropagation engine](/learn-assets/backpropagation/teaching-autodiff.py), [loss implementations](/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/loss-mechanisms.py), and [optimizer recurrence/library comparisons](/learn-assets/gradient-variants/optimizer_library_bridge.py). The preceding [prepared training-loop source](../mini-batches-training-loops-gradient-accumulation/train_iris.py) owns effective-group accumulation; its content is ready, while its updated website implementation may still be pending. The present full-batch diagnostic program remains runnable independently of that publication status.

For continuation, [checkpoint_replay.py](checkpoint_replay.py) owns the process state: `snapshot` records model, optimizer, scheduler, CPU/dropout RNG, ordering RNG, the current permutation/cursor and update count. `restore` reconstructs the same objects and loads their ordinary `state_dict` APIs. The in-memory `torch.save`/`torch.load(weights_only=True)` round trip tests real serialization semantics; saved arrays are copied rather than aliased to parameters that training will subsequently mutate. Individual omission cases hold the other fields fixed, so their divergence has a meaningful cause.

**Extend the checkpoint boundary.** The current program checkpoints after a completed update. Move the checkpoint between two accumulation chunks and specify the additional state needed before implementing that continuation.

<details><summary>Hint and reasoned solution</summary>

At a completed update, gradients have been cleared and the next group can start from model/optimizer/data state. Mid-group, preserve accumulated gradient tensors, the effective group's full denominator, which rows/chunks have already contributed, the remaining order/cursor and any stochastic or AMP scaling state. Restoring weights alone and recomputing only the suffix loses the earlier derivative contributions; recomputing the whole group while retaining old gradients counts them twice. An acceptable implementation either restores this exact mid-group state or deliberately checkpoints only at completed-update boundaries and states that contract. Validate the first resumed derivative and next update, not just final rounded loss. Data-worker prefetch and distributed ownership need additional state beyond this single-process example.

</details>

## 9. Repeating a run and reproducing a conclusion

A seed selects the start of a pseudo-random stream. Saving the generator's state selects its current location. Setting the original seed during a mid-run restart normally returns to the beginning, which is different from resuming the next random draw.

Use separate named generators when independent processes should have independently controlled streams, such as the sample permutation and label corruption here. Record generator states actually used by Python, NumPy and Torch, including relevant device generators. NumPy's `default_rng` objects have their own states; setting the older global NumPy seed does not reset an existing generator object. Dataset code and worker libraries may consume other random streams.

Exact replay also depends on execution. Floating-point addition is not associative; changing reduction order, kernel selection or device can change results even with the same random draws. PyTorch provides deterministic-algorithm controls and documents worker seeding, CUDA benchmarking and device-specific requirements. Those controls can trade speed for repeatability and can raise an error when a requested deterministic implementation is unavailable. [PyTorch's reproducibility note](https://docs.pytorch.org/docs/2.9/notes/randomness.html) explicitly limits guarantees across releases, platforms and CPU/GPU execution.

Distinguish three useful claims:

| Claim | Evidence needed |
| --- | --- |
| “This exact continuation replays in this environment.” | Compare the resumed state and trace to uninterrupted execution under a recorded contract. |
| “This procedure gives similar results across relevant random choices.” | Repeat those choices, retain all planned runs and report variability. |
| “Another person can reproduce the experimental conclusion.” | Supply code, data/provenance, configuration, selection procedure, measurements and enough environmental detail to test the claim. |

The exact CPU replay in section 8 supports the first claim for that program. The Wine initialization repeats support a restricted version of the second. Sharing the packet makes the procedure inspectable and repeatable; evaluating it on a new platform is a new check.

A useful experiment record answers: What was predicted? Which source revision ran? Which input hashes, split IDs and fitted transformations were used? Which model/loss/reduction, optimizer, schedule, batch/update budget and seed streams were fixed? What hardware, precision, versions and determinism settings were used? What quantity was measured, at what checkpoint, and by what selection rule? Where are failed attempts and the actual results? Which inference followed?

Do not overwrite that history with only the best validation run. If you change the budget after seeing a result, record the new decision and keep the old result. Test performance becomes a development signal if repeatedly consulted to choose settings; preserve a final evaluation procedure appropriate to the claim.

## 10. Independent case file

Try the cases before opening the hints or solutions. A strong response cites an observation, competing explanations, a minimal discriminating check, its predicted outcomes and a bounded conclusion.

### Case A: the silent training run

A new scalar regression diagnostic uses \(x=[2,3]\), \(y=[1,2]\), initial \(w=0.25\), mean half-squared error and SGD at 0.2. After backward the gradient is −2.375. After the reported update the weight is still 0.25. Compute the reference weight and give two plausible explanations plus a check that separates them.

<details><summary>Hint</summary>

The derivative has already been independently checked. Inspect the optimizer's actual operation and the parameter object it owns.

</details>

<details><summary>Solution</summary>

The reference gradient is \(((-0.5)(2)+(-1.25)(3))/2=-2.375\), and the next weight is \(0.25-0.2(-2.375)=0.725\). One explanation is that `step()` was never called. Another is that it was called on an optimizer containing an old parameter object after the model was replaced. Capture the executed step and compare the identities of model parameters with optimizer-group entries; record before/after values for both. A zero learning rate is another viable hypothesis whose direct inspection is inexpensive. Nonzero gradients rule out neither explanation. The repaired run must reproduce 0.725 for this changed contract.

</details>

### Case B: a convincing training curve

A three-class classifier fits all its training labels. Validation accuracy is about one third. Your colleague concludes the learning rate must be too small and wants a wider model with a larger learning rate. Design a more informative first investigation. Name an outcome that would favor a data problem and an outcome that would favor further optimization investigation.

<details><summary>Hint</summary>

Training accuracy is measured against a supplied target array. Reconstruct a few row-to-target paths from original input to loss.

</details>

<details><summary>Solution</summary>

Inspect stable row IDs through preprocessing, batching and target construction, compare them with original labels, and verify evaluation uses the same target mapping. Refit a small inspected subset under the correct loop. Discovering a permutation or inconsistent class mapping and improving the controlled run after repairing only that mapping favors a data-interface cause. If alignment and evaluation are correct but a feasible deterministic subset still fails to fit, inspect derivatives and actual updates before widening the model. A valid answer may choose a different cheap test if it states how the outcomes separate plausible hypotheses. The original accuracy observation alone does not choose the learning rate.

</details>

### Case C: validation changes the future

A BatchNorm model runs validation between training updates inside no-grad. Its parameters remain unchanged, but its running mean changes. Explain the mechanism and repair the procedure. Then say why running the same test on a linear-only model could conceal the defect.

<details><summary>Hint</summary>

The state dict contains more than trainable parameters. Identify which switch controls the forward behavior.

</details>

<details><summary>Solution</summary>

The model is still in training mode, so BatchNorm consumes validation-batch statistics and updates buffers during forward. Select evaluation mode before validation, use no-grad for graph control, and select training mode before the next training batch. Verify both buffer preservation and evaluation outputs on a saved copy of the model. A linear-only model has no BatchNorm buffer updates or dropout behavior, so its unchanged values cannot test this mode-sensitive mechanism.

</details>

### Case D: a changed restart calculation

For \(L(w)=(w-4)^2/2\), start at \(w_0=2,v_0=0\), use \(\eta=0.25\) and \(\mu=0.75\). Save after the first update. Calculate the second weight with full state and with reset velocity. Then construct a momentum setting where resetting velocity has no effect.

<details><summary>Hint</summary>

Write the saved pair \((w_1,v_1)\) before continuing. The loss gradient is \(w-4\).

</details>

<details><summary>Solution</summary>

The first gradient is −2, so \(v_1=-2\) and \(w_1=2.5\). Next \(g_1=-1.5\). Full restore gives \(v_2=0.75(-2)-1.5=-3\), so \(w_2=3.25\). Reset velocity gives \(v_2=-1.5\), so \(w_2=2.875\). With momentum zero, the previous velocity is multiplied by zero, and both continuations agree. That null case tests why the state matters, rather than merely memorizing that all saved values always matter.

</details>

### Practical synthesis: write a reproducible diagnosis

Use the supplied offline Wine files. Select one inspected training row and trace its features and class through the split, transform and loss input. Reproduce the seed-7 clean and shuffled-label results. Then propose and run one new diagnostic intervention of your own, such as a different declared target permutation or a changed feasible tiny subset. State your prediction before execution and keep the original settings and outputs. Use training/validation only.

Your report should make another reader able to repeat the intervention: exact input IDs, changed field, held-fixed settings, source/configuration record, expected contrast, observed measurements and interpretation. It should identify an alternative explanation left unresolved. Success is a defensible experiment, including a well-explained null result, rather than beating the supplied validation score. The initial reproduction targets are the section-6 seed-7 values to the displayed precision in the recorded environment; new interventions have no invented answer key.

<details><summary>Example acceptable report structure</summary>

“I changed the fixed 12-example subset and retained the transform fitted on the same 72 training rows, initialization seed 7 and 120-update budget. I predicted the inspected subset would fit because it has no duplicated contradictory examples and the same architecture already fitted the original subset. Here are the exact selected row IDs, the initial/final losses and predictions, and the unchanged reference record. The outcome supports or weakens that local fitting hypothesis. It leaves performance on other distributions unmeasured.” Supply actual numbers from your run in the report; this example supplies a reasoning structure, not a fictional result.

</details>

## 11. Readiness and next study

You have reached the end of **Deep Learning Fundamentals & Architectures**. The module has introduced many architectures; this final lesson supplies a way to decide whether your computation and evidence justify using any of them.

Before marking your own study complete, retrieve these ideas without looking back: why can a nonzero gradient coexist with unchanged weights? Why can fitting shuffled labels succeed? What must be equal when comparing training and validation loss? Which BatchNorm state can change in no-grad? Why can a resumed forward loss match even though the next update will differ? What does repeating one seed or several initializations actually establish?

You are ready to continue when you can diagnose a changed case with a discriminating experiment, reproduce a small forward/update calculation, explain the train/validation/test roles, identify the first diverging checkpoint state and write a record that someone else can run. If one skill is weak, revisit the relevant worked case and solve its changed practice before trying a larger model.

Choose your next study by the work you want to do: revisit the preceding training-mechanics lesson for batching and accumulation; revisit normalization or initialization in this module if layer statistics are difficult; revisit the prepared [end-to-end supervised learning and error-analysis lesson](../end-to-end-supervised-learning-error-analysis/lesson.md) for a broader project decision process; or run a small reproduction of an architecture from this module with your own explicit hypothesis. These review and extension choices let your next question guide further study.

## References and another way to learn it

- [Goodfellow, Bengio and Courville, *Deep Learning*, chapter 11: Practical Methodology](https://www.deeplearningbook.org/contents/guidelines.html). Free textbook chapter connecting metrics, baselines, data decisions and debugging. Best for consolidating the whole process after the first pass; this packet reviewed its actual agenda and debugging discussion. Its architecture-era examples supplement the current runnable API examples here.
- [Stanford CS231n notes: Learning](https://cs231n.github.io/neural-networks-3/). A visual, practical article on gradient checks, small-data checks and monitoring. The gradient and monitoring sections were read. Treat its numeric rules of thumb as historical heuristics to investigate in context; this lesson uses defined computations and controlled evidence.
- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html). An official beginner tutorial route through data, networks, differentiation, optimization and saving/loading. Use it alongside the supplied programs if the API is unfamiliar. The introduction and linked optimization/saving material were reviewed; the tutorial's external dataset exercises were not executed for this packet.
- [PyTorch: Autograd mechanics](https://docs.pytorch.org/docs/2.9/notes/autograd.html), [Reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html), and [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html). Precise API references for graph control, random streams and checkpoints. Relevant sections were read; the first two links intentionally identify the versioned documentation inspected, while the actual calculations ran on the environment recorded above.
- [Stanford CS231n 2017 course syllabus](https://cs231n.stanford.edu/2017/syllabus), especially its linked Training Neural Networks lectures. A video/course alternative for revisiting the broader optimization and regularization mechanisms. The syllabus and official Lecture 7 description were checked; the video was not watched and no timestamp or technical-accuracy claim is based on metadata. The written notes above are the directly reviewed companion.
- [Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf). An advanced research branch on what repeated benchmarking actually measures. The introduction, variance model and recommendations were inspected; reproducing the paper's benchmark suite is outside this lesson.

Research and local experiments were checked on 13 September 2026. The supplied data, exact protocol, full programs and measured records make the examples usable without a network connection. The interactive figures and investigations described here remain specified for a later website implementation.
