# Dropout, DropPath & Stochastic Depth

**Explore as you read.** Edit features/weights, probability, survivor scale, mask grouping, branch position, per-block rates and train/eval mode; inspect retained Monte Carlo prefixes. Update weighted outcome means/variances, gradient routes, call counts, state buffers and saved prediction distributions immediately. Keep the sampled mask fixed while comparing a parameter, with resampling a separate action. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose masking scope and evaluation behavior from their actual effects; distinguish expected active depth from work that was really skipped.


## Learn with some information temporarily missing

Imagine recognizing a handwritten 8. A model might use its upper loop, lower loop, central narrowing and stroke locations. If training makes one combination indispensable, the model may struggle when a new handwriting style changes part of that combination. One possible training intervention is to randomly hide some intermediate values and still ask for the correct digit.

**Dropout** does this temporary hiding. The values return on another pass; parameters are not permanently deleted. **DropPath**, commonly used for a form of **stochastic depth**, hides an entire learned correction in a residual block. Both modify the training problem. Their usefulness must be checked on examples excluded from fitting.

The [previous lesson on residual connections](/learn/path/full-curriculum/residual-connections-skip-connections?module=deep-learning-fundamentals) showed that a direct path can preserve a representation while another path changes it. Here we ask what happens when the correction is sometimes absent.

**First pass:** follow the two-value example, mask geometry, residual branch calculation, train/evaluation mode distinction and real digit comparison; then try practice 1–5. Monte Carlo uncertainty and specialized noise families are optional deeper branches. You need multiplication, averages, a loss and its gradient; these are refreshed where used.

## 1. A mask changes values, then changes an update

Suppose a hidden representation is \(h=[1,2]\). A scalar output uses weights \(w=[1,-0.5]\):

\[
\hat y=w^\top h=1(1)-0.5(2)=0.
\]

Let the target be 1. We use half-squared error \(L=\tfrac12(\hat y-1)^2\), so the unmasked loss is 0.5.

Set the **drop probability** to \(p=0.5\). The keep probability is \(q=1-p=0.5\). Independently for each value, sample a bit: 1 means keep, 0 means hide. Such a bit is a **Bernoulli random variable**. Suppose the sampled mask is \(m=[1,0]\).

Modern inverted dropout multiplies by the mask and divides surviving values by the keep probability:

\[
\widetilde h=\frac{m\odot h}{q}=[2,0],\qquad
\hat y=w^\top\widetilde h=2.
\]

The output moved from 0 to 2; it did not become the target. The sampled loss is again 0.5, now with error in the opposite direction.

Trace the gradient through these actual values:

\[
\frac{\partial L}{\partial w}
=(\hat y-1)\widetilde h=[2,0],
\qquad
\frac{\partial L}{\partial h}
=(\hat y-1)w\odot m/q=[2,0].
\]

A gradient is the local sensitivity of loss to a small change. A gradient-descent step of size 0.1 gives \(w_{\mathrm{new}}=[0.8,-0.5]\). With this same mask, the new output is 1.6 and loss is 0.18. The second weight receives no contribution from this example through the dropped coordinate.

That is not a promise that its optimizer value never changes: other examples, other paths, momentum or weight decay can still contribute. A fresh mask belongs to the next forward pass. Backpropagation must use the mask from the forward computation it differentiates.

**Try a different mask:** keep the original weights and change the mask to $[0,1]$. Follow the changed gradient route: the masked representation becomes $[0,4]$, output −2, error −3 and weight gradient $[0,-12]$. The two masks train different dependencies of the same model.

## 2. Why divide by the keep probability?

For a fixed value \(h_i\),

\[
\mathbb E[\widetilde h_i\mid h_i]
=q(h_i/q)+p(0)=h_i.
\]

The expectation is an average over repeated masks, not a statement about every pass. Here are **all four outcomes** for \(h=[1,2]\), \(p=0.5\):

| Mask | Probability | Masked representation |
| --- | --- | --- |
| [0,0] | 0.25 | [0,0] |
| [0,1] | 0.25 | [0,4] |
| [1,0] | 0.25 | [2,0] |
| [1,1] | 0.25 | [2,4] |

Their weighted mean is \([1,2]\). Their coordinate variances are \([1,4]\). Generally,

\[
\operatorname{Var}(\widetilde h_i\mid h_i)=\frac{p}{1-p}h_i^2.
\]

For \(p=0.25\), the four probabilities are \(0.0625,0.1875,0.1875,0.5625\), in the same row order. They are not uniform. The mean stays \([1,2]\), while variances become \([1/3,4/3]\). Increasing \(p\) changes both how often information disappears and the amplitude of surviving values.

At ordinary evaluation, inverted dropout returns \(h\) directly. It applies neither a random mask nor an extra keep-probability multiplier. Historical implementations instead left training survivors unscaled and multiplied by \(q\) at evaluation. Both conventions need internally consistent initialization/training scale; mixing their evaluation rules is an error. See the explicit current [PyTorch Dropout contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html).

At \(p=0\), training also becomes identity. At \(p=1\), division by zero is invalid; the supplied implementation explicitly returns zeros during training and identity during evaluation. The expectation-preservation formula applies to \(p<1\).

### Preserved means do not imply an unchanged network

Use signed contributions \([1,-1]\), independent masks and \(p=0.5\), then apply ReLU to their sum. Unmasked, the result is \(\max(0,1-1)=0\). Across the four equally likely masks, the results are \(0,0,2,0\), whose mean is 0.5:

\[
\mathbb E[\operatorname{ReLU}(Z)]\ne
\operatorname{ReLU}(\mathbb E[Z]).
\]

Even when a linear output preserves its mean, its expected loss can change. “An ensemble of thinned networks” is a useful interpretation of shared parameters under different masks, not a claim of independently trained models or exact arithmetic averaging by one deterministic nonlinear pass.

The training objective is

\[
\min_\theta\frac1N\sum_{i=1}^N
\mathbb E_m[\ell(f_\theta(x_i;m),y_i)].
\]

Each sampled update estimates this noisy objective. The goal is to learn useful predictions under that perturbation, not to make each hidden unit a complete classifier. Overfitting means fitting sample-specific patterns that generalize poorly; it does not require a gap that widens forever, and parameter count alone does not diagnose it.

## 3. What gets hidden? Geometry matters

A tensor is an array with named axes. For an image representation \([B,C,H,W]\), \(B\) indexes examples, \(C\) feature channels, and \(H,W\) spatial positions. A channel might respond to a learned pattern over the image. Convolution will explain how those maps are built in the next lesson.

| Operation | Independent mask shape | What disappears together |
| --- | --- | --- |
| Element dropout | [B,C,H,W] | One activation value |
| Channel dropout | [B,C,1,1] | A whole feature map for one example |
| Per-example branch dropout | [B,1,1,1] | The whole correction for one example |
| Batchwise branch dropout | [1,1,1,1] | The correction for every example in that batch |

Dimensions of size 1 are **broadcast**: the same bit is repeated along that axis. This small shape choice defines the intervention. On a sequence shaped \([B,T,D]\), a branch mask \([B,1,1]\) is shared across tokens and features for an example. An element mask \([B,T,D]\) makes separate decisions. Neither shape can be inferred from the word “dropout” alone.

Take two examples with two \(2\times2\) channels each, filled with values 1–16 in order. Under channel mask \([1,0]\) for example 1 and \([0,1]\) for example 2, with \(p=0.5\), the surviving maps contain \([[2,4],[6,8]]\) and \([[26,28],[30,32]]\). The other two maps are entirely zero. Under a branch mask \([1,0]\), both maps of example 1 survive and both of example 2 disappear.

**Build the intervention:** make an entire second channel disappear for example 1 while preserving its first channel and all channels of example 2. Choose the mask axes and enter its bits. Then check that no spatial position inside a channel contradicts another.

The difference is more than appearance. For fixed values \([1,2]\) and \(p=0.5\), independent masks give covariance 0; one shared mask gives covariance 2. Shared masking makes values move together. Spatial neighbors can carry redundant evidence, so removing isolated values may leave that evidence nearby. Channel or contiguous-region masking can challenge a different dependency. This motivates comparison, not a rule that element dropout after convolution is always useless. [PyTorch Dropout2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout2d.html) explicitly defines the channel operation; use a four-dimensional batched input here because its three-dimensional interpretation has a version-specific warning.

## 4. Drop the correction while keeping the direct path

A residual block computes \(y=x+F(x)\). With inverted branch dropout,

\[
y=x+\frac{m}{q}F(x).
\]

Let \(x=[2,-1]\), \(F(x)=[0.5,1]\), \(q=0.5\).

| Branch bit | Block output |
| --- | --- |
| 0 | [2,−1] |
| 1 | [3,1] |
| Average | [2.5,0] |

The average equals the unmasked block output for this fixed input. If instead you mask the **whole sum**, a dropped pass gives \([0,0]\), removing the direct path too. It is a different architecture.

For a sampled branch bit, the local derivative is \(I+(m/q)J_F\), where \(J_F\) describes how the correction changes with input. A dropped correction leaves \(I\). A surviving correction can still cancel, shrink or amplify the total derivative, as the preceding residual lesson demonstrated. The preserved path is useful, not an unconditional gradient guarantee.

Terminology varies. Modern libraries commonly call per-example residual-branch masking **DropPath**, and also call it stochastic depth. Torchvision's [stochastic-depth implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html) supports both row and batch modes. The [timm implementation](https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/layers/drop.py) uses one bit per example and has a keep-scaling option. Read the mask and scaling contract instead of assuming two names imply two incompatible algorithms.

### Expected active branches are not measured runtime

For \(L\) residual blocks with drop probabilities \(p_l\), the expected active count is

\[
\mathbb E[A]=\sum_{l=1}^L(1-p_l).
\]

The original stochastic-depth schedule corresponds, in our drop-probability notation, to \(p_l=p_{\max}l/L\), for \(l=1,\ldots,L\). It gives

\[
\mathbb E[A]=L-p_{\max}(L+1)/2.
\]

Four blocks with \(p_{\max}=0.5\) have rates \([0.125,0.25,0.375,0.5]\), giving 2.75 expected active blocks. A zero-first schedule \([0,1/6,1/3,1/2]\) gives 3. These are different conventions, both explicit. For a one-block zero-first schedule, our code uses \([0]\) rather than dividing by \(L-1=0\).

Also count **blocks**, not every layer within them. Huang et al.'s 110-layer example has 54 residual blocks. Their original convention used unscaled surviving branches in training and survival-scaled branches in evaluation. Their speed results involved actually bypassing computation. [Deep Networks with Stochastic Depth, §3](https://arxiv.org/pdf/1603.09382).

In the expression `mask_values(F(x), ...)`, Python has already evaluated `F(x)`. Multiplication by zero cannot undo that work. Batchwise conditional execution can avoid a branch if the decision comes first; per-example skipping may need gathering, scattering and different batch-statistic treatment. Unequal block costs, random generation, memory traffic and hardware scheduling also matter. Expected active depth is a structural quantity, not a speedup benchmark.

## 5. Modes, state and a normalization trap

In PyTorch, `model.train()` enables modules' training behavior; `model.eval()` selects evaluation behavior. `torch.no_grad()` controls recording gradients. It does **not** turn dropout off or stop BatchNorm running-statistic updates.

Ordinary validation uses both evaluation behavior and no gradient recording. The complete experiment calls these explicitly before measuring either training or validation rows. Measuring training data with dropout enabled and validation data with it disabled mixes two forward procedures and can create a misleading “generalization gap.”

BatchNorm stores running means and variances for evaluation. Suppose an activation \(X\) is equally likely to be 1 or 3. Its mean is 2 and variance is 1. With independent inverted dropout at \(q=0.5\),

\[
\mathbb E[\widetilde X^2]=\mathbb E[X^2]/q=5/0.5=10,
\quad \operatorname{Var}(\widetilde X)=10-2^2=6.
\]

The mean was preserved; the variance was not. A BatchNorm downstream can learn statistics of this noisier distribution, then see the clean distribution at evaluation. That is the variance-shift mechanism studied by [Li et al.](https://arxiv.org/abs/1801.05134).

The exact fixture sends \([0,2,0,6]\) through BatchNorm with momentum 1. Its training variance uses divisor 4, giving 6; the stored unbiased running variance uses divisor 3, giving 8. Clean evaluation inputs \([1,3]\) are therefore mapped to approximately \([-0.353553,0.353553]\). Do not confuse population variance 6 with the stored finite-batch estimate 8.

Placing masking after a particular BatchNorm avoids directly masking that layer's input, but later normalization layers may still see altered distributions. LayerNorm and GroupNorm do not have the same running-statistic mismatch, yet they are not immune to masking. LayerNorm of \([1,3]\) is approximately \([-1,1]\); after the mask produces \([2,0]\), it is approximately \([1,-1]\). The representation reversed.

For MC dropout, put the model in evaluation mode first, then selectively enable its dropout modules. Keep BatchNorm in evaluation mode. Our state probe verifies that `no_grad()` in training still increments a BatchNorm counter, while selective dropout activation does not. For functional calls, pass `training=self.training` during ordinary operation; a hardcoded `True` deliberately ignores `eval()`.

## Use the mask contract in a library without changing its meaning

Read `mask_values` in [the complete program](dropout-experiments.py) before the model. It is the scratch implementation: choose the broadcast shape, draw Bernoulli bits once, multiply, divide by keep probability, and bypass sampling in evaluation. Its array work is O(number of activation values); the random mask contains only as many independent entries as its chosen shape. `fixtures` keeps masks fixed for forward/backward arithmetic, while `DigitModel` shows ordinary `nn.Dropout` training. A stochastic sample is not an implementation-equivalence test just because two final losses look close.

The usual interfaces for the four scopes are:

```python
import torch
from torch import nn
from torchvision.ops import stochastic_depth

torch.manual_seed(9)
features = torch.arange(1., 17.).reshape(2, 2, 2, 2)
element = nn.Dropout(p=0.25)
channel = nn.Dropout2d(p=0.25)
print(element(features).shape, channel(features).shape)
for mode in ("row", "batch"):
    branch = stochastic_depth(features, p=0.25, mode=mode, training=True)
    print(mode, branch)
    torch.testing.assert_close(
        stochastic_depth(features, p=0.25, mode=mode, training=False), features)
element.eval()
channel.eval()
torch.testing.assert_close(element(features), features)
torch.testing.assert_close(channel(features), features)
```

This standalone code needs compatible PyTorch/Torchvision versions. It requests the same mask geometry as the scratch implementation, but does not claim the random masks are identical. In `stochastic_depth`, row means one bit per batch member, even when each member contains many tokens or pixels; batch means one bit for the entire supplied tensor. Both mask the supplied **correction**, so the caller still adds the untouched residual input. The [maintained implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html) makes that convention visible. Record the installed version when executing this newly prepared example.

**Independent modification:** add a `locked_features` case for a sequence `[B,T,D]`, with independent bits shaped `[B,1,D]`. Let the caller supply a fixed mask for a deterministic comparison. Return identity in eval, zeros at p1 in training, and `values * mask / (1-p)` otherwise. Then differentiate the sum of the output.

<details><summary>Hint</summary>The forward bit belongs to a feature/example pair; every time step must use the same bit, including backward.</details>

<details><summary>Solution and success criteria</summary>For one example with time rows [1,2] and [3,4], p0.5 and mask [1,0], the result is [2,0] and [6,0]. The gradient of their total with respect to the input is [2,0] on both rows. A fresh backward mask or a `[B,T,D]` draw changes the contract. Test identity evaluation, p0 and p1 separately, and only then use random masks during training. A mask factory is a meaningful customization point; the loss, tensor gradients and optimizer can remain ordinary library operations.</details>

## 6. A complete experiment: does masking help these digits?

Download [dropout-experiments.py](dropout-experiments.py), [digits-400.csv](digits-400.csv) and the [data provenance](data-provenance.md) into one directory. The program uses Python, PyTorch, NumPy and scikit-learn; run:

```sh
python -m pip install torch numpy scikit-learn
python dropout-experiments.py
```

The recorded run used Python 3.12.14, PyTorch 2.14.0 CPU, NumPy 2.3.5 and scikit-learn 1.9.1. The dataset contains 400 real \(8\times8\) UCI digit images, not MNIST: 40 per class. A fixed stratified split uses 280 training and 120 validation examples, seed 22. Pixel values are divided by their known maximum 16. No fitted preprocessing uses validation data, and this small reused teaching split is not an official benchmark or final test.

The program contains two controlled comparisons:

- An MLP: \(64\to64\to64\to10\), tanh hidden activations, element dropout after each hidden activation, \(p\in\{0,0.2,0.5,0.8\}\).
- A residual MLP: a \(64\to64\) tanh stem, four corrections \(F_l(h)=0.5\tanh(W_lh+b_l)\), and a \(64\to10\) head. Compare no branch masking with row or batch masking using zero-first schedules ending at 0.2 or 0.5.

Each family uses the same initial learned parameters for its masking variants at a given seed. The two families have different parameter counts, 8,970 and 21,450, so comparisons between them are not a matched architecture ablation. Every configuration uses Adam at 0.003 for 400 full-batch updates, with three initialization/mask seeds. No augmentation, weight decay, normalization, early stopping or hidden pretrained dependency is included.

The mask producer is implemented explicitly for element, channel, row and batch shapes. The model's forward method makes placement visible. Training minimizes cross-entropy of raw logits; reported losses are deterministic evaluation-mode cross-entropy in natural-log units per example. Saved points at steps 0, 1, 25, 100, 200 and 400 are actual measurements, available in [calculated-inputs.json](calculated-inputs.json).

**Executed final validation results, seed 1:**

| Family | Mask configuration | CE | Correct / 120 |
| --- | --- | --- | --- |
| MLP | none | 0.088034 | 118 |
| MLP | element 0.2 | 0.090138 | 118 |
| MLP | element 0.5 | 0.111671 | 116 |
| MLP | element 0.8 | 0.144736 | 116 |
| Residual | none | 0.136714 | 117 |
| Residual | row, endpoint 0.2 | 0.150095 | 117 |
| Residual | batch, endpoint 0.2 | 0.146876 | 117 |
| Residual | row, endpoint 0.5 | 0.157595 | 117 |
| Residual | batch, endpoint 0.5 | 0.167144 | 116 |

All but the element-0.8 configuration classify all 280 training images correctly; that configuration gets 279. Even high dropout did not force chance-level training accuracy here.

Across seeds, the MLP's no-dropout validation correct count is 117–118, compared with 118 for all three element-0.2 runs. Seed 2's loss improves from 0.071144 to 0.067003 with 0.2, while seeds 1 and 3 slightly worsen. All recorded residual masking variants have worse final validation CE than their corresponding unmasked residual baseline. These observations support a narrow conclusion: masking is not clearly needed for this setup. They do not establish that a different dataset, architecture, schedule or training budget cannot benefit.

**Investigate:** compare the two saved runs with their validation-loss traces visible. Also compare correct counts and training loss. Explain why a smaller training–validation gap alone does not decide the winner. If you change a rate or budget in the program, keep the baseline, retain the new outputs and identify that as another validation experiment. A final generalization claim requires a separate evaluation plan.

## 7. Optional: several predictions from one dropout model

Keep trained dropout active at inference and repeat a forward pass. This is **Monte Carlo dropout**. For classification, each pass produces a probability vector \(p^{(t)}\); average those vectors:

\[
\bar p=\frac1T\sum_{t=1}^Tp^{(t)}.
\]

Average probabilities, not class IDs. Softmax of average logits is generally a different calculation. The supplied `mc_measure` function uses the seed-1 MLP trained with \(p=0.5\), selected in advance for demonstration, and 100 fresh masks. It sets only `nn.Dropout` modules to training mode and restores evaluation afterward.

Deterministic evaluation has CE 0.111671, Brier score 0.040718 and 116/120 correct. The actual MC mean has CE 0.113693, Brier score 0.042566 and the same correct count. Brier here is the mean over examples of the **sum across ten classes** of squared probability errors. Repeated inference did not improve these scores.

Probability spread can reveal sensitivity to learned-feature availability. To separate two kinds of ambiguity, define categorical entropy \(H(p)=-\sum_kp_k\log p_k\), in nats. Compare entropy of the mean with mean entropy:

\[
D=H(\bar p)-\frac1T\sum_tH(p^{(t)}).
\]

If two hypothetical passes give \([0.9,0.1]\) and \([0.1,0.9]\), the mean is \([0.5,0.5]\), entropy 0.693147 and disagreement \(D=0.368064\). If both passes instead give \([0.5,0.5]\), the mean is identical but \(D=0\). The first model's sampled predictions disagree; the second is ambiguous on every pass. This arithmetic is illustrative, separate from the measured digit outputs.

In the actual run, validation specimen source ID 299 is a digit 1 but the mean predicts 6; predictive entropy is 1.216163 and disagreement 0.493401. Source ID 251 is correctly classified as 4, with entropy 0.061497 and disagreement 0.020239. Those two examples help interpret the quantities, but do not validate a universal error-detection threshold.

[Gal and Ghahramani](https://proceedings.mlr.press/v48/gal16.html) give an approximate Bayesian interpretation under a specified variational family and prior/objective relationships. Arbitrary masks added to a model trained without them are not automatically posterior samples. Our experiment measures mask-induced prediction variability; it does not claim an exact Bayesian posterior, calibrated uncertainty or guaranteed detection of unfamiliar inputs.

For regression, spread of sampled prediction means omits observation noise. In a model that explicitly assumes Gaussian observation variance \(\tau^{-1}\), predictive variance includes that term plus variability of the means; \(\tau\) is precision, and \(\tau^{-1}\) is variance. Increasing \(T\) reduces Monte Carlo estimation noise, not model bias or all uncertainty.

A useful application is selecting examples for labeling: disagreement can suggest where another label might help. Another is routing ambiguous inputs for human review. Both require validating the acquisition/deferral policy on the deployment setting. They are possible uses of these quantities, not safety or coverage certificates.

## 8. Optional: choose a noise pattern for a reason

Several related methods answer different questions:

- **DropBlock** hides contiguous regions within feature maps. A \(3\times3\) blank region interrupts local redundant evidence differently from nine scattered zeros. Overlapping blocks and boundaries mean the seed probability for block centers is not simply the final fraction removed. Read the [original DropBlock paper](https://arxiv.org/abs/1810.12890) before implementing its sampling and normalization recipe.
- **DropConnect** masks weights rather than activations. A missing activation removes its contribution to every recipient; missing individual weights can remove different connections to different recipients. [Wan et al.](https://proceedings.mlr.press/v28/wan13.html) develop that distinction.
- **Zoneout** carries selected previous recurrent-state values forward instead of replacing them with zero. If the old state is 0.7 and a proposed update is 0.2, a preserve decision returns 0.7. It is a memory-preserving intervention across time, not ordinary hidden dropout under another name. [Zoneout](https://arxiv.org/abs/1606.01305).
- **Shake-Shake** uses stochastic affine combinations of parallel branches; **ShakeDrop** develops a related residual regularizer with its own stabilization behavior. Their forward/backward recipes require separate study; arbitrary branch noise is not an interchangeable substitute. [Shake-Shake](https://arxiv.org/abs/1705.07485), [ShakeDrop](https://arxiv.org/abs/1802.02375).
- **Gaussian/variational dropout** extends multiplicative noise and can learn noise parameters. Kingma et al.'s local reparameterization and Molchanov et al.'s sparsification are distinct developments from ordinary fixed-rate MC dropout. Additional parameter cost depends on whether noise parameters are shared or per weight; fixed Bernoulli dropout does not double model parameters. [Local reparameterization](https://arxiv.org/abs/1506.02557), [variational sparsification](https://arxiv.org/abs/1701.05369).

Attention probability dropout offers another instructive preview. A normalized row \([0.25,0.75]\), mask \([1,0]\) and \(q=0.5\) becomes \([0.5,0]\), whose sum is 0.5. The operation preserves each weight's expectation, not the row sum on every pass. Renormalizing afterward defines a different operation. The attention lesson will explain the values being mixed; the masking calculation already shows why a sampled result need not be a convex average.

A practical choice starts with the unmasked baseline, the dependency you want to perturb, and a valid validation procedure. Compare a small set of rates and placement choices. Revisit learning rate or training duration if the noisy objective is difficult to fit. Do not copy an architecture's default as a theorem about your data, or assume massive datasets make memorization impossible.

## 9. Practice with changed inputs

### 1. Repair the scaling

A value 3 survives with probability 0.75. A program multiplies survivors by 0.75. What are its expected output and the correct survivor value?

<details><summary>Hint</summary>

Distinguish the chance of survival from the value conditional on survival.

</details>

<details><summary>Worked solution</summary>

Its expectation is \(0.75(3\cdot0.75)=1.6875\). Correct inverted scaling returns \(3/0.75=4\) when kept and 0 otherwise, giving expectation 3.

</details>

### 2. Follow a different update

Let \(h=[2,-1]\), \(w=[0.5,1]\), target 0, \(q=0.5\), mask \([0,1]\), half-squared loss. Find output, weight gradient and weights after an SGD step of 0.1.

<details><summary>Hint</summary>

Form the masked input before differentiating.

</details>

<details><summary>Worked solution</summary>

Masked input \([0,-2]\), output −2, loss 2, gradient \([0,4]\), new weights \([0.5,0.6]\). With the same mask the new output is −1.2 and loss 0.72.

</details>

### 3. Design the mask axes

For \([B,T,D]=[3,5,4]\), hide a feature consistently over all time positions for each example, but allow different examples to keep different features. What mask shape is appropriate?

<details><summary>Hint</summary>

List which axis must share a decision, and which axes need independent decisions.

</details>

<details><summary>Worked solution</summary>

\([3,1,4]\). A \([3,5,4]\) mask varies over time; \([3,1,1]\) hides the whole example's branch; \([1,1,4]\) forces the same feature decisions across examples. Actual recurrent placement needs its own temporal-state reasoning.

</details>

### 4. Catch two validation bugs

A model with dropout and BatchNorm is scored inside `no_grad()` after training. A second engineer “fixes” its randomness by resetting the random seed before each prediction.

<details><summary>Hint</summary>

Separate whether gradients are recorded, whether modules use training behavior, and whether a random draw is repeated.

</details>

<details><summary>Worked solution</summary>

`no_grad()` alone leaves training behavior active. Resetting the seed repeats randomness rather than making the intended deterministic predictor; BatchNorm state can still change. Use `eval()` plus no gradient recording for ordinary validation. For an explicitly requested MC procedure, selectively enable dropout and draw fresh masks without modifying BatchNorm state.

</details>

### 5. Choose from evidence

Model A scores training CE 0.01 and validation CE 0.20. Model B scores 0.30 on both. Which has the smaller gap, and which has the better observed validation loss?

<details><summary>Hint</summary>

Compute the two gaps, then compare the validation objective independently of those gaps.

</details>

<details><summary>Worked solution</summary>

B has zero gap; A has lower validation loss. B's small gap is compatible with underfitting. These values are a hypothetical diagnostic, not the digit measurements. The gap alone is not the selection objective.

</details>

### 6. Count blocks and distinguish conventions

Six blocks use a zero-first schedule ending at drop probability 0.4. Find the rates and expected active count. A programmer computes every correction before masking. Does your answer predict saved computation?

<details><summary>Hint</summary>

Write the endpoint schedule using block indices beginning at zero. Then distinguish contributing branches from executed branch functions.

</details>

<details><summary>Worked solution</summary>

Rates \([0,0.08,0.16,0.24,0.32,0.4]\) sum to 1.2, so expected active count is 4.8. Every correction was computed; 20% fewer active contributions does not imply 20% less computation.

</details>

### 7. Explain a zero uncertainty score

Every MC pass assigns probability 0.99 to the same wrong class. What does low mask disagreement establish?

<details><summary>Hint</summary>

Ask what changes across the sampled predictions and what information about correctness the masks actually provide.

</details>

<details><summary>Worked solution</summary>

The sampled masks agree, not that the prediction is correct or the input familiar. More passes estimate that agreement more precisely. Assess probability quality and any deferral policy against observed outcomes on relevant held-out data.

</details>

## 10. Continue and read another explanation

You can now trace a sampled mask through values, gradients and mode changes, distinguish masking units, and interpret an actual validation comparison. Next, [Convolution, Pooling & Receptive Fields](/learn/path/full-curriculum/convolution-pooling-receptive-fields?module=deep-learning-fundamentals) explains how spatially arranged features are created and combined—the structure that made channel and region masks meaningful here.

For another learning route, [Dive into Deep Learning §5.6](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html) offers a small network diagram and a from-scratch/built-in comparison. Its example uses Fashion-MNIST and a different experiment budget. Use it to connect the masked diagram to code, not as a substitute for checking this lesson's outcomes.

The [2014 JMLR dropout paper](https://jmlr.org/papers/v15/srivastava14a.html) is the historical reference: §§4–5 formalize model/training, §7 studies rates and model averaging, and §9 explores marginalization. Its symbol \(p\) is a **keep** probability; this lesson uses \(p\) for **drop** probability. The exact input-dropout squared-loss penalty is taught in the [Classical ML regularization lesson](/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml); the deep nonlinear objective here should not be silently replaced by a generic L2 penalty. The paper's RBM and unsupervised-pretraining extensions are further probabilistic-model study, not prerequisites for this route.
