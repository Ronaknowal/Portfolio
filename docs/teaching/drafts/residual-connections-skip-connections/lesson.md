# Residual Connections & Skip Connections: Keep a Path, Learn a Correction

Suppose a network has already formed a useful description of an image. Its next block can replace that description completely—or keep it and propose a correction.

A **residual connection** implements the second choice:

\[
\text{new representation}=\text{current representation}+\text{learned correction}.
\]

The extra route carries the current values around a group of operations and adds them at the end. This gives both information and learning signals a direct path through the block. It is a powerful design choice, but the correction can still cancel, amplify, or distort what the direct path carries.

**First pass:** trace one correction and its parameter update, follow the two backward paths, compare operation order and shapes, and run the small digit experiment. Practice 1–4 check those ideas. Scaled residuals, path expansions, architecture families, and the differential-equation connection are deeper branches; you can revisit them after the core.

The [previous initialization lesson](/learn/path/full-curriculum/weight-initialization-xavier-kaiming-p?module=deep-learning-fundamentals) asked how transformations should start. This lesson asks how to connect them. We will use vectors and small MLPs first, so convolution and attention are not prerequisites.

## One block, with numbers you can follow

Write the input as \(x\), the correction function as \(F\), and the output as \(y\):

\[
y=x+F(x).
\]

The straight-through route is an **identity**: it returns exactly the value it receives. The other route may contain several layers. Calling it a **skip** does not mean the computer usually skips calculating \(F(x)\); both routes contribute to an ordinary forward pass.

Consider a two-coordinate representation and a small linear correction:

\[
x=\begin{bmatrix}2\\-1\end{bmatrix},\qquad
W=\begin{bmatrix}0.1&0.2\\0&-0.5\end{bmatrix},\qquad F(x)=Wx.
\]

Follow the two paths:

| Route | Calculation | Value |
|---|---|---|
| Direct | Preserve \(x\) | \([2,-1]\) |
| Correction, coordinate 1 | \(0.1(2)+0.2(-1)\) | 0 |
| Correction, coordinate 2 | \(0(2)-0.5(-1)\) | 0.5 |
| Add coordinate by coordinate | \([2,-1]+[0,0.5]\) | \([2,-0.5]\) |

The first coordinate stayed unchanged. The second moved upward by 0.5. A correction may be positive or negative; the network should be able to remove a feature as well as add one.

**Visual walkthrough:** carry two labeled values along a direct lane, and send the same values through the weight matrix in another lane. Reveal the products, the correction vector, and the addition in order. Then change one weight and predict which output coordinates can change. Use the coordinate labels to explain each change.

If the desired output is \(t=[1,0]\), this correction has not finished the job. Use half the squared distance as the loss:

\[
\mathcal L=\tfrac12\|y-t\|^2
=\tfrac12(1^2+(-0.5)^2)=0.625.
\]

The derivative with respect to \(y\) is \(g=y-t=[1,-0.5]\). For \(F(x)=Wx\), the weight gradient is the outer product \(gx^\top\):

\[
\nabla_W\mathcal L=
\begin{bmatrix}2&-1\\-1&0.5\end{bmatrix}.
\]

One gradient-descent step with learning rate 0.1 gives

\[
W_{\mathrm{new}}=W-0.1\nabla_W\mathcal L
=\begin{bmatrix}-0.1&0.3\\0.1&-0.55\end{bmatrix}.
\]

The new correction is \([-0.5,0.75]\), the new output is \([1.5,-0.25]\), and the loss is 0.15625. The skip path did not learn a weight. It changed the input-output function and therefore the error used to train the correction.

This is a linear example for tracing arithmetic, not a claim that one linear residual layer is more expressive than a general linear layer. Since \(x+Wx=(I+W)x\), both can represent the same linear maps. A practical residual branch usually includes a nonlinearity.

## Why a direct path helps optimization

A deeper model can have many possible representations, but a training algorithm still has to find useful parameters. The historical **degradation problem** was an observation that adding layers to certain plain networks increased their training error. A worse training fit cannot be explained simply by saying the larger model memorized its training data too well.

If extra blocks can represent identity maps, a deeper model can reproduce a shallower model's function. That establishes the existence of an equally good configuration, not a guarantee that an optimizer will reach it. The original ResNet work proposed learning \(H(x)-x\) when the desired block mapping is \(H(x)\), making the identity case \(F(x)=0\) convenient to represent. [He et al., §§1 and 3](https://arxiv.org/pdf/1512.03385)

Several qualifications matter. The identity-based argument requires the added blocks to support that identity construction. A nonlinearity after the addition can change that construction. Zero correction is an available setting, not the default of every randomly initialized block. And a training-error plot alone cannot identify every cause of an optimization difficulty; it does not rule out all gradient-flow problems.

The direct route also changes backpropagation. An addition node sends the upstream derivative into both inputs. At the block input the two contributions meet:

\[
\nabla_x\mathcal L=g+J_F(x)^\top g,
\]

where \(J_F(x)\) is the matrix describing small output changes caused by small input changes. We use column gradients, which is why the transpose appears.

In the numerical example, the direct route contributes \([1,-0.5]\). The correction route contributes \(W^\top g=[0.1,0.45]\). Their total is \([1.1,-0.05]\).

Notice the second coordinate: its gradient became **smaller** after the two routes combined. An unattenuated contribution does not imply that the final sum has a lower bound on its magnitude. This is the distinction between seeing a path in a graph and evaluating the complete derivative.

## The skip cannot promise a nonvanishing total gradient

Use a scalar correction \(F(x)=ax\). The block is \(y=(1+a)x\), with derivative \(1+a\). Ten identical blocks have derivative \((1+a)^{10}\):

| Branch slope \(a\) | One-block derivative | Ten-block derivative |
|---:|---:|---:|
| −1 | 0 | 0 |
| −0.5 | 0.5 | 0.0009765625 |
| 0 | 1 | 1 |
| 0.1 | 1.1 | 2.59374 |
| 1 | 2 | 1024 |

At \(a=-1\), the learned branch cancels the identity exactly. At \(a=-0.5\), sensitivity contracts repeatedly despite a skip in every block. At \(a=1\), it explodes. At \(a=0\), the route is an exact identity.

**Gradient investigation:** choose the correction slope and a number of blocks, then predict contraction, preservation, or growth before revealing the computed gain. Build a stack that still contains every skip but reduces the ten-block gain below 0.001. Then repair that gain by editing the correction. This is a constructive way to understand both the benefit and the limit.

For vector blocks \(x_{k+1}=x_k+F_k(x_k)\), the full Jacobian is the ordered product

\[
J_{\mathrm{total}}=(I+J_{F_{L-1}})\cdots(I+J_{F_0}).
\]

The identity term offers a direct contribution. Other terms can reinforce or cancel it, and matrices generally cannot be reordered. If each residual Jacobian has operator norm at most \(\epsilon<1\), one-block directional gains lie between \(1-\epsilon\) and \(1+\epsilon\). Across \(L\) blocks, even the bound \((1-\epsilon)^L\) can become small. Small per-block changes and depth must be considered together.

The identity-mappings study analyzes these direct routes and compares shortcut and activation choices. Its empirical evidence supports their usefulness; it should not be turned into a universal impossibility of vanishing gradients. [He et al., §§2–4](https://arxiv.org/pdf/1603.05027)

## Operation order changes what is preserved

Compare three formulas:

| Pattern | Output | What happens when \(F=0\)? |
|---|---|---|
| Pure additive residual | \(x+F(x)\) | Exactly \(x\) |
| ReLU after addition | \(\operatorname{ReLU}(x+F(x))\) | \(\operatorname{ReLU}(x)\) |
| LayerNorm after addition | \(\operatorname{LN}(x+F(x))\) | \(\operatorname{LN}(x)\) |

For \(x=[-2,1]\), zero correction gives \([-2,1]\) in the first row and \([0,1]\) in the second. The second row's local Jacobian is \(\operatorname{diag}(0,1)\). Its negative coordinate cannot pass through unchanged.

LayerNorm across the two features produces approximately \([-0.999998,0.999998]\) using epsilon \(10^{-5}\), unit affine scale and zero bias. Adding the same constant to both input coordinates leaves LayerNorm's output unchanged. Its Jacobian therefore removes that common-shift direction, even though the output magnitude looks well controlled.

In **pre-activation** residual designs, normalization and activation live inside the correction path, and the addition has no trailing activation. In **pre-norm** designs, a common form is \(x+F(\operatorname{LN}(x))\). With zero correction, both preserve the raw skip value. The normalization still affects the branch and its derivative.

Moving an operation is not just relabeling a block. A post-activation ResNet can work well; Torchvision's standard BasicBlock and Bottleneck still apply ReLU after addition. The correct conclusion is to understand the chosen design, not to treat every post-activation block as a bug. [Torchvision 0.26 ResNet source](https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/resnet.html)

For a future Transformer lesson, the two patterns are:

\[
\begin{aligned}
\text{pre-norm: }&u=x+\operatorname{Attention}(\operatorname{LN}(x)),\\
&y=u+\operatorname{MLP}(\operatorname{LN}(u));\\
\text{post-norm: }&u=\operatorname{LN}(x+\operatorname{Attention}(x)),\\
&y=\operatorname{LN}(u+\operatorname{MLP}(u)).
\end{aligned}
\]

Treat Attention here as another same-shape transformation; its mechanism comes later. Pre-norm keeps normalization off the direct route. It does not force the accumulated residual stream to have constant variance, or eliminate every need for warmup. Xiong et al. study expected initialization gradients under a particular model and demonstrate useful pre-norm training results; that is narrower than stability at any depth and any learning rate. [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)

## Shape agreement also needs coordinate agreement

Addition combines corresponding entries. For a batch of vectors, a residual output with shape [batch, 32] should be added to another [batch, 32] representation with the intended feature correspondence. A [batch, 1] output may broadcast without a runtime error while performing a quite different operation.

When dimensions change, the skip can use a projection \(P\):

\[
y=P x+F(x).
\]

For example,

\[
P=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}
\quad\text{maps}\quad [2,-1]\ \text{to}\ [2,-1,1].
\]

The correction must now contain three coordinates. This is no longer an identity shortcut. An upstream gradient \([1,2,3]\) returns through the skip as \(P^\top[1,2,3]=[4,5]\). A projection changes both forward representation and backward sensitivity.

For an image tensor [batch, channels, height, width], a future convolutional block may change channels and spatial resolution. A 1×1 convolution can mix channels at each position; stride can select a coarser grid. Both branches must agree on output shape **and spatial alignment**, including padding choices. A projection is one solution; pooling or deliberate padding may fit other designs. Do not insert a learned projection where an identity already expresses the intended connection without considering its effects.

**Shape investigation:** connect named feature sockets, choose identity or an explicit projection, and compute the resulting values. Check the feature labels and actual repeated values as well as the dimension count.

## Start near identity while leaving something able to learn

There are different ways to begin with a small correction.

A two-layer branch can use \(F(x)=W_2\phi(W_1x)\), with random \(W_1\) and zero \(W_2\). The initial correction is zero. The first gradient into \(W_1\) is zero, but \(W_2\) can receive a gradient from the nonzero hidden features. After \(W_2\) moves, the earlier layer can start learning.

Compare this with \(F(x)=\operatorname{ReLU}(W_2\phi(W_1x))\), again with \(W_2=0\). Using PyTorch's zero derivative for ReLU at zero blocks the gradient into \(W_2\) as well. A graph that passes the input through beautifully may have a correction branch that never learns.

The saved fixture uses \(x=[1,2]\), \(W_1=I\), \(W_2=0\), and loss \(\tfrac12\|y\|^2\). With the final linear output, \(\nabla_{W_2}\mathcal L=[[1,2],[2,4]]\). With the extra final ReLU, both weight gradients are zero.

**ReZero** instead uses a trainable scalar:

\[
y=x+\alpha F(x),\qquad \alpha_{\mathrm{initial}}=0.
\]

At initialization, the block's input-output map is identity. For upstream gradient \(g\),

\[
\frac{\partial\mathcal L}{\partial\alpha}=g^\top F(x),
\qquad
\nabla_W\mathcal L=\alpha J_{F,W}^\top g.
\]

The scalar may move before the branch parameters do. It is a weight, not a probability, and may become negative. With \(x=[1,2]\), \(F(x)=[0,0.7]\), target zero, and half squared loss, the scalar gradient is 1.4. SGD at 0.1 moves \(\alpha\) from zero to −0.14. A later update can reach the branch weights. If the scalar gradient is also zero, that first movement is not guaranteed. [ReZero paper, mechanism and scalar example](https://arxiv.org/pdf/2003.04887)

**LayerScale** uses a separate learned scalar for each output feature: \(x+\lambda\odot F(x)\). This gives channels separate update scales. Small nonzero values can allow small branch gradients immediately. Zero per-channel values remain per-channel parameters; they do not lose that flexibility just because their initial values coincide. The CaiT study tested particular small initial values and training recipes, not a universal “all networks above 24 layers need \(10^{-6}\)” law. [LayerScale mechanism and initialization study](https://arxiv.org/pdf/2103.17239)

A fixed factor such as \(1/\sqrt L\) is another possible design experiment. Its rationale often assumes approximately comparable residual contributions. In general,

\[
E\|x+F(x)\|^2=E\|x\|^2+E\|F(x)\|^2+2E[x^\top F(x)],
\]

where the expectations average over the input distribution under study. Correlations can change the growth. Scaling, initialization, normalization, and optimizer choices work together; none can be inferred from depth alone.

## A complete comparison on real handwritten digits

Download [residual-experiments.py](./residual-experiments.py) and [digits-400.csv](./digits-400.csv) into one directory. The [provenance](./data-provenance.md) gives attribution and the exact split. These 400 real 8×8 UCI digit images use pixel values 0–16. Divide by 16, flatten to 64 inputs, and classify digits 0–9.

With PyTorch, NumPy, and scikit-learn installed:

```text
python residual-experiments.py
```

The program contains all data loading, initialization, models, training, diagnostics, and evaluation. It was executed on CPU with PyTorch 2.14.0+cpu, NumPy 2.3.5 and scikit-learn 1.9.1. It downloads no model or data.

Every model starts with a 64→32 linear stem and tanh, and ends with a 32→10 logit head. The stem-only baseline connects those two directly. Other models insert 2, 6, or 12 blocks, each containing

\[
F(h)=W_2\tanh(W_1\operatorname{LN}(h)+b_1)+b_2.
\]

LayerNorm operates across each row's 32 features with its standard affine parameters. The final branch output is linear, so its correction may have either sign.

| Mode | Block output |
|---|---|
| Plain | \(F(h)\) |
| Residual | \(h+F(h)\) |
| Fixed scaled | \(h+F(h)/\sqrt L\) |
| Learned zero gate | \(h+\alpha F(h)\), each block's \(\alpha\) starts at zero |

The learned-gate model retains LayerNorm to keep this comparison matched. It is a ReZero-style gate experiment, not a reproduction of the paper's complete normalization-free recipe.

Read `Refinement.forward`: the important architectural difference is the return expression. `DigitNetwork` constructs the same branch weights for a given seed/block index and the same stem/head for a given seed, across modes and depths. All models use the same stratified 280/120 training/validation split. Each trains for 250 full-batch Adam updates at learning rate 0.003. The three seeds were declared before examining results.

Each ordinary block has 2,176 trainable parameters: two 32×32 weights and two 32 biases, plus 32 LayerNorm scales and 32 offsets. The stem/head have 2,410 total. Thus a 12-block plain, residual, or fixed-scaled model has 28,522 parameters; the learned-gate model has 12 extra scalars. Parameter equality does not imply equal runtime, and deeper configurations do more work per update.

Actual final seed-1 results:

| Blocks | Mode | Training CE | Validation CE | Correct / 120 |
|---:|---|---:|---:|---:|
| 0 | Stem only | 0.027306 | 0.107264 | 117 |
| 2 | Plain | 0.000968 | 0.195945 | 115 |
| 2 | Residual | 0.001076 | 0.112114 | 115 |
| 6 | Plain | 0.000702 | 0.254232 | 116 |
| 6 | Residual | 0.000664 | 0.221809 | 116 |
| 12 | Plain | 0.002000 | 0.599136 | 110 |
| 12 | Residual | 0.000389 | 0.303736 | 111 |
| 12 | Fixed scaled | 0.000405 | 0.144360 | 113 |
| 12 | Learned zero gate | 0.000184 | 0.161761 | 116 |

These results do **not** reproduce the exact historical degradation experiment. All these plain networks fit the training set well at this budget. Their deeper validation results can still worsen. Across seeds, the 12-block residual model scores 111–116 correct; the scaled model 113–116; and the learned-gate model 115–117. The much smaller stem-only baseline scores 117–118. More depth is not necessary for every dataset.

The program records CE and correct counts at updates 0, 1, 25, 100, and 250. It also records layer activation mean squares and gradients of the loss on a fixed 32-example training subset. Those diagnostics answer a specific local question; a larger gradient norm is not automatically more useful. Parameter-displacement records confirm that branches actually changed during fitting.

**Try a controlled change:** first choose one seed and compare plain versus residual at the same depth. Then compare residual versus scaled. Identify which variables are held fixed and which change. For a new experiment, alter one branch placement or initialization, write down your prediction, and keep the resulting record separate. These are validation comparisons; this packet has no final test estimate.

## Many routes are useful, but they are not independent models

For two linear residual blocks, expansion is exact:

\[
(I+W_1)(I+W_0)x=x+W_0x+W_1x+W_1W_0x.
\]

You can identify four paths. With \(L\) linear blocks there are \(2^L\) such products. They share weights and are summed, not independently trained and averaged.

For nonlinear blocks, do not distribute a function over addition. With \(F_0(x)=F_1(x)=x^2\) and \(x=1\), the actual result is

\[
x_1=1+1=2,\qquad x_2=2+2^2=6.
\]

The tempting expression \(x+F_0(x)+F_1(x)+F_1(F_0(x))\) gives 4 instead. The valid telescoping form is \(x_L=x_0+\sum_kF_k(x_k)\); each \(x_k\) already depends on earlier corrections.

The ensemble-of-paths interpretation is useful for thinking about multiple gradient routes and testing robustness, but it is not a theorem that deleting any block is harmless. [Veit et al.](https://arxiv.org/abs/1605.06431)

Our program performs **ablations**: after fitting, omit one residual block at a time and measure the changed validation result without retraining. For seed 1 with six unscaled residual blocks, the full model gets 116/120 correct. Single-block omissions range from 28 to 114 correct. That is a direct counterexample to promising deletion robustness from architecture alone. Different scaled/gated runs can be less sensitive, but those outcomes also depend on training.

These ablations inspect the fitted model. If you use them to choose a pruned architecture, validation information has guided a new selection; a later final evaluation must account for that workflow.

## Connections that solve different problems

**Addition versus concatenation.** Addition mixes aligned coordinates into one vector. Concatenation keeps them in separate positions. If \(x=[2,-1]\) and \(F(x)=[0,0.5]\), addition yields two values \([2,-0.5]\); concatenation yields four \([2,-1,0,0.5]\).

DenseNet supplies each layer with earlier feature maps by concatenation. Starting with \(c_0\) channels and adding \(k\) channels per layer gives \(c_0+Lk\) channels after \(L\) additions of features. Later layers receive more inputs, so stored activations and arithmetic must be assessed with the actual architecture. It is not automatically a more expensive or more accurate substitute at every budget. [DenseNet](https://arxiv.org/abs/1608.06993)

**Gated carry and parallel corrections.** A Highway-style block can compute \(T(x)\odot H(x)+(1-T(x))\odot x\), with a learned gate choosing how much each feature travels along each route. The gate also changes derivatives; a nearly closed carry route is not an identity. ResNeXt instead aggregates several related transformations inside a residual branch. Its **cardinality** counts those transformations. Grouped convolution is a later implementation tool for such structured channel computations. [Highway Networks](https://arxiv.org/abs/1505.00387), [ResNeXt](https://arxiv.org/abs/1611.05431)

**Different spatial scales inside a block.** Res2Net organizes smaller channel groups with hierarchical residual-like connections, allowing different groups to accumulate different spatial contexts. Revisit its diagram after learning receptive fields; for now, recognize that an outer residual connection and the internal design of its correction are separate choices. [Res2Net](https://arxiv.org/abs/1904.01169)

**Remove noise by learning what to subtract.** If a noisy observation is \(z=s+n\), a network can estimate noise \(\hat n(z)\) and return \(z-\hat n(z)\). For an illustrative three-value signal, \(z=[0.2,0.9,0.4]\) and estimated noise \([0.1,-0.1,0.2]\) give \([0.1,1.0,0.2]\). That arithmetic does not prove the estimated noise was correct; clean/noisy examples are needed to train and evaluate the estimator. DnCNN uses residual learning of this kind. It illustrates a residual target at the whole-model level, which need not mean every internal block has a ResNet shortcut. [DnCNN, §III-B](https://arxiv.org/pdf/1608.03981)

**Refinement as a time step.** An update \(x_{k+1}=x_k+\Delta t\,f(x_k)\) resembles the forward-Euler method for a differential equation. For \(f(x)=-x\), it becomes \(x_{k+1}=(1-\Delta t)x_k\). A step of 0.1 contracts by 0.9; a step of 2.5 alternates sign and grows by 1.5 per step. The differential equation decays smoothly, yet an overly large discrete step is unstable. This offers another reason that an identity route alone cannot ensure stability. Neural ODEs build a deeper connection by learning a derivative and using a differential-equation solver; an arbitrary residual network is not automatically a convergent solver. [Neural ODEs](https://arxiv.org/abs/1806.07366)

## What really costs memory and computation

An identity shortcut has no learned weights. Adding two equal-sized tensors performs one addition per entry and requires access to both values at that moment. The shortcut does not require a physical copy just because a diagram shows two lanes.

Backward through a pure addition only needs to route the upstream derivative; it does not need to save the input values to differentiate the addition itself. Other operations in \(F\), their parameter gradients, and the lifetime of values across the branch do require storage. Memory therefore depends on the actual saved tensors and execution schedule, not simply “one saved input per plus sign.”

Activation checkpointing trades some saved intermediates for recomputation. In a simplified equal-cost chain, grouping \(L\) layers into chunks of size \(k\) suggests storage on the order of \(L/k+k\), minimized near \(k=\sqrt L\). Real networks have unequal tensor sizes, workspaces, parameter/optimizer storage, and implementation choices. The simple expression is not a guaranteed whole-device memory bound.

PyTorch's current checkpoint documentation recommends an explicit `use_reentrant=False` and warns that recomputation must agree with the original forward behavior. Random masks and mutable state deserve particular care. [PyTorch checkpoint contract](https://docs.pytorch.org/docs/2.14/checkpoint.html)

The [next lesson](/learn/path/full-curriculum/dropout-droppath-stochastic-depth?module=deep-learning-fundamentals) will randomly mask residual branches. If code first computes `F(x)` and then multiplies it by zero, that branch's forward work has already happened. Avoid translating a masked-path diagram into an invented speedup.

## Practice

### 1. Change the correction

For the opening \(x,W,t\), change only \(W_{11}\) from 0.1 to −0.1, keeping the other three entries fixed. Calculate output and loss before looking at the solution.

<details><summary>Hint</summary>

only the first correction coordinate changes.

</details>

<details><summary>Worked solution</summary>

the correction becomes \([-0.4,0.5]\), output \([1.6,-0.5]\), and loss \(\tfrac12(0.6^2+(-0.5)^2)=0.305\). This improves the initial 0.625, but differs from updating all weights with the gradient.

</details>

### 2. A skip in every block, almost no sensitivity

Ten blocks use \(F(x)=-0.5x\). Find total derivative. Then choose a branch slope that gives total derivative exactly 1.

<details><summary>Hint</summary>

Multiply the local derivatives. A final gain can conceal sign changes along the way.

</details>

<details><summary>Worked solution</summary>

\(0.5^{10}=0.0009765625\). Slope 0 gives each block derivative 1, hence total 1. Slope −2 also gives total 1 for an even ten blocks but reverses sign at each block; it is a useful reminder that a final gain alone does not describe intermediate behavior.

</details>

### 3. A projection is not identity

Use the 3×2 projection above with input \([1,3]\) and upstream gradient \([2,-1,4]\). Find skip output and skip contribution to input gradient.

<details><summary>Hint</summary>

Apply the projection in the forward direction and its transpose to the upstream gradient, coordinate by coordinate.

</details>

<details><summary>Worked solution</summary>

output \([1,3,4]\); gradient \([2+4,-1+4]=[6,3]\). The two-dimensional input receives contributions from three output coordinates.

</details>

### 4. Find the branch that cannot begin learning

Both proposed blocks initially output their input. One ends its zero-initialized correction with a linear map; the other adds a ReLU after that map. Which parameter gradient should you inspect to distinguish them?

<details><summary>Hint</summary>

Identical forward outputs can still put a different final operation on the backward path.

</details>

<details><summary>Worked solution</summary>

inspect the final correction weight's gradient under the same input and loss. In our fixture the linear case gets [[1,2],[2,4]], while the extra-ReLU case gets zero. Check later displacement too; the identity output by itself does not show that a correction will learn.

</details>

### 5. Count features

A dense concatenation group starts with 16 channels and adds 8 channels in each of four layers. How many channels does the fourth layer receive, and how many exist afterward? Contrast addition with a fixed 16-channel residual stream.

<details><summary>Hint</summary>

Count the outputs of preceding layers before including the fourth layer itself.

</details>

<details><summary>Worked solution</summary>

the fourth layer receives \(16+3(8)=40\) channels; afterward there are 48. A shape-preserving additive stream remains 16 channels. Its correction must output 16, even if it uses a different internal width.

</details>

### 6. Interpret a disappointing ablation

A residual classifier loses accuracy when you remove a block. Does this refute the useful direct-gradient route, prove that the block must never be pruned, or justify another experiment?

<details><summary>Hint</summary>

Separate an algebraic property of the graph from the measured intervention on these trained parameters.

</details>

<details><summary>Worked solution</summary>

it shows that this trained model depends on the block under the measured intervention. The derivative identity remains true. Retraining or a different regularization method may change the outcome, but that is another experiment with its own development and final-evaluation protocol. The ablation alone cannot guarantee successful pruning.

</details>

### 7. An optional stability calculation

For \(x_{k+1}=(1-\Delta t)x_k\), find positive step sizes that strictly contract magnitude. Explain what happens at \(\Delta t=2\).

<details><summary>Hint</summary>

require \(|1-\Delta t|<1\).

</details>

<details><summary>Worked solution</summary>

\(0<\Delta t<2\). At 2, each step changes the sign without shrinking magnitude. For larger positive steps, magnitude grows. This is a scalar discretization calculation, not a bound for every trained residual network.

</details>

## Another way to learn

- [Dive into Deep Learning: ResNet and ResNeXt](https://d2l.ai/chapter_convolutional-modern/resnet.html) offers diagrams and a convolutional implementation. Read its function-class argument after the local arithmetic; revisit its image model after the convolution lesson. Operation-count reductions should not be read as measured wall-clock speedups.
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027) is the detailed route through the two path conditions and activation-placement experiments. Distinguish its empirical observations from the exact algebra.
- [Torchvision's ResNet source](https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/resnet.html) lets you identify actual addition, projection, and post-activation lines in a maintained implementation.
- [ReZero](https://arxiv.org/pdf/2003.04887) and [LayerScale](https://arxiv.org/pdf/2103.17239) are useful next readings for the optional learned-scale branch.

The next module topic is [Dropout, DropPath & Stochastic Depth](/learn/path/full-curriculum/dropout-droppath-stochastic-depth?module=deep-learning-fundamentals). You now know exactly where a residual correction sits. The next question is what the model learns when some contributions are deliberately removed during training.
