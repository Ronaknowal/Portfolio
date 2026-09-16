# Backpropagation & Automatic Differentiation

The previous lesson built a network that turns inputs into predictions. Training needs the reverse question: **which adjustable numbers contributed to the error, and how would a small change in each affect it?** Backpropagation answers that question by combining local sensitivities through the computation you already performed.

It computes derivatives. The optimizer then uses those derivatives to change parameters. Keeping these two jobs separate explains why a correct backward pass can accompany an overly large, harmful update—and why a decreasing loss does not certify a correct backward pass.

**First pass:** read §§1–5 and solve practices1–4. You should be able to trace a shared computation, obtain a network's gradients and diagnose a failed numerical check. §6 and practices5–8 are deeper routes through a complete teaching engine, higher derivatives, custom operations and memory tradeoffs. Reading and tracing take roughly60–80minutes; core practice and programs add40–60minutes. The optional engine and advanced practice deserve a separate session.

## 1. A derivative carries the effect of a small change

Start with one scalar neuron:

$$
z=wx+b,\qquad a=\operatorname{ReLU}(z),\qquad
L=\tfrac12(a-y)^2.
$$

The input is $x=2$, weight $w=3$, bias $b=-1$, target $y=1$. Forward computation gives $z=5$, $a=5$, error $a-y=4$, and loss8. The factor1/2 makes the derivative of the squared error especially simple. Later mean-squared-error examples deliberately use a different reduction and retain its factor2.

Suppose $w$ increases a little. The score increases by twice that amount because $x=2$. ReLU has slope1 at the current positive score. Loss changes at a rate of4 with respect to the activation because the current error is4. Combining these effects gives

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
=4\cdot1\cdot2=8.
$$

This is the **chain rule**: multiply sensitivities along a path. A derivative describes the local rate near this state, not the exact effect of any large edit.

[VisualA: forward values above a calculation graph; backward sensitivities below its edges. Distinguish score5, loss8 and gradient8 by labels even when numbers happen to match.]

Working backward:

| Object | Forward value | Sensitivity of loss |
| --- | ---: | ---: |
| $L$ | 8 | $\partial L/\partial L=1$ |
| $a$ | 5 | 4 |
| $z$ | 5 | 4 |
| $w$ | 3 | 8 |
| $b$ | −1 | 4 |
| $x$ | 2 | 12 |

The gradient with respect to the input is meaningful too: it describes how loss changes if the input changes while weights remain fixed. Training normally updates the parameters, not the recorded input. Sensitivity analysis, inverse problems and some explanation methods instead ask about input derivatives. An input gradient describes the model's local behavior; it does not establish a causal explanation of the real world.

A gradient-descent update with learning rate0.1 gives $w=2.2,b=-1.4$. At the same input the new score is3, so the new loss is2. We did not recompute gradients halfway through the simultaneous update: both changes used the same old model.

**InvestigationA — predict, then change the step.** First trace the single neuron. Then use a two-example line fit with $x=(1,2)$, targets(1,3), $w=1,b=0$ and mean squared error. Predict whether one update will reduce loss before revealing its result. Edit the learning rate yourself. At0.1, loss falls from0.5 to0.17; at1, it rises to12.5. At0, nothing changes. Correct derivatives supply a direction locally, not a safe step length automatically.

## 2. Shared computations need a sum of contributions

Programs are graphs because one intermediate result can be reused. Let

$$
u=x\cdot x,\qquad L=u+2u.
$$

At $x=3$, $u=9$ and $L=27$. Loss receives $u$ along two routes: one with coefficient1 and one with coefficient2. Therefore

$$
\bar u=1+2=3,\qquad \bar x=3x+3x=18.
$$

The bar notation $\bar v$ means $\partial L/\partial v$, the sensitivity of the final loss to an intermediate value. It is not another forward value.

There are **two different reasons to add** here. The intermediate $u$ has two consumers, so their contributions accumulate. Then $x$ occupies both input slots of multiplication, so both slot contributions accumulate. Visiting a graph node once must not erase a repeated operand's second contribution.

[VisualB: a fork after $u$, and two distinct input edges from $x$ into the multiply node. Each reverse edge carries its contribution; a small accumulator at $u$ shows1+2, and at $x$ shows9+9.]

**InvestigationB — edit a reused branch.** Change the second branch coefficient from2 to a number you choose. Commit a predicted gradient at $x=3$ before revealing. With coefficient−1, the two branches cancel: forward loss and derivative are both0. With coefficient0, the result is $x^2$, with derivative6. Turning one coefficient down does not necessarily make the gradient smaller in absolute value if it crosses a sign change.

The general reverse rule is

$$
\bar v_i \mathrel{+}= \bar v_j
\frac{\partial v_j}{\partial v_i}.
$$

Read “+=” as “add this path's contribution to any contributions already received.” To apply that rule safely, process an operation only after all downstream users of its result have contributed. This is **reverse topological order**. Forward execution order already respects dependencies; reversing an appropriately recorded operation list is one implementation. A traversal of the reachable graph is another.

**Automatic differentiation**, or AD, applies known derivative rules to elementary operations and combines their numerical values. It does not need to expand one enormous symbolic expression, and it does not perturb every parameter to estimate its effect. It still uses finite-precision arithmetic and the derivative conventions of its primitives. [Baydin and colleagues' AD survey](https://jmlr.org/papers/v18/17-468.html) distinguishes these approaches and develops both accumulation directions.

### The few local rules behind a large network

Let $g$ denote the incoming sensitivity at an operation's output.

| Operation | Contribution to each input | Value needed from forward |
| --- | --- | --- |
| $a+b$ | $g$ to both | None for scalar addition |
| $ab$ | $gb$ to $a$, $ga$ to $b$ | Both operands |
| $\tanh(a)$ | $g(1-\tanh^2(a))$ | Input or tanh output |
| $\operatorname{ReLU}(a)$ | $g\,1[a>0]$ | Positive-input mask |
| $\exp(a)$ | $g\exp(a)$ | Exponential output |
| $\log(a)$, $a>0$ | $g/a$ | Input |

At a nondifferentiable point such as ReLU0, the mathematical derivative is not unique because it does not exist in the ordinary sense. Our engine and PyTorch use0 for this primitive. Do not mistake agreement on that convention for proof of differentiability.

## 3. From individual numbers to tensor gradients

The scalar rules also explain arrays. The main new work is keeping shapes and repeated uses straight.

### Broadcasting forward means summing backward

Suppose a bias $b=(b_1,b_2)$ is added to every row of a3×2 matrix. Each bias entry is used three times. If incoming sensitivities are

$$
G=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix},
\qquad
\bar b=(1+3+5,\;2+4+6)=(9,12).
$$

The backward result has the bias's shape, not the output's shape. More generally, undo newly added axes and expanded size-one axes by summing. Averaging here would be wrong unless an earlier mean reduction supplies that factor.

[VisualC: one bias cell fans into a column of three output cells; reverse arrows collect exactly the three corresponding sensitivities. Show the two different column totals.]

For $Y=AB$, with $A$ shape $N\times D$ and $B$ shape $D\times H$, incoming $G$ has shape $N\times H$. The pullbacks are

$$
\bar A=GB^\top,\qquad \bar B=A^\top G.
$$

For example, $Y_{ij}=\sum_k A_{ik}B_{kj}$. Differentiating with respect to $A_{ik}$ leaves $B_{kj}$, and summing over every affected output $j$ gives the first matrix product. The transpose is a consequence of which dimensions are contracted, not a trick to make a shape error disappear.

### A full two-layer classifier

Here the NumPy teaching engine stores a weight matrix as **input width × output width**. PyTorch's `nn.Linear.weight` stores its transpose, as the previous lesson explained. We state the convention before comparing them.

$$
Z_1=XW_1+b_1,\quad A_1=\tanh(Z_1),\quad
Z_2=A_1W_2+b_2.
$$

For a batch of $N$ rows, input width $D$, hidden width $H$ and $C$ classes:

| Object | Shape |
| --- | --- |
| $X,W_1,b_1$ | $N\times D,\ D\times H,\ H$ |
| $A_1,W_2,b_2$ | $N\times H,\ H\times C,\ C$ |
| logits $Z_2$ and probabilities $P$ | $N\times C$ |
| integer labels $y$ | $N$ |

Mean cross-entropy is $L=-N^{-1}\sum_i\log P_{i,y_i}$. For one row, write it directly as $-z_y+\log\sum_c e^{z_c}$. Differentiating gives $p_c-1[c=y]$; the batch mean supplies1/N:

$$
G_2=(P-\operatorname{onehot}(y))/N.
$$

The remaining reverse pass is

$$
\bar W_2=A_1^\top G_2,\quad \bar b_2=\sum_i(G_2)_{i,:},
\quad \bar A_1=G_2W_2^\top,
$$
$$
G_1=\bar A_1\odot(1-A_1^2),\quad
\bar W_1=X^\top G_1,\quad \bar b_1=\sum_i(G_1)_{i,:}.
$$

The loss becomes a matrix of logit sensitivities, the output layer passes them into hidden activations, tanh changes them according to its slopes, and the first layer collects parameter contributions. ReLU substitutes its positive mask for $1-A_1^2$. A mean squared error $\operatorname{mean}((q-y)^2)$ instead starts with $2(q-y)/K$, where $K$ is the number of averaged entries. The loss definition determines the factor; do not silently drop2 because a different example used half-squared error.

[VisualD: paired forward-shape and backward-shape lanes. Each arrow names the matrix product or reduction, so readers can locate every formula in the program.]

### A real offline training program

The complete download [teaching-autodiff.py](teaching-autodiff.py) contains every operation, graph traversal and training loop; it needs no hidden initialization variables. Keep the accompanying [digits-400.csv](digits-400.csv) beside it. Install `numpy==2.3.5` and `scikit-learn==1.9.1` in a Python3.12 environment, then run:

```text
python teaching-autodiff.py
```

It first learns all four constructed XOR rows using a2→4→1 tanh network and mean squared error, then trains a64→16→10 digit classifier with mean cross-entropy. Parameters begin from the file's explicit seeded normal draws; updates use ordinary full-batch gradient descent. No pretrained network or online input is involved.

The digit data is the same400-image UCI fixture introduced in Perceptrons:28train/12validation images per digit, split seed22, divided by the known pixel bound16. The CSV is a selection from the historical UCI test partition, repartitioned for this lesson; it is neither MNIST nor a new writer-independent benchmark. [Dataset attribution and transformations](data-provenance.md).

The actual executed output includes:

| Updates | XOR mean squared error | Digit training cross-entropy | Digit validation correct /120 |
| ---: | ---: | ---: | ---: |
| 0 | 2.903633 | 3.597477 | 17 |
| 1 | 1.247883 | 3.048748 | 17 |
| 10 | .299061 | 1.957156 | 40 |
| 100 | .113301 | .456258 | 101 |
| 500 | $1.83\times10^{-9}$ | .065882 | 112 |

These columns are two different runs, not competing models on the same task. XOR also continues to2000 updates and reproduces its four targets to roughly machine precision in this run. Its input space contains only those four Boolean rows, so we report truth-table fit, not generalization to a hidden real-world population.

For digits,500 updates give112/120 validation correct. This evaluates the selected instructional model, not the correctness of every derivative. The preceding activation comparison used different width, optimizer, precision and initialization; it is not an isolated comparison of autograd engines.

[VisualE: two clearly separated recorded training traces and the actual XOR outputs. Retain initial loss; random initial logits need not produce $\log C$. A low gradient norm alone is not evidence that a unit is “fully trained” or “dead.”]

## 4. Check derivatives as evidence, not a certificate

A model can learn despite an incorrect derivative. It can also fail despite correct derivatives. Check a backward rule directly on controlled inputs.

A central finite difference estimates a derivative as

$$
D_hf(x)=\frac{f(x+h)-f(x-h)}{2h}.
$$

For a sufficiently smooth function, Taylor expansion gives an $O(h^2)$ truncation term. But subtracting nearby finite-precision values loses significant digits, and division by $h$ amplifies evaluation error. Making $h$ smaller indefinitely can make the estimate worse. Parameter units, value scale, dtype and derivative magnitude all affect a useful perturbation. [Fundamentals of Numerical Computation, §5.5](https://fncbook.com/python/fd-converge/).

Run this complete standard-library calculation:

```python
import math

def central_difference(function, point, step):
    return (function(point + step) - function(point - step)) / (2 * step)

for step in (1e-1, 1e-3, 1e-5, 1e-9, 1e-13, 1e-15):
    estimate = central_difference(math.sin, 1.0, step)
    error = abs(estimate - math.cos(1.0))
    offset_estimate = central_difference(lambda x: 1e12 + x, 1.0, step)
    print(f"{step:.0e}", f"{error:.3e}", f"{offset_estimate:.9f}")
```

Our binary64 calculation gave:

| $h$ | Absolute error for derivative of $\sin(1)$ | Estimated derivative of $10^{12}+x$ at1 |
| ---: | ---: | ---: |
| $10^{-1}$ | $9.001\times10^{-4}$ | .999755859 |
| $10^{-3}$ | $9.005\times10^{-8}$ | .976562500 |
| $10^{-5}$ | $1.114\times10^{-11}$ | 0 |
| $10^{-9}$ | $2.970\times10^{-9}$ | 0 |
| $10^{-13}$ | $1.788\times10^{-4}$ | 0 |
| $10^{-15}$ | .0148092 | 0 |

The second function's mathematical derivative is1 everywhere. At $h=10^{-5}$ both evaluated values round to the same large number, so their difference is0. This does not disprove the derivative; it exposes an unsuitable numerical check. Adding a constant can leave a derivative unchanged while damaging the finite-difference estimate.

**InvestigationF — choose a check you can interpret.** Edit the evaluation point, perturbation and optional large additive offset. Predict whether the smaller perturbation will improve agreement before revealing the actual evaluated pair and derivative estimate. Compare an ordinary sine example with an offset linear function, and a zero-derivative quadratic case. Read the absolute error when the correct derivative is0; relative error with a zero denominator is undefined.

At ReLU0, the symmetric finite difference is0.5 while our derivative convention is0. There is no ordinary derivative there to certify. For a smooth network check, use double precision, fixed input and parameter copies, deterministic state, and perturbations that do not cross a relevant corner. If randomness or running statistics change between the plus and minus evaluations, the calculation compares different functions.

The author checks compared every parameter of a small2→3→2 tanh classifier against PyTorch at the same initialization and the same input. Maximum absolute differences were at most $5.56\times10^{-17}$. Central differences at $h=10^{-5}$ differed by at most $1.16\times10^{-11}$ on those parameters. Separate cases checked repeated operands, broadcasting, matrix multiplication, tanh, ReLU's convention and log/exp composition. The saved check script includes exact inputs, not unnamed “initial” arrays reconstructed after training.

This supports those checked rules, shapes and inputs. It does not certify every graph, dtype, nonfinite value, shape or custom operation. [PyTorch gradcheck](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.gradcheck.gradcheck.html) documents double-precision defaults and issues at nonsmooth points, overlapping storage and nondeterminism. Use absolute and relative tolerances that express the accuracy needed for the actual derivative.

### Forward and backward must describe the same function

For stable cross-entropy, calculate

$$
\log p_c=(z_c-m)-\log\sum_j e^{z_j-m},\qquad m=\max_j z_j.
$$

Do not compute $\log(p_c+\epsilon)$ and keep the unmodified $(p-y)/N$ derivative. That changes the forward objective while leaving the old backward formula. With logits(1000,−1000) and true class1, the stable loss is2000; clipping a tiny probability would instead cap the loss. A stable formulation preserves the intended function across this finite input range.

## 5. Use an autograd framework deliberately

With PyTorch, a leaf tensor whose `requires_grad` flag is true can receive a gradient in its `.grad` field. Operations in ordinary gradient mode construct a graph when needed. A scalar's `backward()` starts from sensitivity1; for a vector output, explicitly supply the output weighting or reduce it to the scalar objective you intend.

This complete program isolates gradient accumulation:

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
loss = x * x
loss.backward(retain_graph=True)
print(x.grad.item())
loss.backward()
print(x.grad.item())
x.grad = None
new_loss = x * x
new_loss.backward()
print(x.grad.item())
```

Expected outputs are6,12,6. PyTorch accumulates leaf gradients; clearing them and recomputing the graph restores one contribution. By contrast, our teaching engine deliberately clears every reachable buffer on each `backward` call. Its second call returns a fresh derivative. An engine's reset policy is an API choice, not a chain-rule difference.

`retain_graph=True` preserves information for another backward through the same forward computation. It does not make the derivative computation differentiable; `create_graph=True` does that. If several losses should contribute together, adding them and making one backward call is often simpler than retaining and repeatedly traversing their shared graph. Holding references to graphs can retain memory; the flag alone does not imply that every new iteration must leak.

| Operation | Meaning | Useful distinction |
| --- | --- | --- |
| `x.detach()` | A value sharing storage without the old derivative path | Does not make an independent copy |
| `torch.no_grad()` | Suppress graph recording for ordinary computations in the context | Does not choose training/evaluation behavior |
| `parameter.requires_grad_(False)` | Exclude that leaf parameter from gradient accumulation | Input gradients may still need a path through its operations |
| `model.eval()` | Select evaluation behavior of modules that have it | Does not disable autograd |

For example, a frozen weight can still multiply an input that requires a gradient; the derivative with respect to that input uses the frozen value. Wrapping the entire computation in `no_grad` would block the desired path. Later Transfer Learning will use this distinction. [PyTorch autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html).

If an intermediate is overwritten before backward needs its old value, differentiation can fail or become invalid. PyTorch tracks many such modifications; the teaching engine instead makes forward arrays read-only and creates new parameter tensors after an update. Do not bypass those protections with hidden mutation or shared writable aliases.

## 6. Deeper routes through AD

### Read the teaching engine as a small system

The [complete engine and two training programs](teaching-autodiff.py) are intended to be read with the derivative tables above. Its main pieces have separate responsibilities:

1. A `Tensor` stores a copied, read-only float64 forward value, gradient buffer, parents and one local pullback.
2. An operation computes its output immediately and installs a closure that will later read the output's incoming gradient.
3. `add_gradient` sums contributions and reduces broadcast dimensions to the parent's shape. It respects the parent's `requires_grad` flag.
4. `backward` visits each reachable node once, clears old buffers, validates the seed and applies pullbacks in reverse order.
5. Each training update creates new parameter leaves from the old values and gradients. The next forward therefore belongs to the new model.

The repeated-operand example explains why a visited set does not remove repeated contributions: the multiply closure still sends one contribution for each operand slot, even when both refer to the same parent. The closure does not run when the operation is first created; its output gradient will only be available during backward.

Here are the central local rules from the full file:

```python
# Inside Tensor.__mul__, after constructing result:
def pullback():
    self.add_gradient(other.data * result.grad)
    other.add_gradient(self.data * result.grad)

# Inside Tensor.__matmul__, for two matrices:
def pullback():
    self.add_gradient(result.grad @ other.data.T)
    other.add_gradient(self.data.T @ result.grad)
```

These are excerpts from the complete downloadable file, not standalone programs. The file includes setup, all supporting methods, strict positive-input validation for log, stable cross-entropy and both data pipelines. Two-dimensional matrix multiplication is its explicit contract; it does not promise arbitrary batched matmul, complex derivatives, GPU execution, mutation tracking, sparse tensors or higher-order differentiation. Its recursive graph traversal is adequate for these small graphs, not an unbounded-depth engine.

The engine's `cross_entropy` computes log-probabilities directly from shifted logits. Its backward uses exactly the derivative of that function. The program checks integer label shape and range, rather than accepting an accidental broadcast target.

Rebuilding this mechanism is educational. For practical large models, established frameworks provide far more primitive coverage, memory scheduling and execution support. Writing a custom operation can still be appropriate when you have a verified specialized implementation; it does not require replacing an entire framework.

### Forward mode and reverse mode answer different directional questions

For a function $f:\mathbb R^n\to\mathbb R^m$, its **Jacobian** $J$ has entry $J_{ij}=\partial f_i/\partial x_j$ and shape $m\times n$.

A **Jacobian-vector product**, $Jv$, asks: if inputs start changing in direction $v$, which way do outputs initially change? Forward-mode AD carries one such tangent alongside each forward value.

A **vector-Jacobian product**, usually written $u^\top J$ or as the column $J^\top u$, asks: how sensitive is a weighted combination of outputs to each input? Reverse mode carries the output weighting $u$ backward. Scalar-loss differentiation uses $u=1$.

Take

$$
f(x_1,x_2)=(x_1x_2,\ \sin x_1,\ x_2^2).
$$

At $x=(0.3,0.7)$,

$$
J=\begin{bmatrix}.7&.3\\\cos(.3)&0\\0&1.4\end{bmatrix}.
$$

For input direction $v=(1,2)$, $Jv=(1.3,0.955336,2.8)$. For output weighting $u=(1,-1,2)$, $J^\top u=(-0.255336,3.1)$. They have different shapes because they answer different questions.

[VisualG: an input-direction arrow and three output-change bars beside an output-weighting vector and two input-sensitivity bars. The Jacobian's rows and columns are labeled by actual variables.]

The following complete CPU program computes these products without constructing the full Jacobian:

```python
import torch

def function(x):
    return torch.stack([x[0] * x[1], torch.sin(x[0]), x[1] ** 2])

x = torch.tensor([0.3, 0.7], dtype=torch.float64)
direction = torch.tensor([1.0, 2.0], dtype=torch.float64)
output_weight = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64)
value, tangent = torch.func.jvp(function, (x,), (direction,))
value, pullback = torch.func.vjp(function, x)
sensitivity = pullback(output_weight)[0]
print(value.tolist())
print(tangent.tolist())
print(sensitivity.tolist())
print(float(output_weight @ tangent), float(sensitivity @ direction))
```

The first result is approximately(0.21,0.295520,0.49). The next two are the products above. Both final inner products equal approximately5.944664, illustrating $u^\top(Jv)=(J^\top u)^\top v$. The author calculation executed these operations in PyTorch2.14.0+cpu. [JVP API](https://docs.pytorch.org/docs/2.14/generated/torch.func.jvp.html), [VJP API](https://docs.pytorch.org/docs/2.14/generated/torch.func.vjp.html).

Obtaining an entire dense Jacobian by separate directional sweeps generally needs $n$ input-basis JVPs or $m$ output-basis VJPs. A scalar loss with many parameters therefore favors one reverse sweep over computing a full gradient by one forward direction per parameter. That is operation-count reasoning, not an exact wall-clock speedup by the number of parameters. Batching, structure, primitive costs and memory affect actual runtime. If you need only one directional derivative, one JVP can be the right request even for a large network.

JAX exposes the same ideas as `jax.jvp`, `jax.vjp`, `jax.grad`, `jax.jacfwd` and `jax.jacrev`. Its [Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) provides an alternate implementation route and full-Jacobian examples. The PyTorch program keeps this lesson runnable in one installed framework; no JAX output is claimed executed here.

### Higher derivatives without storing a Hessian

The **Hessian** of a scalar function is the matrix of its second derivatives. It describes how the gradient changes with the input. A Hessian-vector product $Hv$ often provides the needed curvature information without allocating an $n\times n$ array.

For $f(x_1,x_2)=x_1^2+x_1x_2+3x_2^2$,

$$
\nabla f=(2x_1+x_2,\ x_1+6x_2),\quad
H=\begin{bmatrix}2&1\\1&6\end{bmatrix}.
$$

At(0.3,0.7), the gradient is(1.3,4.5). For $v=(1,2)$, $Hv=(4,13)$.

```python
import torch

x = torch.tensor([0.3, 0.7], requires_grad=True, dtype=torch.float64)
v = torch.tensor([1.0, 2.0], dtype=torch.float64)
loss = x[0] ** 2 + x[0] * x[1] + 3 * x[1] ** 2
gradient = torch.autograd.grad(loss, x, create_graph=True)[0]
hessian_vector = torch.autograd.grad(gradient @ v, x)[0]
print(gradient.tolist())
print(hessian_vector.tolist())
```

The displayed values were checked by the author calculation. Differentiating `gradient.sum()` would produce $H\mathbf1$, not the Hessian diagonal. Similarly, applying a scalar-gradient transformation directly to a vector-valued gradient is not a general way to obtain that diagonal. Use an explicit Jacobian transformation to materialize a Hessian only when needed.

### Custom derivatives and deliberate surrogate gradients

A custom operation pairs a forward function with its backward rule. Here is a complete hard-sigmoid example, including a finite-difference check **away from its corners**:

```python
import torch

class HardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)

    @staticmethod
    def backward(ctx, incoming):
        (x,) = ctx.saved_tensors
        active = (x > -2.5) & (x < 2.5)
        return incoming * 0.2 * active

x = torch.tensor([-3., -2., 0., 2., 3.],
                 requires_grad=True, dtype=torch.float64)
print(torch.autograd.gradcheck(HardSigmoid.apply, (x,)))
HardSigmoid.apply(x).sum().backward()
print(x.grad.tolist())
```

Its expected gradient is(0,.2,.2,.2,0), and the check is expected to pass at those smooth points. At±2.5, the chosen backward is0 but the function has a corner; a central-difference mismatch there is not evidence for a unique true derivative. Independent replay of this complete custom-op program belongs to the finishing phase; its piecewise rule is derived here.

A **straight-through estimator** intentionally uses a surrogate backward—for example, rounding forward but pretending the local slope is1. It is not the exact derivative of rounding. Numerical differentiation of the hard forward function should not be expected to validate that surrogate. Evaluate whether the estimator serves the intended optimization problem, and label the substitution explicitly.

### Accumulating a mean loss across unequal microbatches

If memory permits only part of a batch at once, compute several forward/backward passes before one optimizer step. For a full-batch mean over $N$ independent examples, a microbatch with $n_j$ examples contributes $(n_j/N)$ times its own mean loss.

With $x=(1,2,3,4,5)$, targets $2x$, prediction $wx$ and $w=0$, the full mean-squared-error gradient is−44. Splitting into batches of2 and3 and summing their unweighted means gives−76⅔ instead. The smaller batch was given too much relative weight.

```python
import torch

x = torch.arange(1., 6., dtype=torch.float64)
targets = 2 * x
for mode in ("full", "weighted", "unweighted"):
    weight = torch.tensor(0., requires_grad=True, dtype=torch.float64)
    slices = (slice(None),) if mode == "full" else (slice(0, 2), slice(2, 5))
    for part in slices:
        errors = weight * x[part] - targets[part]
        loss = (errors ** 2).mean()
        if mode == "weighted":
            loss = loss * len(errors) / len(x)
        loss.backward()
    print(mode, weight.grad.item())
```

The author executed these cases: full−44, weighted−44 to roundoff, unweighted−76.666667. In a real loop, zero gradients once before the group and step once after it. Equality assumes an additive objective, unchanged parameters during accumulation and compatible stochastic/state behavior. Batch-dependent normalization can change the function when you change microbatch composition; correct weighting alone does not fix that. The Normalization lesson will make those dependencies visible.

### Activation checkpointing trades storage for recomputation

Reverse mode saves values that later pullbacks need. It need not save every possible intermediate forever. **Activation checkpointing** stores selected boundaries and recomputes omitted forward values when backward reaches that region.

For an ideal chain of $L$ similarly sized layers split into $K$ segments, a simple storage model is proportional to $K+L/K$: retained boundaries plus one segment's intermediates. Choosing $K$ near $\sqrt L$ gives an $O(\sqrt L)$ activation-storage model. Real graphs have unequal tensors, branches, parameters, optimizer state and temporary buffers; this formula is not a total-memory prediction.

[VisualH: an eight-operation timeline showing retained boundary states, discarded intermediates and one regenerated segment during backward. Label recomputation as work, not free reuse.]

A complete deterministic comparison:

```python
import torch
from torch.utils.checkpoint import checkpoint

def block(x):
    return torch.tanh(x @ x.T / x.shape[0])

x = torch.tensor([[0.2, -0.4], [0.7, 0.1]],
                 requires_grad=True, dtype=torch.float64)
ordinary = block(x).sum()
gradient_a = torch.autograd.grad(ordinary, x)[0]
recomputed = checkpoint(block, x, use_reentrant=False).sum()
gradient_b = torch.autograd.grad(recomputed, x)[0]
print(torch.allclose(gradient_a, gradient_b, atol=1e-12, rtol=1e-12))
```

The expected result isTrue for this deterministic block; independent execution of this excerpt is deferred. There is no measured speed or memory claim. Recomputed code must represent the same function: changed global state, device moves or uncontrolled randomness can invalidate equivalence. [PyTorch checkpoint documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html) describes these conditions and its implementations.

At larger scale, activation storage is only one cost. Parameter/gradient/optimizer-state sharding and pipeline scheduling address different bottlenecks; communication compression adds another approximation decision. Their dedicated distributed-training and GPU lessons own those mechanisms. Backprop supplies the derivative dependencies those systems must preserve. No universal rule says activations dominate every model, that checkpointing costs a fixed percentage, or that compressed gradients cause a fixed accuracy tradeoff.

## 7. Practice with changed graphs and failure cases

### 1. Two uses, one parameter

For $u=x^2$, $L=u-0.5u$ at $x=2$, compute the forward values and derivative. Explain both places where reverse contributions add.

<details><summary>Hint</summary>
First collect the two contributions at $u$; then apply the two operand slots in $x\cdot x$.
</details>
<details><summary>Solution</summary>
$u=4,L=2,\bar u=1-.5=.5$. Each multiply input slot contributes $.5(2)=1$, so $\bar x=2$. Visiting $x$ once while dropping one operand contribution incorrectly gives1.
</details>

### 2. Undo a different broadcast

A bias with shape(2,) was added to two rows. Incoming sensitivities are[[-1,0],[2,4]]. What is the bias gradient? If a mean over all four outputs produced those sensitivities, should you divide by4 again?

<details><summary>Hint</summary>
The incoming sensitivities already include all downstream operations.
</details>
<details><summary>Solution</summary>
The gradient is(1,4), a sum down each column. Do not divide again: if the mean supplied a factor1/4, it is already present in the incoming array. Double normalization changes the derivative.
</details>

### 3. One calculation, a dangerous update

For the two-example line fit in investigationA, derive gradients−2 and−1 from the mean-squared-error definition. Explain why the learning-rate1 update increases loss without implying a bad backward rule.

<details><summary>Hint</summary>
The two residuals initially are0 and−1. A mean over two entries cancels the derivative's factor2.
</details>
<details><summary>Solution</summary>
$\partial L/\partial w=(0)(1)+(-1)(2)=-2$ and $\partial L/\partial b=0-1=-1$. A simultaneous step gives(3,1), predictions(4,7), residuals(3,4), mean loss12.5. The gradient is local and the step is too large for this quadratic; finite changes require evaluating the new loss.
</details>

### 4. Interpret two failed gradient checks

One check of ReLU at0 reports AD0 and central difference.5. Another check of $10^{12}+x$ at1 with $h=10^{-5}$ reports AD1 and finite difference0. Are these the same kind of failure? State a useful next action for each.

<details><summary>Hint</summary>
Ask first whether an ordinary derivative exists, then whether the two evaluated values are numerically distinct.
</details>
<details><summary>Solution</summary>
ReLU has a corner: test smooth points on both sides and document the chosen boundary convention. The offset linear function is smooth, but rounding destroys the tiny difference: inspect the actual paired values, change perturbation/scale or remove the irrelevant offset in a separate diagnostic. Neither mismatch alone proves that the implemented local rule is wrong.
</details>

### 5. An input-direction product

For the vector function in §6 at(0.3,0.7), choose $v=(0,1)$. Compute $Jv$, and explain why it is not the gradient of a scalar loss.

<details><summary>Hint</summary>
The direction selects the second Jacobian column.
</details>
<details><summary>Solution</summary>
$Jv=(.3,0,1.4)$. It lists the three output rates when only the second input changes. A scalar-loss gradient would instead require choosing a scalar combination of those outputs and would have two input coordinates.
</details>

### 6. A sum of gradient entries is not a diagonal

For $f=x_1^2+x_1x_2+3x_2^2$, compute the derivative of the sum of gradient entries and compare it with the Hessian diagonal.

<details><summary>Hint</summary>
The summed gradient is $3x_1+7x_2$.
</details>
<details><summary>Solution</summary>
Its derivative is(3,7), equal to $H(1,1)$ for this symmetric Hessian. The diagonal is(2,6). Off-diagonal terms contribute to the row sums and cannot be discarded.
</details>

### 7. Repair an accumulation loop

Suppose a mean loss covers ten examples and microbatches have sizes4,4,2. Give the three weights multiplying the microbatch means. State two conditions under which even these weights do not reproduce one full-batch update.

<details><summary>Hint</summary>
Weight by fraction of examples, and keep the underlying function fixed.
</details>
<details><summary>Solution</summary>
Weights are.4,.4,.2. Updating parameters after each microbatch changes the evaluation state; batch-dependent normalization can change the function. Randomness can also differ unless comparisons control its realization. Small floating-point summation differences can remain even when the mathematical gradients agree.
</details>

### 8. Extend one primitive and expose a wrong rule

Add a sigmoid operation to the teaching engine. Derive its local pullback, check both positive and negative inputs against PyTorch, and check a reused sigmoid output. Then deliberately replace its derivative with a constant and identify a fixture that detects the error. Keep the complete file's reset and broadcast contracts.

<details><summary>Hint</summary>
Use a stable sigmoid forward calculation and cache its result. The pullback multiplies by $\sigma(x)(1-\sigma(x))$. A point far from zero distinguishes this from a constant slope.
</details>
<details><summary>Solution and evaluation criteria</summary>
A correct operation returns the pointwise sigmoid and adds incoming×output×(1−output) to the parent's buffer. At0 its slope is.25; at2 it is about.104994, so a constant.25 rule is exposed. Reusing the result in $s+3s$ multiplies the accumulated input derivative by4. Report actual inputs, dtype, absolute error and a perturbation sweep for a smooth case; the grade depends on these checks and correct graph behavior, not on whether a training curve happens to decrease. No unexecuted extension result is supplied as a measured output.
</details>

## 8. References and alternative routes

- [3Blue1Brown: What is backpropagation really doing?](https://www.3blue1brown.com/lessons/backpropagation/) — creator-hosted video and text companion. The reviewed introductory text explains how desired output changes propagate into hidden contributions; use after the first graph if the algebra feels detached. Video playback was not independently reviewed.
- [Stanford CS231n: Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) — lecture notes for the local-gate view and chain rule. The compound-expression section is a useful alternative to our shared-square example. At nonsmooth ties, use an explicitly stated convention rather than interpreting informal max rules as unique derivatives.
- [Baydin et al.: Automatic Differentiation in Machine Learning](https://jmlr.org/papers/v18/17-468.html) — broad survey of numerical/symbolic distinctions, forward/reverse modes, applications and implementation approaches. Read §§2–3 first; source transformation and research directions are later branches.
- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html), [gradcheck](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.gradcheck.gradcheck.html) and [checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html) — contracts for the APIs and edge cases used here.
- [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) — another programming route to JVPs, VJPs, Jacobians and Hessians; optional if you want to connect the same mathematics to a second framework.
- [Fundamentals of Numerical Computation: Convergence of finite differences](https://fncbook.com/python/fd-converge/) — a deeper explanation of why a perturbation sweep is more informative than one fixed epsilon.

The next topic is **Loss Functions: CE, MSE, Focal, Contrastive & Triplet**. You now know how to differentiate an objective; next, decide what the objective asks the model to get right.
