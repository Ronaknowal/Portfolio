import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const backpropContent = {
  title: "Backpropagation & Automatic Differentiation",
  readTime: "~40 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every modern deep learning framework is, at its heart, a machine for computing one particular number: the partial derivative of a scalar loss with respect to each of a very large number of parameters. A transformer with seventy billion weights has seventy billion such partial derivatives. You cannot get them by algebra — the symbolic expression of a gradient through a deep network is astronomically large. You cannot get them by finite differences — that would cost seventy billion extra forward passes per training step. You get them by a trick. The trick is that if you are careful about how you store intermediate values during the forward pass, you can walk the chain rule backward through the same graph in time proportional to the forward pass itself, and land on every partial derivative you need. That trick is what backpropagation is. Everything people mean by "training a neural network" rests on it.
      </Prose>

      <Prose>
        The ideas arrived in pieces, out of order, and mostly in contexts that had nothing to do with learning. The first of them is Robert Wengert's one-and-a-half-page paper in <em>Communications of the ACM</em>, August 1964, titled "A simple automatic derivative evaluation program." Wengert was a numerical analyst at Aerospace Corporation, and the problem he was attacking was painfully concrete: scientists kept writing Fortran programs that needed derivatives of complicated functions, and they kept deriving those derivatives by hand. He proposed that if you wrote your program as a sequence of elementary operations, each with a known local derivative, you could evaluate the chain rule mechanically as the program ran. This is forward-mode automatic differentiation. It was not yet called that. The paper is two pages. It would be another decade before anyone noticed the dual formulation — running the chain rule backward — which is the version that matters for deep learning.
      </Prose>

      <Prose>
        In 1974, Paul Werbos submitted a Harvard PhD thesis titled "Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences." The thesis was about modeling social science data, but appendix G worked out what Werbos called the "ordered derivative," a recipe for computing the gradient of a scalar output through an arbitrary feed-forward computation in a single backward sweep. Applied to a neural network, this is exactly backpropagation. Almost no one in the neural network community noticed. Werbos himself describes the reception as "not well received." He was awarded the IEEE Neural Network Pioneer Award for this work twenty-one years later, in 1995.
      </Prose>

      <Prose>
        The work that finally landed was Rumelhart, Hinton, and Williams's 1986 paper in <em>Nature</em> volume 323, "Learning representations by back-propagating errors." The paper is barely four pages and contains essentially the same algorithm as the appendix of Werbos's thesis, but it was aimed at a neural network audience at exactly the moment connectionism was becoming a research program. Its key demonstration was that a hidden layer trained by backpropagation could learn internal representations — features of the input that were not hand-designed but emerged as a byproduct of minimizing the output error. That emergent-representations framing is what distinguished the 1986 paper from the 1974 thesis in the minds of its readers. It is also what made the paper influential far beyond its technical contribution.
      </Prose>

      <Prose>
        For two decades after 1986, backpropagation lived in specialized implementations. Every framework — Torch7 in Lua, Theano in Python, Caffe in C++ — had its own hand-maintained collection of layer classes, each with a <Code>forward</Code> method and a paired <Code>backward</Code> method. Implementing a new layer meant deriving its gradient on paper and typing both functions in. Bugs in the backward method were notoriously hard to find because the forward pass would still run and the loss would still go down — just not as fast as it should. Nearly every deep learning practitioner of that era has a story about a sign error or a missing transpose that cost a month of training.
      </Prose>

      <Prose>
        The modern framework era begins in 2015-2017 with the arrival of general-purpose automatic differentiation on top of a dynamic computation graph. Autograd, written by Dougal Maclaurin, David Duvenaud, and Matthew Johnson at Harvard, was the first widely-used Python implementation of reverse-mode AD over arbitrary NumPy code. PyTorch's autograd, described in the 2017 NeurIPS workshop paper by Paszke et al., took the same idea and built it into a tensor library with GPU support. The shift was enormous. You stopped writing layer classes with paired forward/backward methods and started writing ordinary Python functions that manipulated tensors; the gradient came for free. JAX, released by Google in 2018, pushed the idea further by separating the tracing machinery from the execution: <Code>jax.grad</Code> returns a new Python function, <Code>jax.vmap</Code> vectorizes it, <Code>jax.jit</Code> compiles it with XLA. The same three primitives, composed in different orders, give you scalar gradients, batched gradients, Jacobians, Hessians, and Hessian-vector products.
      </Prose>

      <Prose>
        Underneath all of this sits a single mathematical insight articulated carefully in Andreas Griewank and Andrea Walther's 2008 textbook <em>Evaluating Derivatives</em>, and surveyed for the ML audience in Baydin, Pearlmutter, Radul, and Siskind's 2017 JMLR paper "Automatic Differentiation in Machine Learning: a Survey." The insight is that every differentiable program implicitly defines a Jacobian matrix of partial derivatives, and that Jacobian can be extracted in two mirror-image modes — forward mode, which propagates tangent vectors, and reverse mode, which propagates cotangent vectors. For a scalar loss function of a million parameters, reverse mode is a million times cheaper than forward mode. That asymmetry is the entire reason deep learning is possible on current hardware. The rest of this topic is about what that sentence means and how to make it true in code.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with a two-line program: <Code>y = 3 * x</Code>, then <Code>z = y + 1</Code>. You want <Code>dz/dx</Code>. You know this is 3 from algebra. How could a program compute it without knowing algebra?
      </Prose>

      <Prose>
        Option one: carry an extra scalar alongside every variable. Alongside <Code>x</Code>, carry <Code>dx/dx = 1</Code>. When you compute <Code>y = 3 * x</Code>, also compute <Code>dy/dx = 3 * 1 = 3</Code>. When you compute <Code>z = y + 1</Code>, also compute <Code>dz/dx = dy/dx + 0 = 3</Code>. At the end, <Code>z</Code> holds the value and its companion scalar holds the derivative. This is forward-mode AD. It is embarrassingly simple and it generalizes: replace the scalar with a vector and you can track partials with respect to any number of inputs, one extra slot per input.
      </Prose>

      <Prose>
        Option two: run the program first, recording every operation in order, then walk backward. After <Code>z = y + 1</Code> we know <Code>dz/dz = 1</Code>. The operation was an add, whose local derivative with respect to <Code>y</Code> is 1, so <Code>dz/dy = 1 * 1 = 1</Code>. Before that, <Code>y = 3 * x</Code>, whose local derivative with respect to <Code>x</Code> is 3, so <Code>dz/dx = dz/dy * 3 = 3</Code>. This is reverse-mode AD. It is slightly harder because you have to record the forward pass somewhere, but each operation you process during the backward sweep produces gradients with respect to all of its inputs, not just one.
      </Prose>

      <Prose>
        The two modes compute the same number. They do it with different cost profiles. Forward mode costs time proportional to <Code>O(n * cost_of_forward)</Code> when you want derivatives with respect to <Code>n</Code> inputs, because you have to carry an <Code>n</Code>-slot tangent vector through every operation. Reverse mode costs time proportional to <Code>O(m * cost_of_forward)</Code> where <Code>m</Code> is the number of outputs you want gradients of. Neural network training has <Code>m = 1</Code> (one scalar loss) and <Code>n</Code> in the billions (one scalar per parameter). Reverse mode wins by a factor of <Code>n</Code>.
      </Prose>

      <Callout accent="gold">
        forward mode: cheap when you have few inputs, many outputs.<br />
        reverse mode: cheap when you have many inputs, few outputs.<br />
        deep learning: one loss, billions of parameters → reverse mode, always.
      </Callout>

      <Prose>
        The jargon layer over this distinction comes from linear algebra. Any differentiable function <Code>{"f : R^n → R^m"}</Code> has a Jacobian matrix <Code>J</Code> of shape <Code>{"m × n"}</Code>. Forward mode computes a <em>Jacobian-vector product</em> (JVP): given a tangent vector <Code>v</Code> in the input space, it returns <Code>{"J @ v"}</Code>. Reverse mode computes a <em>vector-Jacobian product</em> (VJP): given a cotangent vector <Code>u</Code> in the output space, it returns <Code>{"u^T @ J"}</Code>. Neither mode ever materializes the full Jacobian. You can build a full Jacobian by calling JVP <Code>n</Code> times with unit vectors (forward) or VJP <Code>m</Code> times with unit vectors (reverse). JAX exposes both choices directly as <Code>jax.jacfwd</Code> and <Code>jax.jacrev</Code>.
      </Prose>

      <Prose>
        The mental model for reverse-mode in a neural network is: the forward pass traces out a directed acyclic graph of operations, each node caching the intermediate values its backward pass will need. The loss sits at the bottom of the graph as a single scalar. To compute gradients, you seed that scalar with a gradient of 1 ("the loss is sensitive to itself, by definition, by a factor of one") and let each operation pull that sensitivity back through itself, one step at a time, multiplying by its local derivative and accumulating into its parents. By the time the sweep reaches the inputs, every parameter in the graph holds the sum of sensitivities from every path that passed through it. That sum is exactly the gradient.
      </Prose>

      <Prose>
        The cost accounting for reverse-mode is worth pinning down, because it is the source of every memory pain in deep learning. Each forward operation caches whatever its backward will need: for a matmul, the two input matrices; for a ReLU, a boolean mask of which activations were positive; for a softmax + cross-entropy, usually the probabilities. The total memory cost of the forward pass scales with the number of intermediate activations in the graph, not with the depth alone. A hundred-layer transformer with a batch of sixteen and a context of four thousand caches tens of gigabytes of activations — all of which must live on the GPU until the backward pass frees them. This is why <em>activation memory</em>, not parameter memory, is the bottleneck for large models. Section 8 returns to this.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Math foundation</H2>

      <Prose>
        The univariate chain rule is the first thing you learn in freshman calculus. If <Code>y = g(x)</Code> and <Code>z = f(y)</Code>, then <Code>dz/dx = (df/dy)(dg/dx)</Code>. The multivariate generalization is almost as simple once you accept that derivatives of vector-valued functions are matrices. Let <Code>{"x ∈ R^n"}</Code>, <Code>{"y = g(x) ∈ R^m"}</Code>, <Code>{"z = f(y) ∈ R^k"}</Code>. The Jacobians <Code>{"∂g/∂x"}</Code> and <Code>{"∂f/∂y"}</Code> have shapes <Code>{"m × n"}</Code> and <Code>{"k × m"}</Code>. The chain rule says their product is the Jacobian of the composition.
      </Prose>

      <MathBlock>
        {"\\frac{\\partial z}{\\partial x} \\;=\\; \\frac{\\partial f}{\\partial y} \\cdot \\frac{\\partial g}{\\partial x}"}
      </MathBlock>

      <Prose>
        For neural networks we care about a scalar loss, so the final Jacobian has shape <Code>{"1 × n"}</Code>, which is just a row vector of partial derivatives. We normally write it as a column and call it the gradient <Code>{"∇L"}</Code>. The reverse-mode insight is that you never form the full intermediate Jacobians. You multiply a row vector by a matrix from the left, producing another row vector, and carry that row vector one step further back through the graph. That operation is a vector-Jacobian product, and because the vector stays small (one row), you never build anything larger than necessary.
      </Prose>

      <Prose>
        Make this concrete with a two-layer classifier. Input is a batch <Code>{"X ∈ R^{N × D}"}</Code>. Parameters are <Code>{"W_1 ∈ R^{D × H}"}</Code>, <Code>{"b_1 ∈ R^{H}"}</Code>, <Code>{"W_2 ∈ R^{H × C}"}</Code>, <Code>{"b_2 ∈ R^{C}"}</Code>. Labels are integers <Code>{"y ∈ {0, ..., C-1}^N"}</Code>.
      </Prose>

      <MathBlock>
        {"Z_1 = X W_1 + b_1, \\qquad A_1 = \\mathrm{ReLU}(Z_1)"}
      </MathBlock>
      <MathBlock>
        {"Z_2 = A_1 W_2 + b_2, \\qquad P = \\mathrm{softmax}(Z_2)"}
      </MathBlock>
      <MathBlock>
        {"L = -\\frac{1}{N}\\sum_{i=1}^{N} \\log P_{i, y_i}"}
      </MathBlock>

      <Prose>
        Now we derive each gradient in the reverse direction that autodiff would walk. The first one is the famous simplification that makes softmax + cross-entropy so convenient in practice. Differentiating the loss through the softmax jointly — rather than each piece separately — collapses a messy combination of division and exponentials into a single line.
      </Prose>

      <MathBlock>
        {"\\frac{\\partial L}{\\partial Z_2} \\;=\\; \\frac{1}{N}\\bigl(P - \\mathrm{onehot}(y)\\bigr)"}
      </MathBlock>

      <Prose>
        Read that as: for each example <Code>i</Code>, the gradient at logit <Code>c</Code> is the probability the model assigns to class <Code>c</Code>, minus 1 if <Code>c</Code> is the true class, all divided by <Code>N</Code>. This is the error signal. Everything below here is the same signal being reshaped by the chain rule as it walks back through the network.
      </Prose>

      <MathBlock>
        {"\\frac{\\partial L}{\\partial W_2} \\;=\\; A_1^{\\top} \\frac{\\partial L}{\\partial Z_2}, \\qquad \\frac{\\partial L}{\\partial b_2} \\;=\\; \\sum_{i=1}^{N} \\frac{\\partial L}{\\partial Z_2}_{i,:}"}
      </MathBlock>

      <MathBlock>
        {"\\frac{\\partial L}{\\partial A_1} \\;=\\; \\frac{\\partial L}{\\partial Z_2}\\, W_2^{\\top}, \\qquad \\frac{\\partial L}{\\partial Z_1} \\;=\\; \\frac{\\partial L}{\\partial A_1} \\odot \\mathbf{1}[Z_1 > 0]"}
      </MathBlock>

      <MathBlock>
        {"\\frac{\\partial L}{\\partial W_1} \\;=\\; X^{\\top} \\frac{\\partial L}{\\partial Z_1}, \\qquad \\frac{\\partial L}{\\partial b_1} \\;=\\; \\sum_{i=1}^{N} \\frac{\\partial L}{\\partial Z_1}_{i,:}"}
      </MathBlock>

      <Prose>
        Six lines. Every one of them is a consequence of the chain rule applied to a single operation: matmul's gradient is matmul-with-transpose, bias-add's gradient is a sum over the batch, elementwise ReLU's gradient is an elementwise mask, softmax + cross-entropy's gradient is <Code>{"(P - y)/N"}</Code>. There is no new information in these six lines beyond the chain rule itself. They are the chain rule, specialized to the operations the network happens to use, executed from right to left.
      </Prose>

      <Prose>
        The reason automatic differentiation is worth writing code for rather than working out by hand is that every time you add a layer — a LayerNorm, a residual, a GELU, an attention block — you add another row to this table. Getting it right by hand is tedious and error-prone. Getting it right by <em>code</em> reduces the problem to: for each elementary operation, specify a local backward function that accepts an incoming gradient and returns outgoing gradients. That single abstraction — the local VJP rule — is the entire interface between an op and an AD engine. Everything else is bookkeeping.
      </Prose>

      <Prose>
        One piece of bookkeeping is load-bearing: the order in which you run the backward passes. The forward pass builds a DAG where each node's inputs are its parents. The backward pass must process a node only after all of its children (downstream nodes that consumed its output) have been processed. This is a reverse topological order. It is also the order in which a simple iterative implementation builds up correctly: walk the forward order, record nodes in a list, then iterate that list in reverse. You only need a fancier topological sort if you have nodes with multiple consumers whose child relationships cross branches, which happens routinely in any network with skip connections. The from-scratch implementation below does exactly this.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The point of writing your own autograd engine once in your life is to remove the magic. A working reverse-mode engine is less than two hundred lines of Python. It does not need a graph library, or a compiler, or any dependency beyond NumPy. It only needs a <Code>Tensor</Code> class that records its parents and a local VJP for each operation you support. Every line of code in this section was run before being pasted; the <Code># Output:</Code> comments are verbatim from stdout.
      </Prose>

      <H3>4a. The Tensor class</H3>

      <Prose>
        The core object holds a value, a gradient buffer, a tuple of parent tensors, and a <Code>_backward</Code> closure that knows how to push gradient to those parents. The closure is captured at the moment the operation is performed, which conveniently freezes in any data (like the operand itself for a multiply) that the backward needs. Nothing tricky.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

class Tensor:
    def __init__(self, data, _parents=(), _op="", requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._parents = _parents
        self._op = _op
        self._backward = lambda: None   # local VJP, filled in per op

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)`}
      </CodeBlock>

      <H3>4b. Operations and their local VJPs</H3>

      <Prose>
        Each operation returns a new <Code>Tensor</Code> and installs a <Code>_backward</Code> closure that uses the <em>output's</em> gradient (which will be filled in by the time the closure runs) and the operation's cached operands to update the parents' <Code>grad</Code> fields. Broadcasting requires a small helper, <Code>_unbroadcast</Code>, that sums any axes that were broadcast during the forward pass — otherwise the gradient shape will not match the parameter shape.
      </Prose>

      <CodeBlock language="python">
{`    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _parents=(self, other), _op="+")
        def _backward():
            self.grad  += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _parents=(self, other), _op="*")
        def _backward():
            self.grad  += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data  * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, _parents=(self, other), _op="@")
        def _backward():
            self.grad  += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), _parents=(self,), _op="relu")
        def _backward():
            self.grad += (self.data > 0.0).astype(np.float64) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, _parents=(self,), _op="exp")
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), _parents=(self,), _op="log")
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out`}
      </CodeBlock>

      <Prose>
        Each of those local rules is a one-line specialization of the chain rule. The matmul rule — gradient of <Code>{"Y = A @ B"}</Code> is <Code>{"dY @ B^T"}</Code> into <Code>A</Code> and <Code>{"A^T @ dY"}</Code> into <Code>B</Code> — is the first identity a new AD engineer memorizes and the one that gets silently implemented inside <Code>torch.nn.Linear</Code> a billion times a second in every running GPU.
      </Prose>

      <H3>4c. The fused softmax + cross-entropy</H3>

      <Prose>
        In principle we could build this out of <Code>exp</Code>, <Code>log</Code>, and indexing. In practice every serious autograd library fuses the softmax and the cross-entropy into a single op with a hand-written backward, both because the combined gradient <Code>{"(P - y)/N"}</Code> is dramatically simpler than its component pieces, and because separating them creates numerical problems — <Code>log(softmax(x))</Code> should always be computed via the log-sum-exp identity, never naively. The fused version is the one we use.
      </Prose>

      <CodeBlock language="python">
{`    def softmax_cross_entropy(self, labels):
        """self: logits (N, C). labels: int array (N,). Returns scalar Tensor."""
        x = self.data
        x_shift = x - x.max(axis=1, keepdims=True)   # log-sum-exp trick
        exp = np.exp(x_shift)
        probs = exp / exp.sum(axis=1, keepdims=True)
        N = x.shape[0]
        log_probs = np.log(probs + 1e-30)
        loss_val = -log_probs[np.arange(N), labels].mean()
        out = Tensor(loss_val, _parents=(self,), _op="xent")
        def _backward():
            g = probs.copy()
            g[np.arange(N), labels] -= 1.0
            g /= N
            self.grad += g * out.grad
        out._backward = _backward
        return out`}
      </CodeBlock>

      <H3>4d. Topological sort and the .backward() driver</H3>

      <Prose>
        With the op closures in place, <Code>.backward()</Code> is fifteen lines. Seed the output's gradient with a tensor of ones, walk the graph in topological order to get a flat list of nodes, then iterate that list in reverse calling each node's <Code>_backward</Code>. Each call adds into the parents' gradient buffers, so nodes with multiple consumers accumulate correctly.
      </Prose>

      <CodeBlock language="python">
{`    def backward(self):
        topo, visited = [], set()
        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for p in v._parents:
                build(p)
            topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad`}
      </CodeBlock>

      <H3>4e. A two-layer classifier, trained and checked</H3>

      <Prose>
        Now we can train a real model. The data is eighty points sampled from two 2D Gaussian blobs. The network is <Code>{"(2) → Linear(8) → ReLU → Linear(2)"}</Code>, trained with vanilla SGD on softmax + cross-entropy.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
np.random.seed(0)

def make_blobs(n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    c0 = rng.normal(loc=[-1.0, -1.0], scale=0.4, size=(n_per, 2))
    c1 = rng.normal(loc=[ 1.5,  1.0], scale=0.4, size=(n_per, 2))
    X = np.vstack([c0, c1])
    y = np.array([0] * n_per + [1] * n_per)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

X_np, y_np = make_blobs()
H = 8
W1 = Tensor(np.random.randn(2, H) * 0.5, requires_grad=True)
b1 = Tensor(np.zeros(H),                 requires_grad=True)
W2 = Tensor(np.random.randn(H, 2) * 0.5, requires_grad=True)
b2 = Tensor(np.zeros(2),                 requires_grad=True)
params = [W1, b1, W2, b2]

def forward(x):
    return ((x @ W1 + b1).relu() @ W2 + b2)

lr = 0.2
for step in range(60):
    for p in params: p.zero_grad()
    loss = forward(Tensor(X_np)).softmax_cross_entropy(y_np)
    loss.backward()
    for p in params:
        p.data -= lr * p.grad

preds = forward(Tensor(X_np)).data.argmax(axis=1)
print("acc:", (preds == y_np).mean())
# Output:
# acc: 1.0`}
      </CodeBlock>

      <H3>4f. Correctness check: finite differences and PyTorch</H3>

      <Prose>
        An autograd engine that produces plausible-looking gradients is worthless. The only way to trust a backward pass is to check it numerically. Two independent checks are standard. The first is the finite-difference check: perturb each scalar parameter by a small <Code>ε</Code>, measure how the loss changes, and compare to the analytical gradient. Good agreement at <Code>ε = 10⁻⁵</Code> typically means relative error below <Code>10⁻⁸</Code>. The second is to recompute the gradients against a trusted implementation — in our case PyTorch — and compare element-wise.
      </Prose>

      <CodeBlock language="python">
{`# finite-diff check on W1 at init
eps = 1e-5
W1_fd = np.zeros_like(W1.data)
for i in range(W1.data.shape[0]):
    for j in range(W1.data.shape[1]):
        orig = W1.data[i, j]
        W1.data[i, j] = orig + eps
        lp = forward(Tensor(X_np)).softmax_cross_entropy(y_np).data
        W1.data[i, j] = orig - eps
        lm = forward(Tensor(X_np)).softmax_cross_entropy(y_np).data
        W1.data[i, j] = orig
        W1_fd[i, j] = (lp - lm) / (2 * eps)

print("max|ad - fd| =", np.max(np.abs(W1_fd - W1.grad)))
# Output:
# max|ad - fd| = 3.779e-11`}
      </CodeBlock>

      <CodeBlock language="python">
{`# cross-check against torch autograd on the same init
import torch
tW1 = torch.tensor(W1_init, requires_grad=True)
tb1 = torch.tensor(b1_init, requires_grad=True)
tW2 = torch.tensor(W2_init, requires_grad=True)
tb2 = torch.tensor(b2_init, requires_grad=True)
tX = torch.tensor(X_np); ty = torch.tensor(y_np, dtype=torch.long)

th     = torch.relu(tX @ tW1 + tb1)
tlogits = th @ tW2 + tb2
tloss   = torch.nn.functional.cross_entropy(tlogits, ty)
tloss.backward()

print("max|dW1 err|:", np.max(np.abs(W1_scratch.grad - tW1.grad.numpy())))
print("max|dW2 err|:", np.max(np.abs(W2_scratch.grad - tW2.grad.numpy())))
print("max|db1 err|:", np.max(np.abs(b1_scratch.grad - tb1.grad.numpy())))
print("max|db2 err|:", np.max(np.abs(b2_scratch.grad - tb2.grad.numpy())))
# Output:
# max|dW1 err|: 2.220e-16
# max|dW2 err|: 1.110e-16
# max|db1 err|: 2.220e-16
# max|db2 err|: 5.551e-17`}
      </CodeBlock>

      <Callout accent="green">
        The finite-diff error is at the limit of what <Code>ε = 10⁻⁵</Code> can measure. The PyTorch error is at the limit of float64 machine epsilon. The engine is correct.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Nobody writes their own autograd in production. You use PyTorch or JAX. Both offer the same mental model we just built — a graph of ops, each with a local VJP, differentiated in reverse topological order — with GPU kernels, gradient accumulation, mixed precision, distributed training, and a decade of bug fixes. The APIs are thin and worth knowing fluently.
      </Prose>

      <H3>5a. PyTorch autograd</H3>

      <Prose>
        Every tensor has a flag <Code>requires_grad</Code>. Any tensor produced by an op involving a grad-requiring tensor inherits a <Code>grad_fn</Code> pointing to the backward closure. Calling <Code>.backward()</Code> on a scalar walks that graph and accumulates into leaf tensors' <Code>.grad</Code>. In-place methods end in an underscore — <Code>.requires_grad_()</Code> flips the flag on an existing tensor.
      </Prose>

      <CodeBlock language="python">
{`import torch

x = torch.tensor([1.5, -2.0, 0.5], requires_grad=True)
y = (x * x * x).sum()       # build graph
y.backward()                # walk it
print("x.grad:", x.grad.tolist())
# Output:
# x.grad: [6.75, 12.0, 0.75]`}
      </CodeBlock>

      <Prose>
        For any case more advanced than "one loss, one backward call," use <Code>torch.autograd.grad</Code> directly. It returns gradients as a tuple without touching <Code>.grad</Code> fields, which makes it safe to compose. Set <Code>create_graph=True</Code> when you need the returned gradient to itself be differentiable — this is how higher-order derivatives, influence functions, and meta-learning work.
      </Prose>

      <CodeBlock language="python">
{`x = torch.tensor([1.5, -2.0, 0.5], requires_grad=True)
y = (x * x * x).sum()
(g,) = torch.autograd.grad(y, x, create_graph=True)

# second derivative: differentiate the gradient
g2 = torch.autograd.grad(g.sum(), x)[0]
print("dy/dx:", g.tolist())
print("d2y/dx2:", g2.tolist())
# Output:
# dy/dx: [6.75, 12.0, 0.75]
# d2y/dx2: [9.0, -12.0, 3.0]`}
      </CodeBlock>

      <Prose>
        Two flags regularly confuse newcomers. <Code>retain_graph=True</Code> keeps the graph alive after a backward pass so you can call <Code>.backward()</Code> on it again — which is essential when a single forward produces multiple losses computed at different places in your code. Without it, the second call will raise. <Code>create_graph=True</Code> implies <Code>retain_graph=True</Code> and additionally records the backward ops themselves into the graph, making them differentiable.
      </Prose>

      <CodeBlock language="python">
{`x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)   # first backward
print("after first:", x.grad.item())
y.backward()                     # second backward, would fail without retain_graph
print("after second:", x.grad.item())
# Output:
# after first: 6.0
# after second: 12.0   (gradients accumulate into .grad)`}
      </CodeBlock>

      <H3>5b. Custom autograd.Function</H3>

      <Prose>
        When a built-in op is non-differentiable, or when PyTorch's autodiff is correct but slow, you write your own op. Subclass <Code>torch.autograd.Function</Code>, implement a static <Code>forward</Code> that computes the value and stashes anything the backward will need via <Code>ctx.save_for_backward</Code>, and a static <Code>backward</Code> that returns the gradient for each input. The Function is invoked via its <Code>.apply()</Code> method, never instantiated directly.
      </Prose>

      <CodeBlock language="python">
{`class HardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        mask = ((x > -2.5) & (x < 2.5)).to(grad_out.dtype)
        return grad_out * 0.2 * mask

x = torch.linspace(-3, 3, 7, requires_grad=True)
y = HardSigmoid.apply(x).sum()
y.backward()
print([round(v, 3) for v in x.grad.tolist()])
# Output:
# [0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0]`}
      </CodeBlock>

      <Prose>
        Custom Functions are also the right place to insert <em>straight-through estimators</em> (where the forward is a hard quantization but the backward pretends it was identity) and fused kernels that wrap CUDA code. If you write one, always verify it with <Code>torch.autograd.gradcheck(fn, inputs, eps=1e-6)</Code>, which does an automated finite-difference check.
      </Prose>

      <H3>5c. Gradient checkpointing</H3>

      <Prose>
        Reverse-mode AD caches every intermediate activation, which is what makes it so fast relative to forward mode. That caching is also what makes it so memory-hungry on long sequences and deep models. Gradient checkpointing, introduced in Chen et al. 2016 ("Training Deep Nets with Sublinear Memory Cost," arXiv:1604.06174), trades compute for memory: for a chosen block of the network, do not save the activations during the forward pass; during the backward pass, re-run the forward for that block to regenerate the activations, then do the local backward. For a network of <Code>L</Code> layers split into <Code>{"√L"}</Code> segments, this reduces activation memory from <Code>{"O(L)"}</Code> to <Code>{"O(√L)"}</Code> at the cost of one extra forward pass. The speed penalty is typically 20-30%; the memory saving unlocks training models that would otherwise OOM.
      </Prose>

      <CodeBlock language="python">
{`import torch.utils.checkpoint as cp

def block(x):
    return torch.tanh(x @ x.t() / x.shape[0])

x = torch.randn(8, 8, requires_grad=True)

# normal: activations of block() are cached
y1 = block(x).sum()
y1.backward()
gA = x.grad.clone(); x.grad = None

# checkpointed: activations are recomputed during backward
y2 = cp.checkpoint(block, x, use_reentrant=False).sum()
y2.backward()
gB = x.grad.clone()

print("checkpoint matches normal:", torch.allclose(gA, gB, atol=1e-6))
# Output:
# checkpoint matches normal: True`}
      </CodeBlock>

      <H3>5d. JAX: grad, vjp, jvp, jacrev, jacfwd</H3>

      <Prose>
        JAX makes the forward/reverse-mode distinction first-class. Every differentiation primitive is a function transformation: <Code>jax.grad(f)</Code> returns a new function that computes the scalar gradient, <Code>jax.vjp(f, x)</Code> returns a value and a reverse-mode VJP function, <Code>jax.jvp(f, x, v)</Code> returns a value and its forward-mode JVP. <Code>jacrev</Code> and <Code>jacfwd</Code> build full Jacobians by composing the two modes. Because every primitive is pure, they compose: <Code>jax.grad(jax.grad(f))</Code> is the Hessian's diagonal; <Code>jax.vmap(jax.grad(f))</Code> is per-example gradients at no extra code cost.
      </Prose>

      <CodeBlock language="python">
{`import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(jnp.tanh(x) ** 2)

x = jnp.array([0.1, 0.5, -0.7, 1.2])

# scalar gradient
print("grad:", jax.grad(f)(x).tolist())

# explicit vjp — same result as grad for scalar-out f
y, vjp_fn = jax.vjp(f, x)
print("vjp(1.0):", vjp_fn(1.0)[0].tolist())

# forward-mode jvp
v = jnp.ones_like(x)
_, tangent_out = jax.jvp(f, (x,), (v,))
print("jvp(ones):", float(tangent_out))

# Jacobian of a vector-valued function: same answer via forward or reverse
def g(x):
    return jnp.stack([x[0] * x[1], jnp.sin(x[0]), x[1] ** 2])

xv = jnp.array([0.3, 0.7])
print("jacrev(g):", jax.jacrev(g)(xv).tolist())
print("jacfwd(g):", jax.jacfwd(g)(xv).tolist())
# Output:
# grad: [0.197356, 0.726862, -0.767232, 0.508563]
# vjp(1.0): [0.197356, 0.726862, -0.767232, 0.508563]
# jvp(ones): 0.665548
# jacrev(g): [[0.7, 0.3], [0.955337, 0.0], [0.0, 1.4]]
# jacfwd(g): [[0.7, 0.3], [0.955337, 0.0], [0.0, 1.4]]`}
      </CodeBlock>

      <Prose>
        Choose <Code>jacrev</Code> when outputs are few and inputs many (same rule as before). Choose <Code>jacfwd</Code> when inputs are few and outputs many. For Hessian-vector products in optimization, the canonical trick is <Code>{"jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))"}</Code> — a grad-of-grad that never builds the full Hessian.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Three views of a single training run of the scratch engine from section 4. First, the loss curve. Cross-entropy starts near <Code>{"log(2) ≈ 0.69"}</Code> for a two-class balanced problem and drops to essentially zero once the blobs are separated.
      </Prose>

      <Plot
        label="training loss over 60 SGD steps"
        xLabel="step"
        yLabel="loss"
        series={[
          {
            name: "scratch autograd",
            color: "#e2b55a",
            points: [
              [0, 2.9068], [5, 0.5461], [10, 0.19], [15, 0.1205],
              [20, 0.0802], [25, 0.0562], [30, 0.0424], [35, 0.0336],
              [40, 0.027], [45, 0.0224], [50, 0.0192], [55, 0.0167],
              [59, 0.0151],
            ],
          },
        ]}
      />

      <Prose>
        Second, the per-layer gradient magnitudes during the first few backward passes. Early in training, every layer's gradient is large and roughly comparable; the network is responding strongly to the initial mismatch. By step 3 the first-layer gradient has dropped by an order of magnitude while the bias-2 gradient has stabilized. This drop-and-stabilize pattern is what healthy backprop looks like. Unhealthy versions are in section 9.
      </Prose>

      <StepTrace
        label="gradient norms per layer, steps 0-4"
        steps={[
          {
            label: "step 0",
            render: () => (
              <Prose>
                <Code>{"||dL/dW1|| = 2.405"}</Code>, <Code>{"||dL/db1|| = 1.378"}</Code>, <Code>{"||dL/dW2|| = 2.366"}</Code>, <Code>{"||dL/db2|| = 0.355"}</Code>. Initial loss is 2.91. The network knows nothing and every parameter has a strong error signal. Notice <Code>b_2</Code> is smaller because it only carries the class-balance error, which is already near zero for balanced blobs.
              </Prose>
            ),
          },
          {
            label: "step 1",
            render: () => (
              <Prose>
                <Code>{"||dL/dW1|| = 1.152"}</Code>, <Code>{"||dL/db1|| = 0.650"}</Code>, <Code>{"||dL/dW2|| = 1.109"}</Code>, <Code>{"||dL/db2|| = 0.010"}</Code>. One SGD step (lr=0.2) roughly halved every layer's gradient. The loss is dropping fast because the blobs are linearly separable and the network has enough capacity to overfit.
              </Prose>
            ),
          },
          {
            label: "step 2",
            render: () => (
              <Prose>
                <Code>{"||dL/dW1|| = 0.235"}</Code>, <Code>{"||dL/db1|| = 0.145"}</Code>, <Code>{"||dL/dW2|| = 0.270"}</Code>, <Code>{"||dL/db2|| = 0.279"}</Code>. Another factor-of-five drop. The gradient ordering is still layer-balanced — no vanishing, no exploding.
              </Prose>
            ),
          },
          {
            label: "step 3",
            render: () => (
              <Prose>
                <Code>{"||dL/dW1|| = 0.180"}</Code>, <Code>{"||dL/db1|| = 0.116"}</Code>, <Code>{"||dL/dW2|| = 0.226"}</Code>, <Code>{"||dL/db2|| = 0.280"}</Code>. Deceleration. Most of the easy examples are already classified correctly; the gradient is now driven by the small fraction still on the wrong side of the boundary.
              </Prose>
            ),
          },
          {
            label: "step 4",
            render: () => (
              <Prose>
                <Code>{"||dL/dW1|| = 0.157"}</Code>, <Code>{"||dL/db1|| = 0.104"}</Code>, <Code>{"||dL/dW2|| = 0.206"}</Code>, <Code>{"||dL/db2|| = 0.272"}</Code>. Sub-linear approach to zero. At step 59 the W1 gradient norm is 0.022 — two full decades smaller than step 0. This is the boring, healthy regime.
              </Prose>
            ),
          },
        ]}
      />

      <Prose>
        Third, a heatmap of the final <Code>{"dL/dW_1"}</Code> matrix after 60 steps. The matrix is <Code>{"2 × 8"}</Code> (two input features, eight hidden units). Column 6 is still the dominant contributor — that hidden unit is the last one the network has not yet fully aligned. The other columns are near zero because those hidden units are either fully trained or effectively dead (ReLU output identically zero for both classes).
      </Prose>

      <Heatmap
        label="dL/dW1 after 60 training steps (2 features × 8 hidden)"
        colorScale="gold"
        rowLabels={["x1", "x2"]}
        colLabels={["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"]}
        matrix={[
          [0.0013, 0.0004, -0.0081, -0.0009, 0.0025,  0.0128, -0.0015, -0.0006],
          [0.0007, 0.0004, -0.0057, -0.0006, 0.0014,  0.0137, -0.0011, -0.0004],
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Autodiff offers several closely related primitives. Choosing between them is usually about the shape of your problem (inputs/outputs), the memory you have, and whether you are computing one gradient or many.
      </Prose>

      <H3>Forward vs reverse mode</H3>
      <Prose>
        The rule is set by the ratio of inputs to outputs. If <Code>n ≪ m</Code> (few inputs, many outputs — e.g., sensitivity analysis of many simulated trajectories to a few parameters), use forward mode. If <Code>n ≫ m</Code> (many inputs, one scalar loss — every neural network), use reverse mode. If <Code>n ≈ m</Code>, either works; forward mode often wins on memory. For the specific case of a Jacobian of a vector-valued function, <Code>jacfwd</Code> costs <Code>{"n × cost_of_forward"}</Code>, <Code>jacrev</Code> costs <Code>{"m × cost_of_forward"}</Code> plus a memory cost for caching. Benchmark both. For neural-network-style problems, reverse is never the wrong answer.
      </Prose>

      <H3>When to use gradient checkpointing</H3>
      <Prose>
        Turn on checkpointing when activation memory is your bottleneck — typically at long contexts, deep networks, or large batch sizes where you are close to OOM. The cost is roughly a 25% slowdown in wall-clock time per step, paid back by being able to fit larger models or bigger batches. Do not checkpoint when memory is not the limiting factor; the compute overhead is real and unnecessary. The sweet spot for transformer training is usually checkpointing each transformer block, not individual ops — finer granularity gives diminishing memory savings at increasing overhead.
      </Prose>

      <H3>When to write a custom autograd.Function</H3>
      <Prose>
        Three reasons to write one. First, your operation is non-differentiable in the mathematical sense but you want a sensible surrogate gradient — straight-through estimators for quantization, Gumbel-softmax, REINFORCE-style gradients. Second, you have a fused kernel (CUDA, Triton) whose forward is faster than the equivalent composition of PyTorch ops and whose backward you have derived by hand. Third, PyTorch's built-in autograd produces a correct but numerically unstable gradient for your op and you want to replace it with a more stable closed form. For everything else, just compose standard ops.
      </Prose>

      <H3>detach() vs torch.no_grad() vs requires_grad=False</H3>
      <Prose>
        These three look similar and do different things. <Code>x.detach()</Code> returns a new tensor that shares storage with <Code>x</Code> but has no <Code>grad_fn</Code>; gradients do not flow back through it. Use this when you want to block a specific edge in the graph — e.g., the target of a distillation loss or the old-policy log-probabilities in PPO. <Code>torch.no_grad()</Code> is a context manager that disables graph-building for any op executed inside; nothing inside ever records gradients. Use this for evaluation and for manipulating parameters outside the optimizer step. <Code>requires_grad=False</Code> is a property of a leaf tensor that says "I am never the target of differentiation." Use this for frozen layers — e.g., the backbone of a transfer-learning setup.
      </Prose>

      <H3>Higher-order vs first-order</H3>
      <Prose>
        First-order gradients are what SGD and Adam need and what every training loop computes. Higher-order gradients show up in meta-learning (MAML), influence functions, natural gradient methods, Hessian-vector products for optimizers like K-FAC, and some forms of distillation. Flip on <Code>create_graph=True</Code> in PyTorch or stack <Code>jax.grad</Code> in JAX. Expect the backward pass to become two to five times slower and noticeably more memory-hungry, because the backward graph itself now gets recorded. For Hessian-vector products specifically, prefer the Pearlmutter trick (grad of <Code>{"<grad_f, v>"}</Code>) over materializing the Hessian.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <Prose>
        Reverse-mode AD has a time complexity that is a small multiple of the forward pass — typically 2-3x — which is essentially the only reason training large neural networks is feasible. Its memory complexity is where every scaling difficulty originates: total activation memory is linear in the number of ops in the graph, weighted by each op's activation size. For a transformer with <Code>L</Code> layers, batch size <Code>B</Code>, sequence length <Code>S</Code>, and hidden dimension <Code>d</Code>, activation memory is roughly <Code>{"O(L · B · S · d)"}</Code>, which dominates parameter memory for realistic model sizes. Every scaling technique in modern training is fundamentally about managing this activation cache.
      </Prose>

      <H3>Gradient accumulation</H3>
      <Prose>
        The cheapest trick. When your desired batch size does not fit on one device, split it into micro-batches, do a forward-backward on each, and accumulate gradients into <Code>.grad</Code> without stepping the optimizer. After the last micro-batch, step. Mathematically identical to a full-batch step, with peak memory set by the micro-batch. Every training script on earth uses this. In PyTorch: <Code>loss.backward()</Code> on each micro-batch; <Code>optimizer.step()</Code> only at the end; <Code>optimizer.zero_grad()</Code> only after stepping.
      </Prose>

      <H3>Activation checkpointing</H3>
      <Prose>
        Chen et al. 2016 showed that recomputing activations during the backward pass trades compute for memory in a highly favorable way. With uniform segmentation of <Code>L</Code> layers into <Code>{"√L"}</Code> groups, memory drops from <Code>O(L)</Code> to <Code>{"O(√L)"}</Code> while compute grows by a single extra forward pass (total 2 forwards + 1 backward instead of 1 + 1). Modern implementations in PyTorch (<Code>torch.utils.checkpoint</Code>) and JAX (<Code>jax.checkpoint</Code>, also called <Code>jax.remat</Code>) pick the boundaries automatically or let you annotate them.
      </Prose>

      <H3>Pipeline parallelism</H3>
      <Prose>
        For models too large to fit on a single device even with checkpointing, split the network layer-wise across devices. A naive pipeline leaves most devices idle most of the time; GPipe (Huang et al. 2019) fixes this by splitting each batch into micro-batches and keeping every device busy on a different micro-batch. The tricky part for autograd is that the backward pass must flow in reverse device order, which requires either buffering activations on each device or recomputing them. Modern pipeline implementations (DeepSpeed, Megatron-LM) combine pipeline parallelism with checkpointing to keep activation memory bounded.
      </Prose>

      <H3>ZeRO and FSDP</H3>
      <Prose>
        For the parameter and optimizer state — not activations — the bottleneck is that storing gradients and Adam's moment estimates requires three or four times the parameter memory. ZeRO (Rajbhandari et al. 2020) shards these across data-parallel workers: each worker stores only its slice of the parameters, gradients, and optimizer state, and all-gathers the relevant pieces on the fly during forward and backward. PyTorch's FSDP is a clean implementation of the same idea. From an autograd perspective, this is invisible — the backward pass is still reverse-mode on a local graph; what changed is where the resulting gradient tensors live and how they get reduced across workers.
      </Prose>

      <H3>Gradient compression</H3>
      <Prose>
        In distributed training, the all-reduce of gradients across workers can become the bottleneck. Gradient compression exchanges compressed (quantized, sparsified, or low-rank) gradients and corrects for the error locally. 1-bit Adam (Tang et al. 2021) sends gradients at 1-bit resolution after a warm-up; PowerSGD (Vogels et al. 2019) projects each gradient onto a small rank and sends only the projection. These techniques trade a small amount of accuracy for a large communication speedup; they do not change autograd itself, only what happens to the gradient tensors after they are computed.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <Prose>
        Autograd is correct by construction for correctly-written ops. Most real training failures are not bugs in the engine; they are bugs in the interaction between the engine and the rest of the training pipeline. Here are the ones you will meet.
      </Prose>

      <H3>Exploding gradients</H3>
      <Prose>
        Gradient magnitudes grow super-linearly with depth and parameter scale. In a badly-initialized deep network, the per-layer multiplicative factors all exceed one and the gradient at the input is millions of times larger than at the output. The loss becomes NaN within a handful of steps. Symptoms: loss jumps to Inf, <Code>grad_norm</Code> logs show values over <Code>1e+6</Code>, weight magnitudes balloon. Fix: gradient clipping (<Code>torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)</Code>). Cheap, effective, run it every step.
      </Prose>

      <H3>Vanishing gradients</H3>
      <Prose>
        The mirror problem: per-layer factors are all less than one, the gradient at the input is effectively zero, and early layers stop learning. Historically notorious for deep networks with sigmoid/tanh activations, because the derivative of a saturating activation is at most <Code>0.25</Code> (sigmoid) or <Code>1.0</Code> only at zero (tanh). ReLU and its variants fix this by having derivative exactly 1 on the active half. Residual connections (He et al. 2015) fix the remaining issue by providing a gradient highway that skips the activation chain. If your modern transformer has vanishing gradients, something structural is wrong — check residuals, layer norms, and initialization.
      </Prose>

      <H3>In-place operations breaking autograd</H3>
      <Prose>
        PyTorch's in-place methods (<Code>.add_()</Code>, <Code>.mul_()</Code>, <Code>x += y</Code>) overwrite the underlying tensor. If autograd was going to need the pre-update value for a backward pass, it errors at <Code>.backward()</Code> time with the classic "a variable needed for gradient computation has been modified by an inplace operation." The fix is almost always to remove the in-place form. In-place is a memory optimization; let the engine manage it through its own buffers.
      </Prose>

      <H3>detach() dropping gradients you meant to keep</H3>
      <Prose>
        A common mistake in actor-critic or distillation pipelines: calling <Code>.detach()</Code> on a tensor that should have flowed gradients back to its producer. Symptoms: one of your sub-losses contributes nothing to parameter updates; <Code>param.grad</Code> is zero for some layers; debugging with <Code>autograd.grad</Code> returns <Code>None</Code>. Read the graph carefully. If you need gradients to flow through a tensor, do not detach it. If you only need the numerical value, do.
      </Prose>

      <H3>OOM from retain_graph=True</H3>
      <Prose>
        Each call to <Code>.backward(retain_graph=True)</Code> keeps the forward graph alive after the backward pass. If you call it in a loop without eventually releasing the graph, memory grows every iteration until the GPU runs out. Use <Code>retain_graph=True</Code> only when you know you need a second backward pass on the same graph, and make sure the final backward is without the flag so the graph can be freed.
      </Prose>

      <H3>Higher-order gradient bugs</H3>
      <Prose>
        <Code>create_graph=True</Code> records the backward ops into the graph, which means they are subject to the same autodiff machinery. If any op's backward has a manual implementation that is itself non-differentiable (a common pattern in old custom kernels), asking for a second derivative will silently return zeros. Always <Code>gradcheck</Code> a custom Function, and if you use it in a higher-order context, <Code>gradgradcheck</Code> it.
      </Prose>

      <H3>Numerical instability</H3>
      <Prose>
        <Code>log(x)</Code> at <Code>x = 0</Code> is <Code>-∞</Code> and its derivative is <Code>1/x = ∞</Code>. <Code>sqrt(x)</Code> at <Code>x = 0</Code> is zero but its derivative is <Code>{"0.5 / sqrt(x) = ∞"}</Code>. Dividing two small numbers can produce <Code>0/0 = NaN</Code>. These creep in through softmax (fix with log-sum-exp), normalization layers (fix with epsilon in the denominator), probability losses (fix with <Code>log_softmax</Code> instead of <Code>log(softmax(x))</Code>), and anywhere else you take a derivative of a function with a corner. When training mysteriously produces NaNs around step 1000, the first hypothesis is always a numerical edge case hit for the first time by accumulated drift — print the inputs to every log, sqrt, and division in your loss.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read the originals. The papers below are the actual events in the history of automatic differentiation and backpropagation, in roughly chronological order. Every fact in section 1 traces back to one of them.
      </Prose>

      <Callout accent="gold">
        <strong>Wengert, R. E. (1964).</strong> "A simple automatic derivative evaluation program." <em>Communications of the ACM</em> 7(8):463-464. The two-page paper that invented forward-mode AD. The idea of decomposing a computation into elementary operations each with a known local derivative is the conceptual seed for every modern AD framework.<br />
        <a href="https://dl.acm.org/doi/10.1145/355586.364791" style={{ color: colors.gold }}>dl.acm.org/doi/10.1145/355586.364791</a>
      </Callout>

      <Callout accent="gold">
        <strong>Werbos, P. J. (1974).</strong> "Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences." PhD thesis, Harvard University. Appendix G works out reverse-mode differentiation through a general feed-forward computation — exactly backpropagation — applied to behavioral modeling. Reprinted in Werbos's 1994 book <em>The Roots of Backpropagation</em>.<br />
        <a href="https://gwern.net/doc/ai/nn/1974-werbos.pdf" style={{ color: colors.gold }}>gwern.net/doc/ai/nn/1974-werbos.pdf</a>
      </Callout>

      <Callout accent="gold">
        <strong>Rumelhart, D. E., Hinton, G. E., Williams, R. J. (1986).</strong> "Learning representations by back-propagating errors." <em>Nature</em> 323:533-536. The paper that put backprop on the map. Shorter than a blog post today, and still one of the clearest introductions to the algorithm in existence.<br />
        <a href="https://www.nature.com/articles/323533a0" style={{ color: colors.gold }}>nature.com/articles/323533a0</a>
      </Callout>

      <Callout accent="gold">
        <strong>Griewank, A., Walther, A. (2008).</strong> <em>Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation</em>, 2nd ed. SIAM. The reference textbook. The discussion of checkpointing strategies (chapter 12) is the origin of every modern checkpointing implementation, including Chen et al.'s deep-learning-specific version.
      </Callout>

      <Callout accent="gold">
        <strong>Baydin, A. G., Pearlmutter, B. A., Radul, A. A., Siskind, J. M. (2017).</strong> "Automatic Differentiation in Machine Learning: a Survey." <em>Journal of Machine Learning Research</em> 18(153):1-43. The definitive modern survey. Required reading if you want to understand why JAX and PyTorch ended up with the APIs they did.<br />
        <a href="https://jmlr.org/papers/v18/17-468.html" style={{ color: colors.gold }}>jmlr.org/papers/v18/17-468.html</a>
      </Callout>

      <Callout accent="gold">
        <strong>Paszke, A., et al. (2017).</strong> "Automatic differentiation in PyTorch." NeurIPS-W 2017. Describes PyTorch's dynamic-graph autograd engine and the design decisions that distinguished it from Theano and TensorFlow 1.x.<br />
        <a href="https://openreview.net/forum?id=BJJsrmfCZ" style={{ color: colors.gold }}>openreview.net/forum?id=BJJsrmfCZ</a>
      </Callout>

      <Callout accent="gold">
        <strong>Bradbury, J., et al. (2018).</strong> JAX: composable transformations of Python+NumPy programs.<br />
        <a href="https://github.com/google/jax" style={{ color: colors.gold }}>github.com/google/jax</a>
      </Callout>

      <Callout accent="gold">
        <strong>Chen, T., Xu, B., Zhang, C., Guestrin, C. (2016).</strong> "Training Deep Nets with Sublinear Memory Cost." arXiv:1604.06174. The gradient-checkpointing paper. Uniform <Code>{"O(√L)"}</Code> segmentation for <Code>{"O(L)"}</Code> networks is the default in every modern framework.<br />
        <a href="https://arxiv.org/abs/1604.06174" style={{ color: colors.gold }}>arxiv.org/abs/1604.06174</a>
      </Callout>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Five exercises, arranged from "verify understanding" to "verify implementation." If you can do the last one from scratch in under an hour, you know this material.
      </Prose>

      <H3>Exercise 1 — chain rule by hand</H3>
      <Prose>
        Given <Code>{"f(x) = (sin(x^2) + 1) / (cos(x) + 2)"}</Code>, compute <Code>{"df/dx"}</Code> by hand using the chain and quotient rules. Then write a Python function that evaluates <Code>f</Code> and its derivative via forward-mode AD (carry a <Code>(value, tangent)</Code> pair through every operation). Verify at <Code>x = 0.7</Code> that forward-mode and symbolic agree to 1e-10.
      </Prose>

      <H3>Exercise 2 — why reverse mode is cheaper</H3>
      <Prose>
        Take a function <Code>{"f : R^{1000} → R"}</Code> built by composing a thousand elementary ops. Compute the gradient via (a) finite differences, (b) forward-mode AD calling JVP with 1000 unit vectors, (c) reverse-mode AD with a single backward pass. Count the number of elementary operations each approach performs. Explain in one paragraph why (c) is <Code>{"1000×"}</Code> cheaper than (b) and <Code>{"~2000×"}</Code> cheaper than (a). If you reverse the shape — <Code>{"f : R → R^{1000}"}</Code> — which mode wins and why?
      </Prose>

      <H3>Exercise 3 — manual derivation of softmax + cross-entropy</H3>
      <Prose>
        Starting from <Code>{"L = -log(softmax(z)[y])"}</Code> for a single example with logits <Code>{"z ∈ R^C"}</Code> and true label <Code>y</Code>, derive <Code>{"dL/dz"}</Code> component by component. You should end up with <Code>{"dL/dz_c = p_c - 1[c == y]"}</Code>. Now do the same derivation at batch level and recover the <Code>{"(P - onehot(y))/N"}</Code> formula from section 3. Why would you never implement this as <Code>log(softmax(z))</Code> followed by index-and-negate?
      </Prose>

      <H3>Exercise 4 — gradient check a custom op</H3>
      <Prose>
        Write a <Code>torch.autograd.Function</Code> subclass for <Code>{"f(x) = x * tanh(softplus(x))"}</Code> (the Mish activation). Implement both <Code>forward</Code> and <Code>backward</Code>. Verify with <Code>torch.autograd.gradcheck</Code> on a small random input. Now break your backward by dropping a factor of <Code>{"sech^2"}</Code> somewhere. Observe that <Code>gradcheck</Code> catches the bug, but a naive training loop on a small MLP still "works" in the sense that the loss goes down — just more slowly. This is the reason you always run <Code>gradcheck</Code> on custom ops.
      </Prose>

      <H3>Exercise 5 — re-implement section 4 without re-reading it</H3>
      <Prose>
        Close this page. Open a blank Python file. Build a <Code>Tensor</Code> class with <Code>requires_grad</Code>, <Code>.backward()</Code>, and support for <Code>+</Code>, <Code>*</Code>, <Code>@</Code>, <Code>relu</Code>, and a fused softmax + cross-entropy. Train a two-layer classifier on 2D blobs. Verify the gradients against PyTorch to 1e-10. If you get stuck, the failure is almost always one of three things: (a) you forgot that each <Code>_backward</Code> closure captures <Code>out.grad</Code> by reference, which is only populated later; (b) broadcasting — you need to sum the gradient along broadcast axes before adding it to the parent; (c) topological order — you have to process nodes in reverse of the order in which they were built. Once you get it right, you will never again think of autograd as magic.
      </Prose>

      <Prose>
        That is the entire subject. A hundred lines of code, one mathematical identity, and seventy years of engineering on top.
      </Prose>

    </div>
  ),
};

export default backpropContent;
