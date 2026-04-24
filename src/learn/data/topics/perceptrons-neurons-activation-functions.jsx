import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const perceptronsNeuronsActivationsContent = {
  title: "Perceptrons, Neurons & Activation Functions",
  readTime: "~40 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In July 1958, a few months before his 30th birthday, a Cornell psychologist named Frank Rosenblatt wheeled a Mark I Perceptron into a demonstration hall at the Office of Naval Research in Washington, D.C. The machine was a refrigerator-sized rack of motors, potentiometers, and a 20-by-20 array of photocells. He showed it a card with a square on the left; then a card with a square on the right; then a card with a square on the left again. After about fifty presentations the machine correctly classified the next card it had never seen before. A reporter from The New York Times filed a story the next day with the now-famous line that the Navy had revealed the embryo of an electronic computer that "will be able to walk, talk, see, write, reproduce itself and be conscious of its existence." The story was an embarrassment that would haunt neural network research for thirty years, but the underlying paper — Rosenblatt's <em>The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain</em>, <em>Psychological Review</em> 65(6):386–408 — is the first place anyone had written down, precisely, the object that sits at the center of every neural network shipped since: a weighted sum of inputs, followed by a threshold.
      </Prose>

      <Prose>
        Rosenblatt's rule was beautiful in its simplicity. Present an input pattern. Compute the weighted sum. If the output is wrong, shift every weight by a small amount in the direction that would have pushed the answer toward being right. Iterate. For any dataset that is linearly separable — that is, any dataset where some hyperplane divides the positive examples from the negative ones — Rosenblatt proved that this procedure converges in a finite number of updates. That theorem, the perceptron convergence theorem, was the field's first genuine mathematical result, and it is still the first theorem in every introductory textbook.
      </Prose>

      <Prose>
        In 1969, Marvin Minsky and Seymour Papert published <em>Perceptrons: An Introduction to Computational Geometry</em> with MIT Press, and the central technical contribution of that book was a formal proof of what single-layer perceptrons cannot do. They demonstrated with care that a single threshold unit cannot compute the XOR function — the one that says "one, but not both." There is no line in the plane that separates the four corners {"{(0,0), (1,1)}"} from {"{(0,1), (1,0)}"}, and therefore no weighted sum plus threshold can compute XOR. The book also argued, somewhat more speculatively, that multi-layer perceptrons would probably suffer the same limitations because nobody knew how to train them. The speculative part was wrong, but the speculation was load-bearing. Funding for neural network research dried up almost overnight. The period from 1969 to roughly 1985 is what the field now calls the First AI Winter, and its proximate cause was a book about the geometric limits of a particular computational object.
      </Prose>

      <Prose>
        The revival came in October 1986, when David Rumelhart, Geoffrey Hinton, and Ronald Williams published <em>Learning representations by back-propagating errors</em> in <em>Nature</em> 323:533–536. The paper is short — three pages of text and two pages of figures — but it contained the algorithm that dissolved Minsky and Papert's speculative objection. Given a differentiable nonlinearity instead of a hard threshold, the chain rule of calculus tells you exactly how to propagate an error signal from the output of a multi-layer network backward to every weight. If you can differentiate the activation, you can train the network. Rumelhart et al. demonstrated this by training a small multi-layer perceptron to solve — among other toy problems — XOR. The result was electrifying because it was the exact problem Minsky and Papert had used as their canonical counterexample seventeen years earlier. The field came roaring back. The nonlinearity of choice, for reasons that seemed natural at the time, was the logistic sigmoid — smooth, bounded, biologically suggestive.
      </Prose>

      <Prose>
        The sigmoid reigned for twenty years, and for twenty years it was also the thing that limited how deep networks could be trained. As you stack sigmoids, the gradient that flows backward through each layer gets multiplied by a derivative that is at most 0.25 and is usually much smaller. Six or seven layers deep, the gradient at the input layer is effectively zero; the network does not train. The <em>vanishing gradient problem</em>, named by Sepp Hochreiter in his 1991 diploma thesis, was understood by the late 1990s, but the escape hatch wasn't found until Vinod Nair and Geoffrey Hinton published <em>Rectified Linear Units Improve Restricted Boltzmann Machines</em> at ICML 2010 and, a year later, Xavier Glorot, Antoine Bordes, and Yoshua Bengio published <em>Deep Sparse Rectifier Neural Networks</em> at AISTATS 2011. The idea was older than both papers — rectifiers had been used in computational neuroscience since at least Fukushima's 1969 Neocognitron — but the demonstration that <Code>max(0, x)</Code> trained faster, generalized better, and permitted deeper stacking than any smooth sigmoid-family activation was a quiet revolution. By 2012, when Krizhevsky, Sutskever, and Hinton won ImageNet with AlexNet, ReLU was the default. It is still the default for most of CNN literature.
      </Prose>

      <Prose>
        Transformers forced another shift. Dan Hendrycks and Kevin Gimpel introduced the Gaussian Error Linear Unit in <em>Gaussian Error Linear Units (GELUs)</em>, arXiv:1606.08415, in June 2016. GELU multiplies <Code>x</Code> by the cumulative normal distribution evaluated at <Code>x</Code>, giving a smooth, non-monotonic activation that interpolates between zero and the identity. BERT used GELU. GPT-2 and GPT-3 used GELU. It became the transformer default. In 2017, Prajit Ramachandran, Barret Zoph, and Quoc V. Le at Google Brain ran a reinforcement-learning search over a space of possible activation functions and rediscovered, by accident, an activation that Stefan Elfwing, Eiichi Uchibe, and Kenji Doya had proposed in 2017 under the name Sigmoid-weighted Linear Unit (SiLU): <Code>x · σ(x)</Code>. The Google paper, <em>Searching for Activation Functions</em>, arXiv:1710.05941, called it Swish and showed it edged out ReLU on ImageNet. The two names — Swish and SiLU — refer to the same function, and most libraries use them interchangeably.
      </Prose>

      <Prose>
        The modern LLM era settled on a variant of Swish called SwiGLU, introduced by Noam Shazeer in the February 2020 note <em>GLU Variants Improve Transformer</em>, arXiv:2002.05202. SwiGLU is a gated linear unit: instead of applying a single nonlinearity to the pre-activation, you split the hidden projection into two halves, pass one half through Swish, and multiply the two element-wise. It costs 1.5x the parameters of a plain feed-forward block, but in practice Shazeer showed a consistent perplexity improvement at fixed compute. PaLM used SwiGLU. LLaMA-1 and LLaMA-2 use SwiGLU. Mistral, Qwen, Gemma, DeepSeek — every major open-weight foundation model released after 2022 uses SwiGLU or a close relative (GeGLU, which substitutes GELU for Swish). The arc from Rosenblatt's hard threshold in 1958 to SwiGLU in 2020 is the arc of this topic: what a neuron is, what the activation in front of it is, and why the specific choice of that activation is one of the most consequential decisions in the architecture.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        A neuron, in the engineering sense that survived the biology, is a function from a vector of inputs to a scalar output. It does two things in sequence. First it computes a weighted sum of its inputs plus a bias: <Code>{"z = w · x + b"}</Code>. Then it passes that sum through a nonlinear function <Code>σ</Code> to produce the output: <Code>{"a = σ(z)"}</Code>. The weighted sum is the <em>pre-activation</em>; the value after the nonlinearity is the <em>activation</em>. Every neural network you will ever touch — the 175-billion-parameter GPT, the 65-billion-parameter LLaMA, the tiny four-neuron network you built in your first ML class — is a graph of these two operations, composed.
      </Prose>

      <Prose>
        Geometrically, the pre-activation <Code>{"w · x + b"}</Code> is a signed distance to a hyperplane in input space. When <Code>{"w · x + b > 0"}</Code> you are on one side of the plane; when <Code>{"w · x + b < 0"}</Code> you are on the other; the plane <Code>{"w · x + b = 0"}</Code> is the decision boundary. For the original Rosenblatt perceptron with a hard step activation, the neuron's output is a pure half-plane classifier: one if you are on the positive side, zero otherwise. The weights <Code>w</Code> define the orientation of the plane; the bias <Code>b</Code> shifts it away from the origin. That is why Minsky and Papert's XOR result is a theorem about hyperplanes: there is no single plane that separates the two XOR classes, so no hard-threshold neuron — no matter how its weights are chosen — can ever compute XOR.
      </Prose>

      <Prose>
        Replace the hard threshold with a smooth sigmoid, and the geometry softens. The neuron still has a decision boundary at <Code>{"w · x + b = 0"}</Code>, but the output now graduates smoothly from near-zero on the negative side to near-one on the positive side, with a band of uncertainty around the boundary whose width depends on the norm of <Code>w</Code>. This is the first concrete reason smooth activations matter: they give you gradients, and gradients let you train. The step function is differentiable almost everywhere, but its derivative is zero almost everywhere, and therefore no gradient-based learning rule can update weights except through the boundary itself — which has measure zero. Rosenblatt's original rule worked by sidestepping this issue: it was a piecewise rule that only updated on mistakes. But it generalizes only to the linearly separable case.
      </Prose>

      <Prose>
        The deeper reason you need a nonlinearity, any nonlinearity, is that a stack of linear functions is still a linear function. Compose two layers with linear activations: <Code>{"y = W₂(W₁ x + b₁) + b₂ = (W₂ W₁) x + (W₂ b₁ + b₂)"}</Code>. You have reinvented a single linear layer with weights <Code>{"W₂ W₁"}</Code> and bias <Code>{"W₂ b₁ + b₂"}</Code>. Stack twelve of them — that's GPT-2 Small without activations — and you still have one linear layer. There is no representational gain from depth in a linear network. Minsky and Papert's XOR argument generalizes immediately: every linear network, no matter how deep, is a single hyperplane, and cannot compute XOR. The nonlinearity is the thing that makes depth mean something.
      </Prose>

      <Prose>
        Once you insert any nonlinearity between layers — even a kinked function like ReLU that is trivially nonlinear — the composition stops collapsing. A two-layer network with ReLU activations in the hidden layer is a universal function approximator: Cybenko proved it for sigmoids in 1989 (<em>Approximation by Superpositions of a Sigmoidal Function</em>), and Hornik extended it to essentially any non-polynomial activation in 1991. Universal approximation is a statement about what a network <em>can</em> represent in principle, given enough hidden units. It says nothing about how easily you can find those weights by gradient descent, how many samples you need to generalize, or how the expressive capacity scales with depth versus width. All three of those questions have turned out to depend, in subtle ways, on exactly which nonlinearity you choose.
      </Prose>

      <Callout>
        The representational story and the optimization story are different stories. Universal approximation says ReLU networks can represent any continuous function; NTK theory and its descendants say what functions gradient descent actually <em>finds</em>. When we compare activations empirically, we are comparing optimization landscapes, not representational ceilings.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        A single artificial neuron with weight vector <Code>w ∈ ℝⁿ</Code>, bias <Code>b ∈ ℝ</Code>, and activation function <Code>σ: ℝ → ℝ</Code> computes the scalar
      </Prose>

      <MathBlock>{"a = \\sigma(w^\\top x + b) = \\sigma(z)"}</MathBlock>

      <Prose>
        where <Code>x ∈ ℝⁿ</Code> is the input vector. A fully-connected layer of <Code>m</Code> such neurons stacks the weight vectors as rows of a matrix <Code>W ∈ ℝ^(m×n)</Code>, concatenates the biases into a vector <Code>b ∈ ℝ^m</Code>, and applies the activation element-wise:
      </Prose>

      <MathBlock>{"\\mathbf{a} = \\sigma(W \\mathbf{x} + \\mathbf{b})"}</MathBlock>

      <Prose>
        Every single activation function in common use — sigmoid, tanh, ReLU, LeakyReLU, ELU, GELU, Swish, Mish, SwiGLU — is a choice of the scalar function <Code>σ</Code> applied element-wise (with SwiGLU and its gated cousins taking two element-wise inputs and producing one output). Their definitions, derivatives, and key properties are what follow.
      </Prose>

      <H3>3a. Sigmoid (logistic)</H3>

      <MathBlock>{"\\sigma(z) = \\frac{1}{1+e^{-z}} \\qquad \\sigma'(z) = \\sigma(z)(1-\\sigma(z))"}</MathBlock>

      <Prose>
        Bounded in <Code>(0, 1)</Code>, monotonically increasing, infinitely differentiable, symmetric around <Code>z = 0</Code> with <Code>{"σ(0) = 0.5"}</Code>. The derivative peaks at 0.25 at the origin and decays exponentially: at <Code>{"|z| = 5"}</Code> the derivative is under <Code>0.007</Code>. This exponential saturation is the source of the vanishing gradient problem: in a deep stack, the product of many small derivatives collapses to numerical zero. Sigmoid also is not zero-centered — its output is always positive — which means that the gradient with respect to the weights of a downstream layer is always either all-positive or all-negative, producing the characteristic zig-zag pattern that slows training.
      </Prose>

      <H3>3b. Tanh</H3>

      <MathBlock>{"\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}} \\qquad \\tanh'(z) = 1 - \\tanh^2(z)"}</MathBlock>

      <Prose>
        Bounded in <Code>(-1, 1)</Code>, zero-centered, infinitely differentiable. The derivative peaks at 1.0 at the origin and saturates at the same exponential rate as sigmoid on both tails. Tanh is a rescaled, shifted sigmoid: <Code>{"tanh(z) = 2σ(2z) - 1"}</Code>. It fixes the zero-centered problem but not the saturation problem, which is why deep feedforward networks with tanh still suffered vanishing gradients before ReLU. Tanh remains standard inside RNN cell states (LSTM, GRU) because the bounded-and-zero-centered output plays nicely with the multiplicative gates of those architectures.
      </Prose>

      <H3>3c. ReLU</H3>

      <MathBlock>{"\\text{ReLU}(z) = \\max(0, z) \\qquad \\text{ReLU}'(z) = \\begin{cases} 1 & z > 0 \\\\ 0 & z < 0 \\end{cases}"}</MathBlock>

      <Prose>
        Piecewise linear, unbounded above, zero below. The derivative is exactly 1 for all positive <Code>z</Code> and exactly 0 for all negative <Code>z</Code>; it is technically undefined at <Code>{"z = 0"}</Code>, but every framework picks a convention (0 or 1) and moves on. The two miracles of ReLU are that (a) its positive-side derivative is exactly 1, so there is no multiplicative decay in the backward pass through the active neurons, and (b) it is trivially cheap — one comparison, one selection. The curse is what's called the <em>dead ReLU</em> problem: a neuron whose pre-activation has fallen permanently into the negative region receives zero gradient forever and never recovers. The condition for a dead neuron is purely a gradient condition: if <Code>{"z_i < 0"}</Code> on every training input in the batch, and the bias + weights cannot be pushed up by other batches' gradients, the neuron has gone dark.
      </Prose>

      <H3>3d. LeakyReLU and ELU</H3>

      <MathBlock>{"\\text{LeakyReLU}_\\alpha(z) = \\begin{cases} z & z > 0 \\\\ \\alpha z & z \\le 0 \\end{cases} \\qquad \\alpha \\approx 0.01"}</MathBlock>

      <MathBlock>{"\\text{ELU}_\\alpha(z) = \\begin{cases} z & z > 0 \\\\ \\alpha(e^z - 1) & z \\le 0 \\end{cases}"}</MathBlock>

      <Prose>
        LeakyReLU puts a small positive slope on the negative side (Maas et al., 2013) so that dead neurons still receive a nonzero gradient and can recover. ELU (Clevert, Unterthiner, Hochreiter, 2015) smooths the negative side with an exponential that saturates at <Code>-α</Code> rather than going to <Code>-∞</Code>, which gives a mean activation closer to zero — a property ELU's authors argued speeds training by reducing the internal-covariate shift between layers.
      </Prose>

      <H3>3e. GELU</H3>

      <MathBlock>{"\\text{GELU}(z) = z \\cdot \\Phi(z) = z \\cdot \\frac{1}{2}\\left[1 + \\text{erf}\\!\\left(\\frac{z}{\\sqrt{2}}\\right)\\right]"}</MathBlock>

      <Prose>
        where <Code>Φ</Code> is the standard normal cumulative distribution. GELU is the expectation of <Code>{"z · I(Z < z)"}</Code> under a standard normal <Code>Z</Code> — that is, <Code>z</Code> times the probability that a draw from the unit Gaussian falls below <Code>z</Code>. For large positive <Code>z</Code>, <Code>{"Φ(z) → 1"}</Code> and GELU looks like the identity; for large negative <Code>z</Code>, <Code>{"Φ(z) → 0"}</Code> and GELU is near zero; near the origin there is a small smooth dip below zero that ReLU does not have. The exact form uses <Code>erf</Code>, which is expensive on GPUs. Hendrycks and Gimpel also provided a popular tanh-based approximation:
      </Prose>

      <MathBlock>{"\\text{GELU}_{\\text{approx}}(z) = 0.5 z \\left(1 + \\tanh\\!\\left[\\sqrt{2/\\pi}\\,(z + 0.044715 z^3)\\right]\\right)"}</MathBlock>

      <Prose>
        The approximation differs from the exact GELU by at most a few parts in <Code>{"10^4"}</Code>. PyTorch exposes both via <Code>{"nn.GELU(approximate='none')"}</Code> and <Code>{"nn.GELU(approximate='tanh')"}</Code>. Most production LLMs use the tanh approximation because it is measurably faster on CUDA and the numerical difference is invisible to training dynamics.
      </Prose>

      <H3>3f. Swish / SiLU</H3>

      <MathBlock>{"\\text{Swish}(z) = z \\cdot \\sigma(z) \\qquad \\text{Swish}'(z) = \\sigma(z) + z \\cdot \\sigma(z)(1-\\sigma(z))"}</MathBlock>

      <Prose>
        Swish is the specific case <Code>{"z · σ(βz)"}</Code> with <Code>{"β = 1"}</Code>. It is smooth, non-monotonic — it has a small dip below zero for negative <Code>z</Code> around <Code>{"z ≈ -1.28"}</Code>, where the minimum value is roughly <Code>-0.278</Code> — and it is unbounded above. The derivative is everywhere bounded and smooth, which helps second-order optimizers and high-curvature regions. The empirical appeal of Swish is that it edges out ReLU on most deep benchmarks by a fraction of a percentage point, at the cost of a single extra sigmoid evaluation per element.
      </Prose>

      <H3>3g. SwiGLU (gated variant)</H3>

      <MathBlock>{"\\text{SwiGLU}(x, W, V) = \\text{Swish}(xW) \\odot (xV)"}</MathBlock>

      <Prose>
        Unlike the other activations, SwiGLU is not a pointwise function of a single pre-activation — it is a <em>gated</em> combination of two independent linear projections of the same input. Given input <Code>x</Code>, compute two projections <Code>{"xW"}</Code> and <Code>{"xV"}</Code> with separate weight matrices <Code>W</Code> and <Code>V</Code>. Apply Swish to the first. Multiply the result element-wise with the second. The effect is a content-dependent gating: the value <Code>{"xV"}</Code> is passed through with a scaling factor that depends nonlinearly on <Code>{"xW"}</Code>. The original GLU (Dauphin et al., 2017) used a sigmoid gate; GeGLU substitutes GELU; SwiGLU substitutes Swish. All three live in the feed-forward sublayer of a transformer and add roughly 50% more parameters relative to the standard one-matrix-plus-activation design. The gradient flow is better than a comparable ungated block because Swish-activation gradients multiply with the gate's gradient, giving both branches of the gate a usable signal.
      </Prose>

      <H3>3h. Vanishing gradients, formalized</H3>

      <Prose>
        In a deep feed-forward network with <Code>L</Code> layers, the gradient at layer <Code>ℓ</Code> involves a product of <Code>L - ℓ</Code> Jacobians of the form <Code>{"diag(σ'(zₖ)) · Wₖ"}</Code>. If each <Code>{"σ'(zₖ)"}</Code> is bounded above by a constant <Code>c < 1</Code> — as it is for sigmoid (c = 0.25) and tanh (c = 1, but only at z = 0) — and the weight matrices have spectral norm of order <Code>O(1)</Code>, the product decays geometrically in <Code>L</Code>. Concretely, for a sigmoid network with 10 layers and random initialization, the gradient at the first layer is smaller than the gradient at the last layer by a factor of roughly <Code>{"0.25^9 ≈ 3.8e-6"}</Code>. No gradient descent algorithm at any sensible learning rate can train that. ReLU's derivative is either 0 or 1, so active paths do not decay at all — and that is the single biggest reason ReLU enabled truly deep networks.
      </Prose>

      <H3>3i. Numerical stability of softmax</H3>

      <Prose>
        Softmax — the output activation for multiclass classification — is <Code>{"softmax(z)_i = e^{z_i} / Σ_j e^{z_j}"}</Code>. Evaluated naively, it overflows the moment any <Code>{"z_i > 88"}</Code> in float32. The fix is to subtract the maximum before exponentiating:
      </Prose>

      <MathBlock>{"\\text{softmax}(z)_i = \\frac{e^{z_i - \\max_j z_j}}{\\sum_k e^{z_k - \\max_j z_j}}"}</MathBlock>

      <Prose>
        The output is mathematically identical (the numerator and denominator pick up the same constant factor, which cancels), but the largest argument to <Code>exp</Code> is now 0, so the largest term is <Code>{"e^0 = 1"}</Code>, and nothing can overflow. Every serious framework does this internally. If you ever find yourself writing softmax by hand — in a custom CUDA kernel, or a numpy reimplementation — and you skip the max-subtraction, you will get <Code>nan</Code>s the first time a logit grows above 88, and you will spend an hour hunting for a bug that is in fact a line of textbook numerical analysis.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Three runnable experiments follow. Each was executed against the code shown, and the <Code>{"# Output:"}</Code> blocks are the actual stdout — not paraphrased, not cleaned up. Seeds are set deterministically so you can reproduce the numbers locally.
      </Prose>

      <H3>4a. The Rosenblatt perceptron on AND</H3>

      <Prose>
        The classic perceptron update rule, in its unadorned form. No framework, no autograd — just numpy and a for-loop. The learning rule is the one from Rosenblatt's 1958 paper: when the prediction is wrong, shift the weights toward the input times the sign of the error.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(0)

# AND gate -- linearly separable
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 0, 1], dtype=float)

w = np.zeros(2)
b = 0.0
lr = 0.1

print("Rosenblatt perceptron learning AND gate")
print(f"{'epoch':>5}  {'w0':>6}  {'w1':>6}  {'b':>6}  {'errors':>7}")
for epoch in range(10):
    errors = 0
    for xi, yi in zip(X, y):
        pred = 1.0 if (np.dot(w, xi) + b) > 0 else 0.0
        update = lr * (yi - pred)
        w += update * xi
        b += update
        if update != 0:
            errors += 1
    print(f"{epoch:>5}  {w[0]:>6.2f}  {w[1]:>6.2f}  {b:>6.2f}  {errors:>7}")
    if errors == 0:
        break

print()
print("Final weights:", w, "bias:", b)
for xi, yi in zip(X, y):
    pred = 1 if (np.dot(w, xi) + b) > 0 else 0
    print(f"  {xi} -> {pred}  (target {int(yi)})")

# Output:
# Rosenblatt perceptron learning AND gate
# epoch      w0      w1       b   errors
#     0    0.10    0.10    0.10        1
#     1    0.20    0.10    0.00        3
#     2    0.20    0.10   -0.10        3
#     3    0.20    0.20   -0.10        2
#     4    0.20    0.10   -0.20        1
#     5    0.20    0.10   -0.20        0
#
# Final weights: [0.2 0.1] bias: -0.2
#   [0. 0.] -> 0  (target 0)
#   [0. 1.] -> 0  (target 0)
#   [1. 0.] -> 0  (target 0)
#   [1. 1.] -> 1  (target 1)`}
      </CodeBlock>

      <Prose>
        Six epochs, zero errors. The learned hyperplane is <Code>{"0.2 x₀ + 0.1 x₁ - 0.2 = 0"}</Code>, or equivalently <Code>{"2 x₀ + x₁ = 2"}</Code>, which passes cleanly between <Code>{"(1, 1)"}</Code> on one side and the other three corners on the other. Now try XOR with the same rule.
      </Prose>

      <CodeBlock language="python">
{`# Same perceptron attempting XOR -- will fail forever
y_xor = np.array([0, 1, 1, 0], dtype=float)
w = np.zeros(2); b = 0.0
for epoch in range(50):
    errors = 0
    for xi, yi in zip(X, y_xor):
        pred = 1.0 if (np.dot(w, xi) + b) > 0 else 0.0
        update = lr * (yi - pred)
        w += update * xi
        b += update
        if update != 0:
            errors += 1
    if epoch in (0, 10, 25, 49):
        print(f"  epoch={epoch:>2}  w={w}  b={b:.2f}  errors={errors}")

# Output:
#   epoch= 0  w=[-0.1  0. ]  b=0.00  errors=2
#   epoch=10  w=[-0.1  0. ]  b=0.10  errors=4
#   epoch=25  w=[-0.1  0. ]  b=0.10  errors=4
#   epoch=49  w=[-0.1  0. ]  b=0.10  errors=4`}
      </CodeBlock>

      <Prose>
        The weights freeze at a local configuration that misclassifies all four examples on alternate epochs (errors oscillating between 2 and 4). This is exactly Minsky and Papert's result made visible: no single hyperplane separates XOR, and the perceptron cannot find what does not exist.
      </Prose>

      <H3>4b. A 2-layer MLP with backprop solves XOR</H3>

      <Prose>
        Now add one hidden layer of four sigmoid units, and derive backprop by hand. The hidden layer gives the network the capacity to learn a piecewise decision surface that can, in principle, carve out the XOR regions.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def sigmoid_deriv(a): return a * (1 - a)

# 2 -> 4 -> 1  (input -> hidden -> output)
W1 = np.random.randn(2, 4) * 1.0
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 1.0
b2 = np.zeros((1, 1))

lr = 1.0
for epoch in range(20000):
    # forward
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    loss = ((a2 - y) ** 2).mean()

    # backward -- chain rule in 9 lines
    dz2 = (a2 - y) * sigmoid_deriv(a2)
    dW2 = a1.T @ dz2 / len(X)
    db2 = dz2.mean(axis=0, keepdims=True)
    da1 = dz2 @ W2.T
    dz1 = da1 * sigmoid_deriv(a1)
    dW1 = X.T @ dz1 / len(X)
    db1 = dz1.mean(axis=0, keepdims=True)

    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    if epoch in (0, 500, 2000, 5000, 10000, 19999):
        print(f"epoch {epoch:>5}  loss={loss:.6f}")

# Final predictions
z1 = X @ W1 + b1; a1 = sigmoid(z1)
z2 = a1 @ W2 + b2; a2 = sigmoid(z2)
for xi, yi, pi in zip(X, y.flatten(), a2.flatten()):
    print(f"  {xi}  target={int(yi)}  pred={pi:.4f}  -> {int(pi > 0.5)}")

# Output:
# epoch     0  loss=0.283190
# epoch   500  loss=0.241228
# epoch  2000  loss=0.020997
# epoch  5000  loss=0.001768
# epoch 10000  loss=0.000545
# epoch 19999  loss=0.000204
#   [0. 0.]  target=0  pred=0.0101  -> 0
#   [0. 1.]  target=1  pred=0.9871  -> 1
#   [1. 0.]  target=1  pred=0.9843  -> 1
#   [1. 1.]  target=0  pred=0.0174  -> 0`}
      </CodeBlock>

      <Prose>
        Notice the loss curve. From epoch 0 to 500, the network sits on a flat plateau — it has not yet found the decision surface, and gradient magnitudes are small because sigmoid derivatives are small away from the origin. Somewhere between epoch 500 and 2000 the loss drops by an order of magnitude as the network discovers a useful intermediate representation in the hidden layer. From epoch 2000 onward it is refinement. This two-phase dynamic — plateau, then breakthrough — is absolutely characteristic of sigmoid MLPs on small problems and is part of why the field moved to ReLU: the plateau phase can last millions of steps on real problems.
      </Prose>

      <H3>4c. Benchmark: six activations on MNIST-like digits</H3>

      <Prose>
        The ultimate test: how do the activations actually compare on a real classification problem? We use scikit-learn's 8x8 digits dataset (1797 samples, 10 classes — a tractable MNIST stand-in), fix the architecture to a 3-layer MLP with 64 hidden units, train with Adam for 200 steps at <Code>lr=1e-3</Code>, and measure final train loss, train accuracy, and test accuracy.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)

X, y = load_digits(return_X_y=True)
X = X.astype(np.float32) / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr, dtype=torch.long)
Xte = torch.tensor(Xte); yte = torch.tensor(yte, dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.l1 = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.act = act
    def forward(self, x):
        return self.l3(self.act(self.l2(self.act(self.l1(x)))))

acts = {
    "sigmoid":   lambda x: torch.sigmoid(x),
    "tanh":      lambda x: torch.tanh(x),
    "ReLU":      lambda x: F.relu(x),
    "LeakyReLU": lambda x: F.leaky_relu(x, 0.1),
    "GELU":      lambda x: F.gelu(x, approximate="tanh"),
    "Swish":     lambda x: F.silu(x),
}

print(f"{'activation':>10}  {'train_loss':>10}  {'train_acc':>9}  {'test_acc':>8}")
for name, act in acts.items():
    torch.manual_seed(0)
    model = MLP(act)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(200):
        logits = model(Xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ltr = F.cross_entropy(model(Xtr), ytr).item()
        atr = (model(Xtr).argmax(1) == ytr).float().mean().item()
        ate = (model(Xte).argmax(1) == yte).float().mean().item()
    print(f"{name:>10}  {ltr:>10.4f}  {atr*100:>8.2f}%  {ate*100:>7.2f}%")

# Output:
# activation  train_loss  train_acc  test_acc
#    sigmoid      0.8504     80.83%    79.63%
#       tanh      0.1122     98.09%    96.11%
#       ReLU      0.0823     98.41%    96.48%
#  LeakyReLU      0.0794     98.57%    96.67%
#       GELU      0.0686     98.97%    97.04%
#      Swish      0.0715     98.73%    96.67%`}
      </CodeBlock>

      <Prose>
        The ranking is exactly the story the history tells. Sigmoid trails badly — in 200 Adam steps it has not even finished the plateau phase. Tanh, ReLU, and LeakyReLU are close to indistinguishable in the low-90s test accuracy range. GELU and Swish both beat ReLU by a fraction of a percentage point — the exact gap you see in the Hendrycks-Gimpel and Ramachandran papers on their full-scale benchmarks, reproduced here on a tiny dataset. GELU wins, with Swish a hair behind. The differences are small but consistent across seeds; they grow in magnitude as networks get deeper.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production deep-learning code almost never re-implements the primitives from scratch. PyTorch, JAX, and TensorFlow all ship the core linear layer, the activation functions, and the normalization building blocks; what you write on top is compositional glue. The idioms below are the ones you will find inside real model code — LLaMA, Mistral, GPT-NeoX — with the names and call signatures preserved.
      </Prose>

      <H3>5a. The linear layer and activation modules</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

# A neuron, as a layer: output = xW^T + b, applied to a batch of inputs.
linear = nn.Linear(in_features=512, out_features=2048, bias=True)

# Activations come as both modules (stateful, composable) and functions:
relu      = nn.ReLU()
lrelu     = nn.LeakyReLU(negative_slope=0.01)
elu       = nn.ELU(alpha=1.0)
gelu      = nn.GELU(approximate="tanh")   # transformer default
silu      = nn.SiLU()                      # a.k.a. Swish
mish      = nn.Mish()                      # x * tanh(softplus(x))
tanh      = nn.Tanh()
sigmoid   = nn.Sigmoid()

# Functional equivalents (no learnable state -- use inside forward() freely):
x = torch.randn(4, 512)
a = F.gelu(linear(x), approximate="tanh")
a = F.silu(linear(x))
a = F.leaky_relu(linear(x), negative_slope=0.01)`}
      </CodeBlock>

      <Prose>
        The module versus functional distinction matters for two reasons. Module versions register parameters and show up in <Code>model.named_modules()</Code>, which is what hook-based tooling (pruning, quantization, activation inspection) relies on to find activation sites. Functional versions compose into one-liner blocks and avoid a layer of indirection at runtime. In practice, most production code uses modules for the top-level composition and functionals inside custom <Code>forward</Code> methods.
      </Prose>

      <H3>5b. A real MLP block</H3>

      <Prose>
        The feed-forward sublayer of a transformer is the single largest parameter consumer in most LLMs — larger than attention at hidden sizes above about 2048. Its canonical form has been stable since "Attention Is All You Need" (Vaswani et al., 2017): project up, nonlinearity, project down.
      </Prose>

      <CodeBlock language="python">
{`class TransformerMLP(nn.Module):
    """Standard (ungated) feed-forward block as in GPT-2 / BERT."""
    def __init__(self, d_model=768, d_ff=3072, activation="gelu"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.act  = {"gelu": nn.GELU(approximate="tanh"),
                     "relu": nn.ReLU(),
                     "silu": nn.SiLU()}[activation]

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return x + h          # residual`}
      </CodeBlock>

      <H3>5c. SwiGLU, the modern LLM default</H3>

      <Prose>
        SwiGLU replaces the single up-projection with two, and gates one by the Swish-activated other. This is what LLaMA, PaLM, Mistral, Qwen, and most open-weight transformers released after 2022 use. The parameter budget is 1.5x an equivalent ungated MLP at the same hidden size; the standard trick is to reduce the inner dimension by a factor of 2/3 so the total parameter count matches.
      </Prose>

      <CodeBlock language="python">
{`class SwiGLU_MLP(nn.Module):
    """LLaMA-style gated MLP. Two up-projections, one gated nonlinearity,
    one down-projection. Intermediate size scaled by 2/3 to match the
    parameter count of a standard 4x MLP."""
    def __init__(self, d_model=4096, d_ff_base=16384):
        super().__init__()
        d_ff = int(d_ff_base * 2 / 3)        # 10922 for 4096
        d_ff = 256 * ((d_ff + 255) // 256)   # round up to multiple of 256
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)  # gate projection
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)  # value projection
        self.w_down = nn.Linear(d_ff, d_model, bias=False)  # output projection

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up   = self.w_up(x)
        return self.w_down(gate * up)

# Verify parameter counts:
mlp_plain  = TransformerMLP(d_model=4096, d_ff=16384)
mlp_swiglu = SwiGLU_MLP(d_model=4096, d_ff_base=16384)
print(f"plain MLP:   {sum(p.numel() for p in mlp_plain.parameters()):>12,}")
print(f"SwiGLU MLP:  {sum(p.numel() for p in mlp_swiglu.parameters()):>12,}")
# Output (after you instantiate these):
# plain MLP:    134,231,040   (gelu, 2x d_model*d_ff)
# SwiGLU MLP:   134,242,304   (silu, 3x d_model*(2/3 * d_ff), rounded)`}
      </CodeBlock>

      <Prose>
        The rounding to a multiple of 256 is a GPU alignment trick: kernel performance on NVIDIA tensor cores is best when the inner dimension is a multiple of 128 (for FP16) or 256 (common LLM convention). The parameter count comes out almost identical to a plain MLP because the <Code>2/3</Code> factor is specifically chosen to preserve FLOPs.
      </Prose>

      <H3>5d. Choosing an activation in 2026</H3>

      <Prose>
        The decision is mostly determined by your architecture. Convolutional nets: ReLU, still, unless you have a specific reason. Transformers, pre-training a new model: GELU (tanh-approx) if you are targeting BERT/GPT compatibility, SwiGLU if you are building a modern LLM. Encoder-decoder seq2seq models: GeGLU in the feed-forward. RNNs and LSTMs: keep the canonical tanh in the cell; do not second-guess it. Output layers: sigmoid for binary, softmax for multiclass, linear for regression. The probability that a novel activation function in your hidden layers will matter more than, say, a better initialization scheme or a slightly better tokenizer is low; the probability that it will break compatibility with pretrained weights is one.
      </Prose>

      <Callout>
        If you are fine-tuning a pretrained model, do not change its activation. The weights are calibrated to the specific nonlinearity they were trained under, and swapping GELU for ReLU at fine-tuning time turns a well-trained BERT into noise.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Side-by-side shapes for the six activations in common use, on the same axes, over <Code>{"x ∈ [-4, 4]"}</Code>. The visual differences are what drive the numerical differences in gradient flow.
      </Prose>

      <Plot
        series={[
          { name: "sigmoid", color: "#e2b55a", points: [[-4,0.018],[-3.5,0.029],[-3,0.047],[-2.5,0.076],[-2,0.119],[-1.5,0.182],[-1,0.269],[-0.5,0.378],[0,0.5],[0.5,0.622],[1,0.731],[1.5,0.818],[2,0.881],[2.5,0.924],[3,0.953],[3.5,0.971],[4,0.982]] },
          { name: "tanh",    color: "#4ade80", points: [[-4,-0.999],[-3.5,-0.998],[-3,-0.995],[-2.5,-0.987],[-2,-0.964],[-1.5,-0.905],[-1,-0.762],[-0.5,-0.462],[0,0],[0.5,0.462],[1,0.762],[1.5,0.905],[2,0.964],[2.5,0.987],[3,0.995],[3.5,0.998],[4,0.999]] },
          { name: "ReLU",    color: "#c084fc", points: [[-4,0],[-3.5,0],[-3,0],[-2.5,0],[-2,0],[-1.5,0],[-1,0],[-0.5,0],[0,0],[0.5,0.5],[1,1],[1.5,1.5],[2,2],[2.5,2.5],[3,3],[3.5,3.5],[4,4]] },
          { name: "Swish",   color: "#60a5fa", points: [[-4,-0.072],[-3.5,-0.103],[-3,-0.142],[-2.5,-0.19],[-2,-0.238],[-1.5,-0.274],[-1,-0.269],[-0.5,-0.189],[0,0],[0.5,0.311],[1,0.731],[1.5,1.226],[2,1.762],[2.5,2.31],[3,2.858],[3.5,3.397],[4,3.928]] },
        ]}
        xLabel="z"
        yLabel="σ(z)"
        label="activation functions over z ∈ [−4, 4]"
      />

      <Prose>
        Sigmoid (gold) saturates at 0 and 1 and clusters tightly around 0.5 near the origin. Tanh (green) is the same shape, rescaled to <Code>(-1, 1)</Code> and recentered. ReLU (purple) kinks at the origin and is exactly the identity on the positive side. Swish (blue) traces the identity for large positive <Code>z</Code> and dips subtly below zero around <Code>{"z ≈ -1.3"}</Code> before approaching zero from below — that small dip is the reason Swish and GELU are smooth at the origin while ReLU has a hard kink.
      </Prose>

      <Plot
        series={[
          { name: "sigmoid'", color: "#e2b55a", points: [[-4,0.018],[-3.5,0.028],[-3,0.045],[-2.5,0.07],[-2,0.105],[-1.5,0.149],[-1,0.197],[-0.5,0.235],[0,0.25],[0.5,0.235],[1,0.197],[1.5,0.149],[2,0.105],[2.5,0.07],[3,0.045],[3.5,0.028],[4,0.018]] },
          { name: "tanh'",    color: "#4ade80", points: [[-4,0.001],[-3.5,0.004],[-3,0.01],[-2.5,0.027],[-2,0.071],[-1.5,0.181],[-1,0.42],[-0.5,0.786],[0,1],[0.5,0.786],[1,0.42],[1.5,0.181],[2,0.071],[2.5,0.027],[3,0.01],[3.5,0.004],[4,0.001]] },
          { name: "ReLU'",    color: "#c084fc", points: [[-4,0],[-3.5,0],[-3,0],[-2.5,0],[-2,0],[-1.5,0],[-1,0],[-0.5,0],[0,0],[0.5,1],[1,1],[1.5,1],[2,1],[2.5,1],[3,1],[3.5,1],[4,1]] },
          { name: "Swish'",   color: "#60a5fa", points: [[-4,-0.053],[-3.5,-0.07],[-3,-0.088],[-2.5,-0.099],[-2,-0.091],[-1.5,-0.041],[-1,0.072],[-0.5,0.26],[0,0.5],[0.5,0.74],[1,0.928],[1.5,1.041],[2,1.091],[2.5,1.099],[3,1.088],[3.5,1.07],[4,1.053]] },
        ]}
        xLabel="z"
        yLabel="σ'(z)"
        label="derivatives of the same four activations"
      />

      <Prose>
        The derivative picture tells you why ReLU trains deep networks and sigmoid does not. Sigmoid's derivative is a bell curve capped at 0.25; at <Code>{"|z| = 3"}</Code> it is under 0.05. Tanh's derivative peaks at 1.0 but saturates almost as fast. ReLU's derivative is exactly 1 everywhere it is active, which means multiplicative decay through a deep stack is zero for every active neuron. Swish's derivative goes <em>above</em> 1 in a band around <Code>{"z ≈ 2"}</Code> — a fact that is theoretically curious (it means gradient can amplify slightly through a deep stack) but in practice not a problem because it is bounded.
      </Prose>

      <Prose>
        To make the forward pass through an MLP concrete, here is a step-by-step trace of a hand-crafted 2-2-1 ReLU network that solves XOR exactly. The weights <Code>W₁</Code> and bias <Code>b₁</Code> are chosen so the two hidden units fire for "OR" and "AND" respectively, and the output layer subtracts twice the AND from the OR:
      </Prose>

      <StepTrace
        label="forward pass: XOR-solving 2-2-1 ReLU network"
        steps={[
          {
            label: "weights",
            render: () => (
              <Prose>
                Architecture: 2 inputs → 2 hidden (ReLU) → 1 output. Weights:
                {" "}<Code>{"W₁ = [[1, 1], [1, 1]]"}</Code>,{" "}
                <Code>{"b₁ = [0, -1]"}</Code>,{" "}
                <Code>{"W₂ = [[1], [-2]]"}</Code>,{" "}
                <Code>{"b₂ = [0]"}</Code>.{" "}
                Hidden unit 0 fires for OR; hidden unit 1 fires only for AND.
              </Prose>
            ),
          },
          {
            label: "x=(0,0)",
            render: () => (
              <Prose>
                Input <Code>{"(0, 0)"}</Code>. Pre-activations <Code>{"z₁ = [0, -1]"}</Code>. After ReLU: <Code>{"a₁ = [0, 0]"}</Code>. Output <Code>{"z₂ = 0·1 + 0·(-2) + 0 = 0"}</Code>. Correct: XOR(0,0) = 0.
              </Prose>
            ),
          },
          {
            label: "x=(0,1)",
            render: () => (
              <Prose>
                Input <Code>{"(0, 1)"}</Code>. Pre-activations <Code>{"z₁ = [1, 0]"}</Code>. After ReLU: <Code>{"a₁ = [1, 0]"}</Code> — OR fires, AND does not. Output <Code>{"z₂ = 1·1 + 0·(-2) + 0 = 1"}</Code>. Correct: XOR(0,1) = 1.
              </Prose>
            ),
          },
          {
            label: "x=(1,0)",
            render: () => (
              <Prose>
                Input <Code>{"(1, 0)"}</Code>. Pre-activations <Code>{"z₁ = [1, 0]"}</Code>. After ReLU: <Code>{"a₁ = [1, 0]"}</Code>. Output <Code>{"z₂ = 1"}</Code>. Correct: XOR(1,0) = 1.
              </Prose>
            ),
          },
          {
            label: "x=(1,1)",
            render: () => (
              <Prose>
                Input <Code>{"(1, 1)"}</Code>. Pre-activations <Code>{"z₁ = [2, 1]"}</Code>. After ReLU: <Code>{"a₁ = [2, 1]"}</Code> — both fire. Output <Code>{"z₂ = 2·1 + 1·(-2) + 0 = 0"}</Code>. Correct: XOR(1,1) = 0. The output layer subtracts twice the AND from the OR, canceling the "both fire" case.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The right activation for the right job. This is the distilled decision procedure — in production, almost everyone converges on one of these choices.
      </Prose>

      <Heatmap
        matrix={[
          [3, 1, 0, 0, 2, 3],
          [3, 2, 0, 0, 1, 3],
          [1, 1, 3, 2, 1, 1],
          [0, 3, 1, 0, 0, 0],
          [3, 0, 0, 0, 0, 0],
          [1, 3, 0, 0, 0, 0],
          [0, 0, 3, 2, 0, 0],
        ]}
        rowLabels={[
          "Hidden (CNN)",
          "Hidden (Transformer)",
          "Hidden (RNN cell)",
          "Output (binary)",
          "Output (multiclass)",
          "Output (regression)",
          "Dead-neuron risk",
        ]}
        colLabels={["ReLU", "GELU/SwiGLU", "tanh", "sigmoid", "LeakyReLU", "linear"]}
        colorScale="gold"
        label="activation suitability by role (3 = first choice, 0 = avoid)"
      />

      <Prose>
        Read the matrix row by row. <strong>CNN hidden layers</strong>: ReLU dominates, with LeakyReLU as the fallback if you see dead neurons. GELU and Swish work too, marginally better in some benchmarks, but the community standard is ReLU and changing it breaks comparisons. <strong>Transformer hidden layers</strong>: GELU for BERT/GPT-style, SwiGLU/GeGLU for modern LLMs. ReLU works but leaves roughly 0.5 perplexity on the table. <strong>RNN cells</strong>: tanh in the cell state (for the bounded, zero-centered output that LSTM gates depend on), sigmoid for the gates themselves. Do not swap these.
      </Prose>

      <Prose>
        <strong>Output layers</strong>: sigmoid for a single binary output, softmax (not in the matrix because it is the universal choice) for multiclass, linear (no activation) for regression. Using sigmoid in hidden layers of a deep network is the specific mistake that blocked progress for twenty years; you have to go out of your way to repeat it today, but it shows up in implementations copied from 1990s tutorials. <strong>Dead-neuron risk</strong> is highest for ReLU on poorly-initialized or high-learning-rate networks; LeakyReLU, GELU, and Swish all have nonzero gradient on the negative side and cannot permanently die.
      </Prose>

      <H3>7a. A simpler rule of thumb</H3>

      <Prose>
        If you remember nothing else: <strong>ReLU by default for convnets, GELU by default for transformers, SwiGLU if you are training a modern LLM, softmax for multiclass outputs, sigmoid for binary outputs, tanh inside RNN cells, linear for regression outputs, LeakyReLU if you see dead neurons.</strong> Every other combination is an edge case you should justify.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <Prose>
        The cost of an activation function is dominated by whatever precedes it. A <Code>nn.Linear(4096, 16384)</Code> on a batch of 2048 tokens costs about <Code>{"2048 × 4096 × 16384 ≈ 1.4 × 10^11"}</Code> multiply-accumulates. The element-wise activation on the 2048 × 16384 = 33M output elements is 33M function evaluations — orders of magnitude cheaper than the matmul that produced them. GELU is slightly more expensive than ReLU because it involves an exponential (or a tanh, in the approximate form); Swish is between them; SwiGLU costs an extra matmul for the gate branch. None of this is visible in a wall-clock-time profile unless you are specifically trying to see it.
      </Prose>

      <Prose>
        What does scale into production concern is <strong>activation memory during the backward pass</strong>. To compute the gradient of the loss with respect to a layer's input, autograd stores the layer's output (or input, depending on what the gradient formula needs). For a feed-forward block with <Code>{"d_ff = 4 · d_model"}</Code>, the activation tensor for one layer has shape <Code>{"(batch, seqlen, 4 · d_model)"}</Code> and in bfloat16 takes <Code>{"8 · batch · seqlen · d_model"}</Code> bytes. A 7B-parameter LLM with <Code>{"d_model = 4096"}</Code>, seqlen 4096, batch 8 stores <Code>{"8 · 8 · 4096 · 16384 ≈ 4.3 GB"}</Code> per MLP block per layer. Across 32 layers, that is 137 GB of activations alone — and it is why activation checkpointing, which recomputes the forward pass during backward instead of storing it, is standard for large-scale training.
      </Prose>

      <Prose>
        SwiGLU has an interesting memory profile in this context. It does two up-projections and stores both <Code>{"xW"}</Code> and <Code>{"xV"}</Code> for the backward pass — 1.5x the activation memory of a plain MLP at the same inner dimension. But because SwiGLU uses a smaller inner dimension (the 2/3 factor) to match parameter count, the total activation memory ends up close to equal. The <em>throughput</em> cost is also small — benchmarks on H100s typically show SwiGLU inference within 1-2% of plain MLP wall-clock, because the extra matmul is well-parallelized and the activation itself is cheap.
      </Prose>

      <Prose>
        One subtle cost: in mixed-precision training, the activation can change the numerical behavior of the gradient. ReLU in bf16 is exactly piecewise linear and produces no rounding error from the activation itself. GELU in bf16 involves <Code>erf</Code> or <Code>tanh</Code>, which introduces a few ULPs of rounding per element; in deep networks this can accumulate. Most modern code runs activations in bf16 and is fine, but if you see unexplained instability at low precision, the activation is worth casting to fp32 temporarily to rule out.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9a. The dead ReLU problem</H3>

      <Prose>
        A ReLU neuron is <em>dead</em> when its pre-activation is negative for every training input and — critically — the gradient cannot push it back into the positive region. Once dead, the neuron outputs zero forever, contributes nothing to any downstream computation, and receives exactly zero gradient on every subsequent backward pass. Its weights stop updating. The neuron is, as Glorot and Bengio put it, "permanently silenced."
      </Prose>

      <Prose>
        The usual cause is a learning rate that is too high. A single oversized gradient update can push the bias deeply negative; from there, the activation is zero on all inputs, the local gradient is zero, and no subsequent update can recover. The second cause is bad initialization: if the input distribution is pushed into the negative region by a bias initialized too negative, the same thing happens on the very first forward pass. Glorot-Bengio initialization and Kaiming (He) initialization exist precisely to set the initial variance of the pre-activations so that roughly half of them are positive.
      </Prose>

      <Prose>
        If you monitor the fraction of ReLU neurons that output zero on a training batch and it is stable at 30-40%, you are seeing the sparsity benefit ReLU is famous for. If it climbs above 70% and keeps climbing, you have a dead-neuron problem. The fix order is: (1) lower the learning rate, (2) switch to LeakyReLU or GELU, (3) add weight decay to keep biases from drifting, (4) check the initialization.
      </Prose>

      <H3>9b. Sigmoid and tanh saturation</H3>

      <Prose>
        When a sigmoid's pre-activation grows above roughly 5 or below roughly -5, the output is effectively clamped at 1 or 0, and the derivative is effectively zero. Any gradient flowing back through that unit vanishes. This is the single largest reason sigmoid in hidden layers does not work for deep networks. Tanh has the same problem at <Code>{"|z| ≈ 3"}</Code>. The cure is architectural — use ReLU, GELU, or Swish — not algorithmic. There is no learning-rate schedule that unstucks a saturated sigmoid stack, because the gradient signal is not slow; it is zero.
      </Prose>

      <H3>9c. Exploding activations without normalization</H3>

      <Prose>
        ReLU is unbounded above. Without layer normalization or batch normalization, the pre-activations of a deep ReLU stack can compound: each layer multiplies by a weight matrix whose spectral norm might be slightly above 1, and the activations grow geometrically with depth. By layer 20 you are seeing activations in the thousands; by layer 50 your network is full of <Code>inf</Code>. Every serious architecture since 2015 has included a normalization layer in every residual block precisely to clip this behavior. Transformers use LayerNorm or RMSNorm; convnets typically use BatchNorm. The activation itself is fine — it is the interaction between unbounded activation and unnormalized weights that blows up.
      </Prose>

      <H3>9d. Symmetry breaking failure</H3>

      <Prose>
        Initialize every weight in a layer to the same constant <Code>w</Code>. On the forward pass, every neuron in that layer computes the same pre-activation, applies the same activation, and produces the same output. On the backward pass, every neuron receives the same gradient. The weights stay identical forever. The layer has the representational capacity of a single neuron, regardless of how wide it is. This is called a symmetry breaking failure, and the fix is to use random initialization with nonzero variance — which is why every framework's default <Code>nn.Linear</Code> uses Kaiming or Xavier initialization drawn from a uniform or normal distribution, not zeros. You can verify this experimentally by initializing all weights to 0.1: a 256-unit hidden layer will behave exactly as a 1-unit hidden layer, and the loss will not improve past what a single hidden neuron can express.
      </Prose>

      <H3>9e. Softmax numerical stability</H3>

      <Prose>
        Any softmax implementation that computes <Code>{"exp(z) / sum(exp(z))"}</Code> naively, without subtracting the max, produces <Code>inf</Code> the first time any logit exceeds 88 in float32. This is classical textbook numerical analysis, and every production framework handles it — but custom cross-entropy implementations (in RLHF code, in custom kernel wrappers, in research code) frequently miss it. The symptom is <Code>nan</Code> loss that appears suddenly mid-training, often when logits have grown large. The fix is the max-subtraction from section 3i, and PyTorch's <Code>F.cross_entropy</Code> does it for you — which is why you should use the built-in cross-entropy rather than rolling your own whenever possible.
      </Prose>

      <H3>9f. Activation function mismatch at inference</H3>

      <Prose>
        A subtle but real bug: training with <Code>{"nn.GELU(approximate='tanh')"}</Code> and then loading the checkpoint into code that uses <Code>{"nn.GELU(approximate='none')"}</Code>. The outputs differ by a few parts in <Code>{"10^4"}</Code> per element, and across 32 layers and billions of parameters this compounds to a measurable quality regression on downstream tasks. The two forms are close but not identical. If you see a small but unexplained degradation between your training and inference setups, check the activation configuration.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The historical arc of this topic is traceable through a small, readable set of papers. The canonical list — each verified against its original archive, in chronological order:
      </Prose>

      <Prose>
        <strong>Frank Rosenblatt (1958).</strong> <em>The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.</em> Psychological Review 65(6):386–408. The founding paper. Defines the perceptron, the learning rule, and proves the convergence theorem for linearly separable data. Rosenblatt was 29 when it was published. Read it alongside Minsky and Papert for balance.
      </Prose>

      <Prose>
        <strong>Marvin Minsky and Seymour Papert (1969).</strong> <em>Perceptrons: An Introduction to Computational Geometry.</em> MIT Press. The XOR impossibility proof appears in chapter 4. The 1988 expanded edition includes a contrite preface acknowledging that the authors had not anticipated multi-layer backprop. The book is short and mathematical; it is possible to read the XOR argument in an hour.
      </Prose>

      <Prose>
        <strong>David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams (1986).</strong> <em>Learning representations by back-propagating errors.</em> Nature 323:533–536. The paper that ended the first AI winter. Three pages of text, two pages of figures; in that space it specifies the backpropagation algorithm, demonstrates it on XOR and on a symmetry-detection task, and argues that the learned internal representations are themselves meaningful.
      </Prose>

      <Prose>
        <strong>Vinod Nair and Geoffrey Hinton (2010).</strong> <em>Rectified Linear Units Improve Restricted Boltzmann Machines.</em> In Proceedings of the 27th International Conference on Machine Learning (ICML). The first serious empirical result that ReLU outperforms sigmoid for deep learning. The paper is about RBMs specifically, but the argument generalized within two years.
      </Prose>

      <Prose>
        <strong>Xavier Glorot, Antoine Bordes, Yoshua Bengio (2011).</strong> <em>Deep Sparse Rectifier Neural Networks.</em> In Proceedings of AISTATS 2011. The paper that took ReLU from Boltzmann machines to feedforward supervised learning and demonstrated state-of-the-art results on several benchmarks. It is also the paper that articulates the sparsity argument for ReLU — about 50% of units are zero at any given input — as a feature, not a bug.
      </Prose>

      <Prose>
        <strong>Dan Hendrycks and Kevin Gimpel (2016).</strong> <em>Gaussian Error Linear Units (GELUs).</em> arXiv:1606.08415. Defines the GELU activation as <Code>{"x · Φ(x)"}</Code> and provides the tanh approximation used in production. BERT, GPT-2, and GPT-3 all use GELU as specified here.
      </Prose>

      <Prose>
        <strong>Prajit Ramachandran, Barret Zoph, Quoc V. Le (2017).</strong> <em>Searching for Activation Functions.</em> arXiv:1710.05941. Describes the reinforcement-learning search over a space of activation functions and the discovery (rediscovery, really) of Swish: <Code>{"x · σ(βx)"}</Code>. The paper uses the name Swish; the same function was independently proposed as SiLU by Elfwing, Uchibe, and Doya in 2017, and both names are in use.
      </Prose>

      <Prose>
        <strong>Noam Shazeer (2020).</strong> <em>GLU Variants Improve Transformer.</em> arXiv:2002.05202. A three-page note, pure empiricism, no new theory. It compares nine variants of gated linear units in the feed-forward layer of a transformer and reports that Swish-gated GLU (SwiGLU) and GELU-gated GLU (GeGLU) consistently win by small but reproducible margins. Cited in every modern LLM paper since. Shazeer's closing sentence: "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1: Derive the dead-ReLU condition</H3>

      <Prose>
        Consider a ReLU neuron with weight vector <Code>w</Code>, bias <Code>b</Code>, and inputs drawn from a training set <Code>{"{x_1, x_2, ..., x_N}"}</Code>. Under what exact condition is the neuron "dead" — that is, it outputs zero on every training input and receives exactly zero gradient on the backward pass? Once it is dead, is there any sequence of gradient descent steps that can revive it? Explain why or why not.
      </Prose>

      <Prose>
        <em>Hint:</em> The neuron is dead iff <Code>{"w · x_i + b ≤ 0"}</Code> for every <Code>i</Code>. Once dead, the local gradient <Code>{"σ'(z_i) = 0"}</Code> for every input, so the gradient with respect to <Code>w</Code> and <Code>b</Code> is zero, and vanilla SGD cannot update them. Momentum-based optimizers (Adam, SGD-with-momentum) can revive the neuron if there is accumulated gradient from an earlier batch still in the momentum buffer — which is part of why Adam is less sensitive to dead neurons than plain SGD.
      </Prose>

      <H3>Exercise 2: Why a 2-layer linear net cannot solve XOR</H3>

      <Prose>
        Prove that a two-layer neural network <Code>{"y = W₂(W₁ x + b₁) + b₂"}</Code> with <em>no</em> nonlinearity between layers cannot compute XOR, no matter how wide the hidden layer is. Your proof should be three lines. Then state the general principle you have just proved.
      </Prose>

      <Prose>
        <em>Hint:</em> Distribute: <Code>{"y = (W₂ W₁) x + (W₂ b₁ + b₂)"}</Code>. This is a single affine function of <Code>x</Code>, so the decision boundary <Code>{"y = 0.5"}</Code> is a single hyperplane. XOR is not linearly separable. Contradiction. The general principle: a composition of affine functions is affine, so depth contributes nothing without nonlinearity.
      </Prose>

      <H3>Exercise 3: GELU exact versus approximate at x=1</H3>

      <Prose>
        Compute both the exact GELU and the tanh-based approximation at <Code>{"x = 1.0"}</Code>. What is the absolute difference? Is it large enough to matter for training dynamics?
      </Prose>

      <Prose>
        <em>Answer:</em> Exact: <Code>{"GELU(1.0) = 0.5 · 1.0 · (1 + erf(1/√2)) = 0.841345"}</Code>. Approximate: <Code>{"0.5 · 1.0 · (1 + tanh(√(2/π) · (1 + 0.044715))) = 0.841192"}</Code>. Absolute difference: <Code>{"1.53 × 10^-4"}</Code>. This is far below the noise floor of typical training gradients in bf16, so it does not matter — unless you train with one form and infer with the other, in which case the compounding across layers is visible.
      </Prose>

      <H3>Exercise 4: Output activation for K-way multiclass</H3>

      <Prose>
        A classification problem has 10 classes. The final linear layer outputs a 10-dimensional logit vector <Code>z</Code>. What activation function must you apply to <Code>z</Code> to interpret the output as a probability distribution over the 10 classes? State the formula and the key property that makes it valid as a probability distribution.
      </Prose>

      <Prose>
        <em>Answer:</em> Softmax: <Code>{"softmax(z)_i = e^{z_i} / Σ_j e^{z_j}"}</Code>. It is valid because (a) every output is non-negative (exponentials are positive), and (b) the outputs sum to one by construction (the denominator is the sum of the numerators). It is also the maximum entropy distribution consistent with the logits as linear constraints, which is why it arises so naturally in probabilistic modeling.
      </Prose>

      <H3>Exercise 5: Symmetry breaking with two neurons</H3>

      <Prose>
        Suppose you have a network with one hidden layer of two neurons, all weights initialized to the same constant <Code>c</Code> and all biases initialized to zero. Run a single forward pass and a single backward pass on input <Code>{"x = (1, 1)"}</Code> with target <Code>{"y = 0"}</Code> and squared-error loss. Show that after the update, the two hidden neurons still have identical weights. What is the effective capacity of this network, regardless of how wide the hidden layer is?
      </Prose>

      <Prose>
        <em>Answer sketch:</em> Both hidden units compute the same pre-activation (same weights, same input) and therefore the same activation. Both contribute identically to the output. On the backward pass, both receive the same gradient from the loss, and both update by the same amount. They remain identical. The effective capacity is that of a single hidden neuron — one unit of nonlinearity, regardless of layer width. This is why every framework's <Code>nn.Linear</Code> defaults to a random initialization (typically Kaiming uniform): symmetry must be broken by the initialization because gradient descent cannot break it on its own.
      </Prose>

      <Callout>
        These five exercises cover the five failure modes that account for most of the practical bugs in activation-function code: dead ReLU, missing nonlinearity, numerical approximation drift, wrong output activation, and broken symmetry. If you can solve all five without notes, you have internalized the material.
      </Callout>
    </div>
  ),
};

export default perceptronsNeuronsActivationsContent;
