import { Callout, Code, CodeBlock, H2, Prose } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";

const content = {
  title: "Functional Analysis & RKHS",
  readTime: "~44 min",
  content: () => <div>
    <H2>1. Why study spaces of functions?</H2>
    <Prose>
      In ordinary linear algebra, vectors are finite lists of numbers. In many learning problems, the object we want is a function: a regression curve, a decision boundary, a probability density, or a solution to a differential equation. Functional analysis gives functions vector-space structure—addition, scaling, distance, convergence, and linear operators—so we can reason about infinitely many possible functions without losing mathematical control.
    </Prose>
    <Prose>
      You do not need the full abstract theory to use kernels. The useful bridge is this: a well-chosen function space defines which functions count as simple, and a norm provides a principled way to penalise complexity.
    </Prose>

    <H2>2. Norms, inner products, and Hilbert spaces</H2>
    <Prose>
      A norm measures size; an inner product measures alignment and induces a norm. A Hilbert space is an inner-product space that is complete: every Cauchy sequence of functions whose values should converge has a limit inside the space. Completeness prevents optimisation or approximation from chasing a limit that the chosen space does not contain.
    </Prose>
    <MathBlock>{`\\langle f,g\\rangle_{\\mathcal{H}}, \\qquad \\|f\\|_{\\mathcal{H}}=\\sqrt{\\langle f,f\\rangle_{\\mathcal{H}}}`}</MathBlock>
    <Prose>
      The same pointwise function can have different norms in different spaces. A large RKHS norm often means a function is complex relative to a chosen kernel; it is not a universal statement that the function is physically rough or morally undesirable.
    </Prose>

    <H2>3. Reproducing kernels make point evaluation well behaved</H2>
    <Prose>
      A reproducing-kernel Hilbert space (RKHS) is a Hilbert space of functions where evaluating a function at x is a continuous linear operation. For every input x, there is a representer <Code>K(x, ·)</Code> in the space such that the inner product with it reproduces the function value. This remarkable property turns function evaluation into geometry.
    </Prose>
    <MathBlock>{`f(x)=\\langle f,K(x,\\cdot)\\rangle_{\\mathcal{H}}`}</MathBlock>
    <Prose>
      The kernel K(x, z) is the inner product between the two representers. It behaves like a similarity function, but not every intuitive similarity is a valid kernel. A valid positive-semidefinite kernel produces a non-negative quadratic form for every finite set of inputs and coefficients.
    </Prose>
    <MathBlock>{`\\sum_{i,j}c_i c_j K(x_i,x_j)\\geq0`}</MathBlock>

    <H2>4. The kernel trick is really a modelling trick</H2>
    <Prose>
      A linear model in an implicit feature map phi(x) can become a non-linear model in original input space because <Code>K(x, z) = inner_product(phi(x), phi(z))</Code>. The radial-basis-function (RBF) kernel gives high similarity to nearby points and lower similarity to distant points. Polynomial, string, graph, and domain-specific kernels encode different prior beliefs about what patterns should generalise.
    </Prose>
    <MathBlock>{`K_{\\mathrm{RBF}}(x,z)=\\exp(-\\gamma\\|x-z\\|^2)`}</MathBlock>
    <Callout accent="green" label="Kernel choice is an assumption">
      Gamma in an RBF kernel controls locality. Too large makes each training point influence only a tiny region and can overfit; too small makes the model nearly constant. Feature scaling changes distances, so standardise meaningful continuous features before treating an RBF default as evidence-based.
    </Callout>

    <H2>5. The representer theorem makes infinite spaces computable</H2>
    <Prose>
      Consider minimising a training loss plus a penalty on RKHS norm. Although the search space contains infinitely many functions, the representer theorem says a solution can be written as a finite combination of kernels centred at the training points. The difficult functional problem becomes a finite optimisation over n coefficients.
    </Prose>
    <MathBlock>{`f^*(x)=\\sum_{i=1}^{n}\\alpha_iK(x_i,x)`}</MathBlock>
    <Prose>
      For kernel ridge regression with squared loss, one common objective leads to <Code>alpha = (K + lambda I)^(-1) y</Code>, where K is the n-by-n Gram matrix. This is ordinary linear algebra applied to function learning.
    </Prose>

    <H2>6. A tiny kernel ridge-regression calculation</H2>
    <Prose>
      The code below fits two one-dimensional observations using an RBF kernel and regularisation. It solves the two-by-two system directly so every quantity is visible. The prediction at 0.5 is influenced by both training points, weighted by their learned coefficients and their kernel similarity to the query.
    </Prose>
    <CodeBlock language="python">{`import math

xs, ys = [0.0, 1.0], [0.0, 1.0]
gamma, regularisation = 1.0, 0.1
k01 = math.exp(-gamma * (xs[0] - xs[1]) ** 2)

# Invert K + lambda I for this 2-by-2 example.
a, b = 1 + regularisation, k01
determinant = a * a - b * b
alpha0 = (a * ys[0] - b * ys[1]) / determinant
alpha1 = (-b * ys[0] + a * ys[1]) / determinant

x = 0.5
prediction = (
    alpha0 * math.exp(-gamma * (x - xs[0]) ** 2)
    + alpha1 * math.exp(-gamma * (x - xs[1]) ** 2)
)

print(round(k01, 3))
print(round(alpha0, 3), round(alpha1, 3))
print(round(prediction, 3))`}</CodeBlock>
    <CodeBlock language="output">{`0.368
-0.342 1.024
0.531`}</CodeBlock>
    <Prose>
      Real implementations solve the system using stable linear algebra rather than forming an explicit inverse. Tune gamma and regularisation inside cross-validation; choosing them after repeatedly inspecting the test set turns the test set into training data.
    </Prose>

    <H2>7. Strengths, limits, and practical use</H2>
    <Prose>
      Kernels excel on small-to-medium datasets with a meaningful similarity notion, structured inputs, or limited labels. They offer elegant regularisation and often strong baselines. Their Gram matrix needs O(n²) memory and typical exact solves cost O(n³), which limits scale. Approximate methods such as random Fourier features, Nyström approximations, and iterative solvers can help, but a learned representation may be more practical for large unstructured data.
    </Prose>
    <Prose>
      In high dimensions, naive Euclidean distances can concentrate and RBF similarity can become uninformative. Inspect feature scales, validate against linear and tree baselines, and use kernels designed for the structure you actually have. An RKHS guarantee does not make a bad input representation informative.
    </Prose>
    <Callout label="Practice">
      Fit kernel ridge regression to noisy samples of a smooth one-dimensional curve. Sweep gamma and regularisation on validation data. Which setting underfits, which overfits, and how does the RKHS norm change? Then compare with a linear ridge-regression baseline.
    </Callout>
  </div>,
};

export default content;
