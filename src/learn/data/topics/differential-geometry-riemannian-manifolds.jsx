import { Callout, Code, CodeBlock, H2, Prose } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";

const content = {
  title: "Differential Geometry & Riemannian Manifolds",
  readTime: "~45 min",
  content: () => <div>
    <H2>1. Why flat-space intuition can be wrong</H2>
    <Prose>
      Standard vectors live in flat Euclidean space, where adding two vectors and measuring a straight line are globally meaningful. Many ML objects are constrained or curved: a unit direction lies on a sphere, a rotation lies on a rotation group, a covariance matrix must remain positive definite, and probability distributions have geometry induced by information. Treating these objects as unconstrained vectors can create invalid updates or distances that do not match the problem.
    </Prose>
    <Prose>
      Differential geometry supplies local linear approximations for curved spaces. Riemannian geometry adds a rule for measuring lengths and angles locally, letting us define gradients, shortest paths, and optimisation steps that respect the constraint.
    </Prose>

    <H2>2. Manifolds are locally flat, globally curved</H2>
    <Prose>
      A d-dimensional manifold looks like R^d in a sufficiently small neighbourhood, even if it curves globally. The surface of a sphere is locally like a plane but cannot be flattened onto one plane without distortion everywhere. At a point x, the tangent space T_x M is the linear space of allowed instantaneous directions from x.
    </Prose>
    <MathBlock>{`T_x\\mathcal{M}=\\text{the local linear space of directions at }x`}</MathBlock>
    <Prose>
      On the unit sphere, tangent vectors are perpendicular to the current point: <Code>x^T v = 0</Code>. Moving in an arbitrary ambient direction can leave the sphere, so optimisation must first use a tangent direction and then return to the manifold.
    </Prose>

    <H2>3. A Riemannian metric defines meaningful distance</H2>
    <Prose>
      A Riemannian metric assigns an inner product to each tangent space. It can vary from point to point, which is what allows curved geometry. The metric defines vector length, angle, path length, and therefore geodesics: locally shortest paths that generalise straight lines.
    </Prose>
    <MathBlock>{`\\langle u,v\\rangle_x=u^\\mathsf{T}G(x)v, \\qquad d(x,y)=\\inf_{\\gamma:x\\to y}\\int\\|\\dot{\\gamma}(t)\\|_{\\gamma(t)}dt`}</MathBlock>
    <Prose>
      On a unit sphere, the geodesic distance between unit vectors x and y is <Code>arccos(x^T y)</Code>, not their Euclidean chord length. Orthogonal directions have spherical distance pi/2. For nearby points the two distances are similar; for far-apart points the distinction matters.
    </Prose>

    <H2>4. Riemannian gradients stay within the allowed directions</H2>
    <Prose>
      The Euclidean gradient tells us how a function changes in ambient coordinates. The Riemannian gradient is the tangent vector that represents that change under the manifold metric. On an embedded manifold, a common first step is to project the Euclidean gradient onto the tangent space. A retraction then maps a small tangent step back onto the manifold; it is often a cheap approximation to the exact exponential map.
    </Prose>
    <MathBlock>{`\\operatorname{grad}_{\\mathcal{M}}f(x)=G(x)^{-1}\\nabla f(x), \\qquad P_x(g)=g-(x^\\mathsf{T}g)x\\quad\\text{on the unit sphere}`}</MathBlock>
    <Prose>
      For the sphere, normalising after a small tangent step is a common retraction. It respects the unit-norm constraint, whereas an ordinary Euclidean update generally does not.
    </Prose>
    <CodeBlock language="python">{`import math

x = (1.0, 0.0)       # point on the unit circle
gradient = (1.0, 2.0) # Euclidean gradient at x

dot = sum(a * b for a, b in zip(x, gradient))
tangent_gradient = tuple(g - dot * xi for g, xi in zip(gradient, x))
eta = 0.1
candidate = tuple(xi - eta * gi for xi, gi in zip(x, tangent_gradient))
norm = math.sqrt(sum(value * value for value in candidate))
next_x = tuple(value / norm for value in candidate)  # retraction by normalisation

print(tangent_gradient)
print(tuple(round(value, 3) for value in next_x))
print(round(math.acos(0), 3))  # distance between orthogonal unit vectors`}</CodeBlock>
    <CodeBlock language="output">{`(0.0, 2.0)
(0.981, -0.196)
1.571`}</CodeBlock>
    <Prose>
      The projected gradient removes the radial component that would point off the circle. The new point remains unit length after retraction. The code is a geometric optimisation step, not a general constrained optimiser; other manifolds need their own projection and retraction rules.
    </Prose>

    <H2>5. Exponential maps, logarithm maps, and transport</H2>
    <Prose>
      The exponential map starts at x with tangent vector v and follows the geodesic for one unit of time, landing on the manifold. The logarithm map reverses this locally, turning a nearby point into a tangent displacement. Parallel transport moves a tangent vector along a curve while preserving its geometric meaning. These operations let us compare directions at different points, average manifold-valued data, and define optimisation momentum without pretending all tangent spaces are literally the same plane.
    </Prose>
    <MathBlock>{`\\operatorname{Exp}_x:T_x\\mathcal{M}\\to\\mathcal{M}, \\qquad \\operatorname{Log}_x(y)\\in T_x\\mathcal{M}`}</MathBlock>
    <Prose>
      Exact maps can be expensive or unavailable. Retractions are popular because they match the exponential map to first order and are often sufficient for optimisation, but their approximation error should be tested when steps are large or curvature is strong.
    </Prose>

    <H2>6. Where geometry enters ML</H2>
    <Prose>
      Normalised embeddings and directional statistics live on spheres. Covariance descriptors and Gaussian distributions connect to the manifold of symmetric positive-definite matrices. Hyperbolic embeddings can represent hierarchical data with lower distortion than Euclidean space in some settings. Natural-gradient methods use a Riemannian metric based on information geometry. Geometric deep learning exploits symmetries and non-Euclidean domains such as graphs and meshes.
    </Prose>
    <Callout accent="green" label="Use geometry when the constraint is real">
      A manifold-aware model can preserve rotations, positivity, normalisation, or intrinsic distance that an unconstrained model violates. It does not help merely because the word "manifold" sounds advanced. Compare with a well-tuned Euclidean baseline and validate on the downstream task.
    </Callout>

    <H2>7. A careful geometric-modelling workflow</H2>
    <Prose>
      Identify whether data truly lives on a constrained space or only happens to be represented in one. Choose the metric based on the domain's invariances and cost of distortion. Use stable library implementations for manifold operations, test that updates stay valid, and monitor conditioning near singular or boundary regions. Distinguish a representation's geometry from causal structure: a low-distortion embedding can still encode historical bias or omit decision-relevant variables.
    </Prose>
    <Callout label="Practice">
      Consider unit-normalised text embeddings used for retrieval. Compare cosine similarity, Euclidean distance after normalisation, and raw Euclidean distance. Which quantities are equivalent on the sphere, which are not, and why might normalising before optimisation or retrieval change the model's behaviour?
    </Callout>
  </div>,
};

export default content;
