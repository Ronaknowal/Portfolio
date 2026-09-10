import { Callout, Code, CodeBlock, H2, Prose } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";

const content = {
  title: "Numerical Methods (Finite Differences, Quadrature, Root Finding)",
  readTime: "~44 min",
  content: () => <div>
    <H2>1. Why approximation is a first-class engineering concern</H2>
    <Prose>
      Most useful mathematical problems do not have a convenient closed-form answer. We approximate derivatives to solve differential equations, integrals to compute expectations, roots to calibrate models, and linear systems to fit parameters. Numerical methods provide the algorithms—but their real subject is error: what approximation is being made, how fast does it improve, and when does floating-point arithmetic or an ill-conditioned problem make a plausible answer untrustworthy?
    </Prose>
    <Prose>
      A numerical result deserves a unit, a tolerance, and a verification story. "The code returned a number" is not evidence that it solved the intended mathematical problem.
    </Prose>

    <H2>2. Separate truncation error, round-off, conditioning, and stability</H2>
    <Prose>
      Truncation error comes from replacing an infinite or exact expression with a finite approximation. Round-off comes from finite-precision numbers. Conditioning belongs to the problem: a small input perturbation may cause a large change in the true answer. Stability belongs to the algorithm: it may or may not amplify small computational errors. A stable algorithm cannot make an inherently ill-conditioned problem well-conditioned, but an unstable algorithm can ruin an otherwise manageable one.
    </Prose>
    <MathBlock>{`\\text{forward error}=|\\hat{x}-x|, \\qquad \\text{relative error}=\\frac{|\\hat{x}-x|}{|x|}`}</MathBlock>
    <Callout accent="green" label="Residual is not always error">
      For a linear system A x = b, a small residual <Code>||A x_hat - b||</Code> means the computed vector nearly satisfies the equation. If A is ill-conditioned, x_hat can still be far from the true solution. Check a condition estimate and test sensitivity, not only the residual.
    </Callout>

    <H2>3. Root finding solves equations of the form f(x) = 0</H2>
    <Prose>
      Bisection starts with an interval where a continuous function changes sign and repeatedly halves it. It is slow but guaranteed under that assumption. Newton's method uses a tangent-line update and can converge very quickly near a suitable root, but it can diverge, land on the wrong root, or fail when the derivative is small. Robust production solvers often combine a bracketing method with Newton-like steps.
    </Prose>
    <MathBlock>{`x_{k+1}=x_k-\\frac{f(x_k)}{f'(x_k)}\\quad\\text{(Newton)}, \\qquad |x_k-x^*|\\leq\\frac{b-a}{2^{k+1}}\\quad\\text{(bisection)}`}</MathBlock>
    <Prose>
      To approximate sqrt(2), solve x² - 2 = 0 on the bracket [1, 2]. Thirty bisection iterations give the result below. The bracket itself is valuable: it provides a certificate that the root remains inside under continuity.
    </Prose>

    <H2>4. Finite differences approximate derivatives from nearby values</H2>
    <Prose>
      The forward difference uses one nearby evaluation; the central difference uses symmetric evaluations and cancels the leading error term. For sufficiently smooth f, forward difference has O(h) truncation error and central difference has O(h²). Smaller h is not always better: subtracting almost equal floating-point values magnifies round-off error.
    </Prose>
    <MathBlock>{`f'(x)\\approx\\frac{f(x+h)-f(x)}{h}, \\qquad f'(x)\\approx\\frac{f(x+h)-f(x-h)}{2h}`}</MathBlock>

    <H2>5. Quadrature approximates area and expectation</H2>
    <Prose>
      Numerical integration, or quadrature, approximates an integral using weighted function evaluations. The trapezoidal rule joins values with straight lines. It is simple and second-order accurate for smooth functions on a uniform grid; Simpson's rule and adaptive quadrature can be more efficient when the function is smooth or unevenly difficult across the interval.
    </Prose>
    <MathBlock>{`\\int_a^b f(x)dx\\approx h\\left(\\tfrac12f(a)+\\sum_{i=1}^{n-1}f(a+ih)+\\tfrac12f(b)\\right), \\qquad h=\\frac{b-a}{n}`}</MathBlock>
    <CodeBlock language="python">{`import math

# Bisection: x^2 - 2 = 0 on [1, 2]
lo, hi = 1.0, 2.0
for _ in range(30):
    mid = (lo + hi) / 2
    if mid * mid < 2:
        lo = mid
    else:
        hi = mid
root = (lo + hi) / 2

# Central difference for d/dx sin(x) at x = 1
x, h = 1.0, 0.1
central_derivative = (math.sin(x + h) - math.sin(x - h)) / (2 * h)

# Trapezoidal rule for integral_0^pi sin(x) dx
n, a, b = 4, 0.0, math.pi
step = (b - a) / n
values = [math.sin(a + i * step) for i in range(n + 1)]
trapezoid = step * (0.5 * values[0] + sum(values[1:-1]) + 0.5 * values[-1])

print(round(root, 6))
print(round(central_derivative, 6), round(math.cos(1), 6))
print(round(trapezoid, 6), 2.0)`}</CodeBlock>
    <CodeBlock language="output">{`1.414214
0.539402 0.540302
1.896119 2.0`}</CodeBlock>
    <Prose>
      The displayed errors are expected from deliberately coarse settings. Refine the bisection tolerance, reduce h until round-off begins to dominate, and increase quadrature panels or use adaptivity. A convergence table across multiple step sizes is more informative than one result.
    </Prose>

    <H2>6. Linear algebra is numerical method territory too</H2>
    <Prose>
      Many ML and scientific tasks reduce to solving A x = b or finding eigenvalues. Avoid explicitly computing an inverse for routine solves; use factorizations such as Cholesky, QR, or LU that match matrix structure. Iterative methods such as conjugate gradient exploit sparse symmetric positive-definite systems. Preconditioning can transform a poorly scaled solve into one that converges in practical time.
    </Prose>
    <Prose>
      Automatic differentiation is usually preferable to finite differences for gradients through code because it applies the chain rule accurately to machine precision. Finite differences remain valuable as an independent gradient check, provided h is chosen carefully and the function is deterministic enough to compare.
    </Prose>

    <H2>7. A reliable numerical workflow</H2>
    <Prose>
      Define the mathematical target and acceptable error in domain units. Scale variables so values are not wildly different in magnitude. Choose an algorithm whose assumptions match the function or matrix. Record stopping criteria based on residual, change, and iteration limit; no single criterion is sufficient for every problem. Compare against an analytic solution, a higher-precision computation, or refinement studies where possible. Log failures and non-convergence rather than converting them silently into ordinary outputs.
    </Prose>
    <Callout label="Practice">
      Repeat the central-difference calculation for h values 10^-1 through 10^-12. Plot or tabulate its absolute error against cos(1). At what scale does decreasing h stop helping, and why? Then compare bisection and Newton on a root with a poor Newton starting point.
    </Callout>
  </div>,
};

export default content;
