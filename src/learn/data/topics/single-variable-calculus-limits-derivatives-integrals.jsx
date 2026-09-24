import { Callout, Code, H2, H3, Prose } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { Checkpoint, LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample';
import { ChainLocalChangeLab, ExponentialRateLab, ExtremaCandidatesLab, FundamentalStripFigure, ImproperIntegralLab, LimitGateLab, MotionJourneyFigure, MotionRateLab, ProductIncrementFigure, SignedAccumulationLab, TaylorErrorLab, WeightedRodFigure } from '../../components/lesson-labs/SingleVariableCalculusLabs.jsx';
import { singleVariableCalculusExamples } from '../single-variable-calculus-examples.js';
function CalculusExample({
  id,
  children
}) {
  const example = singleVariableCalculusExamples.find(entry => entry.id === id);
  return <><Prose><strong>Before running:</strong> {example.question}</Prose><RunnableExample example={example}>{children}</RunnableExample></>;
}
function Practice({
  title,
  question,
  hint,
  children
}) {
  return <section className="calculus-practice"><H3>{title}</H3><Prose>{question}</Prose>
    <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>
    <details><summary>Show explained solution</summary>{children}</details>
  </section>;
}
export default {
  title: 'Single-Variable Calculus: Limits, Derivatives & Integrals',
  readTime: '~100 min read + 2–3 hours practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot single-calculus-lesson">
    <LessonIntro prerequisites="Use functions, domains and equations from Algebra; quantified statements from Sets & Logic; and radians and signed trigonometric coordinates from Geometry. We refresh the needed meanings before using them. The Python programs are optional; the diagrams and investigations work here." sections={[['1-ask-three-questions-about-one-journey', 'Position, rate and change'], ['2-make-approaching-precise', 'Limits and continuity'], ['3-extract-the-local-linear-rate', 'Derivatives and their limits'], ['4-build-rules-from-changing-operations', 'Products, chains and inverses'], ['5-connect-local-rates-to-global-decisions', 'Mean values and extrema'], ['6-accumulate-signed-contributions', 'Riemann sums and distance'], ['7-prove-the-bridge-between-rate-and-total', 'Fundamental theorem'], ['8-reverse-operations-to-integrate', 'Integration methods'], ['9-complete-the-exponential-growth-story', 'Logarithms and exponential rates'], ['10-keep-the-remainder-with-the-approximation', 'Taylor and numerical limits'], ['11-check-what-happens-at-the-edge', 'Improper integrals and limit rules'], ['12-transfer-the-same-idea-to-new-quantities', 'Work, mass and rate laws'], ['13-practise-with-changed-assumptions', 'Independent practice'], ['14-connect-the-pieces-and-continue', 'Readiness and resources']]}>A changing quantity has a current value, a local rate and a total change over an interval. Calculus explains how these descriptions fit together. Start with a small journey, then derive the tools needed to make predictions, accumulate contributions and recognize when a tempting calculation is invalid. Read the core route in order; the marked proof and extension branches give additional depth.</LessonIntro>

    <H2>1. Ask three questions about one journey</H2>
    <Prose>A small object travels along a straight track. At each time t, its <strong>position</strong> s(t) tells us where it is relative to a chosen origin. A position can increase or decrease; the object can return to a place it visited earlier. We will use the instructional model s(t)=t³−6t²+9t=t(t−3)² on 0≤t≤4. Time is measured in seconds and position in metres. The polynomial uses the numerical time in that unit; its coefficients carry the corresponding metre/second powers. This is a declared mathematical example, not measured motion.</Prose>
    <LessonTable caption="One position rule, four observations" headers={['Time (s)', 'Position (m)', 'What that says']} rows={[['0', '0', 'Start at the origin'], ['1', '4', 'Four metres to the positive side'], ['3', '0', 'Back at the origin'], ['4', '4', 'Four metres to the positive side again']]} />
    <Prose>Between t=1 and t=3, the position change is 0−4=−4 m. Divide by the elapsed 2 s to obtain an <strong>average velocity</strong> of −2 m/s. The minus sign describes direction. <strong>Speed</strong> is the nonnegative magnitude of velocity; it answers how fast, without the direction.</Prose>
    <MathBlock>{String.raw`\begin{gathered}\text{average velocity on }[a,b]\\
      =\frac{s(b)-s(a)}{b-a},\qquad a<b.
    \end{gathered}`}</MathBlock>
    <Prose>A graph of position against time makes this quotient the slope of a line joining two observations—a <strong>secant</strong>. It is an interval summary. To ask for velocity at one instant, bring the second observation closer. At t=2, s(2)=2. At t=2.5, s(2.5)=0.625, so the average velocity over that half-second is −2.75 m/s. The instantaneous answer will be −3 m/s, but obtaining it requires explaining what “closer” guarantees.</Prose>
    <MotionRateLab />
    <Prose>The total position change over the whole journey is only 4 m. That does not say how far the object travelled: forward and backward motion can cancel in a signed total. We will derive the turning times before using the following three-leg decomposition.</Prose>
    <MotionJourneyFigure />
    <Callout accent="green" label="Three different questions">The value s(t) locates the object. A derivative will describe its local rate of change. An integral will add the contributions over an interval. Keep the quantity and its units attached to each question.</Callout>
    <Prose>For an optional program, save one complete block as <Code>calculus_check.py</Code> and run <Code>python calculus_check.py</Code> with Python 3.12 or a compatible Python 3. Every block supplies its own imports and inputs. Standard-library <Code>Fraction</Code> gives exact rational arithmetic; decimal transcendental outputs use floating-point arithmetic and are labelled accordingly. No program runs automatically in your browser.</Prose>
    <CalculusExample id="motion-secants"><Prose>The fractions expose the entire finite error. At t=2 the quotient is −3+h², so either sign of h gives a value approaching −3. The equal left/right finite slopes here arise from this particular cubic at this point; they are not a rule for every curve.</Prose></CalculusExample>

    <H2>2. Make approaching precise</H2>
    <Prose>A <strong>limit</strong> describes the values a function approaches near an input, whether or not its value at that input agrees. For f(x)=x², inputs near 2 have outputs near 4. Changing only f(2) to 6 leaves every nearby, different input unchanged. The limit is still 4; the new function is not continuous at 2.</Prose>
    <Prose>The notation below says: make the output error smaller than any requested positive tolerance by restricting the input to a sufficiently small, punctured neighborhood. “Punctured” removes the center x=2. The symbol ε, epsilon, is the requested output accuracy; δ, delta, is an input radius we choose after seeing that request.</Prose>
    <MathBlock>{String.raw`\begin{gathered}\lim_{x\to a}f(x)=L\quad\text{means}\\
      \text{for every }\varepsilon>0\text{ there is }\delta>0\\
      \text{such that }0<|x-a|<\delta\\
      \text{implies }|f(x)-L|<\varepsilon.
    \end{gathered}`}</MathBlock>
    <Prose>The order matters: one chosen δ must work for <em>every</em> allowed nearby x, not merely for a favorite sequence of test points. δ can depend on ε and the fixed point a. A sampled graph illustrates this promise; a proof must justify it between the samples too.</Prose>
    <H3>Prove one promise completely</H3>
    <Prose>For x² near 2, factor the output error as |x²−4|=|x−2||x+2|. First keep |x−2|&lt;1. Then 1&lt;x&lt;3 and |x+2|&lt;5. Given any ε&gt;0, choose δ=min(1,ε/5). Every x with 0&lt;|x−2|&lt;δ satisfies |x²−4|&lt;5δ≤ε. That is a proof for all such x. It need not find the largest possible δ; a sufficient radius establishes the limit.</Prose>
    <LimitGateLab />
    <CalculusExample id="limit-guarantee"><Prose>The exact computations confirm several instances of the bound and exhibit a counterexample to an overly large radius. The general factorization supplies the proof; the program alone does not cover every real input.</Prose></CalculusExample>
    <H3>One-sided behavior and continuity</H3>
    <Prose>A <strong>left-hand limit</strong> approaches through x&lt;a; a right-hand limit uses x&gt;a. A two-sided finite limit exists precisely when both sides approach the same finite number, assuming both sides belong to the local domain. In the jump example, the left limit is 4 and the right limit is 6. Defining f(2) cleverly cannot repair that disagreement. At a domain endpoint, use the permitted one-sided approach.</Prose>
    <Prose>Limits may also fail because values grow without bound or keep oscillating. For 1/x near zero the two signs grow in opposite directions; there is no finite two-sided limit. Writing an infinite limit is a statement about eventually exceeding every chosen magnitude, not treating infinity as an ordinary value that can be substituted into arithmetic.</Prose>
    <Prose><strong>Limit laws</strong> let us add and multiply finite limits, and divide them when the denominator's limit is nonzero. They follow by bounding the corresponding errors. Cancellation can reveal a limit at an excluded point: (x²−4)/(x−2)=x+2 for x≠2, so the limit is 4, while the original expression remains undefined at 2. “0/0” after a direct substitution is an unresolved form, not a numerical answer.</Prose>
    <Prose>A function is <strong>continuous at a</strong> when f(a) exists and the limit through its domain equals f(a). Two useful existence theorems are stated here: a continuous function on a closed bounded interval attains a minimum and a maximum; and it takes every value between its endpoint values. These are the <strong>extreme value theorem</strong> and <strong>intermediate value theorem</strong>. Their full completeness foundations belong to Real Analysis. The hypotheses do work: 1/x on the open interval (0,1) has no attained minimum or maximum; a jump can skip intermediate values.</Prose>
    <Checkpoint prompt="Keep f(x)=x² for x≠2, but define f(2)=−10. What are the limit at 2, continuity at 2 and the effect on the nearby guarantee?"><Prose>The limit remains 4 and every punctured-neighborhood guarantee remains unchanged. Continuity fails because f(2)=−10 differs from the limit. A point's value and the behavior nearby are different pieces of information.</Prose></Checkpoint>

    <H2>3. Extract the local linear rate</H2>
    <Prose>For an interior point a, a <strong>derivative</strong> is the finite limit of the secant quotient as its nonzero input increment h approaches zero. The notation f′(a), read “f prime at a,” denotes one number. The function f′ assigns that number at each point where it exists. The notation df/dx names the same derivative operation; its definition is not a division with dx=0.</Prose>
    <MathBlock>{String.raw`f'(a)=\lim_{h\to0}\frac{f(a+h)-f(a)}h.`}</MathBlock>
    <Prose>For f(x)=x², expansion gives f(a+h)−f(a)=2ah+h². With h≠0, divide by h to get 2a+h; its limit is 2a. The derivative is exactly 2a, while 2a+h is an interval approximation. At a=2 and h=0.1, the derivative predicts a change of 0.4; the true change is 0.41.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      f(a+h)&=f(a)+f'(a)h+r(h),\\
      \frac{r(h)}h&\longrightarrow0.
    \end{aligned}`}</MathBlock>
    <Prose>This is the derivative's most useful meaning: one <strong>linear approximation</strong> has an error that becomes negligible relative to the input change. An error merely approaching zero would be too weak. The wrong prediction f(a+h)≈f(a) for a nonflat line has an error tending to zero, yet misses its entire first-order change. For the square, r(h)=h² and r(h)/h=h→0.</Prose>
    <Prose>Expanding our motion polynomial gives the full change below. The first coefficient is velocity; the next two terms explain its finite-increment error.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
      s(t+h)-s(t)\\
      =(3t^2-12t+9)h\\
      {}+(3t-6)h^2+h^3,\\
      s'(t)=3(t-1)(t-3),\\
      s''(t)=6t-12.
    \end{gathered}`}</MathBlock>
    <Prose>The <strong>second derivative</strong> differentiates the first derivative. Here it is acceleration, with units m/s². The derivative's units are output units divided by input units. Higher derivatives repeat the operation when the required derivatives exist.</Prose>
    <H3>Know when the local model fails</H3>
    <Prose>For f(x)=|x| at zero, the quotient |h|/h is 1 for h&gt;0 and −1 for h&lt;0. Both sides are perfectly clear, but they disagree: there is no derivative at the corner. For the real cube-root function at zero, the quotient is |h|^(−2/3), which grows without bound. A vertical tangent can be meaningful geometrically, but it is not a finite derivative under our definition.</Prose>
    <Prose>Differentiability implies continuity: f(a+h)−f(a) is h times a quotient approaching a finite number, so the difference approaches zero. Continuity does not imply differentiability, as |x| shows. Nor does a tangent have to touch without crossing: at the cubic's inflection point t=2, its graph crosses its tangent while the limiting slope is well-defined.</Prose>

    <H2>4. Build rules from changing operations</H2>
    <Prose>Derivative rules let us reuse local reasoning when functions are assembled from smaller operations. Constants have zero change. A sum's change is the sum of its changes, so (f+g)′=f′+g′ and (cf)′=cf′ for constant c. Each rule requires the component derivatives at the point under discussion.</Prose>
    <H3>A product changes along two edges</H3>
    <Prose>When both factors change, (a+Δa)(b+Δb)−ab=bΔa+aΔb+ΔaΔb. If Δa and Δb are each proportional to h to first order, their product contributes only at second order. Divide by h and take the limit: the two edge terms survive.</Prose>
    <ProductIncrementFigure />
    <MathBlock>{String.raw`(fg)'=f'g+fg'.`}</MathBlock>
    <Prose>For a formal check, write the product difference as f(a+h)[g(a+h)−g(a)]+g(a)[f(a+h)−f(a)]. Divide by h. Differentiability gives continuity of f, so f(a+h)→f(a); the quotient limits give the stated rule. Repeated products then prove (xⁿ)′=nxⁿ⁻¹ for positive integers n by induction. For a reciprocal, directly subtract 1/g(a+h)−1/g(a); when g(a)≠0, its derivative is −g′(a)/g(a)². This gives negative integer powers and the quotient rule.</Prose>
    <MathBlock>{String.raw`\left(\frac fg\right)'=\frac{f'g-fg'}{g^2},\qquad g\ne0.`}</MathBlock>
    <H3>A composition passes the change onward</H3>
    <Prose>Let u=g(x) be an intermediate quantity and y=f(u) the output. A small input change h produces approximately g′(x)h in u. The outer function turns that into approximately f′(g(x))g′(x)h. Thus the <strong>chain rule</strong> multiplies the two local sensitivities, with the outer derivative evaluated at the actual intermediate value.</Prose>
    <MathBlock>{String.raw`\frac{d}{dx}f(g(x))=f'(g(x))g'(x).`}</MathBlock>
    <ChainLocalChangeLab />
    <details className="lesson-deeper"><summary>Why the chain proof still works when the inner derivative is zero</summary>
      <Prose>Write Δu=g(x+h)−g(x)=g′(x)h+r_g(h), with r_g(h)/h→0. Differentiability of f means f(u+v)−f(u)=f′(u)v+vE(v), where E(v)→0 and we set E(0)=0. Substitute v=Δu and divide by h. The factor Δu/h tends to g′(x), so it is bounded; E(Δu) tends to zero by continuity of g. Their product vanishes even if g′(x)=0 or some increments give Δu=0. No illegal division by the inner change is needed.</Prose>
    </details>
    <CalculusExample id="local-operations"><Prose>The x=−1/3 case makes u=0. The first-order output prediction is zero, but the positive square remainder remains. The product example also uses a negative increment, showing why the algebra is more general than a picture of added positive strips.</Prose></CalculusExample>
    <H3>Trigonometric, inverse and implicit derivatives</H3>
    <Prose>Trigonometric derivatives use angles measured in <strong>radians</strong>. The unit-circle triangle, sector and tangent-triangle areas give sin h≤h≤tan h for 0&lt;h&lt;π/2. Dividing and using symmetry yields cos h≤sin h/h≤1 near zero, hence sin h/h→1. Also 1−cos h=2sin²(h/2), so |(cos h−1)/h|≤|h|/2→0. The angle-addition identities now derive, rather than guess, the two basic rules:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
      \frac{\sin(x+h)-\sin x}{h}\\
      =\sin x\frac{\cos h-1}{h}\\
      {}+\cos x\frac{\sin h}{h},\\
      (\sin x)'=\cos x,\\
      (\cos x)'=-\sin x.
    \end{gathered}`}</MathBlock>
    <Prose>The squeeze uses the circle's continuous components at zero, visible from their geometry. It would acquire a π/180 scale factor if the numerical input were degrees. Quotient and chain rules give (tan x)′=sec²x wherever cos x≠0, and d[sin(g(x))]/dx=cos(g(x))g′(x).</Prose>
    <Prose>An inverse function reverses input and output changes, so its local slope is the reciprocal of the original nonzero slope. Precisely, let g be continuous and strictly monotone near a, differentiable there with g′(a)≠0. Its inverse is continuous near g(a). For y=g(x), the inverse difference quotient is (x−a)/(g(x)−g(a)); as y→g(a), continuity of the inverse gives x→a, so the quotient tends to 1/g′(a). This establishes inverse differentiability instead of assuming it.</Prose>
    <MathBlock>{String.raw`(g^{-1})'(g(a))=\frac1{g'(a)}.`}</MathBlock>
    <Prose>On the principal branch, (arcsin x)′=1/√(1−x²) for −1&lt;x&lt;1 because the cosine of the inverse angle is positive. Similarly, (arctan x)′=1/(1+x²) for every real x. For x²+y²=25, differentiate along one chosen function branch y(x): 2x+2y y′=0, so y′=−x/y when y≠0. At (3,4), the slope is −3/4. The equation alone has two branches; choosing one matters, and dividing by y at y=0 is invalid.</Prose>

    <H2>5. Connect local rates to global decisions</H2>
    <Prose>A derivative tells us about a point. To draw a conclusion across an interval, state the hypotheses that connect those points. The <strong>mean value theorem</strong> says that a function continuous on [a,b] and differentiable on (a,b) has some interior c whose derivative equals its endpoint average slope.</Prose>
    <MathBlock>{String.raw`\begin{gathered}f'(c)=\frac{f(b)-f(a)}{b-a},\\
      a<c<b.\end{gathered}`}</MathBlock>
    <Prose>This is an existence statement, not a promise to locate c uniquely. If every derivative is positive, every endpoint difference over a positive interval is positive: the function is increasing. If the derivative is zero everywhere on an interval, the function is constant there. If |f′|≤M throughout the interval, then |f(b)−f(a)|≤M|b−a|. A local derivative value at just one point cannot substitute for that interval-wide bound.</Prose>
    <details className="lesson-deeper"><summary>Prove the mean value theorem from an interior extremum</summary>
      <Prose>At an interior local maximum of a differentiable function, small positive-increment quotients are nonpositive and small negative-increment quotients are nonnegative. Their common limit must be zero; a local minimum works with reversed signs. This is Fermat's necessary condition. If a continuous function has equal endpoint values, it either is constant or its attained maximum or minimum differs from that shared value and therefore occurs inside. The extreme value theorem supplies attainment; Fermat supplies a zero derivative. This is Rolle's theorem. For general f, subtract the secant line: g(x)=f(x)−f(a)−m(x−a), where m=[f(b)−f(a)]/(b−a). Then g(a)=g(b)=0. Rolle gives g′(c)=f′(c)−m=0, which proves the result.</Prose>
    </details>
    <H3>Find extrema using the whole allowed domain</H3>
    <Prose>On a closed bounded interval, a continuous function's extrema occur at endpoints or at interior points where its derivative is zero or does not exist. This is a list of <strong>candidates</strong>; compare their values. A stationary point, meaning derivative zero, need not be an extremum. For (t−2)³ the derivative is zero at 2 but the function keeps increasing. A cusp such as |t−2| can attain a minimum where no derivative exists.</Prose>
    <Prose>For the motion cubic, s′=3(t−1)(t−3) is positive before 1, negative between 1 and 3, and positive after 3. Thus the object changes direction at 1 and 3. On [0,4], the values at candidates 0,1,3,4 are 0,4,0,4. Both 0 and 3 attain the global minimum; both 1 and 4 attain the maximum. On a smaller interval, the answer can change completely.</Prose>
    <ExtremaCandidatesLab />
    <CalculusExample id="interval-extrema"><Prose>On [3/2,5/2], neither stationary point lies inside. The function decreases throughout this interval, so its endpoint values determine both extrema. The derivative factorization justifies completeness of the candidate list; the program is not guessing from a sample grid.</Prose></CalculusExample>
    <Prose>For a separate design problem, make a rectangle with perimeter 12 length units. If one side is a, the other is 6−a and the area is A(a)=6a−a². The feasible nondegenerate domain is 0&lt;a&lt;6. A′=6−2a vanishes at a=3, with positive slope before and negative after. The maximum is 9 square units, attained by a square. The boundary values 0 describe degenerate rectangles and explain what happens near the feasible domain's edges.</Prose>
    <Prose>Changing quantities can also be related implicitly. If a circular region has area A=πr², then dA/dt=2πr·dr/dt. At r=3 m and dr/dt=0.2 m/s, its area grows at 1.2π m²/s. The same radius-growth rate produces different area-growth rates at different radii. This is the chain rule with units, not a new formula to memorize.</Prose>

    <H2>6. Accumulate signed contributions</H2>
    <Prose>When velocity is constant over an interval, displacement is velocity×duration. For a changing rate, partition the time interval into smaller pieces, choose a sample time in each piece and add the corresponding rectangles. Their widths have time units, their heights velocity units, and their signed areas have position units.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
      a=t_0<t_1<\cdots<t_n=b,\\
      \Delta t_i=t_i-t_{i-1},\quad \xi_i\in[t_{i-1},t_i],\\
      \text{Riemann sum}=\sum_{i=1}^n f(\xi_i)\Delta t_i.
    \end{gathered}`}</MathBlock>
    <Prose>The sample ξᵢ, read “xi sub i,” may be the left endpoint, midpoint or another point in its subinterval. The <strong>mesh</strong> is the largest subinterval width. A bounded function is Riemann integrable when these sums approach one finite value as the mesh tends to zero, regardless of the chosen tags and sufficiently fine partitions. That value is the <strong>definite integral</strong>, written ∫ from a to b of f(t) dt. Increasing the number of pieces while leaving one large piece unchanged does not ensure shrinking mesh.</Prose>
    <Prose>Every continuous function on a closed bounded interval is integrable. The key reason is that its oscillation on uniformly short intervals becomes uniformly small. Lower and upper rectangle sums differ by at most interval length times that maximum local oscillation, forcing all tagged sums toward the same value. This uses uniform continuity on a compact interval; Real Analysis develops the underlying existence argument.</Prose>
    <H3>Take one finite sum all the way to its limit</H3>
    <Prose>For f(x)=x² on [0,2], use n equal pieces of width 2/n and their right endpoints 2j/n. The exact sum is (8/n³)Σj². The identity Σ from j=1 to n of j²=n(n+1)(2n+1)/6 gives:</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      R_n&=\frac{8}{n^3}\frac{n(n+1)(2n+1)}6\\
      &=\frac83+\frac4n+\frac4{3n^2},\\
      \int_0^2 x^2\,dx&=\lim_{n\to\infty}R_n=\frac83.
    \end{aligned}`}</MathBlock>
    <details className="lesson-deeper"><summary>Recover the sum of squares by telescoping</summary>
      <Prose>The differences j³−(j−1)³=3j²−3j+1 sum to n³ because all intermediate cubes cancel. Therefore n³=3Σj²−3Σj+n. Substitute Σj=n(n+1)/2, which follows by pairing the sequence with its reverse, and solve for Σj². This proves the identity used above rather than inferring it from several computed sums.</Prose>
    </details>
    <CalculusExample id="square-riemann-limit"><Prose>Both positive error terms vanish as n grows. The limit equals 8/3; no displayed finite rectangle sum is being called that exact area.</Prose></CalculusExample>
    <H3>Signed change and total travel require different sums</H3>
    <Prose>A negative rate contributes a negative signed area. For our motion, integrating velocity adds displacement; integrating its magnitude adds distance. Split at the turning times 1 and 3 when evaluating the latter. The three signed legs are +4,−4,+4 m. They sum to 4 m, while their magnitudes sum to 12 m. The absolute value of the net displacement is only 4 m.</Prose>
    <SignedAccumulationLab />
    <CalculusExample id="signed-rectangles"><Prose>Three panels expose errors in both approximations. The four- and eight-panel absolute sums happen to equal 12 even though their signed sums are still inaccurate. That cancellation depends on this function, these intervals and the chosen grids. Agreement in one summary does not certify the whole approximation mechanism.</Prose></CalculusExample>
    <Prose>Linearity of finite sums passes to integrals: constants can be factored out and sums integrated term by term when the integrals exist. Integrals add across adjacent intervals. Reversing bounds negates a signed integral, and equal bounds give zero. These are orientation and accumulation rules, not statements that geometric areas become negative objects.</Prose>

    <H2>7. Prove the bridge between rate and total</H2>
    <Prose>Fix a continuous rate f on [a,b]. For each endpoint x, define A(x)=∫ from a to x of f(t) dt. A fixed endpoint gives one number; allowing x to vary gives an <strong>accumulation function</strong>. Moving x by h adds the strip between x and x+h. Its average height should approach f(x) as the strip narrows.</Prose>
    <FundamentalStripFigure />
    <MathBlock>{String.raw`\begin{gathered}
      \frac{A(x+h)-A(x)}h-f(x)\\
      =\frac1h\int_x^{x+h}[f(t)-f(x)]\,dt.
    \end{gathered}`}</MathBlock>
    <Prose>By continuity at x, every height f(t) in a sufficiently small strip differs from f(x) by less than ε. The absolute value of the integral on the right is then at most ε|h|. Dividing by |h| bounds the average-height error by ε. The same argument works for negative h after accounting for reversed bounds. Hence A′(x)=f(x) at interior points. This is the first part of the <strong>fundamental theorem of calculus</strong>.</Prose>
    <Prose>Now suppose F is an <strong>antiderivative</strong> of f: F′=f on the interval, with continuous endpoint values. The derivative of F−A is zero. The mean value theorem says that difference is constant on the interval, and continuity extends it to the endpoints. Subtracting endpoint values cancels the constant and gives the second part:</Prose>
    <MathBlock>{String.raw`\int_a^b f(t)\,dt=F(b)-F(a).`}</MathBlock>
    <Prose>This is why an accumulation can be calculated by finding a function with the required derivative. It is a theorem connecting two previously different constructions, not a definition that makes their relationship automatic. All antiderivatives on one interval differ by a constant C. In an indefinite integral, +C records that family; in a definite difference, the same constant cancels. On disconnected domains, different components can have different constants.</Prose>
    <H3>Keep the hypothesis with the conclusion</H3>
    <Prose>Continuity of the rate is a sufficient condition for the pointwise derivative result. It cannot be casually replaced with “integrable.” Consider a step that is 0 for t&lt;0 and 1 for t≥0. Its accumulation is flat to the left and has slope 1 to the right, so it has a corner at zero. In fact no everywhere differentiable antiderivative can have exactly this step as its derivative: zero derivative forces a constant left branch, derivative one forces a line of slope one on the right, and continuity at zero joins them into the same corner. The finite integral still exists.</Prose>
    <Prose>For a continuous f and differentiable endpoint functions u(x),v(x), apply the chain rule to two accumulations from the same fixed base:</Prose>
    <MathBlock>{String.raw`\begin{gathered}\frac{d}{dx}\int_{u(x)}^{v(x)}f(t)\,dt\\
      =f(v(x))v'(x)-f(u(x))u'(x).
    \end{gathered}`}</MathBlock>
    <Prose>The lower endpoint subtracts a contribution. For ∫ from x to x² of (1+t²) dt, the derivative is (1+x⁴)2x−(1+x²). At x=2, it is 68−5=63. A variable named t inside the integral is a <strong>dummy variable</strong>; replacing that name does not change the integral or create another free input.</Prose>
    <CalculusExample id="moving-bounds"><Prose>The integral value at x=2 is 62/3, while its derivative is 63. They are different quantities. The exact Fraction secants approach the derivative but need not equal it at finite h.</Prose></CalculusExample>
    <Prose>An initial value fixes the constant: knowing velocity v and starting position s(a) gives s(x)=s(a)+∫ from a to x of v(t) dt. The <strong>average value</strong> of a continuous f on [a,b] is its integral divided by b−a. It is the constant height giving the same signed total. Continuity and the intermediate value theorem ensure that some point attains this average height, even though that point need not be the midpoint.</Prose>
    <Checkpoint prompt="If a continuous velocity is negative throughout an interval but increases, becoming less negative, is the accumulation decreasing or increasing? Is its curve bending upward or downward?"><Prose>Its first derivative equals that negative velocity, so the accumulation decreases. Its increasing slope makes the curve convex, bending upward. Wherever velocity is differentiable, acceleration is nonnegative; it need not be strictly positive at every point. For example, v(t)=−1+t³ on a small interval around zero increases but has acceleration zero at t=0. Decreasing value and increasing slope can happen together.</Prose></Checkpoint>
    <H2>8. Reverse operations to integrate</H2>
    <Prose>Finding an antiderivative often means recognizing a derivative rule in reverse. There is no single elementary recipe that handles every function. Begin by simplifying valid algebra, inspect the domain and choose a method whose mechanism matches the expression. Always differentiate a proposed primitive as a check.</Prose>
    <H3>Substitution reverses a chain</H3>
    <Prose>If f is continuous on an interval containing g([a,b]) and g is continuously differentiable, choose an antiderivative F of f. The chain rule says [F(g(x))]′=f(g(x))g′(x). The fundamental theorem then gives:</Prose>
    <MathBlock>{String.raw`\begin{gathered}\int_a^b f(g(x))g'(x)\,dx\\
      =\int_{g(a)}^{g(b)}f(u)\,du.\end{gathered}`}</MathBlock>
    <Prose>For ∫ from 0 to 2 of 2x(1+x²)³ dx, let u=1+x². Its derivative is 2x, so du=2x dx records the exact differential relation. The bounds become 1 and 5. The integral becomes ∫ from 1 to 5 of u³ du=(5⁴−1)/4=156. The transformed variable, differential and bounds belong to the same calculation.</Prose>
    <Prose>This chain-rule form does not require g to be one-to-one. For example, ∫ from −1 to 1 of 2x cos(x²) dx=sin(1)−sin(1)=0. The intermediate value x² first decreases and then increases, so the signed contributions cancel. That is a different task from collecting nonnegative probability mass from multiple inverse branches, which the next lesson will explain.</Prose>
    <H3>Integration by parts reverses a product</H3>
    <Prose>Integrate the product rule u′v+uv′=(uv)′ and rearrange. With continuously differentiable u and v, the result keeps a boundary contribution:</Prose>
    <MathBlock>{String.raw`\begin{gathered}\int_a^b u(x)v'(x)\,dx\\
      =[u(x)v(x)]_a^b-\int_a^b u'(x)v(x)\,dx.
    \end{gathered}`}</MathBlock>
    <Prose>For ∫ from 0 to 1 of x cos x dx, take u=x and v′=cos x, so u′=1 and v=sin x. The answer is [x sin x]₀¹−∫₀¹sin x dx=sin1+cos1−1. The product's accumulated change is why the boundary term appears. Dropping it is not a harmless choice of integration constant.</Prose>
    <H3>Rewrite the expression without rewriting its domain</H3>
    <Prose>Algebra can separate a rational expression: 1/[x(x+1)]=1/x−1/(x+1), away from x=0 and x=−1. This <strong>partial-fraction decomposition</strong> follows by recombining the right side over a common denominator. It reduces the problem to the missing power-rule case 1/x. The next section builds its logarithmic primitive rather than pretending the denominator can be ignored.</Prose>
    <Prose>Trigonometric substitutions can make a geometric constraint explicit. For ∫₀¹√(1−x²) dx, put x=sin θ on 0≤θ≤π/2. Here cos θ is nonnegative, so √(1−sin²θ)=cos θ and dx=cos θ dθ. The integral is ∫₀^(π/2)cos²θ dθ. Using cos²θ=(1+cos2θ)/2 gives π/4. The selected branch justified removing the square root; outside it, an absolute value would be needed. The answer also agrees with the area of a unit-circle quarter.</Prose>
    <LessonTable caption="Choose an integration operation for a reason" headers={['Expression structure', 'Useful operation', 'What to check']} rows={[['A sum of known derivatives', 'Linearity and primitive rules', 'Each term and its domain'], ['An inner expression and its derivative', 'Substitution', 'Differential and both bounds'], ['A product with one factor simplified by differentiation', 'Integration by parts', 'Boundary term and new integral'], ['A rational expression with factorable denominator', 'Division / partial fractions', 'Excluded poles and valid interval'], ['A square root tied to a circle identity', 'A branch-aware trigonometric substitution', 'Sign of the square root and transformed range']]} />
    <Prose>Some continuous functions, such as exp(−x²), have no antiderivative expressible using a finite combination of elementary functions. The fundamental theorem still defines their accumulation functions and supplies their derivatives. Numerical quadrature can estimate their definite integrals. “No elementary primitive” is not “no integral”; a proof of non-elementary expressibility is a separate symbolic-integration question, not a claim established by failing to guess a formula.</Prose>

    <H2>9. Complete the exponential growth story</H2>
    <Prose>Algebra introduced repeated multiplication, logarithms and the number e, but deliberately did not assume derivative rules. We can now close that connection using integration. For x&gt;0, define L(x)=∫ from 1 to x of 1/t dt. The fundamental theorem gives L′(x)=1/x and L(1)=0. This function will be the natural logarithm.</Prose>
    <MathBlock>{String.raw`L(x)=\int_1^x\frac{dt}{t},\qquad L'(x)=\frac1x.`}</MathBlock>
    <Prose>Its derivative is positive, so L is strictly increasing. Fix a&gt;0 and differentiate L(ax)−L(x): the derivative is a/(ax)−1/x=0. The constant-difference result and evaluation at x=1 give L(ax)=L(a)+L(x). Thus multiplication of positive inputs becomes addition of outputs—the logarithm law, now derived from the integral.</Prose>
    <Prose>Because L(2)&gt;0, repeated use of the product law gives L(2ⁿ)=nL(2), which grows without bound. The reciprocal law gives L(2⁻ⁿ)=−nL(2), unbounded in the other direction. L is continuous and strictly increasing, so it takes every real value exactly once. Its continuous inverse, <strong>exp</strong>, maps each real y to the positive x for which L(x)=y. Define e=exp(1); this agrees with the base convention introduced in Algebra.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
      \ln x=L(x),\qquad \ln(\exp y)=y,\\
      (\exp)'(y)=\frac1{L'(\exp y)}=\exp y.
    \end{gathered}`}</MathBlock>
    <Prose>The inverse-difference-quotient proof from section4 justifies the second line: L′ is nonzero throughout its positive domain. No exponential derivative was assumed in defining L. The product law and injectivity of L give exp(y+z)=exp(y)exp(z), connecting this inverse to repeated powers and extending them continuously.</Prose>
    <details className="lesson-deeper"><summary>Prove the finite-compounding limit from the same integral</summary>
      <Prose>On [1,1+1/n], the height 1/t lies between 1/(1+1/n) and 1. Multiplying by the interval width 1/n and then by n yields 1/(1+1/n)≤n ln(1+1/n)≤1. Both outer bounds approach 1. The logarithm of (1+1/n)ⁿ therefore approaches 1; continuity of exp makes the original expression approach exp(1)=e. A finite table can illustrate this theorem but is not its proof.</Prose>
    </details>
    <H3>Differentiate a continuous rate and interpret it</H3>
    <Prose>For A(t)=A₀exp(kt), the chain rule gives A′=kA. When A is positive, A′/A=k: k is the <strong>instantaneous relative rate</strong>. It has inverse-time units so kt is dimensionless. Over a finite period Δ, the fractional change is exp(kΔ)−1, not kΔ exactly. The latter is only its first-order approximation for small kΔ.</Prose>
    <ExponentialRateLab />
    <Prose>With k=0.2 per time unit, a one-unit gain is exp(0.2)−1≈0.221403, about 22.14%. A genuine 20% one-unit gain corresponds to k=ln(1.2)≈0.182322 per unit. For k&gt;0, doubling time solves exp(kT)=2, giving T=ln2/k. If k&lt;0, the model decays; k=0 is constant, so neither has a positive doubling time under this unchanged model.</Prose>
    <CalculusExample id="exponential-rate"><Prose>The compounding log stays between the proved bounds. The subsequent calculation distinguishes an amount, an amount-per-time derivative, a period fraction and a continuous relative rate. All outputs belong to the explicitly assumed model.</Prose></CalculusExample>
    <LessonTable caption="Complete the real elementary rules with their domains" headers={['Function', 'Derivative', 'Real-domain condition']} rows={[['exp(g(x))', 'exp(g(x))g′(x)', 'g differentiable'], ['ln x', '1/x', 'x>0'], ['ln|x|', '1/x', 'x≠0; each component separately'], ['bˣ = exp(x ln b)', 'bˣ ln b', 'b>0'], ['log_b x = ln x / ln b', '1/(x ln b)', 'x>0, b>0, b≠1'], ['xʳ = exp(r ln x), fixed real r', 'r xʳ⁻¹', 'x>0; other real branches need their own domain check']]} />
    <Prose>The real-power rule follows by differentiating exp(r ln x), not by assuming integer induction covers every real exponent. Its integral is x^(r+1)/(r+1)+C when r≠−1 on the positive domain. The exceptional exponent −1 gives ln x+C. On a negative interval the primitive of 1/x is ln|x|+C.</Prose>
    <CalculusExample id="integration-operations"><Prose>The substitution result is exactly 156. Integration by parts for x exp x gives the primitive (x−1)exp x and definite value 1 on [0,1]. The partial-fraction integral on [1,2] is ln(4/3)≈0.287682072. Neither pole lies in that interval; crossing one would require an improper-integral analysis.</Prose></CalculusExample>

    <H2>10. Keep the remainder with the approximation</H2>
    <Prose>A tangent uses a function's value and first derivative at one point. To incorporate curvature, add a quadratic term; further derivatives add further terms. The <strong>Taylor polynomial</strong> of degree n about a matches the first n derivatives there. Matching those derivatives is a finite local condition, not yet equality with the function.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
      P_n(a+h)=\sum_{j=0}^{n}\frac{f^{(j)}(a)}{j!}h^j,\\
      f(a+h)\\
      =P_n(a+h)+R_n(a+h).
    \end{gathered}`}</MathBlock>
    <Prose>Here f⁽ʲ⁾ is the jth derivative, f⁽⁰⁾ means f itself, and j! is the product of positive integers through j, with 0!=1. The factorials ensure the derivative matching: differentiating hʲ exactly j times gives j!. For a function with n+1 continuous derivatives on the segment from a to a+h, the remainder has the exact integral form:</Prose>
    <MathBlock>{String.raw`\begin{gathered}R_n(x)\\
      =\frac1{n!}\int_a^x (x-t)^n f^{(n+1)}(t)\,dt.\end{gathered}`}</MathBlock>
    <details className="lesson-deeper"><summary>Derive the integral remainder and its bound</summary>
      <Prose>For n=0 the statement is exactly the fundamental theorem: f(x)−f(a)=∫ₐˣf′(t)dt. For n≥1, integrate the proposed remainder by parts, differentiating (x−t)ⁿ/n! and integrating f⁽ⁿ⁺¹⁾. Its boundary contribution is −f⁽ⁿ⁾(a)(x−a)ⁿ/n!, and the remaining integral is Rₙ₋₁(x). Thus Rₙ=Rₙ₋₁−the next Taylor term; induction proves the identity. If |f⁽ⁿ⁺¹⁾|≤M on the whole segment, take absolute values and integrate |x−t|ⁿ over a segment of length |h|. This yields |Rₙ|≤M|h|ⁿ⁺¹/(n+1)!. Reversed integration orientation for h&lt;0 does not change the absolute bound.</Prose>
    </details>
    <MathBlock>{String.raw`|R_n(a+h)|\le\frac{M|h|^{n+1}}{(n+1)!}.`}</MathBlock>
    <Prose>For f(x)=√x near a=4, the linear prediction at x=4.4 is 2+0.4/4=2.1. The true value is about 2.097617696. Since |f″(x)|=1/(4x^(3/2))≤1/32 on [4,4.4], the absolute error is at most (1/32)(0.4²)/2=0.0025. The actual error, about 0.002382304, satisfies that bound. The curvature bound was needed throughout the interval, not only at a.</Prose>
    <TaylorErrorLab />
    <CalculusExample id="taylor-remainders"><Prose>The exponential errors are smaller than their stated upper bounds. At x=1.5, higher even-degree log polynomials become worse in the displayed cases. The function ln(2.5) exists; that fact does not make its power series around zero converge at this input.</Prose></CalculusExample>
    <H3>An infinite series needs a separate conclusion</H3>
    <Prose>For exp x about zero, every derivative at zero is 1. On the segment from 0 to a fixed x, the remainder bound is exp(max(0,x))|x|ⁿ⁺¹/(n+1)!. The ratio of successive bound terms is |x|/(n+2); eventually it is at most 1/2, so the bound tends to zero. Therefore exp x equals its infinite power series at each fixed real x. The proof comes from the vanishing remainder.</Prose>
    <Prose>For ln(1+x), integrate the finite geometric identity 1/(1+t)=Σ from j=0 to n−1 of (−t)ʲ+(−t)ⁿ/(1+t). This gives the degree-n alternating log polynomial plus remainder ∫₀ˣ(−t)ⁿ/(1+t)dt. When |x|&lt;1, its magnitude is at most |x|ⁿ⁺¹/[(n+1)(1−|x|)], which tends to zero. At x=1 a separate bound, 1/(n+1), still gives convergence to ln2. At x=−1 the logarithm is undefined and the terms sum negatively without bound. For |x|&gt;1, the term magnitudes |x|ⁿ/n do not tend to zero, ruling out convergence of this series.</Prose>
    <Prose>Finite differentiability, arbitrarily many derivatives and equality with a Taylor series are different claims. Real Analysis develops sequence/series convergence, uniformity and when derivatives or integrals may pass through an infinite sum. Do not infer those exchanges from a smooth-looking plot.</Prose>
    <H3>A computer takes finite steps in finite arithmetic</H3>
    <Prose>The analytic quotient [exp(h)−1]/h approaches 1. On a computer, exp(h) may round to exactly 1 when h is extremely small, destroying the difference before division. <Code>expm1(h)</Code> computes exp(h)−1 with a numerically suitable method near zero. This is a different issue from the mathematical truncation error.</Prose>
    <CalculusExample id="finite-difference-rounding"><Prose>At the recorded smallest step, direct subtraction reports zero while the analytic derivative is 1. The displayed values are actual Python float results in the verification environment; the precise last digits depend on numerical implementation. Numerical Methods develops step selection, higher-order stencils and error budgets. An ever-smaller step is not a universal accuracy strategy.</Prose></CalculusExample>

    <H2>11. Check what happens at the edge</H2>
    <Prose>An integral over an infinite interval, or across an unbounded integrand, is defined through a limit of ordinary finite integrals. It is called an <strong>improper integral</strong>. A finite answer at every finite cutoff does not decide whether the limit is finite.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      \int_1^\infty x^{-p}\,dx
      &=\lim_{B\to\infty}\int_1^B x^{-p}\,dx,\\
      \int_0^1 x^{-p}\,dx
      &=\lim_{\delta\downarrow0}\int_\delta^1 x^{-p}\,dx.
    \end{aligned}`}</MathBlock>
    <Prose>For p≠1, the primitive is x^(1−p)/(1−p). The tail integral from 1 to B is [B^(1−p)−1]/(1−p). If p&gt;1, the power of B tends to zero and the total is 1/(p−1); if p&lt;1 it grows without bound. At p=1 the finite integral is ln B, which also diverges. So the tail converges exactly when p&gt;1.</Prose>
    <Prose>At the other end, the finite integral from δ to 1 is [1−δ^(1−p)]/(1−p). It has finite limit 1/(1−p) exactly when p&lt;1. At p=1 it is −lnδ and diverges. Thus x^(−1/2) is unbounded at zero but has integral 2 on (0,1]; and 1/x tends to zero at infinity but still has divergent tail area. Height, local singularity and total area answer different questions.</Prose>
    <ImproperIntegralLab />
    <CalculusExample id="improper-cutoffs"><Prose>Discarding [0,10⁻⁶] from x^(−1/2) loses 0.002 of area, which may be too much for an accuracy requirement. The missing amount is computable here; calling an interval “tiny” cannot replace that error calculation.</Prose></CalculusExample>
    <H3>Separate singularities instead of cancelling infinities</H3>
    <Prose>For an interior singularity c, both ∫ₐᶜf and ∫ᶜᵇf must converge independently. Likewise, an integral across both infinite directions requires separate tail convergence. For 1/x on [−1,1], the left improper integral tends to −∞ and the right to +∞; their sum is not a defined real integral. Symmetric cutoffs give zero, but that is a <strong>Cauchy principal value</strong>, a different limiting prescription. It does not repair ordinary convergence.</Prose>
    <details className="lesson-deeper"><summary>A justified finite-endpoint version of l'Hôpital's rule</summary>
      <Prose>Suppose f and g are continuous at a with f(a)=g(a)=0, differentiable on a one-sided punctured interval, and g′ never vanishes there. Suppose also f′(x)/g′(x) approaches a finite L as x approaches a on that side. Then f(x)/g(x) approaches L. Here is the mechanism. For a fixed nearby x, the function H(t)=f(t)g(x)−g(t)f(x) has equal zero values at a and x. Rolle's theorem gives an interior c with f′(c)g(x)−g′(c)f(x)=0. Also g(x)≠0, because otherwise Rolle applied to g would contradict nonvanishing g′. Therefore f(x)/g(x)=f′(c)/g′(c). As x→a, the intermediate c→a too, giving the limit. This is Cauchy's mean-value argument.</Prose>
      <Prose>Apply it, now that exponential derivatives are established, to [exp(x)−1]/x at zero: the derivative ratio is exp(x), tending to 1. Applying the rule repeatedly requires its hypotheses again at every step. Infinity/infinity and infinite-limit variants have their own statements. Do not differentiate numerator and denominator of an ordinary quotient unless an applicable limit theorem actually permits it, and do not use l'Hôpital circularly to prove a limit needed to derive those derivatives.</Prose>
    </details>

    <H2>12. Transfer the same idea to new quantities</H2>
    <Prose>The integral's mechanism is not tied to area measured in square metres. It adds a local contribution multiplied by the corresponding small input increment. The units tell us what has been accumulated.</Prose>
    <H3>Work: force component times displacement</H3>
    <Prose>Along a straight path, a force component F(x) acting in the positive travel direction contributes approximately F(x)Δx to mechanical work over a short displacement. Its total is W=∫F(x)dx. In a declared example, F(x)=3+2x newtons for the numerical position x in metres on [0,2]. The work is [3x+x²]₀²=10 N·m=10 J. This averages a varying force over position, not over time. An opposing force component contributes negative work in this signed convention.</Prose>
    <H3>Mass: density times length, then a balance condition</H3>
    <Prose>A two-metre rod has linear density ρ(x)=2+x kg/m in the chosen numerical metre coordinate. A short piece has approximate mass ρ(x)Δx. Integrating gives total mass M=∫₀²(2+x)dx=6 kg. To find a balance point c, require the signed first moment ∫₀²(x−c)ρ(x)dx to vanish. Rearranging gives c=[∫xρ(x)dx]/M when M&gt;0.</Prose>
    <WeightedRodFigure />
    <Prose>Here the first moment is ∫₀²(2x+x²)dx=20/3 kg·m. Dividing by 6 kg gives c=10/9 m, a little to the right of the geometric midpoint. More mass lies toward the right end, so this direction makes sense. The next topic's weighted expectation will use the same averaging structure with probability mass instead of material mass.</Prose>
    <CalculusExample id="weighted-material"><Prose>Mass, first moment, position and work have different units even though all arise from simple polynomial integrals. Dividing by length would give average density, not the rod's balance point.</Prose></CalculusExample>
    <details className="lesson-deeper"><summary>Two more geometric accumulations: volume and curve length</summary>
      <Prose>If a solid has cross-sectional area A(x), a thin slice has approximate volume A(x)Δx. Its volume is ∫A(x)dx under the usual continuous-slice assumptions. Rotating the region under y=√x, 0≤x≤3, around the x-axis makes disk slices of area πy²=πx. The volume is ∫₀³πx dx=9π/2 in cubic units. The slice's area, not its radius, is what gets multiplied by thickness.</Prose>
      <Prose>For a continuously differentiable graph y=f(x), a short segment has length √(Δx²+Δy²). Divide Δy by Δx and refine the partition; the mean value theorem and continuity of f′ justify the limit ∫√(1+f′(x)²)dx. For y=(2/3)x^(3/2) on [0,3], f′=√x and length is ∫₀³√(1+x)dx=(2/3)[(1+x)^(3/2)]₀³=14/3 in length units. These examples use chosen numerical length units; the coordinate scales must agree when interpreting Euclidean length.</Prose>
    </details>
    <H3>A rate law describes a solution, not an automatic simulation</H3>
    <Prose>The equation y′=ky says the local change is proportional to the current state. Together with y(0)=y₀, it has solution y(t)=y₀exp(kt). To see uniqueness among differentiable solutions on an interval, differentiate exp(−kt)y(t): its derivative is zero, so it stays y₀. This integrating-factor argument also covers y₀=0 without dividing by y.</Prose>
    <Prose>A simple numerical approximation, <strong>Euler's method</strong>, follows the current tangent for a finite time step: y_next=y+hky. That update is not the same as the exact factor exp(kh). For decay, a step so large that 1+hk is negative can make the numerical value negative even while the exact positive solution remains positive.</Prose>
    <CalculusExample id="decay-steps"><Prose>The one-step result is negative and the two-step result collapses to zero. Refining the step improves this example toward 3exp(−2), but accuracy and stability require a method-specific analysis. Ordinary Differential Equations & Linear Systems develops that next branch; a smooth drawn trajectory is not itself a numerical guarantee.</Prose></CalculusExample>

    <H2>13. Practise with changed assumptions</H2>
    <Prose>Try each task before opening its hint. A satisfactory answer explains the operation, checks its domain or theorem hypotheses and interprets the result. The final program gives a changed synthesis example to compare with your own reasoning.</Prose>
    <Practice title="A. Build a different closeness guarantee" question="Prove lim(x→3)x²=9 by choosing a delta from an arbitrary epsilon. Is delta=epsilon always sufficient?" hint="First keep |x−3|<1, then bound the other factor in |x²−9|.">
      <Prose>If |x−3|&lt;1 then 2&lt;x&lt;4 and |x+3|&lt;7. Choose δ=min(1,ε/7). Every allowed x then has |x²−9|&lt;7δ≤ε. δ=ε is not generally sufficient: with ε=0.1, x=3.05 is inside that proposed input radius but |x²−9|=0.3025 exceeds the output tolerance. One explicit witness refutes that choice.</Prose>
    </Practice>
    <Practice title="B. Repair a limit versus value confusion" question="For x≠1 define f(x)=(x²−1)/(x−1), and set f(1)=7. Find the limit and determine the value that would make f continuous. Does the given f have a derivative at 1?" hint="Cancel only where the original denominator is nonzero; then compare the nearby behavior with the assigned point value.">
      <Prose>For x≠1 the function equals x+1, so the limit is 2. Defining f(1)=2 makes it continuous and yields derivative 1 there. The given value 7 makes it discontinuous, so differentiability is impossible. Directly, its quotient is [2+h−7]/h=1−5/h, which has no finite limit.</Prose>
    </Practice>
    <Practice title="C. Diagnose a local-rule shortcut" question="At x=0, find the derivative and the actual change for f(x)=(2x−1)² when h=0.1. A student says squaring gives derivative (2)²=4. What went wrong?" hint="The outer slope depends on its current input, and a derivative is not the finite change.">
      <Prose>The inner slope is 2 and the outer slope at u=−1 is 2u=−2, so f′(0)=−4. The linear change is −0.4. The exact new value is (−0.8)²=0.64, versus f(0)=1, giving change −0.36 and remainder 4h²=0.04. Squaring the inner derivative ignores where the outer function is evaluated and applies neither the product nor chain rule.</Prose>
    </Practice>
    <Practice title="D. Find an optimum with a cusp and endpoints" question="Find the minimum and maximum of |x−1| on [−1,3]. Why would solving f′=0 fail?" hint="Evaluate both endpoints and the point where differentiability fails.">
      <Prose>The candidate x=1 is a cusp with value 0, the global minimum. Both endpoints −1 and 3 have value 2 and attain the maximum. On either differentiable branch the derivative is −1 or 1, so no point solves f′=0. Fermat's zero-derivative condition applies only when the interior extremum is differentiable.</Prose>
    </Practice>
    <Practice title="E. Recover position and total travel from a rate" question="An object starts at position 5 m and has velocity v(t)=2t−4 m/s on [0,4]. Find its final position, net displacement and distance travelled." hint="Integrate the signed rate, then split the absolute-rate calculation where velocity changes sign.">
      <Prose>A position function is 5+t²−4t. The final position is 5 m, so net displacement is zero. The velocity changes sign at t=2, where position is 1 m. The object travels 4 m backwards and 4 m forwards, totalling 8 m. Starting position affects the final position but not these displacement or distance totals.</Prose>
    </Practice>
    <Practice title="F. Keep both moving-bound contributions" question="For G(x)=∫ from x² to 3x of cos t dt, find G′(x) and evaluate it at x=0. Can the integral value there also be the derivative?" hint="Use an accumulation at the upper limit minus an accumulation at the lower limit.">
      <Prose>G′(x)=3cos(3x)−2x cos(x²). Thus G′(0)=3, while G(0)=0 because the two endpoints coincide. A zero interval length at one input does not make the rate at which the bounds separate zero.</Prose>
    </Practice>
    <Practice title="G. Change variable without losing the domain" question="Evaluate ∫ from 1 to 3 of 2x/(1+x²) dx. A proposed answer is ln3−ln1. Repair it." hint="The new variable is 1+x²; both transformed endpoints change.">
      <Prose>Put u=1+x² and du=2x dx. The bounds become 2 and 10, so the answer is ln10−ln2=ln5. All u values here are positive, making the logarithmic primitive valid. The proposed result used new-variable integration with old-variable endpoints.</Prose>
    </Practice>
    <Practice title="H. Restore a missing boundary term" question="A student integrates x sin x on [0,π] by parts and reports ∫₀^π cos x dx=0. What is the correct answer?" hint="Choose v=−cos x and keep [uv] at both endpoints.">
      <Prose>The primitive calculation gives [−x cos x]₀^π+∫₀^πcos x dx. The first term is π and the second is zero, so the answer is π. The missing term records the product's endpoint change; it cannot be absorbed into a cancelled constant.</Prose>
    </Practice>
    <Practice title="I. Translate a decay fraction into a rate" question="A positive model retains 80% of its amount every three time units. Find its constant continuous rate k and its half-life. State the model assumption." hint="exp(3k)=0.8; a half-life H satisfies exp(kH)=1/2.">
      <Prose>Assuming exact exponential decay with constant k, k=ln(0.8)/3≈−0.0743812 per time unit. The half-life is ln(1/2)/k≈9.318851 time units. Dividing −0.2 by 3 gives a finite-fraction approximation, not the exact continuous rate. These conclusions depend on the exponential model; one observed retention interval alone would not establish it.</Prose>
    </Practice>
    <Practice title="J. Bound a changed approximation" question="Use a first-order Taylor approximation to √9.3 about 9. Give an error bound valid over [9,9.3], then state whether the approximation is above or below the true value." hint="f′(9)=1/6 and |f″(x)|=1/(4x^(3/2)); also inspect the sign of f″.">
      <Prose>The approximation is 3+0.3/6=3.05. On this interval |f″|≤1/108, so the absolute error is at most (1/108)(0.3²)/2=1/2400≈0.000416667. The second derivative is negative, so the tangent lies above the concave function on this interval. The true value is about 3.049590136, giving error about 0.000409864, within the bound.</Prose>
    </Practice>
    <Practice title="K. Classify two different improper boundaries" question="Does x^(−3/2) have a finite integral on (0,1]? On [1,infinity)? How much tail area remains beyond B=100?" hint="The same exponent meets different conditions at zero and at infinity.">
      <Prose>The integral at zero diverges because p=3/2≥1. The tail converges because p&gt;1, with total 1/(p−1)=2. Beyond B, the remainder is B^(1−p)/(p−1)=2/√B; at B=100 it is 0.2. A finite tail and a divergent origin can belong to the same function.</Prose>
    </Practice>
    <Practice title="L. Diagnose a principal-value claim" question="Someone evaluates ∫ from −2 to 2 of 1/x by symmetric cutoffs and calls zero the ordinary integral. Give the correct classification and explain which limits must be checked." hint="The two sides of the singularity are separate improper integrals before any addition.">
      <Prose>The left integral tends to −∞ and the right to +∞. Neither has a finite limit, so the ordinary improper integral does not converge. Equal cutoffs cancel to zero and define a principal value, but unequal cutoff rates can produce other finite differences or divergence. The limiting prescription is part of the question.</Prose>
    </Practice>
    <Practice title="M. Solve a changed motion model end to end" question="Use s(t)=t³−3t² on [0,3]. Find all relevant candidates, turning behavior, net displacement, total distance, average velocity and velocity at t=1. Then verify your values with a complete program." hint="Factor the derivative as 3t(t−2), keep both endpoints, and separate signed changes from their magnitudes.">
      <Prose>The interior turning point is 2; t=0 is a stationary endpoint. The function decreases from s(0)=0 to s(2)=−4 and then increases to s(3)=0. Thus the minimum is −4, the maximum 0 occurs at both endpoints, the net displacement is zero and the distance is 8 m. Average velocity is zero; instantaneous velocity at t=1 is −3 m/s. A complete answer must explain how the derivative sign proves there are no hidden turns between these points.</Prose>
      <CalculusExample id="changed-motion"><Prose>Change the final time or initial position only after predicting which outputs should change. An initial-position shift does not alter derivatives, displacement or distance; changing the time interval can alter all three motion summaries.</Prose></CalculusExample>
    </Practice>

    <H2>14. Connect the pieces and continue</H2>
    <Prose>You are ready to continue when you can explain a secant and tangent without dividing by zero; supply an actual neighborhood guarantee for a simple limit; differentiate a composition with its domains; check every extrema candidate; distinguish signed accumulation from total magnitude; and explain the continuity condition connecting an accumulation's derivative to its integrand. You should also be able to choose a valid integration operation and distinguish a finite approximation, its error bound and a claim about an infinite limit.</Prose>
    <Prose>The next topic in this module is <a href="/learn/topic/random-variables-expectation-covariance">Random Variables, Expectation & Covariance</a>. Weighted sums and integrals will describe uncertain outcomes, means and shared variation. The same mass-preserving reasoning used for the rod becomes a probability-weighted average. Follow the module sequence; <a href="/learn/topic/multivariate-calculus-gradients">Multivariate Calculus</a>, <a href="/learn/topic/numerical-methods-finite-differences-quadrature-root-finding">Numerical Methods</a>, <a href="/learn/topic/ordinary-differential-equations-linear-systems">ODEs</a> and <a href="/learn/topic/real-analysis-sequences-modes-of-convergence">Real Analysis</a> are related branches with their own readiness requirements.</Prose>
    <Sources alternatives={<><h4>Another explanation or more practice</h4><ul>
      <li><a href="https://www.3blue1brown.com/lessons/integration/" target="_blank" rel="noreferrer">3Blue1Brown: Integration and the fundamental theorem</a> — a visual companion article for readers who want another route from changing velocity to accumulated area. Its text adaptation was reviewed; its example is distinct from this lesson's reversing journey.</li>
      <li><a href="https://www.youtube.com/watch?v=rfG8ce4nNh0" target="_blank" rel="noreferrer">Grant Sanderson: Integration and the fundamental theorem of calculus</a> — the creator's companion video for the same visual explanation. Use it after the rectangle and accumulation sections; the written companion, not a full viewing of the recording, informed this review.</li>
      <li><a href="https://openstax.org/books/calculus-volume-1/pages/3-1-defining-the-derivative" target="_blank" rel="noreferrer">OpenStax: Defining the Derivative</a> — another beginner explanation with exercises on secants, tangents and difference quotients. Attempt changed problems after solving this lesson's local examples.</li>
    </ul></>}>
      <li><a href="https://openstax.org/books/calculus-volume-1/pages/2-5-the-precise-definition-of-a-limit" target="_blank" rel="noreferrer">OpenStax: The Precise Definition of a Limit</a> — quantified neighborhoods and worked proof practice, useful after the epsilon–delta investigation.</li>
      <li><a href="https://openstax.org/books/calculus-volume-1/pages/4-4-the-mean-value-theorem" target="_blank" rel="noreferrer">OpenStax: The Mean Value Theorem</a> — theorem hypotheses and consequences connecting interval averages to local rates.</li>
      <li><a href="https://openstax.org/books/calculus-volume-1/pages/5-3-the-fundamental-theorem-of-calculus" target="_blank" rel="noreferrer">OpenStax: The Fundamental Theorem of Calculus</a> — both parts and moving-bound examples. Keep the theorem's continuity hypothesis; the broader surrounding “integrable” wording does not justify an everywhere differentiable primitive.</li>
      <li><a href="https://openstax.org/books/calculus-volume-1/pages/6-7-integrals-exponential-functions-and-logarithms" target="_blank" rel="noreferrer">OpenStax: Integrals, Exponential Functions and Logarithms</a> — a rigorous alternative starting from the integral of 1/x, useful for the Algebra-to-Calculus connection.</li>
      <li><a href="https://openstax.org/books/calculus-volume-2/pages/6-3-taylor-and-maclaurin-series" target="_blank" rel="noreferrer">OpenStax: Taylor and Maclaurin Series</a> — finite approximations, remainders and convergence exercises for the deeper branch.</li>
      <li><a href="https://openstax.org/books/calculus-volume-2/pages/6-4-working-with-taylor-series" target="_blank" rel="noreferrer">OpenStax: Working with Taylor Series</a> — the nonelementary Gaussian integral illustrates why an accumulation function can exist without a finite elementary formula. Read the convergence conditions before integrating an infinite series term by term.</li>
      <li><a href="https://openstax.org/books/calculus-volume-2/pages/3-7-improper-integrals" target="_blank" rel="noreferrer">OpenStax: Improper Integrals</a> — endpoint and infinite-tail definitions, convergence examples and further exercises.</li>
    </Sources>
  </div>
};
