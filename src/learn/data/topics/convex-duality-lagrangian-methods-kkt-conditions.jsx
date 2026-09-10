import { Code, H2, H3, Prose } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { Checkpoint, LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample';
import { BoundLadderFigure, KktLogicFigure, ProjectionCertificateLab, ResourceDualAscentLab, ScalarKktLab, SensitivityLab } from '../../components/lesson-labs/DualityKktLabs.jsx';
import { dualityKktExamples as examples } from '../duality-kkt-examples.js';
function Practice({
  title,
  children,
  hint,
  solution
}) {
  return <section className="duality-practice"><H3>{title}</H3>{children}<details><summary>Hint</summary><Prose>{hint}</Prose></details><details><summary>Show reasoning and acceptance checks</summary>{solution}</details></section>;
}
export default {
  title: 'Convex Duality & Lagrangian Methods (KKT Conditions)',
  readTime: '~80 min read + 3–4 hours practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot duality-lesson">
    <LessonIntro prerequisites="Vectors and dot products; Multivariate Calculus's gradients and constrained directions; Convex Optimization's supporting-plane inequality and feasible-candidate bounds. The preceding optimizer and schedule lessons explain how to generate steps; this page asks how constraints and certificates change the problem. Multipliers, infima, duality and KKT are introduced here. No SVM or distributed-systems background is required." sections={[['1-start-with-a-feasible-answer', 'Start with a feasible answer'], ['2-build-a-lower-bound-with-a-price', 'Build a lower bound with a price'], ['3-respect-signs-domains-and-infima', 'Respect signs, domains and infima'], ['4-close-the-gap-with-kkt', 'Close the gap with KKT'], ['5-separate-certificates-from-existence', 'Separate certificates from existence'], ['6-interpret-prices-through-reoptimization', 'Interpret prices through reoptimization'], ['7-coordinate-local-decisions-with-a-shared-price', 'Coordinate local decisions with a shared price'], ['8-check-a-solver-and-reveal-an-ml-dual', 'Check a solver and reveal an ML dual'], ['9-practise-new-constraints-and-certificates', 'Practise new constraints and certificates']]}>
      Finding a low-cost answer is useful. Proving that no feasible answer can be much better is a different achievement. Duality builds lower bounds on the best possible cost; KKT conditions explain when a candidate meets its bound. Work from one small budget problem to equality constraints, sensitivity, a price-based algorithm and a machine-learning application.
    </LessonIntro>

    <H2>1. Start with a feasible answer</H2>
    <Prose>You want to choose a point close to (3,4), but its coordinates must sum to at most 5. Both coordinates are dimensionless in this example. A point is <strong>feasible</strong> if it obeys that restriction. Its <strong>objective</strong>, the quantity to minimize, is squared distance from the target. The original constrained problem is called the <strong>primal problem</strong>.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\min_{(x,y)\in\mathbb R^2} f(x,y),\\
f(x,y)=(x-3)^2+(y-4)^2,\\
g(x,y)=x+y-5\le0.
\end{gathered}`}</MathBlock>
    <Prose>The target (3,4) costs zero but violates the budget by 2. The feasible point (2,2) costs 1+4=5. Therefore the best feasible cost p* is at most 5: an actual feasible answer supplies an <strong>upper bound</strong> on a minimization problem. A low objective at an infeasible point supplies no such bound.</Prose>
    <Prose>We can solve this particular geometry by hand. At the closest feasible point the budget is tight: if there were slack, a small move toward the infeasible target would reduce distance while remaining feasible. Let a=3−x and c=4−y be the two coordinate reductions. The boundary requires a+c=2. Now a²+c²=((a+c)²+(a−c)²)/2=2+(a−c)²/2. It is smallest when a=c=1. Thus the solution is (2,3), with p*=2. This argument permits all real coordinates; nonnegativity is not an extra hidden constraint.</Prose>
    <Prose>For a general budget b, the target is already feasible when b≥7. Otherwise reduce both coordinates by (7−b)/2. This is a projection onto a half-plane: the correction is parallel to its normal vector (1,1). The first program also handles a general nonzero normal vector a and target t, using the correction max(0,aᵀt−b)a/‖a‖².</Prose>
    <MathBlock>{String.raw`\begin{aligned}
s&=\frac{\max(0,7-b)}2,\\
(x^*,y^*)&=(3-s,4-s),\\
p^*(b)&=2s^2=\frac{\max(0,7-b)^2}{2}.
\end{aligned}`}</MathBlock>
    <Prose>All programs below are complete Python examples. Save a whole block as <Code>duality_example.py</Code> and run <Code>python duality_example.py</Code>. Most use NumPy; the last two also use CVXPY and its Clarabel solver. In a chosen Python environment, install missing packages with <Code>python -m pip install numpy cvxpy clarabel</Code>. The displayed outputs were checked with Python 3.12, NumPy 2.3.5, CVXPY 1.9.2 and Clarabel 0.11.1. Solver outputs are rounded and can differ slightly by version or tolerance. The browser labs use bounded deterministic models of the displayed mathematics.</Prose>
    <RunnableExample example={examples.projection}><Prose>The multiplier returned by the helper will acquire its meaning in the next section. Notice the budgets 7 and 9: both return the target with zero cost, but only budget 7 puts it on the boundary. Equal optimal prices will not necessarily mean equal constraint activity.</Prose></RunnableExample>
    <Checkpoint prompt="The point (1,3) is feasible at budget 5 and costs 4. Which statement follows immediately: p*≥4, p*≤4, or p*=4?">
      <Prose>Only p*≤4 follows from that candidate alone. The hand-derived solution proves p*=2, but feasibility by itself does not. Our next goal is to obtain a lower bound without needing to search every feasible point.</Prose>
    </Checkpoint>

    <H2>2. Build a lower bound with a price</H2>
    <Prose>Choose a nonnegative number λ and add λ times the constraint expression to the objective. The result is the <strong>Lagrangian</strong> L. For a feasible point, x+y−b≤0, so this extra term is nonpositive. That elementary sign observation is the source of the bound.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
L&=(x-3)^2+(y-4)^2\\
&\quad+\lambda(x+y-b),\qquad \lambda\ge0.
\end{aligned}`}</MathBlock>
    <Prose>You can interpret λ as a price per unit of budget expression. Going over budget incurs a charge; coming under budget earns a credit. This is not the same as adding a fixed nonnegative violation penalty: L can be less than f at a feasible point. Nor do we minimize jointly over x, y and λ. First fix the price, then ask how small L can be over <em>every</em> real point, including points that violate the original budget.</Prose>
    <Prose>The resulting function of the price is the <strong>dual function</strong> q(λ). In this quadratic example the minimum is attained. Differentiating with respect to the point gives 2(x−3)+λ=0 and 2(y−4)+λ=0. The positive-definite quadratic has a global minimum at z(λ)=(3−λ/2,4−λ/2). Substitution gives the exact lower bound.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
q(\lambda)&=\inf_{(x,y)\in\mathbb R^2}L(x,y,\lambda),\\
z(\lambda)&=(3-\lambda/2,\;4-\lambda/2),\\
q(\lambda)&=\lambda(7-b)-\frac{\lambda^2}{2}.
\end{aligned}`}</MathBlock>
    <Prose>Here <strong>infimum</strong> means greatest lower bound: the best value approached by the function, whether or not a point attains it. When attained, it is an ordinary minimum. The distinction will matter later. For any feasible candidate z and any allowed λ, q(λ)≤L(z,λ)≤f(z). Since this holds for <em>every</em> feasible z, q(λ)≤p*. At budget 5 and λ=1, q=2−1/2=1.5. We now know 1.5≤p*≤5 from λ=1 and candidate (2,2).</Prose>
    <BoundLadderFigure />
    <ProjectionCertificateLab />
    <H3>Choose the strongest price and explain the remaining gap</H3>
    <Prose>The <strong>dual problem</strong> maximizes q over allowed prices. Here q′(λ)=7−b−λ. With λ≥0, its maximum occurs at λ*=max(0,7−b). At budget 5, λ*=2, z(2)=(2,3) and q(2)=2. The feasible cost and lower bound agree, so no feasible answer can improve on that point. We have a certificate, not merely an iteration that stopped changing.</Prose>
    <Prose>For a feasible candidate z and allowed λ with finite q, the difference f(z)−q(λ) bounds how much worse the candidate can be than optimal. In this quadratic we can see exactly what makes that gap remain positive.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
0\le f(z)-p^*&\le f(z)-q(\lambda),\\
f(z)-q(\lambda)&=\underbrace{\|z-z(\lambda)\|^2}_{L(z,\lambda)-q(\lambda)}\\
&\quad+\underbrace{\lambda(b-x-y)}_{f(z)-L(z,\lambda)}.
\end{aligned}`}</MathBlock>
    <Prose>The first term vanishes when the candidate minimizes L. The second vanishes when the price times unused budget is zero. That can happen either because the budget is tight or because its price is zero. These two zero-gap requirements will become stationarity and complementary slackness.</Prose>
    <RunnableExample example={examples.bounds}><Prose>The infeasible candidate (4,4) has f=1 below the true feasible minimum 2. Subtracting q(2)=2 produces −1, which is not a valid negative error bound. The program declines to label it a certificate. A negative multiplier is also declined, even when its particular numerical q happens to be below p*: the inequality-price contract is what establishes the general bound.</Prose></RunnableExample>

    <H2>3. Respect signs, domains and infima</H2>
    <Prose>A general problem has a decision vector x in a stated domain D, inequality constraints gᵢ(x)≤0, and equality constraints hⱼ(x)=0. For each inequality use λᵢ≥0; for each equality use an unrestricted real multiplier νⱼ. Equality terms vanish at feasible points regardless of the sign of ν. The domain D remains part of the infimum: for example, a logarithm's positive-input domain must not silently disappear.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
L(x,\lambda,\nu)&=f(x)+\sum_i\lambda_i g_i(x)\\
&\quad+\sum_j\nu_jh_j(x),\\
q(\lambda,\nu)&=\inf_{x\in D}L(x,\lambda,\nu),\\
d^*&=\sup_{\lambda\ge0,\;\nu\in\mathbb R^r}q(\lambda,\nu)\le p^*.
\end{aligned}`}</MathBlock>
    <Prose>The last inequality is <strong>weak duality</strong>; it does not require convexity. Supremum is the least upper bound, whether or not some multiplier attains it. For finite optimal values, d*=p* is <strong>strong duality</strong>; otherwise p*−d*&gt;0 is a duality gap. A single candidate's f−q is a computable bound gap, which can be positive even when the best primal and dual values agree.</Prose>
    <H3>Equality prices can be negative</H3>
    <Prose>Minimize x²+y² subject to x+y=2. Write h=x+y−2. Then L=x²+y²+ν(x+y−2), and minimizing over x,y gives x=y=−ν/2 and q(ν)=−ν²/2−2ν. Its maximum is at ν=−2, yielding x=y=1 and value 2. Forcing ν≥0 would wrongly exclude the correct certificate.</Prose>
    <Prose>Compare the inequality x+y≥2. To use our convention, write g=2−x−y≤0. Stationarity is 2x−λ=0 and 2y−λ=0; the optimal multiplier is now λ=2. The same geometric point has a different sign convention because the constraint expression changed. The following complete program solves the equality's linear stationarity system and checks both formulations.</Prose>
    <RunnableExample example={examples.equality}><Prose>The block system has upper row 2Ix+Aᵀν=0 and lower row Ax=b. It is useful for this full-row-rank equality problem, not a universal solver for arbitrary singular constraints. The independent practice changes the target and equality normal rather than just rerunning these numbers.</Prose></RunnableExample>
    <H3>An allowed sign may still give q=−∞</H3>
    <Prose>Consider minimizing x subject to x≥1, with x allowed anywhere in ℝ before the constraint is imposed. Write g=1−x≤0. Then L=(1−λ)x+λ. A nonzero slope lets x move in a direction that drives L down without limit. Only λ=1 makes the slope zero.</Prose>
    <MathBlock>{String.raw`q(\lambda)=\begin{cases}
1,&\lambda=1,\\
-\infty,&\lambda\ne1.
\end{cases}`}</MathBlock>
    <Prose>For λ=.5, send x toward −∞; for λ=2, send it toward +∞. Both prices obey λ≥0, but neither gives a finite useful bound. This does not make the primal unbounded: its minimum is 1 at x=1. The effective domain of q consists of multipliers where q is finite; it is more than a sign check. At λ=1, L is identically 1 and every real x minimizes L, but only the feasible complementary choice x=1 is primal-optimal.</Prose>
    <RunnableExample example={examples.domain} />
    <details><summary>Deeper: why the dual is concave, and why a finite bound may be hard to compute</summary>
      <Prose>For a fixed x, L is affine in the multipliers, even if it is nonconvex in x. Let u and v denote two complete multiplier vectors and 0≤t≤1. For every x, L(x,tu+(1−t)v)=tL(x,u)+(1−t)L(x,v)≥tq(u)+(1−t)q(v). Taking the infimum over x gives q(tu+(1−t)v)≥tq(u)+(1−t)q(v): concavity.</Prose>
      <Prose>Thus maximizing the dual is a convex-optimization formulation, but evaluating its global infimum may itself be difficult. A locally minimized L value is generally above q and is not automatically a lower bound on p*. The quadratic examples are tractable because their minimizers are global and explicit. Extended values also need care: an infeasible primal is conventionally assigned p*=+∞, and an objective unbounded below has p*=−∞. A finite-gap argument assumes a nonempty feasible problem and finite relevant values.</Prose>
    </details>
    <Checkpoint prompt="If a numerical routine returns some point with L=10, can you label 10 a lower bound on p*? What additional argument is missing?">
      <Prose>You need a justified global infimum of L, or a separately valid lower bound on that infimum. Merely evaluating L gives an upper bound on its infimum. Even a zero derivative is not enough without assumptions such as convexity of L and the correct domain condition.</Prose>
    </Checkpoint>

    <H2>4. Close the gap with KKT</H2>
    <Prose>The <strong>Karush–Kuhn–Tucker conditions</strong>, abbreviated KKT, organize the requirements for a candidate and its multipliers. Begin with differentiable functions on all of ℝⁿ. A constraint is <strong>active</strong> when its expression equals zero; its slack is −gᵢ(x). An inactive feasible constraint has positive slack.</Prose>
    <LessonTable caption="Four checks on the same point and multipliers" headers={['Condition', 'Equation', 'What it checks']} rows={[['Primal feasibility', 'gᵢ(x)≤0; hⱼ(x)=0', 'The proposed answer obeys the problem.'], ['Dual sign feasibility', 'λᵢ≥0; νⱼ unrestricted', 'Inequality terms have the lower-bound sign.'], ['Stationarity', '∇f + Σλᵢ∇gᵢ + Σνⱼ∇hⱼ = 0', 'The combined Lagrangian derivative balances.'], ['Complementary slackness', 'λᵢgᵢ(x)=0 for every i', 'No positive price is paired with unused slack.']]} />
    <Prose>At the budget optimum (2,3), ∇f=(−2,−2), ∇g=(1,1), and λ=2. The vectors sum to zero. The budget expression is 0, so λg=0. All four checks hold. The objective's preferred local direction points out of the feasible half-plane; the constraint normal accounts for that obstruction. A constrained optimum need not have ∇f=0.</Prose>
    <ScalarKktLab />
    <H3>Why these checks certify a convex problem</H3>
    <Prose>Suppose f and every gᵢ are differentiable convex functions on ℝⁿ and every equality hⱼ is affine. With λ≥0, L is convex in x: nonnegative weights preserve the inequalities' convexity, and an affine term is convex for either sign of ν. Stationarity then makes x a global minimizer of L by the supporting-plane inequality. Therefore L(x,λ,ν)=q(λ,ν). Primal feasibility and complementary slackness make L(x,λ,ν)=f(x). We obtain f(x)=q(λ,ν), so weak duality forces both to equal p*.</Prose>
    <Prose>This is a sufficiency proof for a <em>given</em> KKT pair. No additional Slater condition is needed to validate it. With a retained convex domain D, replace the unconstrained zero-gradient step by “x minimizes L over D,” or the appropriate normal-cone condition. A boundary point of D can minimize L while its ordinary derivative is nonzero. Section 7 handles nonnegative local domains explicitly.</Prose>
    <H3>Active, positive-price and unique are different properties</H3>
    <Prose>Complementarity implies λᵢ&gt;0 ⇒ gᵢ(x)=0. The converse is false. Minimize x² subject to x≥0. At x=0 the constraint is active, yet λ=0 satisfies stationarity 2x−λ=0. Tightening the constraint to x≥δ for δ&gt;0 changes the optimum cost to δ². Zero price here means zero first-order change at δ=0, not no effect for every change.</Prose>
    <Prose>Multipliers can also be nonunique even when the primal answer is unique. Minimize (x+1)² subject to both −x≤0 and −2x≤0. The unique optimum is x=0. Stationarity requires λ₁+2λ₂=2, so (2,0), (1,.5) and (0,1) are all valid nonnegative multiplier pairs. The duplicated restrictions do not identify a unique division of the balancing contribution.</Prose>
    <RunnableExample example={examples.conditions}><Prose>Each output tuple checks primal feasibility, multiplier sign, stationarity and complementarity in that order. In the row (c,x,λ)=(−1,1,4), the derivative balances but the point wastes positive-priced slack. In (−1,−1,0), the unconstrained minimum passes balance and complementarity but is infeasible. No single check can replace the other three.</Prose></RunnableExample>

    <H2>5. Separate certificates from existence</H2>
    <Prose>Knowing how to check a certificate does not prove every optimal point has one. <strong>Constraint qualifications</strong> are assumptions that prevent certain degenerate constraint descriptions and help establish multiplier existence. A useful sufficient example is <strong>Slater's condition</strong>.</Prose>
    <Prose>For our present setting of finite differentiable convex f and gᵢ on all of ℝⁿ, with affine equalities, suppose there is a point satisfying the equalities and every inequality strictly: gᵢ(x)&lt;0. If the primal optimal value is finite, this condition guarantees strong duality and an attained dual optimum. It does not by itself guarantee that the primal infimum is attained. If a primal optimum is also attained, optimal primal and dual points satisfy KKT. More general versions use the relative interior of the common domain and allow refined treatment of affine inequalities; the simple test here is sufficient, not necessary.</Prose>
    <KktLogicFigure />
    <Prose>The original budget problem passes this simple test: for example, choose coordinates whose sum is strictly below b. Its squared-distance objective grows without bound as the point escapes to infinity, so its minimum is attained on the nonempty closed feasible half-plane. In other problems those are separate arguments. Even minimizing eˣ over ℝ has the finite infimum 0 without any point attaining it.</Prose>
    <H3>Equal values can occur without an optimal multiplier</H3>
    <Prose>Minimize x subject to x²≤0. The only feasible point is x=0, so p*=0 and the primal optimum is attained. Both the objective and inequality are convex, but no point satisfies x²&lt;0. The Lagrangian is x+λx². For λ&gt;0 its minimizer is x=−1/(2λ), and q(λ)=−1/(4λ). At λ=0 it is unbounded below.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
q(\lambda)&=\begin{cases}-1/(4\lambda),&\lambda>0,\\-\infty,&\lambda=0,\end{cases}\\
\sup_{\lambda\ge0}q(\lambda)&=0=p^*,\\
\left.\frac{\partial L}{\partial x}\right|_{x=0}&=1\ne0.
\end{aligned}`}</MathBlock>
    <Prose>The dual values approach 0 as λ grows, but no finite λ attains 0 and no KKT multiplier exists at the primal optimum. Strong duality is still true as an equality of optimal values. Failure of Slater did not force a positive gap; it removed a sufficient guarantee of the better behavior.</Prose>
    <Prose>The equivalent constraint x=0, written as an affine equality, describes the same feasible set differently. Its Lagrangian is (1+ν)x; ν=−1 makes it identically zero and supplies a certificate. Constraint representation affects gradients and multiplier existence, even when the allowed points are unchanged.</Prose>
    <RunnableExample example={examples.qualification}><Prose>The first outputs demonstrate the approaching bound and missing finite multiplier. The last output belongs to the optional nonconvex example below; it contrasts nonattainment with a genuinely positive gap.</Prose></RunnableExample>
    <details><summary>Deeper: a positive gap, and the order of minimization and maximization</summary>
      <Prose>Minimize x subject to −x≤0 and x²=1. Only x=1 is feasible, so p*=1. The equality is nonaffine, so this is not a convex formulation. With λ≥0 and free ν, L=νx²+(1−λ)x−ν. For ν&gt;0, completing the square gives q=−ν−(1−λ)²/(4ν), which is negative. For ν=0 the only finite case is λ=1, yielding q=0. For ν&lt;0 it is unbounded below. Thus d*=0 is attained at (λ,ν)=(1,0), leaving a positive gap 1. An attained dual optimum alone does not imply equality of values.</Prose>
      <Prose>For the original constraint formulation, taking the supremum over allowed multipliers at a fixed x makes any violated constraint infinitely costly and leaves a feasible point's cost f(x). Minimizing this result recovers the primal value. Reversing the order gives the dual: maximize the infimum over x. Weak duality says the reversed order cannot exceed the primal value; swapping the order as an equality needs justification.</Prose>
      <Prose>When optimal primal and dual points attain the same finite value, their Lagrangian has a saddle relation: at the selected multipliers it is minimized over x, and at the selected feasible point it is maximized over allowed multipliers. This is a useful connection to minimax methods, but arbitrary nonconvex–nonconcave training games do not inherit these guarantees.</Prose>
    </details>
    <Checkpoint prompt="A convex problem fails the simple strict-feasibility test. Must it have a positive duality gap? If you nevertheless find a valid KKT pair, should you reject it?">
      <Prose>No to both. The x²≤0 example has zero value gap despite failing the test. And the convex sufficiency proof validates any supplied KKT pair under its stated assumptions. A sufficient condition failing is not the same as the desired conclusion being false.</Prose>
    </Checkpoint>

    <H2>6. Interpret prices through reoptimization</H2>
    <Prose>A multiplier can answer a planning question: how might the best achievable cost change if we alter a constraint? Keep the objective fixed and replace g(x)≤0 by g(x)≤u. Positive u relaxes this inequality. Denote the reoptimized primal value by p(u); p(0) is the original best cost. The perturbed Lagrangian is L(x,λ)−λᵀu, so the same λ gives qᵤ(λ)=q₀(λ)−λᵀu.</Prose>
    <Prose>If λ* is an optimal original dual multiplier and strong duality holds there, q₀(λ*)=p(0). Weak duality for the changed problem now gives a global supporting lower estimate. When p is differentiable at 0 in the considered parameter neighborhood, that supporting slope equals its derivative.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
p(u)&\ge p(0)-\lambda^{*\mathsf T}u,\\
\nabla p(0)&=-\lambda^*\quad\text{if differentiable}.
\end{aligned}`}</MathBlock>
    <Prose>The sign follows the chosen right-hand-side convention. Increasing a resource allowance can reduce a minimization cost, so its derivative is nonpositive. An equality perturbation h(x)=v contributes −ν*ᵀv in the same supporting estimate; changing an equality value is not generally a relaxation, and ν may have either sign.</Prose>
    <Prose>Return to the budget b=5. Its optimal price is 2 and p(b)=max(0,7−b)²/2. Increasing the budget by .1 changes the exact cost from 2 to 1.805, a decrease .195. The linear estimate is −2(.1)=−.2. Tightening by .1 changes cost by +.205. The two changes differ because the value curve bends; a local derivative is not an exact finite-change formula.</Prose>
    <SensitivityLab />
    <H3>A corner has supporting prices without a unique derivative</H3>
    <Prose>Minimize t subject to −t≤0 and −t≤u. This is the simple requirement t≥max(0,−u), so p(u)=max(0,−u). Its Lagrangian is (1−λ₁−λ₂)t−λ₂u. A finite infimum requires λ₁+λ₂=1 with both multipliers nonnegative. At u=0 every λ₂ in [0,1] is dual-optimal. It gives the supporting line −λ₂u, but the value curve has left derivative −1 and right derivative 0. There is no single derivative to report.</Prose>
    <Prose>These prices describe alternative supporting slopes. They are not contradictory solver outputs, nor a promise that the same slope predicts every direction. Even a smooth zero-price case can have a second-order cost for tightening, as the b=7 quadratic showed.</Prose>
    <H3>Prices have units and depend on constraint scaling</H3>
    <Prose>If the objective is measured in cost units and a constraint expression in resource units, λ must have cost-per-resource units so λg can be added to f. Replace g≤0 by 10g≤0: the feasible set is unchanged, but the corresponding multiplier becomes λ/10. The balancing vector and price term remain the same. In the budget example λ=2 becomes .2; multiplying .2 by the scaled gradient (10,10) still gives (2,2).</Prose>
    <Prose>Compare multipliers only with their constraint definitions and units. A large raw number is not automatically a more important constraint. For sensitivity after rescaling, also scale the right-hand-side perturbation: the same original resource change δ becomes 10δ in the scaled expression.</Prose>
    <RunnableExample example={examples.sensitivity}><Prose>The printed negative zero is a floating-point display of zero, not a negative price or distinct bound. The corner rows compare actual changed cost with each supporting line. The smooth rows compare actual cost <em>change</em> with the linear change, so their quantities have deliberately different labels.</Prose></RunnableExample>

    <H2>7. Coordinate local decisions with a shared price</H2>
    <Prose>Duality is also a way to organize computation. Suppose two tasks choose nonnegative allocations x₁ and x₂ with a shared budget x₁+x₂≤b. Their individual penalties are (x₁−3)² and 2(x₂−4)²: the second task is more sensitive to missing its target. These are a small modeled resource tradeoff, not a claim about real hardware throughput. If allocations carry physical resource units, the penalty coefficients carry the matching objective-per-resource-squared units.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\min_{x_1,x_2\ge0}f(x),\\
f(x)=(x_1-3)^2+2(x_2-4)^2,\\
x_1+x_2\le b.
\end{gathered}`}</MathBlock>
    <Prose>Keep x₁,x₂≥0 in the domain, and attach λ≥0 only to the shared budget. At a fixed price the Lagrangian separates into two independent minimizations plus the constant −λb. Task 1 minimizes (x₁−3)²+λx₁ over x₁≥0; its unconstrained answer 3−λ/2 is clamped at zero. Task 2 similarly chooses max(0,4−λ/4). Each task needs the common price and its own penalty, not the other task's detailed function.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
x_1(\lambda)&=\max(0,3-\lambda/2),\\
x_2(\lambda)&=\max(0,4-\lambda/4),\\
q(\lambda)&=f(x(\lambda))\\
&\quad+\lambda(x_1(\lambda)+x_2(\lambda)-b),\\
q'(\lambda)&=x_1(\lambda)+x_2(\lambda)-b.
\end{aligned}`}</MathBlock>
    <Prose>The derivative is excess total demand. You can verify it by substituting the local solutions in each interval λ&lt;6, 6&lt;λ&lt;16 and λ&gt;16; the derivatives match at 6 and 16. A <strong>projected dual-gradient ascent</strong> step raises the price when demand exceeds the budget and clips the result to the allowed nonnegative prices.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
r_k&=x_1(\lambda_k)+x_2(\lambda_k)-b,\\
\lambda_{k+1}&=\max(0,\lambda_k+\alpha r_k).
\end{aligned}`}</MathBlock>
    <Prose>At b=5, λ₀=0 gives local allocations (3,4), demand 7 and violation 2. With α=1, λ₁=2. The next local allocations are (2,3.5), still .5 over budget, so λ₂=2.5. The exact optimal price is 8/3; allocations become (5/3,10/3), and cost 8/3. This differs from section 1 because task 2's penalty now has weight 2.</Prose>
    <ResourceDualAscentLab />
    <Prose>For this particular scalar dual, q′ is continuous and has slopes −3/4, −1/4 and 0 in its three price ranges. It is Lipschitz with constant 3/4. Standard projected gradient ascent on this smooth concave function has a sufficient constant-step range 0&lt;α&lt;2/(3/4)=8/3, with an attained optimum for these nonnegative budgets. This bound belongs to this model. At b=5 and α=3, the prices cycle 0→6→0; at α=4 they cycle 0→8→0. A nonnegative price can remain a valid bound while the algorithm fails to converge.</Prose>
    <Prose>Most early local allocations violate the shared budget. Their objective is not a primal upper bound. The example constructs a separate feasible witness: give task 1 min(x₁,b), then give task 2 min(x₂,max(0,b−the first allocation)). This priority repair is simple and explicit, but it is not generally the nearest feasible point or the optimum. Evaluating its objective and subtracting q yields a valid bound gap in exact arithmetic. Floating-point roundoff is separate from that mathematical argument.</Prose>
    <details><summary>Numerical detail: compute a tiny gap without subtracting nearly equal costs</summary>
      <Prose>Let a be the exact local allocation at the selected price, r any feasible repaired allocation, w=(1,2) the quadratic weights and t=(3,4) the targets. Define dᵢ=2wᵢ(aᵢ−tᵢ)+λ. Expanding each quadratic around a gives the following equivalent gap. When aᵢ&gt;0, dᵢ=0; when aᵢ=0, dᵢ≥0 and rᵢ≥0. Every term shown is therefore nonnegative.</Prose>
      <MathBlock>{String.raw`\begin{aligned}
f(r)-q(\lambda)&=\sum_i w_i(r_i-a_i)^2\\
&\quad+\sum_i d_i(r_i-a_i)\\
&\quad+\lambda\!\left(b-\sum_i r_i\right).
\end{aligned}`}</MathBlock>
      <Prose>The helper uses this expression and the priority repair's remaining budget to avoid cancellation. It preserves a genuinely positive tiny gap instead of clamping a negative subtraction to zero. The computation still uses floating point; only the mathematical argument, with its stated feasibility and global-local-minimum conditions, is exact.</Prose>
    </details>
    <RunnableExample example={examples.resource}><Prose>The α=1 iterate has a small but nonzero budget violation after eight updates. Rounded lower and upper values look identical, while the stable gap is about 1.86×10⁻⁹. Use the feasible repair and the full checks, not apparent agreement of printed decimals. The two oscillating runs demonstrate failure of a particular step size, not a failure of weak duality.</Prose></RunnableExample>
    <H3>Where this decomposition is useful</H3>
    <Prose>A communication network can assign prices to shared link capacities. A route's price combines the links it uses; local senders adjust their own rates, while each link updates its price from congestion. The same separation appears in power allocation and scheduling independent workloads under a shared resource budget. The transferable mechanism is a small set of coordinating multipliers, not the literal quadratic penalties used here.</Prose>
    <Prose>Real distributed implementations add communication delays, inexact local solves and finite message precision. A constrained reinforcement-learning formulation can similarly put a multiplier on an expected cost limit, but estimating that cost from data and learning a nonconvex policy do not automatically give this convex model's convergence or a deployment-time safety guarantee. The useful connection is how a constraint influences an objective and how feasibility must still be checked.</Prose>

    <H2>8. Check a solver and reveal an ML dual</H2>
    <Prose>A practical solver can supply a point and multipliers, but its returned status is based on numerical tolerances. With CVXPY, keep an explicit constraint object and read its <Code>dual_value</Code> after a successful solve. Check the original problem's feasibility, multiplier sign, stationarity and complementarity in their original units. Do not interpret a missing dual or an unhandled solver status as a numerical certificate.</Prose>
    <Prose>The following complete solve returns to the original equal-weight budget problem. It checks four residuals, then reconstructs the analytically feasible point (2,3) and compares its objective with the dual formula. This example's known closed form makes reconstruction easy. General problems need a justified feasible witness and a valid global dual evaluation; a small KKT residual alone has no universal conversion into a suboptimality bound.</Prose>
    <RunnableExample example={examples.solver}><Prose>The program uses a nonnegative version of the returned λ for the bound. The algebraically equivalent formula q(λ)=2−(λ−2)²/2 reduces cancellation near the optimum. It still runs in floating point: a displayed gap of zero is a rounded numerical result, while the separate exact calculation λ=2, x=(2,3), f=q=2 is the mathematical certificate. This small example intentionally requires CVXPY's <Code>OPTIMAL</Code> status and checks residuals rather than silently accepting every termination state.</Prose></RunnableExample>
    <H3>Why a classifier's dual uses feature inner products</H3>
    <Prose>A <strong>hard-margin linear support vector machine</strong> chooses a separating plane wᵀa+β=0 for labeled feature vectors aᵢ with labels yᵢ∈&#123;−1,+1&#125;. For this subsection assume the finite training set is linearly separable and contains both classes. Enforce yᵢ(wᵀaᵢ+β)≥1 and minimize ‖w‖²/2. The normalization makes minimizing ‖w‖ equivalent to maximizing the geometric separation between the two margin planes. It is not a guarantee of generalization to new data.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
L&=\tfrac12\|w\|^2\\
&\quad+\sum_i\alpha_i[1-y_i(w^{\mathsf T}a_i+\beta)],\\
\alpha_i&\ge0.
\end{aligned}`}</MathBlock>
    <Prose>Minimizing over w gives w=Σᵢαᵢyᵢaᵢ. Minimizing over the unrestricted offset β forces Σᵢαᵢyᵢ=0; otherwise a nonzero linear slope makes the infimum −∞, exactly as section 3 taught. Substituting these conditions produces the dual below. Feature vectors appear only through pairwise dot products.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
\max_{\alpha}\quad &\sum_i\alpha_i\\
&-\frac12\sum_{i,j}\alpha_i\alpha_j y_i y_j a_i^{\mathsf T}a_j,\\
&\alpha_i\ge0,\qquad \sum_i\alpha_i y_i=0.
\end{aligned}`}</MathBlock>
    <Prose>For a finite separable dataset, scale a separating plane enough to make every margin inequality strict. This supplies a Slater point, and the standard hard-margin formulation attains its optimum. Complementarity says αᵢ&gt;0 can occur only at a tight margin. A tight margin need not have a strictly positive coefficient in a degenerate case; the active-versus-positive distinction still applies.</Prose>
    <RunnableExample example={examples.svm}><Prose>The four two-dimensional points form a small fully reproducible example. The two middle points, (−1,1) and (1,1), carry coefficients .5 each. Their weighted difference reconstructs w=(1,0); the outer points have margin slack 1 and coefficient zero. The program solves both primal and dual independently and checks their relation, rather than assigning a claimed optimum by hand.</Prose></RunnableExample>
    <Prose>The Gram matrix with entries aᵢᵀaⱼ is positive semidefinite because cᵀKc=‖Σᵢcᵢaᵢ‖²≥0. A kernel replaces these dot products with dot products of another feature representation; a valid positive-semidefinite kernel preserves this convex dual structure. An arbitrary similarity table does not. For nonseparable data, a soft-margin formulation introduces slack and a penalty C, changing the dual constraints to 0≤αᵢ≤C. Full kernel choice, soft margins and evaluation belong to the SVM lesson; the purpose here is to derive why the dual has this form.</Prose>
    <Prose>A useful caution about “only support vectors matter”: a zero-coefficient point can be moved without changing an existing optimal certificate only if the changed constraints still accept that certificate. Moving it across a margin can change the solution. Complementarity describes the current optimization problem, not immunity to arbitrary changes in the data.</Prose>

    <H2>9. Practise new constraints and certificates</H2>
    <Prose>Work without the lab's optimal-pair buttons first. A successful answer names the constraint convention, gives the calculation, and explains why its checks are sufficient. Use the complete programs as a way to test a derived answer, not as the only source of it.</Prose>
    <Practice title="A new normal vector" hint="Project the target onto aᵀz=b with a=(1,2). The excess is aᵀt−b, and the normal correction is excess divided by ‖a‖²." solution={<><Prose>Here aᵀt=4, so the excess is 2 and ‖a‖²=5. Subtract (2/5)(1,2) from (2,1): z*=(8/5,1/5). Its cost is 4/25+16/25=4/5. With g=x+2y−2, stationarity gives λ*=4/5: 2(z*−t)+(4/5)a=0. The constraint is tight and λ is positive, so all KKT conditions hold for this convex problem. Substitution into q(λ)=λ(aᵀt−b)−λ²‖a‖²/4 gives 4/5.</Prose><Prose>Acceptance checks: point shape (2,), x+2y=2, positive price, objective and global dual bound both .8. Changing the budget to 4 makes the target feasible on the boundary with zero multiplier.</Prose></>}>
      <Prose>Minimize (x−2)²+(y−1)² subject to x+2y≤2. Derive the point, value and multiplier. Then explain what changes at budget 4. Modify the general half-space program only after calculating the answer.</Prose>
    </Practice>
    <Practice title="Repair an invalid certificate" hint="Use f−q only after checking the constraint and the multiplier sign. A number called L is not necessarily q." solution={<><Prose>The point (4,4) violates x+y≤5; its objective 1 cannot be a primal upper bound. With λ=2 the true lower bound is q=2, so the formal difference −1 is not an error estimate. Moving to the feasible point (2,2) gives f=5 and gap 3. Moving instead to (2,3) gives gap 0. Merely replacing q by L((2,2),1)=4 would be wrong: 4 exceeds the actual optimum 2.</Prose><Prose>If λ is negative, fix the dual sign or justify a different legitimate constraint convention; do not relabel an accidentally small value as a standard inequality-dual certificate.</Prose></>}>
      <Prose>A report says: “At budget 5, candidate (4,4) has cost 1 and λ=2 gives bound 2, so the error is at most −1.” Diagnose the claim and supply both a nonzero-gap and a zero-gap valid replacement.</Prose>
    </Practice>
    <Practice title="An equality with a signed price" hint="Write L=x²+y²+ν(x−y−2). Balance both coordinate derivatives before imposing the equality." solution={<><Prose>Stationarity gives x=−ν/2 and y=ν/2. The equality x−y=2 implies −ν=2, so ν=−2 and (x,y)=(1,−1). The cost is 2. Substitution gives q(ν)=−ν²/2−2ν, maximized at −2. Equality feasibility and global minimization of the convex L close the gap. If the equality is written 2−x+y=0, its multiplier flips to +2; the balancing vector stays the same.</Prose></>}>
      <Prose>Minimize x²+y² subject to x−y=2. Derive the multiplier using both orientations of the equality and explain why a negative answer is allowed.</Prose>
    </Practice>
    <Practice title="Activity, qualification and finite change" hint="Separate the multiplier equation from the value function. For the degenerate example, compute ∂L/∂x at the only feasible point." solution={<><Prose>For min x² with x≥0, x*=0 and λ*=0. The constraint is active; tightening to x≥.2 changes the optimum cost to .04. There is no contradiction with a zero derivative at the original threshold. For min x with x²≤0, the derivative 1+2λx equals 1 at the only feasible point x=0, so no finite multiplier solves stationarity. Nevertheless q=−1/(4λ) approaches p*=0. The second problem exhibits dual nonattainment, not a positive value gap. A supplied KKT pair in a convex problem would still be sufficient without a Slater test.</Prose></>}>
      <Prose>Compare min x² subject to x≥0 with min x subject to x²≤0. Both have the unique optimum x=0. Explain their different multiplier behavior, and calculate the first problem's cost after tightening to x≥.2.</Prose>
    </Practice>
    <Practice title="Price a change without promising its outcome" hint="At b=5 the original optimal price is 2. The supporting line is p(5)−2δ; the exact value uses the changed budget." solution={<><Prose>For δ=.5 the lower estimate is 1; the exact new cost is (7−5.5)²/2=1.125. The linear predicted change is −1, while the actual change is −.875. For δ=−.5 the lower estimate is 3 and the exact cost is 3.125. With the scaled constraint 10(x+y−5)≤u, the original multiplier is .2. The same original relaxation .5 corresponds to u=5, so the supporting change remains −.2×5=−1. Comparing prices 2 and .2 without the expression's units would be misleading.</Prose></>}>
      <Prose>At the original budget 5, give a supporting estimate and the exact reoptimized cost for changes +.5 and −.5. Repeat the price interpretation after multiplying the constraint expression by 10.</Prose>
    </Practice>
    <Practice title="Audit one distributed price step" hint="At price 2 the local allocations are 3−2/2 and 4−2/4. Evaluate the full objective before adding λ times the violation." solution={<><Prose>The local choices are (2,3.5), demand 5.5 and violation .5. Their objective is 1+2(.5)²=1.5, which is not a primal upper bound because they are infeasible. The dual bound is 1.5+2(.5)=2.5. Priority repair gives (2,3), cost 1+2=3; the valid gap is .5. At rate 1 the next price is 2.5. The true optimum 8/3 lies between 2.5 and 3. At rate 3 from price 0, demand 7 raises the price to 6; demand then becomes 2.5, dropping the projected price back to 0. Valid dual bounds do not force progress under this oversized step.</Prose></>}>
      <Prose>For the weighted two-task problem at b=5 and current price 2, compute the local allocation, violation, q, priority repair, upper bound and next price at α=1. Then explain the α=3 cycle from zero without relying on the animation.</Prose>
    </Practice>
    <Prose>You are ready to move on when you can derive a dual function over the correct domain, distinguish active constraints from positive prices, validate a convex KKT pair, and state which existence or sensitivity assumptions are being used. The next module topic, <strong>Second-Order Methods (L-BFGS, K-FAC, Shampoo, Natural Gradient)</strong>, returns to how curvature and geometry guide computational steps. Keep the distinction: an algorithm's update rule and an optimality certificate answer different questions.</Prose>

    <Sources alternatives={<div><p><strong>Another way to learn it:</strong> choose an explanation that addresses the step you found hardest.</p><ul>
      <li><a href="https://see.stanford.edu/Course/EE364A/82" target="_blank" rel="noreferrer">Stephen Boyd, Stanford EE364A Lecture 8 — Lagrangian duality (official video)</a>. Revisit the bound mechanism and why inequality prices have a sign. The official bookmarks place Lagrangians at 01:57 and weak/strong duality at 47:02. Selected companion transcript passages were reviewed; this is not a claim of watching the full recording.</li>
      <li><a href="https://see.stanford.edu/Course/EE364A/83" target="_blank" rel="noreferrer">Stanford EE364A Lecture 9 — KKT and sensitivity (official video)</a>. Useful after sections 4–6: KKT starts around 04:28 and sensitivity around 23:55. The course provides a <a href="https://see.stanford.edu/materials/lsocoee364a/transcripts/ConvexOptimizationI-Lecture09.html" target="_blank" rel="noreferrer">readable transcript alternative</a>; selected KKT and sensitivity passages, plus the official chapter list, were inspected. Keep this page's explicit differentiability and finite-change distinctions alongside the lecture's intuition.</li>
      <li><a href="https://web.stanford.edu/class/ee364b/lectures/decomposition_notes.pdf" target="_blank" rel="noreferrer">Boyd and collaborators — Notes on decomposition methods</a>. A more advanced written route from the two-task price mechanism to shared constraints and network flow. Section 3.2 and the network-flow discussion were read; communication, convergence and recovery details extend beyond the small deterministic lab.</li>
    </ul></div>}>
      <li><a href="https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf" target="_blank" rel="noreferrer">Boyd, Vandenberghe and Nobel — Convex Optimization slides, duality chapter</a>. Primary reference for dual domains, weak/strong duality, KKT sufficiency and necessity, Slater refinements and supporting sensitivity estimates. The targeted duality and sensitivity slides were reviewed, not the complete deck.</li>
      <li><a href="https://www.cvxpy.org/tutorial/advanced/index.html#dual-variables" target="_blank" rel="noreferrer">CVXPY — Dual variables</a>. Direct documentation for explicit constraint objects and their returned <Code>dual_value</Code>. The lesson's numerical residual and reconstruction checks are additional problem-specific work.</li>
      <li><a href="https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation" target="_blank" rel="noreferrer">scikit-learn — SVM mathematical formulation</a>. The current soft-margin formulation helps compare the derived hard-margin dual with the added upper bounds C and kernel Gram matrix. Its API was not executed here; the example solves the stated primal and dual directly in CVXPY.</li>
    </Sources>
  </div>
};
