import { H2, H3, Prose, Code } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import MechanismProgram from '../../components/lesson-labs/MechanismProgram.jsx';
import { mechanismProgram } from '../optimal-transport-mechanism-program.js';
import { Checkpoint, LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample as CompleteExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { TransportGeometryFigure, TransportPlanLab, TransportDualLab, CumulativeTransportLab, SinkhornScalingLab, SinkhornStabilityLab, SinkhornBiasLab, BarycentricProjectionFigure, DisplacementFigure } from '../../components/lesson-labs/OptimalTransportLabs.jsx';
import { optimalTransportExamples as examples } from '../optimal-transport-examples.js';
function Example({
  example
}) {
  return <section><Prose><strong>Before running.</strong> {example.question}</Prose><CompleteExample example={example} /></section>;
}
function Practice({
  title,
  prompt,
  hint,
  children
}) {
  return <section className="lesson-check"><h3>{title}</h3><Prose>{prompt}</Prose><details><summary>Hint</summary><Prose>{hint}</Prose></details><details><summary>Show explained solution</summary>{children}</details></section>;
}
export default {
  title: 'Optimal Transport (Wasserstein Distance, Sinkhorn)',
  readTime: '~75 min read + 2–3 hours practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot optimal-transport-lesson">
    <LessonIntro prerequisites="Weighted sums, probability mass, matrices and basic derivatives. Measure Theory introduced joint laws, marginals and pushforwards; we reconnect those ideas to a finite table before using general measures. Convex Optimization supplies useful review for the later certificate. No prior transport solver is assumed." sections={[['1-compare-distributions-with-a-sense-of-distance', 'Why geometry matters'], ['2-build-a-mass-conserving-plan', 'Construct the ledger'], ['3-distinguish-cost-from-wasserstein-distance', 'Metric, powers and units'], ['4-prove-that-a-plan-is-cheapest', 'Optimality certificates'], ['5-use-order-in-one-dimension', 'Cumulative and quantile transport'], ['6-derive-entropic-scaling', 'Entropy and Sinkhorn steps'], ['7-separate-numerical-error-from-regularization', 'Stability, stopping and bias'], ['8-correct-self-comparison-with-care', 'Three different objectives'], ['9-use-the-plan-and-choose-the-right-extension', 'Maps, applications and variants'], ['10-practise-the-complete-transport-decision', 'Independent practice']]}>Two services finish jobs at different times. One shifts most jobs a little later; another leaves almost everything unchanged but produces a small group of very late jobs. A difference between histogram bars alone does not describe how far the probability moved. Build a mass-moving ledger, prove which ledger is cheapest, then investigate what a fast smoothed approximation actually computes.</LessonIntro>

    <H2>1. Compare distributions with a sense of distance</H2>
    <Prose>Start with an extreme simplification: every job takes exactly zero time in a reference system. In one comparison every job takes .2 time units; in another every job takes 2. Both new distributions put all their mass somewhere the reference puts none. Yet the first change is much smaller on the time axis.</Prose>
    <Prose><strong>Optimal transport</strong> asks how cheaply we could reassign the reference probability mass to reproduce the comparison distribution. Every move has a cost per unit mass. Multiply that cost by the amount moved, add all moves, and choose the cheapest valid plan. The movement is a mathematical comparison; it does not assert that we physically changed the same jobs or discovered their causal histories.</Prose>
    <TransportGeometryFigure />
    <Prose>A <strong>Dirac measure</strong>, written δₓ, puts probability one at the location x. The figure compares δ₀ with δ₀.₂ and δ₂. Under ordinary distance, their transport costs are .2 and 2. Total variation only sees that the occupied locations differ; KL is infinite in the displayed direction because the target assigns zero mass to the reference's atom. These are different questions, not a contest in which one quantity is always best.</Prose>
    <Prose>The location is part of the distribution. Two vectors of weights both equal to <Code>[.5, .5]</Code> can describe different laws if their atoms occupy different places. For categorical labels, numeric IDs do not automatically give meaningful distances. A user ID of 11 is not intrinsically closer to 12 than to 300.</Prose>
    <Prose>The examples below are declared mathematical fixtures. Seven complete programs use Python's standard library; the general linear-programming example uses NumPy and SciPy. Save a block as <Code>transport_example.py</Code> and run <Code>python transport_example.py</Code>. For that optional solver example, install its stated dependencies with <Code>python -m pip install numpy scipy</Code>. The browser computes separate small deterministic models and downloads no Python solver.</Prose>

    <H2>2. Build a mass-conserving plan</H2>
    <Prose>Retain the original small example. The source has half its mass at 0 and half at 2. The target has half at 0 and half at 1. Write source weights as a = (.5, .5) and target weights as b = (.5, .5). We use letters a and b for weights so they cannot be confused with the order of a Wasserstein distance later.</Prose>
    <Prose>A <strong>coupling</strong>, or transport plan, is a table π. Its entry πᵢⱼ is the mass sent from source location xᵢ to target location yⱼ. A row spends one source's available mass. A column receives one target's required mass. Nonnegativity prevents cancelling a bad shipment with negative probability.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\pi_{ij}\geq0,\qquad \sum_j\pi_{ij}=a_i,\\
\sum_i\pi_{ij}=b_j,\\
\sum_i a_i=\sum_j b_j=1.
\end{gathered}`}</MathBlock>
    <Prose>The feasible set Π(a,b) contains every table satisfying these constraints. It is nonempty: sending πᵢⱼ = aᵢbⱼ always works. That particular table is the joint law of independently drawn source and target locations. Transport optimizes over other possible dependence structures while retaining both marginals. Knowing the marginals does not tell us which pairings actually occurred.</Prose>
    <Prose>With ordinary distance, the four costs are 0 → 0: 0, 0 → 1: 1, 2 → 0: 2, and 2 → 1: 1. Keeping the half already at 0 and moving the other half one unit costs .5. Sending both halves across the alternative routes costs .5 × 1 + .5 × 2 = 1.5. Both ledgers conserve mass, but they are not equally good.</Prose>
    <MathBlock>{String.raw`C=\begin{bmatrix}0&1\\2&1\end{bmatrix},
\qquad \pi^*=\begin{bmatrix}.5&0\\0&.5\end{bmatrix}.`}</MathBlock>
    <TransportPlanLab />
    <Prose>The two-by-two constraints leave only one adjustable entry. If the first source weight is a and the first target weight is b, choose t = π₀₀; the other three entries are forced. Here the scalar a and b denote first weights, not the whole vectors.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\pi(t)=
\begin{bmatrix}t&a-t\\b-t&1-a-b+t\end{bmatrix},\\
\max(0,a+b-1)\leq t,\\
t\leq\min(a,b).
\end{gathered}`}</MathBlock>
    <Prose>Each bound is a nonnegativity condition on one entry. In the original .5/.5 example, t ranges from 0 to .5 and the total cost is 1.5 − 2t. Increasing t helps until its upper bound .5. For other costs the slope can reverse or be zero; the investigation exposes all three cases.</Prose>
    <Prose>A coupling may split an atom. A <strong>Monge map</strong> instead assigns each source location one destination: Y = T(X). A source δ₀ cannot be mapped deterministically to half at −1 and half at 1, because T(0) has only one value. A coupling can make that split. For equally weighted clouds with the same number of points, a permutation assignment achieves an optimum, but it need not be unique and other optimal couplings can exist.</Prose>
    <Checkpoint prompt="The source has .2 at 0 and .8 at 2; the target requires .7 at 0 and .3 at 1. Could a plan using only the two diagonal entries satisfy both marginals?">
      <Prose>No. The first source can supply only .2 of the .7 required at target 0. At least .5 must travel from source 2 to target 0. A feasible plan is [[.2, 0], [.5, .3]]. Splitting the second source is essential; a diagonal-only picture would silently violate the column requirement.</Prose>
    </Checkpoint>

    <H2>3. Distinguish cost from Wasserstein distance</H2>
    <Prose>For a declared cost matrix C, the general finite transport problem minimizes the sum of mass times cost. The bracket below is just that sum, not a new operation you must already know.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\mathcal T_C(a,b)=
\min_{\pi\in\Pi(a,b)}\langle C,\pi\rangle,\\
\langle C,\pi\rangle=\sum_{i,j}C_{ij}\pi_{ij}.
\end{gathered}`}</MathBlock>
    <Prose>It is a <strong>linear program</strong>: the objective and marginal equations are linear in the unknown entries, with nonnegative constraints. A finite cost table produces a well-defined optimization problem even when those costs are asymmetric or do not obey a triangle inequality. Calling that arbitrary value a distance would promise additional properties it may not have.</Prose>
    <Prose>A ground <strong>metric</strong> d has nonnegative symmetric distances, is zero only between the same point, and obeys d(x,z) ≤ d(x,y) + d(y,z). To define the r-Wasserstein distance, use the cost Cᵢⱼ = d(xᵢ,yⱼ)ʳ for an order r ≥ 1, then take the rth root of the minimum. Do not apply the power twice.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
C_{ij}=d(x_i,y_j)^r,\\
W_r(\alpha,\beta)=
\left(\min_{\pi\in\Pi(a,b)}
\langle C,\pi\rangle\right)^{1/r}.
\end{gathered}`}</MathBlock>
    <Prose>The laws α and β include their locations and weights. W₁ measures average movement under ordinary distance. W₂ minimizes mean squared displacement and then takes a square root. If coordinates are seconds, W₁ and W₂ both have units of seconds; W₂² has units of seconds squared. For the original fixture, the only occupied nonzero displacement is one unit, so W₁ = .5 but W₂ = √.5 ≈ .7071. The equality of the two unrooted costs in that fixture is a coincidence of its unit-length move.</Prose>
    <Prose>For general probability measures, replace the finite sum with an integral over a joint law π whose marginals are α and β. On a metric space, restrict attention to distributions with finite rth moments if you need a finite Wᵣ. Heavy tails can give an infinite transport distance even when both laws are valid probability distributions.</Prose>
    <details><summary>Deeper: why the triangle inequality needs a root</summary>
      <Prose>In a finite space, take a coupling P from a to b and Q from b to c. Join them through their shared middle mass. For bⱼ &gt; 0, the joint mass of a three-step combination (i,j,k) is PᵢⱼQⱼₖ/bⱼ. Ignore zero middle weights: their entire rows/columns in the two plans are already zero. Summing this three-way table over j gives a valid a-to-c plan.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
R_{ijk}=\frac{P_{ij}Q_{jk}}{b_j}\quad(b_j>0),\\
S_{ik}=\sum_{j:b_j>0}R_{ijk}.
\end{gathered}`}</MathBlock>
      <Prose>Every individual direct distance is at most the sum of its two legs. Minkowski's inequality—the triangle inequality for an rth-power average followed by its root—therefore bounds the rooted cost of S by the sum of the rooted costs of P and Q. Choosing their optima proves the transport triangle inequality. For r = 1 this is just summing the ordinary distance inequality. For r &gt; 1 the root is crucial: squared distance alone already fails the triangle inequality for atoms at 0, 1 and 2, since 4 is greater than 1 + 1.</Prose>
    </details>

    <H2>4. Prove that a plan is cheapest</H2>
    <Prose>Trying many feasible plans does not prove that a cheaper one is absent. A <strong>dual certificate</strong> gives a lower bound valid for every plan. Choose source prices fᵢ and target prices gⱼ such that fᵢ + gⱼ never exceeds the cost Cᵢⱼ of that route. Prices may be negative; they are optimization multipliers, not literal fees paid by a customer.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
f_i+g_j\leq C_{ij}\quad\text{for every route},\\
D(f,g)=\sum_i a_if_i+\sum_j b_jg_j.
\end{gathered}`}</MathBlock>
    <Prose>Multiply every route inequality by its nonnegative mass and add. Because row and column sums are fixed, the weighted sum of prices becomes D(f,g), whatever feasible plan you chose. Thus the price value is a lower bound, while the cost of any feasible plan is an upper bound.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\langle C,\pi\rangle-D(f,g)\\
=\sum_{i,j}\pi_{ij}(C_{ij}-f_i-g_j)\geq0.
\end{gathered}`}</MathBlock>
    <Prose>The quantity Cᵢⱼ − fᵢ − gⱼ is the <strong>slack</strong> for that route. If every occupied route has zero slack, the bounds meet and the plan is optimal. This is complementary slackness in a visible table. In the original example, f = (0,1) and g = (0,0) give lower bound .5; the diagonal plan also costs .5. The two unused routes have slack 1.</Prose>
    <TransportDualLab />
    <Prose>Why can we find a tight certificate? The finite feasible set is nonempty and bounded, and the linear objective is finite. Finite-dimensional linear-programming duality guarantees equal optimal values and attained optima. The general dual is the largest D(f,g) subject to those route inequalities. A quick derivation also explains where the inequalities come from.</Prose>
    <details><summary>Deeper: derive the finite dual from the marginal equations</summary>
      <Prose>Let L be the Lagrangian: the plan cost plus each multiplier times its marginal-equation error. It depends on π, f and g.</Prose>
      <MathBlock>{String.raw`\begin{gathered}
\begin{aligned}
L&=\langle C,\pi\rangle\\
&\quad+f^\top(a-\pi\mathbf1)\\
&\quad+g^\top(b-\pi^\top\mathbf1)\\
&=a^\top f+b^\top g\\
&\quad+\sum_{i,j}\pi_{ij}(C_{ij}-f_i-g_j).
\end{aligned}
\end{gathered}`}</MathBlock>
      <Prose>For fixed f and g, minimize L over nonnegative π without enforcing its marginal equations. If any coefficient of π is negative, increasing that entry sends L to minus infinity. Otherwise the minimum of the last sum is zero. A useful finite lower bound therefore requires fᵢ + gⱼ ≤ Cᵢⱼ, and the remaining value is precisely aᵀf + bᵀg. Maximizing that value yields the dual. The earlier weighted-slack identity is the direct check that avoids relying on a solver's success flag alone.</Prose>
    </details>
    <Example example={examples.twoLocationCertificate} />
    <Prose>The changed fixture gives cost 7/10, not the original .5. Its optimal prices differ because the needed target mass changed. The code considers the two feasible endpoints and the two breakpoints of a piecewise-linear dual; it does not approximate the optimum by an arbitrary grid search.</Prose>
    <Example example={examples.linearProgram} />
    <Prose>The general program flattens a three-by-three plan into nine unknowns, constructs one equation per row and column, and uses SciPy's HiGHS linear-programming interface. One equality is redundant because the totals agree; the solver handles that dependency. It checks feasibility and the returned multiplier certificate. The particular optimizer can differ across solver versions when optima are nonunique, so the displayed invariant results are more useful than promising one selected matrix.</Prose>

    <H2>5. Use order in one dimension</H2>
    <Prose>On a line, we can often avoid a dense solver. Put source and target locations in sorted order. Match as much as possible between the leftmost remaining source and target; when one side is exhausted, advance it. This matches equal ranges of cumulative probability, so it is called <strong>quantile coupling</strong>.</Prose>
    <Prose>For the cost |x − y|ʳ with r ≥ 1, crossed pairings cannot improve the cost. If x ≤ x′ and y ≤ y′, convexity of the displacement cost gives |x − y|ʳ + |x′ − y′|ʳ ≤ |x − y′|ʳ + |x′ − y|ʳ. Replacing a little mass on two crossing routes by noncrossing routes keeps all marginals. Repeating removes crossings and gives the sorted matching rule. Equality can occur, so this does not prove uniqueness in every case.</Prose>
    <Prose>W₁ has an especially useful second view. At a cut between neighboring locations, let Fα(x) and Fβ(x) be the cumulative probabilities to the left. If the source has .3 more mass there, at least .3 must cross that gap to the right. A monotone plan needs no unnecessary movement in both directions. Multiply the magnitude of this imbalance by the gap length and add all gaps.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
W_1(\alpha,\beta)\\
=\int_{\mathbb R}|F_\alpha(x)-F_\beta(x)|\,dx.
\end{gathered}`}</MathBlock>
    <CumulativeTransportLab />
    <Prose>For finite histograms the CDFs are constant inside each gap, so the integral is a sum of rectangle areas. Before the first atom both CDFs are zero; after the last they are one. The sign tells you the net direction; the absolute value prevents opposite gaps from cancelling their transportation work. This CDF-area formula is specific to W₁, not a formula for W₂².</Prose>
    <Prose>More generally, let F⁻¹(u) be the smallest location whose cumulative probability is at least u. Drawing a single uniform u and setting X = Fα⁻¹(u), Y = Fβ⁻¹(u) pairs equal probability ranks. The same construction works for atoms: a plateau in the quantile function can send portions of a source atom to different targets.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
W_r(\alpha,\beta)^r\\
=\int_0^1|F_\alpha^{-1}(u)-F_\beta^{-1}(u)|^r\,du.
\end{gathered}`}</MathBlock>
    <Example example={examples.weightedLine} />
    <Prose>The independent gap sum gives 3/2, agreeing with the sorted plan's ordinary-distance cost. Squaring the route lengths instead gives 5/2, whose square root is 1.581139. Sorting costs O(n log n + m log m) in general; the subsequent weighted sweep uses O(n + m) steps. Equal-weight sorted samples with equal counts simply pair by rank. Do not substitute that simpler rule when weights or counts differ.</Prose>
    <Checkpoint prompt="One forecast puts .9 at 0 seconds and .1 at 10 seconds; another puts everything at 1 second. Both means are1. Is their W1 zero?">
      <Prose>No. The only destination is 1, so .9 must move one second and .1 must move nine seconds. W₁ = .9 + .9 =1.8 seconds. Equal means do not determine equal distributions; the plan makes the difference explicit.</Prose>
    </Checkpoint>

    <H2>6. Derive entropic scaling</H2>
    <Prose>A general plan has nm entries. Even storing every pairwise cost can be expensive for large point clouds. One approach changes the objective so its optimum has a structure we can compute with repeated row and column operations. This is <strong>entropic regularization</strong>; the smoothing is a modelling/computational choice, not a promise that the original optimum was already smooth.</Prose>
    <Prose>For a probability table, its entropy is H(π) = −Σπᵢⱼ log πᵢⱼ, with 0 log 0 defined as zero. Larger entropy means more spread across pairs. We minimize transport cost minus ε times entropy. The extra constant −ε in the following convention does not affect the minimizer, but it does affect a displayed objective value.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
F_\varepsilon(a,b)=\min_{\pi\in\Pi(a,b)}J_\varepsilon(\pi),\\
\begin{aligned}
J_\varepsilon(\pi)&=\langle C,\pi\rangle\\
&\quad+\varepsilon\sum_{i,j}\pi_{ij}(\log\pi_{ij}-1),
\end{aligned}\\
\varepsilon>0.
\end{gathered}`}</MathBlock>
    <Prose>With finite costs and positive weights on the active rows/columns, the solution is unique and positive. Compactness gives existence; strict convexity of the entropy penalty gives uniqueness. Intuitively, introducing a tiny amount into a zero entry provides an initially very large entropy benefit, so a positive feasible reference such as a⊗b prevents a zero entry at the optimum. A zero-weight row must remain zero and can be removed before solving. Infinite-cost forbidden routes require a separate support-feasibility analysis.</Prose>
    <Prose>Add multipliers f and g for the marginal equations and differentiate the regularized objective with respect to a positive πᵢⱼ. The derivative of π(log π −1) is log π. Setting the derivative to zero reveals the product structure:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
C_{ij}+\varepsilon\log\pi_{ij}-f_i-g_j=0,\\
\pi_{ij}
=e^{f_i/\varepsilon}\,
e^{-C_{ij}/\varepsilon}\,
e^{g_j/\varepsilon}
=u_iK_{ij}v_j.
\end{gathered}`}</MathBlock>
    <Prose>K = exp(−C/ε) is a matrix of relative preferences for cheap routes. It generally is not a probability table. The unknown positive factors u and v must make its row and column sums right. Holding v fixed, the row equation is uᵢΣⱼKᵢⱼvⱼ = aᵢ, so uᵢ = aᵢ/(Kv)ᵢ. Then solve the column equations for v using the newly updated u.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
u\leftarrow a\,/\,(Kv),\\
v\leftarrow b\,/\,(K^\top u).
\end{gathered}`}</MathBlock>
    <Prose>All divisions are entry by entry. These are <strong>Sinkhorn iterations</strong>. In exact arithmetic with strictly positive K and active positive marginals, alternating scaling converges to the unique scaled table. After correcting rows, correcting columns generally disturbs those rows again. That is why one pass is not enough.</Prose>
    <SinkhornScalingLab />
    <Example example={examples.originalScaling} />
    <Prose>The plan contains approximately .059601 mass on each off-diagonal route. Its linear transport cost .619 is above the unsmoothed optimum .5. That is not a failed marginal constraint: entropy made the spreading worthwhile in the new objective. The simple loop executes a fixed 100 sweeps; it does not itself report convergence, so the next implementation adds a residual and an explicit budget.</Prose>
    <Prose>For this equal-weight two-by-two case we can independently derive the regularized answer. Write its diagonal entries as t and off-diagonal entries as .5 − t. If Δ = C₀₀ − C₀₁ − C₁₀ + C₁₁, differentiating the one-variable objective gives Δ + 2ε log(t/(.5−t)) =0. Hence:</Prose>
    <MathBlock>{String.raw`t=
\frac{.5}{1+\exp(\Delta/(2\varepsilon))}.`}</MathBlock>
    <Prose>In the original cost matrix Δ = −2 and ε = .5, so t ≈ .440399. The closed form is a useful check on the iterative solver. It is not a general formula for arbitrary weights or larger matrices. In the investigation comparing objective values, this formula supplies the reference independently of the stopped iteration.</Prose>

    <H2>7. Separate numerical error from regularization</H2>
    <Prose>Three different errors can matter. <strong>Optimization error</strong> is what remains because an iteration stopped early. <strong>Regularization bias</strong> is the deliberate difference between the smoothed problem and the unregularized one. <strong>Sampling error</strong> arises when finite observed data approximate unknown population distributions. Reducing one does not automatically reduce the others.</Prose>
    <Prose>Start with feasibility. Check both marginal errors, for example the maximum absolute difference between a computed row/column sum and its required weight. A matrix that fails this check is not yet a feasible plan; its linear term is not a valid primal upper bound. A tiny residual is necessary but should not be advertised as a universal objective-error or population-error certificate.</Prose>
    <Prose>Ordinary exponentials create another problem: exp(−Cᵢⱼ/ε) can round to zero when C/ε is large. A row sum may become zero, leading to division by zero. Instead represent the scaling factors by logarithms. If ℓᵢⱼ = −Cᵢⱼ/ε, Aᵢ = log uᵢ and Bⱼ = log vⱼ, the same updates become:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
A_i\leftarrow\log a_i-
\operatorname{LSE}_j(\ell_{ij}+B_j),\\
B_j\leftarrow\log b_j-
\operatorname{LSE}_i(\ell_{ij}+A_i).
\end{gathered}`}</MathBlock>
    <Prose>LSE is log-sum-exp: for a finite list z, subtract its largest value M before exponentiating, then add M back after taking the logarithm. The largest exponential is then 1 instead of overflowing, and relative weights remain available.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\operatorname{LSE}(z)=M+\log\sum_k e^{z_k-M},\\
M=\max_k z_k.
\end{gathered}`}</MathBlock>
    <SinkhornStabilityLab />
    <Prose>Adding a constant to every cost changes the objective by that constant because total mass is 1, so it leaves the best plan unchanged. The stability investigation uses this exact invariance to distinguish a numerical failure from a mathematical change. Log u and log v also have a harmless gauge freedom: adding the same constant to every log u and subtracting it from every log v leaves π unchanged. Periodic recentering can keep their values manageable.</Prose>
    <Prose>Stable arithmetic does not make a poorly conditioned iteration instantly converge. For small ε, opposing scale factors may require many sweeps. Epsilon continuation—solving a less concentrated problem first and reusing its information—can help in appropriate implementations. Always retain a cap and report failure to meet the requested tolerance. Restoring exp(A + ℓ + B) can still round genuinely tiny entries to zero, so inspect the resulting marginal residual rather than assuming that a log-domain label proves correctness.</Prose>
    <Example example={examples.stableSinkhorn} />
    <Prose>The offset case succeeds with four vanished ordinary kernel entries. The final sharply regularized shifted case does not meet its tolerance within 5,000 sweeps: its remaining marginal error is about .00005. Reporting that result is more useful than quietly describing the candidate as an exact plan. For larger problems, use a maintained solver with clear cost, regularization, tolerance and return-value conventions.</Prose>
    <Prose>The units of ε are the same as the cost, because it trades cost against a dimensionless entropy term. Multiplying all costs and ε by the same positive factor leaves C/ε, and therefore the regularized plan, unchanged. Multiplying costs alone changes the tradeoff. A copied default ε means little without the feature units and cost scale.</Prose>
    <details><summary>Deeper: a finite bound on the linear-cost bias</summary>
      <Prose>Let π* minimize the original cost and πε minimize its entropic version. Comparing the two regularized objectives gives a bound on the extra linear cost. The constant −ε cancels:</Prose>
      <MathBlock>{String.raw`\begin{gathered}
0\leq \langle C,\pi_\varepsilon\rangle
-\langle C,\pi^*\rangle\\
\leq\varepsilon\bigl[H(\pi_\varepsilon)-H(\pi^*)\bigr]\\
\leq\varepsilon\min\{H(a),H(b)\}.
\end{gathered}`}</MathBlock>
      <Prose>For discrete joint laws, H(π) is at least either marginal entropy and at most their sum. The first fact follows from nonnegative conditional entropy; the second from nonnegative KL relative to a⊗b. These give the last inequality. This is a bound for exact optimizers of a finite probability problem, not a certificate for any early iterate. As ε tends to zero, the regularized linear cost approaches the original optimum; if multiple plans are optimal, the entropy term selects the maximum-entropy member in the limit. As ε grows without bound, the plan tends toward the independent coupling a⊗b.</Prose>
    </details>
    <Prose>Dense storage needs O(nm) entries and a dense sweep needs O(nm) arithmetic; the number of sweeps also matters. Exact 1D methods, blocked/lazy pairwise operations and structured kernels address different cases. GPU batching can accelerate suitable kernels but does not remove their statistical assumptions or automatically remove quadratic work. No time curve on this page is presented as a benchmark.</Prose>

    <H2>8. Correct self-comparison with care</H2>
    <Prose>“Sinkhorn cost” is ambiguous unless the author or library says what it returns. The <strong>linear term</strong> is ΣCπ evaluated at a regularized plan. The <strong>full objective</strong> Fε includes the entropy term, so it can be negative under our convention. Neither necessarily vanishes when a distribution is compared with itself. Do not put either raw value into a formula that assumes an ordinary distance.</Prose>
    <Prose>A <strong>Sinkhorn divergence</strong> corrects the two self-comparisons. Use the same cost rule, regularization and full objective convention for all three solves:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
S_\varepsilon(\alpha,\beta)=F_\varepsilon(\alpha,\beta)\\
-\tfrac12F_\varepsilon(\alpha,\alpha)
-\tfrac12F_\varepsilon(\beta,\beta).
\end{gathered}`}</MathBlock>
    <SinkhornBiasLab />
    <Prose>For identical inputs the cancellation is exact algebra. Broader nonnegativity and separation claims need hypotheses on the cost and associated kernel. A standard supported setting is bounded-support Euclidean probability measures with cost ‖x−y‖ or ‖x−y‖² and ε &gt; 0. The relevant kernel exp(−C/ε) has the positivity properties used in the theorem. This does not justify arbitrary asymmetric cost matrices, a general triangle inequality, or calling Sε an unbiased estimate of a population distance. See the annotated original <a href="https://proceedings.mlr.press/v89/feydy19a.html" target="_blank" rel="noreferrer">Sinkhorn-divergence paper</a> for its exact conditions.</Prose>
    <Prose>Some definitions regularize by ε KL(π || a⊗b) instead of our negative entropy. For fixed marginals they select the same plan. In a finite probability table, KL(π || a⊗b) = −H(π) + H(a) + H(b), so those objective values differ by marginal-only terms and the constant in our convention. These terms cancel in the three-way debiasing formula when used consistently. Subtracting only the three linear terms is generally a different quantity.</Prose>
    <Prose>In the default contracted example at ε =1, the linear cross cost is about .738406, the full cross objective about −1.320075, and the debiased divergence about .538778. The unregularized W₂² is .5. Those numbers are all compatible: they answer distinct questions. Removing self-bias does not force exact agreement with unregularized transport at a finite regularization.</Prose>

    <H2>9. Use the plan and choose the right extension</H2>
    <Prose>A scalar cost says how expensive the best movement is. The plan says where the movement went. Inspect both. A small average cost can hide a rare but consequential route, an inappropriate match across labels, or a feature normalization that erased the distinction you cared about.</Prose>
    <H3>A correspondence is not automatically a deterministic mapping</H3>
    <Prose>For a source with positive weight aᵢ, πᵢⱼ/aᵢ is a conditional distribution over destinations. In Euclidean coordinates, its conditional mean is the <strong>barycentric projection</strong> T̄(xᵢ) = Σⱼπᵢⱼyⱼ/aᵢ. This can provide a convenient point-valued summary for colour or feature transfer. It generally does not reproduce the target law when applied to every source.</Prose>
    <BarycentricProjectionFigure />
    <Example example={examples.projection} />
    <Prose>The program makes the loss of spread exact. The target variance is 2; the mapped variance is 1; the remaining conditional variance is 1. This is the conditional-variance decomposition from probability. The projection is a meaningful conditional average, but replacing the conditional distribution by its mean discards information. Neither this coupling nor an unsupervised alignment identifies causal correspondence or guarantees preservation of class labels.</Prose>
    <DisplacementFigure />
    <Prose>More generally, push a coupling through the location (1−t)X + tY to create an interpolation with the correct endpoints. For an optimal Euclidean quadratic coupling this gives a constant-speed W₂ path. Mixing the endpoint measures, (1−t)α + tβ, is another path and describes a different distribution between endpoints.</Prose>
    <details><summary>Deeper: continuous maps and a complete Gaussian calculation</summary>
      <Prose>For Euclidean squared cost, finite second moments and a source absolutely continuous with respect to Lebesgue measure, Brenier's theorem gives a unique optimal map up to source-null sets, represented as the gradient of a convex potential. The absolute-continuity condition is why the single-source-atom splitting counterexample does not contradict the theorem. This is a qualified theorem, not a claim that every empirical point cloud has a unique smooth map.</Prose>
      <Prose>In one dimension, take X = μₐ + σₐZ and Y = μᵦ + σᵦZ using the same standard-normal Z, with positive standard deviations. Their squared displacement expectation is (μₐ−μᵦ)² + (σₐ−σᵦ)². It is minimal because every coupling's covariance is at most σₐσᵦ by Cauchy–Schwarz, and this shared-Z construction attains that upper bound. Thus N(0,1) and N(3,4), where the second parameter denotes variance, have W₂² = 9 +1 =10. Their map is T(x) =3 +2x, and W₂ =√10. The factor 2 changes spread; a mean shift alone would not achieve the required target.</Prose>
    </details>
    <H3>Applications where the representation changes the question</H3>
    <LessonTable caption="Use the geometry to make a concrete comparison" headers={['Setting', 'Useful construction', 'Question still outside the cost']} rows={[['Job completion or arrival times', '1D weighted CDF/quantile transport; W1 in seconds summarizes required timing displacement', 'Mean geometric change is not a deadline violation rate; inspect the actual tail event too.'], ['Colour or feature alignment', 'Use a declared meaningful colour/feature cost and inspect the conditional destinations', 'Averaging destinations can blur modes or labels; unsupervised matching is not causal identity.'], ['Generated versus observed samples', 'A geometric distribution loss can vary when point supports are disjoint', 'A learned critic, empirical sample and finite optimization add distinct approximations; gradients need not be informative everywhere.'], ['Distributions of future returns', 'Quantile comparisons retain more than an expected return', 'A distributional RL algorithm has its own Bellman/control assumptions; this metric alone does not prove them.']]} />
    <Prose>For a rare subgroup of mass η moved distance D while the rest is retained at one shared location, W₁ is ηD in the simple two-atom fixture. Different combinations of rarity and displacement can therefore share the same scalar value. Conversely, small empirical transport cost does not establish population equality: sample size, tails and dimension matter. Define the practical quantity—timing shift, tail risk, label consistency or another decision—before declaring a match successful.</Prose>
    <H3>Three extensions change three different assumptions</H3>
    <Prose><strong>Unbalanced transport</strong> relaxes exact marginal conservation. If one histogram contains 2 units and another 8, a balanced plan cannot satisfy both totals. Normalizing both to 1 deliberately removes that total difference; it is appropriate only when relative composition is the intended question. Another option penalizes changed marginals with a specified discrepancy.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
\min_{\pi\geq0}U_\tau(\pi),\\
\begin{aligned}
U_\tau(\pi)&=\langle C,\pi\rangle\\
&\quad+\tau\,\operatorname{KL}_+(\pi\mathbf1\mid a)\\
&\quad+\tau\,\operatorname{KL}_+(\pi^\top\mathbf1\mid b),
\end{aligned}\\
\operatorname{KL}_+(r\mid a)=\sum_i\phi(r_i,a_i),\\
\phi(s,t)=s\log(s/t)-s+t.
\end{gathered}`}</MathBlock>
    <Prose>This is one particular unregularized unbalanced objective, with τ &gt; 0 controlling the penalty and positive reference amounts in the displayed formula. It is not ordinary KL between normalized probability vectors. The minimizing transported amount can differ from both inputs; the penalty defines the cost of those changes. Other unbalanced/partial formulations impose different rules.</Prose>
    <Example example={examples.relaxedMass} />
    <Prose>In the single-route example the derivative gives m* = √(ab) exp(−c/(2τ)) ≈3.115203. Its positive curvature proves the minimum. That amount exceeds the original 2 and falls below 8 because the stated model permits both marginal changes. It would be an invalid plan if physical mass conservation were required. Increasing τ for unequal totals cannot make both incompatible exact marginals hold simultaneously.</Prose>
    <Prose><strong>Sliced Wasserstein</strong> projects points onto one-dimensional directions and averages their one-dimensional transport costs before the appropriate root. This can make a high-dimensional comparison cheaper. An integral over all unit directions is different from a finite set of projections. A small chosen set can miss entire dependence differences.</Prose>
    <Example example={examples.slicedDirections} />
    <Prose>The two point clouds share their x and y marginals, but opposite diagonals have different dependence. Two axis projections give zero discrepancy. A diagonal direction reveals the difference; the full W₂ is 2. The printed three-direction value is the declared finite approximation, not the exact spherical integral. This links geometric comparison to the earlier warning that matching marginal distributions does not imply matching a joint law.</Prose>
    <Prose><strong>Gromov–Wasserstein</strong> is useful when there is no sensible cross-space point cost but each dataset has meaningful internal distances. A quadratic version compares pairwise distances under a coupling:</Prose>
    <MathBlock>{String.raw`\begin{gathered}
D_{ikjl}=|d_X(x_i,x_k)-d_Y(y_j,y_l)|^2,\\
\min_{\pi\in\Pi(a,b)}
\sum_{i,k,j,l}D_{ikjl}\pi_{ij}\pi_{kl}.
\end{gathered}`}</MathBlock>
    <Prose>Unlike the earlier linear program, this objective is quadratic in π and generally nonconvex. Two equally weighted line segments with points (0,2) and(10,12) have identical internal distances, so an appropriate matching gives zero Gromov cost, although their ordinary W₂ is 10. Ignoring global translation is desirable for some shape comparisons and wrong for a question about absolute physical position. The variant changes what equivalence means; it is not simply a faster ordinary Wasserstein solver.</Prose>
    <Prose>Further study can develop barycenters, dynamic transport and PDEs, statistical estimation, constrained matching or transport-based generative training. The existing <a href="/learn/topic/rectified-flow-flow-matching">Rectified Flow & Flow Matching</a>, <a href="/learn/topic/gan-variants-dcgan-wgan-stylegan-cyclegan-pix2pix">GAN Variants</a> and <a href="/learn/topic/distributional-rl-c51-qr-dqn-iqn">Distributional RL</a> topics own their full algorithms. Their presence in the catalogue does not imply their older pages have this lesson's review status.</Prose>

    <section id="optimal-transport-code-route" aria-label="Vectorized Sinkhorn and POT">
      <H3>Turn the log updates into a reusable dense solver</H3>
      <Prose>The scalar log-domain derivation above remains the mechanism. This complete array implementation applies each row and column update with SciPy's stable log-sum-exp primitive, shifts the common cost baseline, fixes the dual gauge, and checks both marginal residuals. POT receives exactly the same positive probability weights, cost matrix and epsilon with <Code>method="sinkhorn_log"</Code>. Install NumPy 2.3.5, SciPy 1.18.1 and POT 0.9.7; run <Code>python sinkhorn-library-bridge.py</Code>.</Prose>
      <MechanismProgram {...mechanismProgram} title="Complete vectorized log-Sinkhorn and POT comparison" />
      <Prose>The first plan reproduces the equal-weight reference from this lesson; a three-by-two fixture tests unequal masses and rectangular costs. The program recomputes three quantities from the returned plan: transport cost, our entropy objective εΣπ(log π−1), and the version using εΣπ log π. For unit total mass the last two differ by ε, even when their optimizer is identical. POT's plan-returning call is not itself a reported objective. Adding 1000 to every cost preserves the plan and raises its transport objective by 1000.</Prose>
      <Prose>Each dense update costs O(nm); K iterations cost O(Knm), and returning the full plan requires O(nm) storage. This supports strictly positive probability marginals and finite, representable cost/epsilon ranges. Remove zero-mass atoms and reinsert zero rows or columns when extending it. Very small epsilon can exhaust the budget even without underflow: the program exposes that state instead of claiming convergence. Sparse, implicit, accelerated and unbalanced solvers require separate choices; exact one-dimensional transport and the HiGHS LP remain owned by the preceding programs.</Prose>
      <details><summary>Implementation practice: preserve the regularization ratio</summary><Prose>Multiply the costs and epsilon by 3, then repeat with only the costs multiplied. Compare plans, objectives and residuals.</Prose><details><summary>Solution and checks</summary><Prose>Scaling both keeps cost/epsilon and the plan unchanged; all three objective conventions scale by 3. Scaling costs alone sharpens the effective regularization and usually changes the plan. Match both marginals before interpreting a numerical discrepancy as a modelling effect.</Prose></details></details>
      <Prose><a href="https://pythonot.github.io/all.html#ot.sinkhorn" target="_blank" rel="noreferrer">POT's Sinkhorn API</a> supplies the tested method and stop controls. Its stopping statistic differs from the independently checked maximum coordinate residual used here.</Prose>
    </section>
    <H2>10. Practise the complete transport decision</H2>
    <Prose>For each problem, state the locations, weights, cost and exact question before calculating. A correct number for the wrong ground geometry is not a successful solution. Solve independently, then use the hint or compare the reasoning—not just the final decimal.</Prose>
    <Practice title="1. Construct a changed plan and certificate" prompt="Source locations 0,2 have weights (.2,.8); target locations 0,1 have weights (.7,.3). Use ordinary distance. Find every feasible plan using one variable, its minimum cost, and a matching dual lower bound." hint="Use t=π00, then choose between the two feasible endpoints. Try f=(0,2) and g=(0,−1) for the certificate.">
      <Prose>t ranges from 0 to .2 and the plan is [[t,.2−t],[.7−t,.1+t]]. Its cost is 1.7−2t, so the minimum is 1.3 at t=.2, with plan [[.2,0],[.5,.3]]. The suggested prices satisfy all four route inequalities and have value .8×2 +.3×(−1)=1.3. Every occupied route has zero slack. The negative target price is valid; interpreting it as an impossible literal shipping charge would confuse the dual with the physical cost table.</Prose>
    </Practice>
    <Practice title="2. Keep powers and units straight" prompt="A source places .75 at 0 seconds and .25 at 4 seconds. The target puts all mass at 1 second. Compute W1, W2 squared and W2. Would doubling all time coordinates double all three quantities?" hint="The target has only one location, so no optimization choice remains. Average displacement or squared displacement, then take the required root.">
      <Prose>W₁ = .75×1 +.25×3 =1.5 seconds. W₂² =.75×1² +.25×3² =3 seconds squared; W₂ =√3 ≈1.732051 seconds. Doubling coordinates doubles W₁ and W₂ but multiplies W₂² by 4. A library returning the minimum squared cost has not already taken the root.</Prose>
    </Practice>
    <Practice title="3. Explain every CDF-area contribution" prompt="On locations 0,1,3, compare source weights (.5,0,.5) with target weights (0,.75,.25). Calculate W1 by gap areas and construct a plan that attains it." hint="Check cumulative differences on [0,1) and [1,3). Their signs need not agree.">
      <Prose>The first gap has difference .5 and width 1, contribution .5. The second has difference .5−.75=−.25 and width 2, contribution .5. Total W₁ =1. Move .5 from 0 to 1, .25 from 3 to 1 and retain .25 at 3. Its cost .5×1 +.25×2 =1 matches the lower bound. Signed differences alone would cancel and incorrectly suggest zero work.</Prose>
    </Practice>
    <Practice title="4. Repair an invalid optimality argument" prompt="For the original half-mass fixture, someone chooses f=(0,3), g=(0,0), obtains price value 1.5 and concludes the diagonal plan costing .5 cannot be correct. Locate the failure. Then explain why a feasible certificate below .5 would not prove the plan is bad either." hint="Check every f_i+g_j against the corresponding cost. A lower bound is useful only after its inequalities hold.">
      <Prose>For the route 2→0 the prices sum to 3, exceeding cost 2; for 2→1 they sum to 3, exceeding cost 1. The purported bound is invalid. A different feasible but loose certificate would simply fail to prove optimality. The plan may still be optimal—as the valid f=(0,1), g=(0,0) certificate establishes.</Prose>
    </Practice>
    <Practice title="5. Diagnose a scaling run" prompt="You multiply all ground costs by 100, leave epsilon unchanged, and run one row correction followed by one column correction. The column sums look perfect. Explain three reasons this is not evidence that you computed the original regularized optimum accurately. State checks or repairs." hint="Separate a changed mathematical objective, two marginal constraints and floating-point conditioning.">
      <Prose>Keeping ε fixed changes the cost/entropy ratio, so it is a different regularized plan; scale ε by 100 too if preserving that tradeoff is intended. Column correction may disturb rows; inspect both residuals. Larger C/ε can underflow the ordinary kernel or slow scaling; use stable log-domain arithmetic, explicit tolerance/cap and convergence reporting. Neither log-domain arithmetic nor a tiny marginal residual removes deliberate regularization or sampling error.</Prose>
    </Practice>
    <Practice title="6. Derive an equal-weight regularized reference" prompt="Use costs [[0,1],[1,0]], weights (.5,.5) on both sides, and epsilon 1. Derive the diagonal mass t, the linear cost and the full objective in this lesson's convention. Explain the self-comparison divergence." hint="Here Δ=−2 and t=.5/(1+exp(−1)). There are two copies of t and two copies of .5−t in the entropy.">
      <Prose>t ≈.3655292893 and each off-diagonal entry is .1344707107. The linear cost is 2(.5−t)≈.2689414214. The full objective is that cost plus 2t(log t−1)+2(.5−t)(log(.5−t)−1)≈−2.0064088681. This is the self-comparison of a two-atom law one unit apart. Sε =Fε−½Fε−½Fε=0, even though the raw linear cost is positive and the full objective is negative.</Prose>
    </Practice>
    <Practice title="7. Decide what a barycentric projection can preserve" prompt="A source atom at 0 splits .25 to −2 and .75 to 2. Calculate its mean destination and the target variance. What law results from the deterministic mean map, and why is it different?" hint="A single input location has a single mean output. Compare the full conditional distribution with its average.">
      <Prose>The mean destination is .25(−2)+.75(2)=1. The target variance is .25(−2−1)² +.75(2−1)² =3. The mean map produces δ₁ with variance 0, not the two-point target. It preserves the mean but discards conditional spread. The split plan itself is valid; the loss occurs when replacing it by a point summary.</Prose>
    </Practice>
    <Practice title="8. Choose the modelling assumption" prompt="For each case choose ordinary balanced, explicitly unbalanced, or intrinsic-distance transport and defend your choice: two normalized arrival-time distributions; two cell-count histograms with different meaningful totals; two shape descriptions in unrelated coordinate frames. State one quantity each choice might hide." hint="Ask whether total mass and absolute cross-space position are meaningful before choosing a solver.">
      <Prose>Balanced 1D transport fits normalized arrival-time distributions when composition and time displacement are the questions; its mean movement can hide deadline risk. An explicit unbalanced penalty can retain meaningful count differences, but its permitted mass creation/destruction needs justification and a penalty scale. Intrinsic-distance comparison such as Gromov–Wasserstein can compare shapes without a shared coordinate frame, while hiding global translation or orientation that another application might need. These are conditional choices: normalizing counts or registering coordinate frames first could answer different justified questions.</Prose>
    </Practice>
    <Practice title="9. Check an end-to-end comparison" prompt="Create two weighted 1D datasets with unequal numbers of distinct locations. Implement the sorted sweep, verify both marginals, compare its W1 with the exact CDF-area sum, and compare its squared-cost/rooted result with an independent LP. Then duplicate every atom while splitting its weight in half, and rescale coordinates by 3. What should and should not change?" hint="Duplicating representation with split weights preserves the probability law. Scaling coordinates changes metric units, not the normalized weights.">
      <Prose>Acceptance criteria: nonnegative plan entries; each original required row/column total within a declared numerical tolerance; agreement of W₁ with the CDF integral and of the squared minimum with the LP; no change in the represented distribution or optimum under duplicated split atoms. Coordinate scaling by 3 multiplies W₁ and W₂ by 3 and W₂² by 9. If comparing entropic objectives after duplicating atoms, be careful: raw table entropy depends on the representation, while an appropriately consistent KL-reference/debiased distribution comparison needs its conventions checked. Do not infer population equivalence from agreement between two algorithms on the same finite data.</Prose>
    </Practice>
    <Prose>You are ready to move on when you can explain the chosen geometry, construct and certify a small coupling, predict a scaling correction, diagnose a stopped approximation, and state what its resulting correspondence cannot establish. The next topic in this module is <a href="/learn/topic/causal-inference-do-calculus">Causal Inference & Do-Calculus</a>. It asks what can be concluded about interventions—a question that geometric similarity or a low-cost pairing alone does not answer.</Prose>
    <Sources alternatives={<ul>
      <li><a href="https://moore.pims.math.ca/lecture/video/optimal-transport-machine-learning-lecture-1" target="_blank" rel="noreferrer">Gabriel Peyré: Optimal Transport for Machine Learning, PIMS lecture 1</a> — a graduate-level lecture route with links to parts 2 and 3. Institutional speaker/date and course description were checked; full audiovisual playback was not reviewed. Revisit after the coupling and scaling sections if a spoken explanation helps.</li>
      <li><a href="https://www.gpeyre.com/ot4ml/" target="_blank" rel="noreferrer">Optimal Transport for Machine Learners</a> — the author's book, slides and notebook index. The finite dual chapter was inspected; the larger ecosystem is an optional exploration, not a claim that every linked notebook or interactive panel was tested.</li>
    </ul>}>
      <li><a href="https://arxiv.org/html/1803.00567v4" target="_blank" rel="noreferrer">Peyré & Cuturi: Computational Optimal Transport</a> — detailed theory and numerical reference. Scoped coupling, metric, dual and entropic/log-domain sections were checked. Some rendered formulas contain transcription inconsistencies; this lesson's concrete derivations and numerical references were checked independently.</li>
      <li><a href="https://papers.nips.cc/paper_files/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf" target="_blank" rel="noreferrer">Cuturi 2013: Sinkhorn Distances</a> — original machine-learning presentation of entropic matrix scaling. Section 4 uses a reciprocal regularization parameter; its historical runtime results are not a performance guarantee for this lesson.</li>
      <li><a href="https://proceedings.mlr.press/v89/feydy19a.html" target="_blank" rel="noreferrer">Feydy et al. 2019: Sinkhorn Divergences</a> — equations 1–3 and Theorem 1 explain the KL-reference convention, self-bias correction and supported positivity conditions.</li>
      <li><a href="https://pythonot.github.io/quickstart.html" target="_blank" rel="noreferrer">POT 0.9.5 quick start</a> — annotated solver options and examples. Check whether a function returns a plan, a linear value or the full objective; squared Euclidean costs need the corresponding root for W₂. Log-domain options and return conventions were inspected, not every library solver.</li>
    </Sources>
  </div>
};
