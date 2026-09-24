# Real Analysis, Sequences & Modes of Convergence — lesson design

Stable ID `real-analysis-sequences-modes-of-convergence`; Mathematical & Statistical Foundations position 54. Prepared 11 September 2026. This is an implementation design, not a finished lesson or a review certificate. The [original plan](evidence/real-analysis-original-plan.json) records a planned entry with no old body or executable examples. Retain the title: sequences and convergence modes already encompass the added completeness, series and interchange foundations. Preserve the two required prerequisites, Sets/Logic44 and Single-Variable Calculus47, stable identity and module order.

## Learning contract and scope decisions

A learner can already factor elementary expressions, read quantified statements and use derivatives/integrals, but need not know a real-analysis theorem. Begin by asking what a computation must guarantee beyond the displayed samples. Finish by giving a correct all-tail bound, proving or refuting uniform convergence on a stated domain, selecting a valid interchange theorem and producing a changed approximation report. Essential proofs remain visible in the core; supplementary constructions and measure-theoretic arguments use clearly marked depth branches.

The scope grows from the original two outcomes because the actual earlier Calculus lesson explicitly leaves completeness behind the intermediate/extreme value theorems and general series theory to this owner. The new page should close those gaps, not merely repeat the definition of uniform convergence.

| Idea | Actual prior coverage inspected | Owner and decision |
| --- | --- | --- |
| Epsilon witnesses and quantifier order | Sets44's job/reviewer dependence; Calculus47's punctured-neighborhood proof | Introduce sequence/tail notation locally, then translate witness dependence to N(epsilon,x) versus N(epsilon). Include full changed examples. |
| Completeness and continuity existence results | Calculus47 states IVT/EVT and expressly defers their real-number foundations; Functional Analysis39 introduces Cauchy completeness in function spaces | This page supplies supremum, monotone convergence, nested intervals, subsequences, real Cauchy completeness, sequential continuity and compact-interval consequences. Link function-space extensions without importing their models. |
| Infinite series and power-series operations | Counting46's formal coefficient identities; Calculus47's finite Taylor remainder, exponential argument and separately checked logarithm endpoints | Teach partial sums, Cauchy/absolute/conditional convergence, uniform series tests and interior versus endpoint operations here. Explicitly distinguish a formal generating series from evaluating an infinite numerical sum. |
| Moving errors and integral interchange | Measure Theory section6 has n times an indicator, MCT/Fatou/DCT and an exact failure of integral interchange | Use a changed continuous triangular family in the elementary core; prove finite-interval uniform integration and explain its finite-length assumption. Probability/L1 extensions are optional and locally introduced. |
| Fourier jump neighborhoods | Fourier51 supplies projections, square-wave partial sums and qualified convergence; incoming note asks for a fixed versus moving point distinction | Link the continuous-uniform-limit obstruction and a simple analytic jump lower bound. Do not duplicate the complete Fourier engine or treat a sampled maximum as proof. |
| Useful approximation with a nonsmooth target | Earlier finite Taylor examples require derivatives; Numerical Methods38 and Conditioning52 own floating computation and error sources | Bernstein polynomials provide a distinct positive-weight construction and an explicit uniform error bound for a Lipschitz target. This is a worked mathematical application, not a claim that this is the fastest numerical approximator. |
| Probability convergence and uniform integrability | Measure Theory has DCT but no complete modes/UI map; Random Variables48 has expectation and probability laws | Optional bridge after explicit review links. Define the modes, give counterexamples and a qualified UI-to-L1 proof. Do not add these lessons as required edges of the elementary core. |
| Metric spaces, functional approximation and weak PDE limits | Topology40 and Functional Analysis39 are existing owners; PDE56/57 are later owners | Briefly connect real completeness and operator continuity; route convergence/norm/interchange obligations to the numerical PDE author. Do not expand into an unintroduced Sobolev-space course. |

All four incoming Real Analysis notes were read in full. Their adoption is designed here; change their status to implemented only after the actual body and verification exist. The unrelated bit-manipulation inbox entry does not authorize extra analysis content.

## Core teaching sequence and checked mechanisms

### 1. A displayed trend is not an all-tail guarantee

Start with an iterative scalar estimate a_n=3+(-1)^n/(n+1), n>=1. Its error is exactly 1/(n+1). At tolerance 1/10, index 9 is on the excluded boundary and index 10 begins the safe tail. Write the quantifiers after explaining tolerance, chosen index and every later index. The sufficient/minimal index for this family is max(1,floor(1/epsilon)); rational control presets and exact native fractions resolve strict boundaries. A finite plot checks displayed terms; the inequality proves every later term.

Negate convergence in the right order and distinguish boundedness, eventual closeness and monotonicity. The sequence (-1)^n is bounded with two incompatible subsequences. An isolated late disturbance can defeat a claimed empirical certificate even when a finite prefix looked settled. Avoid saying a finite algorithm decides convergence for arbitrary supplied formulas.

### 2. Why a real limit exists

Define upper bound, least upper bound/supremum, maximum and infimum with (0,1), [0,1] and a finite set. Completeness is the real-number axiom that a nonempty set bounded above has a real supremum. Prove the Archimedean property by contradiction using sup(N), then the monotone bounded sequence theorem by choosing a term above sup−epsilon. Existence is distinct from knowing a rate.

Nested dyadic intervals in [1,2] isolate a point whose square is 2. At each step keep the half whose endpoints straddle the squared target; after k steps width is 2^-k and midpoint error is at most half that width. Completeness gives a common point; shrinking widths give uniqueness; continuity of squaring gives the equation. A parity argument proves no rational square root of 2, so being Cauchy among rationals does not force a rational limit. This is not a claim that the floating midpoint implements an exact real number. The browser uses bounded dyadic steps, and native Fraction arithmetic carries the certificate.

### 3. Recognize convergence without guessing the answer

Prove uniqueness by splitting the distance between two alleged limits; convergent sequences are bounded because only a finite prefix remains outside a bounded tail. Explain subsequences with increasing indices. Prove Bolzano–Weierstrass by repeatedly retaining a half interval containing infinitely many terms and selecting increasing indices. A Cauchy sequence asks that all sufficiently late pairs be close. Prove convergence implies Cauchy, and Cauchy implies real convergence using boundedness plus a convergent subsequence and a triangle bound.

Successive differences tending to zero are insufficient: harmonic partial sums have tiny single increments, but the block from N+1 through 2N totals at least 1/2. Contrast the telescoping sum of 1/[k(k+1)] with its exact tail 1/(n+1). Optional tail-envelope branch defines liminf/limsup for bounded sequences; monotone tail bounds converge and equality characterizes convergence. Keep unbounded extended-real cases qualified.

### 4. Continuity on a whole interval

Revisit epsilon/delta continuity briefly, then prove its equivalence with sequential continuity using a contradicting point within 1/n when a delta fails. Prove a continuous function on a closed bounded interval is bounded (otherwise choose points with growing values and use a convergent subsequence), attains its supremum (choose near-supremum points), and is uniformly continuous (contradict with pairs whose distances shrink but outputs stay separated). Define the dependency difference between delta(epsilon,a) and delta(epsilon) for all a.

Supply an IVT supremum proof for a value between endpoints, including endpoint values and both continuity directions at the cut. Closed/bounded assumptions are sufficient hypotheses, not necessary conditions for every particular function. Contrast x² on [0,2], Lipschitz bound 4, with x² on the whole real line: x_n=n and y_n=n+1/n get closer while squared values differ by 2+1/n². This continuous-but-not-uniformly-continuous failure is different from pointwise/uniform convergence of a sequence of functions.

### 5. An infinite sum is a limit of finite sums

Define partial sums, Cauchy tail criterion and the necessary zero-term test. Derive the finite geometric identity and its tail; prove harmonic divergence with dyadic blocks. Explain positive comparison and p-series via integral or grouping estimates, ratio/root tests with the inconclusive value 1, and absolute convergence implying convergence by a Cauchy bound. Alternating decreasing-to-zero magnitudes give bracketing partial sums and next-term error. Conditional convergence is not an inaccurate floating sum: exact infinite rearrangements may change the limit, whereas every permutation of a fixed finite exact sum agrees. Optional rearrangement proof explains why positive and negative subseries both diverge for the alternating harmonic series and how crossing a chosen target yields that target as term sizes vanish.

### 6. One function at a time versus one bound everywhere

Use f_n(x)=x^n on three selectable domains: [0,1], [0,1), and [0,r] for fixed r<1. At each fixed x<1, values approach zero; at x=1 they stay 1. On either domain extending arbitrarily close to 1, the supremum error is 1 and need not be attained. The moving point x_n=2^(-1/n) has error exactly 1/2. On [0,r], the supremum error is r^n and supplies a common index. Render fixed-x and moving-witness views alongside the full domain, with open/closed endpoint symbols and exact labels. The calculation that x_n^n=1/2 supplies the proof; floating plotted samples alone do not.

Define uniform convergence, its negation, and the supremum error. State the equivalence of uniform convergence and the supremum error tending to zero (allow an extended supremum for early unbounded errors). Do not silently require each f_n itself to be bounded: f_n(x)=x+1/n converges uniformly to x on R although the functions are unbounded.

### 7. A narrowing error can evade a grid

On [0,1], n>=2, let T_n(x)=max(1−|nx−1|,0). It is continuous, has support [0,2/n], peak 1 at 1/n and area 1/n. Compare T_n, n*T_n and T_n/n. Every fixed point eventually sees zero, including x=0. Their supremum errors are 1,n,1/n; L1 errors are 1/n,1,1/n²; squared L2 errors are 2/(3n),2n/3,2/(3n³). Derive these by splitting the two linear sides and changing variables. This proves small mean-square error need not give small uniform error.

A coarse grid containing 0 and then 1/m misses the entire triangle when n>2m, although the analytical peak and area are known. The lab shows full domain and local coordinate u=nx with honest axis labels; its exact three-corner polyline cannot itself miss the triangle. A deliberately sampled estimate is labelled separately and may be zero. Explain the distinction between a display failing to sample a feature and the mathematical function losing it.

The Fourier bridge uses a square wave with values −1 left of a jump, +1 right and 0 at it. Every finite trigonometric partial sum is continuous. Any continuous g within a claimed error delta on both sides would require |g(0)+1|<=delta and |g(0)−1|<=delta by continuity; thus delta>=1. The jump point's midpoint value does not repair the neighboring obstruction. Link Fourier for its actual reconstruction, not for an invented finite-grid proof.

### 8. When a limit preserves continuity and integration

Provide an inline three-segment error path f(x)→f_N(x)→f_N(a)→f(a); choose N first for uniform approximation, then delta from continuity of that fixed f_N. This proves continuity of a uniform limit. For Riemann-integrable f_n on finite [a,b], prove the limit is integrable: for a partition, upper-minus-lower sums for f differ from those for f_N by at most 2(b−a)*sup error. Then prove the integral difference bound (b−a)*sup error.

Apply to f_n=x+sin(nx)/n on [0,2]: limiting integral 2 and error at most 2/n, despite a derivative that need not converge. A separate exact integral expression confirms a tighter family-specific error but does not replace the theorem. Compare the previous constant-area triangle, where hypotheses fail. On [0,infinity), g_n=1/n on [0,n] and zero outside converges uniformly to 0 while each integral is 1; finite length did real work. Mention DCT as a differently hypothesized extension and link the actual Measure Theory proof.

### 9. Why derivatives need stronger control

For f_n=sin(nx)/n on [−pi,pi], uniform function error is 1/n, but f'_n=cos(nx) is always 1 at 0 and alternates at pi. The limit's derivative is zero. Contrast g_n=sin(nx)/n²: derivative error <=1/n, so both converge uniformly. A paired height/slope investigation shares n and x; axes keep the changing scales explicit rather than normalizing both curves into the same apparent amplitude.

State and prove a sufficient theorem: f_n in C1([a,b]), f_n'→g uniformly, and f_n(c)→ell at one fixed c imply f_n→f uniformly, f in C1, f'=g. Define f=ell+integral_c^x g, use uniform-limit continuity of g and FTC, and bound sup|f_n−f| by base-value error plus interval length times derivative error. The example f_n=n shows why derivative convergence alone misses drifting constants. Optional rounded-corner family sqrt(x²+1/n²)→|x| has uniform error <=1/n and a nondifferentiable limit; proof uses rationalization and the exact value at zero.

### 10. Build a function from a controlled series

Derive the Weierstrass M-test from the uniformly Cauchy tails. It is sufficient, not necessary. For a power series with radius R, distinguish every compact subinterval |x−a|<=r<R from the entire open interval and its endpoints. Establish a radius using the root limsup or an available ratio; the root limsup is interpreted with 1/0=infinity and 1/infinity=0. Prove interior derivative-series control by choosing r<s<R and bounding k(r/s)^(k−1), while convergence at the center fixes the constant. Integrate on compact subintervals using the preceding theorem. Endpoints require separate arguments.

Use the changed family sum_{k>=1} x^k/k². On [−1,1], tail absolute error <=sum_{k>n}1/k²<=1/n, so the series is uniformly convergent; at x=1 its derivative series is harmonic and diverges. The latter is an endpoint limit, not a contradiction of interior termwise differentiation. Show actual positive tail and derivative partial sums without assuming a Basel closed value. Formal coefficient identities only become numerical identities after convergence has been supplied.

Optional smooth-versus-analytic branch defines h(0)=0 and h(x)=exp(−1/x²) otherwise. Repeated derivatives away from zero are a polynomial in 1/x times the exponential. Prove every such product tends to zero using e^t>=t^m/m! for a sufficiently large m, then inductively show every derivative at zero is zero. The Taylor series is zero, while h(x)>0 off zero. No claim that all smooth functions equal their Taylor series.

### 11. Approximate a corner by averaging nearby values

Give a fully derived positive-weight polynomial construction on [0,1]. Weights b_{n,k}(x)=choose(n,k)x^k(1−x)^(n−k) sum to one, their weighted k/n mean is x and weighted squared deviation is x(1−x)/n. These finite identities can be checked from the binomial theorem without requiring measure theory. Set B_n f(x)=sum b*f(k/n).

For an L-Lipschitz f, a weighted Cauchy–Schwarz bound gives |B_n f(x)−f(x)|<=L sqrt[x(1−x)/n]<=L/(2sqrt n). Use f(x)=|x−.3|, L=1; n=100 guarantees error <=.05 everywhere, not merely at sampled points. Show node values and weights for a selected x, the resulting weighted value, and the whole approximation with its analytic band. Exact endpoints, constant and linear reproduction and changed quadratic identity B_n(x²)=x²+x(1−x)/n are independent verification cases. A different construction, interpolating a few points, does not inherit this theorem.

An optional Weierstrass proof for continuous f splits near and far nodes: uniform continuity bounds nearby differences, and the second moment bounds far weight by 1/(4n*delta²), with all |f|<=M. Therefore the uniform error is at most omega_f(delta)+2M/(4n*delta²). Choose delta, then n; do not confuse polynomial approximation with a Taylor expansion or claim an optimal rate for arbitrary continuous f.

### 12. Optional bridge: probability has several convergence promises

Explicit prerequisites are measurable random variables and expectation from the linked Measure Theory/Random Variables lessons. Define almost-sure, in-probability, Lp (p>=1 finite) and in-distribution convergence. State the same-space requirement for the first three and define distribution convergence via CDF continuity points. Prove Lp→probability with Markov; prove almost-sure→probability by bounded indicators/DCT; derive probability→distribution with the epsilon CDF sandwich and continuity points. A same-distribution alternating ±Z sequence need not converge in probability to Z, because coupling matters.

Use the deterministic typewriter sequence on U in [0,1): n=2^k+j, 0<=j<2^k, X_n=1 on [j/2^k,(j+1)/2^k), zero elsewhere. P(X_n=1)=2^-k→0 and E|X_n|^p=2^-k, but each fixed U is visited once per block and is unvisited elsewhere in every sufficiently large block; thus no almost-sure convergence to zero. The interval sweep needs a native exact boundary test and a visible full row, not a random finite simulation claimed as a theorem.

Define uniform integrability by a common tail expectation cutoff. Show it implies bounded L1 norms but not conversely: X_n=n*1_(0,1/n) has mean absolute value 1 and tail expectation 1 whenever n exceeds the chosen cutoff. An integrable common dominator or a uniform (1+delta)-moment bound suffices, with proof. State and prove the direction probability convergence + UI implies L1: use an almost-sure subsequence to get integrability of the limit by Fatou (construct subsequence with summable failure probabilities); truncate at a common level and bound the bounded middle error by epsilon+2M*P(error>epsilon). Then let n, M and epsilon vary in their valid order. The full equivalence is an annotated reference, not an unexplained required theorem. No martingale convergence theorem is imported.

### 13. Changed independent practice and a final report

Use hint and explained-solution disclosures, not hidden core proofs. Candidates: changed rational tail and strict index; Cauchy failure despite shrinking steps; a supremum not attained; missing compactness; changed geometric ratio and alternating tolerance; x^(2n) on altered domains; a changed triangle height n^alpha and its norm/integral consequences; a false derivative interchange with drifting constants; derivative-series endpoint diagnosis; a Lipschitz approximation budget with L=2 and epsilon=.1; a typewriter boundary; bounded-L1 versus UI; and a complete changed integration certificate.

The capstone computes an approximation to a stated integral with a total deterministic error budget. Use f=|x−c| on [0,1], a Bernstein approximant B_n and its **exact polynomial integral** (1/(n+1))*sum_{k=0}^n f(k/n). Each basis polynomial integrates to 1/(n+1), proved by beta-integral integration by parts or a local factorial recurrence. With rational c, exact Fraction arithmetic verifies the integral and the Lipschitz uniform bound; a sampled error estimate is diagnostic only. Change c, n and the tolerance; the report separates function approximation, exact integration of the approximant and any floating display. A failed sufficient bound means “not certified by this bound,” not “actual error exceeds tolerance.” Supply all inputs, exact answer for integral|x−c|=(c²+(1−c)²)/2, intermediate output and an acceptable changed solution.

Close with actual next module topic Abstract Algebra55 and useful review/application links. Do not skip planned entries.

## Representation contracts

| Location / hurdle | Form and learner action | Encoding and verification |
| --- | --- | --- |
| Sequence tail, §1 | Discrete stem plot with horizontal tolerance band and a chosen all-tail boundary; change tolerance and proposed N | Index is horizontal, signed estimate vertical. Exact rational boundary check and analytic monotone envelope; label finite shown horizon. |
| Completeness, §2 | Step/back dyadic bracket ruler plus endpoint-square table | Bracket interval and width are exact dyadics; local zoom explicitly changes coordinates. Increasing precision does not silently turn rounded digits into exact reals. |
| Cauchy versus steps, §3 | Two tail blocks with their individual increment bars and aggregate gap | Exact Fraction harmonic and telescoping comparisons. Enough terms to reveal a block; no unbounded DOM expansion. |
| Continuity/global domain, §4 | Inline paired x² inputs on bounded versus moving unbounded domains | Actual input/output gaps and domain labels; not a generic label-changing schematic. |
| Quantifier dependence, §6 | Fixed point, moving witness and domain-wide error in aligned plots/readouts | Supremum label is analytical; open endpoint and unattained sup retained. Sampled values checked with high precision. |
| Moving triangle, §7 | Full-domain polygon, explicit u=nx zoom and deliberately coarse sample positions | Exact corner coordinates, area and norms; select amplitude scaling; distinguish full curve from grid estimate. |
| Three-part proof, §8 | Inline error path with two approximation legs and one continuity leg | Labels state N-before-delta dependency; stacked narrow layout preserves order. No unnecessary animation. |
| Derivative interchange, §9 | Paired function/slope plots and guaranteed bands | Shared n/power selector; honest differing vertical units/scales; exact reference at x=0 and pi. |
| Series, §10 | Partial sum and common tail budget; endpoint derivative comparison | Bounded native summation, exact/rational and high-precision oracles; no empirical extrapolation drawn as a theorem. |
| Bernstein construction, §11 | Weighted node strip and approximation/error band | Nonnegative weights sum to one; actual weighted point highlighted. Curve computation memoized by n/target, not inspection cursor. Max n bounded; no huge library. |
| Probability, §12 optional | Dyadic interval sweep with block/within-block index and fixed observer | Half-open intervals and exact dyadic boundaries. All events are analytical; finite visible sweep does not establish almost-sure behavior. |

Static figures and initial investigation states must make ordinary reading useful before controls are touched. The calm existing dark/amber design, Space Grotesk/JetBrains and semantic mathematical colors remain. Apply frontend-design using creator context from `.impeccable.md`; no rebrand or one-lab quota. All controls need keyboard labels, deterministic reset, actual 1440/390/320 layout checks, text equivalents and visible assumptions.

## Native programs, research and verification

Plan complete independent Python3 programs for the tail certificate, exact bracket, Cauchy block, series bounds, domain/witness comparison, triangle grid failure, integral interchange, derivative pair, power-series endpoints, Bernstein weights/integral and optional typewriter/UI. Split or combine by useful workflow, not one program per heading. Execute the actual final strings, store stdout and run changed helper calls. No browser auto-execution of displayed Python. Use standard library fractions/math first; independent verification may use mpmath/SymPy without imposing those on the core learner.

Sources browsed 11 September 2026; mathematical derivations and fixtures are written originally. Inspected scope is deliberately narrower than a whole-course claim:

| Source | Inspected scope and use |
| --- | --- |
| [Lebl, Basic Analysis](https://www.jirka.org/ra/html/ra.html) | Catalogue of chapters1–3/6/11 establishes foundation owners; selected actual §§6.1–6.2 definitions, uniform continuity/integral preservation and derivative/basepoint theorem, compact interior power-series proofs read. Own elementary completeness/sequence proofs will be checked separately. |
| [MIT18.100A lecture notes](https://www.ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/mit18_100af20_lec_full2.pdf) | Lecture24's uniform negation, x^n witness and M-test inspected, plus power-series scope. The full92-page course is not claimed read. |
| [MIT18.100A Lecture24 video](https://www.youtube.com/watch?v=gXPX29KfEc4) and [transcript](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/1vu4GMg5v7TRTz-K943dU10R7KZuWqwZN_transcript.pdf) | Official video metadata verified; selected transcript passages on witness order, interchange and uniform continuity/integration proof read. No playback or invented timestamp. [Lecture23](https://www.youtube.com/watch?v=_HRTdXJgZ0Q) metadata also verified; useful introductory alternative. |
| [MIT Yufei Zhao, Second Moment, §4.7](https://ocw.mit.edu/courses/18-226-probabilistic-methods-in-combinatorics-fall-2022/mit18_226_f22_lec06-07.pdf) | PDF pages27–28, Bernstein weighted approximation and concentration proof inspected. Use the local finite-weight/Cauchy–Schwarz Lipschitz bound; do not copy unrelated combinatorial claims or rate rankings. |
| [Durrett, Probability: Theory and Examples, fifth edition](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf) | §4.6 definition, Theorem4.6.2 moment criterion and Theorem4.6.3 truncation proof inspected; §2.3.2 subsequence criterion and selected §2.3.4 continuity application inspected. Optional advanced reference, not a required martingale course. The services.math hostname failed; the sites.math PDF succeeded. |
| [Lebl, Weierstrass approximation](https://www.jirka.org/ra/html/sec_stoneweier.html) | §11.7.1 theorem and opening convolution construction inspected as a second method; not copied into the main Bernstein workflow and not a claimed full Stone–Weierstrass proof review. |

Before publication, verify exact arithmetic and proof hypotheses, all changed practice answers, actual source imports and bounded computations, formulas parsing/fitting, screen/keyboard interactions and source links. Independently compare Bernstein weights/integrals with exact combinatorics, triangles with symbolic integration, trigonometric/nth-root witnesses with high precision, exact dyadic and strict rational boundaries, and invalid controls/models. Freeze every owned source only after native and actual-font browser review. Root registration/integration and later user acceptance remain distinct.

Semantic implementation ownership: `real-analysis-sequences-modes-of-convergence.jsx`, `real-analysis-models.js`, `real-analysis-examples.js`, `RealAnalysisLabs.jsx`, optional separate `RealAnalysisFigures.jsx` when warranted, and `real-analysis-labs.css`. The authored stable-ID blueprint points here; no cross-topic engine or eager catalogue import. Keep initial computation bounded and derive curves from their stated equations rather than hand-drawn arrays pretending to be measured data.
