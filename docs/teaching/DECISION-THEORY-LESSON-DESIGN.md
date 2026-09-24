# Decision Theory, Risk & Cost-Sensitive Decisions — lesson design

Stable ID: `decision-theory-risk-cost-sensitive-decisions`. Mathematical & Statistical Foundations, position 53. Prepared 11 September 2026. **Root assessed and approved this design; the complete lesson is now implemented.** The [author verification record](DECISION-THEORY-VERIFICATION.md) and its exact source packet distinguish final author checks from later independent review, production integration and user acceptance. The sections below retain the approved design rationale; the implemented disposition at the end records its actual choices and corrections. Root owns shared registration, prerequisite metadata and the rollout ledger.

## 1. Learning contract, baseline and prerequisites

Retain the title and stable identity. The title already covers the proposed transition from uncertain beliefs to justified actions; inspection, provisioning and information acquisition are applications, not reasons to append more nouns. The beginner finish line is to specify what can happen, what can be done and what consequences matter; calculate a feasible action or policy under a declared criterion; and explain what changes if probabilities, costs, information or constraints change. It is a substantial first course in finite statistical decisions, not a promise to solve every economic, strategic or sequential decision problem.

The exact-topic inventory and returned notes were read. The topic is planned with a starting blueprint in `cross-domain-expansion.js`, no previous published body and no native programs. The [original-plan archive](evidence/decision-theory-original-plan.json) preserves the complete CLI response, exact prior source excerpt and source hash before the individual brief. Keep the original prediction/action distinction, loss table, conditional expected losses, cost-driven threshold, abstention, uncertainty and inspection practice; deepen their mechanisms instead of replacing the starting intention. No removed legacy example or executed original program is claimed.

Retain the existing two prerequisites, unchanged:

- **Probability Distributions & Bayes' Theorem**: finite joint mass, conditional probabilities, base rates, likelihood ratios, population versus fitted probabilities.
- **Random Variables, Expectation & Covariance**: transformations before averaging, conditional expectation, total expectation and mean-squared loss. The actual sections 3–4 and 8 were inspected, including their conditions and population/fitted distinction.

Refresh normalization and a weighted sum locally. A learner can follow the finite core without matrix algebra, calculus, an optimization course or Bayesian conjugacy. The optional continuous-summary and utility derivations introduce their elementary slope/concavity facts locally, with calculus links for revision. Finite minimax certificates use weighted averages rather than assuming an LP duality theorem. Bayes' rule is locally enough to understand a prior/posterior in the finite risk example; Bayesian Inference remains an optional deeper link rather than a new hard prerequisite.

Actual adjacent route: Conditioning, Stability & Numerical Analysis52 → Decision Theory53 → **Real Analysis, Sequences & Modes of Convergence54**. Do not skip to a published learning or finance topic. The closing connection can explain that continuous decision rules and convergence of estimated risks need careful limiting arguments, while keeping this page's finite guarantees independent of the later course.

## 2. Ownership review and deliberate boundaries

| Related mechanism and actual scoped source inspection | This lesson's decision | Owner/boundary |
| --- | --- | --- |
| Probability foundation, ``2–3/5: Bayes, odds, finite moments and calibration caveat; Random Variables48, ``4/8: mean under squared loss and conditional/tower identities | Briefly refresh the arithmetic; generalize the loss and action space, then prove conditional minimization using the same weighted groups | Do not duplicate a distributions course or its moment catalogue |
| Bayesian Inference18 actual ``2–3/4/7: posterior updating, shared-parameter batch prediction, prior sensitivity and sensor precision | Use one small integrated batch-loss contrast to show why a decision can need more than a plug-in parameter; link to the actual predictive derivation | Conjugacy/MCMC/VI and posterior computation stay with their existing owners |
| Causal25 actual ``1–2/5/8–10: intervention, identification, paired counterfactuals and estimation | Add a concrete action-changing-outcome distinction and a known synthetic benefit table; state that identifying those intervention probabilities requires its own assumptions | Do-calculus, propensity estimation, experiments and causal identification are linked, not retaught |
| Convex Optimization and Constrained Multi-Objective Optimization actual objective/feasibility/Pareto sections | Teach finite capacity and tail constraints as changes to a decision problem, with an exchange argument and small exact enumeration | Do not reimplement numerical solvers or claim every weighted objective captures stakeholder requirements |
| Calibration & Conformal Prediction published ``1–2.1/3.1–2, currently not individually reviewed | Teach the local distinction between a posterior given full available information, a calibrated coarse score and a finite observed bin rate. A score can rank well without giving correct cost thresholds | Existing destination note already records plot-axis and conditioning defects; append the newly useful decision-resolution bridge without rewriting this unrelated lesson |
| Entropy26 and MI27 teach probabilistic scores/information quantities | Show Brier/log-loss as decisions whose action is a probability report, then separate a better forecast score from this particular operational loss | No repeated information-theory chapter; no claim that lower entropy necessarily buys a useful action |
| MDPs, Bellman Equations & Dynamic Programming: exact inventory says planned, individual design required | Teach a bounded observation-then-action tree and explicitly distinguish it from repeated actions that change future state and information | Full MDP/control/learning policy evaluation remains a later owner, with a [scoped destination bridge](topic-notes/mdps-bellman-equations-dynamic-programming.md) |
| Specialist robust/sequential/game/elicitation theory | Finite ambiguity sets, finite randomization and a concrete worst-risk certificate are included; full complete-class theorems, infinite minimax theory, dynamic coherent risk, games, preference-elicitation experiments and off-policy learning are not claimed | Keep the specialist boundary visible; adapt if authoring reveals a necessary local gap rather than inflating the title or silently declaring a field complete |

Normative costs are stated choices or externally justified requirements, not estimated facts that probability alone determines. Mathematical optimality is relative to them. Use synthetic production/inspection/provisioning examples with declared loss units; avoid medical or personal-financial recommendations. Costs may include delay, resource use and mistakes, but incompatible units cannot simply be added without an explicitly chosen exchange rate. Some requirements belong as constraints rather than a price.

## 3. Continuing example and conventions

Use a manufactured item before release. State Y=0 is sound, Y=1 is faulty; observed information x is available before the action. The action set starts with **release** and **quarantine/rework**. Its rows are actions and columns are states:

| Loss units per item | Sound | Faulty |
| --- | ---: | ---: |
| Release | 0 | 80 |
| Quarantine/rework | 10 | 10 |

Quarantine is assumed to avert the release consequence and always incurs the handling/delay cost. It is not an error-free free prediction: even a correctly quarantined faulty item costs 10. Every diagram, native program and changed exercise must retain this convention until a clearly labelled new matrix is introduced.

At p=P(Y=1|x)=.08, release costs 80p=6.4 in expectation while quarantine costs 10. At p=.20 they cost 16 and 10. Both populations are mostly sound, yet the chosen action changes. The threshold is 10/80=.125. The often-memorized 10/(10+80) would be wrong here because it assumes the true-positive action has zero loss. Show expected contribution cells, not just the final argmin.

Use `r(a|x)` for a conditional expected loss; `δ(x)` for a rule fixed before observing its next input; `R(θ,δ)` for repeated-data risk at a fixed model state θ; and `B(π,δ)` for prior-averaged risk. Keep the variable being averaged and the variables held fixed visually distinct. The everyday phrase “risky outcome” and a tail-risk functional are not silently synonyms for statistical expected-loss risk.

Finite action/state/information sets are the executable core. Finite losses and properly normalized nonnegative masses guarantee the needed expectations; impossible signals have no posterior and contribute zero joint mass. In general spaces, an argmin needs attainment and a measurable selection, and integrability matters. Explain those limits beside the general notation without requiring the later analysis course.

## 4. Learning flow and mechanism map

### 1. A probability is not yet a decision

Introduce item, state, information, action and consequence in that order. A forecast reports a belief; a decision selects something to do. Use the same .08 probability with changed handling cost to expose the role of consequences. A compact branching picture shows decision first, uncertain state second, and a terminal loss, with both branches labelled by the same state law when the action does not change the underlying manufactured condition.

Separate a good decision under its declared model from a fortunate realized outcome. Expected loss is a repeated/model average, not what one item literally costs. A loss of zero today does not retrospectively prove the risky policy optimal. Add a one-step prediction before the full weighted calculation and a complete native matrix evaluator.

### 2. Derive the action regions instead of memorizing .5

Compute `r_a(p)=(1-p)L[a,0]+pL[a,1]`. For two actions, the difference is an affine function: `Δ(p)=(1-p)(L[1,0]-L[0,0])+p(L[1,1]-L[0,1])`. Choose action1 when Δ<0, action0 when Δ>0, and explicitly report the tie set when zero. The usual interior threshold needs opposite endpoint preferences. Parallel/equal risks, dominated actions and endpoints are real cases, not a division-by-zero exception to ignore.

For zero diagonal costs only, derive p≥C_FP/(C_FP+C_FN), specify the tie convention, then derive the likelihood-ratio form using prior odds when the likelihood denominators are nonzero. Treat zero-mass observations separately. Positive common scaling of all losses and state-dependent constants added equally to every action preserve choices; arbitrary rescaling of just one cost does not.

The initial investigation links the editable 2×2 cost cells to exact affine risk lines, a probability cursor and a labelled selected-action strip. The strip is the lower envelope of the lines, not an empirical success-rate graph. A changed flat/dominated preset prevents a misleading universal crossing story.

### 3. Evaluate a rule across possible observations

Move from “after observing x” to choosing a whole rule. Use hidden model state θ∈{low,high}, a binary signal X, P(X=1|low)=.2 and P(X=1|high)=.8. Actions announce low/high under zero-one loss. Enumerate all four deterministic rules. Their risk vectors are (0,1), (1,0), (.2,.2), (.8,.8); with prior P(high)=.3 their prior risks are .3,.7,.2,.8. The inverted-signal rule is dominated, while neither constant rule is uniformly best.

Derive the posterior-weighted loss after each signal and verify its selected rule by expanding the same joint-mass table two ways. Posterior high is 12/19 after X=1 and 3/31 after X=0. The sum over observation groups proves conditional minimization minimizes total risk **when decisions may be selected independently in those groups**. A restricted rule class or coupled budget can prevent that pointwise optimum.

Introduce dominance/admissibility with its quantifiers; “not dominated” is not “best under every criterion.” An optional finite theorem proves that a proper positive prior on every finite state makes a finite-risk Bayes rule admissible: any strict dominance would lower its weighted average. Do not extrapolate this without conditions to arbitrary continuous improper priors. Connect false-positive/false-negative entries to test power and significance without interpreting a p-value as a posterior probability.

### 4. A loss chooses which summary to report

Use one finite demand law Y∈{0,2,10}, probabilities (.2,.5,.3). The mean is 4; an absolute-loss optimum is median2. Under under-provision cost4 and over-provision cost1, a quantity10 minimizes the stated expected cost; its cost is6, compared with10 at quantity2. Introduce under/over residuals, not an unexplained “pinball loss.”

Complete the square for squared error using the prerequisite identity; compare left/right cost slopes or finite differences for absolute/asymmetric loss. General quantile condition is F(q−)≤τ≤F(q), τ=c_under/(c_under+c_over); discrete flat intervals can give nonunique minimizers. State first/second moment requirements and feasible-action constraints. Modes minimize zero-one label error for discrete states; a continuous density mode is not a nontrivial exact-match zero-one optimum.

A provisioned-quantity number line overlays weighted demand stems and separately shaded under/over deficits. Inspecting one candidate reveals every weighted contribution. Include the simple posterior-predictive bridge: uniform θ∈[0,1] followed by two conditionally independent failures gives P(both)=Eθ²=1/3, versus 1/4 after plugging in Eθ=.5. A release loss30 if both fail versus mitigation9 changes from integrated10 to plug-in7.5. This is a changed action consequence of a shared-parameter model, not a repeated conjugacy lesson.

### 5. Defer, inspect or act under a resource limit

First introduce a genuinely different third action. For a separate zero-diagonal classifier matrix, false-negative cost8, false-positive cost2 and fixed abstention loss.6, its risks are8p,2(1-p),.6. Release/class0 is optimal below .075, abstain between .075 and .7, class1 above .7, with explicit endpoint ties. Define abstention as this stated fallback contract, not an automatically perfect human or a free correct answer. State-dependent fallback errors change the row.

Return to the original item matrix and six item probabilities (.02,.06,.10,.20,.35,.60). Unconstrained expected-loss minimization would quarantine the last three. A capacity of at most two slots couples choices: each selected item replaces 80p by10, a saving80p−10. Sort positive savings and prove the top-two rule by exchanging a chosen smaller saving with an unchosen larger one. Total loss is50.4, versus44.4 without that capacity. Linearity of expectation suffices for this fixed additive allocation; independence of faults is not needed here.

A visible allocation board shows items, current expected-loss contributions, saved loss and actual occupied slots. It must not claim that sorting by p works for heterogeneous costs. When resource sizes differ, the small counterexample sizes(3,2,2), benefits(5,3.5,3.5), capacity4 chooses items2+3 with saving7, rather than the single largest benefit5; link the already-taught knapsack owner. Generic greedy benefit-per-size is also not claimed correct. Compare feasible policies rather than adding an unexplained finite penalty to a hard limit.

### 6. Check whether the probabilities and evaluation support the action

Distinguish true conditional probabilities, estimated probabilities, ranking scores and observed finite-bin rates. At true p=.1, reporting .2 crosses the original .125 threshold: the cost becomes10 instead of8. Both reports give the same most-likely class under .5. Binary Brier losses are .09 versus .10; a strictly increasing score transform preserves order but not a numeric threshold.

Give the explicit calibrated-but-coarse counterexample: equally likely groups with true rates .1 and .3 both receive report .2. It is calibrated as a score but is not P(Y=1|full group). With quarantine cost20 and faulty release80, shipping every item costs16 on average; using the group and quarantining only the .3 group costs14. Constant cost thresholds can be optimal among policies using the same calibrated score, without being optimal among all available-information policies. Costs varying within score bins require the corresponding joint conditioning too.

Derive the Brier identity E[(q−Y)²]=p(1−p)+(q−p)² and link log-loss/KL with support conditions. This evaluates probability reports; a better overall score does not guarantee lower loss at every operational threshold. A finite validation workflow evaluates supplied frozen scores, selects a threshold on a validation split with a fixed tie rule, then evaluates once on a separate test set; record all four confusion cells and cost contributions. Include all-accept/all-reject endpoints and repeated scores. A checked proposed fixture uses validation scores(.05,.10,.20,.20,.40,.80), labels(0,1,0,1,0,1): choosing quarantine when score≥.10 gives minimum validation total50. Freeze it before reading test scores(.03,.08,.12,.30,.60,.90), labels(0,0,1,0,1,0): total40, versus100 at threshold.5, although both have four correct class labels out of six. The tiny supplied scores are not asserted to be calibrated; these are teaching calculations, not a performance study. No classifier training or calibration API is needed for the self-contained program.

Show a two-state prior-odds change under explicitly unchanged class-conditional feature laws. A likelihood ratio4 turns prior.1 into posterior4/13, while prior.02 gives4/53. It is a modelled label-shift calculation, not a repair for arbitrary distribution shift. Finite held-out risk is an estimate, not its population value; distinguish binomial proportion intervals from intervals for a general loss mean and link sampling/measurement49 for design and uncertainty. Do not fabricate benchmark superiority.

### 7. Ask what additional information is worth

Return to original prior.08 and costs. The proposed non-destructive test has sensitivity.8 and false-positive probability.1, with no change to item state or later action availability. Joint masses in (negative,positive) order are ((.828,.016),(.092,.064)); positive mass is.156 and posterior16/39, negative posterior4/211. Quarantine positive and release negative. Fold the tree backward: minimize at each decision node, average at chance nodes. Expected loss after using the test is2.84; baseline6.4 gives gross expected sample-information value3.56. With test cost2, total4.84 remains better than6.4.

Perfect state information permits cost.8, so EVPI=5.6 and 0≤EVSI≤EVPI. Prove the lower bound by the ability to ignore a free signal and the upper bound because knowing the state can imitate any signal policy. Those comparisons hold under the same coherent probability/action/loss model. Costs, delay, harmful measurement or forced reactions can invalidate the “free information” comparison. A particular surprising result can make conditional risk larger; the guarantee is pre-observation expected optimal risk.

At prior.02, the same test has gross value.14 and is not worth cost2. At prior.01 it changes beliefs but neither selected action, so value0. Entropy reduction and decision value are different. Do not label a Bayes-risk lower-envelope graph as an entropy curve. Optional branch: garbling a signal cannot improve its optimum if the more informative observer can simulate the garbling; prove this direction in a finite joint table, without asserting the full converse theorem.

Use a topic-specific tree with observation available **before** its action nodes; toggling “must decide before test” collapses the achievable value without pretending that a learner may choose a branch after seeing an unavailable label. All displayed contributions derive from the same joint mass. Zero-probability test results display “impossible under this model,” not a fabricated posterior.

### 8. Separate a forecast from the effect of acting

Use two equally weighted equipment groups. Known synthetic failure probabilities without an intervention are .8 and .3; under a precisely specified preventive action they are .75 and .05. With avoided-failure value50 and action cost6, net benefits are −3.5 and6.5. Choosing solely the group with the highest untreated failure rate selects the wrong intervention under these stipulated effects.

Define potential outcomes Y(a) in plain terms and calculate E[L(a,Y(a))|x]. The supplied intervention probabilities are assumptions in this example; observational P(Y|A=a,x) is not automatically their value. A small side-by-side figure distinguishes acting on a manufactured latent condition from changing a future failure mechanism. Causal25 supplies identification and estimation; the local page teaches the decision once relevant distributions are justified. No clinical or real-device benefit is inferred from these invented values.

### 9. Make uncertainty about the model visible

An interval p∈[.08,.18] is an ambiguity set, not a probability distribution over p and not automatically a confidence interval. The original release worst loss14.4 exceeds quarantine10, so a deterministic minimum-worst-loss criterion quarantines. Report what supports the interval and examine its endpoints because these risks are affine. A coherent posterior over p instead averages the appropriate loss; a single future binary linear loss depends on E[p], whereas the earlier two-event loss depends on E[p²].

Define worst-case regret as loss minus the best action that could be chosen if that model state were known. Distinguish it from the previous criterion with a separate exact table A=(0,10),B=(6,7): worst loss chooses B (7<10), but worst regret chooses A (3<6). Neither objective determines values for the learner. Compare stable action regions, near ties and plausible cost changes before giving a recommendation conditional on assumptions.

### 10. Understand when randomization can help

For a fixed posterior, mixing actions gives a convex average of conditional losses and cannot improve on their minimum; it can reproduce ties. Under worst-model-state evaluation, randomization can hedge. Use loss vectors A=(0,6),B=(4,0),C=(2.5,2.5). Deterministic minimax chooses C with2.5. Choose A with probability.4 and B with.6: risk vector(2.4,2.4). A prior weighting the second state by.4 gives both A/B average2.4 and C2.5, a lower-bound certificate for every mixture, so the achieved2.4 is globally minimax for this finite game.

The risk-plane picture has axes “expected loss if state0” and “expected loss if state1,” not probability. A mixing control travels on the exact A–B segment; expanding a square from the origin makes the maximum-coordinate objective visible. State that Nature chooses a fixed state before the independent random action; an adversary that sees the realized draw changes the game. Optional finite LP formulation names the simplex and state constraints, but actual code enumerates/solves this small case independently and does not rely on an unexplained optimization package.

### 11. Decide which notion of consequence matters

Expected monetary/resource loss is one criterion. If a declared increasing utility u maps a final resource outcome to preference value, maximize E[u(W)], not u(EW). Use equally likely final levels50/150, mean100, u=√W, certainty equivalent approximately93.301270. A sure95 is preferred under that utility despite its smaller mean. The chord picture and the defining concavity inequality explain the result, with domain W≥0 and strictly increasing invertibility for the certainty equivalent. Costs are represented consistently in final outcomes; they are not subtracted after applying an unrelated utility scale.

Positive affine transforms of utility preserve expected-utility choices; an arbitrary increasing transform preserves ordering of sure outcomes but need not preserve lottery preferences. Present utility as a specified preference model, not a measured universal law or complete behavioral theory. Local concavity arguments suffice; full axiomatization/elicitation and prospect-theory experiments are beyond the finish line.

For loss Z∈{0,10,100} with masses(.8,.15,.05), mean6.5 prefers this policy to a sure loss8. Yet a requirement P(Z>20)≤.01 rejects it. At α=.9, VaR is10 and upper-tail CVaR is55: the worst .1 probability includes .05 mass at100 and .05 of the atom at10. The naïve conditional means E[Z|Z≥10]=32.5 and E[Z|Z>10]=100 are neither the answer.

The optional deeper derivation writes CVaRα=min_t {t+E[(Z−t)+]/(1−α)} and proves the finite piecewise-linear minimum via left/right slopes and the mass allocation. Label α as quantile level, not “probability the estimate is correct”; α<1, finite first moment, and the atom convention are explicit. A tail-mass strip lets learners allocate exactly1−α from the worst outcomes and compare mean, threshold and tail average. For arbitrary action-dependent probabilities, no automatic convexity in the action is claimed. Tail objectives and feasibility constraints need not select the same policy.

### 12. Combine the mechanism in a changed allocation-and-test capstone

Use the six-item, two-slot cohort from `5 and offer a **perfect test of at most one item**, costing1, before choosing the two slots. For this test-only extension assume independent Bernoulli item states; otherwise seeing one item could update other probabilities, and the specified model would be incomplete. The best test is item4 (p=.20), not item6 (p=.60).

Without a test the cost is50.4, quarantine items5/6. If item4 tests sound, cost34.4 and still quarantine5/6. If faulty, cost62.4 and quarantine4/6. Pre-test loss is40, plus1=41. Gross information value is10.4; the bad result may be more costly conditionally even though the ex-ante choice helps. The design checker independently enumerates all64 worlds for each candidate test and all feasible small allocation subsets. The actual program must print each candidate's two branch policies/costs, weighted result and selected policy, not only the winning index.

The learner must first solve a changed version: different capacity or test price, an imperfect test, or heterogeneous losses, with explicit update and acceptance cases. One checked changed task raises test price from1 to12: the best tested policy costs52, so choose the no-test50.4 policy. At price10.4 it ties the best tested policy. Do not imply that adding independently computed per-item VOIs solves an arbitrary coupled inspection budget. Explain information timing, fixed precommitted policy and the opportunity to ignore information.

### 13. Independent practice and a usable decision record

Build a worksheet-like final record: target/population/time; available information; feasible actions; state/outcome model and provenance; loss or preference assumptions; criterion; intermediate calculations; selected action and ties; sensitivity; validation/uncertainty; what would invalidate the conclusion. It is the capstone output, not a compliance checklist pasted onto every paragraph.

Readiness means being able to change a loss, information set or constraint and explain the changed reasoning. A finite list of scenarios, a clicked completion marker or the title “Bayes optimal” does not certify practical mastery across all decisions.

## 5. Visual contracts and ordinary reading plan

Preserve the site's calm dark/gold system and existing Space Grotesk/JetBrains families. The frontend-design skill is applied with creator-provided context from `.impeccable.md`; it does not authorize a rebrand. Use conventional diagrams where helpful, not uniform paragraph cards. Shared controls can remain familiar while the geometry changes.

| Representation / placement | Concrete encoding, interaction and purpose | Accessibility/limits and independent check |
| --- | --- | --- |
| Inline release/hold decision tree before `1's formal sum | Decision node precedes chance node; terminal cells show loss and path mass. Identical state probabilities are intentional for an already-manufactured state | On phones stack the two action branches with aligned state/loss columns. Explicit textual path products; no false claim that edge length is probability |
| Editable cost table + lower-envelope graph, `2 | Risk units vertical, fault probability horizontal; endpoint losses determine each line. Show tie/always-one-action states | Exact affine intersections and independent matrix sums; preserve min lines and domain labels at320. No smoothed arbitrary curves |
| Risk “averaging scope” inline diagram and finite signal table, `3 | Highlight one observation column for posterior risk, one state row for procedure risk, whole joint table for prior average | Table headers carry the conditioning variable. Null observation columns identified; direct joint enumeration verifies both averages |
| Demand stems and under/over allocation, `4 | Weighted demand locations and asymmetric distances reveal why different loss functions select different summaries | Slider changes action, cost ratio changes slopes, numeric contribution table remains visible. Median/quantile flat ties are not falsely unique |
| Action/abstention region strip and capacity allocation board, `5 | Three labelled risk regions; separately, selectable actual item slots and loss savings display the coupling | Distinct figures because fallback risk and shared capacity are different operations. Keyboard selection/reset, capacity0/full, negative saving and ties. Exhaustive subset oracle |
| Probability-report/decision comparison, `6 | Same item groups, score values, operational action and expected losses; a compact true-rate/report comparison exposes calibration coarsening | The toy probabilities are modelled, not observed calibration data. Do not render a smooth reliability curve from two invented points |
| Observation-then-decision folding tree, `7 | Show joint mass, posterior, contingent action and branch contribution; reveal or remove the test before the action | Preserve chronology when stacked. Test cost separate. Impossible signals, uninformative/perfect signals and reversed test labels checked from joint distributions |
| Prediction versus intervention paired paths, `8 | Local observed state and hypothetical changed outcome law are different diagrams/labels | The known synthetic effect table is stipulated; do not draw causal arrows as though fitted correlations establish them |
| Finite risk plane and mixing segment, `10 | Axes are state-specific risk, convex segment is randomized mixture, expanding square shows worst risk | Keyboard mix amount and exact values; independent lower-bound certificate. Do not confuse randomization with observing Nature's state |
| Utility chord and tail-mass strip, `11 | Utility curve uses actual sqrt values/chord; separate loss strip allocates worst probability mass including part of an atom | Utility outcome domain and CVaRα<1 explicit. Native integration/finite enumeration/hinge oracle, labels and mass sum checked |
| Capstone contingent allocation tree, `12 | Selected tested item branches to two different feasible slot assignments with aggregate losses | This can reuse the earlier allocation/tree primitives without hiding updated probabilities. Exact64-world oracle, unchanged input/reset, altered capacity/test cost |

Core new structures must be visible in ordinary reading before relying on a control. A static figure can replace a proposed interaction if changing it adds no learning value; conversely add another focused representation if a meaningful hidden step appears during authoring. No final lab count is prescribed.

## 6. Complete native programs and independent practice plan

Use Python3 standard library for the finite core (`fractions`, `itertools`, `math`, seeded `random` only if a simulation adds a distinct lesson). Include save/run instructions before the first code. Every program is complete and independent with declared inputs, intermediate printout, exact/rounded expected stdout and interpretation. Render its stored question explicitly before its code. An optional SciPy LP oracle may be a verification dependency, not a hidden learner requirement; no unstable external classifier API is needed to understand the core.

Proposed programs cover: general finite losses/ties; all observation policies and the two risk averages; mean/median/asymmetric provisioning; abstention plus constrained allocation; frozen validation/test thresholds; shared-parameter batch consequences; test-tree backward induction; causal benefits from a stipulated potential-outcome table; finite robust/randomized criteria; utility and exact atom-aware tails; full contingent capacity capstone. Combine or split only when it improves a coherent runnable workflow. Do not invent outputs before execution or force one program per heading.

Independent practice candidates, each with an optional hint followed by a complete explained solution:

| Changed task | Hint mechanism | Accepted reasoning/result to implement and verify |
| --- | --- | --- |
| Change quarantine cost to12 and release-failure loss to60; inspect p=.15/.25 | Compare full rows, not only error cells | Threshold.2; expected release9/15 versus12; identify exact tie at.2 |
| Repair a claimed C_FP/(C_FP+C_FN) threshold for the original matrix | Read the true-positive cell | It costs10 too; threshold.125, not1/9 |
| Add a state-independent fallback.8 to FN8/FP2 | Compare three affine losses | Lower boundary.1 and upper.6; explain ties and when the middle region can disappear |
| Change prior to.1 in the noisy-signal rule example | Posterior then same joint average | Follow-signal Bayes risk.2 exceeds always-low.1; compute posteriors to check if either observation changes the action |
| Demand support(1,3,9) with masses(.25,.5,.25), under/over ratio3:1 | Accumulate mass toτ=.75 | Quantile minimizers can occupy[3,9]; do not assert a unique q merely because a lower quantile routine returns3 |
| Distinguish calibration from all-information optimality | Split the constant-score population | Reproduce16 versus14 under the stated group/cost fixture; explain why this does not contradict calibration |
| Re-evaluate the original test at prior.02 and cost2 | Compute joint branch contributions first | Gross value.14, so do not buy. A different result can still have a different posterior |
| Find an informative test worth zero to the current action | Make every posterior lie in one action region | Prior.01 in the fixture: belief changes, selected action remains release, EVSI0 |
| Separate model averaging from minimax and minimax regret | Keep criterion order visible | Explain original interval's worst-loss comparison and the A/B regret counterexample; uncertainty set is not a prior |
| Certify the randomized value instead of displaying a grid minimum | Match a mixture upper bound with a state-weighted lower bound | Mix.4/.6, value2.4; explain why Nature observing the draw changes the guarantee |
| Compute CVaR at a changedα cutting a different part of an atom | Allocate exactly1−α mass from the worst tail | Show included fractions and independently verify the hinge representation; contrast conditional means |
| Explain why highest untreated risk need not be highest benefit | Compare Y(0) with Y(1), not p alone | The stipulated groups yield net−3.5/+6.5; identify what real evidence would be needed |
| Change the capstone's price or capacity | Recompute contingent feasible policies | Print all candidates plus no-test baseline and tie rule; specify conditional independence only where needed |
| Diagnose a flawed decision report | Identify unavailable information, invented costs, post-selection test tuning or missing constraint | Give a repaired report with changed calculations, assumptions and explicit unresolved evidence, rather than a generic warning |

These are design candidates, not a fixed problem quota. Open-ended work needs one fully executed acceptable changed answer and a checkable acceptance rule. Add in-section retrieval prompts so the final worksheet does not carry all the assessment.

## 7. Research and learner-resource ledger

Sources were browsed on11September2026. Original derivations and fixtures are the teaching backbone; source annotations below record inspected scope rather than claiming an entire book/course/paper was reviewed.

| Claim or learner purpose | Primary/authoritative source and actual inspected scope | Use and qualification |
| --- | --- | --- |
| Action/information/state/loss and conditional optimality | [Cosma Shalizi, Predictions and Decision Theory,2021](https://www.stat.cmu.edu/~cshalizi/sml/21/lectures/02/lecture-02.html): elements, risk, conditional-minimization proof and alternative losses | Concise written alternative after the first worked calculation. Its fixed-joint-law restriction matters for the causal bridge; no wholesale structure is copied |
| Fixed-parameter procedure risk, Bayes averaging, dominance and minimax | [Will Fithian, Statistical decision theory,2023](https://www.stat.berkeley.edu/~wfithian/courses/stat210a/models.html): definitions and comparison sections | Advanced notation reference. Use checked definitions, not unrelated shorthand claims: its displayed late minimax fraction appears to have a denominator typo, and no blanket UMVU-for-every-convex-loss claim is imported |
| Probability prediction versus thresholded action; held-out threshold selection | [scikit-learn threshold guide](https://scikit-learn.org/stable/modules/classification_threshold.html), ``3.3–3.3.1.4, current page1.9.1 | Practical alternate workflow. Default balanced accuracy is not the lesson's cost function; class labels, data split and objective must be explicit. Actual future library use would require installed-version execution |
| Coarse calibration, proper scores and resolution | [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html), opening note/`1.16.1 and train/calibrator separation | Supports probability-versus-score caveats and finite-bin interpretation. Not a promise that fitting a calibrator yields true conditional probabilities or no shift |
| Proper probability reports and asymmetric quantile decisions | [Gneiting & Raftery2007](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), `3 examples1/3 and `6 quantile scoring | Inspect only these mechanisms. Paper uses positively oriented scores; this lesson explicitly uses losses to minimize. The finite quantile-tie proof is supplied locally |
| Tree order and expected consequences; video alternative | [MIT IDS.333 Unit8 Video3: Constructing Trees](https://ocw.mit.edu/courses/ids-333-risk-and-decision-analysis-fall-2021/resources/unit-8-decision-analysis-video-3/) and its [two-page transcript](https://ocw.mit.edu/courses/ids-333-risk-and-decision-analysis-fall-2021/1-2L0LSIpoNnJ9fZabcMEhn2na6aPrj8o_transcript.pdf), full transcript read | Resource metadata and transcript inspected, not full video playback. It explicitly uses invented example values and distinguishes actions that do/do not alter outcomes; our item fixture and numbers are original |
| Information before a decision; alternate short video | [MIT IDS.333 Unit9 Video2: Value of Test](https://ocw.mit.edu/courses/ids-333-risk-and-decision-analysis-fall-2021/resources/unit-9-value-of-info-video-2/) | Official description/title/embed availability inspected, not full playback/transcript. Publish an honest purpose annotation, no invented timestamp; our finite EVSI inequalities and computations are independently derived |
| Expected utility and certainty equivalent | [MIT14.13 Risk Preferences slides](https://ocw.mit.edu/courses/14-13-psychology-and-economics-spring-2020/30de72ba97b788add155104f2ee49266_MIT14_13S20_lec7_8.pdf), slides9–13 and18–20, plus [Lecture7 resource](https://ocw.mit.edu/courses/14-13-psychology-and-economics-spring-2020/resources/lec7-risk-pref-i/) | Written concavity/expected-utility/CE sections inspected and video metadata verified; full78-slide set/video not reviewed. The declared utility example is not a recommendation about the learner's money |
| Atom-aware upper-tail loss and finite hinge minimum | [Rockafellar & Uryasev, General Loss Distributions](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf), version28Nov2001, `2 definition, Proposition8 discrete formula, Theorem10 and adjoining piecewise-linear discussion | This is the primary basis for discrete-tail conventions; inspect the atom split and finite minimizer, not unrelated portfolio benchmarks. The earlier [continuous-assumption paper](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf), `2, was checked to avoid incorrectly transferring a conditional-mean shorthand to atoms |

Some initial UF/StonyBrook PDF and publisher requests failed; the author-hosted Washington general-distribution PDF was successfully retrieved instead. Do not label failed content as read. The initial MIT `live` URLs failed but canonical `ocw.mit.edu` transcript/slides succeeded. No empirical performance ranking, universal calibration trend or externally supplied loss valuation is inferred from this research.

## 8. Provisional arithmetic and implementation verification plan

[The design-only check record](evidence/decision-theory-design-checks.json) is generated by `scripts/check-decision-theory-design.py`. It checks original/changed inspection priors,60 finite information models including impossible signals, all four signal rules, provisioning costs, a finite minimax certificate, utility arithmetic,6 atom-tail levels against an independent hinge minimum, and every one of64 worlds for each of6 capstone test choices. The exact64-world comparison is independent of the backward conditional policy calculation. These are proposed-fixture checks, **not** final lesson/model/native/browser verification.

During authoring:

1. Build descriptive topic-owned finite models with dense-array, size, normalization, finite-loss and α-domain validation. Do not confuse a floating display tolerance with an exact mathematical tie. Permit negative losses when the contract explicitly interprets gains, or reject them transparently in a bounded control; no hidden clamp.
2. Generate and execute complete Python programs, preserve their stdout and verify changed helper calls. Use exact Fraction/joint enumeration, brute-force feasible policies, independent LP certificate checks, high-precision utility calculations and separate mass-allocation/hinge CVaR oracles. Validate information inequalities only under their stated assumptions, not by clipping negative numerical values.
3. Test risk ordering, losses at all-simplex vertices, ties/dominance, null signals,0/1priors, signal relabelling, perfect/useless tests, fallback always/never preferred, zero/full capacity, heterogeneous costs, repeated scores, all-accept/reject thresholds, atom cutoffs and invalid inputs. Verify the input-data timing of any simulation and all changed practice arithmetic.
4. Read the actual page in normal order; then exercise every distinct investigation at1440/390/320 with actual site fonts, keyboard focus, reset, text equivalents, optional hints and section-anchor arrival. Open the final screenshots. Check endpoint markers, labels, narrow mathematical lines, probabilities summing toone, and decision arrows that cannot imply hidden information.
5. Fingerprint every owned production file, not an assumed six-file list. Separate original-plan/design checks, author verification, independent mathematical/source review, integrated production evidence and user acceptance. Root owns the latter shared registration/build/route steps.

Semantic implementation names: `decision-theory-risk-cost-sensitive-decisions.jsx`, `decision-theory-models.js`, `decision-theory-examples.js`, `DecisionTheoryLabs.jsx`, `DecisionTheoryFigures.jsx` if needed, and `decision-theory-labs.css`. The individual stable-ID blueprint links this design. Math is imported directly only by the consuming lesson/components; no aggregate all-topic runtime, global CSS or eager publication bundle.

The approved design required **no change to the prerequisite list or title**. Its implementation is recorded below. A design, publication or author pass is not independent approval. New ownership discoveries must continue to reach their actual future author.

## 9. Implemented disposition and author corrections

The complete body has 13 sections, eight investigations containing distinct mechanism-specific views, two inline figures, 11 fully executed standalone Python programs and 14 practice/checkpoint groups with separate hints and explained solutions. The finite core and optional derivations follow the proposed flow. Program questions are actually rendered before their code, including the optional shared-parameter program; setup instructions appear before the first example. The loss-summary section includes the complete flat quantile interval argument. The changed capstone prints all six test costs for capacities 1 and 3, both selected branch allocations and the no-test alternative. Existing planned coverage was retained; there was no old body or native program to remove.

The visual forms are purposeful: actual loss endpoints determine two risk lines; fixed-state tiles distinguish the generating experiment from the prior average; demand lengths and an order marker expose shortage/surplus contributions; fallback action regions are separate from coupled capacity slots; test branches show joint mass and the information timing; randomized rules occupy a state-risk plane; a utility chord differs from a probability-mass tail strip; and the final branch allocations recompute their feasible selections. The intervention comparison remains a static paired table/path because its critical distinction is the stipulated causal contrast, not a fake estimator. Mobile branches stack while keeping the underlying probabilities and losses intact. Diagram labels were enlarged after opening phone captures; formulas were explicitly wrapped rather than shrunk.

Author checks and the root's concurrent independent read corrected the following details before freeze:

1. The shared `H2` generates its own text slug and ignores an `id` prop. Root identified the mismatch in the first draft. The topic's 13 intro links now use the actual slugs, and actual keyboard arrival is verified at all three widths. No shared heading component was changed.
2. The first design chose item 4 by a lower-index tie-break without highlighting that item 5 has the same exact test value at capacity 2. Both are now explicitly co-optimal at total 41 for price 1; tests 5 and 6 also tie at capacity 1. Price 10.4 gives a three-way tie including no-test. The body, native output, changed practice and browser candidate list show the ties.
3. Floating sums could hide the rule tie at prior .2/.8, the fallback tie at p=.7, or move an exact cumulative-mass cutoff such as .1+.7 across alpha=.8. Bounded finite decision comparisons now use exact fractions of the written decimal inputs; there is no tolerance-created action tie. Posterior/test joint arithmetic, capacity policies and tail atom allocation use that convention, with explicit normalization of admitted mass-roundoff and rejection if a nonzero exact numerical result cannot be displayed representably. Actual Python uses `Fraction`. The final independent finite oracles check action sets, atom boundaries and conditional policies as well as numerical loss values.
4. Four display equations exceeded the 320px content width. The final forms split conditional-risk averaging, quantile conditions and the CVaR objective into short lines without dropping terms. The Bayes identity uses the previously defined procedure risk in its first sum; the CVaR hinge objective is named locally as Phi-alpha. The original numerical definitions remain unchanged.

5. Root identified that general CVaR's hinge infimum need not be attained at alpha=0 for an integrable loss unbounded below. The displayed general minimum is now restricted to 0 < alpha < 1, while alpha=0 is explicitly the mean extension; the finite-support examples attain their endpoint at or below their smallest loss. The final narrow-screen text and formula were opened and verified.
6. Dense model arrays now require their own supplied entries via `Object.hasOwn`; an inherited prototype value cannot fill a missing element. A separate positive-support certainty-equivalent underflow guard rejects a nonrepresentable result rather than displaying a misleading zero. Actual malformed-input regressions and unchanged ordinary program output passed after these repairs.
7. The fallback action strip explicitly samples 101 probabilities at intervals of .01. Thin ties and boundaries such as .075 can fall between samples. The caption now distinguishes this sampled picture from the directly calculated selected-point risks. No sampled pixel location is offered as an exact continuous boundary.

The finite arithmetic/script and browser records are linked in the verification. The first browser harness also assumed a `pre` renderer; the shared code component actually renders plain text nodes in its two direct block children. The harness was corrected to compare those exact code/output text nodes. That was a test-selector mismatch, not missing lesson code. No untested printed output or hidden dependency was substituted.

The [MDP timing note](topic-notes/mdps-bellman-equations-dynamic-programming.md) and the appended [calibration-resolution note](topic-notes/calibration-conformal-prediction.md) remain routed to their destination authors. This lesson provides the complete local bridges but does not rewrite those owners, claim the MDP lesson exists, or establish population calibration/causal effects from the synthetic data. The first-pass route stays beginner-accessible; the specialist boundaries in section 2 remain deliberate limits, not claims of universal mastery.
