# Bayesian Networks & Causal Graphical Models

Two people call to report that your house alarm is sounding. Is there a burglary? Perhaps. But an earthquake can also trigger the alarm, and a caller sometimes reports a sound incorrectly. You need to combine uncertain clues without treating every clue as an independent vote.

A **Bayesian network** describes a joint probability distribution using small local probability models connected by arrows. Its first job is bookkeeping: state which variables each local model depends on, then combine those models consistently. Once the joint distribution is defined, you can ask many questions of the same model—including questions with missing observations.

There is a second, stronger use for a graph. If its arrows describe how a system is generated, with suitable causal assumptions, it can help predict what happens when a mechanism is changed. Observing an alarm and deliberately switching it on are different events. The first may tell you about a burglary; the second need not.

**Your first route:** read sections 1–6 and do practice 1–6. You should be able to build a small network, calculate a posterior, inspect a path and distinguish observation from intervention. Section 7 develops causal identification and counterfactual reasoning; section 8 explains larger models and inference choices. Return to them after the small calculations feel natural.

You need multiplication, weighted averages and conditional probability. Here \(P(B=1\mid J=1)\) means “among outcomes in which John called, what fraction have a burglary?” A variable is a quantity that can take different values; uppercase letters name variables, lowercase letters their selected values. All alarm and service probabilities below are **constructed teaching models**, not measured crime or engineering statistics. The Wine experiment uses real, attributed observations.

## 1. Build one possible world from local choices

Name five binary variables:

| Variable | State 1 means | Parents |
| --- | --- | --- |
| \(B\) | Burglary | none |
| \(E\) | Earthquake | none |
| \(A\) | Alarm sounds | \(B,E\) |
| \(J\) | John calls | \(A\) |
| \(M\) | Mary calls | \(A\) |

Draw \(B\to A\leftarrow E\), with \(A\to J\) and \(A\to M\). The drawing contains no directed cycle: following arrows can never bring you back to the same node. Such a graph is a **directed acyclic graph**, or DAG. A parent points directly into a node; an ancestor can reach it through one or more arrows.

**Visual — assemble a world.** Place the graph beside a five-slot probability strip. Selecting \(B=1,E=0,A=1,J=1,M=1\) highlights one entry from each local table. Multiply the entries, rather than adding five confidence scores.

Our root probabilities are \(P(B=1)=0.001\) and \(P(E=1)=0.002\). The alarm table is:

| \(B\) | \(E\) | \(P(A=1\mid B,E)\) | \(P(A=0\mid B,E)\) |
| --- | --- | --- | --- |
| 0 | 0 | 0.001 | 0.999 |
| 0 | 1 | 0.29 | 0.71 |
| 1 | 0 | 0.94 | 0.06 |
| 1 | 1 | 0.95 | 0.05 |

John calls with probability 0.90 if the alarm sounds and 0.05 otherwise. Mary's corresponding probabilities are 0.70 and 0.01. Each pair sums to one. This is a **conditional probability table** (CPT): for each parent setting, it gives a complete distribution for the child.

The model's factorization is

\[
P(b,e,a,j,m)=P(b)P(e)P(a\mid b,e)P(j\mid a)P(m\mid a).
\]

For the selected world the probability is

\[
0.001(0.998)(0.94)(0.90)(0.70)=0.0005910156.
\]

That is one world, not yet the probability of burglary given calls. Other worlds can produce the same calls.

Why does this product define a normalized distribution? Imagine generating variables in parent-before-child order: choose \(B,E\), then \(A\), then \(J,M\). Each choice distributes the probability mass arriving at a node among its states. Equivalently, sum the joint over \(J\) and \(M\); their conditional distributions each contribute one. Sum over \(A\), then the roots, and obtain one. Cyclic systems can also have probability models, but this DAG construction is what makes normalization automatic here.

The graph makes specific independence assumptions. John and Mary may be associated before you know whether the alarm sounded, because both respond to it. Once you know \(A\), this model treats their remaining reporting randomness as independent. If they telephone one another, that assumption may fail; a beautifully drawn graph cannot repair a missing dependency.

More generally, the **local Markov property** says that a node is independent of its nondescendants, other than its parents, once its parents are given. A descendant is reachable by following arrows away from the node. This explains which earlier variables a local CPT can leave out of the full probability chain rule; section 4 turns the property into a test for more complicated queries.

For five unconstrained binary variables a joint table has 32 entries and 31 free numbers: one entry is fixed by the sum-to-one constraint. This network stores 20 CPT entries but only **10 free numbers**: one for each root, four for the alarm, two for each caller. The reduction comes from assumptions, not compression without consequences. In general node \(i\), with \(r_i\) states and \(q_i\) parent configurations, contributes \(q_i(r_i-1)\) free CPT parameters.

The term “Bayesian” does not require every fitted network to use Bayesian parameter estimation. The graph can be fitted by maximum likelihood; putting a prior distribution over its parameters is an additional modeling choice.

## 2. Infer a cause by adding compatible worlds

With both calls observed, the desired probability is

\[
P(B=1\mid J=1,M=1)
=\frac{\sum_{e,a}P(1,e,a,1,1)}
{\sum_{b,e,a}P(b,e,a,1,1)}.
\]

There are two operations to recognize. **Marginalization** sums over unobserved alternatives. **Conditioning** keeps outcomes compatible with evidence and renormalizes their remaining mass. Neither operation means selecting the single most likely hidden explanation.

The numerator is \(0.00059224259\). The denominator is \(0.002084100239\), so the posterior is about **0.284172**, or 28.42%. Two calls are substantial evidence compared with the 0.1% prior, but the model still considers no-burglary outcomes collectively more likely.

The following results come from enumerating all 32 worlds:

| Evidence | Burglary posterior |
| --- | ---: |
| None | 0.001000 |
| John calls | 0.016284 |
| John and Mary call | 0.284172 |
| Both call; no earthquake | 0.344200 |
| Alarm definitely sounds | 0.373551 |
| Alarm sounds; earthquake occurs | 0.003268 |
| Alarm sounds; John calls | 0.373551 |

The last three rows explain two different ideas. Discovering an earthquake offers an alternative explanation for the alarm, sharply reducing the burglary posterior in this particular model. This is **explaining away**. In contrast, once the alarm state is already known, John's report adds nothing about burglary: the intervening alarm state screens off that information. “More evidence always increases the posterior” is not a rule.

**Investigation — which clue changes the explanation?** Reveal an earthquake, first with no calls and then with both calls, and compare the burglary posterior with the compatible-world masses. Change John's false-call probability from 0.05 to 0.20; inspect the posterior change directly. Then fix \(A=1\) and alter that false-call entry again. It should have no effect because the selected evidence never uses the \(A=0\) row.

Also try making both John probabilities equal. His call then carries no information about the alarm, so observing only \(J\) returns the burglary prior. If you set John's call probabilities to zero and then assert that he called, the evidence has probability zero: a posterior is undefined, not zero and not a uniform fallback.

### A short executable calculation

Save this as a Python file. It uses only the standard library and prints the two-call posterior; every loop corresponds to a compatible world.

~~~python
from itertools import product

alarm = {(0, 0): .001, (0, 1): .29, (1, 0): .94, (1, 1): .95}
calls_j = [.05, .90]
calls_m = [.01, .70]
mass = [0., 0.]
for b, e, a in product((0, 1), repeat=3):
    p_b = .001 if b else .999
    p_e = .002 if e else .998
    p_a = alarm[b, e] if a else 1 - alarm[b, e]
    mass[b] += p_b * p_e * p_a * calls_j[a] * calls_m[a]
print(mass[1] / sum(mass))
~~~

This is useful as a reference for a tiny network. Repeatedly enumerating every world becomes expensive as the number of variables grows.

## 3. Reuse arithmetic with variable elimination

A **factor** is a table indexed by some variables. It need not itself sum to one. We can multiply factors that share variables, then sum a variable out of their product. The resulting smaller table summarizes exactly what the eliminated variable contributed.

With \(J=M=1\), first combine the earthquake prior and alarm CPT:

\[
g(b,a)=\sum_eP(e)P(a\mid b,e).
\]

| \(B\) | \(A\) | \(g(B,A)\) |
| --- | --- | ---: |
| 0 | 0 | 0.998422 |
| 0 | 1 | 0.001578 |
| 1 | 0 | 0.059980 |
| 1 | 1 | 0.940020 |

Next calculate \(h(b)=P(b)\sum_a g(b,a)P(J=1\mid a)P(M=1\mid a)\). For \(b=1\):

\[
h(1)=0.001[(0.05998)(0.05)(0.01)+(0.94002)(0.90)(0.70)]
=0.00059224259.
\]

For \(b=0\), \(h(0)=0.001491857649\). Finally divide \(h(1)\) by \(h(0)+h(1)\). This is the same answer as enumeration, with intermediate results reused.

**Visual — a factor workbench.** Each factor tile lists its variable axes. Multiplying tiles joins their axes; summing \(E\) collapses its two slices into \(g(B,A)\). Keep the four actual cells visible beside the graph. A step button is useful here to inspect arithmetic, but the earlier evidence investigation is where you test an explanatory hypothesis.

For a general elimination step:

1. Collect every current factor containing the variable.
2. Multiply those factors, aligning shared states.
3. Sum over that variable.
4. Put the resulting factor back alongside untouched factors.

Do not sum a variable out of one factor while leaving another occurrence elsewhere: that would break the dependency they share. Evidence can simplify factors before elimination, and ancestors irrelevant to a query can often be pruned.

The order changes cost, even though correct arithmetic gives the same marginal. Eliminating a highly connected node may create a large factor coupling its neighbors. For an order with induced width \(w\), the largest ordinary dense intermediate over binary variables can have \(2^{w+1}\) entries; with equal \(r\)-state variables replace 2 by \(r\). Width is computed on the undirected interaction graph induced by factors, not simply by counting arrows. For a BN, first connect every pair of parents of the same child and remove arrow directions: this is its **moral graph**, in which every CPT's variables form a connected clique.

**Visual — four-node fill.** In an undirected factor chain \(A-B-C-D\), eliminating endpoints keeps largest factors to two variables. Eliminating \(B\) first temporarily joins \(A,B,C\), creating an \(A-C\) factor. Show 4 versus 8 binary entries. The best achievable induced width is the graph's treewidth. A DAG with a single child of many parents can have a large family factor despite looking shallow.

At width 30, a 31-variable binary float64 factor alone takes \(2^{31}\times8=16\) GiB. Extra copies and other factors add memory. This is an exact storage calculation, not a measured library threshold or a claim that every network of that width is infeasible.

## 4. Read independence from paths

The arrows serve two jobs: they locate CPTs and encode conditional-independence guarantees. **D-separation** is a graphical test for the second job. A path can follow edges in either direction; it is not restricted to directed ancestry.

Inspect the middle node of each three-node segment:

| Pattern | Middle node unobserved | Middle node observed |
| --- | --- | --- |
| Chain \(X\to Z\to Y\), or reversed chain | can carry dependence | blocks this path |
| Fork \(X\leftarrow Z\to Y\) | can carry dependence | blocks this path |
| Collider \(X\to Z\leftarrow Y\) | blocks, unless a descendant of \(Z\) is observed | opens this part of the path |

A path is active when every noncollider along it is unobserved and every collider is either observed or has an observed descendant. Two disjoint variable sets are d-separated by an observation set when **all** connecting paths are blocked. One blocked route does not cancel another active one. For DAG-factorizing distributions, d-separation guarantees conditional independence wherever the conditioning event has support. An active path says the graph does not guarantee independence; numerical parameters can still cancel dependence. **Faithfulness** is the additional assumption that such extra independences do not occur. [Stanford's directed-model notes](https://ermongroup.github.io/cs228-notes/representation/directed/) develop this distinction.

In the alarm graph, \(B\to A\leftarrow E\) is blocked with no observations. Observe \(A\), and it opens. Observe only \(J\), a descendant of \(A\), and it also opens. A checker that tests only whether the collider itself is observed misses this case.

**Investigation — open every route, or block them all.** Choose the query endpoints, mark observed nodes, and predict “graph guarantees independence” or “graph leaves dependence possible.” The graph highlights each path and the first reason it is blocked. Add a second route \(B\to K\to E\) in the small editable graph; now observing \(K\) can block that route while observing \(J\) keeps the collider route open. Remove the second route and compare. Arrowhead direction, observed state and descendant status have separate labels, so color is not the only explanation.

Keep topology separate from numerical strength. Setting every alarm probability equal makes the alarm independent of its stated parents in that distribution, even though the graph still has an active conditioned collider path. The independence comes from the chosen parameters, not from a new d-separation.

For a node \(Y\), its parents, children and its children's other parents form a **Markov blanket**: once those variables are known, the graph guarantees that other nodes add no information about \(Y\). For \(B\), the blanket is \(\{A,E\}\). The earthquake matters in that blanket because learning the common effect can associate its possible causes. This connects to feature selection: a sufficient blanket can make other inputs redundant for a specified distribution. Marginal mutual information or pairwise correlations generally cannot identify that set by themselves, and degenerate distributions may allow smaller blankets.

## 5. Learn the tables, then investigate a real specimen

If a node's parent setting occurs in 10 complete training rows and the child is 1 in three, its maximum-likelihood estimate for that entry is \(3/10\). The same calculation is repeated for each parent setting. The log joint likelihood separates into sums of these local count terms, so fixed-graph, fully observed discrete parameter learning reduces to estimating multinomial tables. Unseen parent settings have no empirical distribution to estimate.

One response is a Dirichlet prior. For child-state counts \(N_k\) and positive prior pseudo-counts \(\alpha_k\), the posterior predictive probability for the next case is

\[
P(X_{\rm new}=k\mid\text{parent setting},D)
=\frac{N_k+\alpha_k}{N+\sum_j\alpha_j}.
\]

For counts \([9,1]\) with \(\alpha=[1,1]\), the probability of state 1 is \(2/12\), rather than \(1/10\). An empty binary row gets \([1/2,1/2]\) with this symmetric prior. These are posterior means/predictive probabilities, not generally MAP estimates: when an interior mode exists, the latter subtracts one from each posterior Dirichlet parameter before normalization. A prior expresses assumptions and can reduce fragile zeros; it does not guarantee accurate probabilities. [pgmpy's parameter-learning guide](https://pgmpy.org/examples/Parameter_Learning_Discrete_BN.html) distinguishes complete-data estimation, prior-based estimation and latent-variable EM.

### Does modeling extra dependencies improve cultivar probabilities?

The [UCI Wine dataset](https://doi.org/10.24432/C5PC7J) contains 178 specimens from three cultivars, with chemical measurements. We use four: alcohol, malic acid, flavanoids and color intensity. This is cultivar recognition, not wine quality prediction. The [offline CSV](wine.csv) retains all 13 original measurement columns and source-order IDs; only the four declared columns enter this experiment. [Provenance and the CC BY 4.0 attribution](data-provenance.md) accompany it.

We deliberately turn each selected measurement into a binary indicator: is it strictly above that feature's **training median**? This makes the learned CPTs inspectable, at the cost of losing detail. Median thresholds are data-processing choices, not chemical laws.

Compare two recipes. **Naive Bayes (NB)** uses cultivar \(C\) as the parent of each feature. **Tree-augmented naive Bayes (TAN)** also allows each feature at most one feature parent. TAN chooses a maximum-weight spanning tree using conditional mutual information \(I(X_i;X_j\mid C)\), then orients that feature tree outward from the first feature and adds \(C\to X_i\). The extra arrows model residual feature associations within cultivar groups; they are not discovered chemical causes. [Friedman, Geiger and Goldszmidt's paper, section 4](https://dang.cs.technion.ac.il/journal_papers/friedman1997Bayesian.pdf), establishes the tree construction.

The score for an edge is

\[
\widehat I(X_i;X_j\mid C)
=\sum_{c,a,b}\widehat P(c,a,b)
\log\frac{\widehat P(a,b\mid c)}
{\widehat P(a\mid c)\widehat P(b\mid c)}.
\]

Zero empirical joint cells contribute zero. This asks whether knowing one feature helps predict the other after accounting for cultivar. The tree restriction keeps the search tractable. With three classes and four binary features, NB has 14 free parameters and TAN has 23; more expressive tables also divide observations into smaller groups. Maximum-likelihood tree selection does not guarantee better predictive performance after smoothing or on new data.

**Protocol fixed before observing scores:** split into training 106, validation 36 and test 36, stratified by cultivar with seeds 61 and 62. Fit medians, tree structure and CPTs on training only. Both models use one pseudo-count per state and the same four features. Choose NB or TAN by lowest validation mean log loss, breaking an exact tie in favor of NB. The prior-only classifier is a reference, not an extra search over settings. Refit the selected recipe on the 142 development specimens and assess it once on the reserved 36.

Log loss is the average \(-\log p\) assigned to the observed class; lower is better. A confident wrong prediction is more costly than a tentative one. Accuracy counts only which class has the largest probability.

| Training-only model | Validation correct | Validation log loss |
| --- | ---: | ---: |
| Prior only | 14/36 | 1.089852 |
| NB | 34/36 | **0.162922** |
| TAN | 34/36 | 0.185233 |

NB wins the declared criterion despite TAN's extra connections. Refit NB scores **33/36 correct and 0.270944 log loss** on test; the development prior alone scores 14/36 and 1.089616. This small, single split cannot establish a universal ordering. No collection dates or grouping metadata support a claim about new regions or later vintages.

**Visual — measured probabilities, not a winner trophy.** Pair each validation specimen's NB and TAN probability assigned to its true cultivar, with a log-loss contribution beneath it. Keep all specimens available, including disagreements and confident errors. The complete saved calculation records row identities and fitted tables.

### Missing a measurement means summing possibilities

The training medians are \([13.05,1.90,2.11,4.75]\). The learned TAN feature parent is alcohol for each of the other three selected measurements, in addition to cultivar. Examine validation specimen 156, whose values are \([13.17,5.19,0.63,7.90]\).

| Visible measurements, fixed training TAN | \(P(C=0)\) | \(P(C=1)\) | \(P(C=2)\) |
| --- | ---: | ---: | ---: |
| None | 0.3303 | 0.4037 | 0.2661 |
| Alcohol only | 0.5935 | 0.0932 | 0.3133 |
| Flavanoids only | 0.0323 | 0.4668 | 0.5009 |
| Alcohol and flavanoids | 0.0499 | 0.1293 | 0.8208 |
| All four | 0.0228 | 0.0130 | 0.9641 |

With only alcohol, cultivar 0 leads. Reveal flavanoids and cultivar 2 leads. To calculate the two-measurement row, sum the joint over both states of malic acid and color intensity, then normalize across cultivar. Do not fill an unknown feature with its more likely state: that discards part of the joint probability mass.

**Investigation — purchase a measurement.** Start with a validation specimen and no visible measurements. Purchase one of two possible measurements, compare how the leading class and uncertainty change, then inspect the table entries used. The acquisition is an actual information choice, with no learner-answer requirement. Edit a visible numerical value across its fixed median to create a counterfactual specimen; label it as an edited input, not another measured record. Edit it within the same bin and the probability must stay identical. Edit a hidden value and it must also stay identical until revealed.

This is a demonstration of conditional inference under a fixed fitted model, not a validated measurement-purchasing policy. Missingness caused by the unobserved value itself may require modeling the missingness mechanism. The visible-subset calculations alone do not establish that ignoring that mechanism is appropriate.

Download [network-experiments.py](network-experiments.py) beside the CSV and run it with Python, NumPy and scikit-learn. It contains the complete counts, tree selection, exact missing-feature marginalization, split protocol and saved numerical examples; it writes [calculated-inputs.json](calculated-inputs.json). Here is its inference mechanism in words you can map to the code: enumerate the 16 possible binary feature states, discard only assignments that contradict visible values, multiply the prior and four CPT entries for each cultivar, add compatible assignments, then normalize. With no visible feature, all conditional tables sum away and the answer is the class prior.

## 6. Changing a mechanism is different from selecting evidence

Now add causal meaning explicitly. Consider a constructed maintenance model:

\[
Z\to X\to Y,\qquad Z\to Y,
\]

where \(Z=1\) denotes high load, \(X=1\) a particular service procedure and \(Y=1\) a later failure. Half the systems have high load. The service procedure is used with probability 0.20 at low load and 0.60 at high load.

| Load \(Z\) | Failure probability without procedure | Failure probability with procedure |
| --- | ---: | ---: |
| 0 | 0.01 | 0.05 |
| 1 | 0.10 | 0.20 |

These numbers intentionally describe a harmful procedure. Names such as “treatment” or “service” do not make an intervention beneficial.

Among serviced systems, 75% have high load: \(0.5(0.6)/[0.5(0.2)+0.5(0.6)]=0.75\). Their failure probability is \(0.25(0.05)+0.75(0.20)=0.1625\). Among unserviced systems, the high-load share is \(0.5(0.4)/[0.5(0.8)+0.5(0.4)]=1/3\), giving failure probability 0.04. The observed difference is 0.1225.

Suppose we instead assign **every** system the procedure while leaving load and the failure mechanism unchanged. The high-load share remains one half, so

\[
P(Y=1\mid do(X=1))=0.5(0.05)+0.5(0.20)=0.125.
\]

Assigning no system the procedure gives 0.055. The causal risk difference is **0.0700**, seven percentage points. The observational risk difference overstates it by \(0.1225-0.0700=0.0525\), not by the treated probability's separate difference of 0.0375.

The **do operator** denotes the intervention that replaces the mechanism assigning \(X\) with a fixed value. In a causally interpreted, fully observed DAG with independent external disturbances, the post-intervention joint is the old factor product with \(P(x\mid pa_X)\) removed, \(X\) fixed, and the other mechanisms retained. This is the truncated factorization. Conditioning on \(X=x\) instead keeps its assignment factor and renormalizes, changing the composition of the selected group.

**Visual — two population lanes.** The observation lane selects serviced units from the original low/high-load mixture; the intervention lane copies the entire 50/50 population and replaces only its assignment mechanism. Area widths show the different weights. The graph must show the cut incoming arrow \(Z\to X\) only in the intervention lane.

**Investigation — change who receives the procedure.** Predict what happens to the observational difference and causal difference if both assignment probabilities become 0.40. Recalculate: the observed difference becomes 0.07, while the causal difference stays 0.07. Changing assignment alone changes selection, not the fixed failure response. Next change one failure-table entry and predict which causal average moves. Keep the two mechanisms separately editable.

A causal graph is an assumption about the system, not a conclusion granted by fitting a BN to observations. A random experiment can justify an assignment mechanism; domain knowledge, temporal constraints and scientific arguments support other arrows or exclusions. Observational fit alone does not establish that replacing a table describes a real intervention.

### Adjustment: block the paths that enter the treatment

In the service model, \(X\leftarrow Z\to Y\) is a backdoor path. Comparing procedure groups within load strata and averaging with the target population's load distribution gives

\[
P(y\mid do(x))=\sum_zP(y\mid x,z)P(z).
\]

The familiar sufficient **backdoor criterion** selects a set containing no descendant of \(X\) and blocking every path from \(X\) to \(Y\) whose first arrow points into \(X\). You also need the required treatment levels to occur in the strata being averaged—**positivity**—and measurements and causal assumptions appropriate to the question.

Try assignment probabilities \([0,1]\): procedure status now identifies load perfectly. The fully specified teaching model still calculates a 0.07 causal difference, but observed data alone never reveal the missing low-load/procedure and high-load/no-procedure response cells. Do not present the adjustment calculation as estimated from those data. This separates a known generative model from identification using available observations.

There can be more than one valid adjustment set. For \(S\to E,\ S\to Y,\ E\to T,\ T\to Y\), the only backdoor route for \(T\)'s effect on \(Y\) is \(T\leftarrow E\leftarrow S\to Y\). Either \(\{E\}\) **or** \(\{S\}\) blocks it; neither contains a descendant of \(T\). Their union also works, with the required support. The graph does not declare the more upstream variable universally better. Measurement quality, cost, support and statistical efficiency can distinguish valid choices.

This is why “adjust for everything available” is unreliable. A collider can open a route, a mediator can remove part of the total effect, and additional variables can worsen overlap. First name the causal quantity, then inspect the paths. The graph-based criterion and its assumptions are developed in [Pearl's causal-inference paper, printed pages 2517–2519](https://ftp.cs.ucla.edu/pub/stat_ser/r416-reprint.pdf).

## 7. Deeper: identify an effect, then ask whose counterfactual it is

### A measured mediator can sometimes bypass an unmeasured cause

Suppose an unobserved \(U\) influences both \(X\) and \(Y\), while \(X\to M\to Y\). There is no direct \(X\to Y\) arrow. Under this graph, the **frontdoor criterion** can identify the total effect using observed \(X,M,Y\):

1. \(M\) intercepts every directed path from \(X\) to \(Y\).
2. No backdoor path from \(X\) to \(M\) is open.
3. Conditioning on \(X\) blocks every backdoor path from \(M\) to \(Y\).

With the required support, the result is

\[
P(y\mid do(x))=\sum_mP(m\mid x)\sum_{x'}P(y\mid m,x')P(x').
\]

The inner average estimates the response to setting the mediator, correcting its association with the upstream treatment. The outer average combines those responses using the mediator distribution induced by setting \(X=x\). Simply conditioning on the mediator is not the same calculation. These assumptions and the formula appear in [Mohan and Pearl's graphical-model tutorial](https://ftp.cs.ucla.edu/pub/stat_ser/uai12-mohan-pearl.pdf).

Here is a fully specified constructed example. Let \(U\) be fair; \(P(X=1\mid U=0,1)=[0.2,0.8]\); \(P(M=1\mid X=0,1)=[0.1,0.9]\). Failure probabilities \(P(Y=1\mid M,U)\) are:

| \(M\) | \(U=0\) | \(U=1\) |
| --- | ---: | ---: |
| 0 | 0.05 | 0.40 |
| 1 | 0.50 | 0.90 |

Generating the joint and then hiding \(U\) gives \(P(X=0)=P(X=1)=0.5\). The observed failure probabilities given \(M,X\) are 0.12 and 0.33 for \(M=0\), and 0.58 and 0.82 for \(M=1\). The inner averages are therefore 0.225 and 0.700. For \(do(X=0)\), weight them by \([0.9,0.1]\) to obtain 0.2725; for \(do(X=1)\), use \([0.1,0.9]\) to obtain 0.6525.

Directly intervening in the complete model gives exactly the same two answers. In contrast, the observational probabilities \(P(Y=1\mid X=0,1)\) are \([0.166,0.771]\). The saved program calculates both routes from all 16 worlds. This is an identification demonstration, not empirical evidence that a real mediator satisfies the criterion.

**Visual — two nested averaging trays.** First show the two \(X'\)-weighted mediator-response mixtures; then place those two results into the \(M\mid X\) mixture. Keep the hidden-\(U\) model available as a separate verification view. Mark the three path assumptions on the graph.

Adding a direct \(X\to Y\) arrow breaks this frontdoor criterion because \(M\) no longer intercepts every directed causal path. An unobserved common cause does not automatically make all causal questions unidentifiable: an observed noncollider can block a longer backdoor route, and a valid frontdoor construction can help when ordinary adjustment cannot. Conversely, observing a mediator does not guarantee frontdoor identification.

### The three do-calculus rules, with their graph operations

For disjoint variable sets \(X,Y,Z,W\), let \(G_{\bar X}\) remove arrows entering \(X\), and \(G_{\underline Z}\) remove arrows leaving \(Z\). Combined subscripts apply both operations. These rules transform expressions when the indicated d-separation holds:

\[
\begin{aligned}
P(y\mid do(x),z,w)&=P(y\mid do(x),w)
&&\text{if }Y\perp Z\mid X,W\text{ in }G_{\bar X};\\
P(y\mid do(x),do(z),w)&=P(y\mid do(x),z,w)
&&\text{if }Y\perp Z\mid X,W\text{ in }G_{\bar X,\underline Z};\\
P(y\mid do(x),do(z),w)&=P(y\mid do(x),w)
&&\text{if }Y\perp Z\mid X,W\text{ in }G_{\bar X,\overline{Z(W)}}.
\end{aligned}
\]

Here \(Z(W)\) contains the \(Z\)-nodes that are not ancestors of any \(W\)-node in \(G_{\bar X}\). When \(W\) is empty, it is all of \(Z\). The first rule removes an irrelevant observation; the second exchanges an action and an observation; the third removes an irrelevant action. Each requires its own modified graph. Ordinary d-separation in the original graph cannot substitute for these tests. [Pearl's rule statements](https://ftp.cs.ucla.edu/pub/stat_ser/r416-reprint.pdf) give the formal conditions.

For example, in a causally sufficient \(X\to Y\) model with no other paths, delete the outgoing arrow from \(X\). Now \(X\) and \(Y\) are separated, so the action/observation exchange justifies \(P(y\mid do(x))=P(y\mid x)\). Add an unobserved common cause and that remaining backdoor route prevents this argument.

Identification asks whether **every** causal model satisfying the assumptions and yielding the same observational distribution agrees on the causal query. Estimation asks how to approximate an identified quantity from finite observations. Do-calculus plus probability operations is complete for the relevant interventional identification problems in the standard acyclic causal framework with allowed latent variables; it is not a promise that every effect, every counterfactual or every feedback system is identified.

### Counterfactuals need information about the same unit across worlds

A structural causal model (SCM) writes variables as assignments such as \(Y=f_Y(X,U_Y)\), with external variables \(U\) describing variation not otherwise represented. Independent external noises in an acyclic, fully observed model yield the familiar causal DAG factorization. Dependent external variables require representing that latent dependence rather than silently multiplying independent noise distributions.

An individual counterfactual uses three steps: infer the external state from what happened (**abduction**), replace the specified assignment (**action**), and run the changed model using that same inferred state (**prediction**). A population intervention averages over the population's external states instead.

Consider randomized binary \(X\) and an independent fair binary \(U\). Compare:

- Model A: \(Y=U\).
- Model B: \(Y=X\mathbin{\mathrm{XOR}}U\), where XOR is 1 exactly when its inputs differ.

Both models have \(P(Y=1\mid X=x)=P(Y=1\mid do(X=x))=1/2\) for either \(x\). They agree on every observational and population-interventional distribution of \(X,Y\). Yet for a unit observed with \(X=0,Y=0\), both infer \(U=0\); changing its \(X\) to 1 yields \(Y=0\) in A and \(Y=1\) in B.

The distinction is the coupling of the same person's outcomes across settings. A conditional probability table alone did not determine it. One may include an unused \(X\to Y\) parent in A if using a common permissive graph; it then exhibits an extra independence rather than faithfulness. Some particular counterfactual quantities can be identified under weaker assumptions than a fully numerically specified SCM, but the population CPTs in this example are insufficient.

**Visual — paired potential outcomes.** Show the two units \(U=0,1\) as persistent row identities, with \(Y_0,Y_1\) in adjacent columns. A uses rows \((0,0),(1,1)\); B uses \((0,1),(1,0)\). Column averages agree, while row-level changes do not. This representation makes the missing information visible without an elaborate simulation.

## 8. Larger networks, structure learning and alternatives

### A graph can predict well without revealing a unique direction

With complete discrete training data, fixed-graph likelihood is easy to evaluate. Structure search can add, remove or reverse an edge while preserving acyclicity, scoring fit against complexity. Under the usual regular-model approximation, a BIC score to **maximize** is \(\ell(\hat\theta)-(k/2)\log n\); equivalently minimize \(-2\ell+k\log n\). Use the actual CPT parameter count for \(k\). This is a statistical approximation with assumptions, not a universal penalty for arbitrary latent or singular models.

Searching many DAGs can be expensive and greedy hill climbing can stop at a local optimum. Parent limits, domain constraints and score caching can help. Conditional-independence approaches such as PC instead remove adjacencies through independence tests and orient what the resulting constraints justify. Under their standard correctness conditions they rely on assumptions such as causal sufficiency, Markovness, faithfulness and suitable independence information; finite samples can yield unstable tests. Latent-variable approaches such as FCI address a different assumption set and represent partially determined structure.

For ordinary DAG independence models, **Markov-equivalent** graphs have the same skeleton and the same unshielded colliders. The skeleton forgets arrowheads; an unshielded collider \(X\to Z\leftarrow Y\) has no \(X-Y\) edge. A chain \(X\to Z\to Y\) and a fork \(X\leftarrow Z\to Y\) both imply \(X\perp Y\mid Z\). Pure observational independence information cannot choose between them. Additional functional assumptions or interventions may identify more; do not claim observational orientation is always impossible under every model class.

**Visual — an equivalence family.** Arrange the three noncollider orientations of a three-node chain together, with their shared independence statement. Put the collider orientation separately. A missing or spurious adjacency is not explained away as membership in the same equivalence class.

If values or variables are missing during fitting, local complete-data counts no longer suffice. EM alternates posterior expected sufficient counts under current parameters and parameter updates. Exact E-steps may be costly; local optima and unidentifiability remain possible. This connects directly to the earlier GMM lesson, where component memberships were hidden rather than graph nodes observed.

### Compile repeated exact queries, or approximate deliberately

A **junction tree** groups interacting variables into clusters connected as a tree. Neighboring clusters exchange functions over their shared variables. Every variable must appear in a connected set of clusters—the running-intersection property—so messages can summarize excluded subproblems consistently. Building such clusters typically involves moralization and triangulation: add fill edges to remove chordless cycles of length four or more, then organize cliques. Triangulation does not remove every graph cycle. Its large clusters are where treewidth reappears.

Sum-product messages are exact on an appropriate tree of factors or clusters. Loopy belief propagation applies similar local updates to a graph with loops; convergence and exactness are no longer automatic. [Stanford's junction-tree chapter](https://ermongroup.github.io/cs228-notes/inference/jt/) is a useful derivation after the factor workbench.

For large discrete models, approximate choices include:

| Method | Main operation | A failure worth checking |
| --- | --- | --- |
| Ancestral sampling | Sample each node after its parents | Rare evidence causes rejection to discard almost everything |
| Likelihood weighting | Fix evidence; weight samples by its likelihood | A few samples may carry almost all weight |
| Gibbs sampling | Resample each variable conditional on its blanket | Strong or deterministic constraints can prevent useful movement |
| Variational inference | Optimize a tractable approximate distribution | The family and objective can miss important dependence or modes |
| Loopy belief propagation | Iterate local messages | Oscillation or a stable but inaccurate fixed point |

For the alarm's two-call evidence, rejection sampling retains about 0.2084% of prior samples on average: roughly 208 of 100,000. Weighting avoids literal rejection but can still have severe weight concentration. For correlated Monte Carlo draws, raw draw count overstates precision; assess effective sample size and exploration. A Gibbs chain constrained to equal binary variables may be unable to change either coordinate alone. Blocked updates or another sampler may be needed. [Stanford's sampling chapter](https://ermongroup.github.io/cs228-notes/inference/sampling/) provides the mechanisms; approximate inference is not certified by a smooth trace.

Also distinguish three query types. A marginal asks for \(P(Y\mid e)\). MPE selects the highest-probability assignment to **all** remaining hidden variables. A marginal-MAP query maximizes over a selected set after summing other hidden variables. Max and sum generally cannot swap. With joint masses \(P(Q=0,H=0)=0.30,\ P(0,1)=0.30,\ P(1,0)=0.39,\ P(1,1)=0.01\), the highest-probability world has \(Q=1\), but the marginal-MAP answer is \(Q=0\) because its combined mass is 0.60.

### Choose a representation for the question

An HMM is a time-unrolled directed model with repeated local transition and emission rules. Its forward algorithm is specialized variable elimination. The next lesson, **Conditional Random Fields**, instead models \(P(\text{labels}\mid\text{observations})\) directly; it need not provide a generative distribution for the observations. This distinction affects which quantities a model can answer.

An undirected Markov random field uses compatibility factors and a partition function. Neither directed nor undirected graphical independence models universally contains the other. A nonchordal undirected cycle and a directed collider illustrate why their conditional-independence semantics differ. Linear-Gaussian BNs replace discrete CPTs with local linear regressions and Gaussian disturbances; they can retain analytic Gaussian inference. Nonlinear or neural conditional distributions broaden expressive power but may require different inference methods.

Probabilistic programming can express richer hierarchical and latent models, but the inference engine still has requirements. Gradient-based HMC is designed for suitable continuous latent spaces, not raw discrete state jumps. Discrete enumeration, marginalization, custom samplers or variational approximations may be appropriate. A language's expressiveness does not make every query computationally easy.

A useful application beyond classification is **diagnosis with selectively missing sensors**: maintain a distribution over underlying faults while measurements arrive. A separate utility model can compare the expected value and cost of a test; probability alone does not decide what action is worthwhile. Reliability planning similarly distinguishes observing a component failure from replacing a component's failure mechanism. Genomic and neural applications can use graphs to state competing explanations, but correlated signals and selection effects must not be relabeled as causal connections.

### A current library route

The self-contained [pgmpy example](pgmpy-example.py) builds the same five-node network using explicit state order, checks its CPTs and asks the two-call query. It follows the currently inspected [DiscreteBayesianNetwork API](https://pgmpy.org/api/generated/models/pgmpy.models.DiscreteBayesianNetwork.html) and [VariableElimination API](https://pgmpy.org/api/generated/inference/pgmpy.inference.VariableElimination.html). Its expected answer is the independently calculated 0.284172 above; this optional library program has **not been executed** in the shared environment.

Do not infer a fixed-value intervention from a method's name alone. Inspect whether an API cuts incoming edges, changes a CPT, sets a particular state or samples an intervention. State ordering and evidence-column ordering also matter: a normalized table can still encode the wrong parent configuration.

## 9. Practice: explain the changed case before calculating

Work through 1–6 before the deeper problems. Each problem changes a mechanism or assumption rather than asking you to copy the preceding trace.

### 1. A different caller

Replace Mary's table by \(P(M=1\mid A=0)=P(M=1\mid A=1)=0.4\). With only Mary calling, what is the burglary posterior? With both John and Mary calling, which earlier posterior should reappear?

<details><summary>Hint</summary>
The factor for Mary's evidence is constant across all remaining worlds.
</details>
<details><summary>Solution</summary>
It cancels between numerator and denominator. Mary alone gives the prior 0.001. Both calls give John's posterior, approximately 0.016284. Equal conditional rows remove Mary's information in this distribution, even though the displayed graph can retain a redundant arrow.
</details>

### 2. A descendant opens a path

Draw \(R\to S\leftarrow T,\ S\to V\to W\). Are \(R,T\) d-separated with no observations, with \(W\) observed, and with both \(S,W\) observed? Does the graph specify a negative correlation in the latter two cases?

<details><summary>Hint</summary>
Ask whether the collider has an observed descendant; then separate “active” from the sign of association.
</details>
<details><summary>Solution</summary>
They are separated with no observations. Both other sets open the path because the collider itself or a descendant is observed. The graph supplies no numerical sign or strength. Explaining away was a result of the alarm's particular probabilities, not a universal negative-correlation theorem.
</details>

### 3. Count a different network

Let a three-state \(C\) parent three binary features \(F_1,F_2,F_3\), and add \(F_1\to F_2\). How many free CPT parameters are there? Compare with a fully unrestricted joint over these four variables.

<details><summary>Hint</summary>
Count each parent configuration, then multiply by child states minus one.
</details>
<details><summary>Solution</summary>
The class contributes 2, \(F_1\) contributes 3, \(F_2\) contributes \(3\times2=6\), and \(F_3\) contributes 3: total 14. The unrestricted joint has \(3\times2^3=24\) entries and 23 free parameters. These count probabilities, not data rows or bytes.
</details>

### 4. A probability query with one hidden variable

You have \(P(C=1)=0.4\), \(P(F=1\mid C=0)=0.2\), \(P(F=1\mid C=1)=0.8\). A downstream measurement \(G\) depends only on \(F\), with \(P(G=1\mid F=0)=0.1\) and \(P(G=1\mid F=1)=0.9\). Calculate \(P(C=1\mid G=1)\) by summing over \(F\).

<details><summary>Hint</summary>
First obtain the two likelihoods \(P(G=1\mid C)\).
</details>
<details><summary>Solution</summary>
They are \(0.8(0.1)+0.2(0.9)=0.26\) for \(C=0\) and \(0.2(0.1)+0.8(0.9)=0.74\) for \(C=1\). Thus the answer is \(0.4(0.74)/[0.6(0.26)+0.4(0.74)]=0.296/0.452\approx0.654867\). Choosing the most likely \(F\) first would solve a different problem.
</details>

### 5. Valid adjustment sets, changed graph

For \(S\to E,\ S\to Y,\ E\to T,\ T\to Y\), check \(\{E\}\) and \(\{S\}\). Now add \(E\to Y\). Which sets among \(\{E\},\{S\},\{E,S\}\) still satisfy the backdoor criterion for the total effect of \(T\)?

<details><summary>Hint</summary>
List the new route that begins \(T\leftarrow E\).
</details>
<details><summary>Solution</summary>
Originally both singleton sets work. The added \(T\leftarrow E\to Y\) route is not blocked by \(S\). Now \(\{E\}\) and \(\{E,S\}\) work, while \(\{S\}\) does not. Required support and correct graph assumptions still apply. Being earlier in the graph is not sufficient to be a valid adjustment variable.
</details>

### 6. A more cautious real-data claim

In the Wine experiment NB and TAN classify the same number of validation specimens correctly, but NB has lower log loss. Explain how this can happen. Would reporting TAN's test result after seeing NB's test loss preserve the declared selection protocol? In the measurement investigation, explain why changing a hidden alcohol value must leave a posterior unchanged.

<details><summary>Hint</summary>
Distinguish the largest-probability label from the full probability vector, and information available to a query from a value stored in a record.
</details>
<details><summary>Solution</summary>
The models can differ in confidence, and may even make different errors with the same total count. Log loss uses the true-class probabilities; accuracy does not. Choosing or emphasizing a second model after inspecting the held-out result reuses test information for selection; a new selection requires new evaluation or a transparently exploratory report. A hidden value is marginalized over, so the query cannot read it until it becomes evidence.
</details>

### 7. Change a causal response, not its assignment

In the service example change the low-load procedure failure probability from 0.05 to 0.01, leaving everything else fixed. What is the new causal difference? Under the original assignment probabilities, what is the observed difference?

<details><summary>Hint</summary>
Use population weights \([1/2,1/2]\) for the intervention and serviced-group weights \([1/4,3/4]\) for the observation.
</details>
<details><summary>Solution</summary>
The intervention probability with the procedure is \(0.5(0.01)+0.5(0.20)=0.105\). Without it, 0.055 remains, so the causal difference is 0.05. Observed serviced risk becomes \(0.25(0.01)+0.75(0.20)=0.1525\); subtracting unchanged 0.04 gives 0.1125. Changing a response cell affects the two averages by different amounts because their weights differ.
</details>

### 8. Marginal answer versus best world

Change the four masses in section 8 to \([0.20,0.25,0.40,0.15]\) in the same \((Q,H)\) order. Find both answers. Why does matching on this example not justify replacing sum with max in general?

<details><summary>Hint</summary>
Compare the largest cell with each row sum.
</details>
<details><summary>Solution</summary>
The most likely world is \((1,0)\), and the marginal-MAP value is also \(Q=1\), whose mass is 0.55 versus 0.45. They coincide here. Section 8's original table is a counterexample to the proposed general shortcut: one coincidence cannot establish an algebraic identity.
</details>

### 9. Break a frontdoor assumption

In section 7's graph add \(U\to M\). Is the displayed frontdoor formula still justified by that criterion? What if instead you add only \(X\to Y\)?

<details><summary>Hint</summary>
Check the \(X\)-to-mediator backdoor and the set of directed \(X\)-to-\(Y\) paths separately.
</details>
<details><summary>Solution</summary>
The first addition opens \(X\leftarrow U\to M\), violating the second condition. The direct \(X\to Y\) addition instead violates the interception condition. Neither change licenses the same formula. Failure of a sufficient criterion alone does not prove every effect unidentifiable; reassess the graph and available assumptions.
</details>

### 10. Find what the probability tables leave unspecified

For the two SCMs in section 7, observe \(X=1,Y=0\). Infer \(U\), then set \(X=0\). What does each model predict? Which population quantities remain equal?

<details><summary>Hint</summary>
Keep each model's inferred external state fixed when changing \(X\).
</details>
<details><summary>Solution</summary>
Model A infers \(U=0\) and still predicts \(Y=0\). Model B infers \(U=1\) and predicts \(Y=1\) after setting \(X=0\). Both still have fair \(Y\) under either intervention and the same observational \(X,Y\) distribution. They disagree about paired, unit-level outcomes, which those distributions did not specify.
</details>

You are ready to continue when you can explain why an unobserved variable is summed out, identify a collider's observed descendant, distinguish a causal query from a predictive one, and justify a small adjustment set. For the deeper route, also distinguish identification from finite-sample estimation and individual counterfactual coupling.

## 10. References and other ways to learn

- **Graph-first course:** [Stanford CS228 notes](https://ermongroup.github.io/cs228-notes/), by Volodymyr Kuleshov and Stefano Ermon with course staff. Begin with directed representation, then variable elimination and junction trees; use the sampling chapter after exact sums are familiar. The course also covers undirected models, latent-variable learning and variational inference. Read the formulas critically: these evolving notes acknowledge possible errors.
- **Worked classifier extension:** [Bayesian Network Classifiers](https://dang.cs.technion.ac.il/journal_papers/friedman1997Bayesian.pdf), Friedman, Geiger and Goldszmidt, 1997. Section 4 explains the conditional-information tree and why a constrained graph search is tractable. Our discretization, split, smoothing choice and measurements are an original small experiment, not a reproduction of the paper's benchmark.
- **Causal graph walkthrough in slide form:** [Graphical Models for Causal Inference](https://ftp.cs.ucla.edu/pub/stat_ser/uai12-mohan-pearl.pdf), Mohan and Pearl. Use the d-separation, intervention and frontdoor diagrams as an alternative visual explanation; the presentation is more formal than this lesson's first pass.
- **Precise causal rules:** [The Mathematics of Causal Inference](https://ftp.cs.ucla.edu/pub/stat_ser/r416-reprint.pdf), Pearl. Read the structural-model setup and printed pages 2517–2519 for d-separation, identification, do-calculus and adjustment. Later mediation and transport sections go beyond this lesson's core route.
- **Runnable library alternative:** [pgmpy discrete inference examples](https://pgmpy.org/examples/Inference_Discrete_BN.html) and [parameter-learning examples](https://pgmpy.org/examples/Parameter_Learning_Discrete_BN.html). They complement our small explicit sums with a reusable API. Documentation inspected September 2026; verify installed-version behavior before relying on copied API calls.
- **Real-data source:** [UCI Wine](https://doi.org/10.24432/C5PC7J), Aeberhard and Forina, with [our extraction and reproducibility details](data-provenance.md). Reuse its measurements to inspect probabilities, not to infer chemical causality or a new-region deployment guarantee.

The next topic in this module is [Conditional Random Fields (CRF)](/learn/path/full-curriculum/conditional-random-fields-crf?module=classical-ml): carry forward factors and normalization, then ask what changes when only the conditional label distribution is modeled. The separate [causal-inference](/learn/path/full-curriculum/causal-inference-do-calculus?module=math-foundations), [Monte Carlo](/learn/path/full-curriculum/monte-carlo-methods-mcmc-metropolis-hastings-hmc-nuts?module=math-foundations) and [variational-inference](/learn/path/full-curriculum/variational-inference?module=math-foundations) lessons extend the deeper branches; none is a reason to skip the module's normal reading sequence.
