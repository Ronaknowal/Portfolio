# PAC Learning & VC Dimension

A classifier can get every training example right and still fail on the next one. That is easy to say. The harder question is **what would make success on a finite sample trustworthy?**

Imagine learning which temperature range makes a simple laboratory indicator turn on. You observe a few temperatures and their on/off outcomes, then choose an interval that fits them. There are infinitely many possible interval endpoints, but the observations constrain those intervals in a highly organized way. Compare that with a lookup table allowed to assign an arbitrary answer at every unseen temperature. Both can fit the sample; their ability to make arbitrary choices beyond it is very different.

PAC learning gives a precise language for the accuracy a learning procedure can achieve from random examples. VC dimension measures one kind of flexibility of a binary prediction family. Together they connect **the examples we observe, the choices a learner is allowed to make, and the error it may make on new examples**.

The preceding Evaluation Metrics lesson defined different kinds of error. Here our core loss is deliberately simple: 0 for a correct binary prediction and 1 for an incorrect one. We will build the learning guarantee from that unit before introducing its equations.

Read sections 1–7 for the core argument. Section 8 turns it into reproducible experiments; sections 9–10 connect the result to model choice, computation and modern generalization theory. The small constructions are worth doing by hand before opening the programs.

## 1. Two different kinds of randomness

Let x be an input and y its observed class, either 0 or 1. A **hypothesis** h is one prediction rule; a **hypothesis class** H is the collection of rules the learner is allowed to choose from. A threshold, an interval, and an arbitrary lookup table define different classes.

The training sample is S = ((x₁,y₁),…,(xₙ,yₙ)). Assume for now that these pairs are independent draws from the same fixed distribution D. A learning algorithm A takes the sample and returns h_S = A(S).

For a fixed rule h, its **population risk** is the chance of making an error on a fresh draw:

\[
R_D(h)=\Pr_{(X,Y)\sim D}(h(X)\ne Y).
\]

Its **empirical risk** is the fraction of observed sample errors:

\[
\widehat R_S(h)=\frac1n\sum_{i=1}^{n}\mathbf1[h(x_i)\ne y_i].
\]

The indicator is 1 when the bracketed statement is true and 0 otherwise. Thus three errors among twenty examples give empirical risk 3/20. Population risk averages over the underlying distribution, which normally is not known exactly.

There are two probability levels. Once a model is fitted, its population risk concerns **a new example**. Before training, different random samples can produce different fitted models and therefore different risks. PAC's confidence concerns **that training-sample draw**.

**[Inline figure: samples outside, future examples inside.]** Several independently drawn training samples each feed the same algorithm. Each fitted rule receives its own fresh-example error meter. Mark a rule as successful if its population error is at most ε; count unsuccessful training draws using δ. Do not draw δ as the fraction of wrong predictions made by one successful model.

For example, “error at most 0.1 with probability at least 0.95” means that the learning procedure produces a rule with population error no greater than 10% on at least 95% of training draws under the stated setting. It does not say 95% of predictions are correct, nor that a particular unlucky sample can be identified from its training score. A randomized learner also contributes its internal randomness to the outer probability when the guarantee includes it.

## 2. The PAC contract, with its assumptions visible

**Probably Approximately Correct** separates two tolerances:

- ε is how much population error is acceptable: the “approximately correct” part.
- δ bounds how often the learning procedure may fail to achieve that target over its random training input: the “probably” part.

In **realizable binary PAC learning**, labels come from some target rule h* in H. Therefore a rule with zero population classification error exists inside the chosen class. A learner PAC-learns H if, for every ε and δ in(0,1), there is a sufficient sample size n_H(ε,δ) such that, for every allowed input distribution and every target h* in H, using at least that many independent examples gives

\[
\Pr_S\big(R_D(A(S))\le\varepsilon\big)\ge1-\delta.
\]

The sample-size guarantee cannot be selected afterward to suit the particular unknown distribution or target. That is the **distribution-free** part of this definition. It does not mean that the realized error is the same for all distributions.

A **consistent** learner chooses a rule that makes no training errors. An **empirical risk minimizer**, or ERM, chooses a rule with the smallest training error in H. Under realizability, an exact ERM is consistent because h* itself makes no errors. Zero training error is a property of the fit; the PAC argument is what connects it to population error.

Realizability can be a useful model for learning an exact threshold or a deterministic labeling rule. It is an assumption to examine, not something certified by fitting a flexible model until training error vanishes. In noisy or misspecified problems, no rule in H may have zero risk. We will introduce the **agnostic** contract in section 4.

A further distinction matters: **statistical learnability** asks whether a suitable sample size and learning rule exist. **Efficient learning** additionally constrains computation and representation size, usually by polynomial bounds. Finite VC dimension characterizes the standard statistical binary setting under suitable regularity conditions; it does not automatically provide a fast optimization algorithm. The original Blumer–Ehrenfeucht–Haussler–Warmuth paper explicitly separates these questions. [Original paper, introduction and sections 2–3](https://mwarmuth.bitbucket.io/pubs/J14.pdf).

## 3. Why a finite class can be learned

Suppose H contains K rules fixed before we see the sample, and the setting is realizable. Take one **bad** rule h whose true error exceeds ε. It can still make zero training errors if every sampled example happens to miss its error region.

For one independent example, the probability of missing that region is at most 1−ε. For n independent examples, it is at most

\[
(1-\varepsilon)^n\le e^{-n\varepsilon}.
\]

This is already the main intuition: repeated independent observations make it difficult for a genuinely bad fixed rule to keep hiding all its mistakes.

But the learner chooses among K rules. We must protect against **any** bad member surviving. The union bound says the probability of at least one event is no larger than the sum of the individual event probabilities:

\[
\Pr(\text{some bad }h\in H\text{ fits every sample label})
\le K e^{-n\varepsilon}.
\]

No independence between the K rules' failure events is required. They can make overlapping mistakes on the same examples. Independence of the sampled observations was used earlier to multiply the per-example probabilities; these are different assumptions.

If the right side is at most δ, then every consistent output is approximately correct on the protected event. Solving gives the sufficient sample condition

\[
n\ge\frac{\ln K+\ln(1/\delta)}{\varepsilon}.
\]

Here ln means natural logarithm. For K=32, ε=.05 and δ=.01, the right side is about 161.42, so 162 examples suffice under this particular bound. This is an **upper bound on sufficient sample size**, not a claim that 161 examples cannot work or that every real dataset needs 162.

**[Inline figure: eliminate bad rules by finding their mistakes.]** Fixed candidate cards carry a population error region in a tiny known world. Sampled points cross out inconsistent candidates. The caption distinguishes an individual survivor probability from the union over the predeclared class. The selected surviving rule is covered because all surviving rules are covered simultaneously.

### An exact four-point world

Take four equally likely inputs 0,1,2,3. The true labels are[0,0,1,1]. Let H contain all 16 binary labelings of these four points. Our fixed learner returns the lexicographically first consistent label vector: it uses observed labels at seen points and predicts 0 at unseen points.

With ε=.25, the learner fails only if it misses **both** positive inputs. Seeing even one positive leaves at most one wrong label, risk .25, which satisfies “at most ε.” Each draw misses both positives with probability 1/2, so

\[
\Pr(R_D(h_S)>.25)=(1/2)^n.
\]

At n=4, the exact failure probability is 1/16=.0625. The generic finite-class upper bound is 16 e⁻¹, larger than 1, so after clipping to the trivial probability limit it says only “at most 1.” The loose bound did not make a false prediction; it simply supplied no useful numerical restriction at that sample size.

At n=24, the exact failure probability is 1/16,777,216. The finite bound is 16 e⁻⁶≈.039660, sufficient for δ=.05 and still much larger than the exact failure probability for this learner in this distribution. Exact behavior, a general sufficient bound, and the smallest possible sample requirement are three different quantities.

### Investigation: which observations remove the uncertainty?

Choose observed inputs in the four-point world and record which unseen labels the fixed learner will predict, its true risk, and whether it meets ε=.25. Reveal the selected candidate and its mistakes. Adding a repeated observation gives no new information about unseen labels; adding a previously unseen positive changes the learned rule.

Then vary the target labels **before drawing a new sample**. The class still contains all 16 rules, but the selected learner's actual failure probability can change. For the all-zero target it is always correct, including before observing any positive. The bound remains a valid class-level statement. The complete program computes all seen-subset probabilities by exact rational arithmetic, so this finite example has no Monte Carlo uncertainty.

## 4. When zero error is impossible: uniform convergence and agnostic learning

Suppose a sensor sometimes gives an ambiguous result even at exactly the same input. Or suppose the true positive region consists of two separated intervals while H allows only one. We should then compare the learner with the best achievable risk **inside H**, rather than demand absolute error approaching zero.

An agnostic PAC guarantee has the form

\[
R_D(h_S)\le\inf_{h\in H}R_D(h)+\varepsilon
\]

with probability at least 1−δ. The infimum is the best risk attainable or approached within the class. The guarantee concerns **excess risk above that reference**. If the best class member has risk .12, a guarantee of excess risk at most .03 means a risk ceiling .15, not .03.

For a fixed rule with independent losses in[0,1], Hoeffding's inequality gives

\[
\Pr\big(|R(h)-\widehat R(h)|>r\big)\le2e^{-2nr^2}.
\]

For K fixed candidate rules, allocate failure probability δ/K to each and apply the union bound:

\[
\Pr\left(\forall h\in H:\ |R(h)-\widehat R(h)|\le r_K\right)\ge1-\delta,
\qquad
r_K=\sqrt{\frac{\ln(2K/\delta)}{2n}}.
\]

This is **uniform convergence**: one event protects every member simultaneously. It covers a member selected after seeing the evaluation scores, provided it belongs to that protected family. Concentration Inequalities developed the individual event and finite union; here uniform protection is what allows selection.

For K=25, n=500 and δ=.05, r_K≈.08311291. For a single fixed rule the radius is .06073615. Selecting among 25 incurs a larger allowance. The models can all be evaluated on the same 500 examples; their prediction errors need not be independent of one another. If they were fitted on a separate training sample, condition on that training sample before applying the held-out bound.

The family must be fixed independently of the evaluation outcomes, or a larger appropriately protected class must contain all possible candidates. Keeping only the final three models after inventing and testing hundreds of outcome-dependent alternatives does not justify paying for K=3. A final independent assessment or an analysis covering the selection procedure is needed.

**[Inline figure: one event covers the selected row.]** A matrix has candidate rules as rows and evaluation examples as columns. Each row's empirical error receives a simultaneous interval. Highlight the selected row only after all intervals are defined. Cross out an attempted replacement K=3 after discarding earlier searched models; retain the actual predeclared family.

### Why ERM pays the radius twice

On the uniform event, compare an ERM h_S with a best class member h* when the minimum exists:

\[
R(h_S)\le\widehat R(h_S)+r_K
\le\widehat R(h^*)+r_K
\le R(h^*)+2r_K.
\]

The first step moves from the chosen rule's empirical to population risk. The middle step uses ERM's defining property. The last moves the comparator's empirical risk back to its population risk. If a minimum does not exist, compare with increasingly near-optimal rules and take the infimum. If optimization stops η above the best empirical risk, add η to the result.

To make 2 r_K≤ε, a sufficient condition is n≥2 ln(2 K/δ)/ε². Contrast the ε⁻² dependence with the ε⁻¹ finite-class realizable argument. They concern different promises. The asymptotic exponents do not say a particular noisy dataset needs exactly 100 times as many examples at ε=.01; constants, class structure, noise assumptions and the reference error matter.

## 5. Infinite choices can still produce few label patterns

An interval's endpoints are real numbers, so there are infinitely many interval rules. Putting K=∞ into the finite union bound is useless. Yet on a finite ordered set of points, many different intervals make **exactly the same predictions**.

Take x₁=.2, x₂=.5, x₃=.8. An interval can label a consecutive block of points positive, or label all negative. There are seven patterns:

| Labels at(.2,.5,.8) | One witnessing interval |
|---|---|
|000| Empty positive region |
|100|[.2,.2]|
|010|[.5,.5]|
|001|[.8,.8]|
|110|[.2,.5]|
|011|[.5,.8]|
|111|[.2,.8]|

The missing pattern is 101. If an interval contains .2 and .8, it contains .5 as well. Trying more random endpoints will not help. The obstruction is structural.

A class **shatters** a set of points if it realizes every possible binary labeling on that same set. Three points have 2³=8 potential labelings. Intervals realize only 7, so they do not shatter this set.

The **VC dimension** is the largest size of a set that the class can shatter, or infinity if arbitrarily large finite sets can be shattered. Its quantifiers matter:

1. Find **some** set of d points.
2. Show that **every** binary labeling of that set has a witness in H.
3. To prove the dimension is exactly d, show that **every** set of d+1 points has at least one impossible labeling.

The witnessing rule can change when the requested labeling changes. One fixed rule is not expected to realize every labeling simultaneously. Conversely, showing that one unfortunate point configuration cannot be shattered does not upper-bound the entire class's VC dimension.

### Thresholds, intervals and half-planes

For increasing thresholds h_a(x)=1[x≥a], one point can receive either label. For x₁<x₂, the labeling 10 is impossible: a≤x₁ implies a≤x₂. Thresholds therefore have VC dimension 1. Allowing either threshold direction defines a different class; the convention is part of the statement.

Intervals shatter any two distinct points: neither, left only, right only, or both. For any three ordered points,101 is impossible. Their VC dimension is 2. Including the empty positive set makes the all-negative rule explicit and gives a consistent default when no positives are observed.

An affine half-plane in the plane predicts 1 when w₁x₁+w₂x₂+b≥0. Three noncollinear points can be shattered: the all-positive/all-negative cases use constant signs, and a line can separate each chosen vertex from the other two. Complementing a separating line gives the complementary patterns. This supplies all 8 labelings.

No four-point set can be shattered. If one point lies in the triangle formed by the other three, making those three positive forces the interior point positive too. If the four points form a convex quadrilateral, alternate the labels around its boundary: the positive diagonal and negative diagonal intersect, so no line can strictly separate the two groups. Degenerate collinear configurations already contain an ordered triple with an impossible alternating pattern. Thus the VC dimension of affine half-planes is 3.

**[Inline figure: a triangle, a quadrilateral, and an interior point.]** The triangle displays witnesses for its 8 patterns. The quadrilateral highlights crossing diagonals for an alternating pattern; the interior-point case shows the convex combination that prevents separation. Add a collinear triple to demonstrate that “there exists a shattered triple” does not mean every triple is shattered.

The program checks 8 feasible triangle patterns,14 feasible square patterns and 6 feasible collinear-triple patterns using linear-program feasibility and verified signed margins. Those finite numerical checks support the drawings. The geometry above supplies the universal upper-bound reasoning; an optimizer failing to find a separator by random search would not be a proof.

### Investigation: build or refute the requested labeling

Place distinct points on a number line, choose threshold or interval rules, and set the desired labels. Before adjusting endpoints, record whether you believe the labeling is possible and give a reason. Reveal a valid witness if one exists; otherwise identify the obstruction in the ordered labels.

Change an impossible interval request 101 to 111, then predict the result. Move all three points without changing their order:101 remains impossible. Move them past one another while keeping labels attached to their IDs: the sorted pattern can change, and so can feasibility. This separates meaningful geometric structure from a purely cosmetic coordinate change.

## 6. The growth function and the price of searching

The **growth function** Π_H(n) is the maximum number of distinct label patterns H can realize on any n-point set. It is not the number of parameter settings, and a count on one arbitrary configuration is only a lower bound on that maximum.

For increasing thresholds on n distinct ordered points, there are n+1 patterns: choose where the positive suffix starts, including before all points or after all points. For intervals, count consecutive positive blocks. There are n choices for a one-point block, n−1 for a two-point block, and so on, plus the all-negative pattern:

\[
\Pi_{\mathrm{intervals}}(n)=1+n+(n-1)+\cdots+1
=1+\frac{n(n+1)}2.
\]

| n | All binary patterns 2ⁿ | Increasing thresholds | Intervals |
|---:|---:|---:|---:|
|1|2|2|2|
|2|4|3|4|
|3|8|4|7|
|4|16|5|11|
|5|32|6|16|
|10|1,024|11|56|

At n=5, intervals realize 16/32=50% of all patterns. Read the numerator and denominator together: 16 realizable patterns out of 32 possible ones. The finite-dimensional class has polynomially many patterns; their fraction among all 2ⁿ patterns becomes small because the denominator is exponential.

**Sauer's lemma** generalizes this counting fact. If VC(H)=d, then

\[
\Pi_H(n)\le\sum_{i=0}^{\min(d,n)}\binom ni.
\]

For 1≤d≤n, this is at most(en/d)ᵈ. For d=0 the class realizes at most one pattern and the expression with division by d should not be used. For n≤d, the maximum is 2ⁿ. The bound is an upper bound, not a claim that every VC-d class attains it.

The recurrence behind the result is instructive. Remove one point from a set. Some patterns on the remaining points admit only one choice for the removed point; others admit both. Count every distinct restricted pattern once using capacity d, then add one extra copy for each pattern admitting both labels. That second collection has capacity at most d−1, because shattering d remaining points with both choices would shatter d+1 original points. This gives the same recurrence as binomial sums. It explains why a missing ability to realize all patterns constrains the later growth.

For n=100,d=5, the exact binomial sum is 79,375,496, compared with 2¹⁰⁰≈1.27×10³⁰. This upper bound alone does not establish that 100 examples yield a useful error guarantee; it must still enter a probabilistic analysis. [Mehta's lecture notes, sections 1–4: examples, Sauer's lemma and its consequence](https://web.uvic.ca/~nmehta/ml_theory_fall2021/lecture12.pdf).

### Why we cannot simply replace K with the observed pattern count

The patterns realized on the training inputs are themselves sample-dependent. Plugging their observed number into a fixed-family bound without further argument skips the reason the theorem works.

The usual proof introduces an independent **ghost sample** used only in the analysis. It relates population-versus-sample discrepancies to discrepancies between two samples, then controls the finitely many patterns on their combined inputs. Random exchanges between the samples and concentration make the counting argument legitimate. The ghost sample is a proof device; a learner need not secretly obtain a second labeled dataset to run ERM.

**[Inline figure: two samples, one pattern restriction.]** A fixed infinite class is restricted to combined sampled inputs; many parameter settings collapse to the same binary row. Exchange markers distinguish the random split used in the proof from an actual model-selection split. A caption says why counting only the selected training outcome would be insufficient.

## 7. What a VC guarantee actually says

For well-behaved binary hypothesis classes with finite VC dimension d, uniform convergence gives distribution-free statistical learning guarantees. “Well-behaved” includes the measurability conditions needed for the relevant random events; the ordinary finite, threshold, interval and half-plane examples here satisfy the standard conditions. The original theorem states this qualification explicitly. It is not a license to extend the result to arbitrary nonmeasurable constructions. [Blumer and colleagues, Theorem 2.1 and AppendixA 1](https://mwarmuth.bitbucket.io/pubs/J14.pdf).

One deliberately conservative explicit uniform bound, with n≥d≥1 and δ in(0,1), is

\[
\Pr_S\left(\forall h\in H:
|R(h)-\widehat R_S(h)|\le
\sqrt{\frac{32\{d\ln(en/d)+\ln(8/\delta)\}}{n}}
\right)\ge1-\delta.
\]

This is the convention used in the program and displayed bound curves. Sharper inequalities can use different constants, so identify the stated bound before comparing numerical values. Applying the result to 0–1 losses is valid because fixed labels simply flip the corresponding prediction bits and do not increase the maximum number of patterns. [Mehta, section 4 explicit VC inequality](https://web.uvic.ca/~nmehta/ml_theory_fall2021/lecture12.pdf).

For d=2,δ=.05, this raw radius is 2.183518 at n=100, .790026 at n=1,000, .277760 at n=10,000, and .095858 at n=100,000. A 0–1 risk gap is already bounded by 1. A radius above 1 is **vacuous** for that comparison: it adds nothing to the trivial range restriction. It is not evidence that the actual error exceeds 1 or that learning is impossible.

For agnostic ERM, apply the earlier three-step argument to get excess risk at most twice the uniform radius. In particular, a gap bound of .1 is not automatically an excess-risk bound of .1. With sharper analyses, the familiar distribution-free agnostic sample scale is proportional to (d+ln(1/δ))/ε² up to constants; the displayed elementary VC route retains an additional logarithmic factor. That sharper sample scale and our conservative displayed bound are different results. [Sharan, lecture 3, page 3, note 1](https://vatsalsharan.github.io/fall23/lec3.pdf).

Realizability permits stronger bounds because a surviving rule must avoid every training error. A classical explicit sufficient condition from Blumer and colleagues is

\[
n\ge\max\left\{
\frac4\varepsilon\log_2\frac2\delta,
\frac{8d}\varepsilon\log_2\frac{13}\varepsilon
\right\}.
\]

The logarithms in this formula are **base 2**, as in that paper. For intervals d=2,δ=.05, the rounded-up sufficient sizes at ε=.2,.1,.05 are 482,1,124,2,568. These conservative general-class guarantees differ from a bound exploiting the particular interval-learning algorithm or a particular input distribution.

The two main lessons are structural. Finite VC dimension prevents unrestricted fitting of all sufficiently large binary patterns, enabling statistical learning. Infinite VC dimension prevents a uniform distribution-free sample guarantee in this standard binary setting. It does **not** mean that no particular distribution or target in an infinite-VC family can be learned; alternative assumptions or target-dependent/nonuniform guarantees are different questions.

There is a useful connection to ensembles here. Later work removes the classical extra logarithmic factor in realizable sample complexity using carefully constructed votes of consistent learners. A vote can lie outside the original class, so a result about such a learner is not automatically a result about every ERM that must return a member of H. Hanneke's construction and Larsen's later bagging analysis connect abstract sample efficiency to combining fits on subsamples. They do not certify arbitrary random-forest defaults or noisy applications under a realizable theorem. [Hanneke's optimal-learning result](https://jmlr.org/papers/volume17/15-389/15-389.pdf), [Larsen's bagging result](https://proceedings.mlr.press/v195/larsen23a.html).

### The unseen-label argument behind the limitation

If a class shatters a large set, imagine a distribution supported on that set with labels chosen independently across its points. A training sample reveals only some labels. At an unobserved point, both labels are still compatible with what was seen; without additional structure, no learner can infer which one was chosen better than chance averaged over these targets.

For d equally likely shattered points and at most n distinct observed points, averaging over target labelings leaves an expected error of at least(d−n)/(2 d). This is an intuition-building lower-bound step, not a complete high-probability sample-complexity theorem. It explains why an arbitrarily large shattered set can defeat any proposed fixed sample size. The full lower-bound argument converts this remaining uncertainty into the appropriate failure probability.

## 8. Compute the constructions and run an honest learning experiment

Download [pac-calculations.py](pac-calculations.py). It is a complete program: exact interval/threshold pattern enumeration, checked two-dimensional separators, exact finite-world probabilities, bound calculations, the one-parameter construction in section 10, and a repeated interval-learning simulation.

For the measured-data branch, also download [banknote-learning-curves.py](banknote-learning-curves.py) and [banknote-subset.csv](banknote-subset.csv). Put the files in one directory. On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1
.venv\Scripts\python.exe pac-calculations.py
.venv\Scripts\python.exe banknote-learning-curves.py
```

On macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1
.venv/bin/python pac-calculations.py
.venv/bin/python banknote-learning-curves.py
```

The author used Python 3.12.14. Both programs run locally and save JSON next to themselves. Every numerical illustration below comes from those runs or an explicitly stated derivation.

### A fitted interval whose true risk we can calculate

Let X be uniform on [0,1] and the target turn on exactly inside [.3,.7]. The learner returns the smallest closed interval containing all observed positive inputs. If no positive input is observed, it returns the empty positive region. This default is essential: inventing a central positive interval would not necessarily be consistent with an all-negative sample.

For observations [.1,.2,.35,.55,.65,.9], the positive inputs are .35,.55,.65, so the learned interval is [.35,.65]. Training error is 0. It misses[.3,.35) and(.65,.7], total length .1. Under the uniform distribution, length equals probability, so its exact population risk is .1.

Add observations .31 and .69: the learned interval expands to [.31,.69] and exact risk falls to .02. Add only negative observations .01 and .99 instead: the interval and risk stay unchanged. With observations [.1,.2,.8,.9], there are no positives; the empty rule is consistent and its true risk is .4.

**[Inline figure: unobserved boundary strips.]** Overlay true interval, learned interval and sampled points on [0,1]. Shade the symmetric-difference regions whose lengths are errors. Use pattern and labels to distinguish true region from fitted region. The caption says that length equals probability only because this constructed input distribution is uniform.

A useful algorithm-specific bound comes from the two interior boundary strips, each of width ε/2, for 0<ε≤.4. If at least one sample lands in each strip, the tight interval misses at most ε of probability. Each strip is missed with probability(1−ε/2)ⁿ; union-bound the two events:

\[
\Pr(R(h_S)>\varepsilon)\le
\min\{1,2(1-\varepsilon/2)^n\}.
\]

This explains **how** observations near both edges control error. It is a sufficient event: a sample can achieve small error even if one chosen strip is empty. For this derivation we used the specified uniform interval world, not an arbitrary banknote distribution.

### Investigation: predict what another observation can change

Edit the sample coordinates while the target remains[.3,.7]. Record the fitted endpoints and population-error direction before revealing the new fit. Compare a new near-boundary positive with a new exterior negative. Then inspect the no-positive case. Every label is generated from the same target, so the realizability assumption is visible.

A separate label-edit mode lets you request positives at .2 and .8 with a negative at .5. Record whether any interval can fit. The correct result is “no consistent interval,” with the middle-point obstruction. The earlier realizable guarantee no longer applies because no member of the chosen class can fit these requested labels.

Now repeat independent training draws. At each n, the author ran 1,000 samples with random generator seed 41 and ε=.1:

| Training size n | Runs with true error>.1 | Mean true error | Two-strip failure bound |
|---:|---:|---:|---:|
|10|743/1,000|.177948|1, trivial|
|20|368/1,000|.092212|.716972|
|50|46/1,000|.039486|.153890|
|100|1/1,000|.019713|.011841|
|200|0/1,000|.009839|.0000701053|

The true risk for each fitted interval is calculated analytically, not estimated from a second large Monte Carlo test set. The failure fractions across training runs are simulation estimates. Zero observed failures at n=200 does not prove failure probability 0. The mathematical bound was derived independently; this experiment illustrates its setting and looseness rather than proving it. An individual draw's error need not decrease when a different, larger sample is drawn.

**[Inline figure: a distribution of fitted risks.]** Plot the retained run-risk summary and ε=.1 cutoff, with counts above the cutoff. Show the failure-frequency table and theoretical bound as different objects with different labels. A theoretical confidence δ, an empirical failure proportion, and a risk quantile must not share one unlabeled “confidence” axis.

### A real development learning curve

The Banknote Authentication data provides four wavelet-derived features and a binary class code. The included subset and source IDs match preceding lessons. Here all 320 designated training rows have labels; the 80 development rows are fixed. The final 80 test rows are **not evaluated by this program**. The task is predicting the source class code, without assigning an unverified genuine/forged meaning. [UCI data record and CC BY 4.0 license](https://archive.ics.uci.edu/dataset/267/banknote+authentication).

The program uses a fixed random permutation of training rows, seed 44, and nested prefixes of 20,40,80,160,320 examples. At each size it fits three predeclared procedures: standardized logistic regression(C=1), standardized RBF SVC(C=1, gamma='scale'), and a decision tree with maximum depth 5. Each scaler is fitted only on its current training prefix. All four features are used.

| n | Logistic regression train correct/n; dev correct/80 | RBF SVC train; dev | Depth 5 tree train; dev |
|---:|---|---|---|
|20|19/20;76/80|19/20;72/80|20/20;63/80|
|40|37/40;77/80|40/40;75/80|40/40;64/80|
|80|74/80;77/80|77/80;79/80|80/80;74/80|
|160|156/160;77/80|158/160;79/80|160/160;76/80|
|320|313/320;79/80|318/320;79/80|317/320;76/80|

These are actual observations from one nested sequence and one shared development set. They show how empirical performance changes for these procedures on these data. They do not measure each family's VC dimension, prove a theorem, or establish that validation accuracy must rise at every increment. The shared development observations are not independent replicate estimates and do not justify invented confidence bands.

**[Inline figure: measured development curves with their counts.]** Plot training and development error against n for each named procedure using the exact table. Put theoretical bound curves in a separate labeled panel, if shown at all; do not fit an “effective VC dimension” to these points and present it as a property of the class.

The practical question is what experiment to run next: collect more representative data, change the representation, investigate errors, or adjust model capacity. Development curves can inform that choice. Final performance assessment belongs after those choices, using an appropriate untouched evaluation set.

## 9. Use capacity without turning it into a model-selection shortcut

A smaller class can make estimation easier while excluding the rule the task needs. A larger class may reduce approximation error and increase the challenge of choosing among its members. Uniform convergence controls the estimation side; it does not tell us the best risk inside every competing family.

Suppose a simple family cannot achieve risk below .2 while a richer family contains a rule with risk .02. A wider bound for the richer family does not prove the simple family will predict better. Conversely, a very flexible family that perfectly fits twenty labels has not supplied evidence about unseen examples merely by achieving zero training error. Compare empirical evidence, task structure and a guarantee's actual assumptions together.

For a finite set of predeclared families, allocate an overall δ across their simultaneous bounds before choosing. More generally, a countable sequence can use positive weights π_j summing to 1 and confidence budgets δπ_j. This is the starting idea behind **structural risk minimization**: compare empirical error plus a justified class penalty, while accounting for the family search. Choosing a narrow family only after inspecting outcomes and then charging for that narrow family alone is not the same procedure.

The distinction also matters in transfer learning. A representation chosen using independent prior data can help make a target task simpler. If it is adapted using the target evaluation labels, the analysis must include that adaptation. “Pretrained” is not a mathematical exemption from selection or distribution assumptions.

A useful statement records the loss, sampling unit, hypothesis family, algorithm, sample size, confidence, theorem and numerical result. If assumptions such as independence or matching deployment distribution fail, the theorem does not directly apply to that deployment question. A bound that is valid but numerically larger than the trivial range is vacuous; an inapplicable bound is a different problem. PAC theory by itself is not a certification of clinical, financial or safety outcomes.

## 10. Deeper connections worth understanding

### Parameter count is not VC dimension

The class of all affine half-spaces in p-dimensional input space has VC dimension p+1; homogeneous separators through the origin form a different class. Axis-aligned boxes in p dimensions have VC dimension 2 p. A union of at most k intervals on the line has VC dimension 2 k: it can realize all patterns on 2 k points, while 2 k+1 alternating points starting and ending positive require k+1 separate positive runs.

These formulas concern the specified unrestricted mathematical classes. They do not license assigning a depth-limited tree a VC dimension from leaf count alone while ignoring input dimension, split family and representation. An RBF SVM's finite number of support vectors in one fit also is not the VC dimension of the unrestricted kernel hypothesis family. Norm/margin restrictions and algorithm-specific analyses need their own definitions.

Here is a surprising exact example. The family h_θ(x)=1[sin(θx)≥0] uses one real parameter but has infinite VC dimension when allowed the inputs and precision below. Take points 1,2,4,…,2ⁿ⁻¹. Given desired labels y₁,…,yₙ, form a binary fraction r whose first n bits are 1−y₁,…,1−yₙ and append bits 01. Set θ=2πr.

Multiplying r by 2ⁱ⁻¹ shifts its binary point so the i-th chosen bit becomes the first fractional bit. If that bit is 0, the fractional cycle lies strictly between 0 and 1/2, where sine is positive; if it is 1, it lies strictly between 1/2 and 1, where sine is negative. The appended bits avoid the zero boundaries. Thus one θ realizes every requested labeling on that n-point set.

For labels[1,0,1,1], r=17/64 and θ≈1.668971. The fractional cycles at x=[1,2,4,8] are[17/64,17/32,1/16,1/8], giving exactly[1,0,1,1]. The complete program checks all 16 four-point labelings using exact fractions.

**[Inline figure: bits becoming sine signs.]** Align the binary fraction, its shifts and the positive/negative semicircle. The point is a representation mechanism, not a wiggly curve chosen to look complicated. Increasing the number of labels requires increasing input range and representational precision. A fixed finite-bit parameter encoding contains only finitely many rules and does not inherit this unrestricted-real-parameter conclusion.

For ReLU networks, precise architecture-dependent bounds involve weights, layers and activation structure. Bartlett and colleagues give an upper bound O(WL log W) and related lower bounds for piecewise-linear networks. This does not make “all neural networks have VC dimension equal to parameter count” correct, nor establish that ordinary training always selects a provably small-capacity subclass. [Primary neural-network VC-dimension result](https://jmlr.org/papers/v20/17-612.html).

### Why other generalization tools exist

**Rademacher complexity** asks how well a fixed function family can correlate with independent random signs on sampled inputs. Its empirical form adapts to the sample geometry, which can provide information that a worst-case shattering count omits. It still requires a clearly defined family and loss; taking the single already-trained model as a new outcome-dependent class does not automatically justify a tiny penalty. The upcoming Rademacher Complexity & Generalization Bounds lesson develops this mechanism and its computation.

**Algorithmic stability** studies how changing one training observation changes the learned predictor's loss. This can exploit how an algorithm chooses among hypotheses rather than control the entire class uniformly. Different stability definitions and bounds have different assumptions; neither every use of SGD nor every regularizer automatically supplies a useful bound. [Bousquet and Elisseeff, stability definitions and generalization framework](https://jmlr.org/papers/volume2/bousquet02a/bousquet02a.pdf).

**PAC-Bayes** uses a distribution over predictors and penalizes its divergence from a suitable prior. The simplest statements require a prior independent of the training sample; data-dependent priors need an appropriate extension or separate data. A distribution over predictors is not automatically the deterministic trained model's guarantee. Dziugaite and Roy demonstrated nonvacuous bounds for particular deep stochastic networks; that is an existence result under a studied setup, not a claim that every modern model receives a tight bound. [Their primary paper](https://arxiv.org/abs/1703.11008).

For real-valued prediction, **pseudo-dimension** introduces a separate comparison threshold at each input and asks which above/below patterns functions can realize. **Fat-shattering** adds a positive separation scale around those thresholds. They extend the capacity question, but meaningful regression guarantees also need assumptions on output/loss ranges or tails. Substituting a dimension into a bounded binary formula does not control arbitrary unbounded squared losses.

### Computation is a separate question

Our interval enumeration requires only counting contiguous blocks. Trying every labeling of n points takes 2ⁿ requests; random trials can find witnesses but cannot generally certify nonexistence. For a fixed finite half-plane labeling, a linear feasibility problem provides a more principled computational check. Establishing a class-wide VC upper bound still requires an argument covering all point configurations.

General capacity computation depends on how a class is represented. Do not attach a single complexity-class label to every neural network, finite table or geometric family. The statistical theorem separates sample sufficiency from finding an ERM efficiently. A useful author or practitioner preserves that separation rather than promise a scalable generic “VC calculator.”

## 11. Practice: state the claim before calculating

### 1. Interpret the two tolerances

A procedure promises population error at most .08 with probability at least .99 over independent training draws. Explain what ε and δ mean, and whether a particular fitted model is promised 99% accuracy.

<details><summary>Hint</summary>One probability concerns new examples given a fitted rule; the other concerns which fitted rule training produces.</details>
<details><summary>Solution</summary>ε=.08 and δ=.01. At least 99% of training draws produce a rule whose population classification error is at most 8%, under the specified setting. The success target is at least 92% population accuracy, not 99%. The statement permits a small fraction of training draws to miss that target and does not identify them from their training scores.</details>

### 2. A family selected on shared validation examples

Twelve candidate predictors are fitted independently of 800 evaluation outcomes. For bounded 0–1 losses and δ=.02, compute the simultaneous two-sided radius. Must the twelve predictors' errors be independent? May we replace 12 with 1 after selecting the best observed rule?

<details><summary>Hint</summary>Use sqrt(ln(2 K/δ)/(2 n)). Distinguish independence of observations from dependence across rules.</details>
<details><summary>Solution</summary>The radius is sqrt(ln 1200/1600)≈.0665679955. The union bound does not require independence across predictors. Independent sampled observations are needed by the fixed-rule concentration argument. Replacing 12 with 1 after selecting on the same outcomes would omit the search; the simultaneous event already covers the selected member.</details>

### 3. Find the missing interval patterns

Four points have coordinates [.1,.3,.6,.9]. How many labelings can a single interval realize? List the impossible patterns. Does shifting every coordinate right by 2 change the answer for unrestricted intervals on the real line?

<details><summary>Hint</summary>Positive labels must form one contiguous run.</details>
<details><summary>Solution</summary>There are 1+4×5/2=11 realizable patterns. The five impossible ones are 0101,1001,1010,1011,1101. Each has separated positive runs. A common translation preserves point order and the available interval witnesses, so the count and feasibility of each ID-attached pattern are unchanged.</details>

### 4. A counterexample is not the whole VC proof

Three collinear plane points cannot be shattered by affine half-planes. Does that prove the class has VC dimension at most 2? Supply the missing reasoning for its actual dimension.

<details><summary>Hint</summary>The lower-bound part of VC dimension is existential over point sets.</details>
<details><summary>Solution</summary>No. A noncollinear triangle can be shattered, giving a lower bound 3. An upper bound requires ruling out every four-point configuration: an interior point versus the surrounding triangle, alternating vertices of a convex quadrilateral, and degenerate cases. The combination gives VC dimension 3. A failure on one triple says only that that triple is not shattered.</details>

### 5. A box proof without a false geometric claim

Show that axis-aligned rectangles in the plane shatter the four points (−1,0),(1,0),(0,−1),(0,1), but cannot shatter any set of five distinct points.

<details><summary>Hint</summary>For the upper bound choose representatives attaining minimum/maximum x and y. A fifth point lies in their bounding box; it need not lie in their convex hull.</details>
<details><summary>Solution</summary>For any nonempty chosen subset of the four cross-shaped points, its tight bounding rectangle includes exactly that subset; use the empty positive set for the empty subset. For any five-point set, choose at most four representatives for the x/y extrema. At least one remaining point lies inside or on their bounding box. Label the representatives positive and that remaining point negative. Any rectangle containing the representatives contains the bounding box and therefore the negative point. This impossible labeling proves the upper bound 4, including coordinate ties. The bounding-box argument does not require the point to lie in the convex hull of the chosen extrema.</details>

### 6. Agnostic risk versus absolute risk

A uniform event bounds all empirical/population gaps by .04. An exact ERM searches a class whose best population risk is .15. What does the standard ERM comparison guarantee? What changes if its empirical optimization is .01 suboptimal?

<details><summary>Hint</summary>The comparison crosses the empirical/population boundary twice.</details>
<details><summary>Solution</summary>Excess risk is at most 2×.04=.08, giving population risk at most .23. With optimization error .01, the ceiling becomes .24. Neither result promises absolute error .08 or .04, and neither says the ceiling equals the actual error.</details>

### 7. Repair a guarantee report

An author says: “The sufficient VC radius is 1.2, therefore the classifier cannot learn. A deeper model has twice as many parameters, so it requires exactly twice as much data. We verified the theorem because no simulated run violated it.” Rewrite the claims correctly.

<details><summary>Hint</summary>Separate vacuity, representation capacity, sufficient bounds and empirical evidence.</details>
<details><summary>Solution</summary>A radius 1.2 adds nothing beyond the 0–1 gap range, so that bound is numerically vacuous; it does not prove poor actual performance or unlearnability. Parameter count alone does not determine VC dimension or a task-specific sample ratio. A finite simulation checks the examples and can expose implementation errors, but cannot prove a distribution-free theorem or zero failure probability. State the exact inequality, family, assumptions and observed simulation counts separately.</details>

### 8. Change the interval experiment

Use target [.25,.75] and observed inputs [.05,.3,.4,.7,.95]. Determine the tight learned interval and exact uniform-input risk. Compare adding .26 with adding .99. Then explain why the same lengths need not equal risk under a nonuniform input distribution.

<details><summary>Hint</summary>Add lengths of missed target pieces. An exterior negative does not move the fitted positive extrema.</details>
<details><summary>Solution</summary>The learned interval is [.3,.7], missing .05 at each end, risk .1. Adding .26 changes it to [.26,.7], risk .01+.05=.06. Adding only .99 leaves [.3,.7] and risk .1 unchanged. For nonuniform X, integrate the probability mass of the disagreement regions rather than their geometric lengths; a short high-density region can matter more than a long low-density one.</details>

### 9. A meaningful learning-curve follow-up

The measured tree fits all 80 training labels but gets 74/80 development labels right. The RBF SVC gets 77/80 training and 79/80 development labels right. What can you conclude, and what should remain undecided?

<details><summary>Hint</summary>These counts come from one shared development set and different algorithms, not measured VC dimensions.</details>
<details><summary>Solution</summary>On this particular development set the SVC makes one error and the tree six. Perfect training fit did not imply the better observed development result. The comparison can motivate paired error inspection and further development experiments. It does not identify either class's VC dimension, establish a universal ranking, quantify independent-run uncertainty, or give an untouched final estimate after selecting a procedure on these same outcomes.</details>

## 12. References and another way to learn

For a visual lecture route, Caltech's Learning From Data course has [Lecture 6: Theory of Generalization](https://www.youtube.com/watch?v=6FWRijsmLtE) and [Lecture 7: The VC Dimension](https://www.youtube.com/watch?v=Dc0sr0kdBVI). The [official course page](https://work.caltech.edu/telecourse.html) identifies their topics and links the recordings. Use them after our finite-family argument, then reconstruct the interval and triangle witnesses yourself. They are extended lectures rather than substitutes for calculating the examples.

For concise mathematical notes, [Mehta's lectures 12–13](https://web.uvic.ca/~nmehta/ml_theory_fall2021/lecture12.pdf) connect concrete shattering examples, Sauer's lemma and an explicit uniform-convergence inequality. [Bartlett's lecture 4](https://www.stat.berkeley.edu/~bartlett/courses/2014spring-cs281bstat241b/lectures/04-notes.pdf) explains why selecting a rule requires more than a fixed-rule concentration statement. Read the notation alongside our candidate-row diagram.

[Blumer, Ehrenfeucht, Haussler and Warmuth's original paper](https://mwarmuth.bitbucket.io/pubs/J14.pdf) is the deeper source for the learnability characterization, sample bounds, geometric algorithms and the separation of statistical from computational learning. Its logarithm convention and regularity assumptions matter. The advanced links in section 10 provide focused next readings on neural-network capacity, stability and PAC-Bayes; they answer different generalization questions and should not be treated as interchangeable shortcuts.

The [data and calculation provenance](data-provenance.md) separates exact constructions, simulated training draws and measured development results. Continue in module order to **Calibration & Conformal Prediction**, where the question changes from aggregate classification error to reliable uncertainty statements about predictions. Rademacher Complexity & Generalization Bounds then returns to the capacity argument with a closer view of the sampled geometry.
