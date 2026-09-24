# Rademacher Complexity & Generalization Bounds

You tried several classifiers and kept the one that made the fewest training mistakes. Some of that improvement may be real structure. Some may come from having enough choices to accommodate accidental details of the sample. How can we put a number on that second possibility?

Try a deliberately impossible prediction task: keep the inputs, replace the answers with independent coin flips, and ask how well the best allowed rule can match them. Repeat with new flips. A class with many ways to adapt can obtain a large average match even though there is no signal to discover. **Rademacher complexity measures this freedom to correlate with random signs.** It helps bound the difference between performance on a training sample and performance on the population that generated it.

The preceding Calibration & Conformal Prediction lesson asked what probabilities and prediction sets promise. This lesson asks what selecting a predictor from a family costs statistically. PAC Learning & VC Dimension supplied the distinction between empirical risk, population risk and a guarantee over repeated samples; we will refresh those ideas locally, then replace worst-case counting with a calculation on the observed sample.

The core route is sections 1–8: calculate the quantity, understand its guarantee, derive useful bounds, and run a small real-data experiment. Section 9 offers optional routes into ensembles, modern neural bounds and alternative frameworks. By the end, you should be able to tell whether a purported “generalization certificate” uses the right class, loss, data and source of randomness—not merely substitute numbers into a formula.

## 1. A three-input noise-matching game

Put three inputs in increasing order: −1, 0, 1. A **positive threshold rule** predicts −1 below a cutoff and +1 at or above it. Whatever real cutoff we choose, its predictions on these three inputs must be one of four rows:

| Rule on the sample | x=−1 | x=0 | x=1 |
|---|---:|---:|---:|
| Cutoff above every input | −1 | −1 | −1 |
| Cutoff between 0 and 1 | −1 | −1 | +1 |
| Cutoff between −1 and 0 | −1 | +1 | +1 |
| Cutoff below every input | +1 | +1 | +1 |

The infinitely many cutoffs have only four distinct effects here. We must include both constant outcomes; restricting cutoffs to observed values would miss the all-negative rule.

Now flip three fair coins. Represent their results by signs σ=(+1,−1,+1). For a prediction row h, calculate

\[
\operatorname{match}(h,\sigma)=\frac{\sigma_1h(x_1)+\sigma_2h(x_2)+\sigma_3h(x_3)}3.
\]

Each agreement contributes +1 and each disagreement contributes −1. The result is 2×the fraction of agreements−1. For this sign pattern, the best threshold agrees twice and disagrees once, giving 1/3. No threshold can produce the alternating row (+1,−1,+1).

There are eight equally likely sign patterns. For the four patterns that already are threshold rows, the best match is 1. For the other four, it is 1/3. The average best match is therefore

\[
\frac{4(1)+4(1/3)}8=\frac23.
\]

That is the empirical Rademacher complexity of this threshold class on these three inputs, using the convention developed below.

**The order of operations matters.** For each new coin-flip pattern, choose its best rule; then average those best scores. If you fix one rule first and average its correlation over fair signs, the answer is zero. Averaging before taking the maximum erases the freedom to adapt that we are trying to measure.

**Investigation: build the best-response table.** Start with the four editable prediction rows and an unsolved sign pattern. Record the winning row and its score before revealing all correlations. Work through the sign patterns, then add a new allowed row and predict whether the average maximum changes. Duplicate an existing row as a separate experiment. Adding a genuinely new option may help; adding another name for the same option cannot.

### Compare classes without inventing a universal ranking

On these same three distinct inputs, exact enumeration gives:

| Allowed predictions | Number of distinct rows | Average best match |
|---|---:|---:|
| Only the fixed all-positive rule | 1 | 0 |
| Either constant sign | 2 | 1/2 |
| Positive thresholds | 4 | 2/3 |
| Thresholds in either orientation | 6 | 5/6 |
| Every possible sign row | 8 | 1 |

Every row in one class is available in the next, so the best achievable match cannot decrease. This is a justified comparison of nested classes on one sample. “Trees are always more complex than linear models” is not justified without specifying outputs, constraints and inputs.

The fixed all-positive predictor has complexity zero even if it predicts the real labels terribly. Complexity describes freedom to adapt, not whether the permitted predictions are appropriate. A broad class can also contain an excellent predictor: high capacity does not force the algorithm to select a bad one. These observations are why a useful risk bound needs a training-loss term as well as a complexity term.

## 2. The definition and the three things held fixed

Let F be a nonempty class of real-valued functions. On inputs S=(x₁,…,xₙ), it produces a set of vectors

\[
F|_S=\{(f(x_1),\ldots,f(x_n)):f\in F\}.
\]

The vertical bar means “restricted to this sample.” We can study these vectors without deciding how each function behaves elsewhere. A **Rademacher variable** is a fair sign: P(σᵢ=−1)=P(σᵢ=+1)=1/2. Draw the n signs independently. Our convention is

\[
\widehat{\mathfrak R}_S(F)
=\mathbb E_\sigma\left[\sup_{f\in F}\frac1n\sum_{i=1}^n\sigma_i f(x_i)\right].
\]

“Supremum” means the largest achievable value, or its limiting value if the class approaches it without attaining it. It becomes an ordinary maximum for our finite prediction table. The expectation is just the average over all 2ⁿ sign patterns, or an approximation to that average when enumeration is too large.

During this calculation, hold fixed the **sample, function class and output definition**. Only the auxiliary signs change. The true target labels are not needed for complexity of a predictor class. They are needed for complexity of a loss class, which is the object in the generalization theorem.

The expected complexity adds another average:

\[
\mathfrak R_n(F)=\mathbb E_{S\sim D^n}\widehat{\mathfrak R}_S(F).
\]

Here a fresh sample of n independent observations is drawn from D before the signs are drawn. Increasing the number of sign draws on one fixed sample estimates its empirical complexity more accurately; it does not average over new datasets.

### Why a convention must accompany a number

Some references put an absolute value inside the supremum, multiply by 2, or both. These are useful related definitions, but the constants and properties must follow the chosen version. The original Bartlett–Mendelson paper uses an absolute-value, 2/n normalization; the definition and theorem here follow [Mohri's lecture on infinite hypothesis sets](https://cs.nyu.edu/~mohri/mls/lecture_3.pdf).

For our singleton {all-positive}, the displayed definition gives zero. Inserting an absolute value gives E|σ₁+σ₂+σ₃|/3=1/2. That is not a harmless formatting change: it measures the enlarged symmetric set consisting of the rule and its negative. For classes already closed under negation, taking the absolute value does not change the per-draw supremum.

Our complexity is nonnegative whenever the expectations exist: the expected maximum is at least the expectation for any fixed member, which is zero. If outputs lie in [−1,1], it is at most 1. For unrestricted real-valued scores there is no universal upper limit of 1; multiplying every output by 10 multiplies complexity by 10.

Several properties follow directly from the noise game:

- Enlarging a class cannot decrease complexity.
- Duplicating a prediction vector changes nothing.
- Adding the same fixed vector to every member changes each sign-pattern score by a common amount whose expectation is zero, so complexity is unchanged.
- Multiplying all outputs by a scalar c multiplies complexity by |c|.
- Adding convex averages—weighted averages with nonnegative weights summing to 1—of existing prediction vectors changes nothing: their average correlation cannot exceed the largest original correlation.

Adding the same **fixed** output shift is different from allowing the learner to choose any intercept. An unbounded freely chosen intercept can make the supremum infinite. An intercept must be fixed, separately bounded, or included in a norm constraint on an augmented feature vector.

Nor must the empirical number decrease every time a new observation is appended. For f_w(x)=wx, |w|≤1, the sample [0] has complexity zero. The sample [0,1] has complexity 1/2. The usual rates concern expected behavior under stated sampling and boundedness conditions, not a monotonicity promise for every observed sample sequence.

## 3. Turn a noise calculation into a risk statement

A training observation is z=(x,y). A predictor f incurs loss ℓ(f(x),y). Its population risk is the expected loss on a new observation from the same population; its empirical risk is the average on the training sample:

\[
R(f)=\mathbb E_{(X,Y)\sim D}\ell(f(X),Y),\qquad
\widehat R_S(f)=\frac1n\sum_i\ell(f(x_i),y_i).
\]

Define the **loss class**

\[
G=\{g_f:(x,y)\mapsto\ell(f(x),y):f\in F\}.
\]

Each member is now a function whose output is an error cost, rather than a prediction score. This distinction is essential. An arbitrary bounded loss need not preserve the complexity of raw predictions. A tiny positive score and a tiny negative score can produce different hard classifications even when their numerical difference is arbitrarily small.

Assume G is fixed before drawing S, every g takes values in [0,1], and S consists of n independent and identically distributed (iid) observations from D. Under the usual measurability conditions, for any δ in (0,1), with probability at least 1−δ over S, simultaneously for every f∈F,

\[
R(f)\leq\widehat R_S(f)+2\widehat{\mathfrak R}_S(G)
+3\sqrt{\frac{\ln(2/\delta)}{2n}}.
\]

This is the empirical-complexity version. A separate expected-complexity version is

\[
R(f)\leq\widehat R_S(f)+2\mathfrak R_n(G)
+\sqrt{\frac{\ln(1/\delta)}{2n}}.
\]

The first adapts to the observed sample but pays for that additional randomness. The second involves an expectation over datasets that is generally unknown. They are not interchangeable plug-in formulas. [The two statements and their proof appear on slides 6–9 of Mohri's lecture](https://cs.nyu.edu/~mohri/mls/lecture_3.pdf).

The three terms answer different questions: how much loss was observed, how much selection freedom the class has on the sample, and how rare an unusually unrepresentative sample we are willing to allow. δ=.05 describes the probability of failure of the simultaneous statement over repeated datasets. It is not a 95% probability assigned to an individual prediction.

The word **simultaneously** lets us choose f using the training data. The chosen model remains one of the members for which the event holds. It does not let us inspect the data, invent an unrestricted new class containing only our chosen model, and claim its singleton complexity is zero. The class was supposed to be fixed before the sample. A representation learned on an independent sample may be conditioned on and frozen; one learned on the same observations requires an analysis covering that learning step.

### A useful exact shortcut for binary classification

If h(x)∈{−1,+1}, y∈{−1,+1}, and the loss is a classification mistake, then

\[
\ell(h(x),y)=\frac{1-yh(x)}2.
\]

The fixed 1/2 term disappears after averaging over signs. Multiplying each fair sign by the fixed −yᵢ produces another independent fair sign. Therefore

\[
\widehat{\mathfrak R}_S(\ell\circ H)=\tfrac12\widehat{\mathfrak R}_{X}(H).
\]

Our threshold example consequently has loss-class complexity 1/3 for any fixed binary target labels, compared with predictor complexity 2/3. This exact identity concerns binary-valued hypotheses. It does not say that the hard-thresholded predictions of a bounded-norm score class have half the complexity of its real scores.

If a bound's right side is 1.24 while the loss is in [0,1], the trivial upper bound 1 is better. The result is **vacuous** numerically. That may reveal loose inequalities, insufficient data, a class that is too broad, or a mismatch between what the theorem measures and the algorithm's useful structure. It does not prove that the model performs badly.

## 4. Why random signs appear in the proof

The proof is easier to follow as an information flow: unknown population average → independent comparison sample → random pair swaps → two noise-matching problems → a concentration statement. The ghost sample is a mathematical device, not additional data that an implementation must collect.

Write Φ(S)=sup_g(Eg−average_S g), the largest optimism of any permitted loss function on S. Introduce S′, another independent sample of the same size. For a fixed g, its average on S′ has expectation Eg. Allowing the choice of g to depend on the realized ghost sample can only increase the expected maximum, giving

\[
\mathbb E_S\Phi(S)
\leq\mathbb E_{S,S'}\sup_g\frac1n\sum_i\big[g(z'_i)-g(z_i)\big].
\]

Pair zᵢ with z′ᵢ. Since both are drawn independently from the same distribution, swapping their positions does not change the joint distribution. Independently choose to swap each pair using a fair sign. This gives the same expected supremum with each difference multiplied by σᵢ.

Now split one supremum into two:

\[
\sup_g\sum_i\sigma_i[g(z'_i)-g(z_i)]
\leq\sup_g\sum_i\sigma_i g(z'_i)
+\sup_g\sum_i(-\sigma_i)g(z_i).
\]

The inequality may be loose because the right side lets two different functions win. Each term, after division by n and averaging, is the expected Rademacher complexity. Thus the expected largest generalization gap is at most twice the expected complexity of G. The factor 2 first appears here; it should not be inserted again at the ghost-sample step.

Finally use **bounded differences**. Replacing one observation changes the average of a [0,1]-valued function by at most 1/n. Taking a supremum preserves that bound, so Φ(S) changes by at most 1/n. McDiarmid's inequality then gives a deviation above its expectation of at most √(ln(1/δ)/(2n)), except on an event of probability δ.

The empirical complexity itself also changes by at most 1/n when one observation changes. Apply a second concentration bound, allocating δ/2 to each event. Replacing its population expectation by its observed value contributes two copies of the deviation term because the complexity is multiplied by 2; concentrating Φ contributes the third. This produces the empirical theorem's coefficient 3.

**Visual explanation:** show a pair of sample columns, with each pair having a swap control, and link every sign to the corresponding subtraction. A second view draws one shared “choose g” box branching into two independently chosen boxes. The learner can see exactly where equality becomes an upper bound. Neither view should suggest that signs replace the actual training labels in the generalization theorem.

The assumptions now have visible jobs. If samples have different distributions, pair swapping need not preserve the distribution. If the loss has no range or tail control, the 1/n change argument fails. If the class changes with the sample, the same replacement proof does not automatically apply. Extensions exist, but require the corresponding theorem.

## 5. Compute or bound complexity using geometry

### A Euclidean norm ball has a closed-form best response

Consider real scores f_w(x)=wᵀx with ‖w‖₂≤B. The Euclidean norm is the vector's length, and the dot product measures alignment. For one sign vector,

\[
\sup_{\|w\|_2\leq B}\frac1n\sum_i\sigma_i w^Tx_i
=\frac1n\sup_{\|w\|_2\leq B}w^T\underbrace{\sum_i\sigma_i x_i}_{v}
=\frac{B}{n}\|v\|_2.
\]

The best vector points along v with length B: w*=Bv/‖v‖₂ if v≠0. If v=0, every allowed w gives zero. We have solved the inner optimization exactly; training a classifier on artificial labels would be unnecessary and would generally solve a different objective.

Taking the average over signs gives the exact empirical quantity B E‖Σσᵢxᵢ‖₂/n. For a convenient upper bound, an average length is at most the square root of the average squared length. Expand that square. Cross terms contain E[σᵢσⱼ]=0 for i≠j, while Eσᵢ²=1. Consequently,

\[
\widehat{\mathfrak R}_S(F_B)
\leq\frac{B}{n}\sqrt{\sum_i\|x_i\|_2^2}
\leq\frac{B\max_i\|x_i\|_2}{\sqrt n}.
\]

The first bound uses the observed feature energy; the second uses the largest observed row norm. If ‖X‖₂≤R holds throughout the population, averaging yields 𝔯ₙ(F_B)≤BR/√n. A maximum observed in a sample is not automatically a bound on every future observation.

### Same lengths, different directions

Take B=1 and two observations. If both are (1,0), the signed sum has length 2 for equal signs and zero for opposite signs. Complexity is (1+0+0+1)/4=.5 after dividing each sum by n=2.

If the observations are (1,0) and (0,1), every signed sum has length √2. Complexity is 1/√2≈.707107. Both samples have the same feature energy and the same energy upper bound .707107. The exact quantity captures directional geometry that this upper bound discards.

**Investigation: steer the signed sum.** Edit two input vectors and commit the direction and value of the maximizing w for a chosen sign pattern. Reveal the vector sum and its supporting point on the norm ball. Then predict the average over all patterns. Rotate both vectors together: their coordinates change but the answer does not. Turn a duplicate vector into a perpendicular one: the answer changes despite unchanged lengths. Double B: every best-response value doubles.

A plot of model coefficients alone can mislead. Scaling all inputs by a positive c and scaling the entire coefficient budget by 1/c leaves the set of possible predictions unchanged. Shrinking coefficients by changing feature units is not free capacity reduction.

### Kernels keep the same calculation in a feature space

A kernel gives inner products of feature vectors: Kᵢⱼ=k(xᵢ,xⱼ)=⟨φ(xᵢ),φ(xⱼ)⟩. For the feature-space norm ball, the same squared-length calculation yields

\[
\widehat{\mathfrak R}_S(F_B)
=\frac{B}{n}\mathbb E_\sigma\sqrt{\sigma^T K\sigma}
\leq\frac{B}{n}\sqrt{\operatorname{tr}K}.
\]

The trace is the sum of diagonal entries. It measures total feature-space squared length. We do not need to materialize an infinite feature vector. With an RBF kernel normalized so k(x,x)=1, this bound is B/√n. Infinite feature dimension therefore does not itself imply an infinite bounded-norm complexity.

For a two-point Gram matrix with diagonal 1 and off-diagonal similarity r, r=0 gives .707107, r=.9 gives .599143, and r=1 gives .5 when B=1. The diagonal-based upper bound stays .707107. Show these as exact four-sign calculations, with the similarity matrix beside the signed feature geometry; do not label them measured SVM accuracies.

The function space associated with the kernel is called a reproducing kernel Hilbert space, or RKHS. Its score family must include its constraints: allowing an unbounded norm has no such finite score bound. A free intercept is still a separate issue. A fitted kernel SVM's norm is obtained from its signed dual coefficients a by ‖w‖²=aᵀKa, not from the number of support vectors. Selecting a radius after seeing data also needs to be covered by the selection argument.

### Finite classes, VC theory and sparse coefficients

For M distinct prediction vectors a¹,…,aᴹ∈ℝⁿ with length at most A, **Massart's finite-class bound** is

\[
\widehat{\mathfrak R}_S(F)\leq\frac{A\sqrt{2\ln M}}n.
\]

For sign-valued functions A=√n, so the familiar expression is √(2 ln M/n). Sixteen fixed binary rules on 100 observations give an upper bound .235482. M=1 gives zero. Counting repeated copies separately only weakens this bound; the actual quantity is unchanged.

One proof explains why a logarithm appears. The exponential of a maximum is at most the sum of exponentials. For a fixed vector a, independence of signs and cosh(u)≤exp(u²/2) bound E exp(λσᵀa) by exp(λ²A²/2). Taking a logarithm gives an upper bound ln(M)/λ+λA²/2 on the expected maximum. Choose λ=√(2 ln M)/A and divide by n. Zero-radius and singleton cases follow directly rather than dividing by zero.

If a binary class has VC dimension 1≤d≤n, Sauer's lemma bounds the number of distinct sample predictions by Σⱼ₌₀ᵈ C(n,j)≤(en/d)ᵈ. Combining that count with Massart recovers a bound √(2d ln(en/d)/n). For d=0 the nonempty class has only one prediction pattern on every sample, so its complexity is zero. When d>n, use the bound 2ⁿ on the number of labelings instead; do not apply the simplified Sauer expression outside its range. Rademacher and VC analyses connect through restrictions to the sample. This particular derivation is not a claim that every Rademacher bound is strictly tighter than every VC result.

For an ℓ₁ coefficient budget ‖w‖₁≤B, the inner optimum instead selects the largest absolute coordinate of v=Σσᵢxᵢ:

\[
\sup_{\|w\|_1\leq B}w^Tv=B\|v\|_\infty.
\]

You can spend the whole absolute-weight budget on that coordinate with the useful sign. Treating the d feature columns and their negatives as 2d vectors, Massart gives

\[
\widehat{\mathfrak R}_S(F_{\ell_1,B})
\leq B\max_i\|x_i\|_\infty\sqrt{\frac{2\ln(2d)}n}.
\]

This is a useful connection to sparse high-dimensional models. The logarithmic dimension term comes with a specific ℓ₁ constraint and coordinate bound; it is not a promise that adding arbitrary features has no cost. [Understanding Machine Learning, chapter 26, develops the finite, Euclidean and ℓ₁ calculations](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf).

## 6. Connect score geometry to losses and margins

A function φ is L-Lipschitz if changing its input by a distance d changes its output by at most Ld. It has a bounded steepness, even when it has corners. For our no-absolute-value convention, coordinatewise L-Lipschitz maps satisfy the contraction inequality

\[
\widehat{\mathfrak R}_S(\ell\circ F)\leq L\widehat{\mathfrak R}_{X}(F),
\]

provided each map a↦ℓ(a,yᵢ) has that Lipschitz constant over the score range in question. Different observations may have different maps because their yᵢ differ. A fixed value at zero need not vanish for this convention; fixed offsets disappear under the sign expectation. Absolute-value versions may have different constants and centering requirements. [The coordinatewise proof is lemma 26.9 in Understanding Machine Learning](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf).

For intuition, condition on every sign except one. The remaining fair sign compares the two best responses obtained by adding and subtracting one coordinate. A Lipschitz map cannot separate those alternatives more than L times their original separation. Repeat this replacement coordinate by coordinate. This argument bounds a supremum over a class; it does not assert that the fitted model's losses individually improve.

| Map, with its input stated | Valid Lipschitz constant | Range issue for the [0,1] risk theorem |
|---|---:|---|
| Hinge margin loss max(0,1−m) | 1 | Unbounded as m→−∞; bound or truncate it before using that theorem |
| Logistic margin loss ln(1+exp(−m)) | 1 | Also unbounded on unrestricted margins |
| Sigmoid σ(s)=1/(1+exp(−s)) | 1/4 | A probability map in [0,1], not the logistic loss |
| Squared error (a−y)², a,y∈[−M,M] | 4M in a | Range [0,4M²]; normalize or use the corresponding range-dependent bound |
| Clipped margin loss below | 1/ρ | Always in [0,1] |

For logistic loss the derivative in m is −1/(1+exp(m)), whose magnitude approaches 1. For squared loss it is 2(a−y), which can have magnitude 4M. This is why borrowing the sigmoid's 1/4 constant for logistic loss, or calling squared loss “2-Lipschitz” without a domain, leads to incorrect numbers. A smaller upper bound for one loss also does not automatically make it the better learning objective for another task.

### Margins make a hard classification decision analyzable

For y∈{−1,+1}, the **margin** m=yf(x) is positive when the score has the correct sign, negative when it has the wrong sign, and zero at the boundary. A large positive margin means more score change is needed to reverse the decision. Pick a fixed positive threshold ρ and define

\[
\phi_\rho(m)=
\begin{cases}
1,&m\leq0,\\
1-m/\rho,&0<m<\rho,\\
0,&m\geq\rho.
\end{cases}
\]

This ramp upper-bounds a classification error, including either consistent tie decision at score zero. It also charges partially for correct predictions close to the boundary. Margins [−.2,.1,.4,1.2] with ρ=.5 produce losses [1,.8,.2,0], whose mean is .5. Only one of the four is an outright wrong sign, but two more are fragile at this chosen scale.

Contraction and the empirical theorem give, simultaneously for f in the fixed score family,

\[
P(Yf(X)\leq0)
\leq\frac1n\sum_i\phi_\rho(y_i f(x_i))
+\frac{2}{\rho}\widehat{\mathfrak R}_{X}(F)
+3\sqrt{\frac{\ln(2/\delta)}{2n}}.
\]

The event on the left counts all zero margins as errors, so it is an upper bound for a classifier with a fixed tie rule. For a norm ball, substitute its geometric complexity bound. Increasing ρ makes more training observations count as small-margin cases but decreases the 1/ρ complexity multiplier. The learner must balance both terms.

**Investigation: move the margin threshold.** Given editable margins, a coefficient budget and feature energy, predict the training ramp loss and the complexity addend before reveal. Move a margin across 0 or ρ and inspect its exact contribution. Then multiply every score, the budget B and ρ by the same positive factor. The predictions, ramp losses and B/ρ factor all stay unchanged. Multiplying scores alone cannot manufacture a more informative scale-normalized guarantee.

### Model selection needs its own accounting

If you predeclare K candidate combinations of norm budget and margin threshold, give each bound failure allowance δ/K. A union bound says all K hold together except on an event of probability at most δ. The confidence term becomes 3√(ln(2K/δ)/(2n)). Now choosing among these candidates does not invalidate their simultaneous bounds.

This is a simple form of **structural risk minimization**: compare empirical fit plus a complexity allowance across specified classes. For a countable collection, allocate failure budgets that sum to δ. Choosing among an unrestricted continuum after the fact requires an appropriate uniform theorem, discretization argument or independent selection procedure. A plain regularization sweep is useful validation; it is not itself a computed Rademacher certificate.

This clarifies the connection to a soft-margin SVM. Its usual objective is one-half the squared coefficient norm plus C times the sum of hinge losses. A smaller C makes margin violations less costly relative to a large norm; it does not enforce a smaller tolerated violation. Changing C changes that tradeoff, while the resulting norm and margin distribution determine quantities useful to a bound. The objective does not directly minimize the exact Rademacher complexity, and a large support-vector count alone does not diagnose either overfitting or excessive regularization. Compare the actual losses, coefficients, margins and held-out performance.

## 7. Estimate without turning a lower estimate into an upper guarantee

The complete [calculation program](complexity_calculations.py) enumerates all signs for the small tables and computes every quantity above. Save it locally and run it with Python and NumPy:

```bash
python -m pip install numpy
python complexity_calculations.py
```

Its summary prints singleton 0, two constants .5, thresholds .6666667, both orientations .8333333, all labels 1, and classification-loss complexity .3333333. The accompanying [checked results](checked-results.json) retain the actual prediction rows, all sign patterns, per-rule correlations, winners and exact-enumeration averages. Decimal display is rounded; the definitions, not typography, determine comparisons.

The finite-class function accepts a hypothesis-by-observation matrix. It calculates every dot product, takes the maximum across hypotheses for each sign pattern, then averages. It deliberately contains no `abs`. The threshold helper includes both extreme cutoffs, groups equal input values consistently and removes duplicate restrictions. The linear helper uses the exact supporting-vector solution rather than an approximate classifier fit.

For n observations, full sign enumeration has 2ⁿ rows. The program limits it to n≤12. At larger n, draw T independent sign patterns and average their exact best-response values. For a Euclidean norm ball each value lies in [0,Q], where

\[
Q=\frac{B}{n}\sum_i\|x_i\|_2.
\]

Hoeffding's inequality gives a useful upper correction. With probability at least 1−η over the sign draws, conditional on the fixed sample,

\[
\widehat{\mathfrak R}_S(F)
\leq\widehat{\mathfrak R}_{MC}
+Q\sqrt{\frac{\ln(1/\eta)}{2T}}.
\]

For the two duplicate unit vectors whose exact complexity is .5, one executed sequence with seed 131 produced:

| Sign draws T | Monte Carlo estimate | One-sided correction, η=.05 | Upper endpoint |
|---|---:|---:|---:|
| 16 | .437500 | .305968 | .743468 |
| 64 | .406250 | .152984 | .559234 |
| 256 | .531250 | .076492 | .607742 |
| 1,024 | .519531 | .038246 | .557777 |

More draws shrink the deterministic correction but do not make a particular estimate approach .5 monotonically. The displayed endpoint is a guarantee for each predeclared draw count separately. Inspecting many endpoints and reporting the most favorable one needs simultaneous or sequential accounting. A standard error describes the sign simulation's variability; it does not measure uncertainty about generalization to new observations.

If a risk theorem fails with probability δ and a separately computed Monte Carlo upper correction fails with probability η, a union bound permits total failure allowance δ+η. Omitting η silently turns an estimate into an asserted upper bound. When an analytic upper bound is already smaller than the corrected Monte Carlo result, use the analytic one.

There is another source of error: the inner optimization. A fitted neural network that achieves correlation .3 proves that the supremum is **at least** .3. Failure to optimize it further does not prove that the class cannot achieve .9. Repeating imperfect optimization many times does not repair the inequality direction. A certificate needs an exact supremum or a justified upper bound on it; an experiment with random labels remains useful evidence about the particular training procedure.

For dense linear scores, one sign draw costs O(nd); T draws cost O(Tnd). Process draws in bounded chunks for large matrices. A kernel Gram matrix requires O(n²) storage if materialized and O(n²) work per dense quadratic form, while the trace bound needs only diagonal values. Finite classes cost O(TnM) using the prediction table; threshold-specific cumulative sums can reduce repeated work after sorting. None of these formulas implies a hardware-independent runtime in seconds.

## 8. A real-data experiment: what a norm budget buys and costs

We now use the [UCI Banknote Authentication dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication), attributed to Volker Lohweg and available under CC BY 4.0. Its four inputs are wavelet-derived statistics; the target is a binary class code. The retained [480-row subset](banknote-subset.csv) preserves source row IDs. We use it to compare constrained predictors and inspect a bound calculation, without inferring unverified operational meaning from the class codes.

### Declare the information flow before fitting

The file is the same fixed subset used in neighboring lessons, but this experiment assigns four explicit roles:

| Rows in retained order | Role | What may use their information |
|---|---|---|
| 1–80 | Representation design | Fit four feature means and standard deviations |
| 81–320 | Predictor fit | Fit coefficients inside each declared norm ball; compute empirical losses and feature energy |
| 321–400 | Validation | Select the coefficient budget using the declared rule |
| 401–480 | Assessment | Report the selected procedure and the predeclared comparison table |

The CSV's historical `pool/dev/test` labels remain in the file; the program's explicit index allocation above governs this experiment. A source row belongs to one role only. Distinct row IDs do not ensure distinct inputs: this subset has four pairs with identical feature vectors. Two pairs cross fitting and validation, one crosses representation design and fitting, and one lies within fitting; none involves assessment. The validation result therefore includes two feature vectors already seen during fitting. We retain this declared finite-corpus study to inspect its mathematics, without presenting the validation score as performance on wholly new feature vectors. A new experiment intended to assess that question should group identical inputs before assigning roles, as in the AutoML lesson.

Freeze the representation fitted on the first 80 observations. For every later input, subtract each coordinate's fitted mean and divide by its fitted standard deviation, then divide by 3 and clip to [−1,1]. Append a constant 1 coordinate for the intercept. The resulting five-dimensional vector has norm at most √5 on every input. Its intercept weight is constrained together with the other weights. Convert class 0 to label −1 and class 1 to label +1. Clipping limits extreme feature influence and changes the available predictor family.

For each B in {.25,.5,1,2,4}, fit

\[
\min_{\|w\|_2\leq B}\frac1{240}\sum_i\ln(1+\exp(-y_iw^T\widetilde x_i)).
\]

The program uses a numerical constrained optimizer for this convex objective, with an analytic gradient, feasibility checks and a stationarity residual. A tiny inward projection repairs solver roundoff at the radius before reporting the actual returned model's objective. It optimizes logistic loss but **evaluates the bounded ramp loss** when applying the theorem. Contraction does not grant unbounded logistic loss the [0,1] theorem for free.

Choose the candidate with the fewest validation classification errors, breaking ties by validation log loss and then smaller B. The assessment labels are not used in that choice. Two margin thresholds, ρ=.5 and ρ=1, are predeclared for the theory comparison, giving K=10 budget/threshold pairs.

### Run the complete experiment

Download [bounded_norm_experiment.py](bounded_norm_experiment.py), [complexity_calculations.py](complexity_calculations.py), and [banknote-subset.csv](banknote-subset.csv) into one directory. Then run:

```bash
python -m pip install numpy scipy scikit-learn
python bounded_norm_experiment.py
```

The author run used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and scikit-learn 1.9.1. The [retained results](experiment-results.json) contain each fitted vector, role IDs, losses, margins, solver checks and bound components. Small numerical differences across versions are possible; the program checks the stated constraints rather than trusting formatted output.

| Budget B | Training errors / 240 | Validation errors / 80 | Assessment errors / 80 | Bound expression, ρ=1 |
|---|---:|---:|---:|---:|
| .25 | 52 | 16 | 11 | 1.293533 |
| .5 | 46 | 14 | 8 | 1.258458 |
| 1 | 36 | 10 | 6 | 1.198668 |
| 2 | 19 | 6 | 5 | 1.129848 |
| 4 | 8 | 4 | 2 | 1.198469 |

The validation rule selects B=4. Its assessment error is 2/80, compared with 29/80 for the majority class chosen from the fit set. Larger budgets improve measured classification performance throughout this particular sweep. We do not manufacture a turn upward in the error curve to make the regularization story look more familiar.

For B=2 and ρ=1, the calculation is completely inspectable:

\[
\underbrace{.474878}_{\text{training ramp}}
+\underbrace{2(2)(.0799496)}_{\text{complexity upper addend}}
+\underbrace{.335172}_{\text{confidence with }K=10,\ \delta=.05}
\approx1.129848.
\]

Here .0799496 is √Σ‖x̃ᵢ‖²/n for the 240 fit vectors. Every candidate bound expression exceeds 1, so this calculation gives no informative numerical certificate. That remains true even though the measured classifier is useful. The expression favors B=2 at ρ=1, whereas validation favors B=4; an upper bound is not a prediction of the assessment error.

The data were sampled without replacement from a fixed source corpus. That finite-corpus design and any acquisition dependence are different from the iid population model of our stated theorem. The table evaluates its expression as a theory diagnostic; it does not certify that a future banknote population is iid or that its risk has a particular bound. A formal deployment certificate would need a justified sampling model and the matching theorem. Independently of that issue, these expressions are already numerically vacuous.

The program also estimates the **unit-ball** empirical score complexity on the fixed fit vectors: .0720833 with 2,048 sign draws. Its conservative one-sided Monte Carlo endpoint is .105220, while the analytic feature-energy upper bound is .0799496. The analytic result is better here. This comparison concerns the same fixed geometry; neither number is the model's assessment error.

**Visual investigation:** align each budget's training-margin distribution with its three bound components and the measured validation counts. Changing ρ updates the ramp contributions and bound terms immediately. Display the raw sum above 1 and the trivial ceiling 1, rather than clipping away the reason the bound is uninformative. Apply the declared validation selection live. A separate action opens the frozen assessment report without asking for an answer; exposure remains recorded across reset.

## 9. Optional deeper routes

### Why averaging many learners can preserve score complexity

Suppose a boosting or voting score is a convex average of base predictions: f=Σⱼαⱼhⱼ, αⱼ≥0 and Σαⱼ=1. For any fixed noise signs, its correlation is a weighted average of the base correlations and cannot exceed their maximum. Since each base learner is itself an allowed average, the two suprema are equal. The convex hull has the same empirical score complexity as the base class.

This is an interesting reason to track margins rather than only the number of fitted components. A large ensemble can improve its margin distribution without automatically increasing the normalized score-class complexity. Thresholding the average into a hard label is discontinuous, so the same conclusion does not transfer directly to its 0/1 loss; the margin argument supplies the missing connection. Unnormalized or signed coefficient sums need their own budget.

### Covering numbers, chaining and local complexity

Counting all functions is wasteful when many make nearly identical predictions. A covering set approximates the prediction vectors within a chosen distance. Coarse covers identify major distinctions; finer covers account for smaller residual differences. **Chaining** combines bounds over multiple scales instead of paying the finest-scale count for every distinction. This can sharpen a simple one-scale Massart/Sauer analysis.

Global complexity also measures functions the learning procedure is unlikely to consider near a good solution. **Local Rademacher analysis** restricts attention to a region, often defined by an excess-loss or variance condition, and solves a relation between that region's radius and its complexity. Faster rates can emerge under additional noise or curvature conditions. Arbitrarily deleting poor-looking hypotheses after observing the sample is not a proof of localization. [Bartlett, Bousquet and Mendelson's local-complexity paper](https://arxiv.org/abs/math/0508275) is a deeper route for this distinction.

Gaussian complexity replaces fair signs with independent standard-normal multipliers. It supports related geometric and comparison arguments but is a different quantity with its own normalization and tail behavior. [Bartlett and Mendelson's structural-results paper](https://jmlr.org/papers/volume3/bartlett02a/bartlett02a.pdf) develops both measures and applications to kernels, networks and trees.

### Neural networks need the class and normalization to be explicit

A sufficiently rich neural architecture may fit many random labelings, yet a particular training procedure can still generalize on structured data. A coarse uniform bound over every permitted network may miss the constraints or preferences that matter. It is equally inaccurate to declare every finite network's VC dimension infinite or to declare every Rademacher-based neural bound useless.

Norm-based results specify network depth, activation properties and layer constraints. Neyshabur, Tomioka and Srebro study group and path norms, including cases where width dependence disappears and cases where it cannot. The bound is not universally “product of norms divided by √n” with all other factors dropped. [Their theorem 1 and its conditions](https://proceedings.mlr.press/v40/Neyshabur15.pdf) show why the precise norm and depth matter.

Later spectral-margin results combine products of layer operator norms with additional complexity factors and normalize by margins. Scaling successive layers or the final scores can make an unnormalized norm or margin look larger without the simple generalization interpretation suggested by that number alone. [Bartlett, Foster and Telgarsky's spectral-margin paper](https://papers.neurips.cc/paper/7204-spectrally-normalized-margin-bounds-for-neural-networks.pdf) makes this comparison on stated architectures and datasets. Its empirical associations do not prove that every increase in a trained network's norm causes worse future performance.

### Compare frameworks by what they control

| Framework | Object constrained | What must accompany an application |
|---|---|---|
| VC/growth analysis | Possible binary prediction patterns | Defined class, sampling assumptions and the applicable finite-sample result |
| Rademacher analysis | Average best noise correlation of score or loss classes | Exact convention, class constraints, loss range and justified computation |
| PAC-Bayes | A distribution Q over predictors relative to a prior P | A valid prior choice, KL term, empirical randomized loss and the specific theorem |
| Algorithmic stability | Change in loss when training data change | Algorithm, neighboring-data definition, expectation/high-probability distinction and smoothness/step assumptions |

PAC-Bayes lets the posterior Q depend on the sample under a bound that is uniform over posteriors. The KL divergence measures how Q redistributes probability relative to P; it is not merely the distance between their means. In the basic theorem the prior P must be independent of that sample; selecting a prior using the same observations needs explicit accounting. Dziugaite and Roy optimize a stochastic network posterior, use a prior mean fixed at random initialization, and account for choosing a prior variance from a discrete family. They also bound the error in estimating the randomized classifier's empirical loss. [Sections 3.1–3.3 describe those separate steps](https://arxiv.org/pdf/1703.11008). Merely moving a prior onto a trained network and declaring the KL small is not valid.

For stability, changing one observation is a perturbation of the **training procedure**, not a random-label fit. A uniform stability guarantee bounds the change in loss over all neighboring datasets and query examples; measuring one leave-one-out change is a diagnostic, not the supremum. Under convex, β-smooth, L-Lipschitz losses and suitable steps ηₜ≤2/β, Hardt, Recht and Singer obtain stability at most (2L²/n)Σₜηₜ. Their nonconvex result uses different conditions and dependence on steps. This is not a universal high-probability gap of 2ε for arbitrary SGD-trained networks. [The convex and nonconvex results are theorems 3.7 and 3.8](https://proceedings.mlr.press/v48/hardt16.pdf).

These tools answer related questions under different constraints. There is no universal ranking that makes one framework the best certificate for every transformer, kernel model or dataset. A meaningful certificate needs an actual theorem, valid information boundaries and computed quantities; a useful learning curve needs a sound evaluation protocol. Each can contribute without being mislabeled as the other.

## 10. Practice with changed problems

### 1. Two inputs, three thresholds

For inputs 2 and 5, the positive-threshold prediction rows are (−1,−1), (−1,+1), (+1,+1). Compute the exact empirical complexity over all four sign patterns. Would adding (+1,−1) change it?

<details><summary>Hint</summary>

For each pattern, take the largest of the three normalized dot products. Do not average each hypothesis's four values first.

</details>

<details><summary>Solution</summary>

The maxima for signs (−,−), (−,+), (+,−), (+,+) are 1,1,0,1. Their average is 3/4. Adding the missing prediction row makes every maximum 1, so complexity becomes 1. Duplicating any existing row would preserve 3/4.

</details>

### 2. A mysterious absolute value

A colleague reports complexity 1/2 for a single fixed all-positive rule on three observations. Your calculation gives zero. Explain how both numbers might have been obtained, and why one cannot be pasted into the other's theorem without checking conventions.

<details><summary>Hint</summary>

List the possible sums of three signs and ask whether the sign of that sum was retained.

</details>

<details><summary>Solution</summary>

The ordinary expected signed sum is zero. The absolute sum has average 1.5, giving .5 after division by 3. Taking the absolute value effectively permits the rule's negative as well. The definition has changed, so factors, centering properties and associated risk statements must be checked rather than mixed.

</details>

### 3. Geometry with a different scale

For x₁=(3,0), x₂=(0,4), and ‖w‖₂≤2, compute the exact score complexity and the feature-energy bound. Does an answer greater than 1 indicate a bug?

<details><summary>Hint</summary>

Every signed sum has the same length. Include B and the division by n.

</details>

<details><summary>Solution</summary>

Every signed sum is (±3,±4), length 5. The inner optimum is 2×5/2=5 for every sign pattern, so the average is 5. The energy bound is also 2√(9+16)/2=5. These are unrestricted real scores, not outputs constrained to [−1,1]; the unit ceiling does not apply. A risk theorem still requires its stated loss range or a valid margin transformation.

</details>

### 4. Compare a finite-class bound correctly

There are eight distinct sign-valued hypotheses on 200 observations. Compute Massart's bound. If 1,000 duplicate copies of those same rows are added to the file, what changes?

<details><summary>Hint</summary>

Use natural logarithms and the number of distinct prediction vectors. Separate a bound from the exact quantity.

</details>

<details><summary>Solution</summary>

The bound is √(2 ln 8/200)≈.144203. The exact complexity may be smaller. Duplicate rows leave every maximum and the exact complexity unchanged. Counting copies inside the logarithm would give a looser but unnecessary upper bound; deduplicating restores the original calculation.

</details>

### 5. Repair a loss comparison

Margins are [−.1,.2,.8] and ρ=.4. Calculate the ramp loss, then explain what is wrong with “logistic loss is 1/4-Lipschitz, therefore its classification guarantee is always four times better than hinge.”

<details><summary>Hint</summary>

Distinguish a sigmoid probability from log(1+exp(−m)), and distinguish a gap bound from a risk comparison.

</details>

<details><summary>Solution</summary>

Ramp losses are [1,.5,0], with mean .5. Logistic margin loss is 1-Lipschitz, while the sigmoid probability map is 1/4-Lipschitz. Hinge and logistic loss are unbounded over unrestricted margins; the [0,1] theorem cannot be applied unchanged. Even valid different loss bounds concern different empirical objectives, ranges and potentially fitted models, so a ratio of one term is not a universal classification comparison.

</details>

### 6. The apparently perfect singleton certificate

An agent memorizes a training set with a flexible model, defines F afterward as just that fitted function, computes complexity zero and claims that only the confidence term is needed. Identify the missing condition and give two valid ways to proceed.

<details><summary>Hint</summary>

Ask when F was fixed relative to the sample used for the theorem.

</details>

<details><summary>Solution</summary>

The class is sample-dependent, so the fixed-class proof does not apply. One option is to analyze a predeclared class covering all possible selected models, with its actual complexity. Another is to freeze the model and evaluate it on genuinely independent data with a fixed-predictor concentration result. An appropriate theorem for data-dependent classes is a further route, but cannot be assumed from the ordinary statement.

</details>

### 7. Diagnose the real experiment

The B=4 candidate has 2/80 assessment errors, but its ρ=1 bound expression is about 1.1985. Does this disprove the theorem? Should we choose B=2 merely because its expression is smaller? Would 10,000 more sign draws solve the problem?

<details><summary>Hint</summary>

Distinguish the population assumptions, the raw upper expression, the declared validation rule and which part uses Monte Carlo.

</details>

<details><summary>Solution</summary>

No. An upper expression above 1 is uninformative, not contradicted by a small measured error; the real fixed-corpus experiment also does not establish the iid deployment assumptions. The supplied selection rule chooses B=4 by validation, and the bound is not a test-error prediction. Its displayed value uses the analytic energy upper bound, so more sign draws would not alter that calculation. A refined complexity analysis could change a bound, while better validation or additional independent data could change the practical evidence; those are separate tasks.

</details>

### 8. Audit a random-label “upper bound”

A neural optimizer reaches mean signed correlation .35 over 200 random-label fits. Its author calls .35 an upper bound on the whole architecture's Rademacher complexity and inserts it into a 95% risk bound. Name the two distinct issues, even if the underlying sample were iid.

<details><summary>Hint</summary>

One issue is the direction of inner optimization error. The other is finite simulation uncertainty.

</details>

<details><summary>Solution</summary>

The achieved correlation is no larger than the supremum; an imperfect optimizer supplies lower evidence about capacity, not a certified upper value. Even if the supremum were exact, averaging only 200 random-sign draws would estimate empirical complexity and need a justified upper correction with its own failure allowance. The class, output bounds and loss transformation must also match the theorem. Increasing the number of imperfect fits addresses neither missing optimization certificate nor the loss-class distinction automatically.

</details>

## 11. Other ways to learn this, and the next connection

[Mohri's lecture slides](https://cs.nyu.edu/~mohri/mls/lecture_3.pdf) are the shortest route through the exact convention and the two high-probability bounds used here. Reconstruct the pair-swap proof alongside slides 6–9, then use the growth-function section to connect back to PAC and VC theory.

[Understanding Machine Learning, chapter 26](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf) provides a longer proof route: Rademacher calculus, contraction, Euclidean and ℓ₁ classes, and SVM guarantees. Its chapter 27 then introduces covering numbers. Keep each theorem's normalization and range assumptions together when comparing it with another source.

[Patrick Rebeschini's Oxford course](https://www.stats.ox.ac.uk/~rebeschi/teaching/AFoL/22/) links notes, slides and recordings for lecture 2, “Maximal Inequalities and Rademacher Complexity,” and lecture 3, “Rademacher Complexity. Examples.” The [lecture 3 notes](https://www.stats.ox.ac.uk/~rebeschi/teaching/AFoL/22/material/lecture03.pdf) are particularly useful for comparing ℓ₂ and ℓ₁ geometry; use the course's recordings as an alternative presentation where accessible.

For research depth, the Bartlett–Mendelson, local-complexity, norm-control, spectral-margin, PAC-Bayes and stability papers linked at the relevant mechanisms each change a specific part of the analysis. Start with the question you want answered instead of treating them as interchangeable certificates.

Next in this module is **ML Problem Formulation, Baselines & Data Leakage**. It returns from these guarantees to choosing the prediction unit, target, baseline and information boundaries of a real project. The connection is direct: a sophisticated complexity calculation cannot repair a mislabeled target, a leaked feature or a test population different from the one named in the claim.
