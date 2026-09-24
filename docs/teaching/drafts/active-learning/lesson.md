# Active Learning: Choosing Which Examples to Label

A classifier can have access to thousands of inputs and only a handful of trustworthy answers. A person might need to read each document, inspect each image, or run each experiment before its label becomes available. If you can afford only thirty more answers, which thirty should you request?

**Active learning lets a learning system help choose its next training examples.** It is useful when obtaining labels is a substantial cost and you can influence which examples receive them. Its success is measured by the quality achieved for a labeling budget, compared with a credible alternative such as random sampling. A complicated selection rule does not automatically save labels.

The previous lesson used unlabeled inputs to support learning and sometimes assigned provisional labels itself. Here the model asks an outside source for an answer. That source is called an **oracle**: a human annotator, a measurement procedure, or, in our reproducible experiment, a function that reveals a stored label only after an example has been selected. “Oracle” describes its role in the algorithm; it does not imply that a real annotator is infallible.

By the end, you should be able to choose and implement a query rule, explain what it is trying to learn, account for every label it uses, and evaluate its benefit without confusing the selected training data with a representative test sample.

The core route is sections 1–7. Section 8 provides deeper connections to gradients, Gaussian processes, and importance weighting. You can complete the practical experiment before taking that branch.

## 1. Separate the data you can inspect from the answers you can use

Suppose each input contains four measurements extracted from a banknote image. A model predicts one of two recorded class codes. You can inspect all four measurements for every available item; you cannot use an item's true class for training until you acquire it.

Keep four collections conceptually separate:

| Collection | What the selection algorithm may access | Purpose |
|---|---|---|
| Labeled training set, L | Inputs and acquired answers | Fit the current model |
| Unlabeled candidate pool, U | Inputs, identifiers, allowed metadata | Choose the next query |
| Development set | Inputs and labels through the evaluation procedure | Compare the predeclared strategies and budgets |
| Final test set | Held out until the method is selected | Assess the selected procedure |

**[Inline figure: an annotation queue.]** Draw a candidate as an input card with its answer concealed. Selection moves its identifier to a pending queue; annotation attaches an answer; accepting that answer moves the card to L; retraining produces the next model. Development and test cards occupy separate lanes. Only the selected card's answer crosses the oracle boundary.

The basic loop is concrete:

1. Fit a model using the currently acquired training labels.
2. Score eligible candidates using their inputs and the fitted model.
3. Select one or a batch, subject to budget and eligibility constraints.
4. Obtain and check those answers.
5. Add the accepted answers to training and refit.
6. Compare progress at the agreed budget checkpoints.

There must be a fit **after the final acquisition**. Otherwise a graph labeled “36 training labels” may actually show a model trained on only 35.

Three settings change the available actions. In **pool-based learning**, choose among already collected inputs. In **stream-based selective sampling**, decide whether to label an arriving input, possibly with a bounded buffer. With **membership queries**, propose an input to be measured or labeled. A proposed combination of experimental conditions may be meaningful; an arbitrary mixture of pixels may have no sensible human label. The setting determines what counts as an eligible question. [Settles, scenarios in the active-learning survey](https://burrsettles.com/pub/settles.activelearning.pdf).

## 2. A small question can eliminate many possible explanations

Start with a problem we can solve completely. A device turns on once an input x reaches a threshold t:

\[
h_t(x)=\begin{cases}0&x<t\\1&x\ge t.\end{cases}
\]

Assume the threshold is one of eight values: 0.5, 1.5, …, 7.5. We already know that x = −1 gives 0 and x = 9 gives 1. Both seed answers agree with every candidate threshold. The unlabeled query pool contains integers 0 through 8.

Each candidate threshold is a **hypothesis**, a possible rule. The **version space** is the collection still consistent with the answers obtained so far. It initially contains all eight hypotheses. These assumptions—finite candidate list, threshold rule, and error-free answers—are part of this example, not assumptions to silently impose on arbitrary datasets.

Consider asking about x = 4. Four thresholds predict 1 and four predict 0. Either answer removes half the candidates. Asking about x = 1 makes a one-versus-seven split. Asking about x = 0 removes none, because every remaining hypothesis predicts 0.

**[Inline figure: eight threshold rulers.]** Align eight horizontal number lines, one per threshold, with a vertical query line at x = 4. The prediction on each ruler is visible beside it. After revealing the answer, cross out contradicted rulers while retaining their identities in the history. Color accompanies explicit 0/1 text.

For a true threshold of 5.5, the balanced sequence is:

| Query | Revealed answer | Remaining thresholds |
|---|---:|---|
| 4 | 0 | 4.5, 5.5, 6.5, 7.5 |
| 6 | 1 | 4.5, 5.5 |
| 5 | 0 | 5.5 |

This uses three additional labels, or five including the two seed labels. Scanning x = 1, 2, 3, 4, 5, 6 from the left uses six additional labels for the same threshold. The gain comes from asking questions whose possible answers distinguish the remaining rules.

There is also a simple calculation behind “balanced.” If M hypotheses are equally plausible, a candidate divides them into groups of sizes m₀ and m₁. The expected number remaining after one answer is

\[
\frac{m_0}{M}m_0+\frac{m_1}{M}m_1
=\frac{m_0^2+m_1^2}{M}.
\]

For the initial eight candidates, the 4/4 split leaves exactly 4. The 1/7 split leaves 6.25 on average under the stated uniform distribution over hypotheses. The 0/8 split leaves 8. These are expectations over possible thresholds, not predictions of how a particular fixed threshold will answer.

### Investigation: spend the next label yourself

Choose an unused query and inspect the live survivor counts for answers 0 and 1. Acquire the oracle answer and inspect which rules were eliminated. Try the balanced route, the left-to-right route, and the legal but uninformative query x = 0.

Now change the candidate threshold list and choose a new query before revealing anything. A query that was balanced for eight evenly spaced hypotheses may be poor for an uneven set. If you enter a contradictory answer, the board should show an empty version space and identify the incompatible observations. That signals a failed assumption or annotation problem, rather than successful identification of a threshold.

The lesson transfers: seek answers that resolve consequential alternatives. The exact three-query result does not transfer to noisy labels, an unrestricted hypothesis family, or arbitrary available inputs. Sampling bias and confidently wrong regions are central difficulties in practical active learning. [Dasgupta, *Two Faces of Active Learning*](https://cseweb.ucsd.edu/~dasgupta/papers/twoface.pdf).

## 3. Uncertainty sampling: where does this model hesitate?

We usually cannot enumerate every plausible model. A cheaper starting point is to use one fitted model's class probabilities.

Let pₖ(x) be its estimated probability for class k. A probability vector such as [0.6, 0.3, 0.1] says how that model distributes its belief among the classes. It is not a list of observed frequencies for that individual example.

Three common scores ask slightly different questions:

| Rule | Calculation | Select |
|---|---|---|
| Least confidence | 1 − largest class probability | Largest score |
| Margin | Largest probability − second largest | Smallest gap |
| Predictive entropy | −Σₖ pₖ ln pₖ | Largest score |

The convention 0 ln 0 = 0 makes zero-probability classes contribute nothing. With natural logarithms, entropy is measured in **nats**. Its maximum for C classes is ln C, attained by equal probabilities. A binary entropy plot reaches ln 2 ≈ 0.693147, not 1 unless it explicitly changes its units or normalizes its scale.

For binary classification, all three rules rank candidates identically when they use the same probabilities and tie handling: they prefer p close to 0.5. For more than two classes, the rankings can differ.

**[Inline figure: three probability-strip candidates.]** A, B, and C each have a segmented strip whose widths are their class probabilities. Show the largest segment, the top-two gap, and entropy alongside the same input. This is a supplied-probability example, not a visualization of a fitted classifier.

| Candidate | Probabilities | Least-confidence score | Margin | Entropy, nats |
|---|---|---:|---:|---:|
| A | [0.50, 0.50, 0] | 0.50 | 0 | 0.693147 |
| B | [0.45, 0.30, 0.25] | 0.55 | 0.15 | 1.067094 |
| C | [0.60, 0.20, 0.20] | 0.40 | 0.40 | 0.950271 |

Margin selects A: its top two classes are tied. Least confidence and entropy select B: no class dominates, and probability is spread across all three classes. Choosing a score is choosing a proxy for usefulness. None of these numbers is the expected improvement in deployment accuracy.

A model may be uncertain because an input is genuinely difficult to distinguish using the available measurements, because its training data are sparse nearby, or because the model is unsuitable. Conversely, it can be confidently wrong in a region it has never properly learned. Asking only about the current boundary can neglect that region indefinitely.

The gradient of binary logistic loss makes one misconception especially visible. If the predicted class-1 probability is p, the observed label is y, and the feature vector is x, the gradient with respect to the weights is (p − y)x. A point predicted with p = 0.98 has a gradient magnitude of 0.02||x|| when y = 1, but 0.98||x|| when y = 0. High confidence does **not** guarantee that the true label would make a small update. Before querying, the model does not know which case it faces.

One way to avoid relying entirely on its confidence is to reserve some queries for random exploration or coverage. This is a policy to evaluate, not a universal best percentage. A rare class cannot be guaranteed representation by stratifying the pool on its unknown true labels; predicted-class quotas are only proxies.

## 4. Disagreement: what would different plausible models say?

Imagine two committees evaluating two inputs. For the first input, every member predicts [0.5, 0.5]. For the second, one member predicts [0.95, 0.05] and another predicts [0.05, 0.95]. Both committees average to [0.5, 0.5]. Their mean predictive entropy is the same, but their reasons differ.

In the first committee, the members share the ambiguity. In the second, they give confident opposing answers. An observed label could help distinguish the second committee's explanations. Shared ambiguity does not prove irreducible noise; all the models might share a misspecification.

**Query by committee** builds several models and selects inputs where their predictions disagree. They might come from bootstrap fits, posterior samples, or deliberately different representations. A committee consisting of duplicates provides no new disagreement information.

There are two useful forms of comparison. With M members and C classes, let vₖ be the number of members voting for class k. **Vote entropy** is −Σₖ(vₖ/M) ln(vₖ/M). It ignores how strongly each member prefers its winning class. Alternatively, retain each member's probability vector p⁽ᵐ⁾ and calculate

\[
\bar p=\frac1M\sum_m p^{(m)},\qquad
D=H(\bar p)-\frac1M\sum_m H(p^{(m)}).
\]

This finite-committee quantity is also the average KL divergence from each member to the mean distribution. It is nonnegative: a mixture is at least as entropic as the average component entropy. Compute zero-probability terms with their limiting values rather than dividing zero by zero.

**[Inline figure: where committee uncertainty comes from.]** Place each member's probability strip above the mean strip. Alongside it, compare H(mean) with the mean member entropy and label the difference. Show the shared-ambiguity and opposing-confidence cases together so the unchanged mean and changed disagreement are visible.

| Committee | H(mean), nats | Mean member entropy | Disagreement D |
|---|---:|---:|---:|
| [0.5, 0.5], [0.5, 0.5] | 0.693147 | 0.693147 | 0 |
| [0.95, 0.05], [0.05, 0.95] | 0.693147 | 0.198515 | 0.494632 |
| [0.95, 0.05], [0.95, 0.05] | 0.198515 | 0.198515 | 0 |

If the members represent samples from the parameter posterior p(θ | L), the corresponding expectation estimates **BALD**, Bayesian Active Learning by Disagreement:

\[
I(Y;\theta\mid x,L)
=H(Y\mid x,L)-\mathbb E_{\theta\mid L}H(Y\mid x,\theta).
\]

Read the formula in the order we just calculated it: uncertainty before selecting an explanation, minus the uncertainty that remains within an explanation. This is expected information about model parameters, under that Bayesian model; it is not automatically expected reduction in an application's error cost. [Houlsby and colleagues, §2 and the BALD identity](https://mlg.eng.cam.ac.uk/pub/pdf/HouHusGha11a.pdf).

### Investigation: keep the average fixed while changing the disagreement

Enter the two opposing probability vectors and inspect the live entropy decomposition. Replace them with two [0.5, 0.5] vectors. The average stays the same and the disagreement becomes zero. Next replace both with [0.95, 0.05]; both predictive entropy and disagreement are now small or zero, respectively.

Add a third member with an editable probability vector and compare the result before and after it joins. The display should show each member's distribution and its contribution, not only a changing scalar. These entered probabilities are mathematical fixtures; they are not the output of a secretly trained Bayesian model.

In the data experiment below, we train three bootstrapped logistic regressions for the committee. Their entropy difference is an ensemble query heuristic. We do not claim those bootstraps are exact Bayesian posterior draws.

For neural networks, one approximate Bayesian route uses repeated stochastic forward passes through a model trained with dropout. The passes must actually evaluate that trained model on the candidate inputs. Generating unrelated random probabilities is not Monte Carlo dropout. This method also incurs multiple forward passes, and its approximation assumptions matter. [Gal, Islam, and Ghahramani, deep Bayesian active learning](https://proceedings.mlr.press/v70/gal17a.html).

## 5. A batch should contain useful differences

Suppose twenty nearly identical inputs sit near the model's boundary. Selecting the twenty highest-entropy inputs may buy essentially the same answer twenty times. Batch selection introduces a second question: what will a candidate add **given the others already chosen**?

Consider four candidate points and one already labeled anchor at (0, 0). These coordinates are a deliberately constructed geometric example. Their probabilities are supplied separately; we have not fitted a model that generated them.

| Candidate | Coordinates | Supplied probability of class 1 |
|---|---|---:|
| A | (0, 1) | 0.50 |
| B | (0.1, 1) | 0.52 |
| C | (0, 4) | 0.70 |
| D | (4, 0) | 0.72 |

The two largest binary entropies belong to A and B, which almost coincide. A geometric alternative chooses a batch to cover the candidate space. If S is the new batch and L contains the existing labeled centers, define the **covering radius**

\[
r(S)=\max_{u\in U}\min_{z\in L\cup S}\|u-z\|_2.
\]

For each candidate, find its nearest selected or already labeled center. Then take the largest of those distances. A small radius means no candidate is geometrically far from a center. It says nothing by itself about whether nearby points share a label.

**[Inline figure: coverage before and after selection.]** Show the anchor as a square and candidates as named circles. Draw a line from each candidate to its closest center, with the longest line labeled as the radius. Use coordinates and a distance table to make the geometry checkable.

**Farthest-first selection** repeatedly chooses the eligible point farthest from the current centers, adds it, and updates all nearest-center distances. Here C and D initially tie at distance 4; using alphabetical order chooses C. D remains distance 4 from its nearest center, so choose D next. The remaining maximum distance is that of B to the original anchor: √1.01 ≈ 1.004988. Choosing A and B by entropy leaves the point D at distance 4 from the anchor, so its radius is 4.

Once selected, a point must be masked out of future selection even when distances are zero. Otherwise coincident inputs can make a careless implementation select the same identifier repeatedly.

### Investigation: design a two-item labeling batch

Select the two points you would actually buy and inspect their covering radius and nearest-center distances immediately, then compare with farthest-first. Move C from (0, 4) to (0, 2). The first geometric choice changes to D, then C; the final radius remains √1.01. The order changed without changing the final radius.

Now place every candidate on the anchor. Every radius is zero. The algorithm should still select two distinct identifiers, but it should not suggest that its geometric choice creates an advantage. Change the probability on a distant point as well: geometric coverage and uncertainty can disagree because they optimize different quantities.

Farthest-first has a factor-two approximation guarantee for this metric covering objective, with the existing centers fixed and the same batch budget. It is **not** a factor-two guarantee for classification error. Learned embeddings can make distance more meaningful, but they can also erase important distinctions; changing the representation changes the geometric problem. [Sener and Savarese, §4.3 and Algorithm 1](https://arxiv.org/pdf/1708.00489).

A practical design can combine uncertainty with diversity, or evaluate them as separate baselines. Mixing scores without an explicit objective and a development comparison merely moves the guesswork into a new formula.

## 6. Run a complete experiment on measured banknote features

We will reuse the real Banknote Authentication feature dataset from the previous lesson, while changing the question from provisional labeling to choosing which answers to request. The source contains four wavelet-derived measurements per image and a recorded class code. We retain codes 0 and 1 without assigning an unverified semantic meaning to them. The data are available under CC BY 4.0. [UCI dataset and attribution](https://archive.ics.uci.edu/dataset/267/banknote+authentication).

Download [the 480-row offline subset](banknote-subset.csv), [the complete experiment program](banknote-active-learning.py), and [the data and split explanation](data-provenance.md) into the same directory. The program runs locally using Python, NumPy, and scikit-learn. For an isolated environment:

On Windows PowerShell, these commands use the environment's interpreter directly:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install numpy==2.3.5 scikit-learn==1.9.1
.venv\Scripts\python.exe banknote-active-learning.py
```

On macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.3.5 scikit-learn==1.9.1
.venv/bin/python banknote-active-learning.py
```

The author execution used Python 3.12.14. The CSV and script are sufficient for the run; it does not fetch a dataset or contact a labeling service.

### The protocol comes before the ranking

The subset contains 320 pool rows, 80 development rows, and 80 final test rows, selected by a recorded random permutation of the source data. The row identifier lets us trace a query back to the source.

For each of five run seeds, start with three labeled pool examples of each class. This is an **oracle-balanced seed design**: it assumes those six labels are available. It is not a claim that six randomly obtained labels will discover both classes. All four strategies receive the same starting set within each run. These five initial sets differ from the single six-example starting set in the preceding semi-supervised lesson.

Fit a standard scaler on the 320 pool inputs, using no pool labels. Transform all splits with that scaler. This grants every strategy the same access to the unlabeled pool's feature distribution. It is a comparison of query strategies under shared preprocessing, not a strict baseline that forbids all use of unlabeled inputs.

Then acquire thirty additional labels, one at a time. Every evaluated predictor is logistic regression with C = 1 and max_iter = 500, fitted on all labels acquired so far. The four selection strategies are:

| Strategy | How it chooses the next eligible row |
|---|---|
| Random | Uniform random index among remaining candidates |
| Entropy | Highest entropy from the current logistic regression |
| Committee | Largest entropy difference from three stratified-bootstrap logistic regressions |
| Farthest | Greatest distance to the nearest labeled input in standardized four-dimensional feature space |

Each committee bootstrap samples within the currently observed classes and preserves their counts, so every member has both classes. The main predictor used to evaluate the committee strategy is still the logistic regression fitted on **all** observed labels. Evaluating a randomly selected committee member would change both the query policy and the evaluated predictor.

The program records the development count after fitting at 6, 7, …, 36 training labels. It records every query identifier, its revealed label, its score where applicable, and the main model's probabilities before acquisition. The test set does not choose a query, a seed, or a stopping point.

### Inspect the boundary in the program

The complete downloadable program contains the entropy calculation, all four strategies, bootstrap construction, split handling, trajectories, and final evaluation. Its central ordering is:

```python
known = np.flatnonzero(labeled >= 0)
model = LogisticRegression(C=1, max_iter=500).fit(x[known], labeled[known])
# Evaluate this fitted model at the current label count.
# If the budget is exhausted, save this final model and stop.
remaining = np.flatnonzero(labeled < 0)
probability = model.predict_proba(x[remaining])
# Select one local index from remaining using the declared strategy.
row = int(remaining[local])
label = int(truth[row])  # Oracle access occurs only after selection.
labeled[row] = label
```

This excerpt explains the control flow; run the complete linked file for the definitions of x, labeled, truth, and the selection branch. Notice that the oracle response changes the next fit, not the already computed score.

In this simulation, development labels are available for comparing the strategies. Those 80 labels—and the 80 test labels—are additional to the six-to-thirty-six fitting labels on the x-axis. A real budget report must account for them too. Reusing an existing audited benchmark is different from obtaining those labels for free.

### What actually happened

The following are the mean numbers correct out of the same 80 development examples, across five run seeds. Fractional counts arise because we average five integer counts.

| Acquired queries after the six seeds | 0 | 5 | 10 | 20 | 30 |
|---|---:|---:|---:|---:|---:|
| Random | 64.2 | 70.2 | 73.0 | 74.8 | 76.4 |
| Entropy | 64.2 | 74.0 | 78.2 | 78.2 | 79.2 |
| Committee | 64.2 | 73.6 | 75.2 | 77.4 | 78.2 |
| Farthest | 64.2 | 74.2 | 74.8 | 76.0 | 77.0 |

**[Inline figure: measured acquisition curves.]** Plot training-label counts 6 through 36 against development accuracy. Include the actual per-run trajectories on request, and the mean curve. Any min–max band describes these five runs, not a confidence interval. Do not smooth the lines into invented observations or force them to increase monotonically.

At the predeclared final budget, entropy has the largest mean development count. Its five final development counts are [79, 79, 79, 80, 79]. The protocol therefore selects entropy, with strategy-order tie handling specified in the code. Only then does the program evaluate that strategy's five final fitted models on the test set. The resulting counts are [79, 79, 79, 80, 79] out of 80, again averaging 79.2/80 = 99%.

The equality of these two five-count lists is an observed coincidence, not an assertion that the development and test rows are identical. The data receipt records distinct source-row sets. These repetitions vary initial training labels and query randomness on a fixed dataset; they do not supply five independent population samples.

This experiment supports a local conclusion: under this split, preprocessing, model, seed design, and budget, entropy gave the strongest final development result among these four candidates. It does not establish that entropy is universally best, that it will save a fixed multiple of annotation cost, or that the resulting system is ready to authenticate financial documents.

### A useful second run

Before changing the experiment, write a hypothesis: for example, “the farthest strategy is sensitive to the input scaling because it uses Euclidean distance.” On a fresh development exercise, compare the existing shared standardization with an explicitly declared alternative, retaining matched initial sets and budgets. Inspect which source-row identifiers change early in the query trace.

The supplied test result belongs to the supplied fixed protocol. If you use the test set to tune these modifications, it becomes development evidence. Keep a separate final evaluation for the procedure you eventually choose. A score increase and a sound evaluation design are different achievements.

## 7. Make the labeling workflow and evaluation agree

A practical query is more than an index in an array. It has an input version, annotation instructions, a cost, an answer status, and a model version that requested it.

**[Inline figure: an annotation record timeline.]** Display candidate → pending → answered → accepted or adjudication → included in a specific model fit. Show a separate abstained state. A pending item cannot be issued twice simply because the next training job started before its answer arrived.

An annotator should be able to say that an input is unreadable or the task is underspecified. Do not silently convert abstention into a negative class. Disagreements can reveal ambiguous definitions, inconsistent annotators, or a legitimately difficult input. Multiple labels or adjudication may help, but they cost time and should follow the problem's quality requirements.

Costs can differ substantially. Suppose two candidate experiments have estimated benefits of 0.12 and 0.18 in the same declared utility units, and costs of 1 and 4 minutes. Their benefit-per-minute ratios are 0.12 and 0.045. A ratio policy prefers the first, while a highest-benefit policy prefers the second. These are constructed utilities, not predictions of real accuracy. A ratio heuristic also does not solve every constrained batch-optimization problem.

For a sequence-labeling system such as the CRF studied earlier, an input might be an entire sentence. Querying its total sequence entropy can favor long sentences partly because they contain more labeling decisions. Alternatives include an explicitly normalized score, requesting a span, or estimating benefit per annotation minute. Each changes the action and the cost model. Preserve the actual annotation unit in the evaluation: “100 sentences” and “100 tokens” are different budgets.

A less obvious application is deciding which physical measurement to buy next. An input is a proposed experimental condition and its answer is the measured response. A useful question may cover a poorly understood region or reduce uncertainty where a downstream decision is sensitive. Asking about an impossible condition or repeatedly measuring an irrelevant extreme can be unhelpful even when a model reports high uncertainty. Experimental constraints belong in eligibility, before ranking.

### Why queried examples make a poor test set

Active learning deliberately changes which labels are collected. If it focuses on hard boundary cases, accuracy on those cases can be lower than accuracy on representative inputs. If it ignores a confidently wrong region, the acquired set can make it look better than it is.

For a transparent arithmetic example, take four equally likely deployment items whose losses for a **fixed** model are [0, 1, 0, 1]. Its population error is 2/4 = 0.5. A query rule that samples them with probabilities [0.1, 0.4, 0.1, 0.4] sees expected loss 0.8. That number answers “how hard are my chosen queries?” rather than “how often does this fixed model fail on a uniform deployment item?”

Keep an evaluation sample drawn for the intended use and split at the correct unit, such as a person, document, or time block. Estimate uncertainty in the performance difference when a decision requires it. A single small holdout can be noisy; repeatedly selecting policies on it also makes its best result optimistic.

### Choose a stopping rule before the last curve looks good

Possible operational stops include a fixed budget, an agreed development target, a marginal-gain rule over specified checkpoints, or a constraint on annotation turnaround. A flat acquisition score alone cannot certify that a model is accurate. All models can agree and still be wrong; a changed query rate in a stream can reflect drift, model updates, or a changing candidate mix.

Account for scoring and retraining as well as labels. Entropy requires model inference over the candidates. A committee multiplies this work by the number of members. Cached farthest-first can initialize nearest-center distances in O(|U||L|d), then update them in O(B|U|d) for a batch of B in d-dimensional space, using O(|U|) stored distances in addition to the data. The simple small-data program recomputes distances for clarity; a large pool warrants chunking, cached updates, and measured resource use.

A stream is not automatically an O(1)-memory system: model state, feature buffers, pending queries, acquired training data, and the retraining policy all matter. A large model may make retraining the dominant cost even when the next label is cheap. Compare methods under an operational budget that reflects the real task.

## 8. Deeper connections: define the benefit you want

### Expected change and expected future error are different objectives

If a candidate's possible labels are y, a model-change score can average the gradient magnitude under the model's current probabilities:

\[
G(x)=\sum_y p(y\mid x,L)\|\nabla_\theta \ell(\theta;x,y)\|.
\]

For binary logistic loss with a fixed feature vector, this becomes 2p(1 − p)||x||. Derive it by weighting the y = 1 gradient magnitude (1 − p)||x|| with probability p, and the y = 0 magnitude p||x|| with probability 1 − p. The result shows why feature scaling affects this score. It also shows the limitation: the expectation trusts the model's probabilities, while the true confidently wrong label can cause a much larger update than the model expects.

An expected-future-risk approach instead asks which label would most improve a target prediction problem:

\[
x^*=\arg\min_x\sum_y p(y\mid x,L)\widehat R\bigl(\operatorname{Fit}(L\cup\{(x,y)\})\bigr).
\]

For a constructed example, candidate A has label probabilities [0.5, 0.5] and estimated post-fit risks [0.10, 0.30], giving 0.20. Candidate B has probabilities [0.9, 0.1] and risks [0.15, 0.20], giving 0.155. This objective chooses B although A has greater predictive entropy. The risks in this table are declared hypothetical inputs to the calculation; obtaining credible risk estimates is the hard part.

Naively evaluating every candidate and every label requires many hypothetical refits and evaluations. Approximations can reduce the cost, but neither a low surrogate risk nor a large gradient guarantees a real improvement. The distinction among uncertainty, model change, and future risk is useful when selecting an acquisition objective; it is not a ladder with one universally best rung.

### Gaussian-process variance reduction measures a specific kind of uncertainty

The GP lesson showed that, with fixed kernel and noise parameters, conditioning changes a Gaussian covariance according to a formula that does not depend on the observed response value. Let c(u, x) be the current posterior covariance between a target u and a candidate x, and let σ² be the new measurement-noise variance. Then the reduction in latent variance at u from observing x is

\[
\Delta v(u;x)=\frac{c(u,x)^2}{c(x,x)+\sigma^2}.
\]

Summing this reduction over a specified set of target locations produces an acquisition score for reducing uncertainty there. It can differ from choosing the location with the largest own variance. If c(u, x) = 0, that measurement reduces no variance at u in this model. Larger observation noise reduces its effect. The result is conditional on the GP model and fixed hyperparameters; refitting them can change the covariance itself.

For example, with c(u, x) = 0.5, c(x, x) = 1, and σ² = 0.25, the variance reduction is 0.25/1.25 = 0.2 in squared response units. If the covariance is zero it is exactly zero. This is a model-based uncertainty calculation, not an empirical error guarantee.

### BADGE joins a gradient representation with batch diversity

For a classifier with feature representation z and softmax probabilities p, choose the model's predicted class ŷ = argmaxₖ pₖ. The cross-entropy gradient with respect to each last-layer class-weight vector is

\[
g_k=(p_k-\mathbf1[k=\hat y])z.
\]

Concatenate those class blocks to make one gradient embedding for the candidate. BADGE uses a diverse sampling procedure based on k-means++ distances in this space. It uses the **single predicted label** to form the gradient; it does not concatenate a separate gradient for every possible hypothetical label. [Ash and colleagues, §3 and Algorithm 1](https://arxiv.org/pdf/1906.03671).

For z = [2, −1] and p = [0.6, 0.3, 0.1], ŷ = 0. The three blocks are [−0.8, 0.4], [0.6, −0.3], and [0.2, −0.1], with total Euclidean norm √1.3 ≈ 1.140175. This makes both the representation and class uncertainty visible in the candidate geometry. The resulting diversity still depends on the current model. It cannot discover distinctions that the representation completely removes.

### Importance weighting can expose selection bias, with explicit assumptions

Return to the fixed four-item loss table from section 7. If an item i is selected with known probability qᵢ > 0 and the target distribution is uniform over four items, the single-query weighted loss is ℓᵢ/(4qᵢ). The possible weighted values are [0, 0.625, 0, 0.625]. Their expectation under the query distribution is

\[
\sum_iq_i\frac{\ell_i}{4q_i}=\frac14\sum_i\ell_i=0.5.
\]

The weighting corrects the expectation in this specific fixed-model sampling experiment. Zero selection probability prevents this correction for excluded items; tiny probabilities can create high variance. The identity alone does not justify evaluating an adaptively trained model on its own selected training data. Practical importance-weighted active-learning algorithms require additional control of sampling and estimation. [Beygelzimer, Dasgupta, and Langford, the IWAL sampling skeleton and guarantees](https://cseweb.ucsd.edu/~dasgupta/papers/iwal-icml.pdf).

### Calibration can matter without changing a query ranking

Probability calibration and acquisition ranking are different properties. For binary logits z, global temperature scaling gives p = sigmoid(z/T) with T > 0. Binary uncertainty is largest for the smallest |z/T|. Since dividing every absolute logit by the same positive T preserves its order, this adjustment preserves the entropy query ranking, including ties. It may change reported probabilities substantially while choosing the same next input. This simple result does not extend unchanged to every multiclass transformation. The upcoming calibration lesson develops the broader probability-quality question. [scikit-learn, temperature scaling definition](https://scikit-learn.org/stable/modules/calibration.html).

## 9. Practice: justify the next question and the resulting claim

Try each task before opening its hint or solution. The goal is to explain the selection and its assumptions, rather than recognize a strategy name.

### 1. An uneven threshold family

The remaining thresholds are [0.5, 1.5, 4.5, 7.5], with equal weights. Available queries are x = 1, 3, 8. Which minimizes the expected number of surviving hypotheses? What happens if you query x = 8?

<details><summary>Hint</summary>Count predictions on either side of each query. Use (m₀² + m₁²)/4.</details>
<details><summary>Solution</summary>At x = 1 the split is 3/1 and the expectation is 10/4 = 2.5. At x = 3 it is 2/2 and the expectation is 8/4 = 2. At x = 8 all four predict 1, so the expectation is 4 and an agreeing answer eliminates nothing. Query x = 3. An answer 0 at x = 8 contradicts every remaining hypothesis; investigate the model family and annotation instead of selecting a nonexistent survivor.</details>

### 2. Same binary uncertainty, different labels

Two candidates have probabilities 0.2 and 0.8 for class 1. Rank them using least confidence, margin, and entropy. What extra fact would break the tie in a useful way?

<details><summary>Hint</summary>Each rule is symmetric around 0.5. A tie in model uncertainty need not be a tie in cost or coverage.</details>
<details><summary>Solution</summary>Both have least-confidence score 0.2, margin 0.6, and entropy −0.2 ln 0.2 − 0.8 ln 0.8 ≈ 0.500402 nats. All rules tie. Eligibility, annotation cost, redundancy with the current batch, or a predeclared random tie breaker can determine the choice. The hidden true label cannot be used to break it before acquisition.</details>

### 3. Two kinds of disagreement

Compare committees [[1, 0], [0, 1]] and [[0.5, 0.5], [0.5, 0.5]]. Calculate predictive entropy and the entropy-difference score. Does the second prove that more labels cannot help?

<details><summary>Hint</summary>Deterministic member entropy is zero; a fair binary distribution has entropy ln 2.</details>
<details><summary>Solution</summary>Both means are [0.5, 0.5] with entropy ln 2. The first committee's average member entropy is zero, so its difference is ln 2. The second has mean member entropy ln 2 and difference zero. Zero committee disagreement says the current members agree; their shared model or representation can still be wrong. It does not prove labels are useless.</details>

### 4. Coincident candidates

There are three unlabeled items with distinct identifiers at the same coordinates as an existing center. The budget is two. What should farthest-first return, and what benefit can you infer?

<details><summary>Hint</summary>Separate distance from eligibility.</details>
<details><summary>Solution</summary>All nearest-center distances and the covering radius are zero. With index-order ties, select identifiers 0 and 1, masking each after selection. The geometry gives no reason to prefer those items or to expect an accuracy gain. Distinct identifiers can even carry different labels when features are incomplete.</details>

### 5. The confident mistake

A logistic model predicts p = 0.99 for an input with feature norm 2. Compute the gradient norm if y = 0 and if y = 1. Then compute its own expected gradient norm before observing y.

<details><summary>Hint</summary>Use |p − y| ||x||, then weight the two cases by the model's probabilities.</details>
<details><summary>Solution</summary>The norms are 1.98 for y = 0 and 0.02 for y = 1. The model's expectation is 0.01 × 1.98 + 0.99 × 0.02 = 0.0396. A large actual update can be assigned very little probability by a confidently wrong model. Expected-change selection inherits that probability model.</details>

### 6. A graph with the wrong horizontal axis

A run starts with eight labels, selects twenty more, and evaluates only before each acquisition. Its last plotted point is labeled “28 training labels.” Find the error and give a correct loop boundary.

<details><summary>Hint</summary>Count how many acquired labels have entered the last fit.</details>
<details><summary>Solution</summary>The last pre-acquisition fit uses eight plus nineteen = 27 labels. Fit and evaluate at acquisition counts 0 through 20 inclusive, stopping after the fit at 28 labels. Alternatively label the existing points 8 through 27, but that leaves the full-budget model unevaluated. Our program follows the first approach.</details>

### 7. A misleading performance report

A team reports 90% accuracy on examples its active learner requested, says it used only fifty labels, and selected its query rule after inspecting the final test scores of twelve rules. Identify three separate problems and propose a repair for each.

<details><summary>Hint</summary>Ask where the evaluation inputs came from, which labels the budget counts, and which decisions the test influenced.</details>
<details><summary>Solution</summary>The queried set is selected and may also be training data; use an independent sample appropriate to deployment. Count seed, acquired, development, test, repeated-annotation, and adjudication labels separately rather than implying only fifty answers were needed. Selecting among twelve rules uses the test as development; freeze the chosen procedure and obtain a fresh final evaluation. None of these repairs guarantees a higher score, but each makes the resulting claim interpretable.</details>

### 8. An independent acquisition experiment

Using the offline program, compare random and entropy acquisition with one predeclared change to the training-label budget, such as fifteen new labels. Keep paired initial sets, the same model, and the same development rows. Before running, predict whether you expect the difference to grow or shrink and explain your reason. With your environment's Python interpreter, run `banknote-active-learning.py --budget 15 --strategies random entropy --development-only`. This command retains the final refit and skips test prediction. Save every query identifier, the actual fit count, and the per-run development difference from the output JSON. Explain whether your result supports the prediction.

<details><summary>Hint</summary>The budget parameter changes the loop bound and final stopping condition together. A mean should be accompanied by the five individual differences. Keep `--development-only` while investigating; the result's final_test list should be empty.</details>
<details><summary>Solution and assessment criteria</summary>A complete answer states the budget and hypothesis before observing results; refits after the fifteenth acquisition; produces five paired development results; checks that each run uses 6 + 15 = 21 acquired training labels; and interprets variation without claiming that five shared-data repetitions are independent population samples. Either direction of the observed difference is acceptable. An explanation that merely announces a winner, omits a final refit, or changes the model for only one strategy does not establish the intended comparison. If further tuning follows, the development set supports that tuning and a separate final evaluation remains necessary.</details>

## 10. Other ways to learn and where to go next

For another explanation of the core idea, use the [CMU active-learning lecture video](https://www.youtube.com/watch?v=2BZhsEakEH8), linked from [the official 10-601 course schedule](https://www.cs.cmu.edu/~ninamf/courses/601sp15/lectures.shtml). It is an alternative lecture route on batch selection, selective sampling, and sampling bias. Work through the threshold exercise here after the lecture rather than treating watching as practice.

For the reasons active learning can fail as well as help, read [Dasgupta's *Two Faces of Active Learning*](https://cseweb.ucsd.edu/~dasgupta/papers/twoface.pdf), especially the sampling-bias discussion and version-space examples. Compare its arguments with the finite threshold board and with the separate evaluation lanes in our experiment.

For a wider taxonomy, [Settles' survey](https://burrsettles.com/pub/settles.activelearning.pdf) provides a map of query settings, acquisition objectives, and annotation constraints. It is a foundational reference rather than a catalogue of current package APIs. The [scikit-activeml documentation](https://scikit-activeml.github.io/latest/index.html) provides an optional software route with pool and stream examples; check the selected release's API and missing-label convention before adapting the experiment.

For the advanced mechanisms, return to [the BALD derivation](https://mlg.eng.cam.ac.uk/pub/pdf/HouHusGha11a.pdf), [the deep dropout application](https://proceedings.mlr.press/v70/gal17a.html), [the core-set objective](https://arxiv.org/pdf/1708.00489), [BADGE's last-layer gradient construction](https://arxiv.org/pdf/1906.03671), or [IWAL's treatment of selection probabilities](https://cseweb.ucsd.edu/~dasgupta/papers/iwal-icml.pdf). Each answers a different deeper question; reading one does not make its assumptions apply to every query policy.

The next topic in this module is **Evaluation Metrics**. Here, a count correct out of eighty was enough to make the acquisition protocol visible. Next we will ask which errors matter, how a threshold changes precision and recall, why ranking and probability quality differ, and how the evaluation measure should match the decision the model supports.
