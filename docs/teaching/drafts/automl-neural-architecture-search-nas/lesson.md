# AutoML & Neural Architecture Search

You have a dataset, several reasonable models, and an afternoon. A logistic model might work. A tree might need less preprocessing. A small neural network might capture a useful interaction. Each choice brings more choices: which columns to transform, how much to regularize, how large a model to fit, and when to stop trying alternatives.

**Automated machine learning, or AutoML, organizes and carries out some of these experiments.** You specify what counts as a valid candidate, how candidates will be judged, and how much work is allowed. A search procedure proposes candidates; an evaluator fits and scores them; a record of the results guides the next decision. Neural architecture search, or NAS, applies this idea to the structure of a neural network.

The most useful question is not “Can software find the best model?” It is “What decisions have I permitted it to make, and what evidence will justify its recommendation?” That question connects everything you have just learned about preprocessing, validation, regularization, feature selection, and imbalanced learning.

**First pass:** read §§1–5 in order, including the small neural-network bridge in §4. You will build a valid search space, understand two ways to spend its budget, and interpret a real experiment. Then use §6 to choose a practical workflow. §7 develops differentiable NAS and cheaper proxies; §8 provides optional library translations. The practice in §9 separates core readiness from deeper mathematics. No prior neural-network course is required for the first pass.

## 1. Begin with the experiment, before the optimizer

Imagine an inspection system that assigns class 0 or class 1 to a measured object. An AutoML system can optimize classification accuracy if that is the criterion you give it. It cannot infer that missing one kind of object costs twelve times as much as reviewing another. The [previous lesson on imbalanced learning](/learn/topic/imbalanced-learning-smote-cost-sensitive-learning) explained why that distinction can change both the model and the decision threshold.

Write the following contract before examining search results:

| Decision | A concrete answer for this lesson's study |
| --- | --- |
| Prediction task | Predict the original numeric class of a banknote feature vector. |
| Evidence available at prediction time | Four supplied numeric image descriptors. |
| Candidate choices | Eleven declared preprocessing/model configurations. |
| Selection criterion | Highest arithmetic mean of three validation-fold accuracies. |
| Separation rule | Identical feature vectors stay in the same data role and CV fold. |
| Work allowance | Three fits per candidate, then two declared refits. |
| Final comparison | Selected candidate versus a predeclared logistic baseline on a separate inspection partition. |
| Protected evidence | A further reserved partition receives no predictions. |

Accuracy is a declared educational objective here, not a claim that all real authentication mistakes have equal consequences. A deployment contract would also specify the consequences of errors, acceptable latency, expected data sources, and what happens when the input is unsuitable.

**[Figure F1: the experiment loop and the evidence boundary.]** Read the diagram from left to right: propose a configuration → fit its preprocessing and model on fitting rows → score on validation rows → update the search record. The validation score returns to the proposer. The inspection and reserved partitions sit outside that loop. A separate arrow reaches inspection only after selection is fixed.

### What is being optimized?

A *configuration* describes choices made outside ordinary model fitting: a model family, preprocessing steps, regularization strength, or hidden-layer widths. Model fitting then estimates the numerical parameters for that configuration. For logistic regression, the regularization strength belongs to the configuration; the fitted coefficients are model parameters.

Let \(\lambda\) describe a valid configuration and \(\mathcal A_\lambda\) its complete fitting procedure. A cross-validation objective to minimize is

\[
\widehat f(\lambda)=\frac1K\sum_{k=1}^{K}
 L\!\left(\mathcal A_\lambda(D_{-k}),D_k\right).
\]

Here \(D_{-k}\) contains the fitting rows for fold \(k\), \(D_k\) contains that fold's validation rows, and \(L\) measures prediction error. For accuracy, use \(L=1-\text{accuracy}\). The selected configuration minimizes this estimated error within the permitted space. This problem is often called *combined algorithm selection and hyperparameter optimization*, abbreviated **CASH**. It does not require any particular optimizer.

The hat on \(\widehat f\) matters. It is an estimate influenced by the dataset, split, random seed, and fitting procedure. Searching more configurations can improve the best observed validation score while exploiting more of that estimate's noise. A separate assessment evaluates the selected procedure; it is not another search surface. The [cross-validation lesson](/learn/topic/cross-validation-hyperparameter-tuning) develops nested evaluation when you need to assess the whole selection procedure across outer folds.

For unequal fold sizes, an unweighted mean of fold accuracies and accuracy pooled over all out-of-fold predictions differ slightly. Either may be a deliberate objective. State which one determines selection, and use the same rule for every candidate. Our experiment records both and selects by the mean of fold accuracies.

### The boundary contains preprocessing too

“I only used the validation labels at the end” is insufficient. Fitting a scaler, imputer, feature selector, or learned encoder on all rows can let validation information enter earlier. The fitting procedure \(\mathcal A_\lambda\) includes these operations. In each fold, fit the complete pipeline on that fold's fitting rows, then transform its validation rows using the fitted objects.

If observations share an object, patient, account, experimental run, or exact duplicated feature record, an ordinary row split may put closely related evidence on both sides. Group according to the deployment question. Chronological forecasting needs a time-respecting design. AutoML can execute such a design; selecting the right design remains part of the scientific problem.

## 2. A search space is a small language of valid experiments

Suppose a form asks for a model family, tree depth, neighbor count, and neural-network width. Most combinations make no sense. A logistic model has no tree depth. A one-layer network has no second-layer width. A good search space encodes those dependencies rather than evaluating an enormous table of meaningless combinations.

For our study, the grammar is:

```text
Choose one family
├─ Logistic regression
│  ├─ preprocessing: raw OR standard scaling
│  └─ C: 0.1 OR 1
├─ Decision tree
│  └─ maximum depth: 2 OR 5
├─ Nearest neighbors
│  ├─ preprocessing: standard scaling
│  └─ neighbors: 3 OR 9
└─ Small neural network
   ├─ preprocessing: standard scaling
   ├─ activation: tanh
   └─ hidden widths: (8) OR (16) OR (8, 8)
```

There are \(2\times2+2+2+3=11\) configurations. Multiplying every option count together would treat inactive choices as real and count configurations that do not exist.

**[Investigation I1: assemble a valid experiment.]** Select a family, then edit only its active settings. Before revealing the configuration count, predict what happens if you add a new tree depth. The total grows by one. Adding a new logistic \(C\) value grows the total by two because it pairs with both preprocessing choices. Changing a retained, inactive tree-depth value while logistic regression is selected changes neither the fitted recipe nor its active candidate count.

This grammar also documents a limitation. Our search cannot discover an SVM, a new feature extraction method, or a network with three hidden layers. Even exhaustive search is exhaustive only within its specified language.

### The sampling rule expresses a preference

For eleven inexpensive configurations, enumeration is transparent. Larger spaces often use random sampling. That still requires a distribution.

Suppose a positive regularization parameter spans \(10^{-4}\) to \(10^2\). Sampling its numeric value uniformly allocates almost all probability to the largest decades. Sampling

\[
u\sim\operatorname{Uniform}(-4,2),\qquad \lambda=10^u
\]

gives each decade equal probability. This is useful when ratios, rather than equal absolute increments, express comparable changes. It is a modeling choice, not a rule for every parameter: integer depths and probabilities often need other distributions.

The family selection rule matters too. Uniformly choosing one of four families and then a valid setting gives each family one quarter of the trials. Uniformly choosing one of the eleven configurations gives logistic regression four elevenths of the trials and a tree two elevenths. Neither is “unbiased” without specifying the intended reference measure.

### Learning where to try next

Random search ignores observed scores when proposing the next configuration. **Bayesian optimization** builds a predictive model, called a *surrogate*, of the objective and uses that model to choose an evaluation. Its uncertainty describes uncertainty about a candidate's objective under its assumptions; it is not the candidate classifier's probability for an individual example.

A common acquisition rule is **expected improvement**. If the best observed loss is \(b\) and a surrogate treats a candidate's unknown loss \(F\) as random, improvement is \(\max(b-F,0)\). The acquisition value is

\[
\operatorname{EI}=\mathbb E[\max(b-F,0)].
\]

Consider a constructed surrogate with \(b=0.4\):

| Candidate | Predicted mean loss | Predicted standard deviation | Expected improvement |
| --- | ---: | ---: | ---: |
| A | 0.35 | 0.02 | 0.050040 |
| B | 0.40 | 0.20 | 0.079788 |
| C | 0.50 | 0 | 0 |

B has a worse predicted mean than A but a larger expected improvement: its uncertain lower-loss possibilities compensate for its other possibilities under this acquisition rule. This is a specific exploration decision, not a promise that B will actually perform better.

**[Investigation I2: which uncertainty is worth an experiment?]** Edit the predicted distributions, record a choice, and then reveal the shaded improvement area to the left of the incumbent loss. Move B's uncertainty toward zero and watch its acquisition value fall. Add the same offset to every predicted mean and the incumbent: expected improvements remain unchanged because improvement depends on differences.

For a Gaussian surrogate prediction \(F\sim\mathcal N(\mu,\sigma^2)\), write \(z=(b-\mu)/\sigma\). Let \(\Phi(z)\) be the probability that a standard normal variable is at most \(z\), and \(\phi(z)=e^{-z^2/2}/\sqrt{2\pi}\) its density. Integrating \((b-f)\) over the part of the density below \(b\) gives

\[
\operatorname{EI}=(b-\mu)\Phi(z)+\sigma\phi(z),\qquad \sigma>0.
\]

The first term measures mean advantage weighted by the probability of improvement; the second accounts for uncertainty in the lower tail. When \(\sigma=0\), use \(\max(b-\mu,0)\). Real classification losses are bounded, so a Gaussian predictive approximation can place some probability outside their physical range. The example exposes the decision calculation without asserting that this approximation is always appropriate.

Gaussian processes are one possible surrogate, developed [later in this module](/learn/topic/gaussian-processes-gp). Tree-based surrogates and density-estimation approaches also support search. Conditional spaces require an appropriate representation of inactive parameters; Gaussian processes are not inherently forbidden from such spaces. Random search, evolutionary methods, and sequential surrogates should be compared using the same space, resources, and evaluation protocol. An optimizer cannot rescue a search space that excludes useful solutions.

## 3. Decide how much evidence to buy for each candidate

Evaluating a configuration is often more expensive than proposing it. For \(n\) candidates and \(K\) folds, ordinary CV requires \(nK\) estimator fits, before final refits. Fits can have very different costs. A count of candidates is not a wall-clock budget; concurrent workers also change wall time without eliminating computational work.

An evaluator can use a cheaper approximation, or **lower fidelity**, such as fewer training epochs, fewer fitting examples, or a smaller input resolution. Its usefulness depends on whether it preserves enough information about the expensive target evaluation.

### Successive halving: spend more on survivors

Here is a fully specified constructed example. Nine candidates begin with one resource unit each. Keep the best third, increase their resource to three units, keep the best third again, and increase the survivor to nine units. Lower loss is better.

| Candidate | Loss at 1 unit | Loss at 3 units | Loss at 9 units |
| --- | ---: | ---: | ---: |
| A | 0.10 | 0.10 | 0.10 |
| B | 0.12 | 0.09 | 0.08 |
| C | 0.13 | 0.08 | 0.07 |
| D | 0.14 | 0.07 | 0.02 |
| E | 0.20 | 0.15 | 0.09 |
| F | 0.25 | 0.20 | 0.10 |
| G | 0.30 | 0.25 | 0.20 |
| H | 0.35 | 0.30 | 0.25 |
| I | 0.40 | 0.30 | 0.20 |

At the first rung, A, B, and C survive. At the second, C survives. The selected candidate finishes at 0.07. D would have reached 0.02, but its slow start eliminated it before that evidence was purchased.

If each rung restarts training, work is \(9(1)+3(3)+1(9)=27\) units. If training can genuinely resume from the retained state, incremental work is \(9(1)+3(3-1)+1(9-3)=21\). Fitting all nine at full resource would cost 81 units under this equal-unit-cost construction. Resume is a property of the actual training procedure and state, not something any estimator's similarly named option guarantees.

**[Investigation I3: rescue a slow starter—or eliminate it.]** Reveal the rung decisions before seeing unpurchased final outcomes. Then inspect the constructed full curves. Edit D's first-rung loss from 0.14 to 0.09 and predict the survivor path. D now survives both cuts and wins at full resource. Adding the same constant to all losses changes their labels but leaves rankings, survivors, and resource use unchanged.

Hyperband repeats this resource-allocation idea across several *brackets*. Some brackets start many candidates cheaply; others start fewer candidates with more evidence before elimination. This hedges the breadth-versus-depth choice. It does not make every early ranking reliable. The [original algorithm and analysis](https://www.jmlr.org/papers/volume18/16-558/16-558.pdf) and [Kevin Jamieson's worked bracket table](https://homes.cs.washington.edu/~jamieson/hyperband.html) make those two loops explicit.

### A cheaper measurement can change the question

A model that is best after one epoch need not be best after fifty. A small-data winner may not be best with all fitting data. A reduced image resolution can remove the very feature that distinguishes two architectures. Weight sharing, introduced in §7, changes how candidate weights are obtained rather than merely shortening an otherwise identical fit.

Before adopting a proxy, compare it with the intended evaluation on a declared set of candidates. Inspect ranking changes and the actual promising region, not just one overall correlation. Save sufficient final evaluations to detect slow starters or proxy-specific advantages. There is no universal correlation threshold or fixed fraction of budget that makes every proxy safe.

**Budget is an evidence policy.** It should state candidate counts or stopping rules, resource per evaluation, repeated seeds where justified, concurrency, and what work is reserved for evaluating selected candidates. Record failed and interrupted trials as well as successful ones. A failed fit is useful diagnostic evidence; it is not a secretly excellent score or a reason to hide that part of the search space.

## 4. Neural architecture search, with the neural part explained locally

A neural network composes parameterized transformations. Start with a single unit. It multiplies input values by learned weights, adds a learned bias, and applies an activation function:

\[
h=\tanh(w_1x_1+\cdots+w_dx_d+b).
\]

The function \(\tanh\) bends and bounds the weighted sum. A layer contains several such units. The next layer receives their outputs. In binary classification, a final sigmoid transforms a final weighted sum into a number between zero and one; a classification threshold converts that number into a class decision. The trained number is not automatically calibrated just because it lies in that interval.

Without nonlinear activations, composing affine layers would still give an affine transformation. A nonlinear hidden layer allows the model to represent interactions and curved boundaries that a single linear score cannot. The later deep-learning module develops training and representation in depth; this is enough machinery to understand what our small search changes.

### Weights and architecture are different decisions

An architecture says which layers and connections exist and how large their intermediate representations are. Training estimates the weights within that structure. Choosing eight hidden units instead of sixteen changes the structure; changing one fitted connection weight does not.

For four input features, a hidden layer of width \(h\), and one output:

\[
\text{parameter count}=(4+1)h+(h+1)=6h+1.
\]

The added ones account for bias parameters. Width eight gives 49 parameters; width sixteen gives 97. Two hidden layers of widths eight and eight give

\[
(4+1)8+(8+1)8+(8+1)1=121.
\]

**[Figure F2: open up the three architectures.]** Follow an input row through shapes \(4\to8\to1\), \(4\to16\to1\), and \(4\to8\to8\to1\). Each connection block displays its matrix shape and bias count. Selecting a block reveals the corresponding multiplication. No mysterious “network size” number is needed.

Searching these three structures is a small, legitimate NAS experiment. It is deliberately restricted: the activation, fitting algorithm, and regularization are fixed. A larger NAS space might include convolutional operations, skip connections, or repeated cells, but each additional choice requires a valid shape rule and an evaluation budget.

### A graph must also be executable

Think of a more general architecture as a directed acyclic computation graph. A node stores an intermediate tensor; an edge applies an operation. Two paths can be added only if their output shapes agree, or if an explicit projection makes them agree. Concatenation joins selected dimensions and changes the downstream shape. An identity edge preserves its input; a zero edge contributes a zero tensor of the required shape.

NAS therefore has three separable components:

| Component | The question it answers | Our small study |
| --- | --- | --- |
| Search space | Which executable architectures are allowed? | Three fixed hidden-width patterns. |
| Search strategy | Which candidate is evaluated next? | All three are included in a declared eleven-candidate study. |
| Performance estimation | How is a candidate judged? | Three group-respecting folds, independently fitted weights. |

Swapping an evolutionary proposer for a Bayesian proposer changes the second component. Using a shared-weight supernetwork changes the third. Claims about “a better NAS method” need to say which components changed.

## 5. A real search you can inspect end to end

The [UCI Banknote Authentication dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication), attributed to Volker Lohweg, provides four numerical image descriptors and a binary class for 1,372 records. The descriptors are variance, skewness, kurtosis (spelled “curtosis” in the source), and entropy of the supplied image-derived measurements. We retain the original class labels 0 and 1; the inspected documentation does not establish which numeric label means genuine or forged.

The source contains **1,348 unique feature vectors**. Eleven vectors occur repeatedly, accounting for 24 additional rows. All identical vectors have matching class labels. We retain every row and keep identical vectors together during splitting. That prevents exact-feature copies from appearing on opposite sides of a boundary. It does not establish physical banknote identity: the source provides no specimen or capture-session identifiers with which to evaluate independence at that level.

The fixed group splits produce:

| Role | Unique feature groups | Rows | Class-1 rows | Use |
| --- | ---: | ---: | ---: | --- |
| Development | 900 | 919 | 407 | Three-fold selection and the final refits. |
| Inspection | 200 | 205 | 91 | Compare two fixed fitted procedures once. |
| Reserved | 248 | 248 | 112 | No predictions or model decisions. |

These are group-stratified partitions: stratification uses one class label per unique feature group. Because group sizes vary, row-level class proportions and fold sizes need not be identical. Keeping duplicate groups intact takes priority over forcing exact row counts.

### Run the complete bounded study

Save the openly licensed source file as `banknote-data.csv` beside the program below. It contains five comma-separated values per row and no header. The [dataset download](https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip) contains `data_banknote_authentication.txt`; renaming that file changes no data. A copy with provenance accompanies this lesson packet.

Use Python with NumPy, scikit-learn, and threadpoolctl, for example in an isolated environment:

```bash
python -m pip install numpy scikit-learn threadpoolctl
python banknote_search.py
```

The author calculation used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, and scikit-learn 1.9.1. Version changes can alter optimizer stopping behavior or floating-point details. The study below is a complete teaching program; its evidence was calculated with the equivalent retained author program. The exact displayed script still receives the normal implementation-phase execution check.

```python
# banknote_search.py
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits


def load_roles():
    data = np.loadtxt(Path(__file__).with_name("banknote-data.csv"), delimiter=",")
    x, y = data[:, :4], data[:, 4].astype(int)
    unique, first, group, counts = np.unique(
        x, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    for g in np.flatnonzero(counts > 1):
        if len(np.unique(y[group == g])) != 1:
            raise ValueError("Inspect conflicting labels within an identical-feature group.")
    group_y = y[first]
    development_groups, rest = train_test_split(
        np.arange(len(unique)), train_size=900, stratify=group_y, random_state=71
    )
    inspection_groups, reserve_groups = train_test_split(
        rest, train_size=200, stratify=group_y[rest], random_state=72
    )

    def rows(group_ids):
        return np.flatnonzero(np.isin(group, group_ids))

    development, inspection, reserve = map(
        rows, (development_groups, inspection_groups, reserve_groups)
    )
    splitter = StratifiedKFold(3, shuffle=True, random_state=73)
    folds = [
        (rows(development_groups[a]), rows(development_groups[b]))
        for a, b in splitter.split(development_groups, group_y[development_groups])
    ]
    return x, y, development, inspection, reserve, folds


def candidate_models():
    choices = []
    for scale in (False, True):
        for c in (0.1, 1.0):
            model = LogisticRegression(
                C=c, solver="lbfgs", max_iter=1000, tol=1e-8
            )
            if scale:
                model = make_pipeline(StandardScaler(), model)
            name = f"logistic-{'standard' if scale else 'raw'}-c{c:g}"
            choices.append((name, model))
    for depth in (2, 5):
        choices.append((f"tree-depth{depth}", DecisionTreeClassifier(
            max_depth=depth, random_state=74
        )))
    for k in (3, 9):
        choices.append((f"neighbors-standard-k{k}", make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=k)
        )))
    for widths in ((8,), (16,), (8, 8)):
        name = "mlp-tanh-" + "x".join(map(str, widths))
        choices.append((name, make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=widths, activation="tanh", solver="lbfgs",
            alpha=0.01, max_iter=1000, tol=1e-7, random_state=74
        ))))
    return choices


def main():
    x, y, development, inspection, reserve, folds = load_roles()
    choices = candidate_models()
    means = []
    print("Role rows:", len(development), len(inspection), len(reserve))
    for name, estimator in choices:
        scores = []
        for fitting, validation in folds:
            model = clone(estimator).fit(x[fitting], y[fitting])
            scores.append(accuracy_score(y[validation], model.predict(x[validation])))
        means.append(float(np.mean(scores)))
        print(name, "folds", np.round(scores, 6), "mean", round(means[-1], 6))

    # max returns the first declared candidate when scores tie.
    selected = max(range(len(choices)), key=lambda i: means[i])
    for role, index in (("selected", selected), ("declared_baseline", 3)):
        name, estimator = choices[index]
        model = clone(estimator).fit(x[development], y[development])
        prediction = model.predict(x[inspection])
        print(role, name, accuracy_score(y[inspection], prediction))
        print(confusion_matrix(y[inspection], prediction, labels=[0, 1]))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
```

Every candidate sees the same three validation groups. `clone` creates a fresh estimator for each fit. A pipeline refits its scaler inside the fold. No scaling operation runs on all development rows before CV. The selected model is then refitted on all development rows, which is appropriate after the configuration is fixed.

### What actually happened

There were 33 fold fits and two final refits: **35 estimator fits**. The retained run produced no fitting warnings.

| Configuration | Mean fold accuracy | Neural parameters, where applicable |
| --- | ---: | ---: |
| Logistic, raw, C = 0.1 | 0.977420 | — |
| Logistic, raw, C = 1 | 0.983883 | — |
| Logistic, standardized, C = 0.1 | 0.964439 | — |
| Logistic, standardized, C = 1 | 0.975266 | — |
| Tree, depth 2 | 0.905760 | — |
| Tree, depth 5 | 0.962198 | — |
| Standardized neighbors, k = 3 | 0.998907 | — |
| Standardized neighbors, k = 9 | 0.994629 | — |
| Standardized tanh network, width 8 | 0.998907 | 49 |
| Standardized tanh network, width 16 | 1.000000 | 97 |
| Standardized tanh network, widths 8, 8 | 0.998907 | 121 |

The width-16 network wins the declared criterion. On the separate 205-row inspection partition, it classifies 205 rows correctly. The predeclared standardized logistic baseline with \(C=1\) classifies 202 correctly. Its confusion matrix, with true classes as rows and predicted classes as columns, is \(\begin{bmatrix}111&3\\0&91\end{bmatrix}\); the selected network's is \(\begin{bmatrix}114&0\\0&91\end{bmatrix}\). Always predicting the development majority class 0 would classify 114 of 205 inspection rows correctly.

**[Figure F3: observed results, with individual errors available.]** The overview shows candidate means with all three fold values rather than invented error bars. A separate inspection panel compares only the two declared models. Selecting an error reveals its supplied features, original row identifier, true numeric label, and recorded prediction. It does not fabricate a source photograph or a newly fitted decision boundary.

Three observations deserve explanation:

- **More layers did not win.** The two-hidden-layer model has more parameters than the width-16 model but makes one out-of-fold error. This small result does not establish a universal advantage for shallow networks; it shows why architecture size is a choice to evaluate.
- **Scaling did not improve every fixed logistic configuration.** At a fixed \(C\), changing feature scales changes the relationship between a coefficient penalty and effects in original feature units. It also affects numerical conditioning. The search is comparing complete procedures, not testing a theorem that preprocessing always helps. The [regularization lesson](/learn/topic/regularization-l1-l2-elastic-net-dropout) explains this geometry.
- **A simple competing family was already strong.** Three-neighbor classification and the smaller networks each make one out-of-fold error. Searching only neural networks would conceal that context.

Perfect inspection accuracy is still a finite observation. The 205 records do not cover new currencies, capture devices, adversarial counterfeits, or future distribution shifts. Exact-feature grouping removes one identifiable leakage route; it does not prove independence of every physical specimen. No confidence interval calculated under independent Bernoulli trials can repair missing specimen identities. The experiment establishes a reproducible comparison on this declared source and partition, with a further reserved set still unscored.

### Replay a budget without pretending to run a new optimizer

**[Investigation I5: the search ledger.]** The recorded candidate table can be revealed in a fixed random order: k = 9 neighbors, the two-layer network, k = 3 neighbors, the width-16 network, standardized logistic C = 1, standardized logistic C = 0.1, raw logistic C = 1, depth-2 tree, width-8 network, raw logistic C = 0.1, depth-5 tree. The order was generated with seed 75.

Advance the budget and inspect whether the newly observed candidate changes the recommended configuration or just adds evidence. With two candidates revealed, the two-layer network is recommended. With three, k = 3 neighbors ties its score and wins the predeclared registry-order tie rule. The best score stays the same even though the recommended model changes. With four, width sixteen becomes the winner.

This is a replay of a fully evaluated finite table. It is not a measured comparison between random search and Bayesian optimization, and it does not erase the 35 fits used to construct the evidence. The inspection outcomes remain separate from the replay: you cannot select a different candidate by browsing its inspection accuracy because those additional predictions were never made.

## 6. Turn search results into a useful learning system

### Search, combine, and transfer are different operations

Selecting the best observed single candidate is only one use of a search history. An ensemble can combine predictions from several fitted models. Diversity matters: averaging two models with identical errors adds little; complementary errors can help. The combination itself must be learned and assessed with appropriate data separation.

In a weighted average of class-1 probabilities,

\[
\widehat p(x)=\sum_m a_m\widehat p_m(x),\qquad a_m\ge0,\quad\sum_m a_m=1,
\]

the weights are another learned choice. Greedy ensemble selection can repeatedly add the candidate that improves the current combination; selection with replacement gives a model extra weight through repeated inclusion. Stacking instead trains a second-level predictor on candidate outputs, normally using out-of-fold predictions for its training inputs. These are distinct procedures. The auto-sklearn chapter of the [AutoML book](https://automl.org/book/) describes greedy ensemble selection, so “take the top few and stack them” is not an accurate description of that original method.

Out-of-fold prediction prevents each row's base prediction from coming from a base model fitted on that same row. It does not magically protect every subsequent search decision. If you repeatedly choose ensembles using the same out-of-fold record, that record has become selection evidence. Preserve an assessment boundary for the complete system.

**Meta-learning** uses experience from previous tasks to guide a new task. It may suggest configurations, learn which task characteristics predict useful choices, or transfer fitted representations. A *portfolio* is a small collection of configurations selected to cover different tasks well.

Consider constructed losses for three configurations on two old tasks:

| Configuration | Old task 1 | Old task 2 | Mean |
| --- | ---: | ---: | ---: |
| A | 0.10 | 0.50 | 0.30 |
| B | 0.50 | 0.10 | 0.30 |
| C | 0.25 | 0.25 | 0.25 |

C is the best single default. But trying A and B and selecting between them on each task gives a mean best loss of 0.10. A useful portfolio covers complementary strengths, rather than simply containing the best average performer twice. On a new task with losses A = 0.40, B = 0.40, C = 0.20, that old portfolio misses the best choice. Transfer requires validation on new tasks; similar dataset summaries are evidence to investigate, not a guarantee of similar model rankings.

Current AutoML systems can also use pretrained tabular models or learned portfolios. Their pretraining data, licensing, resource needs, and task overlap become part of the evaluation. The dedicated [AutoML as meta-learning lesson](/learn/topic/automl-as-meta-learning) develops those cross-task decisions.

### Sometimes the search language is an explanation

An interesting application is the Automatic Statistician: its modeling grammar can combine components representing smooth change, periodicity, or noise in a time series. A search can then return a structured model that supports a verbal explanation, such as a seasonal pattern whose amplitude changes over time. This illustrates why the search space determines what kinds of explanations can be produced.

The interpretation is conditional on the grammar, data, and fitted model. A periodic component is not proof of a physical cause. The [book's Automatic Statistician chapter](https://automl.org/wp-content/uploads/2019/05/AutoML_Book.pdf) describes kernel composition, structure search, and generated descriptions. The later Gaussian-process lesson supplies the probability model underlying those kernels; the model-selection criteria introduced in [regularization](/learn/topic/regularization-l1-l2-elastic-net-dropout) explain why raw fitting likelihood alone favors unnecessary complexity.

### Optimize for the device that will run the result

A model with fewer parameters need not be faster on a specific device. Memory access, tensor shapes, operator implementations, parallelism, and data movement all matter. Multiplication counts and parameter counts are useful descriptors; deployment latency is a measurement with conditions: hardware, software, input shape, batch size, precision, warm-up, and the timing boundary.

**[Investigation I6: explore accuracy and latency.]** A scatterplot compares accuracy to inference latency for five explicitly hypothetical candidates. A has 90% accuracy at 2 ms; B, 94% at 4 ms; C, 96% at 8 ms; D, 93% at 5 ms; E, 89% at 3 ms. A, B, and C form the Pareto frontier. D is worse than B in both objectives; E is worse than A. A 5 ms cap selects B; a 3 ms cap selects A.

A point is *Pareto dominated* if another point is no worse in every objective and strictly better in at least one. The frontier contains choices that require a trade-off. It does not choose the trade-off for you. A soft score that penalizes latency can still prefer an over-budget model. If 5 ms is a hard requirement, filter out infeasible candidates using a defined measurement protocol before choosing among the remainder. The [MnasNet paper](https://arxiv.org/pdf/1807.11626) is a primary example of architecture search that includes target-device latency; its results should not be transplanted as timing predictions for another device.

### A practical decision sequence

1. Establish a useful baseline and a valid deployment-related split. Include error costs or group-specific requirements in the objective when the task demands them.
2. Define a compact space with justified ranges and valid conditional settings. Include a strong simple competitor, not only an expensive family you hope will win.
3. Measure a few representative fits to understand cost and failures. Decide whether enumeration, random proposals, a surrogate, or a fidelity scheduler addresses the actual bottleneck.
4. Record preprocessing, folds, seeds, resources, scores, warnings, and active settings for each trial. Keep the selection rule fixed, including ties.
5. Evaluate the selected procedure with evidence outside its selection loop. Assess deployment latency, memory, and relevant error patterns under their own declared protocols.
6. Preserve the fitted preprocessing and model together, along with input schema and environment information. In production, monitor task-relevant changes and outcomes; a statistical change in input distribution alone does not quantify prediction harm.

No permanent league table can rank AutoML libraries for every dataset and resource limit. As inspected in September 2026, FLAML exposes task, metric, budget, estimator, and resampling controls; AutoGluon's current tabular presets include different ensembles and learned model portfolios; KerasTuner exposes neural hyperparameter spaces and tuners. Microsoft NNI is archived and read-only, so treat it as a historical resource rather than a default maintained choice. The linked documentation in §10 records the current interfaces instead of promising fixed installation times or universal winners.

## 7. Deeper branch: differentiable search and cheaper architecture evidence

This branch connects the graph view in §4 to gradients. It is useful when you want to understand what a differentiable NAS method optimizes, and what changes when a searched network becomes a deployed network.

### Replace a discrete choice with a mixture

Suppose an edge could apply one of several shape-compatible operations \(o_1,\ldots,o_m\). Introduce architecture logits \(\alpha_1,\ldots,\alpha_m\), turn them into softmax weights,

\[
p_i=\frac{e^{\alpha_i}}{\sum_j e^{\alpha_j}},\qquad
\overline o(x)=\sum_i p_i o_i(x),
\]

and evaluate the mixture while searching. The logits are not class probabilities. They control how candidate operations contribute to an intermediate computation.

For a constructed scalar edge at input \(x=2\), let the operations be zero, identity, and negation. Their outputs are \([0,2,-2]\). With logits \([\log2,0,0]\), the probabilities are \([0.5,0.25,0.25]\), so the mixed output is zero. If the target is 1 and loss is \(\tfrac12(\overline o-1)^2\), the loss is 0.5.

The softmax derivative gives

\[
\frac{\partial\overline o}{\partial\alpha_i}
=p_i\big(o_i-\overline o\big),\qquad
\frac{\partial L}{\partial\alpha_i}
=(\overline o-1)p_i\big(o_i-\overline o\big).
\]

The gradient is \([0,-0.5,0.5]\). One gradient step of size 0.4 changes the logits to \([\log2,0.2,-0.2]\), yielding output about 0.199336. The identity operation's contribution increases, which moves the mixture toward the target.

**[Investigation I4: mix operations, then commit to one.]** Edit the input, target, and logits; change the inputs to inspect the mixture and gradient. Adding the same number to all logits leaves the probabilities unchanged. The “commit” action selects a discrete operation and shows the resulting function alongside the mixture. With identity and negation weighted equally and target zero at input two, the mixture has zero loss, while either discrete choice has loss two. Retraining a chosen architecture may change its outcome; this small example isolates why discretization itself can change the function.

The [DARTS paper](https://arxiv.org/pdf/1806.09055) uses continuous mixtures to search cell structures. Its convolutional-cell discretization retains two strong nonzero operations from distinct incoming nodes for each intermediate node; its recurrent-cell construction uses one. This is more specific than independently keeping the largest logit on every possible edge. Shape-compatible search and the final graph-construction rule both belong in a reproducible method.

### The architecture should anticipate trained weights

For a network, operations can have trainable weights \(w\) in addition to architecture variables \(\alpha\). The intended nested problem is

\[
\min_\alpha L_{\mathrm{val}}(w^*(\alpha),\alpha),\qquad
w^*(\alpha)\in\arg\min_w L_{\mathrm{train}}(w,\alpha).
\]

The inner problem asks which weights fit the training data for an architecture. The outer problem asks how that fitted architecture performs on validation data. This is **bilevel optimization**. Validation observations used to optimize architecture are selection data; they cannot simultaneously serve as an untouched final test.

Solving the inner training problem from scratch after every architecture change is expensive. A one-step approximation uses

\[
w'=w-\xi\nabla_wL_{\mathrm{train}}(w,\alpha)
\]

and differentiates \(L_{\mathrm{val}}(w',\alpha)\). Treat the current \(w\) as fixed for this approximation. The chain rule gives

\[
\nabla_\alpha L_{\mathrm{val}}(w',\alpha)
-\xi\nabla^2_{\alpha,w}L_{\mathrm{train}}(w,\alpha)
\nabla_{w'}L_{\mathrm{val}}(w',\alpha).
\]

The first term is the direct effect of architecture on validation loss at the updated weights. The second is the effect of architecture on the training step, which changes those weights. The mixed Hessian multiplies a vector; an implementation need not store a full matrix. A central finite difference of training architecture-gradients at \(w\pm\epsilon v\), divided by \(2\epsilon\), approximates this product for \(v=\nabla_{w'}L_{\mathrm{val}}\), subject to the usual step-size and numerical-error trade-offs.

In the DARTS terminology, setting \(\xi=0\) gives the **first-order approximation**. Keeping the nonzero one-step dependency includes the mixed derivative and is called the second-order approximation. A one-step unroll is therefore not automatically “first order” merely because it contains one training step.

**[Figure F4: three derivatives of the same small problem.]** Use

\[
L_{\mathrm{train}}=\tfrac12(w-\alpha)^2,\qquad
L_{\mathrm{val}}=\tfrac12(w-1)^2.
\]

At \(w=0\), \(\alpha=0.2\), and \(\xi=0.1\), the one-step weight is \(w'=0.02\). The direct architecture derivative at fixed \(w\) is zero: \(\alpha\) does not appear explicitly in the validation formula. The one-step derivative is \((0.02-1)(0.1)=-0.098\). Solving the inner problem exactly gives \(w^*=\alpha\), so the true outer derivative is \(\alpha-1=-0.8\).

These three numbers answer different questions. Even if the current training gradient is zero at a particular point, its derivative with respect to architecture need not be zero. Equality of the current and one-step weight values at that point does not justify dropping the chain-rule term. This distinction prevents a common confusion between evaluating an expression and differentiating the function that produced it.

Nonconvex training can have multiple local solutions, and an approximation may poorly track their response to architecture changes. A large identity-operation weight can also reflect search dynamics, optimization ease, or the relaxation, rather than a universally best final graph. Adding an architecture penalty changes the objective; it does not guarantee that every such failure disappears.

### Independent fits, shared weights, and proxies

In our banknote study, each architecture receives fresh fitted weights in each fold. A *supernetwork* can instead contain many candidate subgraphs and train shared weights. Evaluating a subgraph then reuses part of that state. This saves work but couples candidates: an operation may benefit from how frequently it was sampled, which other paths trained it, and which weights it shares. The shared-weight ranking can differ from the ranking after independent full training.

**[Figure F5: what evidence did this architecture receive?]** Contrast three lanes: independently trained candidate, subgraph evaluated with shared weights, and untrained proxy. Label the state reused in each lane and the target claim that still requires an independently evaluated final architecture. The diagram has no invented speed multiplier.

Reinforcement-learning search can view architecture choices as a sequence of actions with a validation-based reward. Evolutionary search can mutate architectures and select promising descendants. Bayesian optimization can model the architecture-to-score relation. Network morphisms can expand a network while preserving its current function under specific constructions. These are different ways to propose candidates or reuse training; none removes the need to define the final evaluator and compare against a competent simple search under matching conditions.

One unusual proxy, **NASWOT**, examines activation patterns at random initialization without performing ordinary network training. A ReLU activation outputs \(\max(0,z)\) for its incoming weighted sum \(z\); call the unit active when that sum is positive. For a small batch passing through these units, record a binary code indicating which units are active. If there are \(N_A\) recorded units, one kernel counts shared activation decisions:

\[
K_{ij}=N_A-d_H(c_i,c_j),
\]

where \(d_H\) is Hamming distance: the number of positions at which two codes differ. The proposed score uses \(\log\det K\). With codes 110 and 101, \(K=\begin{bmatrix}3&1\\1&3\end{bmatrix}\), so the determinant is 8. Identical codes give a singular matrix and determinant zero. This helps visualize the score's preference for differentiated activation patterns; it does not prove that such differentiation will generalize after training.

The [NASWOT paper](https://proceedings.mlr.press/v139/mellor21a/mellor21a.pdf) evaluates this signal on architecture benchmarks and studies sensitivity to initialization and batches. Its main construction uses actual input mini-batches; Gaussian random inputs are an ablation, not the definition of the method. “Without training” still involves computation, including a forward pass and a matrix calculation. A practical implementation must specify handling of singular kernels and distinguish any numerical regularization from the original exact formula.

### Keep the final comparison honest

Use the same task data, candidate space, fitting budget, and deployment protocol when comparing search methods, unless a changed component is the explicit subject of the experiment. Count search work, proxy evaluation, architecture selection, and final retraining. Separate variability from architecture-training seeds and variability from the search process itself. Repeatedly consulting a public benchmark's test outcomes can turn that benchmark into selection evidence, even when no single training script reads those labels.

The later [dedicated NAS lesson](/learn/topic/neural-architecture-search-nas) develops convolutional cells, search benchmarks, shared-weight implementation, and hardware evaluation. The local mechanism here is complete enough to explain what those methods optimize without pretending that a tiny multilayer perceptron validates a full image-model search system.

## 8. Optional: translate the contract into maintained tools

These examples show how to preserve the contract when adopting a library. They are optional runnable extensions, not sources of the observed scores in §5. Neither FLAML nor Keras/TensorFlow was installed or executed for this content packet; choose and record compatible versions during implementation. Both extensions use only the first development fold from `banknote_search.py`; neither accesses inspection or reserved rows.

### FLAML: make the validation method explicit

Install `flaml[automl]` in the study environment, save this as `banknote_flaml.py` beside the previous files, and run it. A quoted extras expression avoids shell interpretation differences.

```bash
python -m pip install "flaml[automl]"
python banknote_flaml.py
```

```python
# banknote_flaml.py
from flaml import AutoML
from sklearn.metrics import accuracy_score
from banknote_search import load_roles

x, y, development, inspection, reserve, folds = load_roles()
fitting, validation = folds[0]
search = AutoML()
search.fit(
    X_train=x[fitting], y_train=y[fitting],
    X_val=x[validation], y_val=y[validation],
    task="classification", metric="accuracy", eval_method="holdout",
    estimator_list=["lrl2", "rf"], time_budget=30, max_iter=8,
    n_jobs=1, seed=76, retrain_full=False,
)
print("Selected family:", search.best_estimator)
print("Selected settings:", search.best_config)
print("Selection holdout loss:", search.best_loss)
print("Selection holdout accuracy:", accuracy_score(
    y[validation], search.predict(x[validation])
))
```

The 30-second setting is a requested search budget, and eight iterations is an additional bound; actual duration includes library and estimator behavior. `best_loss` is a minimized objective, so for the requested accuracy metric its ideal interpretation is one minus accuracy. The printed accuracy reuses selection data and is labeled accordingly. It is not the three-fold score from §5 or an independent final assessment. FLAML's `auto` evaluation mode can choose holdout or CV, which is why the example specifies its intended method. See the [official task-oriented AutoML documentation](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/) for resampling and estimator controls.

### KerasTuner: a conditional neural search space

This extension searches one or two tanh hidden layers. The second width exists only when depth is two. It uses a separate fitting procedure from §5: minibatch Adam rather than L-BFGS. An epoch is one pass through the fitting data; Adam updates weights using batches. The deep-learning module explains that optimizer. Here its settings are fixed except for a declared learning-rate choice.

Use an environment with TensorFlow, Keras, and KerasTuner, then save and run `banknote_keras_search.py`. The tuner stores its trial state in `banknote-nas-study/conditional-mlp`; keeping the same directory resumes compatible state. Use a new descriptive project name for a different declared experiment.

```bash
python -m pip install tensorflow keras keras-tuner
python banknote_keras_search.py
```

```python
# banknote_keras_search.py
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
from banknote_search import load_roles

keras.utils.set_random_seed(77)
x, y, development, inspection, reserve, folds = load_roles()
fitting, validation = folds[0]
scaler = StandardScaler().fit(x[fitting])
x_fit = scaler.transform(x[fitting]).astype("float32")
x_validation = scaler.transform(x[validation]).astype("float32")


def build_model(hp):
    depth = hp.Choice("depth", [1, 2])
    first_width = hp.Choice("first_width", [8, 16])
    with hp.conditional_scope("depth", [2]):
        second_width = hp.Choice("second_width", [8, 16])
    model = keras.Sequential([
        keras.Input(shape=(4,)),
        keras.layers.Dense(first_width, activation="tanh"),
    ])
    if depth == 2:
        model.add(keras.layers.Dense(second_width, activation="tanh"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    rate = hp.Choice("learning_rate", [0.001, 0.01])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=rate),
        loss="binary_crossentropy", metrics=["accuracy"],
    )
    return model


tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=4,
    executions_per_trial=1, seed=77,
    directory="banknote-nas-study", project_name="conditional-mlp",
    overwrite=False,
)
tuner.search(
    x_fit, y[fitting], validation_data=(x_validation, y[validation]),
    epochs=20, batch_size=32, verbose=0,
)
best_settings = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.get_best_models(1)[0]
print("Selected settings:", best_settings.values)
print("Selection holdout metrics:", best_model.evaluate(
    x_validation, y[validation], return_dict=True, verbose=0
))
```

The `conditional_scope` registers when a setting is active; it does not skip execution of its Python body. An inactive second width can be `None`, which is why the model adds that layer only inside `if depth == 2`. Four trials explore part of a twelve-configuration space: four one-layer combinations and eight two-layer combinations. They do not exhaust it.

The tuner selects using validation performance, including its checkpoint behavior across epochs. That checkpoint choice is part of the selection procedure. A high printed score still needs assessment outside this holdout before supporting a generalization claim. The [conditional-hyperparameter reference](https://keras.io/keras_tuner/api/hyperparameters/) and [complete getting-started guide](https://keras.io/keras_tuner/getting_started/) explain the interface and retrieval of selected settings.

## 9. Practice: design, calculate, and diagnose

Try the core questions 1–6 before the deeper questions. Open a hint or solution when you need it. Changing an answer after a reveal is useful reflection, but it is different from an unaided first attempt.

### 1. Count the experiments

A new space contains logistic regression with three \(C\) values and two preprocessing options; trees with four depths; and neural networks with either one hidden layer of width 6 or 12, or two hidden layers with each width independently 6 or 12. How many valid configurations exist? How many fits does four-fold CV require before refits?

<details><summary>Hint</summary>
Add independent family branches. Multiply settings that are simultaneously active within a branch.
</details>
<details><summary>Solution</summary>
Logistic contributes 6, trees 4, one-layer networks 2, and two-layer networks 4. Total 16 configurations; four-fold CV requires 64 fits. Counting an inactive second width for a one-layer network would double-count its functionally identical configurations.
</details>

### 2. Read the evidence boundary

A colleague standardizes the full development matrix, runs fold-based AutoML on it, then tries five cost thresholds on the final inspection set and reports the best inspection cost. Identify two distinct boundary violations and repair them.

<details><summary>Hint</summary>
Ask which rows fitted the preprocessing and which rows selected the threshold.
</details>
<details><summary>Solution</summary>
The scaler learned validation-fold information before CV; place it inside each candidate pipeline and fit it on each fold's fitting rows. The inspection set selected the threshold; choose thresholds using selection data under the task's cost policy, then assess the fixed model-plus-threshold procedure with separate evidence. Calling the second step “just postprocessing” does not restore independence.
</details>

### 3. Spend a small fidelity budget

Four candidates have losses at resources 1, 2, and 4: A = (0.10, 0.09, 0.08), B = (0.11, 0.08, 0.07), C = (0.12, 0.07, 0.02), D = (0.20, 0.18, 0.15). Keep half at each cut. Which candidate wins? What is the work with restart and with genuine continuation? Which full-resource winner is missed?

<details><summary>Solution</summary>
A and B survive resource 1; B survives resource 2 and finishes at 0.07. Restart work is \(4+2(2)+1(4)=12\); continuation work is \(4+2(2-1)+1(4-2)=8\). C would reach 0.02 but is removed at the first cut. Its good later value cannot inform a real early decision unless that evidence is actually purchased.
</details>

### 4. Count network parameters, including biases

For five inputs, hidden widths 6 and 3, and one binary output, compute every parameter block. Would replacing the two layers with one width-12 layer necessarily improve accuracy?

<details><summary>Solution</summary>
The blocks contain \((5+1)6=36\), \((6+1)3=21\), and \((3+1)1=4\) parameters, totaling 61. A one-layer width-12 model has \((5+1)12+(12+1)=85\). Neither parameter count determines accuracy: representation, optimization, regularization, and data all matter. Compare them under a declared evaluator.
</details>

### 5. Interpret the real search replay

Reveal only the first three candidates in I5. Explain why the recommended candidate can change while the best-so-far score stays flat. Can the selected width-16 model's eventual inspection result be attached to the budget-three recommendation?

<details><summary>Solution</summary>
The two-layer network and k = 3 neighbors tie at 0.998907 mean fold accuracy. The registry-order tie rule favors neighbors when it becomes available. A flat maximum does not imply the selected model is unchanged. The width-16 model has not been revealed at budget three and its inspection result belongs to a different selected procedure. The replay cannot borrow that outcome.
</details>

### 6. Apply a hard deployment constraint

Three hypothetical models have (latency, accuracy) pairs P = (3 ms, 0.92), Q = (6 ms, 0.96), and R = (5 ms, 0.91). Identify the frontier and choose under a 5 ms cap. Explain why adding a finite latency penalty to accuracy need not enforce the cap.

<details><summary>Solution</summary>
P dominates R, while P and Q trade speed for accuracy. The frontier is P and Q; the cap leaves P as the best feasible candidate. A finite penalty allows accuracy gains to compensate for exceeding the cap. Feasibility filtering encodes a hard bound directly, provided the latency measurement itself matches the requirement.
</details>

### 7. Calculate expected improvement

The incumbent loss is 0.3. Candidate U has a deterministic predicted loss of 0.25. Candidate V has Gaussian predicted mean 0.3 and standard deviation 0.1. Which has larger EI? What additional fact would you need before claiming that it will actually improve validation performance?

<details><summary>Solution</summary>
U has EI 0.05. V has \(0.1\phi(0)=0.1/\sqrt{2\pi}\approx0.039894\), so U wins this acquisition comparison. Actual performance requires an evaluation; the surrogate's probability model and acquisition ranking are not observed objective values.
</details>

### 8. Differentiate an operation mixture

At one input, two operations output 3 and −1. Their logits are equal, the target is zero, and loss is half squared error. Find the mixture and both architecture gradients. Then add 7 to both logits.

<details><summary>Solution</summary>
Probabilities are one half, output is 1, and loss is 0.5. Gradients are \(1(0.5)(3-1)=1\) and \(1(0.5)(-1-1)=-1\). Adding a common constant leaves probabilities, output, loss, and gradients unchanged. It changes a redundant coordinate representation, not the mixture.
</details>

### 9. Separate three architecture derivatives

Use \(L_{train}=\tfrac12(w-\alpha)^2\), \(L_{val}=\tfrac12(w-2)^2\), current \(w=0\), \(\alpha=0.5\), and \(\xi=0.2\). Compute the first-order direct derivative, the one-step derivative, and the exact-inner outer derivative.

<details><summary>Solution</summary>
The direct derivative is zero. The one-step weight is 0.1 and its derivative with respect to architecture is 0.2, so the one-step outer derivative is \((0.1-2)(0.2)=-0.38\). The exact inner optimum is \(w^*=\alpha\), yielding derivative \(\alpha-2=-1.5\). These are three distinct functions being differentiated, not rounding differences.
</details>

### 10. Design an experiment that could disappoint you

Choose a small classification task with a documented prediction-time feature set. Declare a simple baseline, two justified model families, a conditional space, grouping or temporal boundaries, selection metric, budget, tie rule, and final assessment. Predict one result that would make you simplify the system. Explain which observation would invalidate your original split rather than merely favor a different optimizer.

<details><summary>Example response and assessment criteria</summary>
A valid response could compare a standardized regularized linear model with bounded-depth trees for repeated measurements from devices, keeping each device in one fold when deployment concerns new devices. It would reserve devices for final assessment and declare latency conditions. A near-tie favoring the simpler baseline could justify choosing it. Discovering that device identifiers were duplicated across roles would require repairing the evaluation boundary and reassessing affected conclusions. A large search score alone is not evidence that the split is valid. Other tasks can satisfy the same criteria with different models and boundaries.
</details>

You are ready to continue when you can distinguish a configuration from fitted weights, count a conditional space, protect fitting and selection boundaries, explain a fidelity failure, and interpret the real result without treating the finite search winner as a universal best model. The deeper questions prepare you to inspect differentiable NAS implementations and proxy claims.

## 10. Connections and other ways to learn

The next topic in this module is [Hidden Markov Models](/learn/topic/hidden-markov-models-hmm). AutoML chooses among learning procedures; an HMM introduces a particular probabilistic structure for observations that arrive in sequence and depend on unobserved states. It will distinguish summing over possible hidden paths from finding one best path. This is a change in modeling assumptions, not simply another knob for the current independent-row classifier.

For focused review, revisit [feature scaling and encoding](/learn/topic/feature-scaling-encoding-imputation), [cross-validation](/learn/topic/cross-validation-hyperparameter-tuning), [regularization](/learn/topic/regularization-l1-l2-elastic-net-dropout), and [feature selection](/learn/topic/feature-selection-importance-shap-permutation-mutual-info). For deeper branches, use [Gaussian processes](/learn/topic/gaussian-processes-gp), [dedicated NAS](/learn/topic/neural-architecture-search-nas), and [AutoML as meta-learning](/learn/topic/automl-as-meta-learning).

- **Foundations and an alternative organized treatment:** the openly licensed [AutoML book edited by Hutter, Kotthoff, and Vanschoren](https://www.automl.org/book/). Chapters 1–3 separate hyperparameter optimization, meta-learning, and architecture search; the auto-sklearn and Automatic Statistician chapters show different uses of a search history and search language. Read the relevant section after its local example rather than treating the whole book as a prerequisite.
- **A compact scheduling explanation:** [Kevin Jamieson's Hyperband page](https://homes.cs.washington.edu/~jamieson/hyperband.html) contains the algorithm, bracket table, and experimental protocol. Its breadth/depth table is a useful second representation of I3. The accompanying [JMLR paper](https://www.jmlr.org/papers/volume18/16-558/16-558.pdf) gives assumptions and analysis.
- **Differentiable NAS:** [Liu, Simonyan, and Yang, DARTS](https://arxiv.org/pdf/1806.09055), §§2.1–2.4. Read alongside the locally derived mixture and scalar bilevel example; distinguish its relaxation, approximation, and discretization stages.
- **A surprising proxy to examine critically:** [Mellor and colleagues, Neural Architecture Search without Training](https://proceedings.mlr.press/v139/mellor21a/mellor21a.pdf), especially its activation-pattern construction and ablations. The two-code calculation here supplies a concrete entry point.
- **Deployment-aware search:** [Tan and colleagues, MnasNet](https://arxiv.org/pdf/1807.11626). Focus on the distinction between measured latency and operation counts, and between a soft objective and an actual feasibility requirement.
- **Observed input and licensing:** [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication), [dataset DOI](https://doi.org/10.24432/C55P57). The accompanying provenance describes duplicate grouping and the limits of the available metadata.
- **Executable alternative learning paths:** [FLAML task-oriented AutoML](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/), [KerasTuner getting started](https://keras.io/keras_tuner/getting_started/), and [its conditional hyperparameters](https://keras.io/keras_tuner/api/hyperparameters/). They supply complete API context for §8; their tutorial scores are not this lesson's measurements.
- **Current broader tabular workflows:** [AutoGluon tabular essentials](https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html). Compare its presets and resource implications against your contract. For historical code, the [NNI repository](https://github.com/microsoft/nni) explicitly records its archived status.

The linked resources were inspected for their substantive explanations, algorithms, or interfaces in September 2026.
